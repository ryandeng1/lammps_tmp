/* ----------------------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   https://www.lammps.org/, Sandia National Laboratories
   LAMMPS development team: developers@lammps.org

   Copyright (2003) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under
   the GNU General Public License.

   See the README file in the top-level LAMMPS directory.
------------------------------------------------------------------------- */

/* ----------------------------------------------------------------------
   Contributing author (triclinic) : Pieter in 't Veld (SNL)
------------------------------------------------------------------------- */

#include <cmath>
#include <cstring>
#include <iostream>
#include <ctime>
#include <iomanip>
#include <algorithm>

#include "atom.h"
#include "atom_vec.h"
#include "bond.h"
#include "comm_brick.h"
#include "compute.h"
#include "domain.h"
#include "dump.h"
#include "error.h"
#include "fix.h"
#include "memory.h"
#include "neighbor.h"
#include "pair.h"
#include "stencil_md_utils.h"
#include "stencil_md.h"
#include "timer.h"
#include <cilk/cilk.h>
#include <sstream>

using namespace LAMMPS_NS;

#define BUFFACTOR 1.5
#define BUFMIN 1024
#define BIG 1.0e20

static constexpr bool TRACK_LAMMPS_DATA = true;

/* ---------------------------------------------------------------------- */

CommBrick::CommBrick(LAMMPS *lmp) :
    Comm(lmp), sendnum(nullptr), recvnum(nullptr), sendproc(nullptr), recvproc(nullptr),
    size_forward_recv(nullptr), size_reverse_send(nullptr), size_reverse_recv(nullptr),
    slablo(nullptr), slabhi(nullptr), multilo(nullptr), multihi(nullptr), multioldlo(nullptr),
    multioldhi(nullptr), cutghostmulti(nullptr), cutghostmultiold(nullptr), pbc_flag(nullptr),
    pbc(nullptr), firstrecv(nullptr), sendlist(nullptr), localsendlist(nullptr),
    maxsendlist(nullptr), buf_send(nullptr), buf_recv(nullptr)
{
  style = 0;
  layout = Comm::LAYOUT_UNIFORM;
  pbc_flag = nullptr;
  init_buffers();
}

/* ---------------------------------------------------------------------- */

CommBrick::~CommBrick()
{
  CommBrick::free_swap();
  if (mode == Comm::MULTI) {
    CommBrick::free_multi();
    memory->destroy(cutghostmulti);
  }

  if (mode == Comm::MULTIOLD) {
    CommBrick::free_multiold();
    memory->destroy(cutghostmultiold);
  }

  if (sendlist)
    for (int i = 0; i < maxswap; i++) memory->destroy(sendlist[i]);
  if (localsendlist) memory->destroy(localsendlist);
  memory->sfree(sendlist);
  memory->destroy(maxsendlist);

  memory->destroy(buf_send);
  memory->destroy(buf_recv);

  for (int i = 0; i < maxswap; i++) {
    memory->destroy(buf_sendlist_stencil_md[i]);
    memory->destroy(buf_recv_sendlist_stencil_md[i]);
  }
  memory->sfree(buf_sendlist_stencil_md);
  memory->sfree(buf_recv_sendlist_stencil_md);

  for (int i = 0; i < NUM_TIMESTEPS_IN_PARALLEL + 1; i++) {
    if (sendlist_stencil_md[i]) {
      for (int j = 0; j < maxswap; j++) {
        memory->destroy(sendlist_stencil_md[i][j]);
        memory->destroy(send_force_stencil_md[i][j]);
        memory->destroy(send_vel_stencil_md[i][j]);
        memory->destroy(send_pos_stencil_md[i][j]);
      }
    }

    if (sendlist_stencil_md_next_dt[i]) {
      for (int j = 0; j < maxswap; j++) {
        memory->destroy(sendlist_stencil_md_next_dt[i][j]);
        memory->destroy(send_force_stencil_md_next_dt[i][j]);
        memory->destroy(send_vel_stencil_md_next_dt[i][j]);
        memory->destroy(send_pos_stencil_md_next_dt[i][j]);
      }
    }

    memory->sfree(sendlist_stencil_md[i]);
    memory->sfree(send_force_stencil_md[i]);
    memory->sfree(send_vel_stencil_md[i]);
    memory->sfree(send_pos_stencil_md[i]);
    memory->destroy(maxsendlist_stencil_md[i]);

    memory->sfree(sendlist_stencil_md_next_dt[i]);
    memory->sfree(send_force_stencil_md_next_dt[i]);
    memory->sfree(send_vel_stencil_md_next_dt[i]);
    memory->sfree(send_pos_stencil_md_next_dt[i]);
    memory->destroy(maxsendlist_stencil_md_next_dt[i]);

    if (second_sendlist_stencil_md[i]) {
      for (int j = 0; j < maxswap; j++) { memory->destroy(second_sendlist_stencil_md[i][j]); }
    }
    memory->sfree(second_sendlist_stencil_md[i]);
    memory->destroy(max_second_sendlist_stencil_md[i]);

    if (second_sendlist_stencil_md_next_dt[i]) {
      for (int j = 0; j < maxswap; j++) { memory->destroy(second_sendlist_stencil_md_next_dt[i][j]); }
    }
    memory->sfree(second_sendlist_stencil_md_next_dt[i]);
    memory->destroy(max_second_sendlist_stencil_md_next_dt[i]);
  }

  memory->destroy(maxsend_stencil_md);
  memory->destroy(maxrecv_stencil_md);
  memory->destroy(maxsend_sendlist_stencil_md);
  memory->destroy(maxrecv_sendlist_stencil_md);
  for (int i = 0; i < maxswap; i++) {
      memory->destroy(buf_send_stencil_md[i]);
      memory->destroy(buf_recv_stencil_md[i]);
  }
  memory->sfree(buf_send_stencil_md);
  memory->sfree(buf_recv_stencil_md);
}

/* ---------------------------------------------------------------------- */
// IMPORTANT: we *MUST* pass "*oldcomm" to the Comm initializer here, as
//            the code below *requires* that the (implicit) copy constructor
//            for Comm is run and thus creating a shallow copy of "oldcomm".
//            The call to Comm::copy_arrays() then converts the shallow copy
//            into a deep copy of the class with the new layout.

CommBrick::CommBrick(LAMMPS * /*lmp*/, Comm *oldcomm) : Comm(*oldcomm)
{
  if (oldcomm->layout == Comm::LAYOUT_TILED)
    error->all(FLERR, "Cannot change to comm_style brick from tiled layout");

  style = 0;
  layout = oldcomm->layout;
  Comm::copy_arrays(oldcomm);
  init_buffers();
}

/* ----------------------------------------------------------------------
   initialize comm buffers and other data structs local to CommBrick
------------------------------------------------------------------------- */

void CommBrick::init_buffers()
{
  multilo = multihi = nullptr;
  cutghostmulti = nullptr;

  multioldlo = multioldhi = nullptr;
  cutghostmultiold = nullptr;

  buf_send = buf_recv = nullptr;

  maxsend = maxrecv = BUFMIN;
  CommBrick::grow_send(maxsend, 2);
  memory->create(buf_recv, maxrecv, "comm:buf_recv");

  nswap = 0;
  maxswap = 26;
  // TODO: technically this only needs to be 26, but need additional tweaks to make it work
  // maxswap = 36;
  CommBrick::allocate_swap(maxswap);

  sendlist = (int **) memory->smalloc(maxswap * sizeof(int *), "comm:sendlist");

  memory->create(maxsendlist, maxswap, "comm:maxsendlist");
  for (int i = 0; i < maxswap; i++) {
    maxsendlist[i] = BUFMIN;
    memory->create(sendlist[i], BUFMIN, "comm:sendlist[i]");
  }

  // stencil_md
  buf_recv_stencil_md = buf_send_stencil_md = nullptr;
  maxsend_stencil_md = maxrecv_stencil_md = nullptr;
  memory->create(maxsend_stencil_md, maxswap, "comm:maxsendlist");
  memory->create(maxrecv_stencil_md, maxswap, "comm:maxsendlist");

  for (int i = 0; i < maxswap; i++) {
    maxsend_stencil_md[i] = BUFMIN;
    maxrecv_stencil_md[i] = BUFMIN;
  }

  buf_send_stencil_md =
      (double **) memory->smalloc(maxswap * sizeof(double *), "comm:bufsend_stencil_md");
  buf_recv_stencil_md =
      (double **) memory->smalloc(maxswap * sizeof(double *), "comm:bufrecv_stencil_md");


  for (int i = 0; i < maxswap; i++) {
    memory->create(buf_send_stencil_md[i], maxsend_stencil_md[i] * NUM_PIPELINE_STAGES, "comm:buf_send_stencil_md");
    memory->create(buf_recv_stencil_md[i], maxrecv_stencil_md[i] * NUM_PIPELINE_STAGES, "comm:buf_recv_stencil_md");
  }

  for (int i = 0; i < NUM_TIMESTEPS_IN_PARALLEL + 1; i++) {
    sendlist_stencil_md[i] =
        (int **) memory->smalloc(maxswap * sizeof(int *), "comm:sendlist_stencil_md");
    send_force_stencil_md[i] =
        (bool **) memory->smalloc(maxswap * sizeof(bool *), "comm:send_force_stencil_md");
    send_vel_stencil_md[i] =
            (bool **) memory->smalloc(maxswap * sizeof(bool *), "comm:send_vel_stencil_md");
    send_pos_stencil_md[i] =
        (bool **) memory->smalloc(maxswap * sizeof(bool *), "comm:send_pos_stencil_md");

    memory->create(maxsendlist_stencil_md[i], maxswap, "comm:maxsendlist_stencil_md");

    sendlist_stencil_md_next_dt[i] =
        (int **) memory->smalloc(maxswap * sizeof(int *), "comm:sendlist_stencil_md");
    send_force_stencil_md_next_dt[i] =
        (bool **) memory->smalloc(maxswap * sizeof(bool *), "comm:send_force_stencil_md");
    send_vel_stencil_md_next_dt[i] =
        (bool **) memory->smalloc(maxswap * sizeof(bool *), "comm:send_vel_stencil_md");
    send_pos_stencil_md_next_dt[i] =
        (bool **) memory->smalloc(maxswap * sizeof(bool *), "comm:send_pos_stencil_md");

    memory->create(maxsendlist_stencil_md_next_dt[i], maxswap, "comm:maxsendlist_stencil_md");

    for (int j = 0; j < maxswap; j++) {
      maxsendlist_stencil_md[i][j] = BUFMIN;
      memory->create(sendlist_stencil_md[i][j], BUFMIN, "comm:sendlist_stencil_md[i]");
      memory->create(send_force_stencil_md[i][j], BUFMIN, "comm:send_force_stencil_md[i]");
      memory->create(send_vel_stencil_md[i][j], BUFMIN, "comm:send_vel_stencil_md[i]");
      memory->create(send_pos_stencil_md[i][j], BUFMIN, "comm:send_pos_stencil_md[i]");

      maxsendlist_stencil_md_next_dt[i][j] = BUFMIN;
      memory->create(sendlist_stencil_md_next_dt[i][j], BUFMIN, "comm:sendlist_stencil_md[i]");
      memory->create(send_force_stencil_md_next_dt[i][j], BUFMIN, "comm:send_force_stencil_md[i]");
      memory->create(send_vel_stencil_md_next_dt[i][j], BUFMIN, "comm:send_vel_stencil_md[i]");
      memory->create(send_pos_stencil_md_next_dt[i][j], BUFMIN, "comm:send_pos_stencil_md[i]");
    }

    second_sendlist_stencil_md[i] =
        (int **) memory->smalloc(maxswap * sizeof(int *), "comm:second_sendlist_stencil_md");
    memory->create(max_second_sendlist_stencil_md[i], maxswap,
                   "comm:second_maxsendlist_stencil_md");

    for (int j = 0; j < maxswap; j++) {
      max_second_sendlist_stencil_md[i][j] = BUFMIN;
      memory->create(second_sendlist_stencil_md[i][j], BUFMIN,
                     "comm:second_sendlist_stencil_md[i]");
    }

    second_sendlist_stencil_md_next_dt[i] =
            (int **) memory->smalloc(maxswap * sizeof(int *), "comm:second_sendlist_stencil_md_next_dt");
    memory->create(max_second_sendlist_stencil_md_next_dt[i], maxswap,
            "comm:second_maxsendlist_stencil_md");

    for (int j = 0; j < maxswap; j++) {
      max_second_sendlist_stencil_md_next_dt[i][j] = BUFMIN;
      memory->create(second_sendlist_stencil_md_next_dt[i][j], BUFMIN,
                     "comm:second_sendlist_stencil_md_next_dt[i]");
    }
  }

  maxsend_sendlist_stencil_md = maxrecv_sendlist_stencil_md = nullptr;
  memory->create(maxsend_sendlist_stencil_md, maxswap, "comm:maxsendlist");
  memory->create(maxrecv_sendlist_stencil_md, maxswap, "comm:maxsendlist");

  for (int i = 0; i < maxswap; i++) {
    maxsend_sendlist_stencil_md[i] = BUFMIN;
    maxrecv_sendlist_stencil_md[i] = BUFMIN;
  }

  buf_sendlist_stencil_md =
      (int **) memory->smalloc(maxswap * sizeof(int *), "comm:bufsend_stencil_md");
  buf_recv_sendlist_stencil_md =
      (int **) memory->smalloc(maxswap * sizeof(int *), "comm:bufsend_stencil_md");
  for (int i = 0; i < maxswap; i++) {
    memory->create(buf_sendlist_stencil_md[i], maxsend_sendlist_stencil_md[i],
                   "comm:buf_send_stencil_md");
    memory->create(buf_recv_sendlist_stencil_md[i], maxrecv_sendlist_stencil_md[i],
                   "comm:buf_send_stencil_md");
  }
}

/* ---------------------------------------------------------------------- */

void CommBrick::init()
{
  Comm::init();

  int bufextra_old = bufextra;
  init_exchange();
  if (bufextra > bufextra_old) grow_send(maxsend + bufextra, 2);

  // memory for multi style communication
  // allocate in setup

  if (mode == Comm::MULTI) {
    // If inconsitent # of collections, destroy any preexisting arrays (may be
    // missized)
    if (ncollections != neighbor->ncollections) {
      ncollections = neighbor->ncollections;
      if (multilo != nullptr) {
        free_multi();
        memory->destroy(cutghostmulti);
      }
    }

    // delete any old user cutoffs if # of collections chanaged
    if (cutusermulti && ncollections != ncollections_cutoff) {
      if (me == 0)
        error->warning(FLERR,
                       "cutoff/multi settings discarded, must be defined"
                       " after customizing collections in neigh_modify");
      memory->destroy(cutusermulti);
      cutusermulti = nullptr;
    }

    if (multilo == nullptr) {
      allocate_multi(maxswap);
      memory->create(cutghostmulti, ncollections, 3, "comm:cutghostmulti");
    }
  }
  if ((mode == Comm::SINGLE || mode == Comm::MULTIOLD) && multilo) {
    free_multi();
    memory->destroy(cutghostmulti);
  }

  // memory for multi/old-style communication

  if (mode == Comm::MULTIOLD && multioldlo == nullptr) {
    allocate_multiold(maxswap);
    memory->create(cutghostmultiold, atom->ntypes + 1, 3, "comm:cutghostmultiold");
  }
  if ((mode == Comm::SINGLE || mode == Comm::MULTI) && multioldlo) {
    free_multiold();
    memory->destroy(cutghostmultiold);
  }
}

/* ----------------------------------------------------------------------
   setup spatial-decomposition communication patterns
   function of neighbor cutoff(s) & cutghostuser & current box size
   single mode sets slab boundaries (slablo,slabhi) based on max cutoff
   multi mode sets collection-dependent slab boundaries (multilo,multihi)
   multi/old mode sets type-dependent slab boundaries (multioldlo,multioldhi)
------------------------------------------------------------------------- */

void CommBrick::setup()
{
  // cutghost[] = max distance at which ghost atoms need to be acquired
  // for orthogonal:
  //   cutghost is in box coords = neigh->cutghost in all 3 dims
  // for triclinic:
  //   neigh->cutghost = distance between tilted planes in box coords
  //   cutghost is in lamda coords = distance between those planes
  // for multi:
  //   cutghostmulti = same as cutghost, only for each atom collection
  // for multi/old:
  //   cutghostmultiold = same as cutghost, only for each atom type

  int i, j;
  int ntypes = atom->ntypes;
  double *prd, *sublo, *subhi;

  double cut = get_comm_cutoff();
  if ((cut == 0.0) && (me == 0))
    error->warning(FLERR,
                   "Communication cutoff is 0.0. No ghost atoms "
                   "will be generated. Atoms may get lost.");

  if (mode == Comm::MULTI) {
    double **cutcollectionsq = neighbor->cutcollectionsq;

    // build collection array for atom exchange
    neighbor->build_collection(0);

    // If using multi/reduce, communicate particles a distance equal
    // to the max cutoff with equally sized or smaller collections
    // If not, communicate the maximum cutoff of the entire collection
    for (i = 0; i < ncollections; i++) {
      if (cutusermulti) {
        cutghostmulti[i][0] = cutusermulti[i];
        cutghostmulti[i][1] = cutusermulti[i];
        cutghostmulti[i][2] = cutusermulti[i];
      } else {
        cutghostmulti[i][0] = 0.0;
        cutghostmulti[i][1] = 0.0;
        cutghostmulti[i][2] = 0.0;
      }

      for (j = 0; j < ncollections; j++) {
        if (multi_reduce && (cutcollectionsq[j][j] > cutcollectionsq[i][i])) continue;
        cutghostmulti[i][0] = MAX(cutghostmulti[i][0], sqrt(cutcollectionsq[i][j]));
        cutghostmulti[i][1] = MAX(cutghostmulti[i][1], sqrt(cutcollectionsq[i][j]));
        cutghostmulti[i][2] = MAX(cutghostmulti[i][2], sqrt(cutcollectionsq[i][j]));
      }
    }
  }

  if (mode == Comm::MULTIOLD) {
    double *cuttype = neighbor->cuttype;
    for (i = 1; i <= ntypes; i++) {
      double tmp = 0.0;
      if (cutusermultiold) tmp = cutusermultiold[i];
      cutghostmultiold[i][0] = MAX(tmp, cuttype[i]);
      cutghostmultiold[i][1] = MAX(tmp, cuttype[i]);
      cutghostmultiold[i][2] = MAX(tmp, cuttype[i]);
    }
  }

  if (triclinic == 0) {
    prd = domain->prd;
    sublo = domain->sublo;
    subhi = domain->subhi;
    cutghost[0] = cutghost[1] = cutghost[2] = cut;
  } else {
    prd = domain->prd_lamda;
    sublo = domain->sublo_lamda;
    subhi = domain->subhi_lamda;
    double *h_inv = domain->h_inv;
    double length0, length1, length2;
    length0 = sqrt(h_inv[0] * h_inv[0] + h_inv[5] * h_inv[5] + h_inv[4] * h_inv[4]);
    cutghost[0] = cut * length0;
    length1 = sqrt(h_inv[1] * h_inv[1] + h_inv[3] * h_inv[3]);
    cutghost[1] = cut * length1;
    length2 = h_inv[2];
    cutghost[2] = cut * length2;
    if (mode == Comm::MULTI) {
      for (i = 0; i < ncollections; i++) {
        cutghostmulti[i][0] *= length0;
        cutghostmulti[i][1] *= length1;
        cutghostmulti[i][2] *= length2;
      }
    }

    if (mode == Comm::MULTIOLD) {
      for (i = 1; i <= ntypes; i++) {
        cutghostmultiold[i][0] *= length0;
        cutghostmultiold[i][1] *= length1;
        cutghostmultiold[i][2] *= length2;
      }
    }
  }

  // recvneed[idim][0/1] = # of procs away I recv atoms from, within cutghost
  //   0 = from left, 1 = from right
  //   do not cross non-periodic boundaries, need[2] = 0 for 2d
  // sendneed[idim][0/1] = # of procs away I send atoms to
  //   0 = to left, 1 = to right
  //   set equal to recvneed[idim][1/0] of neighbor proc
  // maxneed[idim] = max procs away any proc recvs atoms in either direction
  // layout = UNIFORM = uniform sized sub-domains:
  //   maxneed is directly computable from sub-domain size
  //     limit to procgrid-1 for non-PBC
  //   recvneed = maxneed except for procs near non-PBC
  //   sendneed = recvneed of neighbor on each side
  // layout = NONUNIFORM = non-uniform sized sub-domains:
  //   compute recvneed via updown() which accounts for non-PBC
  //   sendneed = recvneed of neighbor on each side
  //   maxneed via Allreduce() of recvneed

  int *periodicity = domain->periodicity;
  int left, right;

  if (layout == Comm::LAYOUT_UNIFORM) {
    maxneed[0] = static_cast<int>(cutghost[0] * procgrid[0] / prd[0]) + 1;
    maxneed[1] = static_cast<int>(cutghost[1] * procgrid[1] / prd[1]) + 1;
    maxneed[2] = static_cast<int>(cutghost[2] * procgrid[2] / prd[2]) + 1;
    if (domain->dimension == 2) maxneed[2] = 0;
    if (!periodicity[0]) maxneed[0] = MIN(maxneed[0], procgrid[0] - 1);
    if (!periodicity[1]) maxneed[1] = MIN(maxneed[1], procgrid[1] - 1);
    if (!periodicity[2]) maxneed[2] = MIN(maxneed[2], procgrid[2] - 1);

    if (!periodicity[0]) {
      recvneed[0][0] = MIN(maxneed[0], myloc[0]);
      recvneed[0][1] = MIN(maxneed[0], procgrid[0] - myloc[0] - 1);
      left = myloc[0] - 1;
      if (left < 0) left = procgrid[0] - 1;
      sendneed[0][0] = MIN(maxneed[0], procgrid[0] - left - 1);
      right = myloc[0] + 1;
      if (right == procgrid[0]) right = 0;
      sendneed[0][1] = MIN(maxneed[0], right);
    } else
      recvneed[0][0] = recvneed[0][1] = sendneed[0][0] = sendneed[0][1] = maxneed[0];

    if (!periodicity[1]) {
      recvneed[1][0] = MIN(maxneed[1], myloc[1]);
      recvneed[1][1] = MIN(maxneed[1], procgrid[1] - myloc[1] - 1);
      left = myloc[1] - 1;
      if (left < 0) left = procgrid[1] - 1;
      sendneed[1][0] = MIN(maxneed[1], procgrid[1] - left - 1);
      right = myloc[1] + 1;
      if (right == procgrid[1]) right = 0;
      sendneed[1][1] = MIN(maxneed[1], right);
    } else
      recvneed[1][0] = recvneed[1][1] = sendneed[1][0] = sendneed[1][1] = maxneed[1];

    if (!periodicity[2]) {
      recvneed[2][0] = MIN(maxneed[2], myloc[2]);
      recvneed[2][1] = MIN(maxneed[2], procgrid[2] - myloc[2] - 1);
      left = myloc[2] - 1;
      if (left < 0) left = procgrid[2] - 1;
      sendneed[2][0] = MIN(maxneed[2], procgrid[2] - left - 1);
      right = myloc[2] + 1;
      if (right == procgrid[2]) right = 0;
      sendneed[2][1] = MIN(maxneed[2], right);
    } else
      recvneed[2][0] = recvneed[2][1] = sendneed[2][0] = sendneed[2][1] = maxneed[2];

  } else {
    recvneed[0][0] = updown(0, 0, myloc[0], prd[0], periodicity[0], xsplit);
    recvneed[0][1] = updown(0, 1, myloc[0], prd[0], periodicity[0], xsplit);
    left = myloc[0] - 1;
    if (left < 0) left = procgrid[0] - 1;
    sendneed[0][0] = updown(0, 1, left, prd[0], periodicity[0], xsplit);
    right = myloc[0] + 1;
    if (right == procgrid[0]) right = 0;
    sendneed[0][1] = updown(0, 0, right, prd[0], periodicity[0], xsplit);

    recvneed[1][0] = updown(1, 0, myloc[1], prd[1], periodicity[1], ysplit);
    recvneed[1][1] = updown(1, 1, myloc[1], prd[1], periodicity[1], ysplit);
    left = myloc[1] - 1;
    if (left < 0) left = procgrid[1] - 1;
    sendneed[1][0] = updown(1, 1, left, prd[1], periodicity[1], ysplit);
    right = myloc[1] + 1;
    if (right == procgrid[1]) right = 0;
    sendneed[1][1] = updown(1, 0, right, prd[1], periodicity[1], ysplit);

    if (domain->dimension == 3) {
      recvneed[2][0] = updown(2, 0, myloc[2], prd[2], periodicity[2], zsplit);
      recvneed[2][1] = updown(2, 1, myloc[2], prd[2], periodicity[2], zsplit);
      left = myloc[2] - 1;
      if (left < 0) left = procgrid[2] - 1;
      sendneed[2][0] = updown(2, 1, left, prd[2], periodicity[2], zsplit);
      right = myloc[2] + 1;
      if (right == procgrid[2]) right = 0;
      sendneed[2][1] = updown(2, 0, right, prd[2], periodicity[2], zsplit);
    } else
      recvneed[2][0] = recvneed[2][1] = sendneed[2][0] = sendneed[2][1] = 0;

    int all[6];
    MPI_Allreduce(&recvneed[0][0], all, 6, MPI_INT, MPI_MAX, world);
    maxneed[0] = MAX(all[0], all[1]);
    maxneed[1] = MAX(all[2], all[3]);
    maxneed[2] = MAX(all[4], all[5]);
  }

  // allocate comm memory

  nswap = 2 * (maxneed[0] + maxneed[1] + maxneed[2]);
  if (nswap > maxswap) grow_swap(nswap);

  // setup parameters for each exchange:
  // sendproc = proc to send to at each swap
  // recvproc = proc to recv from at each swap
  // for mode SINGLE:
  //   slablo/slabhi = boundaries for slab of atoms to send at each swap
  //   use -BIG/midpt/BIG to insure all atoms included even if round-off occurs
  //   if round-off, atoms recvd across PBC can be < or > than subbox boundary
  //   note that borders() only loops over subset of atoms during each swap
  //   treat all as PBC here, non-PBC is handled in borders() via r/s need[][]
  // for mode MULTI:
  //   multilo/multihi is same, with slablo/slabhi for each atom type
  // pbc_flag: 0 = nothing across a boundary, 1 = something across a boundary
  // pbc = -1/0/1 for PBC factor in each of 3/6 orthogonal/triclinic dirs
  // for triclinic, slablo/hi and pbc_border will be used in lamda (0-1) coords
  // 1st part of if statement is sending to the west/south/down
  // 2nd part of if statement is sending to the east/north/up

  int dim, ineed;

  int iswap = 0;
  for (dim = 0; dim < 3; dim++) {
    for (ineed = 0; ineed < 2 * maxneed[dim]; ineed++) {
      pbc_flag[iswap] = 0;
      pbc[iswap][0] = pbc[iswap][1] = pbc[iswap][2] = pbc[iswap][3] = pbc[iswap][4] =
          pbc[iswap][5] = 0;

      if (ineed % 2 == 0) {
        sendproc[iswap] = procneigh[dim][0];
        recvproc[iswap] = procneigh[dim][1];
        if (mode == Comm::SINGLE) {
          if (ineed < 2)
            slablo[iswap] = -BIG;
          else
            slablo[iswap] = 0.5 * (sublo[dim] + subhi[dim]);
          slabhi[iswap] = sublo[dim] + cutghost[dim];
        } else if (mode == Comm::MULTI) {
          for (i = 0; i < ncollections; i++) {
            if (ineed < 2)
              multilo[iswap][i] = -BIG;
            else
              multilo[iswap][i] = 0.5 * (sublo[dim] + subhi[dim]);
            multihi[iswap][i] = sublo[dim] + cutghostmulti[i][dim];
          }
        } else {
          for (i = 1; i <= ntypes; i++) {
            if (ineed < 2)
              multioldlo[iswap][i] = -BIG;
            else
              multioldlo[iswap][i] = 0.5 * (sublo[dim] + subhi[dim]);
            multioldhi[iswap][i] = sublo[dim] + cutghostmultiold[i][dim];
          }
        }
        if (myloc[dim] == 0) {
          pbc_flag[iswap] = 1;
          pbc[iswap][dim] = 1;
          if (triclinic) {
            if (dim == 1)
              pbc[iswap][5] = 1;
            else if (dim == 2)
              pbc[iswap][4] = pbc[iswap][3] = 1;
          }
        }

      } else {
        sendproc[iswap] = procneigh[dim][1];
        recvproc[iswap] = procneigh[dim][0];
        if (mode == Comm::SINGLE) {
          slablo[iswap] = subhi[dim] - cutghost[dim];
          if (ineed < 2)
            slabhi[iswap] = BIG;
          else
            slabhi[iswap] = 0.5 * (sublo[dim] + subhi[dim]);
        } else if (mode == Comm::MULTI) {
          for (i = 0; i < ncollections; i++) {
            multilo[iswap][i] = subhi[dim] - cutghostmulti[i][dim];
            if (ineed < 2)
              multihi[iswap][i] = BIG;
            else
              multihi[iswap][i] = 0.5 * (sublo[dim] + subhi[dim]);
          }
        } else {
          for (i = 1; i <= ntypes; i++) {
            multioldlo[iswap][i] = subhi[dim] - cutghostmultiold[i][dim];
            if (ineed < 2)
              multioldhi[iswap][i] = BIG;
            else
              multioldhi[iswap][i] = 0.5 * (sublo[dim] + subhi[dim]);
          }
        }
        if (myloc[dim] == procgrid[dim] - 1) {
          pbc_flag[iswap] = 1;
          pbc[iswap][dim] = -1;
          if (triclinic) {
            if (dim == 1)
              pbc[iswap][5] = -1;
            else if (dim == 2)
              pbc[iswap][4] = pbc[iswap][3] = -1;
          }
        }
      }

      iswap++;
    }
  }
}

/* ----------------------------------------------------------------------
   walk up/down the extent of nearby processors in dim and dir
   loc = myloc of proc to start at
   dir = 0/1 = walk to left/right
   do not cross non-periodic boundaries
   is not called for z dim in 2d
   return how many procs away are needed to encompass cutghost away from loc
------------------------------------------------------------------------- */

int CommBrick::updown(int dim, int dir, int loc, double prd, int periodicity, double *split)
{
  int index, count;
  double frac, delta;

  if (dir == 0) {
    frac = cutghost[dim] / prd;
    index = loc - 1;
    delta = 0.0;
    count = 0;
    while (delta < frac) {
      if (index < 0) {
        if (!periodicity) break;
        index = procgrid[dim] - 1;
      }
      count++;
      delta += split[index + 1] - split[index];
      index--;
    }

  } else {
    frac = cutghost[dim] / prd;
    index = loc + 1;
    delta = 0.0;
    count = 0;
    while (delta < frac) {
      if (index >= procgrid[dim]) {
        if (!periodicity) break;
        index = 0;
      }
      count++;
      delta += split[index + 1] - split[index];
      index++;
    }
  }

  return count;
}

/* ----------------------------------------------------------------------
   forward communication of atom coords every timestep
   other per-atom attributes may also be sent via pack/unpack routines
------------------------------------------------------------------------- */

void CommBrick::forward_comm(int /*dummy*/)
{
  int n;
  MPI_Request request;
  AtomVec *avec = atom->avec;
  double **x = atom->x;
  double *buf;

  // exchange data with another proc
  // if other proc is self, just copy
  // if comm_x_only set, exchange or copy directly to x, don't unpack

  // comm_x_only means only the position is needed for the pair which is the
  // case for allegro, do not need velocity firstrecv is the position in which
  // received atoms can be added, x is a double**, first dimension is n_local +
  // n_ghost + ___ (we fill in the last ___)
  //

  int num_recv_mpi = 0;

  for (int iswap = 0; iswap < nswap; iswap++) {
    if (sendproc[iswap] != me) {
      if (comm_x_only) {
        if (size_forward_recv[iswap]) {
          buf = x[firstrecv[iswap]];
          MPI_Irecv(buf, size_forward_recv[iswap], MPI_DOUBLE, recvproc[iswap], 0, world, &request);
        }
        n = avec->pack_comm(sendnum[iswap], sendlist[iswap], buf_send, pbc_flag[iswap], pbc[iswap]);
        if (n) MPI_Send(buf_send, n, MPI_DOUBLE, sendproc[iswap], 0, world);
        if (size_forward_recv[iswap]) MPI_Wait(&request, MPI_STATUS_IGNORE);
        num_recv_mpi += size_forward_recv[iswap];
      } else if (ghost_velocity) {
        assert(false);
        if (size_forward_recv[iswap])
          MPI_Irecv(buf_recv, size_forward_recv[iswap], MPI_DOUBLE, recvproc[iswap], 0, world,
                    &request);
        n = avec->pack_comm_vel(sendnum[iswap], sendlist[iswap], buf_send, pbc_flag[iswap],
                                pbc[iswap]);
        if (n) MPI_Send(buf_send, n, MPI_DOUBLE, sendproc[iswap], 0, world);
        if (size_forward_recv[iswap]) MPI_Wait(&request, MPI_STATUS_IGNORE);
        avec->unpack_comm_vel(recvnum[iswap], firstrecv[iswap], buf_recv);
        num_recv_mpi += size_forward_recv[iswap];
      } else {
        assert(false);
        if (size_forward_recv[iswap])
          MPI_Irecv(buf_recv, size_forward_recv[iswap], MPI_DOUBLE, recvproc[iswap], 0, world,
                    &request);
        n = avec->pack_comm(sendnum[iswap], sendlist[iswap], buf_send, pbc_flag[iswap], pbc[iswap]);
        if (n) MPI_Send(buf_send, n, MPI_DOUBLE, sendproc[iswap], 0, world);
        if (size_forward_recv[iswap]) MPI_Wait(&request, MPI_STATUS_IGNORE);
        avec->unpack_comm(recvnum[iswap], firstrecv[iswap], buf_recv);
      }

    } else {
      if (comm_x_only) {
        if (sendnum[iswap])
          avec->pack_comm(sendnum[iswap], sendlist[iswap], x[firstrecv[iswap]], pbc_flag[iswap],
                          pbc[iswap]);
      } else if (ghost_velocity) {
        avec->pack_comm_vel(sendnum[iswap], sendlist[iswap], buf_send, pbc_flag[iswap], pbc[iswap]);
        avec->unpack_comm_vel(recvnum[iswap], firstrecv[iswap], buf_send);
      } else {
        avec->pack_comm(sendnum[iswap], sendlist[iswap], buf_send, pbc_flag[iswap], pbc[iswap]);
        avec->unpack_comm(recvnum[iswap], firstrecv[iswap], buf_send);
      }
    }
  }

  if (TRACK_LAMMPS_DATA) {
      MPI_Allreduce(MPI_IN_PLACE, &num_recv_mpi, 1, MPI_INT, MPI_SUM, world);
      if (comm->me == 0) {
          std::cout << "num recv forward: " << num_recv_mpi << std::endl;
      }
  }
}

/* ----------------------------------------------------------------------
   reverse communication of forces on atoms every timestep
   other per-atom attributes may also be sent via pack/unpack routines
------------------------------------------------------------------------- */

void CommBrick::reverse_comm()
{
  int n;
  MPI_Request request;
  AtomVec *avec = atom->avec;
  double **f = atom->f;
  double *buf;

  // exchange data with another proc
  // if other proc is self, just copy
  // if comm_f_only set, exchange or copy directly from f, don't pack

  /*
  int num_no_force = 0;
  for (int k = atom->nlocal; k < atom->nlocal + atom->nghost; k++) {
      if (fabs(atom->f[k][0]) <= 1e-6) {
          num_no_force++;
      }
  }
  */

  // std::cout << "me: " << comm->me << " lammps no force. num: " << num_no_force << " out of: " << atom->nghost << std::endl;
  int num_recv_mpi = 0;

  for (int iswap = nswap - 1; iswap >= 0; iswap--) {
    if (sendproc[iswap] != me) {
      if (comm_f_only) {
        if (size_reverse_recv[iswap])
          MPI_Irecv(buf_recv, size_reverse_recv[iswap], MPI_DOUBLE, sendproc[iswap], 0, world,
                    &request);
        if (size_reverse_send[iswap]) {
          buf = f[firstrecv[iswap]];
          MPI_Send(buf, size_reverse_send[iswap], MPI_DOUBLE, recvproc[iswap], 0, world);
        }
        if (size_reverse_recv[iswap]) MPI_Wait(&request, MPI_STATUS_IGNORE);
        num_recv_mpi += size_reverse_recv[iswap];
        // std::cout << GREEN << "lammps reverse comm num recv " << size_reverse_recv[iswap] << RESET_COLOR << std::endl;
      } else {
        if (size_reverse_recv[iswap])
          MPI_Irecv(buf_recv, size_reverse_recv[iswap], MPI_DOUBLE, sendproc[iswap], 0, world,
                    &request);
        n = avec->pack_reverse(recvnum[iswap], firstrecv[iswap], buf_send);
        if (n) MPI_Send(buf_send, n, MPI_DOUBLE, recvproc[iswap], 0, world);
        if (size_reverse_recv[iswap]) MPI_Wait(&request, MPI_STATUS_IGNORE);
      }
      avec->unpack_reverse(sendnum[iswap], sendlist[iswap], buf_recv);
    } else {
      if (comm_f_only) {
        if (sendnum[iswap])
          avec->unpack_reverse(sendnum[iswap], sendlist[iswap], f[firstrecv[iswap]]);
      } else {
        avec->pack_reverse(recvnum[iswap], firstrecv[iswap], buf_send);
        avec->unpack_reverse(sendnum[iswap], sendlist[iswap], buf_send);
      }
    }
  }

  if (TRACK_LAMMPS_DATA) {
      MPI_Allreduce(MPI_IN_PLACE, &num_recv_mpi, 1, MPI_INT, MPI_SUM, world);
      if (comm->me == 0) {
          std::cout << "num send reverse: " << num_recv_mpi << std::endl;
      }
  }
}

/* ----------------------------------------------------------------------
   exchange: move atoms to correct processors
   atoms exchanged with all 6 stencil neighbors
   send out atoms that have left my box, receive ones entering my box
   atoms will be lost if not inside a stencil proc's box
     can happen if atom moves outside of non-periodic boundary
     or if atom moves more than one proc away
   this routine called before every reneighboring
   for triclinic, atoms must be in lamda coords (0-1) before exchange is called
------------------------------------------------------------------------- */

void CommBrick::exchange()
{
  int i, m, nsend, nrecv, nrecv1, nrecv2, nlocal;
  double lo, hi, value;
  double **x;
  double *sublo, *subhi;
  MPI_Request request;
  AtomVec *avec = atom->avec;

  // clear global->local map for owned and ghost atoms
  // b/c atoms migrate to new procs in exchange() and
  //   new ghosts are created in borders()
  // map_set() is done at end of borders()
  // clear ghost count and any ghost bonus data internal to AtomVec

  if (map_style != Atom::MAP_NONE) atom->map_clear();
  atom->nghost = 0;
  atom->avec->clear_bonus();

  // insure send buf has extra space for a single atom
  // only need to reset if a fix can dynamically add to size of single atom

  if (maxexchange_fix_dynamic) {
    assert(false);
    int bufextra_old = bufextra;
    init_exchange();
    if (bufextra > bufextra_old) grow_send(maxsend + bufextra, 2);
  }

  // subbox bounds for orthogonal or triclinic

  if (triclinic == 0) {
    sublo = domain->sublo;
    subhi = domain->subhi;
  } else {
    sublo = domain->sublo_lamda;
    subhi = domain->subhi_lamda;
  }

  // loop over dimensions

  int dimension = domain->dimension;

  for (int dim = 0; dim < dimension; dim++) {
    // fill buffer with atoms leaving my box, using < and >=
    // when atom is deleted, fill it in with last atom

    x = atom->x;
    lo = sublo[dim];
    hi = subhi[dim];
    nlocal = atom->nlocal;
    i = nsend = 0;

    while (i < nlocal) {
      if (x[i][dim] < lo || x[i][dim] >= hi) {
        if (nsend > maxsend) grow_send(nsend, 1);
        nsend += avec->pack_exchange(i, &buf_send[nsend]);
        avec->copy(nlocal - 1, i, 1);
        nlocal--;
      } else
        i++;
    }
    atom->nlocal = nlocal;

    // send/recv atoms in both directions
    // send size of message first so receiver can realloc buf_recv if needed
    // if 1 proc in dimension, no send/recv
    //   set nrecv = 0 so buf_send atoms will be lost
    // if 2 procs in dimension, single send/recv
    // if more than 2 procs in dimension, send/recv to both neighbors

    if (procgrid[dim] == 1)
      nrecv = 0;
    else {
      MPI_Sendrecv(&nsend, 1, MPI_INT, procneigh[dim][0], 0, &nrecv1, 1, MPI_INT, procneigh[dim][1],
                   0, world, MPI_STATUS_IGNORE);
      nrecv = nrecv1;
      if (procgrid[dim] > 2) {
        MPI_Sendrecv(&nsend, 1, MPI_INT, procneigh[dim][1], 0, &nrecv2, 1, MPI_INT,
                     procneigh[dim][0], 0, world, MPI_STATUS_IGNORE);
        nrecv += nrecv2;
      }
      if (nrecv > maxrecv) grow_recv(nrecv);

      MPI_Irecv(buf_recv, nrecv1, MPI_DOUBLE, procneigh[dim][1], 0, world, &request);
      MPI_Send(buf_send, nsend, MPI_DOUBLE, procneigh[dim][0], 0, world);
      MPI_Wait(&request, MPI_STATUS_IGNORE);

      if (procgrid[dim] > 2) {
        MPI_Irecv(&buf_recv[nrecv1], nrecv2, MPI_DOUBLE, procneigh[dim][0], 0, world, &request);
        MPI_Send(buf_send, nsend, MPI_DOUBLE, procneigh[dim][1], 0, world);
        MPI_Wait(&request, MPI_STATUS_IGNORE);
      }
    }

    // check incoming atoms to see if they are in my box
    // if so, add to my list
    // box check is only for this dimension,
    //   atom may be passed to another proc in later dims

    m = 0;
    while (m < nrecv) {
      value = buf_recv[m + dim + 1];
      if (value >= lo && value < hi) {
        m += avec->unpack_exchange(&buf_recv[m]);
      } else {
        m += static_cast<int>(buf_recv[m]);
      }
    }
  }

  if (atom->firstgroupname) {
      atom->first_reorder();
  }
}

// send the data from the current process to the domains created by stencil_md
// calls the "lammps" version of atom and comm and clears everything out,
// migrates all the shit over to "stencil_md" send to process that owns the zoid
void CommBrick::exchange_stencil_md_initial_send(std::vector<MPI_Request>& r)
{
  int i, m, nsend, nrecv, nrecv1, nrecv2, nlocal;
  double lo, hi, value;
  double **x;
  double *sublo, *subhi;
  AtomVec *avec = atom->avec;

  // clear global->local map for owned and ghost atoms
  // b/c atoms migrate to new procs in exchange() and
  //   new ghosts are created in borders()
  // map_set() is done at end of borders()
  // clear ghost count and any ghost bonus data internal to AtomVec

  if (map_style != Atom::MAP_NONE) {
      // atom->map_clear();
  }
  // atom->nghost = 0;
  // atom->avec->clear_bonus();

  // insure send buf has extra space for a single atom
  // only need to reset if a fix can dynamically add to size of single atom

  if (maxexchange_fix_dynamic) {
    assert(false);
    int bufextra_old = bufextra;
    init_exchange();
    if (bufextra > bufextra_old) grow_send(maxsend + bufextra, 2);
  }

  // fill buffer with atoms leaving my box, using < and >=
  // when atom is deleted, fill it in with last atom

  x = atom->x;
  nlocal = atom->nlocal;
  i = nsend = 0;

  // send out all atoms from the main process, super inefficient but it's the
  // initial step so do whatever is necessary
  while (i < nlocal) {
    if (nsend > maxsend) {
        grow_send(nsend, 1);
    }
    int tmp = avec->pack_exchange(i, &buf_send[nsend]);
    nsend += tmp;
    // std::cout << "i: " << i << " out of: " << nlocal << " nsend: " << nsend << " maxsend: " << maxsend << " tmp: " << tmp << " buf send: " << buf_send << std::endl;
    // avec->copy(nlocal-1,i,1);
    // nlocal--;
    i++;
  }

  stencil_md_initial_exchange_nsend = nsend;

  // atom->nlocal = nlocal;
  // send atoms to zoids from dep level 0 to 4.
  std::vector<MPI_Request> requests;
  int idx = 0;
  for (int dep = 0; dep < NUM_DEPS; dep++) {
    for (int j = 0; j < lmp->queues[dep].size(); j++) {
      queue_info &zoid = lmp->queues[dep][j];
      int zoid_num = zoid.num;
      MPI_Isend(&stencil_md_initial_exchange_nsend, 1, MPI_INT, zoid_num % comm->nprocs, zoid_num, world, &r[2 * idx]);
      MPI_Isend(buf_send, stencil_md_initial_exchange_nsend, MPI_DOUBLE, zoid_num % comm->nprocs, zoid_num, world, &r[2 * idx + 1]);
      idx++;
    }
  }

  if (atom->firstgroupname) {
    assert(false);
    atom->first_reorder();
  }
}

void CommBrick::borders_stencil_md_initial_send(std::vector<MPI_Request>& r) {
    int i, m, nsend, nrecv, nrecv1, nrecv2, nlocal;
    double lo, hi, value;
    double **x;
    double *sublo, *subhi;
    AtomVec *avec = atom->avec;

    // clear global->local map for owned and ghost atoms
    // b/c atoms migrate to new procs in exchange() and
    //   new ghosts are created in borders()
    // map_set() is done at end of borders()
    // clear ghost count and any ghost bonus data internal to AtomVec

    if (map_style != Atom::MAP_NONE) {
        // atom->map_clear();
    }
    // atom->nghost = 0;
    // atom->avec->clear_bonus();

    // insure send buf has extra space for a single atom
    // only need to reset if a fix can dynamically add to size of single atom

    if (maxexchange_fix_dynamic) {
        assert(false);
        int bufextra_old = bufextra;
        init_exchange();
        if (bufextra > bufextra_old) grow_send(maxsend + bufextra, 2);
    }

    // fill buffer with atoms leaving my box, using < and >=
    // when atom is deleted, fill it in with last atom

    x = atom->x;
    nlocal = atom->nlocal;
    i = nsend = 0;

    // send out all atoms from the main process, super inefficient but it's the
    // initial step so do whatever is necessary
    while (i < nlocal) {
        if (nsend > maxsend) {
            grow_send(nsend, 1);
        }
        // int tmp = avec->pack_border_stencil_md(i, &buf_send[nsend]);
        int tmp = avec->pack_exchange(i, &buf_send[nsend]);
        nsend += tmp;
        // std::cout << "i: " << i << " out of: " << nlocal << " nsend: " << nsend << " maxsend: " << maxsend << " tmp: " << tmp << " buf send: " << buf_send << std::endl;
        // avec->copy(nlocal-1,i,1);
        // nlocal--;
        i++;
    }

    stencil_md_initial_exchange_nsend = nsend;

    // atom->nlocal = nlocal;
    // send atoms to zoids from dep level 0 to 4.
    std::vector<MPI_Request> requests;
    int idx = 0;
    for (int dep = 0; dep < NUM_DEPS; dep++) {
        for (int j = 0; j < lmp->queues[dep].size(); j++) {
            queue_info &zoid = lmp->queues[dep][j];
            int zoid_num = zoid.num;
            MPI_Isend(&stencil_md_initial_exchange_nsend, 1, MPI_INT, zoid_num % comm->nprocs, zoid_num, world, &r[2 * idx]);
            MPI_Isend(buf_send, stencil_md_initial_exchange_nsend, MPI_DOUBLE, zoid_num % comm->nprocs, zoid_num, world, &r[2 * idx + 1]);
            idx++;
        }
    }

    if (atom->firstgroupname) {
        assert(false);
        atom->first_reorder();
    }
}

// send the data from the current process to the domains created by stencil_md
// calls the "lammps" version of atom and comm and clears everything out,
// migrates all the shit over to "stencil_md" send to process that owns the zoid
void CommBrick::exchange_stencil_md_initial_send_to_dep0() {
    int i, m, nsend, nrecv, nrecv1, nrecv2, nlocal;
    double lo, hi, value;
    double **x;
    double *sublo, *subhi;
    MPI_Request request;
    AtomVec *avec = atom->avec;

    // clear global->local map for owned and ghost atoms
    // b/c atoms migrate to new procs in exchange() and
    //   new ghosts are created in borders()
    // map_set() is done at end of borders()
    // clear ghost count and any ghost bonus data internal to AtomVec

    if (map_style != Atom::MAP_NONE) atom->map_clear();
    // atom->nghost = 0;
    // atom->avec->clear_bonus();

    // insure send buf has extra space for a single atom
    // only need to reset if a fix can dynamically add to size of single atom

    if (maxexchange_fix_dynamic) {
        int bufextra_old = bufextra;
        init_exchange();
        if (bufextra > bufextra_old) grow_send(maxsend + bufextra, 2);
    }

    // fill buffer with atoms leaving my box, using < and >=
    // when atom is deleted, fill it in with last atom

    x = atom->x;
    nlocal = atom->nlocal;
    i = nsend = 0;

    // send out all atoms from the main process, super inefficient but it's the
    // initial step so do whatever is necessary
    while (i < nlocal) {
        if (nsend > maxsend) grow_send(nsend, 1);
        nsend += avec->pack_exchange(i, &buf_send[nsend]);
        // avec->copy(nlocal-1,i,1);
        // nlocal--;
        i++;
    }
    // atom->nlocal = nlocal;
    // send atoms to zoids from dep level 0 to 4.
    std::vector<MPI_Request> requests;
    int dep = 0;
    for (int j = 0; j < lmp->queues[dep].size(); j++) {
        queue_info &zoid = lmp->queues[dep][j];
        int zoid_num = zoid.num;
        MPI_Request r1;
        MPI_Request r2;
        MPI_Isend(&nsend, 1, MPI_INT, zoid_num % comm->nprocs, zoid_num, world, &r1);
        MPI_Isend(buf_send, nsend, MPI_DOUBLE, zoid_num % comm->nprocs, zoid_num, world, &r2);
        requests.push_back(r1);
        requests.push_back(r2);
    }

    if (atom->firstgroupname) {
        assert(false);
        atom->first_reorder();
    }
}

void CommBrick::borders_stencil_md_initial_receive_from_lammps(Atom *atom_, Domain *domain_, queue_info &zoid, int timestep) {
    AtomVec *avec = atom_->avec;

    int dimension = domain_->dimension;
    assert(dimension == 3);

    double **x = atom_->x;

    std::unordered_set<tagint> tags_set;
    for (int i = 0; i < atom_->nlocal; i++) {
        tags_set.insert(atom_->tag[i]);
    }

    int zoid_num = zoid.num;
    for (int i = 0; i < comm->nprocs; i++) {
        int nrecv;
        MPI_Recv(&nrecv, 1, MPI_INT, i, zoid_num, world, MPI_STATUS_IGNORE);
        if (nrecv * size_border > maxrecv) { grow_recv(nrecv * size_border); }
        MPI_Recv(buf_recv, nrecv, MPI_DOUBLE, i, zoid_num, world, MPI_STATUS_IGNORE);
        int m = 0;

        while (m < nrecv) {
            bool borders_zoid = true;
            bool in_zoid = true;

            double ghost_pos[3] = {0};

            for (int dim = 0; dim < domain->dimension; dim++) {
                double lo = zoid.zoid.cuts[dim].lower + timestep * zoid.zoid.cuts[dim].slope_lower;
                double hi = zoid.zoid.cuts[dim].upper + timestep * zoid.zoid.cuts[dim].slope_upper;

                double lo_borders = lo - ALLEGRO_SLOPE;
                double hi_borders = hi + ALLEGRO_SLOPE;

                double value = buf_recv[m + dim + 1];
                double value_borders = buf_recv[m + dim + 1];

                // std::cout << "m: " << m << " value: " << value << " lo: " << lo << " hi: " << hi << " prd: " << domain->prd[dim] << std::endl;

                while (value < lo) { value += domain->prd[dim]; }
                while (value >= hi) { value -= domain->prd[dim]; }

                while (value_borders < lo_borders) { value_borders += domain->prd[dim]; }
                while (value_borders >= hi_borders) { value_borders -= domain->prd[dim]; }

                in_zoid = in_zoid && value >= lo && value < hi;
                borders_zoid = borders_zoid && value_borders >= lo_borders && value_borders < hi_borders;

                ghost_pos[dim] = value_borders;
            }

            tagint tag_ = (tagint) ubuf(buf_recv[m + 6 + 1]).i;
            assert(tag_ >= 0 && tag_ <= atom->natoms);
            if (tags_set.find(tag_) != tags_set.end()) {
                m += static_cast<int>(buf_recv[m]);
                continue;
            }
            tags_set.insert(tag_);

            auto begin = std::chrono::high_resolution_clock::now();

            if (borders_zoid && !in_zoid) {
                buf_recv[m + 0 + 1] = ghost_pos[0];
                buf_recv[m + 1 + 1] = ghost_pos[1];
                buf_recv[m + 2 + 1] = ghost_pos[2];
                // m += atom_->avec->unpack_border_stencil_md(&buf_recv[m], atom_);
                // m += atom_->avec->unpack_border_stencil_md(&buf_recv[m], atom_);
                m += atom_->avec->unpack_exchange_stencil_md(&buf_recv[m], atom_, domain_, LAMMPS_SEND_GHOST);
                // atom_->avec->unpack_border_stencil_md(nrecv, atom->nlocal + atom->nghost, buf_recv, atom_);
            } else {
                if (borders_zoid) {
                    assert(in_zoid);
                }
                m += static_cast<int>(buf_recv[m]);
            }
        }
    }

    // domain_->remap_all_stencil_md(atom_);
    if (atom_->firstgroupname) {
        assert(false);
        atom_->first_reorder();
    }

    // reset global->local map
    if (map_style != Atom::MAP_NONE) {
        atom_->map_init_stencil_md();
        atom_->map_set();
        // atom->map_set();
    }
}

void CommBrick::exchange_stencil_md_initial_receive(Atom *atom_, Domain *domain_, queue_info &zoid)
{
  AtomVec *avec = atom_->avec;
  if (map_style != Atom::MAP_NONE) {
      atom_->map_clear();
  }
  atom_->nghost = 0;
  atom_->avec->clear_bonus();

  if (maxexchange_fix_dynamic) {
    assert(false);
    int bufextra_old = bufextra;
    init_exchange();
    if (bufextra > bufextra_old) grow_send(maxsend + bufextra, 2);
  }

  double *sublo = domain_->sublo;
  double *subhi = domain_->subhi;

  int dimension = domain_->dimension;
  assert(dimension == 3);

  double **x = atom_->x;

  std::set<int> tag_set;
  std::map<int, int> tag_to_source;
  int zoid_num = zoid.num;
  int num_in_zoid = 0;
  for (int i = 0; i < comm->nprocs; i++) {
    int nrecv;
    MPI_Recv(&nrecv, 1, MPI_INT, i, zoid_num, world, MPI_STATUS_IGNORE);
    if (nrecv > maxrecv) {
        grow_recv(nrecv);
    }
    MPI_Recv(buf_recv, nrecv, MPI_DOUBLE, i, zoid_num, world, MPI_STATUS_IGNORE);
    int m = 0;
    while (m < nrecv) {
      // try remapping the zoid
      bool in_zoid = true;
      for (int dim = 0; dim < domain->dimension; dim++) {
        double lo = sublo[dim];
        double hi = subhi[dim];
        double value = buf_recv[m + dim + 1];

        while (value < lo) { value += domain->prd[dim]; }
        while (value >= hi) { value -= domain->prd[dim]; }

        in_zoid = in_zoid && value >= lo && value < hi;

        /*
        if (lo < 0) {
          in_zoid = in_zoid && ((value >= domain_->prd[dim] + lo) || (value < hi));
        } else {
          in_zoid = in_zoid && (value >= lo) && (value < hi);
        }
        */
      }

      if (in_zoid) {
        m += atom_->avec->unpack_exchange_stencil_md(&buf_recv[m], atom_, domain_, LAMMPS_SEND_LOCAL);
        num_in_zoid++;
      } else {
        m += static_cast<int>(buf_recv[m]);
      }
    }
  }
  domain_->remap_all_stencil_md(atom_);
  if (atom_->firstgroupname) {
      assert(false);
      std::cout << "first reorder" << std::endl;
      atom_->first_reorder();
  }
}

void CommBrick::exchange_stencil_md_initial_receive_double_buffering(queue_info &zoid, int timestep) {
    assert(false);
    // No need to grow recv buffers as already sent before. This is a complete hack.
    double double_buffering_bounds_lo[3];
    double double_buffering_bounds_hi[3];

    int DOUBLE_BUFFERING_IDX = timestep % DOUBLE_BUFFERING;

    for (int dim = 0; dim < NUM_DIMENSIONS; dim++) {
        double lo = std::numeric_limits<double>::infinity();
        double hi = -std::numeric_limits<double>::infinity();
        for (int t = 0; t < NUM_TIMESTEPS_IN_PARALLEL + 1; t++) {
            double zoid_lo = zoid.zoid.cuts[dim].lower + t * zoid.zoid.cuts[dim].slope_lower;
            double zoid_hi = zoid.zoid.cuts[dim].upper + t * zoid.zoid.cuts[dim].slope_upper;
            lo = std::min<double>(lo, zoid_lo);
            hi = std::max<double>(hi, zoid_hi);
        }

        double_buffering_bounds_lo[dim] = lo;
        double_buffering_bounds_hi[dim] = hi;
    }

    std::set<int> tag_set;
    std::map<int, int> tag_to_source;
    int zoid_num = zoid.num;
    int num_in_zoid = 0;

    for (int i = 0; i < comm->nprocs; i++) {
        int nrecv;
        MPI_Recv(&nrecv, 1, MPI_INT, i, zoid_num, world, MPI_STATUS_IGNORE);
        if (nrecv > maxrecv) {
            grow_recv(nrecv);
        }
        MPI_Recv(buf_recv, nrecv, MPI_DOUBLE, i, zoid_num, world, MPI_STATUS_IGNORE);
        int m = 0;
        while (m < nrecv) {
            // try remapping the zoid
            bool in_zoid = true;
            for (int dim = 0; dim < domain->dimension; dim++) {
                double lo = double_buffering_bounds_lo[dim];
                double hi = double_buffering_bounds_hi[dim];
                double value = buf_recv[m + dim + 1];

                while (value < lo) { value += domain->prd[dim]; }
                while (value > hi) { value -= domain->prd[dim]; }

                in_zoid = in_zoid && value >= lo && value < hi;
            }

            if (in_zoid) {
                // m += atom_->avec->unpack_exchange_stencil_md(&buf_recv[m], atom_, domain_, LAMMPS_SEND_LOCAL);
                auto start_buf = &buf_recv[m];
                int recv_buf_idx = 1;
                double x0 = start_buf[recv_buf_idx++];
                double x1 = start_buf[recv_buf_idx++];
                double x2 = start_buf[recv_buf_idx++];

                double v0 = start_buf[recv_buf_idx++];
                double v1 = start_buf[recv_buf_idx++];
                double v2 = start_buf[recv_buf_idx++];

                tagint tag_ = (tagint) ubuf(start_buf[recv_buf_idx++]).i;
                int type_ = (int) ubuf(start_buf[recv_buf_idx++]).i;
                int mask_ = (int) ubuf(start_buf[recv_buf_idx++]).i;
                int image_ = (imageint) ubuf(start_buf[recv_buf_idx++]).i;

                zoid.x_stencil_md[DOUBLE_BUFFERING_IDX].push_back({x0, x1, x2});
                zoid.v_stencil_md[DOUBLE_BUFFERING_IDX].push_back({v0, v1, v2});
                zoid.f_stencil_md[DOUBLE_BUFFERING_IDX].push_back({0.0, 0.0, 0.0});
                zoid.eval_f_stencil_md[DOUBLE_BUFFERING_IDX].push_back({0.0, 0.0, 0.0});

                if (timestep == 0) {
                    zoid.tag_stencil_md[DOUBLE_BUFFERING_IDX].push_back(tag_);
                    zoid.type_stencil_md[DOUBLE_BUFFERING_IDX].push_back(type_);
                    zoid.mask_stencil_md[DOUBLE_BUFFERING_IDX].push_back(mask_);
                    zoid.image_stencil_md[DOUBLE_BUFFERING_IDX].push_back(image_);
                }

                m += static_cast<int>(buf_recv[m]);
                num_in_zoid++;
            } else {
                m += static_cast<int>(buf_recv[m]);
            }
        }
    }

    assert(num_in_zoid == zoid.x_stencil_md[DOUBLE_BUFFERING_IDX].size());

    auto& zoid_pos = zoid.x_stencil_md[DOUBLE_BUFFERING_IDX];
    auto* zoid_img = zoid.image_stencil_md[DOUBLE_BUFFERING_IDX].data();

    imageint idim, otherdims;

    for (int i = 0; i < num_in_zoid; i++) {
        auto &pos = zoid_pos[i];

        assert(domain->xperiodic);
        assert(domain->yperiodic);
        assert(domain->zperiodic);

        /*
           while (x[i][0] < lo[0]) {
                x[i][0] += period[0];
                idim = image[i] & IMGMASK;
                otherdims = image[i] ^ idim;
                idim--;
                idim &= IMGMASK;
                image[i] = otherdims | idim;
            }
            while (x[i][0] >= hi[0]) {
                x[i][0] -= period[0];
                idim = image[i] & IMGMASK;
                otherdims = image[i] ^ idim;
                idim++;
                idim &= IMGMASK;
                image[i] = otherdims | idim;
            }
            x[i][0] = MAX(x[i][0], lo[0]);
         */


        while (pos.x < double_buffering_bounds_lo[0]) {
            pos.x += domain->prd[0];
            idim = zoid_img[i] & IMGMASK;
            otherdims = zoid_img[i] ^ idim;
            idim--;
            idim &= IMGMASK;
            zoid_img[i] = otherdims | idim;
        }
        while (pos.x >= double_buffering_bounds_hi[0]) {
            pos.x -= domain->prd[0];
            idim = zoid_img[i] & IMGMASK;
            otherdims = zoid_img[i] ^ idim;
            idim++;
            idim &= IMGMASK;
            zoid_img[i] = otherdims | idim;
        }
        zoid_pos[i].x = MAX(zoid_pos[i].x, double_buffering_bounds_lo[0]);

        while (pos.y < double_buffering_bounds_lo[1]) {
            pos.y += domain->prd[1];
            idim = zoid_img[i] & IMGMASK;
            otherdims = zoid_img[i] ^ idim;
            idim--;
            idim &= IMGMASK;
            zoid_img[i] = otherdims | idim;
        }
        while (pos.y >= double_buffering_bounds_hi[1]) {
            pos.y -= domain->prd[1];
            idim = zoid_img[i] & IMGMASK;
            otherdims = zoid_img[i] ^ idim;
            idim++;
            idim &= IMGMASK;
            zoid_img[i] = otherdims | idim;
        }
        zoid_pos[i].y = MAX(zoid_pos[i].y, double_buffering_bounds_lo[1]);

        while (pos.z < double_buffering_bounds_lo[2]) {
            pos.z += domain->prd[2];
            idim = zoid_img[i] & IMGMASK;
            otherdims = zoid_img[i] ^ idim;
            idim--;
            idim &= IMGMASK;
            zoid_img[i] = otherdims | idim;
        }
        while (pos.z >= double_buffering_bounds_hi[2]) {
            pos.z -= domain->prd[2];
            idim = zoid_img[i] & IMGMASK;
            otherdims = zoid_img[i] ^ idim;
            idim++;
            idim &= IMGMASK;
            zoid_img[i] = otherdims | idim;
        }
        zoid_pos[i].z = MAX(zoid_pos[i].z, double_buffering_bounds_lo[2]);

    }

    // Copy the remap_all implementation
    // domain_->remap_all_stencil_md(atom_);
}

// send the border atoms, only for timestep 0
// have it so that it's not the case that every zoid or processor has access to
// all of the atoms at time 0
void CommBrick::borders_stencil_md_initial_send(Atom *atom_, Domain *domain_, queue_info &zoid,
                                                int timestep) {
  int size_with_vel = 9;
  AtomVec *avec = atom_->avec;

  int zoid_num = zoid.num;

  std::vector<int> neighbors;
  // neighbors is size 26, or 3^3 - 1
  for (int i = 0; i < NUM_ZOIDS; i++) {
    if (i != zoid_num && is_close(zoid.where, lmp->zoid_num_to_zoid[i].where)) {
      neighbors.push_back(i);
    }
  }

  for (int i = 0; i < neighbors.size(); i++) {
    int send_zoid_num = neighbors[i];
    auto &send_zoid = lmp->zoid_num_to_zoid[send_zoid_num];

    double *send_zoid_lo = lmp->domain_stencil_md[send_zoid_num][timestep]->sublo;
    double *send_zoid_hi = lmp->domain_stencil_md[send_zoid_num][timestep]->subhi;

    double **x = atom_->x;
    tagint *tag = atom_->tag;
    int nfirst = 0;
    int nlast = atom_->nlocal;
    int nsend = 0;
    int pbc_flag_ = 0;
    int pbc_[3] = {0, 0, 0};
    for (int dim = 0; dim < 3; dim++) {
      if (zoid.where[dim] == RIGHT && send_zoid.where[dim] == PBC) {
        pbc_flag_ = 1;
        pbc_[dim] = -1;
      }
      if (zoid.where[dim] == PBC && send_zoid.where[dim] == RIGHT) {
        pbc_flag_ = 1;
        pbc_[dim] = 1;
      }
    }

    double slack;

    if (timestep == NUM_TIMESTEPS_IN_PARALLEL) {
        slack = (NUM_TIMESTEPS_IN_PARALLEL + 1) * ALLEGRO_SLOPE;
    } else {
        slack = (NUM_TIMESTEPS_IN_PARALLEL + 1 - timestep) * ALLEGRO_SLOPE;
    }

    // TODO: new method, no multi-hop ghosts
    slack = ALLEGRO_SLOPE;

    for (int atom_idx = 0; atom_idx < atom_->nlocal + atom_->nghost; atom_idx++) {
      bool borders_zoid = true;
      bool in_zoid = true;
      for (int dim = 0; dim < 3; dim++) {
        // TODO be aware if 2 * ALLEGRO_SLOPE goes more than the width of a zoid
        // we might have issues and have to reach 'further' into a zoid and
        // modify the neighbors which we iterate over double lo =
        // send_zoid_lo[dim] - (NUM_TIMESTEPS_IN_PARALLEL + 1) * ALLEGRO_SLOPE;
        // double hi = send_zoid_hi[dim] + (NUM_TIMESTEPS_IN_PARALLEL + 1) *
        // ALLEGRO_SLOPE;
        /*
        double lo = send_zoid.zoid.cuts[dim].lower +
            send_zoid.zoid.cuts[dim].slope_lower * timestep_idx -
            (num_timesteps_in_parallel + 1) * allegro_slope;
        double hi = send_zoid.zoid.cuts[dim].upper +
            send_zoid.zoid.cuts[dim].slope_upper * timestep_idx +
            (num_timesteps_in_parallel + 1) * allegro_slope;
        */

        double lo = send_zoid.zoid.cuts[dim].lower +
                      send_zoid.zoid.cuts[dim].slope_lower * timestep;
        double hi = send_zoid.zoid.cuts[dim].upper +
                      send_zoid.zoid.cuts[dim].slope_upper * timestep;

        lo -= ALLEGRO_SLOPE;
        hi += ALLEGRO_SLOPE;

        double value = x[atom_idx][dim];

        double atom_pos_shifted = x[atom_idx][dim] + pbc_[dim] * domain->prd[dim];
        borders_zoid = borders_zoid && (atom_pos_shifted >= lo) && (atom_pos_shifted <= hi);
      }

      if (borders_zoid) {
        if (nsend == maxsendlist[i]) { grow_list(i, nsend); }
        sendlist[i][nsend++] = atom_idx;
      }
    }

    if (nsend * size_with_vel > maxsend_stencil_md[i]) {
      grow_send_stencil_md(nsend * size_with_vel, i, 0);
    }
    // TODO: check pbc_flag, pbc

    int n;
    if (ghost_velocity) {
      assert(false);
      // n = avec->pack_border_vel(nsend, sendlist[i], buf_send_stencil_md[i],
      // pbc_flag[i], pbc[i]);
      n = avec->pack_border_vel(nsend, sendlist[i], buf_send_stencil_md[i], pbc_flag_, pbc_);
    } else {
      n = avec->pack_border_vel(nsend, sendlist[i], buf_send_stencil_md[i], pbc_flag_, pbc_);
      // n = avec->pack_border(nsend, sendlist[i], buf_send_stencil_md[i], pbc_flag_, pbc_);
    }

    sendnum[i] = nsend;

    MPI_Request r1;
    MPI_Request r2;
    MPI_Isend(&sendnum[i], 1, MPI_INT, neighbors[i] % comm->nprocs, send_zoid_num, world, &r1);
    if (n) {
      MPI_Isend(buf_send_stencil_md[i], n, MPI_DOUBLE, neighbors[i] % comm->nprocs, send_zoid_num,
                world, &r2);
    }

    smax = MAX(smax, nsend);
    size_reverse_recv[i] = nsend * size_reverse;
  }

  // For molecular systems we lose some bits for local atom indices due
  // to encoding of special pairs in neighbor lists. Check for overflows.

  if ((atom->molecular != Atom::ATOMIC) && ((atom->nlocal + atom->nghost) > NEIGHMASK))
    error->one(FLERR,
               "Per-processor number of atoms is too large for "
               "molecular neighbor lists");

  /*
    int max = maxreverse*smax;
    for (int i = 0; i < maxswap; i++) {
        if (max > maxsend_stencil_md[i]) {
            grow_send_stencil_md(max,i, 0);
        }
    }

    // reset global->local map
    if (map_style != Atom::MAP_NONE) {
        atom->map_set();
    }
    */
}

void CommBrick::borders_stencil_md_initial_receive(Atom *atom_, Domain *domain_, queue_info &zoid, int timestep) {
  int size_with_vel = 9;
  rmax = 0;
  AtomVec *avec = atom_->avec;

  int zoid_num = zoid.num;

  std::vector<int> neighbors;

  // neighbors is size 26, or 3^3 - 1
  for (int i = 0; i < NUM_ZOIDS; i++) {
    if (i != zoid_num && is_close(zoid.where, lmp->zoid_num_to_zoid[i].where)) {
      neighbors.push_back(i);
    }
  }

  for (int i = 0; i < neighbors.size(); i++) {
    int recv_zoid_num = neighbors[i];
    auto &q_info = lmp->zoid_num_to_zoid[recv_zoid_num];
    int zoid_idx = lmp->zoid_num_to_idx[recv_zoid_num];
    int nrecv;
    MPI_Recv(&nrecv, 1, MPI_INT, neighbors[i] % comm->nprocs, zoid_num, world, MPI_STATUS_IGNORE);
    if (nrecv * size_with_vel > maxrecv_stencil_md[i]) {
      grow_recv_stencil_md(nrecv * size_with_vel, i);
    }

    if (nrecv) {
      MPI_Recv(buf_recv_stencil_md[i], nrecv * size_with_vel, MPI_DOUBLE, neighbors[i] % comm->nprocs,
               zoid_num, world, MPI_STATUS_IGNORE);
    }

    int old_idx = atom_->nlocal + atom_->nghost;

    double *buf = buf_recv_stencil_md[i];

    int num_ghosts_added = 0;
    if (ghost_velocity) {
      assert(false);
      avec->unpack_border_vel(nrecv, atom_->nlocal + atom_->nghost, buf);
    } else {
      // avec->unpack_border(nrecv, atom_->nlocal + atom_->nghost, buf);
      num_ghosts_added = avec->unpack_border_stencil_md(nrecv, atom_->nlocal + atom_->nghost, buf,
                                                        atom_, zoid_num);
      nrecv = num_ghosts_added;
    }

    rmax = MAX(rmax, nrecv);
    recvnum[i] = nrecv;
    firstrecv[i] = atom_->nlocal + atom_->nghost;
    atom_->nghost += num_ghosts_added;
    size_forward_recv[i] = nrecv * size_forward;
    size_reverse_send[i] = nrecv * size_reverse;

    std::set<int> tags;
    int repeat_idx = -1;
    tagint repeat_tag = -1;
    for (int k = 0; k < atom_->nlocal + atom_->nghost; k++) {
      if (tags.find(atom_->tag[k]) != tags.end()) {
        repeat_idx = k;
        repeat_tag = atom_->tag[k];
      }
      tags.insert(atom_->tag[k]);
    }

    if (tags.size() != atom_->nlocal + atom_->nghost) {
      std::cout << "Same atom received from multiple zoids, resulting in a "
                   "duplicate tag on two "
                   "difference indices"
                << std::endl;
      std::cout << "zoid: " << zoid_num << " receiving ghost from: " << recv_zoid_num
                << " repeat shit. " << std::endl;
      std::cout << "Repeat tag: " << repeat_tag << " repeat idx: " << repeat_idx << std::endl;
      assert(false);
    }
  }

  // For molecular systems we lose some bits for local atom indices due
  // to encoding of special pairs in neighbor lists. Check for overflows.

  if ((atom->molecular != Atom::ATOMIC) && ((atom->nlocal + atom->nghost) > NEIGHMASK))
    error->one(FLERR,
               "Per-processor number of atoms is too large for "
               "molecular neighbor lists");

  // TODO: whatever this does causes some error
  //    int max_send = maxreverse*smax;
  //    for (int i = 0; i < maxswap; i++) {
  //        if (max_send > maxsend_stencil_md[i]) {
  //            grow_send_stencil_md(max_send,i, 0);
  //        }
  //    }
  //
  //    // insure send/recv buffers are long enough for all forward & reverse
  //    comm int max_recv = maxreverse*rmax; for (int i = 0; i < maxswap; i++) {
  //        if (max_recv > maxrecv_stencil_md[i]) {
  //            grow_recv_stencil_md(max_recv, i);
  //        }
  //    }

  // reset global->local map
  if (map_style != Atom::MAP_NONE) {
    atom_->map_set();
    // atom->map_set();
  }
}

/* ----------------------------------------------------------------------
   borders: list nearby atoms to send to neighboring procs at every timestep
   one list is created for every swap that will be made
   as list is made, actually do swaps
   this does equivalent of a forward_comm(), so don't need to explicitly
     call forward_comm() on reneighboring timestep
   this routine is called before every reneighboring
   for triclinic, atoms must be in lamda coords (0-1) before borders is called
------------------------------------------------------------------------- */

void CommBrick::borders()
{
  int i, n, itype, icollection, iswap, dim, ineed, twoneed;
  int nsend, nrecv, sendflag, nfirst, nlast, ngroup, nprior;
  double lo, hi;
  int *type;
  int *collection;
  double **x;
  double *buf, *mlo, *mhi;
  MPI_Request request;
  AtomVec *avec = atom->avec;

  // After exchanging/sorting, need to reconstruct collection array for border
  // communication
  if (mode == Comm::MULTI) neighbor->build_collection(0);

  // do swaps over all 3 dimensions

  iswap = 0;
  smax = rmax = 0;

  /*
  std::cout << "me: " << comm->me << " my lo: " << domain->sublo[0] << " " << domain->sublo[1] << " " << domain->sublo[2]
  << " my hi: " << domain->subhi[0] << " " << domain->subhi[1] << " " << domain->subhi[2] << std::endl;
  MPI_Barrier(world);
  */

  for (dim = 0; dim < 3; dim++) {
    nlast = 0;
    twoneed = 2 * maxneed[dim];
    for (ineed = 0; ineed < twoneed; ineed++) {
      // find atoms within slab boundaries lo/hi using <= and >=
      // check atoms between nfirst and nlast
      //   for first swaps in a dim, check owned and ghost
      //   for later swaps in a dim, only check newly arrived ghosts
      // store sent atom indices in sendlist for use in future timesteps

      x = atom->x;
      if (mode == Comm::SINGLE) {
        lo = slablo[iswap];
        hi = slabhi[iswap];
      } else if (mode == Comm::MULTI) {
        collection = neighbor->collection;
        mlo = multilo[iswap];
        mhi = multihi[iswap];
      } else {
        type = atom->type;
        mlo = multioldlo[iswap];
        mhi = multioldhi[iswap];
      }
      if (ineed % 2 == 0) {
        nfirst = nlast;
        nlast = atom->nlocal + atom->nghost;
      }

      nsend = 0;

      // sendflag = 0 if I do not send on this swap
      // sendneed test indicates receiver no longer requires data
      // e.g. due to non-PBC or non-uniform sub-domains

      if (ineed / 2 >= sendneed[dim][ineed % 2])
        sendflag = 0;
      else
        sendflag = 1;

      // find send atoms according to SINGLE vs MULTI
      // all atoms eligible versus only atoms in bordergroup
      // can only limit loop to bordergroup for first sends (ineed < 2)
      // on these sends, break loop in two: owned (in group) and ghost

      if (sendflag) {
        if (!bordergroup || ineed >= 2) {
          if (mode == Comm::SINGLE) {
            for (i = nfirst; i < nlast; i++)
              if (x[i][dim] >= lo && x[i][dim] <= hi) {
                if (nsend == maxsendlist[iswap]) grow_list(iswap, nsend);
                sendlist[iswap][nsend++] = i;
              }
          } else if (mode == Comm::MULTI) {
            for (i = nfirst; i < nlast; i++) {
              icollection = collection[i];
              if (x[i][dim] >= mlo[icollection] && x[i][dim] <= mhi[icollection]) {
                if (nsend == maxsendlist[iswap]) grow_list(iswap, nsend);
                sendlist[iswap][nsend++] = i;
              }
            }
          } else {
            for (i = nfirst; i < nlast; i++) {
              itype = type[i];
              if (x[i][dim] >= mlo[itype] && x[i][dim] <= mhi[itype]) {
                if (nsend == maxsendlist[iswap]) grow_list(iswap, nsend);
                sendlist[iswap][nsend++] = i;
              }
            }
          }

        } else {
          if (mode == Comm::SINGLE) {
            ngroup = atom->nfirst;
            for (i = 0; i < ngroup; i++)
              if (x[i][dim] >= lo && x[i][dim] <= hi) {
                if (nsend == maxsendlist[iswap]) grow_list(iswap, nsend);
                sendlist[iswap][nsend++] = i;
              }
            for (i = atom->nlocal; i < nlast; i++)
              if (x[i][dim] >= lo && x[i][dim] <= hi) {
                if (nsend == maxsendlist[iswap]) grow_list(iswap, nsend);
                sendlist[iswap][nsend++] = i;
              }
          } else if (mode == Comm::MULTI) {
            ngroup = atom->nfirst;
            for (i = 0; i < ngroup; i++) {
              icollection = collection[i];
              if (x[i][dim] >= mlo[icollection] && x[i][dim] <= mhi[icollection]) {
                if (nsend == maxsendlist[iswap]) grow_list(iswap, nsend);
                sendlist[iswap][nsend++] = i;
              }
            }
            for (i = atom->nlocal; i < nlast; i++) {
              icollection = collection[i];
              if (x[i][dim] >= mlo[icollection] && x[i][dim] <= mhi[icollection]) {
                if (nsend == maxsendlist[iswap]) grow_list(iswap, nsend);
                sendlist[iswap][nsend++] = i;
              }
            }
          } else {
            ngroup = atom->nfirst;
            for (i = 0; i < ngroup; i++) {
              itype = type[i];
              if (x[i][dim] >= mlo[itype] && x[i][dim] <= mhi[itype]) {
                if (nsend == maxsendlist[iswap]) grow_list(iswap, nsend);
                sendlist[iswap][nsend++] = i;
              }
            }
            for (i = atom->nlocal; i < nlast; i++) {
              itype = type[i];
              if (x[i][dim] >= mlo[itype] && x[i][dim] <= mhi[itype]) {
                if (nsend == maxsendlist[iswap]) grow_list(iswap, nsend);
                sendlist[iswap][nsend++] = i;
              }
            }
          }
        }
      }

      // pack up list of border atoms

      if (nsend * size_border > maxsend) grow_send(nsend * size_border, 0);
      if (ghost_velocity)
        n = avec->pack_border_vel(nsend, sendlist[iswap], buf_send, pbc_flag[iswap], pbc[iswap]);
      else
        n = avec->pack_border(nsend, sendlist[iswap], buf_send, pbc_flag[iswap], pbc[iswap]);

      if (comm->me == 0) {
          std::cout << "dim: " << dim << " ineed: " << ineed << " iswap: " << iswap << " send to proc: " << sendproc[iswap]
                    << " slablo: " << slablo[iswap] << " slabhi: " << slabhi[iswap] << " nsend: " << nsend << std::endl;
      }

      // swap atoms with other proc
      // no MPI calls except SendRecv if nsend/nrecv = 0
      // put incoming ghosts at end of my atom arrays
      // if swapping with self, simply copy, no messages

      if (sendproc[iswap] != me) {
        MPI_Sendrecv(&nsend, 1, MPI_INT, sendproc[iswap], 0, &nrecv, 1, MPI_INT, recvproc[iswap], 0,
                     world, MPI_STATUS_IGNORE);
        if (nrecv * size_border > maxrecv) grow_recv(nrecv * size_border);
        if (nrecv)
          MPI_Irecv(buf_recv, nrecv * size_border, MPI_DOUBLE, recvproc[iswap], 0, world, &request);
        if (n) MPI_Send(buf_send, n, MPI_DOUBLE, sendproc[iswap], 0, world);
        if (nrecv) MPI_Wait(&request, MPI_STATUS_IGNORE);
        buf = buf_recv;
      } else {
        nrecv = nsend;
        buf = buf_send;
      }

      // unpack buffer

      if (ghost_velocity)
        avec->unpack_border_vel(nrecv, atom->nlocal + atom->nghost, buf);
      else
        avec->unpack_border(nrecv, atom->nlocal + atom->nghost, buf);

      // set all pointers & counters

      smax = MAX(smax, nsend);
      rmax = MAX(rmax, nrecv);
      sendnum[iswap] = nsend;
      recvnum[iswap] = nrecv;
      size_forward_recv[iswap] = nrecv * size_forward;
      size_reverse_send[iswap] = nrecv * size_reverse;
      size_reverse_recv[iswap] = nsend * size_reverse;
      firstrecv[iswap] = atom->nlocal + atom->nghost;
      nprior = atom->nlocal + atom->nghost;
      atom->nghost += nrecv;
      if (neighbor->style == Neighbor::MULTI) neighbor->build_collection(nprior);

      iswap++;
    }
  }

  // For molecular systems we lose some bits for local atom indices due
  // to encoding of special pairs in neighbor lists. Check for overflows.

  if ((atom->molecular != Atom::ATOMIC) && ((atom->nlocal + atom->nghost) > NEIGHMASK))
    error->one(FLERR,
               "Per-processor number of atoms is too large for "
               "molecular neighbor lists");

  // insure send/recv buffers are long enough for all forward & reverse comm

  int max = MAX(maxforward * smax, maxreverse * rmax);
  if (max > maxsend) grow_send(max, 0);
  max = MAX(maxforward * rmax, maxreverse * smax);
  if (max > maxrecv) grow_recv(max);

  // reset global->local map

  if (map_style != Atom::MAP_NONE) atom->map_set();
}

/* ----------------------------------------------------------------------
   forward communication invoked by a Pair
   nsize used only to set recv buffer limit
------------------------------------------------------------------------- */

void CommBrick::forward_comm(Pair *pair)
{
  int iswap, n;
  double *buf;
  MPI_Request request;

  int nsize = pair->comm_forward;

  for (iswap = 0; iswap < nswap; iswap++) {
    // pack buffer

    n = pair->pack_forward_comm(sendnum[iswap], sendlist[iswap], buf_send, pbc_flag[iswap],
                                pbc[iswap]);

    // exchange with another proc
    // if self, set recv buffer to send buffer

    if (sendproc[iswap] != me) {
      if (recvnum[iswap])
        MPI_Irecv(buf_recv, nsize * recvnum[iswap], MPI_DOUBLE, recvproc[iswap], 0, world,
                  &request);
      if (sendnum[iswap]) MPI_Send(buf_send, n, MPI_DOUBLE, sendproc[iswap], 0, world);
      if (recvnum[iswap]) MPI_Wait(&request, MPI_STATUS_IGNORE);
      buf = buf_recv;
    } else
      buf = buf_send;

    // unpack buffer

    pair->unpack_forward_comm(recvnum[iswap], firstrecv[iswap], buf);
  }
}

/* ----------------------------------------------------------------------
   reverse communication invoked by a Pair
   nsize used only to set recv buffer limit
------------------------------------------------------------------------- */

void CommBrick::reverse_comm(Pair *pair)
{
  int iswap, n;
  double *buf;
  MPI_Request request;

  int nsize = MAX(pair->comm_reverse, pair->comm_reverse_off);

  for (iswap = nswap - 1; iswap >= 0; iswap--) {
    // pack buffer

    n = pair->pack_reverse_comm(recvnum[iswap], firstrecv[iswap], buf_send);

    // exchange with another proc
    // if self, set recv buffer to send buffer

    if (sendproc[iswap] != me) {
      if (sendnum[iswap])
        MPI_Irecv(buf_recv, nsize * sendnum[iswap], MPI_DOUBLE, sendproc[iswap], 0, world,
                  &request);
      if (recvnum[iswap]) MPI_Send(buf_send, n, MPI_DOUBLE, recvproc[iswap], 0, world);
      if (sendnum[iswap]) MPI_Wait(&request, MPI_STATUS_IGNORE);
      buf = buf_recv;
    } else
      buf = buf_send;

    // unpack buffer

    pair->unpack_reverse_comm(sendnum[iswap], sendlist[iswap], buf);
  }
}

/* ----------------------------------------------------------------------
   forward communication invoked by a Bond
   nsize used only to set recv buffer limit
------------------------------------------------------------------------- */

void CommBrick::forward_comm(Bond *bond)
{
  int iswap, n;
  double *buf;
  MPI_Request request;

  int nsize = bond->comm_forward;

  for (iswap = 0; iswap < nswap; iswap++) {
    // pack buffer

    n = bond->pack_forward_comm(sendnum[iswap], sendlist[iswap], buf_send, pbc_flag[iswap],
                                pbc[iswap]);

    // exchange with another proc
    // if self, set recv buffer to send buffer

    if (sendproc[iswap] != me) {
      if (recvnum[iswap])
        MPI_Irecv(buf_recv, nsize * recvnum[iswap], MPI_DOUBLE, recvproc[iswap], 0, world,
                  &request);
      if (sendnum[iswap]) MPI_Send(buf_send, n, MPI_DOUBLE, sendproc[iswap], 0, world);
      if (recvnum[iswap]) MPI_Wait(&request, MPI_STATUS_IGNORE);
      buf = buf_recv;
    } else
      buf = buf_send;

    // unpack buffer

    bond->unpack_forward_comm(recvnum[iswap], firstrecv[iswap], buf);
  }
}

/* ----------------------------------------------------------------------
   reverse communication invoked by a Bond
   nsize used only to set recv buffer limit
------------------------------------------------------------------------- */

void CommBrick::reverse_comm(Bond *bond)
{
  int iswap, n;
  double *buf;
  MPI_Request request;

  int nsize = MAX(bond->comm_reverse, bond->comm_reverse_off);

  for (iswap = nswap - 1; iswap >= 0; iswap--) {
    // pack buffer

    n = bond->pack_reverse_comm(recvnum[iswap], firstrecv[iswap], buf_send);

    // exchange with another proc
    // if self, set recv buffer to send buffer

    if (sendproc[iswap] != me) {
      if (sendnum[iswap])
        MPI_Irecv(buf_recv, nsize * sendnum[iswap], MPI_DOUBLE, sendproc[iswap], 0, world,
                  &request);
      if (recvnum[iswap]) MPI_Send(buf_send, n, MPI_DOUBLE, recvproc[iswap], 0, world);
      if (sendnum[iswap]) MPI_Wait(&request, MPI_STATUS_IGNORE);
      buf = buf_recv;
    } else
      buf = buf_send;

    // unpack buffer

    bond->unpack_reverse_comm(sendnum[iswap], sendlist[iswap], buf);
  }
}

/* ----------------------------------------------------------------------
   forward communication invoked by a Fix
   size/nsize used only to set recv buffer limit
   size = 0 (default) -> use comm_forward from Fix
   size > 0 -> Fix passes max size per atom
   the latter is only useful if Fix does several comm modes,
     some are smaller than max stored in its comm_forward
------------------------------------------------------------------------- */

void CommBrick::forward_comm(Fix *fix, int size)
{
  int iswap, n, nsize;
  double *buf;
  MPI_Request request;

  if (size)
    nsize = size;
  else
    nsize = fix->comm_forward;

  for (iswap = 0; iswap < nswap; iswap++) {
    // pack buffer

    n = fix->pack_forward_comm(sendnum[iswap], sendlist[iswap], buf_send, pbc_flag[iswap],
                               pbc[iswap]);

    // exchange with another proc
    // if self, set recv buffer to send buffer

    if (sendproc[iswap] != me) {
      if (recvnum[iswap])
        MPI_Irecv(buf_recv, nsize * recvnum[iswap], MPI_DOUBLE, recvproc[iswap], 0, world,
                  &request);
      if (sendnum[iswap]) MPI_Send(buf_send, n, MPI_DOUBLE, sendproc[iswap], 0, world);
      if (recvnum[iswap]) MPI_Wait(&request, MPI_STATUS_IGNORE);
      buf = buf_recv;
    } else
      buf = buf_send;

    // unpack buffer

    fix->unpack_forward_comm(recvnum[iswap], firstrecv[iswap], buf);
  }
}

/* ----------------------------------------------------------------------
   reverse communication invoked by a Fix
   size/nsize used only to set recv buffer limit
   size = 0 (default) -> use comm_forward from Fix
   size > 0 -> Fix passes max size per atom
   the latter is only useful if Fix does several comm modes,
     some are smaller than max stored in its comm_forward
------------------------------------------------------------------------- */

void CommBrick::reverse_comm(Fix *fix, int size)
{
  int iswap, n, nsize;
  double *buf;
  MPI_Request request;

  if (size)
    nsize = size;
  else
    nsize = fix->comm_reverse;

  for (iswap = nswap - 1; iswap >= 0; iswap--) {
    // pack buffer

    n = fix->pack_reverse_comm(recvnum[iswap], firstrecv[iswap], buf_send);

    // exchange with another proc
    // if self, set recv buffer to send buffer

    if (sendproc[iswap] != me) {
      if (sendnum[iswap])
        MPI_Irecv(buf_recv, nsize * sendnum[iswap], MPI_DOUBLE, sendproc[iswap], 0, world,
                  &request);
      if (recvnum[iswap]) MPI_Send(buf_send, n, MPI_DOUBLE, recvproc[iswap], 0, world);
      if (sendnum[iswap]) MPI_Wait(&request, MPI_STATUS_IGNORE);
      buf = buf_recv;
    } else
      buf = buf_send;

    // unpack buffer

    fix->unpack_reverse_comm(sendnum[iswap], sendlist[iswap], buf);
  }
}

/* ----------------------------------------------------------------------
   reverse communication invoked by a Fix with variable size data
   query fix for pack size to insure buf_send is big enough
   handshake sizes before each Irecv/Send to insure buf_recv is big enough
------------------------------------------------------------------------- */

void CommBrick::reverse_comm_variable(Fix *fix)
{
  int iswap, nsend, nrecv;
  double *buf;
  MPI_Request request;

  for (iswap = nswap - 1; iswap >= 0; iswap--) {
    // pack buffer

    nsend = fix->pack_reverse_comm_size(recvnum[iswap], firstrecv[iswap]);
    if (nsend > maxsend) grow_send(nsend, 0);
    nsend = fix->pack_reverse_comm(recvnum[iswap], firstrecv[iswap], buf_send);

    // exchange with another proc
    // if self, set recv buffer to send buffer

    if (sendproc[iswap] != me) {
      MPI_Sendrecv(&nsend, 1, MPI_INT, recvproc[iswap], 0, &nrecv, 1, MPI_INT, sendproc[iswap], 0,
                   world, MPI_STATUS_IGNORE);

      if (sendnum[iswap]) {
        if (nrecv > maxrecv) grow_recv(nrecv);
        MPI_Irecv(buf_recv, maxrecv, MPI_DOUBLE, sendproc[iswap], 0, world, &request);
      }
      if (recvnum[iswap]) MPI_Send(buf_send, nsend, MPI_DOUBLE, recvproc[iswap], 0, world);
      if (sendnum[iswap]) MPI_Wait(&request, MPI_STATUS_IGNORE);
      buf = buf_recv;
    } else
      buf = buf_send;

    // unpack buffer

    fix->unpack_reverse_comm(sendnum[iswap], sendlist[iswap], buf);
  }
}

/* ----------------------------------------------------------------------
   forward communication invoked by a Compute
   nsize used only to set recv buffer limit
------------------------------------------------------------------------- */

void CommBrick::forward_comm(Compute *compute)
{
  int iswap, n;
  double *buf;
  MPI_Request request;

  int nsize = compute->comm_forward;

  for (iswap = 0; iswap < nswap; iswap++) {
    // pack buffer

    n = compute->pack_forward_comm(sendnum[iswap], sendlist[iswap], buf_send, pbc_flag[iswap],
                                   pbc[iswap]);

    // exchange with another proc
    // if self, set recv buffer to send buffer

    if (sendproc[iswap] != me) {
      if (recvnum[iswap])
        MPI_Irecv(buf_recv, nsize * recvnum[iswap], MPI_DOUBLE, recvproc[iswap], 0, world,
                  &request);
      if (sendnum[iswap]) MPI_Send(buf_send, n, MPI_DOUBLE, sendproc[iswap], 0, world);
      if (recvnum[iswap]) MPI_Wait(&request, MPI_STATUS_IGNORE);
      buf = buf_recv;
    } else
      buf = buf_send;

    // unpack buffer

    compute->unpack_forward_comm(recvnum[iswap], firstrecv[iswap], buf);
  }
}

/* ----------------------------------------------------------------------
   reverse communication invoked by a Compute
   nsize used only to set recv buffer limit
------------------------------------------------------------------------- */

void CommBrick::reverse_comm(Compute *compute)
{
  int iswap, n;
  double *buf;
  MPI_Request request;

  int nsize = compute->comm_reverse;

  for (iswap = nswap - 1; iswap >= 0; iswap--) {
    // pack buffer

    n = compute->pack_reverse_comm(recvnum[iswap], firstrecv[iswap], buf_send);

    // exchange with another proc
    // if self, set recv buffer to send buffer

    if (sendproc[iswap] != me) {
      if (sendnum[iswap])
        MPI_Irecv(buf_recv, nsize * sendnum[iswap], MPI_DOUBLE, sendproc[iswap], 0, world,
                  &request);
      if (recvnum[iswap]) MPI_Send(buf_send, n, MPI_DOUBLE, recvproc[iswap], 0, world);
      if (sendnum[iswap]) MPI_Wait(&request, MPI_STATUS_IGNORE);
      buf = buf_recv;
    } else
      buf = buf_send;

    // unpack buffer

    compute->unpack_reverse_comm(sendnum[iswap], sendlist[iswap], buf);
  }
}

/* ----------------------------------------------------------------------
   forward communication invoked by a Dump
   nsize used only to set recv buffer limit
------------------------------------------------------------------------- */

void CommBrick::forward_comm(Dump *dump)
{
  int iswap, n;
  double *buf;
  MPI_Request request;

  int nsize = dump->comm_forward;

  for (iswap = 0; iswap < nswap; iswap++) {
    // pack buffer

    n = dump->pack_forward_comm(sendnum[iswap], sendlist[iswap], buf_send, pbc_flag[iswap],
                                pbc[iswap]);

    // exchange with another proc
    // if self, set recv buffer to send buffer

    if (sendproc[iswap] != me) {
      if (recvnum[iswap])
        MPI_Irecv(buf_recv, nsize * recvnum[iswap], MPI_DOUBLE, recvproc[iswap], 0, world,
                  &request);
      if (sendnum[iswap]) MPI_Send(buf_send, n, MPI_DOUBLE, sendproc[iswap], 0, world);
      if (recvnum[iswap]) MPI_Wait(&request, MPI_STATUS_IGNORE);
      buf = buf_recv;
    } else
      buf = buf_send;

    // unpack buffer

    dump->unpack_forward_comm(recvnum[iswap], firstrecv[iswap], buf);
  }
}

/* ----------------------------------------------------------------------
   reverse communication invoked by a Dump
   nsize used only to set recv buffer limit
------------------------------------------------------------------------- */

void CommBrick::reverse_comm(Dump *dump)
{
  int iswap, n;
  double *buf;
  MPI_Request request;

  int nsize = dump->comm_reverse;

  for (iswap = nswap - 1; iswap >= 0; iswap--) {
    // pack buffer

    n = dump->pack_reverse_comm(recvnum[iswap], firstrecv[iswap], buf_send);

    // exchange with another proc
    // if self, set recv buffer to send buffer

    if (sendproc[iswap] != me) {
      if (sendnum[iswap])
        MPI_Irecv(buf_recv, nsize * sendnum[iswap], MPI_DOUBLE, sendproc[iswap], 0, world,
                  &request);
      if (recvnum[iswap]) MPI_Send(buf_send, n, MPI_DOUBLE, recvproc[iswap], 0, world);
      if (sendnum[iswap]) MPI_Wait(&request, MPI_STATUS_IGNORE);
      buf = buf_recv;
    } else
      buf = buf_send;

    // unpack buffer

    dump->unpack_reverse_comm(sendnum[iswap], sendlist[iswap], buf);
  }
}

/* ----------------------------------------------------------------------
   forward communication of N values in per-atom array
------------------------------------------------------------------------- */

void CommBrick::forward_comm_array(int nsize, double **array)
{
  int i, j, k, m, iswap, last;
  double *buf;
  MPI_Request request;

  // insure send/recv bufs are big enough for nsize
  // based on smax/rmax from most recent borders() invocation

  if (nsize > maxforward) {
    maxforward = nsize;
    if (maxforward * smax > maxsend) grow_send(maxforward * smax, 0);
    if (maxforward * rmax > maxrecv) grow_recv(maxforward * rmax);
  }

  for (iswap = 0; iswap < nswap; iswap++) {
    // pack buffer

    m = 0;
    for (i = 0; i < sendnum[iswap]; i++) {
      j = sendlist[iswap][i];
      for (k = 0; k < nsize; k++) buf_send[m++] = array[j][k];
    }

    // exchange with another proc
    // if self, set recv buffer to send buffer

    if (sendproc[iswap] != me) {
      if (recvnum[iswap])
        MPI_Irecv(buf_recv, nsize * recvnum[iswap], MPI_DOUBLE, recvproc[iswap], 0, world,
                  &request);
      if (sendnum[iswap])
        MPI_Send(buf_send, nsize * sendnum[iswap], MPI_DOUBLE, sendproc[iswap], 0, world);
      if (recvnum[iswap]) MPI_Wait(&request, MPI_STATUS_IGNORE);
      buf = buf_recv;
    } else
      buf = buf_send;

    // unpack buffer

    m = 0;
    last = firstrecv[iswap] + recvnum[iswap];
    for (i = firstrecv[iswap]; i < last; i++)
      for (k = 0; k < nsize; k++) array[i][k] = buf[m++];
  }
}

/* ----------------------------------------------------------------------
   exchange info provided with all 6 stencil neighbors
------------------------------------------------------------------------- */

int CommBrick::exchange_variable(int n, double *inbuf, double *&outbuf)
{
  int nsend, nrecv, nrecv1, nrecv2;
  MPI_Request request;

  nrecv = n;
  if (nrecv > maxrecv) grow_recv(nrecv);
  memcpy(buf_recv, inbuf, nrecv * sizeof(double));

  // loop over dimensions

  for (int dim = 0; dim < 3; dim++) {
    // no exchange if only one proc in a dimension

    if (procgrid[dim] == 1) continue;

    // send/recv info in both directions using same buf_recv
    // if 2 procs in dimension, single send/recv
    // if more than 2 procs in dimension, send/recv to both neighbors

    nsend = nrecv;
    MPI_Sendrecv(&nsend, 1, MPI_INT, procneigh[dim][0], 0, &nrecv1, 1, MPI_INT, procneigh[dim][1],
                 0, world, MPI_STATUS_IGNORE);
    nrecv += nrecv1;
    if (procgrid[dim] > 2) {
      MPI_Sendrecv(&nsend, 1, MPI_INT, procneigh[dim][1], 0, &nrecv2, 1, MPI_INT, procneigh[dim][0],
                   0, world, MPI_STATUS_IGNORE);
      nrecv += nrecv2;
    } else
      nrecv2 = 0;

    if (nrecv > maxrecv) grow_recv(nrecv);

    MPI_Irecv(&buf_recv[nsend], nrecv1, MPI_DOUBLE, procneigh[dim][1], 0, world, &request);
    MPI_Send(buf_recv, nsend, MPI_DOUBLE, procneigh[dim][0], 0, world);
    MPI_Wait(&request, MPI_STATUS_IGNORE);

    if (procgrid[dim] > 2) {
      MPI_Irecv(&buf_recv[nsend + nrecv1], nrecv2, MPI_DOUBLE, procneigh[dim][0], 0, world,
                &request);
      MPI_Send(buf_recv, nsend, MPI_DOUBLE, procneigh[dim][1], 0, world);
      MPI_Wait(&request, MPI_STATUS_IGNORE);
    }
  }

  outbuf = buf_recv;
  return nrecv;
}

/* ----------------------------------------------------------------------
   realloc the size of the send buffer as needed with BUFFACTOR and bufextra
   flag = 0, don't need to realloc with copy, just free/malloc w/ BUFFACTOR
   flag = 1, realloc with BUFFACTOR
   flag = 2, free/malloc w/out BUFFACTOR
------------------------------------------------------------------------- */

void CommBrick::grow_send(int n, int flag)
{
  if (flag == 0) {
    maxsend = static_cast<int>(BUFFACTOR * n);
    memory->destroy(buf_send);
    memory->create(buf_send, maxsend + bufextra, "comm:buf_send");
  } else if (flag == 1) {
    maxsend = static_cast<int>(BUFFACTOR * n);
    memory->grow(buf_send, maxsend + bufextra, "comm:buf_send");
  } else {
    memory->destroy(buf_send);
    memory->grow(buf_send, maxsend + bufextra, "comm:buf_send");
  }
}

void CommBrick::grow_send_stencil_md(int n, int idx, int flag)
{
  std::cout << "GROW SEND STENCILMD" << std::endl;
  if (flag == 0) {
    maxsend_stencil_md[idx] = static_cast<int>(BUFFACTOR * n);
    memory->destroy(buf_send_stencil_md[idx]);
    buf_send_stencil_md[idx] = memory->create(buf_send_stencil_md[idx],
                                              (maxsend_stencil_md[idx] + bufextra) * NUM_PIPELINE_STAGES, "comm:buf_send");
  } else if (flag == 1) {
    maxsend_stencil_md[idx] = static_cast<int>(BUFFACTOR * n);
    buf_send_stencil_md[idx] =
        memory->grow(buf_send_stencil_md[idx], (maxsend_stencil_md[idx] + bufextra) * NUM_PIPELINE_STAGES, "comm:buf_send");
  } else {
    memory->destroy(buf_send_stencil_md[idx]);
    buf_send_stencil_md[idx] =
        memory->grow(buf_send_stencil_md[idx], (maxsend_stencil_md[idx] + bufextra) * NUM_PIPELINE_STAGES, "comm:buf_send");
  }
}

/* ----------------------------------------------------------------------
   free/malloc the size of the recv buffer as needed with BUFFACTOR
------------------------------------------------------------------------- */

void CommBrick::grow_recv(int n)
{
  maxrecv = static_cast<int>(BUFFACTOR * n);
  memory->destroy(buf_recv);
  memory->create(buf_recv, maxrecv, "comm:buf_recv");
}

void CommBrick::grow_recv_stencil_md(int n, int idx)
{
  std::cout << "GROW RECV STENCILMD" << std::endl;
  maxrecv_stencil_md[idx] = static_cast<int>(BUFFACTOR * n);
  memory->destroy(buf_recv_stencil_md[idx]);
  memory->create(buf_recv_stencil_md[idx], maxrecv_stencil_md[idx] * NUM_PIPELINE_STAGES, "comm:buf_recv");
}

/* ----------------------------------------------------------------------
   realloc the size of the iswap sendlist as needed with BUFFACTOR
------------------------------------------------------------------------- */

void CommBrick::grow_list(int iswap, int n)
{
  maxsendlist[iswap] = static_cast<int>(BUFFACTOR * n);
  memory->grow(sendlist[iswap], maxsendlist[iswap], "comm:sendlist[iswap]");
}

/* ----------------------------------------------------------------------
   realloc the buffers needed for swaps
------------------------------------------------------------------------- */

void CommBrick::grow_swap(int n)
{
  free_swap();
  allocate_swap(n);
  if (mode == Comm::MULTI) {
    free_multi();
    allocate_multi(n);
  }

  if (mode == Comm::MULTIOLD) {
    free_multiold();
    allocate_multiold(n);
  }

  sendlist = (int **) memory->srealloc(sendlist, n * sizeof(int *), "comm:sendlist");
  memory->grow(maxsendlist, n, "comm:maxsendlist");
  for (int i = maxswap; i < n; i++) {
    maxsendlist[i] = BUFMIN;
    memory->create(sendlist[i], BUFMIN, "comm:sendlist[i]");
  }

  // stencil md version of sendlist and maxsendlist
  for (int i = 0; i < NUM_TIMESTEPS_IN_PARALLEL + 1; i++) {
    sendlist_stencil_md[i] =
        (int **) memory->smalloc(n * sizeof(int *), "comm:sendlist_stencil_md");
    send_force_stencil_md[i] =
        (bool **) memory->smalloc(n * sizeof(int *), "comm:send_force_stencil_md[t]");
    send_pos_stencil_md[i] =
        (bool **) memory->smalloc(n * sizeof(int *), "comm:send_pos_stencil_md[t]");
    send_vel_stencil_md[i] =
        (bool **) memory->smalloc(n * sizeof(int *), "comm:send_vel_stencil_md[t]");

    sendlist_stencil_md_next_dt[i] =
        (int **) memory->smalloc(n * sizeof(int *), "comm:sendlist_stencil_md");
    send_force_stencil_md_next_dt[i] =
        (bool **) memory->smalloc(n * sizeof(int *), "comm:send_force_stencil_md[t]");
    send_pos_stencil_md_next_dt[i] =
        (bool **) memory->smalloc(n * sizeof(int *), "comm:send_pos_stencil_md[t]");
    send_vel_stencil_md_next_dt[i] =
        (bool **) memory->smalloc(n * sizeof(int *), "comm:send_vel_stencil_md[t]");

    second_sendlist_stencil_md[i] =
        (int **) memory->smalloc(n * sizeof(int *), "comm:sendlist_stencil_md");
    second_sendlist_stencil_md_next_dt[i] =
        (int **) memory->smalloc(n * sizeof(int *), "comm:sendlist_stencil_md");
    memory->create(maxsendlist_stencil_md[i], maxswap, "comm:maxsendlist_stencil_md");
    memory->create(maxsendlist_stencil_md_next_dt[i], maxswap, "comm:maxsendlist_stencil_md");
    memory->create(max_second_sendlist_stencil_md[i], maxswap, "comm:maxsendlist_stencil_md");
    memory->create(max_second_sendlist_stencil_md_next_dt[i], maxswap, "comm:maxsendlist_stencil_md");

    for (int j = 0; j < n; j++) {
      maxsendlist_stencil_md[i][j] = BUFMIN;
      maxsendlist_stencil_md_next_dt[i][j] = BUFMIN;

      max_second_sendlist_stencil_md[i][j] = BUFMIN;
      max_second_sendlist_stencil_md_next_dt[i][j] = BUFMIN;

      memory->create(sendlist_stencil_md[i][j], BUFMIN, "comm:sendlist_stencil_md[i]");
      memory->create(send_force_stencil_md[i][j], BUFMIN, "comm:send_force_stencil_md[i]");
      memory->create(send_pos_stencil_md[i][j], BUFMIN, "comm:send_pos_stencil_md[i]");
      memory->create(send_vel_stencil_md[i][j], BUFMIN, "comm:send_vel_stencil_md[i]");

      memory->create(sendlist_stencil_md_next_dt[i][j], BUFMIN, "comm:sendlist_stencil_md[i]");
      memory->create(send_force_stencil_md_next_dt[i][j], BUFMIN, "comm:send_force_stencil_md[i]");
      memory->create(send_pos_stencil_md_next_dt[i][j], BUFMIN, "comm:send_pos_stencil_md[i]");
      memory->create(send_vel_stencil_md_next_dt[i][j], BUFMIN, "comm:send_vel_stencil_md[i]");

      memory->create(second_sendlist_stencil_md[i][j], BUFMIN, "comm:sendlist_stencil_md[i]");
      memory->create(second_sendlist_stencil_md_next_dt[i][j], BUFMIN, "comm:sendlist_stencil_md[i]");
    }
  }

  maxswap = n;
}

/* ----------------------------------------------------------------------
   allocation of swap info
------------------------------------------------------------------------- */

void CommBrick::allocate_swap(int n)
{
  memory->create(sendnum, n, "comm:sendnum");
  memory->create(recvnum, n, "comm:recvnum");
  memory->create(sendproc, n, "comm:sendproc");
  memory->create(recvproc, n, "comm:recvproc");
  memory->create(size_forward_recv, n, "comm:size");
  memory->create(size_reverse_send, n, "comm:size");
  memory->create(size_reverse_recv, n, "comm:size");
  memory->create(slablo, n, "comm:slablo");
  memory->create(slabhi, n, "comm:slabhi");
  memory->create(firstrecv, n, "comm:firstrecv");
  memory->create(pbc_flag, n, "comm:pbc_flag");
  memory->create(pbc, n, 6, "comm:pbc");

  // stencil_md
  for (int i = 0; i < NUM_TIMESTEPS_IN_PARALLEL + 1; i++) {
    memory->create(sendnum_stencil_md[i], n, "comm:sendnum_stencil_md");
    memory->create(sendnum_stencil_md_next_dt[i], n, "comm:sendnum_stencil_md");
    memory->create(second_sendnum_stencil_md[i], n, "comm:sendnum_stencil_md");
    memory->create(second_sendnum_stencil_md_next_dt[i], n, "comm:sendnum_stencil_md");
  }
}

/* ----------------------------------------------------------------------
   allocation of multi-collection swap info
------------------------------------------------------------------------- */

void CommBrick::allocate_multi(int n)
{
  multilo = memory->create(multilo, n, ncollections, "comm:multilo");
  multihi = memory->create(multihi, n, ncollections, "comm:multihi");
}

/* ----------------------------------------------------------------------
   allocation of multi/old-type swap info
------------------------------------------------------------------------- */

void CommBrick::allocate_multiold(int n)
{
  multioldlo = memory->create(multioldlo, n, atom->ntypes + 1, "comm:multioldlo");
  multioldhi = memory->create(multioldhi, n, atom->ntypes + 1, "comm:multioldhi");
}

/* ----------------------------------------------------------------------
   free memory for swaps
------------------------------------------------------------------------- */

void CommBrick::free_swap()
{
  memory->destroy(sendnum);
  memory->destroy(recvnum);
  memory->destroy(sendproc);
  memory->destroy(recvproc);
  memory->destroy(size_forward_recv);
  memory->destroy(size_reverse_send);
  memory->destroy(size_reverse_recv);
  memory->destroy(slablo);
  memory->destroy(slabhi);
  memory->destroy(firstrecv);
  memory->destroy(pbc_flag);
  memory->destroy(pbc);

  // stencil_md
  for (int i = 0; i < NUM_TIMESTEPS_IN_PARALLEL + 1; i++) {
    memory->destroy(sendnum_stencil_md[i]);
    memory->destroy(sendnum_stencil_md_next_dt[i]);
    memory->destroy(second_sendnum_stencil_md[i]);
    memory->destroy(second_sendnum_stencil_md_next_dt[i]);
  }
}

/* ----------------------------------------------------------------------
   free memory for multi-collection swaps
------------------------------------------------------------------------- */

void CommBrick::free_multi()
{
  memory->destroy(multilo);
  memory->destroy(multihi);
  multilo = multihi = nullptr;
}

/* ----------------------------------------------------------------------
   free memory for multi/old-type swaps
------------------------------------------------------------------------- */

void CommBrick::free_multiold()
{
  memory->destroy(multioldlo);
  memory->destroy(multioldhi);
  multioldlo = multioldhi = nullptr;
}

/* ----------------------------------------------------------------------
   extract data potentially useful to other classes
------------------------------------------------------------------------- */

void *CommBrick::extract(const char *str, int &dim)
{
  dim = 0;
  if (strcmp(str, "localsendlist") == 0) {
    int i, iswap, isend;
    dim = 1;
    if (!localsendlist)
      memory->create(localsendlist, atom->nlocal, "comm:localsendlist");
    else
      memory->grow(localsendlist, atom->nlocal, "comm:localsendlist");

    for (i = 0; i < atom->nlocal; i++) localsendlist[i] = 0;

    for (iswap = 0; iswap < nswap; iswap++)
      for (isend = 0; isend < sendnum[iswap]; isend++)
        if (sendlist[iswap][isend] < atom->nlocal) localsendlist[sendlist[iswap][isend]] = 1;

    return (void *) localsendlist;
  }

  return nullptr;
}

/* ----------------------------------------------------------------------
   return # of bytes of allocated memory
------------------------------------------------------------------------- */

double CommBrick::memory_usage()
{
  double bytes = 0;
  bytes += (double) nprocs * sizeof(int);    // grid2proc
  for (int i = 0; i < nswap; i++) bytes += memory->usage(sendlist[i], maxsendlist[i]);
  bytes += memory->usage(buf_send, maxsend + bufextra);
  bytes += memory->usage(buf_recv, maxrecv);
  return bytes;
}
