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

  int num_send = 0;

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
        num_send += size_forward_recv[iswap];
        // std::cout << GREEN << "lammps forward comm num recv " << size_forward_recv[iswap] << " other way around? " << n << RESET_COLOR << std::endl;
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

  /*
  MPI_Allreduce(MPI_IN_PLACE, &num_send, 1, MPI_INT, MPI_SUM, world);
  if (comm->me == 0) {
      std::cout << "num send forward: " << num_send << std::endl;
  }
  */
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
  int num_send = 0;

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
        num_send += size_reverse_recv[iswap];
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
  /*
  MPI_Allreduce(MPI_IN_PLACE, &num_send, 1, MPI_INT, MPI_SUM, world);
  if (comm->me == 0) {
      std::cout << "num send reverse: " << num_send << std::endl;
  }
  */
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

    std::set<tagint> tags_set;
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

                if (atom_->nlocal == 0) {
                    std::cout << "Lo: " << lo << " hi: " << hi << std::endl;
                }
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

            // std::cout << "m: " << m << " out of: " << nrecv << " zoid: " << zoid.num << " timestep: " << timestep << std::endl;
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
        while (value > hi) { value -= domain->prd[dim]; }

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

// send ghosts to other zoid for the to compute their second send list which is based on local atoms in previous timesteps
// local t --> ghost t + 1, we need to send ghost t + 1 in order for the receiving zoid to order their local t (or ghost t + 1) to match accordingly
void CommBrick::construct_second_send_list_stencil_md_send(std::array<Atom *, NUM_TIMESTEPS_IN_PARALLEL + 1> &atom_arr, queue_info &zoid,
                                                           std::vector<MPI_Request>& r) {
    int zoid_num = zoid.num;
    std::vector<int>& recv_from = lmp->recv_from_neighbors[zoid_num];

    int idx = 0;
    for (int i = 0; i < recv_from.size(); i++) {
        int recv_zoid_num = recv_from[i];

        // only need to send from t = 1, since ghosts at time 0 never receive information
        int nsend_arr[NUM_TIMESTEPS_IN_PARALLEL + 1] = {0};
        int total = 0;
        for (int t = 0; t < NUM_TIMESTEPS_IN_PARALLEL + 1; t++) {
            int num_segments = zoid.recv_ghost_num_segments[t][i];
            int size_at_timestep = 0;
            for (int j = 0; j < num_segments; j++) {
                int segment_size = zoid.recv_ghost_sizes[t][i][j];
                assert(segment_size >= 0);
                total += segment_size;
                size_at_timestep += segment_size;
            }
            nsend_arr[t] = size_at_timestep;
        }

        if (total > maxsend_stencil_md[i]) {
            grow_send_stencil_md(total, i, 0);
        }

        int buf_idx = 0;
        for (int t = 0; t < NUM_TIMESTEPS_IN_PARALLEL + 1; t++) {
            Atom* atom_ = atom_arr[t];

            int num_segments = zoid.recv_ghost_num_segments[t][i];
            for (int j = 0; j < num_segments; j++) {
                int segment_size = zoid.recv_ghost_sizes[t][i][j];
                int segment_start_idx = zoid.recv_ghost_idxs[t][i][j];
                for (int k = segment_start_idx; k < segment_start_idx + segment_size; k++) {
                    double d = ubuf(atom_->tag[k]).d;
                    buf_send_stencil_md[i][buf_idx++] = d;
                }
            }
        }

        assert(buf_idx == total);

        for (int t = 0; t < NUM_TIMESTEPS_IN_PARALLEL + 1; t++) {
            send_list_sendnum_stencil_md[i][t] = nsend_arr[t];
        }

        // dst << 16 | src
        // int mpi_tag = (recv_zoid_num << 16 | zoid_num);
        int mpi_tag = get_mpi_tag(recv_zoid_num, zoid_num);

        MPI_Isend(send_list_sendnum_stencil_md[i], NUM_TIMESTEPS_IN_PARALLEL + 1, MPI_INT, recv_zoid_num % comm->nprocs, mpi_tag, world, &r[2 * idx]);
        if (total) {
            MPI_Isend(buf_send_stencil_md[i], total, MPI_DOUBLE, recv_zoid_num % comm->nprocs, mpi_tag,
                      world, &r[2 * idx + 1]);
        }
        idx++;
    }
}

// send ghosts to other zoid for the to compute their second send list which is based on local atoms in previous timesteps
// local t --> ghost t + 1, we need to send ghost t + 1 in order for the receiving zoid to order their local t (or ghost t + 1) to match accordingly
void CommBrick::construct_second_send_list_stencil_md_next_dt_send(std::array<Atom *, NUM_TIMESTEPS_IN_PARALLEL + 1> &atom_arr, queue_info &zoid,
                                                                   std::vector<MPI_Request>& r) {
    int zoid_num = zoid.num;
    std::vector<int>& recv_from = lmp->recv_from_neighbors_next_dt[zoid_num];

    int idx = 0;
    for (int i = 0; i < recv_from.size(); i++) {
        int recv_zoid_num = recv_from[i];

        // only need to send from t = 1, since ghosts at time 0 never receive information
        int nsend_arr[NUM_TIMESTEPS_IN_PARALLEL + 1] = {0};
        int total = 0;
        for (int t = 0; t < NUM_TIMESTEPS_IN_PARALLEL + 1; t++) {
            int num_segments = zoid.recv_ghost_num_segments[t][i];
            int size_at_timestep = 0;
            for (int j = 0; j < num_segments; j++) {
                int segment_size = zoid.recv_ghost_sizes[t][i][j];
                assert(segment_size >= 0);
                total += segment_size;
                size_at_timestep += segment_size;
            }
            nsend_arr[t] = size_at_timestep;
        }

        if (total > maxsend_stencil_md[i]) {
            grow_send_stencil_md(total, i, 0);
        }

        int buf_idx = 0;
        for (int t = 0; t < NUM_TIMESTEPS_IN_PARALLEL + 1; t++) {
            Atom* atom_ = atom_arr[NUM_TIMESTEPS_IN_PARALLEL - t];

            int num_segments = zoid.recv_ghost_num_segments[t][i];
            for (int j = 0; j < num_segments; j++) {
                int segment_size = zoid.recv_ghost_sizes[t][i][j];
                int segment_start_idx = zoid.recv_ghost_idxs[t][i][j];
                for (int k = segment_start_idx; k < segment_start_idx + segment_size; k++) {
                    double d = ubuf(atom_->tag[k]).d;
                    buf_send_stencil_md[i][buf_idx++] = d;
                }
            }
        }

        for (int t = 0; t < NUM_TIMESTEPS_IN_PARALLEL + 1; t++) {
            send_list_sendnum_stencil_md[i][t] = nsend_arr[t];
        }

        // int mpi_tag = (recv_zoid_num << 16 | zoid_num);
        int mpi_tag = get_mpi_tag(recv_zoid_num, zoid_num);

        MPI_Isend(send_list_sendnum_stencil_md[i], NUM_TIMESTEPS_IN_PARALLEL + 1, MPI_INT,
                  recv_zoid_num % comm->nprocs, mpi_tag, world, &r[2 * idx]);
        if (total) {
            MPI_Isend(buf_send_stencil_md[i], total, MPI_DOUBLE, recv_zoid_num % comm->nprocs, mpi_tag,
                      world, &r[2 * idx + 1]);
        }
        idx++;
    }
}

// send my ghost atoms to construct send_list_stencil_md
void CommBrick::construct_send_list_stencil_md_send(
        std::array<Atom *, NUM_TIMESTEPS_IN_PARALLEL + 1> &atom_arr, queue_info &zoid,
        std::vector<MPI_Request>& r) {
    int zoid_num = zoid.num;

    int max_atoms = -1;
    for (int i = 0; i < atom_arr.size(); i++) {
        Atom *atom_ = atom_arr[i];
        int num_atoms = atom_->nlocal + atom_->nghost;
        if (num_atoms > max_atoms) { max_atoms = num_atoms; }
    }

    bool *can_send[NUM_TIMESTEPS_IN_PARALLEL + 1];

    for (int t = 0; t < NUM_TIMESTEPS_IN_PARALLEL + 1; t++) {
        can_send[t] = new bool[max_atoms];
        memset(can_send[t], false, max_atoms);
    }

    for (int i = 0; i < lmp->send_to_neighbors[zoid_num].size(); i++) {
        int send_zoid_num = lmp->send_to_neighbors[zoid_num][i];
        queue_info &send_zoid = lmp->zoid_num_to_zoid[send_zoid_num];

        for (int j = 0; j < atom_arr[0]->nlocal; j++) {
            can_send[0][j] = true;
        }

        for (int t = 1; t < NUM_TIMESTEPS_IN_PARALLEL + 1; t++) {
            for (int j = 0; j < atom_arr[t]->nlocal; j++) {
                can_send[t][j] = true;
            }
        }

        for (int t = 0; t < NUM_TIMESTEPS_IN_PARALLEL + 1; t++) {
            Atom *atom_ = atom_arr[t];

            for (int j = 0; j < atom_->nlocal + atom_->nghost; j++) {
                if (j < atom_->nlocal && t < NUM_TIMESTEPS_IN_PARALLEL) {
                    assert(zoid.atom_idx_mapping[t][j] != -1);
                    can_send[t + 1][zoid.atom_idx_mapping[t][j]] = true;
                } else if (t < NUM_TIMESTEPS_IN_PARALLEL && zoid.can_eval_pos[t][j]) {
                    assert(zoid.atom_idx_mapping[t][j] != -1);
                    can_send[t + 1][zoid.atom_idx_mapping[t][j]] = true;
                }
            }
        }
    }

    // send ghost atoms to local
    std::vector<int>& send_to = lmp->send_to_neighbors[zoid_num];
    int upper_bound = 0;
    for (int t = 0; t < NUM_TIMESTEPS_IN_PARALLEL + 1; t++) {
        upper_bound += atom_arr[t]->nghost;
    }

    std::set<IDX_3D> all_ranges;

    int idx = 0;
    for (int i = 0; i < send_to.size(); i++) {
        int send_zoid_num = send_to[i];
        queue_info& send_zoid = lmp->zoid_num_to_zoid[send_zoid_num];

        if (upper_bound > maxsend_stencil_md[i]) {
            grow_send_stencil_md(upper_bound, i, 0);
        }

        int buf_idx = 0;

        // specify idx'es that will send pos + force, and then idxs that will send force ONLY
        std::vector<int> idx_vec_pos[NUM_TIMESTEPS_IN_PARALLEL + 1];
        std::vector<int> idx_vec_force[NUM_TIMESTEPS_IN_PARALLEL + 1];

        for (int t = 0; t < NUM_TIMESTEPS_IN_PARALLEL + 1; t++) {
            Atom* atom_ = atom_arr[t];

            int nsend_stencil_md = 0;
            for (int j = atom_->nlocal; j < atom_->nlocal + atom_->nghost; j++) {
                double *pos = atom_->x[j];
                bool in_zoid = true;
                for (int dim = 0; dim < domain->dimension; dim++) {
                    double lo = send_zoid.zoid.cuts[dim].lower + (t) *send_zoid.zoid.cuts[dim].slope_lower;
                    double hi = send_zoid.zoid.cuts[dim].upper + (t) *send_zoid.zoid.cuts[dim].slope_upper;

                    double value = pos[dim];

                    int pbc_ = 0;
                    if (zoid.where[dim] == RIGHT && send_zoid.where[dim] == PBC) { pbc_ = -1; }

                    if (zoid.where[dim] == PBC && send_zoid.where[dim] == RIGHT) { pbc_ = 1; }

                    double atom_pos_shifted = value + pbc_ * domain->prd[dim];
                    in_zoid = in_zoid && ((atom_pos_shifted >= lo && atom_pos_shifted < hi));
                }

                bool can_eval_check = true;
                for (int dim = 0; dim < 3; dim++) {
                    bool my_zoid_shrinking_dim = zoid.zoid.cuts[dim].slope_lower > 0;
                    bool send_zoid_shrinking_dim = send_zoid.zoid.cuts[dim].slope_lower > 0;

                    double send_lo = send_zoid.zoid.cuts[dim].lower + t * send_zoid.zoid.cuts[dim].slope_lower;
                    double send_hi = send_zoid.zoid.cuts[dim].upper + t * send_zoid.zoid.cuts[dim].slope_upper;
                    bool send_in_bounds = (atom_->x[j][dim] >= send_lo && atom_->x[j][dim] <= send_hi);

                    double my_lo = zoid.zoid.cuts[dim].lower + t * zoid.zoid.cuts[dim].slope_lower;
                    double my_hi = zoid.zoid.cuts[dim].upper + t * zoid.zoid.cuts[dim].slope_upper;
                    bool my_in_bounds = (atom_->x[j][dim] >= my_lo && atom_->x[j][dim] <= my_hi);

                    double send_diff = std::min(fabs(atom_->x[j][dim] - send_lo), fabs(atom_->x[j][dim] - send_hi));
                    double my_diff = std::min(fabs(atom_->x[j][dim] - my_lo), fabs(atom_->x[j][dim] - my_hi));

                    /*
                    can_eval_center = can_eval_center && (in_bounds || (diff <= ALLEGRO_CUTOFF_RADIUS));
                    can_eval_center_debug[dim] = (in_bounds || (diff <= ALLEGRO_CUTOFF_RADIUS));
                    can_eval_center = can_eval_center && in_bounds && (diff > ALLEGRO_CUTOFF_RADIUS);
                    can_eval_center_debug[dim] = (in_bounds && diff > ALLEGRO_CUTOFF_RADIUS);
                    */
                }

                // atoms that are neighbors of neighbors also need to be send
                // technically always send force?

                if (in_zoid) {
                    if (can_send[t][j]) {
                        idx_vec_pos[t].push_back(j);
                    }

                    if (zoid.relevant_atom_idxs[t].find(j) != zoid.relevant_atom_idxs[t].end()) {
                        idx_vec_force[t].push_back(j);
                    } else {
                        assert(PURELY_LOCAL_POTENTIAL);
                    }
                }
            }

            std::vector<int> segment_idxs_force;
            std::vector<int> segment_lengths_force;
            int num_force_segments = get_segments(idx_vec_force[t], segment_idxs_force, segment_lengths_force);

            auto& bounds = stencilMD->GET_BOUNDS(true, t);

            if (bounds.size() > 0) {
                std::set<IDX_3D> ranges;
                for (int idx : idx_vec_force[t]) {
                    double* pos = atom_->x[idx];
                    auto range = get_bin(bounds, pos, domain->boxlo, domain->boxhi);
                    ranges.insert(range);
                    all_ranges.insert(range);
                }
            }

            if (num_force_segments > 10) {
                std::cout << YELLOW << "curr dt zoid: " << zoid.num << " to: " << send_zoid_num << " time: " << t << " num force segments: " << num_force_segments << RESET_COLOR << std::endl;
            }

            zoid.send_force_num_segments[t][i] = num_force_segments;
            zoid.send_force_idxs[t][i] = new int[num_force_segments];
            zoid.send_force_sizes[t][i] = new int[num_force_segments];

            int total_send_force_to_zoid = 0;
            for (int j = 0; j < num_force_segments; j++) {
                zoid.send_force_idxs[t][i][j] = segment_idxs_force[j];
                zoid.send_force_sizes[t][i][j] = segment_lengths_force[j];

                total_send_force_to_zoid += segment_lengths_force[j];
            }

            zoid.send_force_total_num_elems[t][i] = total_send_force_to_zoid;

            std::vector<int> segment_idxs_pos;
            std::vector<int> segment_lengths_pos;
            int num_pos_segments = get_segments(idx_vec_pos[t], segment_idxs_pos, segment_lengths_pos);

            if (num_pos_segments > 100) {
                std::cout << YELLOW << "curr dt zoid: " << zoid.num << " to: " << send_zoid_num << " time: " << t << " num pos_segments segments: " << num_pos_segments << RESET_COLOR << std::endl;
            }

            int total_send_pos_to_zoid = 0;
            zoid.send_pos_num_segments[t][i] = num_pos_segments;
            zoid.send_pos_idxs[t][i] = new int[num_pos_segments];
            zoid.send_pos_sizes[t][i] = new int[num_pos_segments];

            for (int j = 0; j < num_pos_segments; j++) {
                zoid.send_pos_idxs[t][i][j] = segment_idxs_pos[j];
                zoid.send_pos_sizes[t][i][j] = segment_lengths_pos[j];

                total_send_pos_to_zoid += segment_lengths_pos[j];
            }

            zoid.send_pos_total_num_elems[t][i] = total_send_pos_to_zoid;
            sendnum_stencil_md[t][i] = nsend_stencil_md;
        }

        int total = 0;
        for (int t = 0; t < NUM_TIMESTEPS_IN_PARALLEL + 1; t++) {
            total += idx_vec_force[t].size();
            total += idx_vec_pos[t].size();
        }

        if (total > maxsend_stencil_md[i]) {
            grow_send_stencil_md(total, i, 0);
        }

        for (int t = 0; t < NUM_TIMESTEPS_IN_PARALLEL + 1; t++) {
            Atom* atom_ = atom_arr[t];
            for (int idx_force : idx_vec_force[t]) {
                buf_send_stencil_md[i][buf_idx++] = ubuf(atom_->tag[idx_force]).d;
            }
        }

        for (int t = 0; t < NUM_TIMESTEPS_IN_PARALLEL + 1; t++) {
            Atom* atom_ = atom_arr[t];
            for (int idx_pos : idx_vec_pos[t]) {
                buf_send_stencil_md[i][buf_idx++] = ubuf(atom_->tag[idx_pos]).d;
            }
        }

        for (int t = 0; t < NUM_TIMESTEPS_IN_PARALLEL + 1; t++) {
            send_list_sendnum_stencil_md[i][t] = idx_vec_force[t].size();
        }

        for (int t = 0; t < NUM_TIMESTEPS_IN_PARALLEL + 1; t++) {
            send_list_sendnum_stencil_md[i][t + NUM_TIMESTEPS_IN_PARALLEL + 1] = idx_vec_pos[t].size();
        }

        // (dst << 16 | src)
        // int mpi_tag = (send_zoid_num << 16 | zoid_num);
        int mpi_tag = get_mpi_tag(send_zoid_num, zoid_num);

        MPI_Isend(send_list_sendnum_stencil_md[i], 2 * (NUM_TIMESTEPS_IN_PARALLEL + 1),
                  MPI_INT, send_zoid_num % comm->nprocs, mpi_tag, world, &r[2 * idx]);
        if (total) {
            MPI_Isend(buf_send_stencil_md[i], total, MPI_DOUBLE, send_zoid_num % comm->nprocs, mpi_tag,
                      world, &r[2 * idx + 1]);
        }

        idx++;
    }
}

// send my ghost atoms to construct send_list_stencil_md
void CommBrick::construct_send_list_stencil_md_next_dt_send(
        std::array<Atom *, NUM_TIMESTEPS_IN_PARALLEL + 1> &atom_arr, queue_info &zoid,
        std::vector<MPI_Request>& r) {
    int zoid_num = zoid.num;

    int max_atoms = -1;
    for (int i = 0; i < atom_arr.size(); i++) {
        Atom *atom_ = atom_arr[i];
        int num_atoms = atom_->nlocal + atom_->nghost;
        if (num_atoms > max_atoms) { max_atoms = num_atoms; }
    }

    bool *can_send[NUM_TIMESTEPS_IN_PARALLEL + 1];

    for (int t = 0; t < NUM_TIMESTEPS_IN_PARALLEL + 1; t++) {
        can_send[t] = new bool[max_atoms];
        memset(can_send[t], false, max_atoms);
    }

    std::vector<int>& send_to = lmp->send_to_neighbors_next_dt[zoid_num];
    for (int i = 0; i < send_to.size(); i++) {
        int send_zoid_num = send_to[i];
        queue_info &send_zoid = lmp->zoid_num_to_zoid_next_dt[send_zoid_num];

        for (int j = 0; j < atom_arr[NUM_TIMESTEPS_IN_PARALLEL]->nlocal; j++) {
            can_send[0][j] = true;
        }

        for (int t = 1; t < NUM_TIMESTEPS_IN_PARALLEL + 1; t++) {
            for (int j = 0; j < atom_arr[NUM_TIMESTEPS_IN_PARALLEL - t]->nlocal; j++) {
                can_send[t][j] = true;
            }
        }

        for (int t = 0; t < NUM_TIMESTEPS_IN_PARALLEL + 1; t++) {
            Atom *atom_ = atom_arr[NUM_TIMESTEPS_IN_PARALLEL - t];

            for (int j = 0; j < atom_->nlocal + atom_->nghost; j++) {
                if (j < atom_->nlocal && t < NUM_TIMESTEPS_IN_PARALLEL) {
                    assert(zoid.atom_idx_mapping[t][j] != -1);
                    can_send[t + 1][zoid.atom_idx_mapping[t][j]] = true;
                } else if (t < NUM_TIMESTEPS_IN_PARALLEL && zoid.can_eval_pos[t][j]) {
                    assert(zoid.atom_idx_mapping[t][j] != -1);
                    can_send[t + 1][zoid.atom_idx_mapping[t][j]] = true;
                }
            }
        }
    }

    // send ghost atoms to local
    int upper_bound = 0;
    for (int t = 0; t < NUM_TIMESTEPS_IN_PARALLEL + 1; t++) {
        upper_bound += atom_arr[NUM_TIMESTEPS_IN_PARALLEL - t]->nghost;
    }

    int idx = 0;
    for (int i = 0; i < send_to.size(); i++) {
        int send_zoid_num = send_to[i];
        queue_info& send_zoid = lmp->zoid_num_to_zoid_next_dt[send_zoid_num];

        if (upper_bound > maxsend_stencil_md[i]) {
            grow_send_stencil_md(upper_bound, i, 0);
        }

        int buf_idx = 0;

        // specify idx'es that will send pos + force, and then idxs that will send force ONLY
        std::vector<int> idx_vec_force[NUM_TIMESTEPS_IN_PARALLEL + 1];
        std::vector<int> idx_vec_pos[NUM_TIMESTEPS_IN_PARALLEL + 1];

        for (int t = 0; t < NUM_TIMESTEPS_IN_PARALLEL + 1; t++) {
            Atom* atom_ = atom_arr[NUM_TIMESTEPS_IN_PARALLEL - t];

            int nsend_stencil_md = 0;
            for (int j = atom_->nlocal; j < atom_->nlocal + atom_->nghost; j++) {
                double *pos = atom_->x[j];
                bool in_zoid = true;

                for (int dim = 0; dim < domain->dimension; dim++) {
                    double lo = send_zoid.zoid.cuts[dim].lower + (t) *send_zoid.zoid.cuts[dim].slope_lower;
                    double hi = send_zoid.zoid.cuts[dim].upper + (t) *send_zoid.zoid.cuts[dim].slope_upper;

                    double value = pos[dim];

                    int pbc_ = 0;
                    if (zoid.where[dim] == RIGHT && send_zoid.where[dim] == PBC) { pbc_ = -1; }

                    if (zoid.where[dim] == PBC && send_zoid.where[dim] == RIGHT) { pbc_ = 1; }

                    double atom_pos_shifted = value + pbc_ * domain->prd[dim];
                    in_zoid = in_zoid && ((atom_pos_shifted >= lo && atom_pos_shifted < hi));
                }

                // atoms that are neighbors of neighbors also need to be send
                /*
                if (in_zoid) {
                    if (nsend_stencil_md == maxsendlist_stencil_md_next_dt[t][i]) {
                        grow_list_stencil_md_next_dt(i, nsend_stencil_md, t);
                    }
                    sendlist_stencil_md_next_dt[t][i][nsend_stencil_md] = j;
                    send_force_stencil_md_next_dt[t][i][nsend_stencil_md] = in_zoid;
                    send_pos_stencil_md_next_dt[t][i][nsend_stencil_md] = can_send[t][j];
                    send_vel_stencil_md_next_dt[t][i][nsend_stencil_md] = can_send[t][j];
                    // buf_send_stencil_md[i][buf_idx++] = ubuf(atom_->tag[j]).d;
                    nsend_stencil_md++;

                    idx_vec_force[t].push_back(j);

                    if (can_send[t][j]) {
                        idx_vec_pos[t].push_back(j);
                    }
                }
                */

                if (in_zoid) {
                    if (can_send[t][j]) {
                        idx_vec_pos[t].push_back(j);
                    }

                    if (zoid.relevant_atom_idxs[t].find(j) != zoid.relevant_atom_idxs[t].end()) {
                        idx_vec_force[t].push_back(j);
                    } else {
                        assert(PURELY_LOCAL_POTENTIAL);
                    }
                }
            }

            std::vector<int> segment_idxs_force;
            std::vector<int> segment_lengths_force;
            int num_force_segments = get_segments(idx_vec_force[t], segment_idxs_force, segment_lengths_force);

            if (num_force_segments > 10) {
                std::cout << YELLOW << "next dt zoid: " << zoid.num << " to: " << send_zoid_num << " time: " << t << " num force segments: " << num_force_segments << RESET_COLOR << std::endl;
            }

            zoid.send_force_num_segments[t][i] = num_force_segments;
            zoid.send_force_idxs[t][i] = new int[num_force_segments];
            zoid.send_force_sizes[t][i] = new int[num_force_segments];

            int total_send_force_to_zoid = 0;
            for (int j = 0; j < num_force_segments; j++) {
                zoid.send_force_idxs[t][i][j] = segment_idxs_force[j];
                zoid.send_force_sizes[t][i][j] = segment_lengths_force[j];

                total_send_force_to_zoid += segment_lengths_force[j];
            }

            zoid.send_force_total_num_elems[t][i] = total_send_force_to_zoid;

            std::vector<int> segment_idxs_pos;
            std::vector<int> segment_lengths_pos;
            int num_pos_segments = get_segments(idx_vec_pos[t], segment_idxs_pos, segment_lengths_pos);

            if (num_pos_segments > 100) {
                std::cout << YELLOW << "next dt zoid: " << zoid.num << " to: " << send_zoid_num << " time: " << t << " num pos_segments segments: " << num_pos_segments << RESET_COLOR << std::endl;
            }

            zoid.send_pos_num_segments[t][i] = num_pos_segments;
            zoid.send_pos_idxs[t][i] = new int[num_pos_segments];
            zoid.send_pos_sizes[t][i] = new int[num_pos_segments];

            int total_send_pos_to_zoid = 0;
            for (int j = 0; j < num_pos_segments; j++) {
                zoid.send_pos_idxs[t][i][j] = segment_idxs_pos[j];
                zoid.send_pos_sizes[t][i][j] = segment_lengths_pos[j];
                total_send_pos_to_zoid += segment_lengths_pos[j];
            }

            zoid.send_pos_total_num_elems[t][i] = total_send_pos_to_zoid;
            sendnum_stencil_md_next_dt[t][i] = nsend_stencil_md;
        }

        int total = 0;
        for (int t = 0; t < NUM_TIMESTEPS_IN_PARALLEL + 1; t++) {
            total += idx_vec_force[t].size();
            total += idx_vec_pos[t].size();
        }

        if (total > maxsend_stencil_md[i]) {
            grow_send_stencil_md(total, i, 0);
        }

        for (int t = 0; t < NUM_TIMESTEPS_IN_PARALLEL + 1; t++) {
            Atom* atom_ = atom_arr[NUM_TIMESTEPS_IN_PARALLEL - t];
            for (int idx_force_only : idx_vec_force[t]) {
                buf_send_stencil_md[i][buf_idx++] = ubuf(atom_->tag[idx_force_only]).d;
            }
        }

        for (int t = 0; t < NUM_TIMESTEPS_IN_PARALLEL + 1; t++) {
            Atom* atom_ = atom_arr[NUM_TIMESTEPS_IN_PARALLEL - t];
            for (int idx_force_pos : idx_vec_pos[t]) {
                buf_send_stencil_md[i][buf_idx++] = ubuf(atom_->tag[idx_force_pos]).d;
            }
        }

        for (int t = 0; t < NUM_TIMESTEPS_IN_PARALLEL + 1; t++) {
            send_list_sendnum_stencil_md[i][t] = idx_vec_force[t].size();
        }

        for (int t = 0; t < NUM_TIMESTEPS_IN_PARALLEL + 1; t++) {
            send_list_sendnum_stencil_md[i][t + NUM_TIMESTEPS_IN_PARALLEL + 1] = idx_vec_pos[t].size();
        }

        // (dst << 16 | src)
        // int mpi_tag = (send_zoid_num << 16 | zoid_num);
        int mpi_tag = get_mpi_tag(send_zoid_num, zoid_num);

        MPI_Isend(send_list_sendnum_stencil_md[i], 2 * (NUM_TIMESTEPS_IN_PARALLEL + 1),
                  MPI_INT, send_zoid_num % comm->nprocs, mpi_tag, world, &r[2 * idx]);

        if (total) {
            MPI_Isend(buf_send_stencil_md[i], total, MPI_DOUBLE, send_zoid_num % comm->nprocs, mpi_tag,
                      world, &r[2 * idx + 1]);
        }
        idx++;
    }
}

// TODO: right now this is sending local to local
void CommBrick::construct_send_list_stencil_md(
    std::array<Atom *, NUM_TIMESTEPS_IN_PARALLEL + 1> &atom_arr, queue_info &zoid) {

  int zoid_num = zoid.num;
  std::vector<int>& recv_from = lmp->recv_from_neighbors[zoid_num];

  for (int i = 0; i < recv_from.size(); i++) {
    int recv_zoid_num = recv_from[i];
    int nrecv_arr[2 * (NUM_TIMESTEPS_IN_PARALLEL + 1)] = {0};

    // (dst << 16 | src)
    // int mpi_tag = (zoid_num << 16 | recv_zoid_num);
    int mpi_tag = get_mpi_tag(zoid_num, recv_zoid_num);

    MPI_Recv(nrecv_arr, 2 * (NUM_TIMESTEPS_IN_PARALLEL + 1), MPI_INT, recv_zoid_num % comm->nprocs,
             mpi_tag, world, MPI_STATUS_IGNORE);

    int total = 0;
    for (int t = 0; t < 2 * (NUM_TIMESTEPS_IN_PARALLEL + 1); t++) {
        assert(nrecv_arr[t] >= 0);
        total += nrecv_arr[t];
    }

    if (total > maxrecv_stencil_md[i]) {
        grow_recv_stencil_md(total, i);
    }

    if (total) {
        MPI_Recv(buf_recv_stencil_md[i], total, MPI_DOUBLE, recv_zoid_num % comm->nprocs,
                 mpi_tag, world,MPI_STATUS_IGNORE);

        int idx_in_buf = 0;
        for (int t = 0; t < NUM_TIMESTEPS_IN_PARALLEL + 1; t++) {
            Atom* atom_ = atom_arr[t];
            // int nsend_stencil_md = 0;
            int nrecv_force_only = nrecv_arr[t];
            zoid.recv_list_local_force_only[t][i] = new int[nrecv_force_only];
            zoid.recv_list_local_num_force_only[t][i] = nrecv_force_only;

            auto& bounds = stencilMD->GET_BOUNDS(true, t);
            std::set<IDX_3D> ranges;

            for (int j = 0; j < nrecv_force_only; j++) {
                tagint tag_ = (tagint) ubuf(buf_recv_stencil_md[i][idx_in_buf++]).i;
                assert(tag_ >= 0 && tag_ <= atom->natoms);
                if (!atom_->tag_to_idx.count(tag_)) {
                    std::cout << "zoid num: " << zoid_num << " recv from: " << recv_zoid_num << " tag: " << tag_ << " timestep: " << t << std::endl;
                }
                assert(atom_->tag_to_idx.count(tag_));
                int idx = atom_->tag_to_idx[tag_];
                assert(idx < atom_->nlocal);
                zoid.recv_list_local_force_only[t][i][j] = idx;

                if (t != 0 && t < NUM_TIMESTEPS_IN_PARALLEL + 1) {
                    double* pos = atom_->x[idx];
                    auto range = get_bin(bounds, pos, domain->boxlo, domain->boxhi);
                    ranges.insert(range);
                }
            }

            int num_local_in_range = 0;
            for (auto& range : ranges) {
                assert(bounds.size() > 0);
                for (int local_idx = 0; local_idx < atom_->nlocal; local_idx++) {
                    double *pos = atom_->x[local_idx];
                    auto compare = get_bin(bounds, pos, domain->boxlo, domain->boxhi);
                    if (range == compare) {
                        num_local_in_range++;
                    } else {
                        // std::cout << "range: " << std::get<0>(range) << " " << std::get<1>(range) << " " << std::get<2>(range) << " compare: " << std::get<0>(compare) << " " << std::get<1>(compare) << " " << std::get<2>(compare) << std::endl;
                    }
                }
            }
            // assert(num_local_in_range == nrecv_force_only);
        }


        for (int t = 0; t < NUM_TIMESTEPS_IN_PARALLEL + 1; t++) {
            Atom* atom_ = atom_arr[t];

            int nrecv_force_pos = nrecv_arr[t + NUM_TIMESTEPS_IN_PARALLEL + 1];
            zoid.recv_list_local_force_pos[t][i] = new int[nrecv_force_pos];
            zoid.recv_list_local_num_force_pos[t][i] = nrecv_force_pos;

            auto& bounds = stencilMD->GET_BOUNDS(true, t);
            std::set<IDX_3D> ranges;

            for (int j = 0; j < nrecv_force_pos; j++) {
                tagint tag_ = (tagint) ubuf(buf_recv_stencil_md[i][idx_in_buf++]).i;
                assert(tag_ >= 0 && tag_ <= atom->natoms);
                if (!atom_->tag_to_idx.count(tag_)) {
                    std::cout << "zoid num: " << zoid_num << " recv from: " << recv_zoid_num << " tag: " << tag_ << " timestep: " << t << std::endl;
                }
                assert(atom_->tag_to_idx.count(tag_));
                int idx = atom_->tag_to_idx[tag_];
                if (idx >= atom_->nlocal) {
                    std::cout << RED << "zoid num: " << zoid_num << " recv from: " << recv_zoid_num << " t: " << t << " tag: " << tag_ << " idx: " << idx << " nlocal: " << atom_->nlocal << RESET_COLOR << std::endl;
                }
                assert(idx < atom_->nlocal);
                zoid.recv_list_local_force_pos[t][i][j] = idx;

                if (t != 0 && t < NUM_TIMESTEPS_IN_PARALLEL + 1) {
                    double *pos = atom_->x[idx];
                    auto range = get_bin(bounds, pos, domain->boxlo, domain->boxhi);
                    ranges.insert(range);
                }
            }

            /*
            if (ranges.size() == 1) {
                assert(bounds.size() > 0);
                auto& range = *(ranges.begin());
                int num_local_in_range = 0;
                for (int local_idx = 0; local_idx < atom_->nlocal; local_idx++) {
                    double* pos = atom_->x[local_idx];
                    auto compare = get_bin(bounds, pos, domain->boxlo, domain->boxhi);
                    if (range == compare) {
                        num_local_in_range++;
                    }
                }
                assert(num_local_in_range == nrecv_force_pos);
            } else {
            }
            */
        }
    }
  }
}

// ghost to local for next dt's perspective
void CommBrick::construct_send_list_stencil_md_next_dt(
        std::array<Atom *, NUM_TIMESTEPS_IN_PARALLEL + 1> &atom_arr, queue_info &zoid) {

    int zoid_num = zoid.num;
    std::vector<int>& recv_from = lmp->recv_from_neighbors_next_dt[zoid_num];

    for (int i = 0; i < recv_from.size(); i++) {
        int recv_zoid_num = recv_from[i];
        int nrecv_arr[2 * (NUM_TIMESTEPS_IN_PARALLEL + 1)] = {0};

        // int mpi_tag = (zoid_num << 16 | recv_zoid_num);
        int mpi_tag = get_mpi_tag(zoid_num, recv_zoid_num);

        MPI_Recv(nrecv_arr, 2 * (NUM_TIMESTEPS_IN_PARALLEL + 1), MPI_INT, recv_zoid_num % comm->nprocs,
                 mpi_tag, world, MPI_STATUS_IGNORE);

        int total = 0;
        for (int t = 0; t < 2 * (NUM_TIMESTEPS_IN_PARALLEL + 1); t++) {
            assert(nrecv_arr[t] >= 0);
            total += nrecv_arr[t];
        }

        if (total > maxrecv_stencil_md[i]) {
            grow_recv_stencil_md(total, i);
        }

        if (total) {
            MPI_Recv(buf_recv_stencil_md[i], total, MPI_DOUBLE, recv_zoid_num % comm->nprocs,
                     mpi_tag, world,MPI_STATUS_IGNORE);

            int idx_in_buf = 0;

            for (int t = 0; t < NUM_TIMESTEPS_IN_PARALLEL + 1; t++) {
                Atom* atom_ = atom_arr[NUM_TIMESTEPS_IN_PARALLEL - t];
                int nrecv_force_only = nrecv_arr[t];
                zoid.recv_list_local_force_only[t][i] = new int[nrecv_force_only];
                zoid.recv_list_local_num_force_only[t][i] = nrecv_force_only;

                for (int j = 0; j < nrecv_force_only; j++) {
                    tagint tag_ = (tagint) ubuf(buf_recv_stencil_md[i][idx_in_buf++]).i;
                    assert(tag_ >= 0 && tag_ <= atom->natoms);
                    if (!atom_->tag_to_idx.count(tag_)) {
                        std::cout << "zoid num: " << zoid_num << " recv from: " << recv_zoid_num << " tag: " << tag_ << " timestep: " << t << std::endl;
                    }
                    assert(atom_->tag_to_idx.count(tag_));
                    int idx = atom_->tag_to_idx[tag_];
                    assert(idx < atom_->nlocal);
                    zoid.recv_list_local_force_only[t][i][j] = idx;
                }
            }

            for (int t = 0; t < NUM_TIMESTEPS_IN_PARALLEL + 1; t++) {
                Atom* atom_ = atom_arr[NUM_TIMESTEPS_IN_PARALLEL - t];

                int nrecv_force_pos = nrecv_arr[t + NUM_TIMESTEPS_IN_PARALLEL + 1];
                zoid.recv_list_local_force_pos[t][i] = new int[nrecv_force_pos];
                zoid.recv_list_local_num_force_pos[t][i] = nrecv_force_pos;
                for (int j = 0; j < nrecv_force_pos; j++) {
                    tagint tag_ = (tagint) ubuf(buf_recv_stencil_md[i][idx_in_buf++]).i;
                    assert(tag_ >= 0 && tag_ <= atom->natoms);
                    if (!atom_->tag_to_idx.count(tag_)) {
                        std::cout << "zoid num: " << zoid_num << " recv from: " << recv_zoid_num << " tag: " << tag_ << " timestep: " << t << std::endl;
                    }
                    assert(atom_->tag_to_idx.count(tag_));
                    int idx = atom_->tag_to_idx[tag_];
                    if (idx >= atom_->nlocal) {
                        std::cout << RED << "zoid num: " << zoid_num << " recv from: " << recv_zoid_num << " t: " << t << " tag: " << tag_ << " idx: " << idx << " nlocal: " << atom_->nlocal << RESET_COLOR << std::endl;
                    }
                    assert(idx < atom_->nlocal);
                    zoid.recv_list_local_force_pos[t][i][j] = idx;
                }
            }
        }
    }
}

// sendlist for sending positions of my atoms that are ghost atoms to other zoids
// receive ghost atoms? construct local atoms?
void CommBrick::construct_second_send_list_stencil_md(
        std::array<Atom *, NUM_TIMESTEPS_IN_PARALLEL + 1> &atom_arr, queue_info &zoid) {

    int zoid_num = zoid.num;
    auto& send_to = lmp->send_to_neighbors[zoid_num];

    int max_atoms = -1;
    for (int i = 0; i < atom_arr.size(); i++) {
        Atom *atom_ = atom_arr[i];
        int num_atoms = atom_->nlocal + atom_->nghost;
        if (num_atoms > max_atoms) { max_atoms = num_atoms; }
    }

    bool *can_send[NUM_TIMESTEPS_IN_PARALLEL + 1];

    for (int t = 0; t < NUM_TIMESTEPS_IN_PARALLEL + 1; t++) {
        can_send[t] = new bool[max_atoms];
        memset(can_send[t], false, max_atoms * sizeof(bool));
    }

    for (int i = 0; i < lmp->send_to_neighbors[zoid_num].size(); i++) {
        int send_zoid_num = lmp->send_to_neighbors[zoid_num][i];

        // assumed all my atoms are up-to-date in position
        for (int j = 0; j < atom_arr[0]->nlocal; j++) {
            can_send[0][j] = true;
        }

        for (int t = 1; t < NUM_TIMESTEPS_IN_PARALLEL + 1; t++) {
            for (int j = 0; j < atom_arr[t]->nlocal; j++) {
                can_send[t][j] = true;
            }
        }

        for (int t = 0; t < NUM_TIMESTEPS_IN_PARALLEL + 1; t++) {
            Atom *atom_ = atom_arr[t];

            for (int j = 0; j < atom_->nlocal + atom_->nghost; j++) {
                if (j < atom_->nlocal && t < NUM_TIMESTEPS_IN_PARALLEL) {
                    if (zoid.atom_idx_mapping[t][j] == -1) {
                        std::cout << "zoid: " << zoid_num << " t: " << t << " idx: " << j << " out of: " << atom_->nlocal << " tag: " << atom_->tag[j]
                            << " pos: " << atom_->x[j][0] << " " << atom_->x[j][1] << " " << atom_->x[j][2] << std::endl;
                    }
                    assert(zoid.atom_idx_mapping[t][j] != -1);
                    can_send[t + 1][zoid.atom_idx_mapping[t][j]] = true;
                } else if (t < NUM_TIMESTEPS_IN_PARALLEL && zoid.can_eval_pos[t][j]) {
                    assert(zoid.atom_idx_mapping[t][j] != -1);
                    can_send[t + 1][zoid.atom_idx_mapping[t][j]] = true;
                }
            }
        }
    }

    int total_num_segments = 0;

    for (int i = 0; i < send_to.size(); i++) {
        int send_zoid_num = send_to[i];
        int nrecv_arr[NUM_TIMESTEPS_IN_PARALLEL + 1] = {0};

        // int mpi_tag = (zoid_num << 16 | send_zoid_num);
        int mpi_tag = get_mpi_tag(zoid_num, send_zoid_num);

        MPI_Recv(nrecv_arr, NUM_TIMESTEPS_IN_PARALLEL + 1, MPI_INT, send_zoid_num % comm->nprocs,
                 mpi_tag, world, MPI_STATUS_IGNORE);

        int total = 0;
        for (int t = 0; t < NUM_TIMESTEPS_IN_PARALLEL + 1; t++) {
            assert(nrecv_arr[t] >= 0);
            total += nrecv_arr[t];
        }

        if (total > maxrecv_stencil_md[i]) {
            grow_recv_stencil_md(total, i);
        }

        if (total) {
            MPI_Recv(buf_recv_stencil_md[i], total, MPI_DOUBLE, send_zoid_num % comm->nprocs,
                     mpi_tag, world,MPI_STATUS_IGNORE);

            int idx_in_buf = 0;
            for (int t = 0; t < NUM_TIMESTEPS_IN_PARALLEL + 1; t++) {
                Atom* atom_ = atom_arr[t];
                int nsend_stencil_md = 0;

                std::vector<bool> new_segment_types;
                std::vector<int> new_segment_sizes;
                std::vector<int> new_segment_idxs;

                std::vector<int> local_idxs;
                std::vector<int> ghost_idxs;

                int num_curr_local = 0;
                std::vector<int> curr_ghost_idxs_new;

                for (int j = 0; j < nrecv_arr[t]; j++) {
                    tagint tag_ = (tagint) ubuf(buf_recv_stencil_md[i][idx_in_buf++]).i;
                    assert(atom_->tag_to_idx.count(tag_));
                    int idx = atom_->tag_to_idx[tag_];
                    assert(can_send[t][idx]);

                    if (idx < atom_->nlocal) {
                        local_idxs.push_back(idx);
                        num_curr_local++;
                        if (curr_ghost_idxs_new.size() > 0) {
                            std::vector<int> idxs;
                            std::vector<int> sizes;
                            int num_ghost_segments = get_segments(curr_ghost_idxs_new, idxs, sizes);
                            for (int ghost_segment_idx : idxs) {
                                new_segment_types.push_back(GHOST_SEGMENT_TYPE);
                            }
                            for (int ghost_segment_idx : idxs) {
                                new_segment_idxs.push_back(ghost_segment_idx);
                            }
                            for (int ghost_segment_size : sizes) {
                                new_segment_sizes.push_back(ghost_segment_size);
                            }
                            curr_ghost_idxs_new.clear();
                        }
                    } else {
                        ghost_idxs.push_back(idx);
                        curr_ghost_idxs_new.push_back(idx);
                        if (num_curr_local > 0) {
                            new_segment_types.push_back(LOCAL_SEGMENT_TYPE);
                            new_segment_idxs.push_back(-1);
                            new_segment_sizes.push_back(num_curr_local);
                            num_curr_local = 0;
                        }
                    }
                }

                if (num_curr_local > 0) {
                    new_segment_types.push_back(LOCAL_SEGMENT_TYPE);
                    new_segment_idxs.push_back(-1);
                    new_segment_sizes.push_back(num_curr_local);
                    if (curr_ghost_idxs_new.size() > 0) {
                        assert(false);
                    }
                }

                if (curr_ghost_idxs_new.size() > 0) {
                    if (num_curr_local > 0) {
                        assert(false);
                    }
                    std::vector<int> idxs;
                    std::vector<int> sizes;
                    int num_ghost_segments = get_segments(curr_ghost_idxs_new, idxs, sizes);
                    for (int ghost_segment_idx : idxs) {
                        new_segment_types.push_back(GHOST_SEGMENT_TYPE);
                    }
                    for (int ghost_segment_idx : idxs) {
                        new_segment_idxs.push_back(ghost_segment_idx);
                    }
                    for (int ghost_segment_size : sizes) {
                        new_segment_sizes.push_back(ghost_segment_size);
                    }
                }

                int num_local_segments = 0;
                for (int j = 0; j < new_segment_types.size(); j++) {
                    if (new_segment_types[j] == LOCAL_SEGMENT_TYPE) {
                        num_local_segments++;
                    }
                }

                int num_segments = new_segment_types.size();
                int num_local = local_idxs.size();

                zoid.send_num_segments[t][i] = num_segments;
                zoid.send_local_list[t][i] = new int[num_local];
                for (int j = 0; j < num_local; j++) {
                    zoid.send_local_list[t][i][j] = local_idxs[j];
                }

                zoid.send_segment_sizes[t][i] = new int[num_segments];
                zoid.send_segment_types[t][i] = new bool[num_segments];
                zoid.send_segment_idxs[t][i] = new int[num_segments];

                for (int j = 0; j < num_segments; j++) {
                    zoid.send_segment_types[t][i][j] = new_segment_types[j];
                    zoid.send_segment_sizes[t][i][j] = new_segment_sizes[j];
                    zoid.send_segment_idxs[t][i][j] = new_segment_idxs[j];
                }

                int debug_num_local = 0;
                int debug_num_local_segments = 0;
                for (int j = 0; j < num_segments; j++) {
                    if (new_segment_types[j] == LOCAL_SEGMENT_TYPE) {
                        debug_num_local += new_segment_sizes[j];
                        debug_num_local_segments++;
                    }
                }

                if (debug_num_local != local_idxs.size()) {
                    std::cout << "zoid: " << zoid_num << " send to: " << send_zoid_num << " time: " << t << " num local from segments: " << debug_num_local << " num local idxs: " << local_idxs.size() << std::endl;
                    std::cout << "num segments: " << num_segments << " num local segments: " << debug_num_local_segments << std::endl;
                }
                assert(debug_num_local == local_idxs.size());

            }
        } else {
            for (int t = 0; t < NUM_TIMESTEPS_IN_PARALLEL + 1; t++) {
                zoid.send_num_segments[t][i] = 0;
            }
        }
    }
}

// sendlist for sending positions of my atoms that are ghost atoms to other zoids
void CommBrick::construct_second_send_list_stencil_md_next_dt(
    std::array<Atom *, NUM_TIMESTEPS_IN_PARALLEL + 1> &atom_arr, queue_info &zoid) {

    int zoid_num = zoid.num;
    auto& send_to = lmp->send_to_neighbors_next_dt[zoid_num];

    int max_atoms = -1;
    for (int i = 0; i < atom_arr.size(); i++) {
        Atom *atom_ = atom_arr[i];
        int num_atoms = atom_->nlocal + atom_->nghost;
        if (num_atoms > max_atoms) { max_atoms = num_atoms; }
    }

    bool *can_send[NUM_TIMESTEPS_IN_PARALLEL + 1];

    for (int t = 0; t < NUM_TIMESTEPS_IN_PARALLEL + 1; t++) {
        can_send[t] = new bool[max_atoms];
        memset(can_send[t], false, max_atoms * sizeof(bool));
    }

    for (int i = 0; i < send_to.size(); i++) {
        int send_zoid_num = send_to[i];

        // assumed all my atoms are up-to-date in position
        for (int j = 0; j < atom_arr[NUM_TIMESTEPS_IN_PARALLEL]->nlocal; j++) {
            can_send[0][j] = true;
        }

        for (int t = 1; t < NUM_TIMESTEPS_IN_PARALLEL + 1; t++) {
            for (int j = 0; j < atom_arr[NUM_TIMESTEPS_IN_PARALLEL - t]->nlocal; j++) {
                can_send[t][j] = true;
            }
        }

        for (int t = 0; t < NUM_TIMESTEPS_IN_PARALLEL + 1; t++) {
            Atom *atom_ = atom_arr[NUM_TIMESTEPS_IN_PARALLEL - t];

            for (int j = 0; j < atom_->nlocal + atom_->nghost; j++) {
                if (j < atom_->nlocal && t < NUM_TIMESTEPS_IN_PARALLEL) {
                    if (zoid.atom_idx_mapping[t][j] == -1) {
                        std::cout << "zoid: " << zoid_num << " t: " << t << " idx: " << j << " out of: " << atom_->nlocal << " tag: " << atom_->tag[j]
                                  << " pos: " << atom_->x[j][0] << " " << atom_->x[j][1] << " " << atom_->x[j][2] << std::endl;
                    }
                    assert(zoid.atom_idx_mapping[t][j] != -1);
                    can_send[t + 1][zoid.atom_idx_mapping[t][j]] = true;
                } else if (t < NUM_TIMESTEPS_IN_PARALLEL && zoid.can_eval_pos[t][j]) {
                    assert(zoid.atom_idx_mapping[t][j] != -1);
                    can_send[t + 1][zoid.atom_idx_mapping[t][j]] = true;
                }
            }
        }
    }

    int total_num_segments = 0;

    for (int i = 0; i < send_to.size(); i++) {
        int send_zoid_num = send_to[i];
        int nrecv_arr[NUM_TIMESTEPS_IN_PARALLEL + 1] = {0};
        // int mpi_tag = (zoid_num << 16 | send_zoid_num);
        int mpi_tag = get_mpi_tag(zoid_num, send_zoid_num);
        MPI_Recv(nrecv_arr, NUM_TIMESTEPS_IN_PARALLEL + 1, MPI_INT, send_zoid_num % comm->nprocs,
                 mpi_tag, world, MPI_STATUS_IGNORE);


        int total = 0;
        for (int t = 0; t < NUM_TIMESTEPS_IN_PARALLEL + 1; t++) {
            assert(nrecv_arr[t] >= 0);
            total += nrecv_arr[t];
        }

        if (total > maxrecv_stencil_md[i]) {
            grow_recv_stencil_md(total, i);
        }

        if (total) {
            MPI_Recv(buf_recv_stencil_md[i], total, MPI_DOUBLE, send_zoid_num % comm->nprocs,
                     mpi_tag, world,MPI_STATUS_IGNORE);

            int idx_in_buf = 0;
            for (int t = 0; t < NUM_TIMESTEPS_IN_PARALLEL + 1; t++) {
                Atom* atom_ = atom_arr[NUM_TIMESTEPS_IN_PARALLEL - t];
                int nsend_stencil_md = 0;

                std::vector<int> new_segment_types;
                std::vector<int> new_segment_sizes;
                std::vector<int> new_segment_idxs;

                std::vector<int> local_idxs;
                std::vector<int> ghost_idxs;

                int num_curr_local = 0;
                std::vector<int> curr_ghost_idxs_new;

                for (int j = 0; j < nrecv_arr[t]; j++) {
                    tagint tag_ = (tagint) ubuf(buf_recv_stencil_md[i][idx_in_buf++]).i;
                    assert(atom_->tag_to_idx.count(tag_));
                    int idx = atom_->tag_to_idx[tag_];
                    assert(can_send[t][idx]);

                    if (idx < atom_->nlocal) {
                        local_idxs.push_back(idx);
                        num_curr_local++;
                        if (curr_ghost_idxs_new.size() > 0) {
                            std::vector<int> idxs;
                            std::vector<int> sizes;
                            int num_ghost_segments = get_segments(curr_ghost_idxs_new, idxs, sizes);
                            for (int ghost_segment_idx : idxs) {
                                new_segment_types.push_back(GHOST_SEGMENT_TYPE);
                            }

                            for (int ghost_segment_idx : idxs) {
                                new_segment_idxs.push_back(ghost_segment_idx);
                            }

                            for (int ghost_segment_size : sizes) {
                                new_segment_sizes.push_back(ghost_segment_size);
                            }

                            curr_ghost_idxs_new.clear();
                        }
                    } else {
                        ghost_idxs.push_back(idx);
                        curr_ghost_idxs_new.push_back(idx);
                        if (num_curr_local > 0) {
                            new_segment_types.push_back(LOCAL_SEGMENT_TYPE);
                            new_segment_idxs.push_back(-1);
                            new_segment_sizes.push_back(num_curr_local);
                            num_curr_local = 0;
                        }
                    }
                }

                if (num_curr_local > 0) {
                    new_segment_types.push_back(LOCAL_SEGMENT_TYPE);
                    new_segment_idxs.push_back(-1);
                    new_segment_sizes.push_back(num_curr_local);
                    if (curr_ghost_idxs_new.size() > 0) {
                        assert(false);
                    }
                }

                if (curr_ghost_idxs_new.size() > 0) {
                    if (num_curr_local > 0) {
                        assert(false);
                    }
                    std::vector<int> idxs;
                    std::vector<int> sizes;
                    int num_ghost_segments = get_segments(curr_ghost_idxs_new, idxs, sizes);
                    for (int ghost_segment_idx : idxs) {
                        new_segment_types.push_back(GHOST_SEGMENT_TYPE);
                    }
                    for (int ghost_segment_idx : idxs) {
                        new_segment_idxs.push_back(ghost_segment_idx);
                    }
                    for (int ghost_segment_size : sizes) {
                        new_segment_sizes.push_back(ghost_segment_size);
                    }
                }

                int num_segments = new_segment_types.size();
                int num_local = local_idxs.size();

                zoid.send_num_segments[t][i] = num_segments;
                zoid.send_local_list[t][i] = new int[num_local];
                for (int j = 0; j < num_local; j++) {
                    zoid.send_local_list[t][i][j] = local_idxs[j];
                }

                zoid.send_segment_sizes[t][i] = new int[num_segments];
                zoid.send_segment_types[t][i] = new bool[num_segments];
                zoid.send_segment_idxs[t][i] = new int[num_segments];

                for (int j = 0; j < num_segments; j++) {
                    zoid.send_segment_types[t][i][j] = new_segment_types[j];
                    zoid.send_segment_sizes[t][i][j] = new_segment_sizes[j];
                    zoid.send_segment_idxs[t][i][j] = new_segment_idxs[j];
                }

                int debug_num_local = 0;
                int debug_num_local_segments = 0;
                for (int j = 0; j < num_segments; j++) {
                    if (new_segment_types[j] == LOCAL_SEGMENT_TYPE) {
                        debug_num_local += new_segment_sizes[j];
                        debug_num_local_segments++;
                    }
                }

                if (debug_num_local != local_idxs.size()) {
                    std::cout << "zoid: " << zoid_num << " send to: " << send_zoid_num << " time: " << t << " num local from segments: " << debug_num_local << " num local idxs: " << local_idxs.size() << std::endl;
                    std::cout << "num segments: " << num_segments << " num local segments: " << debug_num_local_segments << std::endl;
                }
                assert(debug_num_local == local_idxs.size());
            }
        } else {
            for (int t = 0; t < NUM_TIMESTEPS_IN_PARALLEL + 1; t++) {
                zoid.send_num_segments[t][i] = 0;
            }
        }
    }
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

void CommBrick::send_data_stencil_md(std::array<Atom *, NUM_TIMESTEPS_IN_PARALLEL + 1> &atom_arr,
                                     queue_info &zoid, std::vector<MPI_Request>& send_requests) {
  // TODO: checking send data for last timestep
  int zoid_num = zoid.num;
  int sz = 12;

  auto &send_to_neighbors = lmp->send_to_neighbors[zoid_num];

  for (int i = 0; i < send_to_neighbors.size(); i++) {
    int send_zoid_num = send_to_neighbors[i];
    queue_info &send_zoid = lmp->zoid_num_to_zoid[send_zoid_num];
    int pbc_flag_[3] = {0};
    for (int dim = 0; dim < 3; dim++) {
      if (zoid.where[dim] == RIGHT && send_zoid.where[dim] == PBC) { pbc_flag_[dim] = -1; }

      if (zoid.where[dim] == PBC && send_zoid.where[dim] == RIGHT) { pbc_flag_[dim] = 1; }
    }

    int neighbor_process = send_zoid_num % comm->nprocs;

    /*
    int num_send[NUM_TIMESTEPS_IN_PARALLEL + 1] = {0};

    // figure out nsend
    int num_elems = 0;

    int num_elems_send[NUM_TIMESTEPS_IN_PARALLEL + 1] = {0};
    for (int t = 0; t < NUM_TIMESTEPS_IN_PARALLEL + 1; t++) {
        int num_elems_timestep = 0;
        if (DEBUG_SEND_RECV_DATA) {
          // 3 elems for data, 1 for the debug tag
          int num_force_sizes = 0;
          for (int j = 0; j < zoid.send_force_num_segments[t][i]; j++) {
              num_elems_timestep += zoid.send_force_sizes[t][i][j] * (3 + 1);
              num_force_sizes += zoid.send_force_sizes[t][i][j];
          }

          int num_pos_sizes = 0;
          for (int j = 0; j < zoid.send_pos_num_segments[t][i]; j++) {
              num_elems_timestep += zoid.send_pos_sizes[t][i][j] * 2 * (3 + 1);
              num_pos_sizes += zoid.send_pos_sizes[t][i][j];
          }

          int num_local_sizes = 0;
          int num_ghost_sizes = 0;

          for (int j = 0; j < zoid.send_num_segments[t][i]; j++) {
              int segment_type = zoid.send_segment_types[t][i][j];
              if (segment_type == LOCAL_SEGMENT_TYPE) {
                  num_local_sizes += zoid.send_segment_sizes[t][i][j];
              } else {
                  assert(segment_type == GHOST_SEGMENT_TYPE);
                  num_ghost_sizes += zoid.send_segment_sizes[t][i][j];
              }
          }

          num_elems_timestep += (num_local_sizes + num_ghost_sizes) * 2 * (3 + 1);

          std::cout << YELLOW << "zoid: " << zoid_num << " sending to: " << send_zoid_num << " time: " << t
            << " num force: " << num_force_sizes << " num force segments: " << zoid.send_force_num_segments[t][i]
            << " num pos: " << num_pos_sizes << " num pos segments: " << zoid.send_pos_num_segments[t][i]
            << " num local to ghost: " << num_local_sizes << " num local segments: " << zoid.send_local_num_segments[t][i]
            << " num ghost to ghost: " << num_ghost_sizes << " num ghost segments: " << zoid.send_ghost_num_segments[t][i]
            << RESET_COLOR << std::endl;

          num_elems_send[t] = num_elems_timestep;
      }
    }

    int total_elems_calc = 0;
    for (int t = 0; t < NUM_TIMESTEPS_IN_PARALLEL + 1; t++) {
        total_elems_calc += num_elems_send[t];
    }

    if (total_elems_calc >= maxsend_stencil_md[i]) {
        grow_send_stencil_md(total_elems_calc, i, 0);
    }
    */

    int nsend = 0;
    for (int t = 0; t < NUM_TIMESTEPS_IN_PARALLEL + 1; t++) {
        assert(zoid.num_elems_send[t][i] >= 0);
        nsend += zoid.num_elems_send[t][i];
    }

    if (nsend > maxsend_stencil_md[i]) {
        grow_send_stencil_md(nsend, i, 0);
    }

    int buf_idx = 0;
    for (int t = 0; t < NUM_TIMESTEPS_IN_PARALLEL + 1; t++) {
      Atom *atom_ = atom_arr[t];

      bool debug = false;
      int n = atom_->avec->pack_data_stencil_md(
              zoid.send_force_num_segments[t][i], zoid.send_pos_num_segments[t][i],
              zoid.send_force_idxs[t][i], zoid.send_force_sizes[t][i],
              zoid.send_pos_idxs[t][i], zoid.send_pos_sizes[t][i],
              zoid.send_local_list[t][i],
              zoid.send_num_segments[t][i], zoid.send_segment_types[t][i], zoid.send_segment_idxs[t][i], zoid.send_segment_sizes[t][i],
              &buf_send_stencil_md[i][buf_idx], pbc_flag_, debug
              /*
              zoid.send_local_num_segments[t][i], zoid.send_local_ghost_segments_mapping[t][i], zoid.send_local_sizes[t][i], zoid.send_local_list[t][i],
              zoid.send_ghost_num_segments[t][i], zoid.send_ghost_idxs[t][i], zoid.send_ghost_sizes[t][i],
              &buf_send_stencil_md[i][buf_idx], pbc_flag_, debug
              */
          );

      buf_idx += n;
      // send_list_sendnum_stencil_md[i][t] = n;
      if (n != zoid.num_elems_send[t][i]) {
          std::cout << YELLOW << "zoid: " << zoid_num << " send to: " << send_zoid_num << " time: " << t << " what I have: " << zoid.num_elems_send[t][i] << " what I got: " << n << RESET_COLOR << std::endl;
      }
      assert(n == zoid.num_elems_send[t][i]);
    }

    if (neighbor_process != comm->me) {
      MPI_Request r2;
      // dst << 16 | src
      // int mpi_tag = (send_zoid_num << 16 | zoid_num);
      int mpi_tag = get_mpi_tag(send_zoid_num, zoid_num);
      /*
      MPI_Isend(send_list_sendnum_stencil_md[i], NUM_TIMESTEPS_IN_PARALLEL + 1, MPI_INT, neighbor_process, mpi_tag,
                world, &r1);
      */
      MPI_Isend(buf_send_stencil_md[i], buf_idx, MPI_DOUBLE, neighbor_process, mpi_tag, world,
                  &r2);
      send_requests.push_back(r2);
    } else {
        auto& send_atom_arr = lmp->atom_stencil_md[send_zoid_num];
        int send_zoid_idx = -1;
        for (int k = 0; k < lmp->recv_from_neighbors[send_zoid_num].size(); k++) {
            if (lmp->recv_from_neighbors[send_zoid_num][k] == zoid_num) {
                send_zoid_idx = k;
                break;
            }
        }
        assert(send_zoid_idx != -1);
        int buf_send_idx = 0;
        for (int t = 1; t < NUM_TIMESTEPS_IN_PARALLEL + 1; t++) {
            Atom* atom_ = send_atom_arr[t];
            atom_->avec->unpack_data_stencil_md(
                    send_zoid.recv_list_local_num_force_only[t][send_zoid_idx], send_zoid.recv_list_local_num_force_pos[t][send_zoid_idx],
                    send_zoid.recv_list_local_force_only[t][send_zoid_idx], send_zoid.recv_list_local_force_pos[t][send_zoid_idx],
                    send_zoid.recv_ghost_num_segments[t][send_zoid_idx], send_zoid.recv_ghost_idxs[t][send_zoid_idx], send_zoid.recv_ghost_sizes[t][send_zoid_idx],
                    &buf_send_stencil_md[i][buf_send_idx]);
            buf_send_idx += send_zoid.num_elems_recv[t][send_zoid_idx];
        }
    }
  }
}

void CommBrick::pack_data_to_process_stencil_md(bool curr_dt, int start_timestep, int end_timestep,
                                                std::array<Atom*, NUM_TIMESTEPS_IN_PARALLEL + 1>& atom_arr,
                                                queue_info& zoid, int proc, int pipeline_stage) {
    int zoid_num = zoid.num;
    auto &send_to_neighbors = curr_dt ? lmp->send_to_neighbors[zoid_num] : lmp->send_to_neighbors_next_dt[zoid_num];

    int send_request_vec_idx = 0;

    auto begin = std::chrono::high_resolution_clock::now();

    // pack data into buffer
    int nsend_force = 0;
    std::vector<int> neighbors_in_proc;
    for (int i = 0; i < send_to_neighbors.size(); i++) {
        int neighbor = send_to_neighbors[i];
        if (neighbor % comm->nprocs == proc) {
            neighbors_in_proc.push_back(i);
        }
    }

    if (neighbors_in_proc.size() == 0) {
        return;
    }

    int num_elems_send;
    if (start_timestep == 0) {
        for (int t = start_timestep; t < end_timestep; t++) {
            for (int i = 0; i < send_to_neighbors.size(); i++) {
                int neighbor = send_to_neighbors[i];
                if (neighbor % comm->nprocs == proc) {
                    int num_force_segments = zoid.send_force_num_segments[t][i];
                    for (int j = 0; j < num_force_segments; j++) {
                        nsend_force += zoid.send_force_sizes[t][i][j];
                    }
                }
            }
        }

        int nsend_vel = 0;
        for (int t = start_timestep; t < end_timestep; t++) {
            for (int i = 0; i < send_to_neighbors.size(); i++) {
                int neighbor = send_to_neighbors[i];
                if (neighbor % comm->nprocs == proc) {
                    int num_vel_segments = zoid.send_pos_num_segments[t][i];
                    for (int j = 0; j < num_vel_segments; j++) {
                        nsend_vel += zoid.send_pos_sizes[t][i][j];
                    }
                }
            }
        }

        // count positions
        int nsend_pos = 0;
        for (int t = start_timestep; t < end_timestep; t++) {
            nsend_pos += zoid.num_elems_send_process[t][proc];
        }

        // force = 3 elems + 1 for tag, send pos is 3 elems for pos and 3 elems for velocity
        if (DEBUG_SEND_RECV_DATA) {
            num_elems_send = nsend_force * (3 + 1) + nsend_pos * (3 + 1) + nsend_vel * (3 + 1);
        } else {
            num_elems_send = nsend_force * (3) + nsend_pos * (3) + nsend_vel * (3);
        }
    } else {
        num_elems_send = zoid.num_send_process[proc];
    }

    if (num_elems_send > maxsend_stencil_md[proc]) {
        // grow_send_stencil_md(num_elems_send, proc, 0);
        grow_send_stencil_md(num_elems_send, proc, 1);
    }

    std::vector<int> idxs;
    idxs.push_back(0);
    for (int t = start_timestep; t < end_timestep; t++) {
        idxs.push_back(zoid.num_send_process_timestep[proc][t] + idxs[idxs.size() - 1]);
    }

    int buf_offset = pipeline_stage * maxsend_stencil_md[proc];

    // int buf_idx = 0;
    // TODO: parallelize

    cilk_for (int t = start_timestep; t < end_timestep; t++) {
        Atom* atom_;
        if (curr_dt) {
            atom_ = atom_arr[t];
        } else {
            atom_ = atom_arr[NUM_TIMESTEPS_IN_PARALLEL - t];
        }

        int starting_idx = idxs[t - start_timestep];

        int* pbc_flags_ = nullptr;
        int n = atom_->avec->pack_data_to_process_stencil_md(
                neighbors_in_proc.size(), neighbors_in_proc.data(),
                zoid.send_force_total_num_elems[t], zoid.send_force_num_segments[t], zoid.send_force_idxs[t], zoid.send_force_sizes[t],
                zoid.send_process_num_segments[t][proc], zoid.send_process_segment_types[t][proc],
                zoid.send_process_segment_idxs[t][proc], zoid.send_process_segment_sizes[t][proc],
                zoid.send_pos_total_num_elems[t], zoid.send_pos_num_segments[t], zoid.send_pos_idxs[t], zoid.send_pos_sizes[t],
                zoid.send_process_local_list[t][proc], &buf_send_stencil_md[proc][starting_idx + buf_offset], pbc_flags_);

        // buf_idx += n;
    }
}

void CommBrick::send_data_bins_stencil_md(bool curr_dt, std::array<Atom*, NUM_TIMESTEPS_IN_PARALLEL + 1>& atom_arr,
                                          queue_info& zoid, int start_timestep, int end_timestep) {
    int zoid_num = zoid.num;
    auto &send_to = curr_dt ? lmp->send_to_neighbors[zoid_num] : lmp->send_to_neighbors_next_dt[zoid_num];

    // int buf_idx = 0;
    // TODO: parallelize
    for(int i = 0; i < send_to.size(); i++) {
        int send_zoid_num = send_to[i];
        auto& send_zoid = curr_dt? lmp->zoid_num_to_zoid[send_zoid_num] : lmp->zoid_num_to_zoid_next_dt[send_zoid_num];
        auto& other_atom_arr = lmp->atom_stencil_md[send_zoid_num];

        auto& recv_from = curr_dt ? lmp->recv_from_neighbors[send_zoid_num] : lmp->recv_from_neighbors_next_dt[send_zoid_num];
        auto it = std::find(recv_from.begin(), recv_from.end(), zoid_num);
        assert(it != recv_from.end());
        int recv_idx = -1;
        recv_idx = std::distance(recv_from.begin(), it);
        assert(recv_idx != -1);

        // calculate pbc flags
        for (int t = start_timestep; t < end_timestep; t++) {
            Atom* atom_;
            if (curr_dt) {
                atom_ = atom_arr[t];
            } else {
                atom_ = atom_arr[NUM_TIMESTEPS_IN_PARALLEL - t];
            }

            Atom* other_atom;
            if (curr_dt) {
                other_atom = other_atom_arr[t];
            } else {
                other_atom = other_atom_arr[NUM_TIMESTEPS_IN_PARALLEL - t];
            }

            int pbc_flag_[3] = {0};
            for (int dim = 0; dim < 3; dim++) {
                if (zoid.where[dim] == RIGHT && send_zoid.where[dim] == PBC) { pbc_flag_[dim] = -1; }

                if (zoid.where[dim] == PBC && send_zoid.where[dim] == RIGHT) { pbc_flag_[dim] = 1; }
            }

            auto& bounds = stencilMD->GET_BOUNDS(curr_dt, t);
            std::cout << BOLDCYAN << "zoid: " << zoid.num << " send to: " << send_zoid_num << " time: " << t << RESET_COLOR << std::endl;
            std::stringstream s;
            for (auto& b: bounds) {
                s << b << " ";
            }
            std::cout << BOLDMAGENTA << "MAN BOUNDS: " << s.str() << RESET_COLOR << std::endl;
            /*
            for (int h = 0; h < atom_->nlocal + atom_->nghost; h++) {
                double* pos = atom_->x[h];
                auto bin = get_bin(bounds, pos, domain->boxlo, domain->boxhi);
                std::cout << BOLDGREEN << "TIMESTEP: " << t << " MY ZOID: " << zoid_num << " MY IDX: " << h << " nlocal: " << atom_->nlocal << " tag: " << atom_->tag[h] << " pos: " << pos[0] << " " << pos[1] << " " << pos[2]
                    << " BIN: " << std::get<0>(bin) << " " << std::get<1>(bin) << " " << std::get<2>(bin) << " bin idx: " << get_bin_idx(bin) << RESET_COLOR << std::endl;
            }

            for (int h = 0; h < other_atom->nlocal + other_atom->nghost; h++) {
                double* pos = other_atom->x[h];
                auto bin = get_bin(bounds, pos, domain->boxlo, domain->boxhi);
                std::cout << BOLDYELLOW << "TIMESTEP: " << t << " OTHER ZOID: " << send_zoid_num << " OTHER IDX: " << h << " nlocal: " << other_atom->nlocal << " tag: " << other_atom->tag[h] << " pos: " << pos[0] << " " << pos[1] << " " << pos[2]
                    << " BIN: " << std::get<0>(bin) << " " << std::get<1>(bin) << " " << std::get<2>(bin) << " bin idx: " << get_bin_idx(bin) << RESET_COLOR << std::endl;
            }
            */

            atom_->avec->send_data_bins_stencil_md(zoid.bin_to_idx[t], zoid.bin_to_size[t], send_zoid.bin_to_idx[t], send_zoid.bin_to_size[t],
                                                   zoid.send_force_num_bins[t][i], zoid.send_force_bins[t][i],
                                                   zoid.send_pos_num_bins[t][i], zoid.send_pos_bins[t][i],
                                                   zoid.send_vel_num_bins[t][i], zoid.send_vel_bins[t][i],
                                                   send_zoid.recv_force_num_bins[t][recv_idx], send_zoid.recv_force_bins[t][recv_idx],
                                                   send_zoid.recv_pos_num_bins[t][recv_idx], send_zoid.recv_pos_bins[t][recv_idx],
                                                   send_zoid.recv_vel_num_bins[t][recv_idx], send_zoid.recv_vel_bins[t][recv_idx],
                                                   other_atom->tag, other_atom->f, other_atom->x, other_atom->v, pbc_flag_
                                                   );
        }
    }
}

void CommBrick::recv_data_bins_stencil_md(bool curr_dt, std::array<Atom*, NUM_TIMESTEPS_IN_PARALLEL + 1>& atom_arr,
                                          queue_info& zoid, int start_timestep, int end_timestep) {
    int zoid_num = zoid.num;
    auto &recv_from = curr_dt ? lmp->recv_from_neighbors[zoid_num] : lmp->recv_from_neighbors_next_dt[zoid_num];

    // TODO: parallelize
    for (int i = 0; i < recv_from.size(); i++) {
        int recv_zoid_num = recv_from[i];
        auto& recv_zoid = curr_dt ? lmp->zoid_num_to_zoid[recv_zoid_num] : lmp->zoid_num_to_zoid_next_dt[recv_zoid_num];
        auto& other_atom_arr = lmp->atom_stencil_md[recv_zoid_num];

        auto& send_to = curr_dt ? lmp->send_to_neighbors[recv_zoid_num] : lmp->send_to_neighbors_next_dt[recv_zoid_num];
        auto it = std::find(send_to.begin(), send_to.end(), zoid_num);
        assert(it != send_to.end());
        int send_idx = -1;
        send_idx = std::distance(send_to.begin(), it);
        assert(send_idx != -1);

        // calculate pbc flags
        cilk_for (int t = start_timestep; t < end_timestep; t++) {
            Atom* atom_;
            if (curr_dt) {
                atom_ = atom_arr[t];
            } else {
                atom_ = atom_arr[NUM_TIMESTEPS_IN_PARALLEL - t];
            }

            Atom* other_atom;
            if (curr_dt) {
                other_atom = other_atom_arr[t];
            } else {
                other_atom = other_atom_arr[NUM_TIMESTEPS_IN_PARALLEL - t];
            }

            int pbc_flag_[3] = {0};
            for (int dim = 0; dim < 3; dim++) {
                if (recv_zoid.where[dim] == RIGHT && zoid.where[dim] == PBC) { pbc_flag_[dim] = -1; }

                if (recv_zoid.where[dim] == PBC && zoid.where[dim] == RIGHT) { pbc_flag_[dim] = 1; }
            }

            auto& bounds = stencilMD->GET_BOUNDS(curr_dt, t);
            /*
            std::cout << "zoid: " << zoid.num << " recv from: " << recv_zoid_num << " t: " << t << " curr dt? " << curr_dt
                << " num force bins recv: " << zoid.recv_force_num_bins[t][i] << " num pos bins recv: " << zoid.recv_pos_num_bins[t][i]
                << " num vel bins recv: " << zoid.recv_vel_num_bins[t][i] << std::endl;
            */

            const dbl3_t_stencil_md* _noalias const send_f = (dbl3_t_stencil_md*) other_atom->eval_f_stencil_md[0];
            const dbl3_t_stencil_md* _noalias const send_x = (dbl3_t_stencil_md*) other_atom->x[0];
            const dbl3_t_stencil_md* _noalias const send_v = (dbl3_t_stencil_md*) other_atom->v[0];

            atom_->avec->recv_data_bins_stencil_md(recv_zoid.bin_to_idx[t], recv_zoid.bin_to_size[t],
                                                   zoid.bin_to_idx[t], zoid.bin_to_size[t],
                                                   recv_zoid.send_force_num_bins[t][send_idx], recv_zoid.send_force_bins[t][send_idx],
                                                   recv_zoid.send_pos_num_bins[t][send_idx], recv_zoid.send_pos_bins[t][send_idx],
                                                   recv_zoid.send_vel_num_bins[t][send_idx], recv_zoid.send_vel_bins[t][send_idx],
                                                   zoid.recv_force_num_bins[t][i], zoid.recv_force_bins[t][i],
                                                   zoid.recv_pos_num_bins[t][i], zoid.recv_pos_bins[t][i],
                                                   zoid.recv_vel_num_bins[t][i], zoid.recv_vel_bins[t][i],
                                                   other_atom->tag, send_f, send_x, send_v, pbc_flag_
            );
        }
    }
}

void CommBrick::unpack_self_stencil_md(bool curr_dt, int start_timestep, int end_timestep, queue_info &zoid,
                                       int pipeline_stage) {
    int zoid_num = zoid.num;
    auto& recv_from = curr_dt ? lmp->recv_from_neighbors[zoid_num] : lmp->recv_from_neighbors_next_dt[zoid_num];
    auto& atom_arr = lmp->atom_stencil_md[zoid_num];

    for (int recv_zoid_num : recv_from) {
        if (recv_zoid_num % comm->nprocs == comm->me) {
            auto c_zoid = (CommBrick*) lmp->comm_stencil_md[recv_zoid_num];
            int buf_offset = pipeline_stage * c_zoid->maxsend_stencil_md[comm->me];
            int receive_request_idx = curr_dt ? lmp->recv_from_neighbors_procs_idxs[recv_zoid_num] : lmp->recv_from_neighbors_procs_idxs_next_dt[recv_zoid_num];
            double* buf = c_zoid->buf_send_stencil_md[comm->me];

            assert(zoid_num % comm->nprocs == comm->me);
            queue_info& recv_zoid = curr_dt? lmp->zoid_num_to_zoid[recv_zoid_num] : lmp->zoid_num_to_zoid_next_dt[recv_zoid_num];

            int pbc_flag_[3] = {0};
            for (int dim = 0; dim < 3; dim++) {
                if (recv_zoid.where[dim] == RIGHT && zoid.where[dim] == PBC) { pbc_flag_[dim] = -1; }

                if (recv_zoid.where[dim] == PBC && zoid.where[dim] == RIGHT) { pbc_flag_[dim] = 1; }
            }

            std::vector<int> idxs;
            idxs.push_back(0);
            for (int t = start_timestep; t < end_timestep; t++) {
                if (curr_dt) {
                    idxs.push_back(lmp->num_recv_elems_from_zoid[t][receive_request_idx] + idxs[idxs.size() - 1]);
                } else {
                    idxs.push_back(lmp->num_recv_elems_from_zoid_next_dt[t][receive_request_idx] + idxs[idxs.size() - 1]);
                }
            }

            auto it = std::find(recv_from.begin(), recv_from.end(), recv_zoid_num);
            assert(it != recv_from.end());
            int recv_idx = -1;
            recv_idx = std::distance(recv_from.begin(), it);
            assert(recv_idx != -1);

            cilk_for (int t = start_timestep; t < end_timestep; t++) {
                Atom *atom_;
                int nrecv_force;
                int nrecv_pos;
                // int nrecv_vel;

                if (curr_dt) {
                    atom_ = atom_arr[t];
                    nrecv_force = lmp->num_recv_force_from_zoid[t][receive_request_idx];
                    nrecv_pos = lmp->num_recv_pos_from_zoid[t][receive_request_idx];
                    // nrecv_vel = lmp->num_recv_vel_from_zoid[t][receive_request_idx];
                } else {
                    atom_ = atom_arr[NUM_TIMESTEPS_IN_PARALLEL - t];
                    nrecv_force = lmp->num_recv_force_from_zoid_next_dt[t][receive_request_idx];
                    nrecv_pos = lmp->num_recv_pos_from_zoid_next_dt[t][receive_request_idx];
                    // nrecv_vel = lmp->num_recv_vel_from_zoid_next_dt[t][receive_request_idx];
                }

                int starting_idx = idxs[t - start_timestep];

                atom_->avec->unpack_data_from_process_stencil_md(
                        nrecv_force, nrecv_pos,
                        zoid.recv_process_force_offset[t][recv_idx], zoid.recv_list_local_num_force_only[t][recv_idx],
                        zoid.recv_list_local_force_only[t][recv_idx],
                        zoid.recv_process_num_segments[t][recv_idx], zoid.recv_process_segment_types[t][recv_idx],
                        zoid.recv_process_segment_idxs[t][recv_idx], zoid.recv_process_segment_sizes[t][recv_idx],
                        zoid.recv_process_vel_offset[t][recv_idx], zoid.recv_list_local_num_force_pos[t][recv_idx],
                        zoid.recv_list_local_force_pos[t][recv_idx],
                        zoid.recv_ghost_num_segments[t][recv_idx], zoid.recv_ghost_idxs[t][recv_idx],
                        zoid.recv_ghost_sizes[t][recv_idx],
                        &buf[starting_idx + buf_offset], pbc_flag_);
            }
        }
    }
}

bool CommBrick::send_packed_data_to_process_stencil_md(bool curr_dt, int start_timestep, int end_timestep,
                                                       queue_info& zoid, MPI_Request* request, int proc, int pipeline_stage) {
    int zoid_num = zoid.num;

    // int num_elems_send = zoid.num_send_process[proc];
    int num_elems_send = 0;
    for (int t = start_timestep; t < end_timestep; t++) {
        num_elems_send += zoid.num_send_process_timestep[proc][t];
    }

    auto &send_to_neighbors = curr_dt ? lmp->send_to_neighbors[zoid_num] : lmp->send_to_neighbors_next_dt[zoid_num];
    int buf_offset = pipeline_stage * maxsend_stencil_md[proc];

    if (proc != comm->me) {
        // int mpi_tag = (proc << 16 | zoid_num);
        int mpi_tag = get_mpi_tag(proc, zoid_num, start_timestep, end_timestep);
        MPI_Isend(&buf_send_stencil_md[proc][buf_offset], num_elems_send, MPI_DOUBLE, proc, mpi_tag, world,
                  request);
        return true;
    } else {
        auto& recv_from_neighbor_procs = curr_dt ? lmp->recv_from_neighbors_procs : lmp->recv_from_neighbors_procs_next_dt;
        int receive_request_idx = curr_dt ? lmp->recv_from_neighbors_procs_idxs[zoid_num] : lmp->recv_from_neighbors_procs_idxs_next_dt[zoid_num];
        if (curr_dt) {
            assert(receive_request_idx >= 0 && receive_request_idx < lmp->recv_from_neighbors_procs.size());
        } else {
            assert(receive_request_idx >= 0 && receive_request_idx < lmp->recv_from_neighbors_procs_next_dt.size());
        }

        queue_info& recv_zoid = curr_dt? lmp->zoid_num_to_zoid[zoid_num] : lmp->zoid_num_to_zoid_next_dt[zoid_num];
        auto& zoid_num_idxs_recv = curr_dt ? lmp->recv_zoid_to_my_zoids[zoid_num] : lmp->recv_zoid_to_my_zoids_next_dt[zoid_num];

        cilk_for (int i = 0; i < zoid_num_idxs_recv.size(); i++) {
            int other_zoid_num = zoid_num_idxs_recv[i].first;
            int recv_idx = zoid_num_idxs_recv[i].second;
            auto& other_atom_arr = lmp->atom_stencil_md[other_zoid_num];
            auto& other_recv_from = curr_dt ? lmp->recv_from_neighbors[other_zoid_num] : lmp->recv_from_neighbors_next_dt[other_zoid_num];
            queue_info& other_zoid = curr_dt ? lmp->zoid_num_to_zoid[other_zoid_num] : lmp->zoid_num_to_zoid_next_dt[other_zoid_num];

            int pbc_flag_[3] = {0};
            for (int dim = 0; dim < 3; dim++) {
                if (zoid.where[dim] == RIGHT && other_zoid.where[dim] == PBC) { pbc_flag_[dim] = -1; }

                if (zoid.where[dim] == PBC && other_zoid.where[dim] == RIGHT) { pbc_flag_[dim] = 1; }
            }

            std::vector<int> idxs;
            idxs.push_back(0);
            for (int t = start_timestep; t < end_timestep; t++) {
                if (curr_dt) {
                    idxs.push_back(lmp->num_recv_elems_from_zoid[t][receive_request_idx] + idxs[idxs.size() - 1]);
                } else {
                    idxs.push_back(lmp->num_recv_elems_from_zoid_next_dt[t][receive_request_idx] + idxs[idxs.size() - 1]);
                }
            }

            cilk_for (int t = start_timestep; t < end_timestep; t++) {
                Atom* atom_;
                int nrecv_force;
                int nrecv_pos;
                // int nrecv_vel;

                if (curr_dt) {
                    atom_ = other_atom_arr[t];
                    nrecv_force = lmp->num_recv_force_from_zoid[t][receive_request_idx];
                    nrecv_pos = lmp->num_recv_pos_from_zoid[t][receive_request_idx];
                    // nrecv_vel = lmp->num_recv_vel_from_zoid[t][receive_request_idx];
                } else {
                    atom_ = other_atom_arr[NUM_TIMESTEPS_IN_PARALLEL - t];
                    nrecv_force = lmp->num_recv_force_from_zoid_next_dt[t][receive_request_idx];
                    nrecv_pos = lmp->num_recv_pos_from_zoid_next_dt[t][receive_request_idx];
                    // nrecv_vel = lmp->num_recv_vel_from_zoid_next_dt[t][receive_request_idx];
                }

                int other_starting_idx = idxs[t - start_timestep];

                atom_->avec->unpack_data_from_process_stencil_md(
                        nrecv_force, nrecv_pos,
                        other_zoid.recv_process_force_offset[t][recv_idx], other_zoid.recv_list_local_num_force_only[t][recv_idx], other_zoid.recv_list_local_force_only[t][recv_idx],
                        other_zoid.recv_process_num_segments[t][recv_idx], other_zoid.recv_process_segment_types[t][recv_idx],
                        other_zoid.recv_process_segment_idxs[t][recv_idx], other_zoid.recv_process_segment_sizes[t][recv_idx],
                        other_zoid.recv_process_vel_offset[t][recv_idx], other_zoid.recv_list_local_num_force_pos[t][recv_idx], other_zoid.recv_list_local_force_pos[t][recv_idx],
                        other_zoid.recv_ghost_num_segments[t][recv_idx], other_zoid.recv_ghost_idxs[t][recv_idx], other_zoid.recv_ghost_sizes[t][recv_idx],
                        &buf_send_stencil_md[comm->me][other_starting_idx + buf_offset], pbc_flag_);
            }
        }
        return false;
    }
}


bool CommBrick::send_data_to_process_stencil_md(bool curr_dt, std::array<Atom*, NUM_TIMESTEPS_IN_PARALLEL + 1>& atom_arr,
                                                queue_info& zoid, MPI_Request* request, int proc, bool is_initial, int pipeline_stage) {
    if (!curr_dt) {
        assert(!is_initial);
    }

    int start_timestep;
    int end_timestep;

    // technically only need timesteps 1 to NUM_TIMESTEPS_IN_PARALLEL + 1, but initial setup needs t = 0
    if (is_initial) {
        start_timestep = 0;
        // set this to NUM_TIMESTEPS_IN_PARALLEL + 1 instead of 1 so that the allocations for the future buffers can happen now?
        end_timestep = NUM_TIMESTEPS_IN_PARALLEL + 1;
    } else {
        start_timestep = 1;
        end_timestep = NUM_TIMESTEPS_IN_PARALLEL + 1;
    }

    int zoid_num = zoid.num;
    auto &send_to_neighbors = curr_dt ? lmp->send_to_neighbors[zoid_num] : lmp->send_to_neighbors_next_dt[zoid_num];

    int send_request_vec_idx = 0;

    auto begin = std::chrono::high_resolution_clock::now();

    // pack data into buffer
    int nsend_force = 0;
    std::vector<int> neighbors_in_proc;
    for (int i = 0; i < send_to_neighbors.size(); i++) {
        int neighbor = send_to_neighbors[i];
        if (neighbor % comm->nprocs == proc) {
            neighbors_in_proc.push_back(i);
        }
    }

    if (neighbors_in_proc.size() == 0) {
        return false;
    }

    int num_elems_send;
    if (is_initial) {
        for (int t = start_timestep; t < end_timestep; t++) {
            for (int i = 0; i < send_to_neighbors.size(); i++) {
                int neighbor = send_to_neighbors[i];
                if (neighbor % comm->nprocs == proc) {
                    int num_force_segments = zoid.send_force_num_segments[t][i];
                    for (int j = 0; j < num_force_segments; j++) {
                        nsend_force += zoid.send_force_sizes[t][i][j];
                    }
                }
            }
        }

        int nsend_vel = 0;
        for (int t = start_timestep; t < end_timestep; t++) {
            for (int i = 0; i < send_to_neighbors.size(); i++) {
                int neighbor = send_to_neighbors[i];
                if (neighbor % comm->nprocs == proc) {
                    int num_vel_segments = zoid.send_pos_num_segments[t][i];
                    for (int j = 0; j < num_vel_segments; j++) {
                        nsend_vel += zoid.send_pos_sizes[t][i][j];
                    }
                }
            }
        }

        // count positions
        int nsend_pos = 0;
        for (int t = start_timestep; t < end_timestep; t++) {
            nsend_pos += zoid.num_elems_send_process[t][proc];
        }

        // force = 3 elems + 1 for tag, send pos is 3 elems for pos and 3 elems for velocity
        if (DEBUG_SEND_RECV_DATA) {
            num_elems_send = nsend_force * (3 + 1) + nsend_pos * (3 + 1) + nsend_vel * (3 + 1);
        } else {
            num_elems_send = nsend_force * (3) + nsend_pos * (3) + nsend_vel * (3);
        }
    } else {
        num_elems_send = zoid.num_send_process[proc];
    }

    if (num_elems_send > maxsend_stencil_md[proc]) {
        grow_send_stencil_md(num_elems_send, proc, 0);
    }

    int buf_offset = pipeline_stage * maxsend_stencil_md[proc];

    auto begin_atom_pack = std::chrono::high_resolution_clock::now();
    int buf_idx = 0;
    for (int t = start_timestep; t < end_timestep; t++) {
        // Atom *atom_ = atom_arr[t];
        Atom* atom_;
        if (curr_dt) {
            atom_ = atom_arr[t];
        } else {
            atom_ = atom_arr[NUM_TIMESTEPS_IN_PARALLEL - t];
        }

        int* pbc_flags_ = nullptr;
        int n = atom_->avec->pack_data_to_process_stencil_md(
                neighbors_in_proc.size(), neighbors_in_proc.data(),
                zoid.send_force_total_num_elems[t], zoid.send_force_num_segments[t], zoid.send_force_idxs[t], zoid.send_force_sizes[t],
                zoid.send_process_num_segments[t][proc], zoid.send_process_segment_types[t][proc],
                zoid.send_process_segment_idxs[t][proc], zoid.send_process_segment_sizes[t][proc],
                zoid.send_pos_total_num_elems[t], zoid.send_pos_num_segments[t], zoid.send_pos_idxs[t], zoid.send_pos_sizes[t],
                zoid.send_process_local_list[t][proc], &buf_send_stencil_md[proc][buf_idx + buf_offset], pbc_flags_);

        buf_idx += n;
    }
    auto end_atom_pack = std::chrono::high_resolution_clock::now();
    auto duration_atom_pack = std::chrono::duration_cast<std::chrono::microseconds>(end_atom_pack-begin_atom_pack).count();

    assert(num_elems_send == buf_idx);

    if (proc != comm->me) {
        // int mpi_tag = (proc << 16 | zoid_num);
        int mpi_tag = get_mpi_tag(proc, zoid_num, start_timestep, end_timestep);
        MPI_Isend(&buf_send_stencil_md[proc][buf_offset], buf_idx, MPI_DOUBLE, proc, mpi_tag, world,
                  request);
        return true;
    } else {
        int receive_request_idx = -1;
        auto& recv_from_neighbor_procs = curr_dt ? lmp->recv_from_neighbors_procs : lmp->recv_from_neighbors_procs_next_dt;
        auto it = std::find(recv_from_neighbor_procs.begin(), recv_from_neighbor_procs.end(), zoid_num);
        assert(it != recv_from_neighbor_procs.end());
        receive_request_idx = std::distance(recv_from_neighbor_procs.begin(), it);

        auto begin_unpack = std::chrono::high_resolution_clock::now();
        queue_info& recv_zoid = curr_dt? lmp->zoid_num_to_zoid[zoid_num] : lmp->zoid_num_to_zoid_next_dt[zoid_num];
        auto& zoid_num_idxs_recv = curr_dt ? lmp->recv_zoid_to_my_zoids[zoid_num] : lmp->recv_zoid_to_my_zoids_next_dt[zoid_num];

        for (auto& [other_zoid_num, recv_idx] : zoid_num_idxs_recv) {
            auto& other_atom_arr = lmp->atom_stencil_md[other_zoid_num];
            auto& other_recv_from = curr_dt ? lmp->recv_from_neighbors[other_zoid_num] : lmp->recv_from_neighbors_next_dt[other_zoid_num];
            queue_info& other_zoid = curr_dt ? lmp->zoid_num_to_zoid[other_zoid_num] : lmp->zoid_num_to_zoid_next_dt[other_zoid_num];

            int pbc_flag_[3] = {0};
            for (int dim = 0; dim < 3; dim++) {
                if (zoid.where[dim] == RIGHT && other_zoid.where[dim] == PBC) { pbc_flag_[dim] = -1; }

                if (zoid.where[dim] == PBC && other_zoid.where[dim] == RIGHT) { pbc_flag_[dim] = 1; }
            }

            int other_buf_idx = 0;
            for (int t = start_timestep; t < end_timestep; t++) {
                Atom* atom_;
                int nrecv_force;
                int nrecv_pos;
                int nrecv_vel;

                if (curr_dt) {
                    atom_ = other_atom_arr[t];
                    nrecv_force = lmp->num_recv_force_from_zoid[t][receive_request_idx];
                    nrecv_pos = lmp->num_recv_pos_from_zoid[t][receive_request_idx];
                    nrecv_vel = lmp->num_recv_vel_from_zoid[t][receive_request_idx];
                } else {
                    atom_ = other_atom_arr[NUM_TIMESTEPS_IN_PARALLEL - t];
                    nrecv_force = lmp->num_recv_force_from_zoid_next_dt[t][receive_request_idx];
                    nrecv_pos = lmp->num_recv_pos_from_zoid_next_dt[t][receive_request_idx];
                    nrecv_vel = lmp->num_recv_vel_from_zoid_next_dt[t][receive_request_idx];
                }

                /*
                std::cout << "unpack process SELF UNPACK: " << comm->me << " zoid: " << other_zoid.num << " recv from: " << zoid_num << " timestep: " << t
                          << " pos start: " << nrecv_force << " vel start: " << nrecv_force + nrecv_pos << " vel offset: " << zoid.recv_process_vel_offset[t][recv_idx]
                          << " force offset: " << other_zoid.recv_process_force_offset[t][recv_idx]
                          << " nrecv force: " << nrecv_force << " nrecv pos: " << nrecv_pos << " nrecv vel: " << nrecv_vel
                          << " increment: " << nrecv_force * 4 + nrecv_pos * 4 + nrecv_vel * 4 << std::endl;
                */

                atom_->avec->unpack_data_from_process_stencil_md(
                        nrecv_force, nrecv_pos,
                        other_zoid.recv_process_force_offset[t][recv_idx], other_zoid.recv_list_local_num_force_only[t][recv_idx], other_zoid.recv_list_local_force_only[t][recv_idx],
                        other_zoid.recv_process_num_segments[t][recv_idx], other_zoid.recv_process_segment_types[t][recv_idx],
                        other_zoid.recv_process_segment_idxs[t][recv_idx], other_zoid.recv_process_segment_sizes[t][recv_idx],
                        other_zoid.recv_process_vel_offset[t][recv_idx], other_zoid.recv_list_local_num_force_pos[t][recv_idx], other_zoid.recv_list_local_force_pos[t][recv_idx],
                        other_zoid.recv_ghost_num_segments[t][recv_idx], other_zoid.recv_ghost_idxs[t][recv_idx], other_zoid.recv_ghost_sizes[t][recv_idx],
                        &buf_send_stencil_md[comm->me][other_buf_idx + buf_offset], pbc_flag_);

                if (DEBUG_SEND_RECV_DATA) {
                    other_buf_idx += nrecv_force * (3 + 1) + nrecv_pos * (3 + 1) + nrecv_vel * (3 + 1);
                } else {
                    other_buf_idx += nrecv_force * (3) + nrecv_pos * (3) + nrecv_vel * (3);
                }
            }
        }

        auto end_unpack = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_unpack-begin_unpack).count();

        return false;
    }
}

void CommBrick::receive_data_process_stencil_md(bool curr_dt, int start_timestep, int end_timestep,
                                                MPI_Request* request, int recv_zoid_num, int pipeline_stage) {
    assert(pipeline_stage >= 0 && pipeline_stage < NUM_PIPELINE_STAGES);

    if (recv_zoid_num % comm->nprocs == comm->me) {
        std::cout << "ERROR. proc: " << comm->me << " recv zoid num: " << recv_zoid_num << std::endl;
        assert(false);
        return;
    }

    auto& recv_from_neighbors_procs = curr_dt ? lmp->recv_from_neighbors_procs : lmp->recv_from_neighbors_procs_next_dt;
    auto it = std::find(recv_from_neighbors_procs.begin(), recv_from_neighbors_procs.end(), recv_zoid_num);

    assert(it != recv_from_neighbors_procs.end());

    int idx_recv_zoid = -1;
    idx_recv_zoid = std::distance(recv_from_neighbors_procs.begin(), it);

    assert(idx_recv_zoid != -1);

    // int mpi_tag = (comm->me << 16 | recv_zoid_num);
    int mpi_tag = get_mpi_tag(comm->me, recv_zoid_num, start_timestep, end_timestep);
    int nrecv_force = 0;
    int nrecv_pos = 0;
    int nrecv_vel = 0;

    for (int t = start_timestep; t < end_timestep; t++) {
        if (curr_dt) {
            nrecv_force += lmp->num_recv_force_from_zoid[t][idx_recv_zoid];
            nrecv_pos += lmp->num_recv_pos_from_zoid[t][idx_recv_zoid];
            nrecv_vel += lmp->num_recv_vel_from_zoid[t][idx_recv_zoid];
        } else {
            nrecv_force += lmp->num_recv_force_from_zoid_next_dt[t][idx_recv_zoid];
            nrecv_pos += lmp->num_recv_pos_from_zoid_next_dt[t][idx_recv_zoid];
            nrecv_vel += lmp->num_recv_vel_from_zoid_next_dt[t][idx_recv_zoid];
        }
    }

    int nrecv;
    if (DEBUG_SEND_RECV_DATA) {
        nrecv = nrecv_force * (3 + 1) + nrecv_pos * (3 + 1) + nrecv_vel * (3 + 1);
    } else {
        nrecv = nrecv_force * (3) + nrecv_pos * (3) + nrecv_vel * (3);
    }

    if (nrecv > maxrecv_stencil_md[idx_recv_zoid]) {
        grow_recv_stencil_md(nrecv, idx_recv_zoid);
    }

    int buf_offset = pipeline_stage * maxrecv_stencil_md[idx_recv_zoid];
    assert(idx_recv_zoid >= 0 && idx_recv_zoid < 26);

    MPI_Irecv(&buf_recv_stencil_md[idx_recv_zoid][buf_offset], nrecv, MPI_DOUBLE, recv_zoid_num % comm->nprocs, mpi_tag, world,
              request);
}

void CommBrick::unpack_data_process_zoid_stencil_md(bool curr_dt, queue_info& zoid,
                                                    int start_timestep, int end_timestep, int pipeline_stage) {
    // assume the sender will have done all of the unpacking
    assert(pipeline_stage >= 0 && pipeline_stage < NUM_PIPELINE_STAGES);

    int zoid_num = zoid.num;
    auto &atom_arr = lmp->atom_stencil_md[zoid_num];
    auto& recv_from = curr_dt ? lmp->recv_from_neighbors[zoid_num] : lmp->recv_from_neighbors_next_dt[zoid_num];

    for (int recv_idx = 0; recv_idx < recv_from.size(); recv_idx++) {
        int recv_zoid_num = recv_from[recv_idx];
        if (recv_zoid_num % comm->nprocs == comm->me) {
            continue;
        }
        int receive_request_idx = curr_dt ? lmp->recv_from_neighbors_procs_idxs[recv_zoid_num] : lmp->recv_from_neighbors_procs_idxs_next_dt[recv_zoid_num];
        assert(receive_request_idx >= 0 && receive_request_idx < 26);
        queue_info& recv_zoid = curr_dt? lmp->zoid_num_to_zoid[recv_zoid_num] : lmp->zoid_num_to_zoid_next_dt[recv_zoid_num];

        double* buf;
        int buf_offset;
        if (recv_zoid_num % comm->nprocs == comm->me) {
            auto c_recv_zoid = (CommBrick*) lmp->comm_stencil_md[recv_zoid_num];
            buf = c_recv_zoid->buf_send_stencil_md[comm->me];
            buf_offset = pipeline_stage * c_recv_zoid->maxsend_stencil_md[comm->me];
        } else {
            auto c = (CommBrick*) lmp->comm;
            buf = c->buf_recv_stencil_md[receive_request_idx];
            buf_offset = pipeline_stage * c->maxrecv_stencil_md[receive_request_idx];
        }

        int pbc_flag_[3] = {0};
        for (int dim = 0; dim < 3; dim++) {
            if (recv_zoid.where[dim] == RIGHT && zoid.where[dim] == PBC) { pbc_flag_[dim] = -1; }

            if (recv_zoid.where[dim] == PBC && zoid.where[dim] == RIGHT) { pbc_flag_[dim] = 1; }
        }

        std::vector<int> idxs;
        idxs.push_back(0);
        for (int t = start_timestep; t < end_timestep; t++) {
            if (curr_dt) {
                idxs.push_back(lmp->num_recv_elems_from_zoid[t][receive_request_idx] + idxs[idxs.size() - 1]);
            } else {
                idxs.push_back(
                        lmp->num_recv_elems_from_zoid_next_dt[t][receive_request_idx] + idxs[idxs.size() - 1]);
            }
        }

        if (curr_dt) {
            assert(lmp->recv_from_neighbors_procs[receive_request_idx] == recv_zoid_num);
        } else {
            assert(lmp->recv_from_neighbors_procs_next_dt[receive_request_idx] == recv_zoid_num);
        }

        auto it = std::find(recv_from.begin(), recv_from.end(), recv_zoid_num);
        assert(it != recv_from.end());
        int recv_idx2 = -1;
        recv_idx2 = std::distance(recv_from.begin(), it);
        assert(recv_idx2 != -1);
        assert(recv_idx == recv_idx2);

        cilk_for(int t = start_timestep; t < end_timestep; t++) {
            Atom *atom_;
            int nrecv_force;
            int nrecv_pos;
            // int nrecv_vel;
            if (curr_dt) {
                atom_ = atom_arr[t];
                nrecv_force = lmp->num_recv_force_from_zoid[t][receive_request_idx];
                nrecv_pos = lmp->num_recv_pos_from_zoid[t][receive_request_idx];
                // nrecv_vel = lmp->num_recv_vel_from_zoid[t][receive_request_idx];
            } else {
                atom_ = atom_arr[NUM_TIMESTEPS_IN_PARALLEL - t];
                nrecv_force = lmp->num_recv_force_from_zoid_next_dt[t][receive_request_idx];
                nrecv_pos = lmp->num_recv_pos_from_zoid_next_dt[t][receive_request_idx];
                // nrecv_vel = lmp->num_recv_vel_from_zoid_next_dt[t][receive_request_idx];
            }

            int starting_idx = idxs[t - start_timestep];

            atom_->avec->unpack_data_from_process_stencil_md(
                    nrecv_force, nrecv_pos,
                    zoid.recv_process_force_offset[t][recv_idx], zoid.recv_list_local_num_force_only[t][recv_idx],
                    zoid.recv_list_local_force_only[t][recv_idx],
                    zoid.recv_process_num_segments[t][recv_idx], zoid.recv_process_segment_types[t][recv_idx],
                    zoid.recv_process_segment_idxs[t][recv_idx], zoid.recv_process_segment_sizes[t][recv_idx],
                    zoid.recv_process_vel_offset[t][recv_idx], zoid.recv_list_local_num_force_pos[t][recv_idx],
                    zoid.recv_list_local_force_pos[t][recv_idx],
                    zoid.recv_ghost_num_segments[t][recv_idx], zoid.recv_ghost_idxs[t][recv_idx],
                    zoid.recv_ghost_sizes[t][recv_idx],
                    &buf[starting_idx + buf_offset], pbc_flag_);
        }
    }
}

void CommBrick::unpack_data_process_stencil_md(bool curr_dt, int start_timestep, int end_timestep,
                                               int recv_zoid_num, int pipeline_stage) {
    // assume the sender will have done all of the unpacking
    assert(recv_zoid_num % comm->nprocs != comm->me);
    assert(pipeline_stage >= 0 && pipeline_stage < NUM_PIPELINE_STAGES);

    auto& recv_from_neighbor_procs = curr_dt ? lmp->recv_from_neighbors_procs : lmp->recv_from_neighbors_procs_next_dt;
    int receive_request_idx = curr_dt ? lmp->recv_from_neighbors_procs_idxs[recv_zoid_num] : lmp->recv_from_neighbors_procs_idxs_next_dt[recv_zoid_num];
    if (curr_dt) {
        assert(receive_request_idx >= 0 && receive_request_idx < lmp->recv_from_neighbors_procs.size());
    } else {
        assert(receive_request_idx >= 0 && receive_request_idx < lmp->recv_from_neighbors_procs_next_dt.size());
    }

    auto begin = std::chrono::high_resolution_clock::now();
    queue_info& recv_zoid = curr_dt? lmp->zoid_num_to_zoid[recv_zoid_num] : lmp->zoid_num_to_zoid_next_dt[recv_zoid_num];
    auto& zoid_num_idxs_recv = curr_dt ? lmp->recv_zoid_to_my_zoids[recv_zoid_num] : lmp->recv_zoid_to_my_zoids_next_dt[recv_zoid_num];
    int buf_offset = pipeline_stage * maxrecv_stencil_md[receive_request_idx];
    assert(receive_request_idx >= 0 && receive_request_idx < 26);

    cilk_for (int i = 0; i < zoid_num_idxs_recv.size(); i++) {
        int zoid_num = zoid_num_idxs_recv[i].first;
        int recv_idx = zoid_num_idxs_recv[i].second;

        auto& atom_arr = lmp->atom_stencil_md[zoid_num];
        auto& recv_from = curr_dt ? lmp->recv_from_neighbors[zoid_num] : lmp->recv_from_neighbors_next_dt[zoid_num];
        queue_info& zoid = curr_dt? lmp->zoid_num_to_zoid[zoid_num] : lmp->zoid_num_to_zoid_next_dt[zoid_num];

        int pbc_flag_[3] = {0};
        for (int dim = 0; dim < 3; dim++) {
            if (recv_zoid.where[dim] == RIGHT && zoid.where[dim] == PBC) { pbc_flag_[dim] = -1; }

            if (recv_zoid.where[dim] == PBC && zoid.where[dim] == RIGHT) { pbc_flag_[dim] = 1; }
        }

        std::vector<int> idxs;
        idxs.push_back(0);
        for (int t = start_timestep; t < end_timestep; t++) {
            if (curr_dt) {
                idxs.push_back(lmp->num_recv_elems_from_zoid[t][receive_request_idx] + idxs[idxs.size() - 1]);
            } else {
                idxs.push_back(lmp->num_recv_elems_from_zoid_next_dt[t][receive_request_idx] + idxs[idxs.size() - 1]);
            }
        }

        if (curr_dt) {
            assert(lmp->recv_from_neighbors_procs[receive_request_idx] == recv_zoid_num);
        } else {
            assert(lmp->recv_from_neighbors_procs_next_dt[receive_request_idx] == recv_zoid_num);
        }

        cilk_for (int t = start_timestep; t < end_timestep; t++) {
            Atom* atom_;

            int nrecv_force;
            int nrecv_pos;
            // int nrecv_vel;
            if (curr_dt) {
                atom_ = atom_arr[t];
                nrecv_force = lmp->num_recv_force_from_zoid[t][receive_request_idx];
                nrecv_pos = lmp->num_recv_pos_from_zoid[t][receive_request_idx];
                // nrecv_vel = lmp->num_recv_vel_from_zoid[t][receive_request_idx];
            } else {
                atom_ = atom_arr[NUM_TIMESTEPS_IN_PARALLEL - t];
                nrecv_force = lmp->num_recv_force_from_zoid_next_dt[t][receive_request_idx];
                nrecv_pos = lmp->num_recv_pos_from_zoid_next_dt[t][receive_request_idx];
                // nrecv_vel = lmp->num_recv_vel_from_zoid_next_dt[t][receive_request_idx];
            }

            int starting_idx = idxs[t - start_timestep];

            atom_->avec->unpack_data_from_process_stencil_md(
                    nrecv_force, nrecv_pos,
                    zoid.recv_process_force_offset[t][recv_idx], zoid.recv_list_local_num_force_only[t][recv_idx], zoid.recv_list_local_force_only[t][recv_idx],
                    zoid.recv_process_num_segments[t][recv_idx], zoid.recv_process_segment_types[t][recv_idx],
                    zoid.recv_process_segment_idxs[t][recv_idx], zoid.recv_process_segment_sizes[t][recv_idx],
                    zoid.recv_process_vel_offset[t][recv_idx], zoid.recv_list_local_num_force_pos[t][recv_idx], zoid.recv_list_local_force_pos[t][recv_idx],
                    zoid.recv_ghost_num_segments[t][recv_idx], zoid.recv_ghost_idxs[t][recv_idx], zoid.recv_ghost_sizes[t][recv_idx],
                    &buf_recv_stencil_md[receive_request_idx][starting_idx + buf_offset], pbc_flag_);
        }
    }
}

void CommBrick::send_data_stencil_md_next_dt(std::array<Atom *, NUM_TIMESTEPS_IN_PARALLEL + 1> &atom_arr,
                                             queue_info &zoid) {
    assert(false);
    // TODO: checking send data for last timestep
    int zoid_num = zoid.num;
    int sz = 12;

    auto &send_to_neighbors = lmp->send_to_neighbors_next_dt[zoid_num];

    for (int i = 0; i < send_to_neighbors.size(); i++) {
        int send_zoid_num = send_to_neighbors[i];
        queue_info &send_zoid = lmp->zoid_num_to_zoid_next_dt[send_zoid_num];
        int pbc_flag_[3] = {0};
        for (int dim = 0; dim < 3; dim++) {
            if (zoid.where[dim] == RIGHT && send_zoid.where[dim] == PBC) { pbc_flag_[dim] = -1; }

            if (zoid.where[dim] == PBC && send_zoid.where[dim] == RIGHT) { pbc_flag_[dim] = 1; }
        }

        int neighbor_process = send_zoid_num % comm->nprocs;

        /*
        int num_send[NUM_TIMESTEPS_IN_PARALLEL + 1] = {0};
        int idx = 0;
        int num_elems_send[NUM_TIMESTEPS_IN_PARALLEL + 1] = {0};
        for (int t = 0; t < NUM_TIMESTEPS_IN_PARALLEL + 1; t++) {
            int num_elems_timestep = 0;
            if (DEBUG_SEND_RECV_DATA) {
                // 3 elems for data, 1 for the debug tag
                int num_force_sizes = 0;
                for (int j = 0; j < zoid.send_force_num_segments[t][i]; j++) {
                    num_elems_timestep += zoid.send_force_sizes[t][i][j] * (3 + 1);
                    num_force_sizes += zoid.send_force_sizes[t][i][j];
                }

                int num_pos_sizes = 0;
                for (int j = 0; j < zoid.send_pos_num_segments[t][i]; j++) {
                    num_elems_timestep += zoid.send_pos_sizes[t][i][j] * 2 * (3 + 1);
                    num_pos_sizes += zoid.send_pos_sizes[t][i][j];
                }

                int num_local_sizes = 0;
                int num_ghost_sizes = 0;

                for (int j = 0; j < zoid.send_num_segments[t][i]; j++) {
                    int segment_type = zoid.send_segment_types[t][i][j];
                    if (segment_type == LOCAL_SEGMENT_TYPE) {
                        num_local_sizes += zoid.send_segment_sizes[t][i][j];
                    } else {
                        assert(segment_type == GHOST_SEGMENT_TYPE);
                        num_ghost_sizes += zoid.send_segment_sizes[t][i][j];
                    }
                }

                num_elems_timestep += (num_local_sizes + num_ghost_sizes) * 2 * (3 + 1);

                std::cout << YELLOW << "NEXT DT zoid: " << zoid_num << " sending to: " << send_zoid_num << " time: " << t
                          << " num force: " << num_force_sizes << " num force segments: " << zoid.send_force_num_segments[t][i]
                          << " num pos: " << num_pos_sizes << " num pos segments: " << zoid.send_pos_num_segments[t][i]
                          << " num local to ghost: " << num_local_sizes << " num local segments: " << zoid.send_local_num_segments[t][i]
                          << " num ghost to ghost: " << num_ghost_sizes << " num ghost segments: " << zoid.send_ghost_num_segments[t][i]
                          << RESET_COLOR << std::endl;

                num_elems_send[t] = num_elems_timestep;
            }
        }

        int total_elems_calc = 0;
        for (int t = 0; t < NUM_TIMESTEPS_IN_PARALLEL + 1; t++) {
            total_elems_calc += num_elems_send[t];
        }
        */

        int nsend = 0;
        for (int t = 0; t < NUM_TIMESTEPS_IN_PARALLEL + 1; t++) {
            assert(zoid.num_elems_send[t][i] >= 0);
            nsend += zoid.num_elems_send[t][i];
        }

        if (nsend > maxsend_stencil_md[i]) {
            grow_send_stencil_md(nsend, i, 0);
        }

        int buf_idx = 0;
        for (int t = 0; t < NUM_TIMESTEPS_IN_PARALLEL + 1; t++) {
            Atom *atom_ = atom_arr[NUM_TIMESTEPS_IN_PARALLEL - t];
            bool debug = false;
            int n = atom_->avec->pack_data_stencil_md(
                    zoid.send_force_num_segments[t][i], zoid.send_pos_num_segments[t][i],
                    zoid.send_force_idxs[t][i], zoid.send_force_sizes[t][i],
                    zoid.send_pos_idxs[t][i], zoid.send_pos_sizes[t][i],
                    zoid.send_local_list[t][i],
                    zoid.send_num_segments[t][i], zoid.send_segment_types[t][i], zoid.send_segment_idxs[t][i], zoid.send_segment_sizes[t][i],
                    &buf_send_stencil_md[i][buf_idx], pbc_flag_, debug);

            buf_idx += n;
            // send_list_sendnum_stencil_md[i][t] = n;
        }

        assert(nsend == buf_idx);

        if (neighbor_process != comm->me || true) {
            MPI_Request r1;
            MPI_Request r2;
            // dst << 16 | src
            // int mpi_tag = (send_zoid_num << 16 | zoid_num);
            int mpi_tag = get_mpi_tag(send_zoid_num, zoid_num);
            /*
            MPI_Isend(send_list_sendnum_stencil_md[i], NUM_TIMESTEPS_IN_PARALLEL + 1, MPI_INT, neighbor_process, mpi_tag,
                      world, &r1);
            */
            MPI_Isend(buf_send_stencil_md[i], buf_idx, MPI_DOUBLE, neighbor_process, mpi_tag, world,
                      &r2);
        } else {

        }
    }
}

void CommBrick::unpack_data_stencil_md(std::array<Atom *, NUM_TIMESTEPS_IN_PARALLEL + 1> &atom_arr,
                                        queue_info &zoid, std::vector<MPI_Request>& receive_requests) {
    assert(false);
    int zoid_num = zoid.num;

    int receive_request_idx = 0;
    for (int i = 0; i < lmp->recv_from_neighbors[zoid_num].size(); i++) {
        int recv_zoid_num = lmp->recv_from_neighbors[zoid_num][i];
        int neighbor_process = recv_zoid_num % comm->nprocs;
        if (neighbor_process == comm->me) {
            continue;
        }

        auto begin = std::chrono::high_resolution_clock::now();

        auto start_wait = std::chrono::system_clock::now();
        const std::time_t start_wait_tc = std::chrono::system_clock::to_time_t(start_wait);

        // MPI_Status status;
        // MPI_Wait(&receive_requests[receive_request_idx++], &status);

        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end-begin).count();

        if (duration > 10000) {
            auto now = std::chrono::system_clock::now();
            const std::time_t t_c = std::chrono::system_clock::to_time_t(now);
            std::cout << YELLOW << "me: " << comm->me << " curr dt zoid: " << zoid_num << " recv from: " << recv_zoid_num << " mpi wait duration: "
                << duration << " idx: " << i << " now: " << std::put_time(std::localtime(&t_c), "%F %T.\n")
                << " start wait: " << std::put_time(std::localtime(&start_wait_tc), "%F %T.\n") << RESET_COLOR << std::endl;
        }

        int buf_recv_idx = 0;
        for (int t = 0; t < NUM_TIMESTEPS_IN_PARALLEL + 1; t++) {
            Atom* atom_ = atom_arr[t];
            atom_->avec->unpack_data_stencil_md(
                    zoid.recv_list_local_num_force_only[t][i], zoid.recv_list_local_num_force_pos[t][i],
                    zoid.recv_list_local_force_only[t][i], zoid.recv_list_local_force_pos[t][i],
                    zoid.recv_ghost_num_segments[t][i], zoid.recv_ghost_idxs[t][i], zoid.recv_ghost_sizes[t][i],
                    &buf_recv_stencil_md[i][buf_recv_idx]);
            buf_recv_idx += zoid.num_elems_recv[t][i];
        }

    }
}

void CommBrick::receive_data_stencil_md(std::array<Atom *, NUM_TIMESTEPS_IN_PARALLEL + 1> &atom_arr,
                                        queue_info &zoid, std::vector<MPI_Request>& receive_requests) {
  assert(false);
  int zoid_num = zoid.num;
  int sz = 12;

  for (int i = 0; i < lmp->recv_from_neighbors[zoid_num].size(); i++) {
    int recv_zoid_num = lmp->recv_from_neighbors[zoid_num][i];
    int neighbor_process = recv_zoid_num % comm->nprocs;
    if (neighbor_process == comm->me) {
        continue;
    }

    auto begin_for_zoid = std::chrono::high_resolution_clock::now();

    // int mpi_tag = (zoid_num << 16 | recv_zoid_num);
    int mpi_tag = get_mpi_tag(zoid_num, recv_zoid_num);

    /*
    int num_elems_recv[NUM_TIMESTEPS_IN_PARALLEL + 1] = {0};
    for (int t = 0; t < NUM_TIMESTEPS_IN_PARALLEL + 1; t++) {
        int num_elems_timestep = 0;
        if (DEBUG_SEND_RECV_DATA) {
            // 3 elems for data, 1 for the debug tag
            num_elems_timestep += zoid.recv_list_local_num_force_only[t][i] * (3 + 1);
            num_elems_timestep += zoid.recv_list_local_num_force_pos[t][i] * 2 * (3 + 1);

            int total_ghost_idxs = 0;
            for (int j = 0; j < zoid.recv_ghost_num_segments[t][i]; j++) {
                total_ghost_idxs += zoid.recv_ghost_sizes[t][i][j];
            }

            std::cout << CYAN << "zoid: " << zoid_num << " receiving from: " << recv_zoid_num << " time: " << t
                      << " num force: " << zoid.recv_list_local_num_force_only[t][i]
                      << " num pos: " << zoid.recv_list_local_num_force_pos[t][i]
                      << " num ghost: " << total_ghost_idxs << " num ghost segments: " << zoid.recv_ghost_num_segments[t][i]
                      << RESET_COLOR << std::endl;

            num_elems_timestep += total_ghost_idxs * 2 * (3 + 1);
        }

        num_elems_recv[t] = num_elems_timestep;
    }

    int total_calculated_recv = 0;
    for (int t = 0; t < NUM_TIMESTEPS_IN_PARALLEL + 1; t++) {
        total_calculated_recv += num_elems_recv[t];
    }

    int nrecv[NUM_TIMESTEPS_IN_PARALLEL + 1];
    MPI_Recv(nrecv, NUM_TIMESTEPS_IN_PARALLEL + 1, MPI_INT, neighbor_process, mpi_tag, world,
             MPI_STATUS_IGNORE);

    int total_recv = 0;
    for (int t = 0; t < NUM_TIMESTEPS_IN_PARALLEL + 1; t++) {
        total_recv += nrecv[t];
    }

    if (total_recv != total_calculated_recv) {
        std::cout << RED << "zoid: " << zoid_num << " recv: " << recv_zoid_num << " incorrect calculation. Got: " << total_recv << " calaulcated: " << total_calculated_recv << RESET_COLOR << std::endl;
    }
    assert(total_recv == total_calculated_recv);
    */

    int num_recv = 0;
    for (int t = 0; t < NUM_TIMESTEPS_IN_PARALLEL + 1; t++) {
        num_recv += zoid.num_elems_recv[t][i];
    }

    if (num_recv > maxrecv_stencil_md[i]) { grow_recv_stencil_md(num_recv, i); }

    auto begin_mpi = std::chrono::high_resolution_clock::now();
    MPI_Request request;
    MPI_Irecv(buf_recv_stencil_md[i], num_recv, MPI_DOUBLE, neighbor_process, mpi_tag, world,
              &request);
    receive_requests.push_back(request);

    auto end_mpi = std::chrono::high_resolution_clock::now();
    auto duration_mpi = std::chrono::duration_cast<std::chrono::microseconds>(end_mpi - begin_mpi).count();
  }
}

void CommBrick::receive_data_stencil_md_next_dt(std::array<Atom *, NUM_TIMESTEPS_IN_PARALLEL + 1> &atom_arr,
                                                queue_info &zoid) {
    assert(false);
    int zoid_num = zoid.num;
    int sz = 12;

    for (int i = 0; i < lmp->recv_from_neighbors_next_dt[zoid_num].size(); i++) {
        int recv_zoid_num = lmp->recv_from_neighbors_next_dt[zoid_num][i];
        int neighbor_process = recv_zoid_num % comm->nprocs;
        if (neighbor_process == comm->me) {
            continue;
        }

        int mpi_tag = (zoid_num << 16 | recv_zoid_num);
        // std::cout << RED << " next dt zoid num: " << zoid_num << " receiving from: " << recv_zoid_num << " neighbor process: " << neighbor_process << RESET_COLOR << std::endl;

        /*
        int num_elems_recv[NUM_TIMESTEPS_IN_PARALLEL + 1] = {0};
        for (int t = 0; t < NUM_TIMESTEPS_IN_PARALLEL + 1; t++) {
            int num_elems_timestep = 0;
            if (DEBUG_SEND_RECV_DATA) {
                // 3 elems for data, 1 for the debug tag
                num_elems_timestep += zoid.recv_list_local_num_force_only[t][i] * (3 + 1);
                num_elems_timestep += zoid.recv_list_local_num_force_pos[t][i] * 2 * (3 + 1);

                int total_ghost_idxs = 0;
                for (int j = 0; j < zoid.recv_ghost_num_segments[t][i]; j++) {
                    total_ghost_idxs += zoid.recv_ghost_sizes[t][i][j];
                }

                std::cout << CYAN << "NEXT DT zoid: " << zoid_num << " receiving from: " << recv_zoid_num << " time: " << t
                          << " num force: " << zoid.recv_list_local_num_force_only[t][i]
                          << " num pos: " << zoid.recv_list_local_num_force_pos[t][i]
                          << " num ghost: " << total_ghost_idxs << " num ghost segments: " << zoid.recv_ghost_num_segments[t][i]
                          << RESET_COLOR << std::endl;

                num_elems_timestep += total_ghost_idxs * 2 * (3 + 1);
            }

            num_elems_recv[t] = num_elems_timestep;
        }

        int total_calculated_recv = 0;
        for (int t = 0; t < NUM_TIMESTEPS_IN_PARALLEL + 1; t++) {
            total_calculated_recv += num_elems_recv[t];
        }

        int nrecv[NUM_TIMESTEPS_IN_PARALLEL + 1];
        MPI_Recv(nrecv, NUM_TIMESTEPS_IN_PARALLEL + 1, MPI_INT, neighbor_process, mpi_tag, world,
                 MPI_STATUS_IGNORE);

        int total_recv = 0;
        for (int t = 0; t < NUM_TIMESTEPS_IN_PARALLEL + 1; t++) {
            total_recv += nrecv[t];
        }

        if (total_recv != total_calculated_recv) {
            for (int t = 0; t < NUM_TIMESTEPS_IN_PARALLEL + 1; t++) {
                std::cout << "time: " << t << " nrecv from mpi: " << nrecv[t] << " num calc: " << num_elems_recv[t] << std::endl;
            }
            std::cout << RED << "zoid: " << zoid_num << " recv: " << recv_zoid_num << " incorrect calculation. Got: " << total_recv << " calaulcated: " << total_calculated_recv << RESET_COLOR << std::endl;
        }
        assert(total_recv == total_calculated_recv);

        if (total_recv > maxrecv_stencil_md[i]) { grow_recv_stencil_md(total_recv, i); }
        */

        int nrecv = 0;
        for (int t = 0; t < NUM_TIMESTEPS_IN_PARALLEL + 1; t++) {
            nrecv += zoid.num_elems_recv[t][i];
        }

        if (nrecv > maxrecv_stencil_md[i]) { grow_recv_stencil_md(nrecv, i); }

        MPI_Recv(buf_recv_stencil_md[i], nrecv, MPI_DOUBLE, neighbor_process, mpi_tag, world,
                 MPI_STATUS_IGNORE);

        int buf_recv_idx = 0;
        for (int t = 0; t < NUM_TIMESTEPS_IN_PARALLEL + 1; t++) {
            Atom* atom_ = atom_arr[NUM_TIMESTEPS_IN_PARALLEL - t];
            auto begin = std::chrono::high_resolution_clock::now();
            atom_->avec->unpack_data_stencil_md(
                    zoid.recv_list_local_num_force_only[t][i], zoid.recv_list_local_num_force_pos[t][i],
                    zoid.recv_list_local_force_only[t][i], zoid.recv_list_local_force_pos[t][i],
                    zoid.recv_ghost_num_segments[t][i], zoid.recv_ghost_idxs[t][i], zoid.recv_ghost_sizes[t][i],
                    &buf_recv_stencil_md[i][buf_recv_idx]);
            auto end = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end-begin).count();
            std::cout << "next dt zoid: " << zoid_num << " recv from: " << recv_zoid_num << " time: " << t << " duration: " << duration << std::endl;
            buf_recv_idx += zoid.num_elems_recv[t][i];
        }
    }
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
