// clang-format off
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

#include "comm_brick.h"

#include "atom.h"
#include "atom_vec.h"
#include "bond.h"
#include "compute.h"
#include "domain.h"
#include "dump.h"
#include "error.h"
#include "fix.h"
#include "memory.h"
#include "neighbor.h"
#include "pair.h"

#include <cmath>
#include <cstring>
#include <iostream>
#include "stencil_md.h"

using namespace LAMMPS_NS;

#define BUFFACTOR 1.5
#define BUFMIN 1024
#define BIG 1.0e20

/* ---------------------------------------------------------------------- */

CommBrick::CommBrick(LAMMPS *lmp) :
  Comm(lmp),
  sendnum(nullptr), recvnum(nullptr), sendproc(nullptr), recvproc(nullptr),
  size_forward_recv(nullptr), size_reverse_send(nullptr), size_reverse_recv(nullptr),
  slablo(nullptr), slabhi(nullptr), multilo(nullptr), multihi(nullptr),
  multioldlo(nullptr), multioldhi(nullptr), cutghostmulti(nullptr), cutghostmultiold(nullptr),
  pbc_flag(nullptr), pbc(nullptr), firstrecv(nullptr), sendlist(nullptr),
  localsendlist(nullptr), maxsendlist(nullptr), buf_send(nullptr), buf_recv(nullptr)
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

  if (sendlist) for (int i = 0; i < maxswap; i++) memory->destroy(sendlist[i]);
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
              memory->destroy(send_pos_stencil_md[i][j]);
          }

      }
      memory->sfree(sendlist_stencil_md[i]);
      memory->sfree(send_force_stencil_md[i]);
      memory->sfree(send_pos_stencil_md[i]);
      memory->destroy(maxsendlist_stencil_md[i]);

      if (second_sendlist_stencil_md[i]) {
          for (int j = 0; j < maxswap; j++) {
              memory->destroy(second_sendlist_stencil_md[i][j]);
          }
      }
      memory->sfree(second_sendlist_stencil_md[i]);
      memory->destroy(max_second_sendlist_stencil_md[i]);
  }
}

/* ---------------------------------------------------------------------- */
//IMPORTANT: we *MUST* pass "*oldcomm" to the Comm initializer here, as
//           the code below *requires* that the (implicit) copy constructor
//           for Comm is run and thus creating a shallow copy of "oldcomm".
//           The call to Comm::copy_arrays() then converts the shallow copy
//           into a deep copy of the class with the new layout.

CommBrick::CommBrick(LAMMPS * /*lmp*/, Comm *oldcomm) : Comm(*oldcomm)
{
  if (oldcomm->layout == Comm::LAYOUT_TILED)
    error->all(FLERR,"Cannot change to comm_style brick from tiled layout");

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
  CommBrick::grow_send(maxsend,2);
  memory->create(buf_recv,maxrecv,"comm:buf_recv");

  nswap = 0;
  maxswap = 26;
  CommBrick::allocate_swap(maxswap);

  sendlist = (int **) memory->smalloc(maxswap*sizeof(int *),"comm:sendlist");

  memory->create(maxsendlist,maxswap,"comm:maxsendlist");
  for (int i = 0; i < maxswap; i++) {
    maxsendlist[i] = BUFMIN;
    memory->create(sendlist[i],BUFMIN,"comm:sendlist[i]");
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

  buf_send_stencil_md = (double **) memory->smalloc(maxswap * sizeof(double*), "comm:bufsend_stencil_md");
  buf_recv_stencil_md = (double **) memory->smalloc(maxswap * sizeof(double*), "comm:bufrecv_stencil_md");
  for (int i = 0; i < maxswap; i++) {
      memory->create(buf_send_stencil_md[i], maxsend_stencil_md[i], "comm:buf_send_stencil_md");
      memory->create(buf_recv_stencil_md[i], maxrecv_stencil_md[i], "comm:buf_recv_stencil_md");
  }

  // stencil_md2
  buf_recv2_stencil_md = buf_send2_stencil_md = nullptr;
  maxsend2_stencil_md = maxrecv2_stencil_md = nullptr;
  memory->create(maxsend2_stencil_md, maxswap, "comm:maxsendlist");
  memory->create(maxrecv2_stencil_md, maxswap, "comm:maxsendlist");

  for (int i = 0; i < maxswap; i++) {
    maxsend2_stencil_md[i] = BUFMIN;
    maxrecv2_stencil_md[i] = BUFMIN;
  }

  buf_send2_stencil_md = (double **) memory->smalloc(maxswap * sizeof(double*), "comm:bufsend_stencil_md");
  buf_recv2_stencil_md = (double **) memory->smalloc(maxswap * sizeof(double*), "comm:bufrecv_stencil_md");

  for (int i = 0; i < maxswap; i++) {
      memory->create(buf_send2_stencil_md[i], maxsend2_stencil_md[i], "comm:buf_send2_stencil_md");
      memory->create(buf_recv2_stencil_md[i], maxrecv2_stencil_md[i], "comm:buf_recv_stencil_md");
  }

  for (int i = 0; i < NUM_TIMESTEPS_IN_PARALLEL + 1; i++) {
      sendlist_stencil_md[i] = (int **) memory->smalloc(maxswap*sizeof(int *),"comm:sendlist_stencil_md");
      send_force_stencil_md[i] = (bool **) memory->smalloc(maxswap*sizeof(bool*), "comm:send_force_stencil_md");
      send_pos_stencil_md[i] = (bool **) memory->smalloc(maxswap*sizeof(bool*), "comm:send_pos_stencil_md");

      memory->create(maxsendlist_stencil_md[i],maxswap,"comm:maxsendlist_stencil_md");

      for (int j = 0; j < maxswap; j++) {
          maxsendlist_stencil_md[i][j] = BUFMIN;
          memory->create(sendlist_stencil_md[i][j], BUFMIN, "comm:sendlist_stencil_md[i]");
          memory->create(send_force_stencil_md[i][j], BUFMIN, "comm:send_force_stencil_md[i]");
          memory->create(send_pos_stencil_md[i][j], BUFMIN, "comm:send_pos_stencil_md[i]");
      }

      second_sendlist_stencil_md[i] = (int **) memory->smalloc(maxswap*sizeof(int *),"comm:second_sendlist_stencil_md");
      memory->create(max_second_sendlist_stencil_md[i],maxswap,"comm:second_maxsendlist_stencil_md");

      for (int j = 0; j < maxswap; j++) {
          max_second_sendlist_stencil_md[i][j] = BUFMIN;
          memory->create(second_sendlist_stencil_md[i][j], BUFMIN, "comm:second_sendlist_stencil_md[i]");
      }
  }

  maxsend_sendlist_stencil_md = maxrecv_sendlist_stencil_md = nullptr;
  memory->create(maxsend_sendlist_stencil_md, maxswap, "comm:maxsendlist");
  memory->create(maxrecv_sendlist_stencil_md, maxswap, "comm:maxsendlist");

  for (int i = 0; i < maxswap; i++) {
    maxsend_sendlist_stencil_md[i] = BUFMIN;
    maxrecv_sendlist_stencil_md[i] = BUFMIN;
  }

  buf_sendlist_stencil_md = (int **) memory->smalloc(maxswap * sizeof(int*), "comm:bufsend_stencil_md");
  buf_recv_sendlist_stencil_md = (int **) memory->smalloc(maxswap * sizeof(int*), "comm:bufsend_stencil_md");
  for (int i = 0; i < maxswap; i++) {
    memory->create(buf_sendlist_stencil_md[i], maxsend_sendlist_stencil_md[i], "comm:buf_send_stencil_md");
    memory->create(buf_recv_sendlist_stencil_md[i], maxrecv_sendlist_stencil_md[i], "comm:buf_send_stencil_md");
  }
}

/* ---------------------------------------------------------------------- */

void CommBrick::init()
{
  Comm::init();

  int bufextra_old = bufextra;
  init_exchange();
  if (bufextra > bufextra_old) grow_send(maxsend+bufextra,2);

  // memory for multi style communication
  // allocate in setup

  if (mode == Comm::MULTI) {
    // If inconsitent # of collections, destroy any preexisting arrays (may be missized)
    if (ncollections != neighbor->ncollections) {
      ncollections = neighbor->ncollections;
      if (multilo != nullptr) {
        free_multi();
        memory->destroy(cutghostmulti);
      }
    }

    // delete any old user cutoffs if # of collections chanaged
    if (cutusermulti && ncollections != ncollections_cutoff) {
      if(me == 0) error->warning(FLERR, "cutoff/multi settings discarded, must be defined"
                                        " after customizing collections in neigh_modify");
      memory->destroy(cutusermulti);
      cutusermulti = nullptr;
    }

    if (multilo == nullptr) {
      allocate_multi(maxswap);
      memory->create(cutghostmulti,ncollections,3,"comm:cutghostmulti");
    }
  }
  if ((mode == Comm::SINGLE || mode == Comm::MULTIOLD) && multilo) {
    free_multi();
    memory->destroy(cutghostmulti);
  }

  // memory for multi/old-style communication

  if (mode == Comm::MULTIOLD && multioldlo == nullptr) {
    allocate_multiold(maxswap);
    memory->create(cutghostmultiold,atom->ntypes+1,3,"comm:cutghostmultiold");
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

  int i,j;
  int ntypes = atom->ntypes;
  double *prd,*sublo,*subhi;

  double cut = get_comm_cutoff();
  if ((cut == 0.0) && (me == 0))
    error->warning(FLERR,"Communication cutoff is 0.0. No ghost atoms "
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

      for (j = 0; j < ncollections; j++){
        if (multi_reduce && (cutcollectionsq[j][j] > cutcollectionsq[i][i])) continue;
        cutghostmulti[i][0] = MAX(cutghostmulti[i][0],sqrt(cutcollectionsq[i][j]));
        cutghostmulti[i][1] = MAX(cutghostmulti[i][1],sqrt(cutcollectionsq[i][j]));
        cutghostmulti[i][2] = MAX(cutghostmulti[i][2],sqrt(cutcollectionsq[i][j]));
      }
    }
  }

  if (mode == Comm::MULTIOLD) {
    double *cuttype = neighbor->cuttype;
    for (i = 1; i <= ntypes; i++) {
      double tmp = 0.0;
      if (cutusermultiold) tmp = cutusermultiold[i];
      cutghostmultiold[i][0] = MAX(tmp,cuttype[i]);
      cutghostmultiold[i][1] = MAX(tmp,cuttype[i]);
      cutghostmultiold[i][2] = MAX(tmp,cuttype[i]);
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
    double length0,length1,length2;
    length0 = sqrt(h_inv[0]*h_inv[0] + h_inv[5]*h_inv[5] + h_inv[4]*h_inv[4]);
    cutghost[0] = cut * length0;
    length1 = sqrt(h_inv[1]*h_inv[1] + h_inv[3]*h_inv[3]);
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
  int left,right;

  if (layout == Comm::LAYOUT_UNIFORM) {
    maxneed[0] = static_cast<int> (cutghost[0] * procgrid[0] / prd[0]) + 1;
    maxneed[1] = static_cast<int> (cutghost[1] * procgrid[1] / prd[1]) + 1;
    maxneed[2] = static_cast<int> (cutghost[2] * procgrid[2] / prd[2]) + 1;
    if (domain->dimension == 2) maxneed[2] = 0;
    if (!periodicity[0]) maxneed[0] = MIN(maxneed[0],procgrid[0]-1);
    if (!periodicity[1]) maxneed[1] = MIN(maxneed[1],procgrid[1]-1);
    if (!periodicity[2]) maxneed[2] = MIN(maxneed[2],procgrid[2]-1);

    if (!periodicity[0]) {
      recvneed[0][0] = MIN(maxneed[0],myloc[0]);
      recvneed[0][1] = MIN(maxneed[0],procgrid[0]-myloc[0]-1);
      left = myloc[0] - 1;
      if (left < 0) left = procgrid[0] - 1;
      sendneed[0][0] = MIN(maxneed[0],procgrid[0]-left-1);
      right = myloc[0] + 1;
      if (right == procgrid[0]) right = 0;
      sendneed[0][1] = MIN(maxneed[0],right);
    } else recvneed[0][0] = recvneed[0][1] =
             sendneed[0][0] = sendneed[0][1] = maxneed[0];

    if (!periodicity[1]) {
      recvneed[1][0] = MIN(maxneed[1],myloc[1]);
      recvneed[1][1] = MIN(maxneed[1],procgrid[1]-myloc[1]-1);
      left = myloc[1] - 1;
      if (left < 0) left = procgrid[1] - 1;
      sendneed[1][0] = MIN(maxneed[1],procgrid[1]-left-1);
      right = myloc[1] + 1;
      if (right == procgrid[1]) right = 0;
      sendneed[1][1] = MIN(maxneed[1],right);
    } else recvneed[1][0] = recvneed[1][1] =
             sendneed[1][0] = sendneed[1][1] = maxneed[1];

    if (!periodicity[2]) {
      recvneed[2][0] = MIN(maxneed[2],myloc[2]);
      recvneed[2][1] = MIN(maxneed[2],procgrid[2]-myloc[2]-1);
      left = myloc[2] - 1;
      if (left < 0) left = procgrid[2] - 1;
      sendneed[2][0] = MIN(maxneed[2],procgrid[2]-left-1);
      right = myloc[2] + 1;
      if (right == procgrid[2]) right = 0;
      sendneed[2][1] = MIN(maxneed[2],right);
    } else recvneed[2][0] = recvneed[2][1] =
             sendneed[2][0] = sendneed[2][1] = maxneed[2];

  } else {
    recvneed[0][0] = updown(0,0,myloc[0],prd[0],periodicity[0],xsplit);
    recvneed[0][1] = updown(0,1,myloc[0],prd[0],periodicity[0],xsplit);
    left = myloc[0] - 1;
    if (left < 0) left = procgrid[0] - 1;
    sendneed[0][0] = updown(0,1,left,prd[0],periodicity[0],xsplit);
    right = myloc[0] + 1;
    if (right == procgrid[0]) right = 0;
    sendneed[0][1] = updown(0,0,right,prd[0],periodicity[0],xsplit);

    recvneed[1][0] = updown(1,0,myloc[1],prd[1],periodicity[1],ysplit);
    recvneed[1][1] = updown(1,1,myloc[1],prd[1],periodicity[1],ysplit);
    left = myloc[1] - 1;
    if (left < 0) left = procgrid[1] - 1;
    sendneed[1][0] = updown(1,1,left,prd[1],periodicity[1],ysplit);
    right = myloc[1] + 1;
    if (right == procgrid[1]) right = 0;
    sendneed[1][1] = updown(1,0,right,prd[1],periodicity[1],ysplit);

    if (domain->dimension == 3) {
      recvneed[2][0] = updown(2,0,myloc[2],prd[2],periodicity[2],zsplit);
      recvneed[2][1] = updown(2,1,myloc[2],prd[2],periodicity[2],zsplit);
      left = myloc[2] - 1;
      if (left < 0) left = procgrid[2] - 1;
      sendneed[2][0] = updown(2,1,left,prd[2],periodicity[2],zsplit);
      right = myloc[2] + 1;
      if (right == procgrid[2]) right = 0;
      sendneed[2][1] = updown(2,0,right,prd[2],periodicity[2],zsplit);
    } else recvneed[2][0] = recvneed[2][1] =
             sendneed[2][0] = sendneed[2][1] = 0;

    int all[6];
    MPI_Allreduce(&recvneed[0][0],all,6,MPI_INT,MPI_MAX,world);
    maxneed[0] = MAX(all[0],all[1]);
    maxneed[1] = MAX(all[2],all[3]);
    maxneed[2] = MAX(all[4],all[5]);
  }

  // allocate comm memory

  nswap = 2 * (maxneed[0]+maxneed[1]+maxneed[2]);
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

  int dim,ineed;

  int iswap = 0;
  for (dim = 0; dim < 3; dim++) {
    for (ineed = 0; ineed < 2*maxneed[dim]; ineed++) {
      pbc_flag[iswap] = 0;
      pbc[iswap][0] = pbc[iswap][1] = pbc[iswap][2] =
        pbc[iswap][3] = pbc[iswap][4] = pbc[iswap][5] = 0;

      if (ineed % 2 == 0) {
        sendproc[iswap] = procneigh[dim][0];
        recvproc[iswap] = procneigh[dim][1];
        if (mode == Comm::SINGLE) {
          if (ineed < 2) slablo[iswap] = -BIG;
          else slablo[iswap] = 0.5 * (sublo[dim] + subhi[dim]);
          slabhi[iswap] = sublo[dim] + cutghost[dim];
        } else if (mode == Comm::MULTI) {
          for (i = 0; i < ncollections; i++) {
            if (ineed < 2) multilo[iswap][i] = -BIG;
            else multilo[iswap][i] = 0.5 * (sublo[dim] + subhi[dim]);
            multihi[iswap][i] = sublo[dim] + cutghostmulti[i][dim];
          }
        } else {
          for (i = 1; i <= ntypes; i++) {
            if (ineed < 2) multioldlo[iswap][i] = -BIG;
            else multioldlo[iswap][i] = 0.5 * (sublo[dim] + subhi[dim]);
            multioldhi[iswap][i] = sublo[dim] + cutghostmultiold[i][dim];
          }
        }
        if (myloc[dim] == 0) {
          pbc_flag[iswap] = 1;
          pbc[iswap][dim] = 1;
          if (triclinic) {
            if (dim == 1) pbc[iswap][5] = 1;
            else if (dim == 2) pbc[iswap][4] = pbc[iswap][3] = 1;
          }
        }

      } else {
        sendproc[iswap] = procneigh[dim][1];
        recvproc[iswap] = procneigh[dim][0];
        if (mode == Comm::SINGLE) {
          slablo[iswap] = subhi[dim] - cutghost[dim];
          if (ineed < 2) slabhi[iswap] = BIG;
          else slabhi[iswap] = 0.5 * (sublo[dim] + subhi[dim]);
        } else if (mode == Comm::MULTI) {
          for (i = 0; i < ncollections; i++) {
            multilo[iswap][i] = subhi[dim] - cutghostmulti[i][dim];
            if (ineed < 2) multihi[iswap][i] = BIG;
            else multihi[iswap][i] = 0.5 * (sublo[dim] + subhi[dim]);
          }
        } else {
          for (i = 1; i <= ntypes; i++) {
            multioldlo[iswap][i] = subhi[dim] - cutghostmultiold[i][dim];
            if (ineed < 2) multioldhi[iswap][i] = BIG;
            else multioldhi[iswap][i] = 0.5 * (sublo[dim] + subhi[dim]);
          }
        }
        if (myloc[dim] == procgrid[dim]-1) {
          pbc_flag[iswap] = 1;
          pbc[iswap][dim] = -1;
          if (triclinic) {
            if (dim == 1) pbc[iswap][5] = -1;
            else if (dim == 2) pbc[iswap][4] = pbc[iswap][3] = -1;
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
  int index,count;
  double frac,delta;

  if (dir == 0) {
    frac = cutghost[dim]/prd;
    index = loc - 1;
    delta = 0.0;
    count = 0;
    while (delta < frac) {
      if (index < 0) {
        if (!periodicity) break;
        index = procgrid[dim] - 1;
      }
      count++;
      delta += split[index+1] - split[index];
      index--;
    }

  } else {
    frac = cutghost[dim]/prd;
    index = loc + 1;
    delta = 0.0;
    count = 0;
    while (delta < frac) {
      if (index >= procgrid[dim]) {
        if (!periodicity) break;
        index = 0;
      }
      count++;
      delta += split[index+1] - split[index];
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

  // comm_x_only means only the position is needed for the pair which is the case for allegro, do not need velocity
  // firstrecv is the position in which received atoms can be added, x is a double**, first dimension is n_local + n_ghost + ___ (we fill in the last ___)
  //
  for (int iswap = 0; iswap < nswap; iswap++) {
    if (sendproc[iswap] != me) {
      if (comm_x_only) {
        if (size_forward_recv[iswap]) {
          buf = x[firstrecv[iswap]];
          MPI_Irecv(buf,size_forward_recv[iswap],MPI_DOUBLE,recvproc[iswap],0,world,&request);
        }
        n = avec->pack_comm(sendnum[iswap],sendlist[iswap],buf_send,pbc_flag[iswap],pbc[iswap]);
        if (n) MPI_Send(buf_send,n,MPI_DOUBLE,sendproc[iswap],0,world);
        if (size_forward_recv[iswap]) MPI_Wait(&request,MPI_STATUS_IGNORE);
      } else if (ghost_velocity) {
        if (size_forward_recv[iswap])
          MPI_Irecv(buf_recv,size_forward_recv[iswap],MPI_DOUBLE,recvproc[iswap],0,world,&request);
        n = avec->pack_comm_vel(sendnum[iswap],sendlist[iswap],buf_send,pbc_flag[iswap],pbc[iswap]);
        if (n) MPI_Send(buf_send,n,MPI_DOUBLE,sendproc[iswap],0,world);
        if (size_forward_recv[iswap]) MPI_Wait(&request,MPI_STATUS_IGNORE);
        avec->unpack_comm_vel(recvnum[iswap],firstrecv[iswap],buf_recv);
      } else {
        if (size_forward_recv[iswap])
          MPI_Irecv(buf_recv,size_forward_recv[iswap],MPI_DOUBLE,
                    recvproc[iswap],0,world,&request);
        n = avec->pack_comm(sendnum[iswap],sendlist[iswap],buf_send,pbc_flag[iswap],pbc[iswap]);
        if (n) MPI_Send(buf_send,n,MPI_DOUBLE,sendproc[iswap],0,world);
        if (size_forward_recv[iswap]) MPI_Wait(&request,MPI_STATUS_IGNORE);
        avec->unpack_comm(recvnum[iswap],firstrecv[iswap],buf_recv);
      }

    } else {
      if (comm_x_only) {
        if (sendnum[iswap])
          avec->pack_comm(sendnum[iswap],sendlist[iswap],
                          x[firstrecv[iswap]],pbc_flag[iswap],pbc[iswap]);
      } else if (ghost_velocity) {
        avec->pack_comm_vel(sendnum[iswap],sendlist[iswap],buf_send,pbc_flag[iswap],pbc[iswap]);
        avec->unpack_comm_vel(recvnum[iswap],firstrecv[iswap],buf_send);
      } else {
        avec->pack_comm(sendnum[iswap],sendlist[iswap],buf_send,pbc_flag[iswap],pbc[iswap]);
        avec->unpack_comm(recvnum[iswap],firstrecv[iswap],buf_send);
      }
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

  for (int iswap = nswap-1; iswap >= 0; iswap--) {
    if (sendproc[iswap] != me) {
      if (comm_f_only) {
        if (size_reverse_recv[iswap])
          MPI_Irecv(buf_recv,size_reverse_recv[iswap],MPI_DOUBLE,sendproc[iswap],0,world,&request);
        if (size_reverse_send[iswap]) {
          buf = f[firstrecv[iswap]];
          MPI_Send(buf,size_reverse_send[iswap],MPI_DOUBLE,recvproc[iswap],0,world);
        }
        if (size_reverse_recv[iswap]) MPI_Wait(&request,MPI_STATUS_IGNORE);
      } else {
        if (size_reverse_recv[iswap])
          MPI_Irecv(buf_recv,size_reverse_recv[iswap],MPI_DOUBLE,sendproc[iswap],0,world,&request);
        n = avec->pack_reverse(recvnum[iswap],firstrecv[iswap],buf_send);
        if (n) MPI_Send(buf_send,n,MPI_DOUBLE,recvproc[iswap],0,world);
        if (size_reverse_recv[iswap]) MPI_Wait(&request,MPI_STATUS_IGNORE);
      }
      avec->unpack_reverse(sendnum[iswap],sendlist[iswap],buf_recv);
    } else {
      if (comm_f_only) {
        if (sendnum[iswap])
          avec->unpack_reverse(sendnum[iswap],sendlist[iswap],f[firstrecv[iswap]]);
      } else {
        avec->pack_reverse(recvnum[iswap],firstrecv[iswap],buf_send);
        avec->unpack_reverse(sendnum[iswap],sendlist[iswap],buf_send);
      }
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
  int i,m,nsend,nrecv,nrecv1,nrecv2,nlocal;
  double lo,hi,value;
  double **x;
  double *sublo,*subhi;
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
    int bufextra_old = bufextra;
    init_exchange();
    if (bufextra > bufextra_old) grow_send(maxsend+bufextra,2);
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
        if (nsend > maxsend) grow_send(nsend,1);
        nsend += avec->pack_exchange(i,&buf_send[nsend]);
        avec->copy(nlocal-1,i,1);
        nlocal--;
      } else i++;
    }
    atom->nlocal = nlocal;

    // send/recv atoms in both directions
    // send size of message first so receiver can realloc buf_recv if needed
    // if 1 proc in dimension, no send/recv
    //   set nrecv = 0 so buf_send atoms will be lost
    // if 2 procs in dimension, single send/recv
    // if more than 2 procs in dimension, send/recv to both neighbors

    if (procgrid[dim] == 1) nrecv = 0;
    else {
      MPI_Sendrecv(&nsend,1,MPI_INT,procneigh[dim][0],0,
                   &nrecv1,1,MPI_INT,procneigh[dim][1],0,world,MPI_STATUS_IGNORE);
      nrecv = nrecv1;
      if (procgrid[dim] > 2) {
        MPI_Sendrecv(&nsend,1,MPI_INT,procneigh[dim][1],0,
                     &nrecv2,1,MPI_INT,procneigh[dim][0],0,world,MPI_STATUS_IGNORE);
        nrecv += nrecv2;
      }
      if (nrecv > maxrecv) grow_recv(nrecv);

      MPI_Irecv(buf_recv,nrecv1,MPI_DOUBLE,procneigh[dim][1],0,world,&request);
      MPI_Send(buf_send,nsend,MPI_DOUBLE,procneigh[dim][0],0,world);
      MPI_Wait(&request,MPI_STATUS_IGNORE);

      if (procgrid[dim] > 2) {
        MPI_Irecv(&buf_recv[nrecv1],nrecv2,MPI_DOUBLE,procneigh[dim][0],0,world,&request);
        MPI_Send(buf_send,nsend,MPI_DOUBLE,procneigh[dim][1],0,world);
        MPI_Wait(&request,MPI_STATUS_IGNORE);
      }
    }

    // check incoming atoms to see if they are in my box
    // if so, add to my list
    // box check is only for this dimension,
    //   atom may be passed to another proc in later dims

    m = 0;
    while (m < nrecv) {
      value = buf_recv[m+dim+1];
      if (value >= lo && value < hi) {
          m += avec->unpack_exchange(&buf_recv[m]);
      }
      else {
          m += static_cast<int> (buf_recv[m]);
      }
    }
  }

  if (atom->firstgroupname) atom->first_reorder();
}

// send the data from the current process to the domains created by stencil_md
// calls the "lammps" version of atom and comm and clears everything out, migrates all the shit over to "stencil_md"
// send to process that owns the zoid
void CommBrick::exchange_stencil_md_initial_send() {
    int i,m,nsend,nrecv,nrecv1,nrecv2,nlocal;
    double lo,hi,value;
    double **x;
    double *sublo,*subhi;
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
        if (bufextra > bufextra_old) grow_send(maxsend+bufextra,2);
    }

    // fill buffer with atoms leaving my box, using < and >=
    // when atom is deleted, fill it in with last atom

    x = atom->x;
    nlocal = atom->nlocal;
    i = nsend = 0;

    // send out all atoms from the main process, super inefficient but it's the initial step so do whatever is necessary
    while (i < nlocal) {
        if (nsend > maxsend) grow_send(nsend,1);
        nsend += avec->pack_exchange(i,&buf_send[nsend]);
        // avec->copy(nlocal-1,i,1);
        // nlocal--;
        i++;
    }
    // atom->nlocal = nlocal;
    // send atoms to zoids from dep level 0 to 4.
    std::vector<MPI_Request> requests;
    for (int dep = 0; dep < NUM_DEPS; dep++) {
        for (int j = 0; j < lmp->queues[dep].size(); j++) {
            queue_info& zoid = lmp->queues[dep][j];
            int zoid_num = zoid.num;
            // maybe do isend
            // MPI_Send(&nsend, 1, MPI_INT, zoid_num % comm->nprocs, zoid_num, world);
            MPI_Request r1;
            MPI_Request r2;
            MPI_Isend(&nsend, 1, MPI_INT, zoid_num % comm->nprocs, zoid_num, world, &r1);
            MPI_Isend(buf_send, nsend, MPI_DOUBLE, zoid_num % comm->nprocs, zoid_num, world, &r2);
            requests.push_back(r1);
            requests.push_back(r2);
        }
    }

    if (atom->firstgroupname) {
        assert(false);
        atom->first_reorder();
    }
}

void CommBrick::exchange_stencil_md_initial_receive(Atom* atom_, Domain* domain_, queue_info& zoid) {
    AtomVec *avec = atom_->avec;
    if (map_style != Atom::MAP_NONE) atom_->map_clear();
    atom_->nghost = 0;
    atom_->avec->clear_bonus();

    if (maxexchange_fix_dynamic) {
        assert(false);
        int bufextra_old = bufextra;
        init_exchange();
        if (bufextra > bufextra_old) grow_send(maxsend+bufextra,2);
    }

    double* sublo = domain_->sublo;
    double* subhi = domain_->subhi;

    /*
    for (int i = 0; i < 3; i++) {
        if (fabs(domain_->sublo[i] - zoid.zoid.cuts[i].lower) > 1e-8) {
            std::cout << "sublo: " << domain_->sublo[i] << " lower: " << zoid.zoid.cuts[i].lower << std::endl;
        }
        if (fabs(domain_->subhi[i] - zoid.zoid.cuts[i].upper) > 1e-8) {
            std::cout << "subhi: " << domain_->subhi[i] << " upper: " << zoid.zoid.cuts[i].upper << std::endl;
        }
        assert(fabs(domain_->sublo[i] - zoid.zoid.cuts[i].lower) <= 1e-8);
        assert(fabs(domain_->subhi[i] - zoid.zoid.cuts[i].upper) <= 1e-8);
    }
    */

    int dimension = domain_->dimension;
    assert(dimension == 3);

    double** x = atom_->x;

    std::set<int> tag_set;
    std::map<int, int> tag_to_source;
    int zoid_num = zoid.num;
    int num_in_zoid = 0;
    for (int i = 0; i < comm->nprocs; i++) {
        int nrecv;
        MPI_Recv(&nrecv, 1, MPI_INT, i, zoid_num, world, MPI_STATUS_IGNORE);
        if (nrecv*size_border > maxrecv) {
            grow_recv(nrecv*size_border);
        }
        MPI_Recv(buf_recv,nrecv,MPI_DOUBLE,i,zoid_num,world, MPI_STATUS_IGNORE);
        int m = 0;
        while (m < nrecv) {
            // try remapping the zoid
            bool in_zoid = true;
            for (int dim = 0; dim < domain->dimension; dim++) {
                double lo = sublo[dim];
                double hi = subhi[dim];
                double value = buf_recv[m+dim+1];
                bool prev = true;
                if (lo < 0) {
                    in_zoid = in_zoid && ((value >= domain_->prd[dim] + lo) || (value < hi));
                } else {
                    in_zoid = in_zoid && (value >= lo) && (value < hi);
                }
            }
            if (in_zoid) {
                std::set<int> tag_tmp;
                m += atom_->avec->unpack_exchange_stencil_md(&buf_recv[m], atom_, domain_, tag_tmp);
                num_in_zoid++;
                /*
                if (atom_->tag[atom_->nlocal - 1] == 14021) {
                    std::cout << "RYAN-------- HELP----" << " zoid num: " << zoid.num << std::endl;
                    std::cout << " Lo: " << sublo[0] << " " << sublo[1] << " " << sublo[2] << std::endl;
                    std::cout << " Hi: " << subhi[0] << " " << subhi[1] << " " << subhi[2] << std::endl;
                }
                */
            } else {
                m += static_cast<int> (buf_recv[m]);
            }
        }
    }
    domain_->remap_all_stencil_md(atom_);
    if (atom_->firstgroupname) atom_->first_reorder();
}

/*
void CommBrick::exchange_stencil_md_initial_receive(std::array<Atom*, NUM_TIMESTEPS_IN_PARALLEL + 1> atom_arr, std::array<Domain*, NUM_TIMESTEPS_IN_PARALLEL> domain_arr, queue_info& zoid) {
    for (int i = 0; i < atom_arr.size(); i++) {
        atom_arr[i]->nghost = 0;
        atom_arr[i]->avec->clear_bonus();
        if (map_style != Atom::MAP_NONE) {
            atom_arr[i]->map_clear();
        }
    }

    Atom* atom_ = atom_arr[0];
    Domain* domain_ = domain_arr[0];
    AtomVec *avec = atom_->avec;
    // atom_->nghost = 0;
    // atom_->avec->clear_bonus();

    if (maxexchange_fix_dynamic) {
        assert(false);
        int bufextra_old = bufextra;
        init_exchange();
        if (bufextra > bufextra_old) grow_send(maxsend+bufextra,2);
    }

    double* sublo = domain_->sublo;
    double* subhi = domain_->subhi;

    for (int i = 0; i < 3; i++) {
        assert(fabs(domain_->sublo[i] - zoid.zoid.cuts[i].lower) <= 1e-8);
        assert(fabs(domain_->subhi[i] - zoid.zoid.cuts[i].upper) <= 1e-8);
    }

    int dimension = domain_->dimension;
    assert(dimension == 3);

    double** x = atom_->x;

    std::set<int> tag_set;
    std::map<int, int> tag_to_source;
    int zoid_num = zoid.num;
    int num_in_zoid = 0;
    for (int i = 0; i < comm->nprocs; i++) {
        int nrecv;
        MPI_Recv(&nrecv, 1, MPI_INT, i, zoid_num, world, MPI_STATUS_IGNORE);
        if (nrecv*size_border > maxrecv) {
            grow_recv(nrecv*size_border);
        }
        MPI_Recv(buf_recv,nrecv,MPI_DOUBLE,i,zoid_num,world, MPI_STATUS_IGNORE);
        int m = 0;
        while (m < nrecv) {
            // try remapping the zoid
            bool in_zoid = true;
            for (int dim = 0; dim < domain_->dimension; dim++) {
                double lo = sublo[dim];
                double hi = subhi[dim];
                double value = buf_recv[m+dim+1];
                bool prev = true;
                if (lo < 0) {
                    in_zoid = in_zoid && ((value >= domain_->prd[dim] + lo) || (value < hi));
                } else {
                    in_zoid = in_zoid && (value >= lo) && (value < hi);
                }
            }
            if (in_zoid) {
                std::set<int> tag_tmp;
                m += atom_->avec->unpack_exchange_stencil_md(&buf_recv[m], atom_, domain_, tag_tmp);
                num_in_zoid++;
            } else {
                m += static_cast<int> (buf_recv[m]);
            }
        }
    }

    // fill out the remaining atoms for nlocal
    for (int i = 1; i < atom_arr.size(); i++) {
        Atom* target_atom = atom_arr[i];
        Domain* target_domain = domain_arr[i];
        double* sublo_target = target_domain->sublo;
        double* subhi_target = target_domain->subhi;
        for (int j = 0; j < atom_->nlocal; j++) {
            bool in_zoid = true;
            for (int dim = 0; dim < domain_->dimension; dim++) {
                double lo = sublo_target[dim];
                double hi = subhi_target[dim];
                // double value = buf_recv[m+dim+1];
                double value = atom_->x[j][dim];
                bool prev = true;
                if (lo < 0) {
                    in_zoid = in_zoid && ((value >= domain_->prd[dim] + lo) || (value < hi));
                } else {
                    in_zoid = in_zoid && (value >= lo) && (value < hi);
                }
            }
            if (in_zoid) {
                target_atom->avec->add_local_atom_stencil_md(target_atom, target_domain,
                                                             atom_->x[j], atom_->v[j],
                                                             atom_->tag[j], atom_->type[j],
                                                             atom_->mask[j], atom_->image[j]);
            }
        }
    }

    if (atom_->firstgroupname) {
        assert(false);
        atom_->first_reorder();
    }
}
*/

void CommBrick::construct_send_list_stencil_md(std::array<Atom*, NUM_TIMESTEPS_IN_PARALLEL + 1>& atom_arr, queue_info& zoid) {
    int zoid_num = zoid.num;

    int max_atoms = -1;
    for (int i = 0; i < atom_arr.size(); i++) {
        Atom* atom_ = atom_arr[i];
        int num_atoms = atom_->nlocal + atom_->nghost;
        if (num_atoms > max_atoms) {
            max_atoms = num_atoms;
        }
    }

    bool *can_send[NUM_TIMESTEPS_IN_PARALLEL + 1];

    for (int t = 0; t < NUM_TIMESTEPS_IN_PARALLEL + 1; t++) {
        can_send[t] = new bool[max_atoms];
    }

    for (int i = 0; i < lmp->send_to_neighbors[zoid_num].size(); i++) {
        int send_zoid_num = lmp->send_to_neighbors[zoid_num][i];
        queue_info& send_zoid = lmp->zoid_num_to_zoid[send_zoid_num];

        for (int j = 0; j < NUM_TIMESTEPS_IN_PARALLEL + 1; j++) {
            memset(can_send[j], false, max_atoms);
        }

        for (int j = 0; j < atom_arr[0]->nlocal + atom_arr[0]->nghost; j++) {
            can_send[0][j] = true;
        }

        for (int t = 1; t < NUM_TIMESTEPS_IN_PARALLEL; t++) {
            for (int j = 0; j < atom_arr[t]->nlocal; j++) {
                can_send[t][j] = true;
            }
        }

        for (int t = 0; t < NUM_TIMESTEPS_IN_PARALLEL + 1; t++) {
            int nsend_stencil_md = 0;
            Atom* atom_ = atom_arr[t];

            for (int j = 0; j < atom_->nlocal + atom_->nghost; j++) {
                if (j < atom_->nlocal && t < NUM_TIMESTEPS_IN_PARALLEL) {
                    can_send[t + 1][zoid.atom_idx_mapping[t][j]] = true;
                }

//                if (!can_send[t][j]) {
//                    continue;
//                }

                double* pos = atom_->x[j];
                bool in_zoid = true;
                bool borders_zoid = true;

                for (int dim = 0; dim < domain->dimension; dim++) {
                    double lo = send_zoid.zoid.cuts[dim].lower + (t) * send_zoid.zoid.cuts[dim].slope_lower;
                    double hi = send_zoid.zoid.cuts[dim].upper + (t) * send_zoid.zoid.cuts[dim].slope_upper;

                    double value = pos[dim];

                    int pbc_ = 0;
                    if (zoid.where[dim] == RIGHT && send_zoid.where[dim] == PBC) {
                        pbc_ = -1;
                    }

                    if (zoid.where[dim] == PBC && send_zoid.where[dim] == RIGHT) {
                        pbc_ = 1;
                    }

                    double atom_pos_shifted = value + pbc_ * domain->prd[dim];
                    in_zoid = in_zoid && ((atom_pos_shifted >= lo && atom_pos_shifted <= hi));

                    double borders_lo = lo - ALLEGRO_SLOPE;
                    double borders_hi = hi + ALLEGRO_SLOPE;
                    borders_zoid = borders_zoid && ((atom_pos_shifted >= borders_lo && atom_pos_shifted <= borders_hi));
                }

                // atoms that are neighbors of neighbors also need to be send
                if (in_zoid || borders_zoid) {
                    assert(borders_zoid);
                    if (nsend_stencil_md == maxsendlist_stencil_md[t][i]) {
                        grow_list_stencil_md(i, nsend_stencil_md, t);
                    }
                    sendlist_stencil_md[t][i][nsend_stencil_md] = j;
                    send_force_stencil_md[t][i][nsend_stencil_md] = in_zoid;
                    send_pos_stencil_md[t][i][nsend_stencil_md] = can_send[t][j];

                    if (send_zoid_num == 19 && atom_->tag[j] == 11338 && t == 1) {
                        std::cout << "zoid num: " << zoid_num << std::endl;
                        std::cout << "Send force? " << send_force_stencil_md[t][i][nsend_stencil_md] << std::endl;
                        std::cout << "pos: " << atom_->x[j][0] << " " << atom_->x[j][1] << " " << atom_->x[j][2] << std::endl;

                        for (int dim = 0; dim < domain->dimension; dim++) {
                            double lo = send_zoid.zoid.cuts[dim].lower + (t) * send_zoid.zoid.cuts[dim].slope_lower;
                            double hi = send_zoid.zoid.cuts[dim].upper + (t) * send_zoid.zoid.cuts[dim].slope_upper;
                            std::cout << "send_zoid lo: " << lo << " hi: " << hi << std::endl;
                        }

                        for (int dim = 0; dim < domain->dimension; dim++) {
                            double lo = zoid.zoid.cuts[dim].lower + (t) * zoid.zoid.cuts[dim].slope_lower;
                            double hi = zoid.zoid.cuts[dim].upper + (t) * zoid.zoid.cuts[dim].slope_upper;
                            std::cout << "my zoid lo: " << lo << " hi: " << hi << std::endl;
                        }
                    }
                    /*
                    if (zoid_num == 1 &&
                        send_zoid_num == 19 && atom_->tag[j] == 10429 && t == 1) {
                        std::cout << "Send force? " << send_force_stencil_md[t][i][nsend_stencil_md] << std::endl;
                        std::cout << "pos: " << atom_->x[j][0] << " " << atom_->x[j][1] << " " << atom_->x[j][2] << std::endl;

                        for (int dim = 0; dim < domain->dimension; dim++) {
                            double lo = send_zoid.zoid.cuts[dim].lower + (t) * send_zoid.zoid.cuts[dim].slope_lower;
                            double hi = send_zoid.zoid.cuts[dim].upper + (t) * send_zoid.zoid.cuts[dim].slope_upper;
                            std::cout << "send_zoid lo: " << lo << " hi: " << hi << std::endl;
                        }
                    }
                    */
                    nsend_stencil_md++;
                }
            }

            sendnum_stencil_md[t][i] = nsend_stencil_md;
            assert(nsend_stencil_md >= 0);

            // assert(send_atoms.size() == nsend_stencil_md);
            std::set<int> test_send_atoms;
            for (int h = 0; h < nsend_stencil_md; h++) {
                test_send_atoms.insert(sendlist_stencil_md[t][i][h]);
            }
            assert(test_send_atoms.size() == nsend_stencil_md);
        }



        /*
        for (int t = 0; t < NUM_TIMESTEPS_IN_PARALLEL + 1; t++) {
            std::set<int> send_atoms;
            Atom* atom_ = atom_arr[t];
            int nsend_stencil_md = 0;
            // TODO: since there is overlap, potentially send local->local?
            for (int j = 0; j < atom_->nlocal; j++) {
                // check if local in next_zoid?
                double* pos = atom_->x[j];
                bool in_zoid = true;
                bool borders_zoid = true;

                for (int dim = 0; dim < domain->dimension; dim++) {
                    double lo = send_zoid.zoid.cuts[dim].lower + (t) * send_zoid.zoid.cuts[dim].slope_lower;
                    double hi = send_zoid.zoid.cuts[dim].upper + (t) * send_zoid.zoid.cuts[dim].slope_upper;

                    double value = pos[dim];

                    int pbc_ = 0;
                    if (zoid.where[dim] == RIGHT && send_zoid.where[dim] == PBC) {
                        pbc_ = -1;
                    }

                    if (zoid.where[dim] == PBC && send_zoid.where[dim] == RIGHT) {
                        pbc_ = 1;
                    }

                    double atom_pos_shifted = value + pbc_ * domain->prd[dim];
                    in_zoid = in_zoid && ((atom_pos_shifted >= lo && atom_pos_shifted <= hi));

                    double borders_lo = lo - ALLEGRO_SLOPE;
                    double borders_hi = hi + ALLEGRO_SLOPE;
                    borders_zoid = borders_zoid && ((atom_pos_shifted >= borders_lo && atom_pos_shifted <= borders_hi));
                }

                // atoms that are neighbors of neighbors also need to be send
                if (in_zoid) {
                    assert(borders_zoid);
                    if (nsend_stencil_md == maxsendlist_stencil_md[t][i]) {
                        grow_list_stencil_md(i, nsend_stencil_md, t);
                    }
                    sendlist_stencil_md[t][i][nsend_stencil_md] = j;
                    send_force_stencil_md[t][i][nsend_stencil_md] = !borders_zoid;
                    nsend_stencil_md++;
                    send_atoms.insert(j);
                }
            }

            for (int j = atom_->nlocal; j < atom_->nghost + atom_->nlocal; j++) {
                // for (int j = atom_->nlocal; j < atom_->nghost + atom_->nlocal; j++) {
                // check if local in next_zoid?
                double* pos = atom_->x[j];
                bool in_zoid = true;
                bool borders_zoid = true;

                for (int dim = 0; dim < domain->dimension; dim++) {
                    double lo = send_zoid.zoid.cuts[dim].lower + (t) * send_zoid.zoid.cuts[dim].slope_lower;
                    double hi = send_zoid.zoid.cuts[dim].upper + (t) * send_zoid.zoid.cuts[dim].slope_upper;

                    lo -= ALLEGRO_SLOPE;
                    hi += ALLEGRO_SLOPE;

                    double value = pos[dim];

                    int pbc_ = 0;
                    if (zoid.where[dim] == RIGHT && send_zoid.where[dim] == PBC) {
                        pbc_ = -1;
                    }

                    if (zoid.where[dim] == PBC && send_zoid.where[dim] == RIGHT) {
                        pbc_ = 1;
                    }

                    double atom_pos_shifted = value + pbc_ * domain->prd[dim];
                    in_zoid = in_zoid && ((atom_pos_shifted >= lo && atom_pos_shifted <= hi));

                    double borders_lo = lo - ALLEGRO_SLOPE;
                    double borders_hi = hi + ALLEGRO_SLOPE;
                    borders_zoid = borders_zoid && ((atom_pos_shifted >= borders_lo && atom_pos_shifted <= borders_hi));
                }

                // atoms that are neighbors of neighbors also need to be send
                if (in_zoid) {
                    assert(borders_zoid);
                    if (nsend_stencil_md == maxsendlist_stencil_md[t][i]) {
                        grow_list_stencil_md(i, nsend_stencil_md, t);
                    }
                    sendlist_stencil_md[t][i][nsend_stencil_md] = j;
                    send_force_stencil_md[t][i][nsend_stencil_md] = !borders_zoid;
                    nsend_stencil_md++;
                    send_atoms.insert(j);
                }
            }


            sendnum_stencil_md[t][i] = nsend_stencil_md;
            assert(nsend_stencil_md >= 0);

            // assert(send_atoms.size() == nsend_stencil_md);
            std::set<int> test_send_atoms;
            for (int h = 0; h < nsend_stencil_md; h++) {
                test_send_atoms.insert(sendlist_stencil_md[t][i][h]);
            }
            assert(test_send_atoms.size() == nsend_stencil_md);
        }
        */
    }

    for (int j = 0; j < NUM_TIMESTEPS_IN_PARALLEL + 1; j++) {
        delete[] can_send[j];
    }

}

// after grouping the ghost atoms
void CommBrick::construct_second_send_list_stencil_md_send(std::array<Atom*, NUM_TIMESTEPS_IN_PARALLEL + 1>& atom_arr, queue_info& zoid) {
    /*
    int zoid_num = zoid.num;
    auto& recv_from = lmp->recv_from[zoid_num];

    for (int i = 0; i < recv_from.size(); i++) {
        // receive from send_to
        int send_zoid_num = recv_from[i];
        queue_info& send_zoid = lmp->zoid_num_to_zoid[send_zoid_num];

        int num_send_per_timestep[NUM_TIMESTEPS_IN_PARALLEL + 1] = {0};

        std::vector<int> tags;
        for (int t = 1; t < NUM_TIMESTEPS_IN_PARALLEL; t++) {
            Atom* atom_ = atom_arr[t];
            int nsend_stencil_md = 0;
            int second_nsend_stencil_md = 0;

            // second_recv
            int start_idx = zoid.second_recv_stencil_md[t][i];
            int end_idx;
            if (i == recv_from.size() - 1) {
                end_idx = atom_->nlocal + atom_->nghost;
            } else {
                end_idx = zoid.second_recv_stencil_md[t][i + 1];
            }

            // std::cout << "start idx: " << start_idx << " end_idx: " << end_idx << " i: " << i << " recv_from.size() " << recv_from.size() << std::endl;

            int nsend = end_idx - start_idx;
            num_send_per_timestep[t] = nsend;

            for (int idx = start_idx; idx < end_idx; idx++) {
                tags.push_back(atom_->tag[idx]);
            }
        }

        int total = 0;
        for (int k = 0; k < NUM_TIMESTEPS_IN_PARALLEL + 1; k++) {
            total += num_send_per_timestep[k];
        }
        if (total != tags.size()) {
            std::cout << "total: " << total << " tags size: " << tags.size() << std::endl;
        }
        assert(total == tags.size());
        if (total >= maxsend_sendlist_stencil_md[i]) {
            grow_send_sendlist_stencil_md(total, i, 0);
        }

        for (int k = 0; k < tags.size(); k++) {
            buf_sendlist_stencil_md[i][k] = tags[k];
        }

        MPI_Request r1;
        MPI_Isend(num_send_per_timestep, NUM_TIMESTEPS_IN_PARALLEL + 1, MPI_INT, send_zoid_num % comm->nprocs, send_zoid_num, world, &r1);
        if (total) {
            MPI_Request r2;
            MPI_Isend(buf_sendlist_stencil_md[i], total, MPI_INT, send_zoid_num % comm->nprocs, send_zoid_num, world, &r2);
        }
    }
    */
}

void CommBrick::construct_second_send_list_stencil_md_receive(std::array<Atom*, NUM_TIMESTEPS_IN_PARALLEL + 1>& atom_arr, queue_info& zoid) {
    assert(false);
}

// send the border atoms, only for timestep 0
// have it so that it's not the case that every zoid or processor has access to all of the atoms at time 0
void CommBrick::borders_stencil_md_initial_send(Atom* atom_, Domain* domain_, queue_info& zoid, int timestep_idx) {
    AtomVec *avec = atom_->avec;
    int zoid_num = zoid.num;

    // std::vector<int>& neighbors = lmp->send_to_neighbors[zoid_num];
    std::vector<int> neighbors;
    // neighbors is size 26, or 3^3 - 1
    for (int i = 0; i < NUM_ZOIDS; i++) {
        if (i != zoid_num && is_close(zoid.where, lmp->zoid_num_to_zoid[i].where)) {
            neighbors.push_back(i);
        }
    }

    for (int i = 0; i < neighbors.size(); i++) {
        int send_zoid_num = neighbors[i];
        auto& send_q_info = lmp->zoid_num_to_zoid[send_zoid_num];

        double *send_zoid_lo = lmp->domain_stencil_md[send_zoid_num][timestep_idx]->sublo;
        double *send_zoid_hi = lmp->domain_stencil_md[send_zoid_num][timestep_idx]->subhi;

        double **x = atom_->x;
        tagint *tag = atom_->tag;
        int nfirst = 0;
        int nlast = atom_->nlocal;
        int nsend = 0;
        int pbc_flag_ = 0;
        int pbc_[3] = {0, 0, 0};
        for (int dim = 0; dim < 3; dim++) {
            if (zoid.where[dim] == RIGHT && send_q_info.where[dim] == PBC) {
                pbc_flag_ = 1;
                pbc_[dim] = -1;
            }
            if (zoid.where[dim] == PBC && send_q_info.where[dim] == RIGHT) {
                pbc_flag_ = 1;
                pbc_[dim] = 1;
            }
        }

        for (int atom_idx = 0; atom_idx < atom_->nlocal; atom_idx++) {
            bool borders_zoid = true;
            bool in_zoid = true;
            for (int dim = 0; dim < 3; dim++) {
                // TODO be aware if 2 * ALLEGRO_SLOPE goes more than the width of a zoid
                // we might have issues and have to reach 'further' into a zoid and modify the neighbors which we iterate over
                double lo = send_zoid_lo[dim] - 2 * ALLEGRO_SLOPE;
                double hi = send_zoid_hi[dim] + 2 * ALLEGRO_SLOPE;
                double value = x[atom_idx][dim];

                double atom_pos_shifted = x[atom_idx][dim] + pbc_[dim] * domain->prd[dim];
                borders_zoid = borders_zoid && (atom_pos_shifted >= lo) && (atom_pos_shifted <= hi);
            }

            if (borders_zoid) {
                if (nsend == maxsendlist[i]) {
                    grow_list(i, nsend);
                }
                sendlist[i][nsend++] = atom_idx;
                if (in_zoid) {
                    // std::cout << "Zoid num: " << zoid_num << " to send_zoid_num: " << send_zoid_num << std::endl;
                    // assert(false);
                }
            }
        }

        if (nsend * size_border > maxsend_stencil_md[i]) {
            grow_send_stencil_md(nsend * size_border, i, 0);
        }
        // TODO: check pbc_flag, pbc

        int n;
        if (ghost_velocity) {
            assert(false);
            // n = avec->pack_border_vel(nsend, sendlist[i], buf_send_stencil_md[i], pbc_flag[i], pbc[i]);
            n = avec->pack_border_vel(nsend, sendlist[i], buf_send_stencil_md[i], pbc_flag_, pbc_);
        } else {
            n = avec->pack_border(nsend, sendlist[i], buf_send_stencil_md[i], pbc_flag_, pbc_);
        }

        MPI_Request r1;
        MPI_Request r2;
        MPI_Isend(&nsend, 1, MPI_INT, neighbors[i] % comm->nprocs, send_zoid_num, world, &r1);
        if (n) {
            MPI_Isend(buf_send_stencil_md[i], n, MPI_DOUBLE, neighbors[i] % comm->nprocs, send_zoid_num, world, &r2);
        }

        smax = MAX(smax,nsend);
        sendnum[i] = nsend;
        size_reverse_recv[i] = nsend*size_reverse;
    }

    // For molecular systems we lose some bits for local atom indices due
    // to encoding of special pairs in neighbor lists. Check for overflows.

    if ((atom->molecular != Atom::ATOMIC)
        && ((atom->nlocal + atom->nghost) > NEIGHMASK))
        error->one(FLERR,"Per-processor number of atoms is too large for "
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

int* CommBrick::send_exclude_eval_tags(std::array<Atom *, NUM_TIMESTEPS_IN_PARALLEL + 1> &atom_arr, queue_info &zoid) {
    int zoid_num = zoid.num;
    auto& send_to_neighbors = lmp->send_to_neighbors[zoid_num];


    int sizes[NUM_TIMESTEPS_IN_PARALLEL + 1] = {0};
    std::vector<int> tags_to_send[NUM_TIMESTEPS_IN_PARALLEL + 1];
    for (int t = 0; t < NUM_TIMESTEPS_IN_PARALLEL + 1; t++) {
        Atom* atom_ = atom_arr[t];

        // never re-evaluate local atoms
        for (int i = 0; i < atom_->nlocal; i++) {
            if (atom_->eval_mask_stencil_md[i] == 1 || true) {
                tags_to_send[t].push_back(atom_->tag[i]);
            }
        }

        for (int i = atom_->nlocal; i < atom_->nlocal + atom_->nghost; i++) {
            if (atom_->eval_mask_stencil_md[i] == 0) {
                tags_to_send[t].push_back(atom_->tag[i]);
            }
        }

        sizes[t] = tags_to_send[t].size();
    }

    int total_tags = 0;
    for (int t = 0; t < NUM_TIMESTEPS_IN_PARALLEL + 1; t++) {
        total_tags += sizes[t];
    }

    int* data = new int[total_tags];
    int idx = 0;
    for (int t = 0; t < NUM_TIMESTEPS_IN_PARALLEL + 1; t++) {
        for (int i = 0; i < tags_to_send[t].size(); i++) {
            data[idx++] = tags_to_send[t][i];
        }
    }

    assert(idx == total_tags);

    for (int i = 0; i < lmp->send_to_neighbors[zoid_num].size(); i++) {
        int send_zoid_num = lmp->send_to_neighbors[zoid_num][i];
        MPI_Request r1;
        MPI_Request r2;
        MPI_Isend(sizes, NUM_TIMESTEPS_IN_PARALLEL + 1, MPI_INT, send_zoid_num % comm->nprocs, (send_zoid_num + 100) * 5, world, &r1);
        if (total_tags > 0) {
            MPI_Isend(data, total_tags, MPI_INT, send_zoid_num % comm->nprocs, (send_zoid_num + 100) * 5, world, &r2);
        }
    }

    return data;
}

void CommBrick::receive_exclude_eval_tags(std::array<Atom *, NUM_TIMESTEPS_IN_PARALLEL + 1> &atom_arr, queue_info &zoid) {
    int zoid_num = zoid.num;
    auto& recv_from_neighbors = lmp->recv_from_neighbors[zoid_num];

    int sizes[NUM_TIMESTEPS_IN_PARALLEL + 1] = {0};

    std::vector<int*> bufs;

    for (int i = 0; i < lmp->recv_from_neighbors[zoid_num].size(); i++) {
        int recv_zoid_num = lmp->recv_from_neighbors[zoid_num][i];

        int nrecv[NUM_TIMESTEPS_IN_PARALLEL + 1] = {0};

        MPI_Recv(nrecv, NUM_TIMESTEPS_IN_PARALLEL + 1, MPI_INT, recv_zoid_num % comm->nprocs, (zoid_num + 100) * 5, world, MPI_STATUS_IGNORE);

        int total = 0;
        for (int t = 0; t < NUM_TIMESTEPS_IN_PARALLEL + 1; t++) {
            assert(nrecv[t] >= 0);
            total += nrecv[t];
        }

        int* data = new int[total];

        if (total) {
            MPI_Recv(data, total, MPI_INT,
                     recv_zoid_num % comm->nprocs, (zoid_num + 100) * 5, world, MPI_STATUS_IGNORE);
        }

        int idx = 0;
        for (int t = 0; t < NUM_TIMESTEPS_IN_PARALLEL + 1; t++) {
            Atom* atom_ = atom_arr[t];
            std::set<int> exclude_tags;

            for (int j = 0; j < nrecv[t]; j++) {
                exclude_tags.insert(data[idx++]);
            }

            for (int j = 0; j < atom_->nlocal + atom_->nghost; j++) {
                if (exclude_tags.find(atom_->tag[j]) != exclude_tags.end()) {
                    atom_->eval_mask_stencil_md[j] = 0;
                }
            }
        }

        bufs.push_back(data);
    }

    for (int i = 0; i < bufs.size(); i++) {
        delete[] bufs[i];
    }
}

void CommBrick::send_shared_ghost_stencil_md(Atom* atom_, queue_info& zoid, std::set<int>& indices, std::vector<int>& neighbors, bool duo_or_trio) {
    assert(false);
    int sz = 4;
    int total_sz = indices.size() * sz;

    if (total_sz > maxsend2_stencil_md[duo_or_trio]) {
        grow_send2_stencil_md(total_sz, duo_or_trio, 0);
    }

    bool debug = (zoid.num == 10);

    int n = atom_->avec->pack_shared_ghost_stencil_md(atom_, indices, buf_send2_stencil_md[duo_or_trio], debug);

    for (int i = 0; i < neighbors.size(); i++) {
        int send_zoid_num = neighbors[i];
        int neighbor_process = send_zoid_num % comm->nprocs;

        int num_send = indices.size();
        MPI_Request r1;
        MPI_Request r2;
        MPI_Isend(&num_send, 1, MPI_INT, neighbor_process, send_zoid_num, world, &r1);
        if (num_send) {
            MPI_Isend(buf_send2_stencil_md[duo_or_trio], n, MPI_DOUBLE, neighbor_process, send_zoid_num, world, &r2);
        }
    }
}

void CommBrick::receive_shared_ghost_stencil_md(Atom* atom_, queue_info& zoid, std::vector<int>& neighbors) {
    assert(false);
    int zoid_num = zoid.num;
    int sz = 4;
    for (int i = 0; i < neighbors.size(); i++) {
        int recv_zoid_num = neighbors[i];
        int neighbor_process = recv_zoid_num % comm->nprocs;

        int nrecv = 0;
        MPI_Request r1;
        MPI_Request r2;

        MPI_Recv(&nrecv, 1, MPI_INT, recv_zoid_num % comm->nprocs, zoid_num, world, MPI_STATUS_IGNORE);
        assert(nrecv >= 0);

        if (nrecv * sz > maxrecv2_stencil_md[i]) {
            grow_recv2_stencil_md(nrecv * sz, i);
        }

        if (nrecv) {
            MPI_Recv(buf_recv2_stencil_md[i], nrecv * sz, MPI_DOUBLE,
                     neighbor_process, zoid_num, world, MPI_STATUS_IGNORE);
        }

        std::cout << "me: " << comm->me << " zoid: " << zoid_num << " unpack share ghost from: " << recv_zoid_num << " nrecv: " << nrecv << std::endl;
        atom_->avec->unpack_shared_ghost_stencil_md(atom_, nrecv, buf_recv2_stencil_md[i]);
    }
}

void CommBrick::send_data_stencil_md(std::array<Atom*, NUM_TIMESTEPS_IN_PARALLEL + 1>& atom_arr, queue_info& zoid) {
    int zoid_num = zoid.num;
    int sz = 9;

    auto& send_to_neighbors = lmp->send_to_neighbors[zoid_num];

    for (int i = 0; i < send_to_neighbors.size(); i++) {
        int send_zoid_num = send_to_neighbors[i];
        queue_info& send_zoid = lmp->zoid_num_to_zoid[send_zoid_num];
        int pbc_flag_[3] = {0};
        for (int dim = 0; dim < 3; dim++) {
            if (zoid.where[dim] == RIGHT && send_zoid.where[dim] == PBC) {
                pbc_flag_[dim] = -1;
            }

            if (zoid.where[dim] == PBC && send_zoid.where[dim] == RIGHT) {
                pbc_flag_[dim] = 1;
            }
        }

        int neighbor_process = send_zoid_num % comm->nprocs;

        int num_send[NUM_TIMESTEPS_IN_PARALLEL + 1] = {0};
        int idx = 0;

        // figure out nsend
        int num_elems = 0;


        for (int t = 0; t < NUM_TIMESTEPS_IN_PARALLEL + 1; t++) {
            num_elems += sendnum_stencil_md[t][i];
            assert(sendnum_stencil_md[t][i] >= 0);
        }

        int data_sz = num_elems * sz;
        if (data_sz >= maxsend_stencil_md[i]) {
            grow_send_stencil_md(data_sz, i, 0);
        }

        for (int t = 0; t < NUM_TIMESTEPS_IN_PARALLEL + 1; t++) {
            Atom* atom_ = atom_arr[t];
            int *list = sendlist_stencil_md[t][i];
            for (int h = 0; h < sendnum_stencil_md[t][i]; h++) {
                assert(sendlist_stencil_md[t][i][h] < atom_->nlocal + atom_->nghost);
            }
            int n = atom_->avec->pack_data_stencil_md(sendnum_stencil_md[t][i], sendlist_stencil_md[t][i],
                                                      &buf_send_stencil_md[i][idx], pbc_flag_,
                                                      send_force_stencil_md[t][i], send_pos_stencil_md[t][i]);
            idx += n;
            assert(n == sendnum_stencil_md[t][i] * sz);
            num_send[t] = sendnum_stencil_md[t][i];

//            if (zoid.where[0] == RIGHT && zoid.where[1] == LEFT && zoid.where[2] == LEFT &&
//                send_zoid_num == 19 && t == 1) {
//                for (int h = 0; h < sendnum_stencil_md[t][i]; h++) {
//                    int atom_idx = sendlist_stencil_md[t][i][h];
//                    if (atom_->tag[atom_idx] == 11338 || atom_->tag[atom_idx] == 10429) {
//                        std::cout << "zoid: " << zoid_num << " Idx: " << h << " atom idx: " << atom_idx << " Tag sent: " << atom_->tag[atom_idx]
//                        << " Pos: " << atom_->x[atom_idx][0] << " " << atom_->x[atom_idx][1] << " " << atom_->x[atom_idx][2]
//                        << " force: " << atom_->f[atom_idx][0] << " " << atom_->f[atom_idx][1] << " " << atom_->f[atom_idx][2] << " sent force? " << send_force_stencil_md[t][i][h] << std::endl;
//                    }
//                }
//            }

            for (int h = 0; h < sendnum_stencil_md[t][i]; h++) {
                int atom_idx = sendlist_stencil_md[t][i][h];
                if (atom_->tag[atom_idx] == 10265) {
                    std::cout << "timestep: " << t << " zoid: " << zoid_num << " send zoid: " << send_zoid_num << "  Idx: " << h << " atom idx: " << atom_idx << " Tag sent: " << atom_->tag[atom_idx]
                              << " Pos: " << atom_->x[atom_idx][0] << " " << atom_->x[atom_idx][1] << " " << atom_->x[atom_idx][2]
                              << " force: " << atom_->f[atom_idx][0] << " " << atom_->f[atom_idx][1] << " " << atom_->f[atom_idx][2]
                              << " sent force? " << send_force_stencil_md[t][i][h] << " send pos: " << send_pos_stencil_md[t][i][h] << std::endl;
                }
            }
        }

        if (neighbor_process != comm->me || true) {
            MPI_Request r1;
            MPI_Request r2;
            MPI_Isend(num_send, NUM_TIMESTEPS_IN_PARALLEL + 1, MPI_INT, neighbor_process, send_zoid_num, world, &r1);
            if (num_elems) {
                MPI_Isend(buf_send_stencil_md[i], idx, MPI_DOUBLE, neighbor_process, send_zoid_num, world, &r2);
            }
        } else {
            /*
            queue_info& send_zoid = lmp->zoid_num_to_zoid[send_zoid_num];
            if (!send_zoid.init_first_recv) {
                assert(false);
            }

            int idx2 = 0;
            auto& other_atom_arr = lmp->atom_stencil_md[send_zoid_num];

            int send_idx = -1;
            for (int j = 0; j < lmp->recv_from_neighbors[send_zoid_num].size(); j++) {
                if (lmp->recv_from_neighbors[send_zoid_num][j] == zoid_num) {
                    send_idx = j;
                    break;
                }
            }

            assert(send_idx != -1);
            for (int t = 0; t < NUM_TIMESTEPS_IN_PARALLEL + 1; t++) {
                if (num_send[t]) {
                    Atom* other_atom_ = other_atom_arr[t];
//                    std::cout << "me: " << comm->me << " SELF unpack data for zoid: " << send_zoid_num << " from: " << zoid_num << " timestep: " << t << std::endl;
//                    std::cout << "send index: " << send_idx << " first? " << send_zoid.first_recv_stencil_md2[t][send_idx] << " size? " << send_zoid.first_recv_sz_stencil_md2[t][send_idx] << std::endl;
//                    std::cout << "num_send for t: " << t << " is: " << num_send[t] << " " << sendnum_stencil_md[t][i] << std::endl;
                    other_atom_->avec->unpack_data_stencil_md(other_atom_,
                                                              num_send[t],
                                                              send_zoid.first_recv_stencil_md[t][send_idx],
                                                              &buf_send_stencil_md[i][idx2],
                                                              send_zoid.first_recv_stencil_md2[t][send_idx],
                                                              send_zoid.first_recv_sz_stencil_md2[t][send_idx]);
                    idx2 += num_send[t] * sz;
                }
            }
            */
        }
    }
}

void CommBrick::construct_shared_ghost_send_list_stencil_md(Atom* a, queue_info& zoid, std::vector<int>& x) {
    for (int i = 0; i < x.size(); i++) {
        sendlist_shared_ghost_stencil_md[0].push_back(x[i]);
    }
}

void CommBrick::receive_data_stencil_md(std::array<Atom*, NUM_TIMESTEPS_IN_PARALLEL + 1>& atom_arr, queue_info& zoid) {
    int zoid_num = zoid.num;
    int sz = 9;

    for (int i = 0; i < lmp->recv_from_neighbors[zoid_num].size(); i++) {
        int recv_zoid_num = lmp->recv_from_neighbors[zoid_num][i];
        int neighbor_process = recv_zoid_num % comm->nprocs;
        if (neighbor_process == comm->me) {
            // continue;
        }

        // std::cout << " zoid num: " << zoid_num << " receiving from: " << recv_zoid_num << std::endl;
        int nrecv[NUM_TIMESTEPS_IN_PARALLEL + 1] = {0};
        MPI_Recv(nrecv, NUM_TIMESTEPS_IN_PARALLEL + 1, MPI_INT, neighbor_process, zoid_num, world, MPI_STATUS_IGNORE);

        int total = 0;
        for (int t = 0; t < NUM_TIMESTEPS_IN_PARALLEL + 1; t++) {
            assert(nrecv[t] >= 0);
            total += nrecv[t];
        }

        if (total * sz > maxrecv_stencil_md[i]) {
            grow_recv_stencil_md(total * sz, i);
        }

        if (total) {
            MPI_Recv(buf_recv_stencil_md[i], total * sz, MPI_DOUBLE,
                     neighbor_process, zoid_num, world, MPI_STATUS_IGNORE);
        }

        int idx = 0;
        for (int t = 0; t < NUM_TIMESTEPS_IN_PARALLEL + 1; t++) {
            if (nrecv[t]) {
                Atom* atom_ = atom_arr[t];
                atom_->avec->unpack_data_stencil_md(atom_, nrecv[t], zoid.first_recv_stencil_md[t][i], &buf_recv_stencil_md[i][idx], zoid.first_recv_stencil_md2[t][i], zoid.first_recv_sz_stencil_md2[t][i]);
                idx += nrecv[t] * sz;
            }
        }
    }
}

void CommBrick::borders_stencil_md_initial_receive(Atom* atom_, Domain* domain_, queue_info& zoid) {
    rmax = 0;
    AtomVec *avec = atom_->avec;

    int zoid_num = zoid.num;

    // auto& send_to = lmp->send_to[zoid_num];
    // auto& recv_from = lmp->recv_from[zoid_num];

    std::vector<int> neighbors;
    // std::vector<int>& neighbors = lmp->recv_from_neighbors[zoid_num];

    // neighbors is size 26, or 3^3 - 1
    for (int i = 0; i < NUM_ZOIDS; i++) {
        if (i != zoid_num && is_close(zoid.where, lmp->zoid_num_to_zoid[i].where)) {
            neighbors.push_back(i);
        }
    }

    // auto& recv_from_neighbors = lmp->recv_from_neighbors[zoid_num];
    std::set<int> recv_atoms_per_tag[neighbors.size()];

    for (int i = 0; i < neighbors.size(); i++) {
        int recv_zoid_num = neighbors[i];
        auto &q_info = lmp->zoid_num_to_zoid[recv_zoid_num];
        int zoid_idx = lmp->zoid_num_to_idx[recv_zoid_num];
        int nrecv;
        MPI_Recv(&nrecv, 1, MPI_INT, neighbors[i] % comm->nprocs, zoid_num, world, MPI_STATUS_IGNORE);
        if (nrecv * size_border > maxrecv_stencil_md[i]) {
            grow_recv_stencil_md(nrecv * size_border, i);
        }

        if (nrecv) {
            MPI_Recv(buf_recv_stencil_md[i], nrecv * size_border, MPI_DOUBLE,
                     neighbors[i] % comm->nprocs, zoid_num, world, MPI_STATUS_IGNORE);
        }

        int old_idx = atom_->nlocal + atom_->nghost;

        double *buf = buf_recv_stencil_md[i];

        int num_ghosts_added = 0;
        if (ghost_velocity) {
            assert(false);
            avec->unpack_border_vel(nrecv, atom_->nlocal + atom_->nghost, buf);
        } else {
            // avec->unpack_border(nrecv, atom_->nlocal + atom_->nghost, buf);
            num_ghosts_added = avec->unpack_border_stencil_md(nrecv, atom_->nlocal + atom_->nghost, buf, atom_, zoid_num);
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
            std::cout << "Same atom received from multiple zoids, resulting in a duplicate tag on two difference indices" << std::endl;
            std::cout << "zoid: " << zoid_num << " receiving ghost from: " << recv_zoid_num << " repeat shit. " << std::endl;
            std::cout << "Repeat tag: " << repeat_tag << " repeat idx: " << repeat_idx << std::endl;
            assert(false);
        }
    }

    // For molecular systems we lose some bits for local atom indices due
    // to encoding of special pairs in neighbor lists. Check for overflows.

    if ((atom->molecular != Atom::ATOMIC)
    && ((atom->nlocal + atom->nghost) > NEIGHMASK))
    error->one(FLERR,"Per-processor number of atoms is too large for "
    "molecular neighbor lists");

    // TODO: whatever this does causes some error
//    int max_send = maxreverse*smax;
//    for (int i = 0; i < maxswap; i++) {
//        if (max_send > maxsend_stencil_md[i]) {
//            grow_send_stencil_md(max_send,i, 0);
//        }
//    }
//
//    // insure send/recv buffers are long enough for all forward & reverse comm
//    int max_recv = maxreverse*rmax;
//    for (int i = 0; i < maxswap; i++) {
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

/*
void CommBrick::borders_stencil_md_initial_receive(std::array<Atom*, NUM_TIMESTEPS_IN_PARALLEL + 1> atom_arr,
                                                   std::array<Domain*, NUM_TIMESTEPS_IN_PARALLEL> domain_arr, queue_info& zoid) {
    rmax = 0;
    Atom* atom_ = atom_arr[0];
    AtomVec *avec = atom_->avec;

    int zoid_num = zoid.num;

    std::vector<std::pair<int, queue_info&>> send_to = lmp->send_to[zoid_num];
    std::vector<int> recv_from = lmp->recv_from[zoid_num];

    std::vector<int> neighbors;
    for (int i = 0; i < NUM_ZOIDS; i++) {
        if (i != zoid_num && is_close(zoid.where, lmp->zoid_num_to_zoid[i].where)) {
            neighbors.push_back(i);
        }
    }

    std::sort(neighbors.begin(), neighbors.end());

    for (int i = 0; i < neighbors.size(); i++) {
        int recv_zoid_num = neighbors[i];
        auto &q_info = lmp->zoid_num_to_zoid[recv_zoid_num];
        int zoid_idx = lmp->zoid_num_to_idx[recv_zoid_num];
        int nrecv;
        MPI_Recv(&nrecv, 1, MPI_INT, neighbors[i] % comm->nprocs, zoid_num, world, MPI_STATUS_IGNORE);
        if (nrecv * size_border > maxrecv_stencil_md[i]) {
            grow_recv_stencil_md(nrecv * size_border, i);
        }
        if (nrecv) {
            MPI_Recv(buf_recv_stencil_md[i], nrecv * size_border, MPI_DOUBLE,
                     neighbors[i] % comm->nprocs, zoid_num, world, MPI_STATUS_IGNORE);
        }

        double *buf = buf_recv_stencil_md[i];
        if (ghost_velocity) {
            assert(false);
            avec->unpack_border_vel(nrecv, atom_->nlocal + atom_->nghost, buf);
        } else {
            // avec->unpack_border(nrecv, atom_->nlocal + atom_->nghost, buf);
            avec->unpack_border_stencil_md(nrecv, atom_->nlocal + atom_->nghost, buf, atom_, zoid_num);
        }

        rmax = MAX(rmax, nrecv);
        recvnum[i] = nrecv;
        firstrecv[i] = atom_->nlocal + atom_->nghost;
        atom_->nghost += nrecv;
        size_forward_recv[i] = nrecv * size_forward;
        size_reverse_send[i] = nrecv * size_reverse;
    }

    // For molecular systems we lose some bits for local atom indices due
    // to encoding of special pairs in neighbor lists. Check for overflows.

    if ((atom->molecular != Atom::ATOMIC)
        && ((atom->nlocal + atom->nghost) > NEIGHMASK))
        error->one(FLERR,"Per-processor number of atoms is too large for "
                         "molecular neighbor lists");

    // TODO: whatever this does causes some error
//    int max_send = maxreverse*smax;
//    for (int i = 0; i < maxswap; i++) {
//        if (max_send > maxsend_stencil_md[i]) {
//            grow_send_stencil_md(max_send,i, 0);
//        }
//    }
//
//    // insure send/recv buffers are long enough for all forward & reverse comm
//    int max_recv = maxreverse*rmax;
//    for (int i = 0; i < maxswap; i++) {
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
*/

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
  int i,n,itype,icollection,iswap,dim,ineed,twoneed;
  int nsend,nrecv,sendflag,nfirst,nlast,ngroup,nprior;
  double lo,hi;
  int *type;
  int *collection;
  double **x;
  double *buf,*mlo,*mhi;
  MPI_Request request;
  AtomVec *avec = atom->avec;

  // After exchanging/sorting, need to reconstruct collection array for border communication
  if (mode == Comm::MULTI) neighbor->build_collection(0);

  // do swaps over all 3 dimensions

  iswap = 0;
  smax = rmax = 0;

  for (dim = 0; dim < 3; dim++) {
    nlast = 0;
    twoneed = 2*maxneed[dim];
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

      if (ineed/2 >= sendneed[dim][ineed % 2]) sendflag = 0;
      else sendflag = 1;

      // find send atoms according to SINGLE vs MULTI
      // all atoms eligible versus only atoms in bordergroup
      // can only limit loop to bordergroup for first sends (ineed < 2)
      // on these sends, break loop in two: owned (in group) and ghost

      // std::cout << "borders lo: " << lo << " borders hi: " << hi << " sublo: " << domain->sublo[dim] << " subhi: " << domain->subhi[dim] << std::endl;
      if (sendflag) {
        if (!bordergroup || ineed >= 2) {
          if (mode == Comm::SINGLE) {
            for (i = nfirst; i < nlast; i++)
              if (x[i][dim] >= lo && x[i][dim] <= hi) {
                if (nsend == maxsendlist[iswap]) grow_list(iswap,nsend);
                sendlist[iswap][nsend++] = i;
              }
          } else if (mode == Comm::MULTI) {
            for (i = nfirst; i < nlast; i++) {
              icollection = collection[i];
              if (x[i][dim] >= mlo[icollection] && x[i][dim] <= mhi[icollection]) {
                if (nsend == maxsendlist[iswap]) grow_list(iswap,nsend);
                sendlist[iswap][nsend++] = i;
              }
            }
          } else {
            for (i = nfirst; i < nlast; i++) {
              itype = type[i];
              if (x[i][dim] >= mlo[itype] && x[i][dim] <= mhi[itype]) {
                if (nsend == maxsendlist[iswap]) grow_list(iswap,nsend);
                sendlist[iswap][nsend++] = i;
              }
            }
          }

        } else {
          if (mode == Comm::SINGLE) {
            ngroup = atom->nfirst;
            for (i = 0; i < ngroup; i++)
              if (x[i][dim] >= lo && x[i][dim] <= hi) {
                if (nsend == maxsendlist[iswap]) grow_list(iswap,nsend);
                sendlist[iswap][nsend++] = i;
              }
            for (i = atom->nlocal; i < nlast; i++)
              if (x[i][dim] >= lo && x[i][dim] <= hi) {
                if (nsend == maxsendlist[iswap]) grow_list(iswap,nsend);
                sendlist[iswap][nsend++] = i;
              }
          } else if (mode == Comm::MULTI) {
            ngroup = atom->nfirst;
            for (i = 0; i < ngroup; i++) {
              icollection = collection[i];
              if (x[i][dim] >= mlo[icollection] && x[i][dim] <= mhi[icollection]) {
                if (nsend == maxsendlist[iswap]) grow_list(iswap,nsend);
                sendlist[iswap][nsend++] = i;
              }
            }
            for (i = atom->nlocal; i < nlast; i++) {
              icollection = collection[i];
              if (x[i][dim] >= mlo[icollection] && x[i][dim] <= mhi[icollection]) {
                if (nsend == maxsendlist[iswap]) grow_list(iswap,nsend);
                sendlist[iswap][nsend++] = i;
              }
            }
          } else {
            ngroup = atom->nfirst;
            for (i = 0; i < ngroup; i++) {
              itype = type[i];
              if (x[i][dim] >= mlo[itype] && x[i][dim] <= mhi[itype]) {
                if (nsend == maxsendlist[iswap]) grow_list(iswap,nsend);
                sendlist[iswap][nsend++] = i;
              }
            }
            for (i = atom->nlocal; i < nlast; i++) {
              itype = type[i];
              if (x[i][dim] >= mlo[itype] && x[i][dim] <= mhi[itype]) {
                if (nsend == maxsendlist[iswap]) grow_list(iswap,nsend);
                sendlist[iswap][nsend++] = i;
              }
            }
          }
        }
      }

      // pack up list of border atoms

      if (nsend*size_border > maxsend) grow_send(nsend*size_border,0);
      if (ghost_velocity)
        n = avec->pack_border_vel(nsend,sendlist[iswap],buf_send,pbc_flag[iswap],pbc[iswap]);
      else
        n = avec->pack_border(nsend,sendlist[iswap],buf_send,pbc_flag[iswap],pbc[iswap]);

      // swap atoms with other proc
      // no MPI calls except SendRecv if nsend/nrecv = 0
      // put incoming ghosts at end of my atom arrays
      // if swapping with self, simply copy, no messages

      if (sendproc[iswap] != me) {
        MPI_Sendrecv(&nsend,1,MPI_INT,sendproc[iswap],0,
                     &nrecv,1,MPI_INT,recvproc[iswap],0,world,MPI_STATUS_IGNORE);
        if (nrecv*size_border > maxrecv) grow_recv(nrecv*size_border);
        if (nrecv) MPI_Irecv(buf_recv,nrecv*size_border,MPI_DOUBLE,
                             recvproc[iswap],0,world,&request);
        if (n) MPI_Send(buf_send,n,MPI_DOUBLE,sendproc[iswap],0,world);
        if (nrecv) MPI_Wait(&request,MPI_STATUS_IGNORE);
        buf = buf_recv;
      } else {
        nrecv = nsend;
        buf = buf_send;
      }

      // unpack buffer

      if (ghost_velocity)
        avec->unpack_border_vel(nrecv,atom->nlocal+atom->nghost,buf);
      else
        avec->unpack_border(nrecv,atom->nlocal+atom->nghost,buf);

      // set all pointers & counters

      smax = MAX(smax,nsend);
      rmax = MAX(rmax,nrecv);
      sendnum[iswap] = nsend;
      recvnum[iswap] = nrecv;
      size_forward_recv[iswap] = nrecv*size_forward;
      size_reverse_send[iswap] = nrecv*size_reverse;
      size_reverse_recv[iswap] = nsend*size_reverse;
      firstrecv[iswap] = atom->nlocal + atom->nghost;
      nprior = atom->nlocal + atom->nghost;
      atom->nghost += nrecv;
      if (neighbor->style == Neighbor::MULTI) neighbor->build_collection(nprior);

      iswap++;
    }
  }

  // For molecular systems we lose some bits for local atom indices due
  // to encoding of special pairs in neighbor lists. Check for overflows.

  if ((atom->molecular != Atom::ATOMIC)
      && ((atom->nlocal + atom->nghost) > NEIGHMASK))
    error->one(FLERR,"Per-processor number of atoms is too large for "
               "molecular neighbor lists");

  // insure send/recv buffers are long enough for all forward & reverse comm

  int max = MAX(maxforward*smax,maxreverse*rmax);
  if (max > maxsend) grow_send(max,0);
  max = MAX(maxforward*rmax,maxreverse*smax);
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
  int iswap,n;
  double *buf;
  MPI_Request request;

  int nsize = pair->comm_forward;

  for (iswap = 0; iswap < nswap; iswap++) {

    // pack buffer

    n = pair->pack_forward_comm(sendnum[iswap],sendlist[iswap],buf_send,pbc_flag[iswap],pbc[iswap]);

    // exchange with another proc
    // if self, set recv buffer to send buffer

    if (sendproc[iswap] != me) {
      if (recvnum[iswap])
        MPI_Irecv(buf_recv,nsize*recvnum[iswap],MPI_DOUBLE,recvproc[iswap],0,world,&request);
      if (sendnum[iswap])
        MPI_Send(buf_send,n,MPI_DOUBLE,sendproc[iswap],0,world);
      if (recvnum[iswap]) MPI_Wait(&request,MPI_STATUS_IGNORE);
      buf = buf_recv;
    } else buf = buf_send;

    // unpack buffer

    pair->unpack_forward_comm(recvnum[iswap],firstrecv[iswap],buf);
  }
}

/* ----------------------------------------------------------------------
   reverse communication invoked by a Pair
   nsize used only to set recv buffer limit
------------------------------------------------------------------------- */

void CommBrick::reverse_comm(Pair *pair)
{
  int iswap,n;
  double *buf;
  MPI_Request request;

  int nsize = MAX(pair->comm_reverse,pair->comm_reverse_off);

  for (iswap = nswap-1; iswap >= 0; iswap--) {

    // pack buffer

    n = pair->pack_reverse_comm(recvnum[iswap],firstrecv[iswap],buf_send);

    // exchange with another proc
    // if self, set recv buffer to send buffer

    if (sendproc[iswap] != me) {
      if (sendnum[iswap])
        MPI_Irecv(buf_recv,nsize*sendnum[iswap],MPI_DOUBLE,sendproc[iswap],0,world,&request);
      if (recvnum[iswap])
        MPI_Send(buf_send,n,MPI_DOUBLE,recvproc[iswap],0,world);
      if (sendnum[iswap]) MPI_Wait(&request,MPI_STATUS_IGNORE);
      buf = buf_recv;
    } else buf = buf_send;

    // unpack buffer

    pair->unpack_reverse_comm(sendnum[iswap],sendlist[iswap],buf);
  }
}

/* ----------------------------------------------------------------------
   forward communication invoked by a Bond
   nsize used only to set recv buffer limit
------------------------------------------------------------------------- */

void CommBrick::forward_comm(Bond *bond)
{
  int iswap,n;
  double *buf;
  MPI_Request request;

  int nsize = bond->comm_forward;

  for (iswap = 0; iswap < nswap; iswap++) {

    // pack buffer

    n = bond->pack_forward_comm(sendnum[iswap],sendlist[iswap],buf_send,pbc_flag[iswap],pbc[iswap]);

    // exchange with another proc
    // if self, set recv buffer to send buffer

    if (sendproc[iswap] != me) {
      if (recvnum[iswap])
        MPI_Irecv(buf_recv,nsize*recvnum[iswap],MPI_DOUBLE,recvproc[iswap],0,world,&request);
      if (sendnum[iswap])
        MPI_Send(buf_send,n,MPI_DOUBLE,sendproc[iswap],0,world);
      if (recvnum[iswap]) MPI_Wait(&request,MPI_STATUS_IGNORE);
      buf = buf_recv;
    } else buf = buf_send;

    // unpack buffer

    bond->unpack_forward_comm(recvnum[iswap],firstrecv[iswap],buf);
  }
}

/* ----------------------------------------------------------------------
   reverse communication invoked by a Bond
   nsize used only to set recv buffer limit
------------------------------------------------------------------------- */

void CommBrick::reverse_comm(Bond *bond)
{
  int iswap,n;
  double *buf;
  MPI_Request request;

  int nsize = MAX(bond->comm_reverse,bond->comm_reverse_off);

  for (iswap = nswap-1; iswap >= 0; iswap--) {

    // pack buffer

    n = bond->pack_reverse_comm(recvnum[iswap],firstrecv[iswap],buf_send);

    // exchange with another proc
    // if self, set recv buffer to send buffer

    if (sendproc[iswap] != me) {
      if (sendnum[iswap])
        MPI_Irecv(buf_recv,nsize*sendnum[iswap],MPI_DOUBLE,sendproc[iswap],0,world,&request);
      if (recvnum[iswap])
        MPI_Send(buf_send,n,MPI_DOUBLE,recvproc[iswap],0,world);
      if (sendnum[iswap]) MPI_Wait(&request,MPI_STATUS_IGNORE);
      buf = buf_recv;
    } else buf = buf_send;

    // unpack buffer

    bond->unpack_reverse_comm(sendnum[iswap],sendlist[iswap],buf);
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
  int iswap,n,nsize;
  double *buf;
  MPI_Request request;

  if (size) nsize = size;
  else nsize = fix->comm_forward;

  for (iswap = 0; iswap < nswap; iswap++) {

    // pack buffer

    n = fix->pack_forward_comm(sendnum[iswap],sendlist[iswap],buf_send,pbc_flag[iswap],pbc[iswap]);

    // exchange with another proc
    // if self, set recv buffer to send buffer

    if (sendproc[iswap] != me) {
      if (recvnum[iswap])
        MPI_Irecv(buf_recv,nsize*recvnum[iswap],MPI_DOUBLE,recvproc[iswap],0,world,&request);
      if (sendnum[iswap])
        MPI_Send(buf_send,n,MPI_DOUBLE,sendproc[iswap],0,world);
      if (recvnum[iswap]) MPI_Wait(&request,MPI_STATUS_IGNORE);
      buf = buf_recv;
    } else buf = buf_send;

    // unpack buffer

    fix->unpack_forward_comm(recvnum[iswap],firstrecv[iswap],buf);
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
  int iswap,n,nsize;
  double *buf;
  MPI_Request request;

  if (size) nsize = size;
  else nsize = fix->comm_reverse;

  for (iswap = nswap-1; iswap >= 0; iswap--) {

    // pack buffer

    n = fix->pack_reverse_comm(recvnum[iswap],firstrecv[iswap],buf_send);

    // exchange with another proc
    // if self, set recv buffer to send buffer

    if (sendproc[iswap] != me) {
      if (sendnum[iswap])
        MPI_Irecv(buf_recv,nsize*sendnum[iswap],MPI_DOUBLE,sendproc[iswap],0,world,&request);
      if (recvnum[iswap])
        MPI_Send(buf_send,n,MPI_DOUBLE,recvproc[iswap],0,world);
      if (sendnum[iswap]) MPI_Wait(&request,MPI_STATUS_IGNORE);
      buf = buf_recv;
    } else buf = buf_send;

    // unpack buffer

    fix->unpack_reverse_comm(sendnum[iswap],sendlist[iswap],buf);
  }
}

/* ----------------------------------------------------------------------
   reverse communication invoked by a Fix with variable size data
   query fix for pack size to insure buf_send is big enough
   handshake sizes before each Irecv/Send to insure buf_recv is big enough
------------------------------------------------------------------------- */

void CommBrick::reverse_comm_variable(Fix *fix)
{
  int iswap,nsend,nrecv;
  double *buf;
  MPI_Request request;

  for (iswap = nswap-1; iswap >= 0; iswap--) {

    // pack buffer

    nsend = fix->pack_reverse_comm_size(recvnum[iswap],firstrecv[iswap]);
    if (nsend > maxsend) grow_send(nsend,0);
    nsend = fix->pack_reverse_comm(recvnum[iswap],firstrecv[iswap],buf_send);

    // exchange with another proc
    // if self, set recv buffer to send buffer

    if (sendproc[iswap] != me) {
      MPI_Sendrecv(&nsend,1,MPI_INT,recvproc[iswap],0,
                   &nrecv,1,MPI_INT,sendproc[iswap],0,world,MPI_STATUS_IGNORE);

      if (sendnum[iswap]) {
        if (nrecv > maxrecv) grow_recv(nrecv);
        MPI_Irecv(buf_recv,maxrecv,MPI_DOUBLE,sendproc[iswap],0,world,&request);
      }
      if (recvnum[iswap])
        MPI_Send(buf_send,nsend,MPI_DOUBLE,recvproc[iswap],0,world);
      if (sendnum[iswap]) MPI_Wait(&request,MPI_STATUS_IGNORE);
      buf = buf_recv;
    } else buf = buf_send;

    // unpack buffer

    fix->unpack_reverse_comm(sendnum[iswap],sendlist[iswap],buf);
  }
}

/* ----------------------------------------------------------------------
   forward communication invoked by a Compute
   nsize used only to set recv buffer limit
------------------------------------------------------------------------- */

void CommBrick::forward_comm(Compute *compute)
{
  int iswap,n;
  double *buf;
  MPI_Request request;

  int nsize = compute->comm_forward;

  for (iswap = 0; iswap < nswap; iswap++) {

    // pack buffer

    n = compute->pack_forward_comm(sendnum[iswap],sendlist[iswap],
                                   buf_send,pbc_flag[iswap],pbc[iswap]);

    // exchange with another proc
    // if self, set recv buffer to send buffer

    if (sendproc[iswap] != me) {
      if (recvnum[iswap])
        MPI_Irecv(buf_recv,nsize*recvnum[iswap],MPI_DOUBLE,recvproc[iswap],0,world,&request);
      if (sendnum[iswap])
        MPI_Send(buf_send,n,MPI_DOUBLE,sendproc[iswap],0,world);
      if (recvnum[iswap]) MPI_Wait(&request,MPI_STATUS_IGNORE);
      buf = buf_recv;
    } else buf = buf_send;

    // unpack buffer

    compute->unpack_forward_comm(recvnum[iswap],firstrecv[iswap],buf);
  }
}

/* ----------------------------------------------------------------------
   reverse communication invoked by a Compute
   nsize used only to set recv buffer limit
------------------------------------------------------------------------- */

void CommBrick::reverse_comm(Compute *compute)
{
  int iswap,n;
  double *buf;
  MPI_Request request;

  int nsize = compute->comm_reverse;

  for (iswap = nswap-1; iswap >= 0; iswap--) {

    // pack buffer

    n = compute->pack_reverse_comm(recvnum[iswap],firstrecv[iswap],buf_send);

    // exchange with another proc
    // if self, set recv buffer to send buffer

    if (sendproc[iswap] != me) {
      if (sendnum[iswap])
        MPI_Irecv(buf_recv,nsize*sendnum[iswap],MPI_DOUBLE,sendproc[iswap],0,world,&request);
      if (recvnum[iswap])
        MPI_Send(buf_send,n,MPI_DOUBLE,recvproc[iswap],0,world);
      if (sendnum[iswap]) MPI_Wait(&request,MPI_STATUS_IGNORE);
      buf = buf_recv;
    } else buf = buf_send;

    // unpack buffer

    compute->unpack_reverse_comm(sendnum[iswap],sendlist[iswap],buf);
  }
}

/* ----------------------------------------------------------------------
   forward communication invoked by a Dump
   nsize used only to set recv buffer limit
------------------------------------------------------------------------- */

void CommBrick::forward_comm(Dump *dump)
{
  int iswap,n;
  double *buf;
  MPI_Request request;

  int nsize = dump->comm_forward;

  for (iswap = 0; iswap < nswap; iswap++) {

    // pack buffer

    n = dump->pack_forward_comm(sendnum[iswap],sendlist[iswap],
                                buf_send,pbc_flag[iswap],pbc[iswap]);

    // exchange with another proc
    // if self, set recv buffer to send buffer

    if (sendproc[iswap] != me) {
      if (recvnum[iswap])
        MPI_Irecv(buf_recv,nsize*recvnum[iswap],MPI_DOUBLE,recvproc[iswap],0,world,&request);
      if (sendnum[iswap])
        MPI_Send(buf_send,n,MPI_DOUBLE,sendproc[iswap],0,world);
      if (recvnum[iswap]) MPI_Wait(&request,MPI_STATUS_IGNORE);
      buf = buf_recv;
    } else buf = buf_send;

    // unpack buffer

    dump->unpack_forward_comm(recvnum[iswap],firstrecv[iswap],buf);
  }
}

/* ----------------------------------------------------------------------
   reverse communication invoked by a Dump
   nsize used only to set recv buffer limit
------------------------------------------------------------------------- */

void CommBrick::reverse_comm(Dump *dump)
{
  int iswap,n;
  double *buf;
  MPI_Request request;

  int nsize = dump->comm_reverse;

  for (iswap = nswap-1; iswap >= 0; iswap--) {

    // pack buffer

    n = dump->pack_reverse_comm(recvnum[iswap],firstrecv[iswap],buf_send);

    // exchange with another proc
    // if self, set recv buffer to send buffer

    if (sendproc[iswap] != me) {
      if (sendnum[iswap])
        MPI_Irecv(buf_recv,nsize*sendnum[iswap],MPI_DOUBLE,sendproc[iswap],0,world,&request);
      if (recvnum[iswap])
        MPI_Send(buf_send,n,MPI_DOUBLE,recvproc[iswap],0,world);
      if (sendnum[iswap]) MPI_Wait(&request,MPI_STATUS_IGNORE);
      buf = buf_recv;
    } else buf = buf_send;

    // unpack buffer

    dump->unpack_reverse_comm(sendnum[iswap],sendlist[iswap],buf);
  }
}

/* ----------------------------------------------------------------------
   forward communication of N values in per-atom array
------------------------------------------------------------------------- */

void CommBrick::forward_comm_array(int nsize, double **array)
{
  int i,j,k,m,iswap,last;
  double *buf;
  MPI_Request request;

  // insure send/recv bufs are big enough for nsize
  // based on smax/rmax from most recent borders() invocation

  if (nsize > maxforward) {
    maxforward = nsize;
    if (maxforward*smax > maxsend) grow_send(maxforward*smax,0);
    if (maxforward*rmax > maxrecv) grow_recv(maxforward*rmax);
  }

  for (iswap = 0; iswap < nswap; iswap++) {

    // pack buffer

    m = 0;
    for (i = 0; i < sendnum[iswap]; i++) {
      j = sendlist[iswap][i];
      for (k = 0; k < nsize; k++)
        buf_send[m++] = array[j][k];
    }

    // exchange with another proc
    // if self, set recv buffer to send buffer

    if (sendproc[iswap] != me) {
      if (recvnum[iswap])
        MPI_Irecv(buf_recv,nsize*recvnum[iswap],MPI_DOUBLE,recvproc[iswap],0,world,&request);
      if (sendnum[iswap])
        MPI_Send(buf_send,nsize*sendnum[iswap],MPI_DOUBLE,sendproc[iswap],0,world);
      if (recvnum[iswap]) MPI_Wait(&request,MPI_STATUS_IGNORE);
      buf = buf_recv;
    } else buf = buf_send;

    // unpack buffer

    m = 0;
    last = firstrecv[iswap] + recvnum[iswap];
    for (i = firstrecv[iswap]; i < last; i++)
      for (k = 0; k < nsize; k++)
        array[i][k] = buf[m++];
  }
}

/* ----------------------------------------------------------------------
   exchange info provided with all 6 stencil neighbors
------------------------------------------------------------------------- */

int CommBrick::exchange_variable(int n, double *inbuf, double *&outbuf)
{
  int nsend,nrecv,nrecv1,nrecv2;
  MPI_Request request;

  nrecv = n;
  if (nrecv > maxrecv) grow_recv(nrecv);
  memcpy(buf_recv,inbuf,nrecv*sizeof(double));

  // loop over dimensions

  for (int dim = 0; dim < 3; dim++) {

    // no exchange if only one proc in a dimension

    if (procgrid[dim] == 1) continue;

    // send/recv info in both directions using same buf_recv
    // if 2 procs in dimension, single send/recv
    // if more than 2 procs in dimension, send/recv to both neighbors

    nsend = nrecv;
    MPI_Sendrecv(&nsend,1,MPI_INT,procneigh[dim][0],0,
                 &nrecv1,1,MPI_INT,procneigh[dim][1],0,world,MPI_STATUS_IGNORE);
    nrecv += nrecv1;
    if (procgrid[dim] > 2) {
      MPI_Sendrecv(&nsend,1,MPI_INT,procneigh[dim][1],0,
                   &nrecv2,1,MPI_INT,procneigh[dim][0],0,world,MPI_STATUS_IGNORE);
      nrecv += nrecv2;
    } else nrecv2 = 0;

    if (nrecv > maxrecv) grow_recv(nrecv);

    MPI_Irecv(&buf_recv[nsend],nrecv1,MPI_DOUBLE,procneigh[dim][1],0,world,&request);
    MPI_Send(buf_recv,nsend,MPI_DOUBLE,procneigh[dim][0],0,world);
    MPI_Wait(&request,MPI_STATUS_IGNORE);

    if (procgrid[dim] > 2) {
      MPI_Irecv(&buf_recv[nsend+nrecv1],nrecv2,MPI_DOUBLE,procneigh[dim][0],0,world,&request);
      MPI_Send(buf_recv,nsend,MPI_DOUBLE,procneigh[dim][1],0,world);
      MPI_Wait(&request,MPI_STATUS_IGNORE);
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
    maxsend = static_cast<int> (BUFFACTOR * n);
    memory->destroy(buf_send);
    memory->create(buf_send,maxsend+bufextra,"comm:buf_send");
  } else if (flag == 1) {
    maxsend = static_cast<int> (BUFFACTOR * n);
    memory->grow(buf_send,maxsend+bufextra,"comm:buf_send");
  } else {
    memory->destroy(buf_send);
    memory->grow(buf_send,maxsend+bufextra,"comm:buf_send");
  }
}

void CommBrick::grow_send_stencil_md(int n, int idx, int flag)
{
    if (flag == 0) {
        maxsend_stencil_md[idx] = static_cast<int> (BUFFACTOR * n);
        memory->destroy(buf_send_stencil_md[idx]);
        buf_send_stencil_md[idx] = memory->create(buf_send_stencil_md[idx],maxsend_stencil_md[idx]+bufextra,"comm:buf_send");
    } else if (flag == 1) {
        maxsend_stencil_md[idx] = static_cast<int> (BUFFACTOR * n);
        buf_send_stencil_md[idx] = memory->grow(buf_send_stencil_md[idx],maxsend_stencil_md[idx]+bufextra,"comm:buf_send");
    } else {
        memory->destroy(buf_send_stencil_md[idx]);
        buf_send_stencil_md[idx] = memory->grow(buf_send_stencil_md[idx],maxsend_stencil_md[idx]+bufextra,"comm:buf_send");
    }
}

void CommBrick::grow_send2_stencil_md(int n, int idx, int flag) {
    if (flag == 0) {
        maxsend2_stencil_md[idx] = static_cast<int> (BUFFACTOR * n);
        memory->destroy(buf_send2_stencil_md[idx]);
        buf_send2_stencil_md[idx] = memory->create(buf_send2_stencil_md[idx],maxsend2_stencil_md[idx]+bufextra,"comm:buf_send");
    } else if (flag == 1) {
        assert(false);
        maxsend_stencil_md[idx] = static_cast<int> (BUFFACTOR * n);
        buf_send_stencil_md[idx] = memory->grow(buf_send_stencil_md[idx],maxsend_stencil_md[idx]+bufextra,"comm:buf_send");
    } else {
        assert(false);
        memory->destroy(buf_send_stencil_md[idx]);
        buf_send_stencil_md[idx] = memory->grow(buf_send_stencil_md[idx],maxsend_stencil_md[idx]+bufextra,"comm:buf_send");
    }
}

void CommBrick::grow_send_sendlist_stencil_md(int n, int idx, int flag) {
    if (flag == 0) {
        maxsend_sendlist_stencil_md[idx] = static_cast<int> (BUFFACTOR * n);
        memory->destroy(buf_sendlist_stencil_md[idx]);
        buf_sendlist_stencil_md[idx] = memory->create(buf_sendlist_stencil_md[idx],maxsend_sendlist_stencil_md[idx]+bufextra,"comm:buf_send");
    } else if (flag == 1) {
        maxsend_sendlist_stencil_md[idx] = static_cast<int> (BUFFACTOR * n);
        buf_sendlist_stencil_md[idx] = memory->grow(buf_sendlist_stencil_md[idx],maxsend_sendlist_stencil_md[idx]+bufextra,"comm:buf_send");
    } else {
        memory->destroy(buf_sendlist_stencil_md[idx]);
        buf_sendlist_stencil_md[idx] = memory->grow(buf_sendlist_stencil_md[idx],maxsend_sendlist_stencil_md[idx]+bufextra,"comm:buf_send");
    }
}

/* ----------------------------------------------------------------------
   free/malloc the size of the recv buffer as needed with BUFFACTOR
------------------------------------------------------------------------- */

void CommBrick::grow_recv(int n)
{
  maxrecv = static_cast<int> (BUFFACTOR * n);
  memory->destroy(buf_recv);
  memory->create(buf_recv,maxrecv,"comm:buf_recv");
}

void CommBrick::grow_recv_stencil_md(int n, int idx)
{
    maxrecv_stencil_md[idx] = static_cast<int> (BUFFACTOR * n);
    memory->destroy(buf_recv_stencil_md[idx]);
    memory->create(buf_recv_stencil_md[idx],maxrecv_stencil_md[idx],"comm:buf_recv");
}

void CommBrick::grow_recv2_stencil_md(int n, int idx) {
    maxrecv2_stencil_md[idx] = static_cast<int> (BUFFACTOR * n);
    memory->destroy(buf_recv2_stencil_md[idx]);
    memory->create(buf_recv2_stencil_md[idx],maxrecv2_stencil_md[idx],"comm:buf_recv");
}

void CommBrick::grow_recv_sendlist_stencil_md(int n, int idx) {
    maxrecv_sendlist_stencil_md[idx] = static_cast<int> (BUFFACTOR * n);
    memory->destroy(buf_recv_sendlist_stencil_md[idx]);
    memory->create(buf_recv_sendlist_stencil_md[idx],maxrecv_sendlist_stencil_md[idx],"comm:buf_recv");
}

/* ----------------------------------------------------------------------
   realloc the size of the iswap sendlist as needed with BUFFACTOR
------------------------------------------------------------------------- */

void CommBrick::grow_list(int iswap, int n)
{
  maxsendlist[iswap] = static_cast<int> (BUFFACTOR * n);
  memory->grow(sendlist[iswap],maxsendlist[iswap],"comm:sendlist[iswap]");
}

void CommBrick::grow_list_stencil_md(int iswap, int n, int timestep) {
    // std::cout << "iswap: " << iswap << " n: " << n << " timestep: " << timestep << std::endl;
    maxsendlist_stencil_md[timestep][iswap] = static_cast<int> (BUFFACTOR * n);
    memory->grow(sendlist_stencil_md[timestep][iswap], maxsendlist_stencil_md[timestep][iswap], "comm:sendlist_stencil_md[iswap]");
    memory->grow(send_force_stencil_md[timestep][iswap], maxsendlist_stencil_md[timestep][iswap], "comm:send_force_stencil_md[iswap]");
    memory->grow(send_pos_stencil_md[timestep][iswap], maxsendlist_stencil_md[timestep][iswap], "comm:send_pos_stencil_md[iswap]");
}

void CommBrick::grow_second_list_stencil_md(int iswap, int n, int timestep) {
    // std::cout << "iswap: " << iswap << " n: " << n << " timestep: " << timestep << std::endl;
    max_second_sendlist_stencil_md[timestep][iswap] = static_cast<int> (BUFFACTOR * n);
    memory->grow(second_sendlist_stencil_md[timestep][iswap], max_second_sendlist_stencil_md[timestep][iswap], "comm:sendlist_stencil_md[iswap]");
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

  sendlist = (int **)
    memory->srealloc(sendlist,n*sizeof(int *),"comm:sendlist");
  memory->grow(maxsendlist,n,"comm:maxsendlist");
  for (int i = maxswap; i < n; i++) {
    maxsendlist[i] = BUFMIN;
    memory->create(sendlist[i],BUFMIN,"comm:sendlist[i]");
  }

  // stencil md version of sendlist and maxsendlist
  for (int i = 0; i < NUM_TIMESTEPS_IN_PARALLEL + 1; i++) {
    sendlist_stencil_md[i] = (int **) memory->smalloc(n*sizeof(int *),"comm:sendlist_stencil_md");
    send_force_stencil_md[i] = (bool **) memory->smalloc(n*sizeof(int *),"comm:sendlist_stencil_md");
    send_pos_stencil_md[i] = (bool **) memory->smalloc(n*sizeof(int *),"comm:sendlist_stencil_md");
    second_sendlist_stencil_md[i] = (int **) memory->smalloc(n*sizeof(int *),"comm:sendlist_stencil_md");
    memory->create(maxsendlist_stencil_md[i],maxswap,"comm:maxsendlist_stencil_md");
    memory->create(max_second_sendlist_stencil_md[i],maxswap,"comm:maxsendlist_stencil_md");

    for (int j = 0; j < n; j++) {
        maxsendlist_stencil_md[i][j] = BUFMIN;
        max_second_sendlist_stencil_md[i][j] = BUFMIN;
        memory->create(sendlist_stencil_md[i][j], BUFMIN, "comm:sendlist_stencil_md[i]");
        memory->create(send_force_stencil_md[i][j], BUFMIN, "comm:send_force_stencil_md[i]");
        memory->create(send_pos_stencil_md[i][j], BUFMIN, "comm:send_pos_stencil_md[i]");
        memory->create(second_sendlist_stencil_md[i][j], BUFMIN, "comm:sendlist_stencil_md[i]");
    }
  }

  maxswap = n;
}

/* ----------------------------------------------------------------------
   allocation of swap info
------------------------------------------------------------------------- */

void CommBrick::allocate_swap(int n)
{
  memory->create(sendnum,n,"comm:sendnum");
  memory->create(recvnum,n,"comm:recvnum");
  memory->create(sendproc,n,"comm:sendproc");
  memory->create(recvproc,n,"comm:recvproc");
  memory->create(size_forward_recv,n,"comm:size");
  memory->create(size_reverse_send,n,"comm:size");
  memory->create(size_reverse_recv,n,"comm:size");
  memory->create(slablo,n,"comm:slablo");
  memory->create(slabhi,n,"comm:slabhi");
  memory->create(firstrecv,n,"comm:firstrecv");
  memory->create(pbc_flag,n,"comm:pbc_flag");
  memory->create(pbc,n,6,"comm:pbc");

  // stencil_md
  for (int i = 0; i < NUM_TIMESTEPS_IN_PARALLEL + 1; i++) {
      memory->create(sendnum_stencil_md[i], n, "comm:sendnum_stencil_md");
      // memory->create(second_sendnum_stencil_md[i], n, "comm:second_sendnum_stencil_md");
  }
}

/* ----------------------------------------------------------------------
   allocation of multi-collection swap info
------------------------------------------------------------------------- */

void CommBrick::allocate_multi(int n)
{
  multilo = memory->create(multilo,n,ncollections,"comm:multilo");
  multihi = memory->create(multihi,n,ncollections,"comm:multihi");
}

/* ----------------------------------------------------------------------
   allocation of multi/old-type swap info
------------------------------------------------------------------------- */

void CommBrick::allocate_multiold(int n)
{
  multioldlo = memory->create(multioldlo,n,atom->ntypes+1,"comm:multioldlo");
  multioldhi = memory->create(multioldhi,n,atom->ntypes+1,"comm:multioldhi");
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
      // memory->destroy(second_sendnum_stencil_md[i]);
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
  if (strcmp(str,"localsendlist") == 0) {
    int i, iswap, isend;
    dim = 1;
    if (!localsendlist)
      memory->create(localsendlist,atom->nlocal,"comm:localsendlist");
    else
      memory->grow(localsendlist,atom->nlocal,"comm:localsendlist");

    for (i = 0; i < atom->nlocal; i++)
      localsendlist[i] = 0;

    for (iswap = 0; iswap < nswap; iswap++)
      for (isend = 0; isend < sendnum[iswap]; isend++)
        if (sendlist[iswap][isend] < atom->nlocal)
          localsendlist[sendlist[iswap][isend]] = 1;

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
  bytes += (double)nprocs * sizeof(int);    // grid2proc
  for (int i = 0; i < nswap; i++)
    bytes += memory->usage(sendlist[i],maxsendlist[i]);
  bytes += memory->usage(buf_send,maxsend+bufextra);
  bytes += memory->usage(buf_recv,maxrecv);
  return bytes;
}
