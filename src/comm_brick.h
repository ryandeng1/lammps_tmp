/* -*- c++ -*- ----------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   https://www.lammps.org/, Sandia National Laboratories
   LAMMPS development team: developers@lammps.org

   Copyright (2003) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under
   the GNU General Public License.

   See the README file in the top-level LAMMPS directory.
------------------------------------------------------------------------- */

#ifndef LMP_COMM_BRICK_H
#define LMP_COMM_BRICK_H

#include "comm.h"
#include "stencil_md_utils.h"

namespace LAMMPS_NS {

class CommBrick : public Comm {
 public:
  CommBrick(class LAMMPS *);
  CommBrick(class LAMMPS *, class Comm *);

  ~CommBrick() override;

  void init() override;
  void setup() override;                        // setup 3d comm pattern
  void forward_comm(int dummy = 0) override;    // forward comm of atom coords
  void reverse_comm() override;                 // reverse comm of forces
  void exchange() override;                     // move atoms to new procs
  void borders() override;                      // setup list of atoms to comm

  void exchange_stencil_md_initial_send_to_dep0() override;                     // move atoms to new procs, stencil_md version
  void exchange_stencil_md_initial_send() override;                     // move atoms to new procs, stencil_md version
  void exchange_stencil_md_initial_receive(Atom*, Domain*, queue_info&) override;                     // move atoms to new procs, stencil_md version
  void borders_stencil_md_initial_receive_from_lammps(Atom*, Domain*, queue_info&, int) override;                     // move atoms to new procs, stencil_md version

  void exchange_stencil_md_initial_send_to_zoid(Atom*, queue_info&, queue_info&, int timestep) override;                     // move atoms to new procs, stencil_md version
  void exchange_stencil_md_initial_receive_from_zoid(Atom*, queue_info&, queue_info&, int timestep) override;                     // move atoms to new procs, stencil_md version

  void borders_stencil_md_initial_send(Atom*, Domain*, queue_info&, int) override;                     // move atoms to new procs, stencil_md version
  void borders_stencil_md_initial_receive(Atom*, Domain*, queue_info&, int) override;                     // move atoms to new procs, stencil_md version

  void borders_stencil_md_initial_send_to_zoid(Atom*, Domain*, queue_info&, int, int) override;                     // move atoms to new procs, stencil_md version
  void borders_stencil_md_initial_receive_from_zoid(Atom*, Domain*, queue_info&, int, int) override;                     // move atoms to new procs, stencil_md version

  // void borders_stencil_md_initial_receive(std::array<Atom*, NUM_TIMESTEPS_IN_PARALLEL + 1>, std::array<Domain*, NUM_TIMESTEPS_IN_PARALLEL>, queue_info&) override;                     // move atoms to new procs, stencil_md version
  void send_data_stencil_md(std::array<Atom*, NUM_TIMESTEPS_IN_PARALLEL + 1>& atom_arr, queue_info& zoid, std::vector<MPI_Request>&) override;
  void receive_data_stencil_md(std::array<Atom*, NUM_TIMESTEPS_IN_PARALLEL + 1>& atom_arr, queue_info& zoid, std::vector<MPI_Request>&) override;
  void unpack_data_stencil_md(std::array<Atom*, NUM_TIMESTEPS_IN_PARALLEL + 1>& atom_arr, queue_info& zoid, std::vector<MPI_Request>&) override;

  void unpack_data_process_stencil_md(bool curr_dt, int recv_zoid_num, bool is_initial) override;
  bool send_data_to_process_stencil_md(bool curr_dt, std::array<Atom*, NUM_TIMESTEPS_IN_PARALLEL + 1>&, queue_info&, MPI_Request*, int, bool is_initial, int64_t* pack_duration) override;
  void receive_data_process_stencil_md(bool curr_dt, MPI_Request*, int, bool is_initial) override;

  void construct_send_list_stencil_md(std::array<Atom*, NUM_TIMESTEPS_IN_PARALLEL + 1>& atom_arr, queue_info& zoid) override;
  void construct_send_list_stencil_md_send(std::array<Atom*, NUM_TIMESTEPS_IN_PARALLEL + 1>& atom_arr, queue_info& zoid) override;

  void construct_second_send_list_stencil_md_send(std::array<Atom*, NUM_TIMESTEPS_IN_PARALLEL + 1>& atom_arr, queue_info& zoid) override;

  void construct_send_list_stencil_md_next_dt_send(std::array<Atom*, NUM_TIMESTEPS_IN_PARALLEL + 1>& atom_arr, queue_info& zoid) override;
  void construct_second_send_list_stencil_md_next_dt_send(std::array<Atom*, NUM_TIMESTEPS_IN_PARALLEL + 1>& atom_arr, queue_info& zoid) override;

  void send_data_stencil_md_next_dt(std::array<Atom*, NUM_TIMESTEPS_IN_PARALLEL + 1>& atom_arr, queue_info& zoid) override;
  void receive_data_stencil_md_next_dt(std::array<Atom*, NUM_TIMESTEPS_IN_PARALLEL + 1>& atom_arr, queue_info& zoid) override;
  void construct_send_list_stencil_md_next_dt(std::array<Atom*, NUM_TIMESTEPS_IN_PARALLEL + 1>& atom_arr, queue_info& zoid) override;

  void construct_second_send_list_stencil_md(std::array<Atom*, NUM_TIMESTEPS_IN_PARALLEL + 1>& atom_arr, queue_info& zoid) override;
  void construct_second_send_list_stencil_md_next_dt(std::array<Atom*, NUM_TIMESTEPS_IN_PARALLEL + 1>& atom_arr, queue_info& zoid) override;

  void forward_comm(class Pair *) override;                 // forward comm from a Pair
  void reverse_comm(class Pair *) override;                 // reverse comm from a Pair
  void forward_comm(class Bond *) override;                 // forward comm from a Bond
  void reverse_comm(class Bond *) override;                 // reverse comm from a Bond
  void forward_comm(class Fix *, int size = 0) override;    // forward comm from a Fix
  void reverse_comm(class Fix *, int size = 0) override;    // reverse comm from a Fix
  void reverse_comm_variable(class Fix *) override;         // variable size reverse comm from a Fix
  void forward_comm(class Compute *) override;              // forward from a Compute
  void reverse_comm(class Compute *) override;              // reverse from a Compute
  void forward_comm(class Dump *) override;                 // forward comm from a Dump
  void reverse_comm(class Dump *) override;                 // reverse comm from a Dump

  void forward_comm_array(int, double **) override;            // forward comm of array
  int exchange_variable(int, double *, double *&) override;    // exchange on neigh stencil
  void *extract(const char *, int &) override;
  double memory_usage() override;

 protected:
  int nswap;                            // # of swaps to perform = sum of maxneed
  int recvneed[3][2];                   // # of procs away I recv atoms from
  int sendneed[3][2];                   // # of procs away I send atoms to
  int maxneed[3];                       // max procs away any proc needs, per dim
  int maxswap;                          // max # of swaps memory is allocated for
  int *sendnum, *recvnum;               // # of atoms to send/recv in each swap
  int *sendproc, *recvproc;             // proc to send/recv to/from at each swap
  int *size_forward_recv;               // # of values to recv in each forward comm
  int *size_reverse_send;               // # to send in each reverse comm
  int *size_reverse_recv;               // # to recv in each reverse comm
  double *slablo, *slabhi;              // bounds of slab to send at each swap
  double **multilo, **multihi;          // bounds of slabs for multi-collection swap
  double **multioldlo, **multioldhi;    // bounds of slabs for multi-type swap
  double **cutghostmulti;               // cutghost on a per-collection basis
  double **cutghostmultiold;            // cutghost on a per-type basis
  int *pbc_flag;                        // general flag for sending atoms thru PBC
  int **pbc;                            // dimension flags for PBC adjustments

  int *firstrecv;        // where to put 1st recv atom in each swap
  int **sendlist;        // list of atoms to send in each swap
  int *localsendlist;    // indexed list of local sendlist atoms
  int *maxsendlist;      // max size of send list for each swap

  double *buf_send;        // send buffer for all comm
  double *buf_recv;        // recv buffer for all comm

  double** buf_send_stencil_md;
  double** buf_recv_stencil_md;
  int* maxsend_stencil_md;
  int* maxrecv_stencil_md;

  int send_list_sendnum_stencil_md[26][2 * (NUM_TIMESTEPS_IN_PARALLEL + 1)];

  int stencil_md_initial_exchange_nsend;

  int maxsend, maxrecv;    // current size of send/recv buffer
  int smax, rmax;          // max size in atoms of single borders send/recv

  // NOTE: init_buffers is called from a constructor and must not be made virtual
  void init_buffers();

  int updown(int, int, int, double, int, double *);
  // compare cutoff to procs
  virtual void grow_send(int, int);       // reallocate send buffer
  virtual void grow_recv(int);            // free/allocate recv buffer
  virtual void grow_list(int, int);       // reallocate one sendlist
  virtual void grow_swap(int);            // grow swap, multi, and multi/old arrays
  virtual void allocate_swap(int);        // allocate swap arrays
  virtual void allocate_multi(int);       // allocate multi arrays
  virtual void allocate_multiold(int);    // allocate multi/old arrays
  virtual void free_swap();               // free swap arrays
  virtual void free_multi();              // free multi arrays
  virtual void free_multiold();           // free multi/old arrays

  void grow_send_sendlist_stencil_md(int, int, int);
  void grow_recv_sendlist_stencil_md(int, int);

  void grow_second_sendlist_stencil_md(int, int, int);
  void grow_second_sendlist_stencil_md_next_dt(int, int, int);
  void grow_list_stencil_md(int, int, int);
  void grow_list_stencil_md_next_dt(int, int, int);

  void grow_send_stencil_md(int, int, int);
  void grow_recv_stencil_md(int, int);

  void grow_send2_stencil_md(int, int, int);
  void grow_recv2_stencil_md(int, int);

  int ** sendlist_stencil_md[NUM_TIMESTEPS_IN_PARALLEL + 1];
  int ** sendlist_stencil_md_next_dt[NUM_TIMESTEPS_IN_PARALLEL + 1];

  int ** second_sendlist_stencil_md[NUM_TIMESTEPS_IN_PARALLEL + 1];
  int * max_second_sendlist_stencil_md[NUM_TIMESTEPS_IN_PARALLEL + 1];

  int * maxsendlist_stencil_md[NUM_TIMESTEPS_IN_PARALLEL + 1];
  int * maxsendlist_stencil_md_next_dt[NUM_TIMESTEPS_IN_PARALLEL + 1];

  int* sendnum_stencil_md[NUM_TIMESTEPS_IN_PARALLEL + 1];
  int* sendnum_stencil_md_next_dt[NUM_TIMESTEPS_IN_PARALLEL + 1];

  int* second_sendnum_stencil_md[NUM_TIMESTEPS_IN_PARALLEL + 1];
  int* second_sendnum_stencil_md_next_dt[NUM_TIMESTEPS_IN_PARALLEL + 1];

  int ** second_sendlist_stencil_md_next_dt[NUM_TIMESTEPS_IN_PARALLEL + 1];
  int * max_second_sendlist_stencil_md_next_dt[NUM_TIMESTEPS_IN_PARALLEL + 1];

  int ** buf_sendlist_stencil_md;
  int ** buf_recv_sendlist_stencil_md;
  int * maxsend_sendlist_stencil_md;
  int * maxrecv_sendlist_stencil_md;

  bool** send_force_stencil_md[NUM_TIMESTEPS_IN_PARALLEL + 1];
  bool** send_pos_stencil_md[NUM_TIMESTEPS_IN_PARALLEL + 1];
  bool** send_vel_stencil_md[NUM_TIMESTEPS_IN_PARALLEL + 1];

  bool** send_force_stencil_md_next_dt[NUM_TIMESTEPS_IN_PARALLEL + 1];
  bool** send_pos_stencil_md_next_dt[NUM_TIMESTEPS_IN_PARALLEL + 1];
  bool** send_vel_stencil_md_next_dt[NUM_TIMESTEPS_IN_PARALLEL + 1];

  double** buf_send2_stencil_md;
  double** buf_recv2_stencil_md;
  int* maxsend2_stencil_md;
  int* maxrecv2_stencil_md;

  int* num_elems_recv_process[NUM_TIMESTEPS_IN_PARALLEL + 1];

  // TODO: change this to int*, just use std::vector<int> rn to avoid the headache
  std::vector<int> sendlist_shared_ghost_stencil_md[NUM_TIMESTEPS_IN_PARALLEL];
  // int ** sendlist_shared_ghost_stencil_md[NUM_TIMESTEPS_IN_PARALLEL + 1];
  // int* sendnum_shared_ghost_stencil_md[NUM_TIMESTEPS_IN_PARALLEL + 1];
};

}    // namespace LAMMPS_NS

#endif
