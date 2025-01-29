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

#ifdef INTEGRATE_CLASS
// clang-format off
IntegrateStyle(verlet,Verlet);
// clang-format on
#else

#ifndef LMP_VERLET_H
#define LMP_VERLET_H

#include "integrate.h"
#include "stencil_md_utils.h"

namespace LAMMPS_NS {

class Verlet : public Integrate {
 public:
  Verlet(class LAMMPS *, int, char **);
  void init() override;
  void setup(int flag) override;
  void setup_minimal(int) override;
  void run(int) override;
  void force_clear() override;
  void cleanup() override;

  void setup_stencil_md();
  void setup_stencil_md_many_zoids();

  void sort_ghost_atoms_stencil_md_bins(Atom*, queue_info&, int);
  void construct_send_force_bins(bool curr_dt, Atom*, queue_info& zoid, int);
  void construct_send_pos_bins(bool curr_dt, Atom*, queue_info& zoid, int);
  void construct_send_vel_bins(bool curr_dt, Atom*, queue_info& zoid, int);

  void construct_recv_force_bins(bool curr_dt, Atom*, queue_info& zoid, int);
  void construct_recv_pos_bins(bool curr_dt, Atom*, queue_info& zoid, int);
  void construct_recv_vel_bins(bool curr_dt, Atom*, queue_info& zoid, int);

  void construct_no_comm_bins(bool curr_dt, Atom*, queue_info& zoid, int);
  void construct_bin_to_comm(bool curr_dt, Atom*, queue_info& zoid, int);

  void construct_dtfm_cache(Atom*);

  void construct_bin_to_idx(bool curr_dt, Atom*, queue_info& zoid, int);

  void construct_bin_to_send_zoids(bool curr_dt, Atom*, queue_info& zoid, int);

  void sort_ghost_atoms_stencil_md(Atom*, Atom*, queue_info&, int);
  void group_ghost_atoms_stencil_md(Atom*, Atom*, queue_info&, int);
  void group_ghost_atoms_stencil_md_next_dt(Atom*, Atom*, queue_info&, int);
  virtual void force_clear_stencil_md(Atom*, Force*, Neighbor*);
  // void setup_bins_stencil_md(Atom*, queue_info&, int);
  void setup_atom_arr_stencil_md(std::array<Atom*, NUM_TIMESTEPS_IN_PARALLEL + 1>, Domain*);

  void cleanup_stencil_md();

  template <bool curr_dt>
  void run_stencil_md_zoid(int start_timestep, int start_eval, int end_eval, int zoid_num, double** test_f, double** test_x, double** test_v, bool warmup);

  template <bool curr_dt>
  void run_stencil_md_big_zoid(int start_timestep, int start_eval, int end_eval, int zoid_num, double** test_f, double** test_x, double** test_v, bool warmup);

  template <bool curr_dt>
  void run_stencil_md_dep_templated(int dep, int start_timestep, int start_t, int end_t, int* dep_to_idxs,
                                    std::vector<MPI_Request>* send_requests, std::vector<MPI_Request>& receive_requests,
                                    std::vector<int>* dep_to_wait_idxs, std::vector<int>* dep_to_wait_idxs_next_dt,
                                    double** test_f, double** test_x, double** test_v, int pipeline_stage=0, bool warmup=false);

  template <bool curr_dt>
  void run_stencil_md_pipelined_helper(int starting_timestep,
                                       std::vector<int>* dep_to_wait_idxs, std::vector<int>* dep_to_wait_idxs_next_dt,
                                       double** test_f, double** test_x, double** test_v, bool warmup);

  void run_stencil_md_pipelined(int num_timesteps, std::vector<int>* dep_to_wait_idxs, std::vector<int>* dep_to_wait_idxs_next_dt,
                                double** test_f, double** test_x, double** test_v, bool warmup);

  template <bool curr_dt>
  void run_stencil_md_zoid_no_cilk_for(int start_timestep, int zoid_num,
                                       double** test_f, double** test_x, double** test_v,
                                       std::array<std::atomic<int>, NUM_ZOIDS>& counters, const std::array<int, NUM_ZOIDS>& cache);

  template <bool curr_dt>
  void run_stencil_md_no_cilk_for_helper(int num_timesteps, double** test_f, double** test_x, double** test_v,
                                         std::array<std::atomic<int>, NUM_ZOIDS>& counters, const std::array<int, NUM_ZOIDS>& cache);

  void run_stencil_md_no_cilk_for(int num_timesteps, double** test_f, double** test_x, double** test_v);

  /* BEGIN DOUBLE BUFFERING */
  template <bool curr_dt>
  void run_stencil_md_zoid_many_cuts_everything(int starting_timestep, int dep, queue_info& zoid, int start_t, int end_t,
                                                std::vector<MPI_Request>* send_r, std::vector<MPI_Request>* recv_r,
                                                double** test_f, double** test_x, double** test_v);

  template <bool curr_dt>
  void run_stencil_md_zoid_many_cuts(int starting_timestep, int dep, queue_info& zoid, int start_t, int end_t,
                                     double** test_f, double** test_x, double** test_v);
  template <bool curr_dt>
  void run_stencil_md_many_cuts_helper_dep(int starting_timestep, int dep,
                                           int nproc_recv, MPI_Request* recv_requests,
                                           std::map<std::pair<int, int>, bool> did_recv_map,
                                           std::vector<MPI_Request>& send_requests,
                                           double** test_f, double** test_x, double** test_v);

  template <bool curr_dt>
  void run_stencil_md_many_cuts_helper(int starting_timestep, double** test_f, double** test_x, double** test_v);

  void run_stencil_md_many_cuts(int num_timesteps, double** test_f, double** test_x, double** test_v);

  template <bool curr_dt>
  void run_stencil_md_zoid_double_buffering(int start_timestep, int start_eval, int end_eval, int zoid_num,
                                            double** test_f, double** test_x, double** test_v, bool warmup);

  void run_stencil_md_pipelined_double_buffering(int num_timesteps, std::vector<int>* dep_to_wait_idxs, std::vector<int>* dep_to_wait_idxs_next_dt,
                                                 double** test_f, double** test_x, double** test_v, bool warmup);

  template <bool curr_dt>
  void run_stencil_md_pipelined_double_buffering_helper(int starting_timestep,
                                                        std::vector<int>* dep_to_wait_idxs, std::vector<int>* dep_to_wait_idxs_next_dt,
                                                        double** test_f, double** test_x, double** test_v, bool warmup);

  template <bool curr_dt>
  void run_stencil_md_dep_double_buffering(int dep, int start_timestep, int start_t, int end_t, int* dep_to_idxs,
                                           std::vector<MPI_Request>* send_requests, std::vector<MPI_Request>& receive_requests,
                                           std::vector<int>* dep_to_wait_idxs, std::vector<int>* dep_to_wait_idxs_next_dt,
                                           double** test_f, double** test_x, double** test_v, int pipeline_stage=0, bool warmup=false);

  /* END DOUBLE BUFFERING */

protected:
  int triclinic;    // 0 if domain is orthog, 1 if triclinic
  int torqueflag, extraflag;
};

}    // namespace LAMMPS_NS

#endif
#endif
