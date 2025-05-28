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

  virtual void force_clear_stencil_md(Atom*, Force*, Neighbor*);

  void cleanup_stencil_md();

  /* BEGIN DOUBLE BUFFERING */
  template <bool curr_dt>
  void run_stencil_md_zoid_many_cuts_no_comm(int starting_timestep, int dep, queue_info& zoid, int start_t, int end_t,
                                             std::vector<MPI_Request>* send_r,
                                             double** test_f, double** test_x, double** test_v);

  template <bool curr_dt>
  void run_stencil_md_zoid_many_cuts_no_comm_pipelined(int starting_timestep, int dep, queue_info& zoid,
                                                       int start_t, int end_t, int pipeline_stage,
                                                       std::vector<MPI_Request>* send_r,
                                                       double** test_f, double** test_x, double** test_v);

  template <bool curr_dt>
  void run_stencil_md_zoid_many_cuts_no_comm_new_comm(int starting_timestep, int dep, queue_info& zoid, int start_t, int end_t,
                                                      std::vector<MPI_Request>* send_r,
                                                      double** test_f, double** test_x, double** test_v);

  template <bool curr_dt>
  void unpack_self_wrapper(int starting_timestep, int dep, queue_info& zoid,
                           int start_t, int end_t, std::atomic<int>& counter,
                           std::vector<MPI_Request>* send_r,
                           double** test_f, double** test_x, double** test_v,
                           std::vector<std::atomic_flag>& claimed);

  template <bool curr_dt>
  void unpack_self_wrapper_pipelined(int starting_timestep, int dep, queue_info& zoid,
                                     int start_t, int end_t, int pipeline_stage, std::atomic<int>& counter,
                                     std::vector<MPI_Request>* send_r,
                                     double** test_f, double** test_x, double** test_v,
                                     std::vector<std::atomic_flag>& claimed);

  template <bool curr_dt>
  void unpack_other_wrapper(int starting_timestep, int dep, queue_info& zoid,
                            int recv_zoid_num,
                            int start_t, int end_t, std::atomic<int>& counter,
                            std::vector<MPI_Request>* send_r,
                            double** test_f, double** test_x, double** test_v,
                            std::vector<std::atomic_flag>& claimed);

  template <bool curr_dt>
  void unpack_other_wrapper_pipelined(int starting_timestep, int dep, queue_info& zoid,
                                      int recv_zoid_num,
                                      int start_t, int end_t, int pipeline_stage, std::atomic<int>& counter,
                                      std::vector<MPI_Request>* send_r,
                                      double** test_f, double** test_x, double** test_v,
                                      std::vector<std::atomic_flag>& claimed);

  template <bool curr_dt>
  void run_stencil_md_zoid_many_cuts(int starting_timestep, int dep, queue_info& zoid, int start_t, int end_t,
                                     double** test_f, double** test_x, double** test_v);

  template <bool curr_dt>
  void run_stencil_md_many_cuts_helper_dep(int starting_timestep, int dep,
                                           std::vector<MPI_Request>* recv_requests,
                                           std::vector<MPI_Request>* send_requests,
                                           int start_t, int end_t, int pipeline_stage,
                                           double** test_f, double** test_x, double** test_v);

  template <bool curr_dt>
  void run_stencil_md_many_cuts_helper(int starting_timestep, double** test_f, double** test_x, double** test_v);

  template <bool curr_dt>
  void run_stencil_md_many_cuts_waitany_loop(int dep, std::vector<MPI_Request>& recv_r);

  template <bool curr_dt>
  void run_stencil_md_many_cuts_waitany_spawn_wait_loop(int starting_timestep, double** test_f, double** test_x, double** test_v,
                                                        std::vector<std::atomic_flag>& claimed);

  template <bool curr_dt>
  void run_stencil_md_many_cuts_waitany(int starting_timestep, double** test_f, double** test_x, double** test_v,
                                        std::vector<std::atomic_flag>& claimed);

  template <bool curr_dt>
  void run_stencil_md_many_cuts_waitany_pipelined_helper(int starting_timestep, int dep, int pipeline_stage,
                                                         int start_t, int end_t,
                                                         double **test_f, double **test_x, double **test_v,
                                                         std::vector<MPI_Request>* send_r,
                                                         std::vector<MPI_Request>* recv_r,
                                                         std::vector<std::atomic_flag>& claimed);

  template <bool curr_dt>
  void run_stencil_md_many_cuts_waitany_pipelined(int starting_timestep, double** test_f, double** test_x, double** test_v,
                                                  std::vector<std::atomic_flag>& claimed, std::vector<std::atomic_flag>& claimed2);

  template <bool curr_dt>
  void run_stencil_md_many_cuts_new_comm(int starting_timestep, double** test_f, double** test_x, double** test_v);

  void run_stencil_md_many_cuts(int num_timesteps, double** test_f, double** test_x, double** test_v,
                                std::vector<std::atomic_flag>& claimed);

  void run_stencil_md_many_cuts_pipelined(int num_timesteps, double** test_f, double** test_x, double** test_v,
                                          std::vector<std::atomic_flag>& claimed, std::vector<std::atomic_flag>& claimed2);

protected:
  int triclinic;    // 0 if domain is orthog, 1 if triclinic
  int torqueflag, extraflag;
};

}    // namespace LAMMPS_NS

#endif
#endif
