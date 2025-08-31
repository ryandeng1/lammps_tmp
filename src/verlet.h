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
#include "stencil_md.h"

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
  void run_stencil_md_zoid_many_cuts_no_comm_pipelined_only_next_dep(int starting_timestep, int dep, queue_info& zoid,
                                                                     int start_t, int end_t, int pipeline_stage,
                                                                     std::vector<std::vector<MPI_Request>>& send_r,
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
  void unpack_self_wrapper_pipelined_only_next_dep(int starting_timestep, int dep, queue_info& zoid,
                                                   int start_t, int end_t, int pipeline_stage, std::atomic<int>& counter,
                                                   std::vector<std::vector<MPI_Request>>& send_r,
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
  void unpack_other_wrapper_pipelined_only_next_dep(int starting_timestep, int dep, queue_info& zoid,
                                                    int recv_zoid_num,
                                                    int start_t, int end_t, int pipeline_stage, std::atomic<int>& counter,
                                                    std::vector<std::vector<MPI_Request>>& send_r,
                                                    double** test_f, double** test_x, double** test_v,
                                                    std::vector<std::atomic_flag>& claimed);

  template <bool curr_dt>
  void unpack_data_pipelined_proc_to_proc_helper(int starting_timestep, int dep, queue_info& zoid,
                                                 int start_t, int end_t, int pipeline_stage,
                                                 std::vector<std::vector<MPI_Request>>& send_r,
                                                 double** test_f, double** test_x, double** test_v);
  template <bool curr_dt>
  void unpack_data_pipelined_proc_to_proc_wrapper(int starting_timestep, int dep,
                                                  queue_info& zoid, int recv_zoid_num, int find_idx, int send_dep, int proc, std::atomic<int>& counter,
                                                  int start_t, int end_t, int pipeline_stage,
                                                  std::vector<std::vector<MPI_Request>>& send_r,
                                                  std::atomic_flag& claimed,
                                                  double** test_f, double** test_x, double** test_v);
  template <bool curr_dt>
  void unpack_data_pipelined_proc_to_proc(int starting_timestep, int dep,
                                          int proc, int send_dep,
                                          int start_t, int end_t, int pipeline_stage, std::vector<std::atomic<int>>& counter,
                                          std::vector<std::vector<MPI_Request>>& send_r,
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
  void run_stencil_md_many_cuts_waitany_pipelined_helper_with_proc_to_proc(int starting_timestep, int dep, int pipeline_stage,
                                                                           int start_t, int end_t,
                                                                           double **test_f, double **test_x, double **test_v,
                                                                           std::vector<std::atomic<int>>& recv_neighbor_counts,
                                                                           std::vector<std::vector<MPI_Request>>& send_r,
                                                                           std::vector<MPI_Request>& send_r_proc_to_proc,
                                                                           std::vector<MPI_Request>* recv_r,
                                                                           std::vector<std::atomic_flag>& claimed);

  template <bool curr_dt>
  void run_stencil_md_many_cuts_waitany_pipelined(int starting_timestep, double** test_f, double** test_x, double** test_v,
                                                  std::vector<std::atomic_flag>& claimed, std::vector<std::atomic_flag>& claimed2);

  template <bool curr_dt>
  void run_stencil_md_many_cuts_waitany_pipelined_with_proc_to_proc(int starting_timestep, double** test_f, double** test_x, double** test_v,
                                                                    std::vector<std::vector<MPI_Request>>* send_r,
                                                                    std::vector<std::vector<MPI_Request>>* send_r_proc_to_proc,
                                                                    std::vector<std::atomic<int>>* recv_neighbor_counters,
                                                                    std::vector<std::atomic_flag>& claimed, std::vector<std::atomic_flag>& claimed2);

  void run_stencil_md_many_cuts(int num_timesteps, double** test_f, double** test_x, double** test_v,
                                std::vector<std::atomic_flag>& claimed, std::vector<std::atomic_flag*>& zoid_unpack_self_claimed);

  void run_stencil_md_many_cuts_pipelined(int num_timesteps, double** test_f, double** test_x, double** test_v,
                                          std::vector<std::atomic_flag>& claimed, std::vector<std::atomic_flag>& claimed2);
  template <bool curr_dt>
  void unpack_data_proc_to_proc_wrapper_better_work_queue(int starting_timestep, int dep,
                                        queue_info& zoid, int recv_zoid_num, int find_idx,
                                        int proc, int send_dep,
                                        int start_timestep, int end_timestep, int pipeline_stage,
                                        std::vector<std::atomic<int>>& zoid_recv_neighbor_counters,
                                        std::vector<std::atomic<int>>& dep_counters,
                                        std::vector<std::vector<MPI_Request>>& send_r_zoid_to_zoid,
                                        std::vector<std::vector<MPI_Request>>& send_r_proc_to_proc,
                                        double** test_f, double** test_x, double** test_v,
                                        std::vector<std::atomic_flag>& zoid_claimed,
                                        std::vector<std::atomic_flag>& dep_claimed,
                                        MPIX_Stream_Manager* request_manager, 
                                        std::vector<std::atomic<bool>>& zoid_done);

  template <bool curr_dt>
  void unpack_data_proc_to_proc_better_work_queue(int starting_timestep, int dep,
                                int proc, int send_dep,
                                int start_timestep, int end_timestep, int pipeline_stage,
                                std::vector<std::atomic<int>>& zoid_recv_neighbor_counters,
                                std::vector<std::atomic<int>>& dep_counters,
                                std::vector<std::vector<MPI_Request>>& send_r_zoid_to_zoid,
                                std::vector<std::vector<MPI_Request>>& send_r_proc_to_proc,
                                double** test_f, double** test_x, double** test_v,
                                std::vector<std::atomic_flag>& zoid_claimed,
                                std::vector<std::atomic_flag>& dep_claimed,
                                MPIX_Stream_Manager* request_manager, 
                                std::vector<std::atomic<bool>>& zoid_done);

  template <bool curr_dt>
  void unpack_data_proc_to_proc_wrapper(int starting_timestep, int dep,
                                        queue_info& zoid, int recv_zoid_num, int find_idx,
                                        int proc, int send_dep,
                                        int start_timestep, int end_timestep, int pipeline_stage,
                                        std::vector<std::atomic<int>>& zoid_recv_neighbor_counters,
                                        std::vector<std::atomic<int>>& dep_counters,
                                        std::vector<std::vector<MPI_Request>>& send_r_zoid_to_zoid,
                                        std::vector<std::vector<MPI_Request>>& send_r_proc_to_proc,
                                        double** test_f, double** test_x, double** test_v,
                                        std::vector<std::atomic_flag>& zoid_claimed,
                                        std::vector<std::atomic_flag>& dep_claimed,
                                        MPIX_Stream_Manager* request_manager);

  template <bool curr_dt>
  void unpack_data_proc_to_proc(int starting_timestep, int dep,
                                int proc, int send_dep,
                                int start_timestep, int end_timestep, int pipeline_stage,
                                std::vector<std::atomic<int>>& zoid_recv_neighbor_counters,
                                std::vector<std::atomic<int>>& dep_counters,
                                std::vector<std::vector<MPI_Request>>& send_r_zoid_to_zoid,
                                std::vector<std::vector<MPI_Request>>& send_r_proc_to_proc,
                                double** test_f, double** test_x, double** test_v,
                                std::vector<std::atomic_flag>& zoid_claimed,
                                std::vector<std::atomic_flag>& dep_claimed,
                                MPIX_Stream_Manager* request_manager);
  template <bool curr_dt>
  void run_stencil_md_many_cuts_waitany_with_proc_to_proc(int starting_timestep,
    double** test_f, double** test_x, double** test_v,
    std::vector<std::atomic<int>>& recv_neighbor_counters,
    std::vector<std::vector<MPI_Request>>& send_r,
    std::vector<std::vector<MPI_Request>>& send_r_proc_to_proc,
    std::vector<std::vector<MPI_Request>>& recv_r,
    std::vector<std::vector<MPI_Request>>& recv_r_proc_to_proc,
    std::vector<std::atomic_flag>& claimed);


  template <bool curr_dt>
  void stencil_md_run_zoid_wrapper(int starting_timestep, int dep, queue_info& zoid,
                                    int start_timestep, int end_timestep,
                                    std::vector<std::atomic<int>>& zoid_recv_neighbor_counters, std::vector<std::atomic<int>>& dep_counters,
                                    std::vector<std::vector<MPI_Request>>& send_r_zoid_to_zoid,
                                    std::vector<std::vector<MPI_Request>>& send_r_proc_to_proc,
                                    double** test_f, double** test_x, double** test_v,
                                    std::vector<std::atomic_flag>& zoid_claimed,
                                    std::vector<std::atomic_flag>& dep_claimed,
                                    MPIX_Stream_Manager* request_manager) noexcept;

  template <bool curr_dt>
  void stencil_md_run_zoid_wrapper_better_work_queue(int starting_timestep, int dep, queue_info& zoid,
                                    int start_timestep, int end_timestep,
                                    std::vector<std::atomic<int>>& zoid_recv_neighbor_counters, std::vector<std::atomic<int>>& dep_counters,
                                    std::vector<std::vector<MPI_Request>>& send_r_zoid_to_zoid,
                                    std::vector<std::vector<MPI_Request>>& send_r_proc_to_proc,
                                    double** test_f, double** test_x, double** test_v,
                                    std::vector<std::atomic_flag>& zoid_claimed,
                                    std::vector<std::atomic_flag>& dep_claimed,
                                    MPIX_Stream_Manager* request_manager,
                                    std::vector<std::atomic<bool>>& zoid_done) noexcept;

  template <bool curr_dt>
  void run_stencil_md_many_cuts_unpack_self_wrapper(int starting_timestep, int dep, queue_info& zoid,
    double** test_f, double** test_x, double** test_v,
    std::vector<std::atomic<int>>& zoid_recv_neighbor_counters,
    std::vector<std::atomic<int>>& dep_counters,
    std::vector<std::vector<MPI_Request>>& send_r_zoid_to_zoid,
    std::vector<std::vector<MPI_Request>>& send_r_proc_to_proc,
    std::vector<std::atomic_flag>& zoid_claimed,
    std::vector<std::atomic_flag>& dep_claimed,
    MPIX_Stream_Manager* stream_manager) noexcept;

  template <bool curr_dt>
  void run_stencil_md_many_cuts_process_stream(int starting_timestep, int dep, int stream_num,
    double** test_f, double** test_x, double** test_v,
    std::vector<std::atomic<int>>& zoid_recv_neighbor_counters,
    std::vector<std::atomic<int>>& dep_counters,
    std::vector<std::vector<MPI_Request>>& send_r_zoid_to_zoid,
    std::vector<std::vector<MPI_Request>>& send_r_proc_to_proc,
    std::vector<std::vector<std::vector<MPI_Request>>>& recv_r_zoid_to_zoid_streams,
    std::vector<std::atomic_flag>& zoid_claimed,
    std::vector<std::atomic_flag>& dep_claimed,
    MPIX_Stream_Manager* stream_manager,
    std::vector<std::atomic_flag>& zoid_unpack_claimed) noexcept;

  template <bool curr_dt>
  void run_stencil_md_many_cuts_process_stream_better_work_queue(int starting_timestep, int dep, int stream_num,
    double** test_f, double** test_x, double** test_v,
    std::vector<std::atomic<int>>& zoid_recv_neighbor_counters,
    std::vector<std::atomic<int>>& dep_counters,
    std::vector<std::vector<MPI_Request>>& send_r_zoid_to_zoid,
    std::vector<std::vector<MPI_Request>>& send_r_proc_to_proc,
    std::vector<std::vector<std::vector<MPI_Request>>>& recv_r_zoid_to_zoid_streams,
    std::vector<std::atomic_flag>& zoid_claimed,
    std::vector<std::atomic_flag>& dep_claimed,
    MPIX_Stream_Manager* stream_manager,
    std::vector<std::atomic_flag*>& zoid_unpack_self_claimed,
    std::vector<std::atomic<bool>>& zoid_done,
    std::vector<std::atomic_flag>& zoid_unpack_claimed) noexcept;

  template <bool curr_dt>
  void run_stencil_md_many_cuts_proc_to_proc(int starting_timestep,
    double** test_f, double** test_x, double** test_v,
    std::vector<std::atomic<int>>& zoid_recv_neighbor_counters,
    std::vector<std::atomic<int>>& dep_counters,
    std::vector<std::vector<MPI_Request>>& send_r,
    std::vector<std::vector<MPI_Request>>& send_r_proc_to_proc,
    std::vector<std::vector<std::vector<MPI_Request>>>& recv_r_zoid_to_zoid_streams,
    std::vector<std::vector<MPI_Request>>& recv_r_proc_to_proc,
    std::vector<std::atomic_flag>& zoid_claimed,
    std::vector<std::atomic_flag>& dep_claimed, MPIX_Stream_Manager* request_manager,
    std::vector<std::atomic_flag*>& zoid_unpack_self_claimed
  ) noexcept;

protected:
  int triclinic;    // 0 if domain is orthog, 1 if triclinic
  int torqueflag, extraflag;
};

}    // namespace LAMMPS_NS

#endif
#endif
