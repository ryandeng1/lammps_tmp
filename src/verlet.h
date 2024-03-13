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

  void sort_ghost_atoms_stencil_md(Atom*, Atom*, queue_info&, int);
  void group_ghost_atoms_stencil_md(Atom*, Atom*, queue_info&, int);
  void group_ghost_atoms_stencil_md_next_dt(Atom*, Atom*, queue_info&, int);
  virtual void force_clear_stencil_md(Atom*, Force*, Neighbor*);
  // void setup_bins_stencil_md(Atom*, queue_info&, int);
  void setup_atom_arr_stencil_md(std::array<Atom*, NUM_TIMESTEPS_IN_PARALLEL + 1>, Domain*);

  void sort_ghost_atoms_by_prev_and_next_zoid(Atom*, queue_info&, int);
  void cleanup_stencil_md();

  void run_stencil_md(int start_timestep, std::map<int, std::vector<int>>& dep_to_wait_idxs, std::map<int, std::vector<int>>& dep_to_wait_idxs_next_dt,
                      double**, double**, int64_t* compute_duration, int64_t* send_comm_duration, int64_t* recv_comm_duration,
                      int64_t* modify_duration, int64_t* mpi_duration,
                      int64_t* curr_dt_comm_duration, int64_t* next_dt_comm_duration,
                      int* num_pairs_evaled);

protected:
  int triclinic;    // 0 if domain is orthog, 1 if triclinic
  int torqueflag, extraflag;
};

}    // namespace LAMMPS_NS

#endif
#endif
