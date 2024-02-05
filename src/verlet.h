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

  void setup_stencil_md();
  void group_local_atoms_stencil_md(Atom*, queue_info&, int);
  void group_ghost_atoms_stencil_md(Atom*, Atom*, queue_info&, int);
  void group_local_atoms_stencil_md_next_dt(Atom*, queue_info&, int);
  void group_ghost_atoms_stencil_md_next_dt(Atom*, Atom*, queue_info&, int);
  virtual void force_clear_stencil_md(Atom*, Force*, Neighbor*);
  void setup_bins_stencil_md(Atom*, queue_info&, int);
  void setup_atom_arr_stencil_md(std::array<Atom*, NUM_TIMESTEPS_IN_PARALLEL + 1>, Domain*);

  void sort_ghost_atoms_by_prev_and_next_zoid(Atom*, queue_info&, int);

protected:
  int triclinic;    // 0 if domain is orthog, 1 if triclinic
  int torqueflag, extraflag;
};

}    // namespace LAMMPS_NS

#endif
#endif
