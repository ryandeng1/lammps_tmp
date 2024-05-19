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

/* ----------------------------------------------------------------------
   Contributing author: Axel Kohlmeyer (Temple U)
------------------------------------------------------------------------- */

#ifdef BOND_CLASS
// clang-format off
BondStyle(fene/omp,BondFENEOMP);
// clang-format on
#elifdef BOND_CLASS_STENCIL_MD
// clang-format off
BondStyleStencilMD(fene/omp, BondFENEOMP);
// clang-format on
#else

#ifndef LMP_BOND_FENE_OMP_H
#define LMP_BOND_FENE_OMP_H

#include "bond_fene.h"
#include "thr_omp.h"

namespace LAMMPS_NS {

class BondFENEOMP : public BondFENE, public ThrOMP {

 public:
  BondFENEOMP(class LAMMPS *lmp);
  BondFENEOMP(class LAMMPS *_lmp, class Modify* modify_);
  void compute(int, int) override;

  void compute_stencil_md(int, int, Atom*, bool*, queue_info&, int*, Neighbor*) override;

 private:
  template <int EVFLAG, int EFLAG, int NEWTON_BOND>
  void eval(int ifrom, int ito, ThrData *const thr);

  template <int EVFLAG, int EFLAG, int NEWTON_BOND>
  void eval_stencil_md(int ifrom, int ito, ThrData *const thr, Atom* atom_, Neighbor* neighbor_);
};

}    // namespace LAMMPS_NS

#endif
#endif
