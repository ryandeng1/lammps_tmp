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

#ifdef PAIR_CLASS
// clang-format off
PairStyle(lj/cut/omp,PairLJCutOMP);
// clang-format on
#elifdef PAIR_CLASS_STENCIL_MD
// clang-format off
PairStyleStencilMD(lj/cut/omp, PairLJCutOMP);
// clang-format on
#else

#ifndef LMP_PAIR_LJ_CUT_OMP_H
#define LMP_PAIR_LJ_CUT_OMP_H

#include "pair_lj_cut.h"
#include "thr_omp.h"

namespace LAMMPS_NS {

class PairLJCutOMP : public PairLJCut, public ThrOMP {

 public:
  PairLJCutOMP(class LAMMPS *);
  PairLJCutOMP(class LAMMPS *, class Modify *);

  void compute(int, int) override;
  double memory_usage() override;

  void compute_stencil_md(int, int, Atom*, bool*, queue_info&, int*) override;

 private:
  template <int EVFLAG, int EFLAG, int NEWTON_PAIR>
  void eval(int ifrom, int ito, ThrData *const thr);

  template <int EVFLAG, int EFLAG, int NEWTON_PAIR>
  inline void eval_stencil_md(int ifrom, int ito, ThrData *const thr, Atom* atom_);
};

}    // namespace LAMMPS_NS

#endif
#endif
