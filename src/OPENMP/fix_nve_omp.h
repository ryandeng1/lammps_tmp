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

#ifdef FIX_CLASS
// clang-format off
FixStyle(nve/omp,FixNVEOMP);
// clang-format on
#elifdef FIX_CLASS_STENCIL_MD
// clang-format off
FixStyleStencilMD(nve/omp,FixNVEOMP);
// clang-format on
#else

#ifndef LMP_FIX_NVE_OMP_H
#define LMP_FIX_NVE_OMP_H

#include "fix_nve.h"

namespace LAMMPS_NS {

class FixNVEOMP : public FixNVE {
 public:
  FixNVEOMP(class LAMMPS *, int, char **);
  FixNVEOMP(class LAMMPS *, class Modify*, int, char **);

  void initial_integrate(int) override;
  void final_integrate() override;

  void initial_integrate_stencil_md(int, Atom*, Atom*, int*, bool*) override;
  void final_integrate_stencil_md(Atom*, Atom*, Neighbor*, int*, bool*) override;
};

}    // namespace LAMMPS_NS

#endif
#endif
