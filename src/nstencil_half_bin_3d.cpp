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

#include "nstencil_half_bin_3d.h"

using namespace LAMMPS_NS;

/* ---------------------------------------------------------------------- */

NStencilHalfBin3d::NStencilHalfBin3d(LAMMPS *lmp) : NStencil(lmp) {}

/* ----------------------------------------------------------------------
   create stencil based on bin geometry and cutoff
------------------------------------------------------------------------- */

// default lj cut uses this, so a hack is to create a separate create_stencil_md
void NStencilHalfBin3d::create()
{
  std::cout << "nstencil half bin 3d create lammps" << std::endl;
  int i, j, k;

  nstencil = 0;

  for (k = 0; k <= sz; k++)
    for (j = -sy; j <= sy; j++)
      for (i = -sx; i <= sx; i++)
        if (k > 0 || j > 0 || (j == 0 && i > 0))
          if (bin_distance(i, j, k) < cutneighmaxsq)
            stencil[nstencil++] = k * mbiny * mbinx + j * mbinx + i;
}

// default lj cut uses this, so a hack is to create a separate create_stencil_md one
// will use full stencil to get access to all other atoms
void NStencilHalfBin3d::create_stencil_md() {
    // full bin implementation, will then modify the corresponding npair implementation in npair_half_bin_atomonly_newton.cpp to account for this
    int i, j, k;

    nstencil = 0;

    for (k = -sz; k <= sz; k++)
        for (j = -sy; j <= sy; j++)
            for (i = -sx; i <= sx; i++)
                if (bin_distance(i, j, k) < cutneighmaxsq)
                    stencil[nstencil++] = k * mbiny * mbinx + j * mbinx + i;

    /*
    std::cout << "nstencil half bin 3d create stencil md. sx: " << sx << " sy: " << sy << " sz: " << sz << std::endl;
    int i, j, k;

    nstencil = 0;

    for (k = 0; k <= sz; k++)
        for (j = -sy; j <= sy; j++)
            for (i = -sx; i <= sx; i++)
                if (k > 0 || j > 0 || (j == 0 && i > 0))
                    if (bin_distance(i, j, k) < cutneighmaxsq)
                        stencil[nstencil++] = k * mbiny * mbinx + j * mbinx + i;
    */
}
