// clang-format off
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

#include "omp_compat.h"
#include "fix_nve_omp.h"
#include "atom.h"
#include <cilk/cilk.h>

using namespace LAMMPS_NS;
using namespace FixConst;

typedef struct { double x,y,z; } dbl3_t;

/* ---------------------------------------------------------------------- */

FixNVEOMP::FixNVEOMP(LAMMPS *lmp, int narg, char **arg) :
  FixNVE(lmp, narg, arg) { }

FixNVEOMP::FixNVEOMP(LAMMPS *lmp, Modify* modify_, int narg, char **arg) :
        FixNVEOMP(lmp, narg, arg) { }

/* ----------------------------------------------------------------------
   allow for both per-type and per-atom mass
------------------------------------------------------------------------- */

void FixNVEOMP::initial_integrate(int /* vflag */)
{
  // update v and x of atoms in group

  auto * _noalias const x = (dbl3_t *) atom->x[0];
  auto * _noalias const v = (dbl3_t *) atom->v[0];
  const auto * _noalias const f = (dbl3_t *) atom->f[0];
  const int * const mask = atom->mask;
  const int nlocal = (igroup == atom->firstgroup) ? atom->nfirst : atom->nlocal;

  if (atom->rmass) {
    const double * const rmass = atom->rmass;
#if defined (_OPENMP)
#pragma omp parallel for LMP_DEFAULT_NONE schedule(static)
#endif
    for (int i = 0; i < nlocal; i++)
      if (mask[i] & groupbit) {
        const double dtfm = dtf / rmass[i];
        v[i].x += dtfm * f[i].x;
        v[i].y += dtfm * f[i].y;
        v[i].z += dtfm * f[i].z;
        x[i].x += dtv * v[i].x;
        x[i].y += dtv * v[i].y;
        x[i].z += dtv * v[i].z;
      }

  } else {
    const double * const mass = atom->mass;
    const int * const type = atom->type;
#if defined (_OPENMP)
#pragma omp parallel for LMP_DEFAULT_NONE schedule(static)
#endif
    for (int i = 0; i < nlocal; i++)
      if (mask[i] & groupbit) {
        const double dtfm = dtf / mass[type[i]];
        v[i].x += dtfm * f[i].x;
        v[i].y += dtfm * f[i].y;
        v[i].z += dtfm * f[i].z;
        x[i].x += dtv * v[i].x;
        x[i].y += dtv * v[i].y;
        x[i].z += dtv * v[i].z;
      }
  }
}

void FixNVEOMP::initial_integrate_stencil_md(int /* vflag */, Atom* atom_, Atom* next, int* atom_idx_mapping, bool* can_eval) {
    assert(false);

    /*
    // update v and x of atoms in group
    auto * _noalias const x = (dbl3_t *) atom_->x[0];
    auto * _noalias const next_x = (dbl3_t *) next->x[0];
    auto * _noalias const v = (dbl3_t *) atom_->v[0];
    auto * _noalias const next_v = (dbl3_t *) next->v[0];
    const auto * _noalias const f = (dbl3_t *) atom_->f[0];
    const auto * _noalias const eval_f = (dbl3_t *) atom_->eval_f_stencil_md[0];

    const int * const mask = atom_->mask;
    const int nlocal = (igroup == atom_->firstgroup) ? atom_->nfirst : atom_->nlocal;
    auto& local_dtfm = atom_->local_dtfm;

    if (atom->rmass) {
        assert(false);
        const double * const rmass = atom->rmass;
#if defined (_OPENMP)
#pragma omp parallel for LMP_DEFAULT_NONE schedule(static)
#endif
        for (int i = 0; i < nlocal; i++)
            if (mask[i] & groupbit) {
                const double dtfm = dtf / rmass[i];
                v[i].x += dtfm * f[i].x;
                v[i].y += dtfm * f[i].y;
                v[i].z += dtfm * f[i].z;
                x[i].x += dtv * v[i].x;
                x[i].y += dtv * v[i].y;
                x[i].z += dtv * v[i].z;
            }

    } else {
        const double * const mass = atom->mass;
        const int * const type = atom_->type;

        cilk_for (int i = 0; i < nlocal; i++) {
            // if (mask[i] & groupbit) {
                // const double dtfm = dtf / mass[type[i]];
                const double dtfm = local_dtfm[i];

                int next_idx = atom_idx_mapping[i];

                assert(atom_->tag[i] == next->tag[next_idx]);
                assert(next_idx != -1);

                next_v[next_idx].x = v[i].x + dtfm * (f[i].x + eval_f[i].x);
                next_v[next_idx].y = v[i].y + dtfm * (f[i].y + eval_f[i].y);
                next_v[next_idx].z = v[i].z + dtfm * (f[i].z + eval_f[i].z);

                next_x[next_idx].x = x[i].x + dtv * next_v[next_idx].x;
                next_x[next_idx].y = x[i].y + dtv * next_v[next_idx].y;
                next_x[next_idx].z = x[i].z + dtv * next_v[next_idx].z;
            // }
        }
    }
    */
}

/* ---------------------------------------------------------------------- */

void FixNVEOMP::final_integrate()
{
  // update v of atoms in group

  auto * _noalias const v = (dbl3_t *) atom->v[0];
  const auto * _noalias const f = (dbl3_t *) atom->f[0];
  const int * const mask = atom->mask;
  const int nlocal = (igroup == atom->firstgroup) ? atom->nfirst : atom->nlocal;

  if (atom->rmass) {
    const double * const rmass = atom->rmass;
#if defined (_OPENMP)
#pragma omp parallel for LMP_DEFAULT_NONE schedule(static)
#endif
    for (int i = 0; i < nlocal; i++)
      if (mask[i] & groupbit) {
        const double dtfm = dtf / rmass[i];
        v[i].x += dtfm * f[i].x;
        v[i].y += dtfm * f[i].y;
        v[i].z += dtfm * f[i].z;
      }

  } else {
    const double * const mass = atom->mass;
    const int * const type = atom->type;
#if defined (_OPENMP)
#pragma omp parallel for LMP_DEFAULT_NONE schedule(static)
#endif
    for (int i = 0; i < nlocal; i++)
      if (mask[i] & groupbit) {
        const double dtfm = dtf / mass[type[i]];
        v[i].x += dtfm * f[i].x;
        v[i].y += dtfm * f[i].y;
        v[i].z += dtfm * f[i].z;
      }
  }
}

void FixNVEOMP::final_integrate_stencil_md(Atom* atom_, Atom* next, Neighbor* neighbor_, int* atom_idx_mapping, bool* can_eval) {
    assert(false);
    /*
    // update v of atoms in group

    auto * _noalias const v = (dbl3_t *) atom_->v[0];
    auto * _noalias const next_v = (dbl3_t *) next->v[0];

    const auto * _noalias const f = (dbl3_t *) next->f[0];
    const auto * _noalias const eval_f = (dbl3_t *) next->eval_f_stencil_md[0];
    const int * const mask = next->mask;
    const int nlocal = (igroup == atom_->firstgroup) ? atom_->nfirst : atom_->nlocal;
    const int next_nlocal = next->nlocal;

    if (atom->rmass) {
        assert(false);
        const double * const rmass = atom->rmass;
#if defined (_OPENMP)
#pragma omp parallel for LMP_DEFAULT_NONE schedule(static)
#endif
        for (int i = 0; i < nlocal; i++)
            if (mask[i] & groupbit) {
                const double dtfm = dtf / rmass[i];
                v[i].x += dtfm * f[i].x;
                v[i].y += dtfm * f[i].y;
                v[i].z += dtfm * f[i].z;
            }

    } else {
        const double * const mass = atom->mass;
        const int * const type = next->type;
        auto& local_dtfm = next->local_dtfm;

        cilk_for (int i = 0; i < next_nlocal; i++) {
            // if (mask[i] & groupbit) {
                // const double dtfm = dtf / mass[type[i]];
                const double dtfm = local_dtfm[i];
                next_v[i].x += dtfm * (f[i].x + eval_f[i].x);
                next_v[i].y += dtfm * (f[i].y + eval_f[i].y);
                next_v[i].z += dtfm * (f[i].z + eval_f[i].z);
            // }
        }
    }
    */
}

