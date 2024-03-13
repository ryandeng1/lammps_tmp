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

#include "fix_nve.h"

#include "atom.h"
#include "error.h"
#include "force.h"
#include "respa.h"
#include "update.h"

using namespace LAMMPS_NS;
using namespace FixConst;

/* ---------------------------------------------------------------------- */

FixNVE::FixNVE(LAMMPS *lmp, int narg, char **arg) :
  Fix(lmp, narg, arg)
{
  if (!utils::strmatch(style,"^nve/sphere") && narg < 3)
    error->all(FLERR,"Illegal fix nve command");

  dynamic_group_allow = 1;
  time_integrate = 1;
}

FixNVE::FixNVE(LAMMPS *lmp, Modify* modify_, int narg, char **arg) :
        FixNVE(lmp, narg, arg) {}

/* ---------------------------------------------------------------------- */

int FixNVE::setmask()
{
  int mask = 0;
  mask |= INITIAL_INTEGRATE;
  mask |= FINAL_INTEGRATE;
  mask |= INITIAL_INTEGRATE_RESPA;
  mask |= FINAL_INTEGRATE_RESPA;
  return mask;
}

/* ---------------------------------------------------------------------- */

void FixNVE::init()
{
  dtv = update->dt;
  dtf = 0.5 * update->dt * force->ftm2v;

  if (utils::strmatch(update->integrate_style,"^respa"))
    step_respa = (dynamic_cast<Respa *>(update->integrate))->step;
}

void FixNVE::init_stencil_md(Atom* atom_, Modify* modify_) {
    init();
}

/* ----------------------------------------------------------------------
   allow for both per-type and per-atom mass
------------------------------------------------------------------------- */

void FixNVE::initial_integrate(int /*vflag*/)
{
  double dtfm;

  // update v and x of atoms in group

  double **x = atom->x;
  double **v = atom->v;
  double **f = atom->f;
  double *rmass = atom->rmass;
  double *mass = atom->mass;
  int *type = atom->type;
  int *mask = atom->mask;
  int nlocal = atom->nlocal;
  if (igroup == atom->firstgroup) {
      assert(false);
      nlocal = atom->nfirst;
  }

  if (rmass) {
    assert(false);
    for (int i = 0; i < nlocal; i++)
      if (mask[i] & groupbit) {
        dtfm = dtf / rmass[i];
        v[i][0] += dtfm * f[i][0];
        v[i][1] += dtfm * f[i][1];
        v[i][2] += dtfm * f[i][2];
        x[i][0] += dtv * v[i][0];
        x[i][1] += dtv * v[i][1];
        x[i][2] += dtv * v[i][2];
      }

  } else {
    for (int i = 0; i < nlocal; i++)
      if (mask[i] & groupbit) {
        dtfm = dtf / mass[type[i]];

        v[i][0] += dtfm * f[i][0];
        v[i][1] += dtfm * f[i][1];
        v[i][2] += dtfm * f[i][2];
        x[i][0] += dtv * v[i][0];
        x[i][1] += dtv * v[i][1];
        x[i][2] += dtv * v[i][2];

        // assert(fabs(x0 - x[i][0]) <= ADDITIONAL_CUTOFF);
        // assert(fabs(x1 - x[i][1]) <= ADDITIONAL_CUTOFF);
        // assert(fabs(x2 - x[i][2]) <= ADDITIONAL_CUTOFF);
      }
  }
}

void FixNVE::initial_integrate_stencil_md(int /*vflag*/, Atom* atom_, Atom* next, int* atom_idx_mapping, bool* can_eval) {
    double dtfm;

    // update v and x of atoms in group

    double **x = atom_->x;
    double **v = atom_->v;
    double **f = atom_->f;
    double **eval_f = atom_->eval_f_stencil_md;
    double **next_x = next->x;

    double *rmass = atom->rmass;
    double *mass = atom->mass;
    int *type = atom_->type;
    int *mask = atom_->mask;
    int nlocal = atom_->nlocal;
    if (igroup == atom_->firstgroup) {
        nlocal = atom_->nfirst;
    }

    if (rmass) {
        assert(false);
        for (int i = 0; i < nlocal; i++)
            if (mask[i] & groupbit) {
                dtfm = dtf / rmass[i];
                v[i][0] += dtfm * f[i][0];
                v[i][1] += dtfm * f[i][1];
                v[i][2] += dtfm * f[i][2];
                x[i][0] += dtv * v[i][0];
                x[i][1] += dtv * v[i][1];
                x[i][2] += dtv * v[i][2];
            }

    } else {
        for (int i = 0; i < nlocal; i++) {
            if (mask[i] & groupbit) {
                dtfm = dtf / mass[type[i]];

                v[i][0] += dtfm * (f[i][0] + eval_f[i][0]);
                v[i][1] += dtfm * (f[i][1] + eval_f[i][1]);
                v[i][2] += dtfm * (f[i][2] + eval_f[i][2]);

                int next_idx = atom_idx_mapping[i];
                next_x[next_idx][0] = x[i][0] + dtv * v[i][0];
                next_x[next_idx][1] = x[i][1] + dtv * v[i][1];
                next_x[next_idx][2] = x[i][2] + dtv * v[i][2];
                assert(atom_->tag[i] == next->tag[next_idx]);
            }
        }
    }
}

/* ---------------------------------------------------------------------- */

void FixNVE::final_integrate()
{
  double dtfm;

  // update v of atoms in group

  double **v = atom->v;
  double **f = atom->f;
  double *rmass = atom->rmass;
  double *mass = atom->mass;
  int *type = atom->type;
  int *mask = atom->mask;
  int nlocal = atom->nlocal;
  if (igroup == atom->firstgroup) {
      assert(false);
      nlocal = atom->nfirst;
  }

  if (rmass) {
    assert(false);
    for (int i = 0; i < nlocal; i++)
      if (mask[i] & groupbit) {
        dtfm = dtf / rmass[i];
        v[i][0] += dtfm * f[i][0];
        v[i][1] += dtfm * f[i][1];
        v[i][2] += dtfm * f[i][2];
      }

  } else {
    for (int i = 0; i < nlocal; i++)
      if (mask[i] & groupbit) {
        dtfm = dtf / mass[type[i]];
        v[i][0] += dtfm * f[i][0];
        v[i][1] += dtfm * f[i][1];
        v[i][2] += dtfm * f[i][2];
      }
  }
}

void FixNVE::final_integrate_stencil_md(Atom* atom_, Atom* next, Neighbor* neighbor_, int* atom_idx_mapping, bool* can_eval) {
    double dtfm;

    // update v of atoms in group

    double **v = atom_->v;
    double **f = next->f;
    double **eval_f = next->eval_f_stencil_md;
    double **next_v = next->v;

    double *rmass = atom->rmass;
    double *mass = atom->mass;
    int *type = next->type;
    int *mask = next->mask;

    int nlocal = atom_->nlocal;
    int next_nlocal = next->nlocal;
    if (igroup == atom_->firstgroup) {
        assert(false);
        nlocal = atom_->nfirst;
    }

    if (rmass) {
        assert(false);
        for (int i = 0; i < nlocal; i++)
            if (mask[i] & groupbit) {
                dtfm = dtf / rmass[i];
                v[i][0] += dtfm * f[i][0];
                v[i][1] += dtfm * f[i][1];
                v[i][2] += dtfm * f[i][2];
            }

    } else {
        for (int i = 0; i < nlocal; i++) {
            int next_idx = atom_idx_mapping[i];
            assert(next_idx != -1);
            next_v[next_idx][0] = v[i][0];
            next_v[next_idx][1] = v[i][1];
            next_v[next_idx][2] = v[i][2];
        }

        for (int i = 0; i < next_nlocal; i++) {
            if (mask[i] & groupbit) {
                dtfm = dtf / mass[type[i]];
                next_v[i][0] += dtfm * (f[i][0] + eval_f[i][0]);
                next_v[i][1] += dtfm * (f[i][1] + eval_f[i][1]);
                next_v[i][2] += dtfm * (f[i][2] + eval_f[i][2]);
            }
        }
    }
}

/* ---------------------------------------------------------------------- */

void FixNVE::initial_integrate_respa(int vflag, int ilevel, int /*iloop*/)
{
  dtv = step_respa[ilevel];
  dtf = 0.5 * step_respa[ilevel] * force->ftm2v;

  // innermost level - NVE update of v and x
  // all other levels - NVE update of v

  if (ilevel == 0) initial_integrate(vflag);
  else final_integrate();
}

/* ---------------------------------------------------------------------- */

void FixNVE::final_integrate_respa(int ilevel, int /*iloop*/)
{
  dtf = 0.5 * step_respa[ilevel] * force->ftm2v;
  final_integrate();
}

/* ---------------------------------------------------------------------- */

void FixNVE::reset_dt()
{
  dtv = update->dt;
  dtf = 0.5 * update->dt * force->ftm2v;
}
