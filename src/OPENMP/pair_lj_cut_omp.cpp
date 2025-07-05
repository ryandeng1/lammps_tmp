// clang-format off
/* ----------------------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   https://www.lammps.org/, Sandia National Laboratories
   LAMMPS development team: developers@lammps.org

   This software is distributed under the GNU General Public License.

   See the README file in the top-level LAMMPS directory.
------------------------------------------------------------------------- */

/* ----------------------------------------------------------------------
   Contributing author: Axel Kohlmeyer (Temple U)
------------------------------------------------------------------------- */

#include "pair_lj_cut_omp.h"

#include "atom.h"
#include "comm.h"
#include "force.h"
#include "neigh_list.h"
#include "suffix.h"

#include "omp_compat.h"
#include <cilk/cilk.h>
#include <cilk/cilk_api.h>
#include <cilk/cilkscale.h>
#include <cilk/opadd_reducer.h>
#include <sstream>
#include "stencil_md_utils.h"

using namespace LAMMPS_NS;

static void new_reducer(void* view) {
    {
        ((dbl3_t *)view)->x = 0.0;
        ((dbl3_t *)view)->y = 0.0;
        ((dbl3_t *)view)->z = 0.0;
    }
}

static void merge(void* left, void* right) {
    ((dbl3_t *)left)->x += ((dbl3_t*)right)->x;
    ((dbl3_t *)left)->y += ((dbl3_t*)right)->y;
    ((dbl3_t *)left)->z += ((dbl3_t*)right)->z;
}

// static dbl3_t cilk_reducer(new_reducer, merge) ftmp;

// static cilk::opadd_reducer<int> num_edges = 0;
// static cilk::opadd_reducer<int> num_lammps_edges = 0;

/* ---------------------------------------------------------------------- */

PairLJCutOMP::PairLJCutOMP(LAMMPS *lmp) :
  PairLJCut(lmp), ThrOMP(lmp, THR_PAIR)
{
  suffix_flag |= Suffix::OMP;
  respa_enable = 0;
  cut_respa = nullptr;
}

PairLJCutOMP::PairLJCutOMP(LAMMPS *lmp, Modify* modify_) :
        PairLJCut(lmp), ThrOMP(lmp, modify_, THR_PAIR) {
    suffix_flag |= Suffix::OMP;
    respa_enable = 0;
    cut_respa = nullptr;
}

/* ---------------------------------------------------------------------- */

void PairLJCutOMP::compute(int eflag, int vflag)
{
  // num_edges = 0;
  // num_accepted_edges = 0;
  ev_init(eflag,vflag);

  const int nall = atom->nlocal + atom->nghost;
  const int nthreads = comm->nthreads;
  const int inum = list->inum;

#if defined(_OPENMP)
#pragma omp parallel LMP_DEFAULT_NONE LMP_SHARED(eflag,vflag)
#endif
  {
    int ifrom, ito, tid;

    loop_setup_thr(ifrom, ito, tid, inum, nthreads);
    ThrData *thr = fix->get_thr(tid);
    thr->timer(Timer::START);
    ev_setup_thr(eflag, vflag, nall, eatom, vatom, nullptr, thr);

    if (evflag) {
      if (eflag) {
        if (force->newton_pair) eval<1,1,1>(ifrom, ito, thr);
        else eval<1,1,0>(ifrom, ito, thr);
      } else {
        if (force->newton_pair) eval<1,0,1>(ifrom, ito, thr);
        else eval<1,0,0>(ifrom, ito, thr);
      }
    } else {
      if (force->newton_pair) eval<0,0,1>(ifrom, ito, thr);
      else eval<0,0,0>(ifrom, ito, thr);
    }
    thr->timer(Timer::PAIR);
    reduce_thr(this, eflag, vflag, thr);
  } // end of omp parallel region
}

void PairLJCutOMP::compute_stencil_md(int eflag, int vflag, Atom* atom_, bool* can_eval_center, queue_info& zoid, int* num_eval) {
    ev_init(eflag,vflag);
    const int nall = atom_->nlocal + atom_->nghost;
    const int nlocal = atom_->nlocal;
    const int nthreads = comm->nthreads;
    const int inum = list->inum;

    int newton_pair = force->newton_pair;

    if (USE_ATOMICS) {
        const auto * _noalias const x = (dbl3_t *) atom_->x[0];
        auto * _noalias const f = (dbl3_t *) atom_->eval_f_stencil_md[0];
        const int * _noalias const type = atom_->type;
        const double * _noalias const special_lj = force->special_lj;
        const int * _noalias const ilist = list->ilist;
        const int * _noalias const numneigh = list->numneigh;
        const int * const * const firstneigh = list->firstneigh;

        // loop over neighbors of my atoms
        cilk_for (int ii = 0; ii < nlocal; ii++) {
            const int i = ilist[ii];
            const int itype = type[i];

            const int    * _noalias const jlist = firstneigh[i];
            const double * _noalias const cutsqi = cutsq[itype];
            const double * _noalias const offseti = offset[itype];
            const double * _noalias const lj1i = lj1[itype];
            const double * _noalias const lj2i = lj2[itype];
            const double * _noalias const lj3i = lj3[itype];
            const double * _noalias const lj4i = lj4[itype];

            double xtmp = x[i].x;
            double ytmp = x[i].y;
            double ztmp = x[i].z;
            int jnum = numneigh[i];

            double fxtmp = 0.0;
            double fytmp = 0.0;
            double fztmp = 0.0;

            for (int jj = 0; jj < jnum; jj++) {
                double evdwl = 0.0;
                int j = jlist[jj];
                double factor_lj = special_lj[sbmask(j)];
                j &= NEIGHMASK;

                double delx = xtmp - x[j].x;
                double dely = ytmp - x[j].y;
                double delz = ztmp - x[j].z;
                double rsq = delx*delx + dely*dely + delz*delz;
                int jtype = type[j];

                if (rsq < cutsqi[jtype]) {
                    double r2inv = 1.0/rsq;
                    double r6inv = r2inv*r2inv*r2inv;
                    double forcelj = r6inv * (lj1i[jtype]*r6inv - lj2i[jtype]);
                    double fpair = factor_lj*forcelj*r2inv;

                    fxtmp += delx*fpair;
                    fytmp += dely*fpair;
                    fztmp += delz*fpair;

                    /*
                    ftmp.x += delx*fpair;
                    ftmp.y += dely*fpair;
                    ftmp.z += delz*fpair;
                    */

                    if (newton_pair || j < nlocal) {
                        // f[j].x -= delx*fpair;
                        // f[j].y -= dely*fpair;
                        // f[j].z -= delz*fpair;
                        __atomic_fetch_add(&f[j].x, -delx*fpair, __ATOMIC_RELAXED);
                        __atomic_fetch_add(&f[j].y, -dely*fpair, __ATOMIC_RELAXED);
                        __atomic_fetch_add(&f[j].z, -delz*fpair, __ATOMIC_RELAXED);
                    }

                    /*
                    if (EFLAG) {
                        evdwl = r6inv*(lj3i[jtype]*r6inv-lj4i[jtype]) - offseti[jtype];
                        evdwl *= factor_lj;
                    }

                    if (EVFLAG) {
                        ev_tally_thr(this, i, j, nlocal, NEWTON_PAIR,
                                     evdwl, 0.0, fpair, delx, dely, delz, thr);
                    }
                    */
                }
            }

            __atomic_fetch_add(&f[i].x, fxtmp, __ATOMIC_RELAXED);
            __atomic_fetch_add(&f[i].y, fytmp, __ATOMIC_RELAXED);
            __atomic_fetch_add(&f[i].z, fztmp, __ATOMIC_RELAXED);

            // f[i].x += fxtmp;
            // f[i].y += fytmp;
            // f[i].z += fztmp;
        }

        return;
    }

    if (false && nlocal < 4096) {
        const auto * _noalias const x = (dbl3_t *) atom_->x[0];
        auto * _noalias const f = (dbl3_t *) atom_->eval_f_stencil_md[0];
        const int * _noalias const type = atom_->type;
        const double * _noalias const special_lj = force->special_lj;
        const int * _noalias const ilist = list->ilist;
        const int * _noalias const numneigh = list->numneigh;
        const int * const * const firstneigh = list->firstneigh;

        // loop over neighbors of my atoms
        for (int ii = 0; ii < nlocal; ii++) {
            const int i = ilist[ii];
            const int itype = type[i];

            const int    * _noalias const jlist = firstneigh[i];
            const double * _noalias const cutsqi = cutsq[itype];
            const double * _noalias const offseti = offset[itype];
            const double * _noalias const lj1i = lj1[itype];
            const double * _noalias const lj2i = lj2[itype];
            const double * _noalias const lj3i = lj3[itype];
            const double * _noalias const lj4i = lj4[itype];

            double xtmp = x[i].x;
            double ytmp = x[i].y;
            double ztmp = x[i].z;
            int jnum = numneigh[i];

            double fxtmp = 0.0;
            double fytmp = 0.0;
            double fztmp = 0.0;

            for (int jj = 0; jj < jnum; jj++) {
                double evdwl = 0.0;
                int j = jlist[jj];
                double factor_lj = special_lj[sbmask(j)];
                j &= NEIGHMASK;

                double delx = xtmp - x[j].x;
                double dely = ytmp - x[j].y;
                double delz = ztmp - x[j].z;
                double rsq = delx*delx + dely*dely + delz*delz;
                int jtype = type[j];

                if (rsq < cutsqi[jtype]) {
                    double r2inv = 1.0/rsq;
                    double r6inv = r2inv*r2inv*r2inv;
                    double forcelj = r6inv * (lj1i[jtype]*r6inv - lj2i[jtype]);
                    double fpair = factor_lj*forcelj*r2inv;

                    fxtmp += delx*fpair;
                    fytmp += dely*fpair;
                    fztmp += delz*fpair;

                    /*
                    ftmp.x += delx*fpair;
                    ftmp.y += dely*fpair;
                    ftmp.z += delz*fpair;
                    */

                    if (newton_pair || j < nlocal) {
                        f[j].x -= delx*fpair;
                        f[j].y -= dely*fpair;
                        f[j].z -= delz*fpair;
                    }

                    /*
                    if (EFLAG) {
                        evdwl = r6inv*(lj3i[jtype]*r6inv-lj4i[jtype]) - offseti[jtype];
                        evdwl *= factor_lj;
                    }

                    if (EVFLAG) {
                        ev_tally_thr(this, i, j, nlocal, NEWTON_PAIR,
                                     evdwl, 0.0, fpair, delx, dely, delz, thr);
                    }
                    */
                }
            }

            f[i].x += fxtmp;
            f[i].y += fytmp;
            f[i].z += fztmp;
        }

        return;
    }

    int nthreads_to_use = inum / NUM_WORKERS_PER_THREAD;

    /*
    if (nthreads_to_use < 1) {
        nthreads_to_use = 1;
    }

    if (nthreads_to_use > nthreads) {
        nthreads_to_use = nthreads;
    }
    */

    double **f_ = atom_->eval_f_stencil_md;
    double **torque = atom_->torque;
    double *erforce = atom_->erforce;
    double *desph = atom_->desph;
    double *drho = atom_->drho;

    cilk_for (int tid = 0; tid < nthreads_to_use; tid++) {
        // each thread works on a fixed chunk of atoms.
        const int idelta = 1 + inum / nthreads_to_use;
        int ifrom = tid * idelta;
        int ito = ((ifrom + idelta) > inum) ? inum : ifrom + idelta;

        // loop_setup_thr(ifrom, ito, tid, inum, nthreads);
        ThrData *thr = fix->get_thr(tid);
        thr->timer(Timer::START);
        thr->init_force(nall,f_,torque,erforce,desph,drho);
        ev_setup_thr(eflag, vflag, nall, eatom, vatom, nullptr, thr);

        if (evflag) {
            if (eflag) {
                if (force->newton_pair) {
                    eval_stencil_md<1,1,1>(ifrom, ito, thr, atom_);
                } else {
                    eval_stencil_md<1,1,0>(ifrom, ito, thr, atom_);
                }
            } else {
                if (force->newton_pair) {
                    eval_stencil_md<1,0,1>(ifrom, ito, thr, atom_);
                } else {
                    eval_stencil_md<1,0,0>(ifrom, ito, thr, atom_);
                }
            }
        } else {
            if (force->newton_pair) {
                eval_stencil_md<0,0,1>(ifrom, ito, thr, atom_);
            } else {
                eval_stencil_md<0,0,0>(ifrom, ito, thr, atom_);
            }
        }
        thr->timer(Timer::PAIR);
    }

    if (USE_BOND) {
        return;
    }

    // In bond computation, no need to do reduce here. Can do it later in one go in bond_fene_omp.cpp.
    // try new reduce
    if (nthreads_to_use == 1) {
        return;
    }

    if (!USE_ATOMICS) {
        double* f = &(atom_->eval_f_stencil_md[0][0]);

        int nvals = nall * 3;

        cilk_for (int i = 0; i < nvals; i++) {
            for (int n = 1; n < nthreads_to_use; n++) {
                f[i] += f[n * nvals + i];
            }
        }
    }

    // try new reduce
    /*
    cilk_for (int tid = 0; tid < nthreads_to_use; tid++) {
        ThrData *thr = fix->get_thr(tid);
        reduce_thr_stencil_md(this, eflag, vflag, thr, atom_, nthreads_to_use);
    } // end of omp parallel region
    */
}

template <int EVFLAG, int EFLAG, int NEWTON_PAIR>
void PairLJCutOMP::eval(int iifrom, int iito, ThrData * const thr)
{
  const auto * _noalias const x = (dbl3_t *) atom->x[0];
  auto * _noalias const f = (dbl3_t *) thr->get_f()[0];
  const int * _noalias const type = atom->type;
  const double * _noalias const special_lj = force->special_lj;
  const int * _noalias const ilist = list->ilist;
  const int * _noalias const numneigh = list->numneigh;
  const int * const * const firstneigh = list->firstneigh;

  double xtmp,ytmp,ztmp,delx,dely,delz,fxtmp,fytmp,fztmp;
  double rsq,r2inv,r6inv,forcelj,factor_lj,evdwl,fpair;

  const int nlocal = atom->nlocal;
  int j,jj,jnum,jtype;

  evdwl = 0.0;

  // loop over neighbors of my atoms

  for (int ii = iifrom; ii < iito; ++ii) {
    const int i = ilist[ii];
    const int itype = type[i];
    const int    * _noalias const jlist = firstneigh[i];
    const double * _noalias const cutsqi = cutsq[itype];
    const double * _noalias const offseti = offset[itype];
    const double * _noalias const lj1i = lj1[itype];
    const double * _noalias const lj2i = lj2[itype];
    const double * _noalias const lj3i = lj3[itype];
    const double * _noalias const lj4i = lj4[itype];

    xtmp = x[i].x;
    ytmp = x[i].y;
    ztmp = x[i].z;
    jnum = numneigh[i];
    fxtmp=fytmp=fztmp=0.0;

    for (jj = 0; jj < jnum; jj++) {
      j = jlist[jj];
      factor_lj = special_lj[sbmask(j)];
      j &= NEIGHMASK;
      // num_lammps_edges++;

      delx = xtmp - x[j].x;
      dely = ytmp - x[j].y;
      delz = ztmp - x[j].z;
      rsq = delx*delx + dely*dely + delz*delz;
      jtype = type[j];

      if (rsq < cutsqi[jtype]) {
        r2inv = 1.0/rsq;
        r6inv = r2inv*r2inv*r2inv;
        forcelj = r6inv * (lj1i[jtype]*r6inv - lj2i[jtype]);
        fpair = factor_lj*forcelj*r2inv;

        fxtmp += delx*fpair;
        fytmp += dely*fpair;
        fztmp += delz*fpair;
        if (NEWTON_PAIR || j < nlocal) {
          f[j].x -= delx*fpair;
          f[j].y -= dely*fpair;
          f[j].z -= delz*fpair;
        }

        /*
        if (EFLAG) {
          evdwl = r6inv*(lj3i[jtype]*r6inv-lj4i[jtype]) - offseti[jtype];
          evdwl *= factor_lj;
        }

        if (EVFLAG) ev_tally_thr(this,i,j,nlocal,NEWTON_PAIR,
                                 evdwl,0.0,fpair,delx,dely,delz,thr);
        */
      }
    }
    f[i].x += fxtmp;
    f[i].y += fytmp;
    f[i].z += fztmp;
  }
}

template <int EVFLAG, int EFLAG, int NEWTON_PAIR>
__attribute__((always_inline)) void PairLJCutOMP::eval_stencil_md(int iifrom, int iito, ThrData * const thr, Atom* atom_)
{
    const auto * _noalias const x = (dbl3_t *) atom_->x[0];
    auto * _noalias const f = (dbl3_t *) thr->get_f()[0];
    const int * _noalias const type = atom_->type;
    const double * _noalias const special_lj = force->special_lj;
    const int * _noalias const ilist = list->ilist;
    const int * _noalias const numneigh = list->numneigh;
    const int * const * const firstneigh = list->firstneigh;
    /*
    double xtmp,ytmp,ztmp,delx,dely,delz,fxtmp,fytmp,fztmp;
    double rsq,r2inv,r6inv,forcelj,factor_lj,evdwl,fpair;
    */

    const int nlocal = atom->nlocal;
    // int j,jj,jnum,jtype;

    // loop over neighbors of my atoms

    for (int ii = iifrom; ii < iito; ++ii) {
        const int i = ilist[ii];
        const int itype = type[i];
        const int    * _noalias const jlist = firstneigh[i];
        const double * _noalias const cutsqi = cutsq[itype];
        const double * _noalias const offseti = offset[itype];
        const double * _noalias const lj1i = lj1[itype];
        const double * _noalias const lj2i = lj2[itype];
        const double * _noalias const lj3i = lj3[itype];
        const double * _noalias const lj4i = lj4[itype];

        double xtmp = x[i].x;
        double ytmp = x[i].y;
        double ztmp = x[i].z;
        int jnum = numneigh[i];

        double fxtmp = 0.0;
        double fytmp = 0.0;
        double fztmp = 0.0;

        for (int jj = 0; jj < jnum; jj++) {
            // num_edges++;
            double evdwl = 0.0;
            int j = jlist[jj];
            double factor_lj = special_lj[sbmask(j)];
            j &= NEIGHMASK;

            double delx = xtmp - x[j].x;
            double dely = ytmp - x[j].y;
            double delz = ztmp - x[j].z;
            double rsq = delx*delx + dely*dely + delz*delz;
            int jtype = type[j];

            if (rsq < cutsqi[jtype]) {
                double r2inv = 1.0/rsq;
                double r6inv = r2inv*r2inv*r2inv;
                double forcelj = r6inv * (lj1i[jtype]*r6inv - lj2i[jtype]);
                double fpair = factor_lj*forcelj*r2inv;

                fxtmp += delx*fpair;
                fytmp += dely*fpair;
                fztmp += delz*fpair;

                /*
                ftmp.x += delx*fpair;
                ftmp.y += dely*fpair;
                ftmp.z += delz*fpair;
                */

                if (NEWTON_PAIR || j < nlocal) {
                    f[j].x -= delx*fpair;
                    f[j].y -= dely*fpair;
                    f[j].z -= delz*fpair;
                }

                /*
                if (EFLAG) {
                    evdwl = r6inv*(lj3i[jtype]*r6inv-lj4i[jtype]) - offseti[jtype];
                    evdwl *= factor_lj;
                }

                if (EVFLAG) {
                    ev_tally_thr(this, i, j, nlocal, NEWTON_PAIR,
                                 evdwl, 0.0, fpair, delx, dely, delz, thr);
                }
                */
            }
        }

        f[i].x += fxtmp;
        f[i].y += fytmp;
        f[i].z += fztmp;
    }
}

/* ---------------------------------------------------------------------- */

double PairLJCutOMP::memory_usage()
{
  double bytes = memory_usage_thr();
  bytes += PairLJCut::memory_usage();

  return bytes;
}
