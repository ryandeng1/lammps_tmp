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
  ev_init(eflag,vflag);

  const int nall = atom->nlocal + atom->nghost;
  const int nthreads = comm->nthreads;
  const int inum = list->inum;

  if (LAMMPS_USE_CILK) {
      cilk_for (int tid = 0; tid < comm->nthreads; tid++) {
          // int ifrom, ito, tid;
          int ifrom, ito;
          // each thread works on a fixed chunk of atoms.
          const int idelta = 1 + inum / comm->nthreads;
          ifrom = tid * idelta;
          ito = ((ifrom + idelta) > inum) ? inum : ifrom + idelta;

          // loop_setup_thr(ifrom, ito, tid, inum, nthreads);
          ThrData *thr = fix->get_thr(tid);
          thr->timer(Timer::START);
          ev_setup_thr(eflag, vflag, nall, eatom, vatom, nullptr, thr);

          if (evflag) {
              if (eflag) {
                  if (force->newton_pair) {
                      // eval_stencil_md<1,1,1>(ifrom, ito, thr, atom);
                      eval<1, 1, 1>(ifrom, ito, thr);
                  } else {
                      // eval_stencil_md<1,1,0>(ifrom, ito, thr, atom);
                      eval<1, 1, 0>(ifrom, ito, thr);
                  }
              } else {
                  if (force->newton_pair) {
                      // eval_stencil_md<1,0,1>(ifrom, ito, thr, atom);
                      eval<1,0,1>(ifrom, ito, thr);
                  } else {
                      // eval_stencil_md<1,0,0>(ifrom, ito, thr, atom);
                      eval<1,0,0>(ifrom, ito, thr);
                  }
              }
          } else {
              if (force->newton_pair) {
                  // eval_stencil_md<0,0,1>(ifrom, ito, thr, atom);
                  eval<0,0,1>(ifrom, ito, thr);
              } else {
                  // eval_stencil_md<0,0,0>(ifrom, ito, thr, atom);
                  eval<0,0,0>(ifrom, ito, thr);
              }
          }
          thr->timer(Timer::PAIR);
      } // end of omp parallel region

      // try new reduce

      if (comm->nthreads == 1) {
          return;
      }

      double* f = &(atom->f[0][0]);
      int nvals = nall * 3;

      constexpr int CHUNK_SIZE = 256;

      cilk_for (int i = 0; i < nvals; i += CHUNK_SIZE) {
          for (int n = 1; n < comm->nthreads; n++) {
              for (int j = i; j < nvals && j < i + CHUNK_SIZE; j++) {
                  f[j] += f[n * nvals + j];
              }
          }
      }

      /*
      cilk_for (int i = 0; i < nvals; i++) {
          double t0 = f[i];
          for (int n = 1; n < nworkers; ++n) {
              t0 += f[n * nvals + i];
          }
          f[i] = t0;
      }
      */

      return;
  }


  /*
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
  */
}

void PairLJCutOMP::compute_stencil_md(int eflag, int vflag, Atom* atom_, bool* can_eval_center, queue_info& zoid, int* num_eval) {
    int num_edges = 0;
    ev_init(eflag,vflag);
    const int nall = atom_->nlocal + atom_->nghost;
    const int nlocal = atom_->nlocal;
    const int nthreads = comm->nthreads;
    const int inum = list->inum;

    int nthreads_to_use = inum / NUM_WORKERS_PER_THREAD;

    if (nthreads_to_use < 1) {
        nthreads_to_use = 1;
    }

    if (nthreads_to_use > nthreads) {
        nthreads_to_use = nthreads;
    }

    double **x = atom_->x;
    int *type = atom_->type;
    double *special_lj = force->special_lj;
    int newton_pair = force->newton_pair;

    int* ilist = list->ilist;
    int* numneigh = list->numneigh;
    int** firstneigh = list->firstneigh;

    cilk_for (int tid = 0; tid < nthreads_to_use; tid++) {
        // int ifrom, ito, tid;
        int ifrom, ito;
        // each thread works on a fixed chunk of atoms.
        const int idelta = 1 + inum / nthreads_to_use;
        ifrom = tid * idelta;
        ito = ((ifrom + idelta) > inum) ? inum : ifrom + idelta;
    // cilk_for (int ii = 0; ii < atom_->nlocal; ii++) {
        // cilk::opadd_reducer<double> fxtmp = 0.0;
        // cilk::opadd_reducer<double> fytmp = 0.0;
        // cilk::opadd_reducer<double> fztmp = 0.0;

        // loop_setup_thr(ifrom, ito, tid, inum, nthreads);
        ThrData *thr = fix->get_thr(tid);
        thr->timer(Timer::START);
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

        /*
        int i = ilist[ii];
        double xtmp = x[i][0];
        double ytmp = x[i][1];
        double ztmp = x[i][2];

        int itype = type[i];
        int* jlist = firstneigh[i];
        int jnum = numneigh[i];

        // loop over neighbors of my atoms

        cilk_for (int jj = 0; jj < jnum; jj++) {
            double evdwl = 0.0;

            int tid = __cilkrts_get_worker_number();
            ThrData *thr = fix->get_thr(tid);
            // auto * _noalias const f = (dbl3_t *) thr->get_f()[0];
            double** f = thr->get_f();

            int j = jlist[jj];
            double factor_lj = special_lj[sbmask(j)];
            j &= NEIGHMASK;

            double delx = xtmp - x[j][0];
            double dely = ytmp - x[j][1];
            double delz = ztmp - x[j][2];
            double rsq = delx * delx + dely * dely + delz * delz;
            int jtype = type[j];

            if (rsq < cutsq[itype][jtype]) {
                double r2inv = 1.0 / rsq;
                double r6inv = r2inv * r2inv * r2inv;
                double forcelj = r6inv * (lj1[itype][jtype] * r6inv - lj2[itype][jtype]);
                double fpair = factor_lj * forcelj * r2inv;

                fxtmp += delx * fpair;
                fytmp += dely * fpair;
                fztmp += delz * fpair;

                if (newton_pair || j < nlocal) {
                    f[j][0] -= delx * fpair;
                    f[j][1] -= dely * fpair;
                    f[j][2] -= delz * fpair;
                    // f[j].x -= delx * fpair;
                    // f[j].y -= dely * fpair;
                    // f[j].z -= delz * fpair;
                }

                if (eflag) {
                    evdwl = r6inv * (lj3[itype][jtype] * r6inv - lj4[itype][jtype]) - offset[itype][jtype];
                    evdwl *= factor_lj;
                }

                if (evflag) ev_tally(i, j, nlocal, newton_pair, evdwl, 0.0, fpair, delx, dely, delz);
            }
        }

        int tid = __cilkrts_get_worker_number();
        ThrData *thr = fix->get_thr(tid);
        // auto * _noalias const f = (dbl3_t *) thr->get_f()[0];
        double** f = thr->get_f();

        // f[i].x += fxtmp;
        // f[i].y += fytmp;
        // f[i].z += fztmp;
        f[i][0] += fxtmp;
        f[i][1] += fytmp;
        f[i][2] += fztmp;
        */
    }

    /*
    cilk_for (int tid = 0; tid < nthreads_to_use; tid++) {
        // int ifrom, ito, tid;
        int ifrom, ito;
        // each thread works on a fixed chunk of atoms.
        const int idelta = 1 + inum / nthreads_to_use;
        ifrom = tid * idelta;
        ito = ((ifrom + idelta) > inum) ? inum : ifrom + idelta;

        // loop_setup_thr(ifrom, ito, tid, inum, nthreads);
        ThrData *thr = fix->get_thr(tid);
        thr->timer(Timer::START);
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
        // reduce_thr_stencil_md(this, eflag, vflag, thr, atom_);
    } // end of omp parallel region
    */

    // try new reduce

    if (nthreads_to_use == 1) {
        return;
    }

    double* f = &(atom_->eval_f_stencil_md[0][0]);

    int nvals = nall * 3;

    constexpr int CHUNK_SIZE = 128;

    cilk_for (int i = 0; i < nvals; i += CHUNK_SIZE) {
        for (int n = 1; n < nthreads_to_use; n++) {
            for (int j = i; j < nvals && j < i + CHUNK_SIZE; j++) {
                f[j] += f[n * nvals + j];
            }
        }
    }

    /*
    // #pragma cilk grainsize NUM_WORKERS_PER_THREAD
    cilk_for (int i = 0; i < nvals; i++) {
        double t0 = f[i];
        for (int n = 1; n < nthreads_to_use; ++n) {
            t0 += f[n * nvals + i];
        }
        f[i] = t0;
    }
    */

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
inline void PairLJCutOMP::eval_stencil_md(int iifrom, int iito, ThrData * const thr, Atom* atom_)
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

    cilk::opadd_reducer<double> fxtmp = 0.0;
    cilk::opadd_reducer<double> fytmp = 0.0;
    cilk::opadd_reducer<double> fztmp = 0.0;

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

//        cilk::opadd_reducer<double> fxtmp = 0.0;
//        cilk::opadd_reducer<double> fytmp = 0.0;
//        cilk::opadd_reducer<double> fztmp = 0.0;
        fxtmp = 0.0;
        fytmp = 0.0;
        fztmp = 0.0;
        // double fxtmp = 0.0;
        // double fytmp = 0.0;
        // double fztmp = 0.0;

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
