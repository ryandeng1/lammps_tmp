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
      // wsp_t start_compute = wsp_getworkspan();

      const auto * _noalias const x = (dbl3_t *) atom->x[0];
      const int * _noalias const type = atom->type;
      const double * _noalias const special_lj = force->special_lj;
      const int * _noalias const ilist = list->ilist;
      const int * _noalias const numneigh = list->numneigh;
      const int * const * const firstneigh = list->firstneigh;
      const int nlocal = atom->nlocal;

      bool newton_pair = force->newton_pair;

      cilk_for (int i = 0; i < atom->nlocal; i++) {
          int tid = __cilkrts_get_worker_number();
          ThrData *thr = fix->get_thr(tid);
          auto * _noalias const f = (dbl3_t *) thr->get_f()[0];
          thr->timer(Timer::START);
          ev_setup_thr(eflag, vflag, nall, eatom, vatom, nullptr, thr);

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
                  if (newton_pair || j < nlocal) {
                      f[j].x -= delx*fpair;
                      f[j].y -= dely*fpair;
                      f[j].z -= delz*fpair;
                  }

                  double evdwl = 0.0;
                  if (eflag) {
                      evdwl = r6inv*(lj3i[jtype]*r6inv-lj4i[jtype]) - offseti[jtype];
                      evdwl *= factor_lj;
                  }

                  if (eflag | vflag) ev_tally_thr(this,i,j,nlocal,newton_pair,
                                                  evdwl,0.0,fpair,delx,dely,delz,thr);
              }
          }
          f[i].x += fxtmp;
          f[i].y += fytmp;
          f[i].z += fztmp;
      }

      /*
      #pragma cilk grainsize 1
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
                      eval_stencil_md<1,1,1>(ifrom, ito, thr, atom);
                  } else {
                      eval_stencil_md<1,1,0>(ifrom, ito, thr, atom);
                  }
              } else {
                  if (force->newton_pair) {
                      eval_stencil_md<1,0,1>(ifrom, ito, thr, atom);
                  } else {
                      eval_stencil_md<1,0,0>(ifrom, ito, thr, atom);
                  }
              }
          } else {
              if (force->newton_pair) {
                  eval_stencil_md<0,0,1>(ifrom, ito, thr, atom);
              } else {
                  eval_stencil_md<0,0,0>(ifrom, ito, thr, atom);
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
      */

      /*
      wsp_t end_compute = wsp_getworkspan();
      wsp_t elapsed_compute = wsp_sub(end_compute, start_compute);

      if (comm->me == 0) {
          wsp_dump(elapsed_compute, "potential_calc");
      }
      */

      // #pragma cilk grainsize NUM_WORKERS_PER_THREAD
      double* f = &(atom->f[0][0]);
      int nvals = nall * 3;

      int nworkers = __cilkrts_get_nworkers();

      // wsp_t start_reduce = wsp_getworkspan();
      /*
      cilk_for (int i = 0; i < nvals; i++) {
          double t0 = f[i];
          for (int n = 1; n < nworkers; ++n) {
              t0 += f[n * nvals + i];
          }
          f[i] = t0;
      }
      */

      /*
      wsp_t end_reduce = wsp_getworkspan();
      wsp_t elapsed_reduce = wsp_sub(end_reduce, start_reduce);

      if (comm->me == 0) {
          wsp_dump(elapsed_reduce, "reduce");
      }
      */

      return;
  }


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
    const int nthreads = comm->nthreads;
    const int inum = list->inum;

    int nthreads_to_use = inum / NUM_WORKERS_PER_THREAD;

    if (nthreads_to_use < 1) {
        nthreads_to_use = 1;
    }

    if (nthreads_to_use > nthreads) {
        nthreads_to_use = nthreads;
    }

    // std::cout << "zoid: " << zoid.num << " nthreads: " << nthreads << " num workers to use: " << nthreads_to_use << " num local: " << inum << std::endl;

    /*
    if (true) {
        int nlocal = atom_->nlocal;

        int* ilist = list->ilist;
        int* numneigh = list->numneigh;
        int** firstneigh = list->firstneigh;

        // loop over neighbors of my atoms
        for (int ii = 0; ii < inum; ii++) {
            int i = ilist[ii];
            int* jlist = firstneigh[i];
            int jnum = numneigh[i];

            for (int jj = 0; jj < jnum; jj++) {
                int j = jlist[jj];

                if (num_eval != nullptr && ((zoid.num == 63 && *num_eval == 1))) {
                    std::cout << YELLOW << "zoid: " << zoid.num << " time: " << *num_eval << " i: " << i << " neighbor: " << j << " nlocal: " << atom_->nlocal << " nghost: " << atom_->nlocal + atom_->nghost
                        << " tag i: " << atom_->tag[i] << " tag j: " << atom_->tag[j] << " pos i: " << atom_->x[i][0] << " " << atom_->x[i][1] << " " << atom_->x[i][2]
                        << " pos j: " << atom_->x[j][0] << " " << atom_->x[j][1] << " " << atom_->x[j][2] << RESET_COLOR << std::endl;
                }
            }
        }
    }
    */

/*
#if defined(_OPENMP)
#pragma omp parallel LMP_DEFAULT_NONE LMP_SHARED(eflag,vflag)
#endif
*/

    /*
    cilk_for (int ii = 0; ii < nlocal; ii++) {
        int tid = __cilkrts_get_worker_number();
        ThrData *thr = fix->get_thr(tid);
        auto * _noalias const f = (dbl3_t *) thr->get_f()[0];
        ev_setup_thr(eflag, vflag, nall, eatom, vatom, nullptr, thr);

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

        double evdwl = 0.0;

        for (int jj = 0; jj < jnum; jj++) {
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
                if (force->newton_pair|| j < nlocal) {
                    f[j].x -= delx*fpair;
                    f[j].y -= dely*fpair;
                    f[j].z -= delz*fpair;
                }

                if (eflag) {
                    evdwl = r6inv*(lj3i[jtype]*r6inv-lj4i[jtype]) - offseti[jtype];
                    evdwl *= factor_lj;
                }

                if (eflag | vflag) {
                    ev_tally_thr(this, i, j, nlocal, force->newton_pair,
                                 evdwl, 0.0, fpair, delx, dely, delz, thr);
                }
            }
        }
        f[i].x += fxtmp;
        f[i].y += fytmp;
        f[i].z += fztmp;
    }
    */

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

    // try new reduce

    if (nthreads_to_use == 1) {
        return;
    }

    double* f = &(atom_->eval_f_stencil_md[0][0]);

    int nvals = nall * 3;

    // #pragma cilk grainsize NUM_WORKERS_PER_THREAD
    cilk_for (int i = 0; i < nvals; i++) {
        double t0 = f[i];
        for (int n = 1; n < nthreads_to_use; ++n) {
            t0 += f[n * nvals + i];
        }
        f[i] = t0;
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

        if (EFLAG) {
          evdwl = r6inv*(lj3i[jtype]*r6inv-lj4i[jtype]) - offseti[jtype];
          evdwl *= factor_lj;
        }

        if (EVFLAG) ev_tally_thr(this,i,j,nlocal,NEWTON_PAIR,
                                 evdwl,0.0,fpair,delx,dely,delz,thr);
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

                if (EFLAG) {
                    evdwl = r6inv*(lj3i[jtype]*r6inv-lj4i[jtype]) - offseti[jtype];
                    evdwl *= factor_lj;
                }

                if (EVFLAG) {
                    ev_tally_thr(this, i, j, nlocal, NEWTON_PAIR,
                                 evdwl, 0.0, fpair, delx, dely, delz, thr);
                }
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
