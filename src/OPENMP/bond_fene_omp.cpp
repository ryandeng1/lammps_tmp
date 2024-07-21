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

/* ----------------------------------------------------------------------
   Contributing author: Axel Kohlmeyer (Temple U)
------------------------------------------------------------------------- */

#include "omp_compat.h"
#include "bond_fene_omp.h"

#include "atom.h"
#include "comm.h"
#include "error.h"
#include "force.h"
#include "math_const.h"
#include "neighbor.h"
#include "update.h"

#include <cmath>
#include <cilk/cilk.h>

#include "suffix.h"
using namespace LAMMPS_NS;
using MathConst::MY_CUBEROOT2;

constexpr int target_tag = 12777;

/* ---------------------------------------------------------------------- */

BondFENEOMP::BondFENEOMP(class LAMMPS *lmp)
  : BondFENE(lmp), ThrOMP(lmp,THR_BOND)
{
  suffix_flag |= Suffix::OMP;
}

BondFENEOMP::BondFENEOMP(class LAMMPS *lmp, class Modify* modify_)
        : BondFENE(lmp), ThrOMP(lmp, modify_, THR_BOND)
{
    suffix_flag |= Suffix::OMP;
}

/* ---------------------------------------------------------------------- */

void BondFENEOMP::compute(int eflag, int vflag)
{
  ev_init(eflag,vflag);

  const int nall = atom->nlocal + atom->nghost;
  const int nthreads = comm->nthreads;
  const int inum = neighbor->nbondlist;

  if (LAMMPS_USE_CILK) {
      cilk_for(int tid = 0; tid < nthreads; tid++) {
          const int idelta = 1 + inum / nthreads;
          int ifrom = tid * idelta;
          int ito = ((ifrom + idelta) > inum) ? inum : ifrom + idelta;

          ThrData *thr = fix->get_thr(tid);
          thr->timer(Timer::START);
          ev_setup_thr(eflag, vflag, nall, eatom, vatom, nullptr, thr);

          if (inum > 0) {
              if (evflag) {
                  if (eflag) {
                      if (force->newton_bond) eval<1,1,1>(ifrom, ito, thr);
                      else eval<1,1,0>(ifrom, ito, thr);
                  } else {
                      if (force->newton_bond) eval<1,0,1>(ifrom, ito, thr);
                      else eval<1,0,0>(ifrom, ito, thr);
                  }
              } else {
                  if (force->newton_bond) eval<0,0,1>(ifrom, ito, thr);
                  else eval<0,0,0>(ifrom, ito, thr);
              }
          }
          thr->timer(Timer::BOND);
      }

      if (comm->nthreads == 1) {
          return;
      }

      double* f = &(atom->f[0][0]);
      int nvals = nall * 3;

      constexpr int CHUNK_SIZE = 512;

      cilk_for (int i = 0; i < nvals; i += CHUNK_SIZE) {
          for (int n = 1; n < comm->nthreads; n++) {
              for (int j = i; j < nvals && j < i + CHUNK_SIZE; j++) {
                  f[j] += f[n * nvals + j];
              }
          }
      }

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

    if (inum > 0) {
      if (evflag) {
        if (eflag) {
          if (force->newton_bond) eval<1,1,1>(ifrom, ito, thr);
          else eval<1,1,0>(ifrom, ito, thr);
        } else {
          if (force->newton_bond) eval<1,0,1>(ifrom, ito, thr);
          else eval<1,0,0>(ifrom, ito, thr);
        }
      } else {
        if (force->newton_bond) eval<0,0,1>(ifrom, ito, thr);
        else eval<0,0,0>(ifrom, ito, thr);
      }
    }
    thr->timer(Timer::BOND);
    reduce_thr(this, eflag, vflag, thr);
  } // end of omp parallel region
}

void BondFENEOMP::compute_stencil_md(int eflag, int vflag, Atom* atom_, bool* can_eval_center, queue_info& zoid, int* num_eval, Neighbor* neighbor_) {
    ev_init(eflag,vflag);
    const int nall = atom_->nlocal + atom_->nghost;
    const int nlocal = atom_->nlocal;
    const int nthreads = comm->nthreads;
    const int inum = neighbor_->nbondlist;
    double **f_ = atom_->eval_f_stencil_md;

    int nthreads_to_use = zoid.inum_per_timestep[*num_eval];
    int newton = force->newton_pair;

    if (USE_ATOMICS) {
        const auto * _noalias const x = (dbl3_t *) atom_->x[0];
        auto * _noalias const f = (dbl3_t *) atom_->eval_f_stencil_md[0];
        const int3_t * _noalias const bondlist = (int3_t *) neighbor_->bondlist[0];
        double ebond = 0.0;

        cilk_for (int n = 0; n < inum; n++) {
            int i1 = bondlist[n].a;
            int i2 = bondlist[n].b;
            int type = bondlist[n].t;

            double delx = x[i1].x - x[i2].x;
            double dely = x[i1].y - x[i2].y;
            double delz = x[i1].z - x[i2].z;

            double rsq = delx*delx + dely*dely + delz*delz;
            double r0sq = r0[type] * r0[type];
            double rlogarg = 1.0 - rsq/r0sq;

            // if r -> r0, then rlogarg < 0.0 which is an error
            // issue a warning and reset rlogarg = epsilon
            // if r > 2*r0 something serious is wrong, abort

            if (rlogarg < 0.1) {
                error->warning(FLERR,"FENE bond too long: {} {} {} {:.8}",
                               update->ntimestep,atom->tag[i1],atom->tag[i2],sqrt(rsq));
                /*
                if (check_error_thr((rlogarg <= -3.0),tid,FLERR,"Bad FENE bond"))
                    return;
                */
                assert(false);

                rlogarg = 0.1;
            }

            double fbond = -k[type]/rlogarg;

            // force from LJ term
            double sr2 = 0.0;
            double sr6 = 0.0;

            if (rsq < MY_CUBEROOT2*sigma[type]*sigma[type]) {
                sr2 = sigma[type]*sigma[type]/rsq;
                sr6 = sr2*sr2*sr2;
                fbond += 48.0*epsilon[type]*sr6*(sr6-0.5)/rsq;
            }

            // energy

            if (eflag) {
                ebond = -0.5 * k[type]*r0sq*log(rlogarg);
                if (rsq < MY_CUBEROOT2*sigma[type]*sigma[type])
                    ebond += 4.0*epsilon[type]*sr6*(sr6-1.0) + epsilon[type];
            }

            // apply force to each of 2 atoms

            if (newton || i1 < nlocal) {
                __atomic_fetch_add(&f[i1].x, delx*fbond, __ATOMIC_RELAXED);
                __atomic_fetch_add(&f[i1].y, dely*fbond, __ATOMIC_RELAXED);
                __atomic_fetch_add(&f[i1].z, delz*fbond, __ATOMIC_RELAXED);
                // f[i1].x += delx*fbond;
                // f[i1].y += dely*fbond;
                // f[i1].z += delz*fbond;
            }

            if (newton || i2 < nlocal) {
                // f[i2].x -= delx*fbond;
                // f[i2].y -= dely*fbond;
                // f[i2].z -= delz*fbond;
                __atomic_fetch_add(&f[i2].x, -delx*fbond, __ATOMIC_RELAXED);
                __atomic_fetch_add(&f[i2].y, -dely*fbond, __ATOMIC_RELAXED);
                __atomic_fetch_add(&f[i2].z, -delz*fbond, __ATOMIC_RELAXED);
            }

            /*
            if (evflag) ev_tally_thr(this,i1,i2,nlocal,newton,
                                     ebond,fbond,delx,dely,delz,thr);
            */
        }

        return;
    }

    if (PAIR_USE_BINS) {
        const auto * _noalias const x = (dbl3_t *) atom_->x[0];
        auto * _noalias const f = (dbl3_t *) atom_->eval_f_stencil_md[0];
        double ebond = 0.0;

        for (int dep = 0; dep < 27; dep++) {
            auto &special_bins_at_dep = atom_->special_pair_bins[dep];

            for (int bin_idx = 0; bin_idx < special_bins_at_dep.size(); bin_idx++) {
                auto &bin = special_bins_at_dep[bin_idx];
                auto &idxs = atom_->bin_to_local_idxs[bin];

                for (int idx = 0; idx < idxs.size(); idx++) {
                    int i1 = idxs[idx];
                    assert(i1 >= 0 && i1 < nlocal);

                    auto &lst_bonds = neighbor_->atom_bondlist[i1];
                    for (int j = 0; j < lst_bonds.size(); j++) {
                        auto &bond_info = lst_bonds[j];
                        int i2 = bond_info.first;
                        int type = bond_info.second;

                        double delx = x[i1].x - x[i2].x;
                        double dely = x[i1].y - x[i2].y;
                        double delz = x[i1].z - x[i2].z;

                        double rsq = delx * delx + dely * dely + delz * delz;
                        double r0sq = r0[type] * r0[type];
                        double rlogarg = 1.0 - rsq / r0sq;

                        if (rlogarg < 0.1) {
                            error->warning(FLERR, "FENE bond too long: {} {} {} {:.8}",
                                           update->ntimestep, atom->tag[i1], atom->tag[i2], sqrt(rsq));
                            /*
                            if (check_error_thr((rlogarg <= -3.0),tid,FLERR,"Bad FENE bond"))
                                return;
                            */
                            assert(false);

                            rlogarg = 0.1;
                        }

                        double fbond = -k[type] / rlogarg;

                        // force from LJ term
                        double sr2 = 0.0;
                        double sr6 = 0.0;

                        if (rsq < MY_CUBEROOT2 * sigma[type] * sigma[type]) {
                            sr2 = sigma[type] * sigma[type] / rsq;
                            sr6 = sr2 * sr2 * sr2;
                            fbond += 48.0 * epsilon[type] * sr6 * (sr6 - 0.5) / rsq;
                        }

                        // energy

                        if (eflag) {
                            ebond = -0.5 * k[type] * r0sq * log(rlogarg);
                            if (rsq < MY_CUBEROOT2 * sigma[type] * sigma[type])
                                ebond += 4.0 * epsilon[type] * sr6 * (sr6 - 1.0) + epsilon[type];
                        }

                        // apply force to each of 2 atoms

                        if (newton || i1 < nlocal) {
                            f[i1].x += delx * fbond;
                            f[i1].y += dely * fbond;
                            f[i1].z += delz * fbond;
                        }

                        if (newton || i2 < nlocal) {
                            f[i2].x -= delx * fbond;
                            f[i2].y -= dely * fbond;
                            f[i2].z -= delz * fbond;
                        }
                    }
                }
            }

            auto &bins_at_dep = atom_->pair_bins[dep];

            /*
            std::set<int> idxs_touched_at_dep;
            std::map<int, std::tuple<int, int, int>> idx_to_bin;
            std::map<int, std::set<int>> idx_to_touched_neighbor;
            */

            cilk_for (int bin_idx = 0; bin_idx < bins_at_dep.size(); bin_idx++) {
                auto &bin = bins_at_dep[bin_idx];
                auto &idxs = atom_->bin_to_local_idxs[bin];
                for (int idx = 0; idx < idxs.size(); idx++) {
                    int i1 = idxs[idx];
                    assert(i1 >= 0 && i1 < nlocal);

                    auto &lst_bonds = neighbor_->atom_bondlist[i1];
                    for (int j = 0; j < lst_bonds.size(); j++) {
                        auto &bond_info = lst_bonds[j];
                        int i2 = bond_info.first;
                        int type = bond_info.second;

                        double delx = x[i1].x - x[i2].x;
                        double dely = x[i1].y - x[i2].y;
                        double delz = x[i1].z - x[i2].z;

                        double rsq = delx * delx + dely * dely + delz * delz;
                        double r0sq = r0[type] * r0[type];
                        double rlogarg = 1.0 - rsq / r0sq;

                        if (rlogarg < 0.1) {
                            error->warning(FLERR, "FENE bond too long: {} {} {} {:.8}",
                                           update->ntimestep, atom->tag[i1], atom->tag[i2], sqrt(rsq));
                            /*
                            if (check_error_thr((rlogarg <= -3.0),tid,FLERR,"Bad FENE bond"))
                                return;
                            */
                            assert(false);

                            rlogarg = 0.1;
                        }

                        double fbond = -k[type] / rlogarg;

                        // force from LJ term
                        double sr2 = 0.0;
                        double sr6 = 0.0;

                        if (rsq < MY_CUBEROOT2 * sigma[type] * sigma[type]) {
                            sr2 = sigma[type] * sigma[type] / rsq;
                            sr6 = sr2 * sr2 * sr2;
                            fbond += 48.0 * epsilon[type] * sr6 * (sr6 - 0.5) / rsq;
                        }

                        // energy

                        if (eflag) {
                            ebond = -0.5 * k[type] * r0sq * log(rlogarg);
                            if (rsq < MY_CUBEROOT2 * sigma[type] * sigma[type])
                                ebond += 4.0 * epsilon[type] * sr6 * (sr6 - 1.0) + epsilon[type];
                        }

                        // apply force to each of 2 atoms

                        if (newton || i1 < nlocal) {
                            f[i1].x += delx * fbond;
                            f[i1].y += dely * fbond;
                            f[i1].z += delz * fbond;

                            /*
                            if (idxs_touched_at_dep.find(i1) != idxs_touched_at_dep.end() && bin != idx_to_bin[i1]) {
                                auto& overlap_bin = idx_to_bin[i1];
                                std::cout << "zoid: " << zoid.num << " dep: " << dep << " center idx: " << i1 << " neighbor idx: " << i2 << " tag: " << atom_->tag[i1] << " " << atom_->tag[i2]
                                          << " overlap. bin: " << std::get<0>(overlap_bin) << " " << std::get<1>(overlap_bin) << " " << std::get<2>(overlap_bin)
                                          << " curr bin: " << std::get<0>(bin) << " " << std::get<1>(bin) << " " << std::get<2>(bin) << std::endl;
                                std::cout << "pos: " << atom_->x[i1][0] << " " << atom_->x[i1][1] << " " << atom_->x[i1][2] << " other pos: " << atom_->x[i2][0] << " " << atom_->x[i2][1] << " " << atom_->x[i2][2] << std::endl;

                                auto& other_neighbor_idxs = idx_to_touched_neighbor[j];
                                for (auto& other_neighbor_idx : other_neighbor_idxs) {
                                    std::cout << "other neighbor: " << atom_->x[other_neighbor_idx][0] << " " << atom_->x[other_neighbor_idx][1] << " " << atom_->x[other_neighbor_idx][2]
                                              << " tag: " << atom_->tag[other_neighbor_idx] << std::endl;
                                }
                                assert(false);
                            }

                            idxs_touched_at_dep.insert(i1);
                            idx_to_bin[i1] = bin;
                            idx_to_touched_neighbor[i1].insert(i2);
                            */
                        }

                        if (newton || i2 < nlocal) {
                            f[i2].x -= delx * fbond;
                            f[i2].y -= dely * fbond;
                            f[i2].z -= delz * fbond;
                        }
                    }
                }
            }
        }

        return;
    }


    /*
    int nthreads_to_use = inum / NUM_WORKERS_PER_THREAD;

    if (nthreads_to_use < 1) {
        nthreads_to_use = 1;
    }

    if (nthreads_to_use > nthreads) {
        nthreads_to_use = nthreads;
    }
    */

    cilk_for (int tid = 0; tid < nthreads_to_use; tid++) {
        // each thread works on a fixed chunk of atoms.
        const int idelta = 1 + inum / nthreads_to_use;
        int ifrom = tid * idelta;
        int ito = ((ifrom + idelta) > inum) ? inum : ifrom + idelta;

        // loop_setup_thr(ifrom, ito, tid, inum, nthreads);
        ThrData *thr = fix->get_thr(tid);
        thr->timer(Timer::START);
        ev_setup_thr(eflag, vflag, nall, eatom, vatom, nullptr, thr);

        // pair should init force and do the reduce
        // thr->init_force(nall,f_, nullptr, nullptr, nullptr, nullptr);

        if (evflag) {
            if (eflag) {
                if (force->newton_pair) {
                    eval_stencil_md<1,1,1>(ifrom, ito, thr, atom_, neighbor_);
                } else {
                    eval_stencil_md<1,1,0>(ifrom, ito, thr, atom_, neighbor_);
                }
            } else {
                if (force->newton_pair) {
                    eval_stencil_md<1,0,1>(ifrom, ito, thr, atom_, neighbor_);
                } else {
                    eval_stencil_md<1,0,0>(ifrom, ito, thr, atom_, neighbor_);
                }
            }
        } else {
            if (force->newton_pair) {
                eval_stencil_md<0,0,1>(ifrom, ito, thr, atom_, neighbor_);
            } else {
                eval_stencil_md<0,0,0>(ifrom, ito, thr, atom_, neighbor_);
            }
        }
        thr->timer(Timer::PAIR);
    }

    // try new reduce
    if (nthreads_to_use == 1) {
        return;
    }

    if (!USE_ATOMICS) {
        double* f = &(atom_->eval_f_stencil_md[0][0]);

        int nvals = nall * 3;

        // do not explicitly set chunk size, have cilk figure it out.
        cilk_for (int i = 0; i < nvals; i++) {
            for (int n = 1; n < nthreads_to_use; n++) {
                f[i] += f[n * nvals + i];
            }
        }
    }
}

template <int EVFLAG, int EFLAG, int NEWTON_BOND>
void BondFENEOMP::eval(int nfrom, int nto, ThrData * const thr)
{
  int i1,i2,n,type;
  double delx,dely,delz,ebond,fbond;
  double rsq,r0sq,rlogarg,sr2,sr6;

  const auto * _noalias const x = (dbl3_t *) atom->x[0];
  auto * _noalias const f = (dbl3_t *) thr->get_f()[0];
  const int3_t * _noalias const bondlist = (int3_t *) neighbor->bondlist[0];
  const int nlocal = atom->nlocal;
  const int tid = thr->get_tid();
  ebond = 0.0;

  for (n = nfrom; n < nto; n++) {
    i1 = bondlist[n].a;
    i2 = bondlist[n].b;
    type = bondlist[n].t;

    delx = x[i1].x - x[i2].x;
    dely = x[i1].y - x[i2].y;
    delz = x[i1].z - x[i2].z;

    rsq = delx*delx + dely*dely + delz*delz;
    r0sq = r0[type] * r0[type];
    rlogarg = 1.0 - rsq/r0sq;

    // if r -> r0, then rlogarg < 0.0 which is an error
    // issue a warning and reset rlogarg = epsilon
    // if r > 2*r0 something serious is wrong, abort

    if (rlogarg < 0.1) {
      error->warning(FLERR,"FENE bond too long: {} {} {} {:.8}",
                     update->ntimestep,atom->tag[i1],atom->tag[i2],sqrt(rsq));
      if (check_error_thr((rlogarg <= -3.0),tid,FLERR,"Bad FENE bond"))
        return;

      rlogarg = 0.1;
    }

    fbond = -k[type]/rlogarg;

    // force from LJ term

    if (rsq < MY_CUBEROOT2*sigma[type]*sigma[type]) {
      sr2 = sigma[type]*sigma[type]/rsq;
      sr6 = sr2*sr2*sr2;
      fbond += 48.0*epsilon[type]*sr6*(sr6-0.5)/rsq;
    }

    // energy
    if (EFLAG) {
      ebond = -0.5 * k[type]*r0sq*log(rlogarg);
      if (rsq < MY_CUBEROOT2*sigma[type]*sigma[type])
        ebond += 4.0*epsilon[type]*sr6*(sr6-1.0) + epsilon[type];
    }

    // apply force to each of 2 atoms

    if (NEWTON_BOND || i1 < nlocal) {
      f[i1].x += delx*fbond;
      f[i1].y += dely*fbond;
      f[i1].z += delz*fbond;
    }

    if (NEWTON_BOND || i2 < nlocal) {
      f[i2].x -= delx*fbond;
      f[i2].y -= dely*fbond;
      f[i2].z -= delz*fbond;
    }

    if (EVFLAG) ev_tally_thr(this,i1,i2,nlocal,NEWTON_BOND,
                             ebond,fbond,delx,dely,delz,thr);
  }
}

template <int EVFLAG, int EFLAG, int NEWTON_BOND>
void BondFENEOMP::eval_stencil_md(int nfrom, int nto, ThrData * const thr, Atom* atom_, Neighbor* neighbor_) {
    int i1,i2,n,type;
    double delx,dely,delz,ebond,fbond;
    double rsq,r0sq,rlogarg,sr2,sr6;

    const auto * _noalias const x = (dbl3_t *) atom_->x[0];
    auto * _noalias const f = (dbl3_t *) thr->get_f()[0];
    const int3_t * _noalias const bondlist = (int3_t *) neighbor_->bondlist[0];
    const int nlocal = atom_->nlocal;
    const int tid = thr->get_tid();
    ebond = 0.0;

    for (n = nfrom; n < nto; n++) {
        i1 = bondlist[n].a;
        i2 = bondlist[n].b;
        type = bondlist[n].t;

        delx = x[i1].x - x[i2].x;
        dely = x[i1].y - x[i2].y;
        delz = x[i1].z - x[i2].z;

        rsq = delx*delx + dely*dely + delz*delz;
        r0sq = r0[type] * r0[type];
        rlogarg = 1.0 - rsq/r0sq;

        // if r -> r0, then rlogarg < 0.0 which is an error
        // issue a warning and reset rlogarg = epsilon
        // if r > 2*r0 something serious is wrong, abort

        if (rlogarg < 0.1) {
            error->warning(FLERR,"FENE bond too long: {} {} {} {:.8}",
                           update->ntimestep,atom->tag[i1],atom->tag[i2],sqrt(rsq));
            if (check_error_thr((rlogarg <= -3.0),tid,FLERR,"Bad FENE bond"))
                return;

            rlogarg = 0.1;
        }

        fbond = -k[type]/rlogarg;

        // force from LJ term

        if (rsq < MY_CUBEROOT2*sigma[type]*sigma[type]) {
            sr2 = sigma[type]*sigma[type]/rsq;
            sr6 = sr2*sr2*sr2;
            fbond += 48.0*epsilon[type]*sr6*(sr6-0.5)/rsq;
        }

        // energy

        if (EFLAG) {
            ebond = -0.5 * k[type]*r0sq*log(rlogarg);
            if (rsq < MY_CUBEROOT2*sigma[type]*sigma[type])
                ebond += 4.0*epsilon[type]*sr6*(sr6-1.0) + epsilon[type];
        }

        // apply force to each of 2 atoms

        if (NEWTON_BOND || i1 < nlocal) {
            f[i1].x += delx*fbond;
            f[i1].y += dely*fbond;
            f[i1].z += delz*fbond;
        }

        if (NEWTON_BOND || i2 < nlocal) {
            f[i2].x -= delx*fbond;
            f[i2].y -= dely*fbond;
            f[i2].z -= delz*fbond;
        }

        if (EVFLAG) ev_tally_thr(this,i1,i2,nlocal,NEWTON_BOND,
                                 ebond,fbond,delx,dely,delz,thr);
    }
}
