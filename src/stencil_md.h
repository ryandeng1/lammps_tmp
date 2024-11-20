//
// Created by Ryan Deng on 2/27/24.
//

#pragma once

#include "pointers.h"
#include "bond_fene.h"
#include "pair_lj_cut.h"
#include "pair_lj_cut_omp.h"
#include <cilk/cilk.h>
#include <cilk/cilk_api.h>
#include <cilk/cilksan.h>
#include "force.h"
#include "modify.h"
#include "pair.h"
#include "bond.h"
#include "atom.h"
#include "neighbor.h"
#include "comm.h"
#include "neigh_list.h"
#include "math_const.h"
#include "update.h"
#include "error.h"
#include "domain.h"
#include "thr_omp.h"
#include "fix_langevin.h"
#include <cilk/opadd_reducer.h>

namespace LAMMPS_NS {

class StencilMD : protected Pointers {
public:
    StencilMD(class LAMMPS *lmp) : Pointers(lmp) {}

    void MODIFY_ADD_FIX_STENCIL_MD(int narg, char **arg);

    void MODIFY_ADD_FIX_PACKAGE_STENCIL_MD(const std::string& fixcmd);

    void MODIFY_ADD_COMPUTE_STENCIL_MD(int narg, char **arg);

    void ATOM_STYLE(const std::string &style, int narg, char **arg, int trysuffix);

    void ATOM_SETTINGS();

    void FORCE_SET_SPECIAL(int narg, char **arg);

    void FORCE_CREATE_BOND(const std::string& style, int trysuffix);

    void FORCE_BOND_SETTINGS(int narg, char **arg);

    void FORCE_BOND_COEFF(int narg, char **arg);

    void FORCE_PAIR_COEFF(int narg, char **arg);

    void FORCE_MODIFY_PARAMS(int narg, char **arg);

    void FORCE_CREATE_PAIR(const std::string& style, int trysuffix);

    void FORCE_PAIR_SETTINGS(int narg, char **arg);

    void CREATE();

    void CREATE_NEXT_DT();

    void INIT_ZOIDS();

    void INIT_ZOID_DATA();

    void INIT_ZOID_NEIGHBORS();

    void INIT_DOMAIN_BOUNDS();

    void INIT_ALL();

    void SETUP();

    void MODIFY_PRE_FORCE_SETUP(int);

    void MODIFY_SETUP(int);

    void GET_LOCAL_ATOMS_ZOID();

    void GET_GHOST_ATOMS_ZOID();

    void SORT_LOCAL_ATOMS_BINS();

    void CREATE_ATOM_IDX_MAPPING();

    void BUILD_NEIGHBOR_LIST();
    void BUILD_NEIGHBOR_LIST_NEXT_DT();

    void COMPUTE_NUM_SEND_RECV_PROCESS();

    void COMPARE_POS_AGAINST_LAMMPS(bool curr_dt, int timestep, Atom* atom_, queue_info& zoid, double** test_x);
    void COMPARE_FORCE_AGAINST_LAMMPS(bool curr_dt, int timestep, Atom* atom_, queue_info& zoid, double** test_f);
    void COMPARE_VEL_AGAINST_LAMMPS(bool curr_dt, int timestep, Atom* atom_, queue_info& zoid, double** test_f);

    void SET_INUM_PER_TIMESTEP();
    void SET_INUM_PER_TIMESTEP_NEXT_DT();

    void SET_CLAIMED_ATOMIC_BOOLS();

    std::vector<double>& GET_BOUNDS(bool curr_dt, int timestep);

    std::vector<double>& LAMMPS_GET_BOUNDS(bool curr_dt, int timestep);

    std::vector<double> lammps_bounds[NUM_TIMESTEPS_IN_PARALLEL + 1];
    std::vector<std::size_t> lammps_sorted_bin_indices[NUM_TIMESTEPS_IN_PARALLEL + 1];

    std::vector<double> bounds[NUM_TIMESTEPS_IN_PARALLEL + 1];
    std::vector<std::size_t> sorted_bin_indices[NUM_TIMESTEPS_IN_PARALLEL + 1];

    dbl3_t_stencil_md* lammps_f;
    static constexpr int NUM_ARRAYS = 64;

    std::array<std::vector<std::pair<int, dbl3_t_stencil_md>>, NUM_ARRAYS>* lammps_f_updates;
    int nworkers;

    void INIT_PER_PARTITION_FORCE_ARRAY() {
        return;
        int nall = (atom->nlocal + atom->nghost);
        lammps_f = new dbl3_t_stencil_md[nall * NUM_ARRAYS];
        memset(lammps_f, 0, nall * NUM_ARRAYS * 3 * sizeof(double));

        nworkers = __cilkrts_get_nworkers();
        lammps_f_updates = new std::array<std::vector<std::pair<int, dbl3_t_stencil_md>>, NUM_ARRAYS>[nworkers];
    }

    void lammps_fuse_reduce2() {
        const auto * _noalias const x = (dbl3_t_stencil_md *) atom->x[0];
        auto * _noalias const f = (dbl3_t_stencil_md *) atom->f[0];

        auto pair = (PairLJCutOMP*) force->pair;
        auto bond = (BondFENE*) force->bond;

        const auto* _noalias bondlist = neighbor->atom_bondlist;

        const int * _noalias const ilist = pair->list->ilist;
        const int * _noalias const numneigh = pair->list->numneigh;
        const int * const * const firstneigh = pair->list->firstneigh;
        const double * _noalias const special_lj = force->special_lj;

        auto* spinlocks = atom->spinlocks;
        auto idx_use_atomics = atom->idx_use_atomics;

        assert(pair->list->inum == atom->nlocal);

        const auto* cutsq = pair->cutsq;
        const auto* offset = pair->offset;
        const auto* lj1 = pair->lj1;
        const auto* lj2 = pair->lj2;
        const auto* lj3 = pair->lj3;
        const auto* lj4 = pair->lj4;
        auto newton_pair = force->newton_pair;

        const auto* _noalias const sigma = bond->sigma;
        const auto* _noalias const epsilon = bond->epsilon;
        const auto* _noalias const r0 = bond->r0;
        const auto* _noalias const k = bond->k;

        const int* _noalias const atom_type = atom->type;

        const int nlocal = atom->nlocal;
        const int nall = atom->nlocal + atom->nghost;

        auto& local_bins = atom->local_bins;
        auto& local_bins_idxs = atom->local_bins_idxs;

        assert(local_bins.size() == NUM_ARRAYS);

        cilk_for (int b = 0; b < local_bins.size(); b++) {
            auto& bin = local_bins[b];
            auto& idxs = local_bins_idxs[b];
            auto& updates = lammps_f_updates[__cilkrts_get_worker_number()];

            int start = idxs[0];
            for (int idx = 0; idx < idxs.size(); idx++) {
                int ii = start + idx;
                assert(ii == start + idx);
                const int i = ilist[ii];
                assert(i == ii);
                assert(ii >= 0 && ii < nlocal);
                const int itype = atom_type[i];

                // assert(bin_use_atomics == atom->idx_use_atomics[i]);

                // const int *_noalias const jlist = firstneigh[i];
                const auto& jlist_use_atomics = firstneigh_use_atomics[ii];
                const auto& jlist_no_atomics = firstneigh_no_atomics[ii];

                const auto& lst_bonds_use_atomics = bondlist_use_atomics[i];
                const auto& lst_bonds_no_atomics = bondlist_no_atomics[i];

                const double *_noalias const cutsqi = cutsq[itype];
                const double *_noalias const offseti = offset[itype];
                const double *_noalias const lj1i = lj1[itype];
                const double *_noalias const lj2i = lj2[itype];
                const double *_noalias const lj3i = lj3[itype];
                const double *_noalias const lj4i = lj4[itype];

                double xtmp = x[i].x;
                double ytmp = x[i].y;
                double ztmp = x[i].z;
                int jnum = numneigh[i];

                double fxtmp = 0.0;
                double fytmp = 0.0;
                double fztmp = 0.0;

                for (int jj = 0; jj < jlist_use_atomics.size(); jj++) {
                    double evdwl = 0.0;
                    int j = jlist_use_atomics[jj];
                    double factor_lj = special_lj[pair->sbmask(j)];
                    j &= NEIGHMASK;

                    double delx = xtmp - x[j].x;
                    double dely = ytmp - x[j].y;
                    double delz = ztmp - x[j].z;
                    double rsq = delx * delx + dely * dely + delz * delz;
                    int jtype = atom_type[j];

                    if (rsq < cutsqi[jtype]) {
                        double r2inv = 1.0 / rsq;
                        double r6inv = r2inv * r2inv * r2inv;
                        double forcelj = r6inv * (lj1i[jtype] * r6inv - lj2i[jtype]);
                        double fpair = factor_lj * forcelj * r2inv;

                        fxtmp += delx * fpair;
                        fytmp += dely * fpair;
                        fztmp += delz * fpair;

                        if (newton_pair || j < nlocal) {
                            // spinlocks[j].lock();
                            updates[b].emplace_back(j, dbl3_t_stencil_md{-delx*fpair, -dely*fpair, -delz*fpair});
                            // f[j].x -= delx * fpair;
                            // f[j].y -= dely * fpair;
                            // f[j].z -= delz * fpair;
                            // spinlocks[j].unlock();
                        }
                    }
                }

                for (int jj = 0; jj < jlist_no_atomics.size(); jj++) {
                    double evdwl = 0.0;
                    int j = jlist_no_atomics[jj];
                    double factor_lj = special_lj[pair->sbmask(j)];
                    j &= NEIGHMASK;

                    double delx = xtmp - x[j].x;
                    double dely = ytmp - x[j].y;
                    double delz = ztmp - x[j].z;
                    double rsq = delx * delx + dely * dely + delz * delz;
                    int jtype = atom_type[j];

                    if (rsq < cutsqi[jtype]) {
                        double r2inv = 1.0 / rsq;
                        double r6inv = r2inv * r2inv * r2inv;
                        double forcelj = r6inv * (lj1i[jtype] * r6inv - lj2i[jtype]);
                        double fpair = factor_lj * forcelj * r2inv;

                        fxtmp += delx * fpair;
                        fytmp += dely * fpair;
                        fztmp += delz * fpair;

                        if (newton_pair || j < nlocal) {
                            f[j].x -= delx * fpair;
                            f[j].y -= dely * fpair;
                            f[j].z -= delz * fpair;
                        }
                    }
                }

                // auto& lst_bonds = bondlist[i];
                for (auto& [i2, type] : bondlist_use_atomics[i]) {
                    double delx = xtmp - x[i2].x;
                    double dely = ytmp - x[i2].y;
                    double delz = ztmp - x[i2].z;

                    double rsq = delx * delx + dely * dely + delz * delz;
                    double r0sq = r0[type] * r0[type];
                    double rlogarg = 1.0 - rsq / r0sq;

                    if (rlogarg < 0.1) {
                        error->warning(FLERR, "FENE bond too long: {} {} {} {:.8}",
                                       update->ntimestep, atom->tag[i], atom->tag[i2], sqrt(rsq));
                        //                            if (check_error_thr((rlogarg <= -3.0),tid,FLERR,"Bad FENE bond"))
                        //                                return;
                        assert(false);

                        rlogarg = 0.1;
                    }

                    double fbond = -k[type] / rlogarg;

                    // force from LJ term
                    double sr2 = 0.0;
                    double sr6 = 0.0;

                    if (rsq < MathConst::MY_CUBEROOT2 * sigma[type] * sigma[type]) {
                        sr2 = sigma[type] * sigma[type] / rsq;
                        sr6 = sr2 * sr2 * sr2;
                        fbond += 48.0 * epsilon[type] * sr6 * (sr6 - 0.5) / rsq;
                    }

                    // energy

                    // apply force to each of 2 atoms

                    if (newton_pair || i < nlocal) {
                        fxtmp += delx * fbond;
                        fytmp += dely * fbond;
                        fztmp += delz * fbond;
                    }

                    if (newton_pair || i2 < nlocal) {
                        // spinlocks[i2].lock();
                        updates[b].emplace_back(i2, dbl3_t_stencil_md{-delx*fbond, -dely*fbond, -delz*fbond});
                        // f[i2].x -= delx * fbond;
                        // f[i2].y -= dely * fbond;
                        // f[i2].z -= delz * fbond;
                        // spinlocks[i2].unlock();
                    }
                }

                for (auto& [i2, type] : bondlist_no_atomics[i]) {
                    double delx = xtmp - x[i2].x;
                    double dely = ytmp - x[i2].y;
                    double delz = ztmp - x[i2].z;

                    double rsq = delx * delx + dely * dely + delz * delz;
                    double r0sq = r0[type] * r0[type];
                    double rlogarg = 1.0 - rsq / r0sq;

                    if (rlogarg < 0.1) {
                        error->warning(FLERR, "FENE bond too long: {} {} {} {:.8}",
                                       update->ntimestep, atom->tag[i], atom->tag[i2], sqrt(rsq));
                        //                            if (check_error_thr((rlogarg <= -3.0),tid,FLERR,"Bad FENE bond"))
                        //                                return;
                        assert(false);

                        rlogarg = 0.1;
                    }

                    double fbond = -k[type] / rlogarg;

                    // force from LJ term
                    double sr2 = 0.0;
                    double sr6 = 0.0;

                    if (rsq < MathConst::MY_CUBEROOT2 * sigma[type] * sigma[type]) {
                        sr2 = sigma[type] * sigma[type] / rsq;
                        sr6 = sr2 * sr2 * sr2;
                        fbond += 48.0 * epsilon[type] * sr6 * (sr6 - 0.5) / rsq;
                    }

                    // energy

                    // apply force to each of 2 atoms

                    if (newton_pair || i < nlocal) {
                        fxtmp += delx * fbond;
                        fytmp += dely * fbond;
                        fztmp += delz * fbond;
                    }

                    if (newton_pair || i2 < nlocal) {
                        f[i2].x -= delx * fbond;
                        f[i2].y -= dely * fbond;
                        f[i2].z -= delz * fbond;
                    }
                }

                f[i].x += fxtmp;
                f[i].y += fytmp;
                f[i].z += fztmp;
            }
        }

        // do reduction
        for (int b = 0; b < local_bins.size(); b++) {
            for (int i = 0; i < nworkers; i++) {
                for (auto& [idx, info] : lammps_f_updates[i][b]) {
                    f[idx].x += info.x;
                    f[idx].y += info.y;
                    f[idx].z += info.z;
                }
                lammps_f_updates[i][b].clear();
            }
        }
    }

    void lammps_fuse_reduce() {
        const auto * _noalias const x = (dbl3_t_stencil_md *) atom->x[0];
        auto * _noalias const f = (dbl3_t_stencil_md *) atom->f[0];

        auto pair = (PairLJCutOMP*) force->pair;
        auto bond = (BondFENE*) force->bond;

        const auto* _noalias bondlist = neighbor->atom_bondlist;

        const int * _noalias const ilist = pair->list->ilist;
        const int * _noalias const numneigh = pair->list->numneigh;
        const int * const * const firstneigh = pair->list->firstneigh;
        const double * _noalias const special_lj = force->special_lj;

        auto* spinlocks = atom->spinlocks;
        // auto mutexes = atom->mutexes;

        assert(pair->list->inum == atom->nlocal);

        const auto* cutsq = pair->cutsq;
        const auto* offset = pair->offset;
        const auto* lj1 = pair->lj1;
        const auto* lj2 = pair->lj2;
        const auto* lj3 = pair->lj3;
        const auto* lj4 = pair->lj4;
        auto newton_pair = force->newton_pair;

        const auto* _noalias const sigma = bond->sigma;
        const auto* _noalias const epsilon = bond->epsilon;
        const auto* _noalias const r0 = bond->r0;
        const auto* _noalias const k = bond->k;

        const int* _noalias const atom_type = atom->type;

        const int nlocal = atom->nlocal;
        const int nall = atom->nlocal + atom->nghost;

        auto& local_bins = atom->local_bins;
        auto& local_bins_idxs = atom->local_bins_idxs;

        assert(local_bins.size() == NUM_ARRAYS);

        cilk_for (int b = 0; b < local_bins.size(); b++) {
            auto& bin = local_bins[b];
            auto& idxs = local_bins_idxs[b];
            auto f_arr = &lammps_f[b * nall];
            memset(f_arr, 0, nall * 3 * sizeof(double));

            for (int i = 0; i < nall; i++) {
                assert(fabs(f_arr[i].x) < 1e-6);
                assert(fabs(f_arr[i].y) < 1e-6);
                assert(fabs(f_arr[i].z) < 1e-6);
            }

            int start = idxs[0];
            for (int idx = 0; idx < idxs.size(); idx++) {
                int ii = start + idx;
                assert(ii == start + idx);
                const int i = ilist[ii];
                assert(i == ii);
                assert(ii >= 0 && ii < nlocal);
                const int itype = atom_type[i];

                const int *_noalias const jlist = firstneigh[i];
                const int jnum = numneigh[i];

                const double *_noalias const cutsqi = cutsq[itype];
                const double *_noalias const offseti = offset[itype];
                const double *_noalias const lj1i = lj1[itype];
                const double *_noalias const lj2i = lj2[itype];
                const double *_noalias const lj3i = lj3[itype];
                const double *_noalias const lj4i = lj4[itype];

                double xtmp = x[i].x;
                double ytmp = x[i].y;
                double ztmp = x[i].z;

                double fxtmp = 0.0;
                double fytmp = 0.0;
                double fztmp = 0.0;

                for (int jj = 0; jj < jnum; jj++) {
                    double evdwl = 0.0;
                    int j = jlist[jj];
                    double factor_lj = special_lj[pair->sbmask(j)];
                    j &= NEIGHMASK;

                    double delx = xtmp - x[j].x;
                    double dely = ytmp - x[j].y;
                    double delz = ztmp - x[j].z;
                    double rsq = delx * delx + dely * dely + delz * delz;
                    int jtype = atom_type[j];

                    if (rsq < cutsqi[jtype]) {
                        double r2inv = 1.0 / rsq;
                        double r6inv = r2inv * r2inv * r2inv;
                        double forcelj = r6inv * (lj1i[jtype] * r6inv - lj2i[jtype]);
                        double fpair = factor_lj * forcelj * r2inv;

                        fxtmp += delx * fpair;
                        fytmp += dely * fpair;
                        fztmp += delz * fpair;

                        if (newton_pair || j < nlocal) {
                            f_arr[j].x -= delx * fpair;
                            f_arr[j].y -= dely * fpair;
                            f_arr[j].z -= delz * fpair;
                        }
                    }
                }

                for (auto& [i2, type] : bondlist[i]) {
                    double delx = xtmp - x[i2].x;
                    double dely = ytmp - x[i2].y;
                    double delz = ztmp - x[i2].z;

                    double rsq = delx * delx + dely * dely + delz * delz;
                    double r0sq = r0[type] * r0[type];
                    double rlogarg = 1.0 - rsq / r0sq;

                    if (rlogarg < 0.1) {
                        error->warning(FLERR, "FENE bond too long: {} {} {} {:.8}",
                                       update->ntimestep, atom->tag[i], atom->tag[i2], sqrt(rsq));
                        //                            if (check_error_thr((rlogarg <= -3.0),tid,FLERR,"Bad FENE bond"))
                        //                                return;
                        assert(false);

                        rlogarg = 0.1;
                    }

                    double fbond = -k[type] / rlogarg;

                    // force from LJ term
                    double sr2 = 0.0;
                    double sr6 = 0.0;

                    if (rsq < MathConst::MY_CUBEROOT2 * sigma[type] * sigma[type]) {
                        sr2 = sigma[type] * sigma[type] / rsq;
                        sr6 = sr2 * sr2 * sr2;
                        fbond += 48.0 * epsilon[type] * sr6 * (sr6 - 0.5) / rsq;
                    }

                    // energy

                    // apply force to each of 2 atoms

                    if (newton_pair || i < nlocal) {
                        fxtmp += delx * fbond;
                        fytmp += dely * fbond;
                        fztmp += delz * fbond;
                    }

                    if (newton_pair || i2 < nlocal) {
                        f_arr[i2].x -= delx * fbond;
                        f_arr[i2].y -= dely * fbond;
                        f_arr[i2].z -= delz * fbond;
                    }
                }

                f_arr[i].x += fxtmp;
                f_arr[i].y += fytmp;
                f_arr[i].z += fztmp;
            }
        }

        // do reduction
        #pragma cilk grainsize 512
        cilk_for (int i = 0; i < nall; i++) {
            for (int j = 0; j < NUM_ARRAYS; j++) {
                f[i].x += lammps_f[j * nall + i].x;
                f[i].y += lammps_f[j * nall + i].y;
                f[i].z += lammps_f[j * nall + i].z;
            }
        }
    }

    std::vector<int>* firstneigh_use_atomics;
    std::vector<int>* firstneigh_no_atomics;

    std::vector<std::pair<int, int>>* bondlist_use_atomics;
    std::vector<std::pair<int, int>>* bondlist_no_atomics;

    void lammps_setup_atomic_lists() {
        auto pair = (PairLJCutOMP*) force->pair;
        auto bond = (BondFENE*) force->bond;

        const int * _noalias const ilist = pair->list->ilist;
        const int * _noalias const numneigh = pair->list->numneigh;
        const int * const * const firstneigh = pair->list->firstneigh;

        const int nlocal = atom->nlocal;

        auto& idx_use_atomics = atom->idx_use_atomics;

        /*
        numneigh_use_atomic = new int[nlocal];
        numneigh_no_atomic = new int[nlocal];
        firstneigh_use_atomic = new int*[nlocal];
        firstneigh_no_atomic = new int*[nlocal];
        */

        firstneigh_use_atomics = new std::vector<int>[nlocal];
        firstneigh_no_atomics = new std::vector<int>[nlocal];
        bondlist_use_atomics = new std::vector<std::pair<int, int>>[nlocal];
        bondlist_no_atomics = new std::vector<std::pair<int, int>>[nlocal];

        for (int ii = 0; ii < atom->nlocal; ii++) {
            const int i = ilist[ii];
            assert(i == ii);
            assert(ii >= 0 && ii < nlocal);

            const int *_noalias const jlist = firstneigh[i];
            int jnum = numneigh[i];

            for (int jj = 0; jj < jnum; jj++) {
                int j = jlist[jj];

                if (idx_use_atomics[j]) {
                    firstneigh_use_atomics[ii].push_back(j);
                } else {
                    firstneigh_no_atomics[ii].push_back(j);
                }
            }

            auto& lst_bonds = neighbor->atom_bondlist[ii];
            for (int j = 0; j < lst_bonds.size(); j++) {
                auto &bond_info = lst_bonds[j];
                int i2 = bond_info.first;
                int type = bond_info.second;
                if (atom->idx_use_atomics[i2]) {
                    bondlist_use_atomics[ii].push_back(std::make_pair(i2, type));
                } else {
                    bondlist_no_atomics[ii].push_back(std::make_pair(i2, type));
                }
            }
        }
    }

    void lammps_fuse_force_compute_lammps_bins_split() {
        // auto& bin_bounds = LAMMPS_GET_BOUNDS(true, 0);
        const auto * _noalias const x = (dbl3_t_stencil_md *) atom->x[0];
        auto * _noalias const f = (dbl3_t_stencil_md *) atom->f[0];

        auto pair = (PairLJCutOMP*) force->pair;
        auto bond = (BondFENE*) force->bond;

        const auto* _noalias bondlist = neighbor->atom_bondlist;
        auto& idx_use_atomics = atom->idx_use_atomics;

        const int * _noalias const ilist = pair->list->ilist;
        const int * _noalias const numneigh = pair->list->numneigh;
        const int * const * const firstneigh = pair->list->firstneigh;
        const double * _noalias const special_lj = force->special_lj;

        auto* spinlocks = atom->spinlocks;
        // auto mutexes = atom->mutexes;

        assert(pair->list->inum == atom->nlocal);

        const auto* cutsq = pair->cutsq;
        const auto* offset = pair->offset;
        const auto* lj1 = pair->lj1;
        const auto* lj2 = pair->lj2;
        const auto* lj3 = pair->lj3;
        const auto* lj4 = pair->lj4;
        auto newton_pair = force->newton_pair;

        const auto* _noalias const sigma = bond->sigma;
        const auto* _noalias const epsilon = bond->epsilon;
        const auto* _noalias const r0 = bond->r0;
        const auto* _noalias const k = bond->k;

        const int* _noalias const atom_type = atom->type;

        const int nlocal = atom->nlocal;

        auto& local_bins = atom->local_bins;
        auto& local_bins_idxs = atom->local_bins_idxs;

        int num_atomics_same_bin = 0;
        int num_atomics_diff_bin = 0;
        int num_neighbors_atomics = 0;
        int num_neighbors_no_atomics = 0;

        int num_atomics_within_cutoff = 0;
        int num_no_atomics_within_cutoff = 0;

        for (int b = 0; b < local_bins.size(); b++) {
            auto& bin = local_bins[b];
            auto& idxs = local_bins_idxs[b];

            int start = idxs[0];
            for (int idx = 0; idx < idxs.size(); idx++) {
                int ii = start + idx;
                assert(ii == start + idx);
                const int i = ilist[ii];
                assert(i == ii);
                assert(ii >= 0 && ii < nlocal);
                const int itype = atom_type[i];

                // assert(bin_use_atomics == atom->idx_use_atomics[i]);

                // const int *_noalias const jlist = firstneigh[i];
                const auto& jlist_use_atomics = firstneigh_use_atomics[ii];
                const auto& jlist_no_atomics = firstneigh_no_atomics[ii];

                const auto& lst_bonds_use_atomics = bondlist_use_atomics[i];
                const auto& lst_bonds_no_atomics = bondlist_no_atomics[i];

                num_neighbors_atomics += jlist_use_atomics.size();
                num_neighbors_no_atomics += jlist_no_atomics.size();

                const double *_noalias const cutsqi = cutsq[itype];
                const double *_noalias const offseti = offset[itype];
                const double *_noalias const lj1i = lj1[itype];
                const double *_noalias const lj2i = lj2[itype];
                const double *_noalias const lj3i = lj3[itype];
                const double *_noalias const lj4i = lj4[itype];

                double xtmp = x[i].x;
                double ytmp = x[i].y;
                double ztmp = x[i].z;
                int jnum = numneigh[i];

                double fxtmp = 0.0;
                double fytmp = 0.0;
                double fztmp = 0.0;

                for (int jj = 0; jj < jlist_use_atomics.size(); jj++) {
                    double evdwl = 0.0;
                    int j = jlist_use_atomics[jj];
                    double factor_lj = special_lj[pair->sbmask(j)];
                    j &= NEIGHMASK;

                    double delx = xtmp - x[j].x;
                    double dely = ytmp - x[j].y;
                    double delz = ztmp - x[j].z;
                    double rsq = delx * delx + dely * dely + delz * delz;
                    int jtype = atom_type[j];

                    if (rsq < cutsqi[jtype]) {
                        num_atomics_within_cutoff++;

                        double r2inv = 1.0 / rsq;
                        double r6inv = r2inv * r2inv * r2inv;
                        double forcelj = r6inv * (lj1i[jtype] * r6inv - lj2i[jtype]);
                        double fpair = factor_lj * forcelj * r2inv;

                        fxtmp += delx * fpair;
                        fytmp += dely * fpair;
                        fztmp += delz * fpair;

                        if (newton_pair || j < nlocal) {
                            auto bin_bounds = stencilMD->LAMMPS_GET_BOUNDS(true, 0);
                            auto my_bin = get_bin(bin_bounds, atom->x[i], domain->boxlo, domain->boxhi);
                            auto other_bin = get_bin(bin_bounds, atom->x[j], domain->boxlo, domain->boxhi);
                            if (my_bin != other_bin) {
                                num_atomics_same_bin++;
                            } else {
                                num_atomics_diff_bin++;
                            }
                            assert(idx_use_atomics[j]);
                            spinlocks[j].lock();
                            f[j].x -= delx * fpair;
                            f[j].y -= dely * fpair;
                            f[j].z -= delz * fpair;
                            spinlocks[j].unlock();
                        }
                    }
                }

                for (int jj = 0; jj < jlist_no_atomics.size(); jj++) {
                    double evdwl = 0.0;
                    int j = jlist_no_atomics[jj];
                    double factor_lj = special_lj[pair->sbmask(j)];
                    j &= NEIGHMASK;

                    double delx = xtmp - x[j].x;
                    double dely = ytmp - x[j].y;
                    double delz = ztmp - x[j].z;
                    double rsq = delx * delx + dely * dely + delz * delz;
                    int jtype = atom_type[j];

                    if (rsq < cutsqi[jtype]) {
                        num_no_atomics_within_cutoff++;

                        double r2inv = 1.0 / rsq;
                        double r6inv = r2inv * r2inv * r2inv;
                        double forcelj = r6inv * (lj1i[jtype] * r6inv - lj2i[jtype]);
                        double fpair = factor_lj * forcelj * r2inv;

                        fxtmp += delx * fpair;
                        fytmp += dely * fpair;
                        fztmp += delz * fpair;

                        if (newton_pair || j < nlocal) {
                            assert(!idx_use_atomics[j]);
                            f[j].x -= delx * fpair;
                            f[j].y -= dely * fpair;
                            f[j].z -= delz * fpair;
                        }
                    }
                }

                // auto& lst_bonds = bondlist[i];
                for (auto& [i2, type] : bondlist_use_atomics[i]) {
                    double delx = xtmp - x[i2].x;
                    double dely = ytmp - x[i2].y;
                    double delz = ztmp - x[i2].z;

                    double rsq = delx * delx + dely * dely + delz * delz;
                    double r0sq = r0[type] * r0[type];
                    double rlogarg = 1.0 - rsq / r0sq;

                    if (rlogarg < 0.1) {
                        error->warning(FLERR, "FENE bond too long: {} {} {} {:.8}",
                                       update->ntimestep, atom->tag[i], atom->tag[i2], sqrt(rsq));
                        //                            if (check_error_thr((rlogarg <= -3.0),tid,FLERR,"Bad FENE bond"))
                        //                                return;
                        assert(false);

                        rlogarg = 0.1;
                    }

                    double fbond = -k[type] / rlogarg;

                    // force from LJ term
                    double sr2 = 0.0;
                    double sr6 = 0.0;

                    if (rsq < MathConst::MY_CUBEROOT2 * sigma[type] * sigma[type]) {
                        sr2 = sigma[type] * sigma[type] / rsq;
                        sr6 = sr2 * sr2 * sr2;
                        fbond += 48.0 * epsilon[type] * sr6 * (sr6 - 0.5) / rsq;
                    }

                    // energy

                    // apply force to each of 2 atoms

                    if (newton_pair || i < nlocal) {
                        fxtmp += delx * fbond;
                        fytmp += dely * fbond;
                        fztmp += delz * fbond;
                    }

                    if (newton_pair || i2 < nlocal) {
                        assert(idx_use_atomics[i2]);
                        spinlocks[i2].lock();
                        f[i2].x -= delx * fbond;
                        f[i2].y -= dely * fbond;
                        f[i2].z -= delz * fbond;
                        spinlocks[i2].unlock();
                    }
                }

                for (auto& [i2, type] : bondlist_no_atomics[i]) {
                    double delx = xtmp - x[i2].x;
                    double dely = ytmp - x[i2].y;
                    double delz = ztmp - x[i2].z;

                    double rsq = delx * delx + dely * dely + delz * delz;
                    double r0sq = r0[type] * r0[type];
                    double rlogarg = 1.0 - rsq / r0sq;

                    if (rlogarg < 0.1) {
                        error->warning(FLERR, "FENE bond too long: {} {} {} {:.8}",
                                       update->ntimestep, atom->tag[i], atom->tag[i2], sqrt(rsq));
                        //                            if (check_error_thr((rlogarg <= -3.0),tid,FLERR,"Bad FENE bond"))
                        //                                return;
                        assert(false);

                        rlogarg = 0.1;
                    }

                    double fbond = -k[type] / rlogarg;

                    // force from LJ term
                    double sr2 = 0.0;
                    double sr6 = 0.0;

                    if (rsq < MathConst::MY_CUBEROOT2 * sigma[type] * sigma[type]) {
                        sr2 = sigma[type] * sigma[type] / rsq;
                        sr6 = sr2 * sr2 * sr2;
                        fbond += 48.0 * epsilon[type] * sr6 * (sr6 - 0.5) / rsq;
                    }

                    // energy

                    // apply force to each of 2 atoms

                    if (newton_pair || i < nlocal) {
                        fxtmp += delx * fbond;
                        fytmp += dely * fbond;
                        fztmp += delz * fbond;
                    }

                    if (newton_pair || i2 < nlocal) {
                        assert(!idx_use_atomics[i2]);
                        f[i2].x -= delx * fbond;
                        f[i2].y -= dely * fbond;
                        f[i2].z -= delz * fbond;
                    }
                }

                if (idx_use_atomics[i]) {
                    spinlocks[i].lock();
                    f[i].x += fxtmp;
                    f[i].y += fytmp;
                    f[i].z += fztmp;
                    spinlocks[i].unlock();
                } else {
                    f[i].x += fxtmp;
                    f[i].y += fytmp;
                    f[i].z += fztmp;
                }
            }
        }

        std::cout << "num neighbors atomics within cutoff: " << num_atomics_within_cutoff << " num no atomics within cutoff: " << num_no_atomics_within_cutoff << " num same: " << num_atomics_same_bin << " num diff: " << num_atomics_diff_bin << " num neighbors atomics: " << num_neighbors_atomics << " num neigh no atomics: " << num_neighbors_no_atomics << std::endl;
    }

    void lammps_fuse_force_compute_lammps_bins() {
        // auto& bin_bounds = LAMMPS_GET_BOUNDS(true, 0);
        const auto * _noalias const x = (dbl3_t_stencil_md *) atom->x[0];
        auto * _noalias const f = (dbl3_t_stencil_md *) atom->f[0];

        auto pair = (PairLJCutOMP*) force->pair;
        auto bond = (BondFENE*) force->bond;

        const auto* _noalias bondlist = neighbor->atom_bondlist;
        auto& idx_use_atomics = atom->idx_use_atomics;

        const int * _noalias const ilist = pair->list->ilist;
        const int * _noalias const numneigh = pair->list->numneigh;
        const int * const * const firstneigh = pair->list->firstneigh;
        const double * _noalias const special_lj = force->special_lj;

        auto* spinlocks = atom->spinlocks;
        // auto mutexes = atom->mutexes;

        assert(pair->list->inum == atom->nlocal);

        const auto* cutsq = pair->cutsq;
        const auto* offset = pair->offset;
        const auto* lj1 = pair->lj1;
        const auto* lj2 = pair->lj2;
        const auto* lj3 = pair->lj3;
        const auto* lj4 = pair->lj4;
        auto newton_pair = force->newton_pair;

        const auto* _noalias const sigma = bond->sigma;
        const auto* _noalias const epsilon = bond->epsilon;
        const auto* _noalias const r0 = bond->r0;
        const auto* _noalias const k = bond->k;

        const int* _noalias const atom_type = atom->type;

        const int nlocal = atom->nlocal;

//        std::map<int, std::string> m;
//        m[0] = "PBC";
//        m[1] = "LEFT";
//        m[2] = "MIDDLE";
//        m[3] = "RIGHT";

//        constexpr bool DEBUG_CILK = true;
//        Cilksan_fake_mutex fake_lock;
//        if (DEBUG_CILK) {
//            __cilksan_register_lock_explicit(&fake_lock);
//        }

        auto& local_bins = atom->local_bins;
        auto& local_bins_idxs = atom->local_bins_idxs;

        for (int b = 0; b < local_bins.size(); b++) {
            auto& bin = local_bins[b];
            auto& idxs = local_bins_idxs[b];

            int start = idxs[0];
            for (int idx = 0; idx < idxs.size(); idx++) {
                int ii = start + idx;
                assert(ii == start + idx);
                const int i = ilist[ii];
                assert(i == ii);
                assert(ii >= 0 && ii < nlocal);
                const int itype = atom_type[i];

                // assert(bin_use_atomics == atom->idx_use_atomics[i]);

                const int *_noalias const jlist = firstneigh[i];
                const double *_noalias const cutsqi = cutsq[itype];
                const double *_noalias const offseti = offset[itype];
                const double *_noalias const lj1i = lj1[itype];
                const double *_noalias const lj2i = lj2[itype];
                const double *_noalias const lj3i = lj3[itype];
                const double *_noalias const lj4i = lj4[itype];

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
                    double factor_lj = special_lj[pair->sbmask(j)];
                    j &= NEIGHMASK;

                    double delx = xtmp - x[j].x;
                    double dely = ytmp - x[j].y;
                    double delz = ztmp - x[j].z;
                    double rsq = delx * delx + dely * dely + delz * delz;
                    int jtype = atom_type[j];

                    if (rsq < cutsqi[jtype]) {
                        double r2inv = 1.0 / rsq;
                        double r6inv = r2inv * r2inv * r2inv;
                        double forcelj = r6inv * (lj1i[jtype] * r6inv - lj2i[jtype]);
                        double fpair = factor_lj * forcelj * r2inv;

                        fxtmp += delx * fpair;
                        fytmp += dely * fpair;
                        fztmp += delz * fpair;

                        if (newton_pair || j < nlocal) {
                            if (idx_use_atomics[j]) {
//                                if (DEBUG_CILK) {
//                                    __cilksan_acquire_lock(&fake_lock);
//                                }
                                spinlocks[j].lock();
                                // mutexes[j].lock();
                                f[j].x -= delx * fpair;
                                f[j].y -= dely * fpair;
                                f[j].z -= delz * fpair;
                                spinlocks[j].unlock();
                                // mutexes[j].unlock();
//                                __atomic_fetch_add(&f[j].x, -delx * fpair, __ATOMIC_RELAXED);
//                                __atomic_fetch_add(&f[j].y, -dely * fpair, __ATOMIC_RELAXED);
//                                __atomic_fetch_add(&f[j].z, -delz * fpair, __ATOMIC_RELAXED);

//                                if (DEBUG_CILK) {
//                                    __cilksan_release_lock(&fake_lock);
//                                }
                            } else {
                                f[j].x -= delx * fpair;
                                f[j].y -= dely * fpair;
                                f[j].z -= delz * fpair;
                            }
                        }
                    }
                }

                auto& lst_bonds = bondlist[i];
                for (int j = 0; j < lst_bonds.size(); j++) {
                    auto &bond_info = lst_bonds[j];
                    int i2 = bond_info.first;
                    int type = bond_info.second;

                    double delx = xtmp - x[i2].x;
                    double dely = ytmp - x[i2].y;
                    double delz = ztmp - x[i2].z;

                    double rsq = delx * delx + dely * dely + delz * delz;
                    double r0sq = r0[type] * r0[type];
                    double rlogarg = 1.0 - rsq / r0sq;

                    if (rlogarg < 0.1) {
                        error->warning(FLERR, "FENE bond too long: {} {} {} {:.8}",
                                       update->ntimestep, atom->tag[i], atom->tag[i2], sqrt(rsq));
                        //                            if (check_error_thr((rlogarg <= -3.0),tid,FLERR,"Bad FENE bond"))
                        //                                return;
                        assert(false);

                        rlogarg = 0.1;
                    }

                    double fbond = -k[type] / rlogarg;

                    // force from LJ term
                    double sr2 = 0.0;
                    double sr6 = 0.0;

                    if (rsq < MathConst::MY_CUBEROOT2 * sigma[type] * sigma[type]) {
                        sr2 = sigma[type] * sigma[type] / rsq;
                        sr6 = sr2 * sr2 * sr2;
                        fbond += 48.0 * epsilon[type] * sr6 * (sr6 - 0.5) / rsq;
                    }

                    // energy

                    // apply force to each of 2 atoms

                    if (newton_pair || i < nlocal) {
                        fxtmp += delx * fbond;
                        fytmp += dely * fbond;
                        fztmp += delz * fbond;
                    }

                    if (newton_pair || i2 < nlocal) {
                        if (idx_use_atomics[i2]) {
//                            if (DEBUG_CILK) {
//                                __cilksan_acquire_lock(&fake_lock);
//                            }
                            // __atomic_fetch_add(&f[i2].x, -delx * fbond, __ATOMIC_RELAXED);
                            // __atomic_fetch_add(&f[i2].y, -dely * fbond, __ATOMIC_RELAXED);
                            // __atomic_fetch_add(&f[i2].z, -delz * fbond, __ATOMIC_RELAXED);
                            spinlocks[i2].lock();
                            // mutexes[i2].lock();
                            f[i2].x -= delx * fbond;
                            f[i2].y -= dely * fbond;
                            f[i2].z -= delz * fbond;
                            // mutexes[i2].unlock();
                            spinlocks[i2].unlock();
//                            if (DEBUG_CILK) {
//                                __cilksan_release_lock(&fake_lock);
//                            }
                        } else {
                            f[i2].x -= delx * fbond;
                            f[i2].y -= dely * fbond;
                            f[i2].z -= delz * fbond;
                        }
                    }
                }

                if (idx_use_atomics[i]) {
//                    if (DEBUG_CILK) {
//                        __cilksan_acquire_lock(&fake_lock);
//                    }
                    /*
                    __atomic_fetch_add(&f[i].x, fxtmp, __ATOMIC_RELAXED);
                    __atomic_fetch_add(&f[i].y, fytmp, __ATOMIC_RELAXED);
                    __atomic_fetch_add(&f[i].z, fztmp, __ATOMIC_RELAXED);
                    */
                    spinlocks[i].lock();
                    // mutexes[i].lock();
                    f[i].x += fxtmp;
                    f[i].y += fytmp;
                    f[i].z += fztmp;
                    spinlocks[i].unlock();
                    // mutexes[i].unlock();
//                    if (DEBUG_CILK) {
//                        __cilksan_release_lock(&fake_lock);
//                    }
                } else {
                    f[i].x += fxtmp;
                    f[i].y += fytmp;
                    f[i].z += fztmp;
                }
            }

            // std::cout << "bin: " << m[bin[0]] << " " << m[bin[1]] << " " << m[bin[2]] << " num neighbors: " << num_neighbors << " num in bounds: " << num_in_bounds << " num in bin: " << idxs.size() << " num atomics: " << num_atomics << std::endl;
        }

//        if (DEBUG_CILK) {
//            __cilksan_unregister_lock_explicit(&fake_lock);
//        }
    }

    // Begin methods used for fusing
    // typedef struct { double x,y,z; } dbl3_t;
    void lammps_fuse_force_compute2() {
        const auto * _noalias const x = (dbl3_t_stencil_md *) atom->x[0];
        auto * _noalias const f = (dbl3_t_stencil_md *) atom->f[0];

        auto pair = (PairLJCutOMP*) force->pair;
        auto bond = (BondFENE*) force->bond;

        const int * _noalias const ilist = pair->list->ilist;
        const int * _noalias const numneigh = pair->list->numneigh;
        const int * const * const firstneigh = pair->list->firstneigh;
        const double * _noalias const special_lj = force->special_lj;

        assert(pair->list->inum == atom->nlocal);

        const auto* cutsq = pair->cutsq;
        const auto* offset = pair->offset;
        const auto* lj1 = pair->lj1;
        const auto* lj2 = pair->lj2;
        const auto* lj3 = pair->lj3;
        const auto* lj4 = pair->lj4;
        auto newton_pair = force->newton_pair;

        const auto* sigma = bond->sigma;
        const auto* epsilon = bond->epsilon;
        const auto* r0 = bond->r0;
        const auto* k = bond->k;

        int* atom_type = atom->type;

        int nlocal = atom->nlocal;

        /*
        std::map<int, std::string> m;
        m[LEFT] = "LEFT";
        m[RIGHT] = "RIGHT";
        m[MIDDLE] = "MIDDLE";
        m[PBC] = "PBC";

        constexpr bool DEBUG_CILK = false;
        Cilksan_fake_mutex fake_lock;
        if (DEBUG_CILK) {
            __cilksan_register_lock_explicit(&fake_lock);
        }
        */

        for (int dep = 0; dep < NUM_DEPS_BINS; dep++) {
            auto &partitions_at_dep = atom->dep_to_partitions[dep];
            for (int p = 0; p < partitions_at_dep.size(); p++) {
                auto &partition = partitions_at_dep[p];
                auto &bins = atom->partition_to_bins[partition[0]][partition[1]][partition[2]];
                for (int b = 0; b < bins.size(); b++) {
                    auto &bin = bins[b];
                    // bool bin_use_atomics = atom->bin_to_use_atomic[bin];
                    // bool bin_use_atomics = atom->bin_to_use_atomic[bin[0]][bin[1]][bin[2]];
                    // auto &idxs = atom->bin_to_local_idxs[bin];
                    auto &idxs = atom->bin_to_local_idxs2[bin[0]][bin[1]][bin[2]];
                    int start = idxs[0];

                    for (int idx = 0; idx < idxs.size(); idx++) {
                        int ii = start + idx;
                        assert(ii == start + idx);
                        const int i = ilist[ii];
                        assert(i == ii);
                        assert(ii >= 0 && ii < nlocal);
                        const int itype = atom_type[i];

                        // assert(bin_use_atomics == atom->idx_use_atomics[i]);

                        const int *_noalias const jlist = firstneigh[i];
                        const double *_noalias const cutsqi = cutsq[itype];
                        const double *_noalias const offseti = offset[itype];
                        const double *_noalias const lj1i = lj1[itype];
                        const double *_noalias const lj2i = lj2[itype];
                        const double *_noalias const lj3i = lj3[itype];
                        const double *_noalias const lj4i = lj4[itype];

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
                            double factor_lj = special_lj[pair->sbmask(j)];
                            j &= NEIGHMASK;

                            double delx = xtmp - x[j].x;
                            double dely = ytmp - x[j].y;
                            double delz = ztmp - x[j].z;
                            double rsq = delx * delx + dely * dely + delz * delz;
                            int jtype = atom_type[j];

                            if (rsq < cutsqi[jtype]) {
                                double r2inv = 1.0 / rsq;
                                double r6inv = r2inv * r2inv * r2inv;
                                double forcelj = r6inv * (lj1i[jtype] * r6inv - lj2i[jtype]);
                                double fpair = factor_lj * forcelj * r2inv;

                                fxtmp += delx * fpair;
                                fytmp += dely * fpair;
                                fztmp += delz * fpair;

                                if (newton_pair || j < nlocal) {
                                    if (atom->idx_use_atomics[j]) {
                                        /*
                                        if (DEBUG_CILK) {
                                            __cilksan_acquire_lock(&fake_lock);
                                        }
                                        */
                                        __atomic_fetch_add(&f[j].x, -delx * fpair, __ATOMIC_RELAXED);
                                        __atomic_fetch_add(&f[j].y, -dely * fpair, __ATOMIC_RELAXED);
                                        __atomic_fetch_add(&f[j].z, -delz * fpair, __ATOMIC_RELAXED);
                                        /*
                                        if (DEBUG_CILK) {
                                            __cilksan_release_lock(&fake_lock);
                                        }
                                        */
                                    } else {
                                        f[j].x -= delx * fpair;
                                        f[j].y -= dely * fpair;
                                        f[j].z -= delz * fpair;
                                    }
                                }
                            }
                        }

                        auto &lst_bonds = neighbor->atom_bondlist[i];
                        for (int j = 0; j < lst_bonds.size(); j++) {
                            auto &bond_info = lst_bonds[j];
                            int i2 = bond_info.first;
                            int type = bond_info.second;

                            double delx = xtmp - x[i2].x;
                            double dely = ytmp - x[i2].y;
                            double delz = ztmp - x[i2].z;

                            double rsq = delx * delx + dely * dely + delz * delz;
                            double r0sq = r0[type] * r0[type];
                            double rlogarg = 1.0 - rsq / r0sq;

                            if (rlogarg < 0.1) {
                                error->warning(FLERR, "FENE bond too long: {} {} {} {:.8}",
                                               update->ntimestep, atom->tag[i], atom->tag[i2], sqrt(rsq));
                                //                            if (check_error_thr((rlogarg <= -3.0),tid,FLERR,"Bad FENE bond"))
                                //                                return;
                                assert(false);

                                rlogarg = 0.1;
                            }

                            double fbond = -k[type] / rlogarg;

                            // force from LJ term
                            double sr2 = 0.0;
                            double sr6 = 0.0;

                            if (rsq < MathConst::MY_CUBEROOT2 * sigma[type] * sigma[type]) {
                                sr2 = sigma[type] * sigma[type] / rsq;
                                sr6 = sr2 * sr2 * sr2;
                                fbond += 48.0 * epsilon[type] * sr6 * (sr6 - 0.5) / rsq;
                            }

                            // energy

                            // apply force to each of 2 atoms

                            if (newton_pair || i < nlocal) {
                                fxtmp += delx * fbond;
                                fytmp += dely * fbond;
                                fztmp += delz * fbond;
                            }

                            if (newton_pair || i2 < nlocal) {
                                if (atom->idx_use_atomics[i2]) {
                                    /*
                                    if (DEBUG_CILK) {
                                        __cilksan_acquire_lock(&fake_lock);
                                    }
                                    */
                                    __atomic_fetch_add(&f[i2].x, -delx * fbond, __ATOMIC_RELAXED);
                                    __atomic_fetch_add(&f[i2].y, -dely * fbond, __ATOMIC_RELAXED);
                                    __atomic_fetch_add(&f[i2].z, -delz * fbond, __ATOMIC_RELAXED);
                                    /*
                                    if (DEBUG_CILK) {
                                        __cilksan_release_lock(&fake_lock);
                                    }
                                    */
                                } else {
                                    f[i2].x -= delx * fbond;
                                    f[i2].y -= dely * fbond;
                                    f[i2].z -= delz * fbond;
                                }
                            }
                        }

                        // if (bin_use_atomics) {
                        if (atom->idx_use_atomics[i]) {
                            /*
                            if (DEBUG_CILK) {
                                __cilksan_acquire_lock(&fake_lock);
                            }
                            */
                            __atomic_fetch_add(&f[i].x, fxtmp, __ATOMIC_RELAXED);
                            __atomic_fetch_add(&f[i].y, fytmp, __ATOMIC_RELAXED);
                            __atomic_fetch_add(&f[i].z, fztmp, __ATOMIC_RELAXED);
                            /*
                            if (DEBUG_CILK) {
                                __cilksan_release_lock(&fake_lock);
                            }
                            */
                        } else {
                            f[i].x += fxtmp;
                            f[i].y += fytmp;
                            f[i].z += fztmp;
                        }
                    }
                }
            }
        }

        /*
        if (DEBUG_CILK) {
            __cilksan_unregister_lock_explicit(&fake_lock);
        }
        */
    }

    void lammps_fuse_force_compute_atomics() {
        const auto * _noalias const x = (dbl3_t_stencil_md *) atom->x[0];
        auto * _noalias const f = (dbl3_t_stencil_md *) atom->f[0];

        auto pair = (PairLJCutOMP*) force->pair;
        auto bond = (BondFENE*) force->bond;

        const int * _noalias const ilist = pair->list->ilist;
        const int * _noalias const numneigh = pair->list->numneigh;
        const int * const * const firstneigh = pair->list->firstneigh;
        const double * _noalias const special_lj = force->special_lj;

        assert(pair->list->inum == atom->nlocal);

        const auto* cutsq = pair->cutsq;
        const auto* offset = pair->offset;
        const auto* lj1 = pair->lj1;
        const auto* lj2 = pair->lj2;
        const auto* lj3 = pair->lj3;
        const auto* lj4 = pair->lj4;
        auto newton_pair = force->newton_pair;

        const auto* sigma = bond->sigma;
        const auto* epsilon = bond->epsilon;
        const auto* r0 = bond->r0;
        const auto* k = bond->k;

        int* atom_type = atom->type;

        int nlocal = atom->nlocal;

        cilk_for (int i = 0; i < nlocal; i++) {
            const int itype = atom_type[i];

            const int *_noalias const jlist = firstneigh[i];
            const double *_noalias const cutsqi = cutsq[itype];
            const double *_noalias const offseti = offset[itype];
            const double *_noalias const lj1i = lj1[itype];
            const double *_noalias const lj2i = lj2[itype];
            const double *_noalias const lj3i = lj3[itype];
            const double *_noalias const lj4i = lj4[itype];

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
                double factor_lj = special_lj[pair->sbmask(j)];
                j &= NEIGHMASK;

                double delx = xtmp - x[j].x;
                double dely = ytmp - x[j].y;
                double delz = ztmp - x[j].z;
                double rsq = delx * delx + dely * dely + delz * delz;
                int jtype = atom_type[j];

                if (rsq < cutsqi[jtype]) {
                    double r2inv = 1.0 / rsq;
                    double r6inv = r2inv * r2inv * r2inv;
                    double forcelj = r6inv * (lj1i[jtype] * r6inv - lj2i[jtype]);
                    double fpair = factor_lj * forcelj * r2inv;

                    fxtmp += delx * fpair;
                    fytmp += dely * fpair;
                    fztmp += delz * fpair;

                    if (newton_pair || j < nlocal) {
                        __atomic_fetch_add(&f[j].x, -delx * fpair, __ATOMIC_RELAXED);
                        __atomic_fetch_add(&f[j].y, -dely * fpair, __ATOMIC_RELAXED);
                        __atomic_fetch_add(&f[j].z, -delz * fpair, __ATOMIC_RELAXED);
                    }
                }
            }

            auto &lst_bonds = neighbor->atom_bondlist[i];
            for (int j = 0; j < lst_bonds.size(); j++) {
                auto &bond_info = lst_bonds[j];
                int i2 = bond_info.first;
                int type = bond_info.second;

                double delx = xtmp - x[i2].x;
                double dely = ytmp - x[i2].y;
                double delz = ztmp - x[i2].z;

                double rsq = delx * delx + dely * dely + delz * delz;
                double r0sq = r0[type] * r0[type];
                double rlogarg = 1.0 - rsq / r0sq;

                if (rlogarg < 0.1) {
                    error->warning(FLERR, "FENE bond too long: {} {} {} {:.8}",
                                   update->ntimestep, atom->tag[i], atom->tag[i2], sqrt(rsq));
                    //                            if (check_error_thr((rlogarg <= -3.0),tid,FLERR,"Bad FENE bond"))
                    //                                return;
                    assert(false);

                    rlogarg = 0.1;
                }

                double fbond = -k[type] / rlogarg;

                // force from LJ term
                double sr2 = 0.0;
                double sr6 = 0.0;

                if (rsq < MathConst::MY_CUBEROOT2 * sigma[type] * sigma[type]) {
                    sr2 = sigma[type] * sigma[type] / rsq;
                    sr6 = sr2 * sr2 * sr2;
                    fbond += 48.0 * epsilon[type] * sr6 * (sr6 - 0.5) / rsq;
                }

                // energy

                // apply force to each of 2 atoms

                if (newton_pair || i < nlocal) {
                    fxtmp += delx * fbond;
                    fytmp += dely * fbond;
                    fztmp += delz * fbond;
                }

                if (newton_pair || i2 < nlocal) {
                    __atomic_fetch_add(&f[i2].x, -delx * fbond, __ATOMIC_RELAXED);
                    __atomic_fetch_add(&f[i2].y, -dely * fbond, __ATOMIC_RELAXED);
                    __atomic_fetch_add(&f[i2].z, -delz * fbond, __ATOMIC_RELAXED);
                }
            }

            __atomic_fetch_add(&f[i].x, fxtmp, __ATOMIC_RELAXED);
            __atomic_fetch_add(&f[i].y, fytmp, __ATOMIC_RELAXED);
            __atomic_fetch_add(&f[i].z, fztmp, __ATOMIC_RELAXED);
        }
    }

    void lammps_fuse_force_compute() {
        const auto * _noalias const x = (dbl3_t_stencil_md *) atom->x[0];
        auto * _noalias const f = (dbl3_t_stencil_md *) atom->f[0];

        auto pair = (PairLJCutOMP*) force->pair;
        auto bond = (BondFENE*) force->bond;

        const int * _noalias const ilist = pair->list->ilist;
        const int * _noalias const numneigh = pair->list->numneigh;
        const int * const * const firstneigh = pair->list->firstneigh;
        const double * _noalias const special_lj = force->special_lj;

        assert(pair->list->inum == atom->nlocal);

        const auto* cutsq = pair->cutsq;
        const auto* offset = pair->offset;
        const auto* lj1 = pair->lj1;
        const auto* lj2 = pair->lj2;
        const auto* lj3 = pair->lj3;
        const auto* lj4 = pair->lj4;
        auto newton_pair = force->newton_pair;

        const auto* sigma = bond->sigma;
        const auto* epsilon = bond->epsilon;
        const auto* r0 = bond->r0;
        const auto* k = bond->k;

        int* atom_type = atom->type;

        int nlocal = atom->nlocal;

        std::map<int, std::string> m;
        m[LEFT] = "LEFT";
        m[RIGHT] = "RIGHT";
        m[MIDDLE] = "MIDDLE";
        m[PBC] = "PBC";

        for (int dep = 0; dep < NUM_DEPS_BINS; dep++) {
            auto& partitions_at_dep = atom->dep_to_partitions[dep];

            // int num_atoms = 0;
            cilk_for (int d = 0; d < partitions_at_dep.size(); d++) {
                auto& partition = partitions_at_dep[d];

                // for (int dep_level2 = 0; dep_level2 < dep + 1; dep_level2++) {
                for (int dep_level2 = 0; dep_level2 < atom->num_deps_level2[partition[0]][partition[1]][partition[2]]; dep_level2++) {
                    // auto& partitions_at_dep_level2 = atom->dep_to_partitions_level2[dep_level2];
                    auto& partitions_at_dep_level2 = atom->dep_to_partitions_level2[partition[0]][partition[1]][partition[2]][dep_level2];

                    cilk_for (int d2 = 0; d2 < partitions_at_dep_level2.size(); d2++) {
                        auto& partition_level2 = partitions_at_dep_level2[d2];
                        auto& bins = atom->partition_to_bins_level2[partition[0]][partition[1]][partition[2]][partition_level2[0]][partition_level2[1]][partition_level2[2]];

                        for (int b = 0; b < bins.size(); b++) {
                            auto& bin = bins[b];
                            const auto& idxs = atom->bin_to_local_idxs[bin];
                            int start = idxs[0];
                            for (int idx = 0; idx < idxs.size(); idx++) {
                                // int ii = idxs[idx];
                                int ii = start + idx;
                                assert(ii == start + idx);
                                const int i = ilist[ii];
                                assert(i == ii);
                                assert(ii >= 0 && ii < nlocal);
                                const int itype = atom_type[i];

                                const int *_noalias const jlist = firstneigh[i];
                                const double *_noalias const cutsqi = cutsq[itype];
                                const double *_noalias const offseti = offset[itype];
                                const double *_noalias const lj1i = lj1[itype];
                                const double *_noalias const lj2i = lj2[itype];
                                const double *_noalias const lj3i = lj3[itype];
                                const double *_noalias const lj4i = lj4[itype];

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
                                    double factor_lj = special_lj[pair->sbmask(j)];
                                    j &= NEIGHMASK;

                                    double delx = xtmp - x[j].x;
                                    double dely = ytmp - x[j].y;
                                    double delz = ztmp - x[j].z;
                                    double rsq = delx * delx + dely * dely + delz * delz;
                                    int jtype = atom_type[j];

                                    if (rsq < cutsqi[jtype]) {
                                        double r2inv = 1.0 / rsq;
                                        double r6inv = r2inv * r2inv * r2inv;
                                        double forcelj = r6inv * (lj1i[jtype] * r6inv - lj2i[jtype]);
                                        double fpair = factor_lj * forcelj * r2inv;

                                        fxtmp += delx * fpair;
                                        fytmp += dely * fpair;
                                        fztmp += delz * fpair;

                                        if (newton_pair || j < nlocal) {
                                            f[j].x -= delx * fpair;
                                            f[j].y -= dely * fpair;
                                            f[j].z -= delz * fpair;
                                        }
                                    }
                                }

                                auto& lst_bonds = neighbor->atom_bondlist[i];
                                for (int j = 0; j < lst_bonds.size(); j++) {
                                    auto& bond_info = lst_bonds[j];
                                    int i2 = bond_info.first;
                                    int type = bond_info.second;

                                    double delx = xtmp - x[i2].x;
                                    double dely = ytmp - x[i2].y;
                                    double delz = ztmp - x[i2].z;

                                    double rsq = delx * delx + dely * dely + delz * delz;
                                    double r0sq = r0[type] * r0[type];
                                    double rlogarg = 1.0 - rsq / r0sq;

                                    if (rlogarg < 0.1) {
                                        std::cout << "dep: " << dep << " dep2: " << dep_level2 << " partition level 1: " << m[partition[0]] << " " << m[partition[1]] << " " << m[partition[2]]
                                                  << " partition level 2: " << m[partition_level2[0]] << " " << m[partition_level2[1]] << " " << m[partition_level2[2]] << std::endl;
                                        std::cout << "i: " << i << " other idx: " << i2 << std::endl;
                                        std::cout << "rsq: " << rsq << std::endl;
                                        std::cout << "bin: " << bin[0] << " " << bin[1] << " " << bin[2] << std::endl;
                                        error->warning(FLERR, "FENE bond too long: {} {} {} {:.8}",
                                                       update->ntimestep, atom->tag[i], atom->tag[i2], sqrt(rsq));
//                            if (check_error_thr((rlogarg <= -3.0),tid,FLERR,"Bad FENE bond"))
//                                return;
                                        assert(false);

                                        rlogarg = 0.1;
                                    }

                                    double fbond = -k[type] / rlogarg;

                                    // force from LJ term
                                    double sr2 = 0.0;
                                    double sr6 = 0.0;

                                    if (rsq < MathConst::MY_CUBEROOT2 * sigma[type] * sigma[type]) {
                                        sr2 = sigma[type] * sigma[type] / rsq;
                                        sr6 = sr2 * sr2 * sr2;
                                        fbond += 48.0 * epsilon[type] * sr6 * (sr6 - 0.5) / rsq;
                                    }

                                    // energy

                                    // apply force to each of 2 atoms

                                    if (newton_pair || i < nlocal) {
                                        fxtmp += delx * fbond;
                                        fytmp += dely * fbond;
                                        fztmp += delz * fbond;
                                    }

                                    if (newton_pair || i2 < nlocal) {
                                        f[i2].x -= delx * fbond;
                                        f[i2].y -= dely * fbond;
                                        f[i2].z -= delz * fbond;
                                    }
                                }

                                f[i].x += fxtmp;
                                f[i].y += fytmp;
                                f[i].z += fztmp;
                            }
                        }
                    }
                }

                /*
                auto& bins = atom->partition_to_bins[partition[0]][partition[1]][partition[2]];

                for (int b = 0; b < bins.size(); b++) {
                    auto& bin = bins[b];
                    const auto& idxs = atom->bin_to_local_idxs[bin];
                    int start = idxs[0];
                    for (int idx = 0; idx < idxs.size(); idx++) {
                        // int ii = idxs[idx];
                        int ii = start + idx;
                        assert(ii == start + idx);
                        const int i = ilist[ii];
                        assert(i == ii);
                        assert(ii >= 0 && ii < nlocal);
                        const int itype = atom_type[i];

                        const int *_noalias const jlist = firstneigh[i];
                        const double *_noalias const cutsqi = cutsq[itype];
                        const double *_noalias const offseti = offset[itype];
                        const double *_noalias const lj1i = lj1[itype];
                        const double *_noalias const lj2i = lj2[itype];
                        const double *_noalias const lj3i = lj3[itype];
                        const double *_noalias const lj4i = lj4[itype];

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
                            double factor_lj = special_lj[pair->sbmask(j)];
                            j &= NEIGHMASK;

                            double delx = xtmp - x[j].x;
                            double dely = ytmp - x[j].y;
                            double delz = ztmp - x[j].z;
                            double rsq = delx * delx + dely * dely + delz * delz;
                            int jtype = atom_type[j];

                            if (rsq < cutsqi[jtype]) {
                                double r2inv = 1.0 / rsq;
                                double r6inv = r2inv * r2inv * r2inv;
                                double forcelj = r6inv * (lj1i[jtype] * r6inv - lj2i[jtype]);
                                double fpair = factor_lj * forcelj * r2inv;

                                fxtmp += delx * fpair;
                                fytmp += dely * fpair;
                                fztmp += delz * fpair;

                                if (newton_pair || j < nlocal) {
                                    f[j].x -= delx * fpair;
                                    f[j].y -= dely * fpair;
                                    f[j].z -= delz * fpair;
                                }
                            }
                        }

                        auto& lst_bonds = neighbor->atom_bondlist[i];
                        for (int j = 0; j < lst_bonds.size(); j++) {
                            auto& bond_info = lst_bonds[j];
                            int i2 = bond_info.first;
                            int type = bond_info.second;

                            double delx = xtmp - x[i2].x;
                            double dely = ytmp - x[i2].y;
                            double delz = ztmp - x[i2].z;

                            double rsq = delx * delx + dely * dely + delz * delz;
                            double r0sq = r0[type] * r0[type];
                            double rlogarg = 1.0 - rsq / r0sq;

                            if (rlogarg < 0.1) {
                                error->warning(FLERR, "FENE bond too long: {} {} {} {:.8}",
                                               update->ntimestep, atom->tag[i], atom->tag[i2], sqrt(rsq));
//                            if (check_error_thr((rlogarg <= -3.0),tid,FLERR,"Bad FENE bond"))
//                                return;
                                assert(false);

                                rlogarg = 0.1;
                            }

                            double fbond = -k[type] / rlogarg;

                            // force from LJ term
                            double sr2 = 0.0;
                            double sr6 = 0.0;

                            if (rsq < MathConst::MY_CUBEROOT2 * sigma[type] * sigma[type]) {
                                sr2 = sigma[type] * sigma[type] / rsq;
                                sr6 = sr2 * sr2 * sr2;
                                fbond += 48.0 * epsilon[type] * sr6 * (sr6 - 0.5) / rsq;
                            }

                            // energy

                            // apply force to each of 2 atoms

                            if (newton_pair || i < nlocal) {
                                fxtmp += delx * fbond;
                                fytmp += dely * fbond;
                                fztmp += delz * fbond;
                            }

                            if (newton_pair || i2 < nlocal) {
                                f[i2].x -= delx * fbond;
                                f[i2].y -= dely * fbond;
                                f[i2].z -= delz * fbond;
                            }
                        }

                        f[i].x += fxtmp;
                        f[i].y += fytmp;
                        f[i].z += fztmp;
                    }
                }
                */
            }
        }
    }

    inline void final_integrate_stencil_md_(Atom* next) {
        auto * _noalias const next_v = (dbl3_t *) next->v[0];

        const auto * _noalias const f = (dbl3_t *) next->f[0];
        const auto * _noalias const eval_f = (dbl3_t *) next->eval_f_stencil_md[0];

        const int * const mask = next->mask;
        const int next_nlocal = next->nlocal;

        const double * const mass = atom->mass;
        const int * const type = next->type;
        auto& local_dtfm = next->local_dtfm;

        #pragma cilk grainsize 2048
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

    /*
     *
     * cilk_for (int ii = 0; ii < num_chunks; ++ii) {
        int start_chunk = __cilkrts_get_worker_number() * num_chunks / num_workers;
        for (int c = 0; c < num_chunks; ++c) {
            int s = (c + start_chunk) % num_chunks;
	    if (claimed[s].load()) {
		continue;
	    }
	    bool expected = false;
	    if (claimed[s].compare_exchange_weak(expected, true)) {
		for (int i = s * GRAINSIZE; i < (s + 1) * GRAINSIZE && i < N; i++) {
	            const double dtfm = local_dtfm[i];
        	    v[i].x += dtfm * (f[i].x + eval_f[i].x);
                    v[i].y += dtfm * (f[i].y + eval_f[i].y);
        	    v[i].z += dtfm * (f[i].z + eval_f[i].z);
		}
	    }
	}
    }
     */

    inline void fuse_post_force_final_integrate_stencil_md(Atom* next, Modify* modify_) {
        auto * _noalias const next_v = (dbl3_t *) next->v[0];

        const auto * _noalias const f = (dbl3_t *) next->f[0];
        auto * _noalias const eval_f = (dbl3_t *) next->eval_f_stencil_md[0];

        const int * const mask = next->mask;
        const int next_nlocal = next->nlocal;

        const double * const mass = atom->mass;
        const int * const type = next->type;
        auto& local_dtfm = next->local_dtfm;
        auto claimed = next->claimed;
        auto claimed_int = next->claimed_int;
        auto claimed_flag = next->claimed_flag;

        auto fix_post_force = (FixLangevin*) modify->fix[modify->list_post_force[0]];

        auto gfactor1 = fix_post_force->gfactor1;
        auto gfactor2 = fix_post_force->gfactor2;
        // fix_post_force->compute_target();
        auto tsqrt = fix_post_force->tsqrt;

        #pragma cilk grainsize 2048
        cilk_for (int i = 0; i < next_nlocal; i++) {
            const double dtfm = local_dtfm[i];
            double gamma1 = gfactor1[type[i]];
            double gamma2 = gfactor2[type[i]] * tsqrt;

            double rand_x = 0.6;
            double rand_y = 0.6;
            double rand_z = 0.6;

            // dbl3_t_stencil_md fran = {gamma2 * (rand_x - 0.5), gamma2*(rand_y - 0.5), gamma2 * (rand_z - 0.5)};

            double v_x = next_v[i].x;
            double v_y = next_v[i].y;
            double v_z = next_v[i].z;

            // dbl3_t_stencil_md fdrag = {gamma1 * v_x, gamma1 * v_y, gamma1 * v_z};

            // double f_x = eval_f[i].x + gamma1 * v_x + fran.x;
            // double f_y = eval_f[i].y + gamma1 * v_y + fran.y;
            // double f_z = eval_f[i].z + gamma1 * v_z + fran.z;

            eval_f[i].x += gamma1 * v_x + gamma2 * (rand_x - 0.5);
            eval_f[i].y += gamma1 * v_y + gamma2 * (rand_x - 0.5);
            eval_f[i].z += gamma1 * v_z + gamma2 * (rand_x - 0.5);

            next_v[i].x = v_x + dtfm * (f[i].x + eval_f[i].x);
            next_v[i].y = v_y + dtfm * (f[i].y + eval_f[i].y);
            next_v[i].z = v_z + dtfm * (f[i].z + eval_f[i].z);

            // next_v[i].x = v_x + dtfm * (f[i].x + f_x);
            // next_v[i].y = v_y + dtfm * (f[i].y + f_y);
            // next_v[i].z = v_z + dtfm * (f[i].z + f_z);
        }

        return;
    }

    inline void fuse_post_force_final_integrate_stencil_md_affinity(Atom* next, Modify* modify_) {
        auto * _noalias const next_v = (dbl3_t *) next->v[0];

        const auto * _noalias const f = (dbl3_t *) next->f[0];
        auto * _noalias const eval_f = (dbl3_t *) next->eval_f_stencil_md[0];

        const int * const mask = next->mask;
        const int next_nlocal = next->nlocal;

        const double * const mass = atom->mass;
        const int * const type = next->type;
        auto& local_dtfm = next->local_dtfm;
        auto claimed = next->claimed;
        auto claimed_int = next->claimed_int;
        auto claimed_flag = next->claimed_flag;

        int num_chunks = next_nlocal / MODIFY_GRAINSIZE + 1;
        int num_workers = __cilkrts_get_nworkers();
        num_chunks = num_workers;
        int chunk_size = next_nlocal / num_chunks + 1;
        // int chunk_size = MODIFY_GRAINSIZE;

        auto fix_post_force = (FixLangevin*) modify->fix[modify->list_post_force[0]];

        auto gfactor1 = fix_post_force->gfactor1;
        auto gfactor2 = fix_post_force->gfactor2;
        // fix_post_force->compute_target();
        auto tsqrt = fix_post_force->tsqrt;

        if (num_chunks == 1) {
            for (int i = 0; i < next_nlocal; i++) {
                const double dtfm = local_dtfm[i];
                double gamma1 = gfactor1[type[i]];
                double gamma2 = gfactor2[type[i]] * tsqrt;

                double rand_x = 0.6;
                double rand_y = 0.6;
                double rand_z = 0.6;

                // dbl3_t_stencil_md fran = {gamma2 * (rand_x - 0.5), gamma2*(rand_y - 0.5), gamma2 * (rand_z - 0.5)};

                double v_x = next_v[i].x;
                double v_y = next_v[i].y;
                double v_z = next_v[i].z;

                // dbl3_t_stencil_md fdrag = {gamma1 * v_x, gamma1 * v_y, gamma1 * v_z};

                // double f_x = eval_f[i].x + gamma1 * v_x + fran.x;
                // double f_y = eval_f[i].y + gamma1 * v_y + fran.y;
                // double f_z = eval_f[i].z + gamma1 * v_z + fran.z;

                eval_f[i].x += gamma1 * v_x + gamma2 * (rand_x - 0.5);
                eval_f[i].y += gamma1 * v_y + gamma2 * (rand_x - 0.5);
                eval_f[i].z += gamma1 * v_z + gamma2 * (rand_x - 0.5);

                next_v[i].x = v_x + dtfm * (f[i].x + eval_f[i].x);
                next_v[i].y = v_y + dtfm * (f[i].y + eval_f[i].y);
                next_v[i].z = v_z + dtfm * (f[i].z + eval_f[i].z);

                // next_v[i].x = v_x + dtfm * (f[i].x + f_x);
                // next_v[i].y = v_y + dtfm * (f[i].y + f_y);
                // next_v[i].z = v_z + dtfm * (f[i].z + f_z);
            }

            return;
        }

        #pragma cilk grainsize 1
        cilk_for (int ii = 0; ii < num_chunks; ii++) {
            int start_chunk = __cilkrts_get_worker_number() * num_chunks / num_workers;
            for (int c = 0; c < num_chunks; ++c) {
                int s = (c + start_chunk) % num_chunks;

                if (claimed_flag[s].test(std::memory_order_relaxed)) {
                    continue;
                }
                if (!claimed_flag[s].test_and_set(std::memory_order_relaxed)) {
                    // for (int i = s * MODIFY_GRAINSIZE; i < (s + 1) * MODIFY_GRAINSIZE && i < next_nlocal; i++) {
                    for (int i = s * chunk_size; i < (s + 1) * chunk_size && i < next_nlocal; i++) {
                        const double dtfm = local_dtfm[i];
                        double gamma1 = gfactor1[type[i]];
                        double gamma2 = gfactor2[type[i]] * tsqrt;

                        double rand_x = 0.6;
                        double rand_y = 0.6;
                        double rand_z = 0.6;

                        double v_x = next_v[i].x;
                        double v_y = next_v[i].y;
                        double v_z = next_v[i].z;

                        // dbl3_t_stencil_md fran = {gamma2 * (rand_x - 0.5), gamma2*(rand_y - 0.5), gamma2 * (rand_z - 0.5)};
                        // dbl3_t_stencil_md fdrag = {gamma1 * next_v[i].x, gamma1 * next_v[i].y, gamma1 * next_v[i].z};

                        // eval_f[i].x += fdrag.x + fran.x;
                        // eval_f[i].y += fdrag.y + fran.y;
                        // eval_f[i].z += fdrag.z + fran.z;

                        eval_f[i].x += gamma1 * v_x + gamma2 * (rand_x - 0.5);
                        eval_f[i].y += gamma1 * v_y + gamma2 * (rand_x - 0.5);
                        eval_f[i].z += gamma1 * v_z + gamma2 * (rand_x - 0.5);

                        // next_v[i].x += dtfm * (f[i].x + eval_f[i].x);
                        // next_v[i].y += dtfm * (f[i].y + eval_f[i].y);
                        // next_v[i].z += dtfm * (f[i].z + eval_f[i].z);

                        next_v[i].x = v_x + dtfm * (f[i].x + eval_f[i].x);
                        next_v[i].y = v_y + dtfm * (f[i].y + eval_f[i].y);
                        next_v[i].z = v_z + dtfm * (f[i].z + eval_f[i].z);
                    }
                }
            }
        }

        for (int i = 0; i < num_chunks; i++) {
            claimed_flag[i].clear(std::memory_order_relaxed);
        }
    }

    inline void final_integrate_stencil_md(const std::vector<int>& local_idxs, Atom* next) {
        // update v of atoms in group

        // auto * _noalias const v = (dbl3_t_stencil_md *) atom_->v[0];
        auto * _noalias const next_v = (dbl3_t_stencil_md *) next->v[0];

        const auto * _noalias const f = (dbl3_t_stencil_md *) next->f[0];
        const auto * _noalias const eval_f = (dbl3_t_stencil_md *) next->eval_f_stencil_md[0];
        const int * const mask = next->mask;
        // const int nlocal = atom_->nlocal;
        const int next_nlocal = next->nlocal;

        const double * const mass = atom->mass;
        const int * const type = next->type;

        int start = local_idxs[0];
        double dtf = 0.5 * update->dt * force->ftm2v;

        for (int i = 0; i < local_idxs.size(); i++) {
            // int idx = local_idxs[i];
            int idx = start + i;
            assert(idx == local_idxs[i]);
            if (mask[idx]) {
                // const double dtfm = dtf / mass[type[i]];
                const double dtfm = next->local_dtfm[i];
                next_v[idx].x += dtfm * (f[idx].x + eval_f[idx].x);
                next_v[idx].y += dtfm * (f[idx].y + eval_f[idx].y);
                next_v[idx].z += dtfm * (f[idx].z + eval_f[idx].z);
            }
        }
    }

    inline void post_force_stencil_md_(Atom* atom_, Modify* modify_) {
        auto * _noalias const v = (dbl3_t_stencil_md *) atom_->v[0];
        auto * _noalias const eval_f = (dbl3_t_stencil_md *) atom_->eval_f_stencil_md[0];

        int *type = atom_->type;
        int *mask = atom_->mask;

        int n_post_force = modify_->n_post_force;

        assert(n_post_force == 1);

        // auto fix_post_force = (FixLangevin*) modify_->fix[modify_->list_post_force[0]];
        auto fix_post_force = (FixLangevin*) modify->fix[modify->list_post_force[0]];

        auto gfactor1 = fix_post_force->gfactor1;
        auto gfactor2 = fix_post_force->gfactor2;
        // fix_post_force->compute_target();
        auto tsqrt = fix_post_force->tsqrt;

        const int nlocal = atom_->nlocal;

        #pragma cilk grainsize 2048
        cilk_for (int i = 0; i < nlocal; i++) {
            // these are per-atom variables that get updated. Need to put them here to avoid races.
            // double fdrag[3],fran[3];
            // dbl3_t_stencil_md fdrag, fran;

            // if (mask[i]) {
                double gamma1 = gfactor1[type[i]];
                double gamma2 = gfactor2[type[i]] * tsqrt;

                double rand_x = 0.6;
                double rand_y = 0.6;
                double rand_z = 0.6;

                dbl3_t_stencil_md fran = {gamma2 * (rand_x - 0.5), gamma2*(rand_y - 0.5), gamma2 * (rand_z - 0.5)};
                dbl3_t_stencil_md fdrag = {gamma1 * v[i].x, gamma1 * v[i].y, gamma1 * v[i].z};

                /*
                fran.x = gamma2*(rand_x-0.5);
                fran.y = gamma2*(rand_y-0.5);
                fran.z = gamma2*(rand_z-0.5);
                */

                /*
                fran[0] = gamma2*(random->uniform()-0.5);
                fran[1] = gamma2*(random->uniform()-0.5);
                fran[2] = gamma2*(random->uniform()-0.5);

                fdrag.x = gamma1*v[i].x;
                fdrag.y = gamma1*v[i].y;
                fdrag.z = gamma1*v[i].z;
                */

                eval_f[i].x += fdrag.x + fran.x;
                eval_f[i].y += fdrag.y + fran.y;
                eval_f[i].z += fdrag.z + fran.z;
            // }
        }
    }

    inline void post_force_stencil_md_affinity(Atom* atom_, Modify* modify_) {
        /*
        auto * _noalias const v = (dbl3_t_stencil_md *) atom_->v[0];
        auto * _noalias const eval_f = (dbl3_t_stencil_md *) atom_->eval_f_stencil_md[0];

        int *type = atom_->type;
        int *mask = atom_->mask;

        int n_post_force = modify_->n_post_force;

        assert(n_post_force == 1);

        // auto fix_post_force = (FixLangevin*) modify_->fix[modify_->list_post_force[0]];
        auto fix_post_force = (FixLangevin*) modify->fix[modify->list_post_force[0]];

        auto gfactor1 = fix_post_force->gfactor1;
        auto gfactor2 = fix_post_force->gfactor2;
        // fix_post_force->compute_target();
        auto tsqrt = fix_post_force->tsqrt;

        const int nlocal = atom_->nlocal;
        int num_chunks = nlocal / MODIFY_GRAINSIZE + 1;
        auto claimed = atom_->claimed;
        int num_workers = __cilkrts_get_nworkers();

        #pragma cilk grainsize 1
        cilk_for (int ii = 0; ii < num_chunks; ii++) {
            int start_chunk = __cilkrts_get_worker_number() * num_chunks / num_workers;
            for (int c = 0; c < num_chunks; ++c) {
                int s = (c + start_chunk) % num_chunks;
                if (claimed[s].load()) {
                    continue;
                }
                bool expected = false;
                if (claimed[s].compare_exchange_weak(expected, true, std::memory_order_relaxed)) {
                    for (int i = s * MODIFY_GRAINSIZE; i < (s + 1) * MODIFY_GRAINSIZE && i < nlocal; i++) {
                        double gamma1 = gfactor1[type[i]];
                        double gamma2 = gfactor2[type[i]] * tsqrt;

                        double rand_x = 0.6;
                        double rand_y = 0.6;
                        double rand_z = 0.6;

                        dbl3_t_stencil_md fran = {gamma2 * (rand_x - 0.5), gamma2 * (rand_y - 0.5),
                                                  gamma2 * (rand_z - 0.5)};
                        dbl3_t_stencil_md fdrag = {gamma1 * v[i].x, gamma1 * v[i].y, gamma1 * v[i].z};

                        eval_f[i].x += fdrag.x + fran.x;
                        eval_f[i].y += fdrag.y + fran.y;
                        eval_f[i].z += fdrag.z + fran.z;
                    }
                }
            }
        }

        for (int i = 0; i < num_chunks; i++) {
            claimed[i] = false;
        }
        */
    }

    inline void post_force_stencil_md(const std::vector<int>& local_idxs, Atom* atom_, Modify* modify_) {
        auto * _noalias const v = (dbl3_t_stencil_md *) atom_->v[0];
        auto * _noalias const eval_f = (dbl3_t_stencil_md *) atom_->eval_f_stencil_md[0];

        int *type = atom_->type;
        int *mask = atom_->mask;

        int n_post_force = modify_->n_post_force;

        assert(n_post_force == 1);

        // auto fix_post_force = (FixLangevin*) modify_->fix[modify_->list_post_force[0]];
        auto fix_post_force = (FixLangevin*) modify->fix[modify->list_post_force[0]];

        auto gfactor1 = fix_post_force->gfactor1;
        auto gfactor2 = fix_post_force->gfactor2;
        // fix_post_force->compute_target();
        auto tsqrt = fix_post_force->tsqrt;

        int start = local_idxs[0];
        for (int i = 0; i < local_idxs.size(); i++) {
            int idx = start + i;
            // int idx = local_idxs[i];
            assert(idx == local_idxs[i]);
            // these are per-atom variables that get updated. Need to put them here to avoid races.
            // double fdrag[3],fran[3];
            dbl3_t_stencil_md fdrag, fran;
            double gamma1, gamma2;

            if (mask[idx]) {
                gamma1 = gfactor1[type[idx]];
                gamma2 = gfactor2[type[idx]] * tsqrt;

                double rand_x = 0.6;
                double rand_y = 0.6;
                double rand_z = 0.6;
                fran.x = gamma2*(rand_x-0.5);
                fran.y = gamma2*(rand_y-0.5);
                fran.z = gamma2*(rand_z-0.5);

                /*
                fran[0] = gamma2*(random->uniform()-0.5);
                fran[1] = gamma2*(random->uniform()-0.5);
                fran[2] = gamma2*(random->uniform()-0.5);
                */

                fdrag.x = gamma1*v[idx].x;
                fdrag.y = gamma1*v[idx].y;
                fdrag.z = gamma1*v[idx].z;

                eval_f[idx].x += fdrag.x + fran.x;
                eval_f[idx].y += fdrag.y + fran.y;
                eval_f[idx].z += fdrag.z + fran.z;
            }
        }

    }

    template <bool curr_dt>
    void recv_pos_bins_stencil_md_helper(queue_info& zoid, int timestep, Atom* atom_) {
        int zoid_num = zoid.num;
        auto& recv_from = curr_dt ? lmp->recv_from_neighbors[zoid_num] : lmp->recv_from_neighbors_next_dt[zoid_num];
        auto* _noalias const recv_x = (dbl3_t_stencil_md*) atom_->x[0];

        auto recv_bin_to_idx = zoid.bin_to_idx[timestep];
        auto recv_bin_to_size = zoid.bin_to_size[timestep];

        for (int i = 0; i < recv_from.size(); i++) {
            int recv_zoid_num = recv_from[i];
            auto& recv_zoid = curr_dt ? lmp->zoid_num_to_zoid[recv_zoid_num] : lmp->zoid_num_to_zoid_next_dt[recv_zoid_num];
            auto& other_atom_arr = lmp->atom_stencil_md[recv_zoid_num];
            Atom* other_atom = curr_dt ? other_atom_arr[timestep] : other_atom_arr[NUM_TIMESTEPS_IN_PARALLEL - timestep];
            const dbl3_t_stencil_md* _noalias const send_x = (dbl3_t_stencil_md*) other_atom->x[0];

            int pbc_flag_[3] = {0};
            for (int dim = 0; dim < 3; dim++) {
                if (recv_zoid.where[dim] == RIGHT && zoid.where[dim] == PBC) { pbc_flag_[dim] = -1; }

                if (recv_zoid.where[dim] == PBC && zoid.where[dim] == RIGHT) { pbc_flag_[dim] = 1; }
            }

            auto send_bin_to_idx = recv_zoid.bin_to_idx[timestep];
            auto send_bin_to_size = recv_zoid.bin_to_size[timestep];

            int num_pos_bins = zoid.recv_pos_num_bins[timestep][i];
            auto recv_pos_bins = zoid.recv_pos_bins[timestep][i];

            for (int j = 0; j < num_pos_bins; j++) {
                auto bin = recv_pos_bins[j];
                auto bin_idx = get_bin_idx(bin);

                int send_bin_idx = send_bin_to_idx[bin_idx];
                int send_size = send_bin_to_size[bin_idx];

                int recv_bin_idx = recv_bin_to_idx[bin_idx];
                int recv_size = recv_bin_to_size[bin_idx];

                assert(send_size != -1);
                assert(send_size == recv_size);

                for (int k = 0; k < send_size; k++) {
                    int send_idx = send_bin_idx + k;
                    int recv_idx = recv_bin_idx + k;
                    tagint src_tag = other_atom->tag[send_idx];
                    tagint dst_tag = atom_->tag[recv_idx];
                    assert(src_tag == dst_tag);

                    recv_x[recv_idx].x = send_x[send_idx].x + pbc_flag_[0] * domain->prd[0];
                    recv_x[recv_idx].y = send_x[send_idx].y + pbc_flag_[1] * domain->prd[1];
                    recv_x[recv_idx].z = send_x[send_idx].z + pbc_flag_[2] * domain->prd[2];
                }
            }
        }
    }

    template <bool curr_dt>
    void fuse_initial_integrate_stencil_md_pos_vel(queue_info& zoid, int timestep, Atom* next) {
        auto *_noalias const recv_x = (dbl3_t_stencil_md *) next->x[0];
        auto *_noalias const recv_v = (dbl3_t_stencil_md *) next->v[0];

        auto recv_bin_to_idx = zoid.bin_to_idx[timestep];
        auto recv_bin_to_size = zoid.bin_to_size[timestep];

        auto& comm_local_bins = zoid.comm_local_bins[timestep];
        auto& bin_to_local_idxs = next->bin_to_local_idxs2;

        for (int i = 0; i < comm_local_bins.size(); i++) {
            auto &bin = comm_local_bins[i];
            auto &local_idxs = bin_to_local_idxs[bin[0]][bin[1]][bin[2]];
            auto bin_idx = get_bin_idx(bin);
            int start = local_idxs[0];

            int recv_zoid_num = zoid.bin_to_pos_vel_comm[timestep][i];
            auto &recv_zoid = curr_dt ? lmp->zoid_num_to_zoid[recv_zoid_num]
                                      : lmp->zoid_num_to_zoid_next_dt[recv_zoid_num];
            auto &other_atom_arr = lmp->atom_stencil_md[recv_zoid_num];
            Atom *other_atom = curr_dt ? other_atom_arr[timestep] : other_atom_arr[NUM_TIMESTEPS_IN_PARALLEL - timestep];
            const dbl3_t_stencil_md *_noalias const send_x = (dbl3_t_stencil_md *) other_atom->x[0];
            const dbl3_t_stencil_md *_noalias const send_v = (dbl3_t_stencil_md *) other_atom->v[0];

            int pbc_flag_[3] = {0};
            for (int dim = 0; dim < 3; dim++) {
                if (recv_zoid.where[dim] == RIGHT && zoid.where[dim] == PBC) {
                    pbc_flag_[dim] = -1;
                }

                if (recv_zoid.where[dim] == PBC && zoid.where[dim] == RIGHT) {
                    pbc_flag_[dim] = 1;
                }
            }

            bool can_memcpy = false;
            if (pbc_flag_[0] == 0 && pbc_flag_[1] == 0 && pbc_flag_[2] == 0) {
                can_memcpy = true;
            }

            auto send_bin_to_idx = recv_zoid.bin_to_idx[timestep];
            auto send_bin_to_size = recv_zoid.bin_to_size[timestep];

            int send_bin_idx = send_bin_to_idx[bin_idx];
            int send_size = send_bin_to_size[bin_idx];

            int recv_bin_idx = recv_bin_to_idx[bin_idx];
            int recv_size = recv_bin_to_size[bin_idx];

            assert(recv_bin_idx == start);
            assert(send_size != -1);
            assert(send_size == recv_size);
            assert(send_size == local_idxs.size());

            if (false) {
                memcpy(&recv_x[recv_bin_idx], &send_x[send_bin_idx], 3 * send_size * sizeof(double));
                memcpy(&recv_v[recv_bin_idx], &send_v[send_bin_idx], 3 * send_size * sizeof(double));
            } else {
                for (int k = 0; k < send_size; k++) {
                    int send_idx = send_bin_idx + k;
                    int recv_idx = recv_bin_idx + k;
                    if (DEBUG_SEND_RECV_DATA) {
                        tagint src_tag = other_atom->tag[send_idx];
                        tagint dst_tag = next->tag[recv_idx];
                        assert(src_tag == dst_tag);
                    }

                    recv_x[recv_idx].x = send_x[send_idx].x + pbc_flag_[0] * domain->prd[0];
                    recv_x[recv_idx].y = send_x[send_idx].y + pbc_flag_[1] * domain->prd[1];
                    recv_x[recv_idx].z = send_x[send_idx].z + pbc_flag_[2] * domain->prd[2];

                    recv_v[recv_idx].x = send_v[send_idx].x;
                    recv_v[recv_idx].y = send_v[send_idx].y;
                    recv_v[recv_idx].z = send_v[send_idx].z;
                }
            }

            // memcpy(&next->v[recv_bin_idx][0], &other_atom->v[send_bin_idx][0], 3 * send_size * sizeof(double));
        }
    }

    void initial_integrate_stencil_md_segments(queue_info& zoid, int timestep, Atom* curr, Atom* next, int* atom_idx_mapping) {
        auto * _noalias const curr_x = (dbl3_t_stencil_md *) curr->x[0];
        auto * _noalias const next_x = (dbl3_t_stencil_md *) next->x[0];
        auto * _noalias const curr_v = (dbl3_t_stencil_md *) curr->v[0];
        auto * _noalias const next_v = (dbl3_t_stencil_md *) next->v[0];
        auto * _noalias const curr_f = (dbl3_t_stencil_md *) curr->f[0];
        auto * _noalias const curr_eval_f = (dbl3_t_stencil_md *) curr->eval_f_stencil_md[0];

        const int * const mask = curr->mask;
        const int nlocal = curr->nlocal;

        // const double * const mass = atom->mass;
        // const int * const type = curr->type;

        double dtv = update->dt;
        const auto& local_dtfm = curr->local_dtfm;
        // double dtf = 0.5 * update->dt * force->ftm2v;

        int idx = 0;
        const auto& segment_idxs = curr->atom_idx_mapping_segment_idxs;
        const auto& segment_sizes = curr->atom_idx_mapping_segment_sizes;

        assert(segment_sizes.size() <= 2);

        int my_start_idxs[2] = {0, segment_sizes[0]};

        for (int i = 0; i < segment_idxs.size(); i++) {
            int my_start_idx = my_start_idxs[i];
            int start_idx = segment_idxs[i];
            int size = segment_sizes[i];

            cilk_for (int j = 0; j < size; j++) {
                int idx = my_start_idx + j;
                int next_idx = start_idx + j;
                assert(next_idx == atom_idx_mapping[idx]);
                assert(next_idx != -1);

                // if (mask[idx]) {
                    const double dtfm = local_dtfm[idx];

                    next_v[next_idx].x = curr_v[idx].x + dtfm * (curr_f[idx].x + curr_eval_f[idx].x);
                    next_v[next_idx].y = curr_v[idx].y + dtfm * (curr_f[idx].y + curr_eval_f[idx].y);
                    next_v[next_idx].z = curr_v[idx].z + dtfm * (curr_f[idx].z + curr_eval_f[idx].z);

                    next_x[next_idx].x = curr_x[idx].x + dtv * next_v[next_idx].x;
                    next_x[next_idx].y = curr_x[idx].y + dtv * next_v[next_idx].y;
                    next_x[next_idx].z = curr_x[idx].z + dtv * next_v[next_idx].z;

                    curr_f[idx].x = 0.0;
                    curr_f[idx].y = 0.0;
                    curr_f[idx].z = 0.0;
                    curr_eval_f[idx].x = 0.0;
                    curr_eval_f[idx].y = 0.0;
                    curr_eval_f[idx].z = 0.0;
                // }
            }
        }
    }

    void initial_integrate_stencil_md(queue_info& zoid, int timestep, Atom* curr, Atom* next, int* atom_idx_mapping) {
        auto * _noalias const curr_x = (dbl3_t_stencil_md *) curr->x[0];
        auto * _noalias const next_x = (dbl3_t_stencil_md *) next->x[0];
        auto * _noalias const curr_v = (dbl3_t_stencil_md *) curr->v[0];
        auto * _noalias const next_v = (dbl3_t_stencil_md *) next->v[0];
        auto * _noalias const curr_f = (dbl3_t_stencil_md *) curr->f[0];
        auto * _noalias const curr_eval_f = (dbl3_t_stencil_md *) curr->eval_f_stencil_md[0];

        const int * const mask = curr->mask;
        const int nlocal = curr->nlocal;

        // const double * const mass = atom->mass;
        // const int * const type = curr->type;

        double dtv = update->dt;
        auto& local_dtfm = curr->local_dtfm;
        // double dtf = 0.5 * update->dt * force->ftm2v;

        #pragma cilk grainsize 2048
        cilk_for (int i = 0; i < nlocal; i++) {
            if (mask[i]) {
                // const double dtfm = dtf / mass[type[i]];
                const double dtfm = local_dtfm[i];

                int next_idx = atom_idx_mapping[i];
                // idxs.push_back(next_idx);

                next_v[next_idx].x = curr_v[i].x + dtfm * (curr_f[i].x + curr_eval_f[i].x);
                next_v[next_idx].y = curr_v[i].y + dtfm * (curr_f[i].y + curr_eval_f[i].y);
                next_v[next_idx].z = curr_v[i].z + dtfm * (curr_f[i].z + curr_eval_f[i].z);

                next_x[next_idx].x = curr_x[i].x + dtv * next_v[next_idx].x;
                next_x[next_idx].y = curr_x[i].y + dtv * next_v[next_idx].y;
                next_x[next_idx].z = curr_x[i].z + dtv * next_v[next_idx].z;

                assert(curr->tag[i] == next->tag[next_idx]);
                assert(next_idx != -1);

                curr_f[i].x = 0.0;
                curr_f[i].y = 0.0;
                curr_f[i].z = 0.0;
                curr_eval_f[i].x = 0.0;
                curr_eval_f[i].y = 0.0;
                curr_eval_f[i].z = 0.0;
            }
        }
    }

    void initial_integrate_stencil_md_affinity(queue_info& zoid, int timestep, Atom* curr, Atom* next, int* atom_idx_mapping) {
        auto * _noalias const curr_x = (dbl3_t_stencil_md *) curr->x[0];
        auto * _noalias const next_x = (dbl3_t_stencil_md *) next->x[0];
        auto * _noalias const curr_v = (dbl3_t_stencil_md *) curr->v[0];
        auto * _noalias const next_v = (dbl3_t_stencil_md *) next->v[0];
        auto * _noalias const curr_f = (dbl3_t_stencil_md *) curr->f[0];
        auto * _noalias const curr_eval_f = (dbl3_t_stencil_md *) curr->eval_f_stencil_md[0];

        const int * const mask = curr->mask;
        const int nlocal = curr->nlocal;
        int num_chunks = nlocal / MODIFY_GRAINSIZE + 1;
        auto claimed = curr->claimed;
        auto claimed_int = curr->claimed_int;
        auto claimed_flag = curr->claimed_flag;
        int num_workers = __cilkrts_get_nworkers();
        num_chunks = num_workers;
        int chunk_size = nlocal / num_chunks + 1;

        // const double * const mass = atom->mass;
        // const int * const type = curr->type;

        double dtv = update->dt;
        auto& local_dtfm = curr->local_dtfm;
        // double dtf = 0.5 * update->dt * force->ftm2v;

        // base case
        if (num_chunks == 1) {
            for (int i = 0; i < nlocal; i++) {
                const double dtfm = local_dtfm[i];
                int next_idx = atom_idx_mapping[i];
                next_v[next_idx].x = curr_v[i].x + dtfm * (curr_f[i].x + curr_eval_f[i].x);
                next_v[next_idx].y = curr_v[i].y + dtfm * (curr_f[i].y + curr_eval_f[i].y);
                next_v[next_idx].z = curr_v[i].z + dtfm * (curr_f[i].z + curr_eval_f[i].z);

                curr_f[i].x = 0.0;
                curr_f[i].y = 0.0;
                curr_f[i].z = 0.0;
                curr_eval_f[i].x = 0.0;
                curr_eval_f[i].y = 0.0;
                curr_eval_f[i].z = 0.0;

                next_x[next_idx].x = curr_x[i].x + dtv * next_v[next_idx].x;
                next_x[next_idx].y = curr_x[i].y + dtv * next_v[next_idx].y;
                next_x[next_idx].z = curr_x[i].z + dtv * next_v[next_idx].z;

                assert(curr->tag[i] == next->tag[next_idx]);
                assert(next_idx != -1);
            }

            return;
        }

        #pragma cilk grainsize 1
        cilk_for (int ii = 0; ii < num_chunks; ii++) {
            int start_chunk = __cilkrts_get_worker_number() * num_chunks / num_workers;
            for (int c = 0; c < num_chunks; ++c) {
                int s = (c + start_chunk) % num_chunks;

                if (claimed_flag[s].test(std::memory_order_relaxed)) {
                    continue;
                }
                if (!claimed_flag[s].test_and_set(std::memory_order_relaxed)) {
                    // for (int i = s * MODIFY_GRAINSIZE; i < (s + 1) * MODIFY_GRAINSIZE && i < nlocal; i++) {
                    for (int i = s * chunk_size; i < (s + 1) * chunk_size && i < nlocal; i++) {
                        const double dtfm = local_dtfm[i];
                        int next_idx = atom_idx_mapping[i];
                        next_v[next_idx].x = curr_v[i].x + dtfm * (curr_f[i].x + curr_eval_f[i].x);
                        next_v[next_idx].y = curr_v[i].y + dtfm * (curr_f[i].y + curr_eval_f[i].y);
                        next_v[next_idx].z = curr_v[i].z + dtfm * (curr_f[i].z + curr_eval_f[i].z);

                        curr_f[i].x = 0.0;
                        curr_f[i].y = 0.0;
                        curr_f[i].z = 0.0;
                        curr_eval_f[i].x = 0.0;
                        curr_eval_f[i].y = 0.0;
                        curr_eval_f[i].z = 0.0;

                        next_x[next_idx].x = curr_x[i].x + dtv * next_v[next_idx].x;
                        next_x[next_idx].y = curr_x[i].y + dtv * next_v[next_idx].y;
                        next_x[next_idx].z = curr_x[i].z + dtv * next_v[next_idx].z;

                        assert(curr->tag[i] == next->tag[next_idx]);
                        assert(next_idx != -1);
                    }
                }
            }
        }

        for (int i = 0; i < num_chunks; i++) {
            claimed_flag[i].clear(std::memory_order_relaxed);
        }
    }

    /*
    void initial_integrate_stencil_md_affinity_reverse(queue_info& zoid, int timestep, Atom* curr, Atom* next,
                                                       int* atom_idx_mapping, int* next_atom_idx_mapping,
                                                       std::vector<int>& next_atom_idx_mapping_idxs) {
        auto * _noalias const curr_x = (dbl3_t_stencil_md *) curr->x[0];
        auto * _noalias const next_x = (dbl3_t_stencil_md *) next->x[0];
        auto * _noalias const curr_v = (dbl3_t_stencil_md *) curr->v[0];
        auto * _noalias const next_v = (dbl3_t_stencil_md *) next->v[0];
        auto * _noalias const curr_f = (dbl3_t_stencil_md *) curr->f[0];
        auto * _noalias const curr_eval_f = (dbl3_t_stencil_md *) curr->eval_f_stencil_md[0];

        // const int * const mask = curr->mask;
        // const int next_total = next->nlocal + next->nghost;
        const int next_total = next_atom_idx_mapping_idxs.size();
        int num_chunks = (next_total) / MODIFY_GRAINSIZE + 1;
        auto claimed = next->claimed;
        int num_workers = __cilkrts_get_nworkers();

        // const double * const mass = atom->mass;
        // const int * const type = curr->type;

        double dtv = update->dt;
        auto& local_dtfm = curr->local_dtfm;
        // double dtf = 0.5 * update->dt * force->ftm2v;

        #pragma cilk grainsize 1
        cilk_for (int ii = 0; ii < num_chunks; ii++) {
            int start_chunk = __cilkrts_get_worker_number() * num_chunks / num_workers;
            for (int c = 0; c < num_chunks; ++c) {
                int s = (c + start_chunk) % num_chunks;
                if (claimed[s].load()) {
                    continue;
                }
                bool expected = false;
                if (claimed[s].compare_exchange_weak(expected, true, std::memory_order_relaxed)) {
                    for (int i = s * MODIFY_GRAINSIZE; i < (s + 1) * MODIFY_GRAINSIZE && i < next_total; i++) {
                        // int prev_idx = next_atom_idx_mapping[i];
                        int curr_idx = next_atom_idx_mapping_idxs[i];
                        int prev_idx = next_atom_idx_mapping[curr_idx];
                        assert(prev_idx != -1);
                        // if (prev_idx != -1) {
                            const double dtfm = local_dtfm[prev_idx];
                            // int next_idx = atom_idx_mapping[i];
                            // assert(next_atom_idx_mapping[next_idx] == i);
                            next_v[curr_idx].x = curr_v[prev_idx].x + dtfm * (curr_f[prev_idx].x + curr_eval_f[prev_idx].x);
                            next_v[curr_idx].y = curr_v[prev_idx].y + dtfm * (curr_f[prev_idx].y + curr_eval_f[prev_idx].y);
                            next_v[curr_idx].z = curr_v[prev_idx].z + dtfm * (curr_f[prev_idx].z + curr_eval_f[prev_idx].z);

                            next_x[curr_idx].x = curr_x[prev_idx].x + dtv * next_v[curr_idx].x;
                            next_x[curr_idx].y = curr_x[prev_idx].y + dtv * next_v[curr_idx].y;
                            next_x[curr_idx].z = curr_x[prev_idx].z + dtv * next_v[curr_idx].z;

                            assert(curr->tag[prev_idx] == next->tag[curr_idx]);
                            assert(prev_idx != -1);
                        // }
                    }
                }
            }
        }

        for (int i = 0; i < num_chunks; i++) {
            claimed[i] = false;
        }
    }
    */

    void initial_integrate_stencil_md_bins(const queue_info& zoid, Atom* curr, Atom* next, int* atom_idx_mapping) {
        auto * _noalias const curr_x = (dbl3_t_stencil_md *) curr->x[0];
        auto * _noalias const next_x = (dbl3_t_stencil_md *) next->x[0];
        auto * _noalias const curr_v = (dbl3_t_stencil_md *) curr->v[0];
        auto * _noalias const next_v = (dbl3_t_stencil_md *) next->v[0];
        auto * _noalias const curr_f = (dbl3_t_stencil_md *) curr->f[0];
        auto * _noalias const curr_eval_f = (dbl3_t_stencil_md *) curr->eval_f_stencil_md[0];

        const int * const mask = curr->mask;
        const int nlocal = curr->nlocal;

        // const double * const mass = atom->mass;
        // const int * const type = curr->type;

        double dtv = update->dt;

        auto& local_bins = curr->local_bins;
        auto& local_bins_idxs = curr->local_bins_idxs;

        auto& next_bin_idxs = next->bin_to_local_idxs2;
        auto& local_dtfm = curr->local_dtfm;
        // double dtf = 0.5 * update->dt * force->ftm2v;

        cilk_for (int b = 0; b < local_bins.size(); b++) {
            auto& bin = local_bins[b];
            auto& idxs = local_bins_idxs[b];
            auto& next_idxs = next_bin_idxs[bin[0]][bin[1]][bin[2]];

            int curr_start = idxs[0];
            int next_start = next_idxs[0];

            for (int j = 0; j < idxs.size(); j++) {
                int i = curr_start + j;
                int next_idx = next_start + j;

                assert(i >= 0 && i < curr->nlocal);

                if (mask[i]) {
                    // const double dtfm = dtf / mass[type[i]];
                    const double dtfm = local_dtfm[i];

                    assert(next_idx == atom_idx_mapping[i]);

                    next_v[next_idx].x = curr_v[i].x + dtfm * (curr_f[i].x + curr_eval_f[i].x);
                    next_v[next_idx].y = curr_v[i].y + dtfm * (curr_f[i].y + curr_eval_f[i].y);
                    next_v[next_idx].z = curr_v[i].z + dtfm * (curr_f[i].z + curr_eval_f[i].z);

                    next_x[next_idx].x = curr_x[i].x + dtv * next_v[next_idx].x;
                    next_x[next_idx].y = curr_x[i].y + dtv * next_v[next_idx].y;
                    next_x[next_idx].z = curr_x[i].z + dtv * next_v[next_idx].z;

                    assert(curr->tag[i] == next->tag[next_idx]);
                    assert(next_idx != -1);

                    curr_f[i].x = 0.0;
                    curr_f[i].y = 0.0;
                    curr_f[i].z = 0.0;
                    curr_eval_f[i].x = 0.0;
                    curr_eval_f[i].y = 0.0;
                    curr_eval_f[i].z = 0.0;
                }
            }
        }
    }

    template <bool curr_dt>
    void fuse_initial_integrate_stencil_md(queue_info& zoid, int timestep, Atom* curr, Atom* next, int* atom_idx_mapping) {
        cilk_scope {
            cilk_spawn recv_pos_bins_stencil_md_helper<curr_dt>(zoid, timestep + 1, next);
            cilk_spawn fuse_initial_integrate_stencil_md_pos_vel<curr_dt>(zoid, timestep + 1, next);
            cilk_spawn initial_integrate_stencil_md_segments(zoid, timestep, curr, next, atom_idx_mapping);
            // cilk_spawn initial_integrate_stencil_md(zoid, timestep, curr, next, atom_idx_mapping);
            // cilk_spawn initial_integrate_stencil_md_bins(zoid, curr, next, atom_idx_mapping);
            memset(&curr->f[curr->nlocal][0], 0, (curr->nghost) * 3 * sizeof(double));
        }
    }

    void fuse_force_computation_reduce(queue_info& zoid, int timestep, Atom* next, Neighbor* neigh_next, Force* next_force, Modify* modify_) {
        memset(&next->eval_f_stencil_md[next->nlocal][0], 0, (next->nghost) * 3 * sizeof(double));
        for (int k = 0; k < next->nlocal + next->nghost; k++) {
            assert(fabs(next->eval_f_stencil_md[k][0]) < 1e-6);
            assert(fabs(next->eval_f_stencil_md[k][1]) < 1e-6);
            assert(fabs(next->eval_f_stencil_md[k][2]) < 1e-6);
        }

        int nthreads_to_use = zoid.inum_per_timestep[timestep];

        // begin force computation, inline lj_cut and bond_fene
        assert(PURELY_LOCAL_POTENTIAL);

        const auto * _noalias const x = (dbl3_t_stencil_md *) next->x[0];
        auto * _noalias const f = (dbl3_t_stencil_md *) next->eval_f_stencil_md[0];
        const int * _noalias const type = next->type;
        const double * _noalias const special_lj = force->special_lj;

        auto pair = (PairLJCutOMP*) next_force->pair;
        auto bond = (BondFENE*) next_force->bond;

        const int * _noalias const ilist = pair->list->ilist;
        const int * _noalias const numneigh = pair->list->numneigh;
        const int * const * const firstneigh = pair->list->firstneigh;

        const auto* cutsq = pair->cutsq;
        const auto* offset = pair->offset;
        const auto* lj1 = pair->lj1;
        const auto* lj2 = pair->lj2;
        const auto* lj3 = pair->lj3;
        const auto* lj4 = pair->lj4;
        auto newton_pair = force->newton_pair;

        const auto* sigma = bond->sigma;
        const auto* epsilon = bond->epsilon;
        const auto* r0 = bond->r0;
        const auto* k = bond->k;

        int nlocal = next->nlocal;
        int nall = next->nlocal + next->nghost;

        auto bondlist = neigh_next->atom_bondlist;

        if (nlocal < 512 && false) {
            for (int ii = 0; ii < nlocal; ii++) {
                const int i = ilist[ii];
                assert(i == ii);
                const int itype = type[i];

                const int *_noalias const jlist = firstneigh[i];
                const double *_noalias const cutsqi = cutsq[itype];
                const double *_noalias const offseti = offset[itype];
                const double *_noalias const lj1i = lj1[itype];
                const double *_noalias const lj2i = lj2[itype];
                const double *_noalias const lj3i = lj3[itype];
                const double *_noalias const lj4i = lj4[itype];

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
                    double factor_lj = special_lj[pair->sbmask(j)];
                    j &= NEIGHMASK;

                    double delx = xtmp - x[j].x;
                    double dely = ytmp - x[j].y;
                    double delz = ztmp - x[j].z;
                    double rsq = delx * delx + dely * dely + delz * delz;
                    int jtype = type[j];

                    if (rsq < cutsqi[jtype]) {
                        double r2inv = 1.0 / rsq;
                        double r6inv = r2inv * r2inv * r2inv;
                        double forcelj = r6inv * (lj1i[jtype] * r6inv - lj2i[jtype]);
                        double fpair = factor_lj * forcelj * r2inv;

                        fxtmp += delx * fpair;
                        fytmp += dely * fpair;
                        fztmp += delz * fpair;

                        if (newton_pair || j < nlocal) {
                            f[j].x -= delx * fpair;
                            f[j].y -= dely * fpair;
                            f[j].z -= delz * fpair;
                        }
                    }
                }

                auto& lst_bonds = neigh_next->atom_bondlist[i];
                for (int j = 0; j < lst_bonds.size(); j++) {
                    auto& bond_info = lst_bonds[j];
                    int i2 = bond_info.first;
                    int bond_type = bond_info.second;

                    double delx = xtmp - x[i2].x;
                    double dely = ytmp - x[i2].y;
                    double delz = ztmp - x[i2].z;

                    double rsq = delx * delx + dely * dely + delz * delz;
                    double r0sq = r0[bond_type] * r0[bond_type];
                    double rlogarg = 1.0 - rsq / r0sq;

                    if (rlogarg < 0.1) {
                        error->warning(FLERR, "FENE bond too long: {} {} {} {:.8}",
                                       update->ntimestep, atom->tag[i], atom->tag[i2], sqrt(rsq));
//                            if (check_error_thr((rlogarg <= -3.0),tid,FLERR,"Bad FENE bond"))
//                                return;
                        assert(false);

                        rlogarg = 0.1;
                    }

                    double fbond = -k[bond_type] / rlogarg;

                    // force from LJ term
                    double sr2 = 0.0;
                    double sr6 = 0.0;

                    if (rsq < MathConst::MY_CUBEROOT2 * sigma[bond_type] * sigma[bond_type]) {
                        sr2 = sigma[bond_type] * sigma[bond_type] / rsq;
                        sr6 = sr2 * sr2 * sr2;
                        fbond += 48.0 * epsilon[bond_type] * sr6 * (sr6 - 0.5) / rsq;
                    }

                    // energy

                    // apply force to each of 2 atoms

                    if (newton_pair || i < nlocal) {
                        fxtmp += delx * fbond;
                        fytmp += dely * fbond;
                        fztmp += delz * fbond;
                    }

                    if (newton_pair || i2 < nlocal) {
                        f[i2].x -= delx * fbond;
                        f[i2].y -= dely * fbond;
                        f[i2].z -= delz * fbond;
                    }
                }

                f[i].x += fxtmp;
                f[i].y += fytmp;
                f[i].z += fztmp;

            }

            return;
        }

        cilk_for (int tid = 0; tid < nthreads_to_use; tid++) {
            // each thread works on a fixed chunk of atoms.
            const int idelta = 1 + nlocal / nthreads_to_use;
            int ifrom = tid * idelta;
            int ito = ((ifrom + idelta) > nlocal) ? nlocal : ifrom + idelta;
            auto* thread_local_f = f + tid * nall;
            memset(thread_local_f, 0, nall * 3 * sizeof(double));

            for (int ii = ifrom; ii < ito; ++ii) {
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
                    double factor_lj = special_lj[pair->sbmask(j)];
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
                            thread_local_f[j].x -= delx*fpair;
                            thread_local_f[j].y -= dely*fpair;
                            thread_local_f[j].z -= delz*fpair;
                        }
                    }
                }

                for (auto& [i2, type] : bondlist[i]) {
                    // auto& bond_info = lst_bonds[j];
                    // int i2 = bond_info.first;
                    // int type = bond_info.second;

                    double delx = xtmp - x[i2].x;
                    double dely = ytmp - x[i2].y;
                    double delz = ztmp - x[i2].z;

                    double rsq = delx * delx + dely * dely + delz * delz;
                    double r0sq = r0[type] * r0[type];
                    double rlogarg = 1.0 - rsq / r0sq;

                    if (rlogarg < 0.1) {
                        error->warning(FLERR, "FENE bond too long: {} {} {} {:.8}",
                                       update->ntimestep, atom->tag[i], atom->tag[i2], sqrt(rsq));
//                            if (check_error_thr((rlogarg <= -3.0),tid,FLERR,"Bad FENE bond"))
//                                return;
                        assert(false);

                        rlogarg = 0.1;
                    }

                    double fbond = -k[type] / rlogarg;

                    // force from LJ term
                    double sr2 = 0.0;
                    double sr6 = 0.0;

                    if (rsq < MathConst::MY_CUBEROOT2 * sigma[type] * sigma[type]) {
                        sr2 = sigma[type] * sigma[type] / rsq;
                        sr6 = sr2 * sr2 * sr2;
                        fbond += 48.0 * epsilon[type] * sr6 * (sr6 - 0.5) / rsq;
                    }

                    if (newton_pair || i < nlocal) {
                        fxtmp += delx * fbond;
                        fytmp += dely * fbond;
                        fztmp += delz * fbond;
                    }

                    if (newton_pair || i2 < nlocal) {
                        thread_local_f[i2].x -= delx*fbond;
                        thread_local_f[i2].y -= dely*fbond;
                        thread_local_f[i2].z -= delz*fbond;
                    }
                }

                thread_local_f[i].x += fxtmp;
                thread_local_f[i].y += fytmp;
                thread_local_f[i].z += fztmp;
            }
        }

        if (nthreads_to_use == 1) {
            return;
        }

        double* f_ = &(next->eval_f_stencil_md[0][0]);

        int nvals = nall * 3;

        // do not explicitly set chunk size, have cilk figure it out.
        cilk_for (int i = 0; i < nvals; i++) {
            for (int n = 1; n < nthreads_to_use; n++) {
                f_[i] += f_[n * nvals + i];
            }
        }
    }

    void stencil_md_fuse_force_computation_atomics(queue_info& zoid, int timestep, Atom* next, Neighbor* neigh_next, Force* next_force, Modify* modify_) {
        const auto * _noalias const x = (dbl3_t_stencil_md *) next->x[0];
        auto * _noalias const f = (dbl3_t_stencil_md *) next->eval_f_stencil_md[0];

        // auto pair = (PairLJCutOMP*) next_force->pair;
        auto pair = (PairLJCut*) next_force->pair;
        auto bond = (BondFENE*) next_force->bond;

        const auto* _noalias bondlist = neigh_next->atom_bondlist;
        auto& idx_use_atomics = next->idx_use_atomics;

        const int * _noalias const ilist = pair->list->ilist;
        const int * _noalias const numneigh = pair->list->numneigh;
        const int * const * const firstneigh = pair->list->firstneigh;
        const double * _noalias const special_lj = force->special_lj;

        auto* spinlocks = next->spinlocks;

        assert(pair->list->inum == next->nlocal);

        const auto* cutsq = pair->cutsq;
        const auto* offset = pair->offset;
        const auto* lj1 = pair->lj1;
        const auto* lj2 = pair->lj2;
        const auto* lj3 = pair->lj3;
        const auto* lj4 = pair->lj4;
        auto newton_pair = force->newton_pair;

        const auto* _noalias const sigma = bond->sigma;
        const auto* _noalias const epsilon = bond->epsilon;
        const auto* _noalias const r0 = bond->r0;
        const auto* _noalias const k = bond->k;

        const int* _noalias const atom_type = next->type;

        const int nlocal = next->nlocal;

        constexpr bool USE_ATOMIC_FETCH_ADD = true;

        constexpr int BASE_CASE_SIZE = 1024;

        if (nlocal < BASE_CASE_SIZE) {
            for (int i = 0; i < nlocal; i++) {
                const int itype = atom_type[i];

                const int *_noalias const jlist = firstneigh[i];
                const double *_noalias const cutsqi = cutsq[itype];
                const double *_noalias const offseti = offset[itype];
                const double *_noalias const lj1i = lj1[itype];
                const double *_noalias const lj2i = lj2[itype];
                const double *_noalias const lj3i = lj3[itype];
                const double *_noalias const lj4i = lj4[itype];

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
                    double factor_lj = special_lj[pair->sbmask(j)];
                    j &= NEIGHMASK;

                    double delx = xtmp - x[j].x;
                    double dely = ytmp - x[j].y;
                    double delz = ztmp - x[j].z;
                    double rsq = delx * delx + dely * dely + delz * delz;
                    int jtype = atom_type[j];

                    if (rsq < cutsqi[jtype]) {
                        double r2inv = 1.0 / rsq;
                        double r6inv = r2inv * r2inv * r2inv;
                        double forcelj = r6inv * (lj1i[jtype] * r6inv - lj2i[jtype]);
                        double fpair = factor_lj * forcelj * r2inv;

                        fxtmp += delx * fpair;
                        fytmp += dely * fpair;
                        fztmp += delz * fpair;

                        if (newton_pair || j < nlocal) {
                            f[j].x -= delx * fpair;
                            f[j].y -= dely * fpair;
                            f[j].z -= delz * fpair;
                        }
                    }
                }

                auto& lst_bonds = bondlist[i];
                for (int j = 0; j < lst_bonds.size(); j++) {
                    auto &bond_info = lst_bonds[j];
                    int i2 = bond_info.first;
                    int type = bond_info.second;

                    double delx = xtmp - x[i2].x;
                    double dely = ytmp - x[i2].y;
                    double delz = ztmp - x[i2].z;

                    double rsq = delx * delx + dely * dely + delz * delz;
                    double r0sq = r0[type] * r0[type];
                    double rlogarg = 1.0 - rsq / r0sq;

                    if (rlogarg < 0.1) {
                        error->warning(FLERR, "FENE bond too long: {} {} {} {:.8}",
                                       update->ntimestep, atom->tag[i], atom->tag[i2], sqrt(rsq));
                        //                            if (check_error_thr((rlogarg <= -3.0),tid,FLERR,"Bad FENE bond"))
                        //                                return;
                        assert(false);

                        rlogarg = 0.1;
                    }

                    double fbond = -k[type] / rlogarg;

                    // force from LJ term
                    double sr2 = 0.0;
                    double sr6 = 0.0;

                    if (rsq < MathConst::MY_CUBEROOT2 * sigma[type] * sigma[type]) {
                        sr2 = sigma[type] * sigma[type] / rsq;
                        sr6 = sr2 * sr2 * sr2;
                        fbond += 48.0 * epsilon[type] * sr6 * (sr6 - 0.5) / rsq;
                    }

                    // energy

                    // apply force to each of 2 atoms

                    if (newton_pair || i < nlocal) {
                        fxtmp += delx * fbond;
                        fytmp += dely * fbond;
                        fztmp += delz * fbond;
                    }

                    if (newton_pair || i2 < nlocal) {
                        f[i2].x -= delx * fbond;
                        f[i2].y -= dely * fbond;
                        f[i2].z -= delz * fbond;
                    }
                }

                f[i].x += fxtmp;
                f[i].y += fytmp;
                f[i].z += fztmp;
            }

            return;
        }

        #pragma cilk grainsize 512
        cilk_for (int i = 0; i < nlocal; i++) {
            const int itype = atom_type[i];

            const int *_noalias const jlist = firstneigh[i];
            const double *_noalias const cutsqi = cutsq[itype];
            const double *_noalias const offseti = offset[itype];
            const double *_noalias const lj1i = lj1[itype];
            const double *_noalias const lj2i = lj2[itype];
            const double *_noalias const lj3i = lj3[itype];
            const double *_noalias const lj4i = lj4[itype];

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
                double factor_lj = special_lj[pair->sbmask(j)];
                j &= NEIGHMASK;

                double delx = xtmp - x[j].x;
                double dely = ytmp - x[j].y;
                double delz = ztmp - x[j].z;
                double rsq = delx * delx + dely * dely + delz * delz;
                int jtype = atom_type[j];

                if (rsq < cutsqi[jtype]) {
                    double r2inv = 1.0 / rsq;
                    double r6inv = r2inv * r2inv * r2inv;
                    double forcelj = r6inv * (lj1i[jtype] * r6inv - lj2i[jtype]);
                    double fpair = factor_lj * forcelj * r2inv;

                    fxtmp += delx * fpair;
                    fytmp += dely * fpair;
                    fztmp += delz * fpair;

                    if (newton_pair || j < nlocal) {
                        if (USE_ATOMIC_FETCH_ADD) {
                            __atomic_fetch_add(&f[j].x, -delx*fpair, __ATOMIC_RELAXED);
                            __atomic_fetch_add(&f[j].y, -dely*fpair, __ATOMIC_RELAXED);
                            __atomic_fetch_add(&f[j].z, -delz*fpair, __ATOMIC_RELAXED);
                        } else {
                            spinlocks[j].lock();
                            f[j].x -= delx * fpair;
                            f[j].y -= dely * fpair;
                            f[j].z -= delz * fpair;
                            spinlocks[j].unlock();
                        }
                    }
                }
            }

            auto& lst_bonds = bondlist[i];
            for (int j = 0; j < lst_bonds.size(); j++) {
                auto &bond_info = lst_bonds[j];
                int i2 = bond_info.first;
                int type = bond_info.second;

                double delx = xtmp - x[i2].x;
                double dely = ytmp - x[i2].y;
                double delz = ztmp - x[i2].z;

                double rsq = delx * delx + dely * dely + delz * delz;
                double r0sq = r0[type] * r0[type];
                double rlogarg = 1.0 - rsq / r0sq;

                if (rlogarg < 0.1) {
                    error->warning(FLERR, "FENE bond too long: {} {} {} {:.8}",
                                   update->ntimestep, atom->tag[i], atom->tag[i2], sqrt(rsq));
                    //                            if (check_error_thr((rlogarg <= -3.0),tid,FLERR,"Bad FENE bond"))
                    //                                return;
                    assert(false);

                    rlogarg = 0.1;
                }

                double fbond = -k[type] / rlogarg;

                // force from LJ term
                double sr2 = 0.0;
                double sr6 = 0.0;

                if (rsq < MathConst::MY_CUBEROOT2 * sigma[type] * sigma[type]) {
                    sr2 = sigma[type] * sigma[type] / rsq;
                    sr6 = sr2 * sr2 * sr2;
                    fbond += 48.0 * epsilon[type] * sr6 * (sr6 - 0.5) / rsq;
                }

                // energy

                // apply force to each of 2 atoms

                if (newton_pair || i < nlocal) {
                    fxtmp += delx * fbond;
                    fytmp += dely * fbond;
                    fztmp += delz * fbond;
                }

                if (newton_pair || i2 < nlocal) {
                    if (USE_ATOMIC_FETCH_ADD) {
                        __atomic_fetch_add(&f[i2].x, -delx*fbond, __ATOMIC_RELAXED);
                        __atomic_fetch_add(&f[i2].y, -dely*fbond, __ATOMIC_RELAXED);
                        __atomic_fetch_add(&f[i2].z, -delz*fbond, __ATOMIC_RELAXED);
                    } else {
                        spinlocks[i2].lock();
                        f[i2].x -= delx * fbond;
                        f[i2].y -= dely * fbond;
                        f[i2].z -= delz * fbond;
                        spinlocks[i2].unlock();
                    }
                }
            }

            if (USE_ATOMIC_FETCH_ADD) {
                __atomic_fetch_add(&f[i].x, fxtmp, __ATOMIC_RELAXED);
                __atomic_fetch_add(&f[i].y, fytmp, __ATOMIC_RELAXED);
                __atomic_fetch_add(&f[i].z, fztmp, __ATOMIC_RELAXED);
            } else {
                spinlocks[i].lock();
                f[i].x += fxtmp;
                f[i].y += fytmp;
                f[i].z += fztmp;
                spinlocks[i].unlock();
            }
        }
    }

    void stencil_md_fuse_force_computation_atomics_affinity(queue_info& zoid, int timestep, Atom* next, Neighbor* neigh_next, Force* next_force, Modify* modify_) {
        const auto * _noalias const x = (dbl3_t_stencil_md *) next->x[0];
        auto * _noalias const f = (dbl3_t_stencil_md *) next->eval_f_stencil_md[0];

        // auto pair = (PairLJCutOMP*) next_force->pair;
        auto pair = (PairLJCut*) next_force->pair;
        auto bond = (BondFENE*) next_force->bond;

        const auto* _noalias bondlist = neigh_next->atom_bondlist;

        const int * _noalias const ilist = pair->list->ilist;
        const int * _noalias const numneigh = pair->list->numneigh;
        const int * const * const firstneigh = pair->list->firstneigh;
        const double * _noalias const special_lj = force->special_lj;

        auto* spinlocks = next->spinlocks;

        assert(pair->list->inum == next->nlocal);

        const auto* cutsq = pair->cutsq;
        const auto* offset = pair->offset;
        const auto* lj1 = pair->lj1;
        const auto* lj2 = pair->lj2;
        const auto* lj3 = pair->lj3;
        const auto* lj4 = pair->lj4;
        auto newton_pair = force->newton_pair;

        const auto* _noalias const sigma = bond->sigma;
        const auto* _noalias const epsilon = bond->epsilon;
        const auto* _noalias const r0 = bond->r0;
        const auto* _noalias const k = bond->k;

        const int* _noalias const atom_type = next->type;

        const int nlocal = next->nlocal;

        int num_chunks = nlocal / MODIFY_GRAINSIZE + 1;
        int num_workers = __cilkrts_get_nworkers();
        auto claimed = next->claimed;
        auto claimed_int = next->claimed_int;
        auto claimed_flag = next->claimed_flag;

        num_chunks = num_workers;
        int chunk_size = nlocal / num_chunks + 1;

        if (num_chunks == 1) {
            for (int i = 0; i < nlocal; i++) {
                const int itype = atom_type[i];

                const int *_noalias const jlist = firstneigh[i];
                const double *_noalias const cutsqi = cutsq[itype];
                const double *_noalias const offseti = offset[itype];
                const double *_noalias const lj1i = lj1[itype];
                const double *_noalias const lj2i = lj2[itype];
                const double *_noalias const lj3i = lj3[itype];
                const double *_noalias const lj4i = lj4[itype];

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
                    double factor_lj = special_lj[pair->sbmask(j)];
                    j &= NEIGHMASK;

                    double delx = xtmp - x[j].x;
                    double dely = ytmp - x[j].y;
                    double delz = ztmp - x[j].z;
                    double rsq = delx * delx + dely * dely + delz * delz;
                    int jtype = atom_type[j];

                    if (rsq < cutsqi[jtype]) {
                        double r2inv = 1.0 / rsq;
                        double r6inv = r2inv * r2inv * r2inv;
                        double forcelj = r6inv * (lj1i[jtype] * r6inv - lj2i[jtype]);
                        double fpair = factor_lj * forcelj * r2inv;

                        fxtmp += delx * fpair;
                        fytmp += dely * fpair;
                        fztmp += delz * fpair;

                        if (newton_pair || j < nlocal) {
                            f[j].x -= delx * fpair;
                            f[j].y -= dely * fpair;
                            f[j].z -= delz * fpair;
                        }
                    }
                }

                auto &lst_bonds = bondlist[i];
                for (int j = 0; j < lst_bonds.size(); j++) {
                    auto &bond_info = lst_bonds[j];
                    int i2 = bond_info.first;
                    int type = bond_info.second;

                    double delx = xtmp - x[i2].x;
                    double dely = ytmp - x[i2].y;
                    double delz = ztmp - x[i2].z;

                    double rsq = delx * delx + dely * dely + delz * delz;
                    double r0sq = r0[type] * r0[type];
                    double rlogarg = 1.0 - rsq / r0sq;

                    if (rlogarg < 0.1) {
                        error->warning(FLERR, "FENE bond too long: {} {} {} {:.8}",
                                       update->ntimestep, atom->tag[i], atom->tag[i2], sqrt(rsq));
                        //                            if (check_error_thr((rlogarg <= -3.0),tid,FLERR,"Bad FENE bond"))
                        //                                return;
                        assert(false);

                        rlogarg = 0.1;
                    }

                    double fbond = -k[type] / rlogarg;

                    // force from LJ term
                    double sr2 = 0.0;
                    double sr6 = 0.0;

                    if (rsq < MathConst::MY_CUBEROOT2 * sigma[type] * sigma[type]) {
                        sr2 = sigma[type] * sigma[type] / rsq;
                        sr6 = sr2 * sr2 * sr2;
                        fbond += 48.0 * epsilon[type] * sr6 * (sr6 - 0.5) / rsq;
                    }

                    // energy

                    // apply force to each of 2 atoms

                    if (newton_pair || i < nlocal) {
                        fxtmp += delx * fbond;
                        fytmp += dely * fbond;
                        fztmp += delz * fbond;
                    }

                    if (newton_pair || i2 < nlocal) {
                        f[i2].x -= delx * fbond;
                        f[i2].y -= dely * fbond;
                        f[i2].z -= delz * fbond;
                    }
                }

                f[i].x += fxtmp;
                f[i].y += fytmp;
                f[i].z += fztmp;
            }

            return;
        }

        #pragma cilk grainsize 1
        cilk_for (int ii = 0; ii < num_chunks; ii++) {
            int start_chunk = __cilkrts_get_worker_number() * num_chunks / num_workers;
            for (int c = 0; c < num_chunks; ++c) {
                int s = (c + start_chunk) % num_chunks;

                if (claimed_flag[s].test()) {
                    continue;
                }

                if (!claimed_flag[s].test_and_set(std::memory_order_relaxed)) {
                    // for (int i = s * MODIFY_GRAINSIZE; i < (s + 1) * MODIFY_GRAINSIZE && i < nlocal; i++) {
                    for (int i = s * chunk_size; i < (s + 1) * chunk_size && i < nlocal; i++) {
                        const int itype = atom_type[i];

                        const int *_noalias const jlist = firstneigh[i];
                        const double *_noalias const cutsqi = cutsq[itype];
                        const double *_noalias const offseti = offset[itype];
                        const double *_noalias const lj1i = lj1[itype];
                        const double *_noalias const lj2i = lj2[itype];
                        const double *_noalias const lj3i = lj3[itype];
                        const double *_noalias const lj4i = lj4[itype];

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
                            double factor_lj = special_lj[pair->sbmask(j)];
                            j &= NEIGHMASK;

                            double delx = xtmp - x[j].x;
                            double dely = ytmp - x[j].y;
                            double delz = ztmp - x[j].z;
                            double rsq = delx * delx + dely * dely + delz * delz;
                            int jtype = atom_type[j];

                            if (rsq < cutsqi[jtype]) {
                                double r2inv = 1.0 / rsq;
                                double r6inv = r2inv * r2inv * r2inv;
                                double forcelj = r6inv * (lj1i[jtype] * r6inv - lj2i[jtype]);
                                double fpair = factor_lj * forcelj * r2inv;

                                fxtmp += delx * fpair;
                                fytmp += dely * fpair;
                                fztmp += delz * fpair;

                                if (newton_pair || j < nlocal) {
                                    spinlocks[j].lock();
                                    f[j].x -= delx * fpair;
                                    f[j].y -= dely * fpair;
                                    f[j].z -= delz * fpair;
                                    spinlocks[j].unlock();
                                }
                            }
                        }

                        auto &lst_bonds = bondlist[i];
                        for (int j = 0; j < lst_bonds.size(); j++) {
                            auto &bond_info = lst_bonds[j];
                            int i2 = bond_info.first;
                            int type = bond_info.second;

                            double delx = xtmp - x[i2].x;
                            double dely = ytmp - x[i2].y;
                            double delz = ztmp - x[i2].z;

                            double rsq = delx * delx + dely * dely + delz * delz;
                            double r0sq = r0[type] * r0[type];
                            double rlogarg = 1.0 - rsq / r0sq;

                            if (rlogarg < 0.1) {
                                error->warning(FLERR, "FENE bond too long: {} {} {} {:.8}",
                                               update->ntimestep, atom->tag[i], atom->tag[i2], sqrt(rsq));
                                //                            if (check_error_thr((rlogarg <= -3.0),tid,FLERR,"Bad FENE bond"))
                                //                                return;
                                assert(false);

                                rlogarg = 0.1;
                            }

                            double fbond = -k[type] / rlogarg;

                            // force from LJ term
                            double sr2 = 0.0;
                            double sr6 = 0.0;

                            if (rsq < MathConst::MY_CUBEROOT2 * sigma[type] * sigma[type]) {
                                sr2 = sigma[type] * sigma[type] / rsq;
                                sr6 = sr2 * sr2 * sr2;
                                fbond += 48.0 * epsilon[type] * sr6 * (sr6 - 0.5) / rsq;
                            }

                            // energy

                            // apply force to each of 2 atoms

                            if (newton_pair || i < nlocal) {
                                fxtmp += delx * fbond;
                                fytmp += dely * fbond;
                                fztmp += delz * fbond;
                            }

                            if (newton_pair || i2 < nlocal) {
                                spinlocks[i2].lock();
                                f[i2].x -= delx * fbond;
                                f[i2].y -= dely * fbond;
                                f[i2].z -= delz * fbond;
                                spinlocks[i2].unlock();
                            }
                        }

                        spinlocks[i].lock();
                        f[i].x += fxtmp;
                        f[i].y += fytmp;
                        f[i].z += fztmp;
                        spinlocks[i].unlock();
                    }
                }
            }
        }

        for (int i = 0; i < num_chunks; i++) {
            claimed_flag[i].clear(std::memory_order_relaxed);
        }
    }

    template <bool curr_dt>
    void fuse_force_computation_atomics(queue_info& zoid, int timestep, Atom* next, Neighbor* neigh_next, Force* next_force, Modify* modify_) {
        // memset(&next->eval_f_stencil_md[next->nlocal][0], 0, (next->nghost) * 3 * sizeof(double));
        for (int k = 0; k < next->nlocal + next->nghost; k++) {
            assert(fabs(next->eval_f_stencil_md[k][0]) < 1e-6);
            assert(fabs(next->eval_f_stencil_md[k][1]) < 1e-6);
            assert(fabs(next->eval_f_stencil_md[k][2]) < 1e-6);
        }

        // begin force computation, inline lj_cut and bond_fene
        assert(PURELY_LOCAL_POTENTIAL);

        const auto * _noalias const x = (dbl3_t_stencil_md *) next->x[0];
        auto * _noalias const f = (dbl3_t_stencil_md *) next->eval_f_stencil_md[0];
        const int * _noalias const type = next->type;
        const double * _noalias const special_lj = force->special_lj;

        auto pair = (PairLJCut*) next_force->pair;
        auto bond = (BondFENE*) next_force->bond;

        const int * _noalias const ilist = pair->list->ilist;
        const int * _noalias const numneigh = pair->list->numneigh;
        const int * const * const firstneigh = pair->list->firstneigh;

        const auto* cutsq = pair->cutsq;
        const auto* offset = pair->offset;
        const auto* lj1 = pair->lj1;
        const auto* lj2 = pair->lj2;
        const auto* lj3 = pair->lj3;
        const auto* lj4 = pair->lj4;
        auto newton_pair = force->newton_pair;

        const auto* sigma = bond->sigma;
        const auto* epsilon = bond->epsilon;
        const auto* r0 = bond->r0;
        const auto* k = bond->k;

        int nlocal = next->nlocal;

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
                double factor_lj = special_lj[pair->sbmask(j)];
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
                        __atomic_fetch_add(&f[j].x, -delx*fpair, __ATOMIC_RELAXED);
                        __atomic_fetch_add(&f[j].y, -dely*fpair, __ATOMIC_RELAXED);
                        __atomic_fetch_add(&f[j].z, -delz*fpair, __ATOMIC_RELAXED);
                    }
                }
            }

            auto& lst_bonds = neigh_next->atom_bondlist[i];
            for (int j = 0; j < lst_bonds.size(); j++) {
                auto& bond_info = lst_bonds[j];
                int i2 = bond_info.first;
                int type = bond_info.second;

                // double delx = x[i].x - x[i2].x;
                // double dely = x[i].y - x[i2].y;
                // double delz = x[i].z - x[i2].z;
                double delx = xtmp - x[i2].x;
                double dely = ytmp - x[i2].y;
                double delz = ztmp - x[i2].z;

                double rsq = delx * delx + dely * dely + delz * delz;
                double r0sq = r0[type] * r0[type];
                double rlogarg = 1.0 - rsq / r0sq;

                if (rlogarg < 0.1) {
                    error->warning(FLERR, "FENE bond too long: {} {} {} {:.8}",
                                   update->ntimestep, atom->tag[i], atom->tag[i2], sqrt(rsq));
//                            if (check_error_thr((rlogarg <= -3.0),tid,FLERR,"Bad FENE bond"))
//                                return;
                    assert(false);

                    rlogarg = 0.1;
                }

                double fbond = -k[type] / rlogarg;

                // force from LJ term
                double sr2 = 0.0;
                double sr6 = 0.0;

                if (rsq < MathConst::MY_CUBEROOT2 * sigma[type] * sigma[type]) {
                    sr2 = sigma[type] * sigma[type] / rsq;
                    sr6 = sr2 * sr2 * sr2;
                    fbond += 48.0 * epsilon[type] * sr6 * (sr6 - 0.5) / rsq;
                }

                // energy

                /*
                if (eflag) {
                    ebond = -0.5 * k[type] * r0sq * log(rlogarg);
                    if (rsq < MY_CUBEROOT2 * sigma[type] * sigma[type])
                        ebond += 4.0 * epsilon[type] * sr6 * (sr6 - 1.0) + epsilon[type];
                }
                */

                // apply force to each of 2 atoms

                if (newton_pair || i < nlocal) {
                    // f[i].x += delx * fbond;
                    // f[i].y += dely * fbond;
                    // f[i].z += delz * fbond;
                    fxtmp += delx * fbond;
                    fytmp += dely * fbond;
                    fztmp += delz * fbond;
                }

                if (newton_pair || i2 < nlocal) {
                    __atomic_fetch_add(&f[i2].x, -delx*fbond, __ATOMIC_RELAXED);
                    __atomic_fetch_add(&f[i2].y, -dely*fbond, __ATOMIC_RELAXED);
                    __atomic_fetch_add(&f[i2].z, -delz*fbond, __ATOMIC_RELAXED);
                }
            }

            __atomic_fetch_add(&f[i].x, fxtmp, __ATOMIC_RELAXED);
            __atomic_fetch_add(&f[i].y, fytmp, __ATOMIC_RELAXED);
            __atomic_fetch_add(&f[i].z, fztmp, __ATOMIC_RELAXED);
        }
    }

    template <bool curr_dt>
    void fuse_force_computation(queue_info& zoid, int timestep, Atom* next, Neighbor* neigh_next, Force* next_force, Modify* modify_) {
        memset(&next->eval_f_stencil_md[next->nlocal][0], 0, (next->nghost) * 3 * sizeof(double));
        for (int k = 0; k < next->nlocal + next->nghost; k++) {
            assert(fabs(next->eval_f_stencil_md[k][0]) < 1e-6);
            assert(fabs(next->eval_f_stencil_md[k][1]) < 1e-6);
            assert(fabs(next->eval_f_stencil_md[k][2]) < 1e-6);
        }

        // begin force computation, inline lj_cut and bond_fene
        assert(PURELY_LOCAL_POTENTIAL);

        // const auto * _noalias const x = (dbl3_t_stencil_md *) atom_->x[0];
        // auto * _noalias const f = (dbl3_t_stencil_md *) atom_->eval_f_stencil_md[0];
        // const int * _noalias const type = atom_->type;
        const auto * _noalias const x = (dbl3_t_stencil_md *) next->x[0];
        auto * _noalias const f = (dbl3_t_stencil_md *) next->eval_f_stencil_md[0];
        const int * _noalias const type = next->type;
        const double * _noalias const special_lj = force->special_lj;

        auto pair = (PairLJCut*) next_force->pair;
        auto bond = (BondFENE*) next_force->bond;

        const int * _noalias const ilist = pair->list->ilist;
        const int * _noalias const numneigh = pair->list->numneigh;
        const int * const * const firstneigh = pair->list->firstneigh;

        const auto* cutsq = pair->cutsq;
        const auto* offset = pair->offset;
        const auto* lj1 = pair->lj1;
        const auto* lj2 = pair->lj2;
        const auto* lj3 = pair->lj3;
        const auto* lj4 = pair->lj4;
        auto newton_pair = force->newton_pair;

        const auto* sigma = bond->sigma;
        const auto* epsilon = bond->epsilon;
        const auto* r0 = bond->r0;
        const auto* k = bond->k;

        int nlocal = next->nlocal;

        if (nlocal < 512) {
            for (int ii = 0; ii < nlocal; ii++) {
                const int i = ilist[ii];
                assert(i == ii);
                const int itype = type[i];

                const int *_noalias const jlist = firstneigh[i];
                const double *_noalias const cutsqi = cutsq[itype];
                const double *_noalias const offseti = offset[itype];
                const double *_noalias const lj1i = lj1[itype];
                const double *_noalias const lj2i = lj2[itype];
                const double *_noalias const lj3i = lj3[itype];
                const double *_noalias const lj4i = lj4[itype];

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
                    double factor_lj = special_lj[pair->sbmask(j)];
                    j &= NEIGHMASK;

                    double delx = xtmp - x[j].x;
                    double dely = ytmp - x[j].y;
                    double delz = ztmp - x[j].z;
                    double rsq = delx * delx + dely * dely + delz * delz;
                    int jtype = type[j];

                    if (rsq < cutsqi[jtype]) {
                        double r2inv = 1.0 / rsq;
                        double r6inv = r2inv * r2inv * r2inv;
                        double forcelj = r6inv * (lj1i[jtype] * r6inv - lj2i[jtype]);
                        double fpair = factor_lj * forcelj * r2inv;

                        fxtmp += delx * fpair;
                        fytmp += dely * fpair;
                        fztmp += delz * fpair;

                        if (newton_pair || j < nlocal) {
                            f[j].x -= delx * fpair;
                            f[j].y -= dely * fpair;
                            f[j].z -= delz * fpair;
                        }

//                            if (EFLAG) {
//                                evdwl = r6inv*(lj3i[jtype]*r6inv-lj4i[jtype]) - offseti[jtype];
//                                evdwl *= factor_lj;
//                            }
//
//                            if (EVFLAG) {
//                                ev_tally_thr(this, i, j, nlocal, NEWTON_PAIR,
//                                             evdwl, 0.0, fpair, delx, dely, delz, thr);
//                            }
                    }
                }

                // f[i].x += fxtmp;
                // f[i].y += fytmp;
                // f[i].z += fztmp;

                auto& lst_bonds = neigh_next->atom_bondlist[i];
                for (int j = 0; j < lst_bonds.size(); j++) {
                    auto& bond_info = lst_bonds[j];
                    int i2 = bond_info.first;
                    int bond_type = bond_info.second;

                    // double delx = x[i].x - x[i2].x;
                    // double dely = x[i].y - x[i2].y;
                    // double delz = x[i].z - x[i2].z;
                    double delx = xtmp - x[i2].x;
                    double dely = ytmp - x[i2].y;
                    double delz = ztmp - x[i2].z;

                    double rsq = delx * delx + dely * dely + delz * delz;
                    double r0sq = r0[bond_type] * r0[bond_type];
                    double rlogarg = 1.0 - rsq / r0sq;

                    if (rlogarg < 0.1) {
                        error->warning(FLERR, "FENE bond too long: {} {} {} {:.8}",
                                       update->ntimestep, atom->tag[i], atom->tag[i2], sqrt(rsq));
//                            if (check_error_thr((rlogarg <= -3.0),tid,FLERR,"Bad FENE bond"))
//                                return;
                        assert(false);

                        rlogarg = 0.1;
                    }

                    double fbond = -k[bond_type] / rlogarg;

                    // force from LJ term
                    double sr2 = 0.0;
                    double sr6 = 0.0;

                    if (rsq < MathConst::MY_CUBEROOT2 * sigma[bond_type] * sigma[bond_type]) {
                        sr2 = sigma[bond_type] * sigma[bond_type] / rsq;
                        sr6 = sr2 * sr2 * sr2;
                        fbond += 48.0 * epsilon[bond_type] * sr6 * (sr6 - 0.5) / rsq;
                    }

                    // energy

                    /*
                    if (eflag) {
                        ebond = -0.5 * k[type] * r0sq * log(rlogarg);
                        if (rsq < MY_CUBEROOT2 * sigma[type] * sigma[type])
                            ebond += 4.0 * epsilon[type] * sr6 * (sr6 - 1.0) + epsilon[type];
                    }
                    */

                    // apply force to each of 2 atoms

                    if (newton_pair || i < nlocal) {
                        // f[i].x += delx * fbond;
                        // f[i].y += dely * fbond;
                        // f[i].z += delz * fbond;
                        fxtmp += delx * fbond;
                        fytmp += dely * fbond;
                        fztmp += delz * fbond;
                    }

                    if (newton_pair || i2 < nlocal) {
                        f[i2].x -= delx * fbond;
                        f[i2].y -= dely * fbond;
                        f[i2].z -= delz * fbond;
                    }
                }

                f[i].x += fxtmp;
                f[i].y += fytmp;
                f[i].z += fztmp;

            }

            return;
        }

        for (int dep = 0; dep < next->num_deps; dep++) {
            auto& partitions_at_dep = next->dep_to_partitions[dep];

            /*
            std::set<int> idxs_touched_at_dep;
            std::map<int, std::array<int, 3>> idx_to_partition;
            std::map<int, int> idx_to_touched_neighbor;
            std::map<int, std::tuple<int, int, int>> idx_to_bin;
            */

            // int num_atoms = 0;
            cilk_for (int d = 0; d < partitions_at_dep.size(); d++) {
                auto& partition = partitions_at_dep[d];
                auto& bins = next->partition_to_bins[partition[0]][partition[1]][partition[2]];

                for (int b = 0; b < bins.size(); b++) {
                    auto& bin = bins[b];
                    auto bin_idx = get_bin_idx(bin);
                    auto& idxs = next->bin_to_local_idxs[bin];
                    int start = idxs[0];
                    // num_atoms += idxs.size();
                    for (int idx = 0; idx < idxs.size(); idx++) {
                        // int ii = idxs[idx];
                        int ii = start + idx;
                        assert(ii == idxs[0] + idx);
                        assert(ii >= 0 && ii < nlocal);
                        const int i = ilist[ii];
                        assert(i == ii);
                        const int itype = type[i];

                        const int *_noalias const jlist = firstneigh[i];
                        const double *_noalias const cutsqi = cutsq[itype];
                        const double *_noalias const offseti = offset[itype];
                        const double *_noalias const lj1i = lj1[itype];
                        const double *_noalias const lj2i = lj2[itype];
                        const double *_noalias const lj3i = lj3[itype];
                        const double *_noalias const lj4i = lj4[itype];

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
                            double factor_lj = special_lj[pair->sbmask(j)];
                            j &= NEIGHMASK;

                            double delx = xtmp - x[j].x;
                            double dely = ytmp - x[j].y;
                            double delz = ztmp - x[j].z;
                            double rsq = delx * delx + dely * dely + delz * delz;
                            int jtype = type[j];

                            if (rsq < cutsqi[jtype]) {
                                double r2inv = 1.0 / rsq;
                                double r6inv = r2inv * r2inv * r2inv;
                                double forcelj = r6inv * (lj1i[jtype] * r6inv - lj2i[jtype]);
                                double fpair = factor_lj * forcelj * r2inv;

                                fxtmp += delx * fpair;
                                fytmp += dely * fpair;
                                fztmp += delz * fpair;

                                if (newton_pair || j < nlocal) {
                                    f[j].x -= delx * fpair;
                                    f[j].y -= dely * fpair;
                                    f[j].z -= delz * fpair;

                                    /*
                                    if (idxs_touched_at_dep.find(j) != idxs_touched_at_dep.end() && idx_to_partition[j] != partition) {
                                        std::cout << "curr pos: " << x[j].x << " " << x[j].y << " " << x[j].z << std::endl;
                                        std::cout << "neighbor pos: " << x[i].x << " " << x[i].y << " " << x[i].z << std::endl;
                                        int other_neighbor_idx = idx_to_touched_neighbor[j];
                                        std::map<int, std::string> m;
                                        m[LEFT] = "LEFT";
                                        m[RIGHT] = "RIGHT";
                                        m[MIDDLE] = "MIDDLE";
                                        std::cout << "other neighbor pos: " << x[other_neighbor_idx].x << " " << x[other_neighbor_idx].y << " " << x[other_neighbor_idx].z << std::endl;
                                        std::cout << "curr partition: " << m[partition[0]] << " " << m[partition[1]] << " " << m[partition[2]] << std::endl;
                                        std::cout << "other partition: " << m[idx_to_partition[j][0]] << " " << m[idx_to_partition[j][1]] << " " << m[idx_to_partition[j][2]] << std::endl;
                                        std::cout << "curr bin: " << std::get<0>(bin) << " " << std::get<1>(bin) << " " << std::get<2>(bin) << std::endl;
                                        std::cout << "other bin: " << std::get<0>(idx_to_bin[j]) << " " << std::get<1>(idx_to_bin[j]) << " " << std::get<2>(idx_to_bin[j]) << std::endl;
                                        std::cout << "curr idx: " << j << " nlocal: " << nlocal << std::endl;
                                        assert(false);
                                    }

                                    idxs_touched_at_dep.insert(j);
                                    idx_to_partition[j] = partition;
                                    idx_to_bin[j] = bin;
                                    idx_to_touched_neighbor[j] = i;
                                    */
                                }

//                            if (EFLAG) {
//                                evdwl = r6inv*(lj3i[jtype]*r6inv-lj4i[jtype]) - offseti[jtype];
//                                evdwl *= factor_lj;
//                            }
//
//                            if (EVFLAG) {
//                                ev_tally_thr(this, i, j, nlocal, NEWTON_PAIR,
//                                             evdwl, 0.0, fpair, delx, dely, delz, thr);
//                            }
                            }
                        }

                        // f[i].x += fxtmp;
                        // f[i].y += fytmp;
                        // f[i].z += fztmp;

                        auto& lst_bonds = neigh_next->atom_bondlist[i];
                        for (int j = 0; j < lst_bonds.size(); j++) {
                            auto& bond_info = lst_bonds[j];
                            int i2 = bond_info.first;
                            int type = bond_info.second;

                            // double delx = x[i].x - x[i2].x;
                            // double dely = x[i].y - x[i2].y;
                            // double delz = x[i].z - x[i2].z;
                            double delx = xtmp - x[i2].x;
                            double dely = ytmp - x[i2].y;
                            double delz = ztmp - x[i2].z;

                            double rsq = delx * delx + dely * dely + delz * delz;
                            double r0sq = r0[type] * r0[type];
                            double rlogarg = 1.0 - rsq / r0sq;

                            if (rlogarg < 0.1) {
                                error->warning(FLERR, "FENE bond too long: {} {} {} {:.8}",
                                               update->ntimestep, atom->tag[i], atom->tag[i2], sqrt(rsq));
//                            if (check_error_thr((rlogarg <= -3.0),tid,FLERR,"Bad FENE bond"))
//                                return;
                                assert(false);

                                rlogarg = 0.1;
                            }

                            double fbond = -k[type] / rlogarg;

                            // force from LJ term
                            double sr2 = 0.0;
                            double sr6 = 0.0;

                            if (rsq < MathConst::MY_CUBEROOT2 * sigma[type] * sigma[type]) {
                                sr2 = sigma[type] * sigma[type] / rsq;
                                sr6 = sr2 * sr2 * sr2;
                                fbond += 48.0 * epsilon[type] * sr6 * (sr6 - 0.5) / rsq;
                            }

                            // energy

                            /*
                            if (eflag) {
                                ebond = -0.5 * k[type] * r0sq * log(rlogarg);
                                if (rsq < MY_CUBEROOT2 * sigma[type] * sigma[type])
                                    ebond += 4.0 * epsilon[type] * sr6 * (sr6 - 1.0) + epsilon[type];
                            }
                            */

                            // apply force to each of 2 atoms

                            if (newton_pair || i < nlocal) {
                                // f[i].x += delx * fbond;
                                // f[i].y += dely * fbond;
                                // f[i].z += delz * fbond;
                                fxtmp += delx * fbond;
                                fytmp += dely * fbond;
                                fztmp += delz * fbond;
                            }

                            if (newton_pair || i2 < nlocal) {
                                f[i2].x -= delx * fbond;
                                f[i2].y -= dely * fbond;
                                f[i2].z -= delz * fbond;

                                /*
                                if (idxs_touched_at_dep.find(i2) != idxs_touched_at_dep.end() && idx_to_partition[i2] != partition) {
                                    std::cout << "curr pos: " << x[i2].x << " " << x[i2].y << " " << x[i2].z << std::endl;
                                    std::cout << "neighbor pos: " << x[i].x << " " << x[i].y << " " << x[i].z << std::endl;
                                    int other_neighbor_idx = idx_to_touched_neighbor[i2];
                                    std::map<int, std::string> m;
                                    m[LEFT] = "LEFT";
                                    m[RIGHT] = "RIGHT";
                                    m[MIDDLE] = "MIDDLE";
                                    std::cout << "other neighbor pos: " << x[other_neighbor_idx].x << " " << x[other_neighbor_idx].y << " " << x[other_neighbor_idx].z << std::endl;
                                    std::cout << "curr partition: " << m[partition[0]] << " " << m[partition[1]] << " " << m[partition[2]] << std::endl;
                                    std::cout << "other partition: " << m[idx_to_partition[i2][0]] << " " << m[idx_to_partition[i2][1]] << " " << m[idx_to_partition[i2][2]] << std::endl;
                                    std::cout << "curr bin: " << std::get<0>(bin) << " " << std::get<1>(bin) << " " << std::get<2>(bin) << std::endl;
                                    std::cout << "other bin: " << std::get<0>(idx_to_bin[i2]) << " " << std::get<1>(idx_to_bin[i2]) << " " << std::get<2>(idx_to_bin[i2]) << std::endl;
                                    std::cout << "curr idx: " << i2 << " nlocal: " << nlocal << std::endl;
                                    assert(false);
                                }

                                idxs_touched_at_dep.insert(i2);
                                idx_to_partition[i2] = partition;
                                idx_to_bin[i2] = bin;
                                idx_to_touched_neighbor[i2] = i;
                                */
                            }
                        }

                        f[i].x += fxtmp;
                        f[i].y += fytmp;
                        f[i].z += fztmp;
                    }

                    /*
                    post_force_stencil_md(bin, next, modify_);

                    auto& recv_force_zoids = zoid.bin_to_force_comm[timestep][next->bin_to_local_bins_idx[bin]];
                    for (int j = 0; j < recv_force_zoids.size(); j++) {
                        int recv_zoid_num = recv_force_zoids[j];
                        auto& recv_zoid = curr_dt ? lmp->zoid_num_to_zoid[recv_zoid_num]
                                                  : lmp->zoid_num_to_zoid_next_dt[recv_zoid_num];
                        Atom* other_atom = curr_dt ? lmp->atom_stencil_md[recv_zoid_num][timestep]
                                                   : lmp->atom_stencil_md[recv_zoid_num][NUM_TIMESTEPS_IN_PARALLEL - timestep];

                        const auto *_noalias const send_f = (dbl3_t_stencil_md *) other_atom->eval_f_stencil_md[0];

                        auto send_bin_to_idx = recv_zoid.bin_to_idx[timestep];
                        auto send_bin_to_size = recv_zoid.bin_to_size[timestep];

                        int send_bin_idx = send_bin_to_idx[bin_idx];
                        int send_size = send_bin_to_size[bin_idx];

                        int recv_bin_idx = recv_bin_to_idx[bin_idx];
                        int recv_size = recv_bin_to_size[bin_idx];

                        assert(send_size != -1);
                        assert(send_size == recv_size);

                        for (int h = 0; h < send_size; h++) {
                            int send_idx = send_bin_idx + h;
                            int recv_idx = recv_bin_idx + h;
                            tagint src_tag = other_atom->tag[send_idx];
                            tagint dst_tag = next->tag[recv_idx];
                            assert(src_tag == dst_tag);
                            recv_f[recv_idx].x += send_f[send_idx].x;
                            recv_f[recv_idx].y += send_f[send_idx].y;
                            recv_f[recv_idx].z += send_f[send_idx].z;
                        }
                    }

                    final_integrate_stencil_md(bin, next);
                    */
                }
            }
        }
    }

    template <bool curr_dt>
    void fuse_post_force_stencil_md(queue_info& zoid, int timestep, Atom* atom_, Modify* modify_) {
        auto recv_bin_to_idx = zoid.bin_to_idx[timestep];
        auto recv_bin_to_size = zoid.bin_to_size[timestep];

        auto *_noalias const recv_f = (dbl3_t_stencil_md *) atom_->f[0];

        #pragma cilk grainsize 32
        cilk_for (int i = 0; i < atom_->local_bins.size(); i++) {
            auto& bin = atom_->local_bins[i];
            auto& local_idxs = atom_->local_bins_idxs[i];
            auto bin_idx = get_bin_idx(bin);
            auto& recv_force_zoids = zoid.bin_to_force_comm[timestep][i];

            for (int j = 0; j < recv_force_zoids.size(); j++) {
                int recv_zoid_num = recv_force_zoids[j];
                auto& recv_zoid = curr_dt ? lmp->zoid_num_to_zoid[recv_zoid_num]
                                          : lmp->zoid_num_to_zoid_next_dt[recv_zoid_num];
                Atom* other_atom = curr_dt ? lmp->atom_stencil_md[recv_zoid_num][timestep]
                                           : lmp->atom_stencil_md[recv_zoid_num][NUM_TIMESTEPS_IN_PARALLEL - timestep];

                auto *_noalias const send_f = (dbl3_t_stencil_md *) other_atom->eval_f_stencil_md[0];

                auto send_bin_to_idx = recv_zoid.bin_to_idx[timestep];
                auto send_bin_to_size = recv_zoid.bin_to_size[timestep];

                int send_bin_idx = send_bin_to_idx[bin_idx];
                int send_size = send_bin_to_size[bin_idx];

                int recv_bin_idx = recv_bin_to_idx[bin_idx];
                int recv_size = recv_bin_to_size[bin_idx];

                assert(send_size != -1);
                assert(send_size == recv_size);

                for (int k = 0; k < send_size; k++) {
                    int send_idx = send_bin_idx + k;
                    int recv_idx = recv_bin_idx + k;
                    if (DEBUG_SEND_RECV_DATA) {
                        tagint src_tag = other_atom->tag[send_idx];
                        tagint dst_tag = atom_->tag[recv_idx];
                        assert(src_tag == dst_tag);
                    }
                    recv_f[recv_idx].x += send_f[send_idx].x;
                    recv_f[recv_idx].y += send_f[send_idx].y;
                    recv_f[recv_idx].z += send_f[send_idx].z;

                    send_f[send_idx].x = 0;
                    send_f[send_idx].y = 0;
                    send_f[send_idx].z = 0;
                }
            }

            post_force_stencil_md(local_idxs, atom_, modify_);
            final_integrate_stencil_md(local_idxs, atom_);
        }
    }

    void sort_local_bins(queue_info& zoid, int timestep, Atom* curr, Atom* next) {
        std::vector<size_t> local_bins;
        std::vector<size_t> ghost_bins;

        std::map<int, int> curr_tag_to_idx;
        std::map<int, int> next_tag_to_idx;

        for (int i = 0; i < curr->nlocal + curr->nghost; i++) {
            curr_tag_to_idx[curr->tag[i]] = i;
        }

        for (int i = 0; i < next->nlocal + next->nghost; i++) {
            next_tag_to_idx[next->tag[i]] = i;
        }

        auto& bin_bounds = GET_BOUNDS(true, 0);
        for (int i = 0; i < curr->nlocal; i++) {
            int next_idx = next_tag_to_idx.at(curr->tag[i]);
            bool is_ghost = (next_idx >= next->nlocal);
            auto bin = get_bin(bin_bounds, curr->x[i], domain->boxlo, domain->boxhi);
            auto bin_idx = get_bin_idx(bin);
            if (is_ghost) {
                ghost_bins.push_back(bin_idx);
            } else {
                local_bins.push_back(bin_idx);
            }
        }

        for (auto& local_bin : local_bins) {
            curr->sorted_local_bin_indices.push_back(local_bin);
        }
        for (auto& ghost_bin : ghost_bins) {
            curr->sorted_local_bin_indices.push_back(ghost_bin);
        }
    }

    void sort_ghost_bins(queue_info& zoid, int timestep, Atom* curr, Atom* prev) {
        std::map<int, int> curr_tag_to_idx;
        std::map<int, int> prev_tag_to_idx;

        for (int i = 0; i < curr->nlocal + curr->nghost; i++) {
            curr_tag_to_idx[curr->tag[i]] = i;
        }

        for (int i = 0; i < prev->nlocal + prev->nghost; i++) {
            prev_tag_to_idx[prev->tag[i]] = i;
        }

        std::vector<size_t> local_bins;
        std::vector<size_t> ghost_bins;
        auto& bin_bounds = GET_BOUNDS(true, 0);
        for (int i = curr->nlocal; i < curr->nlocal + curr->nghost; i++) {
            int prev_idx = -1;
            if (prev_tag_to_idx.count(curr->tag[i])) {
                prev_idx = prev_tag_to_idx.at(curr->tag[i]);
            }
            bool is_local = (prev_idx != -1 && prev_idx < prev->nlocal);
            auto bin = get_bin(bin_bounds, curr->x[i], domain->boxlo, domain->boxhi);
            auto bin_idx = get_bin_idx(bin);
            if (is_local) {
                local_bins.push_back(bin_idx);
            } else {
                ghost_bins.push_back(bin_idx);
            }
        }

        for (auto& local_bin : local_bins) {
            curr->sorted_ghost_bin_indices.push_back(local_bin);
        }
        for (auto& ghost_bin : ghost_bins) {
            curr->sorted_ghost_bin_indices.push_back(ghost_bin);
        }
    }
};

}