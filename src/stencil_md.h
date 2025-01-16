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
#include <atomic>
#include <numeric>
#include <sstream>
#include <iomanip>

constexpr bool USE_BREAK = false;

// Helper functions for sort local atoms double buffering
template <typename T, typename Compare>
std::vector<std::size_t> sort_permutation(
        const std::vector<T>& vec,
        Compare compare)
{
    std::vector<std::size_t> p(vec.size());
    std::iota(p.begin(), p.end(), 0);
    std::sort(p.begin(), p.end(),
              [&](std::size_t i, std::size_t j){ return compare(vec[i], vec[j]); });
    return p;
}

template <typename T>
void apply_permutation_in_place(
        std::vector<T>& vec,
        const std::vector<std::size_t>& p)
{
    std::vector<bool> done(vec.size());
    for (std::size_t i = 0; i < vec.size(); ++i)
    {
        if (done[i])
        {
            continue;
        }
        done[i] = true;
        std::size_t prev_j = i;
        std::size_t j = p[i];
        while (i != j)
        {
            std::swap(vec[prev_j], vec[j]);
            done[j] = true;
            prev_j = j;
            j = p[j];
        }
    }
}

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

    void INIT_DOUBLE_BUFFERING_SEND_RECV_DATA();

    void SETUP();

    void MODIFY_PRE_FORCE_SETUP(int);

    void MODIFY_SETUP(int);

    void GET_LOCAL_ATOMS_ZOID();
    void GET_LOCAL_ATOMS_ZOID_DOUBLE_BUFFERING();

    void GET_GHOST_ATOMS_ZOID();
    void GET_GHOST_ATOMS_ZOID_DOUBLE_BUFFERING();

    void SORT_LOCAL_ATOMS_BINS();
    void SORT_LOCAL_ATOMS_DOUBLE_BUFFERING();

    void CREATE_ATOM_IDXS_DOUBLE_BUFFERING();

    void CREATE_ATOM_IDX_MAPPING();

    void BUILD_NEIGHBOR_LIST();
    void BUILD_NEIGHBOR_LIST_NEXT_DT();

    void BUILD_NEIGHBOR_LIST_DOUBLE_BUFFERING();
    void BUILD_BOND_LIST_DOUBLE_BUFFERING();

    void COMPUTE_NUM_SEND_RECV_PROCESS();

    void COMPARE_POS_AGAINST_LAMMPS(bool curr_dt, int timestep, Atom* atom_, queue_info& zoid, double** test_x);
    void COMPARE_FORCE_AGAINST_LAMMPS(bool curr_dt, int timestep, Atom* atom_, queue_info& zoid, double** test_f);
    void COMPARE_VEL_AGAINST_LAMMPS(bool curr_dt, int timestep, Atom* atom_, queue_info& zoid, double** test_f);

    void SET_INUM_PER_TIMESTEP();
    void SET_INUM_PER_TIMESTEP_NEXT_DT();

    void SET_CLAIMED_ATOMIC_BOOLS();

    void CONSTRUCT_START_END_BIG_ZOIDS();
    void CONSTRUCT_START_END_BIG_ZOIDS_HELPER(queue_info& zoid);
    void CONSTRUCT_START_END_BIG_ZOIDS_HELPER_SHRINKING(queue_info& zoid);
    void CONSTRUCT_START_END_BIG_ZOIDS_HELPER_EXPANDING(queue_info& zoid);

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
        auto claimed_flag_struct = next->claimed_flag_struct;

        int num_chunks = next->num_chunks;
        int num_workers = __cilkrts_get_nworkers();

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

        int chunks_per_worker = num_chunks / num_workers;
        int chunk_size = next->chunk_size;

        #pragma cilk grainsize 1
        cilk_for (int ii = 0; ii < num_chunks; ii++) {
            int start_chunk = __cilkrts_get_worker_number() * chunks_per_worker;
            for (int c = 0; c < num_chunks; ++c) {
                int s = (c + start_chunk) % num_chunks;

                if (claimed_flag_struct[s].m.test(std::memory_order_relaxed)) {
                    continue;
                }

                if (!claimed_flag_struct[s].m.test_and_set(std::memory_order_relaxed)) {
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
                    break;
                }
            }
        }

        for (int i = 0; i < num_chunks; i++) {
            claimed_flag_struct[i].m.clear(std::memory_order_relaxed);
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

    inline void post_force_stencil_md_double_buffering(queue_info& zoid, int timestep, Modify* modify_) {
        // const auto& v = zoid.v_stencil_md[timestep % DOUBLE_BUFFERING];
        const auto& v = zoid.v_stencil_md[timestep % 1];
        // auto& eval_f = zoid.eval_f_stencil_md[timestep % DOUBLE_BUFFERING];
        auto& eval_f = zoid.f_stencil_md[timestep % DOUBLE_BUFFERING];

        const auto& type = zoid.type_stencil_md[0];
        const auto& mask = zoid.mask_stencil_md[0];

        int n_post_force = modify_->n_post_force;

        assert(n_post_force == 1);

        // auto fix_post_force = (FixLangevin*) modify_->fix[modify_->list_post_force[0]];
        auto fix_post_force = (FixLangevin*) modify->fix[modify->list_post_force[0]];

        auto gfactor1 = fix_post_force->gfactor1;
        auto gfactor2 = fix_post_force->gfactor2;
        // fix_post_force->compute_target();
        auto tsqrt = fix_post_force->tsqrt;

        const auto& local_idxs = zoid.local_idxs_per_timestep[timestep];
        const int nlocal = local_idxs.size();

        #pragma cilk grainsize 2048
        cilk_for (int idx = 0; idx < nlocal; idx++) {
            int i = local_idxs[idx];
            if (mask[i]) {
                double gamma1 = gfactor1[type[i]];
                double gamma2 = gfactor2[type[i]] * tsqrt;

                double rand_x = 0.6;
                double rand_y = 0.6;
                double rand_z = 0.6;

                dbl3_t_stencil_md fran = {gamma2 * (rand_x - 0.5), gamma2 * (rand_y - 0.5), gamma2 * (rand_z - 0.5)};
                dbl3_t_stencil_md fdrag = {gamma1 * v[i].x, gamma1 * v[i].y, gamma1 * v[i].z};

                eval_f[i].x += fdrag.x + fran.x;
                eval_f[i].y += fdrag.y + fran.y;
                eval_f[i].z += fdrag.z + fran.z;
            }
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
        int num_chunks = curr->num_chunks;
        auto claimed = curr->claimed;
        auto claimed_int = curr->claimed_int;
        auto claimed_flag = curr->claimed_flag;
        auto claimed_flag_struct = curr->claimed_flag_struct;
        int num_workers = __cilkrts_get_nworkers();

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

        int chunks_per_worker = num_chunks / num_workers;
        int chunk_size = curr->chunk_size;

        #pragma cilk grainsize 1
        cilk_for (int ii = 0; ii < num_chunks; ii++) {
            int start_chunk = __cilkrts_get_worker_number() * chunks_per_worker;
            for (int c = 0; c < num_chunks; ++c) {
                int s = (c + start_chunk) % num_chunks;

                if (claimed_flag_struct[s].m.test(std::memory_order_relaxed)) {
                    continue;
                }

                if (!claimed_flag_struct[s].m.test_and_set(std::memory_order_relaxed)) {
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
                    break;
                }
            }
        }

        for (int i = 0; i < num_chunks; i++) {
            claimed_flag_struct[i].m.clear(std::memory_order_relaxed);
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

        int num_chunks = next->num_chunks;
        int num_workers = __cilkrts_get_nworkers();
        auto claimed = next->claimed;
        auto claimed_int = next->claimed_int;
        auto claimed_flag = next->claimed_flag;
        auto claimed_flag_struct = next->claimed_flag_struct;

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

        int chunks_per_worker = num_chunks / num_workers;
        int chunk_size = next->chunk_size;

        #pragma cilk grainsize 1
        cilk_for (int ii = 0; ii < num_chunks; ii++) {
            int start_chunk = __cilkrts_get_worker_number() * chunks_per_worker;
            for (int c = 0; c < num_chunks; ++c) {
                int s = (c + start_chunk) % num_chunks;

                if (claimed_flag_struct[s].m.test(std::memory_order_relaxed)) {
                    continue;
                }

                if (!claimed_flag_struct[s].m.test_and_set(std::memory_order_relaxed)) {
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
                    break;
                }
            }
        }

        for (int i = 0; i < num_chunks; i++) {
            claimed_flag_struct[i].m.clear(std::memory_order_relaxed);
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


    /* Start double buffering code */
    void initial_integrate_stencil_md_affinity_double_buffering(queue_info& zoid, int timestep, Atom* atom_) {
        auto * _noalias x = zoid.x_stencil_md[timestep % DOUBLE_BUFFERING].data();
        auto * _noalias next_x = zoid.x_stencil_md[(timestep + 1) % DOUBLE_BUFFERING].data();

        auto * _noalias v = zoid.v_stencil_md[timestep % 1].data();
        auto * _noalias f = zoid.f_stencil_md[timestep % 1].data();
        auto * _noalias eval_f = zoid.eval_f_stencil_md[timestep % 1].data();

        auto * _noalias mask = zoid.mask_stencil_md[0].data();
        auto * _noalias local_idxs = zoid.local_idxs_per_timestep[timestep].data();
        auto * _noalias type = zoid.type_stencil_md[0].data();

        int nlocal = zoid.local_idxs_per_timestep[timestep].size();

        int num_chunks = atom_->num_chunks;
        // auto claimed_flag_struct = atom_->claimed_flag_struct;
        auto* claimed = zoid.claimed_flags_stencil_md[0];
        int num_workers = __cilkrts_get_nworkers();
        double dtv = update->dt;

        const double * const mass = atom->mass;
        double dtf = 0.5 * update->dt * force->ftm2v;

        const auto& tags = zoid.tag_stencil_md[0];

        int chunks_per_worker = num_chunks / num_workers;
        int chunk_size = atom_->chunk_size;

        #pragma cilk grainsize 1
        cilk_for (int ii = 0; ii < num_chunks; ii++) {
            int start_chunk = __cilkrts_get_worker_number() * chunks_per_worker;
            for (int c = 0; c < num_chunks; ++c) {
                int s = (c + start_chunk) % num_chunks;

                if (claimed[s].test(std::memory_order_relaxed)) {
                    continue;
                }

                if (!claimed[s].test_and_set(std::memory_order_relaxed)) {
                    for (int idx = s * chunk_size; idx < (s + 1) * chunk_size && idx < nlocal; idx++) {
                        int i = local_idxs[idx];

                        double v0 = v[i].x;
                        double v1 = v[i].y;
                        double v2 = v[i].z;

                        const double dtfm = dtf / mass[type[i]];
                        v[i].x += dtfm * (f[i].x + eval_f[i].x);
                        v[i].y += dtfm * (f[i].y + eval_f[i].y);
                        v[i].z += dtfm * (f[i].z + eval_f[i].z);

                        f[i].x = 0.0;
                        f[i].y = 0.0;
                        f[i].z = 0.0;
                        // eval_f[i].x = 0.0;
                        // eval_f[i].y = 0.0;
                        // eval_f[i].z = 0.0;

                        next_x[i].x = x[i].x + dtv * v[i].x;
                        next_x[i].y = x[i].y + dtv * v[i].y;
                        next_x[i].z = x[i].z + dtv * v[i].z;

                    }

                    if (USE_BREAK) {
                        break;
                    }
                }
            }
        }

        for (int i = 0; i < num_chunks; i++) {
            claimed[i].clear(std::memory_order_relaxed);
        }

        /*
        auto& curr_x = zoid.x_stencil_md[timestep % DOUBLE_BUFFERING];
        auto& next_x = zoid.x_stencil_md[(timestep + 1) % DOUBLE_BUFFERING];

        // auto& curr_v = zoid.v_stencil_md[timestep % DOUBLE_BUFFERING];
        // auto& next_v = zoid.v_stencil_md[(timestep + 1) % DOUBLE_BUFFERING];
        auto& curr_v = zoid.v_stencil_md[timestep % 1];
        auto& next_v = zoid.v_stencil_md[(timestep + 1) % 1];

        // auto& curr_f = zoid.f_stencil_md[timestep % DOUBLE_BUFFERING];
        // auto& curr_eval_f = zoid.eval_f_stencil_md[timestep % DOUBLE_BUFFERING];
        auto& curr_f = zoid.f_stencil_md[timestep % 1];
        auto& curr_eval_f = zoid.eval_f_stencil_md[timestep % 1];

        auto& mask = zoid.mask_stencil_md[0];
        const auto& local_idxs = zoid.local_idxs_per_timestep[timestep];

        const auto& type = zoid.type_stencil_md[0];

        int num_chunks = atom_->num_chunks;
        // auto claimed_flag_struct = atom_->claimed_flag_struct;
        auto* claimed = zoid.claimed_flags_stencil_md[0];
        int num_workers = __cilkrts_get_nworkers();
        double dtv = update->dt;

        const double * const mass = atom->mass;
        double dtf = 0.5 * update->dt * force->ftm2v;

        const auto& tags = zoid.tag_stencil_md[0];

        int chunks_per_worker = num_chunks / num_workers;
        int chunk_size = atom_->chunk_size;

        #pragma cilk grainsize 1
        cilk_for (int ii = 0; ii < num_chunks; ii++) {
            int start_chunk = __cilkrts_get_worker_number() * chunks_per_worker;
            for (int c = 0; c < num_chunks; ++c) {
                int s = (c + start_chunk) % num_chunks;

                if (claimed[s].test(std::memory_order_relaxed)) {
                    continue;
                }

                if (!claimed[s].test_and_set(std::memory_order_relaxed)) {
                    for (int idx = s * chunk_size; idx < (s + 1) * chunk_size && idx < local_idxs.size(); idx++) {
                        int i = local_idxs[idx];

                        const double dtfm = dtf / mass[type[i]];
                        next_v[i].x = curr_v[i].x + dtfm * (curr_f[i].x + curr_eval_f[i].x);
                        next_v[i].y = curr_v[i].y + dtfm * (curr_f[i].y + curr_eval_f[i].y);
                        next_v[i].z = curr_v[i].z + dtfm * (curr_f[i].z + curr_eval_f[i].z);

                        curr_f[i].x = 0.0;
                        curr_f[i].y = 0.0;
                        curr_f[i].z = 0.0;
                        curr_eval_f[i].x = 0.0;
                        curr_eval_f[i].y = 0.0;
                        curr_eval_f[i].z = 0.0;

                        next_x[i].x = curr_x[i].x + dtv * next_v[i].x;
                        next_x[i].y = curr_x[i].y + dtv * next_v[i].y;
                        next_x[i].z = curr_x[i].z + dtv * next_v[i].z;
                    }
                    break;
                }
            }
        }

        for (int i = 0; i < num_chunks; i++) {
            claimed[i].clear(std::memory_order_relaxed);
        }
        */
    }

    inline void fuse_post_force_final_integrate_stencil_md_affinity_double_buffering(queue_info& zoid, int timestep, Atom* next, Modify* modify_) {
        auto* _noalias v = zoid.v_stencil_md[timestep % 1].data();
        auto* _noalias f = zoid.f_stencil_md[timestep % 1].data();
        // auto* _noalias eval_f = zoid.eval_f_stencil_md[timestep % 1].data();
        auto* _noalias eval_f = zoid.f_stencil_md[timestep % 1].data();
        auto* _noalias type = zoid.type_stencil_md[0].data();

        const auto& local_idxs = zoid.local_idxs_per_timestep[timestep];

        auto* _noalias claimed = zoid.claimed_flags_stencil_md[0];

        int num_chunks = next->num_chunks;
        int num_workers = __cilkrts_get_nworkers();

        auto fix_post_force = (FixLangevin*) modify->fix[modify->list_post_force[0]];

        auto gfactor1 = fix_post_force->gfactor1;
        auto gfactor2 = fix_post_force->gfactor2;
        // fix_post_force->compute_target();
        auto tsqrt = fix_post_force->tsqrt;

        const double * const mass = atom->mass;
        double dtf = 0.5 * update->dt * force->ftm2v;

        const auto& tags = zoid.tag_stencil_md[0];

        int chunks_per_worker = num_chunks / num_workers;
        int chunk_size = next->chunk_size;
        int nlocal = local_idxs.size();

        #pragma cilk grainsize 1
        cilk_for (int ii = 0; ii < num_chunks; ii++) {
            int start_chunk = __cilkrts_get_worker_number() * chunks_per_worker;
            for (int c = 0; c < num_chunks; ++c) {
                int s = (c + start_chunk) % num_chunks;

                if (claimed[s].test(std::memory_order_relaxed)) {
                    continue;
                }

                if (!claimed[s].test_and_set(std::memory_order_relaxed)) {
                    for (int idx = s * chunk_size; idx < (s + 1) * chunk_size && idx < local_idxs.size(); idx++) {
                        int i = local_idxs[idx];

                        const double dtfm = dtf / mass[type[i]];

                        double gamma1 = gfactor1[type[i]];
                        double gamma2 = gfactor2[type[i]] * tsqrt;

                        double rand_x = 0.6;
                        double rand_y = 0.6;
                        double rand_z = 0.6;

                        double v_x = v[i].x;
                        double v_y = v[i].y;
                        double v_z = v[i].z;

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

                        // v[i].x += dtfm * (f[i].x + eval_f[i].x);
                        // v[i].y += dtfm * (f[i].y + eval_f[i].y);
                        // v[i].z += dtfm * (f[i].z + eval_f[i].z);
                        v[i].x += dtfm * (f[i].x);
                        v[i].y += dtfm * (f[i].y);
                        v[i].z += dtfm * (f[i].z);
                    }
                    if (USE_BREAK) {
                        break;
                    }
                }
            }
        }

        for (int i = 0; i < num_chunks; i++) {
            claimed[i].clear(std::memory_order_relaxed);
        }

        /*
        // auto& next_v = zoid.v_stencil_md[(timestep % DOUBLE_BUFFERING)];
        auto& next_v = zoid.v_stencil_md[(timestep % 1)];
        // auto& f = zoid.f_stencil_md[(timestep % DOUBLE_BUFFERING)];
        // auto& eval_f = zoid.eval_f_stencil_md[(timestep % DOUBLE_BUFFERING)];
        auto& f = zoid.f_stencil_md[(timestep % 1)];
        auto& eval_f = zoid.eval_f_stencil_md[(timestep % 1)];

        auto& mask = zoid.mask_stencil_md[0];
        auto& type = zoid.type_stencil_md[0];

        const auto& local_idxs = zoid.local_idxs_per_timestep[timestep];

        // auto& local_dtfm = next->local_dtfm;
        // auto* claimed_flag_struct = next->claimed_flag_struct;
        auto* claimed = zoid.claimed_flags_stencil_md[0];

        int num_chunks = next->num_chunks;
        int num_workers = __cilkrts_get_nworkers();

        auto fix_post_force = (FixLangevin*) modify->fix[modify->list_post_force[0]];

        auto gfactor1 = fix_post_force->gfactor1;
        auto gfactor2 = fix_post_force->gfactor2;
        // fix_post_force->compute_target();
        auto tsqrt = fix_post_force->tsqrt;

        const double * const mass = atom->mass;
        double dtf = 0.5 * update->dt * force->ftm2v;

        const auto& tags = zoid.tag_stencil_md[0];

        int chunks_per_worker = num_chunks / num_workers;
        int chunk_size = next->chunk_size;

        #pragma cilk grainsize 1
        cilk_for (int ii = 0; ii < num_chunks; ii++) {
            int start_chunk = __cilkrts_get_worker_number() * chunks_per_worker;
            for (int c = 0; c < num_chunks; ++c) {
                int s = (c + start_chunk) % num_chunks;

                if (claimed[s].test(std::memory_order_relaxed)) {
                    continue;
                }

                if (!claimed[s].test_and_set(std::memory_order_relaxed)) {
                    for (int idx = s * chunk_size; idx < (s + 1) * chunk_size && idx < local_idxs.size(); idx++) {
                        int i = local_idxs[idx];

                        const double dtfm = dtf / mass[type[i]];

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
                    break;
                }
            }
        }

        for (int i = 0; i < num_chunks; i++) {
            claimed[i].clear(std::memory_order_relaxed);
        }
        */
    }

    void stencil_md_fuse_force_computation_atomics_affinity_double_buffering(queue_info& zoid, int timestep,
                                                                             Atom* next, Neighbor* neigh_next,
                                                                             Force* next_force, Modify* modify_) {
        const auto * _noalias const x = zoid.x_stencil_md[timestep % DOUBLE_BUFFERING].data();
        // auto * _noalias const f = zoid.eval_f_stencil_md[timestep % 1].data();
        auto * _noalias const f = zoid.f_stencil_md[timestep % 1].data();

        auto pair = (PairLJCut*) next_force->pair;
        auto bond = (BondFENE*) next_force->bond;

        const auto& bond_list = zoid.bond_list[timestep];
        const auto& neighbor_list = zoid.neighbor_list[timestep];

        const int * _noalias const ilist = pair->list->ilist;
        // const int * _noalias const numneigh = pair->list->numneigh;
        // const int * const * const firstneigh = pair->list->firstneigh;
        const double * _noalias const special_lj = force->special_lj;

        // auto* spinlocks = next->spinlocks;
        auto* _noalias spinlocks = zoid.spinlocks_stencil_md[0];

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

        // const int* _noalias const atom_type = next->type;
        const auto& atom_type = zoid.type_stencil_md[0];

        const auto& local_idxs = zoid.local_idxs_per_timestep[timestep];
        const int nlocal = local_idxs.size();

        int num_chunks = next->num_chunks;
        int num_workers = __cilkrts_get_nworkers();
        /*
        auto* claimed = next->claimed;
        auto* claimed_int = next->claimed_int;
        auto* claimed_flag = next->claimed_flag;
        auto* claimed_flag_struct = next->claimed_flag_struct;
        */
        auto* claimed = zoid.claimed_flags_stencil_md[0];

        const auto& tags = zoid.tag_stencil_md[0];

        int chunks_per_worker = num_chunks / num_workers;
        int chunk_size = next->chunk_size;

        // Cilksan_fake_mutex fake_lock;

        // auto* per_worker_updates = zoid.per_worker_force_updates;
        // auto& lock = spinlocks[0];

        #pragma cilk grainsize 1
        cilk_for (int ii = 0; ii < num_chunks; ii++) {
            int worker_number = __cilkrts_get_worker_number();
            int start_chunk = worker_number * chunks_per_worker;

            // auto* updates = per_worker_updates[worker_number];
            // int update_size = 0;

            for (int c = 0; c < num_chunks; ++c) {
                int s = (c + start_chunk) % num_chunks;

                if (claimed[s].test(std::memory_order_relaxed)) {
                    continue;
                }

                if (!claimed[s].test_and_set(std::memory_order_relaxed)) {
                    for (int idx = s * chunk_size; idx < (s + 1) * chunk_size && idx < nlocal; idx++) {
                        int i = local_idxs[idx];

                        const int itype = atom_type[i];

                        // const int *_noalias const jlist = firstneigh[i];
                        const auto& jlist = neighbor_list[i];
                        const double *_noalias const cutsqi = cutsq[itype];
                        const double *_noalias const offseti = offset[itype];
                        const double *_noalias const lj1i = lj1[itype];
                        const double *_noalias const lj2i = lj2[itype];
                        const double *_noalias const lj3i = lj3[itype];
                        const double *_noalias const lj4i = lj4[itype];

                        double xtmp = x[i].x;
                        double ytmp = x[i].y;
                        double ztmp = x[i].z;
                        // int jnum = numneigh[i];
                        int jnum = jlist.size();

                        double fxtmp = 0.0;
                        double fytmp = 0.0;
                        double fztmp = 0.0;

                        for (int jj = 0; jj < jnum; jj++) {
                            double evdwl = 0.0;
                            // int j = jlist[jj];
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
                                    // updates[update_size++] = {j, {-delx * fpair, -dely * fpair, -delz * fpair}};
                                }
                            }
                        }

                        auto& lst_bonds = bond_list[i];
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
                                // Cilksan_fake_lock_guard guard(&fake_lock);
                                spinlocks[i2].lock();
                                f[i2].x -= delx * fbond;
                                f[i2].y -= dely * fbond;
                                f[i2].z -= delz * fbond;
                                spinlocks[i2].unlock();
                                // updates[update_size++] = {i2, {-delx * fbond, -dely * fbond, -delz * fbond}};
                            }
                        }

                        spinlocks[i].lock();
                        f[i].x += fxtmp;
                        f[i].y += fytmp;
                        f[i].z += fztmp;
                        spinlocks[i].unlock();
                    }

                    if (USE_BREAK) {
                        break;
                    }

                    /*
                    bool did = false;
                    for (int try_num = 0; try_num < 10; try_num++) {
                        if (lock.try_lock()) {
                            did = true;
                            for (int i = 0; i < update_size; i++) {
                                auto& p = updates[i];
                                int force_idx = p.first;
                                auto& force_update = p.second;
                                f[force_idx].x += force_update.x;
                                f[force_idx].y += force_update.y;
                                f[force_idx].z += force_update.z;
                            }
                            lock.unlock();
                            update_size = 0;
                            break;
                        }
                    }

                    if (did) {
                        break;
                    }
                    */

                    /*
                    lock.lock();
                    for (int i = 0; i < update_size; i++) {
                        auto& p = updates[i];
                        int force_idx = p.first;
                        auto& force_update = p.second;
                        f[force_idx].x += force_update.x;
                        f[force_idx].y += force_update.y;
                        f[force_idx].z += force_update.z;
                    }
                    lock.unlock();
                    break;
                    */
                }
            }

            /*
            if (update_size > MODIFY_GRAINSIZE * MAX_NEIGHBORS_PER_ATOM) {
                std::cout << "TROUBLE. zoid: " << zoid.num << " timestep: " << timestep << " update size: " << update_size << std::endl;
            }

            if (update_size > 0) {
                lock.lock();
                for (int i = 0; i < update_size; i++) {
                    auto& p = updates[i];
                    int force_idx = p.first;
                    auto& force_update = p.second;
                    f[force_idx].x += force_update.x;
                    f[force_idx].y += force_update.y;
                    f[force_idx].z += force_update.z;
                }
                lock.unlock();
            }
            */
        }

        /*
        for (int i = 0; i < num_chunks; i++) {
            claimed_flag_struct[i].m.clear(std::memory_order_relaxed);
        }
        */

        for (int i = 0; i < num_chunks; i++) {
            claimed[i].clear(std::memory_order_relaxed);
        }
    }

    /* Start double buffering code */
    void stencil_md_initial_integrate_affinity_double_buffering_start_end(queue_info& zoid, int timestep,
                                                                          Atom* atom_, const std::vector<int>& space_cut_idxs) {
        if (space_cut_idxs.size() == 0) {
            return;
        }

        auto * _noalias x = zoid.x_stencil_md[timestep % DOUBLE_BUFFERING].data();
        auto * _noalias next_x = zoid.x_stencil_md[(timestep + 1) % DOUBLE_BUFFERING].data();

        auto * _noalias v = zoid.v_stencil_md[timestep % 1].data();
        auto * _noalias f = zoid.f_stencil_md[timestep % 1].data();
        auto * _noalias eval_f = zoid.eval_f_stencil_md[timestep % 1].data();

        auto * _noalias mask = zoid.mask_stencil_md[0].data();
        auto * _noalias local_idxs = zoid.local_idxs_per_timestep[timestep].data();
        auto * _noalias type = zoid.type_stencil_md[0].data();

        int num_chunks = space_cut_idxs.size() / MODIFY_GRAINSIZE + 1;
        // int num_chunks = atom_->num_chunks;
        // auto claimed_flag_struct = atom_->claimed_flag_struct;
        auto* claimed = zoid.claimed_flags_stencil_md[0];
        int num_workers = __cilkrts_get_nworkers();
        double dtv = update->dt;

        const double * const mass = atom->mass;
        double dtf = 0.5 * update->dt * force->ftm2v;

        const auto& tags = zoid.tag_stencil_md[0];

        int chunks_per_worker = num_chunks / num_workers;
        int chunk_size = atom_->chunk_size;

        int nprocess = space_cut_idxs.size();
        int start = space_cut_idxs[0];

        #pragma cilk grainsize 1
        cilk_for (int ii = 0; ii < num_chunks; ii++) {
            int start_chunk = __cilkrts_get_worker_number() * chunks_per_worker;
            for (int c = 0; c < num_chunks; ++c) {
                int s = (c + start_chunk) % num_chunks;

                if (claimed[s].test(std::memory_order_relaxed)) {
                    continue;
                }

                if (!claimed[s].test_and_set(std::memory_order_relaxed)) {
                    for (int idx = s * chunk_size; idx < (s + 1) * chunk_size && idx < nprocess; idx++) {
                        // int i = local_idxs[idx];
                        // int i = idx_start + idx;
                        // assert(i == local_idxs[idx + start]);
                        int i = space_cut_idxs[idx];
                        // int i = start + idx;

                        double v_x = v[i].x;
                        double v_y = v[i].y;
                        double v_z = v[i].z;

                        const double dtfm = dtf / mass[type[i]];
                        v[i].x += dtfm * (f[i].x + eval_f[i].x);
                        v[i].y += dtfm * (f[i].y + eval_f[i].y);
                        v[i].z += dtfm * (f[i].z + eval_f[i].z);

                        double f_x = eval_f[i].x;
                        double f_y = eval_f[i].y;
                        double f_z = eval_f[i].z;

                        f[i].x = 0.0;
                        f[i].y = 0.0;
                        f[i].z = 0.0;
                        eval_f[i].x = 0.0;
                        eval_f[i].y = 0.0;
                        eval_f[i].z = 0.0;

                        next_x[i].x = x[i].x + dtv * v[i].x;
                        next_x[i].y = x[i].y + dtv * v[i].y;
                        next_x[i].z = x[i].z + dtv * v[i].z;
                    }
                    if (USE_BREAK) {
                        break;
                    }
                }
            }
        }

        for (int i = 0; i < num_chunks; i++) {
            claimed[i].clear(std::memory_order_relaxed);
        }

        return;
    }

    void stencil_md_force_computation_start_end(queue_info& zoid, int timestep,
                                                Atom* next, Neighbor* neigh_next,
                                                Force* next_force, Modify* modify_,
                                                std::vector<int>& space_cut_idxs) {

        if (space_cut_idxs.size() == 0) {
            return;
        }

        const auto * _noalias const x = zoid.x_stencil_md[timestep % DOUBLE_BUFFERING].data();
        auto * _noalias const f = zoid.eval_f_stencil_md[timestep % 1].data();
        auto* _noalias f2 = zoid.f_stencil_md[timestep % 1].data();
        auto* _noalias type = zoid.type_stencil_md[0].data();
        auto * _noalias const v = zoid.v_stencil_md[timestep % 1].data();

        auto pair = (PairLJCut*) next_force->pair;
        auto bond = (BondFENE*) next_force->bond;

        const auto& bond_list = zoid.bond_list[timestep];
        const auto& neighbor_list = zoid.neighbor_list[timestep];

        const int * _noalias const ilist = pair->list->ilist;
        const double * _noalias const special_lj = force->special_lj;

        auto* _noalias spinlocks = zoid.spinlocks_stencil_md[0];

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

        const auto& atom_type = zoid.type_stencil_md[0];

        int num_chunks = space_cut_idxs.size() / MODIFY_GRAINSIZE + 1;
        int num_workers = __cilkrts_get_nworkers();

        auto* claimed = zoid.claimed_flags_stencil_md[0];

        int chunks_per_worker = num_chunks / num_workers;
        int chunk_size = next->chunk_size;

        int nprocess = space_cut_idxs.size();
        int start = space_cut_idxs[0];

        #pragma cilk grainsize 1
        cilk_for (int ii = 0; ii < num_chunks; ii++) {
            int start_chunk = __cilkrts_get_worker_number() * chunks_per_worker;

            for (int c = 0; c < num_chunks; ++c) {
                int s = (c + start_chunk) % num_chunks;

                if (claimed[s].test(std::memory_order_relaxed)) {
                    continue;
                }

                if (!claimed[s].test_and_set(std::memory_order_relaxed)) {
                    for (int idx = s * chunk_size; idx < (s + 1) * chunk_size && idx < nprocess; idx++) {
                        // int i = idx_start + idx;
                        // assert(i == local_idxs[start + idx]);
                        int i = space_cut_idxs[idx];
                        // int i = start + idx;
                        // assert(i == space_cut_idxs[idx]);

                        const int itype = atom_type[i];

                        const auto& jlist = neighbor_list[i];
                        const double *_noalias const cutsqi = cutsq[itype];
                        const double *_noalias const offseti = offset[itype];
                        const double *_noalias const lj1i = lj1[itype];
                        const double *_noalias const lj2i = lj2[itype];
                        const double *_noalias const lj3i = lj3[itype];
                        const double *_noalias const lj4i = lj4[itype];

                        double xtmp = x[i].x;
                        double ytmp = x[i].y;
                        double ztmp = x[i].z;

                        int jnum = jlist.size();

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

                                if (newton_pair) {
                                    spinlocks[j].lock();
                                    f[j].x -= delx * fpair;
                                    f[j].y -= dely * fpair;
                                    f[j].z -= delz * fpair;
                                    spinlocks[j].unlock();
                                }
                            }
                        }

                        auto& lst_bonds = bond_list[i];
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
                            if (newton_pair) {
                                fxtmp += delx * fbond;
                                fytmp += dely * fbond;
                                fztmp += delz * fbond;
                            }

                            if (newton_pair) {
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
                    if (USE_BREAK) {
                        break;
                    }
                }
            }
        }

        for (int i = 0; i < num_chunks; i++) {
            claimed[i].clear(std::memory_order_relaxed);
        }
    }

    inline void fuse_post_force_final_integrate_stencil_md_start_end(queue_info& zoid, int timestep,
                                                                     Atom* next, Modify* modify_, std::vector<int>& space_cut_idxs) {
        if (space_cut_idxs.size() == 0) {
            return;
        }

        auto* _noalias v = zoid.v_stencil_md[timestep % 1].data();
        auto* _noalias f = zoid.f_stencil_md[timestep % 1].data();
        auto* _noalias eval_f = zoid.eval_f_stencil_md[timestep % 1].data();
        auto* _noalias type = zoid.type_stencil_md[0].data();

        auto* _noalias claimed = zoid.claimed_flags_stencil_md[0];

        int num_chunks = space_cut_idxs.size() / MODIFY_GRAINSIZE + 1;
        int num_workers = __cilkrts_get_nworkers();

        auto fix_post_force = (FixLangevin*) modify->fix[modify->list_post_force[0]];

        auto gfactor1 = fix_post_force->gfactor1;
        auto gfactor2 = fix_post_force->gfactor2;
        // fix_post_force->compute_target();
        auto tsqrt = fix_post_force->tsqrt;

        const double * const mass = atom->mass;
        double dtf = 0.5 * update->dt * force->ftm2v;

        const auto& tags = zoid.tag_stencil_md[0];

        int chunks_per_worker = num_chunks / num_workers;
        int chunk_size = next->chunk_size;

        int nprocess = space_cut_idxs.size();
        int start = space_cut_idxs[0];

        #pragma cilk grainsize 1
        cilk_for (int ii = 0; ii < num_chunks; ii++) {
            int start_chunk = __cilkrts_get_worker_number() * chunks_per_worker;
            for (int c = 0; c < num_chunks; ++c) {
                int s = (c + start_chunk) % num_chunks;

                if (claimed[s].test(std::memory_order_relaxed)) {
                    continue;
                }

                if (!claimed[s].test_and_set(std::memory_order_relaxed)) {
                    for (int idx = s * chunk_size; idx < (s + 1) * chunk_size && idx < nprocess; idx++) {
                        // int i = local_idxs[idx];
                        int i = space_cut_idxs[idx];
                        // int i = start + idx;
                        // assert(i == space_cut_idxs[idx]);

                        const double dtfm = dtf / mass[type[i]];

                        double gamma1 = gfactor1[type[i]];
                        double gamma2 = gfactor2[type[i]] * tsqrt;

                        double rand_x = 0.6;
                        double rand_y = 0.6;
                        double rand_z = 0.6;

                        double v_x = v[i].x;
                        double v_y = v[i].y;
                        double v_z = v[i].z;

                        eval_f[i].x += gamma1 * v_x + gamma2 * (rand_x - 0.5);
                        eval_f[i].y += gamma1 * v_y + gamma2 * (rand_x - 0.5);
                        eval_f[i].z += gamma1 * v_z + gamma2 * (rand_x - 0.5);

                        v[i].x += dtfm * (f[i].x + eval_f[i].x);
                        v[i].y += dtfm * (f[i].y + eval_f[i].y);
                        v[i].z += dtfm * (f[i].z + eval_f[i].z);
                    }
                    if (USE_BREAK) {
                        break;
                    }
                }
            }
        }

        for (int i = 0; i < num_chunks; i++) {
            claimed[i].clear(std::memory_order_relaxed);
        }
    }

    template <bool curr_dt>
    void CONSTRUCT_SEND_FORCE_DOUBLE_BUFFERING() {
        auto& my_queue = curr_dt ? lmp->queues : lmp->queues_next_dt;

        for (int dep = 0; dep < NUM_DEPS; dep++) {
            for (int j = 0; j < my_queue[dep].size(); j++) {
                auto& zoid = my_queue[dep][j];
                int zoid_num = zoid.num;
                auto& send_to = curr_dt ? lmp->send_to_neighbors[zoid.num] : lmp->send_to_neighbors_next_dt[zoid.num];
                if (zoid.num % comm->nprocs == comm->me) {
                    if (send_to.size() == 0) {
                        continue;
                    }

                    for (int t = 0; t < NUM_TIMESTEPS_IN_PARALLEL + 1; t++) {
                        zoid.send_force_idxs_double_buffering[t] = new std::vector<int>[send_to.size()];
                    }

                    // technically the same for both timesteps
                    std::unordered_map<int, int> zoid_tag_to_idx;
                    for (int k = 0; k < zoid.tag_stencil_md[0].size(); k++) {
                        zoid_tag_to_idx[zoid.tag_stencil_md[0][k]] = k;
                    }

                    for (int i = 0; i < send_to.size(); i++) {
                        for (int t = 0; t < NUM_TIMESTEPS_IN_PARALLEL + 1; t++) {
                            // std::set<int> send_force_idxs;
                            std::vector<int> send_force_idxs;
                            Atom* atom_ = curr_dt ? lmp->atom_stencil_md[zoid.num][t] : lmp->atom_stencil_md[zoid.num][NUM_TIMESTEPS_IN_PARALLEL - t];

                            int num_segments = zoid.send_force_num_segments[t][i];
                            auto idxs = zoid.send_force_idxs[t][i];
                            auto sizes = zoid.send_force_sizes[t][i];

                            for (int k = 0; k < num_segments; k++) {
                                int segment_idx = idxs[k];
                                int segment_size = sizes[k];
                                for (int h = 0; h < segment_size; h++) {
                                    int idx = segment_idx + h;
                                    int double_buffering_idx = zoid_tag_to_idx[atom_->tag[idx]];
                                    // assert(send_force_idxs.find(double_buffering_idx) == send_force_idxs.end());
                                    assert(std::find(send_force_idxs.begin(), send_force_idxs.end(), double_buffering_idx) == send_force_idxs.end());
                                    assert(zoid_tag_to_idx.count(atom_->tag[idx]));
                                    // send_force_idxs.insert(double_buffering_idx);
                                    send_force_idxs.push_back(double_buffering_idx);
                                }
                            }

                            assert(zoid_tag_to_idx.size() == zoid.tag_stencil_md[0].size());
                            zoid.send_force_idxs_double_buffering[t][i] = std::vector<int>(send_force_idxs.begin(), send_force_idxs.end());
                        }
                    }
                }
            }
        }
    }

    template <bool curr_dt>
    void CONSTRUCT_RECV_FORCE_DOUBLE_BUFFERING() {
        auto& queue = curr_dt ? lmp->queues : lmp->queues_next_dt;

        for (int dep = 0; dep < NUM_DEPS; dep++) {
            for (int j = 0; j < queue[dep].size(); j++) {
                auto &zoid = queue[dep][j];
                int zoid_num = zoid.num;
                if (zoid_num % comm->nprocs == comm->me) {
                    auto &recv_from = curr_dt ? lmp->recv_from_neighbors[zoid.num] : lmp->recv_from_neighbors_next_dt[zoid.num];
                    if (recv_from.size() == 0) {
                        continue;
                    }

                    for (int t = 0; t < NUM_TIMESTEPS_IN_PARALLEL + 1; t++) {
                        zoid.recv_force_idxs_double_buffering[t] = new std::vector<int>[recv_from.size()];
                    }

                    // technically the same for both timesteps
                    std::unordered_map<int, int> zoid_tag_to_idx;
                    for (int k = 0; k < zoid.tag_stencil_md[0].size(); k++) {
                        zoid_tag_to_idx[zoid.tag_stencil_md[0][k]] = k;
                    }

                    for (int i = 0; i < recv_from.size(); i++) {
                        std::set<int> all_idxs;

                        std::set<int> all_idxs_across_timesteps;
                        std::map<int, int> idx_to_timestep;
                        for (int t = 0; t < NUM_TIMESTEPS_IN_PARALLEL + 1; t++) {
                            // std::set<int> recv_force_idxs;
                            std::vector<int> recv_force_idxs;
                            Atom *atom_ = curr_dt ? lmp->atom_stencil_md[zoid.num][t] : lmp->atom_stencil_md[zoid.num][NUM_TIMESTEPS_IN_PARALLEL - t];

                            int nrecv_force = zoid.recv_list_local_num_force_only[t][i];

                            for (int k = 0; k < nrecv_force; k++) {
                                int idx = zoid.recv_list_local_force_only[t][i][k];
                                int double_buffering_idx = zoid_tag_to_idx[atom_->tag[idx]];
                                assert(zoid_tag_to_idx.count(atom_->tag[idx]));
                                // assert(recv_force_idxs.find(double_buffering_idx) == recv_force_idxs.end());
                                // recv_force_idxs.insert(double_buffering_idx);
                                assert(std::find(recv_force_idxs.begin(), recv_force_idxs.end(), double_buffering_idx) == recv_force_idxs.end());
                                recv_force_idxs.push_back(double_buffering_idx);

                                assert(all_idxs.find(double_buffering_idx) == all_idxs.end());
                                all_idxs.insert(double_buffering_idx);
                            }

                            assert(zoid_tag_to_idx.size() == zoid.tag_stencil_md[0].size());
                            zoid.recv_force_idxs_double_buffering[t][i] = recv_force_idxs;

                            for (auto& idx: recv_force_idxs) {
                                if (all_idxs_across_timesteps.find(idx) != all_idxs_across_timesteps.end()) {
                                    std::cout << "FORCES LOCAL zoid: " << zoid.num << " time: " << t
                                              << " DUPLICATE IDX. Prev timestep: " << idx_to_timestep[idx]
                                              << RESET_COLOR << std::endl;
                                    assert(false);
                                }
                                all_idxs_across_timesteps.insert(idx);
                                idx_to_timestep[idx] = t;
                            }
                        }
                    }
                }
            }
        }
    }

    template <bool curr_dt>
    void CONSTRUCT_SEND_POS_DOUBLE_BUFFERING() {
        auto& my_queue = curr_dt ? lmp->queues : lmp->queues_next_dt;
        for (int dep = 0; dep < NUM_DEPS; dep++) {
            for (int j = 0; j < my_queue[dep].size(); j++) {
                auto& zoid = my_queue[dep][j];
                int zoid_num = zoid.num;
                if (zoid.num % comm->nprocs == comm->me) {
                    for (int t = 0; t < NUM_TIMESTEPS_IN_PARALLEL + 1; t++) {
                        zoid.send_pos_idxs_double_buffering[t] = new std::vector<int>[comm->nprocs];
                        Atom* atom_ = curr_dt ? lmp->atom_stencil_md[zoid.num][t] : lmp->atom_stencil_md[zoid.num][NUM_TIMESTEPS_IN_PARALLEL - t];

                        std::unordered_map<int, int> zoid_tag_to_idx;
                        for (int k = 0; k < zoid.tag_stencil_md[0].size(); k++) {
                            zoid_tag_to_idx[zoid.tag_stencil_md[0][k]] = k;
                        }

                        for (int proc = 0; proc < comm->nprocs; proc++) {
                            int num_segments = zoid.send_process_num_segments[t][proc];
                            auto segment_types = zoid.send_process_segment_types[t][proc];
                            auto segment_idxs = zoid.send_process_segment_idxs[t][proc];
                            auto segment_sizes = zoid.send_process_segment_sizes[t][proc];

                            auto local_list = zoid.send_process_local_list[t][proc];

                            std::vector<int> all_idxs;

                            int local_list_idx = 0;
                            for (int k = 0; k < num_segments; k++) {
                                auto segment_type = segment_types[k];
                                auto segment_idx = segment_idxs[k];
                                auto segment_size = segment_sizes[k];

                                if (segment_type == SEND_DATA_PROCESS_LOCAL) {
                                    for (int h = 0; h < segment_size; h++) {
                                        int idx = local_list[local_list_idx++];
                                        all_idxs.push_back(idx);
                                    }
                                } else if (segment_type == SEND_DATA_PROCESS_GHOST) {
                                    for (int h = 0; h < segment_size; h++) {
                                        int idx = segment_idx + h;
                                        all_idxs.push_back(idx);
                                    }
                                } else {
                                    assert(false);
                                }
                            }

                            assert(zoid_tag_to_idx.size() == zoid.tag_stencil_md[0].size());

                            std::vector<int> idxs_in_double_buffering;
                            for (auto& idx : all_idxs) {
                                assert(zoid_tag_to_idx.count(atom_->tag[idx]));
                                idxs_in_double_buffering.push_back(zoid_tag_to_idx[atom_->tag[idx]]);

                            }

                            zoid.send_pos_idxs_double_buffering[t][proc] = idxs_in_double_buffering;
                        }
                    }
                }
            }
        }
    }

    template <bool curr_dt>
    void CONSTRUCT_RECV_POS_DOUBLE_BUFFERING() {
        auto& my_queue = curr_dt ? lmp->queues : lmp->queues_next_dt;
        for (int dep = 0; dep < NUM_DEPS; dep++) {
            for (int j = 0; j < my_queue[dep].size(); j++) {
                auto& zoid = my_queue[dep][j];
                int zoid_num = zoid.num;
                if (zoid.num % comm->nprocs == comm->me) {
                    auto& recv_from = curr_dt ? lmp->recv_from_neighbors[zoid_num] : lmp->recv_from_neighbors_next_dt[zoid_num];

                    std::unordered_map<int, int> zoid_tag_to_idx;
                    for (int k = 0; k < zoid.tag_stencil_md[0].size(); k++) {
                        zoid_tag_to_idx[zoid.tag_stencil_md[0][k]] = k;
                    }

                    std::set<int> all_idxs_across_timesteps;
                    std::map<int, int> idx_to_timestep;
                    std::map<int, std::string> idx_to_type;

                    for (int t = 0; t < NUM_TIMESTEPS_IN_PARALLEL + 1; t++) {
                        // TODO: flatten all of the buf idxs into 1 single vector, vector to vector mapping
                        // for i, buf_idx in buf_idxs: my_idx = my_idxs[i]; f[my_idx] += buf[buf_idx];
                        zoid.recv_pos_local_idxs_double_buffering[t] = new std::vector<int>[recv_from.size()];
                        zoid.recv_pos_ghost_idxs_double_buffering[t] = new std::vector<int>[recv_from.size()];

                        Atom* atom_ = curr_dt ? lmp->atom_stencil_md[zoid.num][t] : lmp->atom_stencil_md[zoid.num][NUM_TIMESTEPS_IN_PARALLEL - t];

                        for (int i = 0; i < recv_from.size(); i++) {
                            auto num_in_local_list = zoid.recv_list_local_num_force_pos[t][i];
                            auto local_list = zoid.recv_list_local_force_pos[t][i];

                            // int num_segments = zoid.recv_process_num_segments[t][i];
                            // auto segment_types = zoid.recv_process_segment_types[t][i];
                            // auto segment_idxs = zoid.recv_process_segment_idxs[t][i];
                            // auto segment_sizes = zoid.recv_process_segment_sizes[t][i];

                            int num_segments = zoid.recv_ghost_num_segments[t][i];
                            auto segment_idxs = zoid.recv_ghost_idxs[t][i];
                            auto segment_sizes = zoid.recv_ghost_sizes[t][i];

                            std::vector<int> local_idxs;
                            std::vector<int> ghost_idxs;

                            for (int k = 0; k < num_in_local_list; k++) {
                                int idx = local_list[k];
                                int double_buffering_idx = zoid_tag_to_idx[atom_->tag[idx]];
                                local_idxs.push_back(double_buffering_idx);
                            }

                            for (int k = 0; k < num_segments; k++) {
                                auto segment_idx = segment_idxs[k];
                                auto segment_size = segment_sizes[k];
                                for (int h = 0; h < segment_size; h++) {
                                    int idx = segment_idx + h;
                                    int double_buffering_idx = zoid_tag_to_idx[atom_->tag[idx]];
                                    ghost_idxs.push_back(double_buffering_idx);
                                }
                            }

                            zoid.recv_pos_local_idxs_double_buffering[t][i] = local_idxs;
                            zoid.recv_pos_ghost_idxs_double_buffering[t][i] = ghost_idxs;

                            /*
                            for (auto& idx: local_idxs) {
                                if (all_idxs_across_timesteps.find(idx) != all_idxs_across_timesteps.end()) {
                                    std::cout << "POS LOCAL zoid: " << zoid.num << " time: " << t
                                        << " DUPLICATE IDX. idx: " << idx << " Prev timestep: " << idx_to_timestep[idx]
                                        << " TYPE: " << idx_to_type[idx] << RESET_COLOR << std::endl;
                                    // assert(false);
                                }
                                all_idxs_across_timesteps.insert(idx);
                                idx_to_timestep[idx] = t;
                                idx_to_type[idx] = "LOCAL";
                            }

                            for (auto& idx: ghost_idxs) {
                                if (all_idxs_across_timesteps.find(idx) != all_idxs_across_timesteps.end()) {
                                    std::cout << "POS GHOST zoid: " << zoid.num << " time: " << t
                                        << " DUPLICATE IDX. " << idx << " Prev timestep: " << idx_to_timestep[idx]
                                        << " TYPE: " << idx_to_type[idx] << RESET_COLOR << std::endl;
                                    // assert(false);
                                }
                                all_idxs_across_timesteps.insert(idx);
                                idx_to_timestep[idx] = t;
                                idx_to_type[idx] = "GHOST";
                            }
                            */
                        }
                    }
                }
            }
        }
    }

    template <bool curr_dt>
    void CONSTRUCT_SEND_VEL_DOUBLE_BUFFERING() {
        auto& my_queue = curr_dt ? lmp->queues : lmp->queues_next_dt;
        for (int dep = 0; dep < NUM_DEPS; dep++) {
            for (int j = 0; j < my_queue[dep].size(); j++) {
                auto& zoid = my_queue[dep][j];
                int zoid_num = zoid.num;
                if (zoid.num % comm->nprocs == comm->me) {
                    auto& send_to = curr_dt ? lmp->send_to_neighbors[zoid_num] : lmp->send_to_neighbors_next_dt[zoid_num];
                    for (int t = 0; t < NUM_TIMESTEPS_IN_PARALLEL + 1; t++) {
                        zoid.send_vel_idxs_double_buffering[t] = new std::vector<int>[send_to.size()];
                        Atom* atom_ = curr_dt ? lmp->atom_stencil_md[zoid.num][t] : lmp->atom_stencil_md[zoid.num][NUM_TIMESTEPS_IN_PARALLEL - t];

                        std::unordered_map<int, int> zoid_tag_to_idx;
                        for (int k = 0; k < zoid.tag_stencil_md[0].size(); k++) {
                            zoid_tag_to_idx[zoid.tag_stencil_md[0][k]] = k;
                        }

                        for (int i = 0; i < send_to.size(); i++) {
                            int num_segments = zoid.send_pos_num_segments[t][i];
                            auto segment_idxs = zoid.send_pos_idxs[t][i];
                            auto segment_sizes = zoid.send_pos_sizes[t][i];

                            std::vector<int> all_idxs;

                            for (int k = 0; k < num_segments; k++) {
                                auto segment_idx = segment_idxs[k];
                                auto segment_size = segment_sizes[k];

                                for (int h = 0; h < segment_size; h++) {
                                    int idx = segment_idx + h;
                                    all_idxs.push_back(idx);
                                }
                            }

                            assert(zoid_tag_to_idx.size() == zoid.tag_stencil_md[0].size());

                            std::vector<int> idxs_in_double_buffering;
                            for (auto& idx : all_idxs) {
                                assert(zoid_tag_to_idx.count(atom_->tag[idx]));
                                idxs_in_double_buffering.push_back(zoid_tag_to_idx[atom_->tag[idx]]);
                            }

                            zoid.send_vel_idxs_double_buffering[t][i] = idxs_in_double_buffering;
                        }
                    }
                }
            }
        }
    }

    template <bool curr_dt>
    void CONSTRUCT_RECV_VEL_DOUBLE_BUFFERING() {
        auto& my_queue = curr_dt ? lmp->queues : lmp->queues_next_dt;
        for (int dep = 0; dep < NUM_DEPS; dep++) {
            for (int j = 0; j < my_queue[dep].size(); j++) {
                auto& zoid = my_queue[dep][j];
                int zoid_num = zoid.num;
                if (zoid.num % comm->nprocs == comm->me) {
                    auto& recv_from = curr_dt ? lmp->recv_from_neighbors[zoid_num] : lmp->recv_from_neighbors_next_dt[zoid_num];

                    std::set<int> all_idxs_across_timesteps;
                    std::map<int, int> idx_to_timestep;
                    for (int t = 0; t < NUM_TIMESTEPS_IN_PARALLEL + 1; t++) {
                        zoid.recv_vel_idxs_double_buffering[t] = new std::vector<int>[recv_from.size()];
                        Atom* atom_ = curr_dt ? lmp->atom_stencil_md[zoid.num][t] : lmp->atom_stencil_md[zoid.num][NUM_TIMESTEPS_IN_PARALLEL - t];

                        std::unordered_map<int, int> zoid_tag_to_idx;
                        for (int k = 0; k < zoid.tag_stencil_md[0].size(); k++) {
                            zoid_tag_to_idx[zoid.tag_stencil_md[0][k]] = k;
                        }

                        for (int i = 0; i < recv_from.size(); i++) {
                            auto nrecv_vel = zoid.recv_list_local_num_force_pos[t][i];
                            auto local_list = zoid.recv_list_local_force_pos[t][i];

                            std::vector<int> all_idxs;

                            for (int k = 0; k < nrecv_vel; k++) {
                                all_idxs.push_back(local_list[k]);
                            }

                            assert(zoid_tag_to_idx.size() == zoid.tag_stencil_md[0].size());

                            std::vector<int> idxs_in_double_buffering;
                            for (auto& idx : all_idxs) {
                                assert(zoid_tag_to_idx.count(atom_->tag[idx]));
                                idxs_in_double_buffering.push_back(zoid_tag_to_idx[atom_->tag[idx]]);
                            }

                            zoid.recv_vel_idxs_double_buffering[t][i] = idxs_in_double_buffering;

                            for (auto& idx: idxs_in_double_buffering) {
                                if (all_idxs_across_timesteps.find(idx) != all_idxs_across_timesteps.end()) {
                                    std::cout << BOLDRED << "LOCAL zoid: " << zoid.num << " time: " << t
                                              << " DUPLICATE IDX. Prev timestep: " << idx_to_timestep[idx]
                                              << RESET_COLOR << std::endl;
                                    assert(false);
                                }
                                all_idxs_across_timesteps.insert(idx);
                                idx_to_timestep[idx] = t;
                            }
                        }
                    }
                }
            }
        }
    }

    void TEST_FORCE_AGAINST_LAMMPS_DOUBLE_BUFFERING(bool curr_dt, int timestep, queue_info& zoid, Atom* atom_, double** test_f) {

        const auto& x = zoid.x_stencil_md[timestep % DOUBLE_BUFFERING];

        // const auto& f = zoid.f_stencil_md[timestep % DOUBLE_BUFFERING];
        // const auto& eval_f = zoid.eval_f_stencil_md[timestep % DOUBLE_BUFFERING];
        const auto& f = zoid.f_stencil_md[timestep % 1];
        const auto& eval_f = zoid.eval_f_stencil_md[timestep % 1];

        std::unordered_map<int, int> zoid_tag_to_idx;
        for (int i = 0; i < zoid.tag_stencil_md[0].size(); i++) {
            zoid_tag_to_idx[zoid.tag_stencil_md[0][i]] = i;
        }

        for (int k = 0; k < atom_->nlocal; k++) {
            // compare forces on local atoms?
            int tag = atom_->tag[k];
            int test_idx = zoid_tag_to_idx[tag];

            double my_f[3] = {f[test_idx].x,
                              f[test_idx].y,
                              f[test_idx].z};

            double my_eval_f[3] = {eval_f[test_idx].x,
                                   eval_f[test_idx].y,
                                   eval_f[test_idx].z};

            for (int dim = 0; dim < 3; dim++) {
                double my_force = my_f[dim] + my_eval_f[dim];
                if (fabs(my_force - test_f[timestep][tag * 3 + dim]) > 1e-6) {
                    if (curr_dt) {
                        std::cout << "------FORCE DIFF--------"
                                  << std::endl;
                    } else {
                        std::cout << "------NEXT DT FORCE DIFF--------"
                                  << std::endl;
                    }

                    std::cout << "idx: " << k << " out of: " << atom_->nlocal << " double buffering idx: " << test_idx << std::endl;
                    std::cout << "Dim: " << dim << " Zoid: " << zoid.num << " timestep: " << timestep << " tag: " << tag << std::endl;
                    std::cout << "what I have: " << my_f[0] + my_eval_f[0] << " " << my_f[1] + my_eval_f[1] << " " << my_f[2] + my_eval_f[2] << std::endl;
                    std::cout << "my f: " << my_f[0] << " " << my_f[1] << " " << my_f[2]
                              << " my eval f: " << my_eval_f[0] << " " << my_eval_f[1] << " " << my_eval_f[2]
                              << std::endl;
                    std::cout << "What does LAMMPS have? "
                              << test_f[timestep][tag * 3 + 0] << " "
                              << test_f[timestep][tag * 3 + 1] << " "
                              << test_f[timestep][tag * 3 + 2]
                              << std::endl;
                    std::cout << "Diff: " << fabs(my_force - test_f[timestep][tag * 3 + dim]) << std::endl;
                    std::cout << "pos: " << x[test_idx].x << " " << x[test_idx].y << " " << x[test_idx].z << std::endl;

                    for (int tmp = 0; tmp < 3; tmp++) {
                        std::cout << "lo: " << zoid.zoid.cuts[tmp].lower +
                        zoid.zoid.cuts[tmp].slope_lower * (timestep % (NUM_TIMESTEPS_IN_PARALLEL + 1))
                        << std::endl;
                        std::cout << "hi: "
                                  << zoid.zoid.cuts[tmp].upper +
                                  zoid.zoid.cuts[tmp].slope_upper * (timestep % (NUM_TIMESTEPS_IN_PARALLEL + 1))
                                  << std::endl;
                    }

                    assert(false);
                }
            }
        }
    }

    void TEST_POS_AGAINST_LAMMPS_DOUBLE_BUFFERING(bool curr_dt, int timestep, queue_info& zoid, Atom* atom_, double** test_x) {
        const auto& x = zoid.x_stencil_md[timestep % DOUBLE_BUFFERING];

        std::unordered_map<int, int> zoid_tag_to_idx;
        for (int i = 0; i < zoid.tag_stencil_md[0].size(); i++) {
            zoid_tag_to_idx[zoid.tag_stencil_md[0][i]] = i;
        }

        for (int k = 0; k < atom_->nlocal; k++) {
            // compare forces on local atoms?
            int tag = atom_->tag[k];
            int test_idx = zoid_tag_to_idx[tag];

            double my_x[3] = {x[test_idx].x, x[test_idx].y, x[test_idx].z};

            for (int dim = 0; dim < 3; dim++) {
                double my_pos = my_x[dim];
                if (my_pos < 0) {
                    my_pos += domain->prd[dim];
                } else if (my_pos >= domain->prd[dim]) {
                    my_pos -= domain->prd[dim];
                }

                double test_pos = test_x[timestep][tag * 3 + dim];
                if (test_pos < 0) {
                    test_pos += domain->prd[dim];
                } else if (test_pos >= domain->prd[dim]) {
                    test_pos -= domain->prd[dim];
                }

                if (fabs(my_pos - test_pos) > 1e-6) {
                    if (curr_dt) {
                        std::cout << "------POS DIFF--------"
                                  << std::endl;
                    } else {
                        std::cout << "------NEXT DT POS DIFF--------"
                                  << std::endl;
                    }

                    std::cout << "my pos: " << my_pos << " test pos: " << test_pos << std::endl;
                    std::cout << "idx: " << k << " out of: " << atom_->nlocal << " atom: " << atom_ << std::endl;
                    std::cout << "Dim: " << dim << " Zoid: " << zoid.num << " timestep: " << timestep << " tag: " << tag << std::endl;
                    std::cout << "what I have: " << my_x[0] << " " << my_x[1] << " " << my_x[2] << std::endl;
                    std::cout << "What does LAMMPS have? "
                              << test_x[timestep][tag * 3 + 0] << " "
                              << test_x[timestep][tag * 3 + 1] << " "
                              << test_x[timestep][tag * 3 + 2]
                              << std::endl;
                    std::cout << "Diff: " << fabs(my_x[dim] - test_x[timestep][tag * 3 + dim]) << std::endl;
                    std::cout << "pos: " << x[test_idx].x << " " << x[test_idx].y << " " << x[test_idx].z << std::endl;

                    for (int tmp = 0; tmp < 3; tmp++) {
                        std::cout << "lo: " << zoid.zoid.cuts[tmp].lower +
                                               zoid.zoid.cuts[tmp].slope_lower * (timestep % (NUM_TIMESTEPS_IN_PARALLEL + 1))
                                  << std::endl;
                        std::cout << "hi: "
                                  << zoid.zoid.cuts[tmp].upper +
                                     zoid.zoid.cuts[tmp].slope_upper * (timestep % (NUM_TIMESTEPS_IN_PARALLEL + 1))
                                  << std::endl;
                    }

                    assert(false);
                }
            }
        }
    }

    void TEST_VEL_AGAINST_LAMMPS_DOUBLE_BUFFERING(bool curr_dt, int timestep, queue_info& zoid, Atom* atom_, double** test_v) {

        const auto& x = zoid.x_stencil_md[timestep % DOUBLE_BUFFERING];
        // const auto& v = zoid.v_stencil_md[timestep % DOUBLE_BUFFERING];
        const auto& v = zoid.v_stencil_md[timestep % 1];

        std::unordered_map<int, int> zoid_tag_to_idx;
        for (int i = 0; i < zoid.tag_stencil_md[0].size(); i++) {
            zoid_tag_to_idx[zoid.tag_stencil_md[0][i]] = i;
        }

        for (int k = 0; k < atom_->nlocal; k++) {
            // compare forces on local atoms?
            int tag = atom_->tag[k];
            int test_idx = zoid_tag_to_idx[tag];

            double my_v[3] = {v[test_idx].x, v[test_idx].y, v[test_idx].z};

            for (int dim = 0; dim < 3; dim++) {
                if (fabs(my_v[dim] - test_v[timestep][tag * 3 + dim]) > 1e-6) {
                    if (curr_dt) {
                        std::cout << "------VEL DIFF--------" << std::endl;
                    } else {
                        std::cout << "------NEXT DT VEL DIFF--------" << std::endl;
                    }

                    std::cout << "idx: " << k << " out of: " << atom_->nlocal << " atom: " << atom_ << " real idx: " << test_idx << std::endl;
                    std::cout << "Dim: " << dim << " Zoid: " << zoid.num << " timestep: " << timestep << " tag: " << tag << std::endl;
                    std::cout << "what I have: " << my_v[0] << " " << my_v[1] << " " << my_v[2] << std::endl;
                    std::cout << "What does LAMMPS have? "
                              << test_v[timestep][tag * 3 + 0] << " "
                              << test_v[timestep][tag * 3 + 1] << " "
                              << test_v[timestep][tag * 3 + 2]
                              << std::endl;
                    std::cout << "What does LAMMPS have prev? "
                              << test_v[timestep - 1][tag * 3 + 0] << " "
                              << test_v[timestep - 1][tag * 3 + 1] << " "
                              << test_v[timestep - 1][tag * 3 + 2]
                              << std::endl;
                    std::cout << "Diff: " << fabs(my_v[dim] - test_v[timestep][tag * 3 + dim]) << std::endl;
                    std::cout << "pos: " << x[test_idx].x << " " << x[test_idx].y << " " << x[test_idx].z << std::endl;

                    for (int tmp = 0; tmp < 3; tmp++) {
                        std::cout << "lo: " << zoid.zoid.cuts[tmp].lower +
                                               zoid.zoid.cuts[tmp].slope_lower * (timestep % (NUM_TIMESTEPS_IN_PARALLEL + 1))
                                  << std::endl;
                        std::cout << "hi: "
                                  << zoid.zoid.cuts[tmp].upper +
                                     zoid.zoid.cuts[tmp].slope_upper * (timestep % (NUM_TIMESTEPS_IN_PARALLEL + 1))
                                  << std::endl;
                    }

                    assert(false);
                }
            }
        }
    }

    void TEST_AGAINST_LAMMPS_FORCE_DOUBLE_BUFFERING_SETUP(double* test_f, queue_info& zoid, int timestep) {
        auto& x = zoid.x_stencil_md[timestep % DOUBLE_BUFFERING];
        // auto& f = zoid.f_stencil_md[timestep % DOUBLE_BUFFERING];
        // auto& eval_f = zoid.eval_f_stencil_md[timestep % DOUBLE_BUFFERING];
        auto& f = zoid.f_stencil_md[timestep % 1];
        auto& eval_f = zoid.eval_f_stencil_md[timestep % 1];
        const auto& local_idxs = zoid.local_idxs_per_timestep[timestep];
        auto& tags = zoid.tag_stencil_md[timestep % DOUBLE_BUFFERING];

        for (int i = 0; i < local_idxs.size(); i++) {
            int idx = local_idxs[i];
            int tag = tags[idx];

            double my_x = f[idx].x + eval_f[idx].x;
            double my_y = f[idx].y + eval_f[idx].y;
            double my_z = f[idx].z + eval_f[idx].z;

            bool all_close = (fabs(test_f[tag * 3] - my_x) < 5e-5) && (fabs(test_f[tag * 3 + 1] - my_y) < 5e-5) && (fabs(test_f[tag * 3 + 2] - my_z) < 5e-5);

            if (!all_close) {
                std::cout << RED << "ERROR ON FORCE zoid: " << zoid.num << " idx: " << idx << " tag: " << tag
                          << " what I have: " << my_x << " " << my_y << " " << my_z
                          << " what lammps has: " << test_f[tag * 3] << " " << test_f[tag * 3 + 1] << " " << test_f[tag * 3 + 2]
                          << " diff: "
                          << fabs(test_f[tag * 3] - my_x) << " " << fabs(test_f[tag * 3 + 1] - my_y) << " " << fabs(test_f[tag * 3 + 2] - my_z)
                          << " breakdown : " << f[idx].x << " " << f[idx].y << " " << f[idx].z
                          << " eval f: " << eval_f[idx].x << " " << eval_f[idx].y << " " << eval_f[idx].z
                          << RESET_COLOR << std::endl;

                std::cout << " pos: " << x[idx].x << " " << x[idx].y << " " << x[idx].z << std::endl;
                assert(false);
            }
        }
    }

    void TEST_FORCE_AGAINST_LAMMPS_DOUBLE_BUFFERING_SPACE_CUT(bool curr_dt, int timestep, queue_info& zoid, Atom* atom_, double** test_f, std::vector<int>& space_cut_idxs) {
        std::set<int> space_cut_idxs_set(space_cut_idxs.begin(), space_cut_idxs.end());

        const auto& x = zoid.x_stencil_md[timestep % DOUBLE_BUFFERING];

        // const auto& f = zoid.f_stencil_md[timestep % DOUBLE_BUFFERING];
        // const auto& eval_f = zoid.eval_f_stencil_md[timestep % DOUBLE_BUFFERING];
        const auto& f = zoid.f_stencil_md[timestep % 1];
        const auto& eval_f = zoid.eval_f_stencil_md[timestep % 1];

        std::unordered_map<int, int> zoid_tag_to_idx;
        for (int i = 0; i < zoid.tag_stencil_md[0].size(); i++) {
            zoid_tag_to_idx[zoid.tag_stencil_md[0][i]] = i;
        }

        for (int k = 0; k < atom_->nlocal; k++) {
            // compare forces on local atoms?
            int tag = atom_->tag[k];
            int test_idx = zoid_tag_to_idx[tag];

            if (space_cut_idxs_set.find(test_idx) == space_cut_idxs_set.end()) {
                continue;
            }

            double my_f[3] = {f[test_idx].x,
                              f[test_idx].y,
                              f[test_idx].z};

            double my_eval_f[3] = {eval_f[test_idx].x,
                                   eval_f[test_idx].y,
                                   eval_f[test_idx].z};

            for (int dim = 0; dim < 3; dim++) {
                double my_force = my_f[dim] + my_eval_f[dim];
                if (fabs(my_force - test_f[timestep][tag * 3 + dim]) > 1e-6) {
                    if (curr_dt) {
                        std::cout << "------FORCE DIFF--------"
                                  << std::endl;
                    } else {
                        std::cout << "------NEXT DT FORCE DIFF--------"
                                  << std::endl;
                    }

                    std::cout << "idx: " << k << " out of: " << atom_->nlocal << " double buffering idx: " << test_idx << std::endl;
                    std::cout << "Dim: " << dim << " Zoid: " << zoid.num << " timestep: " << timestep << " tag: " << tag << std::endl;
                    std::cout << "what I have: " << my_f[0] + my_eval_f[0] << " " << my_f[1] + my_eval_f[1] << " " << my_f[2] + my_eval_f[2] << std::endl;
                    std::cout << "my f: " << my_f[0] << " " << my_f[1] << " " << my_f[2]
                              << " my eval f: " << my_eval_f[0] << " " << my_eval_f[1] << " " << my_eval_f[2]
                              << std::endl;
                    std::cout << "What does LAMMPS have? "
                              << test_f[timestep][tag * 3 + 0] << " "
                              << test_f[timestep][tag * 3 + 1] << " "
                              << test_f[timestep][tag * 3 + 2]
                              << std::endl;
                    std::cout << "Diff: " << fabs(my_force - test_f[timestep][tag * 3 + dim]) << std::endl;
                    std::cout << "pos: " << x[test_idx].x << " " << x[test_idx].y << " " << x[test_idx].z << std::endl;

                    for (int tmp = 0; tmp < 3; tmp++) {
                        std::cout << "lo: " << zoid.zoid.cuts[tmp].lower +
                                               zoid.zoid.cuts[tmp].slope_lower * (timestep % (NUM_TIMESTEPS_IN_PARALLEL + 1))
                                  << std::endl;
                        std::cout << "hi: "
                                  << zoid.zoid.cuts[tmp].upper +
                                     zoid.zoid.cuts[tmp].slope_upper * (timestep % (NUM_TIMESTEPS_IN_PARALLEL + 1))
                                  << std::endl;
                    }

                    assert(false);
                }
            }
        }
    }

    void TEST_POS_AGAINST_LAMMPS_DOUBLE_BUFFERING_SPACE_CUT(bool curr_dt, int timestep, queue_info& zoid, Atom* atom_, double** test_x, std::vector<int>& space_cut_idxs) {
        const auto& x = zoid.x_stencil_md[timestep % DOUBLE_BUFFERING];

        std::set<int> space_cut_idxs_set(space_cut_idxs.begin(), space_cut_idxs.end());

        std::unordered_map<int, int> zoid_tag_to_idx;
        for (int i = 0; i < zoid.tag_stencil_md[0].size(); i++) {
            zoid_tag_to_idx[zoid.tag_stencil_md[0][i]] = i;
        }

        for (int k = 0; k < atom_->nlocal; k++) {
            // compare forces on local atoms?
            int tag = atom_->tag[k];
            int test_idx = zoid_tag_to_idx[tag];

            if (space_cut_idxs_set.find(test_idx) == space_cut_idxs_set.end()) {
                continue;
            }

            double my_x[3] = {x[test_idx].x, x[test_idx].y, x[test_idx].z};

            for (int dim = 0; dim < 3; dim++) {
                double my_pos = my_x[dim];
                if (my_pos < 0) {
                    my_pos += domain->prd[dim];
                } else if (my_pos >= domain->prd[dim]) {
                    my_pos -= domain->prd[dim];
                }

                double test_pos = test_x[timestep][tag * 3 + dim];
                if (test_pos < 0) {
                    test_pos += domain->prd[dim];
                } else if (test_pos >= domain->prd[dim]) {
                    test_pos -= domain->prd[dim];
                }

                if (fabs(my_pos - test_pos) > 1e-6) {
                    if (curr_dt) {
                        std::cout << "------POS DIFF--------"
                                  << std::endl;
                    } else {
                        std::cout << "------NEXT DT POS DIFF--------"
                                  << std::endl;
                    }

                    std::cout << "my pos: " << my_pos << " test pos: " << test_pos << std::endl;
                    std::cout << "idx: " << k << " out of: " << atom_->nlocal << " atom: " << atom_ << " real idx: " << test_idx << std::endl;
                    std::cout << "Dim: " << dim << " Zoid: " << zoid.num << " timestep: " << timestep << " tag: " << tag << std::endl;
                    std::cout << "what I have: " << my_x[0] << " " << my_x[1] << " " << my_x[2] << std::endl;
                    std::cout << "What does LAMMPS have? "
                              << test_x[timestep][tag * 3 + 0] << " "
                              << test_x[timestep][tag * 3 + 1] << " "
                              << test_x[timestep][tag * 3 + 2]
                              << std::endl;
                    std::cout << "Diff: " << fabs(my_x[dim] - test_x[timestep][tag * 3 + dim]) << std::endl;
                    std::cout << "pos: " << x[test_idx].x << " " << x[test_idx].y << " " << x[test_idx].z << std::endl;

                    for (int tmp = 0; tmp < 3; tmp++) {
                        std::cout << "lo: " << zoid.zoid.cuts[tmp].lower +
                                               zoid.zoid.cuts[tmp].slope_lower * (timestep % (NUM_TIMESTEPS_IN_PARALLEL + 1))
                                  << std::endl;
                        std::cout << "hi: "
                                  << zoid.zoid.cuts[tmp].upper +
                                     zoid.zoid.cuts[tmp].slope_upper * (timestep % (NUM_TIMESTEPS_IN_PARALLEL + 1))
                                  << std::endl;
                    }

                    assert(false);
                }
            }
        }
    }

    void TEST_VEL_AGAINST_LAMMPS_DOUBLE_BUFFERING_SPACE_CUT(bool curr_dt, int timestep, queue_info& zoid, Atom* atom_, double** test_v, std::vector<int>& space_cut_idxs) {
        const auto& x = zoid.x_stencil_md[timestep % DOUBLE_BUFFERING];
        // const auto& v = zoid.v_stencil_md[timestep % DOUBLE_BUFFERING];
        const auto& v = zoid.v_stencil_md[timestep % 1];

        std::set<int> space_cut_idxs_set(space_cut_idxs.begin(), space_cut_idxs.end());

        std::unordered_map<int, int> zoid_tag_to_idx;
        for (int i = 0; i < zoid.tag_stencil_md[0].size(); i++) {
            zoid_tag_to_idx[zoid.tag_stencil_md[0][i]] = i;
        }

        for (int k = 0; k < atom_->nlocal; k++) {
            // compare forces on local atoms?
            int tag = atom_->tag[k];
            int test_idx = zoid_tag_to_idx[tag];

            if (space_cut_idxs_set.find(test_idx) == space_cut_idxs_set.end()) {
                continue;
            }

            double my_v[3] = {v[test_idx].x, v[test_idx].y, v[test_idx].z};

            for (int dim = 0; dim < 3; dim++) {
                if (fabs(my_v[dim] - test_v[timestep][tag * 3 + dim]) > 1e-6) {
                    if (curr_dt) {
                        std::cout << "------VEL DIFF--------" << std::endl;
                    } else {
                        std::cout << "------NEXT DT VEL DIFF--------" << std::endl;
                    }

                    std::cout << "idx: " << k << " out of: " << atom_->nlocal << " atom: " << atom_ << " real idx: " << test_idx << std::endl;
                    std::cout << "Dim: " << dim << " Zoid: " << zoid.num << " timestep: " << timestep << " tag: " << tag << std::endl;
                    std::cout << "what I have: " << my_v[0] << " " << my_v[1] << " " << my_v[2] << std::endl;
                    std::cout << "What does LAMMPS have? "
                              << test_v[timestep][tag * 3 + 0] << " "
                              << test_v[timestep][tag * 3 + 1] << " "
                              << test_v[timestep][tag * 3 + 2]
                              << std::endl;
                    if (timestep > 0) {
                        std::cout << "What does LAMMPS have prev? "
                                  << test_v[timestep - 1][tag * 3 + 0] << " "
                                  << test_v[timestep - 1][tag * 3 + 1] << " "
                                  << test_v[timestep - 1][tag * 3 + 2]
                                  << std::endl;
                    }
                    std::cout << "Diff: " << fabs(my_v[dim] - test_v[timestep][tag * 3 + dim]) << std::endl;
                    std::cout << "pos: " << x[test_idx].x << " " << x[test_idx].y << " " << x[test_idx].z << std::endl;

                    for (int tmp = 0; tmp < 3; tmp++) {
                        std::cout << "lo: " << zoid.zoid.cuts[tmp].lower +
                                               zoid.zoid.cuts[tmp].slope_lower * (timestep % (NUM_TIMESTEPS_IN_PARALLEL + 1))
                                  << std::endl;
                        std::cout << "hi: "
                                  << zoid.zoid.cuts[tmp].upper +
                                     zoid.zoid.cuts[tmp].slope_upper * (timestep % (NUM_TIMESTEPS_IN_PARALLEL + 1))
                                  << std::endl;
                    }

                    assert(false);
                }
            }
        }
    }

    /* End double buffering code */
    // PIPELINE Flattening of tasks
    struct task {
        int zoid_num;
        int chunk_num;
    };

    /* Start code for many zoids per dimension */
    std::vector<queue_info> queues_many_cuts[NUM_DEPS];
    std::vector<queue_info> queues_many_cuts_next_dt[NUM_DEPS];
    int NUM_CUTS_PER_DIMENSION = 4;
    int NUM_ZOIDS_PER_DIMENSION = NUM_CUTS_PER_DIMENSION * 2;
    int NUM_ZOIDS_MANY_CUTS = NUM_ZOIDS_PER_DIMENSION * NUM_ZOIDS_PER_DIMENSION * NUM_ZOIDS_PER_DIMENSION;
    queue_info* zoid_num_to_zoid_many_cuts;
    queue_info* zoid_num_to_zoid_many_cuts_next_dt;
    std::vector<int>* send_to_neighbors_many_cuts;
    std::vector<int>* send_to_neighbors_many_cuts_next_dt;
    std::vector<int>* recv_from_neighbors_many_cuts;
    std::vector<int>* recv_from_neighbors_many_cuts_next_dt;

    std::vector<int>* recv_proc_sizes[NUM_DEPS];
    std::vector<int>* send_proc_sizes[NUM_DEPS];

    double** buf_recv_many_cuts;
    double** buf_send_many_cuts;
    int* nrecv_buf_many_cuts;
    int* nsend_buf_many_cuts;

    void INIT_ZOIDS_MANY_CUTS() {
        // TODO: test out more than 1 cut in each dimension
        double width = domain->boxhi[0] - domain->boxlo[0];
        double narrow_base_width = ((width / NUM_CUTS_PER_DIMENSION) - 2 * NUM_TIMESTEPS_IN_PARALLEL * ALLEGRO_SLOPE) / 2;
        double wide_base_width = (width - NUM_CUTS_PER_DIMENSION * narrow_base_width) / NUM_CUTS_PER_DIMENSION;

        std::cout << "narrow base width: " << narrow_base_width << " wide base width: " << wide_base_width << " total: " << NUM_CUTS_PER_DIMENSION * (narrow_base_width + wide_base_width) << " width: " << width << std::endl;

        double first_lo = domain->boxlo[0] - narrow_base_width / 2.0;
        double first_hi = domain->boxlo[0] + narrow_base_width / 2.0;

        double second_lo = first_hi;
        double second_hi = second_lo + wide_base_width;

        std::vector<double> bounds;
        bounds.push_back(first_lo);
        bounds.push_back(second_lo);
        bounds.push_back(second_hi);

        double one_set_width = narrow_base_width + wide_base_width;

        for (int i = 0; i < NUM_CUTS_PER_DIMENSION - 1; i++) {
            double lo = bounds[bounds.size() - 1];
            double hi = lo + narrow_base_width;
            double next_hi = std::min(domain->boxhi[0] - narrow_base_width / 2, hi + wide_base_width);
            bounds.push_back(hi);
            bounds.push_back(next_hi);
        }

        for (int i = 0; i < bounds.size() - 1; i++) {
            for (int j = 0; j < bounds.size() - 1; j++) {
                for (int k = 0; k < bounds.size() - 1; k++) {
                    int num_expanding = (i % 2 == 0) + (j % 2 == 0) + (k % 2 == 0);
                    queue_info zoid;
                    zoid.zoid.cuts[0].lower = bounds[i];
                    zoid.zoid.cuts[1].lower = bounds[j];
                    zoid.zoid.cuts[2].lower = bounds[k];
                    zoid.zoid.cuts[0].upper = bounds[i + 1];
                    zoid.zoid.cuts[1].upper = bounds[j + 1];
                    zoid.zoid.cuts[2].upper = bounds[k + 1];
                    // expanding
                    int factor_i = (i % 2 == 0) ? -1 : 1;
                    int factor_j = (j % 2 == 0) ? -1 : 1;
                    int factor_k = (k % 2 == 0) ? -1 : 1;

                    zoid.zoid.cuts[0].slope_lower = factor_i * ALLEGRO_SLOPE;
                    zoid.zoid.cuts[1].slope_lower = factor_j * ALLEGRO_SLOPE;
                    zoid.zoid.cuts[2].slope_lower = factor_k * ALLEGRO_SLOPE;

                    zoid.zoid.cuts[0].slope_upper = -factor_i * ALLEGRO_SLOPE;
                    zoid.zoid.cuts[1].slope_upper = -factor_j * ALLEGRO_SLOPE;
                    zoid.zoid.cuts[2].slope_upper = -factor_k * ALLEGRO_SLOPE;

                    zoid.where[0] = i;
                    zoid.where[1] = j;
                    zoid.where[2] = k;

                    queues_many_cuts[num_expanding].push_back(zoid);
                }
            }
        }

        int zoid_num = 0;
        for (int dep = 0; dep < NUM_DEPS; dep++) {
            for (int j = 0; j < queues_many_cuts[dep].size(); j++) {
                auto& zoid = queues_many_cuts[dep][j];
                zoid.num = zoid_num;
                zoid_num++;
                int new_dep = NUM_DEPS - 1 - dep;
                queue_info new_zoid;
                for (int dim = 0; dim < 3; dim++) {
                    double new_start =
                            zoid.zoid.cuts[dim].lower +
                            NUM_TIMESTEPS_IN_PARALLEL * zoid.zoid.cuts[dim].slope_lower;
                    double new_end =
                            zoid.zoid.cuts[dim].upper +
                            NUM_TIMESTEPS_IN_PARALLEL * zoid.zoid.cuts[dim].slope_upper;
                    new_zoid.zoid.cuts[dim].lower = new_start;
                    new_zoid.zoid.cuts[dim].upper = new_end;
                    new_zoid.zoid.cuts[dim].slope_lower =
                            -1 * zoid.zoid.cuts[dim].slope_lower;
                    new_zoid.zoid.cuts[dim].slope_upper =
                            -1 * zoid.zoid.cuts[dim].slope_upper;
                }

                new_zoid.num = zoid.num;

                for (int dim = 0; dim < domain->dimension; dim++) {
                    new_zoid.where[dim] = zoid.where[dim];
                }
                queues_many_cuts_next_dt[new_dep].push_back(new_zoid);
            }
        }

        if (comm->me == 0) {
            std::cout << BOLDCYAN << "Lo: " << domain->boxlo[0] << " hi: " << domain->boxhi[0] << RESET_COLOR << std::endl;
            std::stringstream s1;
            for (auto& b : bounds) {
                s1 << b << " ";
            }

            std::cout << BOLDCYAN << "bounds: " << s1.str() << RESET_COLOR << std::endl;
        }

        for (int idx = 0; idx < 2; idx++) {
            // skip for now
            continue;
            auto* queues = (idx == 0) ? queues_many_cuts : queues_many_cuts_next_dt;
            for (int t = 0; t < NUM_TIMESTEPS_IN_PARALLEL + 1; t++) {
                int total_num_atoms_in_bounds = 0;

                std::set<int> atoms_in;
                std::map<int, std::pair<int, int>> tag_to_zoid;

                for (int dep = 0; dep < NUM_DEPS; dep++) {
                    for (int j = 0; j < queues[dep].size(); j++) {
                        auto &zoid = queues[dep][j];
                        int num_atoms_in_bounds = 0;
                        int num_atoms_in_bounds_all = 0;
                        for (int i = 0; i < atom->nlocal; i++) {
                            bool in = true;
                            for (int dim = 0; dim < domain->dimension; dim++) {
                                double pos = atom->x[i][dim];
                                double lo = zoid.zoid.cuts[dim].lower + t * zoid.zoid.cuts[dim].slope_lower;
                                double hi = zoid.zoid.cuts[dim].upper + t * zoid.zoid.cuts[dim].slope_upper;
                                if (hi <= lo) {
                                    std::cout << "dep: " << dep << " j: " << j << " lo: " << lo << " hi: " << hi
                                              << " time: " << t
                                              << " lower: " << zoid.zoid.cuts[dim].lower << " upper: "
                                              << zoid.zoid.cuts[dim].upper
                                              << " slope lower: " << zoid.zoid.cuts[dim].slope_lower << " slope upper: "
                                              << zoid.zoid.cuts[dim].slope_upper << std::endl;
                                }
                                assert(hi > lo);
                                while (pos < lo) {
                                    pos += domain->prd[dim];
                                }
                                while (pos >= hi) {
                                    pos -= domain->prd[dim];
                                }
                                in = in && pos >= lo && pos < hi;
                            }
                            if (in) {
                                if (atoms_in.find(atom->tag[i]) != atoms_in.end()) {
                                    std::cout << "duplicate atom. Found in dep: " << dep << " time: " << t
                                              << " pos: " << atom->x[i][0] << " " << atom->x[i][1] << " "
                                              << atom->x[i][2] << std::endl;
                                    for (int dim = 0; dim < domain->dimension; dim++) {
                                        double lo = zoid.zoid.cuts[dim].lower + t * zoid.zoid.cuts[dim].slope_lower;
                                        double hi = zoid.zoid.cuts[dim].upper + t * zoid.zoid.cuts[dim].slope_upper;
                                        std::cout << "dim: " << dim << " lo: " << lo << " hi: " << hi << std::endl;
                                    }

                                    auto &p = tag_to_zoid.at(atom->tag[i]);
                                    std::cout << "prev: " << p.first << " " << p.second << std::endl;
                                    auto &other_zoid = queues_many_cuts[p.first][p.second];
                                    for (int dim = 0; dim < domain->dimension; dim++) {
                                        double lo = other_zoid.zoid.cuts[dim].lower +
                                                    t * other_zoid.zoid.cuts[dim].slope_lower;
                                        double hi = other_zoid.zoid.cuts[dim].upper +
                                                    t * other_zoid.zoid.cuts[dim].slope_upper;
                                        std::cout << "OTHER ZOID dim: " << dim << " lo: " << lo << " hi: " << hi
                                                  << std::endl;
                                    }

                                    assert(false);
                                }
                                atoms_in.insert(atom->tag[i]);
                                tag_to_zoid[atom->tag[i]] = {dep, j};
                                num_atoms_in_bounds++;
                            }
                        }
                        MPI_Allreduce(&num_atoms_in_bounds, &num_atoms_in_bounds_all, 1, MPI_INT, MPI_SUM, world);
                        total_num_atoms_in_bounds += num_atoms_in_bounds_all;
                    }
                }

                assert(total_num_atoms_in_bounds == atom->natoms);
            }
        }
    }

    void INIT_ZOID_DATA_MANY_CUTS() {
        std::map<int, std::pair<int, int>> zoid_num_to_coord;
        for (int dep = 0; dep < NUM_DEPS; dep++) {
            for (int j = 0; j < queues_many_cuts[dep].size(); j++) {
                auto& zoid = queues_many_cuts[dep][j];
                zoid_num_to_coord[zoid.num] = {dep, j};
            }
        }

        for (int dep = 0; dep < NUM_DEPS; dep++) {
            for (int j = 0; j < queues_many_cuts[dep].size(); j++) {
                queue_info& zoid = queues_many_cuts[dep][j];
                if (zoid.num % comm->nprocs == comm->me) {
                    /* start stuff for 2 timesteps */
                    zoid.per_worker_force_updates = new std::pair<int, dbl3_t_stencil_md>*[__cilkrts_get_nworkers()];
                    for (int i = 0; i < __cilkrts_get_nworkers(); i++) {
                        zoid.per_worker_force_updates[i] = new std::pair<int, dbl3_t_stencil_md>[MODIFY_GRAINSIZE * MAX_NEIGHBORS_PER_ATOM];
                    }

                    zoid.x_stencil_md = new std::vector<dbl3_t_stencil_md>[DOUBLE_BUFFERING];
                    // zoid.v_stencil_md = new std::vector<dbl3_t_stencil_md>[DOUBLE_BUFFERING];
                    // zoid.f_stencil_md = new std::vector<dbl3_t_stencil_md>[DOUBLE_BUFFERING];
                    // zoid.eval_f_stencil_md = new std::vector<dbl3_t_stencil_md>[DOUBLE_BUFFERING];
                    zoid.v_stencil_md = new std::vector<dbl3_t_stencil_md>[1];
                    zoid.f_stencil_md = new std::vector<dbl3_t_stencil_md>[1];
                    zoid.eval_f_stencil_md = new std::vector<dbl3_t_stencil_md>[1];

                    zoid.tag_stencil_md = new std::vector<int>[1];
                    zoid.type_stencil_md = new std::vector<int>[1];
                    zoid.mask_stencil_md = new std::vector<int>[1];
                    zoid.image_stencil_md = new std::vector<int>[1];
                    zoid.spinlocks_stencil_md = new spinlock*[1];
                    zoid.claimed_flags_stencil_md = new std::atomic_flag*[1];

                    zoid.local_idxs_per_timestep = new std::vector<int>[NUM_TIMESTEPS_IN_PARALLEL + 1];
                    zoid.neighbor_list = new std::vector<std::vector<int>>[NUM_TIMESTEPS_IN_PARALLEL + 1];
                    zoid.bond_list = new std::vector<std::vector<std::pair<int, int>>>[NUM_TIMESTEPS_IN_PARALLEL + 1];
                    // zoid.neighbor_list = new std::vector<int>*[1];
                    // zoid.bond_list = new std::vector<std::pair<int, int>>*[1];

                    zoid.send_force_idxs_double_buffering = new std::vector<int>*[NUM_TIMESTEPS_IN_PARALLEL + 1];
                    zoid.recv_force_idxs_double_buffering = new std::vector<int>*[NUM_TIMESTEPS_IN_PARALLEL + 1];

                    zoid.send_pos_idxs_double_buffering = new std::vector<int>*[NUM_TIMESTEPS_IN_PARALLEL + 1];
                    zoid.recv_pos_idxs_double_buffering = new std::vector<int>*[NUM_TIMESTEPS_IN_PARALLEL + 1];
                    zoid.recv_pos_local_idxs_double_buffering = new std::vector<int>*[NUM_TIMESTEPS_IN_PARALLEL + 1];
                    zoid.recv_pos_ghost_idxs_double_buffering = new std::vector<int>*[NUM_TIMESTEPS_IN_PARALLEL + 1];

                    zoid.send_vel_idxs_double_buffering = new std::vector<int>*[NUM_TIMESTEPS_IN_PARALLEL + 1];
                    zoid.recv_vel_idxs_double_buffering = new std::vector<int>*[NUM_TIMESTEPS_IN_PARALLEL + 1];
                    /* end stuff for 2 timesteps */
                }
            }
        }

        for (int dep = 0; dep < NUM_DEPS; dep++) {
            for (int j = 0; j < queues_many_cuts_next_dt[dep].size(); j++) {
                queue_info& zoid = queues_many_cuts_next_dt[dep][j];
                if (zoid.num % comm->nprocs == comm->me) {
                    /* start stuff for 2 timesteps */
                    zoid.per_worker_force_updates = new std::pair<int, dbl3_t_stencil_md>*[__cilkrts_get_nworkers()];
                    for (int i = 0; i < __cilkrts_get_nworkers(); i++) {
                        zoid.per_worker_force_updates[i] = new std::pair<int, dbl3_t_stencil_md>[MODIFY_GRAINSIZE * MAX_NEIGHBORS_PER_ATOM];
                    }
                    zoid.space_cut_idxs = new std::vector<int>*[NUM_TIMESTEPS_IN_PARALLEL + 1];
                    // Copy the main data from the curr_dt zoid
                    auto coord = zoid_num_to_coord[zoid.num];

                    zoid.x_stencil_md = queues_many_cuts[coord.first][coord.second].x_stencil_md;
                    zoid.v_stencil_md = queues_many_cuts[coord.first][coord.second].v_stencil_md;
                    zoid.f_stencil_md = queues_many_cuts[coord.first][coord.second].f_stencil_md;
                    zoid.eval_f_stencil_md = queues_many_cuts[coord.first][coord.second].eval_f_stencil_md;
                    zoid.tag_stencil_md = queues_many_cuts[coord.first][coord.second].tag_stencil_md;
                    zoid.type_stencil_md = queues_many_cuts[coord.first][coord.second].type_stencil_md;
                    zoid.mask_stencil_md = queues_many_cuts[coord.first][coord.second].mask_stencil_md;
                    zoid.image_stencil_md = queues_many_cuts[coord.first][coord.second].image_stencil_md;
                    zoid.spinlocks_stencil_md = queues_many_cuts[coord.first][coord.second].spinlocks_stencil_md;
                    zoid.claimed_flags_stencil_md = queues_many_cuts[coord.first][coord.second].claimed_flags_stencil_md;

                    zoid.local_idxs_per_timestep = new std::vector<int>[NUM_TIMESTEPS_IN_PARALLEL + 1];
                    zoid.neighbor_list = new std::vector<std::vector<int>>[NUM_TIMESTEPS_IN_PARALLEL + 1];
                    zoid.bond_list = new std::vector<std::vector<std::pair<int, int>>>[NUM_TIMESTEPS_IN_PARALLEL + 1];

                    zoid.send_force_idxs_double_buffering = new std::vector<int>*[NUM_TIMESTEPS_IN_PARALLEL + 1];
                    zoid.recv_force_idxs_double_buffering = new std::vector<int>*[NUM_TIMESTEPS_IN_PARALLEL + 1];

                    zoid.send_pos_idxs_double_buffering = new std::vector<int>*[NUM_TIMESTEPS_IN_PARALLEL + 1];
                    zoid.recv_pos_idxs_double_buffering = new std::vector<int>*[NUM_TIMESTEPS_IN_PARALLEL + 1];
                    zoid.recv_pos_local_idxs_double_buffering = new std::vector<int>*[NUM_TIMESTEPS_IN_PARALLEL + 1];
                    zoid.recv_pos_ghost_idxs_double_buffering = new std::vector<int>*[NUM_TIMESTEPS_IN_PARALLEL + 1];

                    zoid.send_vel_idxs_double_buffering = new std::vector<int>*[NUM_TIMESTEPS_IN_PARALLEL + 1];
                    zoid.recv_vel_idxs_double_buffering = new std::vector<int>*[NUM_TIMESTEPS_IN_PARALLEL + 1];
                    /* end stuff for 2 timesteps */
                }
            }
        }

        zoid_num_to_zoid_many_cuts = new queue_info[NUM_ZOIDS_MANY_CUTS];
        for (int dep = 0; dep < NUM_DEPS; dep++) {
            for (int j = 0; j < queues_many_cuts[dep].size(); j++) {
                int zoid_num = queues_many_cuts[dep][j].num;
                zoid_num_to_zoid_many_cuts[zoid_num] = queues_many_cuts[dep][j];
            }
        }

        zoid_num_to_zoid_many_cuts_next_dt = new queue_info[NUM_ZOIDS_MANY_CUTS];
        for (int dep = 0; dep < NUM_DEPS; dep++) {
            for (int j = 0; j < queues_many_cuts_next_dt[dep].size(); j++) {
                int zoid_num = queues_many_cuts_next_dt[dep][j].num;
                zoid_num_to_zoid_many_cuts_next_dt[zoid_num] = queues_many_cuts_next_dt[dep][j];
            }
        }
    }

    bool zoid_many_cuts_is_neighbor(int* where_a, int* where_b, bool print=false) {
        // diff in one spot
        int num_diff = 0;
        int diff = -1;

        for (int dim = 0; dim < domain->dimension; dim++) {
            if (where_a[dim] != where_b[dim]) {
                num_diff++;
                diff = std::abs(where_a[dim] - where_b[dim]);
            }
        }

        if (num_diff != 1) {
            return false;
        }

        return (diff == 1) || (diff == NUM_ZOIDS_PER_DIMENSION - 1);
    }

    void INIT_ZOID_MANY_CUTS_NEIGHBORS() {
        // for neighbors
        send_to_neighbors_many_cuts = new std::vector<int>[NUM_ZOIDS_MANY_CUTS];
        send_to_neighbors_many_cuts_next_dt = new std::vector<int>[NUM_ZOIDS_MANY_CUTS];
        recv_from_neighbors_many_cuts = new std::vector<int>[NUM_ZOIDS_MANY_CUTS];
        recv_from_neighbors_many_cuts_next_dt = new std::vector<int>[NUM_ZOIDS_MANY_CUTS];

        std::map<int, int> zoid_to_dep;
        for (int dep = 0; dep < NUM_DEPS; dep++) {
            for (int j = 0; j < queues_many_cuts[dep].size(); j++) {
                zoid_to_dep[queues_many_cuts[dep][j].num] = dep;
            }
        }

        std::map<int, int> zoid_to_dep_next_dt;
        for (int dep = 0; dep < NUM_DEPS; dep++) {
            for (int j = 0; j < queues_many_cuts_next_dt[dep].size(); j++) {
                zoid_to_dep_next_dt[queues_many_cuts_next_dt[dep][j].num] = dep;
            }
        }

        for (int i = 0; i < NUM_ZOIDS_MANY_CUTS; i++) {
            auto& zoid = zoid_num_to_zoid_many_cuts[i];
            for (int j = 0; j < NUM_ZOIDS_MANY_CUTS; j++) {
                auto& other_zoid = zoid_num_to_zoid_many_cuts[j];
                if (zoid_to_dep[zoid.num] + 1 == zoid_to_dep[other_zoid.num]) {
                    if (zoid_many_cuts_is_neighbor(zoid.where, other_zoid.where)) {
                        send_to_neighbors_many_cuts[i].push_back(j);
                        recv_from_neighbors_many_cuts[j].push_back(i);
                    }
                }
            }
        }

        for (int i = 0; i < NUM_ZOIDS_MANY_CUTS; i++) {
            auto& zoid = zoid_num_to_zoid_many_cuts_next_dt[i];
            for (int j = 0; j < NUM_ZOIDS_MANY_CUTS; j++) {
                auto& other_zoid = zoid_num_to_zoid_many_cuts_next_dt[j];
                if (zoid_to_dep_next_dt[zoid.num] + 1 == zoid_to_dep_next_dt[other_zoid.num]) {
                    if (zoid_many_cuts_is_neighbor(zoid.where, other_zoid.where)) {
                        send_to_neighbors_many_cuts_next_dt[i].push_back(j);
                        recv_from_neighbors_many_cuts_next_dt[j].push_back(i);
                    }
                }
            }
        }

        /*
        if (comm->me == 0) {
            for (int dep = 0; dep < NUM_DEPS; dep++) {
                for (int j = 0; j < queues_many_cuts[dep].size(); j++) {
                    auto& zoid = queues_many_cuts[dep][j];
                    std::cout << BOLDCYAN << "dep: " << dep << " j: " << j << " zoid: " << zoid.num
                              << " num send neighbors: " << send_to_neighbors_many_cuts[zoid.num].size()
                              << " num recv neighbors: " << recv_from_neighbors_many_cuts[zoid.num].size()
                              << RESET_COLOR << std::endl;

                }
            }
        }
        */
    }

    void GET_ATOMS_ZOID_MANY_CUTS() {
        int total_num_entries = atom->natoms + 1;
        auto* all_pos = new double[3 * total_num_entries];
        auto* all_vel = new double[3 * total_num_entries];
        auto* all_type = new int[total_num_entries];
        auto* all_mask = new int[total_num_entries];
        auto* all_image = new int[total_num_entries];

        memset(all_pos, 0, 3 * total_num_entries * sizeof(double));
        memset(all_vel, 0, 3 * total_num_entries * sizeof(double));
        memset(all_type, 0, total_num_entries * sizeof(int));
        memset(all_mask, 0, total_num_entries * sizeof(int));
        memset(all_image, 0, total_num_entries * sizeof(int));

        for (int i = 0; i < atom->nlocal; i++) {
            auto tag = atom->tag[i];
            all_pos[tag * 3 + 0] = atom->x[i][0];
            all_pos[tag * 3 + 1] = atom->x[i][1];
            all_pos[tag * 3 + 2] = atom->x[i][2];

            all_vel[tag * 3 + 0] = atom->v[i][0];
            all_vel[tag * 3 + 1] = atom->v[i][1];
            all_vel[tag * 3 + 2] = atom->v[i][2];

            all_type[tag] = atom->type[i];
            all_mask[tag] = atom->mask[i];
            all_image[tag] = atom->image[i];
        }

        MPI_Allreduce(MPI_IN_PLACE, all_pos, total_num_entries * 3, MPI_DOUBLE, MPI_SUM, world);
        MPI_Allreduce(MPI_IN_PLACE, all_vel, total_num_entries * 3, MPI_DOUBLE, MPI_SUM, world);
        MPI_Allreduce(MPI_IN_PLACE, all_type, total_num_entries, MPI_INT, MPI_SUM, world);
        MPI_Allreduce(MPI_IN_PLACE, all_mask, total_num_entries, MPI_INT, MPI_SUM, world);
        MPI_Allreduce(MPI_IN_PLACE, all_image, total_num_entries, MPI_INT, MPI_SUM, world);

        // process local atoms first
        cilk_for (int dep = 0; dep < NUM_DEPS; dep++) {
            cilk_for (int j = 0; j < queues_many_cuts[dep].size(); j++) {
                auto& zoid = queues_many_cuts[dep][j];
                if (zoid.num % comm->nprocs != comm->me) {
                    continue;
                }
                std::unordered_set<int> tags_in_zoid;
                for (int idx = 1; idx < total_num_entries; idx++) {
                    for (int t = 0; t < NUM_TIMESTEPS_IN_PARALLEL + 1; t++) {
                        bool in_zoid = true;
                        bool borders_zoid = true;

                        double new_pos[3] = {0};
                        double new_pos_borders[3] = {0};

                        for (int dim = 0; dim < domain->dimension; dim++) {
                            double pos = all_pos[idx * 3 + dim];
                            double lo = zoid.zoid.cuts[dim].lower + t * zoid.zoid.cuts[dim].slope_lower;
                            double hi = zoid.zoid.cuts[dim].upper + t * zoid.zoid.cuts[dim].slope_upper;
                            while (pos < lo) {
                                pos += domain->prd[dim];
                            }
                            while (pos >= hi) {
                                pos -= domain->prd[dim];
                            }
                            in_zoid = in_zoid && pos >= lo && pos < hi;
                            new_pos[dim] = pos;

                            double lo_borders = lo - ALLEGRO_SLOPE;
                            double hi_borders = hi + ALLEGRO_SLOPE;
                            while (pos < lo_borders) {
                                pos += domain->prd[dim];
                            }
                            while (pos >= hi_borders) {
                                pos -= domain->prd[dim];
                            }
                            borders_zoid = borders_zoid && pos >= lo_borders && pos < hi_borders;
                            new_pos_borders[dim] = pos;
                        }

                        if (in_zoid || borders_zoid) {
                            if (tags_in_zoid.find(idx) == tags_in_zoid.end()) {
                                tags_in_zoid.insert(idx);
                                if (in_zoid) {
                                    zoid.x_stencil_md[0].push_back({new_pos[0], new_pos[1], new_pos[2]});
                                    zoid.x_stencil_md[1].push_back({new_pos[0], new_pos[1], new_pos[2]});
                                } else if (borders_zoid) {
                                    zoid.x_stencil_md[0].push_back({new_pos_borders[0], new_pos_borders[1], new_pos_borders[2]});
                                    zoid.x_stencil_md[1].push_back({new_pos_borders[0], new_pos_borders[1], new_pos_borders[2]});
                                }
                                zoid.v_stencil_md[0].push_back({all_vel[idx * 3 + 0], all_vel[idx * 3 + 1], all_vel[idx * 3 + 2]});
                                zoid.tag_stencil_md[0].push_back(idx);
                                zoid.type_stencil_md[0].push_back(all_type[idx]);
                                zoid.mask_stencil_md[0].push_back(all_mask[idx]);
                                zoid.image_stencil_md[0].push_back(all_image[idx]);
                                zoid.f_stencil_md[0].push_back({0.0, 0.0, 0.0});
                                zoid.eval_f_stencil_md[0].push_back({0.0, 0.0, 0.0});
                            }
                            break;
                        }
                    }
                }
            }
        }

        delete[] all_pos;
        delete[] all_vel;
        delete[] all_type;
        delete[] all_mask;
        delete[] all_image;
    }

    void SORT_LOCAL_ATOMS_ZOID_MANY_CUTS() {
        // setup lammps code
        double binsize = 0.5 * neighbor->cutneighmax;
        double bininv = 1.0 / binsize;

        int nbinx = static_cast<int> ((domain->boxhi[0] - domain->boxlo[0]) * bininv);
        int nbiny = static_cast<int> ((domain->boxhi[1] - domain->boxlo[1]) * bininv);
        int nbinz = static_cast<int> ((domain->boxhi[2] - domain->boxlo[2]) * bininv);

        double bininvx = nbinx / (domain->boxhi[0] - domain->boxlo[0]);
        double bininvy = nbiny / (domain->boxhi[1] - domain->boxlo[1]);
        double bininvz = nbinz / (domain->boxhi[2] - domain->boxlo[2]);

        cilk_for (int dep = 0; dep < NUM_DEPS; dep++) {
            cilk_for (int j = 0; j < queues_many_cuts[dep].size(); j++) {
                queue_info &zoid = queues_many_cuts[dep][j];
                int zoid_num = zoid.num;
                // receive only if the zoid belongs to me
                if (zoid_num % comm->nprocs == comm->me) {
                    std::map<tagint, int> tag_to_idx;
                    for (int i = 0; i < zoid.tag_stencil_md[0].size(); i++) {
                        tag_to_idx[zoid.tag_stencil_md[0][i]] = i;
                    }

                    auto permutation = sort_permutation(zoid.tag_stencil_md[0],
                                                        [&](const tagint &tag_a, const tagint &tag_b) {
                                                            int idx_a = tag_to_idx[tag_a];
                                                            int idx_b = tag_to_idx[tag_b];

                                                            int num_timesteps_local_a = 0;
                                                            int num_timesteps_local_b = 0;

                                                            std::set<int> timesteps_a;
                                                            std::set<int> timesteps_b;

                                                            for (int t2 = 0; t2 < NUM_TIMESTEPS_IN_PARALLEL + 1; t2++) {
                                                                bool in_zoid = true;
                                                                double pos[3] = {zoid.x_stencil_md[0][idx_a].x,
                                                                                 zoid.x_stencil_md[0][idx_a].y,
                                                                                 zoid.x_stencil_md[0][idx_a].z};

                                                                for (int dim = 0; dim < NUM_DIMENSIONS; dim++) {
                                                                    double lo = zoid.zoid.cuts[dim].lower +
                                                                                t2 * zoid.zoid.cuts[dim].slope_lower;
                                                                    double hi = zoid.zoid.cuts[dim].upper +
                                                                                t2 * zoid.zoid.cuts[dim].slope_upper;
                                                                    if (!(pos[dim] >= lo && pos[dim] < hi)) {
                                                                        in_zoid = false;
                                                                    }
                                                                }

                                                                if (in_zoid) {
                                                                    num_timesteps_local_a++;
                                                                    timesteps_a.insert(t2);
                                                                }
                                                            }

                                                            for (int t2 = 0; t2 < NUM_TIMESTEPS_IN_PARALLEL + 1; t2++) {
                                                                bool in_zoid = true;
                                                                double pos[3] = {zoid.x_stencil_md[0][idx_b].x,
                                                                                 zoid.x_stencil_md[0][idx_b].y,
                                                                                 zoid.x_stencil_md[0][idx_b].z};

                                                                for (int dim = 0; dim < NUM_DIMENSIONS; dim++) {
                                                                    double lo = zoid.zoid.cuts[dim].lower +
                                                                                t2 * zoid.zoid.cuts[dim].slope_lower;
                                                                    double hi = zoid.zoid.cuts[dim].upper +
                                                                                t2 * zoid.zoid.cuts[dim].slope_upper;
                                                                    if (!(pos[dim] >= lo && pos[dim] < hi)) {
                                                                        in_zoid = false;
                                                                    }
                                                                }

                                                                if (in_zoid) {
                                                                    num_timesteps_local_b++;
                                                                    timesteps_b.insert(t2);
                                                                }
                                                            }

                                                            if (num_timesteps_local_a != num_timesteps_local_b) {
                                                                return num_timesteps_local_a > num_timesteps_local_b;
                                                            }

                                                            if (timesteps_a != timesteps_b) {
                                                                return timesteps_a > timesteps_b;
                                                            }

                                                            // USE LAMMPS SORTING
                                                            const auto &pos_a = zoid.x_stencil_md[0][idx_a];
                                                            int ix_a = static_cast<int> ((pos_a.x - domain->boxlo[0]) *
                                                                                         bininvx);
                                                            int iy_a = static_cast<int> ((pos_a.y - domain->boxlo[1]) *
                                                                                         bininvy);
                                                            int iz_a = static_cast<int> ((pos_a.z - domain->boxlo[2]) *
                                                                                         bininvz);

                                                            ix_a = MAX(ix_a, 0);
                                                            iy_a = MAX(iy_a, 0);
                                                            iz_a = MAX(iz_a, 0);
                                                            ix_a = MIN(ix_a, nbinx - 1);
                                                            iy_a = MIN(iy_a, nbiny - 1);
                                                            iz_a = MIN(iz_a, nbinz - 1);
                                                            int ibin_a = iz_a * nbiny * nbinx + iy_a * nbinx + ix_a;

                                                            const auto &pos_b = zoid.x_stencil_md[0][idx_b];
                                                            int ix_b = static_cast<int> ((pos_b.x - domain->boxlo[0]) *
                                                                                         bininvx);
                                                            int iy_b = static_cast<int> ((pos_b.y - domain->boxlo[1]) *
                                                                                         bininvy);
                                                            int iz_b = static_cast<int> ((pos_b.z - domain->boxlo[2]) *
                                                                                         bininvz);

                                                            ix_b = MAX(ix_b, 0);
                                                            iy_b = MAX(iy_b, 0);
                                                            iz_b = MAX(iz_b, 0);
                                                            ix_b = MIN(ix_b, nbinx - 1);
                                                            iy_b = MIN(iy_b, nbiny - 1);
                                                            iz_b = MIN(iz_b, nbinz - 1);
                                                            int ibin_b = iz_b * nbiny * nbinx + iy_b * nbinx + ix_b;

                                                            return ibin_a < ibin_b;
                                                        });

                    apply_permutation_in_place(zoid.x_stencil_md[0], permutation);
                    apply_permutation_in_place(zoid.x_stencil_md[1], permutation);

                    apply_permutation_in_place(zoid.tag_stencil_md[0], permutation);
                    apply_permutation_in_place(zoid.type_stencil_md[0], permutation);
                    apply_permutation_in_place(zoid.image_stencil_md[0], permutation);
                    apply_permutation_in_place(zoid.mask_stencil_md[0], permutation);
                    apply_permutation_in_place(zoid.v_stencil_md[0], permutation);
                }
            }
        }

        cilk_for (int dep = 0; dep < NUM_DEPS; dep++) {
            cilk_for (int j = 0; j < queues_many_cuts[dep].size(); j++) {
                queue_info &zoid = queues_many_cuts[dep][j];
                int zoid_num = zoid.num;
                if (zoid_num % comm->nprocs != comm->me) {
                    continue;
                }
                std::unordered_map<tagint, int> tag_to_idx;
                for (int i = 0; i < zoid.tag_stencil_md[0].size(); i++) {
                    tag_to_idx[zoid.tag_stencil_md[0][i]] = i;
                }

                for (int t = 0; t < NUM_TIMESTEPS_IN_PARALLEL + 1; t++) {
                    for (int i = 0; i < zoid.x_stencil_md[0].size(); i++) {
                        bool in_zoid = true;
                        double pos[3] = {zoid.x_stencil_md[0][i].x,
                                         zoid.x_stencil_md[0][i].y,
                                         zoid.x_stencil_md[0][i].z};

                        double zoid_lo[3] = {0};
                        double zoid_hi[3] = {0};

                        for (int dim = 0; dim < domain->dimension; dim++) {
                            double lo = zoid.zoid.cuts[dim].lower +
                                        t * zoid.zoid.cuts[dim].slope_lower;
                            double hi = zoid.zoid.cuts[dim].upper +
                                        t * zoid.zoid.cuts[dim].slope_upper;
                            in_zoid = in_zoid && pos[dim] >= lo && pos[dim] < hi;

                            zoid_lo[dim] = lo;
                            zoid_hi[dim] = hi;
                        }

                        if (in_zoid) {
                            zoid.local_idxs_per_timestep[t].push_back(i);
                        }
                    }

                    std::vector<int> tmp1;
                    std::vector<int> tmp2;
                    int num_segments = get_segments(zoid.local_idxs_per_timestep[t], tmp1, tmp2);
                    /*
                    std::cout << BOLDYELLOW << "CURR DT dep: " << dep << " zoid: " << zoid.num << " time: " << t << " nlocal: " << zoid.local_idxs_per_timestep[t].size()
                        << " num segments: " << num_segments << RESET_COLOR << std::endl;
                    */
                }
            }
        }

        // debug curr_dt
        int total_num_entries = atom->natoms + 1;
        auto all_tags = new int[total_num_entries];
        memset(all_tags, 0, total_num_entries * sizeof(int));
        for (int t = 0; t < NUM_TIMESTEPS_IN_PARALLEL + 1; t++) {
            for (int dep = 0; dep < NUM_DEPS; dep++) {
                for (int j = 0; j < queues_many_cuts[dep].size(); j++) {
                    auto& zoid = queues_many_cuts[dep][j];
                    int zoid_num = zoid.num;
                    if (zoid_num % comm->nprocs != comm->me) {
                        continue;
                    }

                    auto& local_idxs = zoid.local_idxs_per_timestep[t];
                    for (int i = 0; i < local_idxs.size(); i++) {
                        int idx = local_idxs[i];
                        int tag = zoid.tag_stencil_md[0][idx];
                        all_tags[tag]++;
                    }
                }
            }

            MPI_Allreduce(MPI_IN_PLACE, all_tags, total_num_entries, MPI_INT, MPI_SUM, world);
            assert(all_tags[0] == 0);
            for (int i = 1; i < total_num_entries; i++) {
                if (all_tags[i] != 1) {
                    std::cout << "curr_dt timestep: " << t << " tag: " << i << " all tags: " << all_tags[i] << std::endl;
                }
                assert(all_tags[i] == 1);
            }
            memset(all_tags, 0, total_num_entries * sizeof(int));
        }

        cilk_for (int dep = 0; dep < NUM_DEPS; dep++) {
            cilk_for (int j = 0; j < queues_many_cuts_next_dt[dep].size(); j++) {
                queue_info &zoid = queues_many_cuts_next_dt[dep][j];
                int zoid_num = zoid.num;
                if (zoid_num % comm->nprocs != comm->me) {
                    continue;
                }
                std::unordered_map<tagint, int> tag_to_idx;
                for (int i = 0; i < zoid.tag_stencil_md[0].size(); i++) {
                    tag_to_idx[zoid.tag_stencil_md[0][i]] = i;
                }

                for (int t = 0; t < NUM_TIMESTEPS_IN_PARALLEL + 1; t++) {
                    for (int i = 0; i < zoid.x_stencil_md[0].size(); i++) {
                        bool in_zoid = true;
                        double pos[3] = {zoid.x_stencil_md[0][i].x,
                                         zoid.x_stencil_md[0][i].y,
                                         zoid.x_stencil_md[0][i].z};
                        for (int dim = 0; dim < domain->dimension; dim++) {
                            double lo = zoid.zoid.cuts[dim].lower +
                                        t * zoid.zoid.cuts[dim].slope_lower;
                            double hi = zoid.zoid.cuts[dim].upper +
                                        t * zoid.zoid.cuts[dim].slope_upper;
                            in_zoid = in_zoid && pos[dim] >= lo && pos[dim] < hi;
                        }

                        if (in_zoid) {
                            zoid.local_idxs_per_timestep[t].push_back(i);
                        }
                    }

                    std::vector<int> tmp1;
                    std::vector<int> tmp2;
                    int num_segments = get_segments(zoid.local_idxs_per_timestep[t], tmp1, tmp2);
                    if (num_segments > 1) {
                        /*
                        std::stringstream s1;
                        for (auto& size: tmp2) {
                            s1 << size << " ";
                        }
                        std::cout << BOLDYELLOW << "NEXT DT dep: " << dep << " zoid: " << zoid.num << " time: " << t << " nlocal: " << zoid.local_idxs_per_timestep[t].size()
                                  << " num segments: " << num_segments << " segment sizes: " << s1.str() << RESET_COLOR << std::endl;
                        */
                    }
                }
            }
        }

        // debug next_dt
        memset(all_tags, 0, total_num_entries * sizeof(int));
        for (int t = 0; t < NUM_TIMESTEPS_IN_PARALLEL + 1; t++) {
            for (int dep = 0; dep < NUM_DEPS; dep++) {
                for (int j = 0; j < queues_many_cuts_next_dt[dep].size(); j++) {
                    auto& zoid = queues_many_cuts_next_dt[dep][j];
                    int zoid_num = zoid.num;
                    if (zoid_num % comm->nprocs != comm->me) {
                        continue;
                    }

                    auto& local_idxs = zoid.local_idxs_per_timestep[t];
                    for (int i = 0; i < local_idxs.size(); i++) {
                        int idx = local_idxs[i];
                        int tag = zoid.tag_stencil_md[0][idx];
                        all_tags[tag]++;
                    }
                }
            }

            MPI_Allreduce(MPI_IN_PLACE, all_tags, total_num_entries, MPI_INT, MPI_SUM, world);
            assert(all_tags[0] == 0);
            for (int i = 1; i < total_num_entries; i++) {
                if (all_tags[i] != 1) {
                    std::cout << "next_dt timestep: " << t << " tag: " << i << " all tags: " << all_tags[i] << std::endl;
                }
                assert(all_tags[i] == 1);
            }
            memset(all_tags, 0, total_num_entries * sizeof(int));
        }

        delete[] all_tags;

        std::cout << "GOT LOCAL ATOMS PASSED I THINK. " << std::endl;
    }

    void CREATE_NEIGHBOR_LIST_HELPER(queue_info& zoid, std::vector<int>* neighbor_lst) {
        std::unordered_map<int, int> zoid_tag_to_idx;
        for (int k = 0; k < zoid.tag_stencil_md[0].size(); k++) {
            zoid_tag_to_idx[zoid.tag_stencil_md[0][k]] = k;
        }

        auto& x = zoid.x_stencil_md[0];
        for (int t = 0; t < NUM_TIMESTEPS_IN_PARALLEL + 1; t++) {
            zoid.neighbor_list[t].resize(zoid.x_stencil_md[0].size());

            std::unordered_set<int> local_idxs_set;
            auto &local_idxs = zoid.local_idxs_per_timestep[t];
            local_idxs_set.insert(local_idxs.begin(), local_idxs.end());
            for (int i = 0; i < local_idxs.size(); i++) {
                int idx = local_idxs[i];
                zoid.neighbor_list[t][idx].reserve(20);

                int tag = zoid.tag_stencil_md[0][idx];
                std::set<int> neigh_set;
                neigh_set.insert(neighbor_lst[tag].begin(), neighbor_lst[tag].end());

                double xtmp = x[idx].x;
                double ytmp = x[idx].y;
                double ztmp = x[idx].z;

                for (auto &neigh_tag: neigh_set) {
                    int neigh_idx = zoid_tag_to_idx[neigh_tag];
                    bool neigh_nlocal = (local_idxs_set.find(neigh_idx) != local_idxs_set.end());
                    if (neigh_nlocal) {
                        if (x[neigh_idx].z < ztmp) continue;
                        if (x[neigh_idx].z == ztmp) {
                            if (x[neigh_idx].y < ytmp) continue;
                            if (x[neigh_idx].y == ytmp && x[neigh_idx].x < xtmp) continue;
                        }
                    }

                    // add an edge if the ghost atom is ghost in a shrinking dimension
                    if (!neigh_nlocal) {
                        bool shrinking_out_of_bounds = false;
                        bool expanding_out_of_bounds = false;

                        double neigh_pos[3] = {x[neigh_idx].x, x[neigh_idx].y, x[neigh_idx].z};

                        for (int dim = 0; dim < domain->dimension; dim++) {
                            double lo = zoid.zoid.cuts[dim].lower + t * zoid.zoid.cuts[dim].slope_lower;
                            double hi = zoid.zoid.cuts[dim].upper + t * zoid.zoid.cuts[dim].slope_upper;
                            bool my_dim_shrinking = (zoid.zoid.cuts[dim].slope_lower > 0);
                            bool out_of_bounds = (neigh_pos[dim] < lo || neigh_pos[dim] >= hi);
                            if (my_dim_shrinking && out_of_bounds) {
                                shrinking_out_of_bounds = true;
                            } else if (!my_dim_shrinking && out_of_bounds) {
                                expanding_out_of_bounds = true;
                            }
                        }

                        bool add_ghost_edge = shrinking_out_of_bounds;

                        if (!add_ghost_edge) {
                            continue;
                        }
                    }

                    double delx = xtmp - x[neigh_idx].x;
                    double dely = ytmp - x[neigh_idx].y;
                    double delz = ztmp - x[neigh_idx].z;
                    double rsq = delx*delx + dely*dely + delz*delz;
                    int itype = zoid.type_stencil_md[0][idx];
                    int jtype = zoid.type_stencil_md[0][neigh_idx];

                    if (rsq <= neighbor->cutneighsq[itype][jtype]) {
                        zoid.neighbor_list[t][idx].push_back(neigh_idx);
                    } else {
                        std::cout << "zoid: " << zoid.num << " failed check? "
                            << " x: " << xtmp << " " << ytmp << " " << ztmp
                            << " other x: " << x[neigh_idx].x << " " << x[neigh_idx].y << " " << x[neigh_idx].z
                            << " rsq: " << rsq << " neighbor cut: " << neighbor->cutneighsq[itype][jtype]
                            << std::endl;
                    }
                }
            }
        }
    }

    // Complete hack
    void CREATE_NEIGHBOR_LIST() {
        auto* neighbor_lst = new std::vector<int>[atom->natoms + 1];
        for (int i = 0; i < atom->natoms + 1; i++) {
            neighbor_lst[i].reserve(25);
        }

        auto* list = force->pair->list;
        for (int ii = 0; ii < list->inum; ii++) {
            int i = list->ilist[ii];
            assert(i == ii);
            int numneigh = list->numneigh[i];
            for (int j = 0; j < numneigh; j++) {
                int neigh = list->firstneigh[i][j];
                neighbor_lst[atom->tag[i]].push_back(atom->tag[neigh]);
                // TODO: This might be needed since we do not have visibility into everything
                neighbor_lst[atom->tag[neigh]].push_back(atom->tag[i]);
            }
        }

        auto* counts = new int[comm->nprocs];
        int* displacements = new int[comm->nprocs];

        for (int tag = 1; tag < atom->natoms + 1; tag++) {
            memset(counts, 0, comm->nprocs * sizeof(int));
            memset(displacements, 0, comm->nprocs * sizeof(int));

            counts[comm->me] = neighbor_lst[tag].size();
            MPI_Allreduce(MPI_IN_PLACE, counts, comm->nprocs, MPI_INT, MPI_SUM, world);
            int total_size = 0;
            for (int i = 0; i < comm->nprocs; i++) {
                total_size += counts[i];
            }

            int* my_send = new int[counts[comm->me]];
            int* all_recv = new int[total_size];

            for (int i = 0; i < counts[comm->me]; i++) {
                my_send[i] = neighbor_lst[tag][i];
            }
            displacements[0] = 0;
            for (int i = 1; i < comm->nprocs; i++) {
                displacements[i] = displacements[i - 1] + counts[i - 1];
            }

            MPI_Allgatherv(my_send, counts[comm->me], MPI_INT, all_recv,
                           counts, displacements, MPI_INT, world);

            for (int i = 0; i < total_size; i++) {
                int neigh_tag = all_recv[i];
                neighbor_lst[tag].push_back(neigh_tag);
            }

            delete[] my_send;
            delete[] all_recv;
        }

        delete[] counts;
        delete[] displacements;

        cilk_for (int dep = 0; dep < NUM_DEPS; dep++) {
            cilk_for (int j = 0; j < queues_many_cuts[dep].size(); j++) {
                auto& zoid = queues_many_cuts[dep][j];
                if (zoid.num % comm->nprocs != comm->me) {
                    continue;
                }
                CREATE_NEIGHBOR_LIST_HELPER(zoid, neighbor_lst);
            }
        }

        cilk_for (int dep = 0; dep < NUM_DEPS; dep++) {
            cilk_for (int j = 0; j < queues_many_cuts_next_dt[dep].size(); j++) {
                auto& zoid = queues_many_cuts_next_dt[dep][j];
                if (zoid.num % comm->nprocs != comm->me) {
                    continue;
                }
                CREATE_NEIGHBOR_LIST_HELPER(zoid, neighbor_lst);
            }
        }
    }

    void CREATE_BOND_LIST_HELPER(queue_info& zoid, std::vector<std::pair<int, int>>* bond_lst) {
        std::unordered_map<int, int> zoid_tag_to_idx;
        for (int k = 0; k < zoid.tag_stencil_md[0].size(); k++) {
            zoid_tag_to_idx[zoid.tag_stencil_md[0][k]] = k;
        }

        auto& x = zoid.x_stencil_md[0];
        for (int t = 0; t < NUM_TIMESTEPS_IN_PARALLEL + 1; t++) {
            zoid.bond_list[t].resize(zoid.x_stencil_md[0].size());

            std::unordered_set<int> local_idxs_set;
            auto &local_idxs = zoid.local_idxs_per_timestep[t];
            local_idxs_set.insert(local_idxs.begin(), local_idxs.end());

            for (int i = 0; i < local_idxs.size(); i++) {
                int idx = local_idxs[i];
                zoid.bond_list[t][idx].reserve(20);

                int tag = zoid.tag_stencil_md[0][idx];
                std::unordered_map<int, int> neigh_tag_to_bond_type;
                std::set<int> neigh_set;
                for (auto& [neigh_tag, bond_type] : bond_lst[tag]) {
                    neigh_set.insert(neigh_tag);
                    neigh_tag_to_bond_type[neigh_tag] = bond_type;
                }

                double xtmp = x[idx].x;
                double ytmp = x[idx].y;
                double ztmp = x[idx].z;

                for (auto &neigh_tag: neigh_set) {
                    int neigh_idx = zoid_tag_to_idx[neigh_tag];
                    int bond_type = neigh_tag_to_bond_type[neigh_tag];

                    bool neigh_nlocal = (local_idxs_set.find(neigh_idx) != local_idxs_set.end());
                    if (neigh_nlocal) {
                        if (x[neigh_idx].z < ztmp) continue;
                        if (x[neigh_idx].z == ztmp) {
                            if (x[neigh_idx].y < ytmp) continue;
                            if (x[neigh_idx].y == ytmp && x[neigh_idx].x < xtmp) continue;
                        }
                    }

                    // add an edge if the ghost atom is ghost in a shrinking dimension
                    if (!neigh_nlocal) {
                        bool shrinking_out_of_bounds = false;
                        bool expanding_out_of_bounds = false;

                        double neigh_pos[3] = {x[neigh_idx].x, x[neigh_idx].y, x[neigh_idx].z};

                        for (int dim = 0; dim < domain->dimension; dim++) {
                            double lo = zoid.zoid.cuts[dim].lower + t * zoid.zoid.cuts[dim].slope_lower;
                            double hi = zoid.zoid.cuts[dim].upper + t * zoid.zoid.cuts[dim].slope_upper;
                            bool my_dim_shrinking = (zoid.zoid.cuts[dim].slope_lower > 0);
                            bool out_of_bounds = (neigh_pos[dim] < lo || neigh_pos[dim] >= hi);
                            if (my_dim_shrinking && out_of_bounds) {
                                shrinking_out_of_bounds = true;
                            } else if (!my_dim_shrinking && out_of_bounds) {
                                expanding_out_of_bounds = true;
                            }
                        }

                        bool add_ghost_edge = shrinking_out_of_bounds;

                        if (!add_ghost_edge) {
                            continue;
                        }
                    }

                    double delx = xtmp - x[neigh_idx].x;
                    double dely = ytmp - x[neigh_idx].y;
                    double delz = ztmp - x[neigh_idx].z;
                    double rsq = delx*delx + dely*dely + delz*delz;
                    int itype = zoid.type_stencil_md[0][idx];
                    int jtype = zoid.type_stencil_md[0][neigh_idx];

                    if (rsq <= neighbor->cutneighsq[itype][jtype]) {
                        zoid.bond_list[t][idx].push_back({neigh_idx, bond_type});
                    } else {
                        std::cout << "zoid: " << zoid.num << " bond failed check? "
                                  << " x: " << xtmp << " " << ytmp << " " << ztmp
                                  << " other x: " << x[neigh_idx].x << " " << x[neigh_idx].y << " " << x[neigh_idx].z
                                  << " rsq: " << rsq << " neighbor cut: " << neighbor->cutneighsq[itype][jtype]
                                  << std::endl;
                    }
                }
            }
        }
    }

    // Complete hack
    void CREATE_BOND_LIST() {
        auto* bond_lst = new std::vector<std::pair<int, int>>[atom->natoms + 1];
        for (int i = 0; i < atom->natoms + 1; i++) {
            bond_lst[i].reserve(25);
        }

        for (int i = 0; i < neighbor->nbondlist; i++) {
            int i1 = neighbor->bondlist[i][0];
            int i2 = neighbor->bondlist[i][1];
            int type = neighbor->bondlist[i][2];
            bond_lst[atom->tag[i1]].push_back({atom->tag[i2], type});
            // TODO: This might be needed since we do not have visibility into everything
            bond_lst[atom->tag[i2]].push_back({atom->tag[i1], type});
        }

        auto* counts = new int[comm->nprocs];
        int* displacements = new int[comm->nprocs];

        for (int tag = 1; tag < atom->natoms + 1; tag++) {
            memset(counts, 0, comm->nprocs * sizeof(int));
            memset(displacements, 0, comm->nprocs * sizeof(int));

            counts[comm->me] = bond_lst[tag].size();
            MPI_Allreduce(MPI_IN_PLACE, counts, comm->nprocs, MPI_INT, MPI_SUM, world);
            int total_size = 0;
            for (int i = 0; i < comm->nprocs; i++) {
                total_size += counts[i];
            }

            int* my_send = new int[counts[comm->me]];
            int* my_send_type = new int[counts[comm->me]];

            int* all_recv = new int[total_size];
            int* all_recv_type = new int[total_size];

            for (int i = 0; i < counts[comm->me]; i++) {
                my_send[i] = bond_lst[tag][i].first;
                my_send_type[i] = bond_lst[tag][i].second;
            }

            displacements[0] = 0;
            for (int i = 1; i < comm->nprocs; i++) {
                displacements[i] = displacements[i - 1] + counts[i - 1];
            }

            MPI_Allgatherv(my_send, counts[comm->me], MPI_INT, all_recv,
                           counts, displacements, MPI_INT, world);
            MPI_Allgatherv(my_send_type, counts[comm->me], MPI_INT, all_recv_type,
                           counts, displacements, MPI_INT, world);

            for (int i = 0; i < total_size; i++) {
                int neigh_tag = all_recv[i];
                int neigh_type = all_recv_type[i];
                bond_lst[tag].push_back({neigh_tag, neigh_type});
            }

            delete[] my_send;
            delete[] all_recv;

            delete[] my_send_type;
            delete[] all_recv_type;
        }

        delete[] counts;
        delete[] displacements;

        cilk_for (int dep = 0; dep < NUM_DEPS; dep++) {
            cilk_for (int j = 0; j < queues_many_cuts[dep].size(); j++) {
                auto& zoid = queues_many_cuts[dep][j];
                if (zoid.num % comm->nprocs != comm->me) {
                    continue;
                }

                CREATE_BOND_LIST_HELPER(zoid, bond_lst);
            }
        }

        cilk_for (int dep = 0; dep < NUM_DEPS; dep++) {
            cilk_for (int j = 0; j < queues_many_cuts_next_dt[dep].size(); j++) {
                auto& zoid = queues_many_cuts_next_dt[dep][j];
                if (zoid.num % comm->nprocs != comm->me) {
                    continue;
                }

                CREATE_BOND_LIST_HELPER(zoid, bond_lst);
            }
        }
    }

    void INIT_AFFINITY_AND_LOCKS() {
        for (int dep = 0; dep < NUM_DEPS; dep++) {
            for (int j = 0; j < queues_many_cuts[dep].size(); j++) {
                queue_info& zoid = queues_many_cuts[dep][j];
                int zoid_num = zoid.num;
                if (zoid_num % comm->nprocs == comm->me) {
                    int size = zoid.x_stencil_md[0].size();
                    int num_chunks = size / MODIFY_GRAINSIZE + 1;
                    int chunk_size = MODIFY_GRAINSIZE;

                    zoid.claimed_flags_stencil_md[0] = new std::atomic_flag[num_chunks];

                    for (int i = 0; i < num_chunks; i++) {
                        zoid.claimed_flags_stencil_md[0]->clear();
                    }

                    zoid.spinlocks_stencil_md[0] = new spinlock[size];
                }
            }
        }
    }

    template <bool curr_dt>
    void CONSTRUCT_SEND_FORCE_IDXS_ZOID_MANY_CUTS_HELPER(queue_info& zoid) {
        int zoid_num = zoid.num;
        auto& send_to_neighbors = curr_dt ? send_to_neighbors_many_cuts[zoid_num] : send_to_neighbors_many_cuts_next_dt[zoid_num];

        std::map<int, int> idx_to_zoid;

        for (int t = 0; t < NUM_TIMESTEPS_IN_PARALLEL + 1; t++) {
            zoid.send_force_idxs_double_buffering[t] = new std::vector<int>[send_to_neighbors.size()];

            std::set<int> send_idxs;
            auto& local_idxs = zoid.local_idxs_per_timestep[t];
            std::set<int> local_idxs_set;
            local_idxs_set.insert(local_idxs.begin(), local_idxs.end());

            for (int i = 0; i < zoid.x_stencil_md[0].size(); i++) {
                // only send ghost idxs for forces
                bool is_local = (local_idxs_set.find(i) != local_idxs_set.end());
                if (is_local) {
                    continue;
                }

                auto& pos = zoid.x_stencil_md[0][i];
                double atom_pos[3] = {pos.x, pos.y, pos.z};
                double zoid_lo[3] = {0};
                double zoid_hi[3] = {0};

                bool dim_out_of_bounds[3] = {0};
                int out_of_bounds[3] = {0};

                bool borders_zoid = true;
                for (int dim = 0; dim < domain->dimension; dim++) {
                    double lo = zoid.zoid.cuts[dim].lower + t * zoid.zoid.cuts[dim].slope_lower;
                    double hi = zoid.zoid.cuts[dim].upper + t * zoid.zoid.cuts[dim].slope_upper;
                    double lo_borders = lo - ALLEGRO_SLOPE;
                    double hi_borders = hi + ALLEGRO_SLOPE;
                    borders_zoid = borders_zoid && atom_pos[dim] >= lo_borders && atom_pos[dim] < hi;

                    zoid_lo[dim] = lo;
                    zoid_hi[dim] = hi;

                    if (atom_pos[dim] <= lo || atom_pos[dim] > hi) {
                        dim_out_of_bounds[dim] = true;
                        if (atom_pos[dim] <= lo) {
                            out_of_bounds[dim] = -1;
                        } else if (atom_pos[dim] > hi) {
                            out_of_bounds[dim] = 1;
                        } else {
                            assert(false);
                        }
                    }
                }

                // have to do this check as for later timesteps this might not be the case
                if (!borders_zoid) {
                    continue;
                }

                // TODO: pick first neighbor that allows us to map a path to get the force where it eventually belongs
                for (int j = 0; j < send_to_neighbors.size(); j++) {
                    auto send_zoid_num = send_to_neighbors[j];
                    auto& send_zoid = curr_dt ? zoid_num_to_zoid_many_cuts[send_zoid_num] : zoid_num_to_zoid_many_cuts_next_dt[send_zoid_num];

                    bool found_zoid = false;

                    for (int dim = 0; dim < domain->dimension; dim++) {
                        if (!dim_out_of_bounds[dim]) {
                            continue;
                        }

                        int out_of_bounds_factor = out_of_bounds[dim];
                        int where_diff = zoid.where[dim] - send_zoid.where[dim];

                        if (out_of_bounds_factor == -1 && (where_diff == 1 || where_diff == - (NUM_ZOIDS_PER_DIMENSION - 1))) {
                            idx_to_zoid[i] = send_zoid_num;
                            found_zoid = true;
                        } else if (out_of_bounds_factor == 1 && (where_diff == -1 || where_diff == NUM_ZOIDS_PER_DIMENSION - 1)) {
                            idx_to_zoid[i] = send_zoid_num;
                            found_zoid = true;
                        }
                    }

                    if (found_zoid) {
                        zoid.send_force_idxs_double_buffering[t][j].push_back(i);
                        break;
                    }
                }
            }
        }
    }

    template <bool curr_dt>
    void CONSTRUCT_SEND_FORCE_IDXS_ZOID_MANY_CUTS() {
        auto& queues = curr_dt ? queues_many_cuts : queues_many_cuts_next_dt;

        cilk_for (int dep = 0; dep < NUM_DEPS; dep++) {
            cilk_for (int j = 0; j < queues[dep].size(); j++) {
                auto& zoid = queues[dep][j];
                if (zoid.num % comm->nprocs != comm->me) {
                    continue;
                }

                CONSTRUCT_SEND_FORCE_IDXS_ZOID_MANY_CUTS_HELPER<curr_dt>(zoid);
            }
        }
    }

    template <bool curr_dt>
    void CONSTRUCT_SEND_VEL_IDXS_ZOID_MANY_CUTS_HELPER(queue_info& zoid) {
        int zoid_num = zoid.num;
        auto& send_to_neighbors = curr_dt ? send_to_neighbors_many_cuts[zoid_num] : send_to_neighbors_many_cuts_next_dt[zoid_num];

        std::map<int, int> idx_to_zoid;

        zoid.send_vel_idxs_double_buffering[0] = new std::vector<int>[send_to_neighbors.size()];

        for (int t = 1; t < NUM_TIMESTEPS_IN_PARALLEL + 1; t++) {
            zoid.send_vel_idxs_double_buffering[t] = new std::vector<int>[send_to_neighbors.size()];

            auto& local_idxs = zoid.local_idxs_per_timestep[t];
            std::set<int> local_idxs_set;
            local_idxs_set.insert(local_idxs.begin(), local_idxs.end());

            auto& local_idxs_prev = zoid.local_idxs_per_timestep[t - 1];
            std::set<int> local_idxs_prev_set;
            local_idxs_prev_set.insert(local_idxs_prev.begin(), local_idxs_prev.end());

            for (int i = 0; i < zoid.x_stencil_md[0].size(); i++) {
                // only send ghost idxs for forces
                bool is_local = (local_idxs_set.find(i) != local_idxs_set.end());
                bool is_local_prev = (local_idxs_prev_set.find(i) != local_idxs_prev_set.end());

                /*
                if (!(!is_local && is_local_prev)) {
                    continue;
                }
                */

                // want ghost atoms that were local previously to send to other zoids
                if (is_local || !is_local_prev) {
                    continue;
                }

                auto& pos = zoid.x_stencil_md[0][i];
                double atom_pos[3] = {pos.x, pos.y, pos.z};
                double zoid_lo[3] = {0};
                double zoid_hi[3] = {0};

                bool dim_out_of_bounds[3] = {0};
                int out_of_bounds[3] = {0};

                bool borders_zoid = true;
                for (int dim = 0; dim < domain->dimension; dim++) {
                    double lo = zoid.zoid.cuts[dim].lower + t * zoid.zoid.cuts[dim].slope_lower;
                    double hi = zoid.zoid.cuts[dim].upper + t * zoid.zoid.cuts[dim].slope_upper;
                    double lo_borders = lo - ALLEGRO_SLOPE;
                    double hi_borders = hi + ALLEGRO_SLOPE;
                    borders_zoid = borders_zoid && atom_pos[dim] >= lo_borders && atom_pos[dim] < hi;

                    zoid_lo[dim] = lo;
                    zoid_hi[dim] = hi;

                    if (atom_pos[dim] <= lo || atom_pos[dim] > hi) {
                        dim_out_of_bounds[dim] = true;
                        if (atom_pos[dim] <= lo) {
                            out_of_bounds[dim] = -1;
                        } else if (atom_pos[dim] > hi) {
                            out_of_bounds[dim] = 1;
                        } else {
                            assert(false);
                        }
                    }
                }

                // have to do this check as for later timesteps this might not be the case
                if (!borders_zoid) {
                    continue;
                }

                // TODO: pick first neighbor that allows us to map a path to get the force where it eventually belongs
                for (int j = 0; j < send_to_neighbors.size(); j++) {
                    auto send_zoid_num = send_to_neighbors[j];
                    auto& send_zoid = curr_dt ? zoid_num_to_zoid_many_cuts[send_zoid_num] : zoid_num_to_zoid_many_cuts_next_dt[send_zoid_num];

                    bool found_zoid = false;

                    for (int dim = 0; dim < domain->dimension; dim++) {
                        if (!dim_out_of_bounds[dim]) {
                            continue;
                        }

                        int out_of_bounds_factor = out_of_bounds[dim];
                        int where_diff = zoid.where[dim] - send_zoid.where[dim];

                        if (out_of_bounds_factor == -1 && (where_diff == 1 || where_diff == - (NUM_ZOIDS_PER_DIMENSION - 1))) {
                            idx_to_zoid[i] = send_zoid_num;
                            found_zoid = true;
                        } else if (out_of_bounds_factor == 1 && (where_diff == -1 || where_diff == NUM_ZOIDS_PER_DIMENSION - 1)) {
                            idx_to_zoid[i] = send_zoid_num;
                            found_zoid = true;
                        }
                    }

                    if (found_zoid) {
                        zoid.send_vel_idxs_double_buffering[t][j].push_back(i);
                        break;
                    }
                }
            }
        }
    }

    template <bool curr_dt>
    void CONSTRUCT_SEND_VEL_IDXS_ZOID_MANY_CUTS() {
        auto& queues = curr_dt ? queues_many_cuts : queues_many_cuts_next_dt;
        cilk_for (int dep = 0; dep < NUM_DEPS; dep++) {
            cilk_for (int j = 0; j < queues[dep].size(); j++) {
                auto& zoid = queues[dep][j];
                if (zoid.num % comm->nprocs != comm->me) {
                    continue;
                }

                CONSTRUCT_SEND_VEL_IDXS_ZOID_MANY_CUTS_HELPER<curr_dt>(zoid);
            }
        }
    }

    template <bool curr_dt>
    void CONSTRUCT_SEND_POS_IDXS_ZOID_MANY_CUTS_HELPER(queue_info& zoid) {
        int zoid_num = zoid.num;
        auto& send_to_neighbors = curr_dt ? send_to_neighbors_many_cuts[zoid_num] : send_to_neighbors_many_cuts_next_dt[zoid_num];

        std::map<int, int> idx_to_zoid;

        for (int t = 0; t < NUM_TIMESTEPS_IN_PARALLEL + 1; t++) {
            zoid.send_pos_idxs_double_buffering[t] = new std::vector<int>[send_to_neighbors.size()];

            auto& local_idxs = zoid.local_idxs_per_timestep[t];
            std::set<int> local_idxs_set;
            local_idxs_set.insert(local_idxs.begin(), local_idxs.end());

            std::set<int> local_idxs_prev_set;
            if (t > 0) {
                std::vector<int> local_idxs_prev = zoid.local_idxs_per_timestep[t - 1];
                local_idxs_prev_set.insert(local_idxs_prev.begin(), local_idxs_prev.end());
            }

            for (int i = 0; i < zoid.x_stencil_md[0].size(); i++) {
                // only send ghost idxs for forces
                bool is_local = (local_idxs_set.find(i) != local_idxs_set.end());
                bool is_local_prev = (local_idxs_prev_set.find(i) != local_idxs_prev_set.end());

                if (!(is_local || is_local_prev)) {
                    continue;
                }

                auto& pos = zoid.x_stencil_md[0][i];
                double atom_pos[3] = {pos.x, pos.y, pos.z};
                double zoid_lo[3] = {0};
                double zoid_hi[3] = {0};

                bool dim_out_of_bounds[3] = {0};
                int out_of_bounds[3] = {0};

                bool borders_zoid = true;
                for (int dim = 0; dim < domain->dimension; dim++) {
                    double lo = zoid.zoid.cuts[dim].lower + t * zoid.zoid.cuts[dim].slope_lower;
                    double hi = zoid.zoid.cuts[dim].upper + t * zoid.zoid.cuts[dim].slope_upper;
                    double lo_borders = lo - ALLEGRO_SLOPE;
                    double hi_borders = hi + ALLEGRO_SLOPE;
                    borders_zoid = borders_zoid && atom_pos[dim] >= lo_borders && atom_pos[dim] < hi;

                    zoid_lo[dim] = lo;
                    zoid_hi[dim] = hi;

                    if (atom_pos[dim] <= lo || atom_pos[dim] > hi) {
                        dim_out_of_bounds[dim] = true;
                        if (atom_pos[dim] <= lo) {
                            out_of_bounds[dim] = -1;
                        } else if (atom_pos[dim] > hi) {
                            out_of_bounds[dim] = 1;
                        } else {
                            assert(false);
                        }
                    }
                }

                // have to do this check as for later timesteps this might not be the case
                if (!borders_zoid) {
                    continue;
                }

                // TODO: pick first neighbor that allows us to map a path to get the force where it eventually belongs
                for (int j = 0; j < send_to_neighbors.size(); j++) {
                    auto send_zoid_num = send_to_neighbors[j];
                    auto& send_zoid = curr_dt ? zoid_num_to_zoid_many_cuts[send_zoid_num] :
                            zoid_num_to_zoid_many_cuts_next_dt[send_zoid_num];

                    bool found_zoid = false;

                    for (int dim = 0; dim < domain->dimension; dim++) {
                        if (!dim_out_of_bounds[dim]) {
                            continue;
                        }

                        int out_of_bounds_factor = out_of_bounds[dim];
                        int where_diff = zoid.where[dim] - send_zoid.where[dim];

                        if (out_of_bounds_factor == -1 && (where_diff == 1 || where_diff == - (NUM_ZOIDS_PER_DIMENSION - 1))) {
                            idx_to_zoid[i] = send_zoid_num;
                            found_zoid = true;
                        } else if (out_of_bounds_factor == 1 && (where_diff == -1 || where_diff == NUM_ZOIDS_PER_DIMENSION - 1)) {
                            idx_to_zoid[i] = send_zoid_num;
                            found_zoid = true;
                        }
                    }

                    if (found_zoid) {
                        zoid.send_pos_idxs_double_buffering[t][j].push_back(i);
                        break;
                    }
                }
            }
        }
    }

    template <bool curr_dt>
    void CONSTRUCT_SEND_POS_IDXS_ZOID_MANY_CUTS() {
        auto& queues = curr_dt ? queues_many_cuts : queues_many_cuts_next_dt;

        cilk_for (int dep = 0; dep < NUM_DEPS; dep++) {
            cilk_for (int j = 0; j < queues[dep].size(); j++) {
                auto& zoid = queues[dep][j];
                if (zoid.num % comm->nprocs != comm->me) {
                    continue;
                }

                CONSTRUCT_SEND_POS_IDXS_ZOID_MANY_CUTS_HELPER<curr_dt>(zoid);
            }
        }
    }

    template <bool curr_dt>
    void CONSTRUCT_RECV_POS_IDXS_ZOID_MANY_CUTS() {
        std::vector<MPI_Request> r;
        r.reserve(NUM_ZOIDS_MANY_CUTS * (NUM_TIMESTEPS_IN_PARALLEL + 1) * 4 / comm->nprocs);

        auto& queues = curr_dt ? queues_many_cuts : queues_many_cuts_next_dt;

        std::vector<int>** zoid_send_data[NUM_ZOIDS_MANY_CUTS];
        int** zoid_send_data_sizes[NUM_ZOIDS_MANY_CUTS];

        for (int dep = 0; dep < NUM_DEPS; dep++) {
            for (int j = 0; j < queues[dep].size(); j++) {
                auto& zoid = queues[dep][j];
                if (zoid.num % comm->nprocs != comm->me) {
                    continue;
                }

                zoid_send_data[zoid.num] = new std::vector<int>*[NUM_TIMESTEPS_IN_PARALLEL + 1];
                zoid_send_data_sizes[zoid.num] = new int*[NUM_TIMESTEPS_IN_PARALLEL + 1];

                auto& send_to_neighbors = curr_dt ? send_to_neighbors_many_cuts[zoid.num] :
                        send_to_neighbors_many_cuts_next_dt[zoid.num];

                for (int t = 0; t < NUM_TIMESTEPS_IN_PARALLEL + 1; t++) {
                    zoid_send_data[zoid.num][t] = new std::vector<int>[send_to_neighbors.size()];
                    zoid_send_data_sizes[zoid.num][t] = new int[send_to_neighbors.size()];

                    for (int i = 0; i < send_to_neighbors.size(); i++) {
                        int send_zoid_num = send_to_neighbors[i];

                        auto& send_pos_idxs = zoid.send_pos_idxs_double_buffering[t][i];

                        int mpi_tag = get_mpi_tag(send_zoid_num, zoid.num, t, t);

                        zoid_send_data[zoid.num][t][i].reserve(send_pos_idxs.size());
                        zoid_send_data_sizes[zoid.num][t][i] = send_pos_idxs.size();
                        for (int k = 0; k < send_pos_idxs.size(); k++) {
                            zoid_send_data[zoid.num][t][i].push_back(zoid.tag_stencil_md[0][send_pos_idxs[k]]);
                        }

                        r.emplace_back();
                        MPI_Isend(&zoid_send_data_sizes[zoid.num][t][i], 1, MPI_INT, send_zoid_num % comm->nprocs, mpi_tag, world, &r[r.size() - 1]);

                        if (send_pos_idxs.size() > 0) {
                            r.emplace_back();
                            // std::vector<int> send_pos_tags;
                            // send_pos_tags.reserve(size);
                            MPI_Isend(zoid_send_data[zoid.num][t][i].data(), send_pos_idxs.size(), MPI_INT, send_zoid_num % comm->nprocs, mpi_tag, world, &r[r.size() - 1]);
                        }
                    }
                }
            }
        }

        for (int dep = 0; dep < NUM_DEPS; dep++) {
            for (int j = 0; j < queues[dep].size(); j++) {
                auto& zoid = queues[dep][j];
                if (zoid.num % comm->nprocs != comm->me) {
                    continue;
                }

                std::unordered_map<int, int> tag_to_idx;
                for (int i = 0; i < zoid.tag_stencil_md[0].size(); i++) {
                    tag_to_idx[zoid.tag_stencil_md[0][i]] = i;
                }

                auto& recv_from_neighbors = curr_dt ? recv_from_neighbors_many_cuts[zoid.num]
                        : recv_from_neighbors_many_cuts_next_dt[zoid.num];

                for (int t = 0; t < NUM_TIMESTEPS_IN_PARALLEL + 1; t++) {
                    zoid.recv_pos_idxs_double_buffering[t] = new std::vector<int>[recv_from_neighbors.size()];

                    for (int i = 0; i < recv_from_neighbors.size(); i++) {
                        int recv_zoid_num = recv_from_neighbors[i];

                        int mpi_tag = get_mpi_tag(zoid.num, recv_zoid_num, t, t);

                        int nrecv;
                        MPI_Recv(&nrecv, 1, MPI_INT, recv_zoid_num % comm->nprocs,
                                 mpi_tag, world, MPI_STATUS_IGNORE);

                        if (nrecv) {
                            int* recv_buf = new int[nrecv];
                            MPI_Recv(recv_buf, nrecv, MPI_INT, recv_zoid_num % comm->nprocs,
                                     mpi_tag, world, MPI_STATUS_IGNORE);
                            zoid.recv_pos_idxs_double_buffering[t][i].reserve(nrecv);
                            for (int k = 0; k < nrecv; k++) {
                                zoid.recv_pos_idxs_double_buffering[t][i].push_back(tag_to_idx.at(recv_buf[k]));
                            }
                            delete[] recv_buf;
                        }
                    }
                }
            }
        }

        MPI_Waitall(r.size(), r.data(), MPI_STATUS_IGNORE);

        for (int dep = 0; dep < NUM_DEPS; dep++) {
            for (int j = 0; j < queues[dep].size(); j++) {
                auto &zoid = queues[dep][j];
                if (zoid.num % comm->nprocs != comm->me) {
                    continue;
                }

                for (int t = 0; t < NUM_TIMESTEPS_IN_PARALLEL + 1; t++) {
                    delete[] zoid_send_data[zoid.num][t];
                    delete[] zoid_send_data_sizes[zoid.num][t];
                }

                delete[] zoid_send_data[zoid.num];
                delete[] zoid_send_data_sizes[zoid.num];
            }
        }
    }

    template <bool curr_dt>
    void CONSTRUCT_RECV_FORCE_IDXS_ZOID_MANY_CUTS() {
        std::vector<MPI_Request> r;
        r.reserve(NUM_ZOIDS_MANY_CUTS * (NUM_TIMESTEPS_IN_PARALLEL + 1) * 4 / comm->nprocs);

        auto& queues = curr_dt ? queues_many_cuts : queues_many_cuts_next_dt;

        std::vector<int>** zoid_send_data[NUM_ZOIDS_MANY_CUTS];
        int** zoid_send_data_sizes[NUM_ZOIDS_MANY_CUTS];

        for (int dep = 0; dep < NUM_DEPS; dep++) {
            for (int j = 0; j < queues[dep].size(); j++) {
                auto& zoid = queues[dep][j];
                if (zoid.num % comm->nprocs != comm->me) {
                    continue;
                }

                zoid_send_data[zoid.num] = new std::vector<int>*[NUM_TIMESTEPS_IN_PARALLEL + 1];
                zoid_send_data_sizes[zoid.num] = new int*[NUM_TIMESTEPS_IN_PARALLEL + 1];

                auto& send_to_neighbors = curr_dt ? send_to_neighbors_many_cuts[zoid.num] :
                                          send_to_neighbors_many_cuts_next_dt[zoid.num];

                for (int t = 0; t < NUM_TIMESTEPS_IN_PARALLEL + 1; t++) {
                    zoid_send_data[zoid.num][t] = new std::vector<int>[send_to_neighbors.size()];
                    zoid_send_data_sizes[zoid.num][t] = new int[send_to_neighbors.size()];

                    for (int i = 0; i < send_to_neighbors.size(); i++) {
                        int send_zoid_num = send_to_neighbors[i];

                        auto& send_force_idxs = zoid.send_force_idxs_double_buffering[t][i];

                        int mpi_tag = get_mpi_tag(send_zoid_num, zoid.num, t, t);

                        zoid_send_data[zoid.num][t][i].reserve(send_force_idxs.size());
                        zoid_send_data_sizes[zoid.num][t][i] = send_force_idxs.size();
                        for (int k = 0; k < send_force_idxs.size(); k++) {
                            zoid_send_data[zoid.num][t][i].push_back(zoid.tag_stencil_md[0][send_force_idxs[k]]);
                        }

                        r.emplace_back();
                        MPI_Isend(&zoid_send_data_sizes[zoid.num][t][i], 1, MPI_INT, send_zoid_num % comm->nprocs, mpi_tag, world, &r[r.size() - 1]);

                        if (send_force_idxs.size() > 0) {
                            r.emplace_back();
                            // std::vector<int> send_pos_tags;
                            // send_pos_tags.reserve(size);
                            MPI_Isend(zoid_send_data[zoid.num][t][i].data(), send_force_idxs.size(), MPI_INT, send_zoid_num % comm->nprocs, mpi_tag, world, &r[r.size() - 1]);
                        }
                    }
                }
            }
        }

        for (int dep = 0; dep < NUM_DEPS; dep++) {
            for (int j = 0; j < queues[dep].size(); j++) {
                auto& zoid = queues[dep][j];
                if (zoid.num % comm->nprocs != comm->me) {
                    continue;
                }

                std::unordered_map<int, int> tag_to_idx;
                for (int i = 0; i < zoid.tag_stencil_md[0].size(); i++) {
                    tag_to_idx[zoid.tag_stencil_md[0][i]] = i;
                }

                auto& recv_from_neighbors = curr_dt ? recv_from_neighbors_many_cuts[zoid.num]
                                                    : recv_from_neighbors_many_cuts_next_dt[zoid.num];

                for (int t = 0; t < NUM_TIMESTEPS_IN_PARALLEL + 1; t++) {
                    zoid.recv_force_idxs_double_buffering[t] = new std::vector<int>[recv_from_neighbors.size()];

                    for (int i = 0; i < recv_from_neighbors.size(); i++) {
                        int recv_zoid_num = recv_from_neighbors[i];

                        int mpi_tag = get_mpi_tag(zoid.num, recv_zoid_num, t, t);

                        int nrecv;
                        MPI_Recv(&nrecv, 1, MPI_INT, recv_zoid_num % comm->nprocs,
                                 mpi_tag, world, MPI_STATUS_IGNORE);

                        if (nrecv) {
                            int* recv_buf = new int[nrecv];
                            MPI_Recv(recv_buf, nrecv, MPI_INT, recv_zoid_num % comm->nprocs,
                                     mpi_tag, world, MPI_STATUS_IGNORE);
                            zoid.recv_force_idxs_double_buffering[t][i].reserve(nrecv);
                            for (int k = 0; k < nrecv; k++) {
                                zoid.recv_force_idxs_double_buffering[t][i].push_back(tag_to_idx.at(recv_buf[k]));
                            }
                            delete[] recv_buf;
                        }
                    }
                }
            }
        }

        MPI_Waitall(r.size(), r.data(), MPI_STATUS_IGNORE);

        for (int dep = 0; dep < NUM_DEPS; dep++) {
            for (int j = 0; j < queues[dep].size(); j++) {
                auto &zoid = queues[dep][j];
                if (zoid.num % comm->nprocs != comm->me) {
                    continue;
                }

                for (int t = 0; t < NUM_TIMESTEPS_IN_PARALLEL + 1; t++) {
                    delete[] zoid_send_data[zoid.num][t];
                    delete[] zoid_send_data_sizes[zoid.num][t];
                }

                delete[] zoid_send_data[zoid.num];
                delete[] zoid_send_data_sizes[zoid.num];
            }
        }

        // debug
        for (int dep = 0; dep < NUM_DEPS; dep++) {
            for (int j = 0; j < queues[dep].size(); j++) {
                auto &zoid = queues[dep][j];
                if (zoid.num % comm->nprocs != comm->me) {
                    continue;
                }

                std::set<int> all_idxs;
                std::map<int, std::pair<int, int>> idx_to_zoid_timestep;

                auto& recv_from_neighbors = curr_dt ? recv_from_neighbors_many_cuts[zoid.num]
                                                    : recv_from_neighbors_many_cuts_next_dt[zoid.num];

                for (int t = 0; t < NUM_TIMESTEPS_IN_PARALLEL + 1; t++) {
                    for (int i = 0; i < recv_from_neighbors.size(); i++) {
                        auto& recv_idxs = zoid.recv_force_idxs_double_buffering[t][i];
                        for (int k = 0; k < recv_idxs.size(); k++) {
                            int idx = recv_idxs[k];
                            if (all_idxs.find(idx) != all_idxs.end()) {
                                auto& [map_zoid, map_timestep] = idx_to_zoid_timestep.at(idx);
                                if (map_timestep != t) {
                                    std::cout << "RECV FORCE zoid: " << zoid.num << " timestep: " << t << " IDX: " << idx << " curr recv from: " << recv_from_neighbors[i] << " timestep: " << t
                                              << " already there: " << idx_to_zoid_timestep[idx].first << " timestep: " << idx_to_zoid_timestep[idx].second
                                              << " tag: " << zoid.tag_stencil_md[0][idx]
                                              << std::endl;
                                    assert(false);
                                }
                            }
                            all_idxs.insert(idx);
                            idx_to_zoid_timestep[idx] = {recv_from_neighbors[i], t};
                        }
                    }
                }
            }
        }
    }

    template <bool curr_dt>
    void CONSTRUCT_RECV_VEL_IDXS_ZOID_MANY_CUTS() {
        std::vector<MPI_Request> r;
        r.reserve(NUM_ZOIDS_MANY_CUTS * (NUM_TIMESTEPS_IN_PARALLEL + 1) * 4 / comm->nprocs);

        auto& queues = curr_dt ? queues_many_cuts : queues_many_cuts_next_dt;

        std::vector<int>** zoid_send_data[NUM_ZOIDS_MANY_CUTS];
        int** zoid_send_data_sizes[NUM_ZOIDS_MANY_CUTS];

        for (int dep = 0; dep < NUM_DEPS; dep++) {
            for (int j = 0; j < queues[dep].size(); j++) {
                auto& zoid = queues[dep][j];
                if (zoid.num % comm->nprocs != comm->me) {
                    continue;
                }

                zoid_send_data[zoid.num] = new std::vector<int>*[NUM_TIMESTEPS_IN_PARALLEL + 1];
                zoid_send_data_sizes[zoid.num] = new int*[NUM_TIMESTEPS_IN_PARALLEL + 1];

                auto& send_to_neighbors = curr_dt ? send_to_neighbors_many_cuts[zoid.num] :
                                          send_to_neighbors_many_cuts_next_dt[zoid.num];

                for (int t = 0; t < NUM_TIMESTEPS_IN_PARALLEL + 1; t++) {
                    zoid_send_data[zoid.num][t] = new std::vector<int>[send_to_neighbors.size()];
                    zoid_send_data_sizes[zoid.num][t] = new int[send_to_neighbors.size()];

                    for (int i = 0; i < send_to_neighbors.size(); i++) {
                        int send_zoid_num = send_to_neighbors[i];

                        auto& send_vel_idxs = zoid.send_vel_idxs_double_buffering[t][i];

                        int mpi_tag = get_mpi_tag(send_zoid_num, zoid.num, t, t);

                        zoid_send_data[zoid.num][t][i].reserve(send_vel_idxs.size());
                        zoid_send_data_sizes[zoid.num][t][i] = send_vel_idxs.size();
                        for (int k = 0; k < send_vel_idxs.size(); k++) {
                            zoid_send_data[zoid.num][t][i].push_back(zoid.tag_stencil_md[0][send_vel_idxs[k]]);
                        }

                        r.emplace_back();
                        MPI_Isend(&zoid_send_data_sizes[zoid.num][t][i], 1, MPI_INT, send_zoid_num % comm->nprocs, mpi_tag, world, &r[r.size() - 1]);

                        if (send_vel_idxs.size() > 0) {
                            r.emplace_back();
                            // std::vector<int> send_pos_tags;
                            // send_pos_tags.reserve(size);
                            MPI_Isend(zoid_send_data[zoid.num][t][i].data(), send_vel_idxs.size(), MPI_INT, send_zoid_num % comm->nprocs, mpi_tag, world, &r[r.size() - 1]);
                        }
                    }
                }
            }
        }

        for (int dep = 0; dep < NUM_DEPS; dep++) {
            for (int j = 0; j < queues[dep].size(); j++) {
                auto& zoid = queues[dep][j];
                if (zoid.num % comm->nprocs != comm->me) {
                    continue;
                }

                std::unordered_map<int, int> tag_to_idx;
                for (int i = 0; i < zoid.tag_stencil_md[0].size(); i++) {
                    tag_to_idx[zoid.tag_stencil_md[0][i]] = i;
                }

                auto& recv_from_neighbors = curr_dt ? recv_from_neighbors_many_cuts[zoid.num]
                                                    : recv_from_neighbors_many_cuts_next_dt[zoid.num];

                for (int t = 0; t < NUM_TIMESTEPS_IN_PARALLEL + 1; t++) {
                    zoid.recv_vel_idxs_double_buffering[t] = new std::vector<int>[recv_from_neighbors.size()];

                    for (int i = 0; i < recv_from_neighbors.size(); i++) {
                        int recv_zoid_num = recv_from_neighbors[i];

                        int mpi_tag = get_mpi_tag(zoid.num, recv_zoid_num, t, t);

                        int nrecv;
                        MPI_Recv(&nrecv, 1, MPI_INT, recv_zoid_num % comm->nprocs,
                                 mpi_tag, world, MPI_STATUS_IGNORE);

                        if (nrecv) {
                            int* recv_buf = new int[nrecv];
                            MPI_Recv(recv_buf, nrecv, MPI_INT, recv_zoid_num % comm->nprocs,
                                     mpi_tag, world, MPI_STATUS_IGNORE);
                            zoid.recv_vel_idxs_double_buffering[t][i].reserve(nrecv);
                            for (int k = 0; k < nrecv; k++) {
                                zoid.recv_vel_idxs_double_buffering[t][i].push_back(tag_to_idx.at(recv_buf[k]));
                            }
                            delete[] recv_buf;
                        }
                    }
                }
            }
        }

        MPI_Waitall(r.size(), r.data(), MPI_STATUS_IGNORE);

        for (int dep = 0; dep < NUM_DEPS; dep++) {
            for (int j = 0; j < queues[dep].size(); j++) {
                auto &zoid = queues[dep][j];
                if (zoid.num % comm->nprocs != comm->me) {
                    continue;
                }

                for (int t = 0; t < NUM_TIMESTEPS_IN_PARALLEL + 1; t++) {
                    delete[] zoid_send_data[zoid.num][t];
                    delete[] zoid_send_data_sizes[zoid.num][t];
                }

                delete[] zoid_send_data[zoid.num];
                delete[] zoid_send_data_sizes[zoid.num];
            }
        }

        // debug
        for (int dep = 0; dep < NUM_DEPS; dep++) {
            for (int j = 0; j < queues[dep].size(); j++) {
                auto &zoid = queues[dep][j];
                if (zoid.num % comm->nprocs != comm->me) {
                    continue;
                }

                std::set<int> all_idxs;
                std::map<int, std::pair<int, int>> idx_to_zoid_timestep;

                auto& recv_from_neighbors = curr_dt ? recv_from_neighbors_many_cuts[zoid.num]
                                                    : recv_from_neighbors_many_cuts_next_dt[zoid.num];

                for (int t = 0; t < NUM_TIMESTEPS_IN_PARALLEL + 1; t++) {
                    for (int i = 0; i < recv_from_neighbors.size(); i++) {
                        auto& recv_idxs = zoid.recv_vel_idxs_double_buffering[t][i];
                        for (int k = 0; k < recv_idxs.size(); k++) {
                            int idx = recv_idxs[k];
                            if (all_idxs.find(idx) != all_idxs.end()) {
                                std::cout << "RECV VEL: " << zoid.num << " IDX: " << idx << " curr recv from: " << recv_from_neighbors[i] << " timestep: " << t
                                << " already there: " << idx_to_zoid_timestep[idx].first << " timestep: " << idx_to_zoid_timestep[idx].second
                                << " tag: " << zoid.tag_stencil_md[0][idx]
                                << std::endl;
                            }
                            assert(all_idxs.find(idx) == all_idxs.end());
                            all_idxs.insert(idx);
                            idx_to_zoid_timestep[idx] = {recv_from_neighbors[i], t};
                        }
                    }
                }
            }
        }
    }

    template <bool curr_dt>
    void CONSTRUCT_RECV_PROC_SIZES() {
        auto& queues = curr_dt ? queues_many_cuts : queues_many_cuts_next_dt;

        for (int dep = 1; dep < NUM_DEPS; dep++) {
            recv_proc_sizes[dep] = new std::vector<int>[comm->nprocs];
            // recv from proc
            for (int proc = 0; proc < comm->nprocs; proc++) {
                // look at previous dependency level
                int prev_dep = dep - 1;
                for (int j = 0; j < queues[prev_dep].size(); j++) {
                    auto& send_zoid = queues[prev_dep][j];
                    if (send_zoid.num % comm->nprocs != proc) {
                        continue;
                    }

                    for (int k = 0; k < queues[dep].size(); k++) {
                        auto& my_zoid = queues[prev_dep][j];
                        if (my_zoid.num % comm->nprocs != comm->me) {
                            continue;
                        }
                        auto& recv_neighbors = curr_dt ? recv_from_neighbors_many_cuts[my_zoid.num]
                                : recv_from_neighbors_many_cuts_next_dt[my_zoid.num];

                        auto find = std::find(recv_neighbors.begin(), recv_neighbors.end(), send_zoid.num);
                        if (find != recv_neighbors.end()) {
                            int find_idx = std::distance(recv_neighbors.begin(), find);
                            int nrecv_force = 0;
                            int nrecv_pos = 0;
                            int nrecv_vel = 0;
                            for (int t = 0; t < NUM_TIMESTEPS_IN_PARALLEL + 1; t++) {
                                nrecv_force += my_zoid.recv_force_idxs_double_buffering[t][find_idx].size();
                                nrecv_pos += my_zoid.recv_pos_idxs_double_buffering[t][find_idx].size();
                                nrecv_vel += my_zoid.recv_vel_idxs_double_buffering[t][find_idx].size();
                            }
                            int nrecv_from_send_zoid = nrecv_force + nrecv_pos + nrecv_vel;
                            recv_proc_sizes[dep][proc].push_back(nrecv_from_send_zoid);
                        }
                    }
                }
            }
        }
    }

    void INIT_SEND_RECV_BUFFERS() {
        nsend_buf_many_cuts = new int[comm->nprocs];
        nrecv_buf_many_cuts = new int[comm->nprocs];

        buf_recv_many_cuts = new double*[comm->nprocs];
        buf_send_many_cuts = new double*[comm->nprocs];

        constexpr int INITIAL_SIZE = 1024;

        for (int proc = 0; proc < comm->nprocs; proc++) {
            buf_send_many_cuts[proc] = new double[INITIAL_SIZE];
            buf_recv_many_cuts[proc] = new double[INITIAL_SIZE];
            nsend_buf_many_cuts[proc] = INITIAL_SIZE;
            nrecv_buf_many_cuts[proc] = INITIAL_SIZE;
        }
    }

    void GROW_SEND_MANY_CUTS(int proc, int new_size) {
        assert(new_size > nsend_buf_many_cuts[proc]);

        delete[] buf_send_many_cuts[proc];
        nsend_buf_many_cuts[proc] = 0;

        buf_send_many_cuts[proc] = new double[new_size];
        nsend_buf_many_cuts[proc] = new_size;
    }

    void GROW_RECV_MANY_CUTS(int proc, int new_size) {
        assert(new_size > nrecv_buf_many_cuts[proc]);

        delete[] buf_recv_many_cuts[proc];
        nrecv_buf_many_cuts[proc] = 0;

        buf_recv_many_cuts[proc] = new double[new_size];
        nrecv_buf_many_cuts[proc] = new_size;
    }

    template <bool curr_dt>
    void CONSTRUCT_SEND_PROC_SIZES() {
        auto& queues = curr_dt ? queues_many_cuts : queues_many_cuts_next_dt;

        for (int dep = 0; dep < NUM_DEPS - 1; dep++) {
            send_proc_sizes[dep] = new std::vector<int>[comm->nprocs];
            for (int proc = 0; proc < comm->nprocs; proc++) {
                auto& sizes = send_proc_sizes[dep][proc];

                for (int j = 0; j < queues[dep].size(); j++) {
                    auto& zoid = queues[dep][j];
                    if (zoid.num % comm->nprocs != comm->me) {
                        continue;
                    }

                    auto& send_to_neighbors = curr_dt ? send_to_neighbors_many_cuts[zoid.num] : send_to_neighbors_many_cuts_next_dt[zoid.num];
                    int total_nsend_force = 0;
                    int total_nsend_pos = 0;
                    int total_nsend_vel = 0;

                    if (false && zoid.num == 2 && proc == 0) {
                        std::stringstream s1;
                        for (auto& neigh : send_to_neighbors) {
                            s1 << neigh << " ";
                        }
                        std::cout << "zoid: " << zoid.num << " construct force offset OUTSIDE. neighbors: " << s1.str() << std::endl;
                    }

                    for (int i = 0; i < send_to_neighbors.size(); i++) {
                        int send_zoid_num = send_to_neighbors[i];
                        if (send_zoid_num % comm->nprocs != proc) {
                            continue;
                        }

                        if (false && zoid.num == 2 && proc == 0) {
                            std::cout << "zoid: " << zoid.num << " send to: " << send_zoid_num << " construct force offset." << std::endl;
                        }

                        // send force
                        int nsend_force = 0;
                        for (int t = 0; t < NUM_TIMESTEPS_IN_PARALLEL + 1; t++) {
                            nsend_force += zoid.send_force_idxs_double_buffering[t][i].size();
                        }

                        int nsend_pos = 0;
                        for (int t = 0; t < NUM_TIMESTEPS_IN_PARALLEL + 1; t++) {
                            nsend_pos += zoid.send_pos_idxs_double_buffering[t][i].size();
                        }

                        int nsend_vel = 0;
                        for (int t = 0; t < NUM_TIMESTEPS_IN_PARALLEL + 1; t++) {
                            nsend_vel += zoid.send_vel_idxs_double_buffering[t][i].size();
                        }

                        total_nsend_force += nsend_force;
                        total_nsend_pos += nsend_pos;
                        total_nsend_vel += nsend_vel;

                        int nsend_to_zoid = nsend_force + nsend_pos + nsend_vel;
                    }

                    int total_nsend_from_zoid = total_nsend_force + total_nsend_pos + total_nsend_vel;
                    sizes.push_back(total_nsend_from_zoid);
                }
            }
        }
    }

    template <bool curr_dt>
    void RECEIVE_DATA_PROCESS_ZOID_MANY_CUTS(int dep, int proc, MPI_Request* request) {
        auto& queues = curr_dt ? queues_many_cuts[dep] : queues_many_cuts_next_dt[dep];

        int total_recv_from_proc = 0;
        for (int i = 0; i < recv_proc_sizes[dep][proc].size(); i++) {
            total_recv_from_proc += recv_proc_sizes[dep][proc][i];
        }

        int total_doubles_recv_from_proc = DEBUG_SEND_RECV_DATA ? (3 + 1) * total_recv_from_proc : 3 * total_recv_from_proc;

        if (total_doubles_recv_from_proc > nrecv_buf_many_cuts[proc]) {
            GROW_RECV_MANY_CUTS(proc, total_doubles_recv_from_proc);
        }

        if (total_doubles_recv_from_proc > 0) {
            int mpi_tag = get_mpi_tag(comm->me, proc);
            MPI_Irecv(buf_recv_many_cuts[proc], total_doubles_recv_from_proc, MPI_DOUBLE, proc, mpi_tag, world,
                      request);
        }
    }

    template <bool curr_dt>
    int PACK_DATA_TO_PROC_HELPER(queue_info& zoid, int proc, double* buf) {
        auto& send_to_neighbors = curr_dt ? send_to_neighbors_many_cuts[zoid.num] : send_to_neighbors_many_cuts_next_dt[zoid.num];

        int buf_idx = 0;

        if (zoid.num == 2 && proc == 1) {
            std::stringstream s1;
            for (auto& neigh : send_to_neighbors) {
                s1 << neigh << " ";
            }

            std::cout << "neighbors: " << s1.str() << std::endl;
        }

        for (int i = 0; i < send_to_neighbors.size(); i++) {
            if (send_to_neighbors[i] % comm->nprocs != proc) {
                continue;
            }
            if (zoid.num == 2 && proc == 1) {
                std::cout << "ZOID HERE" << std::endl;
            }
            for (int t = 0; t < NUM_TIMESTEPS_IN_PARALLEL + 1; t++) {
                auto& send_force_idxs = zoid.send_force_idxs_double_buffering[t][i];
                auto& send_pos_idxs = zoid.send_force_idxs_double_buffering[t][i];
                auto& send_vel_idxs = zoid.send_force_idxs_double_buffering[t][i];

                for (int k = 0; k < send_force_idxs.size(); k++) {
                    int idx = send_force_idxs[k];
                    int tag = zoid.tag_stencil_md[0][idx];

                    buf[buf_idx++] = ubuf(tag).d;
                    buf[buf_idx++] = zoid.f_stencil_md[0][idx].x;
                    buf[buf_idx++] = zoid.f_stencil_md[0][idx].y;
                    buf[buf_idx++] = zoid.f_stencil_md[0][idx].z;

                    zoid.f_stencil_md[0][idx].x = 0;
                    zoid.f_stencil_md[0][idx].y = 0;
                    zoid.f_stencil_md[0][idx].z = 0;
                }

                for (int k = 0; k < send_pos_idxs.size(); k++) {
                    int idx = send_pos_idxs[k];
                    int tag = zoid.tag_stencil_md[0][idx];

                    buf[buf_idx++] = ubuf(tag).d;
                    buf[buf_idx++] = zoid.x_stencil_md[0][idx].x;
                    buf[buf_idx++] = zoid.x_stencil_md[0][idx].y;
                    buf[buf_idx++] = zoid.x_stencil_md[0][idx].z;
                }

                for (int k = 0; k < send_vel_idxs.size(); k++) {
                    int idx = send_vel_idxs[k];
                    int tag = zoid.tag_stencil_md[0][idx];

                    buf[buf_idx++] = ubuf(tag).d;
                    buf[buf_idx++] = zoid.v_stencil_md[0][idx].x;
                    buf[buf_idx++] = zoid.v_stencil_md[0][idx].y;
                    buf[buf_idx++] = zoid.v_stencil_md[0][idx].z;
                }
            }
        }

        return buf_idx;
    }

    template <bool curr_dt>
    void PACK_DATA_MANY_CUTS(int dep) {
        auto& queues = curr_dt ? queues_many_cuts : queues_many_cuts_next_dt;

        for (int proc = 0; proc < comm->nprocs; proc++) {
            int total_send_to_proc = 0;
            for (int i = 0; i < send_proc_sizes[dep][proc].size(); i++) {
                total_send_to_proc += send_proc_sizes[dep][proc][i];
            }
            int total_doubles_send_to_proc = DEBUG_SEND_RECV_DATA ? (3 + 1) * total_send_to_proc : 3 * total_send_to_proc;
            if (total_doubles_send_to_proc > nsend_buf_many_cuts[proc]) {
                GROW_SEND_MANY_CUTS(proc, total_doubles_send_to_proc);
            }
        }

        std::vector<int> buf_offsets(comm->nprocs, 0);
        std::vector<int> num_times_packed(comm->nprocs, 0);

        for (int j = 0; j < queues[dep].size(); j++) {
            auto& zoid = queues[dep][j];
            if (zoid.num % comm->nprocs != comm->me) {
                continue;
            }

            for (int proc = 0; proc < comm->nprocs; proc++) {
                int npack = PACK_DATA_TO_PROC_HELPER<curr_dt>(zoid, proc, &buf_send_many_cuts[proc][buf_offsets[proc]]);
                if (npack != send_proc_sizes[dep][proc][num_times_packed[proc]]) {
                    std::cout << "zoid: " << zoid.num << " dep: " << dep << " pack to proc: " << proc
                    << " num packed: " << npack << " nsend: " << send_proc_sizes[dep][proc][num_times_packed[proc]]
                    << " idx: " << num_times_packed[proc] << std::endl;

                    std::stringstream s1;
                    for (auto& size : send_proc_sizes[dep][proc]) {
                        s1 << size << " ";
                    }
                    std::cout << "SIZES: " << s1.str() << std::endl;
                }
                assert(npack == send_proc_sizes[dep][proc][num_times_packed[proc]]);
                num_times_packed[proc]++;
            }
        }
    }

    void FORCE_COMPUTE_ZOID_MANY_CUTS(queue_info& zoid, int timestep) {
        const auto * _noalias const x = zoid.x_stencil_md[timestep % DOUBLE_BUFFERING].data();
        auto * _noalias const f = zoid.f_stencil_md[timestep % 1].data();

        auto pair = (PairLJCut*) force->pair;
        auto bond = (BondFENE*) force->bond;

        const auto& bond_list = zoid.bond_list[timestep];
        const auto& neighbor_list = zoid.neighbor_list[timestep];

        const double * _noalias const special_lj = force->special_lj;

        // auto* spinlocks = next->spinlocks;
        auto* _noalias spinlocks = zoid.spinlocks_stencil_md[0];

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

        const auto& atom_type = zoid.type_stencil_md[0];

        const auto& local_idxs = zoid.local_idxs_per_timestep[timestep];
        const int nlocal = local_idxs.size();

        int num_chunks = nlocal / MODIFY_GRAINSIZE + 1;
        int num_workers = __cilkrts_get_nworkers();

        auto* claimed = zoid.claimed_flags_stencil_md[0];

        const auto& tags = zoid.tag_stencil_md[0];

        int chunks_per_worker = num_chunks / num_workers;
        int chunk_size = MODIFY_GRAINSIZE;

        #pragma cilk grainsize 1
        cilk_for (int ii = 0; ii < num_chunks; ii++) {
            int worker_number = __cilkrts_get_worker_number();
            int start_chunk = worker_number * chunks_per_worker;

            for (int c = 0; c < num_chunks; ++c) {
                int s = (c + start_chunk) % num_chunks;

                if (claimed[s].test(std::memory_order_relaxed)) {
                    continue;
                }

                if (!claimed[s].test_and_set(std::memory_order_relaxed)) {
                    for (int idx = s * chunk_size; idx < (s + 1) * chunk_size && idx < nlocal; idx++) {
                        int i = local_idxs[idx];

                        const int itype = atom_type[i];

                        // const int *_noalias const jlist = firstneigh[i];
                        const auto &jlist = neighbor_list[i];
                        const double *_noalias const cutsqi = cutsq[itype];
                        const double *_noalias const offseti = offset[itype];
                        const double *_noalias const lj1i = lj1[itype];
                        const double *_noalias const lj2i = lj2[itype];
                        const double *_noalias const lj3i = lj3[itype];
                        const double *_noalias const lj4i = lj4[itype];

                        double xtmp = x[i].x;
                        double ytmp = x[i].y;
                        double ztmp = x[i].z;
                        // int jnum = numneigh[i];
                        int jnum = jlist.size();

                        double fxtmp = 0.0;
                        double fytmp = 0.0;
                        double fztmp = 0.0;

                        for (int jj = 0; jj < jnum; jj++) {
                            double evdwl = 0.0;
                            // int j = jlist[jj];
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

                        auto &lst_bonds = bond_list[i];
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

                    if (USE_BREAK) {
                        break;
                    }
                }
            }
        }

        for (int i = 0; i < num_chunks; i++) {
            claimed[i].clear(std::memory_order_relaxed);
        }

    }

    inline void post_force_stencil_md_zoid_many_cuts_setup(queue_info& zoid, int timestep) {
        const auto& v = zoid.v_stencil_md[timestep % 1];
        auto& f = zoid.f_stencil_md[timestep % 1];

        const auto& type = zoid.type_stencil_md[0];
        const auto& mask = zoid.mask_stencil_md[0];

        int n_post_force = modify->n_post_force;

        assert(n_post_force == 1);

        auto fix_post_force = (FixLangevin*) modify->fix[modify->list_post_force[0]];

        auto gfactor1 = fix_post_force->gfactor1;
        auto gfactor2 = fix_post_force->gfactor2;
        // fix_post_force->compute_target();
        auto tsqrt = fix_post_force->tsqrt;

        const auto& local_idxs = zoid.local_idxs_per_timestep[timestep];
        const int nlocal = local_idxs.size();

        #pragma cilk grainsize 2048
        cilk_for (int idx = 0; idx < nlocal; idx++) {
            int i = local_idxs[idx];
            if (mask[i]) {
                double gamma1 = gfactor1[type[i]];
                double gamma2 = gfactor2[type[i]] * tsqrt;

                double rand_x = 0.6;
                double rand_y = 0.6;
                double rand_z = 0.6;

                dbl3_t_stencil_md fran = {gamma2 * (rand_x - 0.5), gamma2 * (rand_y - 0.5), gamma2 * (rand_z - 0.5)};
                dbl3_t_stencil_md fdrag = {gamma1 * v[i].x, gamma1 * v[i].y, gamma1 * v[i].z};

                f[i].x += fdrag.x + fran.x;
                f[i].y += fdrag.y + fran.y;
                f[i].z += fdrag.z + fran.z;
            } else {
                assert(false);
            }
        }
    }
    /* End code for many zoids per dimension */
};

}