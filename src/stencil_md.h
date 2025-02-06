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

static int get_mpi_tag_many_cuts(int dst, int src) {
    return (dst << 14) | src;
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

    void TEST_AGAINST_LAMMPS_POS_DOUBLE_BUFFERING(bool curr_dt, int timestep_to_compare_against,
                                                  double* test_x, queue_info& zoid, int t) {
        auto& local_idxs = zoid.local_idxs_per_timestep[t];
        auto& tags = zoid.tag_stencil_md[0];
        auto& x = zoid.x_stencil_md[t % DOUBLE_BUFFERING];

        for (int i = 0; i < local_idxs.size(); i++) {
            int idx = local_idxs[i];
            int tag = tags[idx];

            double my_x = x[idx].x;
            double my_y = x[idx].y;
            double my_z = x[idx].z;

            /*
            while (my_x < domain->boxlo[0]) {
                my_x += domain->prd[0];
            }
            while (my_y < domain->boxlo[1]) {
                my_y += domain->prd[1];
            }
            while (my_z < domain->boxlo[2]) {
                my_z += domain->prd[2];
            }
            */
            while (my_x < domain->boxlo[0]) {
                my_x += domain->prd[0];
            }
            while (my_x >= domain->boxhi[0]) {
                my_x -= domain->prd[0];
            }
            while (my_y < domain->boxlo[1]) {
                my_y += domain->prd[1];
            }
            while (my_y >= domain->boxhi[1]) {
                my_y -= domain->prd[1];
            }
            while (my_z < domain->boxlo[2]) {
                my_z += domain->prd[2];
            }
            while (my_z >= domain->boxhi[2]) {
                my_z -= domain->prd[2];
            }

            double lammps_x = test_x[tag * 3 + 0];
            double lammps_y = test_x[tag * 3 + 1];
            double lammps_z = test_x[tag * 3 + 2];

            /*
            while (lammps_x < domain->boxlo[0]) {
                lammps_x += domain->prd[0];
            }
            while (lammps_y < domain->boxlo[1]) {
                lammps_y += domain->prd[1];
            }
            while (lammps_z < domain->boxlo[2]) {
                lammps_z += domain->prd[2];
            }
            */

            while (lammps_x < domain->boxlo[0]) {
                lammps_x += domain->prd[0];
            }
            while (lammps_x >= domain->boxhi[0]) {
                lammps_x -= domain->prd[0];
            }
            while (lammps_y < domain->boxlo[1]) {
                lammps_y += domain->prd[1];
            }
            while (lammps_y >= domain->boxhi[1]) {
                lammps_y -= domain->prd[1];
            }
            while (lammps_z < domain->boxlo[2]) {
                lammps_z += domain->prd[2];
            }
            while (lammps_z >= domain->boxhi[2]) {
                lammps_z -= domain->prd[2];
            }

            bool all_close = fabs(lammps_x - my_x) < 5e-5 && fabs(lammps_y - my_y) < 5e-5 && fabs(lammps_z - my_z) < 5e-5;

            if (!all_close) {
                std::stringstream o;
                o << RED << "ERROR ON POS. curr_dt: " << curr_dt << " zoid: " << zoid.num
                << " idx: " << idx << " tag: " << tag
                << " what I have: " << my_x << " " << my_y << " " << my_z
                << " what lammps has: " << lammps_x << " " << lammps_y << " " << lammps_z
                << " diff: "
                << fabs(lammps_x - my_x) << " " << fabs(lammps_y - my_y) << " " << fabs(lammps_z - my_z)
                << " overall timestep: " << timestep_to_compare_against
                << " actual pos: " << x[idx].x << " " << x[idx].y << " " << x[idx].z
                << RESET_COLOR << std::endl;
                std::cout << o.str();
                assert(false);
            }
        }
    }

    void TEST_AGAINST_LAMMPS_VEL_DOUBLE_BUFFERING(bool curr_dt, int timestep_to_compare_against,
                                                  double* test_v, queue_info& zoid, int t) {
        auto& local_idxs = zoid.local_idxs_per_timestep[t];
        auto& tags = zoid.tag_stencil_md[0];
        auto& x = zoid.x_stencil_md[t % DOUBLE_BUFFERING];
        auto& v = zoid.v_stencil_md[0];

        for (int i = 0; i < local_idxs.size(); i++) {
            int idx = local_idxs[i];
            int tag = tags[idx];

            double my_x = v[idx].x;
            double my_y = v[idx].y;
            double my_z = v[idx].z;

            double lammps_x = test_v[tag * 3 + 0];
            double lammps_y = test_v[tag * 3 + 1];
            double lammps_z = test_v[tag * 3 + 2];

            bool all_close = fabs(lammps_x - my_x) < 5e-5 && fabs(lammps_y - my_y) < 5e-5 && fabs(lammps_z - my_z) < 5e-5;

            if (!all_close) {
                std::stringstream o;
                o << RED << "ERROR ON VEL. curr_dt: " << curr_dt << " zoid: " << zoid.num
                  << " idx: " << idx << " tag: " << tag
                  << " what I have: " << my_x << " " << my_y << " " << my_z
                  << " what lammps has: " << lammps_x << " " << lammps_y << " " << lammps_z
                  << " diff: "
                  << fabs(lammps_x - my_x) << " " << fabs(lammps_y - my_y) << " " << fabs(lammps_z - my_z)
                  << " overall timestep: " << timestep_to_compare_against
                  << " pos: " << x[idx].x << " " << x[idx].y << " " << x[idx].z
                  << RESET_COLOR << std::endl;

                std::cout << o.str();

                assert(false);
            }
        }
    }

    void TEST_AGAINST_LAMMPS_FORCE_DOUBLE_BUFFERING(bool curr_dt, int timestep_to_compare_against,
                                                    double* test_f, queue_info& zoid, int t) {
        auto& local_idxs = zoid.local_idxs_per_timestep[t];
        auto& tags = zoid.tag_stencil_md[t % 1];
        auto& x = zoid.x_stencil_md[t % DOUBLE_BUFFERING];
        auto& f = zoid.f_stencil_md[t % 1];

        for (int i = 0; i < local_idxs.size(); i++) {
            int idx = local_idxs[i];
            int tag = tags[idx];

            double my_x = f[idx].x;
            double my_y = f[idx].y;
            double my_z = f[idx].z;

            double lammps_x = test_f[tag * 3 + 0];
            double lammps_y = test_f[tag * 3 + 1];
            double lammps_z = test_f[tag * 3 + 2];

            bool all_close = fabs(lammps_x - my_x) < 5e-5 && fabs(lammps_y - my_y) < 5e-5 && fabs(lammps_z - my_z) < 5e-5;

            if (!all_close) {
                std::stringstream o;
                o << RED << "ERROR ON FORCE. curr_dt: " << curr_dt << " zoid: " << zoid.num
                  << " idx: " << idx << " tag: " << tag
                  << " what I have: " << my_x << " " << my_y << " " << my_z
                  << " what lammps has: " << lammps_x << " " << lammps_y << " " << lammps_z
                  << " diff: "
                  << fabs(lammps_x - my_x) << " " << fabs(lammps_y - my_y) << " " << fabs(lammps_z - my_z)
                  << " overall timestep: " << timestep_to_compare_against << " t: " << t
                  << " pos: " << x[idx].x << " " << x[idx].y << " " << x[idx].z
                  << RESET_COLOR << std::endl;

                std::cout << o.str();

                assert(false);
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

    /* Start code for many zoids per dimension */
    std::vector<queue_info> queues_many_cuts[NUM_DEPS];
    std::vector<queue_info> queues_many_cuts_next_dt[NUM_DEPS];

    std::vector<queue_info> my_queues_many_cuts[NUM_DEPS];
    std::vector<queue_info> my_queues_many_cuts_next_dt[NUM_DEPS];

    static constexpr int NUM_CUTS_PER_DIMENSION = 4;
    static constexpr int NUM_ZOIDS_PER_DIMENSION = NUM_CUTS_PER_DIMENSION * 2;
    static constexpr int NUM_ZOIDS_MANY_CUTS = NUM_ZOIDS_PER_DIMENSION * NUM_ZOIDS_PER_DIMENSION * NUM_ZOIDS_PER_DIMENSION;

    queue_info* zoid_num_to_zoid_many_cuts;
    queue_info* zoid_num_to_zoid_many_cuts_next_dt;
    std::vector<int>* send_to_neighbors_many_cuts;
    std::vector<int>* send_to_neighbors_many_cuts_next_dt;
    std::vector<int>* recv_from_neighbors_many_cuts;
    std::vector<int>* recv_from_neighbors_many_cuts_next_dt;

    std::vector<std::vector<int>> send_to_neighbors_not_my_proc_idxs;
    std::vector<std::vector<int>> send_to_neighbors_not_my_proc_idxs_next_dt;
    std::vector<int> send_to_neighbors_num_not_in_proc;
    std::vector<int> send_to_neighbors_num_not_in_proc_next_dt;

    std::vector<std::vector<int>> recv_from_neighbors_not_my_proc_idxs;
    std::vector<std::vector<int>> recv_from_neighbors_not_my_proc_idxs_next_dt;

    std::map<std::pair<int, int>, int> recv_request_zoid_to_idx[NUM_DEPS];
    std::map<std::pair<int, int>, int> recv_request_zoid_to_idx_next_dt[NUM_DEPS];
    std::map<int, std::pair<int, int>> recv_request_idx_to_zoid[NUM_DEPS];
    std::map<int, std::pair<int, int>> recv_request_idx_to_zoid_next_dt[NUM_DEPS];

    std::vector<int>* recv_proc_zoid_offsets[NUM_ZOIDS_MANY_CUTS];
    std::vector<int>* recv_proc_zoid_sizes[NUM_ZOIDS_MANY_CUTS];
    std::vector<int>* recv_proc_zoid_offsets_next_dt[NUM_ZOIDS_MANY_CUTS];
    std::vector<int>* recv_proc_zoid_sizes_next_dt[NUM_ZOIDS_MANY_CUTS];

    std::vector<std::vector<int>> send_proc_zoid_offsets;
    std::vector<std::vector<int>> send_proc_zoid_sizes;
    std::vector<std::vector<int>> send_proc_zoid_offsets_next_dt;
    std::vector<std::vector<int>> send_proc_zoid_sizes_next_dt;

    double** buf_recv_many_cuts[NUM_DEPS];
    double** buf_send_many_cuts[NUM_DEPS];
    int* nrecv_buf_many_cuts[NUM_DEPS];
    int* nsend_buf_many_cuts[NUM_DEPS];

    std::vector<std::vector<int>> dep_proc_to_recv_zoids[NUM_DEPS];
    std::vector<std::vector<int>> dep_proc_to_recv_zoids_next_dt[NUM_DEPS];

    std::vector<std::vector<std::vector<int>>> dep_proc_recv_zoid_to_find_idxs[NUM_DEPS];
    std::vector<std::vector<std::vector<int>>> dep_proc_recv_zoid_to_find_idxs_next_dt[NUM_DEPS];

    static constexpr int MAX_NEIGHBORS = 26;
    double*** buf_send_zoid_to_zoid[NUM_PIPELINE_STAGES];
    int** nsend_buf_send_zoid_to_zoid[NUM_PIPELINE_STAGES];
    double*** buf_recv_zoid_to_zoid[NUM_PIPELINE_STAGES];
    int** nrecv_buf_recv_zoid_to_zoid[NUM_PIPELINE_STAGES];

    std::map<std::pair<int, int>, int> ZOID_TO_ZOID_TO_VCI_IDX;
    std::map<std::pair<int, int>, int> ZOID_TO_ZOID_TO_VCI_IDX_NEXT_DT;

    std::vector<std::vector<int>> send_zoid_to_zoid_sizes;
    std::vector<std::vector<int>> send_zoid_to_zoid_sizes_next_dt;
    std::vector<std::vector<int>> recv_zoid_to_zoid_sizes;
    std::vector<std::vector<int>> recv_zoid_to_zoid_sizes_next_dt;

    std::vector<std::vector<int>> send_zoid_to_zoid_sizes_setup;
    std::vector<std::vector<int>> recv_zoid_to_zoid_sizes_setup;

    std::vector<std::vector<int>> send_zoid_to_zoid_sizes_pipelined[NUM_PIPELINE_STAGES];
    std::vector<std::vector<int>> send_zoid_to_zoid_sizes_pipelined_next_dt[NUM_PIPELINE_STAGES];
    std::vector<std::vector<int>> recv_zoid_to_zoid_sizes_pipelined[NUM_PIPELINE_STAGES];
    std::vector<std::vector<int>> recv_zoid_to_zoid_sizes_pipelined_next_dt[NUM_PIPELINE_STAGES];

    void INIT_ZOID_MANY_CUTS() {
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

        // zoid_num to dep0 map?
        // TODO: renumber zoids
        std::vector<int> proc_to_zoid_count(comm->nprocs, 0);
        std::map<int, std::vector<std::array<int, 3>>> proc_to_zoids;
        std::map<std::array<int, 3>, int> zoid_to_proc;

        int num_dep0_zoids = queues_many_cuts[0].size();
        int num_dep1_zoids = queues_many_cuts[1].size();
        int num_dep2_zoids = queues_many_cuts[2].size();
        int num_dep3_zoids = queues_many_cuts[3].size();

        int curr_proc = 0;
        for (int i = 1; i < NUM_ZOIDS_PER_DIMENSION; i += 2) {
            for (int j = 1; j < NUM_ZOIDS_PER_DIMENSION; j += 2) {
                for (int k = 1; k < NUM_ZOIDS_PER_DIMENSION; k += 2) {
                    if (proc_to_zoid_count[curr_proc] >= num_dep0_zoids / comm->nprocs) {
                        curr_proc++;
                    }
                    proc_to_zoid_count[curr_proc]++;
                    proc_to_zoids[curr_proc].push_back({i, j, k});
                    zoid_to_proc[{i, j, k}] = curr_proc;
                }
            }
        }

        bool claimed[NUM_ZOIDS_PER_DIMENSION][NUM_ZOIDS_PER_DIMENSION][NUM_ZOIDS_PER_DIMENSION] = {0};

        std::map<std::array<int, 3>, std::set<std::array<int, 3>>> tmp_send_neighbors;
        std::map<std::array<int, 3>, std::set<std::array<int, 3>>> tmp_recv_neighbors;

        for (int i = 0; i < NUM_ZOIDS_PER_DIMENSION; i++) {
            for (int j = 0; j < NUM_ZOIDS_PER_DIMENSION; j++) {
                for (int k = 0; k < NUM_ZOIDS_PER_DIMENSION; k++) {
                    std::array<int, 3> my_pos = {i, j, k};
                    if (i % 2 == 1) {
                        tmp_send_neighbors[my_pos].insert({i - 1, j, k});
                        tmp_send_neighbors[my_pos].insert({(i + 1) % NUM_ZOIDS_PER_DIMENSION, j, k});
                    }
                    if (j % 2 == 1) {
                        tmp_send_neighbors[my_pos].insert({i, j - 1, k});
                        tmp_send_neighbors[my_pos].insert({i, (j + 1) % NUM_ZOIDS_PER_DIMENSION, k});
                    }
                    if (k % 2 == 1) {
                        tmp_send_neighbors[my_pos].insert({i, j, k - 1});
                        tmp_send_neighbors[my_pos].insert({i, j, (k + 1) % NUM_ZOIDS_PER_DIMENSION});
                    }
                }
            }
        }

        for (auto& [zoid, send_zoids] : tmp_send_neighbors) {
            for (auto& z : send_zoids) {
                tmp_recv_neighbors[z].insert(zoid);
            }
        }

        int num_zoids_per_dep[NUM_DEPS] = {num_dep0_zoids, num_dep0_zoids + num_dep1_zoids,
                                           num_dep0_zoids + num_dep1_zoids + num_dep2_zoids, NUM_ZOIDS_MANY_CUTS};

        int dep_to_val[NUM_DEPS] = {0, 2, 4, 6};

        for (int dep = 1; dep < NUM_DEPS; dep++) {
            for (int proc = 0; proc < comm->nprocs; proc++) {
                auto& zoids = proc_to_zoids[proc];
                std::set<std::array<int, 3>> all_neighbors;
                std::map<std::array<int, 3>, int> neighbor_to_count;
                for (auto& z : zoids) {
                    int zoid_dep = (z[0] % 2 == 0) + (z[1] % 2 == 0) + (z[2] % 2 == 0);
                    if (zoid_dep == dep - 1) {
                        auto& neighbors = tmp_send_neighbors[z];
                        for (auto& n : neighbors) {
                            all_neighbors.insert(n);
                            neighbor_to_count[n]++;
                        }
                    }
                }

                // pick out the zoids that have most of their neighbors
                for (auto& [k, v] : neighbor_to_count) {
                    if (v == dep_to_val[dep]) {
                        proc_to_zoids[proc].push_back(k);
                        proc_to_zoid_count[proc]++;
                        claimed[k[0]][k[1]][k[2]] = true;
                    } else if (v > dep_to_val[dep] / 2) {
                        if (!claimed[k[0]][k[1]][k[2]] && proc_to_zoid_count[proc] < (num_zoids_per_dep[dep]) / comm->nprocs) {
                            proc_to_zoids[proc].push_back(k);
                            proc_to_zoid_count[proc]++;
                            claimed[k[0]][k[1]][k[2]] = true;
                        }
                    }
                }
            }

            for (int proc = 0; proc < comm->nprocs; proc++) {
                auto& zoids = proc_to_zoids[proc];
                std::set<std::array<int, 3>> all_neighbors;
                std::map<std::array<int, 3>, int> neighbor_to_count;
                for (auto& z : zoids) {
                    int zoid_dep = (z[0] % 2 == 0) + (z[1] % 2 == 0) + (z[2] % 2 == 0);
                    if (zoid_dep == dep - 1) {
                        auto& neighbors = tmp_send_neighbors[z];
                        for (auto& n : neighbors) {
                            all_neighbors.insert(n);
                            neighbor_to_count[n]++;
                        }
                    }
                }

                // pick out the zoids that have half of their neighbors, tiebreak I guess based on earlier process
                for (auto& [k, v] : neighbor_to_count) {
                    if (v >= dep_to_val[dep] / 2) {
                        if (!claimed[k[0]][k[1]][k[2]] && proc_to_zoid_count[proc] < (num_zoids_per_dep[dep]) / comm->nprocs) {
                            proc_to_zoids[proc].push_back(k);
                            proc_to_zoid_count[proc]++;
                            claimed[k[0]][k[1]][k[2]] = true;
                        }
                    }
                }
            }
        }

        std::set<std::array<int, 3>> test_zoids;
        for (int proc = 0; proc < comm->nprocs; proc++) {
            for (auto& zoid : proc_to_zoids[proc]) {
                test_zoids.insert(zoid);
            }
        }

        std::cout << "test zoids size: " << test_zoids.size() << std::endl;
        assert(test_zoids.size() == NUM_ZOIDS_MANY_CUTS);

        std::map<std::array<int, 3>, int> zoid_to_proc_final;
        for (int proc = 0; proc < comm->nprocs; proc++) {
            for (auto& zoid : proc_to_zoids[proc]) {
                zoid_to_proc_final[zoid] = proc;
            }
        }

        std::vector<int> proc_zoid_counts(comm->nprocs, 0);

        // int zoid_num = 0;
        for (int dep = 0; dep < NUM_DEPS; dep++) {
            for (int j = 0; j < queues_many_cuts[dep].size(); j++) {
                auto& zoid = queues_many_cuts[dep][j];
                auto proc_assigned_to_zoid = zoid_to_proc_final.at({zoid.where[0], zoid.where[1], zoid.where[2]});
                zoid.num = proc_zoid_counts[proc_assigned_to_zoid] * comm->nprocs + proc_assigned_to_zoid;
                proc_zoid_counts[proc_assigned_to_zoid]++;
                // zoid.num = zoid_num;
                // zoid_num++;
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

        std::set<int> all_zoid_nums;
        for (int dep = 0; dep < NUM_DEPS; dep++) {
            for (int j = 0; j < queues_many_cuts[dep].size(); j++) {
                auto &zoid = queues_many_cuts[dep][j];
                int zoid_num = zoid.num;
                assert(zoid_num >= 0 && zoid_num < NUM_ZOIDS_MANY_CUTS);
                all_zoid_nums.insert(zoid_num);
            }
        }

        assert(all_zoid_nums.size() == NUM_ZOIDS_MANY_CUTS);

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
                    zoid.x_stencil_md = new std::vector<dbl3_t_stencil_md>[DOUBLE_BUFFERING];
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

                    zoid.send_vel_idxs_double_buffering = new std::vector<int>*[NUM_TIMESTEPS_IN_PARALLEL + 1];
                    zoid.recv_vel_idxs_double_buffering = new std::vector<int>*[NUM_TIMESTEPS_IN_PARALLEL + 1];

                    constexpr int MAX_NEIGHBORS = 26;
                    zoid.send_force_idxs_double_buffering_flattened = new std::vector<int>[MAX_NEIGHBORS];
                    zoid.send_pos_idxs_double_buffering_flattened = new std::vector<int>*[DOUBLE_BUFFERING];
                    zoid.send_vel_idxs_double_buffering_flattened = new std::vector<int>[MAX_NEIGHBORS];

                    zoid.recv_force_idxs_double_buffering_flattened = new std::vector<int>[MAX_NEIGHBORS];
                    zoid.recv_vel_idxs_double_buffering_flattened = new std::vector<int>[MAX_NEIGHBORS];
                    zoid.recv_pos_idxs_double_buffering_flattened = new std::vector<int>*[DOUBLE_BUFFERING];

                    for (int k = 0; k < DOUBLE_BUFFERING; k++) {
                        zoid.send_pos_idxs_double_buffering_flattened[k] = new std::vector<int>[MAX_NEIGHBORS];
                        zoid.recv_pos_idxs_double_buffering_flattened[k] = new std::vector<int>[MAX_NEIGHBORS];
                    }
                    /* end stuff for 2 timesteps */
                }
            }
        }

        for (int dep = 0; dep < NUM_DEPS; dep++) {
            for (int j = 0; j < queues_many_cuts_next_dt[dep].size(); j++) {
                queue_info& zoid = queues_many_cuts_next_dt[dep][j];
                if (zoid.num % comm->nprocs == comm->me) {
                    /* start stuff for 2 timesteps */

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

                    zoid.send_vel_idxs_double_buffering = new std::vector<int>*[NUM_TIMESTEPS_IN_PARALLEL + 1];
                    zoid.recv_vel_idxs_double_buffering = new std::vector<int>*[NUM_TIMESTEPS_IN_PARALLEL + 1];

                    constexpr int MAX_NEIGHBORS = 26;
                    zoid.send_force_idxs_double_buffering_flattened = new std::vector<int>[MAX_NEIGHBORS];
                    zoid.send_pos_idxs_double_buffering_flattened = new std::vector<int>*[DOUBLE_BUFFERING];
                    zoid.send_vel_idxs_double_buffering_flattened = new std::vector<int>[MAX_NEIGHBORS];

                    zoid.recv_force_idxs_double_buffering_flattened = new std::vector<int>[MAX_NEIGHBORS];
                    zoid.recv_vel_idxs_double_buffering_flattened = new std::vector<int>[MAX_NEIGHBORS];
                    zoid.recv_pos_idxs_double_buffering_flattened = new std::vector<int>*[DOUBLE_BUFFERING];

                    for (int k = 0; k < DOUBLE_BUFFERING; k++) {
                        zoid.send_pos_idxs_double_buffering_flattened[k] = new std::vector<int>[MAX_NEIGHBORS];
                        zoid.recv_pos_idxs_double_buffering_flattened[k] = new std::vector<int>[MAX_NEIGHBORS];
                    }
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

    void INIT_MY_ZOIDS() {
        for (int dep = 0; dep < NUM_DEPS; dep++) {
            for (int j = 0; j < queues_many_cuts[dep].size(); j++) {
                auto zoid = queues_many_cuts[dep][j];
                int zoid_num = zoid.num;
                if (zoid_num % comm->nprocs != comm->me) {
                    continue;
                }

                my_queues_many_cuts[dep].push_back(zoid);
            }
        }

        for (int dep = 0; dep < NUM_DEPS; dep++) {
            for (int j = 0; j < queues_many_cuts_next_dt[dep].size(); j++) {
                auto zoid = queues_many_cuts_next_dt[dep][j];
                int zoid_num = zoid.num;
                if (zoid_num % comm->nprocs != comm->me) {
                    continue;
                }

                my_queues_many_cuts_next_dt[dep].push_back(zoid);
            }
        }

    }

    template <bool curr_dt>
    void SORT_MY_ZOIDS() {
        auto get_dep_curr_dt = [](const queue_info& zoid) {
            return (zoid.where[0] % 2 == 1) + (zoid.where[1] % 2 == 1) + (zoid.where[2] % 2 == 1);
        };

        auto get_dep_next_dt = [](const queue_info& zoid) {
            return (zoid.where[0] % 2 == 1) + (zoid.where[1] % 2 == 1) + (zoid.where[2] % 2 == 1);
        };

        for (int dep = 0; dep < NUM_DEPS; dep++) {
            auto& my_queues_at_dep = curr_dt ? my_queues_many_cuts[dep] : my_queues_many_cuts_next_dt[dep];
            if (dep == 0) {
                std::sort(my_queues_at_dep.begin(), my_queues_at_dep.end(), [&](const auto& zoid_a, const auto& zoid_b) {
                    auto& send_neighbors_a = curr_dt ? send_to_neighbors_many_cuts[zoid_a.num]
                            : send_to_neighbors_many_cuts_next_dt[zoid_a.num];
                    auto& send_neighbors_b = curr_dt ? send_to_neighbors_many_cuts[zoid_b.num]
                            : send_to_neighbors_many_cuts_next_dt[zoid_b.num];

                    // put zoids where they are sending to different zoids
                    int num_neighbors_a = 0;
                    for (int i = 0; i < send_neighbors_a.size(); i++) {
                        int send_zoid_num = send_neighbors_a[i];
                        if (send_zoid_num % comm->nprocs != comm->me) {
                            num_neighbors_a++;
                        }
                    }

                    int num_neighbors_b = 0;
                    for (int i = 0; i < send_neighbors_b.size(); i++) {
                        int send_zoid_num = send_neighbors_b[i];
                        if (send_zoid_num % comm->nprocs != comm->me) {
                            num_neighbors_b++;
                        }
                    }

                    return num_neighbors_a > num_neighbors_b;
                });
            } else {
                std::sort(my_queues_at_dep.begin(), my_queues_at_dep.end(), [&](const auto& zoid_a, const auto& zoid_b) {
                    auto& recv_neighbors_a = recv_from_neighbors_many_cuts[zoid_a.num];
                    auto& recv_neighbors_b = recv_from_neighbors_many_cuts[zoid_b.num];

                    // put zoids where they are sending to different zoids
                    int num_neighbors_a = 0;
                    for (int i = 0; i < recv_neighbors_a.size(); i++) {
                        int recv_zoid_num = recv_neighbors_a[i];
                        if (recv_zoid_num % comm->nprocs != comm->me) {
                            num_neighbors_a++;
                        }
                    }

                    int num_neighbors_b = 0;
                    for (int i = 0; i < recv_neighbors_b.size(); i++) {
                        int recv_zoid_num = recv_neighbors_b[i];
                        if (recv_zoid_num % comm->nprocs != comm->me) {
                            num_neighbors_b++;
                        }
                    }

                    return num_neighbors_a < num_neighbors_b;
                });
            }
        }
    }

    bool zoid_many_cuts_is_neighbor_all_deps(int* where_a, int* where_b) {
        for (int dim = 0; dim < domain->dimension; dim++) {
            if (where_a[dim] != where_b[dim]) {
                int diff = where_a[dim] - where_b[dim];
                // expanding zoid in this dimension based on numbering
                if (where_a[dim] % 2 == 1) {
                    if (diff != -1 && diff != 1) {
                        if (!(where_a[dim] == NUM_ZOIDS_PER_DIMENSION - 1 && where_b[dim] == 0)) {
                            return false;
                        }
                    }
                } else {
                    return false;
                }
            }
        }

        return true;
    }

    bool zoid_many_cuts_is_neighbor_all_deps_next_dt(int* where_a, int* where_b) {
        for (int dim = 0; dim < domain->dimension; dim++) {
            if (where_a[dim] != where_b[dim]) {
                int diff = where_a[dim] - where_b[dim];
                // expanding zoid in this dimension based on numbering
                if (where_a[dim] % 2 == 0) {
                    if (diff != -1 && diff != 1) {
                        if (!(where_a[dim] == 0 && where_b[dim] == NUM_ZOIDS_PER_DIMENSION - 1)) {
                            return false;
                        }
                    }
                } else {
                    return false;
                }
            }
        }

        return true;
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
                /*
                if (zoid_to_dep[zoid.num] + 1 == zoid_to_dep[other_zoid.num]) {
                    if (zoid_many_cuts_is_neighbor(zoid.where, other_zoid.where)) {
                        send_to_neighbors_many_cuts[i].push_back(j);
                        recv_from_neighbors_many_cuts[j].push_back(i);
                    }
                }
                */
                if (zoid_to_dep[zoid.num] < zoid_to_dep[other_zoid.num]) {
                    if (zoid_many_cuts_is_neighbor_all_deps(zoid.where, other_zoid.where)) {
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
                /*
                if (zoid_to_dep_next_dt[zoid.num] + 1 == zoid_to_dep_next_dt[other_zoid.num]) {
                    if (zoid_many_cuts_is_neighbor(zoid.where, other_zoid.where)) {
                        send_to_neighbors_many_cuts_next_dt[i].push_back(j);
                        recv_from_neighbors_many_cuts_next_dt[j].push_back(i);
                    }
                }
                */
                if (zoid_to_dep_next_dt[zoid.num] < zoid_to_dep_next_dt[other_zoid.num]) {
                    if (zoid_many_cuts_is_neighbor_all_deps_next_dt(zoid.where, other_zoid.where)) {
                        send_to_neighbors_many_cuts_next_dt[i].push_back(j);
                        recv_from_neighbors_many_cuts_next_dt[j].push_back(i);
                    }
                }
            }
        }

        auto get_dep_curr_dt = [](queue_info& zoid) {
            return (zoid.where[0] % 2 == 0) + (zoid.where[1] % 2 == 0) + (zoid.where[2] % 2 == 0);
        };

        auto get_dep_next_dt = [](queue_info& zoid) {
            return (zoid.where[0] % 2 == 0) + (zoid.where[1] % 2 == 0) + (zoid.where[2] % 2 == 0);
        };

        for (int i = 0; i < NUM_ZOIDS_MANY_CUTS; i++) {
            std::sort(send_to_neighbors_many_cuts[i].begin(), send_to_neighbors_many_cuts[i].end());
            std::sort(recv_from_neighbors_many_cuts[i].begin(), recv_from_neighbors_many_cuts[i].end());
            std::sort(send_to_neighbors_many_cuts_next_dt[i].begin(), send_to_neighbors_many_cuts_next_dt[i].end());
            std::sort(recv_from_neighbors_many_cuts_next_dt[i].begin(), recv_from_neighbors_many_cuts_next_dt[i].end());
        }

        recv_from_neighbors_not_my_proc_idxs.resize(NUM_ZOIDS_MANY_CUTS);
        for (int i = 0; i < NUM_ZOIDS_MANY_CUTS; i++) {
            if (i % comm->nprocs != comm->me) {
                continue;
            }

            auto& recv_neighbors = recv_from_neighbors_many_cuts[i];
            for (int j = 0; j < recv_neighbors.size(); j++) {
                if (recv_neighbors[j] % comm->nprocs != comm->me) {
                    recv_from_neighbors_not_my_proc_idxs[i].push_back(j);
                }
            }
        }

        recv_from_neighbors_not_my_proc_idxs_next_dt.resize(NUM_ZOIDS_MANY_CUTS);
        for (int i = 0; i < NUM_ZOIDS_MANY_CUTS; i++) {
            if (i % comm->nprocs != comm->me) {
                continue;
            }

            auto& recv_neighbors = recv_from_neighbors_many_cuts_next_dt[i];
            for (int j = 0; j < recv_neighbors.size(); j++) {
                if (recv_neighbors[j] % comm->nprocs != comm->me) {
                    recv_from_neighbors_not_my_proc_idxs_next_dt[i].push_back(j);
                }
            }
        }

        send_to_neighbors_not_my_proc_idxs.resize(NUM_ZOIDS_MANY_CUTS);
        send_to_neighbors_num_not_in_proc.resize(NUM_ZOIDS_MANY_CUTS);

        for (int i = 0; i < NUM_ZOIDS_MANY_CUTS; i++) {
            if (i % comm->nprocs != comm->me) {
                continue;
            }

            int idx = 0;
            auto& send_neighbors = send_to_neighbors_many_cuts[i];
            for (int j = 0; j < send_neighbors.size(); j++) {
                if (send_neighbors[j] % comm->nprocs != comm->me) {
                    send_to_neighbors_not_my_proc_idxs[i].push_back(idx++);
                } else {
                    send_to_neighbors_not_my_proc_idxs[i].push_back(-1);
                }
            }

            send_to_neighbors_num_not_in_proc[i] = idx;
        }

        send_to_neighbors_not_my_proc_idxs_next_dt.resize(NUM_ZOIDS_MANY_CUTS);
        send_to_neighbors_num_not_in_proc_next_dt.resize(NUM_ZOIDS_MANY_CUTS);

        for (int i = 0; i < NUM_ZOIDS_MANY_CUTS; i++) {
            if (i % comm->nprocs != comm->me) {
                continue;
            }

            int idx = 0;
            auto& send_neighbors = send_to_neighbors_many_cuts_next_dt[i];
            for (int j = 0; j < send_neighbors.size(); j++) {
                if (send_neighbors[j] % comm->nprocs != comm->me) {
                    send_to_neighbors_not_my_proc_idxs_next_dt[i].push_back(idx++);
                } else {
                    send_to_neighbors_not_my_proc_idxs_next_dt[i].push_back(-1);
                }
            }

            send_to_neighbors_num_not_in_proc_next_dt[i] = idx;
        }

        for (int dep = 0; dep < NUM_DEPS - 1; dep++) {
            int comm_idx = 0;
            for (int j = 0; j < my_queues_many_cuts[dep].size(); j++) {
                auto& zoid = my_queues_many_cuts[dep][j];
                int zoid_num = zoid.num;
                auto& send_neighbors = send_to_neighbors_many_cuts[zoid_num];
                for (int i = 0; i < send_neighbors.size(); i++) {
                    int send_zoid_num = send_neighbors[i];
                    if (send_zoid_num % comm->nprocs != comm->me) {
                        ZOID_TO_ZOID_TO_VCI_IDX[{zoid_num, send_zoid_num}] = (comm_idx) % NUM_COMMS;
                        comm_idx++;
                    }
                }
            }
        }

        for (int dep = 0; dep < NUM_DEPS - 1; dep++) {
            int comm_idx = 0;
            for (int j = 0; j < my_queues_many_cuts_next_dt[dep].size(); j++) {
                auto& zoid = my_queues_many_cuts_next_dt[dep][j];
                int zoid_num = zoid.num;
                auto& send_neighbors = send_to_neighbors_many_cuts_next_dt[zoid_num];
                for (int i = 0; i < send_neighbors.size(); i++) {
                    int send_zoid_num = send_neighbors[i];
                    if (send_zoid_num % comm->nprocs != comm->me) {
                        ZOID_TO_ZOID_TO_VCI_IDX_NEXT_DT[{zoid_num, send_zoid_num}] = (comm_idx) % NUM_COMMS;
                        comm_idx++;
                    }
                }
            }
        }

        std::vector<int> my_zoids_src;
        std::vector<int> my_zoids_dst;
        std::vector<int> my_zoids_comm_idx;

        for (auto& [k, v] : ZOID_TO_ZOID_TO_VCI_IDX) {
            my_zoids_src.push_back(k.first);
            my_zoids_dst.push_back(k.second);
            my_zoids_comm_idx.push_back(v);
        }

        std::vector<int> counts(comm->nprocs, 0);
        std::vector<int> displacements(comm->nprocs, 0);

        int my_count = my_zoids_src.size();
        MPI_Allgather(&my_count, 1, MPI_INT, counts.data(), 1, MPI_INT, world);

        int total_size = 0;
        for (int proc = 0; proc < comm->nprocs; proc++) {
            total_size += counts[proc];
        }

        displacements[0] = 0;
        for (int proc = 1; proc < comm->nprocs; proc++) {
            displacements[proc] = displacements[proc - 1] + counts[proc - 1];
        }

        std::vector<int> all_src;
        std::vector<int> all_dst;
        std::vector<int> all_comm_idx;
        all_src.resize(total_size);
        all_dst.resize(total_size);
        all_comm_idx.resize(total_size);

        MPI_Allgatherv(my_zoids_src.data(), counts[comm->me], MPI_INT, all_src.data(),
                       counts.data(), displacements.data(), MPI_INT, world);

        MPI_Allgatherv(my_zoids_dst.data(), counts[comm->me], MPI_INT, all_dst.data(),
                       counts.data(), displacements.data(), MPI_INT, world);

        MPI_Allgatherv(my_zoids_comm_idx.data(), counts[comm->me], MPI_INT, all_comm_idx.data(),
                       counts.data(), displacements.data(), MPI_INT, world);

        for (int i = 0; i < all_src.size(); i++) {
            int src = all_src[i];
            int dst = all_dst[i];
            int comm_idx = all_comm_idx[i];
            ZOID_TO_ZOID_TO_VCI_IDX[{src, dst}] = comm_idx;
        }

        std::vector<int> my_zoids_src_next_dt;
        std::vector<int> my_zoids_dst_next_dt;
        std::vector<int> my_zoids_comm_idx_next_dt;

        for (auto& [k, v] : ZOID_TO_ZOID_TO_VCI_IDX_NEXT_DT) {
            my_zoids_src_next_dt.push_back(k.first);
            my_zoids_dst_next_dt.push_back(k.second);
            my_zoids_comm_idx_next_dt.push_back(v);
        }

        std::vector<int> counts_next_dt(comm->nprocs, 0);
        std::vector<int> displacements_next_dt(comm->nprocs, 0);

        int my_count_next_dt = my_zoids_src_next_dt.size();
        counts_next_dt[comm->me] = my_count_next_dt;
        MPI_Allgather(&my_count_next_dt, 1, MPI_INT, counts_next_dt.data(), 1, MPI_INT, world);

        int total_size_next_dt = 0;
        for (int proc = 0; proc < comm->nprocs; proc++) {
            total_size_next_dt += counts_next_dt[proc];
        }

        displacements_next_dt[0] = 0;
        for (int proc = 1; proc < comm->nprocs; proc++) {
            displacements_next_dt[proc] = displacements_next_dt[proc - 1] + counts_next_dt[proc - 1];
        }

        std::vector<int> all_src_next_dt;
        std::vector<int> all_dst_next_dt;
        std::vector<int> all_comm_idx_next_dt;
        all_src_next_dt.resize(total_size_next_dt);
        all_dst_next_dt.resize(total_size_next_dt);
        all_comm_idx_next_dt.resize(total_size_next_dt);

        MPI_Allgatherv(my_zoids_src_next_dt.data(), counts_next_dt[comm->me], MPI_INT, all_src_next_dt.data(),
                       counts_next_dt.data(), displacements_next_dt.data(), MPI_INT, world);

        MPI_Allgatherv(my_zoids_dst_next_dt.data(), counts_next_dt[comm->me], MPI_INT, all_dst_next_dt.data(),
                       counts_next_dt.data(), displacements_next_dt.data(), MPI_INT, world);

        MPI_Allgatherv(my_zoids_comm_idx_next_dt.data(), counts_next_dt[comm->me], MPI_INT, all_comm_idx_next_dt.data(),
                       counts_next_dt.data(), displacements_next_dt.data(), MPI_INT, world);

        for (int i = 0; i < all_src_next_dt.size(); i++) {
            int src = all_src_next_dt[i];
            int dst = all_dst_next_dt[i];
            int comm_idx = all_comm_idx_next_dt[i];
            ZOID_TO_ZOID_TO_VCI_IDX_NEXT_DT[{src, dst}] = comm_idx;
        }

        // determine if comm or no comm needed
        for (int dep = 0; dep < NUM_DEPS; dep++) {
            for (int j = 0; j < my_queues_many_cuts[dep].size(); j++) {
                auto& zoid = my_queues_many_cuts[dep][j];
                int zoid_num = zoid.num;
                auto& recv_neighbors = recv_from_neighbors_many_cuts[zoid_num];
                bool no_comm_needed = true;
                for (int i = 0; i < recv_neighbors.size(); i++) {
                    if (recv_neighbors[i] % comm->nprocs != comm->me) {
                        no_comm_needed = false;
                        break;
                    }
                }

                zoid.no_comm_needed = no_comm_needed;
            }
        }

        for (int dep = 0; dep < NUM_DEPS; dep++) {
            for (int j = 0; j < my_queues_many_cuts_next_dt[dep].size(); j++) {
                auto& zoid = my_queues_many_cuts_next_dt[dep][j];
                auto& recv_neighbors = recv_from_neighbors_many_cuts_next_dt[zoid.num];
                bool no_comm_needed = true;
                for (int i = 0; i < recv_neighbors.size(); i++) {
                    if (recv_neighbors[i] % comm->nprocs != comm->me) {
                        no_comm_needed = false;
                        break;
                    }
                }

                zoid.no_comm_needed = no_comm_needed;
            }
        }

        for (int dep = 1; dep < NUM_DEPS; dep++) {
            int recv_request_idx = 0;
            for (int j = 0; j < my_queues_many_cuts[dep].size(); j++) {
                auto& zoid = my_queues_many_cuts[dep][j];
                auto& recv_neighbors = recv_from_neighbors_many_cuts[zoid.num];
                for (int i = 0; i < recv_neighbors.size(); i++) {
                    if (recv_neighbors[i] % comm->nprocs != comm->me) {
                        recv_request_zoid_to_idx[dep][{recv_neighbors[i], zoid.num}] = recv_request_idx;
                        recv_request_idx_to_zoid[dep][recv_request_idx] = {recv_neighbors[i], zoid.num};
                        recv_request_idx++;
                    }
                }
            }
        }

        for (int dep = 1; dep < NUM_DEPS; dep++) {
            int recv_request_idx = 0;
            for (int j = 0; j < my_queues_many_cuts_next_dt[dep].size(); j++) {
                auto& zoid = my_queues_many_cuts_next_dt[dep][j];
                auto& recv_neighbors = recv_from_neighbors_many_cuts_next_dt[zoid.num];
                for (int i = 0; i < recv_neighbors.size(); i++) {
                    if (recv_neighbors[i] % comm->nprocs != comm->me) {
                        recv_request_zoid_to_idx_next_dt[dep][{recv_neighbors[i], zoid.num}] = recv_request_idx;
                        recv_request_idx_to_zoid_next_dt[dep][recv_request_idx] = {recv_neighbors[i], zoid.num};
                        recv_request_idx++;
                    }
                }
            }
        }
    }

    template <bool curr_dt>
    void INIT_DEP_PROC_RECV_ZOID_DATA() {
        auto& queues = curr_dt ? queues_many_cuts
                : queues_many_cuts_next_dt;

        for (int dep = 1; dep < NUM_DEPS; dep++) {
            if (curr_dt) {
                dep_proc_to_recv_zoids[dep].resize(comm->nprocs);
                dep_proc_recv_zoid_to_find_idxs[dep].resize(comm->nprocs);
            } else {
                dep_proc_to_recv_zoids_next_dt[dep].resize(comm->nprocs);
                dep_proc_recv_zoid_to_find_idxs_next_dt[dep].resize(comm->nprocs);
            }

            for (int proc = 0; proc < comm->nprocs; proc++) {
                if (curr_dt) {
                    dep_proc_recv_zoid_to_find_idxs[dep][proc].resize(NUM_ZOIDS_MANY_CUTS);
                } else {
                    dep_proc_recv_zoid_to_find_idxs_next_dt[dep][proc].resize(NUM_ZOIDS_MANY_CUTS);
                }
            }
        }

        for (int dep = 1; dep < NUM_DEPS; dep++) {
            int send_dep = dep - 1;
            for (int proc = 0; proc < comm->nprocs; proc++) {
                auto& recv_zoids_per_dep = curr_dt ? dep_proc_to_recv_zoids[dep][proc]
                        : dep_proc_to_recv_zoids_next_dt[dep][proc];

                for (int j = 0; j < queues[send_dep].size(); j++) {
                    auto& send_zoid = queues[send_dep][j];
                    int send_zoid_num = send_zoid.num;
                    if (send_zoid.num % comm->nprocs != proc) {
                        continue;
                    }

                    auto& send_neighbors = curr_dt ? send_to_neighbors_many_cuts[send_zoid_num]
                            : send_to_neighbors_many_cuts_next_dt[send_zoid_num];

                    for (int i = 0; i < send_neighbors.size(); i++) {
                        int recv_zoid_num = send_neighbors[i];
                        if (recv_zoid_num % comm->nprocs == comm->me) {
                            if (std::find(recv_zoids_per_dep.begin(), recv_zoids_per_dep.end(),
                                          recv_zoid_num) == recv_zoids_per_dep.end()) {
                                recv_zoids_per_dep.push_back(recv_zoid_num);
                            }

                            auto& find_idxs_per_zoid = curr_dt ? dep_proc_recv_zoid_to_find_idxs[dep][proc][recv_zoid_num]
                                    : dep_proc_recv_zoid_to_find_idxs_next_dt[dep][proc][recv_zoid_num];

                            auto& recv_neighbors = curr_dt ? recv_from_neighbors_many_cuts[recv_zoid_num]
                                    : recv_from_neighbors_many_cuts_next_dt[recv_zoid_num];

                            auto find_it = std::find(recv_neighbors.begin(), recv_neighbors.end(), send_zoid_num);
                            assert(find_it != recv_neighbors.end());
                            int find_idx = std::distance(recv_neighbors.begin(), find_it);

                            if (std::find(find_idxs_per_zoid.begin(), find_idxs_per_zoid.end(),
                                          find_idx) == find_idxs_per_zoid.end()) {
                                find_idxs_per_zoid.push_back(find_idx);
                            }
                        }
                    }
                }
            }
        }
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
                        double zoid_lo[3] = {0};
                        double zoid_hi[3] = {0};
                        double zoid_borders_lo[3] = {0};
                        double zoid_borders_hi[3] = {0};

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

                            zoid_lo[dim] = lo;
                            zoid_hi[dim] = hi;
                            zoid_borders_lo[dim] = lo_borders;
                            zoid_borders_hi[dim] = hi_borders;
                        }

                        if (in_zoid || borders_zoid) {
                            /*
                            if (idx == 255412 || idx == 254682) {
                                std::cout << "FOUND TAG in CONSTRUCTION. tag: " << idx << " zoid: " << zoid.num
                                << " time: " << t
                                << " in zoid? " << in_zoid << " borders zoid? " << borders_zoid
                                << " pos: " << new_pos[0] << " " << new_pos[1] << " " << new_pos[2]
                                << " borders pos: " << new_pos_borders[0] << " " << new_pos_borders[1] << " " << new_pos_borders[2]
                                << " zoid lo: " << zoid_lo[0] << " " << zoid_lo[1] << " " << zoid_lo[2]
                                << " zoid hi: " << zoid_hi[0] << " " << zoid_hi[1] << " " << zoid_hi[2]
                                << " zoid borders lo: " << zoid_borders_lo[0] << " " << zoid_borders_lo[1] << " " << zoid_borders_lo[2]
                                << " zoid borders hi: " << zoid_borders_hi[0] << " " << zoid_borders_hi[1] << " " << zoid_borders_hi[2]
                                << std::endl;
                            }
                            */
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
        std::vector<int> my_neigh_pairs_src;
        std::vector<int> my_neigh_pairs_dst;
        my_neigh_pairs_src.reserve(atom->natoms);
        my_neigh_pairs_dst.reserve(atom->natoms);

        auto* list = force->pair->list;
        for (int ii = 0; ii < list->inum; ii++) {
            int i = list->ilist[ii];
            assert(i == ii);
            int numneigh = list->numneigh[i];
            for (int j = 0; j < numneigh; j++) {
                int neigh = list->firstneigh[i][j];
                // my_neigh_pairs.push_back({atom->tag[i], atom->tag[neigh]});
                my_neigh_pairs_src.push_back(atom->tag[i]);
                my_neigh_pairs_dst.push_back(atom->tag[neigh]);
            }
        }

        std::vector<int> counts(comm->nprocs, 0);
        std::vector<int> displacements(comm->nprocs, 0);

        int my_count = my_neigh_pairs_src.size();
        MPI_Allgather(&my_count, 1, MPI_INT, counts.data(), 1, MPI_INT, world);

        int total_size = 0;
        for (int proc = 0; proc < comm->nprocs; proc++) {
            total_size += counts[proc];
        }

        displacements[0] = 0;
        for (int proc = 1; proc < comm->nprocs; proc++) {
            displacements[proc] = displacements[proc - 1] + counts[proc - 1];
        }

        std::vector<int> all_neigh_pairs_src;
        std::vector<int> all_neigh_pairs_dst;
        all_neigh_pairs_src.resize(total_size);
        all_neigh_pairs_dst.resize(total_size);

        MPI_Allgatherv(my_neigh_pairs_src.data(), counts[comm->me], MPI_INT, all_neigh_pairs_src.data(),
                       counts.data(), displacements.data(), MPI_INT, world);

        MPI_Allgatherv(my_neigh_pairs_dst.data(), counts[comm->me], MPI_INT, all_neigh_pairs_dst.data(),
                       counts.data(), displacements.data(), MPI_INT, world);

        auto* neighbor_lst = new std::vector<int>[atom->natoms + 1];
        for (int i = 0; i < atom->natoms + 1; i++) {
            neighbor_lst[i].reserve(25);
        }

        for (int i = 0; i < all_neigh_pairs_src.size(); i++) {
            int src_tag = all_neigh_pairs_src[i];
            int dst_tag = all_neigh_pairs_dst[i];
            neighbor_lst[src_tag].push_back(dst_tag);
            neighbor_lst[dst_tag].push_back(src_tag);
        }

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

        delete[] neighbor_lst;

        /*
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

        std::vector<int> my_send;
        std::vector<int> all_recv;

        for (int tag = 1; tag < atom->natoms + 1; tag++) {
            memset(counts, 0, comm->nprocs * sizeof(int));
            memset(displacements, 0, comm->nprocs * sizeof(int));

            counts[comm->me] = neighbor_lst[tag].size();
            int send_num = neighbor_lst[tag].size();
            // MPI_Allreduce(MPI_IN_PLACE, counts, comm->nprocs, MPI_INT, MPI_SUM, world);
            MPI_Allgather(&send_num, 1, MPI_INT, counts, 1, MPI_INT, world);
            int total_size = 0;
            for (int i = 0; i < comm->nprocs; i++) {
                total_size += counts[i];
            }

            my_send.resize(counts[comm->me]);
            all_recv.resize(total_size);

            for (int i = 0; i < counts[comm->me]; i++) {
                my_send[i] = neighbor_lst[tag][i];
            }
            displacements[0] = 0;
            for (int i = 1; i < comm->nprocs; i++) {
                displacements[i] = displacements[i - 1] + counts[i - 1];
            }

            MPI_Allgatherv(my_send.data(), counts[comm->me], MPI_INT, all_recv.data(),
                           counts, displacements, MPI_INT, world);

            for (int i = 0; i < total_size; i++) {
                int neigh_tag = all_recv[i];
                neighbor_lst[tag].push_back(neigh_tag);
            }
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

        delete[] neighbor_lst;
        */
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
        std::vector<int> my_bond_pairs_src;
        std::vector<int> my_bond_pairs_dst;
        std::vector<int> my_bond_pairs_type;
        my_bond_pairs_src.reserve(atom->natoms);
        my_bond_pairs_dst.reserve(atom->natoms);
        my_bond_pairs_type.reserve(atom->natoms);

        for (int i = 0; i < neighbor->nbondlist; i++) {
            int i1 = neighbor->bondlist[i][0];
            int i2 = neighbor->bondlist[i][1];
            int type = neighbor->bondlist[i][2];
            my_bond_pairs_src.push_back(atom->tag[i1]);
            my_bond_pairs_dst.push_back(atom->tag[i2]);
            my_bond_pairs_type.push_back(type);
        }

        std::vector<int> counts(comm->nprocs, 0);
        std::vector<int> displacements(comm->nprocs, 0);

        int my_count = my_bond_pairs_src.size();
        MPI_Allgather(&my_count, 1, MPI_INT, counts.data(), 1, MPI_INT, world);

        int total_size = 0;
        for (int proc = 0; proc < comm->nprocs; proc++) {
            total_size += counts[proc];
        }

        displacements[0] = 0;
        for (int proc = 1; proc < comm->nprocs; proc++) {
            displacements[proc] = displacements[proc - 1] + counts[proc - 1];
        }

        std::vector<int> all_bond_pairs_src;
        std::vector<int> all_bond_pairs_dst;
        std::vector<int> all_bond_pairs_type;
        all_bond_pairs_src.resize(total_size);
        all_bond_pairs_dst.resize(total_size);
        all_bond_pairs_type.resize(total_size);

        MPI_Allgatherv(my_bond_pairs_src.data(), counts[comm->me], MPI_INT, all_bond_pairs_src.data(),
                       counts.data(), displacements.data(), MPI_INT, world);
        MPI_Allgatherv(my_bond_pairs_dst.data(), counts[comm->me], MPI_INT, all_bond_pairs_dst.data(),
                       counts.data(), displacements.data(), MPI_INT, world);
        MPI_Allgatherv(my_bond_pairs_type.data(), counts[comm->me], MPI_INT, all_bond_pairs_type.data(),
                       counts.data(), displacements.data(), MPI_INT, world);

        auto* bond_lst = new std::vector<std::pair<int, int>>[atom->natoms + 1];
        for (int i = 0; i < atom->natoms + 1; i++) {
            bond_lst[i].reserve(25);
        }

        for (int i = 0; i < all_bond_pairs_src.size(); i++) {
            int src_tag = all_bond_pairs_src[i];
            int dst_tag = all_bond_pairs_dst[i];
            int type = all_bond_pairs_type[i];

            bond_lst[src_tag].push_back({dst_tag, type});
            bond_lst[dst_tag].push_back({src_tag, type});
        }

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

        delete[] bond_lst;

        /*
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

        std::vector<int> my_send;
        std::vector<int> my_send_type;
        std::vector<int> all_recv;
        std::vector<int> all_recv_type;

        for (int tag = 1; tag < atom->natoms + 1; tag++) {
            memset(counts, 0, comm->nprocs * sizeof(int));
            memset(displacements, 0, comm->nprocs * sizeof(int));

            counts[comm->me] = bond_lst[tag].size();
            int send_num = bond_lst[tag].size();
            // MPI_Allreduce(MPI_IN_PLACE, counts, comm->nprocs, MPI_INT, MPI_SUM, world);
            MPI_Allgather(&send_num, 1, MPI_INT, counts, 1, MPI_INT, world);
            // MPI_Allreduce(MPI_IN_PLACE, counts, comm->nprocs, MPI_INT, MPI_SUM, world);
            int total_size = 0;
            for (int i = 0; i < comm->nprocs; i++) {
                total_size += counts[i];
            }

            my_send.resize(counts[comm->me]);
            my_send_type.resize(counts[comm->me]);
            all_recv.resize(total_size);
            all_recv_type.resize(total_size);

            for (int i = 0; i < counts[comm->me]; i++) {
                my_send[i] = bond_lst[tag][i].first;
                my_send_type[i] = bond_lst[tag][i].second;
            }

            displacements[0] = 0;
            for (int i = 1; i < comm->nprocs; i++) {
                displacements[i] = displacements[i - 1] + counts[i - 1];
            }

            MPI_Allgatherv(my_send.data(), counts[comm->me], MPI_INT, all_recv.data(),
                           counts, displacements, MPI_INT, world);
            MPI_Allgatherv(my_send_type.data(), counts[comm->me], MPI_INT, all_recv_type.data(),
                           counts, displacements, MPI_INT, world);

            for (int i = 0; i < total_size; i++) {
                int neigh_tag = all_recv[i];
                int neigh_type = all_recv_type[i];
                bond_lst[tag].push_back({neigh_tag, neigh_type});
            }
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

        delete[] bond_lst;
        */
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

    // Sending forces is strictly ghost to local. There is no point propagating things.
    template <bool curr_dt>
    void CONSTRUCT_SEND_FORCE_IDXS_ZOID_MANY_CUTS_HELPER(queue_info& zoid) {
        int zoid_num = zoid.num;
        auto& send_neighbors = curr_dt ? send_to_neighbors_many_cuts[zoid_num]
                : send_to_neighbors_many_cuts_next_dt[zoid_num];

        std::map<int, int> idx_to_zoid;
        std::set<int> all_force_neighbors;

        for (int t = 0; t < NUM_TIMESTEPS_IN_PARALLEL + 1; t++) {
            zoid.send_force_idxs_double_buffering[t] = new std::vector<int>[send_neighbors.size()];

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

                auto& pos = zoid.x_stencil_md[t % DOUBLE_BUFFERING][i];
                double atom_pos[3] = {pos.x, pos.y, pos.z};
                double zoid_lo[3] = {0};
                double zoid_hi[3] = {0};

                bool borders_zoid = true;
                for (int dim = 0; dim < domain->dimension; dim++) {
                    double lo = zoid.zoid.cuts[dim].lower + t * zoid.zoid.cuts[dim].slope_lower;
                    double hi = zoid.zoid.cuts[dim].upper + t * zoid.zoid.cuts[dim].slope_upper;
                    double lo_borders = lo - ALLEGRO_SLOPE;
                    double hi_borders = hi + ALLEGRO_SLOPE;
                    borders_zoid = borders_zoid && atom_pos[dim] >= lo_borders && atom_pos[dim] < hi_borders;

                    zoid_lo[dim] = lo;
                    zoid_hi[dim] = hi;
                }

                // have to do this check as for later timesteps this might not be the case
                if (!borders_zoid) {
                    continue;
                }

                // TODO: pick first neighbor that allows us to map a path to get the force where it eventually belongs
                for (int j = 0; j < send_neighbors.size(); j++) {
                    auto send_zoid_num = send_neighbors[j];
                    auto& send_zoid = curr_dt ? zoid_num_to_zoid_many_cuts[send_zoid_num]
                            : zoid_num_to_zoid_many_cuts_next_dt[send_zoid_num];

                    bool in_neighbor_zoid = true;
                    for (int dim = 0; dim < domain->dimension; dim++) {
                        double lo = send_zoid.zoid.cuts[dim].lower + t * send_zoid.zoid.cuts[dim].slope_lower;
                        double hi = send_zoid.zoid.cuts[dim].upper + t * send_zoid.zoid.cuts[dim].slope_upper;
                        double p = atom_pos[dim];
                        while (p < lo) {
                            p += domain->prd[dim];
                        }
                        while (p >= hi) {
                            p -= domain->prd[dim];
                        }

                        in_neighbor_zoid = in_neighbor_zoid && p >= lo && p < hi;
                    }

                    if (in_neighbor_zoid) {
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
        auto& send_neighbors = curr_dt ? send_to_neighbors_many_cuts[zoid_num]
                : send_to_neighbors_many_cuts_next_dt[zoid_num];

        std::map<int, int> idx_to_zoid;

        zoid.send_vel_idxs_double_buffering[0] = new std::vector<int>[send_neighbors.size()];

        for (int t = 1; t < NUM_TIMESTEPS_IN_PARALLEL + 1; t++) {
            zoid.send_vel_idxs_double_buffering[t] = new std::vector<int>[send_neighbors.size()];

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

                // want ghost atoms that were local previously to send to other zoids
                if (is_local || !is_local_prev) {
                    continue;
                }

                auto& pos = zoid.x_stencil_md[0][i];
                double atom_pos[3] = {pos.x, pos.y, pos.z};
                double zoid_lo[3] = {0};
                double zoid_hi[3] = {0};

                bool borders_zoid = true;
                for (int dim = 0; dim < domain->dimension; dim++) {
                    double lo = zoid.zoid.cuts[dim].lower + t * zoid.zoid.cuts[dim].slope_lower;
                    double hi = zoid.zoid.cuts[dim].upper + t * zoid.zoid.cuts[dim].slope_upper;
                    double lo_borders = lo - ALLEGRO_SLOPE;
                    double hi_borders = hi + ALLEGRO_SLOPE;
                    borders_zoid = borders_zoid && atom_pos[dim] >= lo_borders && atom_pos[dim] < hi_borders;

                    zoid_lo[dim] = lo;
                    zoid_hi[dim] = hi;
                }

                // have to do this check as for later timesteps this might not be the case
                if (!borders_zoid) {
                    continue;
                }

                // TODO: pick first neighbor that allows us to map a path to get the force where it eventually belongs
                for (int j = 0; j < send_neighbors.size(); j++) {
                    auto send_zoid_num = send_neighbors[j];
                    auto& send_zoid = curr_dt ? zoid_num_to_zoid_many_cuts[send_zoid_num]
                            : zoid_num_to_zoid_many_cuts_next_dt[send_zoid_num];

                    bool in_neighbor_zoid = true;
                    for (int dim = 0; dim < domain->dimension; dim++) {
                        double lo = send_zoid.zoid.cuts[dim].lower + t * send_zoid.zoid.cuts[dim].slope_lower;
                        double hi = send_zoid.zoid.cuts[dim].upper + t * send_zoid.zoid.cuts[dim].slope_upper;
                        double p = atom_pos[dim];
                        while (p < lo) {
                            p += domain->prd[dim];
                        }
                        while (p >= hi) {
                            p -= domain->prd[dim];
                        }

                        in_neighbor_zoid = in_neighbor_zoid && p >= lo && p < hi;
                    }

                    if (in_neighbor_zoid) {
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

        zoid.send_pos_idxs_double_buffering[0] = new std::vector<int>[send_to_neighbors.size()];

        for (int t = 1; t < NUM_TIMESTEPS_IN_PARALLEL + 1; t++) {
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
                    borders_zoid = borders_zoid && atom_pos[dim] >= lo_borders && atom_pos[dim] < hi_borders;

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

                for (int j = 0; j < send_to_neighbors.size(); j++) {
                    auto send_zoid_num = send_to_neighbors[j];
                    auto& send_zoid = curr_dt ? zoid_num_to_zoid_many_cuts[send_zoid_num] :
                            zoid_num_to_zoid_many_cuts_next_dt[send_zoid_num];

                    bool borders_neighbor_zoid = true;
                    bool in_neighbor_zoid = true;

                    bool correct_out_of_bounds = true;

                    for (int dim = 0; dim < domain->dimension; dim++) {
                        double lo = send_zoid.zoid.cuts[dim].lower + t * send_zoid.zoid.cuts[dim].slope_lower;
                        double hi = send_zoid.zoid.cuts[dim].upper + t * send_zoid.zoid.cuts[dim].slope_upper;
                        double lo_borders = lo - ALLEGRO_SLOPE;
                        double hi_borders = hi + ALLEGRO_SLOPE;

                        double p = atom_pos[dim];
                        while (p < lo_borders) {
                            p += domain->prd[dim];
                        }
                        while (p >= hi_borders) {
                            p -= domain->prd[dim];
                        }

                        borders_neighbor_zoid = borders_neighbor_zoid && p >= lo_borders && p < hi_borders;
                        bool borders_dim = p >= lo_borders && p < hi_borders;

                        while (p < lo) {
                            p += domain->prd[dim];
                        }
                        while (p >= hi) {
                            p -= domain->prd[dim];
                        }
                        in_neighbor_zoid = in_neighbor_zoid && p >= lo && p < hi;
                        bool in_dim = p >= lo && p < hi;

                        if (borders_dim && !in_dim) {
                            if (send_zoid.zoid.cuts[dim].slope_lower < 0) {
                                correct_out_of_bounds = false;
                            }
                        }
                    }

                    if (in_neighbor_zoid) {
                        zoid.send_pos_idxs_double_buffering[t][j].push_back(i);
                    } else if (borders_neighbor_zoid) {
                        if (correct_out_of_bounds) {
                            zoid.send_pos_idxs_double_buffering[t][j].push_back(i);
                        }
                    }

                    /*
                    if (borders_neighbor_zoid) {
                        zoid.send_pos_idxs_double_buffering[t][j].push_back(i);
                        if (t == 1 && zoid.tag_stencil_md[0][i] == 118790) {
                            std::cout << "time: " << t << " zoid: " << zoid.num << " send to: " << send_zoid_num << " is_local: " << is_local
                            << " is local prev: " << is_local_prev << std::endl;
                        }
                        // break;
                    }
                    */
                }
            }
        }

    }

    template <bool curr_dt>
    void CONSTRUCT_SEND_POS_IDXS_ZOID_MANY_CUTS() {
        /*
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
        */

        std::vector<MPI_Request> r;
        r.reserve(NUM_ZOIDS_MANY_CUTS * (NUM_TIMESTEPS_IN_PARALLEL + 1) * 4 / comm->nprocs);

        auto& queues = curr_dt ? queues_many_cuts : queues_many_cuts_next_dt;

        std::vector<int>** zoid_recv_data[NUM_ZOIDS_MANY_CUTS];
        int** zoid_recv_data_sizes[NUM_ZOIDS_MANY_CUTS];

        for (int dep = 0; dep < NUM_DEPS; dep++) {
            for (int j = 0; j < queues[dep].size(); j++) {
                auto& zoid = queues[dep][j];
                if (zoid.num % comm->nprocs != comm->me) {
                    continue;
                }

                zoid_recv_data[zoid.num] = new std::vector<int>*[NUM_TIMESTEPS_IN_PARALLEL + 1];
                zoid_recv_data_sizes[zoid.num] = new int*[NUM_TIMESTEPS_IN_PARALLEL + 1];

                auto& recv_neighbors = curr_dt ? recv_from_neighbors_many_cuts[zoid.num] :
                                          recv_from_neighbors_many_cuts_next_dt[zoid.num];

                for (int t = 1; t < NUM_TIMESTEPS_IN_PARALLEL + 1; t++) {
                    zoid_recv_data[zoid.num][t] = new std::vector<int>[recv_neighbors.size()];
                    zoid_recv_data_sizes[zoid.num][t] = new int[recv_neighbors.size()];

                    for (int i = 0; i < recv_neighbors.size(); i++) {
                        int recv_zoid_num = recv_neighbors[i];

                        auto& recv_pos_idxs = zoid.recv_pos_idxs_double_buffering[t][i];

                        // int mpi_tag = get_mpi_tag(send_zoid_num, zoid.num, t, t);
                        int mpi_tag = get_mpi_tag_many_cuts(recv_zoid_num, zoid.num);

                        zoid_recv_data[zoid.num][t][i].reserve(recv_pos_idxs.size());
                        zoid_recv_data_sizes[zoid.num][t][i] = recv_pos_idxs.size();
                        for (int k = 0; k < recv_pos_idxs.size(); k++) {
                            zoid_recv_data[zoid.num][t][i].push_back(zoid.tag_stencil_md[0][recv_pos_idxs[k]]);
                        }

                        r.emplace_back();
                        MPI_Isend(&zoid_recv_data_sizes[zoid.num][t][i], 1, MPI_INT,
                                  recv_zoid_num % comm->nprocs, mpi_tag, world, &r[r.size() - 1]);

                        if (recv_pos_idxs.size() > 0) {
                            r.emplace_back();
                            // std::vector<int> send_pos_tags;
                            // send_pos_tags.reserve(size);
                            MPI_Isend(zoid_recv_data[zoid.num][t][i].data(), recv_pos_idxs.size(), MPI_INT,
                                      recv_zoid_num % comm->nprocs, mpi_tag, world, &r[r.size() - 1]);
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

                auto& send_neighbors = curr_dt ? send_to_neighbors_many_cuts[zoid.num]
                                                    : send_to_neighbors_many_cuts_next_dt[zoid.num];

                zoid.send_pos_idxs_double_buffering[0] = new std::vector<int>[send_neighbors.size()];

                for (int t = 1; t < NUM_TIMESTEPS_IN_PARALLEL + 1; t++) {
                    zoid.send_pos_idxs_double_buffering[t] = new std::vector<int>[send_neighbors.size()];

                    std::map<int, int> tag_to_zoid;

                    for (int i = 0; i < send_neighbors.size(); i++) {
                        int send_zoid_num = send_neighbors[i];

                        // int mpi_tag = get_mpi_tag(zoid.num, recv_zoid_num, t, t);
                        int mpi_tag = get_mpi_tag_many_cuts(zoid.num, send_zoid_num);

                        int nrecv;
                        MPI_Recv(&nrecv, 1, MPI_INT, send_zoid_num % comm->nprocs,
                                 mpi_tag, world, MPI_STATUS_IGNORE);

                        if (nrecv) {
                            int* recv_buf = new int[nrecv];
                            MPI_Recv(recv_buf, nrecv, MPI_INT, send_zoid_num % comm->nprocs,
                                     mpi_tag, world, MPI_STATUS_IGNORE);
                            zoid.send_pos_idxs_double_buffering[t][i].reserve(nrecv);
                            for (int k = 0; k < nrecv; k++) {
                                zoid.send_pos_idxs_double_buffering[t][i].push_back(tag_to_idx.at(recv_buf[k]));
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

                for (int t = 1; t < NUM_TIMESTEPS_IN_PARALLEL + 1; t++) {
                    delete[] zoid_recv_data[zoid.num][t];
                    delete[] zoid_recv_data_sizes[zoid.num][t];
                }

                delete[] zoid_recv_data[zoid.num];
                delete[] zoid_recv_data_sizes[zoid.num];
            }
        }
    }

    template <bool curr_dt>
    void CONSTRUCT_RECV_POS_IDXS_ZOID_MANY_CUTS_HELPER(queue_info& zoid) {
        int zoid_num = zoid.num;
        auto& recv_neighbors = curr_dt ? recv_from_neighbors_many_cuts[zoid.num]
                : recv_from_neighbors_many_cuts_next_dt[zoid.num];

        zoid.recv_pos_idxs_double_buffering[0] = new std::vector<int>[recv_neighbors.size()];

        for (int t = 1; t < NUM_TIMESTEPS_IN_PARALLEL + 1; t++) {
            zoid.recv_pos_idxs_double_buffering[t] = new std::vector<int>[recv_neighbors.size()];

            auto &local_idxs = zoid.local_idxs_per_timestep[t];
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

                auto &pos = zoid.x_stencil_md[0][i];
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
                    borders_zoid = borders_zoid && atom_pos[dim] >= lo_borders && atom_pos[dim] < hi_borders;

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

                /*
                bool out_of_bounds_expanding = false;
                for (int dim = 0; dim < domain->dimension; dim++) {
                    if (dim_out_of_bounds[dim]) {
                        bool shrinking_dim = (zoid.zoid.cuts[dim].slope_lower > 0);
                        if (!shrinking_dim) {
                            out_of_bounds_expanding = true;
                        }
                    }
                }

                if (out_of_bounds_expanding) {
                    continue;
                }
                */

                // find zoid that had it previously
                for (int j = 0; j < recv_neighbors.size(); j++) {
                    auto recv_zoid_num = recv_neighbors[j];
                    auto& recv_zoid = curr_dt ? zoid_num_to_zoid_many_cuts[recv_zoid_num] :
                                      zoid_num_to_zoid_many_cuts_next_dt[recv_zoid_num];

                    bool in_neighbor_zoid = true;

                    for (int dim = 0; dim < domain->dimension; dim++) {
                        double lo = recv_zoid.zoid.cuts[dim].lower + (t - 1) * recv_zoid.zoid.cuts[dim].slope_lower;
                        double hi = recv_zoid.zoid.cuts[dim].upper + (t - 1) * recv_zoid.zoid.cuts[dim].slope_upper;

                        double p = atom_pos[dim];
                        while (p < lo) {
                            p += domain->prd[dim];
                        }
                        while (p >= hi) {
                            p -= domain->prd[dim];
                        }

                        in_neighbor_zoid = in_neighbor_zoid && p >= lo && p < hi;
                    }

                    if (in_neighbor_zoid) {
                        zoid.recv_pos_idxs_double_buffering[t][j].push_back(i);
                        break;
                    }
                }
            }
        }
    }

    template <bool curr_dt>
    void CONSTRUCT_RECV_POS_IDXS_ZOID_MANY_CUTS() {
        auto& queues = curr_dt ? queues_many_cuts : queues_many_cuts_next_dt;

        cilk_for (int dep = 0; dep < NUM_DEPS; dep++) {
            cilk_for (int j = 0; j < queues[dep].size(); j++) {
                auto& zoid = queues[dep][j];
                if (zoid.num % comm->nprocs != comm->me) {
                    continue;
                }

                CONSTRUCT_RECV_POS_IDXS_ZOID_MANY_CUTS_HELPER<curr_dt>(zoid);
            }
        }

        return;

        /*
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

                        // int mpi_tag = get_mpi_tag(send_zoid_num, zoid.num, t, t);
                        int mpi_tag = get_mpi_tag_many_cuts(send_zoid_num, zoid.num);

                        zoid_send_data[zoid.num][t][i].reserve(send_pos_idxs.size());
                        zoid_send_data_sizes[zoid.num][t][i] = send_pos_idxs.size();
                        for (int k = 0; k < send_pos_idxs.size(); k++) {
                            zoid_send_data[zoid.num][t][i].push_back(zoid.tag_stencil_md[0][send_pos_idxs[k]]);
                        }

                        r.emplace_back();
                        MPI_Isend(&zoid_send_data_sizes[zoid.num][t][i], 1, MPI_INT,
                                  send_zoid_num % comm->nprocs, mpi_tag, world, &r[r.size() - 1]);

                        if (send_pos_idxs.size() > 0) {
                            r.emplace_back();
                            // std::vector<int> send_pos_tags;
                            // send_pos_tags.reserve(size);
                            MPI_Isend(zoid_send_data[zoid.num][t][i].data(), send_pos_idxs.size(), MPI_INT,
                                      send_zoid_num % comm->nprocs, mpi_tag, world, &r[r.size() - 1]);
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

                    std::map<int, int> tag_to_zoid;

                    for (int i = 0; i < recv_from_neighbors.size(); i++) {
                        int recv_zoid_num = recv_from_neighbors[i];

                        // int mpi_tag = get_mpi_tag(zoid.num, recv_zoid_num, t, t);
                        int mpi_tag = get_mpi_tag_many_cuts(zoid.num, recv_zoid_num);

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

                    // find duplicates
                    std::map<int, int> idx_to_count;
                    std::map<int, int> idx_to_zoid;
                    for (int i = 0; i < recv_from_neighbors.size(); i++) {
                        for (auto& idx : zoid.recv_pos_idxs_double_buffering[t][i]) {
                            idx_to_count[idx]++;
                            idx_to_zoid[idx] = recv_from_neighbors[i];
                        }
                    }

                    auto& local_idxs = zoid.local_idxs_per_timestep[t];
                    std::set<int> local_idxs_set;
                    local_idxs_set.insert(local_idxs.begin(), local_idxs.end());

                    for (auto& [k, v] : idx_to_count) {
                        if (t == 1 && v > 1 && dep > 1 && curr_dt) {
                            auto& p = zoid.x_stencil_md[0][k];
                            double zoid_lo[3] = {0};
                            double zoid_hi[3] = {0};
                            for (int dim = 0; dim < domain->dimension; dim++) {
                                zoid_lo[dim] = zoid.zoid.cuts[dim].lower + t * zoid.zoid.cuts[dim].slope_lower;
                                zoid_hi[dim] = zoid.zoid.cuts[dim].upper + t * zoid.zoid.cuts[dim].slope_upper;
                            }
                            if (local_idxs_set.find(k) == local_idxs_set.end()) {
                                auto& recv_zoid = zoid_num_to_zoid_many_cuts[idx_to_zoid[k]];
                                std::cout << BOLDGREEN << "time: " << t << " zoid: " << zoid.num << " recv pos no duplicate idx: "
                                          << k << " num instances: " << v << idx_to_zoid[k]
                                          << " pos: " << p.x << " " << p.y << " " << p.z
                                          << " lo: " << zoid_lo[0] << " " << zoid_lo[1] << " " << zoid_lo[2]
                                          << " hi: " << zoid_hi[0] << " " << zoid_hi[1] << " " << zoid_hi[2]
                                          << " where: " << zoid.where[0] << " " << zoid.where[1] << " " << zoid.where[2]
                                          << " recv zoid where: " << recv_zoid.where[0] << " " << recv_zoid.where[1] << " " << recv_zoid.where[2]
                                          << " find? " << (local_idxs_set.find(k) != local_idxs_set.end())
                                          << RESET_COLOR << std::endl;
                            }
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
        */
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

                        // int mpi_tag = get_mpi_tag(send_zoid_num, zoid.num, t, t);
                        int mpi_tag = get_mpi_tag_many_cuts(send_zoid_num, zoid.num);

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

                        // int mpi_tag = get_mpi_tag(zoid.num, recv_zoid_num, t, t);
                        int mpi_tag = get_mpi_tag_many_cuts(zoid.num, recv_zoid_num);

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

                        // int mpi_tag = get_mpi_tag(send_zoid_num, zoid.num, t, t);
                        int mpi_tag = get_mpi_tag_many_cuts(send_zoid_num, zoid.num);

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

                        // int mpi_tag = get_mpi_tag(zoid.num, recv_zoid_num, t, t);
                        int mpi_tag = get_mpi_tag_many_cuts(zoid.num, recv_zoid_num);

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
    void GET_SEND_STATISTICS() {
        auto& queues = curr_dt ? queues_many_cuts : queues_many_cuts_next_dt;
        for (int dep = 0; dep < NUM_DEPS; dep++) {
            for (int j = 0; j < queues[dep].size(); j++) {
                auto& zoid = queues[dep][j];
                int zoid_num = zoid.num;
                if (zoid_num % comm->nprocs != comm->me) {
                    continue;
                }
                int my_zoid_dep = (zoid.where[0] % 2 == 0) + (zoid.where[1] % 2 == 0) + (zoid.where[2] % 2 == 0);
                auto& send_neighbors = curr_dt ? send_to_neighbors_many_cuts[zoid_num] : send_to_neighbors_many_cuts_next_dt[zoid_num];

                for (int i = 0; i < send_neighbors.size(); i++) {
                    int send_zoid_num = send_neighbors[i];
                    auto& send_zoid = curr_dt ? zoid_num_to_zoid_many_cuts[send_zoid_num]
                            : zoid_num_to_zoid_many_cuts_next_dt[send_zoid_num];
                    int send_zoid_dep = (send_zoid.where[0] % 2 == 0) + (send_zoid.where[1] % 2 == 0) + (send_zoid.where[2] % 2 == 0);
                    if (send_zoid_num % comm->nprocs != comm->me) {
                        int total_send_force = zoid.send_force_idxs_double_buffering_flattened[i].size();
                        int total_send_pos = zoid.send_pos_idxs_double_buffering_flattened[0][i].size() + zoid.send_pos_idxs_double_buffering_flattened[1][i].size();
                        int total_send_vel = zoid.send_vel_idxs_double_buffering_flattened[i].size();
                        std::stringstream s1;
                        s1 << "zoid: " << zoid.num << " send to: " << send_zoid_num << " dep: " << dep << " send to: " << send_zoid_num
                        << " total send force: " << total_send_force * 3 << " total send pos: " << total_send_pos << " total send vel: " << total_send_vel
                        << " all in all total: " << (total_send_force + total_send_pos + total_send_vel) * 3
                        << std::endl;
                        std::cout << s1.str();
                    }
                }
            }
        }
    }

    template <bool curr_dt>
    void CONSTRUCT_RECV_PROC_OFFSETS() {
        auto& queues = curr_dt ? queues_many_cuts : queues_many_cuts_next_dt;

        for (int zoid_num = 0; zoid_num < NUM_ZOIDS_MANY_CUTS; zoid_num++) {
            if (zoid_num % comm->nprocs != comm->me) {
                continue;
            }

            if (curr_dt) {
                int num_recv_neighbors = recv_from_neighbors_many_cuts[zoid_num].size();
                if (num_recv_neighbors > 0) {
                    recv_proc_zoid_offsets[zoid_num] = new std::vector<int>[num_recv_neighbors];
                    recv_proc_zoid_sizes[zoid_num] = new std::vector<int>[num_recv_neighbors];
                    for (int i = 0; i < num_recv_neighbors; i++) {
                        recv_proc_zoid_offsets[zoid_num][i].reserve(25);
                        recv_proc_zoid_sizes[zoid_num][i].reserve(25);
                    }
                }
            } else {
                int num_recv_neighbors = recv_from_neighbors_many_cuts_next_dt[zoid_num].size();
                if (num_recv_neighbors > 0) {
                    recv_proc_zoid_offsets_next_dt[zoid_num] = new std::vector<int>[num_recv_neighbors];
                    recv_proc_zoid_sizes_next_dt[zoid_num] = new std::vector<int>[num_recv_neighbors];
                    for (int i = 0; i < num_recv_neighbors; i++) {
                        recv_proc_zoid_offsets_next_dt[zoid_num][i].reserve(25);
                        recv_proc_zoid_sizes_next_dt[zoid_num][i].reserve(25);
                    }
                }
            }
        }

        for (int proc = 0; proc < comm->nprocs; proc++) {
            for (int send_dep = 0; send_dep < NUM_DEPS - 1; send_dep++) {
                int dep = send_dep + 1;

                int total_nrecv_from_proc = 0;

                int offset = 0;
                for (int j = 0; j < queues[send_dep].size(); j++) {
                    auto &send_zoid = queues[send_dep][j];
                    int send_zoid_num = send_zoid.num;
                    if (send_zoid_num % comm->nprocs != proc) {
                        continue;
                    }

                    auto& send_zoid_neighbors = curr_dt ? send_to_neighbors_many_cuts[send_zoid_num]
                                                        : send_to_neighbors_many_cuts_next_dt[send_zoid_num];

                    for (int neigh : send_zoid_neighbors) {
                        if (neigh % comm->nprocs != comm->me) {
                            continue;
                        }

                        auto& my_zoid = curr_dt ? zoid_num_to_zoid_many_cuts[neigh]
                                                : zoid_num_to_zoid_many_cuts_next_dt[neigh];
                        auto& recv_neighbors = curr_dt ? recv_from_neighbors_many_cuts[my_zoid.num]
                                                       : recv_from_neighbors_many_cuts_next_dt[my_zoid.num];

                        auto find_it = std::find(recv_neighbors.begin(), recv_neighbors.end(), send_zoid_num);
                        assert(find_it != recv_neighbors.end());
                        int find_idx = std::distance(recv_neighbors.begin(), find_it);
                        int nrecv_force = 0;
                        int nrecv_pos = 0;
                        int nrecv_vel = 0;
                        for (int t = 0; t < NUM_TIMESTEPS_IN_PARALLEL + 1; t++) {
                            nrecv_force += my_zoid.recv_force_idxs_double_buffering[t][find_idx].size();
                            nrecv_pos += my_zoid.recv_pos_idxs_double_buffering[t][find_idx].size();
                            nrecv_vel += my_zoid.recv_vel_idxs_double_buffering[t][find_idx].size();
                        }

                        int nrecv_from_send_zoid = nrecv_force + nrecv_pos + nrecv_vel;
                        if (curr_dt) {
                            recv_proc_zoid_offsets[my_zoid.num][find_idx].push_back(offset);
                            recv_proc_zoid_sizes[my_zoid.num][find_idx].push_back(nrecv_from_send_zoid);
                        } else {
                            recv_proc_zoid_offsets_next_dt[my_zoid.num][find_idx].push_back(offset);
                            recv_proc_zoid_sizes_next_dt[my_zoid.num][find_idx].push_back(nrecv_from_send_zoid);
                        }

                        offset += nrecv_from_send_zoid;
                        total_nrecv_from_proc += nrecv_from_send_zoid;
                    }
                }

                int total_doubles_recv_from_proc = DEBUG_SEND_RECV_DATA ? total_nrecv_from_proc * (3 + 1) : total_nrecv_from_proc * 3;
                if (total_doubles_recv_from_proc > nrecv_buf_many_cuts[dep][proc]) {
                    auto before = nrecv_buf_many_cuts[dep][proc];
                    GROW_RECV_MANY_CUTS(dep, proc, total_doubles_recv_from_proc);
                }
            }
        }
    }

    static constexpr int NUM_STREAMS = 24;
    // 64 VCIs so 1 per comm
    static constexpr int NUM_COMMS = 64;
    std::vector<MPI_Comm> all_comms;
    MPIX_Stream all_streams[NUM_STREAMS];
    MPI_Comm stream_comm;

    void INIT_SEND_RECV_BUFFERS_MANY_CUTS() {
        /*
        for (int i = 0; i < NUM_STREAMS; i++) {
            MPIX_Stream_create(MPI_INFO_NULL, &all_streams[i]);
        }

        auto res = MPIX_Stream_comm_create_multiplex(world, NUM_STREAMS, all_streams, &stream_comm);
        assert(res == MPI_SUCCESS);
        */

        constexpr int INITIAL_SIZE = 1024;

        all_comms.resize(NUM_COMMS);
        for (int i = 0; i < NUM_COMMS; i++) {
            MPI_Comm_dup(world, &all_comms[i]);
            MPI_Info comm_info;
            MPI_Info_create(&comm_info);
            MPI_Info_set(comm_info, "mpi_assert_no_any_source", "true");
            MPI_Info_set(comm_info, "mpi_assert_no_any_tag", "true");
            MPI_Info_set(comm_info, "vci", std::to_string(i).c_str());
            MPI_Comm_set_info(all_comms[i], comm_info);
            MPI_Info_free(&comm_info);
        }

        for (int dep = 0; dep < NUM_DEPS; dep++) {
            nsend_buf_many_cuts[dep] = new int[comm->nprocs];
            nrecv_buf_many_cuts[dep] = new int[comm->nprocs];

            buf_recv_many_cuts[dep] = new double*[comm->nprocs];
            buf_send_many_cuts[dep] = new double*[comm->nprocs];


            for (int proc = 0; proc < comm->nprocs; proc++) {
                buf_send_many_cuts[dep][proc] = new double[INITIAL_SIZE];
                buf_recv_many_cuts[dep][proc] = new double[INITIAL_SIZE];
                nsend_buf_many_cuts[dep][proc] = INITIAL_SIZE;
                nrecv_buf_many_cuts[dep][proc] = INITIAL_SIZE;
            }
        }

        for (int p = 0; p < NUM_PIPELINE_STAGES; p++) {
            buf_send_zoid_to_zoid[p] = new double**[NUM_ZOIDS_MANY_CUTS];
            nsend_buf_send_zoid_to_zoid[p] = new int*[NUM_ZOIDS_MANY_CUTS];
            buf_recv_zoid_to_zoid[p] = new double**[NUM_ZOIDS_MANY_CUTS];
            nrecv_buf_recv_zoid_to_zoid[p] = new int*[NUM_ZOIDS_MANY_CUTS];

            for (int zoid_num = 0; zoid_num < NUM_ZOIDS_MANY_CUTS; zoid_num++) {
                if (zoid_num % comm->nprocs != comm->me) {
                    continue;
                }

                buf_send_zoid_to_zoid[p][zoid_num] = new double*[MAX_NEIGHBORS];
                nsend_buf_send_zoid_to_zoid[p][zoid_num] = new int[MAX_NEIGHBORS];
                buf_recv_zoid_to_zoid[p][zoid_num] = new double*[MAX_NEIGHBORS];
                nrecv_buf_recv_zoid_to_zoid[p][zoid_num] = new int[MAX_NEIGHBORS];

                for (int i = 0; i < MAX_NEIGHBORS; i++) {
                    buf_send_zoid_to_zoid[p][zoid_num][i] = new double[INITIAL_SIZE];
                    buf_recv_zoid_to_zoid[p][zoid_num][i] = new double[INITIAL_SIZE];
                    nsend_buf_send_zoid_to_zoid[p][zoid_num][i] = INITIAL_SIZE;
                    nrecv_buf_recv_zoid_to_zoid[p][zoid_num][i] = INITIAL_SIZE;
                }
            }
        }
    }

    void GROW_SEND_MANY_CUTS(int dep, int proc, int size) {
        constexpr double FACTOR = 1.5;
        assert(size > nsend_buf_many_cuts[dep][proc]);

        delete[] buf_send_many_cuts[dep][proc];
        nsend_buf_many_cuts[dep][proc] = 0;

        int new_size = static_cast<int>(size * FACTOR);
        buf_send_many_cuts[dep][proc] = new double[new_size];
        nsend_buf_many_cuts[dep][proc] = static_cast<int>(new_size);
    }

    void GROW_RECV_ZOID_TO_ZOID_MANY_CUTS(int zoid_num, int i, int size, int pipeline_stage) {
        constexpr double FACTOR = 1.5;
        int new_size = static_cast<int>(size * FACTOR);

        assert(size > nrecv_buf_recv_zoid_to_zoid[pipeline_stage][zoid_num][i]);

        if (nrecv_buf_recv_zoid_to_zoid[pipeline_stage][zoid_num][i] > 0) {
            delete[] buf_recv_zoid_to_zoid[pipeline_stage][zoid_num][i];
        }

        buf_recv_zoid_to_zoid[pipeline_stage][zoid_num][i] = new double[new_size];
        nrecv_buf_recv_zoid_to_zoid[pipeline_stage][zoid_num][i] = static_cast<int>(new_size);
    }

    void GROW_SEND_ZOID_TO_ZOID_MANY_CUTS(int zoid_num, int i, int size, int pipeline_stage) {
        constexpr double FACTOR = 1.5;
        int new_size = static_cast<int>(size * FACTOR);
        assert(size > nsend_buf_send_zoid_to_zoid[pipeline_stage][zoid_num][i]);

        if (nsend_buf_send_zoid_to_zoid[pipeline_stage][zoid_num][i] > 0) {
            delete[] buf_send_zoid_to_zoid[pipeline_stage][zoid_num][i];
        }

        buf_send_zoid_to_zoid[pipeline_stage][zoid_num][i] = new double[new_size];
        nsend_buf_send_zoid_to_zoid[pipeline_stage][zoid_num][i] = static_cast<int>(new_size);
    }

    void GROW_RECV_MANY_CUTS(int dep, int proc, int size) {
        constexpr double FACTOR = 1.5;
        assert(size > nrecv_buf_many_cuts[dep][proc]);

        delete[] buf_recv_many_cuts[dep][proc];
        nrecv_buf_many_cuts[dep][proc] = 0;

        int new_size = static_cast<int>(size * FACTOR);

        buf_recv_many_cuts[dep][proc] = new double[new_size];
        nrecv_buf_many_cuts[dep][proc] = static_cast<int>(new_size);
    }

    template <bool curr_dt>
    void CONSTRUCT_SEND_PROC_OFFSETS() {
        auto& queues = curr_dt ? queues_many_cuts : queues_many_cuts_next_dt;

        if (curr_dt) {
            send_proc_zoid_offsets.resize(NUM_ZOIDS_MANY_CUTS);
            send_proc_zoid_sizes.resize(NUM_ZOIDS_MANY_CUTS);
        } else {
            send_proc_zoid_offsets_next_dt.resize(NUM_ZOIDS_MANY_CUTS);
            send_proc_zoid_sizes_next_dt.resize(NUM_ZOIDS_MANY_CUTS);
        }

        for (int zoid_num = 0; zoid_num < NUM_ZOIDS_MANY_CUTS; zoid_num++) {
            if (zoid_num % comm->nprocs != comm->me) {
                continue;
            }

            if (curr_dt) {
                send_proc_zoid_offsets[zoid_num].resize(comm->nprocs);
                send_proc_zoid_sizes[zoid_num].resize(comm->nprocs);
                for (int proc = 0; proc < comm->nprocs; proc++) {
                    send_proc_zoid_sizes[zoid_num][proc] = 0;
                }
            } else {
                send_proc_zoid_offsets_next_dt[zoid_num].resize(comm->nprocs);
                send_proc_zoid_sizes_next_dt[zoid_num].resize(comm->nprocs);
                for (int proc = 0; proc < comm->nprocs; proc++) {
                    send_proc_zoid_sizes_next_dt[zoid_num][proc] = 0;
                }
            }
        }

        for (int proc = 0; proc < comm->nprocs; proc++) {
            for (int send_dep = 0; send_dep < NUM_DEPS - 1; send_dep++) {
                int total_nsend_to_proc = 0;
                int buf_offset = 0;

                for (int j = 0; j < queues[send_dep].size(); j++) {
                    auto &send_zoid = queues[send_dep][j];
                    int send_zoid_num = send_zoid.num;
                    if (send_zoid_num % comm->nprocs != comm->me) {
                        continue;
                    }

                    auto& send_neighbors = curr_dt ? send_to_neighbors_many_cuts[send_zoid_num]
                                                        : send_to_neighbors_many_cuts_next_dt[send_zoid_num];

                    int zoid_nsend_to_proc = 0;
                    int before = buf_offset;

                    for (int i = 0; i < send_neighbors.size(); i++) {
                        int neigh = send_neighbors[i];
                        if (neigh % comm->nprocs != proc) {
                            continue;
                        }

                        int nsend_force = 0;
                        int nsend_pos = 0;
                        int nsend_vel = 0;
                        for (int t = 0; t < NUM_TIMESTEPS_IN_PARALLEL + 1; t++) {
                            nsend_force += send_zoid.send_force_idxs_double_buffering[t][i].size();
                            nsend_pos += send_zoid.send_pos_idxs_double_buffering[t][i].size();
                            nsend_vel += send_zoid.send_vel_idxs_double_buffering[t][i].size();
                        }

                        int nsend_total = nsend_force + nsend_pos + nsend_vel;
                        zoid_nsend_to_proc += nsend_total;

                        buf_offset += nsend_total;
                        total_nsend_to_proc += zoid_nsend_to_proc;
                    }

                    assert(buf_offset - before == zoid_nsend_to_proc);
                    if (curr_dt) {
                        send_proc_zoid_offsets[send_zoid.num][proc] = buf_offset - zoid_nsend_to_proc;
                        send_proc_zoid_sizes[send_zoid.num][proc] = zoid_nsend_to_proc;
                    } else {
                        send_proc_zoid_offsets_next_dt[send_zoid.num][proc] = buf_offset - zoid_nsend_to_proc;
                        send_proc_zoid_sizes_next_dt[send_zoid.num][proc] = zoid_nsend_to_proc;
                    }
                }

                int total_doubles_send_to_proc = DEBUG_SEND_RECV_DATA ? total_nsend_to_proc * (3 + 1) : total_nsend_to_proc * 3;
                if (total_doubles_send_to_proc > nsend_buf_many_cuts[send_dep][proc]) {
                    GROW_SEND_MANY_CUTS(send_dep, proc, total_doubles_send_to_proc);
                }
            }
        }
    }

    template <bool curr_dt>
    void CONSTRUCT_SEND_ZOID_TO_ZOID_SIZES() {
        auto& queues = curr_dt ? queues_many_cuts : queues_many_cuts_next_dt;

        constexpr int start_t = 1;
        constexpr int end_t = NUM_TIMESTEPS_IN_PARALLEL + 1;

        constexpr int DEFAULT_PIPELINE_STAGE = 0;

        if (curr_dt) {
            send_zoid_to_zoid_sizes.resize(NUM_ZOIDS_MANY_CUTS);

            send_zoid_to_zoid_sizes_setup.resize(NUM_ZOIDS_MANY_CUTS);
        } else {
            send_zoid_to_zoid_sizes_next_dt.resize(NUM_ZOIDS_MANY_CUTS);
        }

        for (int zoid_num = 0; zoid_num < NUM_ZOIDS_MANY_CUTS; zoid_num++) {
            if (zoid_num % comm->nprocs != comm->me) {
                continue;
            }

            int num_send_neighbors = curr_dt ? send_to_neighbors_many_cuts[zoid_num].size()
                    : send_to_neighbors_many_cuts_next_dt[zoid_num].size();

            if (num_send_neighbors > 0) {
                if (curr_dt) {
                    send_zoid_to_zoid_sizes[zoid_num].resize(num_send_neighbors);
                    std::fill(send_zoid_to_zoid_sizes[zoid_num].begin(), send_zoid_to_zoid_sizes[zoid_num].end(), 0);

                    send_zoid_to_zoid_sizes_setup[zoid_num].resize(num_send_neighbors);
                    std::fill(send_zoid_to_zoid_sizes_setup[zoid_num].begin(), send_zoid_to_zoid_sizes_setup[zoid_num].end(), 0);
                } else {
                    send_zoid_to_zoid_sizes_next_dt[zoid_num].resize(num_send_neighbors);
                    std::fill(send_zoid_to_zoid_sizes_next_dt[zoid_num].begin(), send_zoid_to_zoid_sizes_next_dt[zoid_num].end(), 0);
                }
            }
        }

        for (int send_dep = 0; send_dep < NUM_DEPS - 1; send_dep++) {
            for (int j = 0; j < queues[send_dep].size(); j++) {
                auto &send_zoid = queues[send_dep][j];
                int send_zoid_num = send_zoid.num;
                if (send_zoid_num % comm->nprocs != comm->me) {
                    continue;
                }

                auto& send_neighbors = curr_dt ? send_to_neighbors_many_cuts[send_zoid_num]
                        : send_to_neighbors_many_cuts_next_dt[send_zoid_num];

                for (int i = 0; i < send_neighbors.size(); i++) {
                    int neigh = send_neighbors[i];
                    int nsend_force = 0;
                    int nsend_pos = 0;
                    int nsend_vel = 0;

                    for (int t = start_t; t < end_t; t++) {
                        nsend_force += send_zoid.send_force_idxs_double_buffering[t][i].size();
                        nsend_pos += send_zoid.send_pos_idxs_double_buffering[t][i].size();
                        nsend_vel += send_zoid.send_vel_idxs_double_buffering[t][i].size();
                    }

                    for (int t = start_t; t < end_t; t++) {
                        for (auto& idx : send_zoid.send_force_idxs_double_buffering[t][i]) {
                            send_zoid.send_force_idxs_double_buffering_flattened[i].push_back(idx);
                        }

                        for (auto& idx : send_zoid.send_vel_idxs_double_buffering[t][i]) {
                            send_zoid.send_vel_idxs_double_buffering_flattened[i].push_back(idx);
                        }

                        for (auto& idx : send_zoid.send_pos_idxs_double_buffering[t][i]) {
                            send_zoid.send_pos_idxs_double_buffering_flattened[t % DOUBLE_BUFFERING][i].push_back(idx);
                        }
                    }

                    int nsend_total = nsend_force + nsend_pos + nsend_vel;

                    if (curr_dt) {
                        send_zoid_to_zoid_sizes[send_zoid.num][i] = nsend_total;

                        int setup_nsend_force = send_zoid.send_force_idxs_double_buffering[0][i].size();
                        send_zoid_to_zoid_sizes_setup[send_zoid.num][i] = setup_nsend_force;
                    } else {
                        send_zoid_to_zoid_sizes_next_dt[send_zoid.num][i] = nsend_total;
                    }

                    int total_doubles_send_to_zoid = DEBUG_SEND_RECV_DATA ? nsend_total * (3 + 1) : nsend_total * 3;
                    if (total_doubles_send_to_zoid > nsend_buf_send_zoid_to_zoid[DEFAULT_PIPELINE_STAGE][send_zoid_num][i]) {
                        GROW_SEND_ZOID_TO_ZOID_MANY_CUTS(send_zoid.num, i, total_doubles_send_to_zoid, DEFAULT_PIPELINE_STAGE);
                    }
                }
            }
        }
    }

    template <bool curr_dt>
    void CONSTRUCT_SEND_ZOID_TO_ZOID_SIZES_PIPELINED() {
        auto& queues = curr_dt ? queues_many_cuts : queues_many_cuts_next_dt;

        int start_t[NUM_PIPELINE_STAGES] = {1, NUM_TIMESTEPS_IN_PARALLEL / 2 + 1};
        int end_t[NUM_PIPELINE_STAGES] = {NUM_TIMESTEPS_IN_PARALLEL / 2 + 1, NUM_TIMESTEPS_IN_PARALLEL + 1};

        for (int p = 0; p < NUM_PIPELINE_STAGES; p++) {
            if (curr_dt) {
                send_zoid_to_zoid_sizes_pipelined[p].resize(NUM_ZOIDS_MANY_CUTS);
            } else {
                send_zoid_to_zoid_sizes_pipelined_next_dt[p].resize(NUM_ZOIDS_MANY_CUTS);
            }

            for (int zoid_num = 0; zoid_num < NUM_ZOIDS_MANY_CUTS; zoid_num++) {
                if (zoid_num % comm->nprocs != comm->me) {
                    continue;
                }

                int num_send_neighbors = curr_dt ? send_to_neighbors_many_cuts[zoid_num].size()
                                                 : send_to_neighbors_many_cuts_next_dt[zoid_num].size();

                if (num_send_neighbors > 0) {
                    if (curr_dt) {
                        send_zoid_to_zoid_sizes_pipelined[p][zoid_num].resize(num_send_neighbors);
                        std::fill(send_zoid_to_zoid_sizes_pipelined[p][zoid_num].begin(),
                                  send_zoid_to_zoid_sizes_pipelined[p][zoid_num].end(), 0);
                    } else {
                        send_zoid_to_zoid_sizes_pipelined_next_dt[p][zoid_num].resize(num_send_neighbors);
                        std::fill(send_zoid_to_zoid_sizes_pipelined_next_dt[p][zoid_num].begin(),
                                  send_zoid_to_zoid_sizes_pipelined_next_dt[p][zoid_num].end(), 0);
                    }
                }
            }
        }

        for (int p = 0; p < NUM_PIPELINE_STAGES; p++) {
            for (int send_dep = 0; send_dep < NUM_DEPS - 1; send_dep++) {
                for (int j = 0; j < queues[send_dep].size(); j++) {
                    auto &send_zoid = queues[send_dep][j];
                    int send_zoid_num = send_zoid.num;
                    if (send_zoid_num % comm->nprocs != comm->me) {
                        continue;
                    }

                    auto& send_neighbors = curr_dt ? send_to_neighbors_many_cuts[send_zoid_num]
                                                   : send_to_neighbors_many_cuts_next_dt[send_zoid_num];

                    for (int i = 0; i < send_neighbors.size(); i++) {
                        int neigh = send_neighbors[i];
                        int nsend_force = 0;
                        int nsend_pos = 0;
                        int nsend_vel = 0;

                        for (int t = start_t[p]; t < end_t[p]; t++) {
                            nsend_force += send_zoid.send_force_idxs_double_buffering[t][i].size();
                            nsend_pos += send_zoid.send_pos_idxs_double_buffering[t][i].size();
                            nsend_vel += send_zoid.send_vel_idxs_double_buffering[t][i].size();
                        }

                        int nsend_total = nsend_force + nsend_pos + nsend_vel;

                        if (curr_dt) {
                            send_zoid_to_zoid_sizes_pipelined[p][send_zoid.num][i] = nsend_total;
                        } else {
                            send_zoid_to_zoid_sizes_pipelined_next_dt[p][send_zoid.num][i] = nsend_total;
                        }

                        int total_doubles_send_to_zoid = DEBUG_SEND_RECV_DATA ? nsend_total * (3 + 1) : nsend_total * 3;
                        if (total_doubles_send_to_zoid > nsend_buf_send_zoid_to_zoid[p][send_zoid_num][i]) {
                            GROW_SEND_ZOID_TO_ZOID_MANY_CUTS(send_zoid.num, i, total_doubles_send_to_zoid, p);
                        }
                    }
                }
            }
        }
    }

    template <bool curr_dt>
    void CONSTRUCT_RECV_ZOID_TO_ZOID_SIZES() {
        auto& queues = curr_dt ? queues_many_cuts : queues_many_cuts_next_dt;

        constexpr int start_t = 1;
        constexpr int end_t = NUM_TIMESTEPS_IN_PARALLEL + 1;

        constexpr int DEFAULT_PIPELINE_STAGE = 0;

        if (curr_dt) {
            recv_zoid_to_zoid_sizes.resize(NUM_ZOIDS_MANY_CUTS);

            recv_zoid_to_zoid_sizes_setup.resize(NUM_ZOIDS_MANY_CUTS);
        } else {
            recv_zoid_to_zoid_sizes_next_dt.resize(NUM_ZOIDS_MANY_CUTS);
        }

        for (int zoid_num = 0; zoid_num < NUM_ZOIDS_MANY_CUTS; zoid_num++) {
            if (zoid_num % comm->nprocs != comm->me) {
                continue;
            }

            auto& zoid = curr_dt ? zoid_num_to_zoid_many_cuts[zoid_num]
                    : zoid_num_to_zoid_many_cuts_next_dt[zoid_num];

            auto& recv_neighbors = curr_dt ? recv_from_neighbors_many_cuts[zoid_num]
                    : recv_from_neighbors_many_cuts_next_dt[zoid_num];

            if (curr_dt) {
                recv_zoid_to_zoid_sizes[zoid_num].resize(recv_neighbors.size());
                recv_zoid_to_zoid_sizes_setup[zoid_num].resize(recv_neighbors.size());
            } else {
                recv_zoid_to_zoid_sizes_next_dt[zoid_num].resize(recv_neighbors.size());
            }

            for (int i = 0; i < recv_neighbors.size(); i++) {
                int recv_zoid_num = recv_neighbors[i];

                int nrecv_from_zoid = 0;
                for (int t = start_t; t < end_t; t++) {
                    nrecv_from_zoid += zoid.recv_force_idxs_double_buffering[t][i].size();
                    nrecv_from_zoid += zoid.recv_pos_idxs_double_buffering[t][i].size();
                    nrecv_from_zoid += zoid.recv_vel_idxs_double_buffering[t][i].size();
                }

                for (int t = start_t; t < end_t; t++) {
                    for (auto& idx : zoid.recv_force_idxs_double_buffering[t][i]) {
                        zoid.recv_force_idxs_double_buffering_flattened[i].push_back(idx);
                    }

                    for (auto& idx : zoid.recv_vel_idxs_double_buffering[t][i]) {
                        zoid.recv_vel_idxs_double_buffering_flattened[i].push_back(idx);
                    }

                    for (auto& idx : zoid.recv_pos_idxs_double_buffering[t][i]) {
                        zoid.recv_pos_idxs_double_buffering_flattened[t % DOUBLE_BUFFERING][i].push_back(idx);
                    }
                }

                if (curr_dt) {
                    recv_zoid_to_zoid_sizes[zoid_num][i] = nrecv_from_zoid;

                    int setup_nrecv_force = zoid.recv_force_idxs_double_buffering[0][i].size();
                    recv_zoid_to_zoid_sizes_setup[zoid_num][i] = setup_nrecv_force;
                } else {
                    recv_zoid_to_zoid_sizes_next_dt[zoid_num][i] = nrecv_from_zoid;
                }

                int total_doubles_recv_from_zoid = DEBUG_SEND_RECV_DATA ? nrecv_from_zoid * (3 + 1) : nrecv_from_zoid * 3;
                if (total_doubles_recv_from_zoid > nrecv_buf_recv_zoid_to_zoid[DEFAULT_PIPELINE_STAGE][zoid_num][i]) {
                    GROW_RECV_ZOID_TO_ZOID_MANY_CUTS(zoid_num, i, total_doubles_recv_from_zoid, DEFAULT_PIPELINE_STAGE);
                }
            }
        }
    }

    template <bool curr_dt>
    void CONSTRUCT_RECV_ZOID_TO_ZOID_SIZES_PIPELINED() {
        auto& queues = curr_dt ? queues_many_cuts : queues_many_cuts_next_dt;

        int start_t[NUM_PIPELINE_STAGES] = {1, NUM_TIMESTEPS_IN_PARALLEL / 2 + 1};
        int end_t[NUM_PIPELINE_STAGES] = {NUM_TIMESTEPS_IN_PARALLEL / 2 + 1, NUM_TIMESTEPS_IN_PARALLEL + 1};

        for (int p = 0; p < NUM_PIPELINE_STAGES; p++) {
            if (curr_dt) {
                recv_zoid_to_zoid_sizes_pipelined[p].resize(NUM_ZOIDS_MANY_CUTS);
            } else {
                recv_zoid_to_zoid_sizes_pipelined_next_dt[p].resize(NUM_ZOIDS_MANY_CUTS);
            }

            for (int zoid_num = 0; zoid_num < NUM_ZOIDS_MANY_CUTS; zoid_num++) {
                if (zoid_num % comm->nprocs != comm->me) {
                    continue;
                }

                auto& zoid = curr_dt ? zoid_num_to_zoid_many_cuts[zoid_num]
                                     : zoid_num_to_zoid_many_cuts_next_dt[zoid_num];

                auto& recv_neighbors = curr_dt ? recv_from_neighbors_many_cuts[zoid_num]
                                               : recv_from_neighbors_many_cuts_next_dt[zoid_num];

                if (curr_dt) {
                    recv_zoid_to_zoid_sizes_pipelined[p][zoid_num].resize(recv_neighbors.size());
                } else {
                    recv_zoid_to_zoid_sizes_pipelined_next_dt[p][zoid_num].resize(recv_neighbors.size());
                }

                for (int i = 0; i < recv_neighbors.size(); i++) {
                    int recv_zoid_num = recv_neighbors[i];

                    int nrecv_from_zoid = 0;
                    for (int t = start_t[p]; t < end_t[p]; t++) {
                        nrecv_from_zoid += zoid.recv_force_idxs_double_buffering[t][i].size();
                        nrecv_from_zoid += zoid.recv_pos_idxs_double_buffering[t][i].size();
                        nrecv_from_zoid += zoid.recv_vel_idxs_double_buffering[t][i].size();
                    }

                    if (curr_dt) {
                        recv_zoid_to_zoid_sizes_pipelined[p][zoid_num][i] = nrecv_from_zoid;
                    } else {
                        recv_zoid_to_zoid_sizes_pipelined_next_dt[p][zoid_num][i] = nrecv_from_zoid;
                    }

                    int total_doubles_recv_from_zoid = DEBUG_SEND_RECV_DATA ? nrecv_from_zoid * (3 + 1) : nrecv_from_zoid * 3;
                    if (total_doubles_recv_from_zoid > nrecv_buf_recv_zoid_to_zoid[p][zoid_num][i]) {
                        GROW_RECV_ZOID_TO_ZOID_MANY_CUTS(zoid_num, i, total_doubles_recv_from_zoid, p);
                    }
                }
            }
        }
    }

    void PACK_AND_SEND_DATA_ZOID_TO_ZOID_SETUP(queue_info& zoid, int dep, std::vector<MPI_Request>& r) {
        auto& send_neighbors = send_to_neighbors_many_cuts[zoid.num];

        constexpr int DEFAULT_PIPELINE_STAGE = 0;
        int zoid_num = zoid.num;

        for (int i = 0; i < send_neighbors.size(); i++) {
            int send_zoid_num = send_neighbors[i];
            int nsend = send_zoid_to_zoid_sizes_setup[zoid_num][i];
            int zoid_ndoubles_send = DEBUG_SEND_RECV_DATA ? nsend * (3 + 1) : nsend * 3;

            if (zoid_ndoubles_send > nsend_buf_send_zoid_to_zoid[DEFAULT_PIPELINE_STAGE][zoid.num][i]) {
                assert(false);
                GROW_SEND_ZOID_TO_ZOID_MANY_CUTS(zoid.num, i, zoid_ndoubles_send, DEFAULT_PIPELINE_STAGE);
            }

            auto* buf = buf_send_zoid_to_zoid[DEFAULT_PIPELINE_STAGE][zoid_num][i];

            int buf_idx = PACK_DATA_MANY_CUTS_HELPER_SETUP(zoid, buf, i, send_zoid_num);

            assert(buf_idx == zoid_ndoubles_send);

            if (buf_idx > 0 && (send_zoid_num % comm->nprocs != comm->me)) {
                int mpi_tag = get_mpi_tag_many_cuts(send_zoid_num, zoid.num);
                r.emplace_back();

                int comm_idx = ZOID_TO_ZOID_TO_VCI_IDX.at({zoid_num, send_zoid_num});
                MPI_Isend(buf, buf_idx, MPI_DOUBLE,
                          send_zoid_num % comm->nprocs, mpi_tag,
                          all_comms[comm_idx], &r[r.size() - 1]);
            }
        }
    }

    template <bool curr_dt>
    void PACK_AND_SEND_DATA_ZOID_TO_ZOID(queue_info& zoid, int dep, int start_t, int end_t, std::vector<MPI_Request>& r) {
        auto& send_neighbors = curr_dt ? send_to_neighbors_many_cuts[zoid.num]
                                       : send_to_neighbors_many_cuts_next_dt[zoid.num];

        constexpr int DEFAULT_PIPELINE_STAGE = 0;
        int zoid_num = zoid.num;

        auto& send_request_idxs = curr_dt ? send_to_neighbors_not_my_proc_idxs[zoid_num]
                : send_to_neighbors_not_my_proc_idxs_next_dt[zoid_num];

        cilk_for (int i = 0; i < send_neighbors.size(); i++) {
            int send_zoid_num = send_neighbors[i];
            int nsend = curr_dt ? send_zoid_to_zoid_sizes[zoid_num][i] : send_zoid_to_zoid_sizes_next_dt[zoid_num][i];
            int zoid_ndoubles_send = DEBUG_SEND_RECV_DATA ? nsend * (3 + 1) : nsend * 3;
            int send_request_idx = send_request_idxs[i];

            if (send_zoid_num % comm->nprocs == comm->me) {
                continue;
            }

            if (zoid_ndoubles_send > nsend_buf_send_zoid_to_zoid[DEFAULT_PIPELINE_STAGE][zoid.num][i]) {
                assert(false);
                GROW_SEND_ZOID_TO_ZOID_MANY_CUTS(zoid.num, i, zoid_ndoubles_send, DEFAULT_PIPELINE_STAGE);
            }

            auto* buf = buf_send_zoid_to_zoid[DEFAULT_PIPELINE_STAGE][zoid_num][i];

            int buf_idx = PACK_DATA_MANY_CUTS_HELPER<curr_dt>(zoid, buf, i, send_zoid_num, start_t, end_t);

            assert(buf_idx == zoid_ndoubles_send);

            if (buf_idx > 0 && (send_zoid_num % comm->nprocs != comm->me)) {
                int mpi_tag = get_mpi_tag_many_cuts(send_zoid_num, zoid.num);
                // r.emplace_back();
                /*
                MPI_Isend(buf, buf_idx, MPI_DOUBLE,
                          send_zoid_num % comm->nprocs, mpi_tag,
                          world, &r[r.size() - 1]);
                */

                int comm_idx = curr_dt ? ZOID_TO_ZOID_TO_VCI_IDX.at({zoid_num, send_zoid_num})
                        : ZOID_TO_ZOID_TO_VCI_IDX_NEXT_DT.at({zoid_num, send_zoid_num});

                assert(send_request_idx != -1);

                MPI_Isend(buf, buf_idx, MPI_DOUBLE,
                          send_zoid_num % comm->nprocs, mpi_tag,
                          all_comms[comm_idx], &r[send_request_idx]);
            }
        }
    }

    template <bool curr_dt, bool is_initial>
    void PACK_AND_SEND_DATA_ZOID_TO_ZOID_PIPELINED(queue_info& zoid, int dep,
                                                   int start_t, int end_t, int pipeline_stage,
                                                   std::vector<MPI_Request>& r) {
        auto& send_neighbors = curr_dt ? send_to_neighbors_many_cuts[zoid.num]
                                       : send_to_neighbors_many_cuts_next_dt[zoid.num];

        int zoid_num = zoid.num;

        for (int i = 0; i < send_neighbors.size(); i++) {
            int send_zoid_num = send_neighbors[i];
            int nsend = curr_dt ? send_zoid_to_zoid_sizes_pipelined[pipeline_stage][zoid_num][i]
                    : send_zoid_to_zoid_sizes_pipelined_next_dt[pipeline_stage][zoid_num][i];

            int zoid_ndoubles_send = DEBUG_SEND_RECV_DATA ? nsend * (3 + 1) : nsend * 3;

            if (zoid_ndoubles_send > nsend_buf_send_zoid_to_zoid[pipeline_stage][zoid.num][i]) {
                assert(false);
                GROW_SEND_ZOID_TO_ZOID_MANY_CUTS(zoid.num, i, zoid_ndoubles_send, pipeline_stage);
            }

            auto* buf = buf_send_zoid_to_zoid[pipeline_stage][zoid_num][i];
            int buf_idx = PACK_DATA_MANY_CUTS_HELPER<curr_dt>(zoid, buf, i, send_zoid_num, start_t, end_t);

            assert(buf_idx == zoid_ndoubles_send);

            if (buf_idx > 0 && (send_zoid_num % comm->nprocs != comm->me)) {
                int mpi_tag = get_mpi_tag_many_cuts(send_zoid_num, zoid.num);
                r.emplace_back();
                /*
                MPI_Isend(buf, buf_idx, MPI_DOUBLE,
                          send_zoid_num % comm->nprocs, mpi_tag,
                          world, &r[r.size() - 1]);
                */

                int comm_idx = curr_dt ? ZOID_TO_ZOID_TO_VCI_IDX.at({zoid_num, send_zoid_num})
                                       : ZOID_TO_ZOID_TO_VCI_IDX_NEXT_DT.at({zoid_num, send_zoid_num});

                MPI_Isend(buf, buf_idx, MPI_DOUBLE,
                          send_zoid_num % comm->nprocs, mpi_tag,
                          all_comms[comm_idx], &r[r.size() - 1]);
            }
        }
    }

    void RECEIVE_DATA_ZOID_TO_ZOID_SETUP(int zoid_num, std::vector<MPI_Request>& r) {
        auto& queues = queues_many_cuts;

        constexpr int DEFAULT_PIPELINE_STAGE = 0;

        auto& recv_neighbors = recv_from_neighbors_many_cuts[zoid_num];

        for (int i = 0; i < recv_neighbors.size(); i++) {
            int recv_zoid_num = recv_neighbors[i];
            if (recv_zoid_num % comm->nprocs == comm->me) {
                continue;
            }

            int recv_size = recv_zoid_to_zoid_sizes_setup[zoid_num][i];

            auto* buf = buf_recv_zoid_to_zoid[DEFAULT_PIPELINE_STAGE][zoid_num][i];

            int total_doubles_recv_from_zoid = DEBUG_SEND_RECV_DATA ? recv_size * (3 + 1) : recv_size * 3;
            if (total_doubles_recv_from_zoid > nrecv_buf_recv_zoid_to_zoid[DEFAULT_PIPELINE_STAGE][zoid_num][i]) {
                assert(false);
                GROW_RECV_ZOID_TO_ZOID_MANY_CUTS(zoid_num, i, total_doubles_recv_from_zoid, DEFAULT_PIPELINE_STAGE);
            }

            if (total_doubles_recv_from_zoid > 0) {
                r.emplace_back();
                int mpi_tag = get_mpi_tag_many_cuts(zoid_num, recv_zoid_num);
                int comm_idx = ZOID_TO_ZOID_TO_VCI_IDX.at({recv_zoid_num, zoid_num});
                MPI_Irecv(buf, total_doubles_recv_from_zoid, MPI_DOUBLE,
                          recv_zoid_num % comm->nprocs, mpi_tag,
                          all_comms[comm_idx], &r[r.size() - 1]);
            }
        }
    }

    template <bool curr_dt>
    void RECEIVE_DATA_ZOID_TO_ZOID_WAITANY(int dep, int zoid_num, std::vector<MPI_Request>& r) {
        auto& queues = curr_dt ? queues_many_cuts : queues_many_cuts_next_dt;

        constexpr int DEFAULT_PIPELINE_STAGE = 0;

        auto& recv_neighbors = curr_dt ? recv_from_neighbors_many_cuts[zoid_num]
                                       : recv_from_neighbors_many_cuts_next_dt[zoid_num];

        for (int i = 0; i < recv_neighbors.size(); i++) {
            int recv_zoid_num = recv_neighbors[i];
            if (recv_zoid_num % comm->nprocs == comm->me) {
                continue;
            }

            int recv_size = curr_dt ? recv_zoid_to_zoid_sizes[zoid_num][i]
                                    : recv_zoid_to_zoid_sizes_next_dt[zoid_num][i];

            auto* buf = buf_recv_zoid_to_zoid[DEFAULT_PIPELINE_STAGE][zoid_num][i];

            int total_doubles_recv_from_zoid = DEBUG_SEND_RECV_DATA ? recv_size * (3 + 1) : recv_size * 3;
            if (total_doubles_recv_from_zoid > nrecv_buf_recv_zoid_to_zoid[DEFAULT_PIPELINE_STAGE][zoid_num][i]) {
                assert(false);
                GROW_RECV_ZOID_TO_ZOID_MANY_CUTS(zoid_num, i, total_doubles_recv_from_zoid, DEFAULT_PIPELINE_STAGE);
            }

            if (total_doubles_recv_from_zoid > 0) {
                int mpi_tag = get_mpi_tag_many_cuts(zoid_num, recv_zoid_num);
                int recv_request_idx = curr_dt ? recv_request_zoid_to_idx[dep].at({recv_zoid_num, zoid_num})
                        : recv_request_zoid_to_idx_next_dt[dep].at({recv_zoid_num, zoid_num});
                int comm_idx = curr_dt ? ZOID_TO_ZOID_TO_VCI_IDX.at({recv_zoid_num, zoid_num})
                                       : ZOID_TO_ZOID_TO_VCI_IDX_NEXT_DT.at({recv_zoid_num, zoid_num});
                MPI_Irecv(buf, total_doubles_recv_from_zoid, MPI_DOUBLE,
                          recv_zoid_num % comm->nprocs, mpi_tag,
                          all_comms[comm_idx], &r[recv_request_idx]);
            }
        }
    }

    template <bool curr_dt>
    void RECEIVE_DATA_ZOID_TO_ZOID(int zoid_num, std::vector<MPI_Request>& r) {
        auto& queues = curr_dt ? queues_many_cuts : queues_many_cuts_next_dt;

        constexpr int DEFAULT_PIPELINE_STAGE = 0;

        auto& recv_neighbors = curr_dt ? recv_from_neighbors_many_cuts[zoid_num]
                : recv_from_neighbors_many_cuts_next_dt[zoid_num];

        for (int i = 0; i < recv_neighbors.size(); i++) {
            int recv_zoid_num = recv_neighbors[i];
            if (recv_zoid_num % comm->nprocs == comm->me) {
                continue;
            }

            int recv_size = curr_dt ? recv_zoid_to_zoid_sizes[zoid_num][i]
                    : recv_zoid_to_zoid_sizes_next_dt[zoid_num][i];

            auto* buf = buf_recv_zoid_to_zoid[DEFAULT_PIPELINE_STAGE][zoid_num][i];

            int total_doubles_recv_from_zoid = DEBUG_SEND_RECV_DATA ? recv_size * (3 + 1) : recv_size * 3;
            if (total_doubles_recv_from_zoid > nrecv_buf_recv_zoid_to_zoid[DEFAULT_PIPELINE_STAGE][zoid_num][i]) {
                assert(false);
                GROW_RECV_ZOID_TO_ZOID_MANY_CUTS(zoid_num, i, total_doubles_recv_from_zoid, DEFAULT_PIPELINE_STAGE);
            }

            if (total_doubles_recv_from_zoid > 0) {
                r.emplace_back();
                int mpi_tag = get_mpi_tag_many_cuts(zoid_num, recv_zoid_num);
                /*
                MPI_Irecv(buf, total_doubles_recv_from_zoid, MPI_DOUBLE,
                          recv_zoid_num % comm->nprocs, mpi_tag, world, &r[r.size() - 1]);
                */
                int comm_idx = curr_dt ? ZOID_TO_ZOID_TO_VCI_IDX.at({recv_zoid_num, zoid_num})
                        : ZOID_TO_ZOID_TO_VCI_IDX_NEXT_DT.at({recv_zoid_num, zoid_num});

                MPI_Irecv(buf, total_doubles_recv_from_zoid, MPI_DOUBLE,
                          recv_zoid_num % comm->nprocs, mpi_tag,
                          all_comms[comm_idx], &r[r.size() - 1]);
                /*
                std::stringstream s1;
                s1 << "me: " << comm->me << " zoid: " << zoid_num << " recv from: " << recv_zoid_num
                << " src: " << recv_zoid_num % NUM_STREAMS << " dst: " << zoid_num % NUM_STREAMS
                << " request addr: " << &r[r.size() - 1]
                << std::endl;
                std::cout << s1.str();
                auto res = MPIX_Stream_irecv(buf, total_doubles_recv_from_zoid, MPI_DOUBLE,
                                  recv_zoid_num % comm->nprocs, mpi_tag,
                                  stream_comm, recv_zoid_num % NUM_STREAMS,
                                  zoid_num % NUM_STREAMS,
                                  &r[r.size() - 1]);
                assert(res == MPI_SUCCESS);
                */
            }
        }
    }

    template <bool curr_dt>
    void RECEIVE_DATA_ZOID_TO_ZOID_PIPELINED(int zoid_num, std::vector<MPI_Request>& r, int start_t, int end_t,
                                             int pipeline_stage) {

        auto& queues = curr_dt ? queues_many_cuts : queues_many_cuts_next_dt;

        auto& recv_neighbors = curr_dt ? recv_from_neighbors_many_cuts[zoid_num]
                                       : recv_from_neighbors_many_cuts_next_dt[zoid_num];

        for (int i = 0; i < recv_neighbors.size(); i++) {
            int recv_zoid_num = recv_neighbors[i];
            if (recv_zoid_num % comm->nprocs == comm->me) {
                continue;
            }

            int recv_size = curr_dt ? recv_zoid_to_zoid_sizes_pipelined[pipeline_stage][zoid_num][i]
                                    : recv_zoid_to_zoid_sizes_pipelined_next_dt[pipeline_stage][zoid_num][i];

            auto* buf = buf_recv_zoid_to_zoid[pipeline_stage][zoid_num][i];

            int total_doubles_recv_from_zoid = DEBUG_SEND_RECV_DATA ? recv_size * (3 + 1) : recv_size * 3;
            if (total_doubles_recv_from_zoid > nrecv_buf_recv_zoid_to_zoid[pipeline_stage][zoid_num][i]) {
                assert(false);
                GROW_RECV_ZOID_TO_ZOID_MANY_CUTS(zoid_num, i, total_doubles_recv_from_zoid, pipeline_stage);
            }

            if (total_doubles_recv_from_zoid > 0) {
                r.emplace_back();
                int mpi_tag = get_mpi_tag_many_cuts(zoid_num, recv_zoid_num);
                /*
                MPI_Irecv(buf, total_doubles_recv_from_zoid, MPI_DOUBLE,
                          recv_zoid_num % comm->nprocs, mpi_tag, world, &r[r.size() - 1]);
                */
                int comm_idx = curr_dt ? ZOID_TO_ZOID_TO_VCI_IDX.at({recv_zoid_num, zoid_num})
                        : ZOID_TO_ZOID_TO_VCI_IDX_NEXT_DT.at({recv_zoid_num, zoid_num});

                MPI_Irecv(buf, total_doubles_recv_from_zoid, MPI_DOUBLE,
                          recv_zoid_num % comm->nprocs, mpi_tag,
                          all_comms[comm_idx], &r[r.size() - 1]);
            }
        }
    }

    template <bool curr_dt>
    void UNPACK_FORCE_MANY_CUTS_HELPER(queue_info& zoid, double* buf, int recv_idx, int recv_zoid_num,
                                       int start_t, int end_t) {
        auto& recv_zoid = curr_dt ? zoid_num_to_zoid_many_cuts[recv_zoid_num]
                                  : zoid_num_to_zoid_many_cuts_next_dt[recv_zoid_num];

        auto& recv_force_idxs = zoid.recv_force_idxs_double_buffering_flattened[recv_idx];

        int num_recv_force = recv_force_idxs.size();

        if (DEBUG_SEND_RECV_DATA) {
            for (int i = 0; i < recv_force_idxs.size(); i++) {
                int buf_idx = i * (3 + 1);
                int idx = recv_force_idxs[i];
                auto target_tag = (tagint) ubuf(buf[buf_idx]).i;
                double f_x = buf[buf_idx + 1];
                double f_y = buf[buf_idx + 2];
                double f_z = buf[buf_idx + 3];

                if (target_tag != zoid.tag_stencil_md[0][idx]) {
                    std::cout << "FORCE TAG WRONG me: " << comm->me
                              << " idx: " << idx << " i: " << i
                              << " my zoid: " << zoid.num << " my proc: " << zoid.num % comm->nprocs
                              << " recv from proc: " << recv_zoid_num % comm->nprocs
                              << " tag I got: " << target_tag << " tag I want: " << zoid.tag_stencil_md[0][idx]
                              << " buf_idx: " << buf_idx
                              << " buf: " << buf
                              << std::endl;
                }
                assert(target_tag == zoid.tag_stencil_md[0][idx]);
                zoid.f_stencil_md[0][idx].x += f_x;
                zoid.f_stencil_md[0][idx].y += f_y;
                zoid.f_stencil_md[0][idx].z += f_z;
            }
        } else {
            for (int i = 0; i < recv_force_idxs.size(); i++) {
                int buf_idx = i * 3;
                int idx = recv_force_idxs[i];
                double f_x = buf[buf_idx];
                double f_y = buf[buf_idx + 1];
                double f_z = buf[buf_idx + 2];

                zoid.f_stencil_md[0][idx].x += f_x;
                zoid.f_stencil_md[0][idx].y += f_y;
                zoid.f_stencil_md[0][idx].z += f_z;
            }
        }
    }

    template <bool curr_dt>
    void UNPACK_POS_VEL_MANY_CUTS_HELPER(queue_info& zoid, double* buf, int recv_idx, int recv_zoid_num,
                                         int start_t, int end_t) {
        auto& recv_zoid = curr_dt ? zoid_num_to_zoid_many_cuts[recv_zoid_num]
                                  : zoid_num_to_zoid_many_cuts_next_dt[recv_zoid_num];

        int pbc_flag_[3] = {0};
        for (int dim = 0; dim < 3; dim++) {
            if (recv_zoid.where[dim] == NUM_ZOIDS_PER_DIMENSION - 1 && zoid.where[dim] == 0) {
                pbc_flag_[dim] = -1;
            }

            if (recv_zoid.where[dim] == 0 && zoid.where[dim] == NUM_ZOIDS_PER_DIMENSION - 1) {
                pbc_flag_[dim] = 1;
            }
        }

        auto& recv_force_idxs = zoid.recv_force_idxs_double_buffering_flattened[recv_idx];
        auto& recv_pos_idxs = zoid.recv_pos_idxs_double_buffering_flattened[0][recv_idx];
        auto& recv_pos_idxs2 = zoid.recv_pos_idxs_double_buffering_flattened[1][recv_idx];
        auto& recv_vel_idxs = zoid.recv_vel_idxs_double_buffering_flattened[recv_idx];

        int num_recv_force = recv_force_idxs.size();
        int num_recv_pos = recv_pos_idxs.size();
        int num_recv_pos2 = recv_pos_idxs2.size();
        int num_recv_vel = recv_vel_idxs.size();

        if (DEBUG_SEND_RECV_DATA) {
            int pos_starting_idx = num_recv_force * (3 + 1);

            for (int i = 0; i < recv_pos_idxs.size(); i++) {
                int idx = recv_pos_idxs[i];
                int buf_idx = pos_starting_idx + i * (3 + 1);
                auto target_tag = (tagint) ubuf(buf[buf_idx]).i;
                double x_x = buf[buf_idx + 1];
                double x_y = buf[buf_idx + 2];
                double x_z = buf[buf_idx + 3];
                assert(target_tag == zoid.tag_stencil_md[0][idx]);
                if (target_tag != zoid.tag_stencil_md[0][idx]) {
                    std::cout << "POS me: " << comm->me << " my zoid: " << recv_zoid.num
                              << " recv from: " << recv_zoid_num
                              << " recv proc: " << recv_zoid_num % comm->nprocs << " tag I got: " << target_tag << " tag I want: " << recv_zoid.tag_stencil_md[0][idx]
                              << std::endl;
                }
                zoid.x_stencil_md[0][idx].x = x_x + pbc_flag_[0] * domain->prd[0];
                zoid.x_stencil_md[0][idx].y = x_y + pbc_flag_[1] * domain->prd[1];
                zoid.x_stencil_md[0][idx].z = x_z + pbc_flag_[2] * domain->prd[2];
            }

            int pos_starting_idx2 = (num_recv_force + num_recv_pos) * (3 + 1);

            for (int i = 0; i < recv_pos_idxs2.size(); i++) {
                int idx = recv_pos_idxs2[i];
                int buf_idx = pos_starting_idx2 + i * (3 + 1);
                auto target_tag = (tagint) ubuf(buf[buf_idx]).i;
                double x_x = buf[buf_idx + 1];
                double x_y = buf[buf_idx + 2];
                double x_z = buf[buf_idx + 3];
                assert(target_tag == zoid.tag_stencil_md[0][idx]);
                zoid.x_stencil_md[1][idx].x = x_x + pbc_flag_[0] * domain->prd[0];
                zoid.x_stencil_md[1][idx].y = x_y + pbc_flag_[1] * domain->prd[1];
                zoid.x_stencil_md[1][idx].z = x_z + pbc_flag_[2] * domain->prd[2];
            }

            int vel_starting_idx = (num_recv_force + num_recv_pos + num_recv_pos2) * (3 + 1);
            for (int i = 0; i < recv_vel_idxs.size(); i++) {
                int idx = recv_vel_idxs[i];
                int buf_idx = vel_starting_idx + i * (3 + 1);
                auto target_tag = (tagint) ubuf(buf[buf_idx]).i;
                double v_x = buf[buf_idx + 1];
                double v_y = buf[buf_idx + 2];
                double v_z = buf[buf_idx + 3];
                assert(target_tag == zoid.tag_stencil_md[0][idx]);
                zoid.v_stencil_md[0][idx].x = v_x;
                zoid.v_stencil_md[0][idx].y = v_y;
                zoid.v_stencil_md[0][idx].z = v_z;
            }
        } else {
            int pos_starting_idx = num_recv_force * 3;

            for (int i = 0; i < recv_pos_idxs.size(); i++) {
                int idx = recv_pos_idxs[i];
                int buf_idx = pos_starting_idx + i * 3;
                double x_x = buf[buf_idx];
                double x_y = buf[buf_idx + 1];
                double x_z = buf[buf_idx + 2];

                zoid.x_stencil_md[0][idx].x = x_x + pbc_flag_[0] * domain->prd[0];
                zoid.x_stencil_md[0][idx].y = x_y + pbc_flag_[1] * domain->prd[1];
                zoid.x_stencil_md[0][idx].z = x_z + pbc_flag_[2] * domain->prd[2];
            }

            int pos_starting_idx2 = (num_recv_force + num_recv_pos) * 3;

            for (int i = 0; i < recv_pos_idxs2.size(); i++) {
                int idx = recv_pos_idxs2[i];
                int buf_idx = pos_starting_idx2 + i * 3;
                double x_x = buf[buf_idx];
                double x_y = buf[buf_idx + 1];
                double x_z = buf[buf_idx + 2];
                zoid.x_stencil_md[1][idx].x = x_x + pbc_flag_[0] * domain->prd[0];
                zoid.x_stencil_md[1][idx].y = x_y + pbc_flag_[1] * domain->prd[1];
                zoid.x_stencil_md[1][idx].z = x_z + pbc_flag_[2] * domain->prd[2];
            }

            int vel_starting_idx = (num_recv_force + num_recv_pos + num_recv_pos2) * 3;
            for (int i = 0; i < recv_vel_idxs.size(); i++) {
                int idx = recv_vel_idxs[i];
                int buf_idx = vel_starting_idx + i * 3;
                double v_x = buf[buf_idx];
                double v_y = buf[buf_idx + 1];
                double v_z = buf[buf_idx + 2];
                zoid.v_stencil_md[0][idx].x = v_x;
                zoid.v_stencil_md[0][idx].y = v_y;
                zoid.v_stencil_md[0][idx].z = v_z;
            }
        }
    }

    template <bool curr_dt>
    void UNPACK_DATA_MANY_CUTS_HELPER(queue_info& zoid, double* buf, int recv_idx, int recv_zoid_num,
                                      int start_t, int end_t) {
        auto& recv_zoid = curr_dt ? zoid_num_to_zoid_many_cuts[recv_zoid_num]
                                  : zoid_num_to_zoid_many_cuts_next_dt[recv_zoid_num];

        int pbc_flag_[3] = {0};
        for (int dim = 0; dim < 3; dim++) {
            if (recv_zoid.where[dim] == NUM_ZOIDS_PER_DIMENSION - 1 && zoid.where[dim] == 0) {
                pbc_flag_[dim] = -1;
            }

            if (recv_zoid.where[dim] == 0 && zoid.where[dim] == NUM_ZOIDS_PER_DIMENSION - 1) {
                pbc_flag_[dim] = 1;
            }
        }

        auto& recv_force_idxs = zoid.recv_force_idxs_double_buffering_flattened[recv_idx];
        auto& recv_pos_idxs = zoid.recv_pos_idxs_double_buffering_flattened[0][recv_idx];
        auto& recv_pos_idxs2 = zoid.recv_pos_idxs_double_buffering_flattened[1][recv_idx];
        auto& recv_vel_idxs = zoid.recv_vel_idxs_double_buffering_flattened[recv_idx];

        int num_recv_force = recv_force_idxs.size();
        int num_recv_pos = recv_pos_idxs.size();
        int num_recv_pos2 = recv_pos_idxs2.size();
        int num_recv_vel = recv_vel_idxs.size();

        if (DEBUG_SEND_RECV_DATA) {
            for (int i = 0; i < recv_force_idxs.size(); i++) {
                int buf_idx = i * (3 + 1);
                int idx = recv_force_idxs[i];
                auto target_tag = (tagint) ubuf(buf[buf_idx]).i;
                double f_x = buf[buf_idx + 1];
                double f_y = buf[buf_idx + 2];
                double f_z = buf[buf_idx + 3];

                if (target_tag != zoid.tag_stencil_md[0][idx]) {
                    std::cout << "FORCE TAG WRONG me: " << comm->me
                              << " idx: " << idx << " i: " << i
                              << " my zoid: " << zoid.num << " my proc: " << zoid.num % comm->nprocs
                              << " recv from proc: " << recv_zoid_num % comm->nprocs
                              << " tag I got: " << target_tag << " tag I want: " << zoid.tag_stencil_md[0][idx]
                              << " buf_idx: " << buf_idx
                              << " buf: " << buf
                              << std::endl;
                }
                assert(target_tag == zoid.tag_stencil_md[0][idx]);
                zoid.f_stencil_md[0][idx].x += f_x;
                zoid.f_stencil_md[0][idx].y += f_y;
                zoid.f_stencil_md[0][idx].z += f_z;
            }

            int pos_starting_idx = num_recv_force * (3 + 1);

            for (int i = 0; i < recv_pos_idxs.size(); i++) {
                int idx = recv_pos_idxs[i];
                int buf_idx = pos_starting_idx + i * (3 + 1);
                auto target_tag = (tagint) ubuf(buf[buf_idx]).i;
                double x_x = buf[buf_idx + 1];
                double x_y = buf[buf_idx + 2];
                double x_z = buf[buf_idx + 3];
                assert(target_tag == zoid.tag_stencil_md[0][idx]);
                if (target_tag != zoid.tag_stencil_md[0][idx]) {
                    std::cout << "POS me: " << comm->me << " my zoid: " << recv_zoid.num
                              << " recv from: " << recv_zoid_num
                              << " recv proc: " << recv_zoid_num % comm->nprocs << " tag I got: " << target_tag << " tag I want: " << recv_zoid.tag_stencil_md[0][idx]
                              << std::endl;
                }
                zoid.x_stencil_md[0][idx].x = x_x + pbc_flag_[0] * domain->prd[0];
                zoid.x_stencil_md[0][idx].y = x_y + pbc_flag_[1] * domain->prd[1];
                zoid.x_stencil_md[0][idx].z = x_z + pbc_flag_[2] * domain->prd[2];
            }

            int pos_starting_idx2 = (num_recv_force + num_recv_pos) * (3 + 1);

            for (int i = 0; i < recv_pos_idxs2.size(); i++) {
                int idx = recv_pos_idxs2[i];
                int buf_idx = pos_starting_idx2 + i * (3 + 1);
                auto target_tag = (tagint) ubuf(buf[buf_idx]).i;
                double x_x = buf[buf_idx + 1];
                double x_y = buf[buf_idx + 2];
                double x_z = buf[buf_idx + 3];
                assert(target_tag == zoid.tag_stencil_md[0][idx]);
                zoid.x_stencil_md[1][idx].x = x_x + pbc_flag_[0] * domain->prd[0];
                zoid.x_stencil_md[1][idx].y = x_y + pbc_flag_[1] * domain->prd[1];
                zoid.x_stencil_md[1][idx].z = x_z + pbc_flag_[2] * domain->prd[2];
            }

            int vel_starting_idx = (num_recv_force + num_recv_pos + num_recv_pos2) * (3 + 1);
            for (int i = 0; i < recv_vel_idxs.size(); i++) {
                int idx = recv_vel_idxs[i];
                int buf_idx = vel_starting_idx + i * (3 + 1);
                auto target_tag = (tagint) ubuf(buf[buf_idx]).i;
                double v_x = buf[buf_idx + 1];
                double v_y = buf[buf_idx + 2];
                double v_z = buf[buf_idx + 3];
                assert(target_tag == zoid.tag_stencil_md[0][idx]);
                zoid.v_stencil_md[0][idx].x = v_x;
                zoid.v_stencil_md[0][idx].y = v_y;
                zoid.v_stencil_md[0][idx].z = v_z;
            }
        } else {
            for (int i = 0; i < recv_force_idxs.size(); i++) {
                int buf_idx = i * 3;
                int idx = recv_force_idxs[i];
                double f_x = buf[buf_idx];
                double f_y = buf[buf_idx + 1];
                double f_z = buf[buf_idx + 2];

                zoid.f_stencil_md[0][idx].x += f_x;
                zoid.f_stencil_md[0][idx].y += f_y;
                zoid.f_stencil_md[0][idx].z += f_z;
            }

            int pos_starting_idx = num_recv_force * 3;

            for (int i = 0; i < recv_pos_idxs.size(); i++) {
                int idx = recv_pos_idxs[i];
                int buf_idx = pos_starting_idx + i * 3;
                double x_x = buf[buf_idx];
                double x_y = buf[buf_idx + 1];
                double x_z = buf[buf_idx + 2];

                zoid.x_stencil_md[0][idx].x = x_x + pbc_flag_[0] * domain->prd[0];
                zoid.x_stencil_md[0][idx].y = x_y + pbc_flag_[1] * domain->prd[1];
                zoid.x_stencil_md[0][idx].z = x_z + pbc_flag_[2] * domain->prd[2];
            }

            int pos_starting_idx2 = (num_recv_force + num_recv_pos) * 3;

            for (int i = 0; i < recv_pos_idxs2.size(); i++) {
                int idx = recv_pos_idxs2[i];
                int buf_idx = pos_starting_idx2 + i * 3;
                double x_x = buf[buf_idx];
                double x_y = buf[buf_idx + 1];
                double x_z = buf[buf_idx + 2];
                zoid.x_stencil_md[1][idx].x = x_x + pbc_flag_[0] * domain->prd[0];
                zoid.x_stencil_md[1][idx].y = x_y + pbc_flag_[1] * domain->prd[1];
                zoid.x_stencil_md[1][idx].z = x_z + pbc_flag_[2] * domain->prd[2];
            }

            int vel_starting_idx = (num_recv_force + num_recv_pos + num_recv_pos2) * 3;
            for (int i = 0; i < recv_vel_idxs.size(); i++) {
                int idx = recv_vel_idxs[i];
                int buf_idx = vel_starting_idx + i * 3;
                double v_x = buf[buf_idx];
                double v_y = buf[buf_idx + 1];
                double v_z = buf[buf_idx + 2];
                zoid.v_stencil_md[0][idx].x = v_x;
                zoid.v_stencil_md[0][idx].y = v_y;
                zoid.v_stencil_md[0][idx].z = v_z;
            }
        }

        /*
        int buf_idx = 0;
        for (int t = start_t; t < end_t; t++) {
            auto& recv_force_idxs = zoid.recv_force_idxs_double_buffering[t][recv_idx];
            auto& recv_pos_idxs = zoid.recv_pos_idxs_double_buffering[t][recv_idx];
            auto& recv_vel_idxs = zoid.recv_vel_idxs_double_buffering[t][recv_idx];

            if (DEBUG_SEND_RECV_DATA) {
                for (int k = 0; k < recv_force_idxs.size(); k++) {
                    int idx = recv_force_idxs[k];
                    auto target_tag = (tagint) ubuf(buf[buf_idx++]).i;
                    double f_x = buf[buf_idx++];
                    double f_y = buf[buf_idx++];
                    double f_z = buf[buf_idx++];

                    if (target_tag != zoid.tag_stencil_md[0][idx]) {
                        std::cout << "FORCE TAG WRONG me: " << comm->me
                                  << " idx: " << idx << " k: " << k
                                  << " my zoid: " << zoid.num << " my proc: " << zoid.num % comm->nprocs
                                  << " recv from zoid: " << recv_zoid_num << " time: " << t
                                  << " recv from proc: " << recv_zoid_num % comm->nprocs
                                  << " tag I got: " << target_tag << " tag I want: " << zoid.tag_stencil_md[0][idx]
                                  << " buf_idx: " << buf_idx
                                  << " buf: " << buf
                                  << std::endl;
                    }
                    assert(target_tag == zoid.tag_stencil_md[0][idx]);
                    zoid.f_stencil_md[0][idx].x += f_x;
                    zoid.f_stencil_md[0][idx].y += f_y;
                    zoid.f_stencil_md[0][idx].z += f_z;
                }

                for (int k = 0; k < recv_pos_idxs.size(); k++) {
                    int idx = recv_pos_idxs[k];
                    auto target_tag = (tagint) ubuf(buf[buf_idx++]).i;
                    double x_x = buf[buf_idx++];
                    double x_y = buf[buf_idx++];
                    double x_z = buf[buf_idx++];
                    assert(target_tag == zoid.tag_stencil_md[0][idx]);
                    if (target_tag != zoid.tag_stencil_md[0][idx]) {
                        std::cout << "POS me: " << comm->me << " my zoid: " << recv_zoid.num
                                  << " recv from: " << recv_zoid_num << " time: " << t
                                  << " recv proc: " << recv_zoid_num % comm->nprocs << " tag I got: " << target_tag << " tag I want: " << recv_zoid.tag_stencil_md[0][idx]
                                  << std::endl;
                    }
                    zoid.x_stencil_md[t % DOUBLE_BUFFERING][idx].x = x_x + pbc_flag_[0] * domain->prd[0];
                    zoid.x_stencil_md[t % DOUBLE_BUFFERING][idx].y = x_y + pbc_flag_[1] * domain->prd[1];
                    zoid.x_stencil_md[t % DOUBLE_BUFFERING][idx].z = x_z + pbc_flag_[2] * domain->prd[2];
                }

                for (int k = 0; k < recv_vel_idxs.size(); k++) {
                    int idx = recv_vel_idxs[k];
                    auto target_tag = (tagint) ubuf(buf[buf_idx++]).i;
                    double v_x = buf[buf_idx++];
                    double v_y = buf[buf_idx++];
                    double v_z = buf[buf_idx++];
                    assert(target_tag == zoid.tag_stencil_md[0][idx]);
                    zoid.v_stencil_md[0][idx].x = v_x;
                    zoid.v_stencil_md[0][idx].y = v_y;
                    zoid.v_stencil_md[0][idx].z = v_z;
                }
            } else {
                for (int k = 0; k < recv_force_idxs.size(); k++) {
                    int idx = recv_force_idxs[k];
                    double f_x = buf[buf_idx++];
                    double f_y = buf[buf_idx++];
                    double f_z = buf[buf_idx++];
                    zoid.f_stencil_md[0][idx].x += f_x;
                    zoid.f_stencil_md[0][idx].y += f_y;
                    zoid.f_stencil_md[0][idx].z += f_z;
                }

                for (int k = 0; k < recv_pos_idxs.size(); k++) {
                    int idx = recv_pos_idxs[k];
                    double x_x = buf[buf_idx++];
                    double x_y = buf[buf_idx++];
                    double x_z = buf[buf_idx++];
                    zoid.x_stencil_md[t % DOUBLE_BUFFERING][idx].x = x_x + pbc_flag_[0] * domain->prd[0];
                    zoid.x_stencil_md[t % DOUBLE_BUFFERING][idx].y = x_y + pbc_flag_[1] * domain->prd[1];
                    zoid.x_stencil_md[t % DOUBLE_BUFFERING][idx].z = x_z + pbc_flag_[2] * domain->prd[2];
                }

                for (int k = 0; k < recv_vel_idxs.size(); k++) {
                    int idx = recv_vel_idxs[k];
                    double v_x = buf[buf_idx++];
                    double v_y = buf[buf_idx++];
                    double v_z = buf[buf_idx++];
                    zoid.v_stencil_md[0][idx].x = v_x;
                    zoid.v_stencil_md[0][idx].y = v_y;
                    zoid.v_stencil_md[0][idx].z = v_z;
                }
            }
        }
        */
    }

    template <bool curr_dt>
    void UNPACK_DATA_MANY_CUTS_HELPER_SELF(queue_info& zoid, int recv_idx, int recv_zoid_num,
                                           int send_idx, int start_t, int end_t) {
        assert(recv_zoid_num % comm->nprocs == comm->me);

        auto &recv_zoid = curr_dt ? zoid_num_to_zoid_many_cuts[recv_zoid_num]
                                  : zoid_num_to_zoid_many_cuts_next_dt[recv_zoid_num];

        int pbc_flag_[3] = {0};
        for (int dim = 0; dim < 3; dim++) {
            if (recv_zoid.where[dim] == NUM_ZOIDS_PER_DIMENSION - 1 && zoid.where[dim] == 0) {
                pbc_flag_[dim] = -1;
            }

            if (recv_zoid.where[dim] == 0 && zoid.where[dim] == NUM_ZOIDS_PER_DIMENSION - 1) {
                pbc_flag_[dim] = 1;
            }
        }

        auto& send_force_idxs = recv_zoid.send_force_idxs_double_buffering_flattened[send_idx];
        auto& send_pos_idxs = recv_zoid.send_pos_idxs_double_buffering_flattened[0][send_idx];
        auto& send_pos_idxs2 = recv_zoid.send_pos_idxs_double_buffering_flattened[1][send_idx];
        auto& send_vel_idxs = recv_zoid.send_vel_idxs_double_buffering_flattened[send_idx];

        auto& recv_force_idxs = zoid.recv_force_idxs_double_buffering_flattened[recv_idx];
        auto& recv_pos_idxs = zoid.recv_pos_idxs_double_buffering_flattened[0][recv_idx];
        auto& recv_pos_idxs2 = zoid.recv_pos_idxs_double_buffering_flattened[1][recv_idx];
        auto& recv_vel_idxs = zoid.recv_vel_idxs_double_buffering_flattened[recv_idx];

        assert(send_force_idxs.size() == recv_force_idxs.size());
        assert(send_pos_idxs.size() == recv_pos_idxs.size());
        assert(send_pos_idxs2.size() == recv_pos_idxs2.size());
        assert(send_vel_idxs.size() == recv_vel_idxs.size());

        cilk_for (int i = 0; i < recv_force_idxs.size(); i++) {
            int recv_force_idx = recv_force_idxs[i];
            int send_force_idx = send_force_idxs[i];

            auto& recv_f = zoid.f_stencil_md[0][recv_force_idx];
            auto& send_f = recv_zoid.f_stencil_md[0][send_force_idx];

            assert(recv_zoid.tag_stencil_md[0][send_force_idx] == zoid.tag_stencil_md[0][recv_force_idx]);
            recv_f.x += send_f.x;
            recv_f.y += send_f.y;
            recv_f.z += send_f.z;

            send_f.x = 0;
            send_f.y = 0;
            send_f.z = 0;
        }

        cilk_for (int i = 0; i < recv_pos_idxs.size(); i++) {
            int recv_pos_idx = recv_pos_idxs[i];
            int send_pos_idx = send_pos_idxs[i];

            auto& recv_pos = zoid.x_stencil_md[0][recv_pos_idx];
            auto& send_pos = recv_zoid.x_stencil_md[0][send_pos_idx];

            assert(recv_zoid.tag_stencil_md[0][send_pos_idx] == zoid.tag_stencil_md[0][recv_pos_idx]);
            recv_pos.x = send_pos.x + pbc_flag_[0] * domain->prd[0];
            recv_pos.y = send_pos.y + pbc_flag_[1] * domain->prd[1];
            recv_pos.z = send_pos.z + pbc_flag_[2] * domain->prd[2];
        }

        cilk_for (int i = 0; i < recv_pos_idxs2.size(); i++) {
            int recv_pos_idx = recv_pos_idxs2[i];
            int send_pos_idx = send_pos_idxs2[i];

            auto& recv_pos = zoid.x_stencil_md[1][recv_pos_idx];
            auto& send_pos = recv_zoid.x_stencil_md[1][send_pos_idx];

            assert(recv_zoid.tag_stencil_md[0][send_pos_idx] == zoid.tag_stencil_md[0][recv_pos_idx]);
            recv_pos.x = send_pos.x + pbc_flag_[0] * domain->prd[0];
            recv_pos.y = send_pos.y + pbc_flag_[1] * domain->prd[1];
            recv_pos.z = send_pos.z + pbc_flag_[2] * domain->prd[2];
        }

        cilk_for (int i = 0; i < recv_vel_idxs.size(); i++) {
            int recv_vel_idx = recv_vel_idxs[i];
            int send_vel_idx = send_vel_idxs[i];

            auto& recv_vel = zoid.v_stencil_md[0][recv_vel_idx];
            auto& send_vel = recv_zoid.v_stencil_md[0][send_vel_idx];

            assert(recv_zoid.tag_stencil_md[0][send_vel_idx] == zoid.tag_stencil_md[0][recv_vel_idx]);
            recv_vel.x = send_vel.x;
            recv_vel.y = send_vel.y;
            recv_vel.z = send_vel.z;
        }
    }

    void UNPACK_DATA_MANY_CUTS_HELPER_SETUP(queue_info& zoid, double* buf, int recv_idx, int recv_zoid_num) {
        constexpr int start_t = 0;
        constexpr int end_t = 1;

        auto& recv_zoid = zoid_num_to_zoid_many_cuts[recv_zoid_num];

        int pbc_flag_[3] = {0};
        for (int dim = 0; dim < 3; dim++) {
            if (recv_zoid.where[dim] == NUM_ZOIDS_PER_DIMENSION - 1 && zoid.where[dim] == 0) {
                pbc_flag_[dim] = -1;
            }

            if (recv_zoid.where[dim] == 0 && zoid.where[dim] == NUM_ZOIDS_PER_DIMENSION - 1) {
                pbc_flag_[dim] = 1;
            }
        }

        int buf_idx = 0;

        for (int t = start_t; t < end_t; t++) {
            auto& recv_force_idxs = zoid.recv_force_idxs_double_buffering[t][recv_idx];

            if (DEBUG_SEND_RECV_DATA) {
                for (int k = 0; k < recv_force_idxs.size(); k++) {
                    int idx = recv_force_idxs[k];
                    auto target_tag = (tagint) ubuf(buf[buf_idx++]).i;
                    double f_x = buf[buf_idx++];
                    double f_y = buf[buf_idx++];
                    double f_z = buf[buf_idx++];

                    if (target_tag != zoid.tag_stencil_md[0][idx]) {
                        std::cout << "FORCE TAG WRONG me: " << comm->me
                                  << " idx: " << idx << " k: " << k
                                  << " my zoid: " << zoid.num << " my proc: " << zoid.num % comm->nprocs
                                  << " recv from zoid: " << recv_zoid_num << " time: " << t
                                  << " recv from proc: " << recv_zoid_num % comm->nprocs
                                  << " tag I got: " << target_tag << " tag I want: " << zoid.tag_stencil_md[0][idx]
                                  << " buf_idx: " << buf_idx
                                  << " buf: " << buf
                                  << std::endl;
                    }
                    assert(target_tag == zoid.tag_stencil_md[0][idx]);
                    zoid.f_stencil_md[0][idx].x += f_x;
                    zoid.f_stencil_md[0][idx].y += f_y;
                    zoid.f_stencil_md[0][idx].z += f_z;
                }
            } else {
                for (int k = 0; k < recv_force_idxs.size(); k++) {
                    int idx = recv_force_idxs[k];
                    double f_x = buf[buf_idx++];
                    double f_y = buf[buf_idx++];
                    double f_z = buf[buf_idx++];
                    zoid.f_stencil_md[0][idx].x += f_x;
                    zoid.f_stencil_md[0][idx].y += f_y;
                    zoid.f_stencil_md[0][idx].z += f_z;
                }
            }
        }
    }

    void UNPACK_DATA_MANY_CUTS_ZOID_SETUP(queue_info& zoid, std::vector<MPI_Request>& r) {
        constexpr int DEFAULT_PIPELINE_STAGE = 0;
        int zoid_num = zoid.num;
        auto& recv_neighbors = recv_from_neighbors_many_cuts[zoid_num];

        auto& not_my_proc_idxs = recv_from_neighbors_not_my_proc_idxs[zoid.num];

        int num_wait = 0;
        while (num_wait < not_my_proc_idxs.size()) {
            int idx;
            MPI_Waitany(r.size(), r.data(), &idx, MPI_STATUS_IGNORE);
            int recv_neighbor_idx = not_my_proc_idxs[idx];

            auto buf = buf_recv_zoid_to_zoid[DEFAULT_PIPELINE_STAGE][zoid_num][recv_neighbor_idx];

            int recv_zoid_num = recv_neighbors[recv_neighbor_idx];
            UNPACK_DATA_MANY_CUTS_HELPER_SETUP(zoid, buf, recv_neighbor_idx, recv_zoid_num);
            num_wait++;
        }

        for (int i = 0; i < recv_neighbors.size(); i++) {
            int recv_zoid_num = recv_neighbors[i];
            if (recv_zoid_num % comm->nprocs != comm->me) {
                continue;
            }

            auto &send_neighbors = send_to_neighbors_many_cuts[recv_zoid_num];
            auto find_it = std::find(send_neighbors.begin(), send_neighbors.end(), zoid_num);
            assert(find_it != send_neighbors.end());
            int find_idx = std::distance(send_neighbors.begin(), find_it);
            assert(send_neighbors[find_idx] == zoid_num);
            auto *buf = buf_send_zoid_to_zoid[DEFAULT_PIPELINE_STAGE][recv_zoid_num][find_idx];
            UNPACK_DATA_MANY_CUTS_HELPER_SETUP(zoid, buf, i, recv_zoid_num);
        }
    }

    template <bool curr_dt>
    int UNPACK_DATA_MANY_CUTS_ZOID_SELF_ONLY(queue_info& zoid, int start_t, int end_t) {
        constexpr int DEFAULT_PIPELINE_STAGE = 0;
        int zoid_num = zoid.num;
        auto& recv_neighbors = curr_dt ? recv_from_neighbors_many_cuts[zoid_num]
                                       : recv_from_neighbors_many_cuts_next_dt[zoid_num];

        int num_neighbors_recv = 0;

        for (int i = 0; i < recv_neighbors.size(); i++) {
            int recv_zoid_num = recv_neighbors[i];
            if (recv_zoid_num % comm->nprocs != comm->me) {
                continue;
            }

            auto &send_neighbors = curr_dt ? send_to_neighbors_many_cuts[recv_zoid_num]
                                           : send_to_neighbors_many_cuts_next_dt[recv_zoid_num];
            auto find_it = std::find(send_neighbors.begin(), send_neighbors.end(), zoid_num);
            assert(find_it != send_neighbors.end());
            int find_idx = std::distance(send_neighbors.begin(), find_it);
            assert(send_neighbors[find_idx] == zoid_num);
            auto *buf = buf_send_zoid_to_zoid[DEFAULT_PIPELINE_STAGE][recv_zoid_num][find_idx];
            UNPACK_DATA_MANY_CUTS_HELPER_SELF<curr_dt>(zoid, i, recv_zoid_num, find_idx, start_t, end_t);

            num_neighbors_recv++;
        }

        return num_neighbors_recv;
    }

    template <bool curr_dt>
    void UNPACK_FORCE_MANY_CUTS_ZOID(queue_info& zoid, int start_t, int end_t) {
        // cilk_scope {
        constexpr int DEFAULT_PIPELINE_STAGE = 0;
        int zoid_num = zoid.num;
        auto& recv_neighbors = curr_dt ? recv_from_neighbors_many_cuts[zoid_num]
                                       : recv_from_neighbors_many_cuts_next_dt[zoid_num];

        auto& not_my_proc_idxs = curr_dt ? recv_from_neighbors_not_my_proc_idxs[zoid.num]
                                         : recv_from_neighbors_not_my_proc_idxs_next_dt[zoid.num];

        for (int i = 0; i < recv_neighbors.size(); i++) {
            if (recv_neighbors[i] % comm->nprocs != comm->me) {
                auto buf = buf_recv_zoid_to_zoid[DEFAULT_PIPELINE_STAGE][zoid_num][i];
                int recv_zoid_num = recv_neighbors[i];
                UNPACK_FORCE_MANY_CUTS_HELPER<curr_dt>(zoid, buf, i,
                                                      recv_zoid_num, start_t, end_t);
            }
        }
    }

    template <bool curr_dt>
    void UNPACK_POS_VEL_MANY_CUTS_ZOID(queue_info& zoid, int recv_zoid_num, int start_t, int end_t) {
        constexpr int DEFAULT_PIPELINE_STAGE = 0;
        int zoid_num = zoid.num;
        auto& recv_neighbors = curr_dt ? recv_from_neighbors_many_cuts[zoid_num]
                                       : recv_from_neighbors_many_cuts_next_dt[zoid_num];

        auto find_it = std::find(recv_neighbors.begin(), recv_neighbors.end(), recv_zoid_num);
        assert(find_it != recv_neighbors.end());
        int find_idx = std::distance(recv_neighbors.begin(), find_it);

        auto buf = buf_recv_zoid_to_zoid[DEFAULT_PIPELINE_STAGE][zoid_num][find_idx];
        UNPACK_POS_VEL_MANY_CUTS_HELPER<curr_dt>(zoid, buf, find_idx, recv_zoid_num, start_t, end_t);
    }

    template <bool curr_dt>
    void UNPACK_DATA_MANY_CUTS_ZOID(queue_info& zoid, std::vector<MPI_Request>& r, int start_t, int end_t) {
        // cilk_scope {
        constexpr int DEFAULT_PIPELINE_STAGE = 0;
        int zoid_num = zoid.num;
        auto& recv_neighbors = curr_dt ? recv_from_neighbors_many_cuts[zoid_num]
        : recv_from_neighbors_many_cuts_next_dt[zoid_num];

        for (int i = 0; i < recv_neighbors.size(); i++) {
            int recv_zoid_num = recv_neighbors[i];
            if (recv_zoid_num % comm->nprocs != comm->me) {
                continue;
            }

            auto &send_neighbors = curr_dt ? send_to_neighbors_many_cuts[recv_zoid_num]
                                           : send_to_neighbors_many_cuts_next_dt[recv_zoid_num];
            auto find_it = std::find(send_neighbors.begin(), send_neighbors.end(), zoid_num);
            assert(find_it != send_neighbors.end());
            int find_idx = std::distance(send_neighbors.begin(), find_it);
            assert(send_neighbors[find_idx] == zoid_num);
            auto *buf = buf_send_zoid_to_zoid[DEFAULT_PIPELINE_STAGE][recv_zoid_num][find_idx];
            // UNPACK_DATA_MANY_CUTS_HELPER<curr_dt>(zoid, buf, i, recv_zoid_num, start_t, end_t);
            UNPACK_DATA_MANY_CUTS_HELPER_SELF<curr_dt>(zoid, i, recv_zoid_num, find_idx, start_t, end_t);
        }

        auto& not_my_proc_idxs = curr_dt ? recv_from_neighbors_not_my_proc_idxs[zoid.num]
                        : recv_from_neighbors_not_my_proc_idxs_next_dt[zoid.num];

        int num_wait = 0;
        while (num_wait < not_my_proc_idxs.size()) {
            int idx;
            MPI_Waitany(r.size(), r.data(), &idx, MPI_STATUS_IGNORE);

            int recv_neighbor_idx = not_my_proc_idxs[idx];

            auto buf = buf_recv_zoid_to_zoid[DEFAULT_PIPELINE_STAGE][zoid_num][recv_neighbor_idx];

            int recv_zoid_num = recv_neighbors[recv_neighbor_idx];
            UNPACK_DATA_MANY_CUTS_HELPER<curr_dt>(zoid, buf, recv_neighbor_idx,
                                                  recv_zoid_num, start_t, end_t);
            num_wait++;
        }

        /*
        int wait_idx = 0;
        for (int i = 0; i < recv_neighbors.size(); i++) {
            int recv_zoid_num = recv_neighbors[i];
            double* buf;

            int num_recv_from_zoid = curr_dt ? recv_zoid_to_zoid_sizes[zoid_num][i]
                    : recv_zoid_to_zoid_sizes_next_dt[zoid_num][i];

            assert(num_recv_from_zoid >= 0);

            if (num_recv_from_zoid == 0) {
                continue;
            }

            if (recv_zoid_num % comm->nprocs != comm->me) {
                buf = curr_dt ? buf_recv_zoid_to_zoid[zoid_num][i]
                              : buf_recv_zoid_to_zoid_next_dt[zoid_num][i];
                int recv_size = curr_dt ? recv_zoid_to_zoid_sizes[zoid_num][i]
                                        : recv_zoid_to_zoid_sizes_next_dt[zoid_num][i];
                int total_doubles_recv_from_zoid = DEBUG_SEND_RECV_DATA ? recv_size * (3 + 1) : recv_size * 3;
                int mpi_tag = get_mpi_tag_many_cuts(zoid_num, recv_zoid_num);
                MPI_Wait(&r[wait_idx++], MPI_STATUS_IGNORE);
                buf = curr_dt ? buf_recv_zoid_to_zoid[zoid_num][i]
                        : buf_recv_zoid_to_zoid_next_dt[zoid_num][i];
            } else {
                auto& send_neighbors = curr_dt ? send_to_neighbors_many_cuts[recv_zoid_num]
                                               : send_to_neighbors_many_cuts_next_dt[recv_zoid_num];
                auto find_it = std::find(send_neighbors.begin(), send_neighbors.end(), zoid_num);
                assert(find_it != send_neighbors.end());
                int find_idx = std::distance(send_neighbors.begin(), find_it);
                assert(send_neighbors[find_idx] == zoid_num);
                buf = curr_dt ? buf_send_zoid_to_zoid[recv_zoid_num][find_idx]
                        : buf_send_zoid_to_zoid_next_dt[recv_zoid_num][find_idx];
            }

            UNPACK_DATA_MANY_CUTS_HELPER<curr_dt, is_initial>(zoid, buf, i, recv_zoid_num, start_t, end_t);
        }
        */
// }
    }

    template <bool curr_dt, bool is_initial>
    void UNPACK_DATA_MANY_CUTS_ZOID_PIPELINED(queue_info& zoid, std::vector<MPI_Request>& r,
                                              int start_t, int end_t, int pipeline_stage) {
        int zoid_num = zoid.num;
        auto& recv_neighbors = curr_dt ? recv_from_neighbors_many_cuts[zoid_num]
                                       : recv_from_neighbors_many_cuts_next_dt[zoid_num];

        auto& not_my_proc_idxs = curr_dt ? recv_from_neighbors_not_my_proc_idxs[zoid.num]
                                         : recv_from_neighbors_not_my_proc_idxs_next_dt[zoid.num];

        int num_wait = 0;
        while (num_wait < not_my_proc_idxs.size()) {
            int idx;
            MPI_Waitany(r.size(), r.data(), &idx, MPI_STATUS_IGNORE);

            int recv_neighbor_idx = not_my_proc_idxs[idx];

            auto buf = buf_recv_zoid_to_zoid[pipeline_stage][zoid_num][recv_neighbor_idx];

            int recv_zoid_num = recv_neighbors[recv_neighbor_idx];
            UNPACK_DATA_MANY_CUTS_HELPER<curr_dt>(zoid, buf, recv_neighbor_idx,
                                                  recv_zoid_num, start_t, end_t);
            num_wait++;
        }

        for (int i = 0; i < recv_neighbors.size(); i++) {
            int recv_zoid_num = recv_neighbors[i];
            if (recv_zoid_num % comm->nprocs != comm->me) {
                continue;
            }

            auto &send_neighbors = curr_dt ? send_to_neighbors_many_cuts[recv_zoid_num]
                                           : send_to_neighbors_many_cuts_next_dt[recv_zoid_num];
            auto find_it = std::find(send_neighbors.begin(), send_neighbors.end(), zoid_num);
            assert(find_it != send_neighbors.end());
            int find_idx = std::distance(send_neighbors.begin(), find_it);
            assert(send_neighbors[find_idx] == zoid_num);
            auto *buf = buf_send_zoid_to_zoid[pipeline_stage][recv_zoid_num][find_idx];
            UNPACK_DATA_MANY_CUTS_HELPER<curr_dt>(zoid, buf, i, recv_zoid_num, start_t, end_t);
        }
    }

    template <bool curr_dt, bool is_initial>
    void UNPACK_DATA_MANY_CUTS(int dep, int proc, int start_t, int end_t) {
        auto& queues = curr_dt ? queues_many_cuts : queues_many_cuts_next_dt;
        auto& zoids_affected = curr_dt ? dep_proc_to_recv_zoids[dep][proc]
                : dep_proc_to_recv_zoids_next_dt[dep][proc];

        #pragma cilk grainsize 1
        cilk_for (int i = 0; i < zoids_affected.size(); i++) {
            int recv_zoid_num = zoids_affected[i];
            assert(recv_zoid_num % comm->nprocs == comm->me);

            auto& recv_zoid = curr_dt ? zoid_num_to_zoid_many_cuts[recv_zoid_num]
                                      : zoid_num_to_zoid_many_cuts_next_dt[recv_zoid_num];

            auto& find_idxs_per_zoid = curr_dt ? dep_proc_recv_zoid_to_find_idxs[dep][proc][recv_zoid_num]
                    : dep_proc_recv_zoid_to_find_idxs_next_dt[dep][proc][recv_zoid_num];

            auto& recv_neighbors = curr_dt ? recv_from_neighbors_many_cuts[recv_zoid_num]
                    : recv_from_neighbors_many_cuts_next_dt[recv_zoid_num];

            for (int j = 0; j < find_idxs_per_zoid.size(); j++) {
                int find_idx = find_idxs_per_zoid[j];
                int send_zoid_num = recv_neighbors[find_idx];

                auto& send_zoid = curr_dt ? zoid_num_to_zoid_many_cuts[send_zoid_num]
                        : zoid_num_to_zoid_many_cuts_next_dt[send_zoid_num];

                int pbc_flag_[3] = {0};
                for (int dim = 0; dim < 3; dim++) {
                    if (send_zoid.where[dim] == NUM_ZOIDS_PER_DIMENSION - 1 && recv_zoid.where[dim] == 0) {
                        pbc_flag_[dim] = -1;
                    }

                    if (send_zoid.where[dim] == 0 && recv_zoid.where[dim] == NUM_ZOIDS_PER_DIMENSION - 1) {
                        pbc_flag_[dim] = 1;
                    }
                }

                int offset = curr_dt ? recv_proc_zoid_offsets[recv_zoid_num][find_idx][0]
                                     : recv_proc_zoid_offsets_next_dt[recv_zoid_num][find_idx][0];
                offset = DEBUG_SEND_RECV_DATA ? offset * (3 + 1) : offset * 3;

                int buf_idx = 0;
                double* buf;
                if (proc != comm->me) {
                    // auto* buf = &buf_recv_many_cuts[dep][proc][offset];
                    buf = &buf_recv_many_cuts[dep][proc][offset];
                } else {
                    buf = &buf_send_many_cuts[dep - 1][proc][offset];
                }

                for (int t = start_t; t < end_t; t++) {
                    auto& recv_force_idxs = recv_zoid.recv_force_idxs_double_buffering[t][find_idx];
                    auto& recv_pos_idxs = recv_zoid.recv_pos_idxs_double_buffering[t][find_idx];
                    auto& recv_vel_idxs = recv_zoid.recv_vel_idxs_double_buffering[t][find_idx];

                    if (DEBUG_SEND_RECV_DATA) {
                        for (int k = 0; k < recv_force_idxs.size(); k++) {
                            int idx = recv_force_idxs[k];
                            auto target_tag = (tagint) ubuf(buf[buf_idx++]).i;
                            double f_x = buf[buf_idx++];
                            double f_y = buf[buf_idx++];
                            double f_z = buf[buf_idx++];
                            if (target_tag != recv_zoid.tag_stencil_md[0][idx]) {
                                std::cout << "FORCE TAG WRONG me: " << comm->me << " my zoid: " << recv_zoid.num << " dep: " << dep
                                          << " recv from zoid: " << send_zoid_num << " time: " << t
                                          << " recv from proc: " << proc << " tag I got: " << target_tag << " tag I want: " << recv_zoid.tag_stencil_md[0][idx]
                                          << " offset: " << offset << " buf value: " << " find idx: " << find_idx
                                          << " buf: " << buf
                                          << std::endl;
                            }
                            assert(target_tag == recv_zoid.tag_stencil_md[0][idx]);
                            if (!is_initial && t == 0) {
                                continue;
                            }
                            if (is_initial && t > 0) {
                                continue;
                            }
                            recv_zoid.f_stencil_md[0][idx].x += f_x;
                            recv_zoid.f_stencil_md[0][idx].y += f_y;
                            recv_zoid.f_stencil_md[0][idx].z += f_z;
                        }

                        for (int k = 0; k < recv_pos_idxs.size(); k++) {
                            int idx = recv_pos_idxs[k];
                            auto target_tag = (tagint) ubuf(buf[buf_idx++]).i;
                            double x_x = buf[buf_idx++];
                            double x_y = buf[buf_idx++];
                            double x_z = buf[buf_idx++];
                            assert(target_tag == recv_zoid.tag_stencil_md[0][idx]);
                            if (target_tag != recv_zoid.tag_stencil_md[0][idx]) {
                                std::cout << "POS me: " << comm->me << " my zoid: " << recv_zoid.num
                                          << " recv from: " << send_zoid_num << " time: " << t
                                          << " recv proc: " << proc << " tag I got: " << target_tag << " tag I want: " << recv_zoid.tag_stencil_md[0][idx]
                                          << " offset: " << offset
                                          << std::endl;
                            }
                            if (!is_initial && t == 0) {
                                continue;
                            }
                            if (is_initial && t > 0) {
                                continue;
                            }
                            recv_zoid.x_stencil_md[t % DOUBLE_BUFFERING][idx].x = x_x + pbc_flag_[0] * domain->prd[0];
                            recv_zoid.x_stencil_md[t % DOUBLE_BUFFERING][idx].y = x_y + pbc_flag_[1] * domain->prd[1];
                            recv_zoid.x_stencil_md[t % DOUBLE_BUFFERING][idx].z = x_z + pbc_flag_[2] * domain->prd[2];
                        }

                        for (int k = 0; k < recv_vel_idxs.size(); k++) {
                            int idx = recv_vel_idxs[k];
                            auto target_tag = (tagint) ubuf(buf[buf_idx++]).i;
                            double v_x = buf[buf_idx++];
                            double v_y = buf[buf_idx++];
                            double v_z = buf[buf_idx++];
                            assert(target_tag == recv_zoid.tag_stencil_md[0][idx]);
                            if (!is_initial && t == 0) {
                                continue;
                            }
                            if (is_initial && t > 0) {
                                continue;
                            }
                            recv_zoid.v_stencil_md[0][idx].x = v_x;
                            recv_zoid.v_stencil_md[0][idx].y = v_y;
                            recv_zoid.v_stencil_md[0][idx].z = v_z;
                        }
                    } else {
                        for (int k = 0; k < recv_force_idxs.size(); k++) {
                            int idx = recv_force_idxs[k];
                            double f_x = buf[buf_idx++];
                            double f_y = buf[buf_idx++];
                            double f_z = buf[buf_idx++];
                            if (!is_initial && t == 0) {
                                continue;
                            }
                            if (is_initial && t > 0) {
                                continue;
                            }
                            recv_zoid.f_stencil_md[0][idx].x += f_x;
                            recv_zoid.f_stencil_md[0][idx].y += f_y;
                            recv_zoid.f_stencil_md[0][idx].z += f_z;
                        }

                        for (int k = 0; k < recv_pos_idxs.size(); k++) {
                            int idx = recv_pos_idxs[k];
                            double x_x = buf[buf_idx++];
                            double x_y = buf[buf_idx++];
                            double x_z = buf[buf_idx++];
                            if (!is_initial && t == 0) {
                                continue;
                            }
                            if (is_initial && t > 0) {
                                continue;
                            }
                            recv_zoid.x_stencil_md[t % DOUBLE_BUFFERING][idx].x = x_x + pbc_flag_[0] * domain->prd[0];
                            recv_zoid.x_stencil_md[t % DOUBLE_BUFFERING][idx].y = x_y + pbc_flag_[1] * domain->prd[1];
                            recv_zoid.x_stencil_md[t % DOUBLE_BUFFERING][idx].z = x_z + pbc_flag_[2] * domain->prd[2];
                        }

                        for (int k = 0; k < recv_vel_idxs.size(); k++) {
                            int idx = recv_vel_idxs[k];
                            double v_x = buf[buf_idx++];
                            double v_y = buf[buf_idx++];
                            double v_z = buf[buf_idx++];
                            if (!is_initial && t == 0) {
                                continue;
                            }
                            if (is_initial && t > 0) {
                                continue;
                            }
                            recv_zoid.v_stencil_md[0][idx].x = v_x;
                            recv_zoid.v_stencil_md[0][idx].y = v_y;
                            recv_zoid.v_stencil_md[0][idx].z = v_z;
                        }
                    }
                }
            }
        }

        return;

        /*
        int send_dep = dep - 1;

        for (int j = 0; j < queues[send_dep].size(); j++) {
            auto& send_zoid = queues[send_dep][j];
            int send_zoid_num = send_zoid.num;
            if (send_zoid_num % comm->nprocs != proc) {
                continue;
            }

            auto& send_neighbors = curr_dt ? send_to_neighbors_many_cuts[send_zoid_num]
                    : send_to_neighbors_many_cuts_next_dt[send_zoid_num];

            for (int i = 0; i < send_neighbors.size(); i++) {
                int recv_zoid_num = send_neighbors[i];
                if (recv_zoid_num % comm->nprocs != comm->me) {
                    continue;
                }

                auto& recv_zoid = curr_dt ? zoid_num_to_zoid_many_cuts[recv_zoid_num]
                        : zoid_num_to_zoid_many_cuts_next_dt[recv_zoid_num];

                int pbc_flag_[3] = {0};
                for (int dim = 0; dim < 3; dim++) {
                    if (send_zoid.where[dim] == NUM_ZOIDS_PER_DIMENSION - 1 && recv_zoid.where[dim] == 0) {
                        pbc_flag_[dim] = -1;
                    }

                    if (send_zoid.where[dim] == 0 && recv_zoid.where[dim] == NUM_ZOIDS_PER_DIMENSION - 1) {
                        pbc_flag_[dim] = 1;
                    }
                }

                auto& recv_neighbors = curr_dt ? recv_from_neighbors_many_cuts[recv_zoid_num]
                        : recv_from_neighbors_many_cuts_next_dt[recv_zoid_num];
                auto find_it = std::find(recv_neighbors.begin(), recv_neighbors.end(), send_zoid_num);
                assert(find_it != recv_neighbors.end());
                int find_idx = std::distance(recv_neighbors.begin(), find_it);
                int offset = curr_dt ? recv_proc_zoid_offsets[recv_zoid_num][find_idx][0]
                        : recv_proc_zoid_offsets_next_dt[recv_zoid_num][find_idx][0];
                offset = DEBUG_SEND_RECV_DATA ? offset * (3 + 1) : offset * 3;

                int buf_idx = 0;
                auto* buf = &buf_recv_many_cuts[dep][proc][offset];

                for (int t = start_t; t < end_t; t++) {
                    auto& recv_force_idxs = recv_zoid.recv_force_idxs_double_buffering[t][find_idx];
                    auto& recv_pos_idxs = recv_zoid.recv_pos_idxs_double_buffering[t][find_idx];
                    auto& recv_vel_idxs = recv_zoid.recv_vel_idxs_double_buffering[t][find_idx];

                    assert(DEBUG_SEND_RECV_DATA);

                    for (int k = 0; k < recv_force_idxs.size(); k++) {
                        int idx = recv_force_idxs[k];
                        double tmp = buf[buf_idx];
                        auto target_tag = (tagint) ubuf(buf[buf_idx++]).i;
                        double f_x = buf[buf_idx++];
                        double f_y = buf[buf_idx++];
                        double f_z = buf[buf_idx++];
                        if (target_tag != recv_zoid.tag_stencil_md[0][idx]) {
                            std::cout << "FORCE TAG WRONG me: " << comm->me << " my zoid: " << recv_zoid.num << " dep: " << dep
                                      << " recv from zoid: " << send_zoid_num << " time: " << t
                                      << " recv from proc: " << proc << " tag I got: " << target_tag << " tag I want: " << recv_zoid.tag_stencil_md[0][idx]
                                      << " offset: " << offset << " buf value: " << tmp << " find idx: " << find_idx
                                      << " buf: " << buf
                                      << std::endl;
                        }
                        assert(target_tag == recv_zoid.tag_stencil_md[0][idx]);
                        if (!is_initial && t == 0) {
                            continue;
                        }
                        if (is_initial && t > 0) {
                            continue;
                        }
                        recv_zoid.f_stencil_md[0][idx].x += f_x;
                        recv_zoid.f_stencil_md[0][idx].y += f_y;
                        recv_zoid.f_stencil_md[0][idx].z += f_z;
                    }

                    for (int k = 0; k < recv_pos_idxs.size(); k++) {
                        int idx = recv_pos_idxs[k];
                        auto target_tag = (tagint) ubuf(buf[buf_idx++]).i;
                        double x_x = buf[buf_idx++];
                        double x_y = buf[buf_idx++];
                        double x_z = buf[buf_idx++];
                        assert(target_tag == recv_zoid.tag_stencil_md[0][idx]);
                        if (target_tag != recv_zoid.tag_stencil_md[0][idx]) {
                            std::cout << "POS me: " << comm->me << " my zoid: " << recv_zoid.num
                                      << " recv from: " << send_zoid_num << " time: " << t
                                      << " recv proc: " << proc << " tag I got: " << target_tag << " tag I want: " << recv_zoid.tag_stencil_md[0][idx]
                                      << " offset: " << offset
                                      << std::endl;
                        }
                        if (!is_initial && t == 0) {
                            continue;
                        }
                        if (is_initial && t > 0) {
                            continue;
                        }
                        recv_zoid.x_stencil_md[t % DOUBLE_BUFFERING][idx].x = x_x + pbc_flag_[0] * domain->prd[0];
                        recv_zoid.x_stencil_md[t % DOUBLE_BUFFERING][idx].y = x_y + pbc_flag_[1] * domain->prd[1];
                        recv_zoid.x_stencil_md[t % DOUBLE_BUFFERING][idx].z = x_z + pbc_flag_[2] * domain->prd[2];
                    }

                    for (int k = 0; k < recv_vel_idxs.size(); k++) {
                        int idx = recv_vel_idxs[k];
                        auto target_tag = (tagint) ubuf(buf[buf_idx++]).i;
                        double v_x = buf[buf_idx++];
                        double v_y = buf[buf_idx++];
                        double v_z = buf[buf_idx++];
                        assert(target_tag == recv_zoid.tag_stencil_md[0][idx]);
                        if (!is_initial && t == 0) {
                            continue;
                        }
                        if (is_initial && t > 0) {
                            continue;
                        }
                        recv_zoid.v_stencil_md[0][idx].x = v_x;
                        recv_zoid.v_stencil_md[0][idx].y = v_y;
                        recv_zoid.v_stencil_md[0][idx].z = v_z;
                    }
                }
            }
        }
        */
    }

    template <bool curr_dt, bool is_initial>
    int PACK_DATA_TO_PROC_HELPER(queue_info& zoid, int proc, double* buf, int offset, int start_t, int end_t) {
        auto& send_neighbors = curr_dt ? send_to_neighbors_many_cuts[zoid.num]
                : send_to_neighbors_many_cuts_next_dt[zoid.num];

        int buf_idx = 0;

        for (int i = 0; i < send_neighbors.size(); i++) {
            if (send_neighbors[i] % comm->nprocs != proc) {
                continue;
            }

            for (int t = start_t; t < end_t; t++) {
                auto& send_force_idxs = zoid.send_force_idxs_double_buffering[t][i];
                auto& send_pos_idxs = zoid.send_pos_idxs_double_buffering[t][i];
                auto& send_vel_idxs = zoid.send_vel_idxs_double_buffering[t][i];

                if (DEBUG_SEND_RECV_DATA) {
                    for (int k = 0; k < send_force_idxs.size(); k++) {
                        int idx = send_force_idxs[k];
                        int tag = zoid.tag_stencil_md[0][idx];

                        buf[buf_idx++] = ubuf(tag).d;
                        buf[buf_idx++] = zoid.f_stencil_md[0][idx].x;
                        buf[buf_idx++] = zoid.f_stencil_md[0][idx].y;
                        buf[buf_idx++] = zoid.f_stencil_md[0][idx].z;

                        if (!is_initial) {
                            zoid.f_stencil_md[0][idx].x = 0;
                            zoid.f_stencil_md[0][idx].y = 0;
                            zoid.f_stencil_md[0][idx].z = 0;
                        }
                    }

                    for (int k = 0; k < send_pos_idxs.size(); k++) {
                        int idx = send_pos_idxs[k];
                        int tag = zoid.tag_stencil_md[0][idx];

                        buf[buf_idx++] = ubuf(tag).d;
                        buf[buf_idx++] = zoid.x_stencil_md[t % DOUBLE_BUFFERING][idx].x;
                        buf[buf_idx++] = zoid.x_stencil_md[t % DOUBLE_BUFFERING][idx].y;
                        buf[buf_idx++] = zoid.x_stencil_md[t % DOUBLE_BUFFERING][idx].z;
                    }

                    for (int k = 0; k < send_vel_idxs.size(); k++) {
                        int idx = send_vel_idxs[k];
                        int tag = zoid.tag_stencil_md[0][idx];

                        buf[buf_idx++] = ubuf(tag).d;
                        buf[buf_idx++] = zoid.v_stencil_md[0][idx].x;
                        buf[buf_idx++] = zoid.v_stencil_md[0][idx].y;
                        buf[buf_idx++] = zoid.v_stencil_md[0][idx].z;
                    }
                } else {
                    for (int k = 0; k < send_force_idxs.size(); k++) {
                        int idx = send_force_idxs[k];

                        buf[buf_idx++] = zoid.f_stencil_md[0][idx].x;
                        buf[buf_idx++] = zoid.f_stencil_md[0][idx].y;
                        buf[buf_idx++] = zoid.f_stencil_md[0][idx].z;

                        if (!is_initial) {
                            zoid.f_stencil_md[0][idx].x = 0;
                            zoid.f_stencil_md[0][idx].y = 0;
                            zoid.f_stencil_md[0][idx].z = 0;
                        }
                    }

                    for (int k = 0; k < send_pos_idxs.size(); k++) {
                        int idx = send_pos_idxs[k];

                        buf[buf_idx++] = zoid.x_stencil_md[t % DOUBLE_BUFFERING][idx].x;
                        buf[buf_idx++] = zoid.x_stencil_md[t % DOUBLE_BUFFERING][idx].y;
                        buf[buf_idx++] = zoid.x_stencil_md[t % DOUBLE_BUFFERING][idx].z;
                    }

                    for (int k = 0; k < send_vel_idxs.size(); k++) {
                        int idx = send_vel_idxs[k];

                        buf[buf_idx++] = zoid.v_stencil_md[0][idx].x;
                        buf[buf_idx++] = zoid.v_stencil_md[0][idx].y;
                        buf[buf_idx++] = zoid.v_stencil_md[0][idx].z;
                    }
                }
            }
        }

        return buf_idx;
    }

    template <bool curr_dt>
    int PACK_DATA_MANY_CUTS_HELPER(queue_info& zoid, double* buf,
                                   int send_idx, int send_zoid_num,
                                   int start_t, int end_t) {

        auto& send_force_idxs = zoid.send_force_idxs_double_buffering_flattened[send_idx];
        auto& send_pos_idxs = zoid.send_pos_idxs_double_buffering_flattened[0][send_idx];
        auto& send_pos_idxs2 = zoid.send_pos_idxs_double_buffering_flattened[1][send_idx];
        auto& send_vel_idxs = zoid.send_vel_idxs_double_buffering_flattened[send_idx];

        int num_send_force = send_force_idxs.size();
        int num_send_pos = send_pos_idxs.size();
        int num_send_pos2 = send_pos_idxs2.size();
        int num_send_vel = send_vel_idxs.size();

        if (DEBUG_SEND_RECV_DATA) {
            for (int i = 0; i < send_force_idxs.size(); i++) {
                int idx = send_force_idxs[i];
                int buf_idx = i * (3 + 1);
                int tag = zoid.tag_stencil_md[0][idx];

                buf[buf_idx] = ubuf(tag).d;
                buf[buf_idx + 1] = zoid.f_stencil_md[0][idx].x;
                buf[buf_idx + 2] = zoid.f_stencil_md[0][idx].y;
                buf[buf_idx + 3] = zoid.f_stencil_md[0][idx].z;

                zoid.f_stencil_md[0][idx].x = 0;
                zoid.f_stencil_md[0][idx].y = 0;
                zoid.f_stencil_md[0][idx].z = 0;
            }

            int pos_starting_idx = num_send_force * (3 + 1);
            for (int i = 0; i < send_pos_idxs.size(); i++) {
                int idx = send_pos_idxs[i];
                int buf_idx = pos_starting_idx + i * (3 + 1);
                int tag = zoid.tag_stencil_md[0][idx];

                buf[buf_idx] = ubuf(tag).d;
                buf[buf_idx + 1] = zoid.x_stencil_md[0][idx].x;
                buf[buf_idx + 2] = zoid.x_stencil_md[0][idx].y;
                buf[buf_idx + 3] = zoid.x_stencil_md[0][idx].z;
            }

            int pos_starting_idx2 = (num_send_force + num_send_pos) * (3 + 1);
            for (int i = 0; i < send_pos_idxs2.size(); i++) {
                int idx = send_pos_idxs2[i];
                int buf_idx = pos_starting_idx2 + i * (3 + 1);
                int tag = zoid.tag_stencil_md[0][idx];

                buf[buf_idx] = ubuf(tag).d;
                buf[buf_idx + 1] = zoid.x_stencil_md[1][idx].x;
                buf[buf_idx + 2] = zoid.x_stencil_md[1][idx].y;
                buf[buf_idx + 3] = zoid.x_stencil_md[1][idx].z;
            }

            int vel_starting_idx = (num_send_force + num_send_pos + num_send_pos2) * (3 + 1);

            for (int i = 0; i < send_vel_idxs.size(); i++) {
                int idx = send_vel_idxs[i];
                int buf_idx = vel_starting_idx + i * (3 + 1);
                int tag = zoid.tag_stencil_md[0][idx];

                buf[buf_idx] = ubuf(tag).d;
                buf[buf_idx + 1] = zoid.v_stencil_md[0][idx].x;
                buf[buf_idx + 2] = zoid.v_stencil_md[0][idx].y;
                buf[buf_idx + 3] = zoid.v_stencil_md[0][idx].z;
            }

            return (num_send_force + num_send_pos + num_send_pos2 + num_send_vel) * (3 + 1);
        } else {
            #pragma cilk grainsize 2048
            cilk_for (int i = 0; i < send_force_idxs.size(); i++) {
                int idx = send_force_idxs[i];
                int buf_idx = i * 3;

                buf[buf_idx] = zoid.f_stencil_md[0][idx].x;
                buf[buf_idx + 1] = zoid.f_stencil_md[0][idx].y;
                buf[buf_idx + 2] = zoid.f_stencil_md[0][idx].z;

                zoid.f_stencil_md[0][idx].x = 0;
                zoid.f_stencil_md[0][idx].y = 0;
                zoid.f_stencil_md[0][idx].z = 0;
            }

            int pos_starting_idx = num_send_force * 3;
            #pragma cilk grainsize 2048
            cilk_for (int i = 0; i < send_pos_idxs.size(); i++) {
                int idx = send_pos_idxs[i];
                int buf_idx = pos_starting_idx + i * 3;

                buf[buf_idx] = zoid.x_stencil_md[0][idx].x;
                buf[buf_idx + 1] = zoid.x_stencil_md[0][idx].y;
                buf[buf_idx + 2] = zoid.x_stencil_md[0][idx].z;
            }

            int pos_starting_idx2 = (num_send_force + num_send_pos) * 3;
            #pragma cilk grainsize 2048
            cilk_for (int i = 0; i < send_pos_idxs2.size(); i++) {
                int idx = send_pos_idxs2[i];
                int buf_idx = pos_starting_idx2 + i * 3;

                buf[buf_idx] = zoid.x_stencil_md[1][idx].x;
                buf[buf_idx + 1] = zoid.x_stencil_md[1][idx].y;
                buf[buf_idx + 2] = zoid.x_stencil_md[1][idx].z;
            }

            int vel_starting_idx = (num_send_force + num_send_pos + num_send_pos2) * 3;

            #pragma cilk grainsize 2048
            cilk_for (int i = 0; i < send_vel_idxs.size(); i++) {
                int idx = send_vel_idxs[i];
                int buf_idx = vel_starting_idx + i * 3;

                buf[buf_idx] = zoid.v_stencil_md[0][idx].x;
                buf[buf_idx + 1] = zoid.v_stencil_md[0][idx].y;
                buf[buf_idx + 2] = zoid.v_stencil_md[0][idx].z;
            }

            return (num_send_force + num_send_pos + num_send_pos2 + num_send_vel) * 3;
        }

        return -1;

        /*
        int buf_idx = 0;
        for (int t = start_t; t < end_t; t++) {
            auto &send_force_idxs = zoid.send_force_idxs_double_buffering[t][send_idx];
            auto &send_pos_idxs = zoid.send_pos_idxs_double_buffering[t][send_idx];
            auto &send_vel_idxs = zoid.send_vel_idxs_double_buffering[t][send_idx];

            if (DEBUG_SEND_RECV_DATA) {
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
                    buf[buf_idx++] = zoid.x_stencil_md[t % DOUBLE_BUFFERING][idx].x;
                    buf[buf_idx++] = zoid.x_stencil_md[t % DOUBLE_BUFFERING][idx].y;
                    buf[buf_idx++] = zoid.x_stencil_md[t % DOUBLE_BUFFERING][idx].z;
                }

                for (int k = 0; k < send_vel_idxs.size(); k++) {
                    int idx = send_vel_idxs[k];
                    int tag = zoid.tag_stencil_md[0][idx];

                    buf[buf_idx++] = ubuf(tag).d;
                    buf[buf_idx++] = zoid.v_stencil_md[0][idx].x;
                    buf[buf_idx++] = zoid.v_stencil_md[0][idx].y;
                    buf[buf_idx++] = zoid.v_stencil_md[0][idx].z;
                }
            } else {
                for (int k = 0; k < send_force_idxs.size(); k++) {
                    int idx = send_force_idxs[k];

                    buf[buf_idx++] = zoid.f_stencil_md[0][idx].x;
                    buf[buf_idx++] = zoid.f_stencil_md[0][idx].y;
                    buf[buf_idx++] = zoid.f_stencil_md[0][idx].z;

                    zoid.f_stencil_md[0][idx].x = 0;
                    zoid.f_stencil_md[0][idx].y = 0;
                    zoid.f_stencil_md[0][idx].z = 0;
                }

                for (int k = 0; k < send_pos_idxs.size(); k++) {
                    int idx = send_pos_idxs[k];

                    buf[buf_idx++] = zoid.x_stencil_md[t % DOUBLE_BUFFERING][idx].x;
                    buf[buf_idx++] = zoid.x_stencil_md[t % DOUBLE_BUFFERING][idx].y;
                    buf[buf_idx++] = zoid.x_stencil_md[t % DOUBLE_BUFFERING][idx].z;
                }

                for (int k = 0; k < send_vel_idxs.size(); k++) {
                    int idx = send_vel_idxs[k];

                    buf[buf_idx++] = zoid.v_stencil_md[0][idx].x;
                    buf[buf_idx++] = zoid.v_stencil_md[0][idx].y;
                    buf[buf_idx++] = zoid.v_stencil_md[0][idx].z;
                }
            }
        }

        return buf_idx;
        */
    }

    int PACK_DATA_MANY_CUTS_HELPER_SETUP(queue_info& zoid, double* buf,
                                         int send_idx, int send_zoid_num) {
        constexpr int start_t = 0;
        constexpr int end_t = 1;

        int buf_idx = 0;
        for (int t = start_t; t < end_t; t++) {
            auto &send_force_idxs = zoid.send_force_idxs_double_buffering[t][send_idx];

            if (DEBUG_SEND_RECV_DATA) {
                for (int k = 0; k < send_force_idxs.size(); k++) {
                    int idx = send_force_idxs[k];
                    int tag = zoid.tag_stencil_md[0][idx];

                    buf[buf_idx++] = ubuf(tag).d;
                    buf[buf_idx++] = zoid.f_stencil_md[0][idx].x;
                    buf[buf_idx++] = zoid.f_stencil_md[0][idx].y;
                    buf[buf_idx++] = zoid.f_stencil_md[0][idx].z;
                }
            } else {
                for (int k = 0; k < send_force_idxs.size(); k++) {
                    int idx = send_force_idxs[k];

                    buf[buf_idx++] = zoid.f_stencil_md[0][idx].x;
                    buf[buf_idx++] = zoid.f_stencil_md[0][idx].y;
                    buf[buf_idx++] = zoid.f_stencil_md[0][idx].z;
                }
            }
        }

        return buf_idx;
    }

    void INITIAL_INTEGRATE_ZOID_MANY_CUTS(queue_info& zoid, int dep, int timestep) {
        auto * _noalias x = zoid.x_stencil_md[timestep % DOUBLE_BUFFERING].data();
        auto * _noalias next_x = zoid.x_stencil_md[(timestep + 1) % DOUBLE_BUFFERING].data();

        auto * _noalias v = zoid.v_stencil_md[timestep % 1].data();
        auto * _noalias f = zoid.f_stencil_md[timestep % 1].data();

        auto * _noalias mask = zoid.mask_stencil_md[0].data();
        auto * _noalias local_idxs = zoid.local_idxs_per_timestep[timestep].data();
        auto * _noalias type = zoid.type_stencil_md[0].data();

        int nlocal = zoid.local_idxs_per_timestep[timestep].size();

        auto dtv = update->dt;
        auto* mass = atom->mass;
        auto dtf = 0.5 * update->dt * force->ftm2v;

        // if ((dep == 0 || dep == NUM_DEPS - 1) && nlocal > MODIFY_GRAINSIZE) {
        if (nlocal > MODIFY_GRAINSIZE) {
            auto* claimed = zoid.claimed_flags_stencil_md[0];
            int num_workers = __cilkrts_get_nworkers();
            int num_chunks = nlocal / MODIFY_GRAINSIZE + 1;
            int chunks_per_worker = num_chunks / num_workers;
            int chunk_size = MODIFY_GRAINSIZE;

            /*
            #pragma cilk grainsize MODIFY_GRAINSIZE
            cilk_for (int idx = 0; idx < nlocal; idx++) {
                int i = local_idxs[idx];

                const double dtfm = dtf / mass[type[i]];
                v[i].x += dtfm * f[i].x;
                v[i].y += dtfm * f[i].y;
                v[i].z += dtfm * f[i].z;

                f[i].x = 0.0;
                f[i].y = 0.0;
                f[i].z = 0.0;

                next_x[i].x = x[i].x + dtv * v[i].x;
                next_x[i].y = x[i].y + dtv * v[i].y;
                next_x[i].z = x[i].z + dtv * v[i].z;
            }
            */

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

                            const double dtfm = dtf / mass[type[i]];
                            v[i].x += dtfm * f[i].x;
                            v[i].y += dtfm * f[i].y;
                            v[i].z += dtfm * f[i].z;

                            f[i].x = 0.0;
                            f[i].y = 0.0;
                            f[i].z = 0.0;

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
        } else {
            for (int idx = 0; idx < nlocal; idx++) {
                int i = local_idxs[idx];

                const double dtfm = dtf / mass[type[i]];

                v[i].x += dtfm * f[i].x;
                v[i].y += dtfm * f[i].y;
                v[i].z += dtfm * f[i].z;

                f[i].x = 0.0;
                f[i].y = 0.0;
                f[i].z = 0.0;

                next_x[i].x = x[i].x + dtv * v[i].x;
                next_x[i].y = x[i].y + dtv * v[i].y;
                next_x[i].z = x[i].z + dtv * v[i].z;
            }
        }
    }

    void FUSE_POST_FORCE_FINAL_INTEGRATE_ZOID_MANY_CUTS(queue_info& zoid, int dep, int timestep) {
        auto * _noalias v = zoid.v_stencil_md[timestep % 1].data();
        auto * _noalias f = zoid.f_stencil_md[timestep % 1].data();

        auto * _noalias mask = zoid.mask_stencil_md[0].data();
        auto * _noalias local_idxs = zoid.local_idxs_per_timestep[timestep].data();
        auto * _noalias type = zoid.type_stencil_md[0].data();

        int nlocal = zoid.local_idxs_per_timestep[timestep].size();

        double dtv = update->dt;

        const double * const mass = atom->mass;
        double dtf = 0.5 * update->dt * force->ftm2v;

        auto fix_post_force = (FixLangevin*) modify->fix[modify->list_post_force[0]];

        auto gfactor1 = fix_post_force->gfactor1;
        auto gfactor2 = fix_post_force->gfactor2;
        // fix_post_force->compute_target();
        auto tsqrt = fix_post_force->tsqrt;

        // if ((dep == 0 || dep == NUM_DEPS - 1) && nlocal > MODIFY_GRAINSIZE) {
        if (nlocal > MODIFY_GRAINSIZE) {
            int num_workers = __cilkrts_get_nworkers();
            int num_chunks = nlocal / MODIFY_GRAINSIZE + 1;
            int chunks_per_worker = num_chunks / num_workers;
            int chunk_size = MODIFY_GRAINSIZE;
            auto* claimed = zoid.claimed_flags_stencil_md[0];

            /*
            #pragma cilk grainsize MODIFY_GRAINSIZE
            cilk_for (int idx = 0; idx < nlocal; idx++) {
                int i = local_idxs[idx];

                int atom_type = type[i];

                const double dtfm = dtf / mass[atom_type];

                double gamma1 = gfactor1[atom_type];
                double gamma2 = gfactor2[atom_type] * tsqrt;

                double rand_x = 0.6;
                double rand_y = 0.6;
                double rand_z = 0.6;

                double v_x = v[i].x;
                double v_y = v[i].y;
                double v_z = v[i].z;

                f[i].x += gamma1 * v_x + gamma2 * (rand_x - 0.5);
                f[i].y += gamma1 * v_y + gamma2 * (rand_x - 0.5);
                f[i].z += gamma1 * v_z + gamma2 * (rand_x - 0.5);

                v[i].x += dtfm * f[i].x;
                v[i].y += dtfm * f[i].y;
                v[i].z += dtfm * f[i].z;
            }
            */

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

                            int atom_type = type[i];

                            const double dtfm = dtf / mass[atom_type];

                            double gamma1 = gfactor1[atom_type];
                            double gamma2 = gfactor2[atom_type] * tsqrt;

                            double rand_x = 0.6;
                            double rand_y = 0.6;
                            double rand_z = 0.6;

                            double v_x = v[i].x;
                            double v_y = v[i].y;
                            double v_z = v[i].z;

                            f[i].x += gamma1 * v_x + gamma2 * (rand_x - 0.5);
                            f[i].y += gamma1 * v_y + gamma2 * (rand_x - 0.5);
                            f[i].z += gamma1 * v_z + gamma2 * (rand_x - 0.5);

                            v[i].x += dtfm * f[i].x;
                            v[i].y += dtfm * f[i].y;
                            v[i].z += dtfm * f[i].z;
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
        } else {
            for (int idx = 0; idx < nlocal; idx++) {
                int i = local_idxs[idx];
                int atom_type = type[i];

                const double dtfm = dtf / mass[atom_type];

                double gamma1 = gfactor1[atom_type];
                double gamma2 = gfactor2[atom_type] * tsqrt;

                double rand_x = 0.6;
                double rand_y = 0.6;
                double rand_z = 0.6;

                double v_x = v[i].x;
                double v_y = v[i].y;
                double v_z = v[i].z;
                double f0 = f[i].x;
                double f1 = f[i].y;
                double f2 = f[i].z;

                f[i].x += gamma1 * v_x + gamma2 * (rand_x - 0.5);
                f[i].y += gamma1 * v_y + gamma2 * (rand_y - 0.5);
                f[i].z += gamma1 * v_z + gamma2 * (rand_z - 0.5);

                v[i].x += dtfm * f[i].x;
                v[i].y += dtfm * f[i].y;
                v[i].z += dtfm * f[i].z;
            }
        }
    }

    void FORCE_COMPUTE_ZOID_MANY_CUTS(queue_info& zoid, int dep, int timestep) {
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

        // if ((dep == 0 || dep == NUM_DEPS - 1) && nlocal > MODIFY_GRAINSIZE) {
        if (nlocal > MODIFY_GRAINSIZE) {
            #pragma cilk grainsize MODIFY_GRAINSIZE
            cilk_for (int idx = 0; idx < nlocal; idx++) {
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

            /*
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
            */
        } else {
            for (int idx = 0; idx < nlocal; idx++) {
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
                            f[j].x -= delx * fpair;
                            f[j].y -= dely * fpair;
                            f[j].z -= delz * fpair;
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
    void DELETE_ZOID_DATA_MANY_CUTS() {
        for (int dep = 0; dep < NUM_DEPS; dep++) {
            for (int j = 0; j < queues_many_cuts[dep].size(); j++) {
                queue_info &zoid = queues_many_cuts[dep][j];
                if (zoid.num % comm->nprocs != comm->me) {
                    continue;
                }
                delete[] zoid.x_stencil_md;
                delete[] zoid.v_stencil_md;
                delete[] zoid.f_stencil_md;
                delete[] zoid.eval_f_stencil_md;
                delete[] zoid.tag_stencil_md;
                delete[] zoid.type_stencil_md;
                delete[] zoid.mask_stencil_md;
                delete[] zoid.image_stencil_md;

                delete[] zoid.spinlocks_stencil_md[0];
                delete[] zoid.spinlocks_stencil_md;
                delete[] zoid.claimed_flags_stencil_md[0];
                delete[] zoid.claimed_flags_stencil_md;

                delete[] zoid.local_idxs_per_timestep;
                delete[] zoid.neighbor_list;
                delete[] zoid.bond_list;

                for (int t = 0; t < NUM_TIMESTEPS_IN_PARALLEL + 1; t++) {
                    delete[] zoid.send_force_idxs_double_buffering[t];
                    delete[] zoid.recv_force_idxs_double_buffering[t];
                    delete[] zoid.send_pos_idxs_double_buffering[t];
                    delete[] zoid.recv_pos_idxs_double_buffering[t];
                    delete[] zoid.send_vel_idxs_double_buffering[t];
                    delete[] zoid.recv_vel_idxs_double_buffering[t];
                }

                delete[] zoid.send_force_idxs_double_buffering;
                delete[] zoid.recv_force_idxs_double_buffering;
                delete[] zoid.send_pos_idxs_double_buffering;
                delete[] zoid.recv_pos_idxs_double_buffering;
                delete[] zoid.send_vel_idxs_double_buffering;
                delete[] zoid.recv_vel_idxs_double_buffering;

                for (int k = 0; k < DOUBLE_BUFFERING; k++) {
                    delete[] zoid.send_pos_idxs_double_buffering_flattened[k];
                    delete[] zoid.recv_pos_idxs_double_buffering_flattened[k];
                }

                delete[] zoid.send_force_idxs_double_buffering_flattened;
                delete[] zoid.send_pos_idxs_double_buffering_flattened;
                delete[] zoid.send_vel_idxs_double_buffering_flattened;
                delete[] zoid.recv_force_idxs_double_buffering_flattened;
                delete[] zoid.recv_pos_idxs_double_buffering_flattened;
                delete[] zoid.recv_vel_idxs_double_buffering_flattened;
            }
        }

        for (int dep = 0; dep < NUM_DEPS; dep++) {
            for (int j = 0; j < queues_many_cuts_next_dt[dep].size(); j++) {
                queue_info &zoid = queues_many_cuts_next_dt[dep][j];
                if (zoid.num % comm->nprocs != comm->me) {
                    continue;
                }

                delete[] zoid.local_idxs_per_timestep;
                delete[] zoid.neighbor_list;
                delete[] zoid.bond_list;

                for (int t = 0; t < NUM_TIMESTEPS_IN_PARALLEL + 1; t++) {
                    delete[] zoid.send_force_idxs_double_buffering[t];
                    delete[] zoid.recv_force_idxs_double_buffering[t];
                    delete[] zoid.send_pos_idxs_double_buffering[t];
                    delete[] zoid.recv_pos_idxs_double_buffering[t];
                    delete[] zoid.send_vel_idxs_double_buffering[t];
                    delete[] zoid.recv_vel_idxs_double_buffering[t];
                }

                delete[] zoid.send_force_idxs_double_buffering;
                delete[] zoid.recv_force_idxs_double_buffering;
                delete[] zoid.send_pos_idxs_double_buffering;
                delete[] zoid.recv_pos_idxs_double_buffering;
                delete[] zoid.send_vel_idxs_double_buffering;
                delete[] zoid.recv_vel_idxs_double_buffering;

                for (int k = 0; k < DOUBLE_BUFFERING; k++) {
                    delete[] zoid.send_pos_idxs_double_buffering_flattened[k];
                    delete[] zoid.recv_pos_idxs_double_buffering_flattened[k];
                }

                delete[] zoid.send_force_idxs_double_buffering_flattened;
                delete[] zoid.send_pos_idxs_double_buffering_flattened;
                delete[] zoid.send_vel_idxs_double_buffering_flattened;
                delete[] zoid.recv_force_idxs_double_buffering_flattened;
                delete[] zoid.recv_pos_idxs_double_buffering_flattened;
                delete[] zoid.recv_vel_idxs_double_buffering_flattened;
            }
        }
    }

    void CLEANUP_PIPELINED_BUFFERS() {
        for (int p = 0; p < NUM_PIPELINE_STAGES; p++) {
            for (int zoid_num = 0; zoid_num < NUM_ZOIDS_MANY_CUTS; zoid_num++) {
                if (zoid_num % comm->nprocs != comm->me) {
                    continue;
                }

                for (int i = 0; i < MAX_NEIGHBORS; i++) {
                    delete[] buf_send_zoid_to_zoid[p][zoid_num][i];
                    delete[] buf_recv_zoid_to_zoid[p][zoid_num][i];
                }

                delete[] buf_send_zoid_to_zoid[p][zoid_num];
                delete[] buf_recv_zoid_to_zoid[p][zoid_num];
                delete[] nsend_buf_send_zoid_to_zoid[p][zoid_num];
                delete[] nrecv_buf_recv_zoid_to_zoid[p][zoid_num];
            }
            delete[] buf_send_zoid_to_zoid[p];
            delete[] buf_recv_zoid_to_zoid[p];
            delete[] nsend_buf_send_zoid_to_zoid[p];
            delete[] nrecv_buf_recv_zoid_to_zoid[p];
        }
    }

    ~StencilMD() {
        /*
        MPI_Comm_free(&stream_comm);

        for (int i = 0; i < NUM_STREAMS; i++) {
            MPIX_Stream_free(&all_streams[i]);
        }
        */
        for (int i = 0; i < NUM_COMMS; i++) {
            MPI_Comm_free(&all_comms[i]);
        }

        for (int i = 0; i < NUM_ZOIDS_MANY_CUTS; i++) {
            if (i % comm->nprocs == comm->me) {
                int num_recv_neighbors = recv_from_neighbors_many_cuts[i].size();
                if (num_recv_neighbors > 0) {
                    delete[] recv_proc_zoid_offsets[i];
                    delete[] recv_proc_zoid_sizes[i];
                }

                int num_recv_neighbors_next_dt = recv_from_neighbors_many_cuts_next_dt[i].size();
                if (num_recv_neighbors_next_dt > 0) {
                    delete[] recv_proc_zoid_offsets_next_dt[i];
                    delete[] recv_proc_zoid_sizes_next_dt[i];
                }
            }
        }

        for (int dep = 0; dep < NUM_DEPS; dep++) {
            for (int proc = 0; proc < comm->nprocs; proc++) {
                delete[] buf_send_many_cuts[dep][proc];
                delete[] buf_recv_many_cuts[dep][proc];
            }

            delete[] buf_send_many_cuts[dep];
            delete[] buf_recv_many_cuts[dep];
            delete[] nsend_buf_many_cuts[dep];
            delete[] nrecv_buf_many_cuts[dep];
        }

        delete[] zoid_num_to_zoid_many_cuts;
        delete[] zoid_num_to_zoid_many_cuts_next_dt;
        delete[] send_to_neighbors_many_cuts;
        delete[] send_to_neighbors_many_cuts_next_dt;
        delete[] recv_from_neighbors_many_cuts;
        delete[] recv_from_neighbors_many_cuts_next_dt;

        CLEANUP_PIPELINED_BUFFERS();
        DELETE_ZOID_DATA_MANY_CUTS();
    }
};

}