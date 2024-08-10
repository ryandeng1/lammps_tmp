//
// Created by Ryan Deng on 2/27/24.
//

#pragma once

#include "pointers.h"
#include "bond_fene.h"
#include "pair_lj_cut.h"
#include <cilk/cilk.h>
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

    void BUILD_NEIGHBOR_LIST();
    void BUILD_NEIGHBOR_LIST_NEXT_DT();

    void COMPUTE_NUM_SEND_RECV_PROCESS();

    void COMPARE_POS_AGAINST_LAMMPS(bool curr_dt, int timestep, Atom* atom_, queue_info& zoid, double** test_x);

    void COMPARE_FORCE_AGAINST_LAMMPS(bool curr_dt, int timestep, Atom* atom_, queue_info& zoid, double** test_f);

    void SET_INUM_PER_TIMESTEP();
    void SET_INUM_PER_TIMESTEP_NEXT_DT();

    std::vector<double>& GET_BOUNDS(bool curr_dt, int timestep);

    std::vector<double> bounds[NUM_TIMESTEPS_IN_PARALLEL + 1];
    std::vector<std::size_t> sorted_bin_indices[NUM_TIMESTEPS_IN_PARALLEL + 1];

    // Begin methods used for fusing
    // typedef struct { double x,y,z; } dbl3_t;

    void initial_integrate_stencil_md(const IDX_3D& bin, Atom* atom_, Atom* next, int* atom_idx_mapping);
    void final_integrate_stencil_md(const IDX_3D& bin, Atom* atom_, Atom* next, int* atom_idx_mapping);
    void post_force_stencil_md(const IDX_3D& bin, Atom* atom_, Modify* modify_);

    template <bool curr_dt>
    void fuse_force_computation(queue_info& zoid, int timestep) {
        int zoid_num = zoid.num;
        auto& atom_arr = lmp->atom_stencil_md[zoid_num];
        Atom* curr = curr_dt ? atom_arr[timestep] : atom_arr[NUM_TIMESTEPS_IN_PARALLEL - timestep];
        Atom* next = curr_dt ? atom_arr[timestep + 1] : atom_arr[NUM_TIMESTEPS_IN_PARALLEL - timestep - 1];
        Neighbor* neigh_next = curr_dt ? lmp->neighbor_stencil_md[zoid_num][timestep + 1] : lmp->neighbor_stencil_md_next_dt[zoid_num][timestep + 1];
        int* atom_idx_mapping = zoid.atom_idx_mapping[timestep];

#ifdef LMP_OPENMP
        Modify* modify_ = curr_dt ? lmp->modify_stencil_md_omp[zoid_num][timestep + 1] : lmp->modify_stencil_md_omp[zoid_num][NUM_TIMESTEPS_IN_PARALLEL - timestep - 1];
#else
        Modify* modify_ = lmp->modify_stencil_md[zoid_num];
#endif

        // begin force computation, inline lj_cut and bond_fene
        assert(PURELY_LOCAL_POTENTIAL);

        Force* next_force = curr_dt ? lmp->force_stencil_md[zoid_num][timestep + 1] : lmp->force_stencil_md_next_dt[zoid_num][timestep + 1];

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

        for (int dep = 0; dep < next->num_deps; dep++) {
            auto& partitions_at_dep = next->dep_to_partitions[dep];

            std::set<int> idxs_touched_at_dep;
            std::map<int, std::array<int, 3>> idx_to_partition;
            std::map<int, int> idx_to_touched_neighbor;
            std::map<int, std::tuple<int, int, int>> idx_to_bin;

            cilk_for (int d = 0; d < partitions_at_dep.size(); d++) {
                auto& partition = partitions_at_dep[d];
                auto& bins = next->partition_to_bins[partition[0]][partition[1]][partition[2]];
                for (int b = 0; b < bins.size(); b++) {
                    auto& bin = bins[b];
                    auto &idxs = next->bin_to_local_idxs[bin];
                    for (int idx = 0; idx < idxs.size(); idx++) {
                        int ii = idxs[idx];
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
                                        std::cout << "zoid num: " << zoid.num << " timestep: " << timestep << std::endl;
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

                        f[i].x += fxtmp;
                        f[i].y += fytmp;
                        f[i].z += fztmp;

                        auto &lst_bonds = neigh_next->atom_bondlist[i];
                        for (int j = 0; j < lst_bonds.size(); j++) {
                            auto &bond_info = lst_bonds[j];
                            int i2 = bond_info.first;
                            int type = bond_info.second;

                            double delx = x[i].x - x[i2].x;
                            double dely = x[i].y - x[i2].y;
                            double delz = x[i].z - x[i2].z;

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
                                f[i].x += delx * fbond;
                                f[i].y += dely * fbond;
                                f[i].z += delz * fbond;
                            }

                            if (newton_pair || i2 < nlocal) {
                                f[i2].x -= delx * fbond;
                                f[i2].y -= dely * fbond;
                                f[i2].z -= delz * fbond;
                            }
                        }
                    }
                }
            }
        }
    }

    template <bool curr_dt>
    void fuse_post_force_stencil_md(queue_info& zoid, int timestep) {
        int zoid_num = zoid.num;
        auto& atom_arr = lmp->atom_stencil_md[zoid_num];
        Atom* curr = curr_dt ? atom_arr[timestep] : atom_arr[NUM_TIMESTEPS_IN_PARALLEL - timestep];
        Atom* next = curr_dt ? atom_arr[timestep + 1] : atom_arr[NUM_TIMESTEPS_IN_PARALLEL - timestep - 1];
        int* atom_idx_mapping = zoid.atom_idx_mapping[timestep];

#ifdef LMP_OPENMP
        Modify* modify_ = curr_dt ? lmp->modify_stencil_md_omp[zoid_num][timestep + 1] : lmp->modify_stencil_md_omp[zoid_num][NUM_TIMESTEPS_IN_PARALLEL - timestep - 1];
#else
        Modify* modify_ = lmp->modify_stencil_md[zoid_num];
#endif

        for (int dep = 0; dep < next->num_deps; dep++) {
            auto& partitions_at_dep = next->dep_to_partitions[dep];

            cilk_for (int d = 0; d < partitions_at_dep.size(); d++) {
                auto& partition = partitions_at_dep[d];
                auto& bins = next->partition_to_bins[partition[0]][partition[1]][partition[2]];
                for (int b = 0; b < bins.size(); b++) {
                    auto& bin = bins[b];
                    post_force_stencil_md(bin, next, modify_);
                    final_integrate_stencil_md(bin, curr, next, atom_idx_mapping);
                    // send_vel_bin_stencil_md<curr_dt>(zoid, timestep + 1, bin);
                }
            }
        }
    }

    template <bool curr_dt>
    void send_vel_bin_stencil_md(queue_info& zoid, int timestep, const IDX_3D& bin) {
        int zoid_num = zoid.num;
        // auto &send_to = curr_dt ? lmp->send_to_neighbors[zoid_num] : lmp->send_to_neighbors_next_dt[zoid_num];
        Atom* send_atom = curr_dt ? lmp->atom_stencil_md[zoid_num][timestep] : lmp->atom_stencil_md[zoid_num][NUM_TIMESTEPS_IN_PARALLEL - timestep];

        auto bin_idx = get_bin_idx(bin);
        int num_send_zoids = zoid.bin_to_num_send_zoids[timestep][bin_idx];
        auto send_zoids = zoid.bin_to_send_zoids[timestep][bin_idx];

        // std::cout << "zoid: " << zoid.num << " timestep: " << timestep << " num send: " << num_send_zoids << std::endl;

        // for (int i = 0; i < send_to.size(); i++) {
        for (int i = 0; i < num_send_zoids; i++) {
            // int recv_zoid_num = send_to[i];
            int recv_zoid_num = send_zoids[i];


            if (recv_zoid_num == 16 && timestep == 1) {
                std::cout << "zoid: " << zoid.num << " sending bin: " << bin[0] << " " << bin[1] << " " << bin[2] << std::endl;
            }

            auto& recv_zoid = curr_dt ? lmp->zoid_num_to_zoid[recv_zoid_num] : lmp->zoid_num_to_zoid_next_dt[recv_zoid_num];

            Atom* recv_atom = curr_dt ? lmp->atom_stencil_md[recv_zoid_num][timestep] : lmp->atom_stencil_md[recv_zoid_num][NUM_TIMESTEPS_IN_PARALLEL - timestep];

            auto send_size = zoid.bin_to_size[timestep][bin_idx];
            auto send_arr_idx = zoid.bin_to_idx[timestep][bin_idx];

            auto recv_size = recv_zoid.bin_to_size[timestep][bin_idx];
            auto recv_arr_idx = recv_zoid.bin_to_idx[timestep][bin_idx];

            assert(send_size == recv_size);

            auto * _noalias const send_v = (dbl3_t_stencil_md *) send_atom->v[0];
            auto * _noalias const recv_v = (dbl3_t_stencil_md *) recv_atom->v[0];

            for (int j = 0; j < send_size; j++) {
                int send_vel_idx = send_arr_idx + j;
                int recv_vel_idx = recv_arr_idx + j;
                tagint src_tag = send_atom->tag[send_vel_idx];
                tagint dst_tag = recv_atom->tag[recv_vel_idx];
                assert(src_tag == dst_tag);
                recv_v[recv_vel_idx].x = send_v[send_vel_idx].x;
                recv_v[recv_vel_idx].y = send_v[send_vel_idx].y;
                recv_v[recv_vel_idx].z = send_v[send_vel_idx].z;
            }
        }
    }
};

}