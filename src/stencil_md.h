//
// Created by Ryan Deng on 2/27/24.
//

#pragma once

#include "angle.h"
#include "pointers.h"
#include "bond_fene.h"
#include "pair_lj_cut.h"
#include "pair_lj_cut_omp.h"
#include "pair_dpd.h"
#include "pair_dpd_omp.h"
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
#include "stencil_md_utils.h"
#include "update.h"
#include "error.h"
#include "domain.h"
#include "thr_omp.h"
#include "fix_langevin.h"
#include <cilk/opadd_reducer.h>
#include <atomic>
#include <limits>
// This is from MPICH??
#include <mpi_proto.h>
#include <numeric>
#include <sstream>
#include <iomanip>
#include <queue>
#include "assert.h"
#include <functional>
#include <random>

#define EPSILON 1.0e-10

constexpr bool USE_BREAK = false;
// constexpr bool NO_LOCKS_BOND = true;

// MinCostFlow class implementing a simple min-cost max-flow using SPFA.
struct MinCostFlow {
    // Edge structure for the flow graph.
    const int INF = std::numeric_limits<int>::max();

    struct Edge {
        int to, rev, cap, cost;
    };

    int n;
    std::vector<std::vector<Edge>> graph;

    MinCostFlow(int n): n(n), graph(n) { }

    // Add an edge from s to t with given capacity and cost.
    void addEdge(int s, int t, int cap, int cost) {
        graph[s].push_back({t, (int)graph[t].size(), cap, cost});
        graph[t].push_back({s, (int)graph[s].size() - 1, 0, -cost});
    }

    // Returns total flow achieved and sets flowCost to the total cost.
    int minCostFlow(int s, int t, int f, int &flowCost) {
        int flow = 0;
        flowCost = 0;
        std::vector<int> dist(n), prev_v(n), prev_e(n);
        while (flow < f) {
            fill(dist.begin(), dist.end(), INF);
            dist[s] = 0;
            std::vector<bool> inQueue(n, false);
            std::queue<int> que;
            que.push(s);
            inQueue[s] = true;

            // SPFA to find shortest path in residual graph.
            while (!que.empty()) {
                int v = que.front();
                que.pop();
                inQueue[v] = false;
                for (int i = 0; i < graph[v].size(); i++) {
                    Edge &e = graph[v][i];
                    if (e.cap > 0 && dist[e.to] > dist[v] + e.cost) {
                        dist[e.to] = dist[v] + e.cost;
                        prev_v[e.to] = v;
                        prev_e[e.to] = i;
                        if (!inQueue[e.to]) {
                            que.push(e.to);
                            inQueue[e.to] = true;
                        }
                    }
                }
            }
            if (dist[t] == INF)
                break;  // No more augmenting paths.

            int d = f - flow;
            for (int v = t; v != s; v = prev_v[v])
                d = std::min(d, graph[prev_v[v]][prev_e[v]].cap);
            flow += d;
            flowCost += d * dist[t];
            for (int v = t; v != s; v = prev_v[v]) {
                Edge &e = graph[prev_v[v]][prev_e[v]];
                e.cap -= d;
                graph[v][e.rev].cap += d;
            }
        }
        return flow;
    }
};

// Returns the Euclidean distance needed to move (x,y,z) into the box
static double distance_to_zoid(double* period,
                               std::array<double, 3> zoid_lo, std::array<double, 3> zoid_hi,
                               std::array<double, 3> pos, bool print=false) {

    double delta[3] = {0};

    for (int dim = 0; dim < 3; dim++) {
        double diff = 0.0;

        if (pos[dim] < zoid_lo[dim]) {
            diff = zoid_lo[dim] - pos[dim];
        } else if (pos[dim] > zoid_hi[dim]) {
            diff = pos[dim] - zoid_hi[dim];
        }

        delta[dim] = diff;

        if (print) {
            std::cout << "dim: " << dim << " lo: " << zoid_lo[dim] << " hi: " << zoid_hi[dim] << " pos: " << pos[dim] << " diff: " << diff << std::endl;
        }
    }

    return std::sqrt(delta[0] * delta[0] + delta[1] * delta[1] + delta[2] * delta[2]);
}

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

    void SETUP();

    void MODIFY_PRE_FORCE_SETUP(int);

    void MODIFY_SETUP(int);

    void GET_LOCAL_ATOMS_ZOID();
    void GET_LOCAL_ATOMS_ZOID_DOUBLE_BUFFERING();

    void GET_GHOST_ATOMS_ZOID();
    void GET_GHOST_ATOMS_ZOID_DOUBLE_BUFFERING();

    void SORT_LOCAL_ATOMS_DOUBLE_BUFFERING();

    void CREATE_ATOM_IDXS_DOUBLE_BUFFERING();

    void BUILD_NEIGHBOR_LIST();
    void BUILD_NEIGHBOR_LIST_NEXT_DT();

    void BUILD_NEIGHBOR_LIST_DOUBLE_BUFFERING();
    void BUILD_BOND_LIST_DOUBLE_BUFFERING();

    void COMPARE_POS_AGAINST_LAMMPS(bool curr_dt, int timestep, Atom* atom_, queue_info& zoid, double** test_x);
    void COMPARE_FORCE_AGAINST_LAMMPS(bool curr_dt, int timestep, Atom* atom_, queue_info& zoid, double** test_f);
    void COMPARE_VEL_AGAINST_LAMMPS(bool curr_dt, int timestep, Atom* atom_, queue_info& zoid, double** test_f);

    void SET_CLAIMED_ATOMIC_BOOLS();

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
                const double dtfm = dtf / mass[type[i]];
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

        if constexpr (EXPERIMENT == DPD) {
            const auto& v = zoid.v_stencil_md[t % DOUBLE_BUFFERING];
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
        } else {
            const auto& v = zoid.v_stencil_md[0];
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
    std::map<std::array<int, 3>, int> zoid_where_to_num;

    std::vector<queue_info> queues_many_cuts[NUM_DEPS];
    std::vector<queue_info> queues_many_cuts_next_dt[NUM_DEPS];

    std::vector<queue_info> my_queues_many_cuts[NUM_DEPS];
    std::vector<queue_info> my_queues_many_cuts_next_dt[NUM_DEPS];

    static constexpr int NUM_CUTS_X = 4;
    static constexpr int NUM_CUTS_Y = 4;
    static constexpr int NUM_CUTS_Z = 4;

    static constexpr int NUM_ZOIDS_X = NUM_CUTS_X * 2;
    static constexpr int NUM_ZOIDS_Y = NUM_CUTS_Y * 2;
    static constexpr int NUM_ZOIDS_Z = NUM_CUTS_Z * 2;

    static constexpr int NUM_ZOIDS_MANY_CUTS = NUM_ZOIDS_X * NUM_ZOIDS_Y * NUM_ZOIDS_Z;

    queue_info* zoid_num_to_zoid_many_cuts;
    queue_info* zoid_num_to_zoid_many_cuts_next_dt;
    std::vector<int> zoid_num_to_dep;
    std::vector<int> zoid_num_to_dep_next_dt;

    std::vector<int>* send_to_neighbors_many_cuts;
    std::vector<int>* send_to_neighbors_many_cuts_next_dt;
    std::vector<int>* recv_from_neighbors_many_cuts;
    std::vector<int>* recv_from_neighbors_many_cuts_next_dt;

    std::vector<std::vector<int>> send_to_neighbors_not_my_proc_idxs;
    std::vector<std::vector<int>> send_to_neighbors_not_my_proc_idxs_next_dt;
    std::vector<int> send_to_neighbors_num_not_in_proc;
    std::vector<int> send_to_neighbors_num_not_in_proc_next_dt;

    std::vector<std::vector<int>> send_to_neighbors_not_my_proc_idxs_only_next_dep[2];
    std::vector<int> send_to_neighbors_num_not_in_proc_only_next_dep[2];

    std::vector<std::vector<int>> recv_from_neighbors_not_my_proc_idxs;
    std::vector<std::vector<int>> recv_from_neighbors_not_my_proc_idxs_next_dt;

    std::map<std::pair<int, int>, int> recv_request_zoid_to_idx[NUM_DEPS];
    std::map<std::pair<int, int>, int> recv_request_zoid_to_idx_next_dt[NUM_DEPS];
    std::map<int, std::pair<int, int>> recv_request_idx_to_zoid[NUM_DEPS];
    std::map<int, std::pair<int, int>> recv_request_idx_to_zoid_next_dt[NUM_DEPS];

    std::map<std::pair<int, int>, int> recv_request_zoid_pair_to_idx[2][NUM_DEPS];
    std::map<int, std::pair<int, int>> recv_request_idx_to_zoid_pair[2][NUM_DEPS];
    std::map<std::pair<int, int>, int> recv_request_proc_pair_to_idx[2][NUM_DEPS];
    std::map<int, std::pair<int, int>> recv_request_idx_to_proc_pair[2][NUM_DEPS];
    std::vector<std::pair<int, int>> dep_to_recv_proc_pairs[2][NUM_DEPS];
    std::vector<int> dep_to_recv_proc_pairs_sizes[2][NUM_DEPS];

    // map from (zoid.num, proc) -> offset for each zoid
    std::vector<std::vector<int>> recv_proc_zoid_offsets[2][NUM_PIPELINE_STAGES];
    std::vector<std::vector<int>> recv_proc_zoid_sizes[2][NUM_PIPELINE_STAGES];

    // Map from proc -> (list of (zoid, recv_zoid, recv_idxs) from that proc)
    std::vector<std::vector<std::array<int, 3>>> recv_zoids_from_proc[2][NUM_PIPELINE_STAGES][NUM_DEPS];

    std::map<std::pair<int, int>, int> recv_request_zoid_to_idx_with_proc_to_proc[2][NUM_PIPELINE_STAGES][NUM_DEPS];
    std::map<int, std::pair<int, int>> recv_request_idx_to_zoid_with_proc_to_proc[2][NUM_PIPELINE_STAGES][NUM_DEPS];
    int nrecv_zoid_to_zoid[2][NUM_PIPELINE_STAGES][NUM_DEPS];

    // map from (zoid.num, proc) -> offsets to each process at (dst_dep)
    std::map<std::pair<int, int>, int> send_proc_zoid_offsets[2][NUM_PIPELINE_STAGES];
    std::map<std::pair<int, int>, int> send_proc_zoid_sizes[2][NUM_PIPELINE_STAGES];
    std::vector<int> send_dep_to_procs[2][NUM_PIPELINE_STAGES][NUM_DEPS];

    std::vector<std::pair<int, int>> dep_to_recv_proc_to_proc[2][NUM_PIPELINE_STAGES][NUM_DEPS];
    std::vector<int> dep_to_recv_proc_to_proc_sizes[2][NUM_PIPELINE_STAGES][NUM_DEPS];

    double** buf_send_proc_to_proc[NUM_PIPELINE_STAGES][NUM_DEPS];
    double** buf_recv_proc_to_proc[NUM_PIPELINE_STAGES][NUM_DEPS];
    int* nsend_buf_proc_to_proc[NUM_PIPELINE_STAGES][NUM_DEPS];
    int* nrecv_buf_proc_to_proc[NUM_PIPELINE_STAGES][NUM_DEPS];

    std::vector<std::vector<int>> dep_proc_to_recv_zoids[NUM_DEPS];
    std::vector<std::vector<int>> dep_proc_to_recv_zoids_next_dt[NUM_DEPS];

    std::vector<std::vector<std::vector<int>>> dep_proc_recv_zoid_to_find_idxs[NUM_DEPS];
    std::vector<std::vector<std::vector<int>>> dep_proc_recv_zoid_to_find_idxs_next_dt[NUM_DEPS];

    static constexpr int MAX_NEIGHBORS = 26;
    double*** buf_send_zoid_to_zoid[NUM_PIPELINE_STAGES];
    int** nsend_buf_send_zoid_to_zoid[NUM_PIPELINE_STAGES];
    double*** buf_recv_zoid_to_zoid[NUM_PIPELINE_STAGES];
    int** nrecv_buf_recv_zoid_to_zoid[NUM_PIPELINE_STAGES];

    std::map<std::pair<int, int>, int> ZOID_TO_ZOID_TO_VCI_IDX[2];
    // std::map<std::pair<int, int>, int> ZOID_TO_ZOID_TO_VCI_IDX_NEXT_DT;

    std::vector<int> zoid_to_stream_num[2];

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

    // map dep to zoids I need to send to
    std::vector<int> dep_to_send_zoids[NUM_DEPS];
    std::vector<int> dep_to_send_zoids_next_dt[NUM_DEPS];

    void INIT_ZOID_MANY_CUTS() {
        // TODO: test out more than 1 cut in each dimension
        assert(domain->dimension == 3);

        std::vector<double> bounds_x;
        std::vector<double> bounds_y;
        std::vector<double> bounds_z;

        for (int dim = 0; dim < domain->dimension; dim++) {
            double width = domain->boxhi[dim] - domain->boxlo[dim];
            int num_cuts_in_dimension;
            if (dim == 0) {
                num_cuts_in_dimension = NUM_CUTS_X;
            } else if (dim == 1) {
                num_cuts_in_dimension = NUM_CUTS_Y;
            } else if (dim == 2) {
                num_cuts_in_dimension = NUM_CUTS_Z;
            } else {
                assert(false);
            }

            double narrow_base_width = ((width / num_cuts_in_dimension) - 2 * NUM_TIMESTEPS_IN_PARALLEL * ALLEGRO_SLOPE) / 2 + 0.1;
            double wide_base_width = (width - num_cuts_in_dimension * narrow_base_width) / num_cuts_in_dimension;

            if (comm->me == 0) {
                std::cout << "dim: " << dim << " narrow base width: " << narrow_base_width << " wide base width: " << wide_base_width
                << " total: " << num_cuts_in_dimension * (narrow_base_width + wide_base_width) << " width: " << width << std::endl;
            }

            double first_lo = domain->boxlo[dim] - narrow_base_width / 2.0;
            double first_hi = domain->boxlo[dim] + narrow_base_width / 2.0;

            double second_lo = first_hi;
            double second_hi = second_lo + wide_base_width;

            std::vector<double> bounds;
            bounds.push_back(first_lo);
            bounds.push_back(second_lo);
            bounds.push_back(second_hi);

            double one_set_width = narrow_base_width + wide_base_width;

            for (int i = 0; i < num_cuts_in_dimension - 1; i++) {
                double lo = bounds[bounds.size() - 1];
                double hi = lo + narrow_base_width;
                double next_hi = std::min(domain->boxhi[0] - narrow_base_width / 2, hi + wide_base_width);
                bounds.push_back(hi);
                bounds.push_back(next_hi);
            }

            if (dim == 0) {
                bounds_x = bounds;
            } else if (dim == 1) {
                bounds_y = bounds;
            } else if (dim == 2) {
                bounds_z = bounds;
            } else {
                assert(false);
            }
        }

        for (int i = 0; i < bounds_x.size() - 1; i++) {
            for (int j = 0; j < bounds_y.size() - 1; j++) {
                for (int k = 0; k < bounds_z.size() - 1; k++) {
                    int num_expanding = (i % 2 == 0) + (j % 2 == 0) + (k % 2 == 0);
                    queue_info zoid;
                    zoid.zoid.cuts[0].lower = bounds_x[i];
                    zoid.zoid.cuts[1].lower = bounds_y[j];
                    zoid.zoid.cuts[2].lower = bounds_z[k];
                    zoid.zoid.cuts[0].upper = bounds_x[i + 1];
                    zoid.zoid.cuts[1].upper = bounds_y[j + 1];
                    zoid.zoid.cuts[2].upper = bounds_z[k + 1];
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

        /*
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
        */

        for (int dep = 0; dep < NUM_DEPS; dep++) {
            for (int j = 0; j < queues_many_cuts[dep].size(); j++) {
                auto& zoid = queues_many_cuts[dep][j];
                zoid.num = zoid_where_to_num.at({zoid.where[0], zoid.where[1], zoid.where[2]});
                assert(zoid.num >= 0 && zoid.num < NUM_ZOIDS_MANY_CUTS);
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

                for (int dim = 0; dim < domain->dimension; dim++) {
                    new_zoid.where[dim] = zoid.where[dim];
                }
                new_zoid.num = zoid.num;

                queues_many_cuts_next_dt[new_dep].push_back(new_zoid);
            }
        }

        if (comm->me == 0) {
            for (int dim = 0; dim < domain->dimension; dim++) {
                std::cout << BOLDCYAN << "dim: " << dim << " Lo: " << domain->boxlo[dim] << " hi: " << domain->boxhi[dim] << RESET_COLOR << std::endl;
                std::vector<double> bounds;
                if (dim == 0) {
                    bounds = bounds_x;
                } else if (dim == 1) {
                    bounds = bounds_y;
                } else if (dim == 2) {
                    bounds = bounds_z;
                } else {
                    assert(false);
                }

                std::stringstream s1;
                for (auto& b : bounds) {
                    s1 << b << " ";
                }

                std::cout << BOLDCYAN << "dim: " << dim << " bounds: " << s1.str() << RESET_COLOR << std::endl;
            }
        }


        for (int dep = 0; dep < NUM_DEPS; dep++) {
            std::sort(queues_many_cuts[dep].begin(), queues_many_cuts[dep].end(), [&](const auto& zoid_a, const auto& zoid_b) {
                return zoid_a.num < zoid_b.num;
            });

            std::sort(queues_many_cuts_next_dt[dep].begin(), queues_many_cuts_next_dt[dep].end(), [&](const auto& zoid_a, const auto& zoid_b) {
                return zoid_a.num < zoid_b.num;
            });
        }
    }

    // Helper function to "split" the bits of a 32-bit integer by inserting two zeros between each bit.
    // Only the lower 21 bits of 'a' are used.
    uint64_t splitBy2(uint32_t a) {
        uint64_t x = a & 0x1fffff; // mask to 21 bits
        x = (x | x << 32) & 0x1f00000000ffffULL;
        x = (x | x << 16) & 0x1f0000ff0000ffULL;
        x = (x | x << 8)  & 0x100f00f00f00f00fULL;
        x = (x | x << 4)  & 0x10c30c30c30c30c3ULL;
        x = (x | x << 2)  & 0x1249249249249249ULL;
        return x;
    }

    // Compute the Morton code for 3D coordinates (x, y, z)
    // by interleaving the bits of x, y, and z.
    uint64_t morton3D(uint32_t x, uint32_t y, uint32_t z) {
        return (splitBy2(z) << 2) | (splitBy2(y) << 1) | splitBy2(x);
    }

    void INIT_ZOIDS_NUMBERING() {
        std::map<std::array<int, 3>, std::set<std::array<int, 3>>> tmp_send_neighbors;
        std::map<std::array<int, 3>, std::set<std::array<int, 3>>> tmp_recv_neighbors;

        for (int i = 0; i < NUM_ZOIDS_X; i++) {
            for (int j = 0; j < NUM_ZOIDS_Y; j++) {
                for (int k = 0; k < NUM_ZOIDS_Z; k++) {
                    std::array<int, 3> my_pos = {i, j, k};
                    if (i % 2 == 1) {
                        tmp_send_neighbors[my_pos].insert({i - 1, j, k});
                        tmp_send_neighbors[my_pos].insert({(i + 1) % NUM_ZOIDS_X, j, k});
                    }
                    if (j % 2 == 1) {
                        tmp_send_neighbors[my_pos].insert({i, j - 1, k});
                        tmp_send_neighbors[my_pos].insert({i, (j + 1) % NUM_ZOIDS_Y, k});
                    }
                    if (k % 2 == 1) {
                        tmp_send_neighbors[my_pos].insert({i, j, k - 1});
                        tmp_send_neighbors[my_pos].insert({i, j, (k + 1) % NUM_ZOIDS_Z});
                    }
                }
            }
        }

        for (auto& [zoid, send_zoids] : tmp_send_neighbors) {
            for (auto& z : send_zoids) {
                tmp_recv_neighbors[z].insert(zoid);
            }
        }

        std::map<int, std::vector<std::array<int, 3>>> proc_to_zoids;
        std::vector<int> proc_to_zoid_count(comm->nprocs, 0);
        std::map<std::array<int, 3>, int> zoid_to_proc;

        bool claimed[NUM_ZOIDS_X][NUM_ZOIDS_Y][NUM_ZOIDS_Y] = {0};

        std::vector<std::array<int, 3>> dep0_zoids;
        for (int x = 0; x < NUM_ZOIDS_X; x++) {
            for (int y = 0; y < NUM_ZOIDS_Y; y++) {
                for (int z = 0; z < NUM_ZOIDS_Z; z++) {
                    if (x % 2 == 1 && y % 2 == 1 && z % 2 == 1) {
                        dep0_zoids.push_back({x, y, z});
                    }
                }
            }
        }

        std::sort(dep0_zoids.begin(), dep0_zoids.end(), [&](const auto& zoid_a, const auto& zoid_b) {
            uint64_t code_a = morton3D(zoid_a[0], zoid_a[1], zoid_a[2]);
            uint64_t code_b = morton3D(zoid_b[0], zoid_b[1], zoid_b[2]);

            return code_a < code_b;
        });


        // this is fixed as it's the number of incoming neighbors
        int dep_to_val[NUM_DEPS] = {0, 2, 4, 6};
        int num_zoids_dep0 = NUM_ZOIDS_X  * NUM_ZOIDS_Y * NUM_ZOIDS_Z / (2 * 2 * 2);
        int num_zoids_dep1 = (NUM_ZOIDS_MANY_CUTS - 2 * num_zoids_dep0) / 2;
        int num_zoids_per_dep[NUM_DEPS] = {num_zoids_dep0, num_zoids_dep1, num_zoids_dep1, num_zoids_dep0};

        assert(dep0_zoids.size() % comm->nprocs == 0);

        if (NUM_ZOIDS_MANY_CUTS % comm->nprocs == 0) {
            for (int i = 0; i < dep0_zoids.size(); i++) {
                int proc = i / (dep0_zoids.size() / comm->nprocs);
                proc_to_zoids[proc].push_back(dep0_zoids[i]);
            }
        } else {
            std::cout << "num zoids: " << NUM_ZOIDS_MANY_CUTS << " nprocs: " << comm->nprocs << std::endl;
            assert(false);
        }

        if (comm->me == 0) {
            for (int proc = 0; proc < comm->nprocs; proc++) {
                std::cout << "proc: " << proc << " num dep0 zoids: " << proc_to_zoids[proc].size() << std::endl;
            }
        }

        constexpr bool TRY_ONLY_MAX_FLOW = true;

        if (TRY_ONLY_MAX_FLOW) {
            for (int dep = 1; dep < NUM_DEPS; dep++) {
                std::vector<std::map<std::array<int, 3>, int>> neighbor_to_count_per_proc(comm->nprocs);
                std::map<std::array<int, 3>, std::vector<int>> unclaimed_zoid_to_procs;

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
                                neighbor_to_count_per_proc[proc][n]++;

                                unclaimed_zoid_to_procs[n].push_back(proc);
                            }
                        }
                    }
                }

                std::vector<std::array<int, 3>> unclaimed_zoids_vec;
                for (auto& [zoid, _] : unclaimed_zoid_to_procs) {
                    unclaimed_zoids_vec.push_back(zoid);
                }

                // TODO: DO MIN-Cost Max Flow HERE
                // L is num balls per bin.
                int M = unclaimed_zoids_vec.size(); // Number of balls.
                int N = comm->nprocs; // Number of bins.
                int L = M / N;
                int source = M + N;
                int sink = M + N + 1;
                int totalNodes = M + N + 2;
                MinCostFlow mcf(totalNodes);

                // Source to ball nodes: capacity = 1, cost = 0.
                for (int i = 0; i < M; i++) {
                    mcf.addEdge(source, i, 1, 0);
                }

                // Ball nodes to bin nodes:
                // For each ball, add an edge to each allowed bin with capacity 1 and cost 0.
                for (int i = 0; i < M; i++) {
                    auto& unclaimed_zoid = unclaimed_zoids_vec[i];
                    for (int bin: unclaimed_zoid_to_procs.at(unclaimed_zoid)) {
                        // Bin node index = M + bin.
                        mcf.addEdge(i, M + bin, 1, -neighbor_to_count_per_proc[bin].at(unclaimed_zoid));
                    }
                }

                // Bin nodes to sink:
                // Base edge: capacity = L, cost = 0.
                // Extra edge: capacity = 1, cost = 1 (penalty for extra ball).
                for (int b = 0; b < N; b++) {
                    mcf.addEdge(M + b, sink, L, 0);
                    mcf.addEdge(M + b, sink, 1, 10);
                }

                // Run min-cost flow to assign all M balls.
                int flowCost = 0;
                int flowAchieved = mcf.minCostFlow(source, sink, M, flowCost);
                if (flowAchieved < M) {
                    std::cout << "Error: Not all balls could be assigned!" << std::endl;
                    assert(false);
                }

                // Determine the assignment by inspecting the flow on edges from ball nodes to bin nodes.
                std::vector<int> assignment(M, -1);
                for (int i = 0; i < M; i++) {
                    for (auto &edge : mcf.graph[i]) {
                        // Edge from ball node i to a bin node: bin nodes are in [M, M+N-1].
                        if (edge.to >= M && edge.to < M + N) {
                            // If the edge was used (original capacity was 1, so if cap==0 it was used).
                            if (edge.cap == 0) {
                                int bin = edge.to - M;
                                assignment[i] = bin;
                                break;
                            }
                        }
                    }
                }

                for (int i = 0; i < M; i++) {
                    auto& unclaimed_zoid = unclaimed_zoids_vec[i];
                    auto proc = assignment[i];
                    assert(proc >= 0 && proc < comm->nprocs);
                    proc_to_zoids[proc].push_back(unclaimed_zoid);
                    proc_to_zoid_count[proc]++;
                    claimed[unclaimed_zoid[0]][unclaimed_zoid[1]][unclaimed_zoid[2]] = true;
                }

                for (int proc = 0; proc < comm->nprocs; proc++) {
                    if (comm->me == 0) {
                        std::cout << "dep: " << dep << " proc: " << proc << " has num zoids: " << proc_to_zoids[proc].size() << std::endl;
                    }
                }

                MPI_Barrier(world);
            }

            std::set<std::array<int, 3>> test_zoids;
            for (int proc = 0; proc < comm->nprocs; proc++) {
                for (auto& zoid : proc_to_zoids[proc]) {
                    test_zoids.insert(zoid);
                }
            }

            if (comm->me == 0) {
                std::cout << "test zoids size: " << test_zoids.size() << std::endl;
            }
            assert(test_zoids.size() == NUM_ZOIDS_MANY_CUTS);

            for (int proc = 0; proc < comm->nprocs; proc++) {
                auto& zoids = proc_to_zoids.at(proc);
                std::sort(zoids.begin(), zoids.end(), [](const auto& zoid_a, const auto& zoid_b) {
                    int dep_a = (zoid_a[0] % 2 == 0) + (zoid_a[1] % 2 == 0) + (zoid_a[2] % 2 == 0);
                    int dep_b = (zoid_b[0] % 2 == 0) + (zoid_b[1] % 2 == 0) + (zoid_b[2] % 2 == 0);
                    return dep_a < dep_b;
                });

                for (int i = 0; i < zoids.size(); i++) {
                    int zoid_num = i * comm->nprocs + proc;
                    assert(zoid_num >= 0 && zoid_num < NUM_ZOIDS_MANY_CUTS);
                    zoid_where_to_num[zoids[i]] = i * comm->nprocs + proc;
                }
            }

            return;
        }
    }

    // START CLAUDE CODE
    // Optimized INIT_ZOIDS_NUMBERING with multi-start approach for balanced assignment

    // Optimized INIT_ZOIDS_NUMBERING with multi-start approach for balanced assignment

    void INIT_ZOIDS_NUMBERING_BALANCED() {
        std::map<std::array<int, 3>, std::set<std::array<int, 3>>> tmp_send_neighbors;
        std::map<std::array<int, 3>, std::set<std::array<int, 3>>> tmp_recv_neighbors;

        // Build neighbor relationships (one-way: odd coords send to even coords)
        for (int i = 0; i < NUM_ZOIDS_X; i++) {
            for (int j = 0; j < NUM_ZOIDS_Y; j++) {
                for (int k = 0; k < NUM_ZOIDS_Z; k++) {
                    std::array<int, 3> my_pos = {i, j, k};
                    if (i % 2 == 1) {
                        tmp_send_neighbors[my_pos].insert({i - 1, j, k});
                        tmp_send_neighbors[my_pos].insert({(i + 1) % NUM_ZOIDS_X, j, k});
                    }
                    if (j % 2 == 1) {
                        tmp_send_neighbors[my_pos].insert({i, j - 1, k});
                        tmp_send_neighbors[my_pos].insert({i, (j + 1) % NUM_ZOIDS_Y, k});
                    }
                    if (k % 2 == 1) {
                        tmp_send_neighbors[my_pos].insert({i, j, k - 1});
                        tmp_send_neighbors[my_pos].insert({i, j, (k + 1) % NUM_ZOIDS_Z});
                    }
                }
            }
        }

        // Build receive relationships
        for (auto& [zoid, send_zoids] : tmp_send_neighbors) {
            for (auto& z : send_zoids) {
                tmp_recv_neighbors[z].insert(zoid);
            }
        }

        // Verify perfect divisibility
        assert(NUM_ZOIDS_MANY_CUTS % comm->nprocs == 0);
        const int zoids_per_proc = NUM_ZOIDS_MANY_CUTS / comm->nprocs;

        // MULTI-START APPROACH: Try multiple initial assignments
        const int NUM_STARTS = 5;  // Number of different initial assignments to try
        
        struct Assignment {
            std::map<int, std::vector<std::array<int, 3>>> proc_to_zoids;
            std::map<std::array<int, 3>, int> zoid_to_proc;
            int edge_cuts;
            std::string method_name;
        };
        
        std::vector<Assignment> assignments;
        
        // METHOD 1: Morton curve (Z-order)
        {
            Assignment morton_assignment;
            morton_assignment.method_name = "Morton Curve";
            
            std::vector<std::pair<uint64_t, std::array<int, 3>>> morton_zoids;
            for (int x = 0; x < NUM_ZOIDS_X; x++) {
                for (int y = 0; y < NUM_ZOIDS_Y; y++) {
                    for (int z = 0; z < NUM_ZOIDS_Z; z++) {
                        uint64_t morton_code = morton3D(x, y, z);
                        morton_zoids.push_back({morton_code, {x, y, z}});
                    }
                }
            }
            
            std::sort(morton_zoids.begin(), morton_zoids.end());
            
            for (int i = 0; i < morton_zoids.size(); i++) {
                int proc = i / zoids_per_proc;
                morton_assignment.proc_to_zoids[proc].push_back(morton_zoids[i].second);
                morton_assignment.zoid_to_proc[morton_zoids[i].second] = proc;
            }
            
            assignments.push_back(morton_assignment);
        }
        
        // Optimize each initial assignment with iterative refinement
        for (auto& assignment : assignments) {
            if (comm->me == 0) {
                std::cout << "\nOptimizing " << assignment.method_name << " assignment..." << std::endl;
            }
            
            optimizeAssignmentWithSwaps(assignment.proc_to_zoids, assignment.zoid_to_proc,
                                    tmp_send_neighbors, tmp_recv_neighbors, 20);
            
            assignment.edge_cuts = countTotalEdgeCuts(assignment.zoid_to_proc, tmp_send_neighbors);
            
            if (comm->me == 0) {
                std::cout << "  Final edge cuts: " << assignment.edge_cuts << std::endl;
            }
        }
        
        // Select best assignment
        Assignment* best_assignment = &assignments[0];
        for (auto& assignment : assignments) {
            if (assignment.edge_cuts < best_assignment->edge_cuts) {
                best_assignment = &assignment;
            }
        }
        
        if (comm->me == 0) {
            std::cout << "\n=== BEST ASSIGNMENT: " << best_assignment->method_name 
                    << " with " << best_assignment->edge_cuts << " edge cuts ===" << std::endl;
        }
        
        // Use the best assignment
        std::map<int, std::vector<std::array<int, 3>>> proc_to_zoids = best_assignment->proc_to_zoids;
        std::map<std::array<int, 3>, int> zoid_to_proc = best_assignment->zoid_to_proc;
        
        std::vector<int> counts_per_proc[NUM_DEPS];
        for (int dep = 0; dep < NUM_DEPS; dep++) {
            counts_per_proc[dep].resize(comm->nprocs, 0);
        }

        // Final numbering assignment
        for (int proc = 0; proc < comm->nprocs; proc++) {
            auto& zoids = proc_to_zoids[proc];
            
            // Sort by dependency level then Morton code
            std::sort(zoids.begin(), zoids.end(), [&](const auto& a, const auto& b) {
                int dep_a = (a[0] % 2 == 0) + (a[1] % 2 == 0) + (a[2] % 2 == 0);
                int dep_b = (b[0] % 2 == 0) + (b[1] % 2 == 0) + (b[2] % 2 == 0);
                if (dep_a != dep_b) return dep_a < dep_b;
                
                return morton3D(a[0], a[1], a[2]) < morton3D(b[0], b[1], b[2]);
            });
            
            for (int i = 0; i < zoids.size(); i++) {
                zoid_where_to_num[zoids[i]] = i * comm->nprocs + proc;
                int dep = (zoids[i][0] % 2 == 0) + (zoids[i][1] % 2 == 0) + (zoids[i][2] % 2 == 0);
                counts_per_proc[dep][proc]++;
            }
        }

        for (int dep = 0; dep < NUM_DEPS; dep++) {
            int expected_count_at_proc;
            if (dep == 0 || dep == 3) {
                expected_count_at_proc = (NUM_ZOIDS_MANY_CUTS / comm->nprocs) / 8;
            } else {
                expected_count_at_proc = ((NUM_ZOIDS_MANY_CUTS / comm->nprocs) / 8) * 3;
            }

            for (int proc = 0; proc < comm->nprocs; proc++) {
                if (counts_per_proc[dep][proc] != expected_count_at_proc) {
                    std::cout << BOLDRED << "ERROR IN BALANCING. dep: " << dep << " PROC: " << proc << " GOT: " << counts_per_proc[dep][proc] << " EXPECTED: " << expected_count_at_proc << RESET_COLOR << std::endl;
                    MPI_Abort(world, 0);
                }
            }
        }

        MPI_Barrier(world);

        // Print detailed statistics
        if (comm->me == 0) {
            std::cout << BOLDGREEN << "FOUND BALANCED ASSIGNMENT" << RESET_COLOR << std::endl;
            printBalancedAssignmentStats(proc_to_zoids, zoid_to_proc, tmp_send_neighbors, zoids_per_proc);
            
            // Also print comparison of all methods
            std::cout << "\n=== Method Comparison ===" << std::endl;
            for (const auto& assignment : assignments) {
                std::cout << std::setw(20) << assignment.method_name 
                        << ": " << assignment.edge_cuts << " edge cuts" << std::endl;
            }
        }
    }

    // Helper function: Find optimal torus decomposition
    void findOptimalTorusDecomposition(int nx, int ny, int nz, int nprocs,
                                    int& px, int& py, int& pz) {
        // Find factorization of nprocs that minimizes surface-to-volume ratio
        int best_surface = INT_MAX;
        
        for (int i = 1; i <= nprocs; i++) {
            if (nprocs % i == 0) {
                int remaining = nprocs / i;
                for (int j = 1; j <= remaining; j++) {
                    if (remaining % j == 0) {
                        int k = remaining / j;
                        
                        // Check if this decomposition divides evenly
                        if (nx % i == 0 && ny % j == 0 && nz % k == 0) {
                            // Calculate surface area (communication volume)
                            // In a torus, each block has 6 faces
                            int block_x = nx / i;
                            int block_y = ny / j;
                            int block_z = nz / k;
                            
                            int surface = 2 * (block_y * block_z + block_x * block_z + block_x * block_y);
                            
                            if (surface < best_surface) {
                                best_surface = surface;
                                px = i; py = j; pz = k;
                            }
                        }
                    }
                }
            }
        }
        
        // If no perfect decomposition found, use approximate
        if (best_surface == INT_MAX) {
            // Simple factorization
            px = 1; py = 1; pz = nprocs;
            for (int i = 2; i <= std::cbrt(nprocs); i++) {
                if (nprocs % i == 0) {
                    px = i;
                    int remaining = nprocs / i;
                    for (int j = i; j <= std::sqrt(remaining); j++) {
                        if (remaining % j == 0) {
                            py = j;
                            pz = remaining / j;
                            break;
                        }
                    }
                }
            }
        }
    }

    // Helper function: Adjust cluster count
    void adjustClusterCount(std::vector<std::vector<int>>& clusters, 
                        int target_clusters, int target_size) {
        // Merge smallest clusters until we have the right number
        while (clusters.size() > target_clusters) {
            // Find two smallest clusters
            int min1 = 0, min2 = 1;
            if (clusters[1].size() < clusters[0].size()) {
                min1 = 1; min2 = 0;
            }
            
            for (int i = 2; i < clusters.size(); i++) {
                if (clusters[i].size() < clusters[min1].size()) {
                    min2 = min1;
                    min1 = i;
                } else if (clusters[i].size() < clusters[min2].size()) {
                    min2 = i;
                }
            }
            
            // Merge min1 into min2
            clusters[min2].insert(clusters[min2].end(), 
                                clusters[min1].begin(), 
                                clusters[min1].end());
            clusters.erase(clusters.begin() + min1);
        }
        
        // Split largest clusters if needed
        while (clusters.size() < target_clusters) {
            // Find largest cluster
            int max_idx = 0;
            for (int i = 1; i < clusters.size(); i++) {
                if (clusters[i].size() > clusters[max_idx].size()) {
                    max_idx = i;
                }
            }
            
            // Split it in half
            std::vector<int> new_cluster;
            int split_point = clusters[max_idx].size() / 2;
            new_cluster.insert(new_cluster.end(),
                            clusters[max_idx].begin() + split_point,
                            clusters[max_idx].end());
            clusters[max_idx].resize(split_point);
            clusters.push_back(new_cluster);
        }
        
        // Balance cluster sizes
        while (true) {
            int min_idx = -1, max_idx = -1;
            int min_size = INT_MAX, max_size = 0;
            
            for (int i = 0; i < clusters.size(); i++) {
                if (clusters[i].size() < min_size) {
                    min_size = clusters[i].size();
                    min_idx = i;
                }
                if (clusters[i].size() > max_size) {
                    max_size = clusters[i].size();
                    max_idx = i;
                }
            }
            
            if (max_size - min_size <= 1) break; // Balanced enough
            
            // Move one element from max to min
            clusters[min_idx].push_back(clusters[max_idx].back());
            clusters[max_idx].pop_back();
        }
    }

    // Helper function: K-means++ initialization
    void initializeKMeansPlusPlus(const std::vector<std::array<int, 3>>& points,
                                std::vector<std::array<double, 3>>& centroids) {
        std::mt19937 rng(42);
        int k = centroids.size();
        
        // Choose first centroid randomly
        std::uniform_int_distribution<> first_dist(0, points.size() - 1);
        int first_idx = first_dist(rng);
        centroids[0] = {(double)points[first_idx][0], 
                    (double)points[first_idx][1], 
                    (double)points[first_idx][2]};
        
        // Choose remaining centroids
        for (int c = 1; c < k; c++) {
            std::vector<double> min_distances(points.size(), std::numeric_limits<double>::infinity());
            double total_dist = 0;
            
            // Calculate distance to nearest centroid for each point
            for (int i = 0; i < points.size(); i++) {
                for (int j = 0; j < c; j++) {
                    double dist = euclideanDistance(points[i], centroids[j]);
                    min_distances[i] = std::min(min_distances[i], dist);
                }
                total_dist += min_distances[i] * min_distances[i]; // D^2 weighting
            }
            
            // Choose next centroid with probability proportional to D^2
            std::uniform_real_distribution<> prob_dist(0, total_dist);
            double threshold = prob_dist(rng);
            double cumsum = 0;
            
            for (int i = 0; i < points.size(); i++) {
                cumsum += min_distances[i] * min_distances[i];
                if (cumsum >= threshold) {
                    centroids[c] = {(double)points[i][0], 
                                (double)points[i][1], 
                                (double)points[i][2]};
                    break;
                }
            }
        }
    }

    // Helper function: Balance K-means clusters
    void balanceKMeansClusters(std::vector<std::vector<int>>& cluster_members,
                            const std::vector<std::array<int, 3>>& points,
                            const std::vector<std::array<double, 3>>& centroids) {
        int k = cluster_members.size();
        int target_size = points.size() / k;
        
        // Move points from oversized to undersized clusters
        bool changed = true;
        while (changed) {
            changed = false;
            
            // Find oversized and undersized clusters
            std::vector<int> oversized, undersized;
            for (int c = 0; c < k; c++) {
                if (cluster_members[c].size() > target_size) {
                    oversized.push_back(c);
                } else if (cluster_members[c].size() < target_size) {
                    undersized.push_back(c);
                }
            }
            
            // Try to move points
            for (int from : oversized) {
                if (cluster_members[from].size() <= target_size) continue;
                
                // Find point in 'from' cluster closest to any undersized cluster
                int best_point_idx = -1;
                int best_to_cluster = -1;
                double best_dist_increase = std::numeric_limits<double>::infinity();
                
                for (int i = 0; i < cluster_members[from].size(); i++) {
                    int point_idx = cluster_members[from][i];
                    double current_dist = euclideanDistance(points[point_idx], centroids[from]);
                    
                    for (int to : undersized) {
                        if (cluster_members[to].size() >= target_size) continue;
                        
                        double new_dist = euclideanDistance(points[point_idx], centroids[to]);
                        double dist_increase = new_dist - current_dist;
                        
                        if (dist_increase < best_dist_increase) {
                            best_dist_increase = dist_increase;
                            best_point_idx = i;
                            best_to_cluster = to;
                        }
                    }
                }
                
                // Move the best point
                if (best_point_idx >= 0 && best_to_cluster >= 0) {
                    int point_to_move = cluster_members[from][best_point_idx];
                    cluster_members[from].erase(cluster_members[from].begin() + best_point_idx);
                    cluster_members[best_to_cluster].push_back(point_to_move);
                    changed = true;
                    break; // Restart the loop
                }
            }
        }
    }

    // Helper function: Assign zoids using min-cost flow
    void assignWithMinCostFlow(const std::vector<std::array<int, 3>>& unclaimed_zoids,
                            std::map<int, std::vector<std::array<int, 3>>>& proc_to_zoids,
                            std::map<std::array<int, 3>, int>& zoid_to_proc,
                            const std::map<std::array<int, 3>, std::set<std::array<int, 3>>>& send_neighbors,
                            const std::map<std::array<int, 3>, std::set<std::array<int, 3>>>& recv_neighbors,
                            int zoids_per_proc) {
        
        int M = unclaimed_zoids.size();
        int N = proc_to_zoids.size();
        
        // Calculate remaining capacity
        std::vector<int> proc_remaining_capacity(N);
        for (int p = 0; p < N; p++) {
            proc_remaining_capacity[p] = zoids_per_proc - proc_to_zoids[p].size();
        }
        
        // Build flow graph
        int source = M + N;
        int sink = M + N + 1;
        MinCostFlow mcf(M + N + 2);
        
        // Source to zoid nodes
        for (int i = 0; i < M; i++) {
            mcf.addEdge(source, i, 1, 0);
        }
        
        // Zoid to processor edges with costs
        for (int i = 0; i < M; i++) {
            auto& zoid = unclaimed_zoids[i];
            
            for (int p = 0; p < N; p++) {
                if (proc_remaining_capacity[p] > 0) {
                    int cost = 0;
                    
                    // Communication cost
                    if (recv_neighbors.find(zoid) != recv_neighbors.end()) {
                        for (const auto& sender : recv_neighbors.at(zoid)) {
                            if (zoid_to_proc.find(sender) != zoid_to_proc.end()) {
                                if (zoid_to_proc.at(sender) == p) {
                                    cost -= 10;
                                } else {
                                    cost += 5;
                                }
                            }
                        }
                    }
                    
                    if (send_neighbors.find(zoid) != send_neighbors.end()) {
                        for (const auto& receiver : send_neighbors.at(zoid)) {
                            if (zoid_to_proc.find(receiver) != zoid_to_proc.end()) {
                                if (zoid_to_proc.at(receiver) == p) {
                                    cost -= 10;
                                } else {
                                    cost += 5;
                                }
                            }
                        }
                    }
                    
                    // Spatial locality
                    auto centroid = calculateProcessorCentroid(proc_to_zoids[p]);
                    cost += static_cast<int>(euclideanDistance(zoid, centroid) * 0.5);
                    
                    mcf.addEdge(i, M + p, 1, cost);
                }
            }
        }
        
        // Processor to sink
        for (int p = 0; p < N; p++) {
            if (proc_remaining_capacity[p] > 0) {
                mcf.addEdge(M + p, sink, proc_remaining_capacity[p], 0);
            }
        }
        
        // Run flow
        int flowCost = 0;
        int flowAchieved = mcf.minCostFlow(source, sink, M, flowCost);
        assert(flowAchieved == M);
        
        // Extract assignments
        for (int i = 0; i < M; i++) {
            for (auto& edge : mcf.graph[i]) {
                if (edge.to >= M && edge.to < M + N && edge.cap == 0) {
                    int proc = edge.to - M;
                    proc_to_zoids[proc].push_back(unclaimed_zoids[i]);
                    zoid_to_proc[unclaimed_zoids[i]] = proc;
                    break;
                }
            }
        }
    }

    // Helper function: Optimize assignment with balanced swaps
    void optimizeAssignmentWithSwaps(std::map<int, std::vector<std::array<int, 3>>>& proc_to_zoids,
                                std::map<std::array<int, 3>, int>& zoid_to_proc,
                                const std::map<std::array<int, 3>, std::set<std::array<int, 3>>>& send_neighbors,
                                const std::map<std::array<int, 3>, std::set<std::array<int, 3>>>& recv_neighbors,
                                int max_iterations) {
        
        bool improved = true;
        int iterations = 0;
        
        while (improved && iterations < max_iterations) {
            improved = false;
            iterations++;
            
            // Try all processor pairs
            for (int p1 = 0; p1 < proc_to_zoids.size(); p1++) {
                for (int p2 = p1 + 1; p2 < proc_to_zoids.size(); p2++) {
                    
                    // Find best swap
                    int best_improvement = 0;
                    int best_i1 = -1, best_i2 = -1;
                    
                    for (int i1 = 0; i1 < proc_to_zoids[p1].size(); i1++) {
                        for (int i2 = 0; i2 < proc_to_zoids[p2].size(); i2++) {
                            auto& z1 = proc_to_zoids[p1][i1];
                            auto& z2 = proc_to_zoids[p2][i2];
                            
                            // Calculate improvement
                            int current_cost = 
                                calculateZoidCommCost(z1, p1, zoid_to_proc, send_neighbors, recv_neighbors) +
                                calculateZoidCommCost(z2, p2, zoid_to_proc, send_neighbors, recv_neighbors);
                            
                            int swap_cost = 
                                calculateZoidCommCost(z1, p2, zoid_to_proc, send_neighbors, recv_neighbors) +
                                calculateZoidCommCost(z2, p1, zoid_to_proc, send_neighbors, recv_neighbors);
                            
                            int improvement = current_cost - swap_cost;
                            if (improvement > best_improvement) {
                                best_improvement = improvement;
                                best_i1 = i1;
                                best_i2 = i2;
                            }
                        }
                    }
                    
                    // Perform swap if beneficial
                    if (best_improvement > 0) {
                        auto z1 = proc_to_zoids[p1][best_i1];
                        auto z2 = proc_to_zoids[p2][best_i2];
                        
                        zoid_to_proc[z1] = p2;
                        zoid_to_proc[z2] = p1;
                        
                        proc_to_zoids[p1][best_i1] = z2;
                        proc_to_zoids[p2][best_i2] = z1;
                        
                        improved = true;
                    }
                }
            }
        }
    }

    // Helper function to calculate communication cost for a single zoid
    int calculateZoidCommCost(
        const std::array<int, 3>& zoid,
        int proc,
        const std::map<std::array<int, 3>, int>& zoid_to_proc,
        const std::map<std::array<int, 3>, std::set<std::array<int, 3>>>& send_neighbors,
        const std::map<std::array<int, 3>, std::set<std::array<int, 3>>>& recv_neighbors) {
        
        int cost = 0;
        
        // Cost for outgoing edges
        if (send_neighbors.find(zoid) != send_neighbors.end()) {
            for (const auto& receiver : send_neighbors.at(zoid)) {
                if (zoid_to_proc.find(receiver) != zoid_to_proc.end() && 
                    zoid_to_proc.at(receiver) != proc) {
                    cost++;
                }
            }
        }
        
        // Cost for incoming edges
        if (recv_neighbors.find(zoid) != recv_neighbors.end()) {
            for (const auto& sender : recv_neighbors.at(zoid)) {
                if (zoid_to_proc.find(sender) != zoid_to_proc.end() && 
                    zoid_to_proc.at(sender) != proc) {
                    cost++;
                }
            }
        }
        
        return cost;
    }

    // Helper function to count total edge cuts
    int countTotalEdgeCuts(
        const std::map<std::array<int, 3>, int>& zoid_to_proc,
        const std::map<std::array<int, 3>, std::set<std::array<int, 3>>>& send_neighbors) {
        
        int edge_cuts = 0;
        
        for (const auto& [sender, receivers] : send_neighbors) {
            if (zoid_to_proc.find(sender) == zoid_to_proc.end()) continue;
            int sender_proc = zoid_to_proc.at(sender);
            
            for (const auto& receiver : receivers) {
                if (zoid_to_proc.find(receiver) == zoid_to_proc.end()) continue;
                int receiver_proc = zoid_to_proc.at(receiver);
                
                if (sender_proc != receiver_proc) {
                    edge_cuts++;
                }
            }
        }
        
        return edge_cuts;
    }

    // Helper function to calculate processor centroid
    std::array<double, 3> calculateProcessorCentroid(const std::vector<std::array<int, 3>>& zoids) {
        std::array<double, 3> centroid = {0, 0, 0};
        if (zoids.empty()) return centroid;
        
        for (const auto& z : zoids) {
            centroid[0] += z[0];
            centroid[1] += z[1];
            centroid[2] += z[2];
        }
        
        centroid[0] /= zoids.size();
        centroid[1] /= zoids.size();
        centroid[2] /= zoids.size();
        
        return centroid;
    }

    // Helper function for Euclidean distance
    double euclideanDistance(const std::array<int, 3>& a, const std::array<double, 3>& b) {
        double dx = a[0] - b[0];
        double dy = a[1] - b[1];
        double dz = a[2] - b[2];
        return std::sqrt(dx*dx + dy*dy + dz*dz);
    }

    // Helper function: Rebalance dependency levels across processors
    void rebalanceDependencyLevels(
        std::vector<std::vector<std::array<int, 3>>>& proc_zoids,
        std::vector<std::vector<int>>& proc_dep_counts,
        const std::vector<int>& target_dep_per_proc,
        const std::map<std::array<int, 3>, std::set<std::array<int, 3>>>& send_neighbors,
        const std::map<std::array<int, 3>, std::set<std::array<int, 3>>>& recv_neighbors) {
        
        int nprocs = proc_zoids.size();
        
        // For each dependency level, balance across processors
        for (int dep = 0; dep < 4; dep++) {
            bool balanced = false;
            int iterations = 0;
            const int MAX_ITERATIONS = 10000;
            
            while (!balanced && iterations < MAX_ITERATIONS) {
                balanced = true;
                iterations++;
                
                // Find processors with excess and deficit for this dependency level
                std::vector<int> excess_procs, deficit_procs;
                
                for (int p = 0; p < nprocs; p++) {
                    int diff = proc_dep_counts[p][dep] - target_dep_per_proc[dep];
                    if (diff > 0) {
                        excess_procs.push_back(p);
                    } else if (diff < 0) {
                        deficit_procs.push_back(p);
                    }
                }
                
                if (excess_procs.empty() || deficit_procs.empty()) {
                    continue; // This dependency level is balanced
                }
                
                // Try to move zoids from excess to deficit processors
                for (int from_proc : excess_procs) {
                    if (proc_dep_counts[from_proc][dep] <= target_dep_per_proc[dep]) {
                        continue;
                    }
                    
                    for (int to_proc : deficit_procs) {
                        if (proc_dep_counts[to_proc][dep] >= target_dep_per_proc[dep]) {
                            continue;
                        }
                        
                        // Find best zoid to swap
                        int best_from_idx = -1;
                        int best_to_idx = -1;
                        int min_disruption = INT_MAX;
                        
                        // Look for zoids to swap
                        for (int i = 0; i < proc_zoids[from_proc].size(); i++) {
                            const auto& from_zoid = proc_zoids[from_proc][i];
                            int from_dep = (from_zoid[0] % 2 == 0) + 
                                        (from_zoid[1] % 2 == 0) + 
                                        (from_zoid[2] % 2 == 0);
                            
                            if (from_dep != dep) continue; // Only move zoids of target dependency
                            
                            // Find a zoid to swap with (different dependency level)
                            for (int j = 0; j < proc_zoids[to_proc].size(); j++) {
                                const auto& to_zoid = proc_zoids[to_proc][j];
                                int to_dep = (to_zoid[0] % 2 == 0) + 
                                            (to_zoid[1] % 2 == 0) + 
                                            (to_zoid[2] % 2 == 0);
                                
                                if (to_dep == dep) continue; // Don't swap same dependency
                                
                                // Check if this swap would help balance other dependency levels
                                if (proc_dep_counts[from_proc][to_dep] >= target_dep_per_proc[to_dep] &&
                                    proc_dep_counts[to_proc][to_dep] <= target_dep_per_proc[to_dep]) {
                                    continue; // Would make other dependency worse
                                }
                                
                                // Calculate communication disruption
                                int disruption = calculateSwapDisruption(
                                    from_zoid, to_zoid, from_proc, to_proc,
                                    proc_zoids, send_neighbors, recv_neighbors);
                                
                                if (disruption < min_disruption) {
                                    min_disruption = disruption;
                                    best_from_idx = i;
                                    best_to_idx = j;
                                }
                            }
                        }
                        
                        // Perform the swap if found
                        if (best_from_idx >= 0 && best_to_idx >= 0) {
                            auto from_zoid = proc_zoids[from_proc][best_from_idx];
                            auto to_zoid = proc_zoids[to_proc][best_to_idx];
                            
                            int from_dep = (from_zoid[0] % 2 == 0) + 
                                        (from_zoid[1] % 2 == 0) + 
                                        (from_zoid[2] % 2 == 0);
                            int to_dep = (to_zoid[0] % 2 == 0) + 
                                        (to_zoid[1] % 2 == 0) + 
                                        (to_zoid[2] % 2 == 0);
                            
                            // Swap
                            proc_zoids[from_proc][best_from_idx] = to_zoid;
                            proc_zoids[to_proc][best_to_idx] = from_zoid;
                            
                            // Update counts
                            proc_dep_counts[from_proc][from_dep]--;
                            proc_dep_counts[from_proc][to_dep]++;
                            proc_dep_counts[to_proc][to_dep]--;
                            proc_dep_counts[to_proc][from_dep]++;
                            
                            balanced = false;
                            break;
                        }
                    }
                    if (!balanced) break;
                }
            }
        }
        
        // Verify dependency balance
        if (comm->me == 0) {
            std::cout << "\nDependency balance after rebalancing:" << std::endl;
            for (int p = 0; p < nprocs; p++) {
                std::cout << "Proc " << p << ": ";
                for (int dep = 0; dep < 4; dep++) {
                    std::cout << "dep" << dep << "=" << proc_dep_counts[p][dep] << " ";
                }
                std::cout << std::endl;
            }
        }
    }

    // Helper function: Calculate disruption from swapping two zoids
    int calculateSwapDisruption(
        const std::array<int, 3>& zoid1,
        const std::array<int, 3>& zoid2,
        int proc1, int proc2,
        const std::vector<std::vector<std::array<int, 3>>>& proc_zoids,
        const std::map<std::array<int, 3>, std::set<std::array<int, 3>>>& send_neighbors,
        const std::map<std::array<int, 3>, std::set<std::array<int, 3>>>& recv_neighbors) {
        
        int disruption = 0;
        
        // Build quick lookup for zoid->processor
        std::map<std::array<int, 3>, int> zoid_to_proc;
        for (int p = 0; p < proc_zoids.size(); p++) {
            for (const auto& z : proc_zoids[p]) {
                zoid_to_proc[z] = p;
            }
        }
        
        // Calculate disruption for zoid1 moving from proc1 to proc2
        if (send_neighbors.find(zoid1) != send_neighbors.end()) {
            for (const auto& neighbor : send_neighbors.at(zoid1)) {
                if (zoid_to_proc.find(neighbor) != zoid_to_proc.end()) {
                    int neighbor_proc = zoid_to_proc[neighbor];
                    if (neighbor_proc == proc1) disruption++; // Local becomes remote
                    if (neighbor_proc == proc2) disruption--; // Remote becomes local
                }
            }
        }
        
        if (recv_neighbors.find(zoid1) != recv_neighbors.end()) {
            for (const auto& neighbor : recv_neighbors.at(zoid1)) {
                if (zoid_to_proc.find(neighbor) != zoid_to_proc.end()) {
                    int neighbor_proc = zoid_to_proc[neighbor];
                    if (neighbor_proc == proc1) disruption++; // Local becomes remote
                    if (neighbor_proc == proc2) disruption--; // Remote becomes local
                }
            }
        }
        
        // Calculate disruption for zoid2 moving from proc2 to proc1
        if (send_neighbors.find(zoid2) != send_neighbors.end()) {
            for (const auto& neighbor : send_neighbors.at(zoid2)) {
                if (zoid_to_proc.find(neighbor) != zoid_to_proc.end()) {
                    int neighbor_proc = zoid_to_proc[neighbor];
                    if (neighbor_proc == proc2) disruption++; // Local becomes remote
                    if (neighbor_proc == proc1) disruption--; // Remote becomes local
                }
            }
        }
        
        if (recv_neighbors.find(zoid2) != recv_neighbors.end()) {
            for (const auto& neighbor : recv_neighbors.at(zoid2)) {
                if (zoid_to_proc.find(neighbor) != zoid_to_proc.end()) {
                    int neighbor_proc = zoid_to_proc[neighbor];
                    if (neighbor_proc == proc2) disruption++; // Local becomes remote
                    if (neighbor_proc == proc1) disruption--; // Remote becomes local
                }
            }
        }
        
        return disruption;
    }

    // Helper function: Balance assignment with minimal disruption
    void balanceWithMinimalDisruption(
        std::map<int, std::vector<std::array<int, 3>>>& proc_to_zoids,
        std::map<std::array<int, 3>, int>& zoid_to_proc,
        const std::map<std::array<int, 3>, std::set<std::array<int, 3>>>& send_neighbors,
        const std::map<std::array<int, 3>, std::set<std::array<int, 3>>>& recv_neighbors,
        int target_per_proc) {
        
        // Identify overloaded and underloaded processors
        std::vector<int> overloaded, underloaded;
        std::vector<int> excess(proc_to_zoids.size());
        
        for (int p = 0; p < proc_to_zoids.size(); p++) {
            int current = proc_to_zoids[p].size();
            excess[p] = current - target_per_proc;
            
            if (excess[p] > 0) {
                overloaded.push_back(p);
            } else if (excess[p] < 0) {
                underloaded.push_back(p);
            }
        }
        
        // Move zoids from overloaded to underloaded processors
        for (int from_proc : overloaded) {
            while (excess[from_proc] > 0 && !underloaded.empty()) {
                // Find best zoid to move (minimizes communication disruption)
                int best_zoid_idx = -1;
                int best_to_proc = -1;
                int min_disruption = INT_MAX;
                
                // Evaluate each zoid in the overloaded processor
                for (int i = 0; i < proc_to_zoids[from_proc].size(); i++) {
                    const auto& zoid = proc_to_zoids[from_proc][i];
                    
                    // Try each underloaded processor
                    for (int to_proc : underloaded) {
                        if (excess[to_proc] >= 0) continue;
                        
                        // Calculate disruption: how many communications would cross processors
                        int disruption = 0;
                        
                        // Check sends from this zoid
                        if (send_neighbors.find(zoid) != send_neighbors.end()) {
                            for (const auto& receiver : send_neighbors.at(zoid)) {
                                if (zoid_to_proc.find(receiver) != zoid_to_proc.end()) {
                                    int recv_proc = zoid_to_proc[receiver];
                                    // Currently local, would become remote
                                    if (recv_proc == from_proc) disruption++;
                                    // Currently remote, would become local
                                    if (recv_proc == to_proc) disruption--;
                                }
                            }
                        }
                        
                        // Check receives to this zoid
                        if (recv_neighbors.find(zoid) != recv_neighbors.end()) {
                            for (const auto& sender : recv_neighbors.at(zoid)) {
                                if (zoid_to_proc.find(sender) != zoid_to_proc.end()) {
                                    int send_proc = zoid_to_proc[sender];
                                    // Currently local, would become remote
                                    if (send_proc == from_proc) disruption++;
                                    // Currently remote, would become local
                                    if (send_proc == to_proc) disruption--;
                                }
                            }
                        }
                        
                        if (disruption < min_disruption) {
                            min_disruption = disruption;
                            best_zoid_idx = i;
                            best_to_proc = to_proc;
                        }
                    }
                }
                
                // Move the best zoid
                if (best_zoid_idx >= 0 && best_to_proc >= 0) {
                    auto zoid = proc_to_zoids[from_proc][best_zoid_idx];
                    
                    // Remove from source processor
                    proc_to_zoids[from_proc].erase(
                        proc_to_zoids[from_proc].begin() + best_zoid_idx);
                    
                    // Add to destination processor
                    proc_to_zoids[best_to_proc].push_back(zoid);
                    zoid_to_proc[zoid] = best_to_proc;
                    
                    // Update excess counts
                    excess[from_proc]--;
                    excess[best_to_proc]++;
                    
                    // Remove processor from underloaded list if it's now balanced
                    if (excess[best_to_proc] == 0) {
                        underloaded.erase(
                            std::remove(underloaded.begin(), underloaded.end(), best_to_proc),
                            underloaded.end());
                    }
                } else {
                    // No good move found, force move the first zoid
                    auto zoid = proc_to_zoids[from_proc][0];
                    int to_proc = underloaded[0];
                    
                    proc_to_zoids[from_proc].erase(proc_to_zoids[from_proc].begin());
                    proc_to_zoids[to_proc].push_back(zoid);
                    zoid_to_proc[zoid] = to_proc;
                    
                    excess[from_proc]--;
                    excess[to_proc]++;
                    
                    if (excess[to_proc] == 0) {
                        underloaded.erase(underloaded.begin());
                    }
                }
            }
        }
        
        // Verify perfect balance
        for (int p = 0; p < proc_to_zoids.size(); p++) {
            assert(proc_to_zoids[p].size() == target_per_proc);
        }
    }

    // Print statistics for balanced assignment
    void printBalancedAssignmentStats(
        const std::map<int, std::vector<std::array<int, 3>>>& proc_to_zoids,
        const std::map<std::array<int, 3>, int>& zoid_to_proc,
        const std::map<std::array<int, 3>, std::set<std::array<int, 3>>>& send_neighbors,
        int expected_zoids_per_proc) {
        
        std::cout << "\n=== Balanced Assignment Statistics ===" << std::endl;
        
        // Verify perfect load balance
        std::cout << "\nLoad Distribution (expecting " << expected_zoids_per_proc << " per proc):" << std::endl;
        for (int p = 0; p < proc_to_zoids.size(); p++) {
            int load = proc_to_zoids.at(p).size();
            std::cout << "Processor " << p << ": " << load << " zoids";
            if (load != expected_zoids_per_proc) {
                std::cout << " [ERROR: IMBALANCED!]";
            }
            std::cout << std::endl;
        }
        
        // Per-dependency-level distribution
        std::cout << "\nPer-dependency-level distribution:" << std::endl;
        std::vector<std::vector<int>> proc_dep_count(proc_to_zoids.size(), std::vector<int>(4, 0));
        std::vector<int> total_dep_count(4, 0);
        
        for (int p = 0; p < proc_to_zoids.size(); p++) {
            for (const auto& zoid : proc_to_zoids.at(p)) {
                int zoid_dep = (zoid[0] % 2 == 0) + (zoid[1] % 2 == 0) + (zoid[2] % 2 == 0);
                proc_dep_count[p][zoid_dep]++;
                total_dep_count[zoid_dep]++;
            }
        }
        
        // Print expected vs actual
        for (int dep = 0; dep < 4; dep++) {
            int expected_per_proc = total_dep_count[dep] / proc_to_zoids.size();
            std::cout << "  Dep " << dep << " (total=" << total_dep_count[dep] 
                    << ", expected=" << expected_per_proc << "/proc): ";
            
            int min_count = INT_MAX, max_count = 0;
            for (int p = 0; p < proc_to_zoids.size(); p++) {
                min_count = std::min(min_count, proc_dep_count[p][dep]);
                max_count = std::max(max_count, proc_dep_count[p][dep]);
            }
            
            if (max_count - min_count <= 1) {
                std::cout << "BALANCED (" << min_count << "-" << max_count << ")" << std::endl;
            } else {
                std::cout << "IMBALANCED (range " << min_count << "-" << max_count << ")" << std::endl;
                // Show distribution
                for (int p = 0; p < proc_to_zoids.size(); p++) {
                    std::cout << "    P" << p << "=" << proc_dep_count[p][dep] << " ";
                }
                std::cout << std::endl;
            }
        }
        
        // Communication analysis
        int total_edges = 0;
        int cut_edges = 0;
        std::map<std::pair<int, int>, int> proc_comm_matrix;
        
        for (const auto& [sender, receivers] : send_neighbors) {
            if (zoid_to_proc.find(sender) == zoid_to_proc.end()) continue;
            int sender_proc = zoid_to_proc.at(sender);
            
            for (const auto& receiver : receivers) {
                if (zoid_to_proc.find(receiver) == zoid_to_proc.end()) continue;
                int receiver_proc = zoid_to_proc.at(receiver);
                
                total_edges++;
                if (sender_proc != receiver_proc) {
                    cut_edges++;
                    proc_comm_matrix[{sender_proc, receiver_proc}]++;
                }
            }
        }
        
        std::cout << "\nCommunication Statistics:" << std::endl;
        std::cout << "Total edges: " << total_edges << std::endl;
        std::cout << "Cut edges (inter-process): " << cut_edges << std::endl;
        std::cout << "Edge locality: " << std::fixed << std::setprecision(2) 
                << (100.0 * (total_edges - cut_edges) / total_edges) << "%" << std::endl;
        
        // Find max communication between any processor pair
        int max_comm = 0;
        std::pair<int, int> max_comm_pair;
        for (const auto& [procs, count] : proc_comm_matrix) {
            if (count > max_comm) {
                max_comm = count;
                max_comm_pair = procs;
            }
        }
        
        std::cout << "\nMax inter-process communication: " << max_comm 
                << " messages (Proc " << max_comm_pair.first 
                << " -> Proc " << max_comm_pair.second << ")" << std::endl;
        
        // Per-processor communication load
        std::cout << "\nPer-processor communication:" << std::endl;
        for (int p = 0; p < proc_to_zoids.size(); p++) {
            int sends_out = 0, receives_in = 0;
            
            for (const auto& zoid : proc_to_zoids.at(p)) {
                // Count sends
                if (send_neighbors.find(zoid) != send_neighbors.end()) {
                    for (const auto& receiver : send_neighbors.at(zoid)) {
                        if (zoid_to_proc.find(receiver) != zoid_to_proc.end() &&
                            zoid_to_proc.at(receiver) != p) {
                            sends_out++;
                        }
                    }
                }
                
                // Count receives
                auto recv_it = std::find_if(send_neighbors.begin(), send_neighbors.end(),
                    [&](const auto& pair) {
                        return pair.second.find(zoid) != pair.second.end();
                    });
                
                if (recv_it != send_neighbors.end()) {
                    for (const auto& [sender, receivers] : send_neighbors) {
                        if (receivers.find(zoid) != receivers.end() &&
                            zoid_to_proc.find(sender) != zoid_to_proc.end() &&
                            zoid_to_proc.at(sender) != p) {
                            receives_in++;
                        }
                    }
                }
            }
            
            std::cout << "  Proc " << p << ": " << sends_out << " sends out, " 
                    << receives_in << " receives in" << std::endl;
        }
    }


    // END CLAUDE CODE
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

                zoid.lo = new std::array<double, 3>[NUM_TIMESTEPS_IN_PARALLEL + 1];
                zoid.hi = new std::array<double, 3>[NUM_TIMESTEPS_IN_PARALLEL + 1];

                for (int t = 0; t < NUM_TIMESTEPS_IN_PARALLEL + 1; t++) {
                    for (int dim = 0; dim < domain->dimension; dim++) {
                        zoid.lo[t][dim] = zoid.zoid.cuts[dim].lower + t * zoid.zoid.cuts[dim].slope_lower;
                        zoid.hi[t][dim] = zoid.zoid.cuts[dim].upper + t * zoid.zoid.cuts[dim].slope_upper;
                    }
                }

                if (zoid.num % comm->nprocs == comm->me) {
                    /* start stuff for 2 timesteps */

                    assert(domain->dimension == 3);

                    zoid.x_stencil_md = new std::vector<dbl3_t_stencil_md>[DOUBLE_BUFFERING];
                    if constexpr (EXPERIMENT == DPD) {
                        zoid.v_stencil_md = new std::vector<dbl3_t_stencil_md>[DOUBLE_BUFFERING];
                    } else {
                        zoid.v_stencil_md = new std::vector<dbl3_t_stencil_md>[1];
                    }
                    zoid.f_stencil_md = new std::vector<dbl3_t_stencil_md>[1];
                    zoid.eval_f_stencil_md = new std::vector<dbl3_t_stencil_md>[1];

                    zoid.tag_stencil_md = new std::vector<int>[1];
                    zoid.type_stencil_md = new std::vector<int>[1];
                    zoid.mask_stencil_md = new std::vector<int>[1];
                    zoid.image_stencil_md = new std::vector<int>[1];
                    zoid.spinlocks_stencil_md = new spinlock*[1];
                    zoid.claimed_flags_stencil_md = new std::atomic_flag*[1];

                    zoid.local_idxs_per_timestep = new std::vector<int>[NUM_TIMESTEPS_IN_PARALLEL + 1];
                    // zoid.local_idxs_per_timestep_segment_idxs = new std::vector<int>[NUM_TIMESTEPS_IN_PARALLEL + 1];
                    // zoid.local_idxs_per_timestep_segment_sizes = new std::vector<int>[NUM_TIMESTEPS_IN_PARALLEL + 1];
                    // zoid.is_local_per_timestep = new std::vector<bool>[NUM_TIMESTEPS_IN_PARALLEL + 1];
                    zoid.neighbor_list = new std::vector<std::vector<int>>[NUM_TIMESTEPS_IN_PARALLEL + 1];
                    zoid.bond_list = new std::vector<std::vector<std::pair<int, int>>>[NUM_TIMESTEPS_IN_PARALLEL + 1];
                    zoid.bond_list_modified = new std::vector<std::tuple<int, int, int>>[NUM_TIMESTEPS_IN_PARALLEL + 1];
                    // zoid.bond_list_modified_num_colors = new std::vector<int>[NUM_TIMESTEPS_IN_PARALLEL + 1];
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

                    zoid.send_force_idxs_double_buffering_flattened_pipelined = new std::vector<int>*[NUM_PIPELINE_STAGES];
                    zoid.send_pos_idxs_double_buffering_flattened_pipelined = new std::vector<int>**[NUM_PIPELINE_STAGES];
                    zoid.send_vel_idxs_double_buffering_flattened_pipelined = new std::vector<int>**[NUM_PIPELINE_STAGES];

                    zoid.recv_force_idxs_double_buffering_flattened_pipelined = new std::vector<int>*[NUM_PIPELINE_STAGES];
                    zoid.recv_pos_idxs_double_buffering_flattened_pipelined = new std::vector<int>**[NUM_PIPELINE_STAGES];
                    zoid.recv_vel_idxs_double_buffering_flattened_pipelined = new std::vector<int>**[NUM_PIPELINE_STAGES];

                    for (int p = 0; p < NUM_PIPELINE_STAGES; p++) {
                        zoid.send_force_idxs_double_buffering_flattened_pipelined[p] = new std::vector<int>[MAX_NEIGHBORS];
                        zoid.send_pos_idxs_double_buffering_flattened_pipelined[p] = new std::vector<int>*[MAX_NEIGHBORS];
                        zoid.send_vel_idxs_double_buffering_flattened_pipelined[p] = new std::vector<int>*[MAX_NEIGHBORS];

                        zoid.recv_force_idxs_double_buffering_flattened_pipelined[p] = new std::vector<int>[MAX_NEIGHBORS];
                        zoid.recv_pos_idxs_double_buffering_flattened_pipelined[p] = new std::vector<int>*[MAX_NEIGHBORS];
                        zoid.recv_vel_idxs_double_buffering_flattened_pipelined[p] = new std::vector<int>*[MAX_NEIGHBORS];

                        for (int k = 0; k < DOUBLE_BUFFERING; k++) {
                            zoid.send_pos_idxs_double_buffering_flattened_pipelined[p][k] = new std::vector<int>[MAX_NEIGHBORS];
                            zoid.recv_pos_idxs_double_buffering_flattened_pipelined[p][k] = new std::vector<int>[MAX_NEIGHBORS];

                            zoid.send_vel_idxs_double_buffering_flattened_pipelined[p][k] = new std::vector<int>[MAX_NEIGHBORS];
                            zoid.recv_vel_idxs_double_buffering_flattened_pipelined[p][k] = new std::vector<int>[MAX_NEIGHBORS];
                        }
                    }

                    /* end stuff for 2 timesteps */
                }
            }
        }

        for (int dep = 0; dep < NUM_DEPS; dep++) {
            for (int j = 0; j < queues_many_cuts_next_dt[dep].size(); j++) {
                queue_info& zoid = queues_many_cuts_next_dt[dep][j];

                zoid.lo = new std::array<double, 3>[NUM_TIMESTEPS_IN_PARALLEL + 1];
                zoid.hi = new std::array<double, 3>[NUM_TIMESTEPS_IN_PARALLEL + 1];

                for (int t = 0; t < NUM_TIMESTEPS_IN_PARALLEL + 1; t++) {
                    for (int dim = 0; dim < domain->dimension; dim++) {
                        zoid.lo[t][dim] = zoid.zoid.cuts[dim].lower + t * zoid.zoid.cuts[dim].slope_lower;
                        zoid.hi[t][dim] = zoid.zoid.cuts[dim].upper + t * zoid.zoid.cuts[dim].slope_upper;
                    }
                }

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
                    // zoid.local_idxs_per_timestep_segment_idxs = new std::vector<int>[NUM_TIMESTEPS_IN_PARALLEL + 1];
                    // zoid.local_idxs_per_timestep_segment_sizes = new std::vector<int>[NUM_TIMESTEPS_IN_PARALLEL + 1];
                    // zoid.is_local_per_timestep = new std::vector<bool>[NUM_TIMESTEPS_IN_PARALLEL + 1];
                    zoid.neighbor_list = new std::vector<std::vector<int>>[NUM_TIMESTEPS_IN_PARALLEL + 1];
                    zoid.bond_list = new std::vector<std::vector<std::pair<int, int>>>[NUM_TIMESTEPS_IN_PARALLEL + 1];
                    zoid.bond_list_modified = new std::vector<std::tuple<int, int, int>>[NUM_TIMESTEPS_IN_PARALLEL + 1];
                    // zoid.bond_list_modified_num_colors = new std::vector<int>[NUM_TIMESTEPS_IN_PARALLEL + 1];

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

                    zoid.send_force_idxs_double_buffering_flattened_pipelined = new std::vector<int>*[NUM_PIPELINE_STAGES];
                    zoid.send_pos_idxs_double_buffering_flattened_pipelined = new std::vector<int>**[NUM_PIPELINE_STAGES];
                    zoid.send_vel_idxs_double_buffering_flattened_pipelined = new std::vector<int>**[NUM_PIPELINE_STAGES];

                    zoid.recv_force_idxs_double_buffering_flattened_pipelined = new std::vector<int>*[NUM_PIPELINE_STAGES];
                    zoid.recv_pos_idxs_double_buffering_flattened_pipelined = new std::vector<int>**[NUM_PIPELINE_STAGES];
                    zoid.recv_vel_idxs_double_buffering_flattened_pipelined = new std::vector<int>**[NUM_PIPELINE_STAGES];

                    for (int p = 0; p < NUM_PIPELINE_STAGES; p++) {
                        zoid.send_force_idxs_double_buffering_flattened_pipelined[p] = new std::vector<int>[MAX_NEIGHBORS];
                        zoid.send_pos_idxs_double_buffering_flattened_pipelined[p] = new std::vector<int>*[MAX_NEIGHBORS];
                        zoid.send_vel_idxs_double_buffering_flattened_pipelined[p] = new std::vector<int>*[MAX_NEIGHBORS];

                        zoid.recv_force_idxs_double_buffering_flattened_pipelined[p] = new std::vector<int>[MAX_NEIGHBORS];
                        zoid.recv_pos_idxs_double_buffering_flattened_pipelined[p] = new std::vector<int>*[MAX_NEIGHBORS];
                        zoid.recv_vel_idxs_double_buffering_flattened_pipelined[p] = new std::vector<int>*[MAX_NEIGHBORS];

                        for (int k = 0; k < DOUBLE_BUFFERING; k++) {
                            zoid.send_pos_idxs_double_buffering_flattened_pipelined[p][k] = new std::vector<int>[MAX_NEIGHBORS];
                            zoid.recv_pos_idxs_double_buffering_flattened_pipelined[p][k] = new std::vector<int>[MAX_NEIGHBORS];

                            zoid.send_vel_idxs_double_buffering_flattened_pipelined[p][k] = new std::vector<int>[MAX_NEIGHBORS];
                            zoid.recv_vel_idxs_double_buffering_flattened_pipelined[p][k] = new std::vector<int>[MAX_NEIGHBORS];
                        }
                    }
                    /* end stuff for 2 timesteps */
                }
            }
        }

        zoid_num_to_zoid_many_cuts = new queue_info[NUM_ZOIDS_MANY_CUTS];
        zoid_num_to_dep.resize(NUM_ZOIDS_MANY_CUTS);

        for (int dep = 0; dep < NUM_DEPS; dep++) {
            for (int j = 0; j < queues_many_cuts[dep].size(); j++) {
                int zoid_num = queues_many_cuts[dep][j].num;
                zoid_num_to_zoid_many_cuts[zoid_num] = queues_many_cuts[dep][j];
                zoid_num_to_dep[zoid_num] = dep;
            }
        }

        zoid_num_to_zoid_many_cuts_next_dt = new queue_info[NUM_ZOIDS_MANY_CUTS];
        zoid_num_to_dep_next_dt.resize(NUM_ZOIDS_MANY_CUTS);

        for (int dep = 0; dep < NUM_DEPS; dep++) {
            for (int j = 0; j < queues_many_cuts_next_dt[dep].size(); j++) {
                int zoid_num = queues_many_cuts_next_dt[dep][j].num;
                zoid_num_to_zoid_many_cuts_next_dt[zoid_num] = queues_many_cuts_next_dt[dep][j];
                zoid_num_to_dep_next_dt[zoid_num] = dep;
            }
        }
    }

    void INIT_MY_ZOIDS() {
        for (int dep = 0; dep < NUM_DEPS; dep++) {
            for (int j = 0; j < queues_many_cuts[dep].size(); j++) {
                auto zoid = queues_many_cuts[dep][j];
                if (zoid.num % comm->nprocs != comm->me) {
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

        /*
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
        */
    }

    bool zoid_many_cuts_is_neighbor_all_deps(int* where_a, int* where_b) {
        for (int dim = 0; dim < domain->dimension; dim++) {
            int num_zoids_in_dimension;
            if (dim == 0) {
                num_zoids_in_dimension = NUM_ZOIDS_X;
            } else if (dim == 1) {
                num_zoids_in_dimension = NUM_ZOIDS_Y;
            } else {
                num_zoids_in_dimension = NUM_ZOIDS_Z;
            }

            if (where_a[dim] != where_b[dim]) {
                int diff = where_a[dim] - where_b[dim];
                // expanding zoid in this dimension based on numbering
                if (where_a[dim] % 2 == 1) {
                    if (diff != -1 && diff != 1) {
                        if (!(where_a[dim] == num_zoids_in_dimension - 1 && where_b[dim] == 0)) {
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
            int num_zoids_in_dimension;
            if (dim == 0) {
                num_zoids_in_dimension = NUM_ZOIDS_X;
            } else if (dim == 1) {
                num_zoids_in_dimension = NUM_ZOIDS_Y;
            } else {
                num_zoids_in_dimension = NUM_ZOIDS_Z;
            }

            if (where_a[dim] != where_b[dim]) {
                int diff = where_a[dim] - where_b[dim];
                // expanding zoid in this dimension based on numbering
                if (where_a[dim] % 2 == 0) {
                    if (diff != -1 && diff != 1) {
                        if (!(where_a[dim] == 0 && where_b[dim] == num_zoids_in_dimension - 1)) {
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
            std::sort(send_to_neighbors_many_cuts_next_dt[i].begin(),
                      send_to_neighbors_many_cuts_next_dt[i].end(), std::greater<int>());
            std::sort(recv_from_neighbors_many_cuts_next_dt[i].begin(),
                      recv_from_neighbors_many_cuts_next_dt[i].end(), std::greater<int>());
        }

        int comm_idx = 0;
        for (int dep = 0; dep < NUM_DEPS - 1; dep++) {
            for (int j = 0; j < my_queues_many_cuts[dep].size(); j++) {
                auto& zoid = my_queues_many_cuts[dep][j];
                int zoid_num = zoid.num;
                auto& send_neighbors = send_to_neighbors_many_cuts[zoid_num];
                for (int i = 0; i < send_neighbors.size(); i++) {
                    int send_zoid_num = send_neighbors[i];
                    if (send_zoid_num % comm->nprocs != comm->me) {
                        ZOID_TO_ZOID_TO_VCI_IDX[1][{zoid_num, send_zoid_num}] = (comm_idx) % NUM_COMMS;
                        comm_idx++;
                    }
                }
                // comm_idx++;
            }
        }

        // int comm_idx = 0;
        comm_idx = 0;
        for (int dep = 0; dep < NUM_DEPS - 1; dep++) {
            for (int j = 0; j < my_queues_many_cuts_next_dt[dep].size(); j++) {
                auto& zoid = my_queues_many_cuts_next_dt[dep][j];
                int zoid_num = zoid.num;
                auto& send_neighbors = send_to_neighbors_many_cuts_next_dt[zoid_num];
                for (int i = 0; i < send_neighbors.size(); i++) {
                    int send_zoid_num = send_neighbors[i];
                    if (send_zoid_num % comm->nprocs != comm->me) {
                        ZOID_TO_ZOID_TO_VCI_IDX[0][{zoid_num, send_zoid_num}] = (comm_idx) % NUM_COMMS;
                        // ZOID_TO_ZOID_TO_VCI_IDX_NEXT_DT[{zoid_num, send_zoid_num}] = (comm_idx) % NUM_COMMS;
                        comm_idx++;
                    }
                }
                // comm_idx++;
            }
        }

        if (USE_STREAMS) {
            zoid_to_stream_num[1].resize(NUM_ZOIDS_MANY_CUTS);
            int stream_idx = 0;
            for (int dep = 0; dep < NUM_DEPS; dep++) {
                for (int j = 0; j < my_queues_many_cuts[dep].size(); j++) {
                    auto& zoid = my_queues_many_cuts[dep][j];
                    int zoid_num = zoid.num;
                    zoid_to_stream_num[1][zoid_num] = stream_idx;
                    assert(stream_idx < NUM_STREAMS);
                    stream_idx++;
                }
            }

            zoid_to_stream_num[0].resize(NUM_ZOIDS_MANY_CUTS);
            stream_idx = 0;
            for (int dep = 0; dep < NUM_DEPS; dep++) {
                for (int j = 0; j < my_queues_many_cuts_next_dt[dep].size(); j++) {
                    auto& zoid = my_queues_many_cuts_next_dt[dep][j];
                    int zoid_num = zoid.num;
                    zoid_to_stream_num[0][zoid_num] = stream_idx;
                    assert(stream_idx < NUM_STREAMS);
                    stream_idx++;
                }
            }
            MPI_Allreduce(MPI_IN_PLACE, zoid_to_stream_num[1].data(), NUM_ZOIDS_MANY_CUTS, MPI_INT, MPI_SUM, world);
            MPI_Allreduce(MPI_IN_PLACE, zoid_to_stream_num[0].data(), NUM_ZOIDS_MANY_CUTS, MPI_INT, MPI_SUM, world);
            for (int i = 0; i < NUM_ZOIDS_MANY_CUTS; i++) {
                assert(zoid_to_stream_num[1][i] >= 0 && zoid_to_stream_num[1][i] < NUM_STREAMS);
                assert(zoid_to_stream_num[0][i] >= 0 && zoid_to_stream_num[0][i] < NUM_STREAMS);
            }
        }

        std::vector<int> my_zoids_src;
        std::vector<int> my_zoids_dst;
        std::vector<int> my_zoids_comm_idx;

        for (auto& [k, v] : ZOID_TO_ZOID_TO_VCI_IDX[1]) {
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

        std::vector<int> comm_counts(NUM_COMMS, 0);

        for (int i = 0; i < all_src.size(); i++) {
            int src = all_src[i];
            int dst = all_dst[i];
            int comm_idx = all_comm_idx[i];
            ZOID_TO_ZOID_TO_VCI_IDX[1][{src, dst}] = comm_idx;
            comm_counts[comm_idx]++;
        }

        if (comm->me == 0) {
            for (int i = 0; i < NUM_COMMS; i++) {
                std::cout << "comm: " << i << " num comms: " << comm_counts[i] << std::endl;
            }
        }

        std::vector<int> my_zoids_src_next_dt;
        std::vector<int> my_zoids_dst_next_dt;
        std::vector<int> my_zoids_comm_idx_next_dt;

        // for (auto& [k, v] : ZOID_TO_ZOID_TO_VCI_IDX_NEXT_DT) {
        for (auto& [k, v] : ZOID_TO_ZOID_TO_VCI_IDX[0]) {
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
            ZOID_TO_ZOID_TO_VCI_IDX[0][{src, dst}] = comm_idx;
            // ZOID_TO_ZOID_TO_VCI_IDX_NEXT_DT[{src, dst}] = comm_idx;
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

        if (comm->me == 0) {
            int total_num_diff_proc = 0;

            for (int i = 0; i < NUM_ZOIDS_MANY_CUTS; i++) {
                auto& zoid = zoid_num_to_zoid_many_cuts[i];
                int dep = (zoid.where[0] % 2 == 0) + (zoid.where[1] % 2 == 0) + (zoid.where[2] % 2 == 0);
                auto& recv_neighbors = recv_from_neighbors_many_cuts[zoid.num];

                int num_diff_proc = 0;
                for (int j = 0; j < recv_neighbors.size(); j++) {
                    if (recv_neighbors[j] % comm->nprocs != zoid.num % comm->nprocs) {
                        auto& recv_zoid = zoid_num_to_zoid_many_cuts[recv_neighbors[j]];
                        int recv_zoid_dep = (recv_zoid.where[0] % 2 == 0) + (recv_zoid.where[1] % 2 == 0) + (recv_zoid.where[2] % 2 == 0);
                        if (recv_zoid_dep == dep - 1) {
                            num_diff_proc++;
                            total_num_diff_proc++;
                        }
                    }
                }
            }

            std::cout << "total num diff proc: " << total_num_diff_proc << std::endl;
        }
    }

    template <bool curr_dt>
    void CONSTRUCT_RECV_PROC_TO_PROC_AND_ZOID_TO_ZOID_RECEIVE_REQUEST_IDXS() {
        int curr_dt_idx = static_cast<int>(curr_dt);
        constexpr int num_p = USE_PIPELINE ? NUM_PIPELINE_STAGES : 1;

        auto& my_queues = curr_dt ? my_queues_many_cuts : my_queues_many_cuts_next_dt;

        // only next dep, start with curr_dt
        for (int p = 0; p < num_p; p++) {
            std::vector<std::set<std::pair<int, int>>> deps_seen(comm->nprocs);

            for (int dep = 1; dep < NUM_DEPS; dep++) {
                int recv_request_idx = 0;
                for (int j = 0; j < my_queues[dep].size(); j++) {
                    auto& zoid = my_queues[dep][j];
                    auto& recv_neighbors = curr_dt ? recv_from_neighbors_many_cuts[zoid.num] 
                            : recv_from_neighbors_many_cuts_next_dt[zoid.num];
                    for (int i = 0; i < recv_neighbors.size(); i++) {
                        int recv_zoid_num = recv_neighbors[i];
                        int recv_zoid_dep = curr_dt ? zoid_num_to_dep[recv_zoid_num] : zoid_num_to_dep_next_dt[recv_zoid_num];
                        if (recv_neighbors[i] % comm->nprocs != comm->me && recv_zoid_dep == dep - 1) {
                            auto pair = std::make_pair(recv_neighbors[i], zoid.num);
                            recv_request_idx_to_zoid_with_proc_to_proc[curr_dt_idx][p][dep][recv_request_idx] = pair;
                            recv_request_zoid_to_idx_with_proc_to_proc[curr_dt_idx][p][dep][pair] = recv_request_idx;
                            recv_request_idx++;
                        }
                    }
                }

                nrecv_zoid_to_zoid[curr_dt_idx][p][dep] = recv_request_idx;

                for (int send_dep = 0; send_dep < dep - 1; send_dep++) {
                    for (int proc = 0; proc < comm->nprocs; proc++) {
                        auto& zoids_recv_from_proc = recv_zoids_from_proc[curr_dt_idx][p][send_dep][proc];

                        if (zoids_recv_from_proc.size() > 0 && deps_seen[proc].find({send_dep, dep}) == deps_seen[proc].end()) {
                            bool does_dep_recv_zoids = false;
                            std::vector<int> zoids_my_dep_recv;

                            for (auto& lst_info : zoids_recv_from_proc) {
                                int recv_zoid_num = lst_info[0];
                                int recv_zoid_dep = curr_dt ? zoid_num_to_dep[recv_zoid_num] : zoid_num_to_dep_next_dt[recv_zoid_num];
                                if (recv_zoid_dep == dep) {
                                    does_dep_recv_zoids = true;
                                    zoids_my_dep_recv.push_back(recv_zoid_num);
                                }
                            }

                            if (does_dep_recv_zoids) {
                                recv_request_idx_to_zoid_with_proc_to_proc[curr_dt_idx][p][dep][recv_request_idx] = {proc, send_dep};
                                recv_request_idx++;

                                dep_to_recv_proc_to_proc[curr_dt_idx][p][dep].emplace_back(send_dep, proc);
                                int nrecv_from_proc = 0;
                                // auto& lst_zoids_from_proc = recv_zoids_from_proc[curr_dt_idx][p][send_dep][proc];
                                for (auto& lst_info : zoids_recv_from_proc) {
                                    int recv_zoid_num = lst_info[0];
                                    int send_zoid_num = lst_info[1];
                                    int find_idx = lst_info[2];
                                    nrecv_from_proc += recv_proc_zoid_sizes[curr_dt_idx][p][recv_zoid_num][find_idx];
                                }
                                dep_to_recv_proc_to_proc_sizes[curr_dt_idx][p][dep].push_back(nrecv_from_proc);

                                for (auto& info : zoids_recv_from_proc) {
                                    int zoid_dep = curr_dt ? zoid_num_to_dep[info[0]] : zoid_num_to_dep_next_dt[info[0]];
                                    deps_seen[proc].insert({send_dep, zoid_dep});
                                }
                            }
                        }

                    }
                }

                if (curr_dt && p == 0) {
                    int total_size = dep_to_recv_proc_to_proc[curr_dt][p][dep].size();
                    MPI_Allreduce(MPI_IN_PLACE, &total_size, 1, MPI_INT, MPI_SUM, world);
                    if (comm->me == 0) {
                        std::cout << "dep: " << dep << " total recv: " << total_size << std::endl;
                    }
                }

                std::set<std::pair<int, int>> test;
                test.insert(dep_to_recv_proc_to_proc[curr_dt_idx][p][dep].begin(), dep_to_recv_proc_to_proc[curr_dt_idx][p][dep].end());
                assert(test.size() == dep_to_recv_proc_to_proc[curr_dt_idx][p][dep].size());
            }
        }
    }

    template <bool curr_dt>
    void CONSTRUCT_RECV_ZOID_PAIR_AND_PROC_PAIR_IDXS() {
        int curr_dt_idx = static_cast<int>(curr_dt);

        auto& my_queues = curr_dt ? my_queues_many_cuts : my_queues_many_cuts_next_dt;

        for (int dep = 1; dep < NUM_DEPS; dep++) {
            int recv_request_idx = 0;
            for (int j = 0; j < my_queues[dep].size(); j++) {
                auto& zoid = my_queues[dep][j];
                auto& recv_neighbors = curr_dt ? recv_from_neighbors_many_cuts[zoid.num] 
                        : recv_from_neighbors_many_cuts_next_dt[zoid.num];
                for (int i = 0; i < recv_neighbors.size(); i++) {
                    int recv_zoid_num = recv_neighbors[i];
                    int recv_zoid_dep = curr_dt ? zoid_num_to_dep[recv_zoid_num] : zoid_num_to_dep_next_dt[recv_zoid_num];
                    if (recv_neighbors[i] % comm->nprocs != comm->me && recv_zoid_dep == dep - 1) {
                        auto pair = std::make_pair(recv_neighbors[i], zoid.num);
                        recv_request_idx_to_zoid_pair[curr_dt_idx][dep][recv_request_idx] = pair;
                        recv_request_zoid_pair_to_idx[curr_dt_idx][dep][pair] = recv_request_idx;
                        recv_request_idx++;
                    }
                }
            }
        }

        // only next dep, start with curr_dt
        std::vector<std::set<std::pair<int, int>>> deps_seen(comm->nprocs);

        for (int dep = 2; dep < NUM_DEPS; dep++) {
            int recv_request_idx = 0;
            for (int send_dep = 0; send_dep < dep - 1; send_dep++) {
                for (int proc = 0; proc < comm->nprocs; proc++) {
                    auto& zoids_recv_from_proc = recv_zoids_from_proc[curr_dt_idx][DEFAULT_PIPELINE_STAGE][send_dep][proc];

                    if (zoids_recv_from_proc.size() > 0 && deps_seen[proc].find({send_dep, dep}) == deps_seen[proc].end()) {
                        bool does_dep_recv_zoids = false;
                        std::vector<int> zoids_my_dep_recv;

                        for (auto& lst_info : zoids_recv_from_proc) {
                            int recv_zoid_num = lst_info[0];
                            int recv_zoid_dep = curr_dt ? zoid_num_to_dep[recv_zoid_num] : zoid_num_to_dep_next_dt[recv_zoid_num];
                            if (recv_zoid_dep == dep) {
                                does_dep_recv_zoids = true;
                                zoids_my_dep_recv.push_back(recv_zoid_num);
                            }
                        }

                        if (does_dep_recv_zoids) {
                            recv_request_idx_to_proc_pair[curr_dt_idx][dep][recv_request_idx] = {proc, send_dep};
                            recv_request_proc_pair_to_idx[curr_dt_idx][dep][{proc, send_dep}] = recv_request_idx;
                            recv_request_idx++;

                            dep_to_recv_proc_pairs[curr_dt_idx][dep].emplace_back(send_dep, proc);
                            int nrecv_from_proc = 0;
                            for (auto& lst_info : zoids_recv_from_proc) {
                                int recv_zoid_num = lst_info[0];
                                int send_zoid_num = lst_info[1];
                                int find_idx = lst_info[2];
                                nrecv_from_proc += recv_proc_zoid_sizes[curr_dt_idx][DEFAULT_PIPELINE_STAGE][recv_zoid_num][find_idx];
                            }
                            dep_to_recv_proc_pairs_sizes[curr_dt_idx][dep].push_back(nrecv_from_proc);

                            for (auto& info : zoids_recv_from_proc) {
                                int zoid_dep = curr_dt ? zoid_num_to_dep[info[0]] : zoid_num_to_dep_next_dt[info[0]];
                                deps_seen[proc].insert({send_dep, zoid_dep});
                            }
                        }
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

    template <bool curr_dt>
    void INIT_DEP_TO_SEND_ZOIDS() {
        auto& queues = curr_dt ? queues_many_cuts
                               : queues_many_cuts_next_dt;

        for (int dep = 1; dep < NUM_DEPS; dep++) {
            for (int j = 0; j < queues[dep].size(); j++) {
                auto& zoid = queues[dep][j];
                if (zoid.num % comm->nprocs == comm->me) {
                    continue;
                }

                auto& recv_neighbors = curr_dt ? recv_from_neighbors_many_cuts[zoid.num]
                        : recv_from_neighbors_many_cuts_next_dt[zoid.num];
                // find
                for (int i = 0; i < recv_neighbors.size(); i++) {
                    int recv_zoid_num = recv_neighbors[i];
                    if (recv_zoid_num % comm->nprocs != comm->me) {
                        continue;
                    }

                    int recv_zoid_dep = curr_dt ? zoid_num_to_dep[recv_zoid_num]
                            : zoid_num_to_dep_next_dt[recv_zoid_num];

                    if (recv_zoid_dep < dep - 1) {
                        auto& vec = curr_dt ? dep_to_send_zoids[dep] : dep_to_send_zoids_next_dt[dep];
                        if (std::find(vec.begin(), vec.end(), recv_zoid_num) == vec.end()) {
                            vec.push_back(recv_zoid_num);
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

                        // double new_pos[3] = {0};
                        std::array<double, 3> new_pos = {0};
                        std::array<double, 3> new_pos_borders = {0};
                        double zoid_lo[3] = {0};
                        double zoid_hi[3] = {0};
                        double zoid_borders_lo[3] = {0};
                        double zoid_borders_hi[3] = {0};

                        for (int dim = 0; dim < domain->dimension; dim++) {
                            double pos = all_pos[idx * 3 + dim];
                            // double lo = zoid.zoid.cuts[dim].lower + t * zoid.zoid.cuts[dim].slope_lower;
                            // double hi = zoid.zoid.cuts[dim].upper + t * zoid.zoid.cuts[dim].slope_upper;
                            double lo = zoid.lo[t][dim];
                            double hi = zoid.hi[t][dim];

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
                            borders_zoid = borders_zoid && pos >= lo_borders && pos <= hi_borders;
                            new_pos_borders[dim] = pos;

                            zoid_lo[dim] = lo;
                            zoid_hi[dim] = hi;
                            zoid_borders_lo[dim] = lo_borders;
                            zoid_borders_hi[dim] = hi_borders;
                        }

                        // double dist_to_zoid = distance_to_zoid(domain->prd, zoid.lo[t], zoid.hi[t], new_pos);
                        double dist_to_zoid = distance_to_zoid(domain->prd, zoid.lo[t], zoid.hi[t], new_pos_borders);
                        borders_zoid = (dist_to_zoid <= ALLEGRO_SLOPE);

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
                                if constexpr (EXPERIMENT == DPD) {
                                    zoid.v_stencil_md[1].push_back({all_vel[idx * 3 + 0], all_vel[idx * 3 + 1], all_vel[idx * 3 + 2]});
                                }
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

    bool in_zoid_helper(double* pos, double* lo, double* hi) {
        bool in_zoid = true;
        for (int dim = 0; dim < domain->dimension; dim++) {
            double lo_ = lo[dim];
            double hi_ = hi[dim];
            double p = pos[dim];
            while (p < lo_) {
                p += domain->prd[dim];
            }
            while (p >= hi_) {
                p -= domain->prd[dim];
            }
            in_zoid = in_zoid && p >= lo_ && p < hi_;
        }

        return in_zoid;
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

                    std::map<int, std::vector<int>> idx_to_zoids;

                    for (int idx = 0; idx < zoid.x_stencil_md[0].size(); idx++) {
                        double pos[3] = {zoid.x_stencil_md[0][idx].x, zoid.x_stencil_md[0][idx].y, zoid.x_stencil_md[0][idx].z};

                        for (int t2 = 1; t2 < NUM_TIMESTEPS_IN_PARALLEL + 1; t2++) {
                            for (int z = 0; z < NUM_ZOIDS_MANY_CUTS; z++) {
                                auto& curr_zoid = zoid_num_to_zoid_many_cuts[z];
                                bool in_zoid = in_zoid_helper(pos, curr_zoid.lo[t2].data(), curr_zoid.hi[t2].data());
                                if (in_zoid) {
                                    idx_to_zoids[idx].push_back(curr_zoid.num);
                                }
                            }
                        }
                    }

                    std::vector<unsigned long> permutation;

                    constexpr bool TRY_ORIGINAL = true;
                    if (TRY_ORIGINAL) {
                        int num_elems = zoid.x_stencil_md[0].size();
                        std::vector<std::vector<int>> idx_to_timesteps_local;
                        idx_to_timesteps_local.resize(num_elems);

                        std::vector<std::vector<int>> idx_to_border_zoids;
                        idx_to_border_zoids.resize(num_elems);

                        std::vector<std::vector<int>> idx_to_border_zoids_next_dt;
                        idx_to_border_zoids_next_dt.resize(num_elems);

                        for (int idx = 0; idx < num_elems; idx++) {
                            double pos[3] = {zoid.x_stencil_md[0][idx].x,
                                             zoid.x_stencil_md[0][idx].y,
                                             zoid.x_stencil_md[0][idx].z};

                            for (int t2 = 0; t2 < NUM_TIMESTEPS_IN_PARALLEL + 1; t2++) {
                                bool in_zoid = in_zoid_helper(pos, zoid.lo[t2].data(), zoid.hi[t2].data());
                                if (in_zoid) {
                                    idx_to_timesteps_local[idx].push_back(t2);

                                    auto& recv_neighbors = recv_from_neighbors_many_cuts[zoid.num];
                                    for (int r = 0; r < recv_neighbors.size(); r++) {
                                        int recv_zoid_num = recv_neighbors[r];
                                        auto& recv_zoid = zoid_num_to_zoid_many_cuts[recv_zoid_num];

                                        double neigh_zoid_borders_lo[3] = {recv_zoid.lo[t2][0] - ALLEGRO_SLOPE, recv_zoid.lo[t2][1] - ALLEGRO_SLOPE, recv_zoid.lo[t2][2] - ALLEGRO_SLOPE};
                                        double neigh_zoid_borders_hi[3] = {recv_zoid.hi[t2][0] + ALLEGRO_SLOPE, recv_zoid.hi[t2][1] + ALLEGRO_SLOPE, recv_zoid.hi[t2][2] + ALLEGRO_SLOPE};

                                        bool in_neighbor_zoid = in_zoid_helper(pos, neigh_zoid_borders_lo, neigh_zoid_borders_hi);
                                        if (in_neighbor_zoid) {
                                            idx_to_border_zoids[idx].push_back(recv_zoid_num);
                                        }
                                    }
                                }
                            }
                        }

                        for (int idx = 0; idx < num_elems; idx++) {
                            double pos[3] = {zoid.x_stencil_md[0][idx].x,
                                             zoid.x_stencil_md[0][idx].y,
                                             zoid.x_stencil_md[0][idx].z};

                            auto& zoid_next_dt = zoid_num_to_zoid_many_cuts_next_dt[zoid.num];
                            for (int t2 = 0; t2 < NUM_TIMESTEPS_IN_PARALLEL + 1; t2++) {
                                bool in_zoid = in_zoid_helper(pos, zoid_next_dt.lo[t2].data(), zoid_next_dt.hi[t2].data());
                                if (in_zoid) {
                                    auto& recv_neighbors = recv_from_neighbors_many_cuts_next_dt[zoid.num];
                                    for (int r = 0; r < recv_neighbors.size(); r++) {
                                        int recv_zoid_num = recv_neighbors[r];
                                        auto& recv_zoid = zoid_num_to_zoid_many_cuts_next_dt[recv_zoid_num];

                                        double neigh_zoid_borders_lo[3] = {recv_zoid.lo[t2][0] - ALLEGRO_SLOPE, recv_zoid.lo[t2][1] - ALLEGRO_SLOPE, recv_zoid.lo[t2][2] - ALLEGRO_SLOPE};
                                        double neigh_zoid_borders_hi[3] = {recv_zoid.hi[t2][0] + ALLEGRO_SLOPE, recv_zoid.hi[t2][1] + ALLEGRO_SLOPE, recv_zoid.hi[t2][2] + ALLEGRO_SLOPE};

                                        bool in_neighbor_zoid = in_zoid_helper(pos, neigh_zoid_borders_lo, neigh_zoid_borders_hi);
                                        if (in_neighbor_zoid) {
                                            idx_to_border_zoids_next_dt[idx].push_back(recv_zoid_num);
                                        }
                                    }
                                }
                            }
                        }

                        permutation = sort_permutation(zoid.tag_stencil_md[0],
                           [&](const tagint &tag_a, const tagint &tag_b) {
                               int idx_a = tag_to_idx[tag_a];
                               int idx_b = tag_to_idx[tag_b];

                               const auto& timesteps_local_a = idx_to_timesteps_local[idx_a];
                               const auto& timesteps_local_b = idx_to_timesteps_local[idx_b];

                               if (timesteps_local_a.size() == 0) {
                                   return false;
                               } else if (timesteps_local_b.size() == 0) {
                                   return true;
                               }

                               if (timesteps_local_a != timesteps_local_b) {
                                   return timesteps_local_a < timesteps_local_b;
                               }

                               /*
                               if (timesteps_local_a.size() != timesteps_local_b.size()) {
                                   return timesteps_local_a.size() > timesteps_local_b.size();
                               }

                               int first_timestep_local_a = timesteps_local_a[0];
                               int first_timestep_local_b = timesteps_local_b[0];

                               int last_timestep_local_a = timesteps_local_a[timesteps_local_a.size() - 1];
                               int last_timestep_local_b = timesteps_local_b[timesteps_local_b.size() - 1];

                               if (first_timestep_local_a != first_timestep_local_b) {
                                   return first_timestep_local_a < first_timestep_local_b;
                               }

                               if (last_timestep_local_a != last_timestep_local_b) {
                                   return last_timestep_local_a < last_timestep_local_b;
                               }
                               */

                               /*
                               const auto& border_zoids_a = idx_to_border_zoids[idx_a];
                               const auto& border_zoids_b = idx_to_border_zoids[idx_b];

                               if (border_zoids_a != border_zoids_b) {
                                   return border_zoids_a < border_zoids_b;
                               }

                               const auto& border_zoids_a_next_dt = idx_to_border_zoids_next_dt[idx_a];
                               const auto& border_zoids_b_next_dt = idx_to_border_zoids_next_dt[idx_b];

                               if (border_zoids_a_next_dt != border_zoids_b_next_dt) {
                                   return border_zoids_a_next_dt < border_zoids_b_next_dt;
                               }

                               const auto& zoids_a = idx_to_zoids.at(idx_a);
                               const auto& zoids_b = idx_to_zoids.at(idx_b);

                               return zoids_a < zoids_b;
                               */

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

                        /*
                        permutation = sort_permutation(zoid.tag_stencil_md[0],
                           [&](const tagint &tag_a, const tagint &tag_b) {
                                int idx_a = tag_to_idx[tag_a];
                                int idx_b = tag_to_idx[tag_b];

                                const auto& timesteps_local_a = idx_to_timesteps_local[idx_a];
                                const auto& timesteps_local_b = idx_to_timesteps_local[idx_b];

                                if (timesteps_local_a.size() == 0) {
                                   return false;
                                } else if (timesteps_local_b.size() == 0) {
                                   return true;
                                }

                                int first_timestep_local_a = timesteps_local_a[0];
                                int first_timestep_local_b = timesteps_local_b[0];

                                int last_timestep_local_a = timesteps_local_a[timesteps_local_a.size() - 1];
                                int last_timestep_local_b = timesteps_local_b[timesteps_local_b.size() - 1];

                                if (first_timestep_local_a != first_timestep_local_b) {
                                    return first_timestep_local_a < first_timestep_local_b;
                                }

                                if (last_timestep_local_a != last_timestep_local_b) {
                                    return last_timestep_local_a < last_timestep_local_b;
                                }

                                // USE LAMMPS SORTING
                                const auto &pos_a = zoid.x_stencil_md[0][idx_a];
                                int ix_a = static_cast<int> ((pos_a.x - domain->boxlo[0]) * bininvx);
                                int iy_a = static_cast<int> ((pos_a.y - domain->boxlo[1]) * bininvy);
                                int iz_a = static_cast<int> ((pos_a.z - domain->boxlo[2]) * bininvz);

                                ix_a = MAX(ix_a, 0);
                                iy_a = MAX(iy_a, 0);
                                iz_a = MAX(iz_a, 0);
                                ix_a = MIN(ix_a, nbinx - 1);
                                iy_a = MIN(iy_a, nbiny - 1);
                                iz_a = MIN(iz_a, nbinz - 1);
                                int ibin_a = iz_a * nbiny * nbinx + iy_a * nbinx + ix_a;

                                const auto &pos_b = zoid.x_stencil_md[0][idx_b];
                                int ix_b = static_cast<int> ((pos_b.x - domain->boxlo[0]) * bininvx);
                                int iy_b = static_cast<int> ((pos_b.y - domain->boxlo[1]) * bininvy);
                                int iz_b = static_cast<int> ((pos_b.z - domain->boxlo[2]) * bininvz);

                                ix_b = MAX(ix_b, 0);
                                iy_b = MAX(iy_b, 0);
                                iz_b = MAX(iz_b, 0);
                                ix_b = MIN(ix_b, nbinx - 1);
                                iy_b = MIN(iy_b, nbiny - 1);
                                iz_b = MIN(iz_b, nbinz - 1);
                                int ibin_b = iz_b * nbiny * nbinx + iy_b * nbinx + ix_b;

                                return ibin_a < ibin_b;
                           });

                        permutation = sort_permutation(zoid.tag_stencil_md[0],
                           [&](const tagint &tag_a, const tagint &tag_b) {
                               int idx_a = tag_to_idx[tag_a];
                               int idx_b = tag_to_idx[tag_b];

                               const auto& timesteps_local_a = idx_to_timesteps_local[idx_a];
                               const auto& timesteps_local_b = idx_to_timesteps_local[idx_b];

                               int num_timesteps_local_a = timesteps_local_a.size();
                               int num_timesteps_local_b = timesteps_local_b.size();

                               if (num_timesteps_local_a != num_timesteps_local_b) {
                                   return num_timesteps_local_a > num_timesteps_local_b;
                               }

                               if (num_timesteps_local_a == 0) {
                                   return false;
                               }

                               if (num_timesteps_local_b == 0) {
                                   return true;
                               }

                               int first_timestep_local_a = timesteps_local_a[0];
                               int first_timestep_local_b = timesteps_local_b[0];

                               if (first_timestep_local_a != first_timestep_local_b) {
                                   return first_timestep_local_a > first_timestep_local_b;
                               }

                               const auto& zoids_a = idx_to_zoids.at(idx_a);
                               const auto& zoids_b = idx_to_zoids.at(idx_b);

                               return zoids_a < zoids_b;
                        });
                        */
                    }

                    apply_permutation_in_place(zoid.x_stencil_md[0], permutation);
                    apply_permutation_in_place(zoid.x_stencil_md[1], permutation);

                    apply_permutation_in_place(zoid.tag_stencil_md[0], permutation);
                    apply_permutation_in_place(zoid.type_stencil_md[0], permutation);
                    apply_permutation_in_place(zoid.image_stencil_md[0], permutation);
                    apply_permutation_in_place(zoid.mask_stencil_md[0], permutation);
                    apply_permutation_in_place(zoid.v_stencil_md[0], permutation);
                    if constexpr (EXPERIMENT == DPD) {
                        apply_permutation_in_place(zoid.v_stencil_md[1], permutation);
                    }
                }
            }
        }

        cilk::opadd_reducer<int> total_num_local_segments = 0;

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
                            double lo = zoid.lo[t][dim];
                            double hi = zoid.hi[t][dim];

                            in_zoid = in_zoid && pos[dim] >= lo && pos[dim] < hi;

                            zoid_lo[dim] = lo;
                            zoid_hi[dim] = hi;
                        }

                        if (in_zoid) {
                            zoid.local_idxs_per_timestep[t].push_back(i);
                        }
                    }

                    /*
                    zoid.is_local_per_timestep[t].resize(zoid.tag_stencil_md[0].size());
                    for (int i = 0; i < zoid.tag_stencil_md[0].size(); i++) {
                        zoid.is_local_per_timestep[t][i] = false;
                    }
                    for (int local_idx : zoid.local_idxs_per_timestep[t]) {
                        zoid.is_local_per_timestep[t][local_idx] = true;
                    }
                    */

                    std::vector<int> segment_idxs;
                    std::vector<int> segment_sizes;
                    int num_segments = get_segments(zoid.local_idxs_per_timestep[t], segment_idxs, segment_sizes);
                    /*
                    std::stringstream o;
                    o << BOLDYELLOW << "CURR DT: " << 1 << "  dep: " << dep << " zoid: " << zoid.num << " time: " << t << " nlocal: " << zoid.local_idxs_per_timestep[t].size()
                        << " num segments: " << num_segments << RESET_COLOR << std::endl;
                    std::cout << o.str();
                    */

                    /*
                    for (int i = 0; i < num_segments; i++) {
                        zoid.local_idxs_per_timestep_segment_idxs[t].push_back(segment_idxs[i]);
                        zoid.local_idxs_per_timestep_segment_sizes[t].push_back(segment_sizes[i]);
                    }
                    */

                    total_num_local_segments += num_segments;
                }
            }
        }

        MPI_Allreduce(MPI_IN_PLACE, &total_num_local_segments, 1, MPI_INT, MPI_SUM, world);

        if (comm->me == 0) {
            std::stringstream o;
            o << "TOTAL NUM LOCAL SEGMENTS: " << total_num_local_segments << std::endl;
            std::cout << o.str();
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


        cilk::opadd_reducer<int> total_num_local_segments_next_dt = 0;
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
                            double lo = zoid.lo[t][dim];
                            double hi = zoid.hi[t][dim];
                            in_zoid = in_zoid && pos[dim] >= lo && pos[dim] < hi;
                        }

                        if (in_zoid) {
                            zoid.local_idxs_per_timestep[t].push_back(i);
                        }
                    }

                    /*
                    zoid.is_local_per_timestep[t].resize(zoid.tag_stencil_md[0].size());
                    for (int i = 0; i < zoid.tag_stencil_md[0].size(); i++) {
                        zoid.is_local_per_timestep[t][i] = false;
                    }
                    for (int local_idx : zoid.local_idxs_per_timestep[t]) {
                        zoid.is_local_per_timestep[t][local_idx] = true;
                    }
                    */

                    std::vector<int> segment_idxs;
                    std::vector<int> segment_sizes;
                    int num_segments = get_segments(zoid.local_idxs_per_timestep[t], segment_idxs, segment_sizes);
                    total_num_local_segments_next_dt += num_segments;

                    /*
                    for (int i = 0; i < num_segments; i++) {
                        zoid.local_idxs_per_timestep_segment_idxs[t].push_back(segment_idxs[i]);
                        zoid.local_idxs_per_timestep_segment_sizes[t].push_back(segment_sizes[i]);
                    }
                    */

                    /*
                    std::stringstream o;
                    o << BOLDYELLOW << "NEXT DT dep: " << dep << " zoid: " << zoid.num << " time: " << t
                    << " num segments: " << num_segments << RESET_COLOR << std::endl;
                    std::cout << o.str();
                    */
                }
            }
        }

        MPI_Allreduce(MPI_IN_PLACE, &total_num_local_segments_next_dt, 1, MPI_INT, MPI_SUM, world);

        if (comm->me == 0) {
            std::stringstream o;
            o << "TOTAL NUM LOCAL SEGMENTS NEXT DT: " << total_num_local_segments_next_dt << std::endl;
            std::cout << o.str();
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

        MPI_Barrier(world);
        if (comm->me == 0) {
            std::cout << "GOT LOCAL ATOMS PASSED I THINK. " << std::endl;
        }
    }

    template <bool newton>
    void CREATE_NEIGHBOR_LIST_HELPER(queue_info& zoid, std::vector<int>* neighbor_lst) {
        std::unordered_map<int, int> zoid_tag_to_idx;
        for (int k = 0; k < zoid.tag_stencil_md[0].size(); k++) {
            zoid_tag_to_idx[zoid.tag_stencil_md[0][k]] = k;
        }

        auto& x = zoid.x_stencil_md[0];
        for (int t = 0; t < NUM_TIMESTEPS_IN_PARALLEL + 1; t++) {
            zoid.neighbor_list[t].resize(zoid.x_stencil_md[0].size());

            std::unordered_set<int> local_idxs_set;
            const auto& local_idxs = zoid.local_idxs_per_timestep[t];
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
                    if (!zoid_tag_to_idx.count(neigh_tag)) {
                        std::cout << "zoid: " << zoid.num << " timestep: " << t << " tag: " << tag << " neigh tag: " << neigh_tag << std::endl;
                        assert(false);
                    }
                    int neigh_idx = zoid_tag_to_idx.at(neigh_tag);
                    bool neigh_nlocal = (local_idxs_set.find(neigh_idx) != local_idxs_set.end());
                    if (neigh_nlocal) {
                        /*
                        if (neigh_idx > idx) {
                            continue;
                        }
                        */
                        if (x[neigh_idx].z < ztmp) continue;
                        if (x[neigh_idx].z == ztmp) {
                            if (x[neigh_idx].y < ytmp) continue;
                            if (x[neigh_idx].y == ytmp && x[neigh_idx].x < xtmp) continue;
                        }
                    }

                    // add an edge if the ghost atom is ghost in a shrinking dimension
                    // if (!neigh_nlocal) {
                    if (!neigh_nlocal && newton) {
                        bool shrinking_out_of_bounds = false;
                        bool expanding_out_of_bounds = false;

                        double neigh_pos[3] = {x[neigh_idx].x, x[neigh_idx].y, x[neigh_idx].z};

                        for (int dim = 0; dim < domain->dimension; dim++) {
                            // double lo = zoid.zoid.cuts[dim].lower + t * zoid.zoid.cuts[dim].slope_lower;
                            // double hi = zoid.zoid.cuts[dim].upper + t * zoid.zoid.cuts[dim].slope_upper;
                            double lo = zoid.lo[t][dim];
                            double hi = zoid.hi[t][dim];
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
                        std::cout << "zoid: " << zoid.num << " neighbor failed check? "
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
    template <bool newton>
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

                /*
                my_neigh_pairs_src.push_back(atom->tag[neigh]);
                my_neigh_pairs_dst.push_back(atom->tag[i]);

                double xdiff = atom->x[i][0] - atom->x[neigh][0];
                double ydiff = atom->x[i][1] - atom->x[neigh][1];
                double zdiff = atom->x[i][2] - atom->x[neigh][2];
                double rsq = xdiff * xdiff + ydiff * ydiff + zdiff * zdiff;
                if (rsq >= 10) {
                    std::cout << "lammps failed check atom. " << rsq << std::endl;
                    assert(false);
                }
                */
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
                CREATE_NEIGHBOR_LIST_HELPER<newton>(zoid, neighbor_lst);
            }
        }

        cilk_for (int dep = 0; dep < NUM_DEPS; dep++) {
            cilk_for (int j = 0; j < queues_many_cuts_next_dt[dep].size(); j++) {
                auto& zoid = queues_many_cuts_next_dt[dep][j];
                if (zoid.num % comm->nprocs != comm->me) {
                    continue;
                }
                CREATE_NEIGHBOR_LIST_HELPER<newton>(zoid, neighbor_lst);
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

    template <bool newton>
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
                        /*
                        if (neigh_idx > idx) {
                            continue;
                        }
                        */
                        if (x[neigh_idx].z < ztmp) continue;
                        if (x[neigh_idx].z == ztmp) {
                            if (x[neigh_idx].y < ytmp) continue;
                            if (x[neigh_idx].y == ytmp && x[neigh_idx].x < xtmp) continue;
                        }
                    }

                    // add an edge if the ghost atom is ghost in a shrinking dimension
                    // if (!neigh_nlocal) {
                    if (!neigh_nlocal && newton) {
                        bool shrinking_out_of_bounds = false;
                        bool expanding_out_of_bounds = false;

                        double neigh_pos[3] = {x[neigh_idx].x, x[neigh_idx].y, x[neigh_idx].z};

                        for (int dim = 0; dim < domain->dimension; dim++) {
                            // double lo = zoid.zoid.cuts[dim].lower + t * zoid.zoid.cuts[dim].slope_lower;
                            // double hi = zoid.zoid.cuts[dim].upper + t * zoid.zoid.cuts[dim].slope_upper;
                            double lo = zoid.lo[t][dim];
                            double hi = zoid.hi[t][dim];
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

            for (int i = 0; i < local_idxs.size(); i++) {
                int idx = local_idxs[i];
                for (auto& p : zoid.bond_list[t][idx]) {
                    zoid.bond_list_modified[t].push_back({idx, p.first, p.second});
                }
            }

            /*
            if (NO_LOCKS_BOND) {
                auto conflict_adj_list = build_conflict_graph(zoid.bond_list_modified[t]);
                auto bond_colors = greedy_coloring(zoid.bond_list_modified[t].size(), conflict_adj_list);
                int max_color = 0;
                if (!bond_colors.empty()) {
                    auto max_it = std::max_element(bond_colors.begin(), bond_colors.end());
                    if (max_it != bond_colors.end()) {
                        max_color = *max_it;
                    }
                }
                int num_colors = (max_color >= 0) ? (max_color + 1) : 0;

                std::vector<std::vector<int>> bond_idxs_by_color(num_colors);
                if (num_colors > 0) {
                    for (int bond_idx = 0; bond_idx < zoid.bond_list_modified[t].size(); ++bond_idx) {
                        int color = bond_colors[bond_idx];
                        if (color >= 0 && color < num_colors) { // Safety check
                            bond_idxs_by_color[color].push_back(bond_idx);
                        } else {
                            std::cerr << "Warning: Bond " << bond_idx << " has invalid color " << color << std::endl;
                        }
                    }
                }

                std::map<std::tuple<int, int, int>, int> bond_to_color;
                std::vector<int> color_counts(num_colors, 0);
                for (int c = 0; c < num_colors; c++) {
                    color_counts[c] = bond_idxs_by_color[c].size();
                    for (auto& idx : bond_idxs_by_color[c]) {
                        // bond_to_color[bond_list[idx]] = c;
                        bond_to_color[zoid.bond_list_modified[t][idx]] = c;
                    }
                }

                std::sort(zoid.bond_list_modified[t].begin(), zoid.bond_list_modified[t].end(),
                          [&](const auto& bond_a, const auto& bond_b) {
                    auto color_a = bond_to_color.at(bond_a);
                    auto color_b = bond_to_color.at(bond_b);
                    if (color_a < color_b) {
                        return true;
                    } else if (color_a > color_b) {
                        return false;
                    }

                    return bond_a < bond_b;
                });

                zoid.bond_list_modified_num_colors[t] = color_counts;
            }
            */
        }
    }

    // Complete hack
    template <bool newton>
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

                CREATE_BOND_LIST_HELPER<newton>(zoid, bond_lst);
            }
        }

        cilk_for (int dep = 0; dep < NUM_DEPS; dep++) {
            cilk_for (int j = 0; j < queues_many_cuts_next_dt[dep].size(); j++) {
                auto& zoid = queues_many_cuts_next_dt[dep][j];
                if (zoid.num % comm->nprocs != comm->me) {
                    continue;
                }

                CREATE_BOND_LIST_HELPER<newton>(zoid, bond_lst);
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
    template <bool curr_dt, bool newton>
    void CONSTRUCT_SEND_FORCE_IDXS_ZOID_MANY_CUTS_HELPER(queue_info& zoid) {
        int zoid_num = zoid.num;
        auto& send_neighbors = curr_dt ? send_to_neighbors_many_cuts[zoid_num]
                : send_to_neighbors_many_cuts_next_dt[zoid_num];

        std::map<int, int> idx_to_zoid;
        std::set<int> all_force_neighbors;

        for (int t = 0; t < NUM_TIMESTEPS_IN_PARALLEL + 1; t++) {
            zoid.send_force_idxs_double_buffering[t] = new std::vector<int>[send_neighbors.size()];

            if (!newton) {
                continue;
            }

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
                std::array<double, 3> atom_pos = {pos.x, pos.y, pos.z};
                double zoid_lo[3] = {0};
                double zoid_hi[3] = {0};

                bool borders_zoid = true;
                for (int dim = 0; dim < domain->dimension; dim++) {
                    // double lo = zoid.zoid.cuts[dim].lower + t * zoid.zoid.cuts[dim].slope_lower;
                    // double hi = zoid.zoid.cuts[dim].upper + t * zoid.zoid.cuts[dim].slope_upper;
                    double lo = zoid.lo[t][dim];
                    double hi = zoid.hi[t][dim];
                    double lo_borders = lo - ALLEGRO_SLOPE;
                    double hi_borders = hi + ALLEGRO_SLOPE;
                    // TODO: Ryan, change this back to <= hi_borders if this doesn't work
                    borders_zoid = borders_zoid && atom_pos[dim] >= lo_borders && atom_pos[dim] < hi_borders;
                    // borders_zoid = borders_zoid && atom_pos[dim] >= lo_borders && atom_pos[dim] <= hi_borders;

                    zoid_lo[dim] = lo;
                    zoid_hi[dim] = hi;
                }

                double dist_to_zoid = distance_to_zoid(domain->prd, zoid.lo[t], zoid.hi[t], atom_pos);
                borders_zoid = (borders_zoid && dist_to_zoid <= ALLEGRO_SLOPE);

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
                        // double lo = send_zoid.zoid.cuts[dim].lower + t * send_zoid.zoid.cuts[dim].slope_lower;
                        // double hi = send_zoid.zoid.cuts[dim].upper + t * send_zoid.zoid.cuts[dim].slope_upper;
                        double lo = send_zoid.lo[t][dim];
                        double hi = send_zoid.hi[t][dim];
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

    template <bool curr_dt, bool newton>
    void CONSTRUCT_SEND_FORCE_IDXS_ZOID_MANY_CUTS() {
        auto& queues = curr_dt ? queues_many_cuts : queues_many_cuts_next_dt;

        cilk_for (int dep = 0; dep < NUM_DEPS; dep++) {
            cilk_for (int j = 0; j < queues[dep].size(); j++) {
                auto& zoid = queues[dep][j];
                if (zoid.num % comm->nprocs != comm->me) {
                    continue;
                }

                CONSTRUCT_SEND_FORCE_IDXS_ZOID_MANY_CUTS_HELPER<curr_dt, newton>(zoid);
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
                std::array<double, 3> atom_pos = {pos.x, pos.y, pos.z};
                double zoid_lo[3] = {0};
                double zoid_hi[3] = {0};

                bool borders_zoid = true;
                for (int dim = 0; dim < domain->dimension; dim++) {
                    // double lo = zoid.zoid.cuts[dim].lower + t * zoid.zoid.cuts[dim].slope_lower;
                    // double hi = zoid.zoid.cuts[dim].upper + t * zoid.zoid.cuts[dim].slope_upper;
                    double lo = zoid.lo[t][dim];
                    double hi = zoid.hi[t][dim];
                    double lo_borders = lo - ALLEGRO_SLOPE;
                    double hi_borders = hi + ALLEGRO_SLOPE;
                    borders_zoid = borders_zoid && atom_pos[dim] >= lo_borders && atom_pos[dim] <= hi_borders;

                    zoid_lo[dim] = lo;
                    zoid_hi[dim] = hi;
                }

                // double dist_to_zoid = distance_to_zoid(domain->prd, zoid.lo[t], zoid.hi[t], atom_pos);
                // borders_zoid = (dist_to_zoid <= ALLEGRO_SLOPE);

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
                        // double lo = send_zoid.zoid.cuts[dim].lower + t * send_zoid.zoid.cuts[dim].slope_lower;
                        // double hi = send_zoid.zoid.cuts[dim].upper + t * send_zoid.zoid.cuts[dim].slope_upper;
                        double lo = send_zoid.lo[t][dim];
                        double hi = send_zoid.hi[t][dim];
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
        /*
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
                    double lo_borders = lo - (ALLEGRO_SLOPE - epsilon);
                    double hi_borders = hi + (ALLEGRO_SLOPE - epsilon);
                    borders_zoid = borders_zoid && atom_pos[dim] >= lo_borders && atom_pos[dim] <= hi_borders;

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

                        borders_neighbor_zoid = borders_neighbor_zoid && p >= lo_borders && p <= hi_borders;
                        bool borders_dim = p >= lo_borders && p <= hi_borders;

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
                }
            }
        }
        */
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

        std::map<int, int> tag_to_timestep_odd;
        std::map<int, int> tag_to_timestep_even;

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
                std::array<double, 3> atom_pos = {pos.x, pos.y, pos.z};
                double zoid_lo[3] = {0};
                double zoid_hi[3] = {0};

                bool dim_out_of_bounds[3] = {0};
                int out_of_bounds[3] = {0};

                bool borders_zoid = true;
                for (int dim = 0; dim < domain->dimension; dim++) {
                    // double lo = zoid.zoid.cuts[dim].lower + t * zoid.zoid.cuts[dim].slope_lower;
                    // double hi = zoid.zoid.cuts[dim].upper + t * zoid.zoid.cuts[dim].slope_upper;
                    double lo = zoid.lo[t][dim];
                    double hi = zoid.hi[t][dim];
                    double lo_borders = lo - ALLEGRO_SLOPE;
                    double hi_borders = hi + ALLEGRO_SLOPE;
                    // borders_zoid = borders_zoid && atom_pos[dim] >= lo_borders && atom_pos[dim] <= hi_borders;
                    borders_zoid = borders_zoid && atom_pos[dim] >= lo_borders && atom_pos[dim] < hi_borders;

                    zoid_lo[dim] = lo;
                    zoid_hi[dim] = hi;

                    if (atom_pos[dim] < lo || atom_pos[dim] >= hi) {
                        dim_out_of_bounds[dim] = true;
                        /*
                        if (atom_pos[dim] <= lo) {
                            out_of_bounds[dim] = -1;
                        } else if (atom_pos[dim] > hi) {
                            out_of_bounds[dim] = 1;
                        } else {
                            assert(false);
                        }
                        */
                    }
                }

                double dist_to_zoid = distance_to_zoid(domain->prd, zoid.lo[t], zoid.hi[t], atom_pos);
                borders_zoid = (borders_zoid && dist_to_zoid <= ALLEGRO_SLOPE);

                // have to do this check as for later timesteps this might not be the case
                if (!borders_zoid) {
                    continue;
                }

                bool keep = false;

                for (int dim = 0; dim < domain->dimension; dim++) {
                    bool shrinking = (zoid.zoid.cuts[dim].slope_lower > 0);
                    if (dim_out_of_bounds[dim] && shrinking) {
                        keep = true;
                    }
                }

                if (!keep && !is_local) {
                    continue;
                }

                // find zoid that had it previously
                for (int j = 0; j < recv_neighbors.size(); j++) {
                    auto recv_zoid_num = recv_neighbors[j];
                    auto& recv_zoid = curr_dt ? zoid_num_to_zoid_many_cuts[recv_zoid_num] :
                                      zoid_num_to_zoid_many_cuts_next_dt[recv_zoid_num];

                    bool in_neighbor_zoid = true;

                    for (int dim = 0; dim < domain->dimension; dim++) {
                        // double lo = recv_zoid.zoid.cuts[dim].lower + (t - 1) * recv_zoid.zoid.cuts[dim].slope_lower;
                        // double hi = recv_zoid.zoid.cuts[dim].upper + (t - 1) * recv_zoid.zoid.cuts[dim].slope_upper;
                        double lo = recv_zoid.lo[t - 1][dim];
                        double hi = recv_zoid.hi[t - 1][dim];

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

                        int tag = zoid.tag_stencil_md[0][i];
                        if (t % 2 == 0) {
                            if (tag_to_timestep_even.count(tag)) {
                                if (tag_to_timestep_even.at(tag) != t) {
                                    int other_t = tag_to_timestep_even.at(tag);
                                    std::cout << std::setprecision(20)
                                              << "curr_dt: " << curr_dt << " zoid: " << zoid.num
                                              << "EVEN tag: " << tag << " timestep: " << t << " overlapping recv pos timestep: " << tag_to_timestep_even.at(tag)
                                              << " pos: " << zoid.x_stencil_md[0][i].x << " " << zoid.x_stencil_md[0][i].y << " " << zoid.x_stencil_md[0][i].z
                                              << " lo: " << zoid.lo[t][0] << " " << zoid.lo[t][1] << " " << zoid.lo[t][2]
                                              << " hi: " << zoid.hi[t][0] << " " << zoid.hi[t][1] << " " << zoid.hi[t][2]
                                              << " other lo: " << zoid.lo[other_t][0] << " " << zoid.lo[other_t][1] << " " << zoid.lo[other_t][2]
                                              << " other hi: " << zoid.hi[other_t][0] << " " << zoid.hi[other_t][1] << " " << zoid.hi[other_t][2]
                                              << std::endl;
                                }
                                assert(tag_to_timestep_even.at(tag) == t);
                            }
                            tag_to_timestep_even[tag] = t;
                        } else {
                            if (tag_to_timestep_odd.count(tag)) {
                                int other_t = tag_to_timestep_odd.at(tag);
                                if (tag_to_timestep_odd.at(tag) != t) {
                                    std::cout << std::setprecision(20)
                                              << "curr_dt: " << curr_dt << " zoid: " << zoid.num
                                              << "ODD tag: " << tag << " timestep: " << t
                                              << " overlapping recv pos timestep: " << tag_to_timestep_odd.at(tag)
                                              << " pos: " << zoid.x_stencil_md[0][i].x << " "
                                              << zoid.x_stencil_md[0][i].y << " " << zoid.x_stencil_md[0][i].z
                                              << " lo: " << zoid.lo[t][0] << " " << zoid.lo[t][1] << " "
                                              << zoid.lo[t][2]
                                              << " hi: " << zoid.hi[t][0] << " " << zoid.hi[t][1] << " "
                                              << zoid.hi[t][2]
                                              << " other lo: " << zoid.lo[other_t][0] << " " << zoid.lo[other_t][1]
                                              << " " << zoid.lo[other_t][2]
                                              << " other hi: " << zoid.hi[other_t][0] << " " << zoid.hi[other_t][1]
                                              << " " << zoid.hi[other_t][2]
                                              << std::endl;
                                }
                                assert(tag_to_timestep_odd.at(tag) == t);
                            }
                            tag_to_timestep_odd[tag] = t;
                        }
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

    template <bool curr_dt, bool newton>
    void CONSTRUCT_RECV_FORCE_IDXS_ZOID_MANY_CUTS() {
        auto& queues = curr_dt ? queues_many_cuts : queues_many_cuts_next_dt;

        if (!newton) {
            for (int dep = 0; dep < NUM_DEPS; dep++) {
                for (int j = 0; j < queues[dep].size(); j++) {
                    auto& zoid = queues[dep][j];
                    if (zoid.num % comm->nprocs != comm->me) {
                        continue;
                    }

                    auto &recv_from_neighbors = curr_dt ? recv_from_neighbors_many_cuts[zoid.num]
                                                        : recv_from_neighbors_many_cuts_next_dt[zoid.num];

                    for (int t = 0; t < NUM_TIMESTEPS_IN_PARALLEL + 1; t++) {
                        zoid.recv_force_idxs_double_buffering[t] = new std::vector<int>[recv_from_neighbors.size()];
                    }
                }
            }

            return;
        }

        std::vector<MPI_Request> r;
        r.reserve(NUM_ZOIDS_MANY_CUTS * (NUM_TIMESTEPS_IN_PARALLEL + 1) * 4 / comm->nprocs);

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

                std::map<int, int> tag_to_timestep;

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

                                if (tag_to_timestep.count(recv_buf[k])) {
                                    if (tag_to_timestep.at(recv_buf[k]) != t) {
                                        int other_t = tag_to_timestep.at(recv_buf[k]);
                                        int tag_idx = tag_to_idx.at(recv_buf[k]);

                                        std::cout << std::setprecision (std::numeric_limits<double>::digits10 + 1)
                                        << "tag: " << recv_buf[k] << " timestep: " << t << " overlapping recv force timestep: " << tag_to_timestep.at(recv_buf[k])
                                        << " pos: " << zoid.x_stencil_md[0][tag_idx].x << " " << zoid.x_stencil_md[0][tag_idx].y << " " << zoid.x_stencil_md[0][tag_idx].z
                                        << " lo: " << zoid.lo[t][0] << " " << zoid.lo[t][1] << " " << zoid.lo[t][2]
                                        << " hi: " << zoid.hi[t][0] << " " << zoid.hi[t][1] << " " << zoid.hi[t][2]
                                        << " other lo: " << zoid.lo[other_t][0] << " " << zoid.lo[other_t][1] << " " << zoid.lo[other_t][2]
                                        << " other hi: " << zoid.hi[other_t][0] << " " << zoid.hi[other_t][1] << " " << zoid.hi[other_t][2]
                                        << std::endl;
                                    }
                                    assert(tag_to_timestep.at(recv_buf[k]) == t);
                                } else {
                                    tag_to_timestep[recv_buf[k]] = t;
                                }
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

                std::map<int, int> tag_to_timestep;

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

                                if (tag_to_timestep.count(recv_buf[k])) {
                                    if (tag_to_timestep.at(recv_buf[k]) != t) {
                                        std::cout << "tag: " << recv_buf[k] << " timestep: " << t
                                                  << " overlapping recv vel timestep: " << tag_to_timestep.at(recv_buf[k])
                                                  << std::endl;
                                    }
                                    assert(tag_to_timestep.at(recv_buf[k]) == t);
                                } else {
                                    tag_to_timestep[recv_buf[k]] = t;
                                }

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
    void CONSTRUCT_NEW_RECV_FORCE_IDXS_FLATTENED() {
        auto& queues = curr_dt ? queues_many_cuts : queues_many_cuts_next_dt;
        auto& my_queues = curr_dt ? my_queues_many_cuts : my_queues_many_cuts_next_dt;

        constexpr int MAX_NEIGHBORS = 26;

        std::vector<std::vector<int>> zoid_send_data[NUM_ZOIDS_MANY_CUTS];
        for (int z = 0; z < NUM_ZOIDS_MANY_CUTS; z++) {
            zoid_send_data[z].resize(MAX_NEIGHBORS);
        }

        std::vector<MPI_Request> r;
        r.reserve(NUM_ZOIDS_MANY_CUTS * MAX_NEIGHBORS);

        for (int dep = 0; dep < NUM_DEPS; dep++) {
            for (int j = 0; j < my_queues[dep].size(); j++) {
                auto& zoid = my_queues[dep][j];
                auto& recv_neighbors = curr_dt ? recv_from_neighbors_many_cuts[zoid.num] : recv_from_neighbors_many_cuts_next_dt[zoid.num];
                for (int i = 0; i < recv_neighbors.size(); i++) {
                    int recv_zoid_num = recv_neighbors[i];
                    auto& recv_force_idxs_flattened = zoid.recv_force_idxs_double_buffering_flattened[i];
                    auto permutation = sort_permutation(recv_force_idxs_flattened, std::less<int>());
                    apply_permutation_in_place(recv_force_idxs_flattened, permutation);
                    zoid_send_data[zoid.num][i].insert(zoid_send_data[zoid.num][i].end(), permutation.begin(), permutation.end());
                    int mpi_tag = get_mpi_tag_many_cuts(recv_zoid_num, zoid.num);
                    r.emplace_back();
                    MPI_Isend(zoid_send_data[zoid.num][i].data(), permutation.size(), MPI_INT, recv_zoid_num % comm->nprocs, mpi_tag, world, &r[r.size() - 1]);
                }
            }
        }

        for (int dep = 0; dep < NUM_DEPS; dep++) {
            for (int j = 0; j < my_queues[dep].size(); j++) {
                auto& zoid = my_queues[dep][j];
                auto& send_neighbors = curr_dt ? send_to_neighbors_many_cuts[zoid.num]
                        : send_to_neighbors_many_cuts_next_dt[zoid.num];
                for (int i = 0; i < send_neighbors.size(); i++) {
                    int send_zoid_num = send_neighbors[i];
                    auto& send_force_idxs_flattened = zoid.send_force_idxs_double_buffering_flattened[i];
                    int mpi_tag = get_mpi_tag_many_cuts(zoid.num, send_zoid_num);
                    std::vector<int> recv_buf;
                    recv_buf.resize(send_force_idxs_flattened.size());
                    MPI_Recv(recv_buf.data(), recv_buf.size(), MPI_INT, send_zoid_num % comm->nprocs,
                             mpi_tag, world, MPI_STATUS_IGNORE);
                    std::vector<size_t> permutation;
                    permutation.insert(permutation.end(), recv_buf.begin(), recv_buf.end());
                    apply_permutation_in_place(send_force_idxs_flattened, permutation);
                }
            }
        }

        MPI_Waitall(r.size(), r.data(), MPI_STATUSES_IGNORE);

        for (int dep = 0; dep < NUM_DEPS; dep++) {
            for (int j = 0; j < my_queues[dep].size(); j++) {
                auto& zoid = my_queues[dep][j];
                auto& recv_neighbors = curr_dt ? recv_from_neighbors_many_cuts[zoid.num] : recv_from_neighbors_many_cuts_next_dt[zoid.num];
                for (int i = 0; i < recv_neighbors.size(); i++) {
                    int recv_zoid_num = recv_neighbors[i];
                    auto& recv_force_idxs_flattened = zoid.recv_force_idxs_double_buffering_flattened[i];

                    std::vector<int> force_segment_idxs;
                    std::vector<int> force_segment_sizes;
                    int num_force_segments = get_segments(zoid.recv_force_idxs_double_buffering_flattened[i],
                                                          force_segment_idxs, force_segment_sizes);
                    std::stringstream o;
                    o << BOLDYELLOW << "curr_dt ? " << curr_dt << " zoid: " << zoid.num << " RECV FROM: " << recv_zoid_num
                      << " NUM FORCE SEGMENTS: " << num_force_segments
                      << RESET_COLOR << std::endl;
                    std::cout << o.str();
                }
            }
        }

        /*
        */
    }

    template <bool curr_dt>
    void CONSTRUCT_NEW_RECV_VEL_IDXS_FLATTENED() {
        auto& queues = curr_dt ? queues_many_cuts : queues_many_cuts_next_dt;
        auto& my_queues = curr_dt ? my_queues_many_cuts : my_queues_many_cuts_next_dt;

        constexpr int MAX_NEIGHBORS = 26;

        std::vector<std::vector<int>> zoid_send_data[NUM_ZOIDS_MANY_CUTS];
        for (int z = 0; z < NUM_ZOIDS_MANY_CUTS; z++) {
            zoid_send_data[z].resize(MAX_NEIGHBORS);
        }

        std::vector<MPI_Request> r;
        r.reserve(NUM_ZOIDS_MANY_CUTS * MAX_NEIGHBORS);

        for (int dep = 0; dep < NUM_DEPS; dep++) {
            for (int j = 0; j < my_queues[dep].size(); j++) {
                auto& zoid = my_queues[dep][j];
                auto& recv_neighbors = curr_dt ? recv_from_neighbors_many_cuts[zoid.num] : recv_from_neighbors_many_cuts_next_dt[zoid.num];
                for (int i = 0; i < recv_neighbors.size(); i++) {
                    int recv_zoid_num = recv_neighbors[i];
                    auto& recv_vel_idxs_flattened = zoid.recv_vel_idxs_double_buffering_flattened[i];
                    auto permutation = sort_permutation(recv_vel_idxs_flattened, std::less<int>());
                    apply_permutation_in_place(recv_vel_idxs_flattened, permutation);
                    zoid_send_data[zoid.num][i].insert(zoid_send_data[zoid.num][i].end(), permutation.begin(), permutation.end());
                    int mpi_tag = get_mpi_tag_many_cuts(recv_zoid_num, zoid.num);
                    r.emplace_back();
                    MPI_Isend(zoid_send_data[zoid.num][i].data(), permutation.size(), MPI_INT, recv_zoid_num % comm->nprocs, mpi_tag, world, &r[r.size() - 1]);
                }
            }
        }

        for (int dep = 0; dep < NUM_DEPS; dep++) {
            for (int j = 0; j < my_queues[dep].size(); j++) {
                auto& zoid = my_queues[dep][j];
                auto& send_neighbors = curr_dt ? send_to_neighbors_many_cuts[zoid.num]
                                               : send_to_neighbors_many_cuts_next_dt[zoid.num];
                for (int i = 0; i < send_neighbors.size(); i++) {
                    int send_zoid_num = send_neighbors[i];
                    auto& send_vel_idxs_flattened = zoid.send_vel_idxs_double_buffering_flattened[i];
                    int mpi_tag = get_mpi_tag_many_cuts(zoid.num, send_zoid_num);
                    std::vector<int> recv_buf;
                    recv_buf.resize(send_vel_idxs_flattened.size());
                    MPI_Recv(recv_buf.data(), recv_buf.size(), MPI_INT, send_zoid_num % comm->nprocs,
                             mpi_tag, world, MPI_STATUS_IGNORE);
                    std::vector<size_t> permutation;
                    permutation.insert(permutation.end(), recv_buf.begin(), recv_buf.end());
                    apply_permutation_in_place(send_vel_idxs_flattened, permutation);
                }
            }
        }

        MPI_Waitall(r.size(), r.data(), MPI_STATUSES_IGNORE);

        for (int dep = 0; dep < NUM_DEPS; dep++) {
            for (int j = 0; j < my_queues[dep].size(); j++) {
                auto& zoid = my_queues[dep][j];
                auto& recv_neighbors = curr_dt ? recv_from_neighbors_many_cuts[zoid.num] : recv_from_neighbors_many_cuts_next_dt[zoid.num];
                for (int i = 0; i < recv_neighbors.size(); i++) {
                    int recv_zoid_num = recv_neighbors[i];
                    auto& recv_vel_idxs_flattened = zoid.recv_vel_idxs_double_buffering_flattened[i];

                    std::vector<int> vel_segment_idxs;
                    std::vector<int> vel_segment_sizes;
                    int num_vel_segments = get_segments(zoid.recv_vel_idxs_double_buffering_flattened[i],
                                                        vel_segment_idxs, vel_segment_sizes);

                    std::stringstream o2;
                    o2 << BOLDGREEN << "curr_dt ? " << curr_dt << " zoid: " << zoid.num << " RECV FROM: " << recv_zoid_num
                       << " NUM VEL SEGMENTS: " << num_vel_segments
                       << RESET_COLOR << std::endl;
                    std::cout << o2.str();
                }
            }
        }
    }

    template <bool curr_dt>
    void CONSTRUCT_NEW_RECV_POS_IDXS_FLATTENED() {
        auto& queues = curr_dt ? queues_many_cuts : queues_many_cuts_next_dt;
        auto& my_queues = curr_dt ? my_queues_many_cuts : my_queues_many_cuts_next_dt;

        constexpr int MAX_NEIGHBORS = 26;

        std::vector<std::vector<int>> zoid_send_data0[NUM_ZOIDS_MANY_CUTS];
        std::vector<std::vector<int>> zoid_send_data1[NUM_ZOIDS_MANY_CUTS];
        for (int z = 0; z < NUM_ZOIDS_MANY_CUTS; z++) {
            zoid_send_data0[z].resize(MAX_NEIGHBORS);
            zoid_send_data1[z].resize(MAX_NEIGHBORS);
        }

        std::vector<MPI_Request> r;
        r.reserve(NUM_ZOIDS_MANY_CUTS * MAX_NEIGHBORS);

        for (int dep = 0; dep < NUM_DEPS; dep++) {
            for (int j = 0; j < my_queues[dep].size(); j++) {
                auto& zoid = my_queues[dep][j];
                auto& recv_neighbors = curr_dt ? recv_from_neighbors_many_cuts[zoid.num] : recv_from_neighbors_many_cuts_next_dt[zoid.num];
                for (int i = 0; i < recv_neighbors.size(); i++) {
                    int recv_zoid_num = recv_neighbors[i];
                    auto& recv_pos_idxs_flattened0 = zoid.recv_pos_idxs_double_buffering_flattened[0][i];
                    auto permutation0 = sort_permutation(recv_pos_idxs_flattened0, std::less<int>());
                    apply_permutation_in_place(recv_pos_idxs_flattened0, permutation0);
                    zoid_send_data0[zoid.num][i].insert(zoid_send_data0[zoid.num][i].end(), permutation0.begin(), permutation0.end());
                    int mpi_tag = get_mpi_tag_many_cuts(recv_zoid_num, zoid.num);
                    r.emplace_back();
                    MPI_Isend(zoid_send_data0[zoid.num][i].data(), permutation0.size(), MPI_INT,
                              recv_zoid_num % comm->nprocs, mpi_tag, world, &r[r.size() - 1]);

                    auto& recv_pos_idxs_flattened1 = zoid.recv_pos_idxs_double_buffering_flattened[1][i];
                    auto permutation1 = sort_permutation(recv_pos_idxs_flattened1, std::less<int>());
                    apply_permutation_in_place(recv_pos_idxs_flattened1, permutation1);
                    zoid_send_data1[zoid.num][i].insert(zoid_send_data1[zoid.num][i].end(), permutation1.begin(), permutation1.end());
                    r.emplace_back();
                    MPI_Isend(zoid_send_data1[zoid.num][i].data(), permutation1.size(), MPI_INT,
                              recv_zoid_num % comm->nprocs, mpi_tag, world, &r[r.size() - 1]);
                }
            }
        }

        for (int dep = 0; dep < NUM_DEPS; dep++) {
            for (int j = 0; j < my_queues[dep].size(); j++) {
                auto& zoid = my_queues[dep][j];
                auto& send_neighbors = curr_dt ? send_to_neighbors_many_cuts[zoid.num]
                                               : send_to_neighbors_many_cuts_next_dt[zoid.num];
                for (int i = 0; i < send_neighbors.size(); i++) {
                    int send_zoid_num = send_neighbors[i];
                    auto& send_pos_idxs_flattened0 = zoid.send_pos_idxs_double_buffering_flattened[0][i];
                    int mpi_tag = get_mpi_tag_many_cuts(zoid.num, send_zoid_num);
                    std::vector<int> recv_buf0;
                    recv_buf0.resize(send_pos_idxs_flattened0.size());
                    MPI_Recv(recv_buf0.data(), recv_buf0.size(), MPI_INT, send_zoid_num % comm->nprocs,
                             mpi_tag, world, MPI_STATUS_IGNORE);
                    std::vector<size_t> permutation0;
                    permutation0.insert(permutation0.end(), recv_buf0.begin(), recv_buf0.end());
                    apply_permutation_in_place(send_pos_idxs_flattened0, permutation0);

                    auto& send_pos_idxs_flattened1 = zoid.send_pos_idxs_double_buffering_flattened[1][i];
                    std::vector<int> recv_buf1;
                    recv_buf1.resize(send_pos_idxs_flattened1.size());
                    MPI_Recv(recv_buf1.data(), recv_buf1.size(), MPI_INT, send_zoid_num % comm->nprocs,
                             mpi_tag, world, MPI_STATUS_IGNORE);
                    std::vector<size_t> permutation1;
                    permutation1.insert(permutation1.end(), recv_buf1.begin(), recv_buf1.end());
                    apply_permutation_in_place(send_pos_idxs_flattened1, permutation1);
                }
            }
        }

        MPI_Waitall(r.size(), r.data(), MPI_STATUSES_IGNORE);

        for (int dep = 0; dep < NUM_DEPS; dep++) {
            for (int j = 0; j < my_queues[dep].size(); j++) {
                auto& zoid = my_queues[dep][j];
                auto& recv_neighbors = curr_dt ? recv_from_neighbors_many_cuts[zoid.num] : recv_from_neighbors_many_cuts_next_dt[zoid.num];
                for (int i = 0; i < recv_neighbors.size(); i++) {
                    std::vector<int> pos_segment_idxs0;
                    std::vector<int> pos_segment_sizes0;
                    int num_pos_segments0 = get_segments(zoid.recv_pos_idxs_double_buffering_flattened[0][i],
                                                        pos_segment_idxs0, pos_segment_sizes0);

                    std::vector<int> pos_segment_idxs1;
                    std::vector<int> pos_segment_sizes1;
                    int num_pos_segments1 = get_segments(zoid.recv_pos_idxs_double_buffering_flattened[1][i],
                                                         pos_segment_idxs1, pos_segment_sizes1);

                    std::stringstream o;
                    o << BOLDGREEN << "curr_dt ? " << curr_dt << " zoid: " << zoid.num << " RECV FROM: " << recv_neighbors[i]
                    << " NUM POS SEGMENTS 0: " << num_pos_segments0
                    << " NUM POS SEGMENTS 1: " << num_pos_segments1
                    << RESET_COLOR << std::endl;
                    std::cout << o.str();
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

                int all_neigh_send_f = 0;
                int all_neigh_send_x = 0;
                int all_neigh_send_v = 0;

                for (int i = 0; i < send_neighbors.size(); i++) {
                    int send_zoid_num = send_neighbors[i];
                    auto& send_zoid = curr_dt ? zoid_num_to_zoid_many_cuts[send_zoid_num]
                            : zoid_num_to_zoid_many_cuts_next_dt[send_zoid_num];
                    int send_zoid_dep = (send_zoid.where[0] % 2 == 0) + (send_zoid.where[1] % 2 == 0) + (send_zoid.where[2] % 2 == 0);
                    if (send_zoid_num % comm->nprocs != comm->me) {
                        int total_send_force = zoid.send_force_idxs_double_buffering_flattened[i].size();
                        int total_send_pos = zoid.send_pos_idxs_double_buffering_flattened[0][i].size() + zoid.send_pos_idxs_double_buffering_flattened[1][i].size();
                        int total_send_vel = zoid.send_vel_idxs_double_buffering_flattened[i].size();

                        if (send_zoid_dep == my_zoid_dep + 1) {
                            all_neigh_send_f += total_send_force;
                            all_neigh_send_x += total_send_pos;
                            all_neigh_send_v += total_send_vel;

                            std::stringstream s1;
                            s1 << "zoid: " << zoid.num << " curr_dt: " << curr_dt << " dep: " << my_zoid_dep << " send: " << send_zoid_dep << " total num doubles: " << (total_send_force + total_send_pos + total_send_vel) * 3 << " force: " << total_send_force << " pos: " << total_send_pos << " vel: " << total_send_vel << std::endl;
                            std::cout << s1.str();
                        }
                    }
                }

                int total_send_ndoubles = (all_neigh_send_f + all_neigh_send_x + all_neigh_send_v) * 3;

                /*
                if (total_send_ndoubles > 30000) {
                    std::stringstream o;
                    o << "zoid: " << zoid.num << " dep: " << dep << " where: " << zoid.where[0] << " " << zoid.where[1] << " " << zoid.where[2]
                      << " total send force: " << all_neigh_send_f * 3
                      << " total send pos: " << all_neigh_send_x * 3
                      << " total send vel: " << all_neigh_send_v * 3
                      << " all in all total: " << total_send_ndoubles
                      << std::endl;
                    std::cout << o.str();
                }
                */
            }
        }

        for (int dep = 0; dep < NUM_DEPS - 1; dep++) {
            for (int j = 0; j < my_queues_many_cuts[dep].size(); j++) {
                auto& zoid = my_queues_many_cuts[dep][j];
                std::vector<int> procs;
                auto& send_neighbors = send_to_neighbors_many_cuts[zoid.num];
                for (int i = 0; i < send_neighbors.size(); i++) {
                    if (send_neighbors[i] % comm->nprocs != comm->me) {
                        auto& send_zoid = zoid_num_to_zoid_many_cuts[send_neighbors[i]];
                        int send_zoid_dep = (send_zoid.where[0] % 2 == 0) + (send_zoid.where[1] % 2 == 0) + (send_zoid.where[2] % 2 == 0);
                        if (send_zoid_dep == dep + 1) {
                            procs.push_back(send_neighbors[i] % comm->nprocs);
                        }
                    }
                }

                /*
                std::stringstream s1;
                for (auto& p : procs) {
                    s1 << p << " ";
                }

                std::stringstream o;
                o << "me: " << comm->me << " dep: " << dep << " talk to procs: " << s1.str() << std::endl;
                std::cout << o.str();
                */
            }

            MPI_Barrier(world);
        }
    }

    template <bool curr_dt>
    void GET_RECV_STATISTICS() {
        auto& queues = curr_dt ? queues_many_cuts : queues_many_cuts_next_dt;
        int total_recv_proc = 0;
        int total_recv_proc_mpi = 0;

        for (int dep = 0; dep < NUM_DEPS; dep++) {
            for (int j = 0; j < queues[dep].size(); j++) {
                auto& zoid = queues[dep][j];
                int zoid_num = zoid.num;
                if (zoid_num % comm->nprocs != comm->me) {
                    continue;
                }
                auto& recv_neighbors = curr_dt ? recv_from_neighbors_many_cuts[zoid_num] : recv_from_neighbors_many_cuts_next_dt[zoid_num];

                // int total_recv_doubles = 0;
                // int total_recv_doubles_mpi = 0;

                for (int i = 0; i < recv_neighbors.size(); i++) {
                    int recv_zoid_num = recv_neighbors[i];
                    int total_recv_force = zoid.recv_force_idxs_double_buffering_flattened[i].size();
                    int total_recv_pos = zoid.recv_pos_idxs_double_buffering_flattened[0][i].size() + zoid.recv_pos_idxs_double_buffering_flattened[1][i].size();
                    int total_recv_vel = zoid.recv_vel_idxs_double_buffering_flattened[i].size();
                    int total_recv = total_recv_force + total_recv_pos + total_recv_vel;
                    if (recv_zoid_num % comm->nprocs == comm->me) {
                        // total_recv_doubles += total_recv;
                        total_recv_proc += total_recv;
                    } else {
                        // total_recv_doubles_mpi += total_recv;
                        total_recv_proc_mpi += total_recv;
                        total_recv_proc += total_recv;
                    }
                }
            }
        }

        MPI_Allreduce(MPI_IN_PLACE, &total_recv_proc, 1, MPI_INT, MPI_SUM, world);
        MPI_Allreduce(MPI_IN_PLACE, &total_recv_proc_mpi, 1, MPI_INT, MPI_SUM, world);
        if (comm->me == 0) {
            std::cout << "total recv: " << total_recv_proc << " total recv through MPI: " << total_recv_proc_mpi << " dt: " << NUM_TIMESTEPS_IN_PARALLEL << std::endl;
        }
    }

    template <bool curr_dt>
    void CONSTRUCT_RECV_PROC_TO_PROC_OFFSETS_PIPELINED_HELPER(int pipeline_stage) {
        if (!USE_PIPELINE && pipeline_stage != DEFAULT_PIPELINE_STAGE) {
            return;
        }

        constexpr int start_timestep = USE_PIPELINE ? start_t[pipeline_stage] : default_start_t;
        constexpr int end_timestep = USE_PIPELINE ? end_t[pipeline_stage] : default_end_t;

        auto& queues = curr_dt ? queues_many_cuts : queues_many_cuts_next_dt;

        constexpr int curr_dt_idx = static_cast<int>(curr_dt);

        recv_proc_zoid_offsets[curr_dt_idx][pipeline_stage].resize(NUM_ZOIDS_MANY_CUTS);
        recv_proc_zoid_sizes[curr_dt_idx][pipeline_stage].resize(NUM_ZOIDS_MANY_CUTS);

        for (int zoid_num = 0; zoid_num < NUM_ZOIDS_MANY_CUTS; zoid_num++) {
            if (zoid_num % comm->nprocs != comm->me) {
                continue;
            }

            auto& recv_neighbors = curr_dt ? recv_from_neighbors_many_cuts[zoid_num]
                : recv_from_neighbors_many_cuts_next_dt[zoid_num];

            recv_proc_zoid_offsets[curr_dt_idx][pipeline_stage][zoid_num].resize(recv_neighbors.size());
            recv_proc_zoid_sizes[curr_dt_idx][pipeline_stage][zoid_num].resize(recv_neighbors.size());
        }

        for (int dep = 0; dep < NUM_DEPS - 1; dep++) {
            recv_zoids_from_proc[curr_dt_idx][pipeline_stage][dep].resize(comm->nprocs);
        }

        for (int send_dep = 0; send_dep < NUM_DEPS - 1; send_dep++) {
            std::vector<int> nsend_per_proc(comm->nprocs, 0);
            std::vector<int> offsets_per_proc(comm->nprocs, 0);

            for (int j = 0; j < queues[send_dep].size(); j++) {
                auto& send_zoid = queues[send_dep][j];
                int send_zoid_proc = send_zoid.num % comm->nprocs;
                if (send_zoid_proc == comm->me) {
                    continue;
                }

                auto& send_zoid_neighbors = curr_dt ? send_to_neighbors_many_cuts[send_zoid.num]
                                            : send_to_neighbors_many_cuts_next_dt[send_zoid.num];

                for (int neigh : send_zoid_neighbors) {
                    // only look at neighbors that belong to this process.
                    if (neigh % comm->nprocs != comm->me) {
                        continue;
                    }

                    int neigh_zoid_dep = curr_dt ? zoid_num_to_dep[neigh] : zoid_num_to_dep_next_dt[neigh];

                    if (neigh_zoid_dep == send_dep + 1) {
                        continue;
                    }

                    auto& recv_zoid = curr_dt ? zoid_num_to_zoid_many_cuts[neigh] 
                            : zoid_num_to_zoid_many_cuts_next_dt[neigh];

                    auto& recv_neighbors = curr_dt ? recv_from_neighbors_many_cuts[recv_zoid.num]
                                                   : recv_from_neighbors_many_cuts_next_dt[recv_zoid.num];

                    auto find_it = std::find(recv_neighbors.begin(), recv_neighbors.end(), send_zoid.num);
                    assert(find_it != recv_neighbors.end());
                    int find_idx = std::distance(recv_neighbors.begin(), find_it);

                    int nrecv_force = 0;
                    int nrecv_pos = 0;
                    int nrecv_vel = 0;
                    for (int t = start_timestep; t < end_timestep; t++) {
                        nrecv_force += recv_zoid.recv_force_idxs_double_buffering[t][find_idx].size();
                        nrecv_pos += recv_zoid.recv_pos_idxs_double_buffering[t][find_idx].size();
                        if constexpr (EXPERIMENT == DPD) {
                            nrecv_vel += recv_zoid.recv_pos_idxs_double_buffering[t][find_idx].size();
                        } else {
                            nrecv_vel += recv_zoid.recv_vel_idxs_double_buffering[t][find_idx].size();
                        }
                    }

                    int nrecv_from_send_zoid = nrecv_force + nrecv_pos + nrecv_vel;

                    recv_proc_zoid_offsets[curr_dt_idx][pipeline_stage][neigh][find_idx] = offsets_per_proc[send_zoid_proc];
                    recv_proc_zoid_sizes[curr_dt_idx][pipeline_stage][neigh][find_idx] = nrecv_from_send_zoid;

                    int zoid_dep = curr_dt ? zoid_num_to_dep[neigh] : zoid_num_to_dep_next_dt[neigh];

                    if (nrecv_from_send_zoid > 0) {
                        std::array<int, 3> arr = {recv_zoid.num, send_zoid.num, find_idx};
                        recv_zoids_from_proc[curr_dt_idx][pipeline_stage][send_dep][send_zoid_proc].emplace_back(arr);
                    }

                    offsets_per_proc[send_zoid_proc] += nrecv_from_send_zoid;
                    nsend_per_proc[send_zoid_proc] += nrecv_from_send_zoid;
                }
            }

            for (int proc = 0; proc < comm->nprocs; proc++) {
                int total_doubles_recv_from_proc = DEBUG_SEND_RECV_DATA ? nsend_per_proc[proc] * (3 + 1)
                    : nsend_per_proc[proc] * 3;
                if (total_doubles_recv_from_proc > nrecv_buf_proc_to_proc[pipeline_stage][send_dep][proc]) {
                    GROW_RECV_PROC_TO_PROC_MANY_CUTS(pipeline_stage, send_dep, proc, total_doubles_recv_from_proc);
                }
            }
        }
    }

    template <bool curr_dt>
    void CONSTRUCT_RECV_PROC_TO_PROC_OFFSETS_PIPELINED() {
       for (int p = 0; p < NUM_PIPELINE_STAGES; p++) {
           CONSTRUCT_RECV_PROC_TO_PROC_OFFSETS_PIPELINED_HELPER<curr_dt>(p);
       }
    }

    static constexpr int NUM_STREAMS = 8;
    // 64 VCIs so 1 per comm
    static constexpr int NUM_COMMS = 16;
    std::vector<MPI_Comm> all_comms;
    MPIX_Stream all_streams[NUM_STREAMS];
    MPI_Comm stream_comms[NUM_STREAMS];
    MPI_Comm stream_comm;

    MPI_Comm proc_to_proc_pipelined_comms[NUM_PIPELINE_STAGES][NUM_DEPS];

    static constexpr bool USE_STREAMS = true;

    void INIT_SEND_RECV_BUFFERS_MANY_CUTS() {
        if (USE_STREAMS) {
            for (int i = 0; i < NUM_STREAMS; i++) {
                MPIX_Stream_create(MPI_INFO_NULL, &all_streams[i]);
                MPIX_Stream_comm_create(world, all_streams[i], &stream_comms[i]);
            }
            auto res = MPIX_Stream_comm_create_multiplex(world, NUM_STREAMS, all_streams, &stream_comm);
            // assert(res == MPI_SUCCESS);
        }

        constexpr int INITIAL_SIZE = 1024;

        all_comms.resize(NUM_COMMS);
        for (int i = 0; i < NUM_COMMS; i++) {
            MPI_Comm_dup(world, &all_comms[i]);
            MPI_Info comm_info;
            MPI_Info_create(&comm_info);
            MPI_Info_set(comm_info, "mpi_assert_no_any_source", "true");
            MPI_Info_set(comm_info, "mpi_assert_no_any_tag", "true");
            MPI_Info_set(comm_info, "mpi_assert_allow_overtaking", "true");
            MPI_Info_set(comm_info, "mpi_assert_exact_length", "true");
            MPI_Comm_set_info(all_comms[i], comm_info);
            MPI_Info_free(&comm_info);
        }

        for (int p = 0; p < NUM_PIPELINE_STAGES; p++) {
            for (int dep = 0; dep < NUM_DEPS; dep++) {
                MPI_Comm_dup(world, &proc_to_proc_pipelined_comms[p][dep]);
                MPI_Info comm_info;
                MPI_Info_create(&comm_info);
                MPI_Info_set(comm_info, "mpi_assert_no_any_source", "true");
                MPI_Info_set(comm_info, "mpi_assert_no_any_tag", "true");
                MPI_Info_set(comm_info, "mpi_assert_allow_overtaking", "true");
                MPI_Info_set(comm_info, "mpi_assert_exact_length", "true");
                // MPI_Info_set(comm_info, "vci", std::to_string(i).c_str());
                MPI_Comm_set_info(proc_to_proc_pipelined_comms[p][dep], comm_info);
                MPI_Info_free(&comm_info);
            }
        }

        for (int p = 0; p < NUM_PIPELINE_STAGES; p++) {
            for (int send_dep = 0; send_dep < NUM_DEPS; send_dep++) {
                nsend_buf_proc_to_proc[p][send_dep] = new int[comm->nprocs];
                nrecv_buf_proc_to_proc[p][send_dep] = new int[comm->nprocs];

                buf_send_proc_to_proc[p][send_dep] = new double*[comm->nprocs];
                buf_recv_proc_to_proc[p][send_dep] = new double*[comm->nprocs];

                for (int proc = 0; proc < comm->nprocs; proc++) {
                    nsend_buf_proc_to_proc[p][send_dep][proc] = 0;
                    nrecv_buf_proc_to_proc[p][send_dep][proc] = 0;
                }
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

    void GROW_SEND_PROC_TO_PROC_MANY_CUTS(int pipeline_stage, int send_dep, int proc, int size) {
        constexpr double FACTOR = 1.5;
        assert(size > nsend_buf_proc_to_proc[pipeline_stage][send_dep][proc]);

        if (nsend_buf_proc_to_proc[pipeline_stage][send_dep][proc] > 0) {
            delete[] buf_send_proc_to_proc[pipeline_stage][send_dep][proc];
            nsend_buf_proc_to_proc[pipeline_stage][send_dep][proc] = 0;
        }

        int new_size = static_cast<int>(size * FACTOR);
        buf_send_proc_to_proc[pipeline_stage][send_dep][proc] = new double[new_size];
        nsend_buf_proc_to_proc[pipeline_stage][send_dep][proc] = static_cast<int>(new_size);
    }

  void GROW_RECV_PROC_TO_PROC_MANY_CUTS(int pipeline_stage, int send_dep, int proc, int size) {
      constexpr double FACTOR = 1.5;
      assert(size > nrecv_buf_proc_to_proc[pipeline_stage][send_dep][proc]);

      if (nrecv_buf_proc_to_proc[pipeline_stage][send_dep][proc] > 0) {
          delete[] buf_recv_proc_to_proc[pipeline_stage][send_dep][proc];
          nrecv_buf_proc_to_proc[pipeline_stage][send_dep][proc] = 0;
      }

      int new_size = static_cast<int>(size * FACTOR);

      buf_recv_proc_to_proc[pipeline_stage][send_dep][proc] = new double[new_size];
      nrecv_buf_proc_to_proc[pipeline_stage][send_dep][proc] = static_cast<int>(new_size);
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

    template <bool curr_dt>
    void CONSTRUCT_SEND_PROC_TO_PROC_OFFSETS_PIPELINED_HELPER(int pipeline_stage) {
        if (!USE_PIPELINE && pipeline_stage != DEFAULT_PIPELINE_STAGE) {
            return;
        }

        constexpr int curr_dt_idx = static_cast<int>(curr_dt);
        constexpr int start_timestep = USE_PIPELINE ? start_t[pipeline_stage] : default_start_t;
        constexpr int end_timestep = USE_PIPELINE ? end_t[pipeline_stage] : default_end_t;

        auto& queues = curr_dt ? my_queues_many_cuts : my_queues_many_cuts_next_dt;

        for (int zoid_num = 0; zoid_num < NUM_ZOIDS_MANY_CUTS; zoid_num++) {
            if (zoid_num % comm->nprocs != comm->me) {
                continue;
            }
        }

        for (int send_dep = 0; send_dep < NUM_DEPS - 1; send_dep++) {
            std::vector<int> offsets_per_proc(comm->nprocs, 0);
            std::vector<int> dep_nsend_to_proc(comm->nprocs, 0);

            for (int j = 0; j < queues[send_dep].size(); j++) {
                auto& send_zoid = queues[send_dep][j];
                int send_zoid_num = send_zoid.num;

                auto& send_neighbors = curr_dt ? send_to_neighbors_many_cuts[send_zoid_num]
                                                : send_to_neighbors_many_cuts_next_dt[send_zoid_num];

                std::vector<int> zoid_nsend_to_proc(comm->nprocs, 0);

                for (int i = 0; i < send_neighbors.size(); i++) {
                    int neigh = send_neighbors[i];
                    int neigh_dep = curr_dt ? zoid_num_to_dep[neigh] : zoid_num_to_dep_next_dt[neigh];
                    int neigh_proc = neigh % comm->nprocs;

                    if (neigh_proc == comm->me || neigh_dep == send_dep + 1) {
                        continue;
                    }

                    int nsend_force = 0;
                    int nsend_pos = 0;
                    int nsend_vel = 0;
                    for (int t = start_timestep; t < end_timestep; t++) {
                        nsend_force += send_zoid.send_force_idxs_double_buffering[t][i].size();
                        nsend_pos += send_zoid.send_pos_idxs_double_buffering[t][i].size();
                        if constexpr (EXPERIMENT == DPD) {
                            nsend_vel += send_zoid.send_pos_idxs_double_buffering[t][i].size();
                        } else {
                            nsend_vel += send_zoid.send_vel_idxs_double_buffering[t][i].size();
                        }
                    }

                    int nsend_total = nsend_force + nsend_pos + nsend_vel;
                    dep_nsend_to_proc[neigh_proc] += nsend_total;
                    zoid_nsend_to_proc[neigh_proc] += nsend_total;
                }

                for (int proc = 0; proc < comm->nprocs; proc++) {
                    int nsend = zoid_nsend_to_proc[proc];
                    if (proc == comm->me) {
                        assert(nsend == 0);
                    }

                    auto pair = std::make_pair(send_zoid.num, proc);
                    send_proc_zoid_offsets[curr_dt_idx][pipeline_stage][pair] = offsets_per_proc[proc];
                    send_proc_zoid_sizes[curr_dt_idx][pipeline_stage][pair] = nsend;
                    offsets_per_proc[proc] += nsend;
                }
            }

            for (int proc = 0; proc < comm->nprocs; proc++) {
                int num_sent = dep_nsend_to_proc[proc];
                if (proc == comm->me) {
                    assert(num_sent == 0);
                }

                if (num_sent > 0) {
                    send_dep_to_procs[curr_dt_idx][pipeline_stage][send_dep].push_back(proc);
                }

                int total_doubles_send_to_proc = DEBUG_SEND_RECV_DATA ? num_sent * (3 + 1) : num_sent * 3;
                if (total_doubles_send_to_proc > nsend_buf_proc_to_proc[pipeline_stage][send_dep][proc]) {
                    GROW_SEND_PROC_TO_PROC_MANY_CUTS(pipeline_stage, send_dep, proc, total_doubles_send_to_proc);
                }
            }
        }
    }

    // This is for zoid to proc for messages that aren't immediately needed by my neighbor zoids.
    template <bool curr_dt>
    void CONSTRUCT_SEND_PROC_TO_PROC_OFFSETS_PIPELINED() {
        for (int p = 0; p < NUM_PIPELINE_STAGES; p++) {
            CONSTRUCT_SEND_PROC_TO_PROC_OFFSETS_PIPELINED_HELPER<curr_dt>(p);
        }
    }

    template <bool curr_dt>
    void CONSTRUCT_SEND_ZOID_TO_ZOID_SIZES() {
        auto& queues = curr_dt ? queues_many_cuts : queues_many_cuts_next_dt;
        constexpr int curr_dt_idx = static_cast<int>(curr_dt);

        if (curr_dt) {
            send_zoid_to_zoid_sizes.resize(NUM_ZOIDS_MANY_CUTS);

            send_zoid_to_zoid_sizes_setup.resize(NUM_ZOIDS_MANY_CUTS);

            send_to_neighbors_not_my_proc_idxs.resize(NUM_ZOIDS_MANY_CUTS);
            send_to_neighbors_num_not_in_proc.resize(NUM_ZOIDS_MANY_CUTS);
        } else {
            send_zoid_to_zoid_sizes_next_dt.resize(NUM_ZOIDS_MANY_CUTS);

            send_to_neighbors_not_my_proc_idxs_next_dt.resize(NUM_ZOIDS_MANY_CUTS);
            send_to_neighbors_num_not_in_proc_next_dt.resize(NUM_ZOIDS_MANY_CUTS);
        }

        send_to_neighbors_not_my_proc_idxs_only_next_dep[curr_dt_idx].resize(NUM_ZOIDS_MANY_CUTS);
        send_to_neighbors_num_not_in_proc_only_next_dep[curr_dt_idx].resize(NUM_ZOIDS_MANY_CUTS);


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

                int send_neighbor_not_in_proc_idx = 0;

                int send_neighbor_not_in_proc_only_next_dep_idx = 0;

                for (int i = 0; i < send_neighbors.size(); i++) {
                    int neigh = send_neighbors[i];
                    int nsend_force = 0;
                    int nsend_pos = 0;
                    int nsend_vel = 0;

                    for (int t = default_start_t; t < default_end_t; t++) {
                        nsend_force += send_zoid.send_force_idxs_double_buffering[t][i].size();
                        nsend_pos += send_zoid.send_pos_idxs_double_buffering[t][i].size();
                        nsend_vel += send_zoid.send_vel_idxs_double_buffering[t][i].size();
                    }

                    for (int t = default_start_t; t < default_end_t; t++) {
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

                    std::vector<int> force_segment_idxs;
                    std::vector<int> force_segment_sizes;
                    int num_force_segments = get_segments(send_zoid.send_force_idxs_double_buffering_flattened[i],
                                                          force_segment_idxs, force_segment_sizes);
                    /*
                    std::cout << BOLDYELLOW << "zoid: " << send_zoid_num << " SEND TO: " << neigh
                    << " NUM FORCE SEGMENTS: " << num_force_segments
                    << RESET_COLOR << std::endl;
                    */

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

                    if (curr_dt) {
                        if (neigh % comm->nprocs != comm->me && total_doubles_send_to_zoid > 0) {
                            send_to_neighbors_not_my_proc_idxs[send_zoid.num].push_back(send_neighbor_not_in_proc_idx++);
                        } else {
                            send_to_neighbors_not_my_proc_idxs[send_zoid.num].push_back(-1);
                        }
                    } else {
                        if (neigh % comm->nprocs != comm->me && total_doubles_send_to_zoid > 0) {
                            send_to_neighbors_not_my_proc_idxs_next_dt[send_zoid.num].push_back(send_neighbor_not_in_proc_idx++);
                        } else {
                            send_to_neighbors_not_my_proc_idxs_next_dt[send_zoid.num].push_back(-1);
                        }
                    }

                    int neigh_dep = curr_dt ? zoid_num_to_dep[neigh] : zoid_num_to_dep_next_dt[neigh];
                    if (neigh % comm->nprocs != comm->me && neigh_dep == send_dep + 1) {
                        send_to_neighbors_not_my_proc_idxs_only_next_dep[curr_dt_idx][send_zoid.num].push_back(send_neighbor_not_in_proc_only_next_dep_idx++);
                    } else {
                        send_to_neighbors_not_my_proc_idxs_only_next_dep[curr_dt_idx][send_zoid.num].push_back(-1);
                    }
                }

                if (curr_dt) {
                    send_to_neighbors_num_not_in_proc[send_zoid.num] = send_neighbor_not_in_proc_idx;
                } else {
                    send_to_neighbors_num_not_in_proc_next_dt[send_zoid.num] = send_neighbor_not_in_proc_idx;
                }

                send_to_neighbors_num_not_in_proc_only_next_dep[curr_dt_idx][send_zoid.num] = send_neighbor_not_in_proc_only_next_dep_idx;
            }
        }
    }

    template <bool curr_dt>
    void CONSTRUCT_SEND_ZOID_TO_ZOID_SIZES_PIPELINED() {
        auto& queues = curr_dt ? queues_many_cuts : queues_many_cuts_next_dt;

        constexpr int num_p = USE_PIPELINE ? NUM_PIPELINE_STAGES : 1;

        for (int p = 0; p < num_p; p++) {
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

        for (int p = 0; p < num_p; p++) {
            constexpr int start_timestep = USE_PIPELINE ? start_t[p] : default_start_t;
            constexpr int end_timestep = USE_PIPELINE ? end_t[p] : default_end_t;

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

                        for (int t = start_timestep; t < end_timestep; t++) {
                            nsend_force += send_zoid.send_force_idxs_double_buffering[t][i].size();
                            nsend_pos += send_zoid.send_pos_idxs_double_buffering[t][i].size();
                            if constexpr (EXPERIMENT == DPD) {
                                nsend_vel += send_zoid.send_pos_idxs_double_buffering[t][i].size();
                            } else {
                                nsend_vel += send_zoid.send_vel_idxs_double_buffering[t][i].size();
                            }
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

                        for (int t = start_timestep; t < end_timestep; t++) {
                            for (auto& idx : send_zoid.send_force_idxs_double_buffering[t][i]) {
                                send_zoid.send_force_idxs_double_buffering_flattened_pipelined[p][i].push_back(idx);
                            }

                            if constexpr (EXPERIMENT == DPD) {
                                for (auto& idx : send_zoid.send_pos_idxs_double_buffering[t][i]) {
                                    send_zoid.send_vel_idxs_double_buffering_flattened_pipelined[p][t % DOUBLE_BUFFERING][i].push_back(idx);
                                }
                            } else {
                                for (auto& idx : send_zoid.send_vel_idxs_double_buffering[t][i]) {
                                    send_zoid.send_vel_idxs_double_buffering_flattened_pipelined[p][0][i].push_back(idx);
                                }
                            }

                            for (auto& idx : send_zoid.send_pos_idxs_double_buffering[t][i]) {
                                send_zoid.send_pos_idxs_double_buffering_flattened_pipelined[p][t % DOUBLE_BUFFERING][i].push_back(idx);
                            }
                        }
                    }
                }
            }
        }
    }

    template <bool curr_dt>
    void CONSTRUCT_RECV_ZOID_TO_ZOID_SIZES() {
        auto& queues = curr_dt ? queues_many_cuts : queues_many_cuts_next_dt;

        if (curr_dt) {
            recv_zoid_to_zoid_sizes.resize(NUM_ZOIDS_MANY_CUTS);

            recv_zoid_to_zoid_sizes_setup.resize(NUM_ZOIDS_MANY_CUTS);

            recv_from_neighbors_not_my_proc_idxs.resize(NUM_ZOIDS_MANY_CUTS);
        } else {
            recv_zoid_to_zoid_sizes_next_dt.resize(NUM_ZOIDS_MANY_CUTS);

            recv_from_neighbors_not_my_proc_idxs_next_dt.resize(NUM_ZOIDS_MANY_CUTS);
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

                int nrecv_force = 0;
                int nrecv_pos = 0;
                int nrecv_vel = 0;
                for (int t = default_start_t; t < default_end_t; t++) {
                    nrecv_force += zoid.recv_force_idxs_double_buffering[t][i].size();
                    nrecv_pos += zoid.recv_pos_idxs_double_buffering[t][i].size();
                    nrecv_vel += zoid.recv_vel_idxs_double_buffering[t][i].size();
                }

                int nrecv_from_zoid = nrecv_force + nrecv_pos + nrecv_vel;

                for (int t = default_start_t; t < default_end_t; t++) {
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

                if (total_doubles_recv_from_zoid > 0 && recv_zoid_num % comm->nprocs != comm->me) {
                    if (curr_dt) {
                        recv_from_neighbors_not_my_proc_idxs[zoid_num].push_back(i);
                    } else {
                        recv_from_neighbors_not_my_proc_idxs_next_dt[zoid_num].push_back(i);
                    }
                }
            }
        }
    }

    template <bool curr_dt>
    void CONSTRUCT_RECV_ZOID_TO_ZOID_SIZES_PIPELINED() {
        constexpr int num_p = USE_PIPELINE ? NUM_PIPELINE_STAGES : 1;

        for (int p = 0; p < num_p; p++) {
            constexpr int start_timestep = USE_PIPELINE ? start_t[p] : default_start_t;
            constexpr int end_timestep = USE_PIPELINE ? end_t[p] : default_end_t;

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
                    for (int t = start_timestep; t < end_timestep; t++) {
                        nrecv_from_zoid += zoid.recv_force_idxs_double_buffering[t][i].size();
                        nrecv_from_zoid += zoid.recv_pos_idxs_double_buffering[t][i].size();
                        if constexpr (EXPERIMENT == DPD) {
                            nrecv_from_zoid += zoid.recv_pos_idxs_double_buffering[t][i].size();
                        } else {
                            nrecv_from_zoid += zoid.recv_vel_idxs_double_buffering[t][i].size();
                        }
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

                    for (int t = start_timestep; t < end_timestep; t++) {
                        for (auto& idx : zoid.recv_force_idxs_double_buffering[t][i]) {
                            zoid.recv_force_idxs_double_buffering_flattened_pipelined[p][i].push_back(idx);
                        }

                        if constexpr (EXPERIMENT == DPD) {
                            for (auto& idx : zoid.recv_pos_idxs_double_buffering[t][i]) {
                                zoid.recv_vel_idxs_double_buffering_flattened_pipelined[p][t % DOUBLE_BUFFERING][i].push_back(idx);
                            }
                        } else {
                            for (auto& idx : zoid.recv_vel_idxs_double_buffering[t][i]) {
                                zoid.recv_vel_idxs_double_buffering_flattened_pipelined[p][0][i].push_back(idx);
                            }
                        }

                        for (auto& idx : zoid.recv_pos_idxs_double_buffering[t][i]) {
                            zoid.recv_pos_idxs_double_buffering_flattened_pipelined[p][t % DOUBLE_BUFFERING][i].push_back(idx);
                        }
                    }
                }
            }
        }
    }

    void PACK_AND_SEND_DATA_ZOID_TO_ZOID_SETUP(queue_info& zoid, int dep, std::vector<MPI_Request>& r) {
        auto& send_neighbors = send_to_neighbors_many_cuts[zoid.num];

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

            // if (buf_idx > 0 && (send_zoid_num % comm->nprocs != comm->me)) {
            // TODO: Setup always send data even if 0.
            if ((send_zoid_num % comm->nprocs != comm->me)) {
                int mpi_tag = get_mpi_tag_many_cuts(send_zoid_num, zoid.num);
                r.emplace_back();

                int comm_idx = ZOID_TO_ZOID_TO_VCI_IDX[1].at({zoid_num, send_zoid_num});

                MPI_Isend(buf, buf_idx, MPI_DOUBLE,
                            send_zoid_num % comm->nprocs, mpi_tag,
                            all_comms[comm_idx], &r[r.size() - 1]);
            }
        }
    }

    template <bool curr_dt>
    void PACK_AND_SEND_DATA_ZOID_TO_ZOID(queue_info& zoid, int dep, int start_t, int end_t, std::vector<MPI_Request>& r) {
        constexpr int curr_dt_idx = static_cast<int>(curr_dt);
        auto& send_neighbors = curr_dt ? send_to_neighbors_many_cuts[zoid.num]
                                       : send_to_neighbors_many_cuts_next_dt[zoid.num];

        int zoid_num = zoid.num;

        auto& send_request_idxs = curr_dt ? send_to_neighbors_not_my_proc_idxs[zoid_num]
                : send_to_neighbors_not_my_proc_idxs_next_dt[zoid_num];

        // cilk_for (int i = 0; i < send_neighbors.size(); i++) {
        for (int i = 0; i < send_neighbors.size(); i++) {
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

                assert(send_request_idx != -1);

                assert(ZOID_TO_ZOID_TO_VCI_IDX[curr_dt_idx].count({zoid_num, send_zoid_num}));
                int comm_idx = ZOID_TO_ZOID_TO_VCI_IDX[curr_dt_idx].at({zoid_num, send_zoid_num});
                // int comm_idx = curr_dt ? ZOID_TO_ZOID_TO_VCI_IDX.at({zoid_num, send_zoid_num}) : ZOID_TO_ZOID_TO_VCI_IDX_NEXT_DT.at({zoid_num, send_zoid_num});
                MPI_Isend(buf, buf_idx, MPI_DOUBLE,
                            send_zoid_num % comm->nprocs, mpi_tag,
                            all_comms[comm_idx], &r[send_request_idx]);
            }
        }
    }

    // pack all zoids, but only send to next dep
    template <bool curr_dt>
    void PACK_AND_SEND_DATA_ZOID_TO_ZOID_REVISED(queue_info& zoid, int dep, int start_t, int end_t, std::vector<MPI_Request>& r) {
        constexpr int curr_dt_idx = static_cast<int>(curr_dt);
        if (dep == NUM_DEPS - 1) {
            return;
        }
        auto& send_neighbors = curr_dt ? send_to_neighbors_many_cuts[zoid.num]
                                       : send_to_neighbors_many_cuts_next_dt[zoid.num];

        int zoid_num = zoid.num;

        auto& send_request_idxs = curr_dt ? send_to_neighbors_not_my_proc_idxs[zoid_num]
                                          : send_to_neighbors_not_my_proc_idxs_next_dt[zoid_num];

        cilk_for (int i = 0; i < send_neighbors.size(); i++) {
            int send_zoid_num = send_neighbors[i];
            int nsend = curr_dt ? send_zoid_to_zoid_sizes[zoid_num][i] : send_zoid_to_zoid_sizes_next_dt[zoid_num][i];
            int zoid_ndoubles_send = DEBUG_SEND_RECV_DATA ? nsend * (3 + 1) : nsend * 3;
            int send_request_idx = send_request_idxs[i];
            int send_zoid_dep = curr_dt ? zoid_num_to_dep[send_zoid_num] : zoid_num_to_dep_next_dt[send_zoid_num];

            if (send_zoid_num % comm->nprocs == comm->me) {
                continue;
            }

            if (zoid_ndoubles_send > nsend_buf_send_zoid_to_zoid[DEFAULT_PIPELINE_STAGE][zoid.num][i]) {
                assert(false);
                GROW_SEND_ZOID_TO_ZOID_MANY_CUTS(zoid.num, i, zoid_ndoubles_send, DEFAULT_PIPELINE_STAGE);
            }

            auto *buf = buf_send_zoid_to_zoid[DEFAULT_PIPELINE_STAGE][zoid_num][i];

            int buf_idx = PACK_DATA_MANY_CUTS_HELPER<curr_dt>(zoid, buf, i, send_zoid_num, start_t, end_t);
        }

        for (int i = 0; i < send_neighbors.size(); i++) {
            int send_zoid_num = send_neighbors[i];
            int nsend = curr_dt ? send_zoid_to_zoid_sizes[zoid_num][i] : send_zoid_to_zoid_sizes_next_dt[zoid_num][i];
            int zoid_ndoubles_send = DEBUG_SEND_RECV_DATA ? nsend * (3 + 1) : nsend * 3;
            int send_request_idx = send_request_idxs[i];
            int send_zoid_dep = curr_dt ? zoid_num_to_dep[send_zoid_num] : zoid_num_to_dep_next_dt[send_zoid_num];

            if (send_zoid_num % comm->nprocs == comm->me) {
                continue;
            }

            if (zoid_ndoubles_send > nsend_buf_send_zoid_to_zoid[DEFAULT_PIPELINE_STAGE][zoid.num][i]) {
                assert(false);
                GROW_SEND_ZOID_TO_ZOID_MANY_CUTS(zoid.num, i, zoid_ndoubles_send, DEFAULT_PIPELINE_STAGE);
            }

            auto *buf = buf_send_zoid_to_zoid[DEFAULT_PIPELINE_STAGE][zoid_num][i];

            if (send_zoid_dep == dep + 1 && zoid_ndoubles_send > 0 && (send_zoid_num % comm->nprocs != comm->me))  {
            // if (zoid_ndoubles_send > 0 && (send_zoid_num % comm->nprocs != comm->me))  {
                int mpi_tag = get_mpi_tag_many_cuts(send_zoid_num, zoid.num);

                assert(send_request_idx != -1);

                assert(ZOID_TO_ZOID_TO_VCI_IDX[curr_dt_idx].count({zoid_num, send_zoid_num}));
                int comm_idx = ZOID_TO_ZOID_TO_VCI_IDX[curr_dt_idx].at({zoid_num, send_zoid_num});
                // int comm_idx = curr_dt ? ZOID_TO_ZOID_TO_VCI_IDX.at({zoid_num, send_zoid_num}) : ZOID_TO_ZOID_TO_VCI_IDX_NEXT_DT.at({zoid_num, send_zoid_num});
                MPI_Isend(buf, zoid_ndoubles_send, MPI_DOUBLE,
                            send_zoid_num % comm->nprocs, mpi_tag,
                            all_comms[comm_idx], &r[send_request_idx]);
            }
        }
    }

    template <bool curr_dt>
    void SEND_DATA_ZOID_TO_ZOID_TO_DEP_REVISED(queue_info& zoid, int send_dep, int start_t, int end_t, std::vector<MPI_Request>& r) {
        auto& send_neighbors = curr_dt ? send_to_neighbors_many_cuts[zoid.num]
                                       : send_to_neighbors_many_cuts_next_dt[zoid.num];

        constexpr int curr_dt_idx = static_cast<int>(curr_dt);

        int zoid_num = zoid.num;

        auto& send_request_idxs = curr_dt ? send_to_neighbors_not_my_proc_idxs[zoid_num]
                                          : send_to_neighbors_not_my_proc_idxs_next_dt[zoid_num];

        // cilk_for (int i = 0; i < send_neighbors.size(); i++) {
        for (int i = 0; i < send_neighbors.size(); i++) {
            int send_zoid_num = send_neighbors[i];
            int nsend = curr_dt ? send_zoid_to_zoid_sizes[zoid_num][i] : send_zoid_to_zoid_sizes_next_dt[zoid_num][i];
            int zoid_ndoubles_send = DEBUG_SEND_RECV_DATA ? nsend * (3 + 1) : nsend * 3;
            int send_request_idx = send_request_idxs[i];
            int my_zoid_dep = curr_dt ? zoid_num_to_dep[zoid_num] : zoid_num_to_dep_next_dt[zoid_num];
            int send_zoid_dep = curr_dt ? zoid_num_to_dep[send_zoid_num] : zoid_num_to_dep_next_dt[send_zoid_num];

            if (my_zoid_dep == send_dep - 1) {
                continue;
            }

            if (send_zoid_dep != send_dep) {
                continue;
            }

            if (send_zoid_num % comm->nprocs == comm->me) {
                continue;
            }

            if (zoid_ndoubles_send > nsend_buf_send_zoid_to_zoid[DEFAULT_PIPELINE_STAGE][zoid.num][i]) {
                assert(false);
                GROW_SEND_ZOID_TO_ZOID_MANY_CUTS(zoid.num, i, zoid_ndoubles_send, DEFAULT_PIPELINE_STAGE);
            }

            auto* buf = buf_send_zoid_to_zoid[DEFAULT_PIPELINE_STAGE][zoid_num][i];

            // int buf_idx = PACK_DATA_MANY_CUTS_HELPER<curr_dt>(zoid, buf, i, send_zoid_num, start_t, end_t);
            int buf_idx = zoid_ndoubles_send;

            assert(buf_idx == zoid_ndoubles_send);

            if ((send_zoid_num % comm->nprocs != comm->me))  {
                int mpi_tag = get_mpi_tag_many_cuts(send_zoid_num, zoid.num);

                assert(send_request_idx != -1);

                assert(ZOID_TO_ZOID_TO_VCI_IDX[curr_dt_idx].count({zoid_num, send_zoid_num}));
                int comm_idx = ZOID_TO_ZOID_TO_VCI_IDX[curr_dt_idx].at({zoid_num, send_zoid_num});
                // int comm_idx = curr_dt ? ZOID_TO_ZOID_TO_VCI_IDX.at({zoid_num, send_zoid_num}) : ZOID_TO_ZOID_TO_VCI_IDX_NEXT_DT.at({zoid_num, send_zoid_num});
                MPI_Isend(buf, buf_idx, MPI_DOUBLE,
                            send_zoid_num % comm->nprocs, mpi_tag,
                            all_comms[comm_idx], &r[send_request_idx]);
            }
        }
    }

    template <bool curr_dt>
    void SEND_DATA_ZOID_TO_ZOID_TO_DEP_REVISED_PIPELINED(queue_info& zoid, int send_dep,
                                                         int start_t, int end_t, int pipeline_stage, std::vector<MPI_Request>& r) {
        constexpr int curr_dt_idx = static_cast<int>(curr_dt);
        auto& send_neighbors = curr_dt ? send_to_neighbors_many_cuts[zoid.num]
                                       : send_to_neighbors_many_cuts_next_dt[zoid.num];

        int zoid_num = zoid.num;

        auto& send_request_idxs = curr_dt ? send_to_neighbors_not_my_proc_idxs[zoid_num]
                                          : send_to_neighbors_not_my_proc_idxs_next_dt[zoid_num];

        for (int i = 0; i < send_neighbors.size(); i++) {
            int send_zoid_num = send_neighbors[i];
            int nsend = curr_dt ? send_zoid_to_zoid_sizes_pipelined[pipeline_stage][zoid_num][i]
                    : send_zoid_to_zoid_sizes_pipelined_next_dt[pipeline_stage][zoid_num][i];
            int zoid_ndoubles_send = DEBUG_SEND_RECV_DATA ? nsend * (3 + 1) : nsend * 3;
            int send_request_idx = send_request_idxs[i];
            int my_zoid_dep = curr_dt ? zoid_num_to_dep[zoid_num] : zoid_num_to_dep_next_dt[zoid_num];
            int send_zoid_dep = curr_dt ? zoid_num_to_dep[send_zoid_num] : zoid_num_to_dep_next_dt[send_zoid_num];

            if (my_zoid_dep == send_dep - 1) {
                continue;
            }

            if (send_zoid_dep != send_dep) {
                continue;
            }

            if (send_zoid_num % comm->nprocs == comm->me) {
                continue;
            }

            if (zoid_ndoubles_send > nsend_buf_send_zoid_to_zoid[pipeline_stage][zoid.num][i]) {
                assert(false);
                GROW_SEND_ZOID_TO_ZOID_MANY_CUTS(zoid.num, i, zoid_ndoubles_send, pipeline_stage);
            }

            auto* buf = buf_send_zoid_to_zoid[pipeline_stage][zoid_num][i];

            int buf_idx = zoid_ndoubles_send;

            assert(buf_idx == zoid_ndoubles_send);

            if ((send_zoid_num % comm->nprocs != comm->me))  {
                int mpi_tag = get_mpi_tag_many_cuts(send_zoid_num, zoid.num);

                assert(send_request_idx != -1);

                assert(ZOID_TO_ZOID_TO_VCI_IDX[curr_dt_idx].count({zoid_num, send_zoid_num}));
                int comm_idx = ZOID_TO_ZOID_TO_VCI_IDX[curr_dt_idx].at({zoid_num, send_zoid_num});
                // int comm_idx = curr_dt ? ZOID_TO_ZOID_TO_VCI_IDX.at({zoid_num, send_zoid_num}) : ZOID_TO_ZOID_TO_VCI_IDX_NEXT_DT.at({zoid_num, send_zoid_num});
                MPI_Isend(buf, buf_idx, MPI_DOUBLE,
                          send_zoid_num % comm->nprocs, mpi_tag,
                          all_comms[comm_idx], &r[send_request_idx]);
            }
        }
    }

    template <bool curr_dt>
    void PACK_AND_SEND_DATA_ZOID_TO_ZOID_PIPELINED(queue_info& zoid, int dep,
                                                   int start_t, int end_t, int pipeline_stage,
                                                   std::vector<MPI_Request>& r) {
        auto& send_neighbors = curr_dt ? send_to_neighbors_many_cuts[zoid.num]
                                       : send_to_neighbors_many_cuts_next_dt[zoid.num];

        auto& send_request_idxs = curr_dt ? send_to_neighbors_not_my_proc_idxs[zoid.num]
                                          : send_to_neighbors_not_my_proc_idxs_next_dt[zoid.num];
        constexpr int curr_dt_idx = static_cast<int>(curr_dt);

        int zoid_num = zoid.num;

        cilk_for (int i = 0; i < send_neighbors.size(); i++) {
            int send_zoid_num = send_neighbors[i];
            int nsend = curr_dt ? send_zoid_to_zoid_sizes_pipelined[pipeline_stage][zoid_num][i]
                    : send_zoid_to_zoid_sizes_pipelined_next_dt[pipeline_stage][zoid_num][i];
            int zoid_ndoubles_send = DEBUG_SEND_RECV_DATA ? nsend * (3 + 1) : nsend * 3;
            int send_request_idx = send_request_idxs[i];
            int send_zoid_dep = curr_dt ? zoid_num_to_dep[send_zoid_num] : zoid_num_to_dep_next_dt[send_zoid_num];

            if (send_zoid_num % comm->nprocs == comm->me) {
                continue;
            }

            if (zoid_ndoubles_send > nsend_buf_send_zoid_to_zoid[pipeline_stage][zoid.num][i]) {
                assert(false);
                GROW_SEND_ZOID_TO_ZOID_MANY_CUTS(zoid.num, i, zoid_ndoubles_send, pipeline_stage);
            }

            auto *buf = buf_send_zoid_to_zoid[pipeline_stage][zoid_num][i];

            int buf_idx = PACK_DATA_MANY_CUTS_HELPER_PIPELINED<curr_dt>(zoid, buf, i, send_zoid_num, start_t, end_t, pipeline_stage);
        }

        for (int i = 0; i < send_neighbors.size(); i++) {
            int send_zoid_num = send_neighbors[i];
            int nsend = curr_dt ? send_zoid_to_zoid_sizes_pipelined[pipeline_stage][zoid_num][i]
                                : send_zoid_to_zoid_sizes_pipelined_next_dt[pipeline_stage][zoid_num][i];
            int zoid_ndoubles_send = DEBUG_SEND_RECV_DATA ? nsend * (3 + 1) : nsend * 3;
            int send_request_idx = send_request_idxs[i];
            int send_zoid_dep = curr_dt ? zoid_num_to_dep[send_zoid_num] : zoid_num_to_dep_next_dt[send_zoid_num];

            if (send_zoid_num % comm->nprocs == comm->me) {
                continue;
            }

            if (zoid_ndoubles_send > nsend_buf_send_zoid_to_zoid[pipeline_stage][zoid.num][i]) {
                assert(false);
                GROW_SEND_ZOID_TO_ZOID_MANY_CUTS(zoid.num, i, zoid_ndoubles_send, pipeline_stage);
            }

            auto *buf = buf_send_zoid_to_zoid[pipeline_stage][zoid_num][i];

            // if (zoid_ndoubles_send > 0 && (send_zoid_num % comm->nprocs != comm->me))  {
            if (send_zoid_dep == dep + 1 && zoid_ndoubles_send > 0 && (send_zoid_num % comm->nprocs != comm->me))  {
                int mpi_tag = get_mpi_tag_many_cuts(send_zoid_num, zoid.num);

                assert(send_request_idx != -1);

                assert(ZOID_TO_ZOID_TO_VCI_IDX[curr_dt_idx].count({zoid_num, send_zoid_num}));
                int comm_idx = ZOID_TO_ZOID_TO_VCI_IDX[curr_dt_idx].at({zoid_num, send_zoid_num});
                // int comm_idx = curr_dt ? ZOID_TO_ZOID_TO_VCI_IDX.at({zoid_num, send_zoid_num}) : ZOID_TO_ZOID_TO_VCI_IDX_NEXT_DT.at({zoid_num, send_zoid_num});
                MPI_Isend(buf, zoid_ndoubles_send, MPI_DOUBLE,
                          send_zoid_num % comm->nprocs, mpi_tag,
                          all_comms[comm_idx], &r[send_request_idx]);
            }
        }

        /*
        for (int i = 0; i < send_neighbors.size(); i++) {
            int send_zoid_num = send_neighbors[i];
            int nsend = curr_dt ? send_zoid_to_zoid_sizes_pipelined[pipeline_stage][zoid_num][i]
                    : send_zoid_to_zoid_sizes_pipelined_next_dt[pipeline_stage][zoid_num][i];

            int zoid_ndoubles_send = DEBUG_SEND_RECV_DATA ? nsend * (3 + 1) : nsend * 3;
            int send_request_idx = send_request_idxs[i];

            if (zoid_ndoubles_send > nsend_buf_send_zoid_to_zoid[pipeline_stage][zoid.num][i]) {
                assert(false);
                GROW_SEND_ZOID_TO_ZOID_MANY_CUTS(zoid.num, i, zoid_ndoubles_send, pipeline_stage);
            }

            auto* buf = buf_send_zoid_to_zoid[pipeline_stage][zoid_num][i];
            int buf_idx = PACK_DATA_MANY_CUTS_HELPER_PIPELINED<curr_dt>(zoid, buf, i, send_zoid_num, start_t, end_t, pipeline_stage);

            assert(buf_idx == zoid_ndoubles_send);

            if (buf_idx > 0 && (send_zoid_num % comm->nprocs != comm->me)) {
                int mpi_tag = get_mpi_tag_many_cuts(send_zoid_num, zoid.num);

                int comm_idx = curr_dt ? ZOID_TO_ZOID_TO_VCI_IDX.at({zoid_num, send_zoid_num})
                                       : ZOID_TO_ZOID_TO_VCI_IDX_NEXT_DT.at({zoid_num, send_zoid_num});

                MPI_Isend(buf, buf_idx, MPI_DOUBLE,
                          send_zoid_num % comm->nprocs, mpi_tag,
                          all_comms[comm_idx], &r[send_request_idx]);
            }
        }
        */
    }

    template <bool curr_dt>
    void PACK_AND_SEND_DATA_PIPELINED_ONLY_NEXT_DEP(queue_info& zoid, int send_dep,
                                                    int start_t, int end_t, int pipeline_stage,
                                                    std::vector<MPI_Request>& r) {

        constexpr int curr_dt_idx = static_cast<int>(curr_dt);

        auto& send_neighbors = curr_dt ? send_to_neighbors_many_cuts[zoid.num]
            : send_to_neighbors_many_cuts_next_dt[zoid.num];

        auto& send_request_idxs = send_to_neighbors_not_my_proc_idxs_only_next_dep[curr_dt_idx][zoid.num];

        int zoid_num = zoid.num;

        const auto& procs_to_send_to = send_dep_to_procs[curr_dt_idx][pipeline_stage][send_dep];

        cilk_for (int i = 0; i < procs_to_send_to.size(); i++) {
            PACK_DATA_PROC_TO_PROC_HELPER<curr_dt>(zoid, pipeline_stage, send_dep, procs_to_send_to[i]);
        }

        cilk_for (int i = 0; i < send_neighbors.size(); i++) {
            int send_zoid_num = send_neighbors[i];
            int nsend = curr_dt ? send_zoid_to_zoid_sizes_pipelined[pipeline_stage][zoid_num][i]
                : send_zoid_to_zoid_sizes_pipelined_next_dt[pipeline_stage][zoid_num][i];
            int zoid_ndoubles_send = DEBUG_SEND_RECV_DATA ? nsend * (3 + 1) : nsend * 3;
            int send_zoid_dep = curr_dt ? zoid_num_to_dep[send_zoid_num] : zoid_num_to_dep_next_dt[send_zoid_num];

            if (send_zoid_num % comm->nprocs == comm->me) {
                continue;
            }

            if (send_zoid_dep != send_dep + 1) {
                continue;
            }

            if (zoid_ndoubles_send > nsend_buf_send_zoid_to_zoid[pipeline_stage][zoid.num][i]) {
                assert(false);
                GROW_SEND_ZOID_TO_ZOID_MANY_CUTS(zoid.num, i, zoid_ndoubles_send, pipeline_stage);
            }

            auto *buf = buf_send_zoid_to_zoid[pipeline_stage][zoid_num][i];

            int buf_idx = PACK_DATA_MANY_CUTS_HELPER_PIPELINED<curr_dt>(zoid, buf, i, send_zoid_num, start_t, end_t, pipeline_stage);
        }

        for (int i = 0; i < send_neighbors.size(); i++) {
            int send_zoid_num = send_neighbors[i];
            int nsend = curr_dt ? send_zoid_to_zoid_sizes_pipelined[pipeline_stage][zoid_num][i]
                : send_zoid_to_zoid_sizes_pipelined_next_dt[pipeline_stage][zoid_num][i];
            int zoid_ndoubles_send = DEBUG_SEND_RECV_DATA ? nsend * (3 + 1) : nsend * 3;
            int send_request_idx = send_request_idxs[i];
            int send_zoid_dep = curr_dt ? zoid_num_to_dep[send_zoid_num] : zoid_num_to_dep_next_dt[send_zoid_num];
            if (send_zoid_num % comm->nprocs == comm->me) {
                continue;
            }

            if (zoid_ndoubles_send > nsend_buf_send_zoid_to_zoid[pipeline_stage][zoid.num][i]) {
                assert(false);
                GROW_SEND_ZOID_TO_ZOID_MANY_CUTS(zoid.num, i, zoid_ndoubles_send, pipeline_stage);
            }

            auto *buf = buf_send_zoid_to_zoid[pipeline_stage][zoid_num][i];
            if (send_zoid_dep == send_dep + 1 && zoid_ndoubles_send > 0 && (send_zoid_num % comm->nprocs != comm->me))  {
                int mpi_tag = get_mpi_tag_many_cuts(send_zoid_num, zoid.num);
                assert(send_request_idx != -1);
                assert(ZOID_TO_ZOID_TO_VCI_IDX[curr_dt_idx].count({zoid_num, send_zoid_num}));
                int comm_idx = ZOID_TO_ZOID_TO_VCI_IDX[curr_dt_idx].at({zoid_num, send_zoid_num});
                if (USE_STREAMS) {
                    int src_stream_idx = zoid_to_stream_num[curr_dt_idx][zoid_num];
                    int dst_stream_idx = zoid_to_stream_num[curr_dt_idx][send_zoid_num];
                    MPIX_Stream_isend(buf, zoid_ndoubles_send, MPI_DOUBLE, send_zoid_num % comm->nprocs, mpi_tag, stream_comm,
                        src_stream_idx, dst_stream_idx, &r[send_request_idx]);
                } else {
                    MPI_Isend(buf, zoid_ndoubles_send, MPI_DOUBLE,
                            send_zoid_num % comm->nprocs, mpi_tag,
                            all_comms[comm_idx], &r[send_request_idx]);
                }
            }
        }
    }

    template <bool curr_dt>
    void SEND_DATA_PROC_TO_PROC_PIPELINED(int pipeline_stage, int send_dep,
                                          std::vector<MPI_Request>& r) {

        constexpr int curr_dt_idx = static_cast<int>(curr_dt);

        const auto& procs_to_send_to = send_dep_to_procs[curr_dt_idx][pipeline_stage][send_dep];

        auto& queue = curr_dt ? my_queues_many_cuts[send_dep] : my_queues_many_cuts_next_dt[send_dep];

        int total_num_procs = 0;

        for (int p = 0; p < procs_to_send_to.size(); p++) {
            int proc = procs_to_send_to[p];
            assert(proc != comm->me);
            int total_nsend = 0;
            for (int j = 0; j < queue.size(); j++) {
                auto& zoid = queue[j];
                auto pair = std::make_pair(zoid.num, proc);
                int size = send_proc_zoid_sizes[curr_dt_idx][pipeline_stage][pair];
                total_nsend += size;
            }

            total_nsend = DEBUG_SEND_RECV_DATA ? total_nsend * (3 + 1) : total_nsend * 3;

            auto *buf = buf_send_proc_to_proc[pipeline_stage][send_dep][proc];

            if (total_nsend > 0)  {
                int mpi_tag = get_mpi_tag_many_cuts(proc, comm->me);
                int send_request_idx = p;

                assert(send_request_idx != -1);

                // std::stringstream s1;
                // s1 << "curr_dt: " << curr_dt << " SEND. proc: " << comm->me << " to: " << proc << " pipeline stage: " << pipeline_stage << " count: " << total_nsend << " tag: " << mpi_tag << " send dep: " << send_dep << std::endl;
                // std::cout << s1.str();

                MPI_Isend(buf, total_nsend, MPI_DOUBLE,
                          proc, mpi_tag,
                          proc_to_proc_pipelined_comms[pipeline_stage][send_dep], &r[send_request_idx]);

                total_num_procs++;
            }
        }
    }

    template <bool curr_dt>
    void PACK_DATA_WITH_PROC_TO_PROC(queue_info& zoid, int send_dep, int start_timestep, int end_timestep, int pipeline_stage, std::vector<std::atomic<int>>& zoid_counters) {

        assert(pipeline_stage == DEFAULT_PIPELINE_STAGE);
        constexpr int curr_dt_idx = static_cast<int>(curr_dt);

        auto& send_neighbors = curr_dt ? send_to_neighbors_many_cuts[zoid.num]
            : send_to_neighbors_many_cuts_next_dt[zoid.num];

        auto& send_request_idxs = send_to_neighbors_not_my_proc_idxs_only_next_dep[curr_dt_idx][zoid.num];

        int zoid_num = zoid.num;

        const auto& procs_to_send_to = send_dep_to_procs[curr_dt_idx][pipeline_stage][send_dep];

        cilk_scope {
            for (int i = 0; i < procs_to_send_to.size(); i++) {
                cilk_spawn PACK_DATA_PROC_TO_PROC_HELPER<curr_dt>(zoid, pipeline_stage, send_dep, procs_to_send_to[i]);
            }

            for (int i = 0; i < send_neighbors.size(); i++) {
                int send_zoid_num = send_neighbors[i];
                int nsend = curr_dt ? send_zoid_to_zoid_sizes_pipelined[pipeline_stage][zoid_num][i]
                    : send_zoid_to_zoid_sizes_pipelined_next_dt[pipeline_stage][zoid_num][i];
                int zoid_ndoubles_send = DEBUG_SEND_RECV_DATA ? nsend * (3 + 1) : nsend * 3;
                int send_zoid_dep = curr_dt ? zoid_num_to_dep[send_zoid_num] : zoid_num_to_dep_next_dt[send_zoid_num];

                if (send_zoid_num % comm->nprocs == comm->me) {
                    cilk_spawn [&]() {
                        auto& recv_zoid = curr_dt ? zoid_num_to_zoid_many_cuts[send_zoid_num] : zoid_num_to_zoid_many_cuts_next_dt[send_zoid_num];
                        auto& recv_neighbors = curr_dt ? recv_from_neighbors_many_cuts[send_zoid_num] : recv_from_neighbors_many_cuts_next_dt[send_zoid_num];
                        auto find_it = std::find(recv_neighbors.begin(), recv_neighbors.end(), zoid_num);
                        assert(find_it != recv_neighbors.end());
                        int find_idx = std::distance(recv_neighbors.begin(), find_it);
                        assert(recv_neighbors[find_idx] == zoid_num);
                        UNPACK_DATA_MANY_CUTS_HELPER_SELF_PIPELINED<curr_dt>(recv_zoid, find_idx, zoid.num, i, start_timestep, end_timestep, pipeline_stage);
                        zoid_counters[send_zoid_num]--;
                    }();
                    continue;
                }

                if (send_zoid_dep != send_dep + 1) {
                    continue;
                }

                if (zoid_ndoubles_send > nsend_buf_send_zoid_to_zoid[pipeline_stage][zoid.num][i]) {
                    assert(false);
                    GROW_SEND_ZOID_TO_ZOID_MANY_CUTS(zoid.num, i, zoid_ndoubles_send, pipeline_stage);
                }

                auto *buf = buf_send_zoid_to_zoid[pipeline_stage][zoid_num][i];

                cilk_spawn PACK_DATA_MANY_CUTS_HELPER_PIPELINED<curr_dt>(zoid, buf, i, send_zoid_num, start_timestep, end_timestep, pipeline_stage);
            }
        }
    }

    template <bool curr_dt>
    void SEND_DATA_ZOID_TO_ZOID(queue_info& zoid, int send_dep, int pipeline_stage, std::vector<MPI_Request>& r) {
        assert(USE_STREAMS);
        constexpr int curr_dt_idx = static_cast<int>(curr_dt);

        auto& send_neighbors = curr_dt ? send_to_neighbors_many_cuts[zoid.num]
            : send_to_neighbors_many_cuts_next_dt[zoid.num];

        auto& send_request_idxs = send_to_neighbors_not_my_proc_idxs_only_next_dep[curr_dt_idx][zoid.num];

        int zoid_num = zoid.num;
        int src_stream_idx = zoid_to_stream_num[curr_dt_idx][zoid_num];

        std::vector<int> proc_counts(comm->nprocs, 0);

        for (int i = 0; i < send_neighbors.size(); i++) {
            int send_zoid_num = send_neighbors[i];
            int nsend = curr_dt ? send_zoid_to_zoid_sizes_pipelined[pipeline_stage][zoid_num][i]
                : send_zoid_to_zoid_sizes_pipelined_next_dt[pipeline_stage][zoid_num][i];
            int zoid_ndoubles_send = DEBUG_SEND_RECV_DATA ? nsend * (3 + 1) : nsend * 3;
            int send_request_idx = send_request_idxs[i];
            int send_zoid_dep = curr_dt ? zoid_num_to_dep[send_zoid_num] : zoid_num_to_dep_next_dt[send_zoid_num];
            if (send_zoid_num % comm->nprocs == comm->me) {
                continue;
            }

            if (zoid_ndoubles_send > nsend_buf_send_zoid_to_zoid[pipeline_stage][zoid.num][i]) {
                assert(false);
                GROW_SEND_ZOID_TO_ZOID_MANY_CUTS(zoid.num, i, zoid_ndoubles_send, pipeline_stage);
            }

            auto *buf = buf_send_zoid_to_zoid[pipeline_stage][zoid_num][i];
            if (send_zoid_dep == send_dep + 1 && zoid_ndoubles_send > 0 && (send_zoid_num % comm->nprocs != comm->me))  {
                proc_counts[send_zoid_num % comm->nprocs]++;
                int mpi_tag = get_mpi_tag_many_cuts(send_zoid_num, zoid.num);
                assert(send_request_idx != -1);
                int dst_stream_idx = zoid_to_stream_num[curr_dt_idx][send_zoid_num];
                MPIX_Stream_isend(buf, zoid_ndoubles_send, MPI_DOUBLE, send_zoid_num % comm->nprocs, mpi_tag, stream_comm,
                    src_stream_idx, dst_stream_idx, &r[send_request_idx]);
            }
        }
    }

    void RECEIVE_DATA_ZOID_TO_ZOID_SETUP(int zoid_num, std::vector<MPI_Request>& r) {
        auto& queues = queues_many_cuts;

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

            // if (total_doubles_recv_from_zoid > 0) {
            // Setup just take anything, it won't matter too much.
            if (true) {
                r.emplace_back();
                int mpi_tag = get_mpi_tag_many_cuts(zoid_num, recv_zoid_num);

                assert(ZOID_TO_ZOID_TO_VCI_IDX[1].count({recv_zoid_num, zoid_num}));
                int comm_idx = ZOID_TO_ZOID_TO_VCI_IDX[1].at({recv_zoid_num, zoid_num});
                MPI_Irecv(buf, total_doubles_recv_from_zoid, MPI_DOUBLE,
                            recv_zoid_num % comm->nprocs, mpi_tag,
                            all_comms[comm_idx], &r[r.size() - 1]);
            }
        }
    }

    template <bool curr_dt>
    void RECEIVE_DATA_ZOID_TO_ZOID_WAITANY(int dep, int zoid_num, std::vector<MPI_Request>& r) {
        constexpr int curr_dt_idx = static_cast<int>(curr_dt);
        auto& queues = curr_dt ? queues_many_cuts : queues_many_cuts_next_dt;

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
                if (curr_dt) {
                    assert(recv_request_zoid_to_idx[dep].count({recv_zoid_num, zoid_num}));
                } else {
                    assert(recv_request_zoid_to_idx_next_dt[dep].count({recv_zoid_num, zoid_num}));
                }

                int recv_request_idx = curr_dt ? recv_request_zoid_to_idx[dep].at({recv_zoid_num, zoid_num})
                                               : recv_request_zoid_to_idx_next_dt[dep].at({recv_zoid_num, zoid_num});
                assert(ZOID_TO_ZOID_TO_VCI_IDX[curr_dt_idx].count({recv_zoid_num, zoid_num}));
                int comm_idx = ZOID_TO_ZOID_TO_VCI_IDX[curr_dt_idx].at({recv_zoid_num, zoid_num});
                // int comm_idx = curr_dt ? ZOID_TO_ZOID_TO_VCI_IDX.at({recv_zoid_num, zoid_num}) : ZOID_TO_ZOID_TO_VCI_IDX_NEXT_DT.at({recv_zoid_num, zoid_num});
                MPI_Irecv(buf, total_doubles_recv_from_zoid, MPI_DOUBLE,
                            recv_zoid_num % comm->nprocs, mpi_tag,
                            all_comms[comm_idx], &r[recv_request_idx]);
            }
        }
    }

    template <bool curr_dt>
    void RECEIVE_DATA_ZOID_TO_ZOID_WAITANY_PIPELINED(int dep, int zoid_num, int pipeline_stage, std::vector<MPI_Request>& r) {
        auto& queues = curr_dt ? queues_many_cuts : queues_many_cuts_next_dt;

        auto& recv_neighbors = curr_dt ? recv_from_neighbors_many_cuts[zoid_num]
                                       : recv_from_neighbors_many_cuts_next_dt[zoid_num];

        constexpr int curr_dt_idx = static_cast<int>(curr_dt);

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
                int mpi_tag = get_mpi_tag_many_cuts(zoid_num, recv_zoid_num);
                int recv_request_idx = curr_dt ? recv_request_zoid_to_idx[dep].at({recv_zoid_num, zoid_num})
                                               : recv_request_zoid_to_idx_next_dt[dep].at({recv_zoid_num, zoid_num});
                assert(ZOID_TO_ZOID_TO_VCI_IDX[curr_dt_idx].count({recv_zoid_num, zoid_num}));
                int comm_idx = ZOID_TO_ZOID_TO_VCI_IDX[curr_dt_idx].at({recv_zoid_num, zoid_num});
                // int comm_idx = curr_dt ? ZOID_TO_ZOID_TO_VCI_IDX.at({recv_zoid_num, zoid_num}) : ZOID_TO_ZOID_TO_VCI_IDX_NEXT_DT.at({recv_zoid_num, zoid_num});
                MPI_Irecv(buf, total_doubles_recv_from_zoid, MPI_DOUBLE,
                            recv_zoid_num % comm->nprocs, mpi_tag,
                            all_comms[comm_idx], &r[recv_request_idx]);
            }
        }
    }

    template <bool curr_dt>
    void RECEIVE_DATA_ZOID_TO_ZOID_WAITANY_PIPELINED_ONLY_NEXT_DEP(int dep, int zoid_num, int pipeline_stage, std::vector<MPI_Request>& r) {
        auto& queues = curr_dt ? queues_many_cuts : queues_many_cuts_next_dt;
        auto& recv_neighbors = curr_dt ? recv_from_neighbors_many_cuts[zoid_num] : recv_from_neighbors_many_cuts_next_dt[zoid_num];
        constexpr int curr_dt_idx = static_cast<int>(curr_dt);

        for (int i = 0; i < recv_neighbors.size(); i++) {
            int recv_zoid_num = recv_neighbors[i];
            if (recv_zoid_num % comm->nprocs == comm->me) {
                continue;
            }

            int recv_zoid_dep = curr_dt ? zoid_num_to_dep[recv_zoid_num] : zoid_num_to_dep_next_dt[recv_zoid_num];

            if (recv_zoid_dep != dep - 1) {
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
                int mpi_tag = get_mpi_tag_many_cuts(zoid_num, recv_zoid_num);
                assert(recv_request_zoid_to_idx_with_proc_to_proc[curr_dt_idx][pipeline_stage][dep].count({recv_zoid_num, zoid_num}));
                int recv_request_idx = recv_request_zoid_to_idx_with_proc_to_proc[curr_dt_idx][pipeline_stage][dep].at({recv_zoid_num, zoid_num});
                assert(ZOID_TO_ZOID_TO_VCI_IDX[curr_dt_idx].count({recv_zoid_num, zoid_num}));
                int comm_idx = ZOID_TO_ZOID_TO_VCI_IDX[curr_dt_idx].at({recv_zoid_num, zoid_num});
                // int comm_idx = curr_dt ? ZOID_TO_ZOID_TO_VCI_IDX.at({recv_zoid_num, zoid_num}) : ZOID_TO_ZOID_TO_VCI_IDX_NEXT_DT.at({recv_zoid_num, zoid_num});
                if (USE_STREAMS) {
                    int src_stream_idx = zoid_to_stream_num[curr_dt_idx][recv_zoid_num];
                    int dst_stream_idx = zoid_to_stream_num[curr_dt_idx][zoid_num];
                    /*
                    MPIX_Stream_irecv(buf, total_doubles_recv_from_zoid, MPI_DOUBLE, recv_zoid_num % comm->nprocs,
                    mpi_tag, stream_comm, src_stream_idx, dst_stream_idx, &r[recv_request_idx]);
                    */
                    MPIX_Stream_recv(buf, total_doubles_recv_from_zoid, MPI_DOUBLE, recv_zoid_num % comm->nprocs, mpi_tag,
                        stream_comm, src_stream_idx, dst_stream_idx, MPI_STATUS_IGNORE);
                } else {
                    MPI_Irecv(buf, total_doubles_recv_from_zoid, MPI_DOUBLE,
                            recv_zoid_num % comm->nprocs, mpi_tag,
                            all_comms[comm_idx], &r[recv_request_idx]);
                }
            }
        }
    }

    template <bool curr_dt>
    void RECEIVE_DATA_PROC_TO_PROC_PIPELINED(int pipeline_stage, int dep, std::vector<MPI_Request>& r) {
        auto& queue = curr_dt ? my_queues_many_cuts[dep] : my_queues_many_cuts_next_dt[dep];

        constexpr int curr_dt_idx = static_cast<int>(curr_dt);
        int nrecv_zoid_to_zoid = stencilMD->nrecv_zoid_to_zoid[curr_dt_idx][pipeline_stage][dep];

        int request_arr_idx = nrecv_zoid_to_zoid;

        auto& lst_recv = dep_to_recv_proc_to_proc[curr_dt_idx][pipeline_stage][dep];
        auto& lst_recv_sizes = dep_to_recv_proc_to_proc_sizes[curr_dt_idx][pipeline_stage][dep];

        int total_recv_procs = 0;

        for (int i = 0; i < lst_recv.size(); i++) {
            auto& [send_dep, proc] = lst_recv[i];
            int size = lst_recv_sizes[i];
            int nrecv_from_proc = DEBUG_SEND_RECV_DATA ? size * (3 + 1) : size * 3;
            if (proc == comm->me) {
                assert(nrecv_from_proc == 0);
            }
            assert(nrecv_from_proc > 0);
            int mpi_tag = get_mpi_tag_many_cuts(comm->me, proc);
            int recv_request_idx = request_arr_idx;

            total_recv_procs++;

            MPI_Irecv(buf_recv_proc_to_proc[pipeline_stage][send_dep][proc], nrecv_from_proc, MPI_DOUBLE,
                        proc, mpi_tag,
                        proc_to_proc_pipelined_comms[pipeline_stage][send_dep], &r[request_arr_idx++]);
        }
    }

    template <bool curr_dt>
    int RECEIVE_DATA_ZOID_TO_ZOID(int dep, queue_info& zoid, int pipeline_stage, std::vector<MPI_Request>& r) {
        assert(USE_STREAMS);
        int zoid_num = zoid.num;
        auto& queues = curr_dt ? queues_many_cuts : queues_many_cuts_next_dt;
        auto& recv_neighbors = curr_dt ? recv_from_neighbors_many_cuts[zoid_num] : recv_from_neighbors_many_cuts_next_dt[zoid_num];
        constexpr int curr_dt_idx = static_cast<int>(curr_dt);

        int dst_stream_idx = zoid_to_stream_num[curr_dt_idx][zoid_num];

        std::vector<int> proc_counts(comm->nprocs, 0);

        int num_recv_neighbors = 0;

        for (int i = 0; i < recv_neighbors.size(); i++) {
            int recv_zoid_num = recv_neighbors[i];
            if (recv_zoid_num % comm->nprocs == comm->me) {
                continue;
            }

            int recv_zoid_dep = curr_dt ? zoid_num_to_dep[recv_zoid_num] : zoid_num_to_dep_next_dt[recv_zoid_num];

            if (recv_zoid_dep != dep - 1) {
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
                proc_counts[recv_zoid_num % comm->nprocs]++;
                int mpi_tag = get_mpi_tag_many_cuts(zoid_num, recv_zoid_num);
                assert(recv_request_zoid_to_idx_with_proc_to_proc[curr_dt_idx][pipeline_stage][dep].count({recv_zoid_num, zoid_num}));
                int recv_request_idx = recv_request_zoid_to_idx_with_proc_to_proc[curr_dt_idx][pipeline_stage][dep].at({recv_zoid_num, zoid_num});
                int src_stream_idx = zoid_to_stream_num[curr_dt_idx][recv_zoid_num];
                MPIX_Stream_recv(buf, total_doubles_recv_from_zoid, MPI_DOUBLE, recv_zoid_num % comm->nprocs, mpi_tag,
                    stream_comm, src_stream_idx, dst_stream_idx, MPI_STATUS_IGNORE);
                UNPACK_POS_VEL_MANY_CUTS_ZOID_PIPELINED<curr_dt>(zoid, recv_zoid_num, default_start_t, default_end_t, DEFAULT_PIPELINE_STAGE);
                num_recv_neighbors++;
            }
        }

        return num_recv_neighbors;
    }

    std::atomic<bool> done; 

    void MPIX_START_PROGRESS_THREAD() {
        while (true) {
            if (done) {
                break;
            }
            for (int i = 0; i < NUM_STREAMS; i++) {
                MPIX_Stream_progress(all_streams[i]);
            }
            #ifdef __SSE__
                            __builtin_ia32_pause();
            #endif
            #ifdef __aarch64__
                            __builtin_arm_yield();
            #endif
        }
    }

    void MPIX_STOP_PROGRESS_THREAD() {
        done = true;
    }

    template <bool curr_dt>
    void RECEIVE_DATA_PROC_TO_PROC_AND_ZOID_TO_ZOID(int dep, std::vector<MPI_Request>& recv_r, std::vector<MPI_Request>& recv_r_proc_to_proc) {
        auto& my_queues = curr_dt ? my_queues_many_cuts : my_queues_many_cuts_next_dt;
        constexpr int curr_dt_idx = static_cast<int>(curr_dt);

        // recv data zoid to zoid
        for (int j = 0; j < my_queues[dep].size(); j++) {
            auto& zoid = my_queues[dep][j];
            auto& recv_neighbors = curr_dt ? recv_from_neighbors_many_cuts[zoid.num] 
                : recv_from_neighbors_many_cuts_next_dt[zoid.num];

            for (int i = 0; i < recv_neighbors.size(); i++) {
                int recv_zoid_num = recv_neighbors[i];
                if (recv_zoid_num % comm->nprocs == comm->me) {
                    continue;
                }

                int recv_zoid_dep = curr_dt ? zoid_num_to_dep[recv_zoid_num] : zoid_num_to_dep_next_dt[recv_zoid_num];

                if (recv_zoid_dep != dep - 1) {
                    continue;
                }

                int recv_size = curr_dt ? recv_zoid_to_zoid_sizes_pipelined[DEFAULT_PIPELINE_STAGE][zoid.num][i]
                        : recv_zoid_to_zoid_sizes_pipelined_next_dt[DEFAULT_PIPELINE_STAGE][zoid.num][i];

                int other_recv_size = curr_dt ? recv_zoid_to_zoid_sizes[zoid.num][i] : recv_zoid_to_zoid_sizes_next_dt[zoid.num][i];
                assert(recv_size == other_recv_size);

                auto* buf = buf_recv_zoid_to_zoid[DEFAULT_PIPELINE_STAGE][zoid.num][i];

                int total_doubles_recv_from_zoid = DEBUG_SEND_RECV_DATA ? recv_size * (3 + 1) : recv_size * 3;
                if (total_doubles_recv_from_zoid > nrecv_buf_recv_zoid_to_zoid[DEFAULT_PIPELINE_STAGE][zoid.num][i]) {
                    assert(false);
                    GROW_RECV_ZOID_TO_ZOID_MANY_CUTS(zoid.num, i, total_doubles_recv_from_zoid, DEFAULT_PIPELINE_STAGE);
                }

                if (total_doubles_recv_from_zoid > 0) {
                    int mpi_tag = get_mpi_tag_many_cuts(zoid.num, recv_zoid_num);
                    assert(recv_request_zoid_pair_to_idx[curr_dt_idx][dep].count({recv_zoid_num, zoid.num}));
                    int recv_request_idx = recv_request_zoid_pair_to_idx[curr_dt_idx][dep].at({recv_zoid_num, zoid.num});
                    assert(ZOID_TO_ZOID_TO_VCI_IDX[curr_dt_idx].count({recv_zoid_num, zoid.num}));
                    int comm_idx = ZOID_TO_ZOID_TO_VCI_IDX[curr_dt_idx].at({recv_zoid_num, zoid.num});
                    // int comm_idx = curr_dt ? ZOID_TO_ZOID_TO_VCI_IDX.at({recv_zoid_num, zoid.num})  : ZOID_TO_ZOID_TO_VCI_IDX_NEXT_DT.at({recv_zoid_num, zoid.num});
                    MPI_Irecv(buf, total_doubles_recv_from_zoid, MPI_DOUBLE,
                          recv_zoid_num % comm->nprocs, mpi_tag,
                          all_comms[comm_idx], &recv_r[recv_request_idx]);
                }
            }
        }

        auto& lst_recv = dep_to_recv_proc_pairs[curr_dt_idx][dep];
        auto& lst_recv_sizes = dep_to_recv_proc_pairs_sizes[curr_dt_idx][dep];

        for (int i = 0; i < lst_recv.size(); i++) {
            auto& [send_dep, proc] = lst_recv[i];
            int size = lst_recv_sizes[i];
            int nrecv_from_proc = DEBUG_SEND_RECV_DATA ? size * (3 + 1) : size * 3;
            assert(proc != comm->me && nrecv_from_proc > 0);
            int mpi_tag = get_mpi_tag_many_cuts(comm->me, proc);
            assert(recv_request_proc_pair_to_idx[curr_dt_idx][dep].count({proc, send_dep}));

            int recv_proc_to_proc_idx = recv_request_proc_pair_to_idx[curr_dt_idx][dep][{proc, send_dep}];
            assert(recv_proc_to_proc_idx == i);
            MPI_Irecv(buf_recv_proc_to_proc[DEFAULT_PIPELINE_STAGE][send_dep][proc], nrecv_from_proc, MPI_DOUBLE,
                        proc, mpi_tag,
                        proc_to_proc_pipelined_comms[DEFAULT_PIPELINE_STAGE][send_dep], 
                        &recv_r_proc_to_proc[recv_proc_to_proc_idx]);
        }
    }

    template <bool curr_dt>
    void RECEIVE_DATA_PROC_TO_PROC(int dep, std::vector<MPI_Request>& recv_r_proc_to_proc) {
        auto& my_queues = curr_dt ? my_queues_many_cuts : my_queues_many_cuts_next_dt;
        constexpr int curr_dt_idx = static_cast<int>(curr_dt);

        auto& lst_recv = dep_to_recv_proc_pairs[curr_dt_idx][dep];
        auto& lst_recv_sizes = dep_to_recv_proc_pairs_sizes[curr_dt_idx][dep];

        for (int i = 0; i < lst_recv.size(); i++) {
            auto& [send_dep, proc] = lst_recv[i];
            int size = lst_recv_sizes[i];
            int nrecv_from_proc = DEBUG_SEND_RECV_DATA ? size * (3 + 1) : size * 3;
            assert(proc != comm->me && nrecv_from_proc > 0);
            int mpi_tag = get_mpi_tag_many_cuts(comm->me, proc);
            assert(recv_request_proc_pair_to_idx[curr_dt_idx][dep].count({proc, send_dep}));

            int recv_proc_to_proc_idx = recv_request_proc_pair_to_idx[curr_dt_idx][dep][{proc, send_dep}];
            assert(recv_proc_to_proc_idx == i);
            MPI_Irecv(buf_recv_proc_to_proc[DEFAULT_PIPELINE_STAGE][send_dep][proc], nrecv_from_proc, MPI_DOUBLE,
                        proc, mpi_tag,
                        proc_to_proc_pipelined_comms[DEFAULT_PIPELINE_STAGE][send_dep], 
                        &recv_r_proc_to_proc[recv_proc_to_proc_idx]);
        }
    }

    template <bool curr_dt>
    void RECEIVE_DATA_ZOID_TO_ZOID_PIPELINED(int zoid_num, std::vector<MPI_Request>& r, int start_t, int end_t,
                                             int pipeline_stage) {
        constexpr int curr_dt_idx = static_cast<int>(curr_dt);

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
                assert(ZOID_TO_ZOID_TO_VCI_IDX[curr_dt_idx].count({recv_zoid_num, zoid_num}));
                int comm_idx = ZOID_TO_ZOID_TO_VCI_IDX[curr_dt_idx].at({recv_zoid_num, zoid_num});
                // int comm_idx = curr_dt ? ZOID_TO_ZOID_TO_VCI_IDX.at({recv_zoid_num, zoid_num}) : ZOID_TO_ZOID_TO_VCI_IDX_NEXT_DT.at({recv_zoid_num, zoid_num});

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
                              << " recv from zoid: " << recv_zoid_num
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
    void UNPACK_FORCE_MANY_CUTS_HELPER_PIPELINED(queue_info& zoid, double* buf, int recv_idx, int recv_zoid_num,
                                                 int start_t, int end_t, int pipeline_stage) {
        auto& recv_zoid = curr_dt ? zoid_num_to_zoid_many_cuts[recv_zoid_num]
                                  : zoid_num_to_zoid_many_cuts_next_dt[recv_zoid_num];

        auto& recv_force_idxs = zoid.recv_force_idxs_double_buffering_flattened_pipelined[pipeline_stage][recv_idx];

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
                              << " recv from zoid: " << recv_zoid_num
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
            /*
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
            */
            auto* _noalias const buf_ = (dbl3_t_stencil_md*) buf;
            #pragma cilk grainsize 2048
            cilk_for (int i = 0; i < recv_force_idxs.size(); i++) {
                int idx = recv_force_idxs[i];
                const auto& f_ = buf_[i];
                zoid.f_stencil_md[0][idx].x += f_.x;
                zoid.f_stencil_md[0][idx].y += f_.y;
                zoid.f_stencil_md[0][idx].z += f_.z;
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
            int num_zoids_in_dimension;
            if (dim == 0) {
                num_zoids_in_dimension = NUM_ZOIDS_X;
            } else if (dim == 1) {
                num_zoids_in_dimension = NUM_ZOIDS_Y;
            } else {
                num_zoids_in_dimension = NUM_ZOIDS_Z;
            }

            if (recv_zoid.where[dim] == num_zoids_in_dimension - 1 && zoid.where[dim] == 0) {
                pbc_flag_[dim] = -1;
            }

            if (recv_zoid.where[dim] == 0 && zoid.where[dim] == num_zoids_in_dimension - 1) {
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
    void UNPACK_POS_VEL_MANY_CUTS_HELPER_PIPELINED(queue_info& zoid, double* buf, int recv_idx, int recv_zoid_num,
                                                   int start_t, int end_t, int pipeline_stage) {
        auto& recv_zoid = curr_dt ? zoid_num_to_zoid_many_cuts[recv_zoid_num]
                                  : zoid_num_to_zoid_many_cuts_next_dt[recv_zoid_num];

        int pbc_flag_[3] = {0};
        for (int dim = 0; dim < 3; dim++) {
            int num_zoids_in_dimension;
            if (dim == 0) {
                num_zoids_in_dimension = NUM_ZOIDS_X;
            } else if (dim == 1) {
                num_zoids_in_dimension = NUM_ZOIDS_Y;
            } else {
                num_zoids_in_dimension = NUM_ZOIDS_Z;
            }

            if (recv_zoid.where[dim] == num_zoids_in_dimension - 1 && zoid.where[dim] == 0) {
                pbc_flag_[dim] = -1;
            }

            if (recv_zoid.where[dim] == 0 && zoid.where[dim] == num_zoids_in_dimension - 1) {
                pbc_flag_[dim] = 1;
            }
        }

        auto& recv_force_idxs = zoid.recv_force_idxs_double_buffering_flattened_pipelined[pipeline_stage][recv_idx];
        auto& recv_pos_idxs = zoid.recv_pos_idxs_double_buffering_flattened_pipelined[pipeline_stage][0][recv_idx];
        auto& recv_pos_idxs2 = zoid.recv_pos_idxs_double_buffering_flattened_pipelined[pipeline_stage][1][recv_idx];

        int num_recv_force = recv_force_idxs.size();
        int num_recv_pos = recv_pos_idxs.size();
        int num_recv_pos2 = recv_pos_idxs2.size();

        if (DEBUG_SEND_RECV_DATA) {
            int pos_starting_idx = num_recv_force * (3 + 1);

            for (int i = 0; i < recv_pos_idxs.size(); i++) {
                int idx = recv_pos_idxs[i];
                int buf_idx = pos_starting_idx + i * (3 + 1);
                auto target_tag = (tagint) ubuf(buf[buf_idx]).i;
                double x_x = buf[buf_idx + 1];
                double x_y = buf[buf_idx + 2];
                double x_z = buf[buf_idx + 3];
                if (target_tag != zoid.tag_stencil_md[0][idx]) {
                    std::cout << "POS me: " << comm->me << " my zoid: " << zoid.num
                              << " recv from: " << recv_zoid_num
                              << " recv proc: " << recv_zoid_num % comm->nprocs 
                              << " tag I got: " << target_tag 
                              << " tag I want: " << zoid.tag_stencil_md[0][idx]
                              << " pos starting idx: " << pos_starting_idx
                              << " i: " << i
                              << " buf idx: " << buf_idx
                              << " pipeline stage: " << pipeline_stage
                              << std::endl;
                }
                assert(target_tag == zoid.tag_stencil_md[0][idx]);
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
        } else {
            auto* _noalias const buf_ = (dbl3_t_stencil_md*) buf;
            int pos_starting_idx = num_recv_force;

            auto * _noalias x0_ = zoid.x_stencil_md[0].data();
            auto * _noalias x1_ = zoid.x_stencil_md[1].data();

            #pragma cilk grainsize 2048
            cilk_for (int i = 0; i < recv_pos_idxs.size(); i++) {
                int idx = recv_pos_idxs[i];
                int buf_idx = pos_starting_idx + i;
                const auto& x_ = buf_[buf_idx];

                /*
                zoid.x_stencil_md[0][idx].x = x_.x + pbc_flag_[0] * domain->prd[0];
                zoid.x_stencil_md[0][idx].y = x_.y + pbc_flag_[1] * domain->prd[1];
                zoid.x_stencil_md[0][idx].z = x_.z + pbc_flag_[2] * domain->prd[2];
                */
                x0_[idx].x = x_.x + pbc_flag_[0] * domain->prd[0];
                x0_[idx].y = x_.y + pbc_flag_[1] * domain->prd[1];
                x0_[idx].z = x_.z + pbc_flag_[2] * domain->prd[2];
            }

            int pos_starting_idx2 = (num_recv_force + num_recv_pos);

            #pragma cilk grainsize 2048
            cilk_for (int i = 0; i < recv_pos_idxs2.size(); i++) {
                int idx = recv_pos_idxs2[i];
                int buf_idx = pos_starting_idx2 + i;
                const auto& x_ = buf_[buf_idx];
                /*
                zoid.x_stencil_md[1][idx].x = x_.x + pbc_flag_[0] * domain->prd[0];
                zoid.x_stencil_md[1][idx].y = x_.y + pbc_flag_[1] * domain->prd[1];
                zoid.x_stencil_md[1][idx].z = x_.z + pbc_flag_[2] * domain->prd[2];
                */
                x1_[idx].x = x_.x + pbc_flag_[0] * domain->prd[0];
                x1_[idx].y = x_.y + pbc_flag_[1] * domain->prd[1];
                x1_[idx].z = x_.z + pbc_flag_[2] * domain->prd[2];
            }
        }

        auto& recv_vel_idxs = zoid.recv_vel_idxs_double_buffering_flattened_pipelined[pipeline_stage][0][recv_idx];
        if (DEBUG_SEND_RECV_DATA) {
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
            auto* _noalias const buf_ = (dbl3_t_stencil_md*) buf;
            auto * _noalias v0_ = zoid.v_stencil_md[0].data();

            int vel_starting_idx = (num_recv_force + num_recv_pos + num_recv_pos2);
            #pragma cilk grainsize 2048
            cilk_for (int i = 0; i < recv_vel_idxs.size(); i++) {
                int idx = recv_vel_idxs[i];
                int buf_idx = vel_starting_idx + i;
                const auto& v_ = buf_[buf_idx];
                /*
                zoid.v_stencil_md[0][idx].x = v_.x;
                zoid.v_stencil_md[0][idx].y = v_.y;
                zoid.v_stencil_md[0][idx].z = v_.z;
                */
                v0_[idx].x = v_.x;
                v0_[idx].y = v_.y;
                v0_[idx].z = v_.z;
            }
        }

        if constexpr (EXPERIMENT == DPD) {
            auto& recv_vel_idxs2 = zoid.recv_vel_idxs_double_buffering_flattened_pipelined[pipeline_stage][1][recv_idx];

            int num_recv_vel = recv_vel_idxs.size();

            if (DEBUG_SEND_RECV_DATA) {
                int vel_starting_idx2 = (num_recv_force + num_recv_pos + num_recv_pos2 + num_recv_vel) * (3 + 1);
                for (int i = 0; i < recv_vel_idxs2.size(); i++) {
                    int idx = recv_vel_idxs2[i];
                    int buf_idx = vel_starting_idx2 + i * (3 + 1);
                    auto target_tag = (tagint) ubuf(buf[buf_idx]).i;
                    double v_x = buf[buf_idx + 1];
                    double v_y = buf[buf_idx + 2];
                    double v_z = buf[buf_idx + 3];
                    assert(target_tag == zoid.tag_stencil_md[0][idx]);
                    zoid.v_stencil_md[1][idx].x = v_x;
                    zoid.v_stencil_md[1][idx].y = v_y;
                    zoid.v_stencil_md[1][idx].z = v_z;
                }
            } else {
                auto* _noalias const buf_ = (dbl3_t_stencil_md*) buf;
                auto * _noalias v1_ = zoid.v_stencil_md[1].data();

                int vel_starting_idx2 = (num_recv_force + num_recv_pos + num_recv_pos2 + num_recv_vel);
                #pragma cilk grainsize 2048
                cilk_for (int i = 0; i < recv_vel_idxs2.size(); i++) {
                    int idx = recv_vel_idxs2[i];
                    int buf_idx = vel_starting_idx2 + i;
                    const auto& v_ = buf_[buf_idx];
                    /*
                    zoid.v_stencil_md[1][idx].x = v_.x;
                    zoid.v_stencil_md[1][idx].y = v_.y;
                    zoid.v_stencil_md[1][idx].z = v_.z;
                    */
                    v1_[idx].x = v_.x;
                    v1_[idx].y = v_.y;
                    v1_[idx].z = v_.z;
                }
            }
        }

        /*
        auto& recv_force_idxs = zoid.recv_force_idxs_double_buffering_flattened_pipelined[pipeline_stage][recv_idx];
        auto& recv_pos_idxs = zoid.recv_pos_idxs_double_buffering_flattened_pipelined[pipeline_stage][0][recv_idx];
        auto& recv_pos_idxs2 = zoid.recv_pos_idxs_double_buffering_flattened_pipelined[pipeline_stage][1][recv_idx];

        int num_recv_force = recv_force_idxs.size();
        int num_recv_pos = recv_pos_idxs.size();
        int num_recv_pos2 = recv_pos_idxs2.size();

        auto& recv_vel_idxs = zoid.recv_vel_idxs_double_buffering_flattened_pipelined[pipeline_stage][recv_idx];
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
            auto* _noalias const buf_ = (dbl3_t_stencil_md*) buf;
            int pos_starting_idx = num_recv_force;

            #pragma cilk grainsize 2048
            cilk_for (int i = 0; i < recv_pos_idxs.size(); i++) {
                int idx = recv_pos_idxs[i];
                int buf_idx = pos_starting_idx + i;
                const auto& x_ = buf_[buf_idx];

                zoid.x_stencil_md[0][idx].x = x_.x + pbc_flag_[0] * domain->prd[0];
                zoid.x_stencil_md[0][idx].y = x_.y + pbc_flag_[1] * domain->prd[1];
                zoid.x_stencil_md[0][idx].z = x_.z + pbc_flag_[2] * domain->prd[2];
            }

            int pos_starting_idx2 = (num_recv_force + num_recv_pos);

            #pragma cilk grainsize 2048
            cilk_for (int i = 0; i < recv_pos_idxs2.size(); i++) {
                int idx = recv_pos_idxs2[i];
                int buf_idx = pos_starting_idx2 + i;
                const auto& x_ = buf_[buf_idx];
                zoid.x_stencil_md[1][idx].x = x_.x + pbc_flag_[0] * domain->prd[0];
                zoid.x_stencil_md[1][idx].y = x_.y + pbc_flag_[1] * domain->prd[1];
                zoid.x_stencil_md[1][idx].z = x_.z + pbc_flag_[2] * domain->prd[2];
            }

            int vel_starting_idx = (num_recv_force + num_recv_pos + num_recv_pos2);
            #pragma cilk grainsize 2048
            cilk_for (int i = 0; i < recv_vel_idxs.size(); i++) {
                int idx = recv_vel_idxs[i];
                int buf_idx = vel_starting_idx + i;
                const auto& v_ = buf_[buf_idx];
                zoid.v_stencil_md[0][idx].x = v_.x;
                zoid.v_stencil_md[0][idx].y = v_.y;
                zoid.v_stencil_md[0][idx].z = v_.z;
            }
        }
        */
    }

    template <bool curr_dt>
    void UNPACK_DATA_MANY_CUTS_HELPER(queue_info& zoid, double* buf, int recv_idx, int recv_zoid_num,
                                      int start_t, int end_t) {
        auto& recv_zoid = curr_dt ? zoid_num_to_zoid_many_cuts[recv_zoid_num]
                                  : zoid_num_to_zoid_many_cuts_next_dt[recv_zoid_num];

        int pbc_flag_[3] = {0};
        for (int dim = 0; dim < 3; dim++) {
            int num_zoids_in_dimension;
            if (dim == 0) {
                num_zoids_in_dimension = NUM_ZOIDS_X;
            } else if (dim == 1) {
                num_zoids_in_dimension = NUM_ZOIDS_Y;
            } else {
                num_zoids_in_dimension = NUM_ZOIDS_Z;
            }

            if (recv_zoid.where[dim] == num_zoids_in_dimension - 1 && zoid.where[dim] == 0) {
                pbc_flag_[dim] = -1;
            }

            if (recv_zoid.where[dim] == 0 && zoid.where[dim] == num_zoids_in_dimension - 1) {
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
            cilk_for (int i = 0; i < recv_force_idxs.size(); i++) {
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

            cilk_for (int i = 0; i < recv_pos_idxs.size(); i++) {
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

            cilk_for (int i = 0; i < recv_pos_idxs2.size(); i++) {
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
            cilk_for (int i = 0; i < recv_vel_idxs.size(); i++) {
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
            int num_zoids_in_dimension;
            if (dim == 0) {
                num_zoids_in_dimension = NUM_ZOIDS_X;
            } else if (dim == 1) {
                num_zoids_in_dimension = NUM_ZOIDS_Y;
            } else {
                num_zoids_in_dimension = NUM_ZOIDS_Z;
            }

            if (recv_zoid.where[dim] == num_zoids_in_dimension - 1 && zoid.where[dim] == 0) {
                pbc_flag_[dim] = -1;
            }

            if (recv_zoid.where[dim] == 0 && zoid.where[dim] == num_zoids_in_dimension - 1) {
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

    template <bool curr_dt>
    void UNPACK_DATA_MANY_CUTS_HELPER_SELF_PIPELINED(queue_info& zoid, int recv_idx, int recv_zoid_num,
                                                     int send_idx, int start_t, int end_t, int pipeline_stage) {

        assert(recv_zoid_num % comm->nprocs == comm->me);

        auto &recv_zoid = curr_dt ? zoid_num_to_zoid_many_cuts[recv_zoid_num]
                                  : zoid_num_to_zoid_many_cuts_next_dt[recv_zoid_num];

        int pbc_flag_[3] = {0};
        for (int dim = 0; dim < 3; dim++) {
            int num_zoids_in_dimension;
            if (dim == 0) {
                num_zoids_in_dimension = NUM_ZOIDS_X;
            } else if (dim == 1) {
                num_zoids_in_dimension = NUM_ZOIDS_Y;
            } else {
                num_zoids_in_dimension = NUM_ZOIDS_Z;
            }

            if (recv_zoid.where[dim] == num_zoids_in_dimension - 1 && zoid.where[dim] == 0) {
                pbc_flag_[dim] = -1;
            }

            if (recv_zoid.where[dim] == 0 && zoid.where[dim] == num_zoids_in_dimension - 1) {
                pbc_flag_[dim] = 1;
            }
        }

        auto& send_force_idxs = recv_zoid.send_force_idxs_double_buffering_flattened_pipelined[pipeline_stage][send_idx];
        auto& send_pos_idxs = recv_zoid.send_pos_idxs_double_buffering_flattened_pipelined[pipeline_stage][0][send_idx];
        auto& send_pos_idxs2 = recv_zoid.send_pos_idxs_double_buffering_flattened_pipelined[pipeline_stage][1][send_idx];

        auto& recv_force_idxs = zoid.recv_force_idxs_double_buffering_flattened_pipelined[pipeline_stage][recv_idx];
        auto& recv_pos_idxs = zoid.recv_pos_idxs_double_buffering_flattened_pipelined[pipeline_stage][0][recv_idx];
        auto& recv_pos_idxs2 = zoid.recv_pos_idxs_double_buffering_flattened_pipelined[pipeline_stage][1][recv_idx];

        auto * _noalias const recv_f_ = zoid.f_stencil_md[0].data();
        auto * _noalias const send_f_ = recv_zoid.f_stencil_md[0].data();

        auto * _noalias const recv_x0_ = zoid.x_stencil_md[0].data();
        auto * _noalias const send_x0_ = recv_zoid.x_stencil_md[0].data();

        auto * _noalias const recv_x1_ = zoid.x_stencil_md[1].data();
        auto * _noalias const send_x1_ = recv_zoid.x_stencil_md[1].data();

        #pragma cilk grainsize 2048
        cilk_for (int i = 0; i < recv_force_idxs.size(); i++) {
            int recv_force_idx = recv_force_idxs[i];
            int send_force_idx = send_force_idxs[i];

            // auto& recv_f = zoid.f_stencil_md[0][recv_force_idx];
            // auto& send_f = recv_zoid.f_stencil_md[0][send_force_idx];
            auto& recv_f = recv_f_[recv_force_idx];
            auto& send_f = send_f_[send_force_idx];

            assert(recv_zoid.tag_stencil_md[0][send_force_idx] == zoid.tag_stencil_md[0][recv_force_idx]);
            recv_f.x += send_f.x;
            recv_f.y += send_f.y;
            recv_f.z += send_f.z;

            send_f.x = 0;
            send_f.y = 0;
            send_f.z = 0;
        }

        #pragma cilk grainsize 2048
        cilk_for (int i = 0; i < recv_pos_idxs.size(); i++) {
            int recv_pos_idx = recv_pos_idxs[i];
            int send_pos_idx = send_pos_idxs[i];

            // auto& recv_pos = zoid.x_stencil_md[0][recv_pos_idx];
            // auto& send_pos = recv_zoid.x_stencil_md[0][send_pos_idx];
            auto& recv_pos = recv_x0_[recv_pos_idx];
            auto& send_pos = send_x0_[send_pos_idx];

            assert(recv_zoid.tag_stencil_md[0][send_pos_idx] == zoid.tag_stencil_md[0][recv_pos_idx]);
            recv_pos.x = send_pos.x + pbc_flag_[0] * domain->prd[0];
            recv_pos.y = send_pos.y + pbc_flag_[1] * domain->prd[1];
            recv_pos.z = send_pos.z + pbc_flag_[2] * domain->prd[2];
        }

        #pragma cilk grainsize 2048
        cilk_for (int i = 0; i < recv_pos_idxs2.size(); i++) {
            int recv_pos_idx = recv_pos_idxs2[i];
            int send_pos_idx = send_pos_idxs2[i];

            // auto& recv_pos = zoid.x_stencil_md[1][recv_pos_idx];
            // auto& send_pos = recv_zoid.x_stencil_md[1][send_pos_idx];
            auto& recv_pos = recv_x1_[recv_pos_idx];
            auto& send_pos = send_x1_[send_pos_idx];

            assert(recv_zoid.tag_stencil_md[0][send_pos_idx] == zoid.tag_stencil_md[0][recv_pos_idx]);
            recv_pos.x = send_pos.x + pbc_flag_[0] * domain->prd[0];
            recv_pos.y = send_pos.y + pbc_flag_[1] * domain->prd[1];
            recv_pos.z = send_pos.z + pbc_flag_[2] * domain->prd[2];
        }


        assert(send_force_idxs.size() == recv_force_idxs.size());
        assert(send_pos_idxs.size() == recv_pos_idxs.size());
        assert(send_pos_idxs2.size() == recv_pos_idxs2.size());

        auto& send_vel_idxs = recv_zoid.send_vel_idxs_double_buffering_flattened_pipelined[pipeline_stage][0][send_idx];
        auto& recv_vel_idxs = zoid.recv_vel_idxs_double_buffering_flattened_pipelined[pipeline_stage][0][recv_idx];
        assert(send_vel_idxs.size() == recv_vel_idxs.size());

        auto * _noalias const recv_v0_ = zoid.v_stencil_md[0].data();
        auto * _noalias const send_v0_ = recv_zoid.v_stencil_md[0].data();

        #pragma cilk grainsize 2048
        cilk_for (int i = 0; i < recv_vel_idxs.size(); i++) {
            int recv_vel_idx = recv_vel_idxs[i];
            int send_vel_idx = send_vel_idxs[i];

            // auto& recv_vel = zoid.v_stencil_md[0][recv_vel_idx];
            // auto& send_vel = recv_zoid.v_stencil_md[0][send_vel_idx];
            auto& recv_vel = recv_v0_[recv_vel_idx];
            auto& send_vel = send_v0_[send_vel_idx];

            assert(recv_zoid.tag_stencil_md[0][send_vel_idx] == zoid.tag_stencil_md[0][recv_vel_idx]);

            recv_vel.x = send_vel.x;
            recv_vel.y = send_vel.y;
            recv_vel.z = send_vel.z;
        }

        if constexpr (EXPERIMENT == DPD) {
            auto& send_vel_idxs2 = recv_zoid.send_vel_idxs_double_buffering_flattened_pipelined[pipeline_stage][1][send_idx];
            auto& recv_vel_idxs2 = zoid.recv_vel_idxs_double_buffering_flattened_pipelined[pipeline_stage][1][recv_idx];
            assert(send_vel_idxs2.size() == recv_vel_idxs2.size());

            auto * _noalias const recv_v1_ = zoid.v_stencil_md[1].data();
            auto * _noalias const send_v1_ = recv_zoid.v_stencil_md[1].data();

            #pragma cilk grainsize 2048
            cilk_for (int i = 0; i < recv_vel_idxs2.size(); i++) {
                int recv_vel_idx = recv_vel_idxs2[i];
                int send_vel_idx = send_vel_idxs2[i];

                // auto& recv_vel = zoid.v_stencil_md[1][recv_vel_idx];
                // auto& send_vel = recv_zoid.v_stencil_md[1][send_vel_idx];
                auto& recv_vel = recv_v1_[recv_vel_idx];
                auto& send_vel = send_v1_[send_vel_idx];

                assert(recv_zoid.tag_stencil_md[0][send_vel_idx] == zoid.tag_stencil_md[0][recv_vel_idx]);
                recv_vel.x = send_vel.x;
                recv_vel.y = send_vel.y;
                recv_vel.z = send_vel.z;
            }
        }

        /*
        auto& send_force_idxs = recv_zoid.send_force_idxs_double_buffering_flattened_pipelined[pipeline_stage][send_idx];
        auto& send_pos_idxs = recv_zoid.send_pos_idxs_double_buffering_flattened_pipelined[pipeline_stage][0][send_idx];
        auto& send_pos_idxs2 = recv_zoid.send_pos_idxs_double_buffering_flattened_pipelined[pipeline_stage][1][send_idx];
        auto& send_vel_idxs = recv_zoid.send_vel_idxs_double_buffering_flattened_pipelined[pipeline_stage][send_idx];

        auto& recv_force_idxs = zoid.recv_force_idxs_double_buffering_flattened_pipelined[pipeline_stage][recv_idx];
        auto& recv_pos_idxs = zoid.recv_pos_idxs_double_buffering_flattened_pipelined[pipeline_stage][0][recv_idx];
        auto& recv_pos_idxs2 = zoid.recv_pos_idxs_double_buffering_flattened_pipelined[pipeline_stage][1][recv_idx];
        auto& recv_vel_idxs = zoid.recv_vel_idxs_double_buffering_flattened_pipelined[pipeline_stage][recv_idx];

        assert(send_force_idxs.size() == recv_force_idxs.size());
        assert(send_pos_idxs.size() == recv_pos_idxs.size());
        assert(send_pos_idxs2.size() == recv_pos_idxs2.size());
        assert(send_vel_idxs.size() == recv_vel_idxs.size());

        #pragma cilk grainsize 2048
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

        #pragma cilk grainsize 2048
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

        #pragma cilk grainsize 2048
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

        #pragma cilk grainsize 2048
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
        */
    }

    void UNPACK_DATA_MANY_CUTS_HELPER_SETUP(queue_info& zoid, double* buf, int recv_idx, int recv_zoid_num) {
        constexpr int start_t = 0;
        constexpr int end_t = 1;

        auto& recv_zoid = zoid_num_to_zoid_many_cuts[recv_zoid_num];

        int pbc_flag_[3] = {0};
        for (int dim = 0; dim < 3; dim++) {
            int num_zoids_in_dimension;
            if (dim == 0) {
                num_zoids_in_dimension = NUM_ZOIDS_X;
            } else if (dim == 1) {
                num_zoids_in_dimension = NUM_ZOIDS_Y;
            } else {
                num_zoids_in_dimension = NUM_ZOIDS_Z;
            }

            if (recv_zoid.where[dim] == num_zoids_in_dimension - 1 && zoid.where[dim] == 0) {
                pbc_flag_[dim] = -1;
            }

            if (recv_zoid.where[dim] == 0 && zoid.where[dim] == num_zoids_in_dimension - 1) {
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
        int zoid_num = zoid.num;
        auto& recv_neighbors = recv_from_neighbors_many_cuts[zoid_num];

        auto& not_my_proc_idxs = recv_from_neighbors_not_my_proc_idxs[zoid.num];
        assert(r.size() == not_my_proc_idxs.size());

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
    int UNPACK_DATA_MANY_CUTS_ZOID_SELF_ONLY_PIPELINED(queue_info& zoid, int start_t, int end_t, int pipeline_stage) {
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
            UNPACK_DATA_MANY_CUTS_HELPER_SELF_PIPELINED<curr_dt>(zoid, i, recv_zoid_num, find_idx, start_t, end_t, pipeline_stage);

            num_neighbors_recv++;
        }

        return num_neighbors_recv;
    }

    template <bool curr_dt>
    void UNPACK_FORCE_MANY_CUTS_ZOID(queue_info& zoid, int start_t, int end_t) {
        // cilk_scope {
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
    void UNPACK_FORCE_MANY_CUTS_ZOID_PIPELINED(queue_info& zoid, int start_t, int end_t, int pipeline_stage) {
        int zoid_num = zoid.num;
        auto& recv_neighbors = curr_dt ? recv_from_neighbors_many_cuts[zoid_num]
                                       : recv_from_neighbors_many_cuts_next_dt[zoid_num];

        for (int i = 0; i < recv_neighbors.size(); i++) {
            if (recv_neighbors[i] % comm->nprocs != comm->me) {
                auto buf = buf_recv_zoid_to_zoid[pipeline_stage][zoid_num][i];
                int recv_zoid_num = recv_neighbors[i];
                UNPACK_FORCE_MANY_CUTS_HELPER_PIPELINED<curr_dt>(zoid, buf, i,
                                                                 recv_zoid_num, start_t, end_t, pipeline_stage);
            }
        }
    }

    template <bool curr_dt>
    void UNPACK_FORCE_MANY_CUTS_ZOID_PIPELINED_ONLY_NEXT_DEP(queue_info& zoid, int dep, int start_t, int end_t, int pipeline_stage) {
        int zoid_num = zoid.num;
        auto& recv_neighbors = curr_dt ? recv_from_neighbors_many_cuts[zoid_num]
            : recv_from_neighbors_many_cuts_next_dt[zoid_num];

        constexpr int curr_dt_idx = static_cast<int>(curr_dt);

        for (int i = 0; i < recv_neighbors.size(); i++) {
            if (recv_neighbors[i] % comm->nprocs != comm->me) {
                int recv_zoid_num = recv_neighbors[i];
                int recv_zoid_dep = curr_dt ? zoid_num_to_dep[recv_zoid_num] : zoid_num_to_dep_next_dt[recv_zoid_num];
                if (recv_zoid_dep == dep - 1) {
                    auto buf = buf_recv_zoid_to_zoid[pipeline_stage][zoid_num][i];
                    UNPACK_FORCE_MANY_CUTS_HELPER_PIPELINED<curr_dt>(zoid, buf, i,
                                                                     recv_zoid_num, start_t, end_t, pipeline_stage);
                } else {
                    auto buf = buf_recv_proc_to_proc[pipeline_stage][recv_zoid_dep][recv_zoid_num % comm->nprocs];
                    int offset = recv_proc_zoid_offsets[curr_dt_idx][pipeline_stage][zoid.num][i];
                    offset = DEBUG_SEND_RECV_DATA ? offset * (3 + 1) : offset * 3;
                    UNPACK_FORCE_MANY_CUTS_HELPER_PIPELINED<curr_dt>(zoid, buf + offset, i,
                                                                     recv_zoid_num, start_t, end_t, pipeline_stage);
                }
            }
        }
    }

    template <bool curr_dt>
    void UNPACK_POS_VEL_MANY_CUTS_ZOID(queue_info& zoid, int recv_zoid_num, int start_t, int end_t) {
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
    void UNPACK_POS_VEL_MANY_CUTS_ZOID_PIPELINED(queue_info& zoid, int recv_zoid_num, int start_t, int end_t, int pipeline_stage) {
        int zoid_num = zoid.num;
        auto& recv_neighbors = curr_dt ? recv_from_neighbors_many_cuts[zoid_num]
                                       : recv_from_neighbors_many_cuts_next_dt[zoid_num];

        auto find_it = std::find(recv_neighbors.begin(), recv_neighbors.end(), recv_zoid_num);
        assert(find_it != recv_neighbors.end());
        int find_idx = std::distance(recv_neighbors.begin(), find_it);

        auto buf = buf_recv_zoid_to_zoid[pipeline_stage][zoid_num][find_idx];
        UNPACK_POS_VEL_MANY_CUTS_HELPER_PIPELINED<curr_dt>(zoid, buf, find_idx, recv_zoid_num, start_t, end_t, pipeline_stage);
    }

    template <bool curr_dt>
    void UNPACK_POS_VEL_MANY_CUTS_ZOID_PIPELINED_PROC_TO_PROC(queue_info& zoid, int send_dep, int proc,
                                                              int recv_zoid_num, int find_idx, int start_t, int end_t, int pipeline_stage) {
        constexpr int curr_dt_idx = static_cast<int>(curr_dt);
        auto& recv_neighbors = curr_dt ? recv_from_neighbors_many_cuts[zoid.num]
            : recv_from_neighbors_many_cuts_next_dt[zoid.num];
        assert(recv_neighbors[find_idx] == recv_zoid_num);

        auto buf = buf_recv_proc_to_proc[pipeline_stage][send_dep][proc];
        int offset = recv_proc_zoid_offsets[curr_dt_idx][pipeline_stage][zoid.num][find_idx];
        int zoid_dep = curr_dt ? zoid_num_to_dep[zoid.num] : zoid_num_to_dep_next_dt[zoid.num];
        offset = DEBUG_SEND_RECV_DATA ? offset * (3 + 1) : offset * 3;
        UNPACK_POS_VEL_MANY_CUTS_HELPER_PIPELINED<curr_dt>(zoid, buf + offset, find_idx, recv_zoid_num, start_t, end_t, pipeline_stage);
    }

    template <bool curr_dt>
    void UNPACK_DATA_MANY_CUTS_ZOID(queue_info& zoid, std::vector<MPI_Request>& r, int start_t, int end_t) {
        // cilk_scope {
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

    template <bool curr_dt>
    int PACK_DATA_PROC_TO_PROC_HELPER(queue_info& zoid, int pipeline_stage, int dep, int proc) {
        if (proc == comm->me) {
            return 0;
        }

        constexpr int curr_dt_idx = static_cast<int>(curr_dt);
        constexpr int start_timestep = USE_PIPELINE ? start_t[pipeline_stage] : default_start_t;
        constexpr int end_timestep = USE_PIPELINE ? end_t[pipeline_stage] : default_end_t;

        auto& queues = curr_dt ? my_queues_many_cuts[dep] : my_queues_many_cuts_next_dt[dep];
        auto pair = std::make_pair(zoid.num, proc);

        auto buf = buf_send_proc_to_proc[pipeline_stage][dep][proc];

        assert(send_proc_zoid_offsets[curr_dt_idx][pipeline_stage].count(pair));
        int offset = send_proc_zoid_offsets[curr_dt_idx][pipeline_stage].at(pair);
        offset = DEBUG_SEND_RECV_DATA ? offset * (3 + 1) : offset * 3;

        assert(send_proc_zoid_sizes[curr_dt_idx][pipeline_stage].count(pair));
        int size = send_proc_zoid_sizes[curr_dt_idx][pipeline_stage].at(pair);
        size = DEBUG_SEND_RECV_DATA ? size * (3 + 1) : size * 3;

        auto& send_neighbors = curr_dt ? send_to_neighbors_many_cuts[zoid.num] : send_to_neighbors_many_cuts_next_dt[zoid.num];

        int zoid_ndoubles_send = 0;

        for (int i = 0; i < send_neighbors.size(); i++) {
            int send_zoid_num = send_neighbors[i];
            int send_zoid_dep = curr_dt ? zoid_num_to_dep[send_zoid_num] : zoid_num_to_dep_next_dt[send_zoid_num];
            if (send_zoid_num % comm->nprocs != proc || send_zoid_dep == dep + 1) {
                continue;
            }

            int buf_idx = PACK_DATA_MANY_CUTS_HELPER_PIPELINED<curr_dt>(zoid, buf + offset + zoid_ndoubles_send, i, send_zoid_num,
                                                            start_timestep, end_timestep, pipeline_stage);

            zoid_ndoubles_send += buf_idx;
        }

        if (zoid_ndoubles_send > nsend_buf_proc_to_proc[pipeline_stage][dep][proc]) {
            assert(false);
            GROW_SEND_PROC_TO_PROC_MANY_CUTS(pipeline_stage, dep, proc, zoid_ndoubles_send);
        }

        assert(zoid_ndoubles_send == size);
        return zoid_ndoubles_send;
    }

    template <bool curr_dt>
    int PACK_DATA_MANY_CUTS_HELPER_PIPELINED(queue_info& zoid, double* buf,
                                             int send_idx, int send_zoid_num,
                                             int start_t, int end_t,
                                             int pipeline_stage) {
        auto& send_force_idxs = zoid.send_force_idxs_double_buffering_flattened_pipelined[pipeline_stage][send_idx];
        auto& send_pos_idxs = zoid.send_pos_idxs_double_buffering_flattened_pipelined[pipeline_stage][0][send_idx];
        auto& send_pos_idxs2 = zoid.send_pos_idxs_double_buffering_flattened_pipelined[pipeline_stage][1][send_idx];
        auto& send_vel_idxs = zoid.send_vel_idxs_double_buffering_flattened_pipelined[pipeline_stage][0][send_idx];

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
        } else {
            auto * _noalias f_ = zoid.f_stencil_md[0].data();
            auto * _noalias x0_ = zoid.x_stencil_md[0].data();
            auto * _noalias x1_ = zoid.x_stencil_md[1].data();
            auto * _noalias v_ = zoid.v_stencil_md[0].data();

            #pragma cilk grainsize 2048
            cilk_for (int i = 0; i < send_force_idxs.size(); i++) {
                int idx = send_force_idxs[i];
                int buf_idx = i * 3;

                buf[buf_idx] = f_[idx].x;
                buf[buf_idx + 1] = f_[idx].y;
                buf[buf_idx + 2] = f_[idx].z;

                f_[idx].x = 0;
                f_[idx].y = 0;
                f_[idx].z = 0;

                /*
                buf[buf_idx] = zoid.f_stencil_md[0][idx].x;
                buf[buf_idx + 1] = zoid.f_stencil_md[0][idx].y;
                buf[buf_idx + 2] = zoid.f_stencil_md[0][idx].z;

                zoid.f_stencil_md[0][idx].x = 0;
                zoid.f_stencil_md[0][idx].y = 0;
                zoid.f_stencil_md[0][idx].z = 0;
                */
            }

            int pos_starting_idx = num_send_force * 3;
            #pragma cilk grainsize 2048
            cilk_for (int i = 0; i < send_pos_idxs.size(); i++) {
                int idx = send_pos_idxs[i];
                int buf_idx = pos_starting_idx + i * 3;

                /*
                buf[buf_idx] = zoid.x_stencil_md[0][idx].x;
                buf[buf_idx + 1] = zoid.x_stencil_md[0][idx].y;
                buf[buf_idx + 2] = zoid.x_stencil_md[0][idx].z;
                */
                buf[buf_idx] = x0_[idx].x;
                buf[buf_idx + 1] = x0_[idx].y;
                buf[buf_idx + 2] = x0_[idx].z;
            }

            int pos_starting_idx2 = (num_send_force + num_send_pos) * 3;
            #pragma cilk grainsize 2048
            cilk_for (int i = 0; i < send_pos_idxs2.size(); i++) {
                int idx = send_pos_idxs2[i];
                int buf_idx = pos_starting_idx2 + i * 3;

                /*
                buf[buf_idx] = zoid.x_stencil_md[1][idx].x;
                buf[buf_idx + 1] = zoid.x_stencil_md[1][idx].y;
                buf[buf_idx + 2] = zoid.x_stencil_md[1][idx].z;
                */
                buf[buf_idx] = x1_[idx].x;
                buf[buf_idx + 1] = x1_[idx].y;
                buf[buf_idx + 2] = x1_[idx].z;
            }

            int vel_starting_idx = (num_send_force + num_send_pos + num_send_pos2) * 3;

            #pragma cilk grainsize 2048
            cilk_for (int i = 0; i < send_vel_idxs.size(); i++) {
                int idx = send_vel_idxs[i];
                int buf_idx = vel_starting_idx + i * 3;

                /*
                buf[buf_idx] = zoid.v_stencil_md[0][idx].x;
                buf[buf_idx + 1] = zoid.v_stencil_md[0][idx].y;
                buf[buf_idx + 2] = zoid.v_stencil_md[0][idx].z;
                */
                buf[buf_idx] = v_[idx].x;
                buf[buf_idx + 1] = v_[idx].y;
                buf[buf_idx + 2] = v_[idx].z;
            }
        }

        if constexpr (EXPERIMENT == DPD) {
            auto& send_vel_idxs2 = zoid.send_vel_idxs_double_buffering_flattened_pipelined[pipeline_stage][1][send_idx];
            int num_send_vel2 = send_vel_idxs2.size();

            if (DEBUG_SEND_RECV_DATA) {
                int vel_starting_idx2 = (num_send_force + num_send_pos + num_send_pos2 + num_send_vel) * (3 + 1);

                for (int i = 0; i < send_vel_idxs2.size(); i++) {
                    int idx = send_vel_idxs2[i];
                    int buf_idx = vel_starting_idx2 + i * (3 + 1);
                    int tag = zoid.tag_stencil_md[0][idx];

                    buf[buf_idx] = ubuf(tag).d;
                    buf[buf_idx + 1] = zoid.v_stencil_md[1][idx].x;
                    buf[buf_idx + 2] = zoid.v_stencil_md[1][idx].y;
                    buf[buf_idx + 3] = zoid.v_stencil_md[1][idx].z;
                }

                return (num_send_force + num_send_pos + num_send_pos2 + num_send_vel) * (3 + 1);
            } else {
                int vel_starting_idx2 = (num_send_force + num_send_pos + num_send_pos2 + num_send_vel) * 3;
                auto * _noalias v_ = zoid.v_stencil_md[1].data();

                #pragma cilk grainsize 2048
                cilk_for (int i = 0; i < send_vel_idxs2.size(); i++) {
                    int idx = send_vel_idxs2[i];
                    int buf_idx = vel_starting_idx2 + i * 3;

                    /*
                    buf[buf_idx] = zoid.v_stencil_md[1][idx].x;
                    buf[buf_idx + 1] = zoid.v_stencil_md[1][idx].y;
                    buf[buf_idx + 2] = zoid.v_stencil_md[1][idx].z;
                    */
                    buf[buf_idx] = v_[idx].x;
                    buf[buf_idx + 1] = v_[idx].y;
                    buf[buf_idx + 2] = v_[idx].z;
                }

                return (num_send_force + num_send_pos + num_send_pos2 + num_send_vel + num_send_vel2) * 3;
            }
        }

        if (DEBUG_SEND_RECV_DATA) {
            return (num_send_force + num_send_pos + num_send_pos2 + num_send_vel) * (3 + 1);
        }
        
        return (num_send_force + num_send_pos + num_send_pos2 + num_send_vel) * 3;

        // return -1;
        /*
        auto& send_force_idxs = zoid.send_force_idxs_double_buffering_flattened_pipelined[pipeline_stage][send_idx];
        auto& send_pos_idxs = zoid.send_pos_idxs_double_buffering_flattened_pipelined[pipeline_stage][0][send_idx];
        auto& send_pos_idxs2 = zoid.send_pos_idxs_double_buffering_flattened_pipelined[pipeline_stage][1][send_idx];
        auto& send_vel_idxs = zoid.send_vel_idxs_double_buffering_flattened_pipelined[pipeline_stage][send_idx];

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

        assert(false);
        return -1;
        */
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

    void NVE_INITIAL_INTEGRATE_ZOID_MANY_CUTS(queue_info& zoid, int dep, int timestep) {
        auto * _noalias x = zoid.x_stencil_md[timestep % DOUBLE_BUFFERING].data();
        auto * _noalias next_x = zoid.x_stencil_md[(timestep + 1) % DOUBLE_BUFFERING].data();

        auto * _noalias v = zoid.v_stencil_md[timestep % 1].data();
        auto * _noalias next_v = v;

        if constexpr (EXPERIMENT == DPD) {
            v = zoid.v_stencil_md[timestep % DOUBLE_BUFFERING].data();
            next_v = zoid.v_stencil_md[(timestep + 1) % DOUBLE_BUFFERING].data();
        }

        auto * _noalias f = zoid.f_stencil_md[timestep % 1].data();

        auto * _noalias mask = zoid.mask_stencil_md[0].data();
        auto * _noalias local_idxs = zoid.local_idxs_per_timestep[timestep].data();
        auto * _noalias type = zoid.type_stencil_md[0].data();

        int nlocal = zoid.local_idxs_per_timestep[timestep].size();

        auto dtv = update->dt;
        auto* mass = atom->mass;
        auto dtf = 0.5 * update->dt * force->ftm2v;

        // if ((dep == 0 || dep == NUM_DEPS - 1) && nlocal > MODIFY_GRAINSIZE) {
        if (nlocal > 4096 && (dep == 0 || dep == 3)) {
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
                            if constexpr (EXPERIMENT == DPD) {
                                next_v[i].x = v[i].x + dtfm * f[i].x;
                                next_v[i].y = v[i].y + dtfm * f[i].y;
                                next_v[i].z = v[i].z + dtfm * f[i].z;
                            } else {
                                v[i].x += dtfm * f[i].x;
                                v[i].y += dtfm * f[i].y;
                                v[i].z += dtfm * f[i].z;
                            }

                            f[i].x = 0.0;
                            f[i].y = 0.0;
                            f[i].z = 0.0;

                            if constexpr (EXPERIMENT == DPD) {
                                next_x[i].x = x[i].x + dtv * next_v[i].x;
                                next_x[i].y = x[i].y + dtv * next_v[i].y;
                                next_x[i].z = x[i].z + dtv * next_v[i].z;
                            } else {
                                next_x[i].x = x[i].x + dtv * v[i].x;
                                next_x[i].y = x[i].y + dtv * v[i].y;
                                next_x[i].z = x[i].z + dtv * v[i].z;
                            }
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

                if constexpr (EXPERIMENT == DPD) {
                    next_v[i].x = v[i].x + dtfm * f[i].x;
                    next_v[i].y = v[i].y + dtfm * f[i].y;
                    next_v[i].z = v[i].z + dtfm * f[i].z;
                } else {
                    v[i].x += dtfm * f[i].x;
                    v[i].y += dtfm * f[i].y;
                    v[i].z += dtfm * f[i].z;
                }

                f[i].x = 0.0;
                f[i].y = 0.0;
                f[i].z = 0.0;

                if constexpr (EXPERIMENT == DPD) {
                    next_x[i].x = x[i].x + dtv * next_v[i].x;
                    next_x[i].y = x[i].y + dtv * next_v[i].y;
                    next_x[i].z = x[i].z + dtv * next_v[i].z;
                } else {
                    next_x[i].x = x[i].x + dtv * v[i].x;
                    next_x[i].y = x[i].y + dtv * v[i].y;
                    next_x[i].z = x[i].z + dtv * v[i].z;
                }
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
        if (nlocal > 4096 && (dep == 0 || dep == 3)) {
            /*
            int num_workers = __cilkrts_get_nworkers();
            int num_chunks = nlocal / MODIFY_GRAINSIZE + 1;
            int chunks_per_worker = num_chunks / num_workers;
            int chunk_size = MODIFY_GRAINSIZE;
            auto* claimed = zoid.claimed_flags_stencil_md[0];
            */

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

            /*
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
            */
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

    void NVE_FINAL_INTEGRATE_ZOID_MANY_CUTS(queue_info& zoid, int dep, int timestep) {
        auto * _noalias v = zoid.v_stencil_md[timestep % 1].data();
        if constexpr (EXPERIMENT == DPD) {
            v = zoid.v_stencil_md[timestep % DOUBLE_BUFFERING].data();
        }

        auto * _noalias f = zoid.f_stencil_md[timestep % 1].data();

        auto * _noalias mask = zoid.mask_stencil_md[0].data();
        auto * _noalias local_idxs = zoid.local_idxs_per_timestep[timestep].data();
        auto * _noalias type = zoid.type_stencil_md[0].data();

        int nlocal = zoid.local_idxs_per_timestep[timestep].size();

        double dtv = update->dt;

        const double * const mass = atom->mass;
        double dtf = 0.5 * update->dt * force->ftm2v;

        if (nlocal > MODIFY_GRAINSIZE) {
            int num_workers = __cilkrts_get_nworkers();
            int num_chunks = nlocal / MODIFY_GRAINSIZE + 1;
            int chunks_per_worker = num_chunks / num_workers;
            int chunk_size = MODIFY_GRAINSIZE;
            auto* claimed = zoid.claimed_flags_stencil_md[0];

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

                double rand_x = 0.6;
                double rand_y = 0.6;
                double rand_z = 0.6;

                v[i].x += dtfm * f[i].x;
                v[i].y += dtfm * f[i].y;
                v[i].z += dtfm * f[i].z;
            }
        }
    }

    /**
     * @brief Performs greedy graph coloring on a given graph.
     *
     * Assigns a color (non-negative integer) to each vertex such that no two
     * adjacent vertices share the same color. It uses the smallest available
     * color for each vertex based on its already colored neighbors.
     *
     * @param num_vertices The total number of vertices in the graph (labeled 0 to num_vertices-1).
     * @param adj_list An adjacency list representation of the graph.
     * adj_list[i] contains a vector of vertices adjacent to vertex i.
     * @return A vector where the i-th element is the color assigned to vertex i.
     * Colors are 0-indexed integers. Returns an empty vector if num_vertices is 0.
     */
    std::vector<int> greedy_coloring(int num_vertices, const std::vector<std::vector<int>>& adj_list) {
        assert(num_vertices > 0);

        // Initialize colors for all vertices. -1 indicates uncolored.
        std::vector<int> result_colors(num_vertices, -1);

        // Process vertices one by one (in default order 0, 1, ..., n-1)
        for (int u = 0; u < num_vertices; ++u) {
            // Keep track of colors used by neighbors of u that are already colored
            std::unordered_set<int> neighbor_colors;

            // Iterate through neighbors of the current vertex u
            for (int neighbor : adj_list[u]) {
                // If the neighbor has already been assigned a color
                if (result_colors[neighbor] != -1) {
                    neighbor_colors.insert(result_colors[neighbor]);
                }
            }

            // Find the smallest non-negative integer (color) that is
            // not present in the set of neighbor colors.
            int current_color = 0;
            while (neighbor_colors.count(current_color)) {
                current_color++;
            }

            // Assign the found color to the current vertex u
            result_colors[u] = current_color;
        }

        return result_colors;
    }

    // --- Build Bond Conflict Graph ---
    /**
     * @brief Builds the adjacency list for the bond conflict graph.
     *
     * Nodes in the conflict graph represent bonds. An edge exists if
     * the corresponding bonds share an atom.
     *
     * @param bonds The list of bonds in the molecule.
     * @param num_bonds Total number of bonds.
     * @return Adjacency list for the conflict graph.
     */
    std::vector<std::vector<int>> build_conflict_graph(const std::vector<std::tuple<int, int, int>>& bond_list) {
        int num_bonds = bond_list.size();
        std::vector<std::vector<int>> conflict_adj_list(num_bonds);

        std::vector<std::pair<int, int>> bonds;
        bonds.reserve(num_bonds);
        for (int i = 0; i < num_bonds; i++) {
            int i1 = std::get<0>(bond_list[i]);
            int i2 = std::get<1>(bond_list[i]);
            bonds.push_back(std::make_pair(i1, i2));
        }

        for (int k1 = 0; k1 < num_bonds; ++k1) {
            for (int k2 = k1 + 1; k2 < num_bonds; ++k2) {
                const auto& bond1 = bonds[k1]; // (i1, j1)
                const auto& bond2 = bonds[k2]; // (i2, j2)

                // Check for shared atom
                if (bond1.first == bond2.first || bond1.first == bond2.second ||
                    bond1.second == bond2.first || bond1.second == bond2.second)
                {
                    // Add edge in the conflict graph
                    conflict_adj_list[k1].push_back(k2);
                    conflict_adj_list[k2].push_back(k1);
                }
            }
        }
        return conflict_adj_list;
    }

    using update_t = std::pair<int, dbl3_t_stencil_md>;
    using sparse_updates_t = std::vector<update_t>;

    // Merge two sorted sparse lists into one
    static void merge_sparse(sparse_updates_t &A, sparse_updates_t& B) {
        sparse_updates_t out;
        out.reserve(A.size() + B.size());
        auto ia = A.begin();
        auto ib = B.begin();
        while (ia!=A.end() && ib!=B.end()) {
            if (ia->first < ib->first)            out.push_back(*ia++);
            else if (ib->first < ia->first)       out.push_back(*ib++);
            else {
                // same index → sum
                dbl3_t_stencil_md t{ia->second.x+ib->second.x,
                         ia->second.y+ib->second.y,
                         ia->second.z+ib->second.z};
                out.emplace_back(ia->first, t);
                ++ia; ++ib;
            }
        }
        out.insert(out.end(), ia, A.end());
        out.insert(out.end(), ib, B.end());
        A.swap(out);
    }

    static void sparse_updates_identity(void* view) {
        // *static_cast<sparse_updates_t  **>(view) = new sparse_updates_t();
        // return {};
    }

    static void sparse_updates_reduce(void* left, void* right) {
        merge_sparse(*static_cast<sparse_updates_t*>(left), *static_cast<sparse_updates_t*>(right));
    }

    void BOND_FENE_FORCE_COMPUTE_ZOID_MANY_CUTS(queue_info& zoid, int dep, int timestep) {
        const auto * _noalias const x = zoid.x_stencil_md[timestep % DOUBLE_BUFFERING].data();
        auto * _noalias const f = zoid.f_stencil_md[timestep % 1].data();

        auto pair = (PairLJCut*) force->pair;
        auto bond = (BondFENE*) force->bond;

        // const auto& bond_list = zoid.bond_list[timestep];
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
        // auto newton_pair = force->newton_pair;
        constexpr bool NEWTON_PAIR = true;

        const auto* _noalias const sigma = bond->sigma;
        const auto* _noalias const epsilon = bond->epsilon;
        const auto* _noalias const r0 = bond->r0;
        const auto* _noalias const k = bond->k;

        const auto& atom_type = zoid.type_stencil_md[0];

        const auto& local_idxs = zoid.local_idxs_per_timestep[timestep];
        // const auto& is_local_idx = zoid.is_local_per_timestep[timestep];
        const int nlocal = local_idxs.size();

        int num_chunks = nlocal / MODIFY_GRAINSIZE + 1;
        int num_workers = __cilkrts_get_nworkers();

        auto* claimed = zoid.claimed_flags_stencil_md[0];

        const auto& tags = zoid.tag_stencil_md[0];

        int chunks_per_worker = num_chunks / num_workers;
        int chunk_size = MODIFY_GRAINSIZE;

        // if ((dep == 0 || dep == NUM_DEPS - 1) && nlocal > MODIFY_GRAINSIZE) {
        if (nlocal > 4096) {
            /*
            auto& segment_idxs = zoid.local_idxs_per_timestep_segment_idxs[timestep];
            auto& segment_sizes = zoid.local_idxs_per_timestep_segment_sizes[timestep];

            for (int idx = 0; idx < segment_idxs.size(); idx++) {
                int segment_idx = segment_idxs[idx];
                int segment_size = segment_sizes[idx];
                for (int k = 0; k < segment_size; k++) {
                    int i = segment_idx + k;
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

                            if (NEWTON_PAIR || is_local_idx[j]) {
                                spinlocks[j].lock();
                                f[j].x -= delx * fpair;
                                f[j].y -= dely * fpair;
                                f[j].z -= delz * fpair;
                                spinlocks[j].unlock();
                            }
                        }
                    }

                    spinlocks[i].lock();
                    f[i].x += fxtmp;
                    f[i].y += fytmp;
                    f[i].z += fztmp;
                    spinlocks[i].unlock();
                }
            }
            */

            #pragma cilk grainsize 1024
            cilk_for(int idx = 0; idx < nlocal; idx++) {
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

                        // if (NEWTON_PAIR || is_local_idx[j]) {
                        if (NEWTON_PAIR) {
                            spinlocks[j].lock();
                            f[j].x -= delx * fpair;
                            f[j].y -= dely * fpair;
                            f[j].z -= delz * fpair;
                            spinlocks[j].unlock();
                        }
                    }
                }

                spinlocks[i].lock();
                f[i].x += fxtmp;
                f[i].y += fytmp;
                f[i].z += fztmp;
                spinlocks[i].unlock();
            }
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

                        // if (NEWTON_PAIR || is_local_idx[j]) {
                        if (NEWTON_PAIR) {
                            f[j].x -= delx * fpair;
                            f[j].y -= dely * fpair;
                            f[j].z -= delz * fpair;
                        }
                    }
                }

                f[i].x += fxtmp;
                f[i].y += fytmp;
                f[i].z += fztmp;
            }
        }

        auto& bond_list = zoid.bond_list_modified[timestep];
        int nbonds = bond_list.size();

        if (nbonds > 4096) {
            #pragma cilk grainsize 1024
            cilk_for (int i = 0; i < nbonds; i++) {
                auto& tup = bond_list[i];
                int i1 = std::get<0>(tup);
                int i2 = std::get<1>(tup);
                int type = std::get<2>(tup);

                double delx = x[i1].x - x[i2].x;
                double dely = x[i1].y - x[i2].y;
                double delz = x[i1].z - x[i2].z;

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

                // if (NEWTON_PAIR || is_local_idx[i1]) {
                if (NEWTON_PAIR) {
                    spinlocks[i1].lock();
                    f[i1].x += delx * fbond;
                    f[i1].y += dely * fbond;
                    f[i1].z += delz * fbond;
                    spinlocks[i1].unlock();
                }

                // if (NEWTON_PAIR || is_local_idx[i2]) {
                if (NEWTON_PAIR) {
                    spinlocks[i2].lock();
                    f[i2].x -= delx * fbond;
                    f[i2].y -= dely * fbond;
                    f[i2].z -= delz * fbond;
                    spinlocks[i2].unlock();
                }
            }
        } else {
            for (int i = 0; i < nbonds; i++) {
                auto& tup = bond_list[i];
                int i1 = std::get<0>(tup);
                int i2 = std::get<1>(tup);
                int type = std::get<2>(tup);

                double delx = x[i1].x - x[i2].x;
                double dely = x[i1].y - x[i2].y;
                double delz = x[i1].z - x[i2].z;

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

                // if (NEWTON_PAIR || is_local_idx[i1]) {
                if (NEWTON_PAIR) {
                    f[i1].x += delx * fbond;
                    f[i1].y += dely * fbond;
                    f[i1].z += delz * fbond;
                }

                // if (NEWTON_PAIR || is_local_idx[i2]) {
                if (NEWTON_PAIR) {
                    f[i2].x -= delx * fbond;
                    f[i2].y -= dely * fbond;
                    f[i2].z -= delz * fbond;
                }
            }
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

    void LJ_FORCE_COMPUTE_ZOID_MANY_CUTS(queue_info& zoid, int dep, int timestep) {
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
        constexpr int newton_pair = USE_NEWTON;

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

                spinlocks[i].lock();
                f[i].x += fxtmp;
                f[i].y += fytmp;
                f[i].z += fztmp;
                spinlocks[i].unlock();
            }
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

                f[i].x += fxtmp;
                f[i].y += fytmp;
                f[i].z += fztmp;
            }
        }
    }

    void DPD_FORCE_COMPUTE_ZOID_MANY_CUTS(queue_info& zoid, int dep, int timestep) {
        const auto * _noalias const x = zoid.x_stencil_md[timestep % DOUBLE_BUFFERING].data();
        auto * _noalias const f = zoid.f_stencil_md[timestep % 1].data();
        auto * _noalias const v = zoid.v_stencil_md[timestep % DOUBLE_BUFFERING].data();

        auto pair = (PairDPD*) force->pair;

        const auto& bond_list = zoid.bond_list[timestep];
        const auto& neighbor_list = zoid.neighbor_list[timestep];

        double *special_lj = force->special_lj;
        double dtinvsqrt = 1.0/sqrt(update->dt);

        auto* _noalias spinlocks = zoid.spinlocks_stencil_md[0];

        const auto* cutsq = pair->cutsq;
        auto newton_pair = USE_NEWTON;

        const auto& atom_type = zoid.type_stencil_md[0];

        const auto& local_idxs = zoid.local_idxs_per_timestep[timestep];
        const int nlocal = local_idxs.size();
        auto special_sqrt = pair->special_sqrt;
        auto cut = pair->cut;

        #pragma cilk grainsize MODIFY_GRAINSIZE
        cilk_for (int idx = 0; idx < nlocal; idx++) {
            int i = local_idxs[idx];

            const int itype = atom_type[i];
            const auto &jlist = neighbor_list[i];

            double xtmp = x[i].x;
            double ytmp = x[i].y;
            double ztmp = x[i].z;
            double vxtmp = v[i].x;
            double vytmp = v[i].y;
            double vztmp = v[i].z;
            int jnum = jlist.size();

            double fxtmp = 0.0;
            double fytmp = 0.0;
            double fztmp = 0.0;

            for (int jj = 0; jj < jnum; jj++) {
                int j = jlist[jj];
                double factor_dpd = special_lj[pair->sbmask(j)];
                double factor_sqrt = special_sqrt[pair->sbmask(j)];
                j &= NEIGHMASK;

                double delx = xtmp - x[j].x;
                double dely = ytmp - x[j].y;
                double delz = ztmp - x[j].z;
                double rsq = delx * delx + dely * dely + delz * delz;
                int jtype = atom_type[j];

                if (rsq < cutsq[itype][jtype]) {
                    double r = sqrt(rsq);
                    if (r < EPSILON) continue;     // r can be 0.0 in DPD systems
                    double rinv = 1.0/r;
                    double delvx = vxtmp - v[j].x;
                    double delvy = vytmp - v[j].y;
                    double delvz = vztmp - v[j].z;
                    double dot = delx*delvx + dely*delvy + delz*delvz;
                    double wd = 1.0 - r/cut[itype][jtype];
                    double randnum = 0.6;

                    double fpair = pair->a0[itype][jtype]*wd;
                    fpair -= pair->gamma[itype][jtype]*wd*wd*dot*rinv;
                    fpair *= factor_dpd;
                    fpair += factor_sqrt*pair->sigma[itype][jtype]*wd*randnum*dtinvsqrt;
                    fpair *= rinv;

                    fxtmp += delx*fpair;
                    fytmp += dely*fpair;
                    fztmp += delz*fpair;

                    if (USE_NEWTON) {
                        spinlocks[j].lock();
                        f[j].x -= delx * fpair;
                        f[j].y -= dely * fpair;
                        f[j].z -= delz * fpair;
                        spinlocks[j].unlock();
                    }
                }
            }

            spinlocks[i].lock();
            f[i].x += fxtmp;
            f[i].y += fytmp;
            f[i].z += fztmp;
            spinlocks[i].unlock();
        }
    }

    /* End code for many zoids per dimension */
    void DELETE_ZOID_DATA_MANY_CUTS() {
        for (int dep = 0; dep < NUM_DEPS; dep++) {
            for (int j = 0; j < queues_many_cuts[dep].size(); j++) {
                queue_info &zoid = queues_many_cuts[dep][j];

                delete[] zoid.lo;
                delete[] zoid.hi;

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
                // delete[] zoid.local_idxs_per_timestep_segment_idxs;
                // delete[] zoid.local_idxs_per_timestep_segment_sizes;
                // delete[] zoid.is_local_per_timestep;
                delete[] zoid.neighbor_list;
                delete[] zoid.bond_list;
                delete[] zoid.bond_list_modified;
                // delete[] zoid.bond_list_modified_num_colors;

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

                for (int p = 0; p < NUM_PIPELINE_STAGES; p++) {
                    for (int k = 0; k < DOUBLE_BUFFERING; k++) {
                        delete[] zoid.send_pos_idxs_double_buffering_flattened_pipelined[p][k];
                        delete[] zoid.recv_pos_idxs_double_buffering_flattened_pipelined[p][k];

                        delete[] zoid.send_vel_idxs_double_buffering_flattened_pipelined[p][k];
                        delete[] zoid.recv_vel_idxs_double_buffering_flattened_pipelined[p][k];
                    }

                    delete[] zoid.send_force_idxs_double_buffering_flattened_pipelined[p];
                    delete[] zoid.send_pos_idxs_double_buffering_flattened_pipelined[p];
                    delete[] zoid.send_vel_idxs_double_buffering_flattened_pipelined[p];

                    delete[] zoid.recv_force_idxs_double_buffering_flattened_pipelined[p];
                    delete[] zoid.recv_pos_idxs_double_buffering_flattened_pipelined[p];
                    delete[] zoid.recv_vel_idxs_double_buffering_flattened_pipelined[p];

                }

                delete[] zoid.send_force_idxs_double_buffering_flattened_pipelined;
                delete[] zoid.send_pos_idxs_double_buffering_flattened_pipelined;
                delete[] zoid.send_vel_idxs_double_buffering_flattened_pipelined;

                delete[] zoid.recv_force_idxs_double_buffering_flattened_pipelined;
                delete[] zoid.recv_pos_idxs_double_buffering_flattened_pipelined;
                delete[] zoid.recv_vel_idxs_double_buffering_flattened_pipelined;
            }
        }

        for (int dep = 0; dep < NUM_DEPS; dep++) {
            for (int j = 0; j < queues_many_cuts_next_dt[dep].size(); j++) {
                queue_info &zoid = queues_many_cuts_next_dt[dep][j];

                delete[] zoid.lo;
                delete[] zoid.hi;

                if (zoid.num % comm->nprocs != comm->me) {
                    continue;
                }

                delete[] zoid.local_idxs_per_timestep;
                // delete[] zoid.local_idxs_per_timestep_segment_idxs;
                // delete[] zoid.local_idxs_per_timestep_segment_sizes;
                // delete[] zoid.is_local_per_timestep;
                delete[] zoid.neighbor_list;
                delete[] zoid.bond_list;
                delete[] zoid.bond_list_modified;
                // delete[] zoid.bond_list_modified_num_colors;

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

                for (int p = 0; p < NUM_PIPELINE_STAGES; p++) {
                    for (int k = 0; k < DOUBLE_BUFFERING; k++) {
                        delete[] zoid.send_pos_idxs_double_buffering_flattened_pipelined[p][k];
                        delete[] zoid.recv_pos_idxs_double_buffering_flattened_pipelined[p][k];

                        delete[] zoid.send_vel_idxs_double_buffering_flattened_pipelined[p][k];
                        delete[] zoid.recv_vel_idxs_double_buffering_flattened_pipelined[p][k];
                    }

                    delete[] zoid.send_force_idxs_double_buffering_flattened_pipelined[p];
                    delete[] zoid.send_pos_idxs_double_buffering_flattened_pipelined[p];
                    delete[] zoid.send_vel_idxs_double_buffering_flattened_pipelined[p];

                    delete[] zoid.recv_force_idxs_double_buffering_flattened_pipelined[p];
                    delete[] zoid.recv_pos_idxs_double_buffering_flattened_pipelined[p];
                    delete[] zoid.recv_vel_idxs_double_buffering_flattened_pipelined[p];

                }

                delete[] zoid.send_force_idxs_double_buffering_flattened_pipelined;
                delete[] zoid.send_pos_idxs_double_buffering_flattened_pipelined;
                delete[] zoid.send_vel_idxs_double_buffering_flattened_pipelined;

                delete[] zoid.recv_force_idxs_double_buffering_flattened_pipelined;
                delete[] zoid.recv_pos_idxs_double_buffering_flattened_pipelined;
                delete[] zoid.recv_vel_idxs_double_buffering_flattened_pipelined;
            }
        }
    }

    void CLEANUP_PIPELINED_BUFFERS() {
        for (int p = 0; p < NUM_PIPELINE_STAGES; p++) {
            for (int zoid_num = 0; zoid_num < NUM_ZOIDS_MANY_CUTS ; zoid_num++) {
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
        if (ONLY_RUN_LAMMPS) {
            return;
        }

        if (USE_STREAMS) {
            MPI_Comm_free(&stream_comm);

            for (int i = 0; i < NUM_STREAMS; i++) {
                MPIX_Stream_free(&all_streams[i]);
            }
        }

        for (int i = 0; i < NUM_COMMS; i++) {
            MPI_Comm_free(&all_comms[i]);
        }

        for (int p = 0; p < NUM_PIPELINE_STAGES; p++) {
            for (int dep = 0; dep < NUM_DEPS; dep++) {
                MPI_Comm_free(&proc_to_proc_pipelined_comms[p][dep]);
            }
        }

        for (int p = 0; p < NUM_PIPELINE_STAGES; p++) {
            for (int send_dep = 0; send_dep < NUM_DEPS; send_dep++) {
                for (int proc = 0; proc < comm->nprocs; proc++) {
                    int send_sz = nsend_buf_proc_to_proc[p][send_dep][proc];
                    if (send_sz > 0) {
                        delete[] buf_send_proc_to_proc[p][send_dep][proc];
                    }

                    int recv_sz = nrecv_buf_proc_to_proc[p][send_dep][proc];
                    if (recv_sz > 0) {
                        delete[] buf_recv_proc_to_proc[p][send_dep][proc];
                    }
                }

                delete[] buf_send_proc_to_proc[p][send_dep];
                delete[] buf_recv_proc_to_proc[p][send_dep];
                delete[] nsend_buf_proc_to_proc[p][send_dep];
                delete[] nrecv_buf_proc_to_proc[p][send_dep];
            }
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