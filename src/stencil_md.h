//
// Created by Ryan Deng on 2/27/24.
//

#pragma once

#include "angle.h"
#include "math_special.h"
#include "pointers.h"
#include "bond_fene.h"
#include "pair_lj_cut.h"
#include "pair_lj_cut_omp.h"
#include "pair_dpd.h"
#include "pair_dpd_omp.h"
#include <chrono>
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
#include <string>
#include <thread>
#include "pair_sw.h"
#include "pair_tersoff.h"
#include "pair_eam.h"

#define EPSILON 1.0e-10

constexpr bool USE_BREAK = false;
constexpr bool USE_STREAMS = true;
constexpr int NUM_STREAMS = 24;
constexpr int NUM_PROGRESS_STREAM_ITER = 10;

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

class MPIX_Stream_Manager {
    public:
        std::vector<MPIX_Stream> streams;
        std::vector<MPI_Comm> comms;
        std::vector<spinlock> m;
        MPI_Comm stream_comm;
        int num_streams;
        std::atomic<bool> done;
        std::atomic<bool> start;
        spinlock global_lock;
    
    MPIX_Stream_Manager(int num_streams) : streams(num_streams, MPIX_STREAM_NULL), comms(num_streams, MPI_COMM_NULL), m(num_streams) {
        if (USE_STREAMS) {
            for (int i = 0; i < num_streams; i++) {
                MPIX_Stream_create(MPI_INFO_NULL, &streams[i]);
                // MPIX_Stream_comm_create(MPI_COMM_WORLD, streams[i], &comms[i]);
            }
            MPIX_Stream_comm_create_multiplex(MPI_COMM_WORLD, num_streams, streams.data(), &stream_comm);
        }

        this->num_streams = num_streams;

        done = false;
        start = false;
    }

    ~MPIX_Stream_Manager() {
        if (USE_STREAMS) {
            MPI_Comm_free(&stream_comm);

            for (int i = 0; i < num_streams; i++) {
                // MPI_Comm_free(&comms[i]);
                MPIX_Stream_free(&streams[i]);
            }
        }
    }
};

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
                std::stringstream s1;
                s1 << RED << "ERROR ON FORCE zoid: " << zoid.num << " idx: " << idx << " tag: " << tag
                          << " what I have: " << my_x << " " << my_y << " " << my_z
                          << " what lammps has: " << test_f[tag * 3] << " " << test_f[tag * 3 + 1] << " " << test_f[tag * 3 + 2]
                          << " diff: "
                          << fabs(test_f[tag * 3] - my_x) << " " << fabs(test_f[tag * 3 + 1] - my_y) << " " << fabs(test_f[tag * 3 + 2] - my_z)
                          << " breakdown : " << f[idx].x << " " << f[idx].y << " " << f[idx].z
                          << " eval f: " << eval_f[idx].x << " " << eval_f[idx].y << " " << eval_f[idx].z
                          << " pos: " << x[idx].x << " " << x[idx].y << " " << x[idx].z
                          << RESET_COLOR << std::endl;

                std::cout << s1.str();
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

    static constexpr int NUM_CUTS_X = 6;
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

    std::vector<std::vector<int>> recv_request_idx_to_zoid_per_zoid[2];
    std::vector<std::map<int, int>> recv_request_zoid_to_idx_per_zoid[2];

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
    std::map<std::pair<int, int>, std::pair<int, int>> zoid_to_zoid_to_stream_num[2];
    std::map<std::pair<int, int>, std::pair<int, int>> receiver_dep_proc_to_stream_num[2][NUM_DEPS];
    std::map<std::pair<int, int>, std::pair<int, int>> sender_dep_proc_to_stream_num[2];

    // map curr_dt -> recv_dep -> stream_num -> vector of zoid pairs
    std::vector<std::vector<std::pair<int, int>>> stream_num_to_zoid_pairs[2][NUM_DEPS];
    std::vector<std::vector<std::pair<int, int>>> stream_num_to_dep_proc_pairs[2][NUM_DEPS];
    std::map<std::pair<int, int>, int> zoid_pair_to_recv_request_idx_streams[2];

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
        // Helper function to compute 3D Hilbert curve ordering
        class HilbertCurve3D {
        public:
            static uint64_t xyz2h(int x, int y, int z, int order) {
                uint64_t h = 0;
                for (int i = order - 1; i >= 0; i--) {
                    int xi = (x >> i) & 1;
                    int yi = (y >> i) & 1;
                    int zi = (z >> i) & 1;
                    h = (h << 3) | (xi * 4 + yi * 2 + zi);
                }
                return h;
            }
        };

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

        // Improved INIT_ZOIDS_NUMBERING_BALANCED implementation
        void INIT_ZOIDS_NUMBERING_BALANCED_IMPROVED() {
            // Build neighbor relationships (same as original)
            std::map<std::array<int, 3>, std::set<std::array<int, 3>>> tmp_send_neighbors;
            std::map<std::array<int, 3>, std::set<std::array<int, 3>>> tmp_recv_neighbors;
            
            // Build communication graph with periodic boundaries
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
            
            // Group zoids by dependency level
            std::vector<std::vector<std::array<int, 3>>> zoids_by_dep(NUM_DEPS);
            for (int i = 0; i < NUM_ZOIDS_X; i++) {
                for (int j = 0; j < NUM_ZOIDS_Y; j++) {
                    for (int k = 0; k < NUM_ZOIDS_Z; k++) {
                        int dep = (i % 2 == 0) + (j % 2 == 0) + (k % 2 == 0);
                        zoids_by_dep[dep].push_back({i, j, k});
                    }
                }
            }
            
            // Verify we can balance perfectly
            assert(NUM_ZOIDS_MANY_CUTS % comm->nprocs == 0);
            const int zoids_per_proc = NUM_ZOIDS_MANY_CUTS / comm->nprocs;
            
            // Calculate expected zoids per dependency level per processor
            std::vector<int> expected_per_dep(NUM_DEPS);
            for (int dep = 0; dep < NUM_DEPS; dep++) {
                assert(zoids_by_dep[dep].size() % comm->nprocs == 0);
                expected_per_dep[dep] = zoids_by_dep[dep].size() / comm->nprocs;
            }
            
            struct Assignment {
                std::map<int, std::vector<std::array<int, 3>>> proc_to_zoids;
                std::map<std::array<int, 3>, int> zoid_to_proc;
                std::vector<std::vector<std::array<int, 3>>> proc_dep_zoids; // [proc][dep] -> zoids
                int edge_cuts;
                int max_comm_volume;
                std::string method_name;
            };
            
            std::vector<Assignment> assignments;
            
            // METHOD 1: Improved Morton Curve with dependency awareness
            {
                Assignment morton_assignment;
                morton_assignment.method_name = "Dependency-Aware Morton";
                morton_assignment.proc_dep_zoids.resize(comm->nprocs, std::vector<std::array<int, 3>>(NUM_DEPS));
                
                // Sort each dependency level by Morton code
                for (int dep = 0; dep < NUM_DEPS; dep++) {
                    std::vector<std::pair<uint64_t, std::array<int, 3>>> morton_zoids;
                    for (const auto& zoid : zoids_by_dep[dep]) {
                        uint64_t morton_code = morton3D(zoid[0], zoid[1], zoid[2]);
                        morton_zoids.push_back({morton_code, zoid});
                    }
                    std::sort(morton_zoids.begin(), morton_zoids.end());
                    
                    // Assign to processors maintaining balance
                    for (int i = 0; i < morton_zoids.size(); i++) {
                        int proc = i / expected_per_dep[dep];
                        morton_assignment.proc_to_zoids[proc].push_back(morton_zoids[i].second);
                        morton_assignment.zoid_to_proc[morton_zoids[i].second] = proc;
                    }
                }
                
                assignments.push_back(morton_assignment);
            }
            
            // METHOD 2: Hilbert Curve ordering
            {
                Assignment hilbert_assignment;
                hilbert_assignment.method_name = "Dependency-Aware Hilbert";
                hilbert_assignment.proc_dep_zoids.resize(comm->nprocs, std::vector<std::array<int, 3>>(NUM_DEPS));
                
                int order = std::ceil(std::log2(std::max({NUM_ZOIDS_X, NUM_ZOIDS_Y, NUM_ZOIDS_Z})));
                
                for (int dep = 0; dep < NUM_DEPS; dep++) {
                    std::vector<std::pair<uint64_t, std::array<int, 3>>> hilbert_zoids;
                    for (const auto& zoid : zoids_by_dep[dep]) {
                        uint64_t hilbert_code = HilbertCurve3D::xyz2h(zoid[0], zoid[1], zoid[2], order);
                        hilbert_zoids.push_back({hilbert_code, zoid});
                    }
                    std::sort(hilbert_zoids.begin(), hilbert_zoids.end());
                    
                    for (int i = 0; i < hilbert_zoids.size(); i++) {
                        int proc = i / expected_per_dep[dep];
                        hilbert_assignment.proc_to_zoids[proc].push_back(hilbert_zoids[i].second);
                        hilbert_assignment.zoid_to_proc[hilbert_zoids[i].second] = proc;
                    }
                }
                
                assignments.push_back(hilbert_assignment);
            }
            
            // METHOD 3: 3D Block Decomposition
            {
                Assignment block_assignment;
                block_assignment.method_name = "3D Block Decomposition";
                block_assignment.proc_dep_zoids.resize(comm->nprocs, std::vector<std::array<int, 3>>(NUM_DEPS));
                
                // Find optimal 3D processor grid
                int px, py, pz;
                findOptimal3DDecomposition(NUM_ZOIDS_X, NUM_ZOIDS_Y, NUM_ZOIDS_Z, comm->nprocs, px, py, pz);
                
                // For each dependency level, assign zoids to processor blocks
                for (int dep = 0; dep < NUM_DEPS; dep++) {
                    // Create a temporary assignment for this dep level
                    std::map<int, std::vector<std::array<int, 3>>> dep_proc_zoids;
                    
                    for (const auto& zoid : zoids_by_dep[dep]) {
                        // Map zoid to processor block
                        int proc_x = (zoid[0] * px) / NUM_ZOIDS_X;
                        int proc_y = (zoid[1] * py) / NUM_ZOIDS_Y;
                        int proc_z = (zoid[2] * pz) / NUM_ZOIDS_Z;
                        int proc = proc_x + proc_y * px + proc_z * px * py;
                        
                        dep_proc_zoids[proc].push_back(zoid);
                    }
                    
                    // Rebalance if needed to ensure exact balance
                    rebalanceDepLevel(dep_proc_zoids, expected_per_dep[dep], tmp_send_neighbors);
                    
                    // Add to main assignment
                    for (int proc = 0; proc < comm->nprocs; proc++) {
                        for (const auto& zoid : dep_proc_zoids[proc]) {
                            block_assignment.proc_to_zoids[proc].push_back(zoid);
                            block_assignment.zoid_to_proc[zoid] = proc;
                        }
                    }
                }
                
                assignments.push_back(block_assignment);
            }
            
            // METHOD 4: Graph Partitioning with METIS-like approach
            /*
            {
                Assignment graph_assignment;
                graph_assignment.method_name = "Graph Partitioning";
                graph_assignment.proc_dep_zoids.resize(comm->nprocs, std::vector<std::array<int, 3>>(NUM_DEPS));
                
                // For each dependency level, use a greedy graph partitioning
                for (int dep = 0; dep < NUM_DEPS; dep++) {
                    greedyGraphPartition(zoids_by_dep[dep], expected_per_dep[dep], 
                                    tmp_send_neighbors, tmp_recv_neighbors,
                                    graph_assignment.proc_to_zoids, 
                                    graph_assignment.zoid_to_proc);
                }
                
                assignments.push_back(graph_assignment);
            }
            */
            
            // Optimize each assignment with local search
            for (auto& assignment : assignments) {
                if (comm->me == 0) {
                    std::cout << "\nOptimizing " << assignment.method_name << " assignment..." << std::endl;
                }
                
                // Use dependency-aware optimization
                optimizeAssignmentDepAware(assignment.proc_to_zoids, assignment.zoid_to_proc,
                                        tmp_send_neighbors, tmp_recv_neighbors, 
                                        expected_per_dep, 50);
                
                // Calculate metrics
                assignment.edge_cuts = countTotalEdgeCuts(assignment.zoid_to_proc, tmp_send_neighbors);
                assignment.max_comm_volume = calculateMaxCommVolume(assignment.proc_to_zoids, 
                                                                assignment.zoid_to_proc,
                                                                tmp_send_neighbors, tmp_recv_neighbors);
                
                if (comm->me == 0) {
                    std::cout << "  Edge cuts: " << assignment.edge_cuts 
                            << ", Max comm volume: " << assignment.max_comm_volume << std::endl;
                }
            }
            
            // Select best assignment based on multiple criteria
            Assignment* best_assignment = &assignments[0];
            double best_score = std::numeric_limits<double>::max();
            
            for (auto& assignment : assignments) {
                // Weighted score: prioritize edge cuts but also consider comm volume balance
                double score = assignment.edge_cuts + 0.1 * assignment.max_comm_volume;
                if (score < best_score) {
                    best_score = score;
                    best_assignment = &assignment;
                }
            }
            
            if (comm->me == 0) {
                std::cout << "\n=== BEST ASSIGNMENT: " << best_assignment->method_name 
                        << " (score: " << best_score << ") ===" << std::endl;
            }
            
            // Use the best assignment
            std::map<int, std::vector<std::array<int, 3>>> proc_to_zoids = best_assignment->proc_to_zoids;
            std::map<std::array<int, 3>, int> zoid_to_proc = best_assignment->zoid_to_proc;
            
            // OPTIONAL: Reorder processors to group communicating ones close together
            // Set this to false if performance degrades
            constexpr bool ENABLE_PROCESSOR_REORDERING = false;
            
            if (ENABLE_PROCESSOR_REORDERING) {
                // Build processor connectivity graph based on zoid neighbors
                std::vector<std::set<int>> proc_neighbors(comm->nprocs);
                std::vector<std::map<int, int>> proc_comm_weight(comm->nprocs);
            
                // Count communications between each processor pair (including periodic boundaries)
                for (int proc = 0; proc < comm->nprocs; proc++) {
                    for (const auto& zoid : proc_to_zoids[proc]) {
                        // Check send neighbors
                        if (tmp_send_neighbors.count(zoid)) {
                            for (const auto& neighbor : tmp_send_neighbors.at(zoid)) {
                                if (zoid_to_proc.count(neighbor)) {
                                    int neighbor_proc = zoid_to_proc.at(neighbor);
                                    if (neighbor_proc != proc) {
                                        proc_neighbors[proc].insert(neighbor_proc);
                                        proc_comm_weight[proc][neighbor_proc]++;
                                    }
                                }
                            }
                        }
                        // Check receive neighbors
                        if (tmp_recv_neighbors.count(zoid)) {
                            for (const auto& neighbor : tmp_recv_neighbors.at(zoid)) {
                                if (zoid_to_proc.count(neighbor)) {
                                    int neighbor_proc = zoid_to_proc.at(neighbor);
                                    if (neighbor_proc != proc) {
                                        proc_neighbors[proc].insert(neighbor_proc);
                                        proc_comm_weight[proc][neighbor_proc]++;
                                    }
                                }
                            }
                        }
                    }
                }
            
                // Reorder processors using breadth-first traversal to keep communicating procs close
                std::vector<int> old_to_new_proc(comm->nprocs, -1);
                std::vector<bool> proc_assigned(comm->nprocs, false);
                int next_new_proc = 0;
                
                // Start with processor that has the most communications
                int start_proc = 0;
                int max_comms = 0;
                for (int proc = 0; proc < comm->nprocs; proc++) {
                    int total_comms = 0;
                    for (const auto& [other_proc, weight] : proc_comm_weight[proc]) {
                        total_comms += weight;
                    }
                    if (total_comms > max_comms) {
                        max_comms = total_comms;
                        start_proc = proc;
                    }
                }
            
                // BFS traversal to assign new processor numbers
                std::queue<int> to_visit;
                to_visit.push(start_proc);
                proc_assigned[start_proc] = true;
                old_to_new_proc[start_proc] = next_new_proc++;
                
                while (!to_visit.empty() && next_new_proc < comm->nprocs) {
                    int current_proc = to_visit.front();
                    to_visit.pop();
                    
                    // Sort neighbors by communication weight (heaviest first)
                    std::vector<std::pair<int, int>> weighted_neighbors;
                    for (int neighbor : proc_neighbors[current_proc]) {
                        if (!proc_assigned[neighbor]) {
                            int weight = proc_comm_weight[current_proc][neighbor];
                            weighted_neighbors.push_back({weight, neighbor});
                        }
                    }
                    std::sort(weighted_neighbors.begin(), weighted_neighbors.end(), std::greater<>());
                    
                    // Assign new processor numbers to neighbors in order of communication weight
                    for (const auto& [weight, neighbor] : weighted_neighbors) {
                        if (!proc_assigned[neighbor]) {
                            proc_assigned[neighbor] = true;
                            old_to_new_proc[neighbor] = next_new_proc++;
                            to_visit.push(neighbor);
                        }
                    }
                }
                
                // Assign any remaining processors that have no communications
                for (int proc = 0; proc < comm->nprocs; proc++) {
                    if (!proc_assigned[proc]) {
                        old_to_new_proc[proc] = next_new_proc++;
                    }
                }
                
                // Apply the processor reordering
                std::map<int, std::vector<std::array<int, 3>>> reordered_proc_to_zoids;
                for (int old_proc = 0; old_proc < comm->nprocs; old_proc++) {
                    int new_proc = old_to_new_proc[old_proc];
                    reordered_proc_to_zoids[new_proc] = proc_to_zoids[old_proc];
                }
                proc_to_zoids = reordered_proc_to_zoids;
                
                // Update zoid_to_proc mapping
                for (int new_proc = 0; new_proc < comm->nprocs; new_proc++) {
                    for (const auto& zoid : proc_to_zoids[new_proc]) {
                        zoid_to_proc[zoid] = new_proc;
                    }
                }
                
                if (comm->me == 0) {
                    std::cout << "Processor reordering completed to minimize communication distance." << std::endl;
                    
                    // Print processor neighbor statistics
                    std::vector<int> proc_neighbor_counts(comm->nprocs);
                    for (int proc = 0; proc < comm->nprocs; proc++) {
                        proc_neighbor_counts[proc] = proc_neighbors[proc].size();
                    }
                    auto [min_neighbors, max_neighbors] = std::minmax_element(proc_neighbor_counts.begin(), proc_neighbor_counts.end());
                    std::cout << "Processor communication partners: min=" << *min_neighbors 
                             << ", max=" << *max_neighbors << std::endl;
                }
            } // End of ENABLE_PROCESSOR_REORDERING
            
            // Assign global numbering maintaining round-robin across processors
            for (int proc = 0; proc < comm->nprocs; proc++) {
                auto& zoids = proc_to_zoids[proc];
                
                // Sort by dependency level then by space-filling curve
                std::sort(zoids.begin(), zoids.end(), [&](const auto& a, const auto& b) {
                    int dep_a = (a[0] % 2 == 0) + (a[1] % 2 == 0) + (a[2] % 2 == 0);
                    int dep_b = (b[0] % 2 == 0) + (b[1] % 2 == 0) + (b[2] % 2 == 0);
                    if (dep_a != dep_b) return dep_a < dep_b;
                    
                    return morton3D(a[0], a[1], a[2]) < morton3D(b[0], b[1], b[2]);
                });
                
                for (int i = 0; i < zoids.size(); i++) {
                    zoid_where_to_num[zoids[i]] = i * comm->nprocs + proc;
                }
            }
            
            // Verify balance
            verifyDependencyBalance(proc_to_zoids, expected_per_dep);
            
            // Print statistics
            if (comm->me == 0) {
                printDetailedAssignmentStats(proc_to_zoids, zoid_to_proc, 
                                        tmp_send_neighbors, tmp_recv_neighbors,
                                        expected_per_dep);
            }
        }

        // Helper function: Find optimal 3D decomposition
        void findOptimal3DDecomposition(int nx, int ny, int nz, int nprocs,
                                    int& px, int& py, int& pz) {
            double best_score = std::numeric_limits<double>::max();
            
            for (int i = 1; i <= nprocs; i++) {
                if (nprocs % i == 0) {
                    int remaining = nprocs / i;
                    for (int j = 1; j <= remaining; j++) {
                        if (remaining % j == 0) {
                            int k = remaining / j;
                            
                            // Score based on aspect ratio and communication volume
                            double aspect_ratio = std::max({(double)i/j, (double)j/k, (double)k/i});
                            double comm_volume = 2.0 * (ny*nz/j/k + nx*nz/i/k + nx*ny/i/j);
                            double score = aspect_ratio * comm_volume;
                            
                            if (score < best_score) {
                                best_score = score;
                                px = i; py = j; pz = k;
                            }
                        }
                    }
                }
            }
        }

        // Helper function: Rebalance a dependency level
        void rebalanceDepLevel(std::map<int, std::vector<std::array<int, 3>>>& proc_zoids,
                            int target_per_proc,
                            const std::map<std::array<int, 3>, std::set<std::array<int, 3>>>& send_neighbors) {
            // Find overloaded and underloaded processors
            std::vector<int> overloaded, underloaded;
            
            for (int proc = 0; proc < comm->nprocs; proc++) {
                int load = proc_zoids[proc].size();
                if (load > target_per_proc) {
                    overloaded.push_back(proc);
                } else if (load < target_per_proc) {
                    underloaded.push_back(proc);
                }
            }
            
            // Move zoids from overloaded to underloaded processors
            for (int over : overloaded) {
                while (proc_zoids[over].size() > target_per_proc && !underloaded.empty()) {
                    // Find best zoid to move (minimize edge cuts)
                    int best_idx = -1;
                    int best_under = -1;
                    int min_cut_increase = std::numeric_limits<int>::max();
                    
                    for (int idx = 0; idx < proc_zoids[over].size(); idx++) {
                        const auto& zoid = proc_zoids[over][idx];
                        
                        for (int under : underloaded) {
                            if (proc_zoids[under].size() >= target_per_proc) continue;
                            
                            // Calculate cut increase if we move this zoid
                            int cut_increase = 0;
                            if (send_neighbors.count(zoid)) {
                                for (const auto& neighbor : send_neighbors.at(zoid)) {
                                    // Check if neighbor is in the same processor
                                    bool in_over = std::find(proc_zoids[over].begin(), 
                                                        proc_zoids[over].end(), 
                                                        neighbor) != proc_zoids[over].end();
                                    bool in_under = std::find(proc_zoids[under].begin(), 
                                                            proc_zoids[under].end(), 
                                                            neighbor) != proc_zoids[under].end();
                                    
                                    if (in_over) cut_increase++;
                                    if (in_under) cut_increase--;
                                }
                            }
                            
                            if (cut_increase < min_cut_increase) {
                                min_cut_increase = cut_increase;
                                best_idx = idx;
                                best_under = under;
                            }
                        }
                    }
                    
                    // Move the best zoid
                    if (best_idx != -1) {
                        auto zoid = proc_zoids[over][best_idx];
                        proc_zoids[over].erase(proc_zoids[over].begin() + best_idx);
                        proc_zoids[best_under].push_back(zoid);
                        
                        // Update underloaded list
                        if (proc_zoids[best_under].size() >= target_per_proc) {
                            underloaded.erase(std::remove(underloaded.begin(), 
                                                        underloaded.end(), 
                                                        best_under), 
                                            underloaded.end());
                        }
                    } else {
                        break; // No good move found
                    }
                }
            }
        }

        // Helper function: Greedy graph partitioning
        void greedyGraphPartition(const std::vector<std::array<int, 3>>& zoids,
                                int target_per_proc,
                                const std::map<std::array<int, 3>, std::set<std::array<int, 3>>>& send_neighbors,
                                const std::map<std::array<int, 3>, std::set<std::array<int, 3>>>& recv_neighbors,
                                std::map<int, std::vector<std::array<int, 3>>>& proc_to_zoids,
                                std::map<std::array<int, 3>, int>& zoid_to_proc) {
            
            std::vector<bool> assigned(zoids.size(), false);
            std::vector<std::vector<int>> proc_zoid_indices(comm->nprocs);
            
            // Start with seed zoids distributed across processors
            for (int proc = 0; proc < comm->nprocs; proc++) {
                if (proc < zoids.size()) {
                    // Pick seeds that are far apart
                    int seed_idx = (proc * zoids.size()) / comm->nprocs;
                    proc_zoid_indices[proc].push_back(seed_idx);
                    assigned[seed_idx] = true;
                }
            }
            
            // Grow regions greedily
            while (std::accumulate(assigned.begin(), assigned.end(), 0) < zoids.size()) {
                for (int proc = 0; proc < comm->nprocs; proc++) {
                    if (proc_zoid_indices[proc].size() >= target_per_proc) continue;
                    
                    // Find best unassigned zoid to add to this processor
                    int best_idx = -1;
                    int max_connections = -1;
                    
                    for (int i = 0; i < zoids.size(); i++) {
                        if (assigned[i]) continue;
                        
                        // Count connections to zoids already in this processor
                        int connections = 0;
                        const auto& zoid = zoids[i];
                        
                        for (int j : proc_zoid_indices[proc]) {
                            const auto& proc_zoid = zoids[j];
                            
                            // Check if they're neighbors
                            if (send_neighbors.count(zoid) && 
                                send_neighbors.at(zoid).count(proc_zoid)) {
                                connections++;
                            }
                            if (recv_neighbors.count(zoid) && 
                                recv_neighbors.at(zoid).count(proc_zoid)) {
                                connections++;
                            }
                        }
                        
                        if (connections > max_connections) {
                            max_connections = connections;
                            best_idx = i;
                        }
                    }
                    
                    if (best_idx != -1) {
                        proc_zoid_indices[proc].push_back(best_idx);
                        assigned[best_idx] = true;
                    }
                }
            }
            
            // Convert to required format
            for (int proc = 0; proc < comm->nprocs; proc++) {
                for (int idx : proc_zoid_indices[proc]) {
                    proc_to_zoids[proc].push_back(zoids[idx]);
                    zoid_to_proc[zoids[idx]] = proc;
                }
            }
        }

        // Helper function: Optimize assignment with dependency awareness
        void optimizeAssignmentDepAware(std::map<int, std::vector<std::array<int, 3>>>& proc_to_zoids,
                                    std::map<std::array<int, 3>, int>& zoid_to_proc,
                                    const std::map<std::array<int, 3>, std::set<std::array<int, 3>>>& send_neighbors,
                                    const std::map<std::array<int, 3>, std::set<std::array<int, 3>>>& recv_neighbors,
                                    const std::vector<int>& expected_per_dep,
                                    int max_iterations) {
            
            for (int iter = 0; iter < max_iterations; iter++) {
                bool improved = false;
                
                // Try swapping zoids between processor pairs
                for (int p1 = 0; p1 < proc_to_zoids.size(); p1++) {
                    for (int p2 = p1 + 1; p2 < proc_to_zoids.size(); p2++) {
                        
                        // Find best swap that maintains dependency balance
                        int best_improvement = 0;
                        int best_i1 = -1, best_i2 = -1;
                        
                        for (int i1 = 0; i1 < proc_to_zoids[p1].size(); i1++) {
                            for (int i2 = 0; i2 < proc_to_zoids[p2].size(); i2++) {
                                auto& z1 = proc_to_zoids[p1][i1];
                                auto& z2 = proc_to_zoids[p2][i2];
                                
                                // Check if swap maintains dependency balance
                                int dep1 = (z1[0] % 2 == 0) + (z1[1] % 2 == 0) + (z1[2] % 2 == 0);
                                int dep2 = (z2[0] % 2 == 0) + (z2[1] % 2 == 0) + (z2[2] % 2 == 0);
                                
                                if (dep1 != dep2) continue; // Only swap within same dep level
                                
                                // Calculate edge cut improvement
                                int current_cuts = 0, new_cuts = 0;
                                
                                // Count current cuts for z1
                                if (send_neighbors.count(z1)) {
                                    for (const auto& neighbor : send_neighbors.at(z1)) {
                                        if (zoid_to_proc.count(neighbor) && 
                                            zoid_to_proc.at(neighbor) != p1) {
                                            current_cuts++;
                                        }
                                        if (zoid_to_proc.count(neighbor) && 
                                            zoid_to_proc.at(neighbor) != p2) {
                                            new_cuts++;
                                        }
                                    }
                                }
                                
                                // Count current cuts for z2
                                if (send_neighbors.count(z2)) {
                                    for (const auto& neighbor : send_neighbors.at(z2)) {
                                        if (zoid_to_proc.count(neighbor) && 
                                            zoid_to_proc.at(neighbor) != p2) {
                                            current_cuts++;
                                        }
                                        if (zoid_to_proc.count(neighbor) && 
                                            zoid_to_proc.at(neighbor) != p1) {
                                            new_cuts++;
                                        }
                                    }
                                }
                                
                                int improvement = current_cuts - new_cuts;
                                if (improvement > best_improvement) {
                                    best_improvement = improvement;
                                    best_i1 = i1;
                                    best_i2 = i2;
                                }
                            }
                        }
                        
                        // Perform best swap
                        if (best_improvement > 0) {
                            auto z1 = proc_to_zoids[p1][best_i1];
                            auto z2 = proc_to_zoids[p2][best_i2];
                            
                            proc_to_zoids[p1][best_i1] = z2;
                            proc_to_zoids[p2][best_i2] = z1;
                            
                            zoid_to_proc[z1] = p2;
                            zoid_to_proc[z2] = p1;
                            
                            improved = true;
                        }
                    }
                }
                
                if (!improved) break;
            }
        }

        // Helper function: Calculate maximum communication volume
        int calculateMaxCommVolume(const std::map<int, std::vector<std::array<int, 3>>>& proc_to_zoids,
                                const std::map<std::array<int, 3>, int>& zoid_to_proc,
                                const std::map<std::array<int, 3>, std::set<std::array<int, 3>>>& send_neighbors,
                                const std::map<std::array<int, 3>, std::set<std::array<int, 3>>>& recv_neighbors) {
            
            std::vector<int> comm_volumes(proc_to_zoids.size(), 0);
            
            for (int proc = 0; proc < proc_to_zoids.size(); proc++) {
                std::set<int> comm_partners;
                
                for (const auto& zoid : proc_to_zoids.at(proc)) {
                    // Count outgoing communications
                    if (send_neighbors.count(zoid)) {
                        for (const auto& neighbor : send_neighbors.at(zoid)) {
                            if (zoid_to_proc.count(neighbor)) {
                                int neighbor_proc = zoid_to_proc.at(neighbor);
                                if (neighbor_proc != proc) {
                                    comm_partners.insert(neighbor_proc);
                                }
                            }
                        }
                    }
                    
                    // Count incoming communications
                    if (recv_neighbors.count(zoid)) {
                        for (const auto& neighbor : recv_neighbors.at(zoid)) {
                            if (zoid_to_proc.count(neighbor)) {
                                int neighbor_proc = zoid_to_proc.at(neighbor);
                                if (neighbor_proc != proc) {
                                    comm_partners.insert(neighbor_proc);
                                }
                            }
                        }
                    }
                }
                
                comm_volumes[proc] = comm_partners.size();
            }
            
            return *std::max_element(comm_volumes.begin(), comm_volumes.end());
        }

        // Helper function: Verify dependency balance
        void verifyDependencyBalance(const std::map<int, std::vector<std::array<int, 3>>>& proc_to_zoids,
                                    const std::vector<int>& expected_per_dep) {
            std::vector<std::vector<int>> proc_dep_count(comm->nprocs, std::vector<int>(NUM_DEPS, 0));
            
            for (int proc = 0; proc < comm->nprocs; proc++) {
                for (const auto& zoid : proc_to_zoids.at(proc)) {
                    int dep = (zoid[0] % 2 == 0) + (zoid[1] % 2 == 0) + (zoid[2] % 2 == 0);
                    proc_dep_count[proc][dep]++;
                }
            }
            
            for (int proc = 0; proc < comm->nprocs; proc++) {
                for (int dep = 0; dep < NUM_DEPS; dep++) {
                    if (proc_dep_count[proc][dep] != expected_per_dep[dep]) {
                        std::cout << BOLDRED << "ERROR: Processor " << proc 
                                << " dep " << dep << " has " << proc_dep_count[proc][dep]
                                << " zoids, expected " << expected_per_dep[dep] 
                                << RESET_COLOR << std::endl;
                        MPI_Abort(world, 1);
                    }
                }
            }
        }

        // Helper function: Print detailed statistics
        void printDetailedAssignmentStats(const std::map<int, std::vector<std::array<int, 3>>>& proc_to_zoids,
                                        const std::map<std::array<int, 3>, int>& zoid_to_proc,
                                        const std::map<std::array<int, 3>, std::set<std::array<int, 3>>>& send_neighbors,
                                        const std::map<std::array<int, 3>, std::set<std::array<int, 3>>>& recv_neighbors,
                                        const std::vector<int>& expected_per_dep) {
            
            std::cout << "\n=== Detailed Assignment Statistics ===" << std::endl;
            
            // Edge cut analysis
            int total_edge_cuts = countTotalEdgeCuts(zoid_to_proc, send_neighbors);
            std::cout << "Total edge cuts: " << total_edge_cuts << std::endl;
            
            // Communication volume analysis
            std::vector<int> send_volumes(comm->nprocs, 0);
            std::vector<int> recv_volumes(comm->nprocs, 0);
            std::vector<std::set<int>> comm_partners(comm->nprocs);
            
            for (int proc = 0; proc < comm->nprocs; proc++) {
                for (const auto& zoid : proc_to_zoids.at(proc)) {
                    if (send_neighbors.count(zoid)) {
                        for (const auto& neighbor : send_neighbors.at(zoid)) {
                            if (zoid_to_proc.count(neighbor)) {
                                int neighbor_proc = zoid_to_proc.at(neighbor);
                                if (neighbor_proc != proc) {
                                    send_volumes[proc]++;
                                    comm_partners[proc].insert(neighbor_proc);
                                }
                            }
                        }
                    }
                    
                    if (recv_neighbors.count(zoid)) {
                        for (const auto& neighbor : recv_neighbors.at(zoid)) {
                            if (zoid_to_proc.count(neighbor)) {
                                int neighbor_proc = zoid_to_proc.at(neighbor);
                                if (neighbor_proc != proc) {
                                    recv_volumes[proc]++;
                                }
                            }
                        }
                    }
                }
            }
            
            // Print communication statistics
            std::cout << "\nCommunication volume per processor:" << std::endl;
            for (int proc = 0; proc < std::min(8, (int)comm->nprocs); proc++) {
                std::cout << "  Proc " << proc << ": " 
                        << send_volumes[proc] << " sends, "
                        << recv_volumes[proc] << " recvs, "
                        << comm_partners[proc].size() << " partners" << std::endl;
            }
            if (comm->nprocs > 8) {
                std::cout << "  ... (showing first 8 processors)" << std::endl;
            }
            
            // Print min/max statistics
            auto [min_send, max_send] = std::minmax_element(send_volumes.begin(), send_volumes.end());
            auto [min_recv, max_recv] = std::minmax_element(recv_volumes.begin(), recv_volumes.end());
            
            std::cout << "\nCommunication balance:" << std::endl;
            std::cout << "  Send volume: min=" << *min_send << ", max=" << *max_send 
                    << ", imbalance=" << (double)*max_send / *min_send << std::endl;
            std::cout << "  Recv volume: min=" << *min_recv << ", max=" << *max_recv 
                    << ", imbalance=" << (double)*max_recv / *min_recv << std::endl;
            
            // Dependency level distribution
            std::cout << "\nDependency level distribution:" << std::endl;
            std::vector<std::vector<int>> dep_distribution(NUM_DEPS, std::vector<int>(comm->nprocs, 0));
            
            for (int proc = 0; proc < comm->nprocs; proc++) {
                for (const auto& zoid : proc_to_zoids.at(proc)) {
                    int dep = (zoid[0] % 2 == 0) + (zoid[1] % 2 == 0) + (zoid[2] % 2 == 0);
                    dep_distribution[dep][proc]++;
                }
            }
            
            for (int dep = 0; dep < NUM_DEPS; dep++) {
                std::cout << "  Dep " << dep << ": expected=" << expected_per_dep[dep] 
                        << ", actual=" << dep_distribution[dep][0] << " (all procs should match)" << std::endl;
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

                    int nworkers = __cilkrts_get_nworkers();
                    zoid.per_worker_force_updates = new dbl3_t_stencil_md*[nworkers];
                    zoid.global_to_local_idx = new std::vector<int>[NUM_TIMESTEPS_IN_PARALLEL + 1];
                    zoid.local_to_global_idx = new std::vector<int>[NUM_TIMESTEPS_IN_PARALLEL + 1];

                    zoid.rho_stencil_md = new std::vector<double>[DOUBLE_BUFFERING];
                    zoid.fp_stencil_md = new std::vector<double>[DOUBLE_BUFFERING];
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
                    zoid.local_and_one_hop_ghost_idxs_per_timestep = new std::vector<int>[NUM_TIMESTEPS_IN_PARALLEL + 1];
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

                    if constexpr (EXPERIMENT == EAM) {
                        zoid.send_fp_idxs_double_buffering = new std::vector<int>*[NUM_TIMESTEPS_IN_PARALLEL + 1];
                        zoid.recv_fp_idxs_double_buffering = new std::vector<int>*[NUM_TIMESTEPS_IN_PARALLEL + 1];
                        zoid.send_rho_idxs_double_buffering = new std::vector<int>*[NUM_TIMESTEPS_IN_PARALLEL + 1];
                        zoid.recv_rho_idxs_double_buffering = new std::vector<int>*[NUM_TIMESTEPS_IN_PARALLEL + 1];
                    }

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

                    zoid.neigh_short = new std::vector<std::vector<int>>[1];
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

                    zoid.per_worker_force_updates = queues_many_cuts[coord.first][coord.second].per_worker_force_updates;
                    zoid.global_to_local_idx = new std::vector<int>[NUM_TIMESTEPS_IN_PARALLEL + 1];
                    zoid.local_to_global_idx = new std::vector<int>[NUM_TIMESTEPS_IN_PARALLEL + 1];

                    zoid.rho_stencil_md = queues_many_cuts[coord.first][coord.second].rho_stencil_md;
                    zoid.fp_stencil_md = queues_many_cuts[coord.first][coord.second].fp_stencil_md;

                    zoid.neigh_short = queues_many_cuts[coord.first][coord.second].neigh_short;

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
                    zoid.local_and_one_hop_ghost_idxs_per_timestep = new std::vector<int>[NUM_TIMESTEPS_IN_PARALLEL + 1];

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

                    if constexpr (EXPERIMENT == EAM) {
                        zoid.send_fp_idxs_double_buffering = new std::vector<int>*[NUM_TIMESTEPS_IN_PARALLEL + 1];
                        zoid.recv_fp_idxs_double_buffering = new std::vector<int>*[NUM_TIMESTEPS_IN_PARALLEL + 1];
                        zoid.send_rho_idxs_double_buffering = new std::vector<int>*[NUM_TIMESTEPS_IN_PARALLEL + 1];
                        zoid.recv_rho_idxs_double_buffering = new std::vector<int>*[NUM_TIMESTEPS_IN_PARALLEL + 1];
                    }

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

                std::cout << "dep: " << dep << " zoid: " << zoid.num << " num diff proc: " << num_diff_proc << std::endl;
            }

            std::cout << "total num diff proc: " << total_num_diff_proc << std::endl;
        }
    }

    template <bool curr_dt>
    void INIT_MPIX_STREAM_DATA() {
        constexpr int curr_dt_idx = static_cast<int>(curr_dt);
        auto& my_queues = curr_dt ? my_queues_many_cuts : my_queues_many_cuts_next_dt;

        zoid_to_stream_num[curr_dt_idx].resize(NUM_ZOIDS_MANY_CUTS);
        int stream_idx = 0;
        for (int dep = 0; dep < NUM_DEPS; dep++) {
            for (int j = 0; j < my_queues[dep].size(); j++) {
                auto& zoid = my_queues[dep][j];
                int zoid_num = zoid.num;
                zoid_to_stream_num[curr_dt_idx][zoid_num] = stream_idx % NUM_STREAMS;
                stream_idx++;
            }
        }

        MPI_Allreduce(MPI_IN_PLACE, zoid_to_stream_num[curr_dt_idx].data(), NUM_ZOIDS_MANY_CUTS, MPI_INT, MPI_SUM, world);

        // This is the more recent version that is trying to be implemented.
        std::map<std::pair<int, int>, int> zoid_to_zoid_to_send_stream_num;
        std::map<std::pair<int, int>, int> zoid_to_zoid_to_recv_stream_num;

        std::map<std::tuple<int, int, int>, int> send_dep_proc_to_recv_proc_send_stream_num;
        std::map<std::tuple<int, int, int, int>, int> send_dep_proc_to_recv_dep_proc_recv_stream_num;

        for (int dep = 0; dep < NUM_DEPS; dep++) {
            std::vector<std::vector<std::pair<int, int>>> zoid_to_zoid_per_proc_send(comm->nprocs);
            std::vector<std::vector<std::pair<int, int>>> zoid_to_zoid_per_proc_recv(comm->nprocs);

            for (int j = 0; j < my_queues[dep].size(); j++) {
                auto& zoid  = my_queues[dep][j];
                auto& send_neighbors = curr_dt ? send_to_neighbors_many_cuts[zoid.num] : send_to_neighbors_many_cuts_next_dt[zoid.num];
                auto& recv_neighbors = curr_dt ? recv_from_neighbors_many_cuts[zoid.num] : recv_from_neighbors_many_cuts_next_dt[zoid.num];

                for (int i = 0; i < send_neighbors.size(); i++) {
                    int send_zoid_num = send_neighbors[i];
                    if (send_zoid_num % comm->nprocs == comm->me) {
                        continue;
                    }

                    int send_zoid_dep = curr_dt ? zoid_num_to_dep[send_zoid_num] : zoid_num_to_dep_next_dt[send_zoid_num];
                    if (send_zoid_dep != dep + 1) {
                        continue;
                    }

                    int send_proc = send_zoid_num % comm->nprocs;
                    zoid_to_zoid_per_proc_send[send_proc].push_back({zoid.num, send_zoid_num});
                }

                for (int i = 0; i < recv_neighbors.size(); i++) {
                    int recv_zoid_num = recv_neighbors[i];
                    if (recv_zoid_num % comm->nprocs == comm->me) {
                        continue;
                    }

                    int recv_zoid_dep = curr_dt ? zoid_num_to_dep[recv_zoid_num] : zoid_num_to_dep_next_dt[recv_zoid_num];
                    if (recv_zoid_dep != dep - 1) {
                        continue;
                    }

                    int recv_proc = recv_zoid_num % comm->nprocs;
                    zoid_to_zoid_per_proc_recv[recv_proc].push_back({recv_zoid_num, zoid.num});
                }
            }

            const auto& procs_to_send_to = send_dep_to_procs[curr_dt_idx][DEFAULT_PIPELINE_STAGE][dep];

            int send_stream_idx = 0;
            for (int proc = 0; proc < comm->nprocs; proc++) {
                for (int j = 0; j < zoid_to_zoid_per_proc_send[proc].size(); j++) {
                    auto pair = zoid_to_zoid_per_proc_send[proc][j];
                    zoid_to_zoid_to_send_stream_num[pair] = (send_stream_idx % NUM_STREAMS);
                    // stream_loads[send_stream_idx]++;
                    send_stream_idx++;
                }

                if (std::find(procs_to_send_to.begin(), procs_to_send_to.end(), proc) != procs_to_send_to.end()) {
                    auto tup = std::make_tuple(dep, comm->me, proc);
                    send_dep_proc_to_recv_proc_send_stream_num[tup] = (send_stream_idx % NUM_STREAMS);
                    // stream_loads[send_stream_idx]++;
                    send_stream_idx++;
                }
            }

            std::map<std::pair<int, int>, std::set<int>> proc_pair_to_streams;
            std::vector<int> stream_loads(NUM_STREAMS, 0);

            const auto& recv_proc_pairs = dep_to_recv_proc_pairs[curr_dt_idx][dep];

            for (int proc = 0; proc < comm->nprocs; proc++) {
                if (proc == comm->me) {
                    continue;
                }

                auto proc_pair = std::make_pair(proc, comm->me);

                int recv_stream_idx = 0;
                assert(zoid_to_zoid_per_proc_recv[proc].size() <= NUM_STREAMS);
                for (int j = 0; j < zoid_to_zoid_per_proc_recv[proc].size(); j++) {
                    auto pair = zoid_to_zoid_per_proc_recv[proc][j];
                    // zoid_to_zoid_to_recv_stream_num[pair] = recv_stream_idx;
                    assert(recv_stream_idx < NUM_STREAMS);
                    recv_stream_idx++;
                }

                for (auto& [send_dep, send_proc] : recv_proc_pairs) {
                    if (proc == send_proc) {
                        auto tup = std::make_tuple(send_dep, proc, dep, comm->me);
                        assert(!send_dep_proc_to_recv_dep_proc_recv_stream_num.count(tup));
                        // send_dep_proc_to_recv_dep_proc_recv_stream_num[tup] = recv_stream_idx;
                        recv_stream_idx++;
                    }
                }

                if (recv_stream_idx > NUM_STREAMS) {
                    std::stringstream s1;
                    s1 << BOLDRED << "WARNING me: " << comm->me << " dep: " << dep << " proc: " << proc
                    << " size: " << zoid_to_zoid_per_proc_recv[proc].size() << " overall size: " << recv_stream_idx << RESET_COLOR << std::endl;
                    std::cout << s1.str();
                    // MPI_Abort(world, 0);
                }

                for (int j = 0; j < zoid_to_zoid_per_proc_recv[proc].size(); j++) {
                    auto pair = zoid_to_zoid_per_proc_recv[proc][j];
                    int best_stream = -1;
                    int min_load = INT_MAX;
                    for (int s = 0; s < stream_loads.size(); s++) {
                        // Skip if this stream is already used for this proc 
                        if (proc_pair_to_streams[proc_pair].count(s) > 0) {
                            // TODO: Ryan, maybe with nonblocking MPIX_Stream_progress this will work
                            // continue;
                        }
                        if (stream_loads[s] < min_load) {
                            min_load = stream_loads[s];
                            best_stream = s;
                        }
                    }

                    assert(best_stream != -1);
                    if (best_stream == -1) {
                        std::stringstream s1;
                        s1 << BOLDRED << "ERROR me: " << comm->me << " dep: " << dep << " proc: " << proc
                        << " size: " << zoid_to_zoid_per_proc_recv[proc].size() << RESET_COLOR << std::endl;
                        std::cout << s1.str();
                        MPI_Abort(world, 0);
                    }
                    zoid_to_zoid_to_recv_stream_num[pair] = best_stream;
                    stream_loads[best_stream]++;
                    proc_pair_to_streams[proc_pair].insert(best_stream);
                }

                for (auto& [send_dep, send_proc] : recv_proc_pairs) {
                    if (proc == send_proc) {
                        auto tup = std::make_tuple(send_dep, proc, dep, comm->me);
                        assert(!send_dep_proc_to_recv_dep_proc_recv_stream_num.count(tup));

                        int best_stream = -1;
                        int min_load = INT_MAX;
                        for (int s = 0; s < stream_loads.size(); s++) {
                            // Skip if this stream is already used for this proc 
                            if (proc_pair_to_streams[proc_pair].count(s) > 0) {
                                // TODO: Ryan, maybe with nonblocking MPIX_Stream_progress this will work
                                // continue;
                            }
                            if (stream_loads[s] < min_load) {
                                min_load = stream_loads[s];
                                best_stream = s;
                            }
                        }
                        assert(best_stream != -1);
                        if (best_stream == -1) {
                            std::stringstream s1;
                            s1 << BOLDRED << "ERROR me: " << comm->me << " dep: " << dep << " proc: " << proc
                            << " size: " << zoid_to_zoid_per_proc_recv[proc].size() << RESET_COLOR << std::endl;
                            std::cout << s1.str();
                            MPI_Abort(world, 0);
                        }
                        send_dep_proc_to_recv_dep_proc_recv_stream_num[tup] = best_stream;
                        stream_loads[best_stream]++;
                        proc_pair_to_streams[proc_pair].insert(best_stream);
                    }
                }
            }
        }

        std::map<std::pair<int, int>, int> all_zoid_to_zoid_to_send_stream_num;

        {
            std::vector<int> my_zoids_src;
            std::vector<int> my_zoids_dst;
            std::vector<int> my_send_stream_idxs;

            for (auto& [k, v] : zoid_to_zoid_to_send_stream_num) {
                my_zoids_src.push_back(k.first);
                my_zoids_dst.push_back(k.second);
                my_send_stream_idxs.push_back(v);
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
            std::vector<int> all_send_stream_idxs;
            all_src.resize(total_size);
            all_dst.resize(total_size);
            all_send_stream_idxs.resize(total_size);

            MPI_Allgatherv(my_zoids_src.data(), counts[comm->me], MPI_INT, all_src.data(),
                        counts.data(), displacements.data(), MPI_INT, world);

            MPI_Allgatherv(my_zoids_dst.data(), counts[comm->me], MPI_INT, all_dst.data(),
                        counts.data(), displacements.data(), MPI_INT, world);

            MPI_Allgatherv(my_send_stream_idxs.data(), counts[comm->me], MPI_INT, all_send_stream_idxs.data(),
                        counts.data(), displacements.data(), MPI_INT, world);

            for (int i = 0; i < all_src.size(); i++) {
                int src = all_src[i];
                int dst = all_dst[i];
                int stream_num = all_send_stream_idxs[i];
                all_zoid_to_zoid_to_send_stream_num[{src, dst}] = stream_num;
            }
        }

        std::map<std::pair<int, int>, int> all_zoid_to_zoid_to_recv_stream_num;
        
        {
            std::vector<int> my_zoids_src;
            std::vector<int> my_zoids_dst;
            std::vector<int> my_recv_stream_idxs;

            for (auto& [k, v] : zoid_to_zoid_to_recv_stream_num) {
                my_zoids_src.push_back(k.first);
                my_zoids_dst.push_back(k.second);
                my_recv_stream_idxs.push_back(v);
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
            std::vector<int> all_recv_stream_idxs;
            all_src.resize(total_size);
            all_dst.resize(total_size);
            all_recv_stream_idxs.resize(total_size);

            MPI_Allgatherv(my_zoids_src.data(), counts[comm->me], MPI_INT, all_src.data(),
                        counts.data(), displacements.data(), MPI_INT, world);

            MPI_Allgatherv(my_zoids_dst.data(), counts[comm->me], MPI_INT, all_dst.data(),
                        counts.data(), displacements.data(), MPI_INT, world);

            MPI_Allgatherv(my_recv_stream_idxs.data(), counts[comm->me], MPI_INT, all_recv_stream_idxs.data(),
                        counts.data(), displacements.data(), MPI_INT, world);

            for (int i = 0; i < all_src.size(); i++) {
                int src = all_src[i];
                int dst = all_dst[i];
                int stream_num = all_recv_stream_idxs[i];
                all_zoid_to_zoid_to_recv_stream_num[{src, dst}] = stream_num;
            }
        }
        
        std::map<std::tuple<int, int, int>, int> all_send_dep_proc_to_send_stream_num;

        {
            std::vector<int> my_send_dep;
            std::vector<int> my_send_proc;
            std::vector<int> my_recv_proc;
            std::vector<int> my_send_stream_idxs;

            for (auto& [tup, v] : send_dep_proc_to_recv_proc_send_stream_num) {
                my_send_dep.push_back(std::get<0>(tup));
                my_send_proc.push_back(std::get<1>(tup));
                my_recv_proc.push_back(std::get<2>(tup));
                my_send_stream_idxs.push_back(v);
            }

            std::vector<int> counts(comm->nprocs, 0);
            std::vector<int> displacements(comm->nprocs, 0);

            int my_count = my_send_dep.size();
            MPI_Allgather(&my_count, 1, MPI_INT, counts.data(), 1, MPI_INT, world);

            int total_size = 0;
            for (int proc = 0; proc < comm->nprocs; proc++) {
                total_size += counts[proc];
            }

            displacements[0] = 0;
            for (int proc = 1; proc < comm->nprocs; proc++) {
                displacements[proc] = displacements[proc - 1] + counts[proc - 1];
            }

            std::vector<int> all_send_dep;
            std::vector<int> all_send_proc;
            std::vector<int> all_recv_proc;
            std::vector<int> all_send_stream_idxs;
            all_send_dep.resize(total_size);
            all_send_proc.resize(total_size);
            all_recv_proc.resize(total_size);
            all_send_stream_idxs.resize(total_size);

            MPI_Allgatherv(my_send_dep.data(), counts[comm->me], MPI_INT, all_send_dep.data(),
                        counts.data(), displacements.data(), MPI_INT, world);

            MPI_Allgatherv(my_send_proc.data(), counts[comm->me], MPI_INT, all_send_proc.data(),
                        counts.data(), displacements.data(), MPI_INT, world);

            MPI_Allgatherv(my_recv_proc.data(), counts[comm->me], MPI_INT, all_recv_proc.data(),
                        counts.data(), displacements.data(), MPI_INT, world);

            MPI_Allgatherv(my_send_stream_idxs.data(), counts[comm->me], MPI_INT, all_send_stream_idxs.data(),
                        counts.data(), displacements.data(), MPI_INT, world);

            for (int i = 0; i < all_send_stream_idxs.size(); i++) {
                int send_dep = all_send_dep[i];
                int send_proc = all_send_proc[i];
                int recv_proc = all_recv_proc[i];
                int send_stream_num = all_send_stream_idxs[i];
                auto tup = std::make_tuple(send_dep, send_proc, recv_proc);
                assert(!all_send_dep_proc_to_send_stream_num.count(tup));
                all_send_dep_proc_to_send_stream_num[tup] = send_stream_num;
            }
        }

        std::map<std::tuple<int, int, int, int>, int> all_send_dep_proc_to_recv_dep_proc_recv_stream_num;

        {
            std::vector<int> my_send_dep;
            std::vector<int> my_send_proc;
            std::vector<int> my_recv_dep;
            std::vector<int> my_recv_proc;
            std::vector<int> my_recv_stream_idxs;

            for (auto& [tup, v] : send_dep_proc_to_recv_dep_proc_recv_stream_num) {
                my_send_dep.push_back(std::get<0>(tup));
                my_send_proc.push_back(std::get<1>(tup));
                my_recv_dep.push_back(std::get<2>(tup));
                my_recv_proc.push_back(std::get<3>(tup));
                my_recv_stream_idxs.push_back(v);
            }

            std::vector<int> counts(comm->nprocs, 0);
            std::vector<int> displacements(comm->nprocs, 0);

            int my_count = my_send_dep.size();
            MPI_Allgather(&my_count, 1, MPI_INT, counts.data(), 1, MPI_INT, world);

            int total_size = 0;
            for (int proc = 0; proc < comm->nprocs; proc++) {
                total_size += counts[proc];
            }

            displacements[0] = 0;
            for (int proc = 1; proc < comm->nprocs; proc++) {
                displacements[proc] = displacements[proc - 1] + counts[proc - 1];
            }

            std::vector<int> all_send_dep;
            std::vector<int> all_send_proc;
            std::vector<int> all_recv_dep;
            std::vector<int> all_recv_proc;
            std::vector<int> all_recv_stream_idxs;

            all_send_dep.resize(total_size);
            all_send_proc.resize(total_size);
            all_recv_dep.resize(total_size);
            all_recv_proc.resize(total_size);
            all_recv_stream_idxs.resize(total_size);

            MPI_Allgatherv(my_send_dep.data(), counts[comm->me], MPI_INT, all_send_dep.data(),
                        counts.data(), displacements.data(), MPI_INT, world);

            MPI_Allgatherv(my_send_proc.data(), counts[comm->me], MPI_INT, all_send_proc.data(),
                        counts.data(), displacements.data(), MPI_INT, world);

            MPI_Allgatherv(my_recv_dep.data(), counts[comm->me], MPI_INT, all_recv_dep.data(),
                        counts.data(), displacements.data(), MPI_INT, world);

            MPI_Allgatherv(my_recv_proc.data(), counts[comm->me], MPI_INT, all_recv_proc.data(),
                        counts.data(), displacements.data(), MPI_INT, world);

            MPI_Allgatherv(my_recv_stream_idxs.data(), counts[comm->me], MPI_INT, all_recv_stream_idxs.data(),
                        counts.data(), displacements.data(), MPI_INT, world);

            for (int i = 0; i < all_recv_stream_idxs.size(); i++) {
                int send_dep = all_send_dep[i];
                int send_proc = all_send_proc[i];
                int recv_dep = all_recv_dep[i];
                int recv_proc = all_recv_proc[i];
                int recv_stream_num = all_recv_stream_idxs[i];
                auto tup = std::make_tuple(send_dep, send_proc, recv_dep, recv_proc);
                all_send_dep_proc_to_recv_dep_proc_recv_stream_num[tup] = recv_stream_num;
            }
        }

        assert(all_zoid_to_zoid_to_send_stream_num.size() == all_zoid_to_zoid_to_recv_stream_num.size());
        for (auto& [zoid_pair, send_stream_num] : all_zoid_to_zoid_to_send_stream_num) {
            assert(all_zoid_to_zoid_to_recv_stream_num.count(zoid_pair));
            auto recv_stream_num = all_zoid_to_zoid_to_recv_stream_num.at(zoid_pair);
            zoid_to_zoid_to_stream_num[curr_dt_idx][zoid_pair] = {send_stream_num, recv_stream_num};
        }

        for (int dep = 1; dep < NUM_DEPS; dep++) {
            stream_num_to_zoid_pairs[curr_dt_idx][dep].resize(NUM_STREAMS);
        }

        for (auto& [zoid_pair, stream_pair]: zoid_to_zoid_to_stream_num[curr_dt_idx]) {
            int src_zoid_num = zoid_pair.first;
            int dst_zoid_num = zoid_pair.second;
            int dst_zoid_dep = curr_dt ? zoid_num_to_dep[dst_zoid_num] : zoid_num_to_dep_next_dt[dst_zoid_num];
            int dst_stream_num = stream_pair.second;
            if (dst_zoid_num % comm->nprocs == comm->me) {
                assert(src_zoid_num % comm->nprocs != comm->me);
                stream_num_to_zoid_pairs[curr_dt_idx][dst_zoid_dep][dst_stream_num].push_back({src_zoid_num, dst_zoid_num});
            }
        }

        for (int dep = 1; dep < NUM_DEPS; dep++) {
            for (int j = 0; j < NUM_STREAMS; j++) {
                auto& zoid_pairs = stream_num_to_zoid_pairs[curr_dt_idx][dep][j];
                for (int i = 0; i < zoid_pairs.size(); i++) {
                    zoid_pair_to_recv_request_idx_streams[curr_dt_idx][zoid_pairs[i]] = i;
                }
            }
        }

        assert(all_zoid_to_zoid_to_send_stream_num.size() == all_zoid_to_zoid_to_recv_stream_num.size());
        for (auto& [zoid_pair, send_stream_num] : all_zoid_to_zoid_to_send_stream_num) {
            assert(all_zoid_to_zoid_to_recv_stream_num.count(zoid_pair));
            auto recv_stream_num = all_zoid_to_zoid_to_recv_stream_num.at(zoid_pair);
            zoid_to_zoid_to_stream_num[curr_dt_idx][zoid_pair] = {send_stream_num, recv_stream_num};
        }

        for (auto& [send_tup, send_stream_num] : all_send_dep_proc_to_send_stream_num) {
            auto [send_dep, send_proc, recv_proc] = send_tup;
            bool recv_found = false;
            bool send_found = false;
            for (auto& [tup, recv_stream_num] : all_send_dep_proc_to_recv_dep_proc_recv_stream_num) {
                if (send_dep == std::get<0>(tup) && send_proc == std::get<1>(tup) && recv_proc == std::get<3>(tup)) {
                    if (recv_proc == comm->me) {
                        int recv_dep = std::get<2>(tup);
                        receiver_dep_proc_to_stream_num[curr_dt_idx][recv_dep][{send_dep, send_proc}] = {send_stream_num, recv_stream_num};
                        assert(!recv_found);
                        recv_found = true;
                    }

                    if (send_proc == comm->me) {
                        assert(!sender_dep_proc_to_stream_num[curr_dt_idx].count({send_dep, recv_proc}));
                        sender_dep_proc_to_stream_num[curr_dt_idx][{send_dep, recv_proc}] = {send_stream_num, recv_stream_num};
                    }
                }
            }
        }

        for (int dep = 1; dep < NUM_DEPS; dep++) {
            stream_num_to_dep_proc_pairs[curr_dt_idx][dep].resize(NUM_STREAMS);
            // for (auto& [dep_proc_pair, stream_pair]: send_dep_proc_to_stream_num[curr_dt_idx][dep]) {
            for (auto& [tup, recv_stream_num] : all_send_dep_proc_to_recv_dep_proc_recv_stream_num) {
                auto& [send_dep, send_proc, recv_dep, recv_proc] = tup;
                if (recv_proc == comm->me && recv_dep == dep) {
                    stream_num_to_dep_proc_pairs[curr_dt_idx][dep][recv_stream_num].push_back({send_dep, send_proc});
                }
            }
        }
    }

    template <bool curr_dt>
    void CONSTRUCT_PER_ZOID_RECV_REQUEST_IDXS() {
        auto& my_queues = curr_dt ? my_queues_many_cuts : my_queues_many_cuts_next_dt;
        constexpr int curr_dt_idx = static_cast<int>(curr_dt);

        recv_request_idx_to_zoid_per_zoid[curr_dt_idx].resize(stencilMD->NUM_ZOIDS_MANY_CUTS);
        recv_request_zoid_to_idx_per_zoid[curr_dt_idx].resize(stencilMD->NUM_ZOIDS_MANY_CUTS);

        constexpr int NUM_NEIGHBORS_PER_DEP[NUM_DEPS] = {0, 2, 4, 6};

        for (int dep = 1; dep < NUM_DEPS; dep++) {
            for (int j = 0; j < my_queues[dep].size(); j++) {
                auto& zoid = my_queues[dep][j];

                auto& recv_neighbors = curr_dt ? recv_from_neighbors_many_cuts[zoid.num] : recv_from_neighbors_many_cuts_next_dt[zoid.num];
                int recv_request_idx = 0;
                for (int i = 0; i < recv_neighbors.size(); i++) {
                    int recv_zoid_num = recv_neighbors[i];
                    int recv_zoid_dep = curr_dt ? zoid_num_to_dep[recv_zoid_num] : zoid_num_to_dep_next_dt[recv_zoid_num];
                    if (recv_zoid_num % comm->nprocs == comm->me) {
                        continue;
                    }

                    if (recv_zoid_dep != dep - 1) {
                        continue;
                    }

                    recv_request_idx_to_zoid_per_zoid[curr_dt_idx][zoid.num].push_back(recv_zoid_num);
                    recv_request_zoid_to_idx_per_zoid[curr_dt_idx][zoid.num][recv_zoid_num] = recv_request_idx;
                    recv_request_idx++;
                }
            }
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
                        constexpr bool IS_MANY_BODY = (EXPERIMENT == SW) || (EXPERIMENT == TERSOFF) || (EXPERIMENT == EAM);
                        if constexpr (!IS_MANY_BODY) {
                            double dist_to_zoid = distance_to_zoid(domain->prd, zoid.lo[t], zoid.hi[t], new_pos_borders);
                            borders_zoid = (dist_to_zoid <= ALLEGRO_SLOPE);
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
                                if constexpr (EXPERIMENT == DPD) {
                                    zoid.v_stencil_md[1].push_back({all_vel[idx * 3 + 0], all_vel[idx * 3 + 1], all_vel[idx * 3 + 2]});
                                }
                                zoid.tag_stencil_md[0].push_back(idx);
                                zoid.type_stencil_md[0].push_back(all_type[idx]);
                                zoid.mask_stencil_md[0].push_back(all_mask[idx]);
                                zoid.image_stencil_md[0].push_back(all_image[idx]);
                                zoid.f_stencil_md[0].push_back({0.0, 0.0, 0.0});
                                zoid.eval_f_stencil_md[0].push_back({0.0, 0.0, 0.0});
                                zoid.rho_stencil_md[0].push_back(0);
                                zoid.fp_stencil_md[0].push_back(0);
                            }
                            break;
                        }
                    }
                }

                zoid.neigh_short[0].resize(zoid.x_stencil_md[0].size());
                for (int i = 0; i < zoid.x_stencil_md[0].size(); i++) {
                    zoid.neigh_short[0][i].reserve(1024);
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

    std::pair<bool, double> MANY_BODY_CHECK_IF_ATOM_NEEDS_NEIGHBOR(queue_info& zoid, std::array<double, 3>& zoid_lo, std::array<double, 3>& zoid_hi, std::array<double, 3>& atom_pos) {
        constexpr double NEIGHBOR_CUTOFF = ALLEGRO_SLOPE / 2;

        for (int dim = 0;  dim < 3; dim++) {
            bool my_dim_expanding = (zoid.zoid.cuts[dim].slope_lower < 0);
            double lo = zoid_lo[dim];
            double hi = zoid_hi[dim];
            double pos = atom_pos[dim];
            bool in_bounds = (pos >= lo && pos < hi);
            double dist_to_boundary = std::min(fabs(pos - lo), fabs(pos - hi));

            if (my_dim_expanding && in_bounds && dist_to_boundary < NEIGHBOR_CUTOFF) {
                return std::make_pair(false, -1);
            }

            if (my_dim_expanding && !in_bounds) {
                return std::make_pair(false, -1);
            }
        }

        // easy case, if no dims expanding --> in dep 0 always return true
        return std::make_pair(true, 1);
    }

    template <bool newton>
    void CREATE_NEIGHBOR_LIST_HELPER_SW(queue_info& zoid, std::vector<int>* neighbor_lst, bool print=false) {
        std::unordered_map<int, int> zoid_tag_to_idx;
        for (int k = 0; k < zoid.tag_stencil_md[0].size(); k++) {
            zoid_tag_to_idx[zoid.tag_stencil_md[0][k]] = k;
        }

        constexpr double NEIGHBOR_CUTOFF = ALLEGRO_SLOPE / 2;

        auto& x = zoid.x_stencil_md[0];
        for (int t = 0; t < NUM_TIMESTEPS_IN_PARALLEL + 1; t++) {
            zoid.neighbor_list[t].resize(zoid.x_stencil_md[0].size());

            std::unordered_set<int> local_idxs_set;
            const auto& local_idxs = zoid.local_idxs_per_timestep[t];
            local_idxs_set.insert(local_idxs.begin(), local_idxs.end());

            std::set<int> one_hop_ghost_neigh_idxs;

            for (int i = 0; i < local_idxs.size(); i++) {
                int idx = local_idxs[i];
                zoid.neighbor_list[t][idx].reserve(20);

                std::array<double, 3> atom_pos = {x[idx].x, x[idx].y, x[idx].z};

                auto [atom_needs_neighbors, dist_to_boundary] = MANY_BODY_CHECK_IF_ATOM_NEEDS_NEIGHBOR(zoid, zoid.lo[t], zoid.hi[t], atom_pos);
                int tag = zoid.tag_stencil_md[0][idx];
                std::set<int> neigh_set;
                neigh_set.insert(neighbor_lst[tag].begin(), neighbor_lst[tag].end());

                if (!atom_needs_neighbors) {
                    continue;
                }

                double xtmp = x[idx].x;
                double ytmp = x[idx].y;
                double ztmp = x[idx].z;

                for (auto &neigh_tag: neigh_set) {
                    if (!zoid_tag_to_idx.count(neigh_tag)) {
                        std::cout << "zoid: " << zoid.num << " timestep: " << t << " tag: " << tag << " neigh tag: " << neigh_tag << std::endl;
                        assert(false);
                    }

                    int neigh_idx = zoid_tag_to_idx.at(neigh_tag);

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

            for (int i = 0; i < zoid.x_stencil_md[0].size(); i++) {
                if (local_idxs_set.find(i) != local_idxs_set.end()) {
                    continue;
                }

                std::array<double, 3> atom_pos = {x[i].x, x[i].y, x[i].z};

                bool borders = true;
                double dist_dim[3] = {0};
                for (int dim = 0; dim < 3; dim++) {
                    double lo = zoid.lo[t][dim];
                    double hi = zoid.hi[t][dim];
                    double pos = atom_pos[dim];

                    if (pos < lo && lo - pos >= NEIGHBOR_CUTOFF) {
                        borders = false;
                        dist_dim[dim] = lo - pos;
                    }

                    if (pos >= hi && pos - hi >= NEIGHBOR_CUTOFF) {
                        borders = false;
                        dist_dim[dim] = pos - hi;
                    }
                }

                if (borders) {
                    one_hop_ghost_neigh_idxs.insert(i);
                }
	        }

            for (auto& one_hop_ghost_neigh_idx : one_hop_ghost_neigh_idxs) {
                zoid.neighbor_list[t][one_hop_ghost_neigh_idx].reserve(20);
                std::array<double, 3> atom_pos = {x[one_hop_ghost_neigh_idx].x, x[one_hop_ghost_neigh_idx].y, x[one_hop_ghost_neigh_idx].z};

                int tag = zoid.tag_stencil_md[0][one_hop_ghost_neigh_idx];
                std::set<int> neigh_set;
                neigh_set.insert(neighbor_lst[tag].begin(), neighbor_lst[tag].end());

                double xtmp = x[one_hop_ghost_neigh_idx].x;
                double ytmp = x[one_hop_ghost_neigh_idx].y;
                double ztmp = x[one_hop_ghost_neigh_idx].z;

                auto [atom_needs_neighbors, dist_to_boundary] = MANY_BODY_CHECK_IF_ATOM_NEEDS_NEIGHBOR(zoid, zoid.lo[t], zoid.hi[t], atom_pos);

                if (!atom_needs_neighbors) {
                    continue;
                }

                for (auto &neigh_tag: neigh_set) {
                    if (!zoid_tag_to_idx.count(neigh_tag)) {
                        std::cout << "zoid: " << zoid.num << " timestep: " << t << " tag: " << tag << " neigh tag: " << neigh_tag << std::endl;
                        assert(false);
                    }

                    int neigh_idx = zoid_tag_to_idx.at(neigh_tag);

                    double delx = xtmp - x[neigh_idx].x;
                    double dely = ytmp - x[neigh_idx].y;
                    double delz = ztmp - x[neigh_idx].z;
                    double rsq = delx*delx + dely*dely + delz*delz;
                    int itype = zoid.type_stencil_md[0][one_hop_ghost_neigh_idx];
                    int jtype = zoid.type_stencil_md[0][neigh_idx];

                    if (rsq <= neighbor->cutneighsq[itype][jtype]) {
                        zoid.neighbor_list[t][one_hop_ghost_neigh_idx].push_back(neigh_idx);
                    } else {
                        std::cout << "zoid: " << zoid.num << " neighbor failed check? "
                            << " x: " << xtmp << " " << ytmp << " " << ztmp
                            << " other x: " << x[neigh_idx].x << " " << x[neigh_idx].y << " " << x[neigh_idx].z
                            << " rsq: " << rsq << " neighbor cut: " << neighbor->cutneighsq[itype][jtype]
                            << std::endl;
                    }
                }
            }

            for (auto& local_idx : local_idxs_set) {
                zoid.local_and_one_hop_ghost_idxs_per_timestep[t].push_back(local_idx);
            }
            for (auto& one_hop_ghost_neigh_idx : one_hop_ghost_neigh_idxs) {
                zoid.local_and_one_hop_ghost_idxs_per_timestep[t].push_back(one_hop_ghost_neigh_idx);
            }
            std::sort(zoid.local_and_one_hop_ghost_idxs_per_timestep[t].begin(), zoid.local_and_one_hop_ghost_idxs_per_timestep[t].end(), std::less<int>());
        }
    }

    // basically create_neighbor_list_sw but half neighbor list
    template <bool newton>
    void CREATE_NEIGHBOR_LIST_HELPER_EAM(queue_info& zoid, std::vector<int>* neighbor_lst, bool print=false) {
        constexpr int target_tag = 181712;
        constexpr int target_tag2 = 186753;

        // This is a half neighbor list as LAMMPS uses half neighbor lists for EAM
        std::unordered_map<int, int> zoid_tag_to_idx;
        for (int k = 0; k < zoid.tag_stencil_md[0].size(); k++) {
            zoid_tag_to_idx[zoid.tag_stencil_md[0][k]] = k;
        }

        constexpr double NEIGHBOR_CUTOFF = ALLEGRO_SLOPE / 2;

        auto& x = zoid.x_stencil_md[0];
        for (int t = 0; t < NUM_TIMESTEPS_IN_PARALLEL + 1; t++) {
            zoid.neighbor_list[t].resize(zoid.x_stencil_md[0].size());

            std::unordered_set<int> local_idxs_set;
            const auto& local_idxs = zoid.local_idxs_per_timestep[t];
            local_idxs_set.insert(local_idxs.begin(), local_idxs.end());

            std::set<int> one_hop_ghost_neigh_idxs;

            for (int i = 0; i < local_idxs.size(); i++) {
                int idx = local_idxs[i];
                zoid.neighbor_list[t][idx].reserve(20);

                std::array<double, 3> atom_pos = {x[idx].x, x[idx].y, x[idx].z};

                auto [atom_needs_neighbors, dist_to_boundary] = MANY_BODY_CHECK_IF_ATOM_NEEDS_NEIGHBOR(zoid, zoid.lo[t], zoid.hi[t], atom_pos);
                int tag = zoid.tag_stencil_md[0][idx];
                std::set<int> neigh_set;
                neigh_set.insert(neighbor_lst[tag].begin(), neighbor_lst[tag].end());

                // if ((tag == target_tag || tag == target_tag2) && t == 0 && print) {
                //     std::stringstream s1;
                //     s1 << BOLDCYAN << "FOUND TAG MAIN LOCAL LOOP. zoid: " << zoid.num << " tag: " << tag
                //     << " pos: " << atom_pos[0] << " " << atom_pos[1] << " " << atom_pos[2]
                //     << " lo: " << zoid.lo[t][0] << " " << zoid.lo[t][1] << " " << zoid.lo[t][2]
                //     << " hi: " << zoid.hi[t][0] << " " << zoid.hi[t][1] << " " << zoid.hi[t][2]
                //     << " needs neighbor? " << atom_needs_neighbors
                //     << RESET_COLOR << std::endl;
                //     std::cout << s1.str();
                // }

                if (!atom_needs_neighbors) {
                    continue;
                }

                double xtmp = x[idx].x;
                double ytmp = x[idx].y;
                double ztmp = x[idx].z;

                for (auto &neigh_tag: neigh_set) {
                    if (!zoid_tag_to_idx.count(neigh_tag)) {
                        std::cout << "zoid: " << zoid.num << " timestep: " << t << " tag: " << tag << " neigh tag: " << neigh_tag << std::endl;
                        assert(false);
                    }

                    int neigh_idx = zoid_tag_to_idx.at(neigh_tag);
                    bool neigh_local = (local_idxs_set.find(neigh_idx) != local_idxs_set.end());
                    std::array<double, 3> neigh_pos = {x[neigh_idx].x, x[neigh_idx].y, x[neigh_idx].z};
                    auto [neigh_needs_neighbors, dist_to_boundary] = MANY_BODY_CHECK_IF_ATOM_NEEDS_NEIGHBOR(zoid, zoid.lo[t], zoid.hi[t], neigh_pos);

                    // if ((tag == target_tag || tag == target_tag2) && t == 0 && print) {
                    //     std::stringstream s1;
                    //     s1 << BOLDCYAN << "FOUND TAG MAIN LOCAL LOOP. zoid: " << zoid.num << " tag: " << tag << " " << neigh_tag
                    //     << " pos: " << atom_pos[0] << " " << atom_pos[1] << " " << atom_pos[2]
                    //     << " pos: " << neigh_pos[0] << " " << neigh_pos[1] << " " << neigh_pos[2]
                    //     << " needs neighbor? " << atom_needs_neighbors
                    //     << " neigh local? " << neigh_local
                    //     << " neigh needs neighbors? " << neigh_needs_neighbors
                    //     << RESET_COLOR << std::endl;
                    //     std::cout << s1.str();
                    // }

                    if (!neigh_needs_neighbors) {
                        continue;
                    }

                    // half neighbor list so tiebreak local-local pairs
                    if (neigh_local && neigh_needs_neighbors) {
                        if (x[neigh_idx].z < ztmp) continue;
                        if (x[neigh_idx].z == ztmp) {
                            if (x[neigh_idx].y < ytmp) continue;
                            if (x[neigh_idx].y == ytmp && x[neigh_idx].x < xtmp) continue;
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

            for (int i = 0; i < zoid.x_stencil_md[0].size(); i++) {
                if (local_idxs_set.find(i) != local_idxs_set.end()) {
                    continue;
                }

                std::array<double, 3> atom_pos = {x[i].x, x[i].y, x[i].z};

                bool borders = true;
                double dist_dim[3] = {0};
                for (int dim = 0; dim < 3; dim++) {
                    double lo = zoid.lo[t][dim];
                    double hi = zoid.hi[t][dim];
                    double pos = atom_pos[dim];

                    if (pos < lo && lo - pos >= NEIGHBOR_CUTOFF) {
                        borders = false;
                        dist_dim[dim] = lo - pos;
                    }

                    if (pos >= hi && pos - hi >= NEIGHBOR_CUTOFF) {
                        borders = false;
                        dist_dim[dim] = pos - hi;
                    }
                }

                if (borders) {
                    one_hop_ghost_neigh_idxs.insert(i);
                }
	        }

            for (auto& one_hop_ghost_neigh_idx : one_hop_ghost_neigh_idxs) {
                zoid.neighbor_list[t][one_hop_ghost_neigh_idx].reserve(20);
                std::array<double, 3> atom_pos = {x[one_hop_ghost_neigh_idx].x, x[one_hop_ghost_neigh_idx].y, x[one_hop_ghost_neigh_idx].z};

                int tag = zoid.tag_stencil_md[0][one_hop_ghost_neigh_idx];
                std::set<int> neigh_set;
                neigh_set.insert(neighbor_lst[tag].begin(), neighbor_lst[tag].end());

                double xtmp = x[one_hop_ghost_neigh_idx].x;
                double ytmp = x[one_hop_ghost_neigh_idx].y;
                double ztmp = x[one_hop_ghost_neigh_idx].z;

                auto [atom_needs_neighbors, dist_to_boundary] = MANY_BODY_CHECK_IF_ATOM_NEEDS_NEIGHBOR(zoid, zoid.lo[t], zoid.hi[t], atom_pos);

                // if ((tag == target_tag || tag == target_tag2) && t == 0 && print) {
                //     std::stringstream s1;
                //     s1 << BOLDYELLOW << "ONE HOP GHOST FOUND IDX. zoid: " << zoid.num << " tag: " << tag
                //     << " pos: " << atom_pos[0] << " " << atom_pos[1] << " " << atom_pos[2]
                //     << " lo: " << zoid.lo[t][0] << " " << zoid.lo[t][1] << " " << zoid.lo[t][2]
                //     << " hi: " << zoid.hi[t][0] << " " << zoid.hi[t][1] << " " << zoid.hi[t][2]
                //     << " needs neighbor? " << atom_needs_neighbors
                //     << RESET_COLOR << std::endl;
                //     std::cout << s1.str();
                // }

                if (!atom_needs_neighbors) {
                    continue;
                }

                for (auto &neigh_tag: neigh_set) {
                    if (!zoid_tag_to_idx.count(neigh_tag)) {
                        std::cout << "zoid: " << zoid.num << " timestep: " << t << " tag: " << tag << " neigh tag: " << neigh_tag << std::endl;
                        assert(false);
                    }

                    int neigh_idx = zoid_tag_to_idx.at(neigh_tag);
                    int neigh_local = (local_idxs_set.find(neigh_idx) != local_idxs_set.end());
                    std::array<double, 3> neigh_pos = {x[neigh_idx].x, x[neigh_idx].y, x[neigh_idx].z};
                    auto [neigh_needs_neighbors, dist_to_boundary] = MANY_BODY_CHECK_IF_ATOM_NEEDS_NEIGHBOR(zoid, zoid.lo[t], zoid.hi[t], neigh_pos);
                    bool is_neigh_one_hop_ghost = (one_hop_ghost_neigh_idxs.find(neigh_idx) != one_hop_ghost_neigh_idxs.end());

                    // if ((tag == target_tag || tag == target_tag2) && t == 0 && print) {
                    //     std::stringstream s1;
                    //     s1 << BOLDYELLOW << "ONE HOP GHOST NEIGHBOR. zoid: " << zoid.num << " tag: " << tag << " " << neigh_tag
                    //     << " pos: " << atom_pos[0] << " " << atom_pos[1] << " " << atom_pos[2]
                    //     << " pos: " << neigh_pos[0] << " " << neigh_pos[1] << " " << neigh_pos[2]
                    //     << " lo: " << zoid.lo[t][0] << " " << zoid.lo[t][1] << " " << zoid.lo[t][2]
                    //     << " hi: " << zoid.hi[t][0] << " " << zoid.hi[t][1] << " " << zoid.hi[t][2]
                    //     << " atom needs neighbor? " << atom_needs_neighbors
                    //     << " neigh local? " << neigh_local
                    //     << " neigh needs neighbor? " << neigh_needs_neighbors
                    //     << " is one hop ghost? " << (one_hop_ghost_neigh_idxs.find(neigh_idx) != one_hop_ghost_neigh_idxs.end())
                    //     << RESET_COLOR << std::endl;
                    //     std::cout << s1.str();
                    // }

                    // I am a one-hop ghost. Possible cases: I am pointing to local, another one-hop ghost or a two-hop ghost.
                    // if pointing to a local, then I want to make sure that that local atom has not been evaluated yet.

                    if (!neigh_needs_neighbors) {
                        continue;
                    }

                    if (neigh_local && neigh_needs_neighbors) {
                        continue;
                    }

                    // Point to another one-hop ghost, pick one to tiebreak
                    if (neigh_needs_neighbors && is_neigh_one_hop_ghost) {
                        if (x[neigh_idx].z < ztmp) continue;
                        if (x[neigh_idx].z == ztmp) {
                            if (x[neigh_idx].y < ytmp) continue;
                            if (x[neigh_idx].y == ytmp && x[neigh_idx].x < xtmp) continue;
                        }
                    } 

                    double delx = xtmp - x[neigh_idx].x;
                    double dely = ytmp - x[neigh_idx].y;
                    double delz = ztmp - x[neigh_idx].z;
                    double rsq = delx*delx + dely*dely + delz*delz;
                    int itype = zoid.type_stencil_md[0][one_hop_ghost_neigh_idx];
                    int jtype = zoid.type_stencil_md[0][neigh_idx];

                    if (rsq <= neighbor->cutneighsq[itype][jtype]) {
                        zoid.neighbor_list[t][one_hop_ghost_neigh_idx].push_back(neigh_idx);
                    } else {
                        std::cout << "zoid: " << zoid.num << " neighbor failed check? "
                            << " x: " << xtmp << " " << ytmp << " " << ztmp
                            << " other x: " << x[neigh_idx].x << " " << x[neigh_idx].y << " " << x[neigh_idx].z
                            << " rsq: " << rsq << " neighbor cut: " << neighbor->cutneighsq[itype][jtype]
                            << std::endl;
                    }
                }
            }

            for (auto& local_idx : local_idxs_set) {
                zoid.local_and_one_hop_ghost_idxs_per_timestep[t].push_back(local_idx);
            }
            for (auto& one_hop_ghost_neigh_idx : one_hop_ghost_neigh_idxs) {
                zoid.local_and_one_hop_ghost_idxs_per_timestep[t].push_back(one_hop_ghost_neigh_idx);
            }
            std::sort(zoid.local_and_one_hop_ghost_idxs_per_timestep[t].begin(), zoid.local_and_one_hop_ghost_idxs_per_timestep[t].end(), std::less<int>());
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
        for (int ii = 0; ii < list->inum + list->gnum; ii++) {
            int i = list->ilist[ii];
            assert(i == ii);
            int numneigh = list->numneigh[i];
            for (int j = 0; j < numneigh; j++) {
                int neigh = list->firstneigh[i][j];
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

        std::vector<int> claim_one_hop_ghosts(atom->natoms + 1, 0);

        cilk_for (int dep = 0; dep < NUM_DEPS; dep++) {
            cilk_for (int j = 0; j < queues_many_cuts[dep].size(); j++) {
                auto& zoid = queues_many_cuts[dep][j];
                if (zoid.num % comm->nprocs != comm->me) {
                    continue;
                }

                if constexpr (EXPERIMENT == SW || EXPERIMENT == TERSOFF) {
                    CREATE_NEIGHBOR_LIST_HELPER_SW<newton>(zoid, neighbor_lst);
                } else if constexpr (EXPERIMENT == EAM) {
                    CREATE_NEIGHBOR_LIST_HELPER_EAM<newton>(zoid, neighbor_lst, true);
                } else {
                    CREATE_NEIGHBOR_LIST_HELPER<newton>(zoid, neighbor_lst);
                }
            }
        }

        cilk_for (int dep = 0; dep < NUM_DEPS; dep++) {
            cilk_for (int j = 0; j < queues_many_cuts_next_dt[dep].size(); j++) {
                auto& zoid = queues_many_cuts_next_dt[dep][j];
                if (zoid.num % comm->nprocs != comm->me) {
                    continue;
                }

                if constexpr (EXPERIMENT == SW || EXPERIMENT == TERSOFF) {
                    CREATE_NEIGHBOR_LIST_HELPER_SW<newton>(zoid, neighbor_lst);
                } else if constexpr (EXPERIMENT == EAM) {
                    CREATE_NEIGHBOR_LIST_HELPER_EAM<newton>(zoid, neighbor_lst);
                } else {
                    CREATE_NEIGHBOR_LIST_HELPER<newton>(zoid, neighbor_lst);
                }
            }
        }

        for (int t = 1; t < NUM_TIMESTEPS_IN_PARALLEL + 1; t++) {
            int total_neighbors = 0;
            for (int dep = 0; dep < NUM_DEPS; dep++) {
                for (int j = 0; j < my_queues_many_cuts[dep].size(); j++) {
                    auto& zoid = my_queues_many_cuts[dep][j];
                    auto& neighbor_list = zoid.neighbor_list[t];
                    for (int i = 0; i < zoid.local_idxs_per_timestep[t].size(); i++) {
                        int idx = zoid.local_idxs_per_timestep[t][i];
                        int num_neigh = neighbor_list[idx].size();
                        total_neighbors += num_neigh;
                    }
                }
            }

            MPI_Allreduce(MPI_IN_PLACE, &total_neighbors, 1, MPI_INT, MPI_SUM, world);
            if (comm->me == 0) {
                std::cout << "t: " << t << " num atoms: " << atom->natoms << " num neighbors: " << total_neighbors << " neighbors per atom: " << total_neighbors * 1.0 / atom->natoms << std::endl;
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
                    constexpr int GRAINSIZE = 128;
                    int num_chunks = size / GRAINSIZE + 1;
                    int chunk_size = GRAINSIZE;

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

                constexpr bool IS_MANY_BODY = (EXPERIMENT == SW) || (EXPERIMENT == TERSOFF) || (EXPERIMENT == EAM);
                if constexpr (!IS_MANY_BODY) {
                    double dist_to_zoid = distance_to_zoid(domain->prd, zoid.lo[t], zoid.hi[t], atom_pos);
                    borders_zoid = (borders_zoid && dist_to_zoid <= ALLEGRO_SLOPE);
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
    void CONSTRUCT_SEND_POS_IDXS_ZOID_MANY_CUTS() {
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

                constexpr bool IS_MANY_BODY = (EXPERIMENT == SW) || (EXPERIMENT == TERSOFF) || (EXPERIMENT == EAM);

                if constexpr (!IS_MANY_BODY) {
                    double dist_to_zoid = distance_to_zoid(domain->prd, zoid.lo[t], zoid.hi[t], atom_pos);
                    borders_zoid = (borders_zoid && dist_to_zoid <= ALLEGRO_SLOPE);
                }

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
    void CONSTRUCT_SEND_FP_IDXS_ZOID_MANY_CUTS() {
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

                for (int t = 0; t < NUM_TIMESTEPS_IN_PARALLEL + 1; t++) {
                    zoid_recv_data[zoid.num][t] = new std::vector<int>[recv_neighbors.size()];
                    zoid_recv_data_sizes[zoid.num][t] = new int[recv_neighbors.size()];

                    for (int i = 0; i < recv_neighbors.size(); i++) {
                        int recv_zoid_num = recv_neighbors[i];

                        auto& recv_fp_idxs = zoid.recv_fp_idxs_double_buffering[t][i];

                        // int mpi_tag = get_mpi_tag(send_zoid_num, zoid.num, t, t);
                        int mpi_tag = get_mpi_tag_many_cuts(recv_zoid_num, zoid.num);

                        zoid_recv_data[zoid.num][t][i].reserve(recv_fp_idxs.size());
                        zoid_recv_data_sizes[zoid.num][t][i] = recv_fp_idxs.size();
                        for (int k = 0; k < recv_fp_idxs.size(); k++) {
                            zoid_recv_data[zoid.num][t][i].push_back(zoid.tag_stencil_md[0][recv_fp_idxs[k]]);
                        }

                        r.emplace_back();
                        MPI_Isend(&zoid_recv_data_sizes[zoid.num][t][i], 1, MPI_INT,
                                  recv_zoid_num % comm->nprocs, mpi_tag, world, &r[r.size() - 1]);

                        if (recv_fp_idxs.size() > 0) {
                            r.emplace_back();
                            // std::vector<int> send_pos_tags;
                            // send_pos_tags.reserve(size);
                            MPI_Isend(zoid_recv_data[zoid.num][t][i].data(), recv_fp_idxs.size(), MPI_INT,
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

                for (int t = 0; t < NUM_TIMESTEPS_IN_PARALLEL + 1; t++) {
                    zoid.send_fp_idxs_double_buffering[t] = new std::vector<int>[send_neighbors.size()];

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
                            zoid.send_fp_idxs_double_buffering[t][i].reserve(nrecv);
                            for (int k = 0; k < nrecv; k++) {
                                zoid.send_fp_idxs_double_buffering[t][i].push_back(tag_to_idx.at(recv_buf[k]));
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
    void CONSTRUCT_RECV_FP_IDXS_ZOID_MANY_CUTS_HELPER(queue_info& zoid) {
        assert(EXPERIMENT == EAM);
        int zoid_num = zoid.num;
        auto& recv_neighbors = curr_dt ? recv_from_neighbors_many_cuts[zoid.num]
                : recv_from_neighbors_many_cuts_next_dt[zoid.num];

        constexpr bool NEIGHBOR_CUTOFF = ALLEGRO_SLOPE / 2;

        std::map<int, int> tag_to_timestep_odd;
        std::map<int, int> tag_to_timestep_even;

        for (int t = 0; t < NUM_TIMESTEPS_IN_PARALLEL + 1; t++) {
            zoid.recv_fp_idxs_double_buffering[t] = new std::vector<int>[recv_neighbors.size()];

            auto &local_idxs = zoid.local_idxs_per_timestep[t];
            std::set<int> local_idxs_set;
            local_idxs_set.insert(local_idxs.begin(), local_idxs.end());

            auto& local_and_one_hop_ghost_idxs = zoid.local_and_one_hop_ghost_idxs_per_timestep[t];
            std::set<int> local_and_one_hop_ghost_idxs_set;
            local_and_one_hop_ghost_idxs_set.insert(local_and_one_hop_ghost_idxs.begin(), local_and_one_hop_ghost_idxs.end());

            for (int i = 0; i < zoid.x_stencil_md[0].size(); i++) {
                auto &pos = zoid.x_stencil_md[0][i];
                std::array<double, 3> atom_pos = {pos.x, pos.y, pos.z};
                double zoid_lo[3] = {0};
                double zoid_hi[3] = {0};

                bool dim_out_of_bounds[3] = {0};
                int out_of_bounds[3] = {0};

                bool borders_zoid = true;
                for (int dim = 0; dim < domain->dimension; dim++) {
                    double lo = zoid.lo[t][dim];
                    double hi = zoid.hi[t][dim];
                    double lo_borders = lo - NEIGHBOR_CUTOFF;
                    double hi_borders = hi + NEIGHBOR_CUTOFF;
                    borders_zoid = borders_zoid && atom_pos[dim] >= lo_borders && atom_pos[dim] < hi_borders;

                    zoid_lo[dim] = lo;
                    zoid_hi[dim] = hi;

                    if (atom_pos[dim] < lo || atom_pos[dim] >= hi) {
                        dim_out_of_bounds[dim] = true;
                    }
                }

                double dist_to_zoid = distance_to_zoid(domain->prd, zoid.lo[t], zoid.hi[t], atom_pos);
                borders_zoid = (borders_zoid && dist_to_zoid <= NEIGHBOR_CUTOFF);

                // have to do this check as for later timesteps this might not be the case
                if (!borders_zoid) {
                    continue;
                }

                // if i is not in my set, that means someone else evaluated the fp
                bool keep = (local_and_one_hop_ghost_idxs_set.find(i) == local_and_one_hop_ghost_idxs_set.end());

                // Do not recv data if I am going to evaluate it fully
                if (!keep) {
                    continue;
                }

                auto [needs_neighbor, _ ] = MANY_BODY_CHECK_IF_ATOM_NEEDS_NEIGHBOR(zoid, zoid.lo[t], zoid.hi[t], atom_pos);
                if (needs_neighbor) {
                    std::stringstream s1;
                    s1 << BOLDRED << "idx: " << i << " pos: " << atom_pos[0] << " " << atom_pos[1] << " " << atom_pos[2] 
                    << " in local? " << (local_idxs_set.find(i) != local_idxs_set.end())
                    << RESET_COLOR << std::endl;
                    std::cout << s1.str();
                }
                assert(!needs_neighbor);

                // find zoid that had it previously
                for (int j = 0; j < recv_neighbors.size(); j++) {
                    auto recv_zoid_num = recv_neighbors[j];
                    auto& recv_zoid = curr_dt ? zoid_num_to_zoid_many_cuts[recv_zoid_num] :
                                      zoid_num_to_zoid_many_cuts_next_dt[recv_zoid_num];

                    std::array<double, 3> adjusted_atom_pos;

                    for (int dim = 0; dim < domain->dimension; dim++) {
                        double lo = recv_zoid.lo[t][dim];
                        double hi = recv_zoid.hi[t][dim];

                        double p = atom_pos[dim];
                        while (p < lo) {
                            p += domain->prd[dim];
                        }
                        while (p >= hi) {
                            p -= domain->prd[dim];
                        }

                        adjusted_atom_pos[dim] = p;
                    }

                    auto [atom_needs_neighbors, _ ]  = MANY_BODY_CHECK_IF_ATOM_NEEDS_NEIGHBOR(recv_zoid, recv_zoid.lo[t], recv_zoid.hi[t], adjusted_atom_pos);

                    if (atom_needs_neighbors) {
                        zoid.recv_fp_idxs_double_buffering[t][j].push_back(i);

                        int tag = zoid.tag_stencil_md[0][i];
                        if (t % 2 == 0) {
                            if (tag_to_timestep_even.count(tag)) {
                                if (tag_to_timestep_even.at(tag) != t) {
                                    int other_t = tag_to_timestep_even.at(tag);
                                    std::cout << std::setprecision(20)
                                              << "curr_dt: " << curr_dt << " zoid: " << zoid.num
                                              << " EVEN tag: " << tag << " timestep: " << t << " overlapping recv pos timestep: " << tag_to_timestep_even.at(tag)
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
    void CONSTRUCT_RECV_FP_IDXS_ZOID_MANY_CUTS() {
        auto& queues = curr_dt ? queues_many_cuts : queues_many_cuts_next_dt;

        for (int dep = 0; dep < NUM_DEPS; dep++) {
            for (int j = 0; j < queues[dep].size(); j++) {
                auto& zoid = queues[dep][j];
                if (zoid.num % comm->nprocs != comm->me) {
                    continue;
                }

                CONSTRUCT_RECV_FP_IDXS_ZOID_MANY_CUTS_HELPER<curr_dt>(zoid);
            }
        }

        return;
    }

    // Sending forces is strictly ghost to local. There is no point propagating things.
    template <bool curr_dt, bool newton>
    void CONSTRUCT_SEND_RHO_IDXS_ZOID_MANY_CUTS_HELPER(queue_info& zoid) {
        int zoid_num = zoid.num;
        auto& send_neighbors = curr_dt ? send_to_neighbors_many_cuts[zoid_num]
                : send_to_neighbors_many_cuts_next_dt[zoid_num];

        std::map<int, int> idx_to_zoid;
        std::set<int> all_force_neighbors;

        for (int t = 0; t < NUM_TIMESTEPS_IN_PARALLEL + 1; t++) {
            zoid.send_rho_idxs_double_buffering[t] = new std::vector<int>[send_neighbors.size()];

            if (!newton) {
                continue;
            }

            std::set<int> send_idxs;
            auto& local_idxs = zoid.local_idxs_per_timestep[t];
            std::set<int> local_idxs_set;
            local_idxs_set.insert(local_idxs.begin(), local_idxs.end());

            auto& local_and_one_hop_ghost_idxs = zoid.local_and_one_hop_ghost_idxs_per_timestep[t];
            std::set<int> local_and_one_hop_ghost_idxs_set;
            local_and_one_hop_ghost_idxs_set.insert(local_and_one_hop_ghost_idxs.begin(), local_and_one_hop_ghost_idxs.end());

            for (int i = 0; i < zoid.x_stencil_md[0].size(); i++) {
                // only send ghost idxs for forces
                bool is_local = (local_idxs_set.find(i) != local_idxs_set.end());
                if (is_local) {
                    continue;
                }

                auto& pos = zoid.x_stencil_md[t % DOUBLE_BUFFERING][i];
                std::array<double, 3> atom_pos = {pos.x, pos.y, pos.z};

                bool two_hop_ghost = true;
                constexpr double NEIGHBOR_CUTOFF = ALLEGRO_SLOPE / 2;
                for (int dim = 0; dim < domain->dimension; dim++) {
                    // double lo = zoid.zoid.cuts[dim].lower + t * zoid.zoid.cuts[dim].slope_lower;
                    // double hi = zoid.zoid.cuts[dim].upper + t * zoid.zoid.cuts[dim].slope_upper;
                    double lo = zoid.lo[t][dim];
                    double hi = zoid.hi[t][dim];
                    double lo_one_hop = lo - NEIGHBOR_CUTOFF;
                    double hi_one_hop = hi + NEIGHBOR_CUTOFF;

                    double lo_two_hop = lo - ALLEGRO_SLOPE;
                    double hi_two_hop = hi + ALLEGRO_SLOPE;

                    if (atom_pos[dim] < lo) {
                        two_hop_ghost = two_hop_ghost && atom_pos[dim] >= lo_two_hop && atom_pos[dim] < lo_one_hop;
                    }

                    if (atom_pos[dim] >= hi) {
                        two_hop_ghost = two_hop_ghost && atom_pos[dim] >= hi_one_hop && atom_pos[dim] < hi_one_hop;
                    }
                }

                // have to do this check as for later timesteps this might not be the case
                if (!two_hop_ghost) {
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
                        zoid.send_rho_idxs_double_buffering[t][j].push_back(i);
                        break;
                    }
                }
            }
        }
    }

    template <bool curr_dt, bool newton>
    void CONSTRUCT_SEND_RHO_IDXS_ZOID_MANY_CUTS() {
        auto& queues = curr_dt ? queues_many_cuts : queues_many_cuts_next_dt;

        cilk_for (int dep = 0; dep < NUM_DEPS; dep++) {
            cilk_for (int j = 0; j < queues[dep].size(); j++) {
                auto& zoid = queues[dep][j];
                if (zoid.num % comm->nprocs != comm->me) {
                    continue;
                }

                CONSTRUCT_SEND_RHO_IDXS_ZOID_MANY_CUTS_HELPER<curr_dt, newton>(zoid);
            }
        }
    }

    template <bool curr_dt, bool newton>
    void CONSTRUCT_RECV_RHO_IDXS_ZOID_MANY_CUTS() {
        assert(EXPERIMENT == EAM);
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
                        zoid.recv_rho_idxs_double_buffering[t] = new std::vector<int>[recv_from_neighbors.size()];
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

                        auto& send_rho_idxs = zoid.send_rho_idxs_double_buffering[t][i];

                        // int mpi_tag = get_mpi_tag(send_zoid_num, zoid.num, t, t);
                        int mpi_tag = get_mpi_tag_many_cuts(send_zoid_num, zoid.num);

                        zoid_send_data[zoid.num][t][i].reserve(send_rho_idxs.size());
                        zoid_send_data_sizes[zoid.num][t][i] = send_rho_idxs.size();
                        for (int k = 0; k < send_rho_idxs.size(); k++) {
                            zoid_send_data[zoid.num][t][i].push_back(zoid.tag_stencil_md[0][send_rho_idxs[k]]);
                        }

                        r.emplace_back();
                        MPI_Isend(&zoid_send_data_sizes[zoid.num][t][i], 1, MPI_INT, send_zoid_num % comm->nprocs, mpi_tag, world, &r[r.size() - 1]);

                        if (send_rho_idxs.size() > 0) {
                            r.emplace_back();
                            // std::vector<int> send_pos_tags;
                            // send_pos_tags.reserve(size);
                            MPI_Isend(zoid_send_data[zoid.num][t][i].data(), send_rho_idxs.size(), MPI_INT, send_zoid_num % comm->nprocs, mpi_tag, world, &r[r.size() - 1]);
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
                    zoid.recv_rho_idxs_double_buffering[t] = new std::vector<int>[recv_from_neighbors.size()];

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
                                zoid.recv_rho_idxs_double_buffering[t][i].push_back(tag_to_idx.at(recv_buf[k]));

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
                        auto& recv_idxs = zoid.recv_rho_idxs_double_buffering[t][i];
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

    // 64 VCIs so 1 per comm
    static constexpr int NUM_COMMS = 1;
    std::vector<MPI_Comm> all_comms;

    MPI_Comm proc_to_proc_pipelined_comms[NUM_PIPELINE_STAGES][NUM_DEPS];

    void INIT_SEND_RECV_BUFFERS_MANY_CUTS() {
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

            if constexpr (EXPERIMENT == EAM) {
                int nsend_rho = zoid.send_rho_idxs_double_buffering[0][i].size();
                int nsend_fp = zoid.send_fp_idxs_double_buffering[0][i].size();
                zoid_ndoubles_send += DEBUG_SEND_RECV_DATA ? nsend_rho * 2 : nsend_rho;
                zoid_ndoubles_send += DEBUG_SEND_RECV_DATA ? nsend_fp * 2 : nsend_fp;
            }

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
                            world, &r[r.size() - 1]);
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
                MPI_Isend(buf, zoid_ndoubles_send, MPI_DOUBLE,
                        send_zoid_num % comm->nprocs, mpi_tag,
                        all_comms[comm_idx], &r[send_request_idx]);
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

                MPI_Isend(buf, total_nsend, MPI_DOUBLE,
                          proc, mpi_tag,
                          proc_to_proc_pipelined_comms[pipeline_stage][send_dep], &r[send_request_idx]);

                total_num_procs++;
            }
        }
    }

    template <bool curr_dt>
    void SEND_DATA_PROC_TO_PROC(int pipeline_stage, int send_dep,
                                std::vector<MPI_Request>& r, MPIX_Stream_Manager* manager) {

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

                // manager->m.lock();
                // MPI_Isend(buf, total_nsend, MPI_DOUBLE,
                //           proc, mpi_tag,
                //           proc_to_proc_pipelined_comms[pipeline_stage][send_dep], &r[send_request_idx]);
                // manager->m.unlock();

                // int src_stream_idx = NUM_STREAMS - 2;
                // int dst_stream_idx = NUM_STREAMS - 1;

                assert(sender_dep_proc_to_stream_num[curr_dt_idx].count({send_dep, proc}));
                auto [src_stream_idx, dst_stream_idx] = sender_dep_proc_to_stream_num[curr_dt_idx].at({send_dep, proc});

                if (USE_STREAMS) {
                    manager->m[src_stream_idx].lock();
                    auto res = MPIX_Stream_isend(buf, total_nsend, MPI_DOUBLE, proc, mpi_tag + send_dep, manager->stream_comm, src_stream_idx, dst_stream_idx, &r[send_request_idx]);
                    manager->m[src_stream_idx].unlock();
                    assert(res == MPI_SUCCESS);

                    // for (int i = 0; i < NUM_PROGRESS_STREAM_ITER; i++) {
                    //     manager->m[src_stream_idx].lock();
                    //     MPIX_Stream_progress(manager->streams[src_stream_idx]);
                    //     manager->m[src_stream_idx].unlock();
                    // }
                } else {
                    MPI_Isend(buf, total_nsend, MPI_DOUBLE, proc, mpi_tag, all_comms[dst_stream_idx], &r[send_request_idx]);
                }

                total_num_procs++;
            }
        }
    }

    template <bool curr_dt>
    void PACK_DATA_WITH_PROC_TO_PROC(queue_info& zoid, int send_dep, int start_timestep, int end_timestep, int pipeline_stage,
        MPIX_Stream_Manager* stream_manager, std::vector<MPI_Request>& send_r_zoid_to_zoid) {

        assert(pipeline_stage == DEFAULT_PIPELINE_STAGE);
        constexpr int curr_dt_idx = static_cast<int>(curr_dt);

        auto& send_neighbors = curr_dt ? send_to_neighbors_many_cuts[zoid.num]
            : send_to_neighbors_many_cuts_next_dt[zoid.num];

        auto& send_request_idxs = send_to_neighbors_not_my_proc_idxs_only_next_dep[curr_dt_idx][zoid.num];

        int zoid_num = zoid.num;

        const auto& procs_to_send_to = send_dep_to_procs[curr_dt_idx][pipeline_stage][send_dep];

        for (int i = 0; i < send_neighbors.size(); i++) {
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

            cilk_spawn [this](int dep, queue_info& zoid, double zoid_ndoubles_send, int i, int send_zoid_num, int start_timestep, int end_timestep,
                int pipeline_stage, MPIX_Stream_Manager* manager, int send_request_idx, std::vector<MPI_Request>& r) noexcept {
                int zoid_num = zoid.num;
                auto *buf = buf_send_zoid_to_zoid[pipeline_stage][zoid_num][i];
                PACK_DATA_MANY_CUTS_HELPER_PIPELINED<curr_dt>(zoid, buf, i, send_zoid_num, start_timestep, end_timestep, pipeline_stage);
                int mpi_tag = get_mpi_tag_many_cuts(send_zoid_num, zoid.num);
                auto [src_stream_idx, dst_stream_idx] = zoid_to_zoid_to_stream_num[curr_dt_idx].at({zoid_num, send_zoid_num});

                if (USE_STREAMS) {
                    manager->m[src_stream_idx].lock();
                    auto res = MPIX_Stream_isend(buf, zoid_ndoubles_send, MPI_DOUBLE, send_zoid_num % comm->nprocs, mpi_tag, manager->stream_comm,
                        src_stream_idx, dst_stream_idx, &r[send_request_idx]);
                    manager->m[src_stream_idx].unlock();
                    assert(res == MPI_SUCCESS);

                    for (int i = 0; i < NUM_PROGRESS_STREAM_ITER; i++) {
                        if (manager->m[src_stream_idx].try_lock()) {
                            MPIX_Stream_progress(manager->streams[src_stream_idx]);
                            manager->m[src_stream_idx].unlock();
                        }
                    }
                } else {
                    MPI_Isend(buf, zoid_ndoubles_send, MPI_DOUBLE, send_zoid_num % comm->nprocs, mpi_tag, 
                        all_comms[dst_stream_idx], &r[send_request_idx]);
                }
            }(send_dep, zoid, zoid_ndoubles_send, i, send_zoid_num, start_timestep, end_timestep, pipeline_stage, stream_manager, send_request_idxs[i], send_r_zoid_to_zoid);
        }

        for (int i = 0; i < procs_to_send_to.size(); i++) {
            PACK_DATA_PROC_TO_PROC_HELPER<curr_dt>(zoid, pipeline_stage, send_dep, procs_to_send_to[i]);
        }
    }

    template <bool curr_dt>
    void SEND_DATA_ZOID_TO_ZOID(queue_info& zoid, int send_dep, int pipeline_stage, std::vector<MPI_Request>& r, MPIX_Stream_Manager* manager) {
        constexpr int curr_dt_idx = static_cast<int>(curr_dt);

        auto& send_neighbors = curr_dt ? send_to_neighbors_many_cuts[zoid.num]
            : send_to_neighbors_many_cuts_next_dt[zoid.num];

        auto& send_request_idxs = send_to_neighbors_not_my_proc_idxs_only_next_dep[curr_dt_idx][zoid.num];

        int zoid_num = zoid.num;

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

                assert(zoid_to_zoid_to_stream_num[curr_dt_idx].count({zoid_num, send_zoid_num}));
                auto [src_stream_idx, dst_stream_idx] = zoid_to_zoid_to_stream_num[curr_dt_idx].at({zoid_num, send_zoid_num});

                if (USE_STREAMS) {
                    manager->m[src_stream_idx].lock();
                    auto res = MPIX_Stream_isend(buf, zoid_ndoubles_send, MPI_DOUBLE, send_zoid_num % comm->nprocs, mpi_tag, manager->stream_comm,
                        src_stream_idx, dst_stream_idx, &r[send_request_idx]);
                    manager->m[src_stream_idx].unlock();
                    assert(res == MPI_SUCCESS);
                    /*
                    manager->m[dst_stream_idx].lock();
                    auto res = MPI_Isend(buf, zoid_ndoubles_send, MPI_DOUBLE, send_zoid_num % comm->nprocs, mpi_tag, manager->comms[dst_stream_idx], &r[send_request_idx]);
                    manager->m[dst_stream_idx].unlock();
                    assert(res == MPI_SUCCESS);
                    */
                } else {
                    MPI_Isend(buf, zoid_ndoubles_send, MPI_DOUBLE, send_zoid_num % comm->nprocs, mpi_tag, 
                        all_comms[dst_stream_idx], &r[send_request_idx]);
                }
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

            if constexpr (EXPERIMENT == EAM) {
                auto& zoid = zoid_num_to_zoid_many_cuts[zoid_num];
                int nrecv_rho = zoid.recv_rho_idxs_double_buffering[0][i].size();
                int nrecv_fp = zoid.recv_fp_idxs_double_buffering[0][i].size();
                total_doubles_recv_from_zoid += DEBUG_SEND_RECV_DATA ? nrecv_rho * 2 : nrecv_rho;
                total_doubles_recv_from_zoid += DEBUG_SEND_RECV_DATA ? nrecv_fp * 2 : nrecv_fp;
            }

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
                            world, &r[r.size() - 1]);
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
                MPI_Irecv(buf, total_doubles_recv_from_zoid, MPI_DOUBLE,
                        recv_zoid_num % comm->nprocs, mpi_tag,
                        all_comms[comm_idx], &r[recv_request_idx]);
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
    int RECEIVE_DATA_ZOID_TO_ZOID(int dep, queue_info& zoid, int pipeline_stage, std::vector<MPI_Request>& r, MPIX_Stream_Manager* manager) {
        __builtin_unreachable();
        /*
        int zoid_num = zoid.num;
        auto& queues = curr_dt ? queues_many_cuts : queues_many_cuts_next_dt;
        auto& recv_neighbors = curr_dt ? recv_from_neighbors_many_cuts[zoid_num] : recv_from_neighbors_many_cuts_next_dt[zoid_num];
        constexpr int curr_dt_idx = static_cast<int>(curr_dt);


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
                assert(recv_request_zoid_to_idx_per_zoid[curr_dt_idx][zoid.num].count(recv_zoid_num));
                int recv_request_idx = recv_request_zoid_to_idx_per_zoid[curr_dt_idx][zoid.num].at(recv_zoid_num);
                int src_stream_idx = zoid_to_stream_num[curr_dt_idx][recv_zoid_num];
                int dst_stream_idx = zoid_to_stream_num[curr_dt_idx][zoid_num];

                assert(zoid_to_zoid_to_stream_num[curr_dt_idx].count({recv_zoid_num, zoid_num}));
                // auto [src_stream_idx, dst_stream_idx] = zoid_to_zoid_to_stream_num[curr_dt_idx].at({recv_zoid_num, zoid_num});
                // int recv_request_idx = zoid_pair_to_recv_request_idx_streams[curr_dt_idx][{recv_zoid_num, zoid_num}];
                manager->m[dst_stream_idx].lock();
                MPIX_Stream_irecv(buf, total_doubles_recv_from_zoid, MPI_DOUBLE, recv_zoid_num % comm->nprocs, mpi_tag,
                    manager->stream_comm, src_stream_idx, dst_stream_idx, &r[recv_request_idx]);
                manager->m[dst_stream_idx].unlock();
                num_recv_neighbors++;
            }
        }

        return num_recv_neighbors;
        */
        return -1;
    }

    template <bool curr_dt>
    void RECEIVE_DATA_PROC_TO_PROC_AND_ZOID_TO_ZOID_STREAMS(int dep, int stream_num, int pipeline_stage, std::vector<MPI_Request>& r, MPIX_Stream_Manager* manager) {
        constexpr int curr_dt_idx = static_cast<int>(curr_dt);

        int recv_request_idx = 0;
        auto& zoid_pairs = stream_num_to_zoid_pairs[curr_dt_idx][dep][stream_num];

        for (int i = 0; i < zoid_pairs.size(); i++) {
            auto [recv_zoid_num, zoid_num] = zoid_pairs[i];
            assert(zoid_to_zoid_to_stream_num[curr_dt_idx].count({recv_zoid_num, zoid_num}));
            auto [src_stream_idx, dst_stream_idx] = zoid_to_zoid_to_stream_num[curr_dt_idx].at({recv_zoid_num, zoid_num});

            assert(dst_stream_idx == stream_num);

            int mpi_tag = get_mpi_tag_many_cuts(zoid_num, recv_zoid_num);
            auto& recv_neighbors = curr_dt ? recv_from_neighbors_many_cuts[zoid_num] : recv_from_neighbors_many_cuts_next_dt[zoid_num];
            auto find_it = std::find(recv_neighbors.begin(), recv_neighbors.end(), recv_zoid_num);
            assert(find_it != recv_neighbors.end());
            int find_idx = std::distance(recv_neighbors.begin(), find_it);
            auto* buf = buf_recv_zoid_to_zoid[pipeline_stage][zoid_num][find_idx];
            int recv_size = curr_dt ? recv_zoid_to_zoid_sizes_pipelined[pipeline_stage][zoid_num][find_idx]
                : recv_zoid_to_zoid_sizes_pipelined_next_dt[pipeline_stage][zoid_num][find_idx];
            int total_doubles_recv_from_zoid = DEBUG_SEND_RECV_DATA ? recv_size * (3 + 1) : recv_size * 3;
            if (total_doubles_recv_from_zoid > nrecv_buf_recv_zoid_to_zoid[pipeline_stage][zoid_num][find_idx]) {
                assert(false);
                GROW_RECV_ZOID_TO_ZOID_MANY_CUTS(zoid_num, find_idx, total_doubles_recv_from_zoid, pipeline_stage);
            }

            if (USE_STREAMS) {
                manager->m[stream_num].lock();
                MPIX_Stream_irecv(buf, total_doubles_recv_from_zoid, MPI_DOUBLE, recv_zoid_num % comm->nprocs, mpi_tag,
                    manager->stream_comm, src_stream_idx, stream_num, &r[recv_request_idx]);
                manager->m[stream_num].unlock();

                // for (int i = 0; i < NUM_PROGRESS_STREAM_ITER; i++) {
                //     if (manager->m[stream_num].try_lock()) {
                //         MPIX_Stream_progress(manager->streams[stream_num]);
                //         manager->m[stream_num].unlock();
                //     }
                // }
            } else {
                MPI_Irecv(buf, total_doubles_recv_from_zoid, MPI_DOUBLE, recv_zoid_num % comm->nprocs, mpi_tag, all_comms[stream_num], &r[recv_request_idx]);
            }

            recv_request_idx++;
        }

        auto& dep_proc_pairs = stream_num_to_dep_proc_pairs[curr_dt_idx][dep][stream_num];
        for (int i = 0; i < dep_proc_pairs.size(); i++) {
            auto [send_dep, send_proc] = dep_proc_pairs[i];
            assert(send_proc != comm->me);
            int size = -1;

            auto& lst_proc_send_dep_info = dep_to_recv_proc_to_proc[curr_dt_idx][pipeline_stage][dep];
            for (int j = 0; j < lst_proc_send_dep_info.size(); j++) {
                auto [send_dep_info, send_proc_info] = lst_proc_send_dep_info[j];
                if (send_dep_info == send_dep && send_proc_info == send_proc) {
                    size = dep_to_recv_proc_to_proc_sizes[curr_dt_idx][pipeline_stage][dep][j];
                }
            }

            if (size == -1) {
                std::stringstream s1;
                s1 << BOLDRED << "me: " << comm->me << " size: " << size << " dep: " << dep
                << " send dep: " << send_dep << " send proc: " << send_proc << " lst info size?? " << lst_proc_send_dep_info.size()
                << RESET_COLOR << std::endl;
                std::cout << s1.str();
            }
            assert(size != -1);

            int nrecv_from_proc = DEBUG_SEND_RECV_DATA ? size * (3 + 1) : size * 3;
            int mpi_tag = get_mpi_tag_many_cuts(comm->me, send_proc);
            auto* buf = buf_recv_proc_to_proc[pipeline_stage][send_dep][send_proc];

            assert(receiver_dep_proc_to_stream_num[curr_dt_idx][dep].count({send_dep, send_proc}));
            auto [src_stream_idx, dst_stream_idx] = receiver_dep_proc_to_stream_num[curr_dt_idx][dep].at({send_dep, send_proc});

            assert(dst_stream_idx == stream_num);

            // if (send_proc == 8 && mpi_tag == 229384) {
            //     std::stringstream s1;
            //     s1 << BOLDRED << "curr_dt: " << curr_dt << " me: " << comm->me << " dep: " << dep
            //     << " proc to proc: " << send_proc << " to: " << comm->me << " ndoubles: " << nrecv_from_proc
            //     << RESET_COLOR << std::endl;
            //     std::cout << s1.str();
            // }

            if (USE_STREAMS) {
                manager->m[stream_num].lock();
                MPIX_Stream_irecv(buf, nrecv_from_proc, MPI_DOUBLE, send_proc, mpi_tag + send_dep,
                    manager->stream_comm, src_stream_idx, stream_num, &r[recv_request_idx]);
                // MPI_Irecv(buf, nrecv_from_proc, MPI_DOUBLE, send_proc, mpi_tag, manager->comms[stream_num], &r[recv_request_idx]);
                manager->m[stream_num].unlock();
            } else {
                MPI_Irecv(buf, nrecv_from_proc, MPI_DOUBLE, send_proc, mpi_tag, all_comms[stream_num], &r[recv_request_idx]);
            }

            recv_request_idx++;
        }
    }

    void MPIX_START_PROGRESS_THREAD(MPIX_Stream_Manager* manager) {
        while (true) {
            if (manager->done) {
                break;
            }

            if (manager->global_lock.try_lock()) {
                for (int i = 0; i < manager->num_streams; i++) {
                    if (manager->m[i].try_lock()) {
                        MPIX_Stream_progress(manager->streams[i]);
                        manager->m[i].unlock();
                    }
                }
                manager->global_lock.unlock();
            }

            std::this_thread::sleep_for(std::chrono::microseconds(1));
        }
    }

    void MPIX_STOP_PROGRESS_THREAD(MPIX_Stream_Manager* manager) {
        manager->done = true;
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
                                                     int send_idx, int start_t, int end_t, int pipeline_stage, bool unpack_force=true) {

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

        if (unpack_force) {
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

                if constexpr (EXPERIMENT == EAM) {
                    auto& recv_rho_idxs = zoid.recv_rho_idxs_double_buffering[t][recv_idx];
                    for (int k = 0; k < recv_rho_idxs.size(); k++) {
                        int idx = recv_rho_idxs[k];
                        auto target_tag = (tagint) ubuf(buf[buf_idx++]).i;
                        double rho = buf[buf_idx++];

                        if (target_tag != zoid.tag_stencil_md[0][idx]) {
                            std::cout << "RHO TAG WRONG me: " << comm->me
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
                        zoid.rho_stencil_md[0][idx] += rho;
                    }

                    auto& recv_fp_idxs = zoid.recv_fp_idxs_double_buffering[t][recv_idx];
                    for (int k = 0; k < recv_rho_idxs.size(); k++) {
                        int idx = recv_fp_idxs[k];
                        auto target_tag = (tagint) ubuf(buf[buf_idx++]).i;
                        double fp = buf[buf_idx++];

                        if (target_tag != zoid.tag_stencil_md[0][idx]) {
                            std::cout << "RHO TAG WRONG me: " << comm->me
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
                        zoid.fp_stencil_md[0][idx] = fp;
                    }
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

                if constexpr (EXPERIMENT == EAM) {
                    auto& recv_rho_idxs = zoid.recv_rho_idxs_double_buffering[t][recv_idx];
                    for (int k = 0; k < recv_rho_idxs.size(); k++) {
                        int idx = recv_rho_idxs[k];
                        double rho = buf[buf_idx++];
                        zoid.rho_stencil_md[0][idx] += rho;
                    }

                    auto& recv_fp_idxs = zoid.recv_fp_idxs_double_buffering[t][recv_idx];
                    for (int k = 0; k < recv_fp_idxs.size(); k++) {
                        int idx = recv_fp_idxs[k];
                        double fp = buf[buf_idx++];
                        zoid.fp_stencil_md[0][idx] = fp;
                    }
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
    void UNPACK_FORCE_MANY_CUTS_ZOID_PIPELINED_ONLY_NEXT_DEP(queue_info& zoid, int dep, int start_t, int end_t, int pipeline_stage, bool unpack_self_force=false) {
        int zoid_num = zoid.num;
        auto& recv_neighbors = curr_dt ? recv_from_neighbors_many_cuts[zoid_num]
            : recv_from_neighbors_many_cuts_next_dt[zoid_num];

        constexpr int curr_dt_idx = static_cast<int>(curr_dt);

        for (int i = 0; i < recv_neighbors.size(); i++) {
            int recv_zoid_num = recv_neighbors[i];
            if (recv_zoid_num % comm->nprocs != comm->me) {
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

            if (recv_zoid_num % comm->nprocs == comm->me && unpack_self_force) {
                auto& recv_zoid = curr_dt ? zoid_num_to_zoid_many_cuts[recv_zoid_num] : zoid_num_to_zoid_many_cuts_next_dt[recv_zoid_num];
                auto &send_neighbors = curr_dt ? send_to_neighbors_many_cuts[recv_zoid_num] : send_to_neighbors_many_cuts_next_dt[recv_zoid_num];
                auto find_it = std::find(send_neighbors.begin(), send_neighbors.end(), zoid_num);
                assert(find_it != send_neighbors.end());
                int send_idx = std::distance(send_neighbors.begin(), find_it);
                auto& send_force_idxs = recv_zoid.send_force_idxs_double_buffering_flattened_pipelined[pipeline_stage][send_idx];
                auto& recv_force_idxs = zoid.recv_force_idxs_double_buffering_flattened_pipelined[pipeline_stage][i];
                auto * _noalias const recv_f_ = zoid.f_stencil_md[0].data();
                auto * _noalias const send_f_ = recv_zoid.f_stencil_md[0].data();
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
                                                            start_timestep, end_timestep, pipeline_stage, offset + zoid_ndoubles_send);

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
                                             int pipeline_stage, int buf_offset=0) {
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

            int pos_starting_idx = num_send_force * 3;
            int pos_starting_idx2 = (num_send_force + num_send_pos) * 3;
            int vel_starting_idx = (num_send_force + num_send_pos + num_send_pos2) * 3;

            /*
            cilk_scope {
                cilk_spawn [&]() noexcept {
                    for (int i = 0; i < send_force_idxs.size(); i++) {
                        int idx = send_force_idxs[i];
                        int buf_idx = i * 3;

                        buf[buf_idx] = f_[idx].x;
                        buf[buf_idx + 1] = f_[idx].y;
                        buf[buf_idx + 2] = f_[idx].z;

                        f_[idx].x = 0;
                        f_[idx].y = 0;
                        f_[idx].z = 0;
                    }
                }();

                cilk_spawn [&]() noexcept {
                    for (int i = 0; i < send_pos_idxs.size(); i++) {
                        int idx = send_pos_idxs[i];
                        int buf_idx = pos_starting_idx + i * 3;

                        buf[buf_idx] = x0_[idx].x;
                        buf[buf_idx + 1] = x0_[idx].y;
                        buf[buf_idx + 2] = x0_[idx].z;
                    }
                }();

                cilk_spawn [&]() noexcept {
                    for (int i = 0; i < send_pos_idxs2.size(); i++) {
                        int idx = send_pos_idxs2[i];
                        int buf_idx = pos_starting_idx2 + i * 3;

                        buf[buf_idx] = x1_[idx].x;
                        buf[buf_idx + 1] = x1_[idx].y;
                        buf[buf_idx + 2] = x1_[idx].z;
                    }
                }();

                cilk_spawn [&]() noexcept {
                    for (int i = 0; i < send_vel_idxs.size(); i++) {
                        int idx = send_vel_idxs[i];
                        int buf_idx = vel_starting_idx + i * 3;

                        buf[buf_idx] = v_[idx].x;
                        buf[buf_idx + 1] = v_[idx].y;
                        buf[buf_idx + 2] = v_[idx].z;
                    }
                }();
            }
            */
            for (int i = 0; i < send_force_idxs.size(); i++) {
                int idx = send_force_idxs[i];
                int buf_idx = i * 3;

                buf[buf_idx] = f_[idx].x;
                buf[buf_idx + 1] = f_[idx].y;
                buf[buf_idx + 2] = f_[idx].z;

                f_[idx].x = 0;
                f_[idx].y = 0;
                f_[idx].z = 0;
            }

            for (int i = 0; i < send_pos_idxs.size(); i++) {
                int idx = send_pos_idxs[i];
                int buf_idx = pos_starting_idx + i * 3;

                buf[buf_idx] = x0_[idx].x;
                buf[buf_idx + 1] = x0_[idx].y;
                buf[buf_idx + 2] = x0_[idx].z;
            }

            for (int i = 0; i < send_pos_idxs2.size(); i++) {
                int idx = send_pos_idxs2[i];
                int buf_idx = pos_starting_idx2 + i * 3;

                buf[buf_idx] = x1_[idx].x;
                buf[buf_idx + 1] = x1_[idx].y;
                buf[buf_idx + 2] = x1_[idx].z;
            }

            for (int i = 0; i < send_vel_idxs.size(); i++) {
                int idx = send_vel_idxs[i];
                int buf_idx = vel_starting_idx + i * 3;

                buf[buf_idx] = v_[idx].x;
                buf[buf_idx + 1] = v_[idx].y;
                buf[buf_idx + 2] = v_[idx].z;
            }

            /*
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
            }

            #pragma cilk grainsize 2048
            cilk_for (int i = 0; i < send_pos_idxs.size(); i++) {
                int idx = send_pos_idxs[i];
                int buf_idx = pos_starting_idx + i * 3;

                buf[buf_idx] = x0_[idx].x;
                buf[buf_idx + 1] = x0_[idx].y;
                buf[buf_idx + 2] = x0_[idx].z;
            }

            #pragma cilk grainsize 2048
            cilk_for (int i = 0; i < send_pos_idxs2.size(); i++) {
                int idx = send_pos_idxs2[i];
                int buf_idx = pos_starting_idx2 + i * 3;

                buf[buf_idx] = x1_[idx].x;
                buf[buf_idx + 1] = x1_[idx].y;
                buf[buf_idx + 2] = x1_[idx].z;
            }


            #pragma cilk grainsize 2048
            cilk_for (int i = 0; i < send_vel_idxs.size(); i++) {
                int idx = send_vel_idxs[i];
                int buf_idx = vel_starting_idx + i * 3;

                buf[buf_idx] = v_[idx].x;
                buf[buf_idx + 1] = v_[idx].y;
                buf[buf_idx + 2] = v_[idx].z;
            }
            */
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

                if constexpr (EXPERIMENT == EAM) {
                    auto& send_rho_idxs = zoid.send_rho_idxs_double_buffering[t][send_idx];
                    for (int k = 0; k < send_rho_idxs.size(); k++) {
                        int idx = send_rho_idxs[k];
                        int tag = zoid.tag_stencil_md[0][idx];
                        buf[buf_idx++] = ubuf(tag).d;
                        buf[buf_idx++] = zoid.rho_stencil_md[0][idx];
                    }

                    auto& send_fp_idxs = zoid.send_fp_idxs_double_buffering[t][send_idx];
                    for (int k = 0; k < send_fp_idxs.size(); k++) {
                        int idx = send_fp_idxs[k];
                        int tag = zoid.tag_stencil_md[0][idx];
                        buf[buf_idx++] = ubuf(tag).d;
                        buf[buf_idx++] = zoid.fp_stencil_md[0][idx];
                    }
                }
            } else {
                for (int k = 0; k < send_force_idxs.size(); k++) {
                    int idx = send_force_idxs[k];

                    buf[buf_idx++] = zoid.f_stencil_md[0][idx].x;
                    buf[buf_idx++] = zoid.f_stencil_md[0][idx].y;
                    buf[buf_idx++] = zoid.f_stencil_md[0][idx].z;
                }

                if constexpr (EXPERIMENT == EAM) {
                    auto& send_rho_idxs = zoid.send_rho_idxs_double_buffering[t][send_idx];
                    for (int k = 0; k < send_rho_idxs.size(); k++) {
                        int idx = send_rho_idxs[k];
                        buf[buf_idx++] = zoid.rho_stencil_md[0][idx];
                    }

                    auto& send_fp_idxs = zoid.send_fp_idxs_double_buffering[t][send_idx];
                    for (int k = 0; k < send_fp_idxs.size(); k++) {
                        int idx = send_fp_idxs[k];
                        buf[buf_idx++] = zoid.fp_stencil_md[0][idx];
                    }
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

    void SW_FORCE_COMPUTE_ZOID_MANY_CUTS(queue_info& zoid, int dep, int timestep) {
        const auto * _noalias const x = zoid.x_stencil_md[timestep % DOUBLE_BUFFERING].data();
        auto * _noalias const f = zoid.f_stencil_md[timestep % 1].data();

        const auto& neighbor_list = zoid.neighbor_list[timestep];
        auto* _noalias spinlocks = zoid.spinlocks_stencil_md[0];

        // const auto& local_idxs = zoid.local_idxs_per_timestep[timestep];
        const auto& local_idxs = zoid.local_and_one_hop_ghost_idxs_per_timestep[timestep];
        // const auto& is_local_idx = zoid.is_local_per_timestep[timestep];
        const int nlocal = local_idxs.size();

        const auto& tags = zoid.tag_stencil_md[0];
        const auto& atom_type = zoid.type_stencil_md[0];

        // auto* neighshort = zoid.neigh_short[0].data();
        // int num_neigh_short = zoid.neigh_short[0].capacity();

        auto& neigh_short = zoid.neigh_short[0];

        PairSW* pair_sw = (PairSW*) force->pair;
        auto map = pair_sw->map;
        auto elem3param = pair_sw->elem3param;
        auto params = pair_sw->params;

        // loop over full neighbor list of my atoms

        double powerp = params->powerp;
        double powerq = params->powerq;
        double cut = params->cut;
        double sigma = params->sigma;
        double c1 = params->c1;
        double c2 = params->c2;
        double c3 = params->c3;
        double c4 = params->c4;

        #pragma cilk grainsize 1024
        cilk_for (int ii = 0; ii < nlocal; ii++) {
            int i = local_idxs[ii];

            int itag = tags[i];
            int itype = map[atom_type[i]];
            double xtmp = x[i].x;
            double ytmp = x[i].y;
            double ztmp = x[i].z;
            double fxtmp = 0;
            double fytmp = 0;
            double fztmp = 0;

            // two-body interactions, skip half of them
            auto* neigh_short_atom = neigh_short[i].data();
            int num_neigh_short = neigh_short[i].capacity();

            auto& neigh_list = neighbor_list[i];
            int num_neigh = neigh_list.size();

            int numshort = 0;
            for (int jj = 0; jj < num_neigh; jj++) {
                int j = neigh_list[jj];
                j &= NEIGHMASK;
                double delx = xtmp - x[j].x;
                double dely = ytmp - x[j].y;
                double delz = ztmp - x[j].z;
                double rsq = delx*delx + dely*dely + delz*delz;
                int jtype = map[atom_type[j]];
                int ijparam = elem3param[itype][jtype][jtype];
                if (rsq >= params[ijparam].cutsq) {
                    continue;
                } else {
                    neigh_short_atom[numshort++] = j;
                    assert(numshort <= num_neigh_short);
                }

                int jtag = tags[j];
                if (itag > jtag) {
                    if ((itag+jtag) % 2 == 0) continue;
                } else if (itag < jtag) {
                    if ((itag+jtag) % 2 == 1) continue;
                } else {
                    if (x[j].z < ztmp) continue;
                    if (x[j].z == ztmp && x[j].y < ytmp) continue;
                    if (x[j].z == ztmp && x[j].y == ytmp && x[j].x < xtmp) continue;
                }

                // two-body implementation
                double r = sqrt(rsq);
                double rinvsq = 1.0/rsq;
                double rp = pow(r, -powerp);
                double rq = pow(r, -powerq);
                double rainv = 1.0 / (r - cut);
                double rainvsq = rainv * rainv * r;
                double expsrainv = exp(sigma * rainv);
                double fpair = (c1*rp - c2*rq + (c3*rp -c4*rq) * rainvsq) * expsrainv * rinvsq;
                // if (eflag) eng = (param->c5*rp - param->c6*rq) * expsrainv;
                fxtmp += delx*fpair;
                fytmp += dely*fpair;
                fztmp += delz*fpair;
                spinlocks[j].lock();
                f[j].x -= delx*fpair;
                f[j].y -= dely*fpair;
                f[j].z -= delz*fpair;
                spinlocks[j].unlock();
                // if (evflag) ev_tally(i,j,nlocal,newton_pair, evdwl,0.0,fpair,delx,dely,delz);
            }

            int jnumm1 = numshort - 1;

            for (int jj = 0; jj < jnumm1; jj++) {
                int j = neigh_short_atom[jj];
                int jtype = map[atom_type[j]];
                int ijparam = elem3param[itype][jtype][jtype];
                double delr1[3] = {x[j].x - xtmp, x[j].y - ytmp, x[j].z - ztmp};
                double rsq1 = delr1[0]*delr1[0] + delr1[1]*delr1[1] + delr1[2]*delr1[2];

                double r1 = sqrt(rsq1);
                double rinvsq1 = 1.0/rsq1;
                auto& paramsij = params[ijparam];
                double rainv1 = 1.0/(r1 - paramsij.cut);
                double gsrainv1 = paramsij.sigma_gamma * rainv1;
                double gsrainvsq1 = gsrainv1 * rainv1 / r1;
                double expgsrainv1 = exp(gsrainv1);

                double fjxtmp = 0;
                double fjytmp = 0;
                double fjztmp = 0;

                for (int kk = jj+1; kk < numshort; kk++) {
                    int k = neigh_short_atom[kk];
                    int ktype = map[atom_type[k]];
                    int ikparam = elem3param[itype][ktype][ktype];
                    int ijkparam = elem3param[itype][jtype][ktype];

                    auto& paramik = params[ikparam];
                    auto& paramijk = params[ijkparam];

                    double delr2[3] = {x[k].x - xtmp, x[k].y - ytmp, x[k].z - ztmp};
                    double rsq2 = delr2[0]*delr2[0] + delr2[1]*delr2[1] + delr2[2]*delr2[2];

                    double r2 = sqrt(rsq2);
                    double rinvsq2 = 1.0/rsq2;
                    double rainv2 = 1.0/(r2 - paramik.cut);
                    double gsrainv2 = paramik.sigma_gamma * rainv2;
                    double gsrainvsq2 = gsrainv2*rainv2/r2;
                    double expgsrainv2 = exp(gsrainv2);

                    double rinv12 = 1.0/(r1*r2);
                    double cs = (delr1[0]*delr2[0] + delr1[1]*delr2[1] + delr1[2]*delr2[2]) * rinv12;
                    double delcs = cs - paramijk.costheta;
                    double delcssq = delcs*delcs;

                    double facexp = expgsrainv1*expgsrainv2;

                    // facrad = sqrt(paramij->lambda_epsilon*paramik->lambda_epsilon) *
                    //          facexp*delcssq;

                    double facrad = paramijk.lambda_epsilon * facexp*delcssq;
                    double frad1 = facrad*gsrainvsq1;
                    double frad2 = facrad*gsrainvsq2;
                    double facang = paramijk.lambda_epsilon2 * facexp*delcs;
                    double facang12 = rinv12*facang;
                    double csfacang = cs*facang;
                    double csfac1 = rinvsq1*csfacang;

                    dbl3_t_stencil_md fj = {delr1[0]*(frad1+csfac1)-delr2[0]*facang12,
                        delr1[1]*(frad1+csfac1)-delr2[1]*facang12, 
                        delr1[2]*(frad1+csfac1)-delr2[2]*facang12};

                    double csfac2 = rinvsq2*csfacang;

                    dbl3_t_stencil_md fk = {delr2[0]*(frad2+csfac2)-delr1[0]*facang12, 
                        delr2[1]*(frad2+csfac2)-delr1[1]*facang12,
                        delr2[2]*(frad2+csfac2)-delr1[2]*facang12};

                    fxtmp -= fj.x + fk.x;
                    fytmp -= fj.y + fk.y;
                    fztmp -= fj.z + fk.z;
                    fjxtmp += fj.x;
                    fjytmp += fj.y;
                    fjztmp += fj.z;

                    spinlocks[k].lock();
                    f[k].x += fk.x;
                    f[k].y += fk.y;
                    f[k].z += fk.z;
                    spinlocks[k].unlock();

                    // if (evflag) ev_tally3(i,j,k,evdwl,0.0,fj,fk,delr1,delr2);
                }

                spinlocks[j].lock();
                f[j].x += fjxtmp;
                f[j].y += fjytmp;
                f[j].z += fjztmp;
                spinlocks[j].unlock();
            }

            spinlocks[i].lock();
            f[i].x += fxtmp;
            f[i].y += fytmp;
            f[i].z += fztmp;
            spinlocks[i].unlock();
        }
        // if (vflag_fdotr) virial_fdotr_compute();
    }

    void EAM_FORCE_COMPUTE_ZOID_MANY_CUTS(queue_info& zoid, int dep, int timestep) {
        assert(EXPERIMENT == EAM);
        const auto * _noalias const x = zoid.x_stencil_md[timestep % DOUBLE_BUFFERING].data();
        auto * _noalias const f = zoid.f_stencil_md[timestep % 1].data();
        auto * _noalias const rho = zoid.rho_stencil_md[timestep % DOUBLE_BUFFERING].data();
        auto * _noalias const fp = zoid.fp_stencil_md[timestep % DOUBLE_BUFFERING].data();

        const auto& neighbor_list = zoid.neighbor_list[timestep];
        auto* _noalias spinlocks = zoid.spinlocks_stencil_md[0];

        const auto& local_and_ghost_idxs = zoid.local_and_one_hop_ghost_idxs_per_timestep[timestep];
        const int nlocal_and_ghost = local_and_ghost_idxs.size();

        // TODO: Ryan, I only am evaluating the local atoms, NOT the local + 1 hop ghost atoms.
        const auto& local_idxs = zoid.local_idxs_per_timestep[timestep];
        const int nlocal = local_idxs.size();

        const auto& tags = zoid.tag_stencil_md[0];
        const auto& atom_type = zoid.type_stencil_md[0];

        PairEAM* pair_eam = (PairEAM*) force->pair;
        double cutforcesq = pair_eam->cutforcesq;
        double rdr = pair_eam->rdr;
        int nr = pair_eam->nr;
        auto rhor_spline = pair_eam->rhor_spline;
        auto type2rhor = pair_eam->type2rhor;
        double rdrho = pair_eam->rdrho;
        int nrho = pair_eam->nrho;
        auto type2frho = pair_eam->type2frho;
        auto frho_spline = pair_eam->frho_spline;
        auto z2r_spline = pair_eam->z2r_spline;
        auto type2z2r = pair_eam->type2z2r;
        auto scale = pair_eam->scale;

        // memset(rho, 0, sizeof(double) * zoid.rho_stencil_md[0].size());

        constexpr int target_tag = 186753;
        
        // set rho to 0??
        for (int ii = 0; ii < nlocal_and_ghost; ii++) {
            int i = local_and_ghost_idxs[ii];

            int itag = tags[i];
            double xtmp = x[i].x;
            double ytmp = x[i].y;
            double ztmp = x[i].z;
            int itype = atom_type[i];

            auto& neigh_list = neighbor_list[i];
            int num_neigh = neigh_list.size();

            for (int jj = 0; jj < num_neigh; jj++) {
                int j = neigh_list[jj];
                j &= NEIGHMASK;

                double delx = xtmp - x[j].x;
                double dely = ytmp - x[j].y;
                double delz = ztmp - x[j].z;
                double rsq = delx*delx + dely*dely + delz*delz;

                if (rsq < cutforcesq) {
                    int jtype = atom_type[j];
                    double r = sqrt(rsq);
                    double p = r * rdr + 1.0;
                    int m = static_cast<int>(p);
                    m = MIN(m, nr - 1);
                    p -= m;
                    p = MIN(p, 1.0);
                    auto coeff = rhor_spline[type2rhor[jtype][itype]][m];
                    rho[i] += ((coeff[3]*p + coeff[4])*p + coeff[5])*p + coeff[6];
                    if (true) {
                        auto coeff = rhor_spline[type2rhor[itype][jtype]][m];
                        rho[j] += ((coeff[3]*p + coeff[4])*p + coeff[5])*p + coeff[6];
                    }

                    // if (tags[i] == target_tag || tags[j] == target_tag) {
                    //     std::stringstream s1;
                    //     s1 << BOLDMAGENTA << "STENCILMD FOUND RHO. zoid: " << zoid.num << " tags: " << tags[i] << " " << tags[j]
                    //     << " pos: " << x[i].x << " " << x[i].y << " " << x[i].z << " pos: " << x[j].x << " " << x[j].y << " " << x[j].z
                    //     << " lo: " << zoid.lo[0][0] << " " << zoid.lo[0][1] << " " << zoid.lo[0][2]
                    //     << " hi: " << zoid.hi[0][0] << " " << zoid.hi[0][1] << " " << zoid.hi[0][2]
                    //     << RESET_COLOR << std::endl;
                    //     std::cout << s1.str();
                    // }
                }
            }
        }

        // got all the rho's, compute fp
        for (int ii = 0; ii < nlocal_and_ghost; ii++) {
            int i = local_and_ghost_idxs[ii];
            int itype = atom_type[i];

            double p = rho[i] * rdrho + 1.0;
            int m = static_cast<int>(p);
            m = MAX(1, MIN(m, nrho - 1));
            p -= m;
            p = MIN(p, 1.0);
            auto coeff = frho_spline[type2frho[itype]][m];
            fp[i] = (coeff[0]*p + coeff[1])*p + coeff[2];
            // if (eflag) {
            //     phi = ((coeff[3]*p + coeff[4])*p + coeff[5])*p + coeff[6];
            //     if (rho[i] > rhomax) phi += fp[i] * (rho[i]-rhomax);
            //     phi *= scale[type[i]][type[i]];
            //     if (eflag_global) eng_vdwl += phi;
            //     if (eflag_atom) eatom[i] += phi;
            // }
        }

        for (int ii = 0; ii < nlocal_and_ghost; ii++) {
            int i = local_and_ghost_idxs[ii];

            double xtmp = x[i].x;
            double ytmp = x[i].y;
            double ztmp = x[i].z;

            int itype = atom_type[i];
            auto& neigh_list = neighbor_list[i];
            int num_neigh = neigh_list.size();

            double fxtmp = 0;
            double fytmp = 0;
            double fztmp = 0;

            for (int jj = 0; jj < num_neigh; jj++) {
                int j = neigh_list[jj];
                j &= NEIGHMASK;

                double delx = xtmp - x[j].x;
                double dely = ytmp - x[j].y;
                double delz = ztmp - x[j].z;
                double rsq = delx * delx + dely * dely + delz * delz;

                if (rsq < cutforcesq) {
                    // TODO: numforce?
                    int jtype = atom_type[j];
                    double r = sqrt(rsq);
                    double p = r * rdr + 1.0;
                    int m = static_cast<int>(p);
                    m = MIN(m, nr - 1);
                    p -= m;
                    p = MIN(p, 1.0);

                    // rhoip = derivative of (density at atom j due to atom i)
                    // rhojp = derivative of (density at atom i due to atom j)
                    // phi = pair potential energy
                    // phip = phi'
                    // z2 = phi * r
                    // z2p = (phi * r)' = (phi' r) + phi
                    // psip needs both fp[i] and fp[j] terms since r_ij appears in two
                    //   terms of embed eng: Fi(sum rho_ij) and Fj(sum rho_ji)
                    //   hence embed' = Fi(sum rho_ij) rhojp + Fj(sum rho_ji) rhoip
                    // scale factor can be applied by thermodynamic integration

                    auto coeff = rhor_spline[type2rhor[itype][jtype]][m];
                    auto rhoip = (coeff[0]*p + coeff[1])*p + coeff[2];
                    coeff = rhor_spline[type2rhor[jtype][itype]][m];
                    auto rhojp = (coeff[0]*p + coeff[1])*p + coeff[2];
                    coeff = z2r_spline[type2z2r[itype][jtype]][m];
                    auto z2p = (coeff[0]*p + coeff[1])*p + coeff[2];
                    auto z2 = ((coeff[3]*p + coeff[4])*p + coeff[5])*p + coeff[6];

                    double recip = 1.0/r;
                    double phi = z2*recip;
                    double phip = z2p*recip - phi*recip;
                    double psip = fp[i]*rhojp + fp[j]*rhoip + phip;
                    double fpair = -scale[itype][jtype]*psip*recip;

                    fxtmp += delx * fpair;
                    fytmp += dely * fpair;
                    fztmp += delz * fpair;

                    spinlocks[j].lock();
                    f[j].x -= delx * fpair;
                    f[j].y -= dely * fpair;
                    f[j].z -= delz * fpair;
                    spinlocks[j].unlock();

                    // if (tags[i] == target_tag || tags[j] == target_tag) {
                    //     std::stringstream s1;
                    //     s1 << BOLDGREEN << "STENCILMD FOUND. zoid: " << zoid.num << " tags: " << tags[i] << " " << tags[j]
                    //     << " pos: " << x[i].x << " " << x[i].y << " " << x[i].z << " pos: " << x[j].x << " " << x[j].y << " " << x[j].z
                    //     << " rho: " << rho[i] << " " << rho[j]
                    //     << " fp: " << fp[i] << " " << fp[j]
                    //     << " lo: " << zoid.lo[0][0] << " " << zoid.lo[0][1] << " " << zoid.lo[0][2]
                    //     << " hi: " << zoid.hi[0][0] << " " << zoid.hi[0][1] << " " << zoid.hi[0][2]
                    //     << RESET_COLOR << std::endl;
                    //     std::cout << s1.str();
                    // }

                    // if (eflag) evdwl = scale[itype][jtype]*phi;
                    // if (evflag) ev_tally(i,j,nlocal,newton_pair,evdwl,0.0,fpair,delx,dely,delz);
                }
            }

            spinlocks[i].lock();
            f[i].x += fxtmp;
            f[i].y += fytmp;
            f[i].z += fztmp;
            spinlocks[i].unlock();
        }

        // for (int ii = 0; ii < nlocal_and_ghost; ii++) {
        //     int i = local_and_ghost_idxs[ii];
        //     if (tags[i] == target_tag) {
        //         std::stringstream s1;
        //         s1 << BOLDYELLOW << "FOUND TAG LOOK AT FORCE: " << tags[i] << " in zoid: " << zoid.num
        //         << " rho: " << rho[i] << " fp: " << fp[i]
        //         << " force: " << f[i].x << " " << f[i].y << " " << f[i].z
        //         << RESET_COLOR << std::endl;
        //         std::cout << s1.str();
        //     }
        // }
    }


    void TERSOFF_FORCE_COMPUTE_ZOID_MANY_CUTS(queue_info& zoid, int dep, int timestep) {
        const auto * _noalias const x = zoid.x_stencil_md[timestep % DOUBLE_BUFFERING].data();
        auto * _noalias const f = zoid.f_stencil_md[timestep % 1].data();

        constexpr int shift_flag = 0;

        const auto& neighbor_list = zoid.neighbor_list[timestep];
        auto* _noalias spinlocks = zoid.spinlocks_stencil_md[0];

        // const auto& local_idxs = zoid.local_idxs_per_timestep[timestep];
        const auto& local_idxs = zoid.local_and_one_hop_ghost_idxs_per_timestep[timestep];
        // const auto& is_local_idx = zoid.is_local_per_timestep[timestep];
        const int nlocal = local_idxs.size();

        const auto& tags = zoid.tag_stencil_md[0];
        const auto& atom_type = zoid.type_stencil_md[0];

        // auto* neighshort = zoid.neigh_short[0].data();
        
        // int num_neigh_short = zoid.neigh_short[0].capacity();

        PairTersoff* pair_tersoff = (PairTersoff*) force->pair;
        auto map = pair_tersoff->map;
        auto elem3param = pair_tersoff->elem3param;
        auto params = pair_tersoff->params;

        double cutshortsq = pair_tersoff->cutmax * pair_tersoff->cutmax;

        double ters_R = params[0].bigr;
        double ters_D = params[0].bigd;
        double lam1 = params[0].lam1;
        double lam2 = params[0].lam2;
        double lam3 = params[0].lam3;
        double biga = params[0].biga;
        int powermint = params[0].powermint;

        double ters_c = params[0].c * params[0].c;
        double ters_d = params[0].d * params[0].d;
        double h = params[0].h;
        double beta = params[0].beta;
        double gamma = params[0].gamma;

        double bigr = params[0].bigr;
        double bigd = params[0].bigd;
        double bigb = params[0].bigb;

        double c1 = params[0].c1;
        double c2 = params[0].c2;
        double c3 = params[0].c3;
        double c4 = params[0].c4;

        double powern = params[0].powern;

        auto& neigh_short = zoid.neigh_short[0];

        // loop over full neighbor list of my atoms

        #pragma cilk grainsize 1024
        cilk_for (int ii = 0; ii < nlocal; ii++) {
            int i = local_idxs[ii];

            int itag = tags[i];
            int itype = map[atom_type[i]];
            double xtmp = x[i].x;
            double ytmp = x[i].y;
            double ztmp = x[i].z;
            double fxtmp = 0;
            double fytmp = 0;
            double fztmp = 0;

            auto* neigh_short_atom = neigh_short[i].data();
            int num_neigh_short = neigh_short[i].capacity();

            // two-body interactions, skip half of them

            auto& neigh_list = neighbor_list[i];
            int num_neigh = neigh_list.size();

            int numshort = 0;
            for (int jj = 0; jj < num_neigh; jj++) {
                int j = neigh_list[jj];
                j &= NEIGHMASK;
                double delx = xtmp - x[j].x;
                double dely = ytmp - x[j].y;
                double delz = ztmp - x[j].z;
                double rsq = delx*delx + dely*dely + delz*delz;

                if (rsq < cutshortsq) {
                    neigh_short_atom[numshort++] = j;
                    assert(numshort <= num_neigh_short);
                }

                int jtag = tags[j];
                if (itag > jtag) {
                    if ((itag+jtag) % 2 == 0) continue;
                } else if (itag < jtag) {
                    if ((itag+jtag) % 2 == 1) continue;
                } else {
                    if (x[j].z < ztmp) continue;
                    if (x[j].z == ztmp && x[j].y < ytmp) continue;
                    if (x[j].z == ztmp && x[j].y == ytmp && x[j].x < xtmp) continue;
                }

                int jtype = map[atom_type[j]];
                int ijparam = elem3param[itype][jtype][jtype];
                if (rsq >= params[ijparam].cutsq) continue;

                // repulsive calculation
                double r = sqrt(rsq);
                // ters_fc 
                double ters_fc;
                double ters_fc_d;
                if (r < ters_R - ters_D) {
                    ters_fc = 1;
                    ters_fc_d = 0;
                } else if (r > ters_R + ters_D) {
                    ters_fc = 0;
                    ters_fc_d = 0;
                } else {
                    ters_fc = 0.5*(1.0 - sin(MathConst::MY_PI2*(r - ters_R)/ters_D));
                    ters_fc_d =  -(MathConst::MY_PI4/ters_D) * cos(MathConst::MY_PI2*(r - ters_R)/ters_D);
                }

                double repulsive_exp = exp(-lam1 * r);
                double fpair = -biga * repulsive_exp * (ters_fc_d - ters_fc * lam1) / r;

                fxtmp += delx*fpair;
                fytmp += dely*fpair;
                fztmp += delz*fpair;

                spinlocks[j].lock();
                f[j].x -= delx*fpair;
                f[j].y -= dely*fpair;
                f[j].z -= delz*fpair;
                spinlocks[j].unlock();
                // if (evflag) ev_tally(i,j,nlocal,newton_pair, evdwl,0.0,fpair,delx,dely,delz);
            }

            for (int jj = 0; jj < numshort; jj++) {
                int j = neigh_short_atom[jj];
                int jtype = map[atom_type[j]];
                int iparam_ij = elem3param[itype][jtype][jtype];
                dbl3_t_stencil_md delr1 = {x[j].x - xtmp, x[j].y - ytmp, x[j].z - ztmp};
                double rsq1 = delr1.x*delr1.x + delr1.y*delr1.y + delr1.z*delr1.z;

                if (rsq1 >= params[iparam_ij].cutsq) {
                    continue;
                }

                double r1 = sqrt(rsq1);
                const double r1inv = 1.0/r1;
                dbl3_t_stencil_md r1hat = {r1inv * delr1.x, r1inv * delr1.y, r1inv * delr1.z};

                double fjxtmp = 0;
                double fjytmp = 0;
                double fjztmp = 0;
                double zeta_ij = 0;

                // accumulate bondorder zeta for each i-j interaction via loop over k
                for (int kk = 0; kk < numshort; kk++) {
                    if (jj == kk) continue;
                    int k = neigh_short_atom[kk];
                    int ktype = map[atom_type[k]];
                    int iparam_ijk = elem3param[itype][jtype][ktype];

                    dbl3_t_stencil_md delr2 = {x[k].x - xtmp, x[k].y - ytmp, x[k].z - ztmp};
                    double rsq2 = delr2.x*delr2.x + delr2.y*delr2.y + delr2.z*delr2.z;

                    if (rsq2 >= params[iparam_ijk].cutsq) continue;

                    double r2 = sqrt(rsq2);
                    double r2inv = 1.0/r2;
                    dbl3_t_stencil_md r2hat = {r2inv * delr2.x, r2inv * delr2.y, r2inv * delr2.z};

                    // zeta calculation
                    {
                        double costheta = r1hat.x * r2hat.x + r1hat.y * r2hat.y + r1hat.z * r2hat.z;
                        double arg;
                        if (powermint == 3) {
                            double tmp = lam3 * (r1 - r2);
                            arg = MathSpecial::cube(tmp);
                        } else {
                            arg = lam3 * (r1-r2);
                        }

                        double ex_delr;
                        if (arg > 69.0776) ex_delr = 1.e30;
                        else if (arg < -69.0776) ex_delr = 0.0;
                        else ex_delr = exp(arg);

                        double ters_fc_ik;
                        if (r2 < ters_R - ters_D) {
                            ters_fc_ik = 1;
                        } else if (r2 > ters_R + ters_D) {
                            ters_fc_ik = 0;
                        } else {
                            ters_fc_ik = 0.5*(1.0 - sin(MathConst::MY_PI2*(r2 - ters_R)/ters_D));
                        }

                        double hcth = h - costheta;
                        double ters_gijk = gamma * (1.0 + ters_c / ters_d - ters_c / (ters_d + hcth * hcth));
                        zeta_ij += ters_fc_ik * ters_gijk * ex_delr;
                    }
                }

                // force_zeta

                double fforce;
                double prefactor;

                {
                    double ters_fc_r1;
                    double ters_fc_r1_d;
                    if (r1 < ters_R - ters_D) {
                        ters_fc_r1 = 1;
                        ters_fc_r1_d = 0;
                    } else if (r1 > ters_R + ters_D) {
                        ters_fc_r1 = 0;
                        ters_fc_r1_d = 0;
                    } else {
                        ters_fc_r1 = 0.5*(1.0 - sin(MathConst::MY_PI2*(r1 - ters_R)/ters_D));
                        ters_fc_r1_d = -(MathConst::MY_PI4/ters_D) * cos(MathConst::MY_PI2*(r1 - ters_R)/ters_D);
                    }

                    double fa;
                    double fa_d;
                    if (r1 > bigr + bigd) {
                        fa = 0;
                        fa_d = 0;
                    } else {
                        fa = -bigb * exp(-lam2 * r1) * ters_fc_r1;
                        fa_d = bigb * exp(-lam2 * r1) * (lam2 * ters_fc_r1 - ters_fc_r1_d);
                    }

                    double bij;
                    double bij_d;
                    double ters_bij_tmp = beta * zeta_ij;
                    if (ters_bij_tmp > c1) {
                        bij = 1.0 / sqrt(ters_bij_tmp);
                        bij_d = beta * -0.5*pow(ters_bij_tmp,-1.5);
                    } else if (ters_bij_tmp > c2) {
                        bij = (1.0 - pow(ters_bij_tmp, -powern) / (2.0*powern))/sqrt(ters_bij_tmp);
                        bij_d = beta * (-0.5*pow(ters_bij_tmp,-1.5) *
                            // error in negligible 2nd term fixed 9/30/2015
                            // (1.0 - 0.5*(1.0 +  1.0/(2.0*param->powern)) *
                            (1.0 - (1.0 +  1.0/(2.0*powern)) *
                            pow(ters_bij_tmp,-powern)));
                    } else if (ters_bij_tmp < c4) {
                        bij = 1;
                        bij_d = 0;
                    } else if (ters_bij_tmp < c3) {
                        bij = 1.0 - pow(ters_bij_tmp,powern)/(2.0*powern);
                        bij_d = -0.5*beta * pow(ters_bij_tmp,powern-1.0);
                    } else {
                        double tmp_n = pow(ters_bij_tmp, powern);
                        bij = pow(1.0 + tmp_n, -1.0/(2.0*powern));
                        bij_d = -0.5 * pow(1.0+tmp_n, -1.0-(1.0/(2.0*powern)))*tmp_n / zeta_ij;
                    }

                    fforce = 0.5 * bij * fa_d;
                    prefactor = -0.5 * fa * bij_d;
                }

                double fpair = fforce * r1inv;
                double delx = delr1.x * fpair;
                double dely = delr1.y * fpair;
                double delz = delr1.z * fpair;

                fxtmp += delx;
                fytmp += dely;
                fztmp += delz;
                fjxtmp -= delx;
                fjytmp -= dely;
                fjztmp -= delz;

                for (int kk = 0; kk < numshort; kk++) {
                    if (jj == kk) continue;
                    int k = neigh_short_atom[kk];
                    int ktype = map[atom_type[k]];
                    int iparam_ijk = elem3param[itype][jtype][ktype];
                    dbl3_t_stencil_md delr2 = {x[k].x - xtmp, x[k].y - ytmp, x[k].z - ztmp};
                    double rsq2 = delr2.x*delr2.x + delr2.y*delr2.y + delr2.z*delr2.z;

                    if (rsq2 >= params[iparam_ijk].cutsq) continue;

                    double r2 = sqrt(rsq2);
                    double r2inv = 1.0/r2;
                    dbl3_t_stencil_md r2hat = {r2inv * delr2.x, r2inv * delr2.y, r2inv * delr2.z};

                    // attractive calculation/ters_zetaterm_d
                    {
                        double ters_fc_r2;
                        double ters_fc_r2_d;
                        if (r2 < ters_R - ters_D) {
                            ters_fc_r2 = 1;
                            ters_fc_r2_d = 0;
                        } else if (r2 > ters_R + ters_D) {
                            ters_fc_r2 = 0;
                            ters_fc_r2_d = 0;
                        } else {
                            ters_fc_r2 = 0.5*(1.0 - sin(MathConst::MY_PI2*(r2 - ters_R)/ters_D));
                            ters_fc_r2_d = -(MathConst::MY_PI4/ters_D) * cos(MathConst::MY_PI2*(r2 - ters_R)/ters_D);
                        }

                        double tmp;
                        if (powermint == 3) {
                            tmp = MathSpecial::cube(lam3 * (r1 - r2));
                        } else {
                            tmp = lam3 * (r1 - r2);
                        }

                        double ex_delr;
                        double ex_delr_d;

                        if (tmp > 69.0776) ex_delr = 1.e30;
                        else if (tmp < -69.0776) ex_delr = 0.0;
                        else ex_delr = exp(tmp);

                        if (powermint == 3) {
                            ex_delr_d = 3.0 * MathSpecial::cube(lam3) * MathSpecial::square(r1 - r2) * ex_delr;
                        } else {
                            ex_delr_d = lam3 * ex_delr;
                        }

                        double costheta = r1hat.x * r2hat.x + r1hat.y * r2hat.y + r1hat.z * r2hat.z;
                        // ters_gijk
                        double hcth = h - costheta;
                        double ters_gijk = gamma * (1.0 + ters_c / ters_d - ters_c / (ters_d + hcth * hcth));

                        // ters_gijk_d
                        double numerator = -2.0 * ters_c * hcth;
                        double denominator = 1.0 / (ters_d + hcth * hcth);
                        double ters_gijk_d = gamma * numerator * denominator * denominator;

                        // costheta_d
                        double costheta_d;
                        dbl3_t_stencil_md dcosdrj = {-costheta * r1hat.x + r2hat.x, -costheta * r1hat.y + r2hat.y, -costheta * r1hat.z + r2hat.z};
                        dcosdrj.x *= r1inv;
                        dcosdrj.y *= r1inv;
                        dcosdrj.z *= r1inv;

                        dbl3_t_stencil_md dcosdrk = {-costheta * r2hat.x + r1hat.x, -costheta * r2hat.y + r1hat.y, -costheta * r2hat.z + r1hat.z};
                        dcosdrk.x *= r2inv;
                        dcosdrk.y *= r2inv;
                        dcosdrk.z *= r2inv;

                        dbl3_t_stencil_md dcosdri = {dcosdrj.x + dcosdrk.x, dcosdrj.y + dcosdrk.y, dcosdrj.z + dcosdrk.z};
                        dcosdri.x *= -1.0;
                        dcosdri.y *= -1.0;
                        dcosdri.z *= -1.0;

                        double scale1 = -ters_fc_r2_d * ters_gijk * ex_delr;
                        double scale2 = ters_fc_r2 * ters_gijk_d * ex_delr;
                        double scale3 = ters_fc_r2 * ters_gijk * ex_delr_d;
                        double scale4 = -scale3;

                        dbl3_t_stencil_md dri = {scale1 * r2hat.x, scale1 * r2hat.y, scale1 * r2hat.z};
                        dri.x += scale2 * dcosdri.x;
                        dri.y += scale2 * dcosdri.y;
                        dri.z += scale2 * dcosdri.z;

                        dri.x += scale3 * r2hat.x;
                        dri.y += scale3 * r2hat.y;
                        dri.z += scale3 * r2hat.z;

                        dri.x += scale4 * r1hat.x;
                        dri.y += scale4 * r1hat.y;
                        dri.z += scale4 * r1hat.z;

                        dri.x *= prefactor;
                        dri.y *= prefactor;
                        dri.z *= prefactor;

                        dbl3_t_stencil_md drj = {scale2 * dcosdrj.x, scale2 * dcosdrj.y, scale2 * dcosdrj.z};
                        drj.x += scale3 * r1hat.x;
                        drj.y += scale3 * r1hat.y;
                        drj.z += scale3 * r1hat.z;

                        drj.x *= prefactor;
                        drj.y *= prefactor;
                        drj.z *= prefactor;
                        
                        dbl3_t_stencil_md drk = {-scale1 * r2hat.x, -scale1 * r2hat.y, -scale1 * r2hat.z};
                        drk.x += scale2 * dcosdrk.x;
                        drk.y += scale2 * dcosdrk.y;
                        drk.z += scale2 * dcosdrk.z;

                        drk.x += scale4 * r2hat.x;
                        drk.y += scale4 * r2hat.y;
                        drk.z += scale4 * r2hat.z;

                        drk.x *= prefactor;
                        drk.y *= prefactor;
                        drk.z *= prefactor;

                        fxtmp += dri.x;
                        fytmp += dri.y;
                        fztmp += dri.z;

                        fjxtmp += drj.x;
                        fjytmp += drj.y;
                        fjztmp += drj.z;

                        spinlocks[k].lock();
                        f[k].x += drk.x;
                        f[k].y += drk.y;
                        f[k].z += drk.z;
                        spinlocks[k].unlock();
                    }
                }

                spinlocks[j].lock();
                f[j].x += fjxtmp;
                f[j].y += fjytmp;
                f[j].z += fjztmp;
                spinlocks[j].unlock();
            }

            spinlocks[i].lock();
            f[i].x += fxtmp;
            f[i].y += fytmp;
            f[i].z += fztmp;
            spinlocks[i].unlock();
        }
        // if (vflag_fdotr) virial_fdotr_compute();
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

        const auto* _noalias cutsq = pair->cutsq;
        // const auto* _noalias offset = pair->offset;
        const auto* _noalias lj1 = pair->lj1;
        const auto* _noalias lj2 = pair->lj2;
        // const auto* _noalias lj3 = pair->lj3;
        // const auto* _noalias lj4 = pair->lj4;
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

        auto* _noalias claimed = zoid.claimed_flags_stencil_md[0];

        const auto& tags = zoid.tag_stencil_md[0];

        int chunks_per_worker = num_chunks / num_workers;
        int chunk_size = MODIFY_GRAINSIZE;

        constexpr int PAIR_GRAINSIZE = 512;
        constexpr int BOND_GRAINSIZE = 512;

        if (nlocal > PAIR_GRAINSIZE) {
            #pragma cilk grainsize PAIR_GRAINSIZE
            cilk_for(int idx = 0; idx < nlocal; idx++) {
                int i = local_idxs[idx];

                const int itype = atom_type[i];

                // const int *_noalias const jlist = firstneigh[i];
                const auto &jlist = neighbor_list[i];
                const double *_noalias const cutsqi = cutsq[itype];
                // const double *_noalias const offseti = offset[itype];
                const double *_noalias const lj1i = lj1[itype];
                const double *_noalias const lj2i = lj2[itype];
                // const double *_noalias const lj3i = lj3[itype];
                // const double *_noalias const lj4i = lj4[itype];

                double xtmp = x[i].x;
                double ytmp = x[i].y;
                double ztmp = x[i].z;
                // int jnum = numneigh[i];
                int jnum = jlist.size();

                double fxtmp = 0.0;
                double fytmp = 0.0;
                double fztmp = 0.0;

                for (int jj = 0; jj < jnum; jj++) {
                    // double evdwl = 0.0;
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
                // const double *_noalias const offseti = offset[itype];
                const double *_noalias const lj1i = lj1[itype];
                const double *_noalias const lj2i = lj2[itype];
                // const double *_noalias const lj3i = lj3[itype];
                // const double *_noalias const lj4i = lj4[itype];

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

        if (nbonds > BOND_GRAINSIZE) {
            #pragma cilk grainsize BOND_GRAINSIZE
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
                // double sr2 = 0.0;
                // double sr6 = 0.0;

                if (rsq < MathConst::MY_CUBEROOT2 * sigma[type] * sigma[type]) {
                    double sr2 = sigma[type] * sigma[type] / rsq;
                    double sr6 = sr2 * sr2 * sr2;
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

    template <bool curr_dt>
    void INIT_PER_WORKER_ARRAYS() {
        const int nworkers = __cilkrts_get_nworkers();
        auto& my_queues = curr_dt ? my_queues_many_cuts : my_queues_many_cuts_next_dt;
        for (int dep = 0; dep < NUM_DEPS; dep++) {
            for (int j = 0; j < my_queues[dep].size(); j++) {
                auto& zoid = my_queues[dep][j];
                if (curr_dt) {
                    for (int w = 0; w < nworkers; w++) {
                        zoid.per_worker_force_updates[w] = new dbl3_t_stencil_md[zoid.x_stencil_md[0].size()];
                        for (int i = 0; i < zoid.x_stencil_md[0].size(); i++) {
                            zoid.per_worker_force_updates[w][i].x = 0;
                            zoid.per_worker_force_updates[w][i].y = 0;
                            zoid.per_worker_force_updates[w][i].z = 0;
                        }
                    }
                }
                for (int t = 0; t < NUM_TIMESTEPS_IN_PARALLEL + 1; t++) {
                    std::set<int> idxs;
                    auto& local_idxs = zoid.local_idxs_per_timestep[t];
                    auto& neighbor_list = zoid.neighbor_list[t];
                    for (int i = 0; i < local_idxs.size(); i++) {
                        int local_idx = local_idxs[i];
                        idxs.insert(local_idx);
                        for (auto& neigh_idx : neighbor_list[local_idx]) {
                            idxs.insert(neigh_idx);
                        }
                    }

                    // For bond_fene potential, add in bond idxs
                    if constexpr (EXPERIMENT == BOND_FENE) {
                        auto& bond_list = zoid.bond_list_modified[t];
                        for (auto& [i1, i2, type] : bond_list) {
                            idxs.insert(i1);
                            idxs.insert(i2);
                        }
                    }

                    zoid.global_to_local_idx[t].resize(zoid.x_stencil_md[0].size(), -1);
                    zoid.local_to_global_idx[t].resize(idxs.size(), -1);

                    int local_idx = 0;
                    for (auto& idx : idxs) {
                        zoid.global_to_local_idx[t][idx] = local_idx++;
                    }

                    local_idx = 0;
                    for (auto& idx : idxs) {
                        zoid.local_to_global_idx[t][local_idx++] = idx;
                    }

                    assert(local_idx == idxs.size());
                    for (int i = 0; i < local_idx; i++) {
                        int global_idx = zoid.local_to_global_idx[t][i];
                        assert(zoid.global_to_local_idx[t][global_idx] == i);
                    }
                }
            }
        }
    }

    void LJ_FORCE_COMPUTE_ZOID_MANY_CUTS(queue_info& zoid, int dep, int timestep) {
        const auto * _noalias const x = zoid.x_stencil_md[timestep % DOUBLE_BUFFERING].data();
        auto * _noalias const f = zoid.f_stencil_md[timestep % 1].data();

        auto pair = (PairLJCut*) force->pair;

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
        constexpr bool newton_pair = USE_NEWTON;

        const auto& atom_type = zoid.type_stencil_md[0];

        const auto& local_idxs = zoid.local_idxs_per_timestep[timestep];
        const int nlocal = local_idxs.size();

        constexpr int GRAINSIZE = 1024;
        constexpr int SMALL_GRAINSIZE = 128;

        constexpr bool USE_MEMORY = true;

        if (USE_MEMORY) {
            constexpr int LJ_GRAINSIZE = 512;
            if (nlocal <= LJ_GRAINSIZE) {
                for (int idx = 0; idx < nlocal; idx++) {
                    int i = local_idxs[idx];

                    const int itype = atom_type[i];

                    const auto &jlist = neighbor_list[i];
                    const double *_noalias const cutsqi = cutsq[itype];
                    const double *_noalias const offseti = offset[itype];
                    const double *_noalias const lj1i = lj1[itype];
                    const double *_noalias const lj2i = lj2[itype];
                    // const double *_noalias const lj3i = lj3[itype];
                    // const double *_noalias const lj4i = lj4[itype];

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

                return;
            }

            int nworkers = __cilkrts_get_nworkers();
            auto* claimed = zoid.claimed_flags_stencil_md[0];
            auto* per_worker_force_updates = zoid.per_worker_force_updates;
            int num_chunks = nlocal / LJ_GRAINSIZE + 1;
            int chunks_per_worker = num_chunks / nworkers;

            auto& global_to_local_idx = zoid.global_to_local_idx[timestep];
            auto& local_to_global_idx = zoid.local_to_global_idx[timestep];
            int num_local_to_global = local_to_global_idx.size();
            std::vector<int> workers_used(nworkers, 0);

            #pragma cilk grainsize 1
            cilk_for (int ii = 0; ii < num_chunks; ii++) {
                int worker_number = __cilkrts_get_worker_number();
                int start_chunk = worker_number * chunks_per_worker;
                auto* worker_local_updates = per_worker_force_updates[worker_number];

                for (int c = 0; c < num_chunks; ++c) {
                    int s = (c + start_chunk) % num_chunks;

                    if (claimed[s].test(std::memory_order_relaxed)) {
                        continue;
                    }

                    if (!claimed[s].test_and_set(std::memory_order_relaxed)) {
                        workers_used[worker_number] = 1;
                        for (int idx = s * MODIFY_GRAINSIZE; idx < (s + 1) * MODIFY_GRAINSIZE && idx < nlocal; idx++) {
                            int i = local_idxs[idx];

                            const int itype = atom_type[i];

                            const auto &jlist = neighbor_list[i];
                            const double *_noalias const cutsqi = cutsq[itype];
                            const double *_noalias const offseti = offset[itype];
                            const double *_noalias const lj1i = lj1[itype];
                            const double *_noalias const lj2i = lj2[itype];
                            // const double *_noalias const lj3i = lj3[itype];
                            // const double *_noalias const lj4i = lj4[itype];

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

                                    if (true) {
                                        int local_idx = global_to_local_idx[j];
                                        assert(local_idx != -1);
                                        auto& worker_local_f = worker_local_updates[local_idx];
                                        worker_local_f.x -= delx * fpair;
                                        worker_local_f.y -= dely * fpair;
                                        worker_local_f.z -= delz * fpair;
                                    }
                                }
                            }

                            f[i].x += fxtmp;
                            f[i].y += fytmp;
                            f[i].z += fztmp;
                        }
                    }
                }
            }

            for (int i = 0; i < num_chunks; i++) {
                claimed[i].clear(std::memory_order_relaxed);
            }

            #pragma cilk grainsize LJ_GRAINSIZE
            cilk_for (int local_idx = 0; local_idx < num_local_to_global; local_idx++) {
                int global_idx = local_to_global_idx[local_idx];
                assert(global_idx >= 0);
                for (int w = 0; w < nworkers; w++) {
                    if (!workers_used[w]) {
                        continue;
                    }
                    auto& worker_local_f = per_worker_force_updates[w][local_idx];
                    f[global_idx].x += worker_local_f.x;
                    f[global_idx].y += worker_local_f.y;
                    f[global_idx].z += worker_local_f.z;

                    worker_local_f.x = 0;
                    worker_local_f.y = 0;
                    worker_local_f.z = 0;
                }
            }

            return;
        }

        if ((dep == 0 && (timestep >  2)) || (dep == 3 && (timestep <= 2))) {
            #pragma cilk grainsize SMALL_GRAINSIZE
            cilk_for (int idx = 0; idx < nlocal; idx++) {
                int i = local_idxs[idx];

                const int itype = atom_type[i];

                const auto &jlist = neighbor_list[i];
                const double *_noalias const cutsqi = cutsq[itype];
                const double *_noalias const offseti = offset[itype];
                const double *_noalias const lj1i = lj1[itype];
                const double *_noalias const lj2i = lj2[itype];
                // const double *_noalias const lj3i = lj3[itype];
                // const double *_noalias const lj4i = lj4[itype];

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
        } else if (nlocal > GRAINSIZE) {
            #pragma cilk grainsize GRAINSIZE 
            cilk_for (int idx = 0; idx < nlocal; idx++) {
                int i = local_idxs[idx];

                const int itype = atom_type[i];

                const auto &jlist = neighbor_list[i];
                const double *_noalias const cutsqi = cutsq[itype];
                const double *_noalias const offseti = offset[itype];
                const double *_noalias const lj1i = lj1[itype];
                const double *_noalias const lj2i = lj2[itype];
                // const double *_noalias const lj3i = lj3[itype];
                // const double *_noalias const lj4i = lj4[itype];

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

                const auto &jlist = neighbor_list[i];
                const double *_noalias const cutsqi = cutsq[itype];
                const double *_noalias const offseti = offset[itype];
                const double *_noalias const lj1i = lj1[itype];
                const double *_noalias const lj2i = lj2[itype];
                // const double *_noalias const lj3i = lj3[itype];
                // const double *_noalias const lj4i = lj4[itype];

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

        constexpr bool USE_MEMORY = true;
        if (false) {
            int nworkers = __cilkrts_get_nworkers();
            auto* claimed = zoid.claimed_flags_stencil_md[0];
            auto* per_worker_force_updates = zoid.per_worker_force_updates;
            // for (int w = 0; w < nworkers; w++) {
            //     memset(per_worker_force_updates[w], 0, sizeof(dbl3_t_stencil_md) * zoid.x_stencil_md[0].size());
            // }
            int num_chunks = nlocal / MODIFY_GRAINSIZE + 1;
            int chunks_per_worker = num_chunks / nworkers;

            auto& global_to_local_idx = zoid.global_to_local_idx[timestep];
            auto& local_to_global_idx = zoid.local_to_global_idx[timestep];
            int num_local_to_global = local_to_global_idx.size();
            std::vector<int> workers_used(nworkers, 0);

            #pragma cilk grainsize 1
            cilk_for (int ii = 0; ii < num_chunks; ii++) {
                int worker_number = __cilkrts_get_worker_number();
                int start_chunk = worker_number * chunks_per_worker;
                auto* worker_local_updates = per_worker_force_updates[worker_number];

                for (int c = 0; c < num_chunks; ++c) {
                    int s = (c + start_chunk) % num_chunks;

                    if (claimed[s].test(std::memory_order_relaxed)) {
                        continue;
                    }

                    if (!claimed[s].test_and_set(std::memory_order_relaxed)) {
                        workers_used[worker_number] = 1;
                        for (int idx = s * MODIFY_GRAINSIZE; idx < (s + 1) * MODIFY_GRAINSIZE && idx < nlocal; idx++) {
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

                                    int local_idx = global_to_local_idx[j];
                                    assert(local_idx != -1);
                                    auto& worker_local_f = worker_local_updates[local_idx];
                                    worker_local_f.x -= delx * fpair;
                                    worker_local_f.y -= dely * fpair;
                                    worker_local_f.z -= delz * fpair;
                                }
                            }

                            f[i].x += fxtmp;
                            f[i].y += fytmp;
                            f[i].z += fztmp;
                        }
                    }
                }
            }

            for (int i = 0; i < num_chunks; i++) {
                claimed[i].clear(std::memory_order_relaxed);
            }

            #pragma cilk grainsize 1024
            cilk_for (int local_idx = 0; local_idx < num_local_to_global; local_idx++) {
                int global_idx = local_to_global_idx[local_idx];
                assert(global_idx >= 0);
                for (int w = 0; w < nworkers; w++) {
                    if (!workers_used[w]) {
                        continue;
                    }
                    auto& worker_local_f = per_worker_force_updates[w][local_idx];
                    f[global_idx].x += worker_local_f.x;
                    f[global_idx].y += worker_local_f.y;
                    f[global_idx].z += worker_local_f.z;

                    worker_local_f.x = 0;
                    worker_local_f.y = 0;
                    worker_local_f.z = 0;
                }
            }

            return;
        }

        #pragma cilk grainsize 1024
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

                delete[] zoid.neigh_short;

                for (int w = 0; w < __cilkrts_get_nworkers(); w++) {
                    delete[] zoid.per_worker_force_updates[w];
                }
                delete[] zoid.per_worker_force_updates;
                delete[] zoid.global_to_local_idx;
                delete[] zoid.local_to_global_idx;

                delete[] zoid.fp_stencil_md;
                delete[] zoid.rho_stencil_md;
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
                delete[] zoid.local_and_one_hop_ghost_idxs_per_timestep;
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

                delete[] zoid.global_to_local_idx;
                delete[] zoid.local_to_global_idx;

                delete[] zoid.local_idxs_per_timestep;
                delete[] zoid.local_and_one_hop_ghost_idxs_per_timestep;
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