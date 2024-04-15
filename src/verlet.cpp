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

#include "verlet.h"

#include "accelerator_kokkos.h"
#include "accelerator_omp.h"
#include "angle.h"
#include "atom.h"
#include "atom_vec.h"
#include "bond.h"
#include "citeme.h"
#include "comm.h"
#include "dihedral.h"
#include "error.h"
#include "fix.h"
#include "force.h"
#include "group.h"
#include "improper.h"
#include "info.h"
#include "input.h"
#include "kspace.h"
#include "lmppython.h"
#include "modify.h"
#include "neighbor.h"
#include "output.h"
#include "stencil_md_utils.h"
#include "timer.h"
#include "universe.h"
#include "update.h"
#include "variable.h"
#include "version.h"

#include <mpi.h>
#include <cmath>
#include <cstring>
#include <map>
#include <future>
#include <algorithm>
#include "pair_lj_cut.h"

#include "stencil_md.h"

#include <fstream>
#include <iostream>
#include <cilk/cilk.h>
#include <cilk/cilk_api.h>
#include <unordered_map>
#include <sstream>
#include <cilk/opadd_reducer.h>

#include <CGAL/spatial_sort.h>
#include <CGAL/Exact_predicates_inexact_constructions_kernel.h>
#include <CGAL/point_generators_3.h>
#include <CGAL/hilbert_sort.h>

typedef CGAL::Exact_predicates_inexact_constructions_kernel K;
typedef K::Point_3                                          Point;
typedef CGAL::Creator_uniform_3<double,Point>               Creator;

using namespace LAMMPS_NS;

static constexpr bool TEST_AGAINST_LAMMPS_LOCAL = TEST_AGAINST_LAMMPS;
constexpr bool USE_DEP_TO_WAIT_IDXS = true;
static int64_t unpack_duration = 0;
static int64_t send_comm_duration = 0;
static int64_t recv_comm_duration = 0;
static int64_t compute_duration = 0;
static int64_t modify_duration = 0;
static int64_t modify_pre_force_duration = 0;
static int64_t mpi_duration = 0;
static int64_t curr_dt_comm_duration = 0;
static int64_t next_dt_comm_duration = 0;
static int64_t send_pack_duration = 0;
static int64_t misc_time = 0;

static int64_t unpack_self_time = 0;

static int64_t pre_recv_time = 0;

// based on dep of zoids I want to eval
static int64_t curr_dt_compute_dep_time[NUM_DEPS] = {0};
static int64_t next_dt_compute_dep_time[NUM_DEPS] = {0};
static int64_t curr_dt_num_atoms[NUM_DEPS] = {0};
static int64_t next_dt_num_atoms[NUM_DEPS] = {0};

static std::vector<int64_t> curr_dt_compute_dep_times_vec[NUM_DEPS];
static std::vector<int64_t> next_dt_compute_dep_times_vec[NUM_DEPS];

static cilk::opadd_reducer<int64_t> compute_duration_cilk = 0;
static cilk::opadd_reducer<int64_t> modify_duration_cilk = 0;
static cilk::opadd_reducer<int64_t> modify_pre_force_duration_cilk = 0;
static cilk::opadd_reducer<int64_t> send_pack_duration_cilk = 0;

static std::vector<int64_t> lammps_forward_comm_times;
static std::vector<int64_t> lammps_reverse_comm_times;

static int64_t SIZES[NUM_DEPS] = {1, 3, 3, 1};

/* ---------------------------------------------------------------------- */

Verlet::Verlet(LAMMPS* lmp, int narg, char** arg) : Integrate(lmp, narg, arg) {}

/* ----------------------------------------------------------------------
   initialization before run
------------------------------------------------------------------------- */

double mean(const std::vector<int64_t> &v)
{
    int64_t sum = 0;

    for (auto &each: v)
        sum += each;

    return (double) sum / v.size();
}

double sd(const std::vector<int64_t> &v)
{
    double square_sum_of_difference = 0;
    double mean_var = mean(v);
    auto len = v.size();

    double tmp;
    for (auto &each: v) {
        tmp = each - mean_var;
        square_sum_of_difference += tmp * tmp;
    }

    return std::sqrt(square_sum_of_difference / (len - 1));
}

int atom_coord_to_bin(double* lo, double* hi, double* pos) {

}

void Verlet::init() {
    Integrate::init();

    // warn if no fixes doing time integration

    bool do_time_integrate = false;
    for (const auto& fix : modify->get_fix_list())
        if (fix->time_integrate)
            do_time_integrate = true;

    if (!do_time_integrate && (comm->me == 0))
        error->warning(FLERR,
                       "No fixes with time integration, atoms won't move");

    // virial_style:
    // VIRIAL_PAIR if computed explicitly in pair via sum over pair interactions
    // VIRIAL_FDOTR if computed implicitly in pair by
    //   virial_fdotr_compute() via sum over ghosts

    if (force->newton_pair)
        virial_style = VIRIAL_FDOTR;
    else
        virial_style = VIRIAL_PAIR;

    // setup lists of computes for global and per-atom PE and pressure

    ev_setup();

    // detect if fix omp is present for clearing force arrays

    if (modify->get_fix_by_id("package_omp"))
        external_force_clear = 1;

    // set flags for arrays to clear in force_clear()

    torqueflag = extraflag = 0;
    if (atom->torque_flag)
        torqueflag = 1;
    if (atom->avec->forceclearflag)
        extraflag = 1;

    // orthogonal vs triclinic simulation box

    triclinic = domain->triclinic;
}

/* ----------------------------------------------------------------------
   setup before run
------------------------------------------------------------------------- */

void Verlet::setup(int flag) {
    if (comm->me == 0 && screen) {
        fputs("Setting up Verlet run ...\n", screen);
        if (flag) {
            fmt::print(screen,
                       "  Unit style    : {}\n"
                       "  Current step  : {}\n"
                       "  Time step     : {}\n",
                       update->unit_style, update->ntimestep, update->dt);
            timer->print_timeout(screen);
        }
    }

    if (lmp->kokkos) {
        error->all(FLERR, "KOKKOS package requires run_style verlet/kk");
    }

    update->setupflag = 1;

    // setup domain, communication and neighboring
    // acquire ghosts
    // build neighbor lists

    atom->setup();
    modify->setup_pre_exchange();
    if (triclinic)
        domain->x2lamda(atom->nlocal);
    domain->pbc();
    domain->reset_box();
    comm->setup();
    if (neighbor->style)
        neighbor->setup_bins();
    comm->exchange();
    if (atom->sortfreq > 0) {
        atom->sort();
    }
    comm->borders();
    if (triclinic)
        domain->lamda2x(atom->nlocal + atom->nghost);
    domain->image_check();
    domain->box_too_small_check();
    modify->setup_pre_neighbor();
    neighbor->build(1);
    modify->setup_post_neighbor();
    neighbor->ncalls = 0;

    // compute all forces

    force->setup();
    ev_set(update->ntimestep);
    force_clear();
    modify->setup_pre_force(vflag);

    if (pair_compute_flag) {
        force->pair->compute(eflag, vflag);
    } else if (force->pair) {
        force->pair->compute_dummy(eflag, vflag);
    }

    if (atom->molecular != Atom::ATOMIC) {
        if (force->bond)
            force->bond->compute(eflag, vflag);
        if (force->angle)
            force->angle->compute(eflag, vflag);
        if (force->dihedral)
            force->dihedral->compute(eflag, vflag);
        if (force->improper)
            force->improper->compute(eflag, vflag);
    }

    if (force->kspace) {
        assert(false);
        force->kspace->setup();
        if (kspace_compute_flag)
            force->kspace->compute(eflag, vflag);
        else
            force->kspace->compute_dummy(eflag, vflag);
    }

    modify->setup_pre_reverse(eflag, vflag);
    if (force->newton) {
        comm->reverse_comm();
    } else {
        assert(false);
    }

    modify->setup(vflag);
    output->setup(flag);
    update->setupflag = 0;

    std::cout << GREEN << "------------------- LAMMPS SETUP DONE -------------------------" << RESET_COLOR << std::endl;
    if (!ONLY_RUN_LAMMPS) {
        setup_stencil_md();
    }
}

void atom_reorder_stencil_md(Atom* atom_, int* current, int* permute, int start,
                             int end) {
    atom_->avec->grow_stencil_md(0, atom_);
    int empty;
    for (int i = start; i < end; i++) {
        if (current[i] == permute[i]) {
            continue;
        }
        atom_->avec->copy(i, end, 0);
        empty = i;
        while (permute[empty] != i) {
            atom_->avec->copy(permute[empty], empty, 0);
            empty = current[empty] = permute[empty];
        }
        atom_->avec->copy(end, empty, 0);
        current[empty] = permute[empty];
    }
}

void atom_reorder_ghost_stencil_md(Atom* atom_, int* current, int* permute,
                                   int start, int end, int offset) {
    atom_->avec->grow_stencil_md(0, atom_);
    int empty;
    for (int i = start; i < end; i++) {
        if (current[i] == permute[i]) {
            continue;
        }
        atom_->avec->copy(i + offset, end + offset, 0);
        empty = i;
        while (permute[empty] != i) {
            atom_->avec->copy(permute[empty] + offset, empty + offset, 0);
            empty = current[empty] = permute[empty];
        }
        atom_->avec->copy(end + offset, empty + offset, 0);
        current[empty] = permute[empty];
    }
}

void Verlet::sort_ghost_atoms_stencil_md(Atom* atom_, Atom* prev,
                                         queue_info& zoid, int timestep) {
    int zoid_num = zoid.num;

    queue_info& zoid_next_dt = lmp->zoid_num_to_zoid_next_dt[zoid_num];

    int* current = new int[atom_->nghost];
    int* permute = new int[atom_->nghost];

    for (int i = 0; i < atom_->nghost; i++) {
        current[i] = i;
    }

    std::map<int, std::vector<int>> idx_to_zoids;

    std::map<int, std::vector<int>> idx_to_borders_zoids;

    for (int i = 0; i < atom_->nghost; i++) {
        int idx = atom_->nlocal + i;
        double new_pos[3] = {atom_->x[idx][0], atom_->x[idx][1],
                             atom_->x[idx][2]};
        for (int dim = 0; dim < 3; dim++) {
            if (new_pos[dim] < 0) {
                new_pos[dim] += domain->prd[dim];
            }
            if (new_pos[dim] > domain->prd[dim]) {
                new_pos[dim] -= domain->prd[dim];
            }
        }

        int target_zoid_prev = -1;
        int target_zoid_next = -1;

        int target_zoid_curr = -1;
        int target_zoid_curr_next_dt = -1;

        for (int k = 0; k < NUM_ZOIDS; k++) {
            bool in_zoid_curr = true;

            queue_info& zoid_tmp = lmp->zoid_num_to_zoid_next_dt[k];
            for (int dim = 0; dim < domain->dimension; dim++) {
                double value = atom_->x[idx][dim];

                double lo_curr = zoid_tmp.zoid.cuts[dim].lower +
                                 (timestep)*zoid_tmp.zoid.cuts[dim].slope_lower;
                double hi_curr = zoid_tmp.zoid.cuts[dim].upper +
                                 (timestep)*zoid_tmp.zoid.cuts[dim].slope_upper;

                double sub = value - domain->prd[dim];
                double add = value + domain->prd[dim];

                bool at_least_one_curr = (sub >= lo_curr && sub <= hi_curr) ||
                                         (add >= lo_curr && add <= hi_curr) ||
                                         (value >= lo_curr && value <= hi_curr);

                in_zoid_curr = in_zoid_curr && at_least_one_curr;
            }

            if (in_zoid_curr) {
                assert(target_zoid_curr_next_dt == -1);
                target_zoid_curr_next_dt = k;
            }
        }

        for (int k = 0; k < NUM_ZOIDS; k++) {
            bool in_zoid_prev = true;
            bool in_zoid_next = true;

            bool in_zoid_curr = true;

            bool borders_zoid = true;

            queue_info& zoid_tmp = lmp->zoid_num_to_zoid[k];
            for (int dim = 0; dim < domain->dimension; dim++) {
                double value = atom_->x[idx][dim];

                double lo_prev =
                    zoid_tmp.zoid.cuts[dim].lower +
                    (timestep - 1) * zoid_tmp.zoid.cuts[dim].slope_lower;
                double hi_prev =
                    zoid_tmp.zoid.cuts[dim].upper +
                    (timestep - 1) * zoid_tmp.zoid.cuts[dim].slope_upper;

                // TODO: check that this somehow works?
                if (timestep == 0) {
                    queue_info& zoid_other_dt =
                        lmp->zoid_num_to_zoid_next_dt[k];
                    lo_prev = zoid_other_dt.zoid.cuts[dim].lower +
                              (NUM_TIMESTEPS_IN_PARALLEL - 1) *
                                  zoid_other_dt.zoid.cuts[dim].slope_lower;
                    hi_prev = zoid_other_dt.zoid.cuts[dim].upper +
                              (NUM_TIMESTEPS_IN_PARALLEL - 1) *
                                  zoid_other_dt.zoid.cuts[dim].slope_upper;
                }

                double lo_next =
                    zoid_tmp.zoid.cuts[dim].lower +
                    (timestep + 1) * zoid_tmp.zoid.cuts[dim].slope_lower;
                double hi_next =
                    zoid_tmp.zoid.cuts[dim].upper +
                    (timestep + 1) * zoid_tmp.zoid.cuts[dim].slope_upper;

                // TODO: debug
                if (timestep == NUM_TIMESTEPS_IN_PARALLEL) {
                    queue_info& zoid_other_dt =
                        lmp->zoid_num_to_zoid_next_dt[k];
                    lo_next = zoid_other_dt.zoid.cuts[dim].lower +
                              (1) * zoid_other_dt.zoid.cuts[dim].slope_lower;
                    hi_next = zoid_other_dt.zoid.cuts[dim].upper +
                              (1) * zoid_other_dt.zoid.cuts[dim].slope_upper;
                }

                double lo_curr = zoid_tmp.zoid.cuts[dim].lower +
                                 (timestep)*zoid_tmp.zoid.cuts[dim].slope_lower;
                double hi_curr = zoid_tmp.zoid.cuts[dim].upper +
                                 (timestep)*zoid_tmp.zoid.cuts[dim].slope_upper;

                int pbc_ = 0;
                if (zoid.where[dim] == RIGHT && zoid_tmp.where[dim] == PBC) {
                    pbc_ = -1;
                }

                if (zoid.where[dim] == PBC && zoid_tmp.where[dim]) {
                    pbc_ = 1;
                }

                double atom_pos_shifted = value + pbc_ * domain->prd[dim];

                double sub = value - domain->prd[dim];
                double add = value + domain->prd[dim];

                bool at_least_one_prev = (sub >= lo_prev && sub <= hi_prev) ||
                                         (add >= lo_prev && add <= hi_prev) ||
                                         (value >= lo_prev && value <= hi_prev);

                bool at_least_one_curr = (sub >= lo_curr && sub <= hi_curr) ||
                                         (add >= lo_curr && add <= hi_curr) ||
                                         (value >= lo_curr && value <= hi_curr);

                bool at_least_one_next = (sub >= lo_next && sub <= hi_next) ||
                                         (add >= lo_next && add <= hi_next) ||
                                         (value >= lo_next && value <= hi_next);

                /*
                in_zoid_prev = in_zoid_prev && ((atom_pos_shifted >= lo_prev && atom_pos_shifted <= hi_prev));
                in_zoid_next = in_zoid_next && ((atom_pos_shifted >= lo_next && atom_pos_shifted <= hi_next));
                in_zoid_curr = in_zoid_curr && ((atom_pos_shifted >= lo_curr && atom_pos_shifted <= hi_curr));
                */

                in_zoid_prev = in_zoid_prev && at_least_one_prev;
                in_zoid_curr = in_zoid_curr && at_least_one_curr;
                in_zoid_next = in_zoid_next && at_least_one_next;

                // double lo_borders = lo_curr - ALLEGRO_SLOPE;
                // double hi_borders = hi_curr + ALLEGRO_SLOPE;
                double lo_borders = lo_curr - ALLEGRO_CUTOFF_RADIUS;
                double hi_borders = hi_curr + ALLEGRO_CUTOFF_RADIUS;

                bool at_least_one_borders =
                    (sub >= lo_borders && sub <= hi_borders) ||
                    (add >= lo_borders && add <= hi_borders) ||
                    (value >= lo_borders && value <= hi_borders);
                borders_zoid = borders_zoid && at_least_one_borders;
            }

            borders_zoid = borders_zoid && !in_zoid_curr;

            if (borders_zoid) {
                idx_to_borders_zoids[i].push_back(k);
            }

            if (in_zoid_prev) {
                assert(target_zoid_prev == -1);
                target_zoid_prev = k;
            }

            // disable sorting by curr_zoid
            if (in_zoid_curr) {
                assert(target_zoid_curr == -1);
                target_zoid_curr = k;
            }

            if (in_zoid_next) {
                assert(target_zoid_next == -1);
                target_zoid_next = k;
            }
        }

        // group by target_zoid_prev/target_zoid_next is meant for receiving forces in as few chunks as possible
        if (timestep > 0) {
            idx_to_zoids[i].push_back(target_zoid_prev);
        } else {
            idx_to_zoids[i].push_back(-1);
        }

        if (timestep < NUM_TIMESTEPS_IN_PARALLEL) {
            idx_to_zoids[i].push_back(target_zoid_next);
        } else {
            idx_to_zoids[i].push_back(-1);
        }

        // group by target_zoid_curr/target_zoid_curr_next_dt is meant for sending ghost atoms (forces? mostly)
        // however, forces cannot be duplicated, focus on the data that can be avoided to be replicateD?
        idx_to_zoids[i].push_back(target_zoid_curr);
        idx_to_zoids[i].push_back(target_zoid_curr_next_dt);
    }

    std::vector<int> ghost_idxs;
    for (int i = 0; i < atom_->nghost; i++) {
        ghost_idxs.push_back(i);
    }

    std::set<int>& curr_dt_set = zoid.relevant_atom_tags[timestep];
    std::set<int>& next_dt_set =
        zoid_next_dt.relevant_atom_tags[NUM_TIMESTEPS_IN_PARALLEL - timestep];

    // setup bins to sort by pos?

    std::stable_sort(
        ghost_idxs.begin(), ghost_idxs.end(), [&](const int& a, const int& b) {
            /*
            bool a_curr_dt_relevant =
                (curr_dt_set.find(atom_->tag[a + atom_->nlocal]) !=
                 curr_dt_set.end());
            bool b_curr_dt_relevant =
                (curr_dt_set.find(atom_->tag[b + atom_->nlocal]) !=
                 curr_dt_set.end());

            bool a_next_dt_relevant =
                (next_dt_set.find(atom_->tag[a + atom_->nlocal]) !=
                 next_dt_set.end());
            bool b_next_dt_relevant =
                (next_dt_set.find(atom_->tag[b + atom_->nlocal]) !=
                 next_dt_set.end());

            if (a_curr_dt_relevant && !b_curr_dt_relevant) {
                return true;
            }

            if (!a_curr_dt_relevant && b_curr_dt_relevant) {
                return false;
            }

            if (a_next_dt_relevant && !b_next_dt_relevant) {
                return true;
            }

            if (!a_next_dt_relevant && b_next_dt_relevant) {
                return false;
            }
            */

            std::vector<int>& vec_a = idx_to_zoids[a];
            std::vector<int>& vec_b = idx_to_zoids[b];
            assert(vec_a.size() == vec_b.size());

            int min_vec_size = vec_a.size();

            for (int i = 0; i < 2; i++) {
                int a_proc = vec_a[i] % comm->nprocs;
                int b_proc = vec_b[i] % comm->nprocs;
                if (a_proc < b_proc) {
                    return true;
                } else if (a_proc > b_proc) {
                    return false;
                } else {
                    continue;
                }
            }

            bool a_same_curr_prev = (vec_a[0] == vec_a[2]);
            bool b_same_curr_prev = (vec_b[0] == vec_b[2]);

            if (a_same_curr_prev && !b_same_curr_prev) {
                return true;
            }

            if (!a_same_curr_prev && b_same_curr_prev) {
                return false;
            }

            bool a_same_curr_next = (vec_a[1] == vec_a[3]);
            bool b_same_curr_next = (vec_b[1] == vec_b[3]);

            if (a_same_curr_next && !b_same_curr_next) {
                return true;
            }

            if (!a_same_curr_next && b_same_curr_next) {
                return false;
            }

            for (int i = 0; i < min_vec_size; i++) {
                if (vec_a[i] < vec_b[i]) {
                    return true;
                } else if (vec_a[i] > vec_b[i]) {
                    return false;
                } else {
                    continue;
                }
            }

            // compare borders?
            std::vector<int>& borders_vec_a = idx_to_borders_zoids[a];
            std::vector<int>& borders_vec_b = idx_to_borders_zoids[b];
            int min_size = std::min(borders_vec_a.size(), borders_vec_b.size());

            for (int i = 0; i < min_size; i++) {
                int a_proc = borders_vec_a[i] % comm->nprocs;
                int b_proc = borders_vec_a[i] % comm->nprocs;
                if (a_proc < b_proc) {
                    return true;
                } else if (a_proc > b_proc) {
                    return false;
                } else {
                    continue;
                }
            }

            for (int i = 0; i < min_size; i++) {
                if (borders_vec_a[i] < borders_vec_b[i]) {
                    return true;
                } else if (borders_vec_a[i] > borders_vec_b[i]) {
                    return false;
                } else {
                    continue;
                }
            }

            if (borders_vec_a.size() < borders_vec_b.size()) {
                return true;
            } else if (borders_vec_a.size() > borders_vec_b.size()) {
                return false;
            }

            /*
            double* pos_a = atom_->x[a + atom_->nlocal];
            Point pa(pos_a[0], pos_a[1], pos_a[2]);

            double* pos_b = atom_->x[b + atom_->nlocal];
            Point pb(pos_b[0], pos_b[1], pos_b[2]);

            std::vector<Point> vec_points;
            vec_points.push_back(pa);
            vec_points.push_back(pb);

            CGAL::spatial_sort(vec_points.begin(), vec_points.end());
            if (fabs(vec_points[0][0] - pa[0]) <= 1e-5 && fabs(vec_points[0][1] - pa[1]) <= 1e-5 && fabs(vec_points[0][2] - pa[2]) <= 1e-5) {
                return true;
            } else {
                return false;
            }
            */
            return a < b;

            // return atom_->tag[a + atom_->nlocal] < atom_->tag[b + atom_->nlocal];
        });

    for (int i = 0; i < atom_->nghost; i++) {
        permute[i] = ghost_idxs[i];
    }

    atom_reorder_ghost_stencil_md(atom_, current, permute, 0, atom_->nghost,
                                  atom_->nlocal);

    bool debug = (zoid.num == 18 && timestep == 1);
    if (false) {
        for (int i = 0; i < atom_->nghost; i++) {
            int new_idx = i + atom_->nlocal;
            int old_idx = ghost_idxs[i];
            std::stringstream idx_to_zoids_str;
            for (auto& x : idx_to_zoids[old_idx]) {
                idx_to_zoids_str << x << " ";
            }

            std::stringstream idx_to_borders_zoids_str;
            for (auto& x : idx_to_borders_zoids[old_idx]) {
                idx_to_borders_zoids_str << x << " ";
            }

            std::cout << "new idx: " << i + atom_->nlocal << " old idx: " << old_idx << " idx to zoids: " << idx_to_zoids_str.str() << " borders: " << idx_to_borders_zoids_str.str()
                << " pos: " << atom_->x[new_idx][0] << " " << atom_->x[new_idx][1] << " " << atom_->x[new_idx][2] << std::endl;
        }
    }

    delete[] current;
    delete[] permute;
}

// TODO: Sort ghost atoms by zoid in previous timestep and zoid in next timestep
// Relay this information to zoids for their sendlists, we only need to ensure ghosts are contiguous. Local atoms we can try to make some compromises since
// there are so few local atoms compared to ghost. TBD though.
// this would mean that on the next dt, there will be segments, hopefully not too many, but we shall see I guess
void Verlet::group_ghost_atoms_stencil_md(Atom* atom_, Atom* prev,
                                          queue_info& zoid, int timestep) {
    int zoid_num = zoid.num;

    std::map<int, std::vector<int>> neighbor_to_idxs;

    auto& recv_from = lmp->recv_from_neighbors[zoid.num];

    std::vector<int> test_nonrelevant_idxs;
    for (int i = 0; i < atom_->nghost; i++) {
        int actual_idx = i + atom_->nlocal;
        int tag = atom_->tag[actual_idx];

        std::set<int>& curr_dt_set = zoid.relevant_atom_tags[timestep];

        queue_info& next_dt_zoid = lmp->zoid_num_to_zoid_next_dt[zoid_num];
        std::set<int>& next_dt_set =
            next_dt_zoid
                .relevant_atom_tags[NUM_TIMESTEPS_IN_PARALLEL - timestep];
        bool curr_dt_relevant = (curr_dt_set.find(tag) != curr_dt_set.end());
        bool next_dt_relevant = (next_dt_set.find(tag) != next_dt_set.end());
    }

    std::vector<int> test_segment_idxs;
    std::vector<int> test_segment_sizes;
    int test_num_segments = get_segments(test_nonrelevant_idxs,
                                         test_segment_idxs, test_segment_sizes);

    for (int i = 0; i < atom_->nghost; i++) {
        int actual_idx = i + atom_->nlocal;
        double* pos = atom_->x[actual_idx];

        for (int j = 0; j < recv_from.size(); j++) {
            int recv_zoid_num = recv_from[j];
            queue_info& recv_from_zoid = lmp->zoid_num_to_zoid[recv_zoid_num];

            double lo_track[3] = {0};
            double hi_track[3] = {0};

            double lo_prev_track[3] = {0};
            double hi_prev_track[3] = {0};

            // recv_from is neighbor
            bool pbc_flags[3];
            // technically check if borders zoid
            bool in_zoid_prev = true;

            for (int dim = 0; dim < domain->dimension; dim++) {
                bool in_zoid_curr_dim = true;

                double value = pos[dim];

                int pbc_ = 0;
                if (zoid.where[dim] == RIGHT &&
                    recv_from_zoid.where[dim] == PBC) {
                    pbc_ = -1;
                }
                if (zoid.where[dim] == PBC &&
                    recv_from_zoid.where[dim] == RIGHT) {
                    pbc_ = 1;
                }

                double atom_pos_shifted = value + pbc_ * domain->prd[dim];

                double lo_prev =
                    recv_from_zoid.zoid.cuts[dim].lower +
                    (timestep - 1) * recv_from_zoid.zoid.cuts[dim].slope_lower;
                double hi_prev =
                    recv_from_zoid.zoid.cuts[dim].upper +
                    (timestep - 1) * recv_from_zoid.zoid.cuts[dim].slope_upper;

                double sub = value - domain->prd[dim];
                double add = value + domain->prd[dim];

                bool at_least_one_prev = (sub >= lo_prev && sub <= hi_prev) ||
                                         (add >= lo_prev && add <= hi_prev) ||
                                         (value >= lo_prev && value <= hi_prev);

                double lo_curr = zoid.zoid.cuts[dim].lower +
                                 (timestep)*zoid.zoid.cuts[dim].slope_lower;
                double hi_curr = zoid.zoid.cuts[dim].upper +
                                 (timestep)*zoid.zoid.cuts[dim].slope_upper;

                // if expanding zoid AND other zoid shrinking, use current instead of past, only need values for actual shrinking?
                // must be a ghost atom that wasn't local last timestep somehow
                if (timestep > 0) {
                    // in_zoid_prev = in_zoid_prev && ((atom_pos_shifted >= lo_prev && atom_pos_shifted <= hi_prev));
                    in_zoid_prev = in_zoid_prev && at_least_one_prev;
                } else {
                    in_zoid_prev = false;
                }

                // if (!(value >= lo_curr && value <= hi_curr) && zoid.zoid.cuts[dim].slope_lower < 0 && timestep < NUM_TIMESTEPS_IN_PARALLEL) {
                if (!(value >= lo_curr && value <= hi_curr) &&
                    zoid.zoid.cuts[dim].slope_lower < 0 && !PURELY_LOCAL_POTENTIAL) {
                    in_zoid_prev = false;
                }
            }

            // TODO: this might not be the best
            if (zoid.relevant_atom_idxs[timestep].find(actual_idx) ==
                zoid.relevant_atom_idxs[timestep].end()) {
                // assert(!PURELY_LOCAL_POTENTIAL);
                in_zoid_prev = false;
            }

            if (in_zoid_prev) {
                neighbor_to_idxs[recv_zoid_num].push_back(actual_idx);
            }
        }
    }

    bool debug = (zoid.num == 18 && timestep == 1);
    if (false) {
        for (auto& [recv_zoid_num, idxs] : neighbor_to_idxs) {
            std::stringstream debug_idx;
            for (auto& idx : idxs) {
                std::cout << "zoid: " << zoid.num << " recv from: " << recv_zoid_num << " idx: " << idx << " pos: " << atom_->x[idx][0] << " " << atom_->x[idx][1] << " " << atom_->x[idx][2]
                    << " relevant? " << std::endl;
            }
        }

        assert(false);
    }


    for (int i = 0; i < recv_from.size(); i++) {
        int recv_from_zoid_num = recv_from[i];
        if (neighbor_to_idxs[recv_from_zoid_num].size() == 0) {
            zoid.recv_ghost_idxs[timestep][i] = new int[1];
            zoid.recv_ghost_idxs[timestep][i][0] = 0;
            zoid.recv_ghost_sizes[timestep][i] = new int[1];
            zoid.recv_ghost_sizes[timestep][i][0] = 0;
            zoid.recv_ghost_num_segments[timestep][i] = 0;
        } else {
            std::vector<int> segment_idxs;
            std::vector<int> segment_lengths;

            auto& idx_vec = neighbor_to_idxs[recv_from_zoid_num];

            int num_segments =
                get_segments(idx_vec, segment_idxs, segment_lengths);

            /*
            if (num_segments > 10) {
                std::cout << GREEN << "CURR DT zoid: " << zoid_num
                          << " recv from: " << recv_from_zoid_num
                          << " time: " << timestep
                          << " num segments: " << num_segments << RESET_COLOR
                          << std::endl;
                std::cout << "segment idxs: " << segment_idxs << std::endl;
                std::cout << "segment lengths? " << segment_lengths
                          << std::endl;
            }
            */

            zoid.recv_ghost_idxs[timestep][i] = new int[num_segments];
            zoid.recv_ghost_sizes[timestep][i] = new int[num_segments];
            zoid.recv_ghost_num_segments[timestep][i] = num_segments;

            for (int j = 0; j < segment_idxs.size(); j++) {
                zoid.recv_ghost_idxs[timestep][i][j] = segment_idxs[j];
                zoid.recv_ghost_sizes[timestep][i][j] = segment_lengths[j];
            }
        }
    }
}

// does the same thing as group_ghost but without the reordering as that has been done already
void Verlet::group_ghost_atoms_stencil_md_next_dt(Atom* atom_, Atom* prev,
                                                  queue_info& zoid,
                                                  int timestep) {
    std::map<int, std::vector<int>> neighbor_to_idxs;

    auto& recv_from = lmp->recv_from_neighbors_next_dt[zoid.num];

    std::set<int>& next_dt_set = zoid.relevant_atom_tags[timestep];
    queue_info& curr_dt_zoid = lmp->zoid_num_to_zoid[zoid.num];
    std::set<int>& curr_dt_set =
        curr_dt_zoid.relevant_atom_tags[NUM_TIMESTEPS_IN_PARALLEL - timestep];

    std::vector<int> test_nonrelevant_idxs;
    for (int i = 0; i < atom_->nghost; i++) {
        int actual_idx = i + atom_->nlocal;
        int tag = atom_->tag[actual_idx];
        bool curr_dt_relevant = (curr_dt_set.find(tag) != curr_dt_set.end());
        bool next_dt_relevant = (next_dt_set.find(tag) != next_dt_set.end());
    }

    std::vector<int> test_segment_idxs;
    std::vector<int> test_segment_sizes;
    int test_num_segments = get_segments(test_nonrelevant_idxs,
                                         test_segment_idxs, test_segment_sizes);
    /*
    if (test_nonrelevant_idxs.size() > 0) {
        std::cout << "next dt test num segments: " << test_num_segments
                  << " idxs: " << test_segment_idxs
                  << " sizes: " << test_segment_sizes << std::endl;
    }
    */

    for (int i = 0; i < atom_->nghost; i++) {
        int actual_idx = i + atom_->nlocal;
        double* pos = atom_->x[actual_idx];

        for (int j = 0; j < recv_from.size(); j++) {
            int recv_zoid_num = recv_from[j];
            queue_info& recv_from_zoid =
                lmp->zoid_num_to_zoid_next_dt[recv_zoid_num];

            double lo_track[3] = {0};
            double hi_track[3] = {0};

            double lo_prev_track[3] = {0};
            double hi_prev_track[3] = {0};

            // recv_from is neighbor
            bool pbc_flags[3];
            // technically check if borders zoid
            bool in_zoid_prev = true;

            for (int dim = 0; dim < domain->dimension; dim++) {
                bool in_zoid_curr_dim = true;

                double value = pos[dim];

                int pbc_ = 0;
                if (zoid.where[dim] == RIGHT &&
                    recv_from_zoid.where[dim] == PBC) {
                    pbc_ = -1;
                }
                if (zoid.where[dim] == PBC &&
                    recv_from_zoid.where[dim] == RIGHT) {
                    pbc_ = 1;
                }

                double atom_pos_shifted = value + pbc_ * domain->prd[dim];

                double lo_prev =
                    recv_from_zoid.zoid.cuts[dim].lower +
                    (timestep - 1) * recv_from_zoid.zoid.cuts[dim].slope_lower;
                double hi_prev =
                    recv_from_zoid.zoid.cuts[dim].upper +
                    (timestep - 1) * recv_from_zoid.zoid.cuts[dim].slope_upper;

                double sub = value - domain->prd[dim];
                double add = value + domain->prd[dim];

                bool at_least_one_prev = (sub >= lo_prev && sub <= hi_prev) ||
                                         (add >= lo_prev && add <= hi_prev) ||
                                         (value >= lo_prev && value <= hi_prev);

                // if expanding zoid AND other zoid shrinking, use current instead of past, only need values for actual shrinking?
                // must be a ghost atom that wasn't local last timestep somehow
                if (timestep > 0) {
                    // in_zoid_prev = in_zoid_prev && ((atom_pos_shifted >= lo_prev && atom_pos_shifted <= hi_prev));
                    in_zoid_prev = in_zoid_prev && at_least_one_prev;
                } else {
                    in_zoid_prev = false;
                }

                double lo_curr = zoid.zoid.cuts[dim].lower +
                                 (timestep)*zoid.zoid.cuts[dim].slope_lower;
                double hi_curr = zoid.zoid.cuts[dim].upper +
                                 (timestep)*zoid.zoid.cuts[dim].slope_upper;

                // if (!(value >= lo_curr && value <= hi_curr) && zoid.zoid.cuts[dim].slope_lower < 0 && timestep < NUM_TIMESTEPS_IN_PARALLEL) {
                if (!(value >= lo_curr && value <= hi_curr) &&
                    zoid.zoid.cuts[dim].slope_lower < 0 && !PURELY_LOCAL_POTENTIAL) {
                    in_zoid_prev = false;
                }
            }

            // TODO: this might not be the best
            if (zoid.relevant_atom_tags[timestep].find(
                    atom_->tag[actual_idx]) ==
                zoid.relevant_atom_tags[timestep].end()) {
                // assert(!PURELY_LOCAL_POTENTIAL);
                in_zoid_prev = false;
            }

            if (in_zoid_prev) {
                neighbor_to_idxs[recv_zoid_num].push_back(actual_idx);
            }
        }
    }

    for (int i = 0; i < recv_from.size(); i++) {
        int recv_from_zoid_num = recv_from[i];
        if (neighbor_to_idxs[recv_from_zoid_num].size() == 0) {
            zoid.recv_ghost_idxs[timestep][i] = new int[1];
            zoid.recv_ghost_idxs[timestep][i][0] = 0;
            zoid.recv_ghost_sizes[timestep][i] = new int[1];
            zoid.recv_ghost_sizes[timestep][i][0] = 0;
            zoid.recv_ghost_num_segments[timestep][i] = 0;
        } else {
            std::vector<int> segment_idxs;
            std::vector<int> segment_lengths;

            auto& idx_vec = neighbor_to_idxs[recv_from_zoid_num];

            int num_segments =
                get_segments(idx_vec, segment_idxs, segment_lengths);

            /*
            if (num_segments > 10) {
                std::cout << MAGENTA << "NEXT DT zoid: " << zoid.num
                          << " recv from: " << recv_from_zoid_num
                          << " time: " << timestep
                          << " num segments: " << num_segments
                          << "segment idxs: " << segment_idxs
                          << " segment lengths? " << segment_lengths
                          << RESET_COLOR << std::endl;
            }
            */

            zoid.recv_ghost_idxs[timestep][i] = new int[num_segments];
            zoid.recv_ghost_sizes[timestep][i] = new int[num_segments];
            zoid.recv_ghost_num_segments[timestep][i] = num_segments;

            for (int j = 0; j < num_segments; j++) {
                zoid.recv_ghost_idxs[timestep][i][j] = segment_idxs[j];
                zoid.recv_ghost_sizes[timestep][i][j] = segment_lengths[j];
            }
        }
    }
}

void setup_can_eval_center_mapping_stencil_md(
    std::array<Atom*, NUM_TIMESTEPS_IN_PARALLEL + 1>& atom_arr,
    queue_info& zoid) {
    for (int t = 0; t < NUM_TIMESTEPS_IN_PARALLEL + 1; t++) {
        Atom* atom_ = atom_arr[t];
        int total = atom_->nlocal + atom_->nghost;
        zoid.can_eval_center[t] = new bool[total];
        zoid.can_eval_pos[t] = new bool[total];

        for (int i = 0; i < total; i++) {
            zoid.can_eval_center[t][i] = false;
            zoid.can_eval_pos[t][i] = false;
        }

        for (int i = 0; i < atom_->nlocal; i++) {
            zoid.can_eval_pos[t][i] = true;
        }

        for (int i = 0; i < atom_->nlocal + atom_->nghost; i++) {
            double* pos = atom_->x[i];
            int diffs[3] = {0};
            bool can_eval_center = true;

            bool can_eval_center_debug[3] = {false, false, false};
            double lo_debug[3] = {0};
            double hi_debug[3] = {0};

            for (int dim = 0; dim < 3; dim++) {
                bool shrinking_dim = zoid.zoid.cuts[dim].slope_lower > 0;
                double lo = zoid.zoid.cuts[dim].lower +
                            t * zoid.zoid.cuts[dim].slope_lower;
                double hi = zoid.zoid.cuts[dim].upper +
                            t * zoid.zoid.cuts[dim].slope_upper;

                bool in_bounds = (pos[dim] >= lo && pos[dim] <= hi);

                double diff =
                    std::min(fabs(pos[dim] - lo), fabs(pos[dim] - hi));

                if (shrinking_dim) {
                    can_eval_center =
                        can_eval_center &&
                        (in_bounds || (diff <= ALLEGRO_CUTOFF_RADIUS));
                    can_eval_center_debug[dim] =
                        (in_bounds || (diff <= ALLEGRO_CUTOFF_RADIUS));
                } else {
                    can_eval_center = can_eval_center && in_bounds &&
                                      (diff > ALLEGRO_CUTOFF_RADIUS);
                    can_eval_center_debug[dim] =
                        (in_bounds && diff > ALLEGRO_CUTOFF_RADIUS);
                }
                lo_debug[dim] = lo;
                hi_debug[dim] = hi;
            }

            if (PURELY_LOCAL_POTENTIAL) {
                // can_eval_center = (i < atom_->nlocal && can_eval_center);
                can_eval_center = (i < atom_->nlocal);
            }

            zoid.can_eval_center[t][i] = can_eval_center;
        }
    }
}

void setup_can_eval_center_tag_mapping_stencil_md(
    std::array<Atom*, NUM_TIMESTEPS_IN_PARALLEL + 1>& atom_arr,
    queue_info& zoid) {
    for (int t = 0; t < NUM_TIMESTEPS_IN_PARALLEL + 1; t++) {
        Atom* atom_ = atom_arr[t];

        for (int i = 0; i < atom_->nlocal + atom_->nghost; i++) {
            double* pos = atom_->x[i];
            int diffs[3] = {0};
            bool can_eval_center = true;

            bool can_eval_center_debug[3] = {false, false, false};
            double lo_debug[3] = {0};
            double hi_debug[3] = {0};

            for (int dim = 0; dim < 3; dim++) {
                bool shrinking_dim = zoid.zoid.cuts[dim].slope_lower > 0;
                double lo = zoid.zoid.cuts[dim].lower +
                            t * zoid.zoid.cuts[dim].slope_lower;
                double hi = zoid.zoid.cuts[dim].upper +
                            t * zoid.zoid.cuts[dim].slope_upper;

                bool in_bounds = (pos[dim] >= lo && pos[dim] <= hi);

                double diff =
                    std::min(fabs(pos[dim] - lo), fabs(pos[dim] - hi));

                if (shrinking_dim) {
                    can_eval_center =
                        can_eval_center &&
                        (in_bounds || (diff <= ALLEGRO_CUTOFF_RADIUS));
                    can_eval_center_debug[dim] =
                        (in_bounds || (diff <= ALLEGRO_CUTOFF_RADIUS));
                } else {
                    can_eval_center = can_eval_center && in_bounds &&
                                      (diff > ALLEGRO_CUTOFF_RADIUS);
                    can_eval_center_debug[dim] =
                        (in_bounds && diff > ALLEGRO_CUTOFF_RADIUS);
                }
                lo_debug[dim] = lo;
                hi_debug[dim] = hi;
            }

            if (PURELY_LOCAL_POTENTIAL) {
                // can_eval_center = (i < atom_->nlocal && can_eval_center);
                can_eval_center = (i < atom_->nlocal);
            }

            if (can_eval_center) {
                zoid.can_eval_center_tags[t].insert(atom_->tag[i]);
            }
        }
    }
}

void setup_can_eval_center_tag_mapping_stencil_md_next_dt(
    std::array<Atom*, NUM_TIMESTEPS_IN_PARALLEL + 1>& atom_arr,
    queue_info& zoid) {
    for (int t = 0; t < NUM_TIMESTEPS_IN_PARALLEL + 1; t++) {
        Atom* atom_ = atom_arr[NUM_TIMESTEPS_IN_PARALLEL - t];

        for (int i = 0; i < atom_->nlocal + atom_->nghost; i++) {
            double* pos = atom_->x[i];
            int diffs[3] = {0};
            bool can_eval_center = true;

            bool can_eval_center_debug[3] = {false, false, false};
            double lo_debug[3] = {0};
            double hi_debug[3] = {0};

            for (int dim = 0; dim < 3; dim++) {
                bool shrinking_dim = zoid.zoid.cuts[dim].slope_lower > 0;
                double lo = zoid.zoid.cuts[dim].lower +
                            t * zoid.zoid.cuts[dim].slope_lower;
                double hi = zoid.zoid.cuts[dim].upper +
                            t * zoid.zoid.cuts[dim].slope_upper;

                bool in_bounds = (pos[dim] >= lo && pos[dim] <= hi);

                double diff =
                    std::min(fabs(pos[dim] - lo), fabs(pos[dim] - hi));

                if (shrinking_dim) {
                    can_eval_center =
                        can_eval_center &&
                        (in_bounds || (diff <= ALLEGRO_CUTOFF_RADIUS));
                    can_eval_center_debug[dim] =
                        (in_bounds || (diff <= ALLEGRO_CUTOFF_RADIUS));
                } else {
                    can_eval_center = can_eval_center && in_bounds &&
                                      (diff > ALLEGRO_CUTOFF_RADIUS);
                    can_eval_center_debug[dim] =
                        (in_bounds && diff > ALLEGRO_CUTOFF_RADIUS);
                }
                lo_debug[dim] = lo;
                hi_debug[dim] = hi;
            }

            if (PURELY_LOCAL_POTENTIAL) {
                // can_eval_center = (i < atom_->nlocal && can_eval_center);
                can_eval_center = (i < atom_->nlocal);
            }

            if (can_eval_center) {
                zoid.can_eval_center_tags[t].insert(atom_->tag[i]);
            }
        }
    }
}

void setup_can_eval_center_mapping_stencil_md_next_dt(
    std::array<Atom*, NUM_TIMESTEPS_IN_PARALLEL + 1>& atom_arr,
    queue_info& zoid) {

    for (int t = 0; t < NUM_TIMESTEPS_IN_PARALLEL + 1; t++) {
        Atom* atom_ = atom_arr[NUM_TIMESTEPS_IN_PARALLEL - t];
        int total = atom_->nlocal + atom_->nghost;
        zoid.can_eval_center[t] = new bool[total];
        zoid.can_eval_pos[t] = new bool[total];

        for (int i = 0; i < total; i++) {
            zoid.can_eval_center[t][i] = false;
            zoid.can_eval_pos[t][i] = false;
        }

        for (int i = 0; i < atom_->nlocal; i++) {
            zoid.can_eval_pos[t][i] = true;
        }

        for (int i = 0; i < atom_->nlocal + atom_->nghost; i++) {
            double* pos = atom_->x[i];
            int diffs[3] = {0};
            bool can_eval_center = true;

            for (int dim = 0; dim < 3; dim++) {
                bool shrinking_dim = zoid.zoid.cuts[dim].slope_lower > 0;
                double lo = zoid.zoid.cuts[dim].lower +
                            t * zoid.zoid.cuts[dim].slope_lower;
                double hi = zoid.zoid.cuts[dim].upper +
                            t * zoid.zoid.cuts[dim].slope_upper;

                bool in_bounds = (pos[dim] >= lo && pos[dim] <= hi);

                double diff =
                    std::min(fabs(pos[dim] - lo), fabs(pos[dim] - hi));

                if (shrinking_dim) {
                    can_eval_center =
                        can_eval_center &&
                        (in_bounds || (diff <= ALLEGRO_CUTOFF_RADIUS));
                } else {
                    can_eval_center = can_eval_center && in_bounds &&
                                      (diff > ALLEGRO_CUTOFF_RADIUS);
                }
            }

            if (PURELY_LOCAL_POTENTIAL) {
                // can_eval_center = (i < atom_->nlocal && can_eval_center);
                can_eval_center = (i < atom_->nlocal);
            }

            zoid.can_eval_center[t][i] = can_eval_center;
        }
    }
}

void setup_atom_relevant_idxs_stencil_md(
    std::array<Atom*, NUM_TIMESTEPS_IN_PARALLEL + 1>& atom_arr,
    std::array<Force*, NUM_TIMESTEPS_IN_PARALLEL + 1>& force_arr,
    queue_info& zoid) {
    for (int t = 0; t < NUM_TIMESTEPS_IN_PARALLEL + 1; t++) {
        Force* force_ = force_arr[t];
        NeighList* list = force_->pair->list;
        Atom* atom_ = atom_arr[t];

        for (int i = 0; i < atom_->nlocal + atom_->nghost; i++) {
            if (PURELY_LOCAL_POTENTIAL && !TRY_PRECOMPUTE_RELEVANT_ATOM_IDX) {
                zoid.relevant_atom_idxs[t].insert(i);
                continue;
            }
            if (zoid.can_eval_center[t][i]) {
                zoid.relevant_atom_idxs[t].insert(i);

                int list_idx = list->ilist[i];
                if (list_idx != i) {
                    std::cout << " zoid: " << zoid.num << " time: " << t
                              << " list idx: " << list_idx << " i: " << i
                              << std::endl;
                }
                assert(list_idx == i);
                int num_neigh = list->numneigh[i];
                int* neigh_list = list->firstneigh[i];

                for (int neigh_idx = 0; neigh_idx < num_neigh; neigh_idx++) {
                    int neigh = neigh_list[neigh_idx];
                    double* my_pos = atom_->x[i];
                    double* neigh_pos = atom_->x[neigh];
                    double delx = my_pos[0] - neigh_pos[0];  // xj - xi
                    double dely = my_pos[1] - neigh_pos[1];  // xj - xi
                    double delz = my_pos[2] - neigh_pos[2];  // xj - xi
                    double rsq = delx * delx + dely * dely + delz * delz;
                    // if (rsq < ALLEGRO_CUTOFF_RADIUS * ALLEGRO_CUTOFF_RADIUS && rsq > 1e-20) {
                    zoid.relevant_atom_idxs[t].insert(neigh);
                }
            }
        }
    }
}

void setup_atom_relevant_tags_stencil_md(
    std::array<Atom*, NUM_TIMESTEPS_IN_PARALLEL + 1>& atom_arr,
    std::array<Force*, NUM_TIMESTEPS_IN_PARALLEL + 1>& force_arr,
    queue_info& zoid) {
    for (int t = 0; t < NUM_TIMESTEPS_IN_PARALLEL + 1; t++) {
        Force* force_ = force_arr[t];
        NeighList* list = force_->pair->list;
        Atom* atom_ = atom_arr[t];

        for (int i = 0; i < atom_->nlocal + atom_->nghost; i++) {
            if (PURELY_LOCAL_POTENTIAL && !TRY_PRECOMPUTE_RELEVANT_ATOM_IDX) {
                zoid.relevant_atom_tags[t].insert(atom_->tag[i]);
                continue;
            }
            if (zoid.can_eval_center_tags[t].find(atom_->tag[i]) !=
                zoid.can_eval_center_tags[t].end()) {
                zoid.relevant_atom_tags[t].insert(atom_->tag[i]);
                assert(i < atom_->nlocal);

                int list_idx = list->ilist[i];
                if (list_idx != i) {
                    std::cout << " zoid: " << zoid.num << " time: " << t
                              << " list idx: " << list_idx << " i: " << i
                              << std::endl;
                }
                assert(list_idx == i);
                int num_neigh = list->numneigh[i];
                int* neigh_list = list->firstneigh[i];

                for (int neigh_idx = 0; neigh_idx < num_neigh; neigh_idx++) {
                    int neigh = neigh_list[neigh_idx];
                    double* my_pos = atom_->x[i];
                    double* neigh_pos = atom_->x[neigh];
                    double delx = my_pos[0] - neigh_pos[0];  // xj - xi
                    double dely = my_pos[1] - neigh_pos[1];  // xj - xi
                    double delz = my_pos[2] - neigh_pos[2];  // xj - xi
                    double rsq = delx * delx + dely * dely + delz * delz;
                    zoid.relevant_atom_tags[t].insert(atom_->tag[neigh]);
                }
            }
        }
    }
}

void setup_atom_relevant_tags_stencil_md_next_dt(
    std::array<Atom*, NUM_TIMESTEPS_IN_PARALLEL + 1>& atom_arr,
    std::array<Force*, NUM_TIMESTEPS_IN_PARALLEL + 1>& force_arr,
    queue_info& zoid) {
    for (int t = 0; t < NUM_TIMESTEPS_IN_PARALLEL + 1; t++) {
        Force* force_;
        if (PURELY_LOCAL_POTENTIAL) {
            force_ = force_arr[t];
        } else {
            force_ = force_arr[NUM_TIMESTEPS_IN_PARALLEL - t];
        }

        NeighList* list = force_->pair->list;
        Atom* atom_ = atom_arr[NUM_TIMESTEPS_IN_PARALLEL - t];

        for (int i = 0; i < atom_->nlocal + atom_->nghost; i++) {
            if (PURELY_LOCAL_POTENTIAL && !TRY_PRECOMPUTE_RELEVANT_ATOM_IDX) {
                zoid.relevant_atom_tags[t].insert(atom_->tag[i]);
                continue;
            }

            if (zoid.can_eval_center_tags[t].find(atom_->tag[i]) !=
                zoid.can_eval_center_tags[t].end()) {
                zoid.relevant_atom_tags[t].insert(atom_->tag[i]);

                int list_idx = list->ilist[i];
                if (list_idx != i) {
                    std::cout << "i: " << i << " list idx: " << list_idx << " inum: " << list->inum << " nlocal: " << atom_->nlocal << std::endl;
                }
                assert(list_idx == i);
                int num_neigh = list->numneigh[i];
                int* neigh_list = list->firstneigh[i];

                for (int neigh_idx = 0; neigh_idx < num_neigh; neigh_idx++) {
                    int neigh = neigh_list[neigh_idx];
                    double* my_pos = atom_->x[i];
                    double* neigh_pos = atom_->x[neigh];
                    double delx = my_pos[0] - neigh_pos[0];  // xj - xi
                    double dely = my_pos[1] - neigh_pos[1];  // xj - xi
                    double delz = my_pos[2] - neigh_pos[2];  // xj - xi
                    double rsq = delx * delx + dely * dely + delz * delz;
                    // if (rsq < ALLEGRO_CUTOFF_RADIUS * ALLEGRO_CUTOFF_RADIUS && rsq > 1e-20) {
                    zoid.relevant_atom_tags[t].insert(atom_->tag[neigh]);
                }
            }
        }
    }
}

void setup_atom_relevant_idxs_stencil_md_next_dt(
    std::array<Atom*, NUM_TIMESTEPS_IN_PARALLEL + 1>& atom_arr,
    std::array<Force*, NUM_TIMESTEPS_IN_PARALLEL + 1>& force_arr,
    queue_info& zoid) {
    for (int t = 0; t < NUM_TIMESTEPS_IN_PARALLEL + 1; t++) {
        Force* force_;
        if (PURELY_LOCAL_POTENTIAL) {
            force_ = force_arr[t];
        } else {
            force_ = force_arr[NUM_TIMESTEPS_IN_PARALLEL - t];
        }
        NeighList* list = force_->pair->list;
        Atom* atom_ = atom_arr[NUM_TIMESTEPS_IN_PARALLEL - t];

        for (int i = 0; i < atom_->nlocal + atom_->nghost; i++) {
            if (PURELY_LOCAL_POTENTIAL && !TRY_PRECOMPUTE_RELEVANT_ATOM_IDX) {
                zoid.relevant_atom_idxs[t].insert(i);
                continue;
            }
            if (zoid.can_eval_center[t][i]) {
                zoid.relevant_atom_idxs[t].insert(i);

                int list_idx = list->ilist[i];
                assert(list_idx == i);
                int num_neigh = list->numneigh[i];
                int* neigh_list = list->firstneigh[i];

                for (int neigh_idx = 0; neigh_idx < num_neigh; neigh_idx++) {
                    int neigh = neigh_list[neigh_idx];
                    double* my_pos = atom_->x[i];
                    double* neigh_pos = atom_->x[neigh];
                    double delx = my_pos[0] - neigh_pos[0];  // xj - xi
                    double dely = my_pos[1] - neigh_pos[1];  // xj - xi
                    double delz = my_pos[2] - neigh_pos[2];  // xj - xi
                    double rsq = delx * delx + dely * dely + delz * delz;
                    // if (rsq < ALLEGRO_CUTOFF_RADIUS * ALLEGRO_CUTOFF_RADIUS && rsq > 1e-20) {
                    zoid.relevant_atom_idxs[t].insert(neigh);
                }
            }
        }
    }
}

// create an arr, permute_esque
void setup_atom_pos_mapping_stencil_md(
    std::array<Atom*, NUM_TIMESTEPS_IN_PARALLEL + 1>& atom_arr,
    queue_info& zoid) {
    int max_atoms = -1;
    for (int i = 0; i < atom_arr.size(); i++) {
        Atom* atom_ = atom_arr[i];
        int num_atoms = atom_->nlocal + atom_->nghost;
        if (num_atoms > max_atoms) {
            max_atoms = num_atoms;
        }
    }

    for (int i = 0; i < NUM_TIMESTEPS_IN_PARALLEL + 1; i++) {
        zoid.atom_idx_mapping[i] = new int[max_atoms];
    }

    for (int i = 0; i < NUM_TIMESTEPS_IN_PARALLEL + 1; i++) {
        for (int j = 0; j < max_atoms; j++) {
            zoid.atom_idx_mapping[i][j] = -1;
        }
    }

    for (int t = 0; t < NUM_TIMESTEPS_IN_PARALLEL; t++) {
        Atom* atom_ = atom_arr[t];
        Atom* next = atom_arr[t + 1];
        std::map<int, int> curr_tag_to_idx;
        std::map<int, int> next_tag_to_idx;

        for (int i = 0; i < atom_->nlocal + atom_->nghost; i++) {
            int tag = atom_->tag[i];
            curr_tag_to_idx[tag] = i;
        }
        for (int i = 0; i < next->nlocal + next->nghost; i++) {
            int tag = next->tag[i];
            next_tag_to_idx[tag] = i;
        }

        for (int i = 0; i < atom_->nlocal + atom_->nghost; i++) {
            int next_idx = -1;
            int curr_tag = atom_->tag[i];
            if (next_tag_to_idx.count(curr_tag)) {
                next_idx = next_tag_to_idx[curr_tag];
            }

            zoid.atom_idx_mapping[t][i] = next_idx;
        }
    }
}

// same as before except `next` is t - 1 rather than t + 1
void setup_atom_pos_mapping_stencil_md_next_dt(
    std::array<Atom*, NUM_TIMESTEPS_IN_PARALLEL + 1>& atom_arr,
    queue_info& zoid) {
    int max_atoms = -1;
    for (int i = 0; i < atom_arr.size(); i++) {
        Atom* atom_ = atom_arr[i];
        int num_atoms = atom_->nlocal + atom_->nghost;
        if (num_atoms > max_atoms) {
            max_atoms = num_atoms;
        }
    }

    for (int i = 0; i < NUM_TIMESTEPS_IN_PARALLEL + 1; i++) {
        zoid.atom_idx_mapping[i] = new int[max_atoms];
    }

    for (int i = 0; i < NUM_TIMESTEPS_IN_PARALLEL + 1; i++) {
        for (int j = 0; j < max_atoms; j++) {
            zoid.atom_idx_mapping[i][j] = -1;
        }
    }

    for (int t = 0; t < NUM_TIMESTEPS_IN_PARALLEL; t++) {
        int idx = NUM_TIMESTEPS_IN_PARALLEL - t;
        int idx_next = NUM_TIMESTEPS_IN_PARALLEL - t - 1;

        Atom* atom_ = atom_arr[idx];
        Atom* next = atom_arr[idx_next];

        std::map<int, int> curr_tag_to_idx;
        std::map<int, int> next_tag_to_idx;

        for (int i = 0; i < atom_->nlocal + atom_->nghost; i++) {
            int tag = atom_->tag[i];
            curr_tag_to_idx[tag] = i;
        }
        for (int i = 0; i < next->nlocal + next->nghost; i++) {
            int tag = next->tag[i];
            next_tag_to_idx[tag] = i;
        }

        for (int i = 0; i < atom_->nlocal + atom_->nghost; i++) {
            int next_idx = -1;
            int curr_tag = atom_->tag[i];
            if (next_tag_to_idx.count(curr_tag)) {
                next_idx = next_tag_to_idx[curr_tag];
            }

            zoid.atom_idx_mapping[t][i] = next_idx;
        }
    }
}

void Verlet::setup_stencil_md() {
    // assert(3 * 2 * ALLEGRO_SLOPE * NUM_TIMESTEPS_IN_PARALLEL <= domain->prd[0]);
    stencilMD->INIT_ZOIDS();
    stencilMD->INIT_ZOID_DATA();
    stencilMD->INIT_ZOID_NEIGHBORS();

    if (comm->me == 0) {
        std::map<int, std::set<int>> test;
        for (int dep = 0; dep < NUM_DEPS; dep++) {
            std::cout << "num zoids in dep: " << dep
                      << " is: " << lmp->queues[dep].size() << std::endl;
        }

        for (int i = 0; i < NUM_ZOIDS; i++) {
            for (int recipient : lmp->send_to_neighbors[i]) {
                test[recipient].insert(i);
            }
        }


        for (int i = 0; i < NUM_ZOIDS; i++) {
            /*
            if (test[i].size() != lmp->recv_from_neighbors[i].size()) {
                std::cout << "ZOID NUM: " << i << " sent: " << test[i]
                          << " recv: " << lmp->recv_from_neighbors[i]
                          << std::endl;
            }
            */
            assert(test[i].size() == lmp->recv_from_neighbors[i].size());
        }

        // test
        std::map<int, std::set<int>> test_next_dt;
        for (int dep = 0; dep < NUM_DEPS; dep++) {
            std::cout << "num zoids in dep: " << dep
                      << " is: " << lmp->queues_next_dt[dep].size()
                      << std::endl;
        }

        for (int i = 0; i < NUM_ZOIDS; i++) {
            for (int recipient : lmp->send_to_neighbors_next_dt[i]) {
                test_next_dt[recipient].insert(i);
            }
        }

        for (int i = 0; i < NUM_ZOIDS; i++) {
            /*
            if (test_next_dt[i].size() !=
                lmp->recv_from_neighbors_next_dt[i].size()) {
                std::cout << "ZOID NUM: " << i << " sent: " << test[i]
                          << " recv: " << lmp->recv_from_neighbors_next_dt[i]
                          << std::endl;
            }
            */
            assert(test_next_dt[i].size() ==
                   lmp->recv_from_neighbors_next_dt[i].size());
        }
    }

    // print out send_to_recv info
    /*
    if (comm->me == 0) {
        for (int i = 0; i < NUM_ZOIDS; i++) {
            queue_info& zoid = lmp->zoid_num_to_zoid[i];
            std::cout << "zoid: " << i << " lo: " << zoid.zoid.cuts[0].lower
                      << " " << zoid.zoid.cuts[1].lower << " "
                      << zoid.zoid.cuts[2].lower << std::endl;
            std::cout << "zoid: " << i << " hi: " << zoid.zoid.cuts[0].upper
                      << " " << zoid.zoid.cuts[1].upper << " "
                      << zoid.zoid.cuts[2].upper << std::endl;
        }
        for (int i = 0; i < NUM_ZOIDS; i++) {
            std::cout << "zoid: " << i
                      << " send_to: " << lmp->send_to_neighbors[i]
                      << " recv from: " << lmp->recv_from_neighbors[i]
                      << std::endl;
        }
        std::cout << "----------------------------------------------"
                  << std::endl;
        for (int i = 0; i < NUM_ZOIDS; i++) {
            std::cout << "zoid: " << i << " send_to next dt: "
                      << lmp->send_to_neighbors_next_dt[i]
                      << " recv from next_dt: "
                      << lmp->recv_from_neighbors_next_dt[i] << std::endl;
        }

        for (int i = 0; i < NUM_ZOIDS; i++) {
            if (get_zoid_dep(i) == 3) {
                std::vector<int> recv_vec;
                for (int r : lmp->recv_from_neighbors[i]) {
                    if (get_zoid_dep(r) == 2) {
                        recv_vec.push_back(r % comm->nprocs);
                    }
                }

                std::cout << "recv from for zoid: " << i
                          << " is: " << lmp->recv_from_neighbors[i]
                          << " recv procs: " << recv_vec << std::endl;
            }
        }
    }
    */

    stencilMD->INIT_DOMAIN_BOUNDS();

    stencilMD->INIT_ALL();

    stencilMD->SETUP();

    MPI_Barrier(world);

    stencilMD->GET_LOCAL_ATOMS_ZOID();

    stencilMD->GET_GHOST_ATOMS_ZOID();

    for (int zoid_num = 0; zoid_num < NUM_ZOIDS; zoid_num++) {
        if (zoid_num % comm->nprocs == comm->me) {
            for (int t = 0; t < NUM_TIMESTEPS_IN_PARALLEL + 1; t++) {
                std::cout << "zoid: " << zoid_num << " time: " << t << " nlocal: " << lmp->atom_stencil_md[zoid_num][t]->nlocal << " num ghost: " << lmp->atom_stencil_md[zoid_num][t]->nghost << std::endl;
            }
        }
    }

    /*
    // TODO: this was the original way of doing it, for all timesteps
    // get border atoms
    for (int t = 0; t < NUM_TIMESTEPS_IN_PARALLEL + 1; t++) {
        comm->exchange_stencil_md_initial_send();
        for (int dep = 0; dep < NUM_DEPS; dep++) {
            for (int j = 0; j < lmp->queues[dep].size(); j++) {
                queue_info &zoid = lmp->queues[dep][j];
                int zoid_num = zoid.num;
                // receive only if the zoid belongs to me
                if (zoid_num % comm->nprocs == comm->me) {
                    Atom* first = lmp->atom_stencil_md[zoid_num][t];
                    lmp->comm_stencil_md[zoid_num]->borders_stencil_md_initial_receive_from_lammps(first,
                                                                                                   lmp->domain_stencil_md[zoid_num][t],
                                                                                                   zoid, t);
                }
            }
        }

        MPI_Barrier(world);
    }
    */

    // check to make sure each timestep has all of the local atoms needed
    for (int t = 0; t < NUM_TIMESTEPS_IN_PARALLEL + 1; t++) {
        int total = 0;
        for (int dep = 0; dep < NUM_DEPS; dep++) {
            for (int j = 0; j < lmp->queues[dep].size(); j++) {
                queue_info& zoid = lmp->queues[dep][j];
                if (zoid.num % comm->nprocs == comm->me) {
                    total += lmp->atom_stencil_md[zoid.num][t]->nlocal;
                }
            }
        }

        MPI_Allreduce(MPI_IN_PLACE, &total, 1, MPI_INT, MPI_SUM, world);
        if (total != atom->natoms) {
            std::cout << "timestep: " << t << " num atoms I have: " << total
                      << " num atoms: " << atom->natoms << std::endl;
        }
        assert(total == atom->natoms);
    }

    MPI_Barrier(world);

    if (comm->me == 0) {
        for (int dep = 0; dep < NUM_DEPS; dep++) {
            for (int j = 0; j < lmp->queues[dep].size(); j++) {
                queue_info& zoid = lmp->queues[dep][j];
                int zoid_num = zoid.num;
                // receive only if the zoid belongs to me
                std::cout << "zoid num: " << zoid.num << " bounds" << std::endl;
                for (int dim = 0; dim < 3; dim++) {
                    std::cout << "lo: " << zoid.zoid.cuts[dim].lower
                              << " hi: " << zoid.zoid.cuts[dim].upper
                              << std::endl;
                }
            }
        }

        for (int dep = 0; dep < NUM_DEPS; dep++) {
            for (int j = 0; j < lmp->queues_next_dt[dep].size(); j++) {
                queue_info& zoid = lmp->queues_next_dt[dep][j];
                int zoid_num = zoid.num;
                // receive only if the zoid belongs to me
                std::cout << "next dt zoid num: " << zoid.num << " bounds"
                          << std::endl;
                for (int dim = 0; dim < 3; dim++) {
                    std::cout << "lo: " << zoid.zoid.cuts[dim].lower
                              << " hi: " << zoid.zoid.cuts[dim].upper
                              << std::endl;
                }
            }
        }
    }

    for (int t = 0; t < NUM_TIMESTEPS_IN_PARALLEL + 1; t++) {
        for (int dep = 0; dep < NUM_DEPS; dep++) {
            for (int j = 0; j < lmp->queues[dep].size(); j++) {
                queue_info& zoid = lmp->queues[dep][j];
                int zoid_num = zoid.num;
                // receive only if the zoid belongs to me
                if (zoid_num % comm->nprocs == comm->me) {
                    Atom* atom_ = lmp->atom_stencil_md[zoid_num][t];

                    std::set<int> tags;
                    for (int i = 0; i < atom_->nlocal; i++) {
                        tags.insert(atom_->tag[i]);
                    }
                    // ensure no repeat tags
                    assert(tags.size() == atom_->nlocal);
                }
            }
        }
    }

    if (comm->me == 0) {
        /*
        for (int i = 0; i < NUM_ZOIDS; i++) {
            std::cout << "send to for zoid: " << i
                      << " is: " << lmp->send_to_neighbors[i] << std::endl;
            std::cout << "recv from for zoid: " << i
                      << " is: " << lmp->recv_from_neighbors[i] << std::endl;
            // std::cout << "send to next dt for zoid: " << i << " is: " << lmp->send_to_neighbors_next_dt[i] << std::endl;
            // std::cout << "recv from next dt for zoid: " << i << " is: " << lmp->recv_from_neighbors_next_dt[i] << std::endl;
        }
        */
    }

    if (comm->me == 0) {
        std::cout << YELLOW
                  << "--------------------------------------------------------------------------------------------------"
                  << RESET_COLOR << std::endl;
    }

    stencilMD->MODIFY_SETUP(vflag);
    stencilMD->BUILD_NEIGHBOR_LIST();
    stencilMD->BUILD_NEIGHBOR_LIST_NEXT_DT();

    for (int t = 0; t < NUM_TIMESTEPS_IN_PARALLEL + 1; t++) {
        for (int dep = 0; dep < NUM_DEPS; dep++) {
            for (int j = 0; j < lmp->queues[dep].size(); j++) {
                queue_info& zoid = lmp->queues[dep][j];
                int zoid_num = zoid.num;
                if (zoid_num % comm->nprocs == comm->me) {
                    /*
                    lmp->neighbor_stencil_md[zoid_num][t]
                        ->setup_bins_stencil_md(
                            lmp->atom_stencil_md[zoid_num][t],
                            lmp->domain_stencil_md[zoid_num][t],
                            lmp->comm_stencil_md[zoid_num]);
                    lmp->neighbor_stencil_md[zoid_num][t]->build_stencil_md(
                        1, lmp->atom_stencil_md[zoid_num][t],
                        lmp->domain_stencil_md[zoid_num][t],
                        lmp->comm_stencil_md[zoid_num], zoid);
                    lmp->neighbor_stencil_md[zoid_num][t]->ncalls = 0;

                    MPI_Barrier(world);
                    */

                    Force* force_ = lmp->force_stencil_md[zoid_num][t];
                    force_->setup();

                    force_clear_stencil_md(
                        lmp->atom_stencil_md[zoid_num][t], force_,
                        lmp->neighbor_stencil_md[zoid_num][t]);
                }
            }
        }
    }

    for (int dep = 0; dep < NUM_DEPS; dep++) {
        for (int j = 0; j < lmp->queues[dep].size(); j++) {
            queue_info& zoid = lmp->queues[dep][j];
            int zoid_num = zoid.num;
            if (zoid_num % comm->nprocs == comm->me) {
                setup_can_eval_center_tag_mapping_stencil_md(
                    lmp->atom_stencil_md[zoid_num], zoid);
                setup_atom_relevant_tags_stencil_md(
                    lmp->atom_stencil_md[zoid_num],
                    lmp->force_stencil_md[zoid_num], zoid);
            }
        }
    }

    for (int dep = 0; dep < NUM_DEPS; dep++) {
        for (int j = 0; j < lmp->queues_next_dt[dep].size(); j++) {
            queue_info& zoid = lmp->queues_next_dt[dep][j];
            int zoid_num = zoid.num;
            if (zoid_num % comm->nprocs == comm->me) {
                setup_can_eval_center_tag_mapping_stencil_md_next_dt(
                    lmp->atom_stencil_md[zoid_num], zoid);
                if (PURELY_LOCAL_POTENTIAL) {
                    setup_atom_relevant_tags_stencil_md_next_dt(
                            lmp->atom_stencil_md[zoid_num],
                            lmp->force_stencil_md_next_dt[zoid_num], zoid);
                } else {
                    setup_atom_relevant_tags_stencil_md_next_dt(
                            lmp->atom_stencil_md[zoid_num],
                            lmp->force_stencil_md[zoid_num], zoid);

                }
            }
        }
    }

    // separate sorting ghost atoms and grouping them
    for (int t = 0; t < NUM_TIMESTEPS_IN_PARALLEL + 1; t++) {
        for (int dep = 0; dep < NUM_DEPS; dep++) {
            for (int j = 0; j < lmp->queues[dep].size(); j++) {
                queue_info& zoid = lmp->queues[dep][j];
                int zoid_num = zoid.num;
                assert(zoid_num >= 0 && zoid_num < NUM_ZOIDS);
                // receive only if the zoid belongs to me
                if (zoid_num % comm->nprocs == comm->me) {
                    Atom* atom_ = lmp->atom_stencil_md[zoid_num][t];
                    sort_ghost_atoms_stencil_md(atom_, NULL, zoid, t);
                }
            }
        }
    }

    stencilMD->BUILD_NEIGHBOR_LIST();
    stencilMD->BUILD_NEIGHBOR_LIST_NEXT_DT();

    for (int t = 0; t < NUM_TIMESTEPS_IN_PARALLEL + 1; t++) {
        for (int dep = 0; dep < NUM_DEPS; dep++) {
            for (int j = 0; j < lmp->queues[dep].size(); j++) {
                queue_info& zoid = lmp->queues[dep][j];
                int zoid_num = zoid.num;
                if (zoid_num % comm->nprocs == comm->me) {
                    /*
                    lmp->neighbor_stencil_md[zoid_num][t]
                        ->setup_bins_stencil_md(
                            lmp->atom_stencil_md[zoid_num][t],
                            lmp->domain_stencil_md[zoid_num][t],
                            lmp->comm_stencil_md[zoid_num]);
                    lmp->neighbor_stencil_md[zoid_num][t]->build_stencil_md(
                            1, lmp->atom_stencil_md[zoid_num][t],
                            lmp->domain_stencil_md[zoid_num][t],
                            lmp->comm_stencil_md[zoid_num], zoid);
                    lmp->neighbor_stencil_md[zoid_num][t]->ncalls = 0;
                    */

                    Force* force_ = lmp->force_stencil_md[zoid_num][t];
                    force_->setup();

                    force_clear_stencil_md(
                        lmp->atom_stencil_md[zoid_num][t], force_,
                        lmp->neighbor_stencil_md[zoid_num][t]);
                }
            }
        }
    }

    for (int dep = 0; dep < NUM_DEPS; dep++) {
        for (int j = 0; j < lmp->queues[dep].size(); j++) {
            queue_info& zoid = lmp->queues[dep][j];
            int zoid_num = zoid.num;
            if (zoid_num % comm->nprocs == comm->me) {
                setup_atom_pos_mapping_stencil_md(
                    lmp->atom_stencil_md[zoid_num], zoid);
                setup_can_eval_center_mapping_stencil_md(
                    lmp->atom_stencil_md[zoid_num], zoid);
                setup_atom_relevant_idxs_stencil_md(
                    lmp->atom_stencil_md[zoid_num],
                    lmp->force_stencil_md[zoid_num], zoid);

                /*
                for (int t = 0; t < NUM_TIMESTEPS_IN_PARALLEL + 1; t++) {
                    Atom* atom_ = lmp->atom_stencil_md[zoid_num][t];
                    std::cout
                        << YELLOW << "zoid: " << zoid.num << " time: " << t
                        << " num relevant: "
                        << zoid.relevant_atom_idxs[t].size()
                        << " out of: " << atom_->nlocal
                        << " nlocal, and: " << atom_->nlocal + atom_->nghost
                        << " total. " << RESET_COLOR << std::endl;
                }
                */

                /*
                int count = 0;
                int nlocal = 0;
                int nghost = 0;
                std::vector<int> num_eval_timesteps;
                for (int t = 1; t < NUM_TIMESTEPS_IN_PARALLEL + 1; t++) {
                    Atom* atom_ = lmp->atom_stencil_md[zoid_num][t];
                    nlocal += atom_->nlocal;
                    nghost += atom_->nghost;
                    int num_eval_timestep = 0;
                    for (int k = 0; k < atom_->nlocal + atom_->nghost; k++) {
                        if (zoid.can_eval_center[t][k]) {
                            count++;
                            num_eval_timestep++;
                        }
                    }
                    num_eval_timesteps.push_back(num_eval_timestep);
                }

                std::cout << GREEN << "how many can eval through time? zoid: " << zoid.num
                    << " eval? " << count << " local: " << nlocal << " nghost: " << nghost
                    << " num eval per timestep: " << num_eval_timesteps << RESET_COLOR << std::endl;
                */
            }
        }
    }

    // setup same things for next_dt
    for (int dep = 0; dep < NUM_DEPS; dep++) {
        for (int j = 0; j < lmp->queues_next_dt[dep].size(); j++) {
            queue_info& zoid = lmp->queues_next_dt[dep][j];
            int zoid_num = zoid.num;
            if (zoid_num % comm->nprocs == comm->me) {
                setup_atom_pos_mapping_stencil_md_next_dt(
                    lmp->atom_stencil_md[zoid_num], zoid);
                setup_can_eval_center_mapping_stencil_md_next_dt(
                    lmp->atom_stencil_md[zoid_num], zoid);
                if (PURELY_LOCAL_POTENTIAL) {
                    setup_atom_relevant_idxs_stencil_md_next_dt(
                            lmp->atom_stencil_md[zoid_num],
                            lmp->force_stencil_md_next_dt[zoid_num], zoid);
                } else {
                    setup_atom_relevant_idxs_stencil_md_next_dt(
                            lmp->atom_stencil_md[zoid_num],
                            lmp->force_stencil_md[zoid_num], zoid);
                }
            }
        }
    }

    // group ghost, sort ghost atoms
    for (int t = 0; t < NUM_TIMESTEPS_IN_PARALLEL + 1; t++) {
        for (int dep = 0; dep < NUM_DEPS; dep++) {
            for (int j = 0; j < lmp->queues[dep].size(); j++) {
                queue_info& zoid = lmp->queues[dep][j];
                int zoid_num = zoid.num;
                assert(zoid_num >= 0 && zoid_num < NUM_ZOIDS);
                // receive only if the zoid belongs to me
                if (zoid_num % comm->nprocs == comm->me) {
                    Atom* atom_ = lmp->atom_stencil_md[zoid_num][t];

                    int num_recv_from =
                        lmp->recv_from_neighbors[zoid_num].size();
                    assert(num_recv_from >= 0 && num_recv_from <= NUM_ZOIDS);
                    zoid.recv_ghost_idxs[t] = new int*[num_recv_from];
                    zoid.recv_ghost_sizes[t] = new int*[num_recv_from];
                    zoid.recv_ghost_num_segments[t] = new int[num_recv_from];
                    group_ghost_atoms_stencil_md(atom_, NULL, zoid, t);

                    zoid.recv_list_local[t] = new int*[num_recv_from];
                    zoid.recv_list_local_size[t] = new int[num_recv_from];

                    zoid.recv_list_local_force_only[t] =
                        new int*[num_recv_from];
                    zoid.recv_list_local_force_pos[t] = new int*[num_recv_from];
                    zoid.recv_list_local_num_force_only[t] =
                        new int[num_recv_from];
                    zoid.recv_list_local_num_force_pos[t] =
                        new int[num_recv_from];

                    for (int idx = 0; idx < atom_->nlocal + atom_->nghost;
                         idx++) {
                        if (atom_->tag_to_idx.count(atom_->tag[idx])) {
                            std::cout << "repeat tag. zoid: " << zoid_num
                                      << std::endl;
                        }
                        assert(!atom_->tag_to_idx.count(atom_->tag[idx]));
                        atom_->tag_to_idx[atom_->tag[idx]] = idx;
                    }
                }
            }
        }
    }

    // group ghost atoms for next dt
    for (int t = 0; t < NUM_TIMESTEPS_IN_PARALLEL + 1; t++) {
        for (int dep = 0; dep < NUM_DEPS; dep++) {
            for (int j = 0; j < lmp->queues_next_dt[dep].size(); j++) {
                queue_info& zoid = lmp->queues_next_dt[dep][j];
                int zoid_num = zoid.num;
                // receive only if the zoid belongs to me
                if (zoid_num % comm->nprocs == comm->me) {
                    Atom* atom_ =
                        lmp->atom_stencil_md[zoid_num]
                                            [NUM_TIMESTEPS_IN_PARALLEL - t];

                    int num_recv_from =
                        lmp->recv_from_neighbors_next_dt[zoid_num].size();

                    zoid.recv_ghost_idxs[t] = new int*[num_recv_from];
                    zoid.recv_ghost_sizes[t] = new int*[num_recv_from];
                    zoid.recv_ghost_num_segments[t] = new int[num_recv_from];
                    group_ghost_atoms_stencil_md_next_dt(atom_, NULL, zoid, t);

                    zoid.recv_list_local[t] = new int*[num_recv_from];
                    zoid.recv_list_local_size[t] = new int[num_recv_from];

                    zoid.recv_list_local_force_only[t] =
                        new int*[num_recv_from];
                    zoid.recv_list_local_force_pos[t] = new int*[num_recv_from];
                    zoid.recv_list_local_num_force_only[t] =
                        new int[num_recv_from];
                    zoid.recv_list_local_num_force_pos[t] =
                        new int[num_recv_from];
                }
            }
        }
    }

    MPI_Barrier(world);

    int stencil_md_nghost = 0;
    for (int t = 0; t < NUM_TIMESTEPS_IN_PARALLEL + 1; t++) {
        for (int zoid_num = 0; zoid_num < NUM_ZOIDS; zoid_num++) {
            if (zoid_num % comm->nprocs == comm->me) {
                Atom* atom_ = lmp->atom_stencil_md[zoid_num][t];
                // std::cout << CYAN << "zoid num: " << zoid_num << " time: " << t << " nlocal: " << atom_->nlocal << " nghost: " << atom_->nghost << RESET_COLOR << std::endl;
                stencil_md_nghost += atom_->nghost;
            }
        }
    }

    // Now need to communicate with ther zoids to construct send_list and second_send_list
    std::vector<MPI_Request> r[NUM_ZOIDS];
    for (int dep = 0; dep < NUM_DEPS; dep++) {
        for (int j = 0; j < lmp->queues[dep].size(); j++) {
            queue_info& zoid = lmp->queues[dep][j];
            int zoid_num = zoid.num;
            // receive only if the zoid belongs to me
            if (zoid_num % comm->nprocs == comm->me) {
                r[zoid_num] = std::move(std::vector<MPI_Request>(2 * lmp->recv_from_neighbors[zoid_num].size(), MPI_REQUEST_NULL));
                lmp->comm_stencil_md[zoid_num]
                    ->construct_second_send_list_stencil_md_send(
                        lmp->atom_stencil_md[zoid_num], zoid, r[zoid_num]);
            }
        }
    }

    for (int dep = 0; dep < NUM_DEPS; dep++) {
        for (int j = 0; j < lmp->queues[dep].size(); j++) {
            queue_info& zoid = lmp->queues[dep][j];
            int zoid_num = zoid.num;
            // receive only if the zoid belongs to me
            if (zoid_num % comm->nprocs == comm->me) {
                int num_send_neighbors =
                    lmp->send_to_neighbors[zoid_num].size();
                for (int t = 0; t < NUM_TIMESTEPS_IN_PARALLEL + 1; t++) {
                    zoid.send_segment_sizes[t] = new int*[num_send_neighbors];
                    zoid.send_segment_types[t] = new int*[num_send_neighbors];
                    zoid.send_segment_idxs[t] = new int*[num_send_neighbors];
                    zoid.send_num_segments[t] = new int[num_send_neighbors];
                    zoid.send_local_list[t] = new int*[num_send_neighbors];
                }
                lmp->comm_stencil_md[zoid_num]
                    ->construct_second_send_list_stencil_md(
                        lmp->atom_stencil_md[zoid_num], zoid);
            }
        }
    }

    for (int dep = 0; dep < NUM_DEPS; dep++) {
        for (int j = 0; j < lmp->queues[dep].size(); j++) {
            queue_info& zoid = lmp->queues[dep][j];
            int zoid_num = zoid.num;
            // receive only if the zoid belongs to me
            if (zoid_num % comm->nprocs == comm->me) {
                MPI_Waitall(r[zoid_num].size(), r[zoid_num].data(), MPI_STATUSES_IGNORE);
            }
        }
    }

    MPI_Barrier(world);

    std::vector<MPI_Request> r2_arr[NUM_ZOIDS];
    // construct local list now, ghost to local?
    for (int dep = 0; dep < NUM_DEPS; dep++) {
        for (int j = 0; j < lmp->queues[dep].size(); j++) {
            queue_info& zoid = lmp->queues[dep][j];
            int zoid_num = zoid.num;
            if (zoid_num % comm->nprocs == comm->me) {
                int num_send_neighbors =
                    lmp->send_to_neighbors[zoid_num].size();
                for (int t = 0; t < NUM_TIMESTEPS_IN_PARALLEL + 1; t++) {
                    zoid.send_force_idxs[t] = new int*[num_send_neighbors];
                    zoid.send_force_sizes[t] = new int*[num_send_neighbors];
                    zoid.send_force_num_segments[t] = new int[num_send_neighbors];
                    zoid.send_force_total_num_elems[t] = new int[num_send_neighbors];

                    zoid.send_pos_idxs[t] = new int*[num_send_neighbors];
                    zoid.send_pos_sizes[t] = new int*[num_send_neighbors];
                    zoid.send_pos_num_segments[t] = new int[num_send_neighbors];
                    zoid.send_pos_total_num_elems[t] = new int[num_send_neighbors];
                }
                r2_arr[zoid_num] = std::move(std::vector<MPI_Request>(2 * lmp->send_to_neighbors[zoid_num].size(), MPI_REQUEST_NULL));
                lmp->comm_stencil_md[zoid_num]
                    ->construct_send_list_stencil_md_send(
                        lmp->atom_stencil_md[zoid_num], zoid, r2_arr[zoid_num]);
            }
        }
    }

    for (int dep = 0; dep < NUM_DEPS; dep++) {
        for (int j = 0; j < lmp->queues[dep].size(); j++) {
            queue_info& zoid = lmp->queues[dep][j];
            int zoid_num = zoid.num;
            if (zoid_num % comm->nprocs == comm->me) {
                lmp->comm_stencil_md[zoid_num]->construct_send_list_stencil_md(
                    lmp->atom_stencil_md[zoid_num], zoid);
            }
        }
    }

    MPI_Barrier(world);
    for (int dep = 0; dep < NUM_DEPS; dep++) {
        for (int j = 0; j < lmp->queues[dep].size(); j++) {
            queue_info& zoid = lmp->queues[dep][j];
            int zoid_num = zoid.num;
            if (zoid_num % comm->nprocs == comm->me) {
                MPI_Waitall(r2_arr[zoid_num].size(), r2_arr[zoid_num].data(), MPI_STATUSES_IGNORE);
            }
        }
    }

    // next dt construct lists
    std::vector<MPI_Request> r_next_dt[NUM_ZOIDS];
    for (int dep = 0; dep < NUM_DEPS; dep++) {
        for (int j = 0; j < lmp->queues_next_dt[dep].size(); j++) {
            queue_info& zoid = lmp->queues_next_dt[dep][j];
            int zoid_num = zoid.num;
            if (zoid_num % comm->nprocs == comm->me) {
                r_next_dt[zoid_num] = std::move(std::vector<MPI_Request>(2 * lmp->recv_from_neighbors_next_dt[zoid_num].size(), MPI_REQUEST_NULL));
                lmp->comm_stencil_md[zoid_num]
                    ->construct_second_send_list_stencil_md_next_dt_send(
                        lmp->atom_stencil_md[zoid_num], zoid, r_next_dt[zoid_num]);
            }
        }
    }

    for (int dep = 0; dep < NUM_DEPS; dep++) {
        for (int j = 0; j < lmp->queues_next_dt[dep].size(); j++) {
            queue_info& zoid = lmp->queues_next_dt[dep][j];
            int zoid_num = zoid.num;
            // receive only if the zoid belongs to me
            if (zoid_num % comm->nprocs == comm->me) {
                int num_send_neighbors =
                    lmp->send_to_neighbors_next_dt[zoid_num].size();
                for (int t = 0; t < NUM_TIMESTEPS_IN_PARALLEL + 1; t++) {
                    zoid.send_segment_sizes[t] = new int*[num_send_neighbors];
                    zoid.send_segment_types[t] = new int*[num_send_neighbors];
                    zoid.send_segment_idxs[t] = new int*[num_send_neighbors];
                    zoid.send_num_segments[t] = new int[num_send_neighbors];
                    zoid.send_local_list[t] = new int*[num_send_neighbors];
                }
                lmp->comm_stencil_md[zoid_num]
                    ->construct_second_send_list_stencil_md_next_dt(
                        lmp->atom_stencil_md[zoid_num], zoid);
            }
        }
    }

    for (int dep = 0; dep < NUM_DEPS; dep++) {
        for (int j = 0; j < lmp->queues_next_dt[dep].size(); j++) {
            queue_info& zoid = lmp->queues_next_dt[dep][j];
            int zoid_num = zoid.num;
            // receive only if the zoid belongs to me
            if (zoid_num % comm->nprocs == comm->me) {
                MPI_Waitall(r_next_dt[zoid_num].size(), r_next_dt[zoid_num].data(), MPI_STATUSES_IGNORE);
            }
        }
    }

    MPI_Barrier(world);

    std::vector<MPI_Request> r2_next_dt[NUM_ZOIDS];
    for (int dep = 0; dep < NUM_DEPS; dep++) {
        for (int j = 0; j < lmp->queues_next_dt[dep].size(); j++) {
            queue_info& zoid = lmp->queues_next_dt[dep][j];
            int zoid_num = zoid.num;
            if (zoid_num % comm->nprocs == comm->me) {
                int num_send_neighbors =
                    lmp->send_to_neighbors_next_dt[zoid_num].size();
                for (int t = 0; t < NUM_TIMESTEPS_IN_PARALLEL + 1; t++) {
                    zoid.send_force_idxs[t] = new int*[num_send_neighbors];
                    zoid.send_force_sizes[t] = new int*[num_send_neighbors];
                    zoid.send_force_num_segments[t] = new int[num_send_neighbors];
                    zoid.send_force_total_num_elems[t] = new int[num_send_neighbors];

                    zoid.send_pos_idxs[t] = new int*[num_send_neighbors];
                    zoid.send_pos_sizes[t] = new int*[num_send_neighbors];
                    zoid.send_pos_num_segments[t] = new int[num_send_neighbors];
                    zoid.send_pos_total_num_elems[t] = new int[num_send_neighbors];
                }
                r2_next_dt[zoid_num] = std::move(std::vector<MPI_Request>(2 * lmp->send_to_neighbors_next_dt[zoid_num].size(), MPI_REQUEST_NULL));
                lmp->comm_stencil_md[zoid_num]
                    ->construct_send_list_stencil_md_next_dt_send(
                        lmp->atom_stencil_md[zoid_num], zoid, r2_next_dt[zoid_num]);
            }
        }
    }

    for (int dep = 0; dep < NUM_DEPS; dep++) {
        for (int j = 0; j < lmp->queues_next_dt[dep].size(); j++) {
            queue_info& zoid = lmp->queues_next_dt[dep][j];
            int zoid_num = zoid.num;
            // receive only if the zoid belongs to me
            if (zoid_num % comm->nprocs == comm->me) {
                lmp->comm_stencil_md[zoid_num]
                    ->construct_send_list_stencil_md_next_dt(
                        lmp->atom_stencil_md[zoid_num], zoid);
            }
        }
    }

    for (int dep = 0; dep < NUM_DEPS; dep++) {
        for (int j = 0; j < lmp->queues_next_dt[dep].size(); j++) {
            queue_info& zoid = lmp->queues_next_dt[dep][j];
            int zoid_num = zoid.num;
            // receive only if the zoid belongs to me
            if (zoid_num % comm->nprocs == comm->me) {
                MPI_Waitall(r2_next_dt[zoid_num].size(), r2_next_dt[zoid_num].data(), MPI_STATUSES_IGNORE);
            }
        }
    }

    MPI_Barrier(world);

    int mul_factor;
    if (DEBUG_SEND_RECV_DATA) {
        // 3 elements for pos/vel/force + 1 for tag
        mul_factor = 3 + 1;
    } else {
        mul_factor = 3;
    }

    int total_recv = 0;
    for (int dep = 0; dep < NUM_DEPS; dep++) {
        for (int j = 0; j < lmp->queues[dep].size(); j++) {
            queue_info& zoid = lmp->queues[dep][j];
            int zoid_num = zoid.num;
            // receive only if the zoid belongs to me
            if (zoid_num % comm->nprocs == comm->me) {
                std::vector<int>& send_to = lmp->send_to_neighbors[zoid_num];
                std::vector<int>& recv_from =
                    lmp->recv_from_neighbors[zoid_num];

                for (int t = 0; t < NUM_TIMESTEPS_IN_PARALLEL + 1; t++) {
                    zoid.num_elems_send[t] = new int[send_to.size()];
                    zoid.num_elems_recv[t] = new int[recv_from.size()];
                }

                for (int i = 0; i < send_to.size(); i++) {
                    int send_zoid_num = send_to[i];

                    for (int t = 0; t < NUM_TIMESTEPS_IN_PARALLEL + 1; t++) {
                        int num_elems_timestep = 0;

                        // 3 elems for data, 1 for the debug tag
                        int num_force_sizes = 0;
                        for (int k = 0;
                             k < zoid.send_force_num_segments[t][i]; k++) {
                            num_elems_timestep +=
                                zoid.send_force_sizes[t][i][k] * mul_factor;
                            num_force_sizes +=
                                zoid.send_force_sizes[t][i][k];
                        }

                        int num_pos_sizes = 0;
                        for (int k = 0;
                             k < zoid.send_pos_num_segments[t][i]; k++) {
                            num_elems_timestep +=
                                zoid.send_pos_sizes[t][i][k] * 2 * mul_factor;
                            num_pos_sizes += zoid.send_pos_sizes[t][i][k];
                        }

                        int num_local_sizes = 0;
                        int num_ghost_sizes = 0;

                        int num_ghost_segments = 0;

                        for (int k = 0; k < zoid.send_num_segments[t][i];
                             k++) {
                            int segment_type =
                                zoid.send_segment_types[t][i][k];
                            if (segment_type == LOCAL_SEGMENT_TYPE) {
                                num_local_sizes +=
                                    zoid.send_segment_sizes[t][i][k];
                            } else {
                                assert(segment_type == GHOST_SEGMENT_TYPE);
                                num_ghost_sizes +=
                                    zoid.send_segment_sizes[t][i][k];
                                num_ghost_segments++;
                            }
                        }

                        num_elems_timestep +=
                            (num_local_sizes + num_ghost_sizes) * mul_factor;

                        zoid.num_elems_send[t][i] = num_elems_timestep;
                        if (num_ghost_segments >= 20) {
                            // std::cout << GREEN << "zoid num: " << zoid_num << " send to: " << send_zoid_num << " time: " << t << " debug num ghost segments: " << num_ghost_segments << std::endl;
                        }
                    }
                }

                int zoid_recv_ghost_pos = 0;
                int zoid_recv_local_pos = 0;
                int zoid_recv_local_force = 0;

                for (int i = 0; i < recv_from.size(); i++) {
                    int recv_zoid_num = recv_from[i];

                    for (int t = 0; t < NUM_TIMESTEPS_IN_PARALLEL + 1; t++) {
                        int num_elems_timestep = 0;
                        // 3 elems for data, 1 for the debug tag
                        num_elems_timestep +=
                            zoid.recv_list_local_num_force_only[t][i] * mul_factor;
                        num_elems_timestep +=
                            zoid.recv_list_local_num_force_pos[t][i] * 2 * mul_factor;

                        int total_ghost_idxs = 0;
                        for (int k = 0;
                             k < zoid.recv_ghost_num_segments[t][i]; k++) {
                            total_ghost_idxs +=
                                zoid.recv_ghost_sizes[t][i][k];
                        }

                        num_elems_timestep += total_ghost_idxs * mul_factor;

                        zoid_recv_ghost_pos += total_ghost_idxs * mul_factor;
                        zoid_recv_local_force +=
                            zoid.recv_list_local_num_force_only[t][i] * mul_factor;
                        zoid_recv_local_pos +=
                            zoid.recv_list_local_num_force_pos[t][i] * 2 * mul_factor;

                        zoid.num_elems_recv[t][i] = num_elems_timestep;
                        if (recv_zoid_num % comm->nprocs != comm->me) {
                            total_recv += zoid.num_elems_recv[t][i];
                        }
                    }
                }

                std::cout << "zoid: " << zoid.num
                          << " recv ghost pos: " << zoid_recv_ghost_pos
                          << " recv local pos: " << zoid_recv_local_pos
                          << " recv local force: " << zoid_recv_local_force
                          << std::endl;
            }
        }
    }

    // compute number of elements recv, used just to get a sense of the communication volume induced by stencil md
    for (int dep = 0; dep < NUM_DEPS; dep++) {
        for (int j = 0; j < lmp->queues_next_dt[dep].size(); j++) {
            queue_info& zoid = lmp->queues_next_dt[dep][j];
            int zoid_num = zoid.num;
            // receive only if the zoid belongs to me
            if (zoid_num % comm->nprocs == comm->me) {
                std::vector<int>& send_to =
                    lmp->send_to_neighbors_next_dt[zoid_num];
                std::vector<int>& recv_from =
                    lmp->recv_from_neighbors_next_dt[zoid_num];

                for (int t = 0; t < NUM_TIMESTEPS_IN_PARALLEL + 1; t++) {
                    zoid.num_elems_send[t] = new int[send_to.size()];
                    zoid.num_elems_recv[t] = new int[recv_from.size()];
                }

                for (int i = 0; i < send_to.size(); i++) {
                    int send_zoid_num = send_to[i];

                    for (int t = 0; t < NUM_TIMESTEPS_IN_PARALLEL + 1; t++) {
                        int num_elems_timestep = 0;
                        // 3 elems for data, 1 for the debug tag
                        for (int k = 0;
                             k < zoid.send_force_num_segments[t][i]; k++) {
                            num_elems_timestep +=
                                zoid.send_force_sizes[t][i][k] * mul_factor;
                        }

                        for (int k = 0;
                             k < zoid.send_pos_num_segments[t][i]; k++) {
                            num_elems_timestep +=
                                zoid.send_pos_sizes[t][i][k] * 2 * mul_factor;
                        }

                        int num_local_sizes = 0;
                        int num_ghost_sizes = 0;

                        int num_ghost_segments = 0;

                        for (int k = 0; k < zoid.send_num_segments[t][i];
                             k++) {
                            int segment_type =
                                zoid.send_segment_types[t][i][k];
                            if (segment_type == LOCAL_SEGMENT_TYPE) {
                                num_local_sizes +=
                                    zoid.send_segment_sizes[t][i][k];
                            } else {
                                assert(segment_type == GHOST_SEGMENT_TYPE);
                                num_ghost_sizes +=
                                    zoid.send_segment_sizes[t][i][k];
                                num_ghost_segments++;
                            }
                        }

                        num_elems_timestep +=
                            (num_local_sizes + num_ghost_sizes) * mul_factor;

                        zoid.num_elems_send[t][i] = num_elems_timestep;
                        /*
                        std::cout << MAGENTA << "NEXT DT zoid: " << zoid.num << " send to: " << send_zoid_num << " time: " << t << " num send force: "
                                  << zoid.send_force_num_segments[t][i] << " num send pos: " << zoid.send_pos_num_segments[t][i]
                                  << " send ghost num segments: " << num_ghost_segments << RESET_COLOR << std::endl;
                        */
                    }
                }

                for (int i = 0; i < recv_from.size(); i++) {
                    int recv_zoid_num = recv_from[i];

                    for (int t = 0; t < NUM_TIMESTEPS_IN_PARALLEL + 1; t++) {
                        int num_elems_timestep = 0;
                        // 3 elems for data, 1 for the debug tag
                        num_elems_timestep +=
                            zoid.recv_list_local_num_force_only[t][i] * mul_factor;
                        num_elems_timestep +=
                            zoid.recv_list_local_num_force_pos[t][i] * 2 * mul_factor;

                        int total_ghost_idxs = 0;
                        for (int k = 0;
                             k < zoid.recv_ghost_num_segments[t][i]; k++) {
                            total_ghost_idxs +=
                                zoid.recv_ghost_sizes[t][i][k];
                        }

                        num_elems_timestep += total_ghost_idxs * mul_factor;

                        zoid.num_elems_recv[t][i] = num_elems_timestep;

                        /*
                        std::cout << CYAN << "NEXT DT zoid: " << zoid.num << " recv from: " << recv_zoid_num << " time: " << t << " num recv: "
                                  << zoid.recv_ghost_num_segments[t][i] << RESET_COLOR << std::endl;
                        */
                    }
                }
            }
        }
    }

    // begin stuff for curr dt
    std::map<int, std::map<int, std::vector<int>>>
        zoid_to_idx_to_send_zoids[NUM_TIMESTEPS_IN_PARALLEL + 1];

    for (int t = 0; t < NUM_TIMESTEPS_IN_PARALLEL + 1; t++) {
        for (int dep = 0; dep < NUM_DEPS; dep++) {
            for (int j = 0; j < lmp->queues[dep].size(); j++) {
                queue_info& zoid = lmp->queues[dep][j];
                int zoid_num = zoid.num;
                if (zoid.num % comm->nprocs == comm->me) {
                    // map segment_idx + size to zoids
                    std::map<std::pair<int, int>, std::vector<int>>
                        segment_mapping_to_zoids;

                    Atom* atom_ = lmp->atom_stencil_md[zoid_num][t];
                    auto& send_to = lmp->send_to_neighbors[zoid_num];

                    for (int i = 0; i < send_to.size(); i++) {
                        // ghost to local
                        int num_send_pos_segments =
                            zoid.send_pos_num_segments[t][i];
                        for (int k = 0; k < num_send_pos_segments; k++) {
                            int size = zoid.send_pos_sizes[t][i][k];
                            int start_idx = zoid.send_pos_idxs[t][i][k];
                            segment_mapping_to_zoids[{start_idx, size}]
                                .push_back(send_to[i]);
                            for (int h = 0; h < size; h++) {
                                int idx = start_idx + h;
                                zoid_to_idx_to_send_zoids[t][zoid_num][idx]
                                    .push_back(send_to[i]);
                            }
                        }

                        int* local_list = zoid.send_local_list[t][i];
                        int local_list_idx = 0;
                        for (int k = 0; k < zoid.send_num_segments[t][i]; k++) {
                            int segment_type = zoid.send_segment_types[t][i][k];
                            if (segment_type == GHOST_SEGMENT_TYPE) {
                                int ghost_size =
                                    zoid.send_segment_sizes[t][i][k];
                                int ghost_idx = zoid.send_segment_idxs[t][i][k];
                                segment_mapping_to_zoids[{ghost_idx,
                                                          ghost_size}]
                                    .push_back(send_to[i]);
                                for (int h = 0; h < ghost_size; h++) {
                                    int idx = ghost_idx + h;
                                    zoid_to_idx_to_send_zoids[t][zoid_num][idx]
                                        .push_back(send_to[i]);
                                }
                            } else {
                                int local_size =
                                    zoid.send_segment_sizes[t][i][k];
                                for (int h = 0; h < local_size; h++) {
                                    int local_idx =
                                        local_list[local_list_idx++];
                                    zoid_to_idx_to_send_zoids
                                        [t][zoid_num][local_idx]
                                            .push_back(send_to[i]);
                                }
                            }
                        }
                    }

                    bool debug = false;

                    if (debug) {
                        std::map<int, std::set<int>> ghost_idxs_sent_to_procs;
                        std::map<int, std::vector<int>>
                            local_idxs_sent_to_procs;

                        std::map<int, std::set<int>>
                            ghost_to_local_idxs_sent_to_procs;

                        bool print = (zoid_num == 4 && t == 1);

                        for (int i = 0; i < send_to.size(); i++) {
                            bool print2 =
                                print && send_to[i] % comm->nprocs == 4;
                            int neighbor_proc = send_to[i] % comm->nprocs;
                            int* local_list = zoid.send_local_list[t][i];
                            int num_segments = zoid.send_num_segments[t][i];
                            int local_list_idx = 0;

                            for (int k = 0; k < num_segments; k++) {
                                int segment_type =
                                    zoid.send_segment_types[t][i][k];
                                if (segment_type == GHOST_SEGMENT_TYPE) {
                                    int ghost_size =
                                        zoid.send_segment_sizes[t][i][k];
                                    int ghost_idx =
                                        zoid.send_segment_idxs[t][i][k];
                                    for (int h = 0; h < ghost_size; h++) {
                                        int idx = ghost_idx + h;
                                        ghost_idxs_sent_to_procs[neighbor_proc]
                                            .insert(idx);
                                        /*
                                        if (print && print2) {
                                            std::cout
                                                << YELLOW
                                                << "zoid: " << zoid_num
                                                << " send to: " << send_to[i]
                                                << " ghost segment number: "
                                                << k << " idx: " << idx
                                                << " send to: "
                                                << zoid_to_idx_to_send_zoids
                                                       [t][zoid_num][idx]
                                                << " pos: " << atom_->x[idx][0]
                                                << " " << atom_->x[idx][1]
                                                << " " << atom_->x[idx][2]
                                                << " tag: " << atom_->tag[idx]
                                                << RESET_COLOR << std::endl;
                                        }
                                        */
                                    }
                                } else {
                                    int local_size =
                                        zoid.send_segment_sizes[t][i][k];
                                    for (int h = 0; h < local_size; h++) {
                                        int local_idx =
                                            local_list[local_list_idx++];

                                        auto& vec = local_idxs_sent_to_procs
                                            [neighbor_proc];

                                        /*
                                        if (print && print2) {
                                            std::cout
                                                << CYAN << "zoid: " << zoid_num
                                                << " send to: " << send_to[i]
                                                << " real local segment "
                                                   "number: "
                                                << k << " idx: " << local_idx
                                                << " send to: "
                                                << zoid_to_idx_to_send_zoids
                                                       [t][zoid_num][local_idx]
                                                << " pos: "
                                                << atom_->x[local_idx][0] << " "
                                                << atom_->x[local_idx][1] << " "
                                                << atom_->x[local_idx][2]
                                                << " tag: "
                                                << atom_->tag[local_idx]
                                                << RESET_COLOR << std::endl;
                                        }
                                        */

                                        if (std::find(vec.begin(), vec.end(),
                                                      local_idx) == vec.end()) {
                                            local_idxs_sent_to_procs
                                                [neighbor_proc]
                                                    .push_back(local_idx);
                                        }
                                    }
                                }
                            }

                            int num_send_pos_segments =
                                zoid.send_pos_num_segments[t][i];
                            for (int k = 0; k < num_send_pos_segments; k++) {
                                int size = zoid.send_pos_sizes[t][i][k];
                                int start_idx = zoid.send_pos_idxs[t][i][k];
                                for (int h = 0; h < size; h++) {
                                    int idx = start_idx + h;
                                    ghost_idxs_sent_to_procs[neighbor_proc]
                                        .insert(idx);
                                    ghost_to_local_idxs_sent_to_procs
                                        [neighbor_proc]
                                            .insert(idx);
                                    /*
                                    if (print && print2) {
                                        std::cout
                                            << YELLOW
                                            << "idx to send zids not size 1 "
                                               "wtfel lzoid: "
                                            << zoid_num
                                            << " send to: " << send_to[i]
                                            << " send ghost to local segment "
                                               "number: "
                                            << k << " idx: " << idx
                                            << " send to: "
                                            << zoid_to_idx_to_send_zoids
                                                   [t][zoid_num][idx]
                                            << " pos: " << atom_->x[idx][0]
                                            << " " << atom_->x[idx][1] << " "
                                            << atom_->x[idx][2]
                                            << " tag: " << atom_->tag[idx]
                                            << RESET_COLOR << std::endl;
                                    }
                                    */
                                }
                            }
                        }

                        for (int k = 0; k < comm->nprocs; k++) {
                            std::vector<int> vec_idxs;

                            if (ghost_idxs_sent_to_procs.find(k) !=
                                ghost_idxs_sent_to_procs.end()) {
                                for (int idx : local_idxs_sent_to_procs[k]) {
                                    vec_idxs.push_back(idx);
                                }
                                for (int idx : ghost_idxs_sent_to_procs[k]) {
                                    vec_idxs.push_back(idx);
                                }

                                std::vector<int> ghost_vec_idxs;
                                for (int idx : vec_idxs) {
                                    if (idx >= atom_->nlocal) {
                                        ghost_vec_idxs.push_back(idx);
                                    }
                                }

                                std::vector<int> tmp_idxs;
                                std::vector<int> tmp_lengths;

                                int tmp_num_segments =
                                    get_segments(ghost_vec_idxs, tmp_idxs,
                                                 tmp_lengths, false);
                                if (k != comm->me) {
                                    /*
                                    if (tmp_num_segments >= 15) {
                                        std::cout
                                            << "curr dt zoid: " << zoid_num
                                            << " send to proc: " << k
                                            << " time: " << t
                                            << " tmp num segments for all the "
                                               "ghosts sent in one batch: "
                                            << tmp_num_segments
                                            << " idxs: " << tmp_idxs
                                            << " lengths: " << tmp_lengths
                                            << std::endl;
                                    }
                                    */
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    MPI_Barrier(world);

    std::map<std::pair<int, int>, int>
        send_buf_num_tags[NUM_TIMESTEPS_IN_PARALLEL + 1];
    std::map<std::pair<int, int>, std::vector<int>>
        send_buf_tags[NUM_TIMESTEPS_IN_PARALLEL + 1];
    std::vector<MPI_Request> send_buf_requests;

    std::map<std::pair<int, int>, std::vector<int>>
        force_offset_idxs[NUM_TIMESTEPS_IN_PARALLEL + 1];
    std::map<std::pair<int, int>, int>
        pos_offsets[NUM_TIMESTEPS_IN_PARALLEL + 1];

    std::map<std::pair<int, int>, std::vector<int>>
        vel_offset_idxs[NUM_TIMESTEPS_IN_PARALLEL + 1];

    // send ghost idx to send buf idx per process to the zoids

    for (int t = 0; t < NUM_TIMESTEPS_IN_PARALLEL + 1; t++) {
        lmp->num_recv_force_from_zoid[t] =
            new int[lmp->recv_from_neighbors_procs.size()];
        memset(lmp->num_recv_force_from_zoid[t], -1,
               lmp->recv_from_neighbors_procs.size());
        lmp->num_recv_pos_from_zoid[t] =
            new int[lmp->recv_from_neighbors_procs.size()];
        memset(lmp->num_recv_pos_from_zoid[t], -1,
               lmp->recv_from_neighbors_procs.size());
        lmp->num_recv_vel_from_zoid[t] =
            new int[lmp->recv_from_neighbors_procs.size()];
        memset(lmp->num_recv_vel_from_zoid[t], -1,
               lmp->recv_from_neighbors_procs.size());
        lmp->num_recv_elems_from_zoid[t] = new int[lmp->recv_from_neighbors_procs.size()];
    }

    for (int t = 0; t < NUM_TIMESTEPS_IN_PARALLEL + 1; t++) {
        for (int dep = 0; dep < NUM_DEPS; dep++) {
            for (int j = 0; j < lmp->queues[dep].size(); j++) {
                queue_info& zoid = lmp->queues[dep][j];
                int zoid_num = zoid.num;
                std::vector<int>& send_to = lmp->send_to_neighbors[zoid_num];

                if (zoid_num % comm->nprocs == comm->me) {
                    Atom* atom_ = lmp->atom_stencil_md[zoid_num][t];

                    for (int proc = 0; proc < comm->nprocs; proc++) {
                        std::vector<int> send_process_local_idxs;
                        std::vector<int> send_process_local_sizes;

                        for (int i = 0; i < send_to.size(); i++) {
                            int send_zoid_num = send_to[i];
                            if (send_zoid_num % comm->nprocs == proc) {
                                int* local_list = zoid.send_local_list[t][i];
                                int num_segments = zoid.send_num_segments[t][i];
                                int local_list_idx = 0;
                                for (int k = 0; k < num_segments; k++) {
                                    int segment_type =
                                        zoid.send_segment_types[t][i][k];
                                    if (segment_type == LOCAL_SEGMENT_TYPE) {
                                        int local_size =
                                            zoid.send_segment_sizes[t][i][k];

                                        int actual_size = 0;
                                        for (int h = 0; h < local_size; h++) {
                                            int local_idx =
                                                local_list[local_list_idx++];
                                            if (zoid_to_idx_to_send_zoids
                                                    [t][zoid_num][local_idx]
                                                        .size() == 1) {
                                                if (std::find(
                                                        send_process_local_idxs
                                                            .begin(),
                                                        send_process_local_idxs
                                                            .end(),
                                                        local_idx) ==
                                                    send_process_local_idxs
                                                        .end()) {
                                                    send_process_local_idxs
                                                        .push_back(local_idx);
                                                    actual_size++;
                                                } else {
                                                    assert(false);
                                                }
                                            }
                                        }

                                        // send_process_local_sizes.push_back(local_size);
                                        if (actual_size > 0) {
                                            send_process_local_sizes.push_back(
                                                actual_size);
                                        }
                                    }
                                }
                            }
                        }

                        for (int i = 0; i < send_to.size(); i++) {
                            int send_zoid_num = send_to[i];
                            if (send_zoid_num % comm->nprocs == proc) {
                                int* local_list = zoid.send_local_list[t][i];
                                int num_segments = zoid.send_num_segments[t][i];
                                int local_list_idx = 0;
                                for (int k = 0; k < num_segments; k++) {
                                    int segment_type =
                                        zoid.send_segment_types[t][i][k];
                                    if (segment_type == LOCAL_SEGMENT_TYPE) {
                                        int local_size =
                                            zoid.send_segment_sizes[t][i][k];

                                        int actual_size = 0;
                                        for (int h = 0; h < local_size; h++) {
                                            int local_idx =
                                                local_list[local_list_idx++];

                                            if (std::find(
                                                    send_process_local_idxs
                                                        .begin(),
                                                    send_process_local_idxs
                                                        .end(),
                                                    local_idx) ==
                                                send_process_local_idxs.end()) {
                                                send_process_local_idxs
                                                    .push_back(local_idx);
                                                actual_size++;
                                            }
                                        }

                                        // send_process_local_sizes.push_back(local_size);
                                        if (actual_size > 0) {
                                            send_process_local_sizes.push_back(
                                                actual_size);
                                        }
                                    }
                                }
                            }
                        }

                        int total_size = 0;
                        for (int size : send_process_local_sizes) {
                            total_size += size;
                        }

                        assert(total_size == send_process_local_idxs.size());

                        std::set<int> send_process_ghost_idxs_set;

                        for (int i = 0; i < send_to.size(); i++) {
                            int send_zoid_num = send_to[i];
                            if (send_zoid_num % comm->nprocs == proc) {
                                int* local_list = zoid.send_local_list[t][i];
                                int num_segments = zoid.send_num_segments[t][i];
                                int local_list_idx = 0;
                                for (int k = 0; k < num_segments; k++) {
                                    int segment_type =
                                        zoid.send_segment_types[t][i][k];
                                    if (segment_type == GHOST_SEGMENT_TYPE) {
                                        int ghost_size =
                                            zoid.send_segment_sizes[t][i][k];
                                        int ghost_idx =
                                            zoid.send_segment_idxs[t][i][k];
                                        for (int h = 0; h < ghost_size; h++) {
                                            int idx = ghost_idx + h;
                                            assert(idx < atom_->nlocal +
                                                             atom_->nghost);
                                            send_process_ghost_idxs_set.insert(
                                                idx);
                                        }
                                    }
                                }

                                int num_send_pos_segments =
                                    zoid.send_pos_num_segments[t][i];
                                for (int k = 0; k < num_send_pos_segments;
                                     k++) {
                                    int size = zoid.send_pos_sizes[t][i][k];
                                    int start_idx = zoid.send_pos_idxs[t][i][k];
                                    for (int h = 0; h < size; h++) {
                                        int idx = start_idx + h;
                                        assert(idx <
                                               atom_->nlocal + atom_->nghost);
                                        send_process_ghost_idxs_set.insert(idx);
                                    }
                                }
                            }
                        }

                        std::vector<int> send_process_ghost_idxs_vec;
                        for (int idx : send_process_ghost_idxs_set) {
                            send_process_ghost_idxs_vec.push_back(idx);
                        }

                        std::vector<int> send_process_ghost_segment_idxs;
                        std::vector<int> send_process_ghost_segment_sizes;
                        // int num_ghost_segments = get_segments(send_process_ghost_idxs, send_process_ghost_segment_idxs, send_process_ghost_segment_sizes);
                        int num_ghost_segments =
                            get_segments(send_process_ghost_idxs_vec,
                                         send_process_ghost_segment_idxs,
                                         send_process_ghost_segment_sizes);

                        int num_local_segments =
                            send_process_local_sizes.size();

                        // construct send process information
                        zoid.send_process_num_segments[t][proc] =
                            num_local_segments + num_ghost_segments;
                        zoid.send_process_segment_types[t][proc] =
                            new int[num_local_segments + num_ghost_segments];
                        zoid.send_process_segment_idxs[t][proc] =
                            new int[num_local_segments + num_ghost_segments];
                        zoid.send_process_segment_sizes[t][proc] =
                            new int[num_local_segments + num_ghost_segments];

                        zoid.send_process_local_list[t][proc] =
                            new int[send_process_local_idxs.size()];
                        for (int k = 0; k < send_process_local_idxs.size();
                             k++) {
                            zoid.send_process_local_list[t][proc][k] =
                                send_process_local_idxs[k];
                        }

                        int num_elems_send = 0;
                        for (int k = 0; k < num_local_segments; k++) {
                            zoid.send_process_segment_types[t][proc][k] =
                                SEND_DATA_PROCESS_LOCAL;
                            zoid.send_process_segment_idxs[t][proc][k] = -2;
                            zoid.send_process_segment_sizes[t][proc][k] =
                                send_process_local_sizes[k];

                            num_elems_send += send_process_local_sizes[k];
                        }

                        for (int k = 0; k < num_ghost_segments; k++) {
                            int segment_num = k + num_local_segments;
                            zoid.send_process_segment_types[t][proc]
                                                           [segment_num] =
                                SEND_DATA_PROCESS_GHOST;
                            zoid.send_process_segment_idxs[t][proc]
                                                          [segment_num] =
                                send_process_ghost_segment_idxs[k];
                            zoid.send_process_segment_sizes[t][proc]
                                                           [segment_num] =
                                send_process_ghost_segment_sizes[k];

                            if (send_process_ghost_segment_idxs[k] +
                                    send_process_ghost_segment_sizes[k] >
                                atom_->nlocal + atom_->nghost) {
                                std::cout << RED << "zoid: " << zoid_num
                                          << " send to proc: " << proc
                                          << " segment num: " << segment_num
                                          << " time: " << t << " segment idx: "
                                          << send_process_ghost_segment_idxs[k]
                                          << " size: "
                                          << send_process_ghost_segment_sizes[k]
                                          << " num local: " << atom_->nlocal
                                          << " total: "
                                          << atom_->nlocal + atom_->nghost
                                          << RESET_COLOR << std::endl;
                                for (int h = 0; h < num_ghost_segments; h++) {
                                    std::cout
                                        << "ghost segment: " << h << " idx: "
                                        << send_process_ghost_segment_idxs[h]
                                        << " size: "
                                        << send_process_ghost_segment_sizes[h]
                                        << std::endl;
                                }
                            }
                            assert(send_process_ghost_segment_idxs[k] +
                                       send_process_ghost_segment_sizes[k] <=
                                   atom_->nlocal + atom_->nghost);

                            num_elems_send +=
                                send_process_ghost_segment_sizes[k];
                        }

                        zoid.num_elems_send_process[t][proc] = num_elems_send;

                        for (int idx : send_process_local_idxs) {
                            send_buf_tags[t][{zoid_num, proc}].push_back(
                                atom_->tag[idx]);
                        }

                        for (int idx : send_process_ghost_idxs_vec) {
                            send_buf_tags[t][{zoid_num, proc}].push_back(
                                atom_->tag[idx]);
                        }

                        send_buf_num_tags[t][{zoid_num, proc}] =
                            send_process_local_idxs.size() +
                            send_process_ghost_idxs_vec.size();

                        // 0 offset for the first force sent
                        force_offset_idxs[t][{zoid_num, proc}].push_back(0);
                        int num_send_force = 0;
                        for (int i = 0; i < send_to.size(); i++) {
                            if (send_to[i] % comm->nprocs == proc) {
                                int num_force_segments =
                                    zoid.send_force_num_segments[t][i];
                                int send_force_size = 0;
                                for (int k = 0; k < num_force_segments; k++) {
                                    int segment_size =
                                        zoid.send_force_sizes[t][i][k];
                                    send_force_size += segment_size;
                                }

                                int prev_size = force_offset_idxs
                                    [t][{zoid_num, proc}]
                                    [force_offset_idxs[t][{zoid_num, proc}]
                                         .size() -
                                     1];
                                force_offset_idxs[t][{zoid_num, proc}]
                                    .push_back(send_force_size + prev_size);

                                num_send_force += send_force_size;
                            }
                        }

                        int num_send_vel = 0;
                        vel_offset_idxs[t][{zoid_num, proc}].push_back(0);
                        for (int i = 0; i < send_to.size(); i++) {
                            if (send_to[i] % comm->nprocs == proc) {
                                int num_vel_segments =
                                    zoid.send_pos_num_segments[t][i];
                                int send_vel_size = 0;
                                for (int k = 0; k < num_vel_segments; k++) {
                                    int segment_size =
                                        zoid.send_pos_sizes[t][i][k];
                                    send_vel_size += segment_size;
                                }

                                int prev_size = vel_offset_idxs[t][{
                                    zoid_num,
                                    proc}][vel_offset_idxs[t][{zoid_num, proc}]
                                               .size() -
                                           1];
                                vel_offset_idxs[t][{zoid_num, proc}].push_back(
                                    send_vel_size + prev_size);

                                num_send_vel += send_vel_size;
                            }
                        }

                        int pos_offset = num_send_force;

                        pos_offsets[t][{zoid_num, proc}] = pos_offset;

                        int vel_offset_vec_idx = 0;
                        int force_offset_vec_idx = 0;
                        for (int i = 0; i < send_to.size(); i++) {
                            int send_zoid_num = send_to[i];
                            if (send_zoid_num % comm->nprocs == proc) {
                                int mpi_tag = (send_to[i] << 16) | zoid_num;
                                MPI_Request r1;
                                MPI_Request r2;
                                MPI_Request r3;
                                MPI_Request r4;
                                MPI_Request r5;
                                MPI_Request r6;
                                MPI_Request r7;
                                MPI_Request r8;

                                MPI_Isend(
                                    &send_buf_num_tags[t][{zoid_num, proc}], 1,
                                    MPI_INT, proc, mpi_tag, world, &r1);

                                MPI_Isend(
                                    send_buf_tags[t][{zoid_num, proc}].data(),
                                    send_buf_num_tags[t][{zoid_num, proc}],
                                    MPI_INT, proc, mpi_tag, world, &r2);

                                MPI_Isend(
                                    &force_offset_idxs[t][{zoid_num, proc}]
                                                      [force_offset_vec_idx++],
                                    1, MPI_INT, proc, mpi_tag, world, &r3);

                                MPI_Isend(&pos_offsets[t][{zoid_num, proc}], 1,
                                          MPI_INT, proc, mpi_tag, world, &r4);

                                MPI_Isend(
                                    &force_offset_idxs
                                        [t][{zoid_num, proc}]
                                        [force_offset_idxs[t][{zoid_num, proc}]
                                             .size() -
                                         1],
                                    1, MPI_INT, proc, mpi_tag, world, &r5);

                                MPI_Isend(&zoid.num_elems_send_process[t][proc],
                                          1, MPI_INT, proc, mpi_tag, world,
                                          &r6);

                                MPI_Isend(
                                    &vel_offset_idxs[t][{zoid_num, proc}]
                                                    [vel_offset_vec_idx++],
                                    1, MPI_INT, proc, mpi_tag, world, &r7);

                                MPI_Isend(
                                    &vel_offset_idxs
                                        [t][{zoid_num, proc}]
                                        [vel_offset_idxs[t][{zoid_num, proc}]
                                             .size() -
                                         1],
                                    1, MPI_INT, proc, mpi_tag, world, &r8);

                                send_buf_requests.push_back(r1);
                                send_buf_requests.push_back(r2);
                                send_buf_requests.push_back(r3);
                                send_buf_requests.push_back(r4);
                                send_buf_requests.push_back(r5);
                                send_buf_requests.push_back(r6);
                                send_buf_requests.push_back(r7);
                                send_buf_requests.push_back(r8);
                            }
                        }
                    }
                }
            }
        }

        for (int dep = 0; dep < NUM_DEPS; dep++) {
            for (int j = 0; j < lmp->queues[dep].size(); j++) {
                queue_info& zoid = lmp->queues[dep][j];
                int zoid_num = zoid.num;
                std::vector<int>& recv_from =
                    lmp->recv_from_neighbors[zoid_num];

                if (zoid_num % comm->nprocs == comm->me) {
                    Atom* atom_ = lmp->atom_stencil_md[zoid_num][t];
                    zoid.recv_process_segment_types[t] =
                        new int*[recv_from.size()];
                    zoid.recv_process_segment_idxs[t] =
                        new int*[recv_from.size()];
                    zoid.recv_process_segment_sizes[t] =
                        new int*[recv_from.size()];
                    zoid.recv_process_num_segments[t] =
                        new int[recv_from.size()];

                    // send idxs belonging to that zoid?
                    zoid.recv_process_force_offset[t] =
                        new int[recv_from.size()];
                    zoid.recv_process_pos_offset[t] = new int[recv_from.size()];
                    zoid.recv_process_vel_offset[t] = new int[recv_from.size()];

                    for (int i = 0; i < recv_from.size(); i++) {
                        int recv_zoid_num = recv_from[i];
                        int recv_proc = recv_zoid_num % comm->nprocs;

                        int mpi_tag = (zoid_num << 16 | recv_zoid_num);
                        int num_tags = 0;
                        MPI_Recv(&num_tags, 1, MPI_INT, recv_proc, mpi_tag,
                                 world, MPI_STATUS_IGNORE);

                        int* recv_tags = new int[num_tags];
                        MPI_Recv(recv_tags, num_tags, MPI_INT, recv_proc,
                                 mpi_tag, world, MPI_STATUS_IGNORE);

                        int force_offset_idx = 0;
                        MPI_Recv(&force_offset_idx, 1, MPI_INT, recv_proc,
                                 mpi_tag, world, MPI_STATUS_IGNORE);

                        int num_force_recv = 0;
                        MPI_Recv(&num_force_recv, 1, MPI_INT, recv_proc,
                                 mpi_tag, world, MPI_STATUS_IGNORE);

                        // offset within buffer of forces, which one belongs to this zoid
                        int force_offset = 0;
                        MPI_Recv(&force_offset, 1, MPI_INT, recv_proc, mpi_tag,
                                 world, MPI_STATUS_IGNORE);

                        int num_pos_recv = 0;
                        MPI_Recv(&num_pos_recv, 1, MPI_INT, recv_proc, mpi_tag,
                                 world, MPI_STATUS_IGNORE);

                        int vel_offset = 0;
                        MPI_Recv(&vel_offset, 1, MPI_INT, recv_proc, mpi_tag,
                                 world, MPI_STATUS_IGNORE);

                        int num_vel_recv = 0;
                        MPI_Recv(&num_vel_recv, 1, MPI_INT, recv_proc, mpi_tag,
                                 world, MPI_STATUS_IGNORE);

                        for (int k = 0;
                             k < lmp->recv_from_neighbors_procs.size(); k++) {
                            if (lmp->recv_from_neighbors_procs[k] ==
                                recv_from[i]) {
                                lmp->num_recv_force_from_zoid[t][k] =
                                    num_force_recv;
                                lmp->num_recv_pos_from_zoid[t][k] =
                                    num_pos_recv;
                                lmp->num_recv_vel_from_zoid[t][k] =
                                    num_vel_recv;
                                if (DEBUG_SEND_RECV_DATA) {
                                    lmp->num_recv_elems_from_zoid[t][k] = num_force_recv * (3 + 1) + num_pos_recv * (3 + 1) + num_vel_recv * (3 + 1);
                                } else {
                                    lmp->num_recv_elems_from_zoid[t][k] = num_force_recv * (3) + num_pos_recv * (3) + num_vel_recv * (3);
                                }
                            }
                        }

                        zoid.recv_process_force_offset[t][i] = force_offset_idx;
                        zoid.recv_process_vel_offset[t][i] = vel_offset;
                        zoid.recv_process_pos_offset[t][i] = num_force_recv;

                        std::map<int, int> tag_to_idx_in_buf;
                        for (int k = 0; k < num_tags; k++) {
                            tag_to_idx_in_buf[recv_tags[k]] = k;
                        }

                        bool print = false;

                        // this is sorted??
                        std::vector<int> local_buf_idxs;
                        for (int k = 0;
                             k < zoid.recv_list_local_num_force_pos[t][i];
                             k++) {
                            int idx = zoid.recv_list_local_force_pos[t][i][k];
                            int buf_idx = tag_to_idx_in_buf[atom_->tag[idx]];
                            local_buf_idxs.push_back(buf_idx);
                        }

                        std::vector<int> local_segment_idxs_buf;
                        std::vector<int> local_segment_sizes_buf;

                        int num_local_segments_buf =
                            get_segments(local_buf_idxs, local_segment_idxs_buf,
                                         local_segment_sizes_buf);

                        std::vector<int> ghost_buf_idxs;

                        std::set<int> ghost_buf_idxs_set;
                        for (int k = 0; k < zoid.recv_ghost_num_segments[t][i];
                             k++) {
                            int segment_size = zoid.recv_ghost_sizes[t][i][k];
                            int segment_idx = zoid.recv_ghost_idxs[t][i][k];
                            for (int h = 0; h < segment_size; h++) {
                                int idx = segment_idx + h;
                                int buf_idx =
                                    tag_to_idx_in_buf[atom_->tag[idx]];
                                ghost_buf_idxs.push_back(buf_idx);

                                ghost_buf_idxs_set.insert(buf_idx);
                                if (print) {
                                    std::cout
                                        << CYAN
                                        << "debug recv ghost segment: " << k
                                        << " idx: " << idx
                                        << " buf idx: " << buf_idx
                                        << " tag: " << atom_->tag[idx]
                                        << RESET_COLOR << std::endl;
                                }
                            }
                        }

                        assert(ghost_buf_idxs.size() ==
                               ghost_buf_idxs_set.size());

                        std::vector<int> ghost_segment_idxs_buf;
                        std::vector<int> ghost_segment_sizes_buf;

                        int num_ghost_segments_buf =
                            get_segments(ghost_buf_idxs, ghost_segment_idxs_buf,
                                         ghost_segment_sizes_buf);

                        zoid.recv_process_num_segments[t][i] =
                            num_local_segments_buf + num_ghost_segments_buf;
                        zoid.recv_process_segment_types[t][i] =
                            new int[num_local_segments_buf +
                                    num_ghost_segments_buf];
                        zoid.recv_process_segment_idxs[t][i] =
                            new int[num_local_segments_buf +
                                    num_ghost_segments_buf];
                        zoid.recv_process_segment_sizes[t][i] =
                            new int[num_local_segments_buf +
                                    num_ghost_segments_buf];

                        for (int k = 0; k < num_local_segments_buf; k++) {
                            zoid.recv_process_segment_types[t][i][k] =
                                RECV_DATA_PROCESS_LOCAL;
                            zoid.recv_process_segment_idxs[t][i][k] =
                                local_segment_idxs_buf[k];
                            zoid.recv_process_segment_sizes[t][i][k] =
                                local_segment_sizes_buf[k];
                        }

                        for (int k = 0; k < num_ghost_segments_buf; k++) {
                            int segment_num = k + num_local_segments_buf;
                            zoid.recv_process_segment_types[t][i][segment_num] =
                                RECV_DATA_PROCESS_GHOST;
                            zoid.recv_process_segment_idxs[t][i][segment_num] =
                                ghost_segment_idxs_buf[k];
                            zoid.recv_process_segment_sizes[t][i][segment_num] =
                                ghost_segment_sizes_buf[k];
                        }

                        /*
                        if (true) {
                            std::cout << YELLOW << "curr dt zoid: " << zoid.num << " time: " << t << " num local segments: "
                                      << num_local_segments_buf
                                      << " num ghost segments: "
                                      << num_ghost_segments_buf << " num recv ghost: " << zoid.recv_ghost_num_segments[t][i] << " recv from: " << recv_from[i] << RESET_COLOR << std::endl;
                        }
                        */

                        delete[] recv_tags;
                    }
                }
            }
        }

        MPI_Barrier(world);
    }

    MPI_Waitall(send_buf_requests.size(), send_buf_requests.data(),
                MPI_STATUSES_IGNORE);

    // begin next dt stuff,desperately needs cleanup
    std::map<int, std::map<int, std::vector<int>>>
        zoid_to_idx_to_send_zoids_next_dt[NUM_TIMESTEPS_IN_PARALLEL + 1];

    for (int t = 0; t < NUM_TIMESTEPS_IN_PARALLEL + 1; t++) {
        for (int dep = 0; dep < NUM_DEPS; dep++) {
            for (int j = 0; j < lmp->queues_next_dt[dep].size(); j++) {
                queue_info& zoid = lmp->queues_next_dt[dep][j];
                int zoid_num = zoid.num;
                if (zoid.num % comm->nprocs == comm->me) {
                    // map segment_idx + size to zoids
                    std::map<std::pair<int, int>, std::vector<int>>
                        segment_mapping_to_zoids;

                    Atom* atom_ =
                        lmp->atom_stencil_md[zoid_num]
                                            [NUM_TIMESTEPS_IN_PARALLEL - t];
                    auto& send_to = lmp->send_to_neighbors_next_dt[zoid_num];

                    for (int i = 0; i < send_to.size(); i++) {
                        // ghost to local
                        int num_send_pos_segments =
                            zoid.send_pos_num_segments[t][i];
                        for (int k = 0; k < num_send_pos_segments; k++) {
                            int size = zoid.send_pos_sizes[t][i][k];
                            int start_idx = zoid.send_pos_idxs[t][i][k];
                            segment_mapping_to_zoids[{start_idx, size}]
                                .push_back(send_to[i]);
                            for (int h = 0; h < size; h++) {
                                int idx = start_idx + h;
                                zoid_to_idx_to_send_zoids_next_dt
                                    [t][zoid_num][idx]
                                        .push_back(send_to[i]);
                            }
                        }

                        int* local_list = zoid.send_local_list[t][i];
                        int local_list_idx = 0;
                        for (int k = 0; k < zoid.send_num_segments[t][i]; k++) {
                            int segment_type = zoid.send_segment_types[t][i][k];
                            if (segment_type == GHOST_SEGMENT_TYPE) {
                                int ghost_size =
                                    zoid.send_segment_sizes[t][i][k];
                                int ghost_idx = zoid.send_segment_idxs[t][i][k];
                                segment_mapping_to_zoids[{ghost_idx,
                                                          ghost_size}]
                                    .push_back(send_to[i]);
                                for (int h = 0; h < ghost_size; h++) {
                                    int idx = ghost_idx + h;
                                    zoid_to_idx_to_send_zoids_next_dt
                                        [t][zoid_num][idx]
                                            .push_back(send_to[i]);
                                }
                            } else {
                                int local_size =
                                    zoid.send_segment_sizes[t][i][k];
                                for (int h = 0; h < local_size; h++) {
                                    int local_idx =
                                        local_list[local_list_idx++];
                                    zoid_to_idx_to_send_zoids_next_dt
                                        [t][zoid_num][local_idx]
                                            .push_back(send_to[i]);
                                }
                            }
                        }
                    }

                    bool debug = true;

                    if (debug) {
                        std::map<int, std::set<int>> ghost_idxs_sent_to_procs;
                        std::map<int, std::vector<int>>
                            local_idxs_sent_to_procs;

                        bool print = (zoid_num == 32 && t == 0);

                        for (int i = 0; i < send_to.size(); i++) {
                            bool print2 =
                                print && send_to[i] % comm->nprocs == 6;
                            int neighbor_proc = send_to[i] % comm->nprocs;
                            int* local_list = zoid.send_local_list[t][i];
                            int num_segments = zoid.send_num_segments[t][i];
                            int local_list_idx = 0;

                            for (int k = 0; k < num_segments; k++) {
                                int segment_type =
                                    zoid.send_segment_types[t][i][k];
                                if (segment_type == GHOST_SEGMENT_TYPE) {
                                    int ghost_size =
                                        zoid.send_segment_sizes[t][i][k];
                                    int ghost_idx =
                                        zoid.send_segment_idxs[t][i][k];
                                    for (int h = 0; h < ghost_size; h++) {
                                        int idx = ghost_idx + h;
                                        ghost_idxs_sent_to_procs[neighbor_proc]
                                            .insert(idx);
                                        /*
                                        if (print && print2) {
                                            std::cout
                                                << YELLOW
                                                << "zoid: " << zoid_num
                                                << " send to: " << send_to[i]
                                                << " ghost segment number: "
                                                << k << " idx: " << idx
                                                << " send to: "
                                                << zoid_to_idx_to_send_zoids
                                                       [t][zoid_num][idx]
                                                << " pos: " << atom_->x[idx][0]
                                                << " " << atom_->x[idx][1]
                                                << " " << atom_->x[idx][2]
                                                << " tag: " << atom_->tag[idx]
                                                << RESET_COLOR << std::endl;
                                        }
                                        */
                                    }
                                } else {
                                    int local_size =
                                        zoid.send_segment_sizes[t][i][k];
                                    for (int h = 0; h < local_size; h++) {
                                        int local_idx =
                                            local_list[local_list_idx++];

                                        auto& vec = local_idxs_sent_to_procs
                                            [neighbor_proc];

                                        /*
                                        if (print && print2) {
                                            std::cout
                                                << CYAN << "zoid: " << zoid_num
                                                << " send to: " << send_to[i]
                                                << " real local segment "
                                                   "number: "
                                                << k << " idx: " << local_idx
                                                << " send to: "
                                                << zoid_to_idx_to_send_zoids
                                                       [t][zoid_num][local_idx]
                                                << " pos: "
                                                << atom_->x[local_idx][0] << " "
                                                << atom_->x[local_idx][1] << " "
                                                << atom_->x[local_idx][2]
                                                << " tag: "
                                                << atom_->tag[local_idx]
                                                << RESET_COLOR << std::endl;
                                        }
                                        */

                                        if (std::find(vec.begin(), vec.end(),
                                                      local_idx) == vec.end()) {
                                            local_idxs_sent_to_procs
                                                [neighbor_proc]
                                                    .push_back(local_idx);
                                        }
                                    }
                                }
                            }

                            int num_send_pos_segments =
                                zoid.send_pos_num_segments[t][i];
                            for (int k = 0; k < num_send_pos_segments; k++) {
                                int size = zoid.send_pos_sizes[t][i][k];
                                int start_idx = zoid.send_pos_idxs[t][i][k];
                                for (int h = 0; h < size; h++) {
                                    int idx = start_idx + h;
                                    ghost_idxs_sent_to_procs[neighbor_proc]
                                        .insert(idx);
                                    /*
                                    if (print && print2) {
                                        std::cout
                                            << YELLOW
                                            << "idx to send zids not size 1 "
                                               "wtfel lzoid: "
                                            << zoid_num
                                            << " send to: " << send_to[i]
                                            << " send ghost to local segment "
                                               "number: "
                                            << k << " idx: " << idx
                                            << " send to: "
                                            << zoid_to_idx_to_send_zoids
                                                   [t][zoid_num][idx]
                                            << " pos: " << atom_->x[idx][0]
                                            << " " << atom_->x[idx][1] << " "
                                            << atom_->x[idx][2]
                                            << " tag: " << atom_->tag[idx]
                                            << RESET_COLOR << std::endl;
                                    }
                                    */
                                }
                            }
                        }

                        for (int k = 0; k < comm->nprocs; k++) {
                            std::vector<int> vec_idxs;

                            if (ghost_idxs_sent_to_procs.find(k) !=
                                ghost_idxs_sent_to_procs.end()) {
                                for (int idx : local_idxs_sent_to_procs[k]) {
                                    vec_idxs.push_back(idx);
                                }
                                for (int idx : ghost_idxs_sent_to_procs[k]) {
                                    vec_idxs.push_back(idx);
                                }

                                std::vector<int> ghost_vec_idxs;
                                for (int idx : vec_idxs) {
                                    if (idx >= atom_->nlocal) {
                                        ghost_vec_idxs.push_back(idx);
                                    }
                                }

                                std::vector<int> tmp_idxs;
                                std::vector<int> tmp_lengths;

                                int tmp_num_segments =
                                    get_segments(ghost_vec_idxs, tmp_idxs,
                                                 tmp_lengths, false);
                                if (k != comm->me) {
                                    /*
                                    if (tmp_num_segments >= 15) {
                                        std::cout
                                            << "zoid: " << zoid_num
                                            << " send to proc: " << k
                                            << " nlocal? " << atom_->nlocal
                                            << " time: " << t
                                            << " tmp num segments for all the "
                                               "ghosts sent in one batch: "
                                            << tmp_num_segments
                                            << " idxs: " << tmp_idxs
                                            << " lengths: " << tmp_lengths
                                            << std::endl;
                                    }
                                    */
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    MPI_Barrier(world);

    // start next dt sending

    std::map<std::pair<int, int>, int>
        send_buf_num_tags_next_dt[NUM_TIMESTEPS_IN_PARALLEL + 1];
    std::map<std::pair<int, int>, std::vector<int>>
        send_buf_tags_next_dt[NUM_TIMESTEPS_IN_PARALLEL + 1];
    std::vector<MPI_Request> send_buf_requests_next_dt;

    std::map<std::pair<int, int>, std::vector<int>>
        force_offset_idxs_next_dt[NUM_TIMESTEPS_IN_PARALLEL + 1];
    std::map<std::pair<int, int>, int>
        pos_offsets_next_dt[NUM_TIMESTEPS_IN_PARALLEL + 1];

    std::map<std::pair<int, int>, std::vector<int>>
        vel_offset_idxs_next_dt[NUM_TIMESTEPS_IN_PARALLEL + 1];

    // send ghost idx to send buf idx per process to the zoids

    for (int t = 0; t < NUM_TIMESTEPS_IN_PARALLEL + 1; t++) {
        lmp->num_recv_force_from_zoid_next_dt[t] =
            new int[lmp->recv_from_neighbors_procs_next_dt.size()];
        memset(lmp->num_recv_force_from_zoid_next_dt[t], -1,
               lmp->recv_from_neighbors_procs_next_dt.size());
        lmp->num_recv_pos_from_zoid_next_dt[t] =
            new int[lmp->recv_from_neighbors_procs_next_dt.size()];
        memset(lmp->num_recv_pos_from_zoid_next_dt[t], -1,
               lmp->recv_from_neighbors_procs_next_dt.size());
        lmp->num_recv_vel_from_zoid_next_dt[t] =
            new int[lmp->recv_from_neighbors_procs_next_dt.size()];
        memset(lmp->num_recv_vel_from_zoid_next_dt[t], -1,
               lmp->recv_from_neighbors_procs_next_dt.size());
        lmp->num_recv_elems_from_zoid_next_dt[t] = new int[lmp->recv_from_neighbors_procs_next_dt.size()];
    }

    for (int t = 0; t < NUM_TIMESTEPS_IN_PARALLEL + 1; t++) {
        for (int dep = 0; dep < NUM_DEPS; dep++) {
            for (int j = 0; j < lmp->queues_next_dt[dep].size(); j++) {
                queue_info& zoid = lmp->queues_next_dt[dep][j];
                int zoid_num = zoid.num;
                std::vector<int>& send_to =
                    lmp->send_to_neighbors_next_dt[zoid_num];

                if (zoid_num % comm->nprocs == comm->me) {
                    Atom* atom_ =
                        lmp->atom_stencil_md[zoid_num]
                                            [NUM_TIMESTEPS_IN_PARALLEL - t];

                    for (int proc = 0; proc < comm->nprocs; proc++) {
                        std::vector<int> send_process_local_idxs;
                        std::vector<int> send_process_local_sizes;

                        for (int i = 0; i < send_to.size(); i++) {
                            int send_zoid_num = send_to[i];
                            if (send_zoid_num % comm->nprocs == proc) {
                                int* local_list = zoid.send_local_list[t][i];
                                int num_segments = zoid.send_num_segments[t][i];
                                int local_list_idx = 0;
                                for (int k = 0; k < num_segments; k++) {
                                    int segment_type =
                                        zoid.send_segment_types[t][i][k];
                                    if (segment_type == LOCAL_SEGMENT_TYPE) {
                                        int local_size =
                                            zoid.send_segment_sizes[t][i][k];

                                        int actual_size = 0;
                                        for (int h = 0; h < local_size; h++) {
                                            int local_idx =
                                                local_list[local_list_idx++];
                                            if (zoid_to_idx_to_send_zoids_next_dt
                                                    [t][zoid_num][local_idx]
                                                        .size() == 1) {
                                                if (std::find(
                                                        send_process_local_idxs
                                                            .begin(),
                                                        send_process_local_idxs
                                                            .end(),
                                                        local_idx) ==
                                                    send_process_local_idxs
                                                        .end()) {
                                                    send_process_local_idxs
                                                        .push_back(local_idx);
                                                    actual_size++;
                                                } else {
                                                    assert(false);
                                                }
                                            }
                                        }

                                        // send_process_local_sizes.push_back(local_size);
                                        if (actual_size > 0) {
                                            send_process_local_sizes.push_back(
                                                actual_size);
                                        }
                                    }
                                }
                            }
                        }

                        for (int i = 0; i < send_to.size(); i++) {
                            int send_zoid_num = send_to[i];
                            if (send_zoid_num % comm->nprocs == proc) {
                                int* local_list = zoid.send_local_list[t][i];
                                int num_segments = zoid.send_num_segments[t][i];
                                int local_list_idx = 0;
                                for (int k = 0; k < num_segments; k++) {
                                    int segment_type =
                                        zoid.send_segment_types[t][i][k];
                                    if (segment_type == LOCAL_SEGMENT_TYPE) {
                                        int local_size =
                                            zoid.send_segment_sizes[t][i][k];

                                        int actual_size = 0;
                                        for (int h = 0; h < local_size; h++) {
                                            int local_idx =
                                                local_list[local_list_idx++];

                                            if (std::find(
                                                    send_process_local_idxs
                                                        .begin(),
                                                    send_process_local_idxs
                                                        .end(),
                                                    local_idx) ==
                                                send_process_local_idxs.end()) {
                                                send_process_local_idxs
                                                    .push_back(local_idx);
                                                actual_size++;
                                            }
                                        }

                                        // send_process_local_sizes.push_back(local_size);
                                        if (actual_size > 0) {
                                            send_process_local_sizes.push_back(
                                                actual_size);
                                        }
                                    }
                                }
                            }
                        }

                        int total_size = 0;
                        for (int size : send_process_local_sizes) {
                            total_size += size;
                        }

                        assert(total_size == send_process_local_idxs.size());

                        std::set<int> send_process_ghost_idxs_set;

                        for (int i = 0; i < send_to.size(); i++) {
                            int send_zoid_num = send_to[i];
                            if (send_zoid_num % comm->nprocs == proc) {
                                int* local_list = zoid.send_local_list[t][i];
                                int num_segments = zoid.send_num_segments[t][i];
                                int local_list_idx = 0;
                                for (int k = 0; k < num_segments; k++) {
                                    int segment_type =
                                        zoid.send_segment_types[t][i][k];
                                    if (segment_type == GHOST_SEGMENT_TYPE) {
                                        int ghost_size =
                                            zoid.send_segment_sizes[t][i][k];
                                        int ghost_idx =
                                            zoid.send_segment_idxs[t][i][k];
                                        for (int h = 0; h < ghost_size; h++) {
                                            int idx = ghost_idx + h;
                                            assert(idx < atom_->nlocal +
                                                             atom_->nghost);
                                            send_process_ghost_idxs_set.insert(
                                                idx);
                                        }
                                    }
                                }

                                int num_send_pos_segments =
                                    zoid.send_pos_num_segments[t][i];
                                for (int k = 0; k < num_send_pos_segments;
                                     k++) {
                                    int size = zoid.send_pos_sizes[t][i][k];
                                    int start_idx = zoid.send_pos_idxs[t][i][k];
                                    for (int h = 0; h < size; h++) {
                                        int idx = start_idx + h;
                                        assert(idx <
                                               atom_->nlocal + atom_->nghost);
                                        send_process_ghost_idxs_set.insert(idx);
                                    }
                                }
                            }
                        }

                        std::vector<int> send_process_ghost_idxs_vec;
                        for (int idx : send_process_ghost_idxs_set) {
                            send_process_ghost_idxs_vec.push_back(idx);
                        }

                        std::vector<int> send_process_ghost_segment_idxs;
                        std::vector<int> send_process_ghost_segment_sizes;
                        // int num_ghost_segments = get_segments(send_process_ghost_idxs, send_process_ghost_segment_idxs, send_process_ghost_segment_sizes);
                        int num_ghost_segments =
                            get_segments(send_process_ghost_idxs_vec,
                                         send_process_ghost_segment_idxs,
                                         send_process_ghost_segment_sizes);

                        int num_local_segments =
                            send_process_local_sizes.size();

                        // construct send process information
                        zoid.send_process_num_segments[t][proc] =
                            num_local_segments + num_ghost_segments;
                        zoid.send_process_segment_types[t][proc] =
                            new int[num_local_segments + num_ghost_segments];
                        zoid.send_process_segment_idxs[t][proc] =
                            new int[num_local_segments + num_ghost_segments];
                        zoid.send_process_segment_sizes[t][proc] =
                            new int[num_local_segments + num_ghost_segments];

                        zoid.send_process_local_list[t][proc] =
                            new int[send_process_local_idxs.size()];
                        for (int k = 0; k < send_process_local_idxs.size();
                             k++) {
                            zoid.send_process_local_list[t][proc][k] =
                                send_process_local_idxs[k];
                        }

                        int num_elems_send = 0;
                        ;
                        for (int k = 0; k < num_local_segments; k++) {
                            zoid.send_process_segment_types[t][proc][k] =
                                SEND_DATA_PROCESS_LOCAL;
                            zoid.send_process_segment_idxs[t][proc][k] = -2;
                            zoid.send_process_segment_sizes[t][proc][k] =
                                send_process_local_sizes[k];

                            num_elems_send += send_process_local_sizes[k];
                        }

                        for (int k = 0; k < num_ghost_segments; k++) {
                            int segment_num = k + num_local_segments;
                            zoid.send_process_segment_types[t][proc]
                                                           [segment_num] =
                                SEND_DATA_PROCESS_GHOST;
                            zoid.send_process_segment_idxs[t][proc]
                                                          [segment_num] =
                                send_process_ghost_segment_idxs[k];
                            zoid.send_process_segment_sizes[t][proc]
                                                           [segment_num] =
                                send_process_ghost_segment_sizes[k];

                            if (send_process_ghost_segment_idxs[k] +
                                    send_process_ghost_segment_sizes[k] >
                                atom_->nlocal + atom_->nghost) {
                                std::cout << RED << "NEXT DT zoid: " << zoid_num
                                          << " send to proc: " << proc
                                          << " segment num: " << segment_num
                                          << " time: " << t << " segment idx: "
                                          << send_process_ghost_segment_idxs[k]
                                          << " size: "
                                          << send_process_ghost_segment_sizes[k]
                                          << " num local: " << atom_->nlocal
                                          << " total: "
                                          << atom_->nlocal + atom_->nghost
                                          << RESET_COLOR << std::endl;
                                for (int h = 0; h < num_ghost_segments; h++) {
                                    std::cout
                                        << "ghost segment: " << h << " idx: "
                                        << send_process_ghost_segment_idxs[h]
                                        << " size: "
                                        << send_process_ghost_segment_sizes[h]
                                        << std::endl;
                                }
                            }
                            assert(send_process_ghost_segment_idxs[k] +
                                       send_process_ghost_segment_sizes[k] <=
                                   atom_->nlocal + atom_->nghost);

                            num_elems_send +=
                                send_process_ghost_segment_sizes[k];
                        }

                        zoid.num_elems_send_process[t][proc] = num_elems_send;

                        for (int idx : send_process_local_idxs) {
                            send_buf_tags_next_dt[t][{zoid_num, proc}]
                                .push_back(atom_->tag[idx]);
                        }

                        for (int idx : send_process_ghost_idxs_vec) {
                            send_buf_tags_next_dt[t][{zoid_num, proc}]
                                .push_back(atom_->tag[idx]);
                        }

                        send_buf_num_tags_next_dt[t][{zoid_num, proc}] =
                            send_process_local_idxs.size() +
                            send_process_ghost_idxs_vec.size();

                        // 0 offset for the first force sent
                        force_offset_idxs_next_dt[t][{zoid_num, proc}]
                            .push_back(0);
                        int num_send_force = 0;
                        for (int i = 0; i < send_to.size(); i++) {
                            if (send_to[i] % comm->nprocs == proc) {
                                int num_force_segments =
                                    zoid.send_force_num_segments[t][i];
                                int send_force_size = 0;
                                std::vector<int> force_sizes;
                                for (int k = 0; k < num_force_segments; k++) {
                                    int segment_size =
                                        zoid.send_force_sizes[t][i][k];
                                    send_force_size += segment_size;
                                    force_sizes.push_back(segment_size);
                                }

                                int prev_size = force_offset_idxs_next_dt[t][{
                                    zoid_num, proc}][force_offset_idxs_next_dt
                                                         [t][{zoid_num, proc}]
                                                             .size() -
                                                     1];
                                force_offset_idxs_next_dt[t][{zoid_num, proc}]
                                    .push_back(send_force_size + prev_size);

                                num_send_force += send_force_size;
                            }
                        }

                        int num_send_vel = 0;
                        vel_offset_idxs_next_dt[t][{zoid_num, proc}].push_back(
                            0);
                        for (int i = 0; i < send_to.size(); i++) {
                            if (send_to[i] % comm->nprocs == proc) {
                                int num_vel_segments =
                                    zoid.send_pos_num_segments[t][i];
                                int send_vel_size = 0;
                                for (int k = 0; k < num_vel_segments; k++) {
                                    int segment_size =
                                        zoid.send_pos_sizes[t][i][k];
                                    send_vel_size += segment_size;
                                }

                                int prev_size = vel_offset_idxs_next_dt[t][{
                                    zoid_num,
                                    proc}][vel_offset_idxs_next_dt[t][{zoid_num,
                                                                       proc}]
                                               .size() -
                                           1];
                                vel_offset_idxs_next_dt[t][{zoid_num, proc}]
                                    .push_back(send_vel_size + prev_size);

                                num_send_vel += send_vel_size;
                            }
                        }

                        int pos_offset = num_send_force;

                        pos_offsets_next_dt[t][{zoid_num, proc}] = pos_offset;

                        int vel_offset_vec_idx = 0;
                        int force_offset_vec_idx = 0;
                        for (int i = 0; i < send_to.size(); i++) {
                            int send_zoid_num = send_to[i];
                            if (send_zoid_num % comm->nprocs == proc) {
                                int mpi_tag = (send_to[i] << 16) | zoid_num;
                                MPI_Request r1;
                                MPI_Request r2;
                                MPI_Request r3;
                                MPI_Request r4;
                                MPI_Request r5;
                                MPI_Request r6;
                                MPI_Request r7;
                                MPI_Request r8;

                                MPI_Isend(&send_buf_num_tags_next_dt[t][{
                                              zoid_num, proc}],
                                          1, MPI_INT, proc, mpi_tag, world,
                                          &r1);

                                MPI_Isend(
                                    send_buf_tags_next_dt[t][{zoid_num, proc}]
                                        .data(),
                                    send_buf_num_tags_next_dt[t]
                                                             [{zoid_num, proc}],
                                    MPI_INT, proc, mpi_tag, world, &r2);

                                MPI_Isend(&force_offset_idxs_next_dt[t][{
                                              zoid_num,
                                              proc}][force_offset_vec_idx++],
                                          1, MPI_INT, proc, mpi_tag, world,
                                          &r3);

                                MPI_Isend(
                                    &pos_offsets_next_dt[t][{zoid_num, proc}],
                                    1, MPI_INT, proc, mpi_tag, world, &r4);

                                MPI_Isend(
                                    &force_offset_idxs_next_dt
                                        [t][{zoid_num, proc}]
                                        [force_offset_idxs_next_dt[t][{zoid_num,
                                                                       proc}]
                                             .size() -
                                         1],
                                    1, MPI_INT, proc, mpi_tag, world, &r5);

                                MPI_Isend(&zoid.num_elems_send_process[t][proc],
                                          1, MPI_INT, proc, mpi_tag, world,
                                          &r6);

                                MPI_Isend(
                                    &vel_offset_idxs_next_dt[t][{
                                        zoid_num, proc}][vel_offset_vec_idx++],
                                    1, MPI_INT, proc, mpi_tag, world, &r7);

                                MPI_Isend(
                                    &vel_offset_idxs_next_dt
                                        [t][{zoid_num, proc}]
                                        [vel_offset_idxs_next_dt[t][{zoid_num,
                                                                     proc}]
                                             .size() -
                                         1],
                                    1, MPI_INT, proc, mpi_tag, world, &r8);

                                send_buf_requests_next_dt.push_back(r1);
                                send_buf_requests_next_dt.push_back(r2);
                                send_buf_requests_next_dt.push_back(r3);
                                send_buf_requests_next_dt.push_back(r4);
                                send_buf_requests_next_dt.push_back(r5);
                                send_buf_requests_next_dt.push_back(r6);
                                send_buf_requests_next_dt.push_back(r7);
                                send_buf_requests_next_dt.push_back(r8);
                            }
                        }
                    }
                }
            }
        }

        for (int dep = 0; dep < NUM_DEPS; dep++) {
            for (int j = 0; j < lmp->queues_next_dt[dep].size(); j++) {
                queue_info& zoid = lmp->queues_next_dt[dep][j];
                int zoid_num = zoid.num;
                std::vector<int>& recv_from =
                    lmp->recv_from_neighbors_next_dt[zoid_num];

                if (zoid_num % comm->nprocs == comm->me) {
                    Atom* atom_ =
                        lmp->atom_stencil_md[zoid_num]
                                            [NUM_TIMESTEPS_IN_PARALLEL - t];
                    zoid.recv_process_segment_types[t] =
                        new int*[recv_from.size()];
                    zoid.recv_process_segment_idxs[t] =
                        new int*[recv_from.size()];
                    zoid.recv_process_segment_sizes[t] =
                        new int*[recv_from.size()];
                    zoid.recv_process_num_segments[t] =
                        new int[recv_from.size()];

                    // send idxs belonging to that zoid?
                    zoid.recv_process_force_offset[t] =
                        new int[recv_from.size()];
                    zoid.recv_process_pos_offset[t] = new int[recv_from.size()];
                    zoid.recv_process_vel_offset[t] = new int[recv_from.size()];

                    for (int i = 0; i < recv_from.size(); i++) {
                        int recv_zoid_num = recv_from[i];
                        int recv_proc = recv_zoid_num % comm->nprocs;

                        int mpi_tag = (zoid_num << 16 | recv_zoid_num);
                        int num_tags = 0;
                        MPI_Recv(&num_tags, 1, MPI_INT, recv_proc, mpi_tag,
                                 world, MPI_STATUS_IGNORE);

                        int* recv_tags = new int[num_tags];
                        MPI_Recv(recv_tags, num_tags, MPI_INT, recv_proc,
                                 mpi_tag, world, MPI_STATUS_IGNORE);

                        int force_offset_idx = 0;
                        MPI_Recv(&force_offset_idx, 1, MPI_INT, recv_proc,
                                 mpi_tag, world, MPI_STATUS_IGNORE);

                        int num_force_recv = 0;
                        MPI_Recv(&num_force_recv, 1, MPI_INT, recv_proc,
                                 mpi_tag, world, MPI_STATUS_IGNORE);

                        // offset within buffer of forces, which one belongs to this zoid
                        int force_offset = 0;
                        MPI_Recv(&force_offset, 1, MPI_INT, recv_proc, mpi_tag,
                                 world, MPI_STATUS_IGNORE);

                        int num_pos_recv = 0;
                        MPI_Recv(&num_pos_recv, 1, MPI_INT, recv_proc, mpi_tag,
                                 world, MPI_STATUS_IGNORE);

                        int vel_offset = 0;
                        MPI_Recv(&vel_offset, 1, MPI_INT, recv_proc, mpi_tag,
                                 world, MPI_STATUS_IGNORE);

                        int num_vel_recv = 0;
                        MPI_Recv(&num_vel_recv, 1, MPI_INT, recv_proc, mpi_tag,
                                 world, MPI_STATUS_IGNORE);

                        for (int k = 0;
                             k < lmp->recv_from_neighbors_procs_next_dt.size();
                             k++) {
                            if (lmp->recv_from_neighbors_procs_next_dt[k] ==
                                recv_from[i]) {
                                lmp->num_recv_force_from_zoid_next_dt[t][k] =
                                    num_force_recv;
                                lmp->num_recv_pos_from_zoid_next_dt[t][k] =
                                    num_pos_recv;
                                lmp->num_recv_vel_from_zoid_next_dt[t][k] =
                                    num_vel_recv;

                                if (DEBUG_SEND_RECV_DATA) {
                                    lmp->num_recv_elems_from_zoid_next_dt[t][k] = num_force_recv * (3 + 1) + num_pos_recv * (3 + 1) + num_vel_recv * (3 + 1);
                                } else {
                                    lmp->num_recv_elems_from_zoid_next_dt[t][k] = num_force_recv * (3) + num_pos_recv * (3) + num_vel_recv * (3);
                                }
                            }
                        }

                        zoid.recv_process_force_offset[t][i] = force_offset_idx;
                        zoid.recv_process_vel_offset[t][i] = vel_offset;
                        zoid.recv_process_pos_offset[t][i] = num_force_recv;

                        assert(vel_offset >= 0 && vel_offset <= 1000000);

                        std::map<int, int> tag_to_idx_in_buf;
                        for (int k = 0; k < num_tags; k++) {
                            tag_to_idx_in_buf[recv_tags[k]] = k;
                        }

                        // this is sorted??
                        std::vector<int> local_buf_idxs;
                        for (int k = 0;
                             k < zoid.recv_list_local_num_force_pos[t][i];
                             k++) {
                            int idx = zoid.recv_list_local_force_pos[t][i][k];
                            int buf_idx = tag_to_idx_in_buf[atom_->tag[idx]];
                            local_buf_idxs.push_back(buf_idx);
                        }

                        std::vector<int> local_segment_idxs_buf;
                        std::vector<int> local_segment_sizes_buf;

                        int num_local_segments_buf =
                            get_segments(local_buf_idxs, local_segment_idxs_buf,
                                         local_segment_sizes_buf);

                        std::vector<int> ghost_buf_idxs;

                        std::set<int> ghost_buf_idxs_set;
                        for (int k = 0; k < zoid.recv_ghost_num_segments[t][i];
                             k++) {
                            int segment_size = zoid.recv_ghost_sizes[t][i][k];
                            int segment_idx = zoid.recv_ghost_idxs[t][i][k];
                            for (int h = 0; h < segment_size; h++) {
                                int idx = segment_idx + h;
                                int buf_idx =
                                    tag_to_idx_in_buf[atom_->tag[idx]];
                                ghost_buf_idxs.push_back(buf_idx);

                                ghost_buf_idxs_set.insert(buf_idx);
                            }
                        }

                        assert(ghost_buf_idxs.size() ==
                               ghost_buf_idxs_set.size());

                        std::vector<int> ghost_segment_idxs_buf;
                        std::vector<int> ghost_segment_sizes_buf;

                        int num_ghost_segments_buf =
                            get_segments(ghost_buf_idxs, ghost_segment_idxs_buf,
                                         ghost_segment_sizes_buf);

                        zoid.recv_process_num_segments[t][i] =
                            num_local_segments_buf + num_ghost_segments_buf;
                        zoid.recv_process_segment_types[t][i] =
                            new int[num_local_segments_buf +
                                    num_ghost_segments_buf];
                        zoid.recv_process_segment_idxs[t][i] =
                            new int[num_local_segments_buf +
                                    num_ghost_segments_buf];
                        zoid.recv_process_segment_sizes[t][i] =
                            new int[num_local_segments_buf +
                                    num_ghost_segments_buf];

                        for (int k = 0; k < num_local_segments_buf; k++) {
                            zoid.recv_process_segment_types[t][i][k] =
                                RECV_DATA_PROCESS_LOCAL;
                            zoid.recv_process_segment_idxs[t][i][k] =
                                local_segment_idxs_buf[k];
                            zoid.recv_process_segment_sizes[t][i][k] =
                                local_segment_sizes_buf[k];
                        }

                        for (int k = 0; k < num_ghost_segments_buf; k++) {
                            int segment_num = k + num_local_segments_buf;
                            zoid.recv_process_segment_types[t][i][segment_num] =
                                RECV_DATA_PROCESS_GHOST;
                            zoid.recv_process_segment_idxs[t][i][segment_num] =
                                ghost_segment_idxs_buf[k];
                            zoid.recv_process_segment_sizes[t][i][segment_num] =
                                ghost_segment_sizes_buf[k];
                        }

                        delete[] recv_tags;
                    }
                }
            }
        }

        MPI_Barrier(world);
    }

    MPI_Waitall(send_buf_requests_next_dt.size(),
                send_buf_requests_next_dt.data(), MPI_STATUSES_IGNORE);
    MPI_Barrier(world);

    // compute individual zoid numbers
    int zoid_nrecv_force = 0;
    int zoid_nrecv_vel = 0;
    int zoid_nrecv_pos = 0;

    std::map<int, std::vector<int>> force_idx_to_recv_zoids[NUM_ZOIDS];

    for (int dep = 0; dep < NUM_DEPS; dep++) {
        for (int j = 0; j < lmp->queues[dep].size(); j++) {
            queue_info& zoid = lmp->queues[dep][j];
            if (zoid.num % comm->nprocs == comm->me) {
                auto& recv_from = lmp->recv_from_neighbors[zoid.num];
                for (int t = 1; t < NUM_TIMESTEPS_IN_PARALLEL + 1; t++) {
                    Atom* atom_ = lmp->atom_stencil_md[zoid.num][t];
                    for (int i = 0; i < recv_from.size(); i++) {
                        int recv_zoid_num = recv_from[i];
                        if (recv_zoid_num % comm->nprocs != comm->me) {
                            int num_force =
                                zoid.recv_list_local_num_force_only[t][i];
                            int num_pos =
                                zoid.recv_list_local_num_force_pos[t][i];
                            int num_vel = num_pos;

                            int ghost_pos = 0;
                            for (int k = 0;
                                 k < zoid.recv_ghost_num_segments[t][i]; k++) {
                                ghost_pos += zoid.recv_ghost_sizes[t][i][k];
                            }

                            if (t == NUM_TIMESTEPS_IN_PARALLEL) {
                                for (int k = 0; k < num_force; k++) {
                                    int idx =
                                        zoid.recv_list_local_force_only[t][i]
                                                                       [k];
                                    force_idx_to_recv_zoids[zoid.num][idx]
                                        .push_back(recv_zoid_num);
                                }
                            }

                            zoid_nrecv_force += num_force;
                            zoid_nrecv_vel += num_vel;
                            zoid_nrecv_pos += num_pos + ghost_pos;

                            /*
                            std::cout << MAGENTA << "zoid: " << zoid.num << " recv from: " << recv_zoid_num << " time: " << t
                                << " num force: " << num_force << " num pos: " << num_pos << " num vel: " << num_vel << RESET_COLOR << std::endl;
                            */
                        }
                    }
                }
            }
        }
    }

    int nrecv_force = 0;
    int nrecv_pos = 0;
    int nrecv_vel = 0;

    for (int k = 0; k < lmp->recv_from_neighbors_procs.size(); k++) {
        if (lmp->recv_from_neighbors_procs[k] % comm->nprocs != comm->me) {
            for (int t = 1; t < NUM_TIMESTEPS_IN_PARALLEL + 1; t++) {
                nrecv_force += lmp->num_recv_force_from_zoid[t][k];
                nrecv_pos += lmp->num_recv_pos_from_zoid[t][k];
                nrecv_vel += lmp->num_recv_vel_from_zoid[t][k];
            }
        }
    }

    int nrecv = nrecv_force * 3 + nrecv_pos * 3 + nrecv_vel * 3;

    std::cout << "process: " << comm->me
              << " nrecv force: " << nrecv_force * (3)
              << " nrecv pos: " << nrecv_pos * (3)
              << " nrecv vel: " << nrecv_vel * 3 << " nrecv total: " << nrecv
              << " zoid calc force. " << zoid_nrecv_force * 3
              << " zoid calc pos: " << zoid_nrecv_pos * 3
              << " zoid calc vel: " << zoid_nrecv_vel * 3 << std::endl;

    int res = 0;
    MPI_Allreduce(&nrecv, &res, 1, MPI_INT, MPI_SUM, world);

    if (comm->me == 0) {
        std::cout << "num total recv across all processes: " << res
                  << std::endl;
    }

    MPI_Barrier(world);

    stencilMD->COMPUTE_NUM_SEND_RECV_PROCESS();

    // compute force and then clear everything
    constexpr bool DO_WARMUP_PAIR_CALC = false;

    if (DO_WARMUP_PAIR_CALC) {
        std::cout << "Prepping forces for each timestep" << std::endl;
        for (int t = 0; t < NUM_TIMESTEPS_IN_PARALLEL + 1; t++) {
            for (int dep = 0; dep < NUM_DEPS; dep++) {
                for (int j = 0; j < lmp->queues[dep].size(); j++) {
                    queue_info& zoid = lmp->queues[dep][j];
                    int zoid_num = zoid.num;
                    if (zoid_num % comm->nprocs == comm->me) {
                        /*
                        AtomKokkos* atomKK_ =
                            (AtomKokkos*)lmp->atom_stencil_md[zoid_num][t];
                        */
                        Force* force_ = lmp->force_stencil_md[zoid_num][t];
                        /*
                        atomKK_->sync_stencil_md(
                            force->pair->execution_space,
                            force->pair->datamask_read,
                            lmp->atom_stencil_md[zoid_num][t]);
                        */
                        // Warm up?
                        force_->pair->compute_stencil_md(
                            eflag, vflag, lmp->atom_stencil_md[zoid_num][t],
                            zoid.can_eval_center[t],
                            lmp->zoid_num_to_zoid[zoid_num], nullptr);
                        force_clear_stencil_md(
                            lmp->atom_stencil_md[zoid_num][t], force_,
                            lmp->neighbor_stencil_md[zoid_num][t]);
                        /*
                        atomKK_->modified_stencil_md(
                            force_->pair->execution_space,
                            force_->pair->datamask_modify,
                            lmp->atom_stencil_md[zoid_num][t]);
                        */
                    }
                }
            }
        }

        std::cout << "done prepping forces" << std::endl;
    }

    // doing a round of next_dt communication to prep the buffers to have enough size
    std::cout << "prep next dt send" << std::endl;

    std::unordered_map<int, std::vector<int>> dep_to_wait_idxs_next_dt;
    std::set<int> zoids_already_waiting_on_next_dt;
    std::unordered_map<int, std::vector<int>> zoid_to_wait_idxs_next_dt;

    for (int dep = 1; dep < NUM_DEPS; dep++) {
        int num_zoids = 0;
        std::vector<int> dep_recv_zoids;
        for (int i = 0; i < lmp->recv_from_neighbors_procs_next_dt.size();
             i++) {
            int recv_zoid_num = lmp->recv_from_neighbors_procs_next_dt[i];
            if (recv_zoid_num % comm->nprocs != comm->me) {
                for (int j = 0; j < lmp->queues_next_dt[dep].size(); j++) {
                    queue_info& zoid = lmp->queues_next_dt[dep][j];
                    int zoid_num = zoid.num;
                    if (zoid_num % comm->nprocs == comm->me) {
                        auto& recv_from =
                                lmp->recv_from_neighbors_next_dt[zoid_num];
                        if (std::find(recv_from.begin(), recv_from.end(),
                                      recv_zoid_num) != recv_from.end() &&
                            zoids_already_waiting_on_next_dt.find(
                                    recv_zoid_num) ==
                            zoids_already_waiting_on_next_dt.end()) {
                            dep_to_wait_idxs_next_dt[dep].push_back(i);
                            zoids_already_waiting_on_next_dt.insert(
                                    recv_zoid_num);
                            dep_recv_zoids.push_back(recv_zoid_num);
                            zoid_to_wait_idxs_next_dt[zoid_num].push_back(i);
                            break;
                        }
                    }
                }
            }
        }
    }

    std::map<int, int> zoid_num_to_num_procs_next_dt;
    for (int zoid_num = 0; zoid_num < NUM_ZOIDS; zoid_num++) {
        if (zoid_num % comm->nprocs == comm->me) {
            int num_procs = 0;
            for (int proc = 0; proc < comm->nprocs; proc++) {
                if (proc == comm->me) {
                    continue;
                }
                for (int i = 0;
                     i < lmp->send_to_neighbors_next_dt[zoid_num].size(); i++) {
                    if (lmp->send_to_neighbors_next_dt[zoid_num][i] %
                        comm->nprocs ==
                        proc) {
                        num_procs++;
                        break;
                    }
                }
            }
            assert(num_procs >= 0 && num_procs < comm->nprocs);
            zoid_num_to_num_procs_next_dt[zoid_num] = num_procs;
        }
    }

    stencilMD->MODIFY_PRE_FORCE_SETUP(vflag);

    // start compute
    std::vector<MPI_Request> send_requests_next_dt[NUM_ZOIDS];
    std::vector<std::future<void>> send_request_threads_next_dt;
    for (int zoid_num = 0; zoid_num < NUM_ZOIDS; zoid_num++) {
        if (zoid_num % comm->nprocs == comm->me) {
            send_requests_next_dt[zoid_num] = std::move(std::vector<MPI_Request>(zoid_num_to_num_procs_next_dt[zoid_num], MPI_REQUEST_NULL));
        }
    }

    int num_zoids_recv_from_next_dt = lmp->recv_from_neighbors_procs_next_dt.size();
    std::vector<MPI_Request> receive_requests_next_dt(
            num_zoids_recv_from_next_dt, MPI_REQUEST_NULL);

    for (int i = 0; i < lmp->recv_from_neighbors_procs_next_dt.size(); i++) {
        int recv_zoid_num = lmp->recv_from_neighbors_procs_next_dt[i];
        if (recv_zoid_num % comm->nprocs != comm->me) {
            comm->receive_data_process_stencil_md(false,
                                                  &receive_requests_next_dt[i], recv_zoid_num, false);
        }
    }

    for (int dep = 0; dep < NUM_DEPS; dep++) {
        if (dep > 0) {
            auto begin = std::chrono::high_resolution_clock::now();
            for (int idx: dep_to_wait_idxs_next_dt[dep]) {
                int recv_zoid_num = lmp->recv_from_neighbors_procs_next_dt[idx];
                MPI_Wait(&receive_requests_next_dt[idx], MPI_STATUS_IGNORE);
                comm->unpack_data_process_stencil_md(false, recv_zoid_num, false);
            }
        }

        for (int j = 0; j < lmp->queues_next_dt[dep].size(); j++) {
            queue_info& zoid = lmp->queues_next_dt[dep][j];
            int zoid_num = zoid.num;
            if (zoid_num % comm->nprocs == comm->me) {
                if (dep < NUM_DEPS - 1) {
                    Comm *comm_ = lmp->comm_stencil_md[zoid_num];
                    auto &atom_arr = lmp->atom_stencil_md[zoid_num];

                    int vec_idx = 0;
                    for (int proc = 0; proc < comm->nprocs; proc++) {
                        bool sent = comm_->send_data_to_process_stencil_md(false,
                                                                           atom_arr,
                                                                           lmp->zoid_num_to_zoid_next_dt[zoid_num],
                                                                           &send_requests_next_dt[zoid_num][vec_idx],
                                                                           proc, false, &send_pack_duration);
                        if (sent) {
                            // send_request_threads_next_dt.emplace_back(std::async(std::launch::async,
                            vec_idx++;
                        }
                    }
                }
            }
        }
    }

    for (int i = 0; i < NUM_ZOIDS; i++) {
        if (send_requests_next_dt[i].size() > 0) {
            MPI_Waitall(send_requests_next_dt[i].size(), send_requests_next_dt[i].data(), MPI_STATUSES_IGNORE);
        }
    }

    // compute force but only for the first timestep
    int num_zoids_recv_from = lmp->recv_from_neighbors_procs.size();
    std::vector<MPI_Request> receive_requests(
            lmp->recv_from_neighbors_procs.size(), MPI_REQUEST_NULL);

    for (int i = 0; i < lmp->recv_from_neighbors_procs.size(); i++) {
        int recv_zoid_num = lmp->recv_from_neighbors_procs[i];
        if (recv_zoid_num % comm->nprocs != comm->me) {
            comm->receive_data_process_stencil_md(true, &receive_requests[i], recv_zoid_num, true);
        }
    }

    // map dependency levels to number of zoids to wait on
    std::vector<int> dep_to_wait_idxs[NUM_DEPS];
    std::set<int> zoids_already_waiting_on;

    for (int dep = 1; dep < NUM_DEPS; dep++) {
        int num_zoids = 0;
        for (int i = 0; i < lmp->recv_from_neighbors_procs.size(); i++) {
            int recv_zoid_num = lmp->recv_from_neighbors_procs[i];
            if (recv_zoid_num % comm->nprocs != comm->me) {
                for (int j = 0; j < lmp->queues[dep].size(); j++) {
                    queue_info& zoid = lmp->queues[dep][j];
                    int zoid_num = zoid.num;
                    if (zoid_num % comm->nprocs == comm->me) {
                        auto& recv_from = lmp->recv_from_neighbors[zoid_num];
                        if (std::find(recv_from.begin(), recv_from.end(),
                                      recv_zoid_num) != recv_from.end() &&
                            zoids_already_waiting_on.find(recv_zoid_num) ==
                                zoids_already_waiting_on.end()) {
                            dep_to_wait_idxs[dep].push_back(i);
                            zoids_already_waiting_on.insert(recv_zoid_num);
                            break;
                        }
                    }
                }
            }
        }
    }

    std::vector<MPI_Request> send_requests[NUM_ZOIDS];
    std::vector<std::thread> send_request_threads;

    for (int zoid_num = 0; zoid_num < NUM_ZOIDS; zoid_num++) {
        if (zoid_num % comm->nprocs == comm->me) {
            int num_procs = 0;
            for (int proc = 0; proc < comm->nprocs; proc++) {
                if (proc == comm->me) {
                    continue;
                }
                for (int i = 0; i < lmp->send_to_neighbors[zoid_num].size();
                     i++) {
                    if (lmp->send_to_neighbors[zoid_num][i] % comm->nprocs ==
                        proc) {
                        num_procs++;
                        break;
                    }
                }
            }
            assert(num_procs >= 0 && num_procs < comm->nprocs);
            if (num_procs > 0) {
                send_requests[zoid_num] =
                    std::vector<MPI_Request>(num_procs, MPI_REQUEST_NULL);
            }
        }
    }

    for (int dep = 0; dep < NUM_DEPS; dep++) {
        if (dep > 0) {
            for (int idx : dep_to_wait_idxs[dep]) {
                int recv_zoid_num = lmp->recv_from_neighbors_procs[idx];
                MPI_Wait(&receive_requests[idx], MPI_STATUS_IGNORE);
                comm->unpack_data_process_stencil_md(true, recv_zoid_num, true);
            }
        }

        for (int j = 0; j < lmp->queues[dep].size(); j++) {
            queue_info& zoid = lmp->queues[dep][j];
            int zoid_num = zoid.num;
            if (zoid_num % comm->nprocs == comm->me) {
                Atom* atom_ = lmp->atom_stencil_md[zoid_num][0];
                Force* force_ = lmp->force_stencil_md[zoid_num][0];
                // todo: eflag and vflag might cause some issues
                // TODO: compute force for each pair in parallel

                force_->pair->compute_stencil_md(
                    eflag, vflag, lmp->atom_stencil_md[zoid_num][0],
                    zoid.can_eval_center[0], lmp->zoid_num_to_zoid[zoid_num],
                    nullptr);

                if (dep < NUM_DEPS - 1) {
                    Comm *comm_ = lmp->comm_stencil_md[zoid_num];

                    int vec_idx = 0;

                    for (int proc = 0; proc < comm->nprocs; proc++) {
                        bool sent = comm_->send_data_to_process_stencil_md(true,
                                                                           lmp->atom_stencil_md[zoid_num],
                                                                           lmp->zoid_num_to_zoid[zoid_num],
                                                                           &send_requests[zoid_num][vec_idx], proc,
                                                                           true, &send_pack_duration);
                        if (sent) {
                            vec_idx++;
                        }
                    }
                }
            }
        }
    }

    for (int i = 0; i < NUM_ZOIDS; i++) {
        if (send_requests[i].size() > 0) {
            MPI_Waitall(send_requests[i].size(), send_requests[i].data(), MPI_STATUSES_IGNORE);
        }
    }

    double* send_f = new double[(atom->natoms + 1) * 3];
    for (int i = 0; i < (atom->natoms + 1) * 3; i++) {
        send_f[i] = 0.0;
    }

    for (int i = 0; i < atom->nlocal; i++) {
        int tag = atom->tag[i];
        if (!(tag >= 0 && tag <= atom->natoms)) {
            std::cout << "ERROR. "
                      << " idx: " << i << " out of nlocal: " << atom->nlocal
                      << " tag: " << atom->tag[i] << std::endl;
        }
        assert(tag >= 0 && tag <= atom->natoms);
        send_f[tag * 3 + 0] = atom->f[i][0];
        send_f[tag * 3 + 1] = atom->f[i][1];
        send_f[tag * 3 + 2] = atom->f[i][2];
    }

    double* recv_f = new double[(atom->natoms + 1) * 3];

    for (int i = 0; i < (atom->natoms + 1) * 3; i++) {
        recv_f[i] = 0.0;
    }

    MPI_Allreduce(send_f, recv_f, (atom->natoms + 1) * 3, MPI_DOUBLE, MPI_SUM,
                  world);

    for (int i = 0; i < atom->nlocal; i++) {
        int tag = atom->tag[i];
        for (int dim = 0; dim < 3; dim++) {
            if (fabs(recv_f[tag * 3 + dim] - atom->f[i][dim]) > 5e-5) {
                std::cout << "error in recv tag: " << tag << " idx: " << i
                          << " dim: " << dim
                          << " what I have: " << atom->f[i][dim]
                          << " what I got: " << recv_f[tag * 3 + dim]
                          << " diff: "
                          << fabs(recv_f[tag * 3 + dim] - atom->f[i][dim])
                          << std::endl;
            }
            assert(fabs(recv_f[tag * 3 + dim] - atom->f[i][dim]) <= 5e-5);
        }
    }

    MPI_Barrier(world);

    double x_ = 0.0;
    double y_ = 0.0;
    double z_ = 0.0;
    for (int dep = 0; dep < NUM_DEPS; dep++) {
        for (int j = 0; j < lmp->queues[dep].size(); j++) {
            queue_info& zoid = lmp->queues[dep][j];
            int zoid_num = zoid.num;
            if (zoid_num % comm->nprocs == comm->me) {
                Atom* atom_ = lmp->atom_stencil_md[zoid_num][0];
                for (int k = 0; k < atom_->nlocal; k++) {
                    x_ += atom_->f[k][0] + atom_->eval_f_stencil_md[k][0];
                    y_ += atom_->f[k][1] + atom_->eval_f_stencil_md[k][1];
                    z_ += atom_->f[k][2] + atom_->eval_f_stencil_md[k][2];
                }
            }
        }
    }

    double total_x = 0;
    double total_y = 0;
    double total_z = 0;
    MPI_Reduce(&x_, &total_x, 1, MPI_DOUBLE, MPI_SUM, 0, world);
    MPI_Reduce(&y_, &total_y, 1, MPI_DOUBLE, MPI_SUM, 0, world);
    MPI_Reduce(&z_, &total_z, 1, MPI_DOUBLE, MPI_SUM, 0, world);
    if (comm->me == 0) {
        std::cout << "STENCIL MD Sum x: " << total_x << " Sum y: " << total_y
                  << " Sum z: " << total_z << std::endl;
    }

    int total_evaled = 0;
    for (int dep = 0; dep < NUM_DEPS; dep++) {
        for (int j = 0; j < lmp->queues[dep].size(); j++) {
            queue_info& zoid = lmp->queues[dep][j];
            int zoid_num = zoid.num;
            if (zoid_num % comm->nprocs == comm->me) {
                Atom* atom_ = lmp->atom_stencil_md[zoid_num][0];
                // Atom *next = lmp->atom_stencil_md[zoid_num][1];
                for (int idx = 0; idx < atom_->nlocal; idx++) {
                    int tag = atom_->tag[idx];
                    for (int dim = 0; dim < 3; dim++) {
                        double my_force = atom_->f[idx][dim] +
                                          atom_->eval_f_stencil_md[idx][dim];
                        if (fabs(recv_f[tag * 3 + dim] - my_force) > 5e-5) {
                            std::cout
                                << "zoid: " << zoid_num << " idx: " << idx
                                << " tag: " << atom_->tag[idx]
                                << " dim: " << dim
                                << " what I have: " << my_force
                                << " what lammps has: " << recv_f[tag * 3 + dim]
                                << " diff: "
                                << fabs(recv_f[tag * 3 + dim] - my_force)
                                << std::endl;
                            std::cout << "tag: " << atom_->tag[idx]
                                      << " pos: " << atom_->x[idx][0] << " "
                                      << atom_->x[idx][1] << " "
                                      << atom_->x[idx][2] << std::endl;

                            std::cout << "recv force: " << atom_->f[idx][dim]
                                      << " eval force: "
                                      << atom_->eval_f_stencil_md[idx][dim]
                                      << " my force: " << my_force << std::endl;
                            assert(fabs(my_force - recv_f[tag * 3 + dim]) <= 5e-5);
                        }
                    }
                    total_evaled++;
                }
            }
        }
    }

    int total_atoms_evaled = 0;
    MPI_Allreduce(&total_evaled, &total_atoms_evaled, 1, MPI_INT, MPI_SUM,
                  world);

    std::cout << "atoms evaled: " << total_atoms_evaled
              << " total number of atoms: " << atom->natoms << std::endl;

    MPI_Barrier(world);

    delete[] send_f;
    delete[] recv_f;

    double total_temp_for_me = 0;

    stencilMD->MODIFY_SETUP(vflag);
    /*
    for (int dep = 0; dep < NUM_DEPS; dep++) {
        for (int j = 0; j < lmp->queues[dep].size(); j++) {
            queue_info& zoid = lmp->queues[dep][j];
            int zoid_num = zoid.num;
            if (zoid_num % comm->nprocs == comm->me) {
                lmp->modify_stencil_md[zoid_num]->setup(vflag);
            }
        }
    }
    */


    std::cout << GREEN << "-------- SETUP STENCIL MD PASSED ---------"
              << RESET_COLOR << std::endl;
    MPI_Barrier(world);

    /*
    modify->setup(vflag);
    output->setup(flag);
    update->setupflag = 0;
    */
}

/* ----------------------------------------------------------------------
   setup without output
   flag = 0 = just force calculation
   flag = 1 = reneighbor and force calculation
------------------------------------------------------------------------- */

void Verlet::setup_minimal(int flag) {
    update->setupflag = 1;

    // setup domain, communication and neighboring
    // acquire ghosts
    // build neighbor lists

    if (flag) {
        modify->setup_pre_exchange();
        if (triclinic)
            domain->x2lamda(atom->nlocal);
        domain->pbc();
        domain->reset_box();
        comm->setup();
        if (neighbor->style)
            neighbor->setup_bins();
        comm->exchange();
        comm->borders();
        if (triclinic)
            domain->lamda2x(atom->nlocal + atom->nghost);
        domain->image_check();
        domain->box_too_small_check();
        modify->setup_pre_neighbor();
        neighbor->build(1);
        modify->setup_post_neighbor();
        neighbor->ncalls = 0;
    }

    // compute all forces

    ev_set(update->ntimestep);
    force_clear();
    modify->setup_pre_force(vflag);

    if (pair_compute_flag)
        force->pair->compute(eflag, vflag);
    else if (force->pair)
        force->pair->compute_dummy(eflag, vflag);

    if (atom->molecular != Atom::ATOMIC) {
        if (force->bond)
            force->bond->compute(eflag, vflag);
        if (force->angle)
            force->angle->compute(eflag, vflag);
        if (force->dihedral)
            force->dihedral->compute(eflag, vflag);
        if (force->improper)
            force->improper->compute(eflag, vflag);
    }

    if (force->kspace) {
        force->kspace->setup();
        if (kspace_compute_flag)
            force->kspace->compute(eflag, vflag);
        else
            force->kspace->compute_dummy(eflag, vflag);
    }

    modify->setup_pre_reverse(eflag, vflag);
    if (force->newton)
        comm->reverse_comm();

    modify->setup(vflag);
    update->setupflag = 0;
}

/* ----------------------------------------------------------------------
   run for N steps
------------------------------------------------------------------------- */

void Verlet::run(int n) {
    // TODO: This is meant to maximize spending time ONLY on what I am tracking
    eflag = 0; vflag = 0;

    bigint ntimestep;
    int nflag, sortflag;

    int n_post_integrate = modify->n_post_integrate;
    int n_pre_exchange = modify->n_pre_exchange;
    int n_pre_neighbor = modify->n_pre_neighbor;
    int n_post_neighbor = modify->n_post_neighbor;
    int n_pre_force = modify->n_pre_force;
    int n_pre_reverse = modify->n_pre_reverse;
    int n_post_force_any = modify->n_post_force_any;
    int n_end_of_step = modify->n_end_of_step;

    if (atom->sortfreq > 0)
        sortflag = 1;
    else
        sortflag = 0;

    int test_num_timesteps = n + 1;
    double* test_f[test_num_timesteps];
    double* test_x[test_num_timesteps];

    for (int i = 0; i < test_num_timesteps; i++) {
        test_f[i] = new double[3 * (atom->natoms + 1)];
        test_x[i] = new double[3 * (atom->natoms + 1)];
        for (int j = 0; j < 3 * (atom->natoms + 1); j++) {
            test_f[i][j] = 0.0;
            test_x[i][j] = 0.0;
        }
    }

    double* send_f = new double[3 * (atom->natoms + 1)];
    for (int i = 0; i < 3 * (atom->natoms + 1); i++) {
        send_f[i] = 0;
    }

    double* send_x = new double[3 * (atom->natoms + 1)];
    for (int i = 0; i < 3 * (atom->natoms + 1); i++) {
        send_x[i] = 0;
    }

    int64_t lammps_compute_duration = 0;
    int64_t lammps_comm_duration = 0;
    int64_t lammps_forward_comm_duration = 0;
    int64_t lammps_reverse_comm_duration = 0;
    int64_t lammps_modify_duration = 0;
    int64_t lammps_modify_pre_force_duration = 0;

    // for (int i = 0; i < n; i++) {
    auto begin_lammps = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < n + 1; i++) {
        /*
        if (timer->check_timeout(i)) {
            assert(false);
            update->nsteps = i;
            break;
        }
        */

        // ntimestep = ++update->ntimestep;
        // ev_set(ntimestep);

        // initial time integration

        timer->stamp();

        // Begin stencil md code
        if (TEST_AGAINST_LAMMPS_LOCAL) {
            // memset(send_f, 0, sizeof(send_f));
            for (int j = 0; j < 3 * (atom->natoms + 1); j++) {
                send_f[j] = 0;
            }
            for (int j = 0; j < 3 * (atom->natoms + 1); j++) {
                send_x[j] = 0;
            }

            for (int j = 0; j < atom->nlocal; j++) {
                int tag = atom->tag[j];
                assert(tag >= 0 && tag <= atom->natoms);
                send_f[tag * 3 + 0] = atom->f[j][0];
                send_f[tag * 3 + 1] = atom->f[j][1];
                send_f[tag * 3 + 2] = atom->f[j][2];

                send_x[tag * 3 + 0] = atom->x[j][0];
                send_x[tag * 3 + 1] = atom->x[j][1];
                send_x[tag * 3 + 2] = atom->x[j][2];
            }

            MPI_Allreduce(send_f, test_f[i], (atom->natoms + 1) * 3, MPI_DOUBLE,
                          MPI_SUM, world);

            MPI_Allreduce(send_x, test_x[i], (atom->natoms + 1) * 3, MPI_DOUBLE,
                          MPI_SUM, world);
        }

        if (i == n) {
            break;
        }
        // end stencil md code

        auto begin_m = std::chrono::high_resolution_clock::now();
        modify->initial_integrate(vflag);
        auto end_m = std::chrono::high_resolution_clock::now();
        auto duration_m = std::chrono::duration_cast<std::chrono::microseconds>(end_m - begin_m).count();
        lammps_modify_duration += duration_m;
        if (n_post_integrate)
            modify->post_integrate();
        timer->stamp(Timer::MODIFY);

        // regular communication vs neighbor list rebuild

        nflag = neighbor->decide();

        if (nflag == 0) {
            timer->stamp();
            auto begin = std::chrono::high_resolution_clock::now();
            comm->forward_comm();
            auto end = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count();
            lammps_comm_duration += duration;
            lammps_forward_comm_duration += duration;
            lammps_forward_comm_times.push_back(duration);
            timer->stamp(Timer::COMM);
        } else {
            assert(false);
            if (n_pre_exchange) {
                timer->stamp();
                modify->pre_exchange();
                timer->stamp(Timer::MODIFY);
            }
            if (triclinic)
                domain->x2lamda(atom->nlocal);
            domain->pbc();
            if (domain->box_change) {
                domain->reset_box();
                comm->setup();
                if (neighbor->style)
                    neighbor->setup_bins();
            }
            timer->stamp();
            comm->exchange();
            if (sortflag && ntimestep >= atom->nextsort)
                atom->sort();
            comm->borders();
            if (triclinic)
                domain->lamda2x(atom->nlocal + atom->nghost);
            timer->stamp(Timer::COMM);
            if (n_pre_neighbor) {
                modify->pre_neighbor();
                timer->stamp(Timer::MODIFY);
            }
            neighbor->build(1);
            timer->stamp(Timer::NEIGH);
            if (n_post_neighbor) {
                modify->post_neighbor();
                timer->stamp(Timer::MODIFY);
            }
        }

        // force computations
        // important for pair to come before bonded contributions
        // since some bonded potentials tally pairwise energy/virial
        // and Pair:ev_tally() needs to be called before any tallying

        force_clear();

        timer->stamp();

        if (n_pre_force) {
            auto begin = std::chrono::high_resolution_clock::now();
            modify->pre_force(vflag);
            auto end = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count();
            lammps_modify_duration += duration;
            lammps_modify_pre_force_duration += duration;
            timer->stamp(Timer::MODIFY);
        }

        if (pair_compute_flag) {
            auto begin = std::chrono::high_resolution_clock::now();
            force->pair->compute(eflag, vflag);
            auto end = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count();
            lammps_compute_duration += duration;
            timer->stamp(Timer::PAIR);
        }

        if (atom->molecular != Atom::ATOMIC) {
            if (force->bond)
                force->bond->compute(eflag, vflag);
            if (force->angle)
                force->angle->compute(eflag, vflag);
            if (force->dihedral)
                force->dihedral->compute(eflag, vflag);
            if (force->improper)
                force->improper->compute(eflag, vflag);
            timer->stamp(Timer::BOND);
        }

        if (kspace_compute_flag) {
            force->kspace->compute(eflag, vflag);
            timer->stamp(Timer::KSPACE);
        }

        if (n_pre_reverse) {
            modify->pre_reverse(eflag, vflag);
            timer->stamp(Timer::MODIFY);
        }

        // reverse communication of forces
        if (force->newton) {
            auto begin = std::chrono::high_resolution_clock::now();
            comm->reverse_comm();
            auto end = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count();
            lammps_comm_duration += duration;
            lammps_reverse_comm_duration += duration;
            lammps_reverse_comm_times.push_back(duration);
            timer->stamp(Timer::COMM);
        }

        // force modifications, final time integration, diagnostics
        if (n_post_force_any)
            modify->post_force(vflag);

        auto begin_m2 = std::chrono::high_resolution_clock::now();
        modify->final_integrate();
        auto end_m2 = std::chrono::high_resolution_clock::now();
        auto duration_m2 = std::chrono::duration_cast<std::chrono::microseconds>(end_m2 - begin_m2).count();
        lammps_modify_duration += duration_m2;
        if (n_end_of_step) {
            modify->end_of_step();
        }
        timer->stamp(Timer::MODIFY);

        // all output

        /*
        if (ntimestep == output->next) {
            timer->stamp();
            output->write(ntimestep);
            timer->stamp(Timer::OUTPUT);
        }
        */
    }

    auto end_lammps = std::chrono::high_resolution_clock::now();
    auto duration_lammps = std::chrono::duration_cast<std::chrono::microseconds>(end_lammps - begin_lammps).count();

    int64_t total_duration_lammps = 0;
    MPI_Allreduce(&duration_lammps, &total_duration_lammps, 1, MPI_INT64_T, MPI_SUM, world);

    std::cout << "lammps total just running the thing: " << duration_lammps << " microseconds. " << " total duration: " << total_duration_lammps << std::endl;

    int64_t total_comm_duration = 0;
    int64_t total_forward_comm_duration = 0;
    int64_t total_reverse_comm_duration = 0;

    int64_t total_compute_duration = 0;
    int64_t total_modify_duration = 0;
    int64_t total_modify_pre_force_duration = 0;


    MPI_Allreduce(&lammps_comm_duration, &total_comm_duration, 1, MPI_INT64_T, MPI_SUM, world);
    MPI_Allreduce(&lammps_forward_comm_duration, &total_forward_comm_duration, 1, MPI_INT64_T, MPI_SUM, world);
    MPI_Allreduce(&lammps_reverse_comm_duration, &total_reverse_comm_duration, 1, MPI_INT64_T, MPI_SUM, world);

    MPI_Allreduce(&lammps_compute_duration, &total_compute_duration, 1, MPI_INT64_T, MPI_SUM, world);
    MPI_Allreduce(&lammps_modify_duration, &total_modify_duration, 1, MPI_INT64_T, MPI_SUM, world);
    MPI_Allreduce(&lammps_modify_pre_force_duration, &total_modify_pre_force_duration, 1, MPI_INT64_T, MPI_SUM, world);

    if (comm->me == 0) {
        std::cout << GREEN << "process: " << comm->me
                  << " LAMMPS COMM DURATION: " << lammps_comm_duration << " forward: " << lammps_forward_comm_duration
                  << " reverse: " << lammps_reverse_comm_duration
                  << " microseconds. " << " total comm duration: " << total_comm_duration
                  << " total forward comm: " << total_forward_comm_duration << " total reverse comm: "
                  << total_reverse_comm_duration << RESET_COLOR << std::endl;

        std::cout << YELLOW
                  << "lammps compute duration: " << lammps_compute_duration
                  << " microseconds. " << " total compute duration: " << total_compute_duration << RESET_COLOR
                  << std::endl;

        std::cout << YELLOW
                  << "lammps modify duration: " << lammps_modify_duration << " pre force duration: "
                  << lammps_modify_pre_force_duration
                  << " microseconds. " << " total modify duration: " << total_modify_duration
                  << " total modify pre force duration: " << total_modify_pre_force_duration << RESET_COLOR
                  << std::endl;
    }

    /*
    if (comm->me == 0) {
        for (int dep = 0; dep < NUM_DEPS; dep++) {
            std::ostringstream stringStream;
            stringStream << "curr_dt_dep_" << dep;
            std::string copyOfStr = stringStream.str();
            std::ofstream f(copyOfStr);

            for (int k = 0; k < curr_dt_compute_dep_times_vec[dep].size(); k++) {
                if (k == lammps_forward_comm_times.size() - 1) {
                    f << curr_dt_compute_dep_times_vec[dep][k];
                } else {
                    f << curr_dt_compute_dep_times_vec[dep][k] << ",";
                }
            }

            f.close();
        }

        for (int dep = 0; dep < NUM_DEPS; dep++) {
            std::ostringstream stringStream;
            stringStream << "next_dt_dep_" << dep;
            std::string copyOfStr = stringStream.str();
            std::ofstream f(copyOfStr);

            for (int k = 0; k < next_dt_compute_dep_times_vec[dep].size(); k++) {
                if (k == lammps_forward_comm_times.size() - 1) {
                    f << next_dt_compute_dep_times_vec[dep][k];
                } else {
                    f << next_dt_compute_dep_times_vec[dep][k] << ",";
                }
            }

            f.close();
        }
    }
    */

    delete[] send_f;
    delete[] send_x;

    MPI_Barrier(world);
    if (ONLY_RUN_LAMMPS) {
        return;
    }

    // setup data structures to run stencil md
    // map dependency levels to number of zoids to wait on
    std::vector<int> dep_to_wait_idxs[NUM_DEPS];
    std::set<int> zoids_already_waiting_on;
    std::map<int, std::vector<int>> zoid_to_wait_idxs;

    for (int dep = 1; dep < NUM_DEPS; dep++) {
        int num_zoids = 0;
        std::vector<int> dep_recv_zoids;
        for (int i = 0; i < lmp->recv_from_neighbors_procs.size(); i++) {
            int recv_zoid_num = lmp->recv_from_neighbors_procs[i];
            if (recv_zoid_num % comm->nprocs != comm->me) {
                for (int j = 0; j < lmp->queues[dep].size(); j++) {
                    queue_info& zoid = lmp->queues[dep][j];
                    int zoid_num = zoid.num;
                    if (zoid_num % comm->nprocs == comm->me) {
                        auto& recv_from = lmp->recv_from_neighbors[zoid_num];
                        if (std::find(recv_from.begin(), recv_from.end(),
                                      recv_zoid_num) != recv_from.end() &&
                            zoids_already_waiting_on.find(recv_zoid_num) == zoids_already_waiting_on.end()) {
                            dep_to_wait_idxs[dep].push_back(i);
                            zoids_already_waiting_on.insert(recv_zoid_num);
                            dep_recv_zoids.push_back(recv_zoid_num);
                            zoid_to_wait_idxs[zoid_num].push_back(i);
                            break;
                        }
                    }
                }
            }
        }
    }

    std::cout << "------------------------------------------------------------------------" << std::endl;
    for (int dep = 0; dep < NUM_DEPS; dep++) {
        int num_directly_wait_on = 0;
        for (auto& idx : dep_to_wait_idxs[dep]) {
            int recv_zoid_num = lmp->recv_from_neighbors_procs[idx];
            // std::cout << "Wait on idx: " << idx << " zoid: " << recv_zoid_num << std::endl;

            if (get_zoid_dep(recv_zoid_num) == dep - 1 && (recv_zoid_num % comm->nprocs != comm->me)) {
                num_directly_wait_on++;
            }
        }

        int num_wait_on_normally = 0;
        for (int j = 0; j < lmp->queues[dep].size(); j++) {
            int zoid_num = lmp->queues[dep][j].num;
            if (zoid_num % comm->nprocs == comm->me) {
                for (auto& recv_from_zoid_num : lmp->recv_from_neighbors[zoid_num]) {
                    if (recv_from_zoid_num % comm->nprocs != comm->me) {
                        num_wait_on_normally++;
                    }
                }
            }
        }

        // std::cout << "me: " << comm->me << " dep: " << dep << " num wait on: " << dep_to_wait_idxs[dep].size() << " num directly wait on: " << num_directly_wait_on << " num wait on normally: " << num_wait_on_normally << std::endl;
    }

    /*
    for (int zoid_num = 0; zoid_num < NUM_ZOIDS; zoid_num++) {
        if (zoid_num % comm->nprocs == comm->me) {
            if (dep_to_wait_idxs[get_zoid_dep(zoid_num)].size() >= 9) {
                std::vector<int> tmp;
                for (int wait_idx : zoid_to_wait_idxs[zoid_num]) {
                    int wait_zoid = lmp->recv_from_neighbors_procs[wait_idx];
                    if (get_zoid_dep(wait_zoid) == get_zoid_dep(zoid_num) - 1) {
                        tmp.push_back(wait_idx);
                    }
                }

                std::cout << CYAN << "zoid num: " << zoid_num << " num directly wait on: " << tmp.size() << " num recv from: " << lmp->recv_from_neighbors[zoid_num].size() << RESET_COLOR << std::endl;
                for (int wait_idx : tmp) {
                    int wait_zoid = lmp->recv_from_neighbors_procs[wait_idx];
                    std::cout << BLUE << "me: " << comm->me << " zoid: " << zoid_num << " waiting on zoid: " << lmp->recv_from_neighbors_procs[wait_idx] << RESET_COLOR << std::endl;
                }
            }
        }
    }
    */

    MPI_Barrier(world);

    // map dependency levels to number of zoids to wait on
    std::vector<int> dep_to_wait_idxs_next_dt[NUM_DEPS];
    std::set<int> zoids_already_waiting_on_next_dt;
    std::map<int, std::vector<int>> zoid_to_wait_idxs_next_dt;

    for (int dep = 1; dep < NUM_DEPS; dep++) {
        int num_zoids = 0;
        std::vector<int> dep_recv_zoids;
        for (int i = 0; i < lmp->recv_from_neighbors_procs_next_dt.size(); i++) {
            int recv_zoid_num = lmp->recv_from_neighbors_procs_next_dt[i];
            if (recv_zoid_num % comm->nprocs != comm->me) {
                for (int j = 0; j < lmp->queues_next_dt[dep].size(); j++) {
                    queue_info& zoid = lmp->queues_next_dt[dep][j];
                    int zoid_num = zoid.num;
                    if (zoid_num % comm->nprocs == comm->me) {
                        auto& recv_from =
                                lmp->recv_from_neighbors_next_dt[zoid_num];
                        if (std::find(recv_from.begin(), recv_from.end(),
                                      recv_zoid_num) != recv_from.end() &&
                            zoids_already_waiting_on_next_dt.find(
                                    recv_zoid_num) ==
                            zoids_already_waiting_on_next_dt.end()) {
                            dep_to_wait_idxs_next_dt[dep].push_back(i);
                            zoids_already_waiting_on_next_dt.insert(
                                    recv_zoid_num);
                            dep_recv_zoids.push_back(recv_zoid_num);
                            zoid_to_wait_idxs_next_dt[zoid_num].push_back(i);
                            break;
                        }
                    }
                }
            }
        }
    }

    int zoid_num_to_num_procs[NUM_ZOIDS];
    for (int zoid_num = 0; zoid_num < NUM_ZOIDS; zoid_num++) {
        if (zoid_num % comm->nprocs == comm->me) {
            int num_procs = 0;
            for (int proc = 0; proc < comm->nprocs; proc++) {
                if (proc == comm->me) {
                    continue;
                }
                for (int i = 0; i < lmp->send_to_neighbors[zoid_num].size();
                     i++) {
                    if (lmp->send_to_neighbors[zoid_num][i] % comm->nprocs ==
                        proc) {
                        num_procs++;
                        break;
                    }
                }
            }
            assert(num_procs >= 0 && num_procs < comm->nprocs);
            zoid_num_to_num_procs[zoid_num] = num_procs;
        }
    }

    int zoid_num_to_num_procs_next_dt[NUM_ZOIDS];
    for (int zoid_num = 0; zoid_num < NUM_ZOIDS; zoid_num++) {
        if (zoid_num % comm->nprocs == comm->me) {
            int num_procs = 0;
            for (int proc = 0; proc < comm->nprocs; proc++) {
                if (proc == comm->me) {
                    continue;
                }
                for (int i = 0;
                     i < lmp->send_to_neighbors_next_dt[zoid_num].size(); i++) {
                    if (lmp->send_to_neighbors_next_dt[zoid_num][i] %
                        comm->nprocs ==
                        proc) {
                        num_procs++;
                        break;
                    }
                }
            }
            assert(num_procs >= 0 && num_procs < comm->nprocs);
            zoid_num_to_num_procs_next_dt[zoid_num] = num_procs;
        }
    }

    MPI_Barrier(world);

    auto begin = std::chrono::high_resolution_clock::now();
    for (int t = 0; t < n; t += 2 * NUM_TIMESTEPS_IN_PARALLEL) {
        if (USE_DEP_TO_WAIT_IDXS) {
            run_stencil_md(t, dep_to_wait_idxs, dep_to_wait_idxs_next_dt, zoid_num_to_num_procs, zoid_num_to_num_procs_next_dt,
                           test_f, test_x);
        } else {
            /*
            run_stencil_md(t, zoid_to_wait_idxs, zoid_to_wait_idxs_next_dt, zoid_num_to_num_procs, zoid_num_to_num_procs_next_dt,
                           test_f, test_x);
            */
        }
    }
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count();
    int64_t total_duration_stencil_md = 0;
    MPI_Allreduce(&duration, &total_duration_stencil_md, 1, MPI_INT64_T, MPI_SUM, world);

    std::cout << "me: " << comm->me << " stencil md total just running the thing: " << duration << " microseconds. " << " unpack duration? " << unpack_duration << " total duration: " << total_duration_stencil_md << std::endl;

    /*
    int64_t stencil_md_total_compute_time_curr_dt_dep[NUM_DEPS] = {0};
    int64_t stencil_md_total_compute_time_next_dt_dep[NUM_DEPS] = {0};

    int64_t stencil_md_total_num_atoms_curr_dt_dep[NUM_DEPS] = {0};
    int64_t stencil_md_total_num_atoms_next_dt_dep[NUM_DEPS] = {0};
    for (int dep = 0; dep < NUM_DEPS; dep++) {
        int64_t total_compute_time_curr_dt_dep = 0;
        int64_t total_compute_time_next_dt_dep = 0;

        MPI_Allreduce(&curr_dt_compute_dep_time[dep], &total_compute_time_curr_dt_dep, 1, MPI_INT64_T, MPI_SUM, world);
        MPI_Allreduce(&next_dt_compute_dep_time[dep], &total_compute_time_next_dt_dep, 1, MPI_INT64_T, MPI_SUM, world);

        stencil_md_total_compute_time_curr_dt_dep[dep] = total_compute_time_curr_dt_dep;
        stencil_md_total_compute_time_next_dt_dep[dep] = total_compute_time_next_dt_dep;

        int64_t total_num_atoms_curr_dt_dep = 0;
        int64_t total_num_atoms_next_dt_dep = 0;

        MPI_Allreduce(&curr_dt_num_atoms[dep], &total_num_atoms_curr_dt_dep, 1, MPI_INT64_T, MPI_SUM, world);
        MPI_Allreduce(&next_dt_num_atoms[dep], &total_num_atoms_next_dt_dep, 1, MPI_INT64_T, MPI_SUM, world);

        stencil_md_total_num_atoms_curr_dt_dep[dep] += total_num_atoms_curr_dt_dep;
        stencil_md_total_num_atoms_next_dt_dep[dep] += total_num_atoms_next_dt_dep;
    }

    if (comm->me == 0) {
        int64_t stencil_md_total_compute_time = 0;
        for (int dep = 0; dep < NUM_DEPS; dep++) {
            std::cout << CYAN << "CURR DT DEP: " << dep << " total time: " << stencil_md_total_compute_time_curr_dt_dep[dep]
                << " NUM ATOMS: " << stencil_md_total_num_atoms_curr_dt_dep[dep] << " ratio: " << (double) stencil_md_total_num_atoms_curr_dt_dep[dep] / stencil_md_total_compute_time_curr_dt_dep[dep] << RESET_COLOR << std::endl;

            stencil_md_total_compute_time += stencil_md_total_compute_time_curr_dt_dep[dep];
        }
        for (int dep = 0; dep < NUM_DEPS; dep++) {
            std::cout << CYAN << "NEXT DT DEP: " << dep << " total time: " << stencil_md_total_compute_time_next_dt_dep[dep]
                << " NUM ATOMS: " << stencil_md_total_num_atoms_next_dt_dep[dep] << " ratio: " << (double)  stencil_md_total_num_atoms_next_dt_dep[dep] / stencil_md_total_compute_time_next_dt_dep[dep] << RESET_COLOR << std::endl;
            stencil_md_total_compute_time += stencil_md_total_compute_time_next_dt_dep[dep];
        }


        std::cout << CYAN << " TOTAL COMPUTE TIME: " << stencil_md_total_compute_time << RESET_COLOR << std::endl;

        for (int dep = 0; dep < NUM_DEPS; dep++) {
            std::cout << GREEN << "ME: " << comm->me << " curr dt mean: " << mean(curr_dt_compute_dep_times_vec[dep]) << " sd: " << sd(curr_dt_compute_dep_times_vec[dep])
                      << " next dt mean: " << mean(next_dt_compute_dep_times_vec[dep]) << " sd: " << sd(next_dt_compute_dep_times_vec[dep]) << RESET_COLOR << std::endl;
        }
    }
    */

    int64_t stencil_md_total_send_comm_duration = 0;
    int64_t stencil_md_total_recv_comm_duration = 0;
    int64_t stencil_md_total_compute_duration = 0;
    int64_t stencil_md_total_modify_duration = 0;
    int64_t stencil_md_total_modify_pre_force_duration = 0;
    int64_t stencil_md_total_mpi_duration = 0;
    int64_t stencil_md_total_curr_dt_comm_duration = 0;
    int64_t stencil_md_total_next_dt_comm_duration = 0;
    int64_t stencil_md_total_send_pack_duration = 0;
    int64_t stencil_md_total_misc_duration = 0;
    int64_t stencil_md_total_pre_recv_time = 0;
    int64_t stencil_md_total_unpack_self_time = 0;

    // int64_t my_compute_duration = curr_dt_compute_duration + next_dt_compute_duration;

    MPI_Allreduce(&compute_duration, &stencil_md_total_compute_duration, 1, MPI_INT64_T, MPI_SUM, world);
    MPI_Allreduce(&modify_duration, &stencil_md_total_modify_duration, 1, MPI_INT64_T, MPI_SUM, world);
    MPI_Allreduce(&modify_pre_force_duration, &stencil_md_total_modify_pre_force_duration, 1, MPI_INT64_T, MPI_SUM, world);

    MPI_Allreduce(&mpi_duration, &stencil_md_total_mpi_duration, 1, MPI_INT64_T, MPI_SUM, world);
    MPI_Allreduce(&misc_time, &stencil_md_total_misc_duration, 1, MPI_INT64_T, MPI_SUM, world);
    MPI_Allreduce(&unpack_self_time, &stencil_md_total_unpack_self_time, 1, MPI_INT64_T, MPI_SUM, world);

    MPI_Allreduce(&pre_recv_time, &stencil_md_total_pre_recv_time, 1, MPI_INT64_T, MPI_SUM, world);
    // MPI_Allreduce(&curr_dt_comm_duration, &stencil_md_total_curr_dt_comm_duration, 1, MPI_INT64_T, MPI_SUM, world);
    // MPI_Allreduce(&next_dt_comm_duration, &stencil_md_total_next_dt_comm_duration, 1, MPI_INT64_T, MPI_SUM, world);

    MPI_Allreduce(&send_comm_duration, &stencil_md_total_send_comm_duration, 1, MPI_INT64_T, MPI_SUM, world);
    MPI_Allreduce(&recv_comm_duration, &stencil_md_total_recv_comm_duration, 1, MPI_INT64_T, MPI_SUM, world);

    MPI_Allreduce(&send_pack_duration, &stencil_md_total_send_pack_duration, 1, MPI_INT64_T, MPI_SUM, world);

    if (comm->me == 0) {
        std::cout << GREEN << "process: " << comm->me << " STENCIL MD LOCAL COMM DURATION: " << send_comm_duration + recv_comm_duration
                  << " LOCAL COMPUTE: " << compute_duration
                  << " TOTAL COMPUTE: " << stencil_md_total_compute_duration
                  << " TOTAL COMM: " << stencil_md_total_send_comm_duration + stencil_md_total_recv_comm_duration
                  << " TOTAL SEND COMM: " << stencil_md_total_send_comm_duration << " TOTAL RECV COMM: " << stencil_md_total_recv_comm_duration
                  << " TOTAL MODIFY: " << stencil_md_total_modify_duration << " TOTAL MODIFY PRE FORCE: " << stencil_md_total_modify_pre_force_duration
                  << " TOTAL MPI DURATION: " << stencil_md_total_mpi_duration
                  << " TOTAL SEND PACK DURATION: " << stencil_md_total_send_pack_duration
                  << " TOTAL MISC DURATION: " << stencil_md_total_misc_duration
                  << " TOTAL PRE-RECV TIME: " << stencil_md_total_pre_recv_time
                  << " TOTAL UNPACK SELF TIME: " << stencil_md_total_unpack_self_time << RESET_COLOR << std::endl;

        std::cout << YELLOW << "CURR DT TOTAL COMM DURATION: " << stencil_md_total_curr_dt_comm_duration
                  << " NEXT DT COMM DURATION: " << stencil_md_total_next_dt_comm_duration << RESET_COLOR << std::endl;
    }

    for (int i = 0; i < test_num_timesteps; i++) {
        delete[] test_f[i];
        delete[] test_x[i];
    }
}

// assume already have all the data necessary to run the zoid
template <bool curr_dt>
void Verlet::run_stencil_md_zoid(int starting_timestep, int zoid_num, int num_procs, double** test_f, double** test_x) {
    int n_pre_force = modify->n_pre_force;
    int n_post_force_any = modify->n_post_force_any;
    int n_end_of_step = modify->n_end_of_step;

    queue_info& zoid = curr_dt ? lmp->zoid_num_to_zoid[zoid_num] : lmp->zoid_num_to_zoid_next_dt[zoid_num];
    auto& atom_arr = lmp->atom_stencil_md[zoid_num];
    int** atom_idx_mapping = zoid.atom_idx_mapping;

    auto begin_m = std::chrono::high_resolution_clock::now();
    cilk_for (int t = 0; t < NUM_TIMESTEPS_IN_PARALLEL; t++) {
        Atom* atom_next_timestep = curr_dt ? atom_arr[t + 1] : atom_arr[NUM_TIMESTEPS_IN_PARALLEL - t - 1];
        Modify* modify_ = curr_dt ? lmp->modify_stencil_md_omp[zoid_num][t + 1] : lmp->modify_stencil_md_omp[zoid_num][NUM_TIMESTEPS_IN_PARALLEL - t - 1];
        modify_->pre_force_stencil_md(vflag, atom_next_timestep);
    }
    auto end_m = std::chrono::high_resolution_clock::now();
    auto duration_m = std::chrono::duration_cast<std::chrono::microseconds>(end_m - begin_m).count();
    modify_pre_force_duration_cilk += duration_m;

    for (int t = 0; t < NUM_TIMESTEPS_IN_PARALLEL; t++) {
        Atom* atom_ = curr_dt ? atom_arr[t] : atom_arr[NUM_TIMESTEPS_IN_PARALLEL - t];
        Atom* atom_next_timestep = curr_dt ? atom_arr[t + 1] : atom_arr[NUM_TIMESTEPS_IN_PARALLEL - t - 1];
#ifdef LMP_OPENMP
        Modify* modify_ = curr_dt ? lmp->modify_stencil_md_omp[zoid_num][t + 1] : lmp->modify_stencil_md_omp[zoid_num][NUM_TIMESTEPS_IN_PARALLEL - t - 1];
#else
        Modify* modify_ = lmp->modify_stencil_md[zoid_num];
#endif
        if (TEST_AGAINST_LAMMPS_LOCAL) {
            int timestep_to_compare_against = curr_dt ? starting_timestep + t : starting_timestep + NUM_TIMESTEPS_IN_PARALLEL + t;
            for (int k = 0; k < atom_->nlocal; k++) {
                int tag = atom_->tag[k];
                double* x_ = atom_->x[k];
                for (int dim = 0; dim < 3; dim++) {
                    double val = x_[dim];
                    if (val < 0) {
                        val += domain->prd[dim];
                    } else if (val >= domain->prd[dim]) {
                        val -= domain->prd[dim];
                    }

                    double test_val = test_x[timestep_to_compare_against][tag * 3 + dim];
                    if (test_val < 0) {
                        test_val += domain->prd[dim];
                    } else if (test_val >= domain->prd[dim]) {
                        test_val -= domain->prd[dim];
                    }

                    if (fabs(val - test_val) > 1e-6) {
                        if (curr_dt) {
                            std::cout << "-------POS DIFF--------"
                                      << std::endl;
                        } else {
                            std::cout << "-------NEXT DT POS DIFF--------"
                                      << std::endl;
                        }
                        std::cout << "idx: " << k
                                  << " out of: " << atom_->nlocal
                                  << std::endl;
                        std::cout
                                << "Dim: " << dim << " Zoid: " << zoid_num
                                << " timestep: " << timestep_to_compare_against << " tag: " << tag
                                << " different. " << std::endl;
                        std::cout << "What I have: " << x_[0] << " "
                                  << x_[1] << " " << x_[2] << std::endl;
                        std::cout << "What does LAMMPS have? "
                                  << test_x[t][tag * 3 + 0] << " "
                                  << test_x[t][tag * 3 + 1] << " "
                                  << test_x[t][tag * 3 + 2]
                                  << std::endl;
                        std::cout
                                << "Diff: "
                                << fabs(x_[dim] - test_x[t][tag * 3 + dim])
                                << std::endl;
                        std::cout << "pos: " << atom_->x[k][0] << " "
                                  << atom_->x[k][1] << " "
                                  << atom_->x[k][2] << std::endl;

                        for (int tmp = 0; tmp < 3; tmp++) {
                            std::cout
                                    << "lo: "
                                    << zoid.zoid.cuts[tmp].lower +
                                       zoid.zoid.cuts[tmp].slope_lower *
                                       t
                                    << std::endl;
                            std::cout
                                    << "hi: "
                                    << zoid.zoid.cuts[tmp].upper +
                                       zoid.zoid.cuts[tmp].slope_upper *
                                       t
                                    << std::endl;
                        }
                        assert(false);
                    }
                }
            }

            for (int k = 0; k < atom_->nlocal; k++) {
                // compare forces on local atoms?
                int tag = atom_->tag[k];
                for (int dim = 0; dim < 3; dim++) {
                    double my_force = atom_->f[k][dim] +
                                      atom_->eval_f_stencil_md[k][dim];
                    if (fabs(my_force - test_f[timestep_to_compare_against][tag * 3 + dim]) >
                        1e-6) {
                        if (curr_dt) {
                            std::cout << "------FORCE DIFF--------"
                                      << std::endl;
                        } else {
                            std::cout << "------NEXT DT FORCE DIFF--------"
                                      << std::endl;
                        }
                        std::cout << "idx: " << k
                                  << " out of: " << atom_->nlocal
                                  << std::endl;
                        std::cout
                                << "Dim: " << dim << " Zoid: " << zoid_num
                                << " timestep: " << timestep_to_compare_against << " tag: " << tag
                                << " different. " << std::endl;
                        std::cout << "what I have f: " << atom_->f[k][0]
                                  << " " << atom_->f[k][1] << " "
                                  << atom_->f[k][2] << std::endl;
                        std::cout
                                << "what I have eval "
                                << atom_->eval_f_stencil_md[k][0] << " "
                                << atom_->eval_f_stencil_md[k][1] << " "
                                << atom_->eval_f_stencil_md[k][2]
                                << std::endl;
                        std::cout << "what I have: "
                                  << atom_->f[k][0] +
                                     atom_->eval_f_stencil_md[k][0]
                                  << " "
                                  << atom_->f[k][1] +
                                     atom_->eval_f_stencil_md[k][1]
                                  << " "
                                  << atom_->f[k][2] +
                                     atom_->eval_f_stencil_md[k][2]
                                  << std::endl;
                        std::cout << "What does LAMMPS have? "
                                  << test_f[t][tag * 3 + 0] << " "
                                  << test_f[t][tag * 3 + 1] << " "
                                  << test_f[t][tag * 3 + 2]
                                  << std::endl;
                        std::cout
                                << "Diff: "
                                << fabs(my_force - test_f[t][tag * 3 + dim])
                                << std::endl;
                        std::cout << "pos: " << atom_->x[k][0] << " "
                                  << atom_->x[k][1] << " "
                                  << atom_->x[k][2] << std::endl;

                        for (int tmp = 0; tmp < 3; tmp++) {
                            std::cout
                                    << "lo: "
                                    << zoid.zoid.cuts[tmp].lower +
                                       zoid.zoid.cuts[tmp].slope_lower *
                                       t
                                    << std::endl;
                            std::cout
                                    << "hi: "
                                    << zoid.zoid.cuts[tmp].upper +
                                       zoid.zoid.cuts[tmp].slope_upper *
                                       t
                                    << std::endl;
                        }

                        assert(false);
                    }
                }
            }
        }

        auto begin_m = std::chrono::high_resolution_clock::now();
        // updates positions in atom_next_timestep
        modify_->initial_integrate_stencil_md(
                vflag, atom_, atom_next_timestep, atom_idx_mapping[t],
                zoid.can_eval_pos[t]);
        auto end_m = std::chrono::high_resolution_clock::now();
        auto duration_m = std::chrono::duration_cast<std::chrono::microseconds>(end_m - begin_m).count();
        modify_duration_cilk += duration_m;

        if (TEST_AGAINST_LAMMPS_LOCAL) {
            int timestep_to_compare_against = curr_dt ? starting_timestep + t + 1 : starting_timestep + NUM_TIMESTEPS_IN_PARALLEL + t + 1;
            for (int k = 0; k < atom_next_timestep->nlocal; k++) {
                int tag = atom_next_timestep->tag[k];
                double* x_ = atom_next_timestep->x[k];
                for (int dim = 0; dim < 3; dim++) {
                    double val = x_[dim];
                    if (val < 0) {
                        val += domain->prd[dim];
                    } else if (val >= domain->prd[dim]) {
                        val -= domain->prd[dim];
                    }

                    double test_val = test_x[timestep_to_compare_against][tag * 3 + dim];
                    if (test_val < 0) {
                        test_val += domain->prd[dim];
                    } else if (test_val >= domain->prd[dim]) {
                        test_val -= domain->prd[dim];
                    }

                    if (fabs(val - test_val) > 1e-6) {
                        std::cout << "-------POS DIFF NEXT--------"
                                  << std::endl;
                        std::cout
                                << "idx: " << k
                                << " out of: " << atom_next_timestep->nlocal
                                << std::endl;
                        std::cout
                                << "Dim: " << dim << " Zoid: " << zoid_num
                                << " timestep: " << timestep_to_compare_against << " tag: " << tag
                                << " different. " << std::endl;
                        std::cout << "What I have: " << x_[0] << " "
                                  << x_[1] << " " << x_[2] << std::endl;
                        std::cout << "What does LAMMPS have? "
                                  << test_x[t + 1][tag * 3 + 0] << " "
                                  << test_x[t + 1][tag * 3 + 1] << " "
                                  << test_x[t + 1][tag * 3 + 2]
                                  << std::endl;
                        std::cout << "Diff: " << fabs(val - test_val)
                                  << std::endl;
                        std::cout << "me val: " << val
                                  << " test_val: " << test_val
                                  << std::endl;
                        std::cout
                                << "pos: " << atom_next_timestep->x[k][0]
                                << " " << atom_next_timestep->x[k][1] << " "
                                << atom_next_timestep->x[k][2] << std::endl;
                        std::cout
                                << "can eval center? "
                                << zoid.can_eval_center[t + 1][k]
                                << " relevant? "
                                << (zoid.relevant_atom_idxs[t + 1].find(
                                        k) !=
                                    zoid.relevant_atom_idxs[t + 1].end())
                                << std::endl;
                        if (atom_->tag_to_idx.count(
                                atom_next_timestep->tag[k])) {
                            int prev_idx =
                                    atom_->tag_to_idx[atom_next_timestep
                                            ->tag[k]];
                            std::cout
                                    << "prev nlocal: " << atom_->nlocal
                                    << " prev idx: " << prev_idx
                                    << " prev can eval center? "
                                    << zoid.can_eval_center[t][prev_idx]
                                    << " relevant? "
                                    << (zoid.relevant_atom_idxs[t].find(
                                            prev_idx) !=
                                        zoid.relevant_atom_idxs[t].end())
                                    << std::endl;
                        } else {
                            std::cout << "could not find prev tag"
                                      << std::endl;
                        }

                        for (int tmp = 0; tmp < 3; tmp++) {
                            std::cout
                                    << "lo: "
                                    << zoid.zoid.cuts[tmp].lower +
                                       zoid.zoid.cuts[tmp].slope_lower *
                                       (t + 1)
                                    << std::endl;
                            std::cout
                                    << "hi: "
                                    << zoid.zoid.cuts[tmp].upper +
                                       zoid.zoid.cuts[tmp].slope_upper *
                                       (t + 1)
                                    << std::endl;
                        }
                        assert(false);
                    }
                }
            }
        }

        if (n_pre_force) {
            // auto begin_m = std::chrono::high_resolution_clock::now();
            // modify_->pre_force_stencil_md(vflag, atom_next_timestep);
            // auto end_m = std::chrono::high_resolution_clock::now();
            // auto duration_m = std::chrono::duration_cast<std::chrono::microseconds>(end_m - begin_m).count();
            // modify_pre_force_duration_cilk[__cilkrts_get_worker_number()] += duration_m;
            // modify_pre_force_duration += duration_m;
            // modify_duration += duration_m;
            // timer->stamp(Timer::MODIFY);
        }

        if (pair_compute_flag) {
            Force* next_force;
            if (!PURELY_LOCAL_POTENTIAL) {
                next_force = curr_dt ? lmp->force_stencil_md[zoid_num][t + 1] : lmp->force_stencil_md[zoid_num][NUM_TIMESTEPS_IN_PARALLEL - t - 1];
            } else {
                next_force = curr_dt ? lmp->force_stencil_md[zoid_num][t + 1] : lmp->force_stencil_md_next_dt[zoid_num][t + 1];
            }

            int* atom_idx_mapping_ = zoid.atom_idx_mapping[t + 1];
            auto begin = std::chrono::high_resolution_clock::now();
            int timestep = t + 1;
            next_force->pair->compute_stencil_md(
                    eflag, vflag, atom_next_timestep,
                    zoid.can_eval_center[t + 1],
                    zoid, &timestep);
            auto end = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count();
            compute_duration_cilk += duration;
            /*
            if (curr_dt) {
                // curr_dt_compute_dep_time[get_zoid_dep(zoid_num)] += duration;
                // curr_dt_num_atoms[get_zoid_dep(zoid_num)] += atom_next_timestep->nlocal;
                // curr_dt_compute_dep_times_vec[get_zoid_dep(zoid_num)].push_back(duration);
            } else {
                next_dt_compute_dep_time[get_zoid_dep_next_dt(zoid_num)] += duration;
                next_dt_num_atoms[get_zoid_dep_next_dt(zoid_num)] += atom_next_timestep->nlocal;
                next_dt_compute_dep_times_vec[get_zoid_dep_next_dt(zoid_num)].push_back(duration);
            }
            */
        }

        // reverse communication of forces
        if (force->newton) {
            /*
              std::cout << "Force newton on" << std::endl;
              Kokkos::fence();
              comm->reverse_comm();
              timer->stamp(Timer::COMM);
            */
        }

        // force modifications, final time integration, diagnostics

        if (n_post_force_any) {
            assert(false);
            modify->post_force(vflag);
        }

        auto begin_m2 = std::chrono::high_resolution_clock::now();
        modify_->final_integrate_stencil_md(
                atom_, atom_next_timestep, neighbor, atom_idx_mapping[t],
                zoid.can_eval_pos[t + 1]);
        auto end_m2 = std::chrono::high_resolution_clock::now();
        auto duration_m2 = std::chrono::duration_cast<std::chrono::microseconds>(end_m2 - begin_m2).count();
        // modify_duration += duration_m2;
        modify_duration_cilk += duration_m2;

        if (n_end_of_step) {
            assert(false);
            modify->end_of_step();
        }
        // timer->stamp(Timer::MODIFY);

        // all output

        /*
        if (ntimestep == output->next) {
            assert(false);
            timer->stamp();
            output->write(ntimestep);
            timer->stamp(Timer::OUTPUT);
        }
        */
    }
}

void Verlet::run_stencil_md(int starting_timestep, std::vector<int>* dep_to_wait_idxs, std::vector<int>* dep_to_wait_idxs_next_dt,
                            int* zoid_num_to_num_procs, int* zoid_num_to_num_procs_next_dt,
                            double** test_f, double** test_x) {
    eflag = 0; vflag = 0;
    bigint ntimestep;
    int nflag, sortflag;

    int n_post_integrate = modify->n_post_integrate;
    int n_pre_exchange = modify->n_pre_exchange;
    int n_pre_neighbor = modify->n_pre_neighbor;
    int n_post_neighbor = modify->n_post_neighbor;
    int n_pre_force = modify->n_pre_force;
    int n_pre_reverse = modify->n_pre_reverse;
    int n_post_force_any = modify->n_post_force_any;
    int n_end_of_step = modify->n_end_of_step;

    if (atom->sortfreq > 0)
        sortflag = 1;
    else
        sortflag = 0;

    // change the positions of atoms for debugging purposes
    /*
    if (TEST_AGAINST_LAMMPS_LOCAL) {
        for (int dep = 0; dep < NUM_DEPS; dep++) {
            for (int j = 0; j < lmp->queues[dep].size(); j++) {
                queue_info &zoid = lmp->queues[dep][j];
                int zoid_num = zoid.num;
                if (zoid_num % comm->nprocs == comm->me) {
                    for (int t = 1; t < NUM_TIMESTEPS_IN_PARALLEL + 1; t++) {
                        Atom *atom_ = lmp->atom_stencil_md[zoid_num][t];
                        zoid.debug_atom_pos[t] =
                                new double[(atom_->nlocal + atom_->nghost) * 3];
                        for (int k = 0; k < atom_->nlocal + atom_->nghost; k++) {
                            for (int dim = 0; dim < 3; dim++) {
                                zoid.debug_atom_pos[t][k * 3 + dim] = atom_->x[k][dim];
                                atom_->x[k][dim] = -1000;
                            }
                        }
                    }
                }
            }
        }
    }
    */

    /*
    std::vector<int> curr_dt_tmp;
    std::vector<int> next_dt_tmp;
    for (auto& x : lmp->recv_from_neighbors_procs) {
        if (x % comm->nprocs != comm->me) {
            curr_dt_tmp.push_back(x);
        }
    }
    for (auto& x : lmp->recv_from_neighbors_procs_next_dt) {
        if (x % comm->nprocs != comm->me) {
            next_dt_tmp.push_back(x);
        }
    }
    std::cout << "ME: " << comm->me << " NUM RECV LAUNCH CURR DT: " << lmp->recv_from_neighbors_procs.size() << " NEXT DT: " << lmp->recv_from_neighbors_procs_next_dt.size()
        << " CURR DT FILTERED SIZE: " << curr_dt_tmp.size() << " NEXT DT FILTERED SIZE: " << next_dt_tmp.size() << std::endl;
    */

    MPI_Barrier(world);

    int num_zoids_recv_from = lmp->recv_from_neighbors_procs.size();
    std::vector<std::future<void>> receive_request_futures;
    std::vector<MPI_Request> receive_requests(
        lmp->recv_from_neighbors_procs.size(), MPI_REQUEST_NULL);

    /*
    for (int i = 0; i < lmp->recv_from_neighbors_procs.size(); i++) {
        int recv_zoid_num = lmp->recv_from_neighbors_procs[i];
        if (recv_zoid_num % comm->nprocs != comm->me) {
            comm->receive_data_process_stencil_md(true, &receive_requests[i], recv_zoid_num, false);
        }
    }
    */

    auto begin_r = std::chrono::high_resolution_clock::now();
    int recv_idx = 0;
    for (int dep = 0; dep < NUM_DEPS; dep++) {
        for (int idx : dep_to_wait_idxs[dep]) {
            int recv_zoid_num = lmp->recv_from_neighbors_procs[idx];
            comm->receive_data_process_stencil_md(true, &receive_requests[recv_idx++], recv_zoid_num, false);
        }
    }

    // start compute
    std::vector<MPI_Request> send_requests[NUM_ZOIDS];
    std::vector<std::future<void>> send_request_threads;

    for (int zoid_num = 0; zoid_num < NUM_ZOIDS; zoid_num++) {
        if (zoid_num % comm->nprocs == comm->me) {
            // send_requests[zoid_num] = std::move(std::vector<MPI_Request>(zoid_num_to_num_procs[zoid_num], MPI_REQUEST_NULL));
            send_requests[zoid_num] = std::move(std::vector<MPI_Request>(comm->nprocs, MPI_REQUEST_NULL));
        }
    }

    auto end_r = std::chrono::high_resolution_clock::now();
    auto duration_r = std::chrono::duration_cast<std::chrono::microseconds>(end_r - begin_r).count();
    pre_recv_time += duration_r;

    /*
    for (int dep = 0; dep < NUM_DEPS; dep++) {
        std::stringstream curr_dt_debug;
        for (auto& idx: dep_to_wait_idxs[dep]) {
            curr_dt_debug << lmp->recv_from_neighbors_procs[idx] << " ";
        }
        std::stringstream next_dt_debug;
        for (auto& idx: dep_to_wait_idxs_next_dt[dep]) {
            next_dt_debug << lmp->recv_from_neighbors_procs_next_dt[idx] << " ";
        }
        std::cout << YELLOW << "curr dt dep: " << dep << " size: " << dep_to_wait_idxs[dep].size() << " wait zoids: " << curr_dt_debug.str() << RESET_COLOR << std::endl;
        std::cout << GREEN << "next dt dep: " << dep << " size: " << dep_to_wait_idxs_next_dt[dep].size() << " wait zoids: " << next_dt_debug.str() << RESET_COLOR << std::endl;
    }
    */

    int running_recv_idx = 0;
    for (int dep = 0; dep < NUM_DEPS; dep++) {
        // start compute curr dt

        if (USE_DEP_TO_WAIT_IDXS) {
            if (dep > 0) {
                auto begin = std::chrono::high_resolution_clock::now();

                auto begin_mpi = std::chrono::high_resolution_clock::now();
                MPI_Waitall(dep_to_wait_idxs[dep].size(), &receive_requests[running_recv_idx], MPI_STATUSES_IGNORE);
                running_recv_idx += dep_to_wait_idxs[dep].size();
                auto end_mpi = std::chrono::high_resolution_clock::now();
                auto duration_mpi = std::chrono::duration_cast<std::chrono::microseconds>(
                        end_mpi - begin_mpi).count();
                mpi_duration += duration_mpi;

                for (int idx: dep_to_wait_idxs[dep]) {
                // cilk_for (int i = 0 ; i < dep_to_wait_idxs[dep].size(); i++) {
                    // int idx = dep_to_wait_idxs[dep][i];
                    int recv_zoid_num = lmp->recv_from_neighbors_procs[idx];
                    /*
                    MPI_Wait(&receive_requests[idx], MPI_STATUS_IGNORE);
                    */
                    auto begin_unpack = std::chrono::high_resolution_clock::now();
                    comm->unpack_data_process_stencil_md(true, recv_zoid_num, false);
                    auto end_unpack = std::chrono::high_resolution_clock::now();
                    auto duration_unpack = std::chrono::duration_cast<std::chrono::microseconds>(
                            end_unpack - begin_unpack).count();
                    unpack_duration += duration_unpack;
                }

                auto end = std::chrono::high_resolution_clock::now();
                auto duration =
                        std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count();
                recv_comm_duration += duration;
            }
        }

        compute_duration_cilk = 0;
        modify_duration_cilk = 0;
        modify_pre_force_duration_cilk = 0;
        send_pack_duration_cilk = 0;

        cilk_for (int j = 0; j < lmp->queues[dep].size(); j++) {
        // for (int j = 0; j < lmp->queues[dep].size(); j++) {
            queue_info& zoid = lmp->queues[dep][j];
            int zoid_num = zoid.num;
            if (zoid_num % comm->nprocs != comm->me) {
                continue;
            }

            /*
            if (!USE_DEP_TO_WAIT_IDXS) {
                auto begin = std::chrono::high_resolution_clock::now();
                for (int idx: dep_to_wait_idxs[zoid_num]) {
                    int recv_zoid_num = lmp->recv_from_neighbors_procs[idx];
                    auto begin_mpi = std::chrono::high_resolution_clock::now();
                    MPI_Wait(&receive_requests[idx], MPI_STATUS_IGNORE);
                    auto end_mpi = std::chrono::high_resolution_clock::now();
                    auto duration_mpi = std::chrono::duration_cast<std::chrono::microseconds>(
                            end_mpi - begin_mpi).count();
                    // mpi_duration += duration_mpi;
                    auto begin_unpack = std::chrono::high_resolution_clock::now();
                    comm->unpack_data_process_stencil_md(true, recv_zoid_num, false);
                    auto end_unpack = std::chrono::high_resolution_clock::now();
                    auto duration_unpack = std::chrono::duration_cast<std::chrono::microseconds>(
                            end_unpack - begin_unpack).count();
                    // unpack_duration += duration_unpack;
                    curr_dt_wait_dep[get_zoid_dep(recv_zoid_num)] += duration_mpi;
                }

                auto end = std::chrono::high_resolution_clock::now();
                auto duration =
                        std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count();
                // recv_comm_duration += duration;
                // curr_dt_comm_duration += duration;
                // curr_dt_dep_time[dep] += duration;
            }
            */

            auto& atom_arr = lmp->atom_stencil_md[zoid_num];
            int** atom_idx_mapping = lmp->queues[dep][j].atom_idx_mapping;

            run_stencil_md_zoid<true>(starting_timestep, zoid_num, -1, test_f, test_x);

            if (dep < NUM_DEPS - 1) {
                auto begin = std::chrono::high_resolution_clock::now();
                Comm* comm_ = lmp->comm_stencil_md[zoid_num];

                int vec_idx = 0;
                cilk_for (int proc = 0; proc < comm->nprocs; proc++) {
                // for (int proc = 0; proc < comm->nprocs; proc++) {
                    /*
                    bool sent = comm_->send_data_to_process_stencil_md(true,
                        atom_arr, lmp->zoid_num_to_zoid[zoid_num],
                        &send_requests[zoid_num][vec_idx], proc, false, &send_pack_duration);
                    if (sent) {
                        vec_idx++;
                    }
                    */
                    comm_->pack_data_to_process_stencil_md(true,
                                                           lmp->atom_stencil_md[zoid_num],
                                                           zoid, proc, false);
                }
                auto end = std::chrono::high_resolution_clock::now();
                auto duration =
                        std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count();
                send_pack_duration_cilk += duration;
            }
        }

        modify_duration += modify_duration_cilk / SIZES[dep];
        compute_duration += compute_duration_cilk / SIZES[dep];
        modify_pre_force_duration += modify_pre_force_duration_cilk / SIZES[dep];
        send_pack_duration += send_pack_duration_cilk / SIZES[dep];

        if (dep < NUM_DEPS - 1) {
            for (int j = 0; j < lmp->queues[dep].size(); j++) {
                queue_info& zoid = lmp->queues[dep][j];
                int zoid_num = zoid.num;
                if (zoid_num % comm->nprocs != comm->me) {
                    continue;
                }
                auto begin = std::chrono::high_resolution_clock::now();
                Comm* comm_ = lmp->comm_stencil_md[zoid_num];
                int vec_idx = 0;
                for (int proc = 0; proc < comm->nprocs; proc++) {
                    if (proc != comm->me
                        && lmp->send_to_neighbors_procs[zoid_num].find(proc) != lmp->send_to_neighbors_procs[zoid_num].end()) {
                        bool sent = comm_->send_packed_data_to_process_stencil_md(true, zoid, &send_requests[zoid_num][proc], proc);
                        /*
                        assert(sent);
                        if (sent) {
                            vec_idx++;
                        }
                        */
                    }
                }
                auto end = std::chrono::high_resolution_clock::now();
                auto duration =
                        std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count();
                send_comm_duration += duration;

                auto begin2 = std::chrono::high_resolution_clock::now();
                if (lmp->send_to_neighbors_procs[zoid_num].find(comm->me) != lmp->send_to_neighbors_procs[zoid_num].end()) {
                    comm_->send_packed_data_to_process_stencil_md(true, zoid, nullptr, comm->me);
                }
                auto end2 = std::chrono::high_resolution_clock::now();
                auto duration2 =
                        std::chrono::duration_cast<std::chrono::microseconds>(end2 - begin2).count();
                unpack_self_time += duration2;
            }
        }
    }

    // clear force on everything except last timestep of initial,
    auto begin_misc = std::chrono::high_resolution_clock::now();
    cilk_for (int i = 0; i < NUM_ZOIDS; i++) {
        if (i % comm->nprocs == comm->me) {
            cilk_for (int t = 0; t < NUM_TIMESTEPS_IN_PARALLEL; t++) {
                Atom* atom_ = lmp->atom_stencil_md[i][t];
                int nall = atom_->nlocal + atom_->nghost;
                memset(&atom_->f[0][0], 0, (nall) * 3 * sizeof(double));
                memset(&atom_->eval_f_stencil_md[0][0], 0, (nall) * 3 * sizeof(double));
            }
        }
    }

    /*
    if (TEST_AGAINST_LAMMPS_LOCAL) {
        // change the positions of atoms for debugging purposes
        for (int zoid_num = 0; zoid_num < NUM_ZOIDS; zoid_num++) {
            if (zoid_num % comm->nprocs == comm->me) {
                queue_info& zoid = lmp->zoid_num_to_zoid_next_dt[zoid_num];
                for (int t = 1; t < NUM_TIMESTEPS_IN_PARALLEL + 1; t++) {
                    Atom* atom_ = lmp->atom_stencil_md[zoid_num][NUM_TIMESTEPS_IN_PARALLEL - t];
                    zoid.debug_atom_pos[t] =
                            new double[(atom_->nlocal + atom_->nghost) * 3];
                    for (int k = 0; k < atom_->nlocal + atom_->nghost; k++) {
                        for (int dim = 0; dim < 3; dim++) {
                            zoid.debug_atom_pos[t][k * 3 + dim] = atom_->x[k][dim];
                            atom_->x[k][dim] = -1000;
                        }
                    }
                }
            }
        }
    }
    */

    for (int i = 0; i < NUM_ZOIDS; i++) {
        if (send_requests[i].size() > 0) {
            MPI_Waitall(send_requests[i].size(), send_requests[i].data(), MPI_STATUSES_IGNORE);
        }
    }

    auto end_misc = std::chrono::high_resolution_clock::now();
    auto duration_misc =
            std::chrono::duration_cast<std::chrono::microseconds>(end_misc - begin_misc).count();
    misc_time += duration_misc;

    int num_zoids_recv_from_next_dt = lmp->recv_from_neighbors_procs_next_dt.size();
    std::vector<std::future<void>> receive_request_futures_next_dt;
    std::vector<MPI_Request> receive_requests_next_dt(
        num_zoids_recv_from_next_dt, MPI_REQUEST_NULL);

    /*
    for (int i = 0; i < lmp->recv_from_neighbors_procs_next_dt.size(); i++) {
        int recv_zoid_num = lmp->recv_from_neighbors_procs_next_dt[i];
        if (recv_zoid_num % comm->nprocs != comm->me) {
            comm->receive_data_process_stencil_md(false,
                                                  &receive_requests_next_dt[i], recv_zoid_num, false);
        }
    }
    */

    begin_r = std::chrono::high_resolution_clock::now();
    recv_idx = 0;
    for (int dep = 0; dep < NUM_DEPS; dep++) {
        for (int idx : dep_to_wait_idxs_next_dt[dep]) {
            int recv_zoid_num = lmp->recv_from_neighbors_procs_next_dt[idx];
            comm->receive_data_process_stencil_md(false, &receive_requests_next_dt[recv_idx++], recv_zoid_num, false);
        }
    }

    // start compute
    std::vector<MPI_Request> send_requests_next_dt[NUM_ZOIDS];
    std::vector<std::future<void>> send_request_threads_next_dt;
    for (int zoid_num = 0; zoid_num < NUM_ZOIDS; zoid_num++) {
        if (zoid_num % comm->nprocs == comm->me) {
            // send_requests_next_dt[zoid_num] = std::move(std::vector<MPI_Request>(zoid_num_to_num_procs_next_dt[zoid_num], MPI_REQUEST_NULL));
            send_requests_next_dt[zoid_num] = std::move(std::vector<MPI_Request>(comm->nprocs, MPI_REQUEST_NULL));
        }
    }

    end_r = std::chrono::high_resolution_clock::now();
    duration_r = std::chrono::duration_cast<std::chrono::microseconds>(end_r - begin_r).count();
    pre_recv_time += duration_r;

    running_recv_idx = 0;
    // start compute
    for (int dep = 0; dep < NUM_DEPS; dep++) {
        if (dep > 0) {
            if (USE_DEP_TO_WAIT_IDXS) {
                auto begin = std::chrono::high_resolution_clock::now();

                auto begin_mpi = std::chrono::high_resolution_clock::now();
                MPI_Waitall(dep_to_wait_idxs_next_dt[dep].size(), &receive_requests_next_dt[running_recv_idx], MPI_STATUSES_IGNORE);
                running_recv_idx += dep_to_wait_idxs_next_dt[dep].size();
                auto end_mpi = std::chrono::high_resolution_clock::now();
                auto duration_mpi = std::chrono::duration_cast<std::chrono::microseconds>(
                        end_mpi - begin_mpi).count();
                mpi_duration += duration_mpi;

                for (int idx: dep_to_wait_idxs_next_dt[dep]) {
                // cilk_for (int i = 0; i < dep_to_wait_idxs_next_dt[dep].size(); i++) {
                    // int idx = dep_to_wait_idxs_next_dt[dep][i];
                    int recv_zoid_num = lmp->recv_from_neighbors_procs_next_dt[idx];
                    /*
                    auto begin_mpi = std::chrono::high_resolution_clock::now();
                    MPI_Wait(&receive_requests_next_dt[idx], MPI_STATUS_IGNORE);
                    auto end_mpi = std::chrono::high_resolution_clock::now();
                    auto duration_mpi = std::chrono::duration_cast<std::chrono::microseconds>(
                            end_mpi - begin_mpi).count();
                    mpi_duration += duration_mpi;
                    */
                    auto begin_unpack = std::chrono::high_resolution_clock::now();
                    comm->unpack_data_process_stencil_md(false, recv_zoid_num, false);
                    auto end_unpack = std::chrono::high_resolution_clock::now();
                    auto duration_unpack = std::chrono::duration_cast<std::chrono::microseconds>(
                            end_unpack - begin_unpack).count();
                    unpack_duration += duration_unpack;
                }

                auto end = std::chrono::high_resolution_clock::now();
                auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count();
                recv_comm_duration += duration;
                next_dt_comm_duration += duration;
            }
        }

        compute_duration_cilk = 0;
        modify_duration_cilk = 0;
        modify_pre_force_duration_cilk = 0;
        send_pack_duration_cilk = 0;

        cilk_for (int j = 0; j < lmp->queues_next_dt[dep].size(); j++) {
        // for (int j = 0; j < lmp->queues_next_dt[dep].size(); j++) {
            queue_info &zoid = lmp->queues_next_dt[dep][j];
            int zoid_num = zoid.num;
            if (zoid_num % comm->nprocs != comm->me) {
                continue;
            }

            /*
            if (!USE_DEP_TO_WAIT_IDXS) {
                auto begin = std::chrono::high_resolution_clock::now();
                for (int idx: dep_to_wait_idxs_next_dt[zoid_num]) {
                    int recv_zoid_num = lmp->recv_from_neighbors_procs_next_dt[idx];
                    auto begin_mpi = std::chrono::high_resolution_clock::now();
                    MPI_Wait(&receive_requests_next_dt[idx], MPI_STATUS_IGNORE);
                    auto end_mpi = std::chrono::high_resolution_clock::now();
                    auto duration_mpi = std::chrono::duration_cast<std::chrono::microseconds>(
                            end_mpi - begin_mpi).count();
                    mpi_duration += duration_mpi;
                    auto begin_unpack = std::chrono::high_resolution_clock::now();
                    comm->unpack_data_process_stencil_md(false, recv_zoid_num, false);
                    auto end_unpack = std::chrono::high_resolution_clock::now();
                    auto duration_unpack = std::chrono::duration_cast<std::chrono::microseconds>(
                            end_unpack - begin_unpack).count();
                    unpack_duration += duration_unpack;
                    next_dt_wait_dep[get_zoid_dep_next_dt(recv_zoid_num)] += duration_mpi;
                }
                auto end = std::chrono::high_resolution_clock::now();
                auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count();
                recv_comm_duration += duration;
                next_dt_comm_duration += duration;
                next_dt_dep_time[dep] += duration;
            }
            */

            auto &atom_arr = lmp->atom_stencil_md[zoid_num];
            int **atom_idx_mapping = zoid.atom_idx_mapping;

            run_stencil_md_zoid<false>(starting_timestep, zoid_num, -1, test_f, test_x);

            // send data
            if (dep < NUM_DEPS - 1) {
                Comm *comm_ = lmp->comm_stencil_md[zoid_num];
                auto begin = std::chrono::high_resolution_clock::now();

                int vec_idx = 0;
                cilk_for (int proc = 0; proc < comm->nprocs; proc++) {
                    comm_->pack_data_to_process_stencil_md(false,
                                                           atom_arr,
                                                           lmp->zoid_num_to_zoid_next_dt[zoid_num],
                                                           proc, false);
                }

                auto end = std::chrono::high_resolution_clock::now();
                auto duration =
                        std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count();
                send_pack_duration_cilk += duration;
            }
        }

        modify_duration += modify_duration_cilk / SIZES[dep];
        compute_duration += compute_duration_cilk / SIZES[dep];
        modify_pre_force_duration += modify_pre_force_duration_cilk / SIZES[dep];
        send_pack_duration += send_pack_duration_cilk / SIZES[dep];

        if (dep < NUM_DEPS - 1) {
            for (int j = 0; j < lmp->queues_next_dt[dep].size(); j++) {
                queue_info& zoid = lmp->queues_next_dt[dep][j];
                int zoid_num = zoid.num;
                if (zoid_num % comm->nprocs != comm->me) {
                    continue;
                }

                auto begin = std::chrono::high_resolution_clock::now();
                Comm* comm_ = lmp->comm_stencil_md[zoid_num];
                int vec_idx = 0;
                for (int proc = 0; proc < comm->nprocs; proc++) {
                    if (proc != comm->me
                            && lmp->send_to_neighbors_procs_next_dt[zoid_num].find(proc) != lmp->send_to_neighbors_procs_next_dt[zoid_num].end()) {
                        // bool sent = comm_->send_packed_data_to_process_stencil_md(false, zoid, &send_requests_next_dt[zoid_num][vec_idx], proc);
                        bool sent = comm_->send_packed_data_to_process_stencil_md(false, zoid, &send_requests_next_dt[zoid_num][proc], proc);
                    }
                }
                auto end = std::chrono::high_resolution_clock::now();
                auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count();
                send_comm_duration += duration;

                auto begin2 = std::chrono::high_resolution_clock::now();
                if (lmp->send_to_neighbors_procs_next_dt[zoid_num].find(comm->me) != lmp->send_to_neighbors_procs_next_dt[zoid_num].end()) {
                    comm_->send_packed_data_to_process_stencil_md(false, zoid, nullptr, comm->me);
                }
                auto end2 = std::chrono::high_resolution_clock::now();
                auto duration2 =
                        std::chrono::duration_cast<std::chrono::microseconds>(end2 - begin2).count();
                unpack_self_time += duration2;
            }
        }
    }

    // clear force on everything except first timestep
    // this should get optimized to be `memset` with -O3
    begin_misc = std::chrono::high_resolution_clock::now();
    cilk_for (int i = 0; i < NUM_ZOIDS; i++) {
        if (i % comm->nprocs == comm->me) {
            cilk_for (int t = 1; t < NUM_TIMESTEPS_IN_PARALLEL + 1; t++) {
                Atom* atom_ = lmp->atom_stencil_md[i][t];
                int nall = atom_->nlocal + atom_->nghost;
                memset(&atom_->f[0][0], 0, (nall) * 3 * sizeof(double));
                memset(&atom_->eval_f_stencil_md[0][0], 0, (nall) * 3 * sizeof(double));
            }
        }
    }

    // cleanup MPI_Request objects
    for (int i = 0; i < NUM_ZOIDS; i++) {
        if (send_requests_next_dt[i].size() > 0) {
            MPI_Waitall(send_requests_next_dt[i].size(), send_requests_next_dt[i].data(), MPI_STATUSES_IGNORE);
        }
    }

    end_misc = std::chrono::high_resolution_clock::now();
    duration_misc =
            std::chrono::duration_cast<std::chrono::microseconds>(end_misc - begin_misc).count();
    misc_time += duration_misc;
}

/* ---------------------------------------------------------------------- */

void Verlet::cleanup() {
    modify->post_run();
    domain->box_too_small_check();
    update->update_time();
}

/* ----------------------------------------------------------------------
   clear force on own & ghost atoms
   clear other arrays as needed
------------------------------------------------------------------------- */

void Verlet::force_clear() {
    size_t nbytes;

    if (external_force_clear)
        return;

    // clear force on all particles
    // if either newton flag is set, also include ghosts
    // when using threads always clear all forces.

    int nlocal = atom->nlocal;

    if (neighbor->includegroup == 0) {
        nbytes = sizeof(double) * nlocal;
        if (force->newton)
            nbytes += sizeof(double) * atom->nghost;

        if (nbytes) {
            memset(&atom->f[0][0], 0, 3 * nbytes);
            if (torqueflag)
                memset(&atom->torque[0][0], 0, 3 * nbytes);
            if (extraflag)
                atom->avec->force_clear(0, nbytes);
        }

        // neighbor includegroup flag is set
        // clear force only on initial nfirst particles
        // if either newton flag is set, also include ghosts

    } else {
        nbytes = sizeof(double) * atom->nfirst;

        if (nbytes) {
            memset(&atom->f[0][0], 0, 3 * nbytes);
            if (torqueflag)
                memset(&atom->torque[0][0], 0, 3 * nbytes);
            if (extraflag)
                atom->avec->force_clear(0, nbytes);
        }

        if (force->newton) {
            nbytes = sizeof(double) * atom->nghost;

            if (nbytes) {
                memset(&atom->f[nlocal][0], 0, 3 * nbytes);
                if (torqueflag)
                    memset(&atom->torque[nlocal][0], 0, 3 * nbytes);
                if (extraflag)
                    atom->avec->force_clear(nlocal, nbytes);
            }
        }
    }
}

void Verlet::force_clear_stencil_md(Atom* atom_, Force* force_,
                                    Neighbor* neighbor_) {
    /*
    size_t nbytes;

    if (external_force_clear) {
        return;
    }

    // clear force on all particles
    // if either newton flag is set, also include ghosts
    // when using threads always clear all forces.

    int nlocal = atom_->nlocal;

    if (neighbor_->includegroup == 0) {
        nbytes = sizeof(double) * nlocal;
        if (force_->newton) {
            nbytes += sizeof(double) * atom_->nghost;
        }

        if (nbytes) {
            memset(&atom_->f[0][0],0,3*nbytes);
            memset(&atom_->eval_f_stencil_md[0][0],0,3*nbytes);
            if (torqueflag) memset(&atom_->torque[0][0],0,3*nbytes);
            if (extraflag) atom_->avec->force_clear(0,nbytes);
        }

        // neighbor includegroup flag is set
        // clear force only on initial nfirst particles
        // if either newton flag is set, also include ghosts

    } else {
        nbytes = sizeof(double) * atom_->nfirst;

        if (nbytes) {
            memset(&atom_->f[0][0],0,3*nbytes);
            memset(&atom_->eval_f_stencil_md[0][0],0,3*nbytes);
            if (torqueflag) memset(&atom_->torque[0][0],0,3*nbytes);
            if (extraflag) atom_->avec->force_clear(0,nbytes);
        }

        if (force_->newton) {
            nbytes = sizeof(double) * atom_->nghost;

            if (nbytes) {
                memset(&atom_->f[nlocal][0],0,3*nbytes);
                memset(&atom_->eval_f_stencil_md[nlocal][0],0,3*nbytes);
                if (torqueflag) memset(&atom_->torque[nlocal][0],0,3*nbytes);
                if (extraflag) atom_->avec->force_clear(nlocal,nbytes);
            }
        }
    }
    */

    memset(&atom_->f[0][0], 0, (atom_->nlocal + atom_->nghost) * comm->nthreads * sizeof(double) * 3);
    memset(&atom_->eval_f_stencil_md[0][0], 0, (atom_->nlocal + atom_->nghost) * comm->nthreads * sizeof(double) * 3);
}

void Verlet::cleanup_stencil_md() {
    /*
    for (int dep = 0; dep < NUM_DEPS; dep++) {
        for (int j = 0; j < lmp->queues[dep].size(); j++) {
            queue_info& zoid = lmp->queues[dep][j];
            int zoid_num = zoid.num;
            if (zoid_num % comm->nprocs == comm->me) {
                int num_recv_from = lmp->recv_from_neighbors[zoid_num].size();
                int num_send_to = lmp->send_to_neighbors[zoid_num].size();

                for (int t = 0; t < NUM_TIMESTEPS_IN_PARALLEL + 1; t++) {
                    delete[] zoid.can_eval_center[t];
                    delete[] zoid.can_eval_pos[t];
                    delete[] zoid.debug_atom_pos[t];

                    for (int i = 0; i < num_recv_from; i++) {
                        delete[] zoid.recv_list_local[t][i];

                        delete[] zoid.recv_list_local_force_only[t][i];
                        delete[] zoid.recv_list_local_force_pos[t][i];
                    }
                    delete[] zoid.recv_list_local[t];
                    delete[] zoid.recv_list_local_size[t];

                    delete[] zoid.recv_list_local_force_only[t];
                    delete[] zoid.recv_list_local_force_pos[t];
                    delete[] zoid.recv_list_local_num_force_only[t];
                    delete[] zoid.recv_list_local_num_force_pos[t];

                    for (int i = 0; i < num_send_to; i++) {
                        // for send list
                        delete[] zoid.send_force_idxs[t][i];
                        delete[] zoid.send_force_sizes[t][i];

                        delete[] zoid.send_pos_idxs[t][i];
                        delete[] zoid.send_pos_sizes[t][i];

                        delete[] zoid.send_segment_sizes[t][i];
                        delete[] zoid.send_segment_types[t][i];
                        delete[] zoid.send_segment_idxs[t][i];
                        delete[] zoid.send_num_segments[t][i];

                        delete[] zoid.send_local_list[t][i];
                    }

                    delete[] zoid.send_force_idxs[t];
                    delete[] zoid.send_force_sizes[t];

                    delete[] zoid.send_pos_idxs[t];
                    delete[] zoid.send_pos_sizes[t];
                    delete[] zoid.send_force_num_segments[t];
                    delete[] zoid.send_pos_num_segments[t];

                    delete[] zoid.send_segment_sizes[t];
                    delete[] zoid.send_segment_types[t];
                    delete[] zoid.send_segment_idxs[t];
                    delete[] zoid.send_num_segments[t];

                    delete[] zoid.send_local_list[t];
                }
            }
        }
    }
    */
}
