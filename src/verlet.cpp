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
#include <iomanip>

using namespace LAMMPS_NS;

static constexpr bool TEST_AGAINST_LAMMPS_LOCAL = TEST_AGAINST_LAMMPS;

static cilk::opadd_reducer<int64_t> unpack_duration = 0;
static cilk::opadd_reducer<int64_t> send_comm_duration = 0;
static cilk::opadd_reducer<int64_t> recv_comm_duration = 0;
static cilk::opadd_reducer<int64_t> pair_duration = 0;
static cilk::opadd_reducer<int64_t> bond_duration = 0;
static cilk::opadd_reducer<int64_t> modify_initial_duration = 0;
static cilk::opadd_reducer<int64_t> modify_final_duration = 0;
static cilk::opadd_reducer<int64_t> modify_pre_force_duration = 0;
static cilk::opadd_reducer<int64_t> modify_post_force_duration = 0;
static cilk::opadd_reducer<int64_t> mpi_duration = 0;
static cilk::opadd_reducer<int64_t> send_pack_duration = 0;
static cilk::opadd_reducer<int64_t> misc_time = 0;
static cilk::opadd_reducer<int64_t> unpack_self_time = 0;
static cilk::opadd_reducer<int64_t> pre_recv_time = 0;

static std::vector<int64_t> lammps_forward_comm_times;
static std::vector<int64_t> lammps_reverse_comm_times;

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

    if (!ONLY_RUN_STENCIL_MD) {
        modify->setup_pre_force(vflag);
    }

    if (pair_compute_flag) {
        if (!ONLY_RUN_STENCIL_MD) {
            force->pair->compute(eflag, vflag);
        }
    } else if (force->pair) {
        force->pair->compute_dummy(eflag, vflag);
    }

    if (atom->molecular != Atom::ATOMIC) {
        if (!ONLY_RUN_STENCIL_MD) {
            if (force->bond) {
                force->bond->compute(eflag, vflag);
            }
        }
        if (force->angle) {
            assert(false);
            force->angle->compute(eflag, vflag);
        }
        if (force->dihedral) {
            assert(false);
            force->dihedral->compute(eflag, vflag);
        }
        if (force->improper) {
            assert(false);
            force->improper->compute(eflag, vflag);
        }
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

    if (LAMMPS_USE_BINS) {
        assert(false);
        // lammps setup
        auto& lammps_bin_bounds = stencilMD->LAMMPS_GET_BOUNDS(true, 0);
        auto& lammps_sorted_bin_indices = stencilMD->lammps_sorted_bin_indices[0];
        domain->pbc();
        comm->exchange();
        atom->lammps_sort_local_bins(lammps_bin_bounds, lammps_sorted_bin_indices);
        comm->borders();
        neighbor->build(1);
        atom->setup_lammps_pair_bins();
        neighbor->setup_stencil_md_bond_bins(atom);

        stencilMD->lammps_setup_atomic_lists();

        stencilMD->INIT_PER_PARTITION_FORCE_ARRAY();
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
            if (new_pos[dim] < domain->boxlo[dim]) {
                new_pos[dim] += domain->prd[dim];
            }
            if (new_pos[dim] > domain->boxhi[dim]) {
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

                bool at_least_one_curr = (sub >= lo_curr && sub < hi_curr) ||
                                         (add >= lo_curr && add < hi_curr) ||
                                         (value >= lo_curr && value < hi_curr);

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

                // double atom_pos_shifted = value + pbc_ * domain->prd[dim];

                double sub = value - domain->prd[dim];
                double add = value + domain->prd[dim];

                bool at_least_one_prev = (sub >= lo_prev && sub < hi_prev) ||
                                         (add >= lo_prev && add < hi_prev) ||
                                         (value >= lo_prev && value < hi_prev);

                bool at_least_one_curr = (sub >= lo_curr && sub < hi_curr) ||
                                         (add >= lo_curr && add < hi_curr) ||
                                         (value >= lo_curr && value < hi_curr);

                bool at_least_one_next = (sub >= lo_next && sub < hi_next) ||
                                         (add >= lo_next && add < hi_next) ||
                                         (value >= lo_next && value < hi_next);

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
                if (target_zoid_prev != -1) {
                    std::cout << "tag: " << atom_->tag[idx] << std::endl;
                    std::cout << "period: " << domain->prd[0] << " " << domain->prd[1] << " " << domain->prd[2]
                        << " boxlo: " << domain->boxlo[0] << " " << domain->boxlo[1] << " " << domain->boxlo[2]
                        << " boxhi: " << domain->boxhi[0] << " " << domain->boxhi[1] << " " << domain->boxhi[2] << std::endl;
                    std::cout << "old pos: " << atom_->x[idx][0] << " " << atom_->x[idx][1] << " " << atom_->x[idx][2]
                        << " pos: " << new_pos[0] << " " << new_pos[1] << " " << new_pos[2] << std::endl;
                    std::cout << "target zoid prev: " << target_zoid_prev << " curr zoid: " << k << " time: " << timestep << std::endl;
                    queue_info& prev_zoid = lmp->zoid_num_to_zoid[target_zoid_prev];
                    queue_info& new_zoid = lmp->zoid_num_to_zoid[k];
                    for (int dim = 0; dim < 3; dim++) {
                        std::cout << "me: " << comm->me << " prev zoid lo: " << prev_zoid.zoid.cuts[dim].lower + timestep * prev_zoid.zoid.cuts[dim].slope_lower
                            << " prev zoid hi: " << prev_zoid.zoid.cuts[dim].upper + timestep * prev_zoid.zoid.cuts[dim].slope_upper << std::endl;
                    }
                    for (int dim = 0; dim < 3; dim++) {
                        std::cout << "me: " << comm->me << " new zoid lo: " << new_zoid.zoid.cuts[dim].lower + timestep * new_zoid.zoid.cuts[dim].slope_lower
                                  << " new zoid hi: " << new_zoid.zoid.cuts[dim].upper + timestep * new_zoid.zoid.cuts[dim].slope_upper << std::endl;
                    }
                }
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
            /*
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
            */

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

    delete[] current;
    delete[] permute;

    for (int idx = 0; idx < atom_->nlocal + atom_->nghost; idx++) {
        if (atom_->tag_to_idx.count(atom_->tag[idx])) {
            std::cout << "idx: " << idx << " repeat tag. tag: " << atom_->tag[idx] << " zoid: " << zoid_num
                      << std::endl;
        }
        assert(!atom_->tag_to_idx.count(atom_->tag[idx]));
        atom_->tag_to_idx[atom_->tag[idx]] = idx;
    }
}

void Verlet::sort_ghost_atoms_stencil_md_bins(Atom* atom_, queue_info& zoid, int timestep) {
    int zoid_num = zoid.num;

    int* current = new int[atom_->nghost];
    int* permute = new int[atom_->nghost];

    for (int i = 0; i < atom_->nghost; i++) {
        current[i] = i;
    }

    std::vector<int> ghost_idxs;
    for (int i = 0; i < atom_->nghost; i++) {
        ghost_idxs.push_back(i);
    }

    auto& bin_bounds = stencilMD->GET_BOUNDS(true, timestep);
    // auto& sorted_bin_indices = stencilMD->sorted_bin_indices[timestep];
    auto& sorted_bin_indices = atom_->sorted_ghost_bin_indices;

    // std::map<std::array<int, 3>, Data_vector> bin_to_data_points;
    // std::map<std::tuple<int, int, int>, Data_vector> bin_to_data_points;
    std::map<IDX_3D, Data_vector> bin_to_data_points;

    for (int i = atom_->nlocal; i < atom_->nlocal + atom_->nghost; i++) {
        double* pos = atom_->x[i];
        double new_pos[3];
        for (int j = 0; j < 3; j++) {
            double new_pos_dim = pos[j];
            if (new_pos_dim < domain->boxlo[j]) {
                new_pos_dim += domain->prd[j];
            }
            if (new_pos_dim >= domain->boxhi[j]) {
                new_pos_dim -= domain->prd[j];
            }
            new_pos[j] = new_pos_dim;
        }
        auto bin = get_bin(bin_bounds, pos, domain->boxlo, domain->boxhi);
        // bin_to_data_points[bin].push_back(std::make_pair(Point(pos[0], pos[1], pos[2]), i));
        bin_to_data_points[bin].push_back(std::make_pair(Point(new_pos[0], new_pos[1], new_pos[2]), i));
    }

    std::map<int, int> bin_to_idx;
    for (int i = 0; i < sorted_bin_indices.size(); i++) {
        bin_to_idx[sorted_bin_indices[i]] = i;
    }

    std::sort(
            ghost_idxs.begin(), ghost_idxs.end(), [&](const int& a, const int& b) {
                int idx_a = atom_->nlocal + a;
                int idx_b = atom_->nlocal + b;

                double* pos_a = atom_->x[idx_a];
                double* pos_b = atom_->x[idx_b];
                auto bin_a = get_bin(bin_bounds, pos_a, domain->boxlo, domain->boxhi);
                auto bin_b = get_bin(bin_bounds, pos_b, domain->boxlo, domain->boxhi);

                int bin_idx_a = get_bin_idx(bin_a);
                int bin_idx_b = get_bin_idx(bin_b);

                // int find_idx_a = std::distance(sorted_bin_indices.begin(), std::find(sorted_bin_indices.begin(), sorted_bin_indices.end(), bin_idx_a));
                // assert(find_idx_a < sorted_bin_indices.size());

                // int find_idx_b = std::distance(sorted_bin_indices.begin(), std::find(sorted_bin_indices.begin(), sorted_bin_indices.end(), bin_idx_b));
                // assert(find_idx_b < sorted_bin_indices.size());
                int find_idx_a = bin_to_idx[bin_idx_a];
                int find_idx_b = bin_to_idx[bin_idx_b];

                /*
                if (bin_a < bin_b) {
                    return true;
                } else if (bin_a > bin_b) {
                    return false;
                }
                */

                if (find_idx_a < find_idx_b) {
                    return true;
                } else if (find_idx_a > find_idx_b) {
                    return false;
                }

                std::array<std::vector<double>, 3> bounds;
                for (int dim = 0; dim < 3; dim++) {
                    /*
                    if (bin_a[dim] == bin_bounds.size() - 1) {
                        bounds[dim].push_back(bin_bounds[bin_a[dim]]);
                        bounds[dim].push_back(bin_bounds[0] + domain->prd[dim]);
                    } else {
                        bounds[dim].push_back(bin_bounds[bin_a[dim]]);
                        bounds[dim].push_back(bin_bounds[bin_a[dim] + 1]);
                    }
                    */
                    bounds[dim].push_back(bin_bounds[bin_a[dim]]);
                }

                double new_pos_a[3];
                double new_pos_b[3];

                for (int dim = 0; dim < 3; dim++) {
                    double new_pos_a_ = pos_a[dim];
                    if (new_pos_a_ > domain->boxhi[dim]) {
                        new_pos_a_ -= domain->prd[dim];
                    }
                    if (new_pos_a_ < domain->boxlo[dim]) {
                        new_pos_a_ += domain->prd[dim];
                    }
                    new_pos_a[dim] = new_pos_a_;
                }

                for (int dim = 0; dim < 3; dim++) {
                    double new_pos_b_ = pos_b[dim];
                    if (new_pos_b_ > domain->boxhi[dim]) {
                        new_pos_b_ -= domain->prd[dim];
                    }
                    if (new_pos_b_ < domain->boxlo[dim]) {
                        new_pos_b_ += domain->prd[dim];
                    }
                    new_pos_b[dim] = new_pos_b_;
                }

                // double dist_a = min_dist_to_boundary(bounds, {pos_a[0], pos_a[1], pos_a[2]});
                // double dist_b = min_dist_to_boundary(bounds, {pos_b[0], pos_b[1], pos_b[2]});
                double dist_a = min_dist_to_boundary(bounds, {new_pos_a[0], new_pos_a[1], new_pos_a[2]});
                double dist_b = min_dist_to_boundary(bounds, {new_pos_b[0], new_pos_b[1], new_pos_b[2]});

                // return dist_a < dist_b;
                return atom_->tag[idx_a] < atom_->tag[idx_b];

                /*
                assert(bin_a == bin_b);

                auto& data_points = bin_to_data_points[bin_a];

                find_idx_a = std::distance(data_points.begin(), std::find_if(data_points.begin(), data_points.end(), [&](auto& elem) {
                    return elem.second == idx_a;
                }));

                assert(find_idx_a < data_points.size());

                find_idx_b = std::distance(data_points.begin(), std::find_if(data_points.begin(), data_points.end(), [&](auto& elem) {
                    return elem.second == idx_b;
                }));

                assert(find_idx_b < data_points.size());

                return find_idx_a < find_idx_b;
                */

                /*
                bool close = fabs(pos_a[0] - pos_b[0]) <= 1e-5 && fabs(pos_a[1] - pos_b[1]) <= 1e-5 && fabs(pos_a[2] - pos_b[2]) <= 1e-5;
                if (close) {
                    assert(idx_a == idx_b);
                    return false;
                }

                return atom_->tag[idx_a] < atom_->tag[idx_b];

                /*
                std::vector<Point> p(2);
                Point p_a(pos_a[0], pos_a[1], pos_a[2]);
                Point p_b(pos_b[0], pos_b[1], pos_b[2]);
                p[0] = p_a;
                p[1] = p_b;

                CGAL::hilbert_sort(p.begin(), p.end());

                bool compare = fabs(p[0].x() - p_a.x()) <= 1e-5 && fabs(p[0].y() - p_a.y()) <= 1e-5 && fabs(p[0].z() - p_a.z()) <= 1e-5;
                return compare;
                */
            });

    for (int i = 0; i < atom_->nghost; i++) {
        permute[i] = ghost_idxs[i];
    }

    atom_reorder_ghost_stencil_md(atom_, current, permute, 0, atom_->nghost,
                                  atom_->nlocal);

    for (int i = atom_->nlocal; i < atom_->nlocal + atom_->nghost; i++) {
        double* pos = atom_->x[i];
        auto bin = get_bin(bin_bounds, pos, domain->boxlo, domain->boxhi);
        atom_->bin_to_local_idxs2[bin[0]][bin[1]][bin[2]].push_back(i);
    }

    delete[] current;
    delete[] permute;

    for (int idx = 0; idx < atom_->nlocal + atom_->nghost; idx++) {
        if (atom_->tag_to_idx.count(atom_->tag[idx])) {
            std::cout << "idx: " << idx << " repeat tag. tag: " << atom_->tag[idx] << " zoid: " << zoid_num
                      << std::endl;
        }
        assert(!atom_->tag_to_idx.count(atom_->tag[idx]));
        atom_->tag_to_idx[atom_->tag[idx]] = idx;
    }
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

            bool in_zoid_curr = true;

            for (int dim = 0; dim < domain->dimension; dim++) {
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

                bool at_least_one_prev = (sub >= lo_prev && sub < hi_prev) ||
                                         (add >= lo_prev && add < hi_prev) ||
                                         (value >= lo_prev && value < hi_prev);

                double lo_curr = recv_from_zoid.zoid.cuts[dim].lower +
                                 (timestep)*zoid.zoid.cuts[dim].slope_lower;
                double hi_curr = recv_from_zoid.zoid.cuts[dim].upper +
                                 (timestep)*zoid.zoid.cuts[dim].slope_upper;

                bool at_least_one_curr = (sub >= lo_curr && sub < hi_curr) ||
                                         (add >= lo_curr && add < hi_curr) ||
                                         (value >= lo_curr && value < hi_curr);

                // if expanding zoid AND other zoid shrinking, use current instead of past, only need values for actual shrinking?
                // must be a ghost atom that wasn't local last timestep somehow
                // TODO: Ryan please check
                if (timestep > 0) {
                    // in_zoid_prev = in_zoid_prev && ((atom_pos_shifted >= lo_prev && atom_pos_shifted <= hi_prev));
                    in_zoid_prev = in_zoid_prev && at_least_one_prev;
                } else {
                    in_zoid_prev = false;
                }

                // if (!(value >= lo_curr && value <= hi_curr) && zoid.zoid.cuts[dim].slope_lower < 0 && timestep < NUM_TIMESTEPS_IN_PARALLEL) {
                if (!(value >= lo_curr && value < hi_curr) &&
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

            bool in_zoid_curr = true;

            for (int dim = 0; dim < domain->dimension; dim++) {
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

                bool at_least_one_prev = (sub >= lo_prev && sub < hi_prev) ||
                                         (add >= lo_prev && add < hi_prev) ||
                                         (value >= lo_prev && value < hi_prev);

                double lo_curr = recv_from_zoid.zoid.cuts[dim].lower +
                                 (timestep)*recv_from_zoid.zoid.cuts[dim].slope_lower;
                double hi_curr = recv_from_zoid.zoid.cuts[dim].upper +
                                 (timestep)*recv_from_zoid.zoid.cuts[dim].slope_upper;

                bool at_least_one_curr = (sub >= lo_curr && sub < hi_curr) ||
                                         (add >= lo_curr && add < hi_curr) ||
                                         (value >= lo_curr && value < hi_curr);

                // if expanding zoid AND other zoid shrinking, use current instead of past, only need values for actual shrinking?
                // must be a ghost atom that wasn't local last timestep somehow
                if (timestep > 0) {
                    // in_zoid_prev = in_zoid_prev && ((atom_pos_shifted >= lo_prev && atom_pos_shifted <= hi_prev));
                    in_zoid_prev = in_zoid_prev && at_least_one_prev;
                } else {
                    in_zoid_prev = false;
                    // TODO: timestep 0 case
                }

                // if (!(value >= lo_curr && value <= hi_curr) && zoid.zoid.cuts[dim].slope_lower < 0 && timestep < NUM_TIMESTEPS_IN_PARALLEL) {
                if (!(value >= lo_curr && value < hi_curr) &&
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

void Verlet::construct_send_force_bins(bool curr_dt, Atom* atom_, queue_info &zoid, int timestep) {
    int zoid_num = zoid.num;
    auto& bounds = stencilMD->GET_BOUNDS(curr_dt, timestep);
    auto& send_to = curr_dt ? lmp->send_to_neighbors[zoid_num] : lmp->send_to_neighbors_next_dt[zoid_num];

    zoid.send_force_num_bins[timestep] = new int[send_to.size()];
    // zoid.send_force_bins[timestep] = new std::tuple<int, int, int>*[send_to.size()];
    zoid.send_force_bins[timestep] = new IDX_3D*[send_to.size()];

    for (int i = 0; i < send_to.size(); i++) {
        // std::set<std::tuple<int, int, int>> bins;
        std::set<IDX_3D> bins;
        int send_zoid_num = send_to[i];
        int num_send_force_segments = zoid.send_force_num_segments[timestep][i];
        for (int j = 0; j < num_send_force_segments; j++) {
            int segment_idx = zoid.send_force_idxs[timestep][i][j];
            int segment_size = zoid.send_force_sizes[timestep][i][j];
            for (int h = 0; h < segment_size; h++) {
                int idx = segment_idx + h;
                double* pos = atom_->x[idx];
                auto bin = get_bin(bounds, pos, domain->boxlo, domain->boxhi);
                bins.insert(bin);
            }
        }
        int num_bins = bins.size();
        zoid.send_force_num_bins[timestep][i] = num_bins;
        // zoid.send_force_bins[timestep][i] = new std::tuple<int, int, int>[num_bins];
        zoid.send_force_bins[timestep][i] = new IDX_3D[num_bins];
        // std::vector<std::tuple<int, int, int>> bins_vec(bins.begin(), bins.end());
        std::vector<IDX_3D> bins_vec(bins.begin(), bins.end());

        for (int j = 0; j < num_bins; j++) {
            zoid.send_force_bins[timestep][i][j] = bins_vec[j];
        }
    }
}

void Verlet::construct_send_pos_bins(bool curr_dt, Atom* atom_, queue_info &zoid, int timestep) {
    int zoid_num = zoid.num;
    auto& bounds = stencilMD->GET_BOUNDS(curr_dt, timestep);
    auto& send_to = curr_dt ? lmp->send_to_neighbors[zoid_num] : lmp->send_to_neighbors_next_dt[zoid_num];

    zoid.send_pos_num_bins[timestep] = new int[send_to.size()];
    // zoid.send_pos_bins[timestep] = new std::tuple<int, int, int>*[send_to.size()];
    zoid.send_pos_bins[timestep] = new IDX_3D*[send_to.size()];

    // first gather local_to_ghost
    for (int i = 0; i < send_to.size(); i++) {
        // std::set<std::tuple<int, int, int>> bins;
        std::set<IDX_3D> bins;
        int send_zoid_num = send_to[i];
        int num_recv_ghost_segments = zoid.send_num_segments[timestep][i];

        int local_idx = 0;
        for (int j = 0; j < num_recv_ghost_segments; j++) {
            int segment_type = zoid.send_segment_types[timestep][i][j];
            int segment_idx = zoid.send_segment_idxs[timestep][i][j];
            int segment_size = zoid.send_segment_sizes[timestep][i][j];
            for (int h = 0; h < segment_size; h++) {
                int idx;
                if (segment_type == LOCAL_SEGMENT_TYPE) {
                    idx = zoid.send_local_list[timestep][i][local_idx++];
                } else {
                    idx = segment_idx + h;
                }

                double* pos = atom_->x[idx];
                auto bin = get_bin(bounds, pos, domain->boxlo, domain->boxhi);
                bins.insert(bin);
            }
        }

        int num_ghost_to_local_segments = zoid.send_pos_num_segments[timestep][i];
        for (int j = 0; j < num_ghost_to_local_segments; j++) {
            int segment_idx = zoid.send_pos_idxs[timestep][i][j];
            int segment_size = zoid.send_pos_sizes[timestep][i][j];
            for (int h = 0; h < segment_size; h++) {
                int idx = segment_idx + h;
                double* pos = atom_->x[idx];
                auto bin = get_bin(bounds, pos, domain->boxlo, domain->boxhi);
                bins.insert(bin);
            }
        }

        int num_bins = bins.size();
        zoid.send_pos_num_bins[timestep][i] = num_bins;
        // zoid.send_pos_bins[timestep][i] = new std::tuple<int, int, int>[num_bins];
        // std::vector<std::tuple<int, int, int>> bins_vec(bins.begin(), bins.end());
        zoid.send_pos_bins[timestep][i] = new IDX_3D[num_bins];
        std::vector<IDX_3D> bins_vec(bins.begin(), bins.end());

        for (int j = 0; j < num_bins; j++) {
            zoid.send_pos_bins[timestep][i][j] = bins_vec[j];
        }
    }
}

void Verlet::construct_send_vel_bins(bool curr_dt, Atom* atom_, queue_info &zoid, int timestep) {
    int zoid_num = zoid.num;
    auto& bounds = stencilMD->GET_BOUNDS(curr_dt, timestep);
    auto& send_to = curr_dt ? lmp->send_to_neighbors[zoid_num] : lmp->send_to_neighbors_next_dt[zoid_num];

    zoid.send_vel_num_bins[timestep] = new int[send_to.size()];
    // zoid.send_vel_bins[timestep] = new std::tuple<int, int, int>*[send_to.size()];
    zoid.send_vel_bins[timestep] = new IDX_3D*[send_to.size()];

    // first gather local_to_ghost
    for (int i = 0; i < send_to.size(); i++) {
        // std::set<std::tuple<int, int, int>> bins;
        std::set<IDX_3D> bins;
        int send_zoid_num = send_to[i];

        int num_recv_local_segments = zoid.send_pos_num_segments[timestep][i];
        for (int j = 0; j < num_recv_local_segments; j++) {
            int segment_idx = zoid.send_pos_idxs[timestep][i][j];
            int segment_size = zoid.send_pos_sizes[timestep][i][j];
            for (int h = 0; h < segment_size; h++) {
                int idx = segment_idx + h;
                double* pos = atom_->x[idx];
                auto bin = get_bin(bounds, pos, domain->boxlo, domain->boxhi);
                bins.insert(bin);
            }
        }

        int num_bins = bins.size();
        zoid.send_vel_num_bins[timestep][i] = num_bins;
        // zoid.send_vel_bins[timestep][i] = new std::tuple<int, int, int>[num_bins];
        zoid.send_vel_bins[timestep][i] = new IDX_3D[num_bins];
        // std::vector<std::tuple<int, int, int>> bins_vec(bins.begin(), bins.end());
        std::vector<IDX_3D> bins_vec(bins.begin(), bins.end());

        for (int j = 0; j < num_bins; j++) {
            zoid.send_vel_bins[timestep][i][j] = bins_vec[j];
        }
    }
}

void Verlet::construct_recv_force_bins(bool curr_dt, Atom* atom_, queue_info &zoid, int timestep) {
    int zoid_num = zoid.num;
    auto& bounds = stencilMD->GET_BOUNDS(curr_dt, timestep);
    auto& recv_from = curr_dt ? lmp->recv_from_neighbors[zoid_num] : lmp->recv_from_neighbors_next_dt[zoid_num];

    zoid.recv_force_num_bins[timestep] = new int[recv_from.size()];
    // zoid.recv_force_bins[timestep] = new std::tuple<int, int, int>*[recv_from.size()];
    zoid.recv_force_bins[timestep] = new IDX_3D*[recv_from.size()];

    for (int i = 0; i < recv_from.size(); i++) {
        // std::set<std::tuple<int, int, int>> bins;
        std::set<IDX_3D> bins;
        int recv_zoid_num = recv_from[i];
        int num_recv_force = zoid.recv_list_local_num_force_only[timestep][i];

        for (int j = 0; j < num_recv_force; j++) {
            int idx = zoid.recv_list_local_force_only[timestep][i][j];
            double* pos = atom_->x[idx];
            auto bin = get_bin(bounds, pos, domain->boxlo, domain->boxhi);
            bins.insert(bin);
        }

        int num_bins = bins.size();
        zoid.recv_force_num_bins[timestep][i] = num_bins;
        // zoid.recv_force_bins[timestep][i] = new std::tuple<int, int, int>[num_bins];
        // std::vector<std::tuple<int, int, int>> bins_vec(bins.begin(), bins.end());
        zoid.recv_force_bins[timestep][i] = new IDX_3D[num_bins];
        std::vector<IDX_3D> bins_vec(bins.begin(), bins.end());

        for (int j = 0; j < num_bins; j++) {
            zoid.recv_force_bins[timestep][i][j] = bins_vec[j];
        }
    }
}

void Verlet::construct_recv_vel_bins(bool curr_dt, Atom* atom_, queue_info &zoid, int timestep) {
    int zoid_num = zoid.num;
    auto& bounds = stencilMD->GET_BOUNDS(curr_dt, timestep);
    auto& recv_from = curr_dt ? lmp->recv_from_neighbors[zoid_num] : lmp->recv_from_neighbors_next_dt[zoid_num];

    zoid.recv_vel_num_bins[timestep] = new int[recv_from.size()];
    // zoid.recv_vel_bins[timestep] = new std::tuple<int, int, int>*[recv_from.size()];
    zoid.recv_vel_bins[timestep] = new IDX_3D*[recv_from.size()];

    for (int i = 0; i < recv_from.size(); i++) {
        // std::set<std::tuple<int, int, int>> bins;
        std::set<IDX_3D> bins;
        int recv_zoid_num = recv_from[i];
        int num_recv_vel = zoid.recv_list_local_num_force_pos[timestep][i];

        for (int j = 0; j < num_recv_vel; j++) {
            int idx = zoid.recv_list_local_force_pos[timestep][i][j];
            double* pos = atom_->x[idx];
            auto bin = get_bin(bounds, pos, domain->boxlo, domain->boxhi);
            bins.insert(bin);
        }

        int num_bins = bins.size();
        zoid.recv_vel_num_bins[timestep][i] = num_bins;
        // zoid.recv_vel_bins[timestep][i] = new std::tuple<int, int, int>[num_bins];
        zoid.recv_vel_bins[timestep][i] = new IDX_3D[num_bins];
        // std::vector<std::tuple<int, int, int>> bins_vec(bins.begin(), bins.end());
        std::vector<IDX_3D> bins_vec(bins.begin(), bins.end());

        for (int j = 0; j < num_bins; j++) {
            zoid.recv_vel_bins[timestep][i][j] = bins_vec[j];
        }
    }
}

// Right now, only looks at ghost-pos bins
void Verlet::construct_recv_pos_bins(bool curr_dt, Atom* atom_, queue_info &zoid, int timestep) {
    int zoid_num = zoid.num;
    auto& bounds = stencilMD->GET_BOUNDS(curr_dt, timestep);
    auto& recv_from = curr_dt ? lmp->recv_from_neighbors[zoid_num] : lmp->recv_from_neighbors_next_dt[zoid_num];

    zoid.recv_pos_num_bins[timestep] = new int[recv_from.size()];
    zoid.recv_pos_bins[timestep] = new IDX_3D*[recv_from.size()];

    // first gather local_to_ghost
    for (int i = 0; i < recv_from.size(); i++) {
        std::set<IDX_3D> bins;
        int send_zoid_num = recv_from[i];

        int num_recv_ghost_segments = zoid.recv_ghost_num_segments[timestep][i];
        for (int j = 0; j < num_recv_ghost_segments; j++) {
            int segment_idx = zoid.recv_ghost_idxs[timestep][i][j];
            int segment_size = zoid.recv_ghost_sizes[timestep][i][j];
            for (int h = 0; h < segment_size; h++) {
                int idx = segment_idx + h;

                double* pos = atom_->x[idx];
                auto bin = get_bin(bounds, pos, domain->boxlo, domain->boxhi);
                bins.insert(bin);
            }
        }

        /*
        int num_recv_local = zoid.recv_list_local_num_force_pos[timestep][i];
        for (int j = 0; j < num_recv_local; j++) {
            int idx = zoid.recv_list_local_force_pos[timestep][i][j];
            double* pos = atom_->x[idx];
            auto bin = get_bin(bounds, pos, domain->boxlo, domain->boxhi);
            bins.insert(bin);
        }
        */

        int num_bins = bins.size();
        zoid.recv_pos_num_bins[timestep][i] = num_bins;
        zoid.recv_pos_bins[timestep][i] = new IDX_3D[num_bins];
        std::vector<IDX_3D> bins_vec(bins.begin(), bins.end());

        for (int j = 0; j < num_bins; j++) {
            zoid.recv_pos_bins[timestep][i][j] = bins_vec[j];
        }
    }
}

void Verlet::construct_no_comm_bins(bool curr_dt, Atom* atom_, queue_info &zoid, int timestep) {
    int zoid_num = zoid.num;
    auto& recv_from = curr_dt ? lmp->recv_from_neighbors[zoid_num] : lmp->recv_from_neighbors_next_dt[zoid_num];
    auto& bounds = stencilMD->GET_BOUNDS(curr_dt, timestep);

    std::set<IDX_3D> all_comm_bins;

    std::set<IDX_3D> all_force_bins;
    std::set<IDX_3D> all_pos_vel_bins;

    std::map<IDX_3D, std::vector<int>> bin_to_force_comm;
    std::map<IDX_3D, int> bin_to_pos_vel_comm;

    // first gather local_to_ghost
    for (int i = 0; i < recv_from.size(); i++) {
        int num_recv_force_bins = zoid.recv_force_num_bins[timestep][i];
        for (int j = 0; j < num_recv_force_bins; j++) {
            auto& force_bin = zoid.recv_force_bins[timestep][i][j];
            all_comm_bins.insert(force_bin);
            bin_to_force_comm[force_bin].push_back(recv_from[i]);

            all_force_bins.insert(force_bin);
        }

        // these are ghost pos bins
        int num_recv_pos_bins = zoid.recv_pos_num_bins[timestep][i];
        for (int j = 0; j < num_recv_pos_bins; j++) {
            auto pos_bin = zoid.recv_pos_bins[timestep][i][j];
            all_comm_bins.insert(pos_bin);
        }

        int num_recv_pos_vel_bins = zoid.recv_vel_num_bins[timestep][i];
        for (int j = 0; j < num_recv_pos_vel_bins; j++) {
            auto vel_bin = zoid.recv_vel_bins[timestep][i][j];
            all_comm_bins.insert(vel_bin);
            assert(!bin_to_pos_vel_comm.count(vel_bin));
            bin_to_pos_vel_comm[vel_bin] = recv_from[i];

            all_pos_vel_bins.insert(vel_bin);
        }
    }

    for (int i = 0; i < atom_->nlocal; i++) {
        auto bin = get_bin(bounds, atom_->x[i], domain->boxlo, domain->boxhi);
        if (all_pos_vel_bins.find(bin) == all_pos_vel_bins.end()) {
            if (std::find(zoid.no_comm_local_bins[timestep].begin(), zoid.no_comm_local_bins[timestep].end(), bin)
                == zoid.no_comm_local_bins[timestep].end()) {
                zoid.no_comm_local_bins[timestep].push_back(bin);
            }
        } else {
            if (std::find(zoid.comm_local_bins[timestep].begin(), zoid.comm_local_bins[timestep].end(), bin)
                == zoid.comm_local_bins[timestep].end()) {
                zoid.comm_local_bins[timestep].push_back(bin);
            }
        }
    }

    if (zoid.comm_local_bins[timestep].size() > 0) {
        zoid.bin_to_pos_vel_comm[timestep] = new int[zoid.comm_local_bins[timestep].size()];
        for (int i = 0; i < zoid.comm_local_bins[timestep].size(); i++) {
            auto& comm_bin = zoid.comm_local_bins[timestep][i];
            zoid.bin_to_pos_vel_comm[timestep][i] = bin_to_pos_vel_comm[comm_bin];
            assert(bin_to_pos_vel_comm.count(comm_bin));
        }
    }

    zoid.local_bins_comm[timestep].reserve(atom_->local_bins.size());
    zoid.bin_to_force_comm[timestep] = new std::vector<int>[atom_->local_bins.size()];

    for (int i = 0; i < atom_->local_bins.size(); i++) {
        auto& bin = atom_->local_bins[i];
        if (all_pos_vel_bins.find(bin) == all_pos_vel_bins.end()) {
            zoid.local_bins_comm[timestep].push_back(false);
        } else {
            zoid.local_bins_comm[timestep].push_back(true);
        }

        zoid.bin_to_force_comm[timestep][i] = bin_to_force_comm[bin];
    }
}

void Verlet::construct_dtfm_cache(Atom* atom_) {
    double dtv = update->dt;
    double dtf = 0.5 * update->dt * force->ftm2v;

    const double * const mass = atom->mass;
    const int * const type = atom_->type;
    atom_->local_dtfm.reserve(atom_->nlocal);
    for (int i = 0; i < atom_->nlocal; i++) {
        const double dtfm = dtf / mass[type[i]];
        atom_->local_dtfm.push_back(dtfm);
    }
}

void Verlet::construct_bin_to_idx(bool curr_dt, Atom* atom_, queue_info &zoid, int timestep) {
    int zoid_num = zoid.num;
    auto& bounds = stencilMD->GET_BOUNDS(curr_dt, timestep);

    assert(atom_->nlocal + atom_->nghost > 0);

    std::set<IDX_3D> all_bins;

    auto prev_bin = get_bin(bounds, atom_->x[0], domain->boxlo, domain->boxhi);
    int prev_idx = 0;
    for (int i = 1; i < atom_->nlocal + atom_->nghost; i++) {
        auto curr_bin = get_bin(bounds, atom_->x[i], domain->boxlo, domain->boxhi);
        if (curr_bin != prev_bin) {
            // int num_in_prev_bin = i - 1 - prev_idx;
            int num_in_prev_bin = i - prev_idx;
            int bin_idx = get_bin_idx(prev_bin);
            assert(bin_idx < NUM_BINS * NUM_BINS * NUM_BINS);
            if (zoid.bin_to_idx[timestep][bin_idx] != -1) {
                std::stringstream bounds_str;
                for (auto& b: bounds) {
                    bounds_str << b << " ";
                }
                std::cout << "BOUNDS: " << bounds_str.str() << std::endl;
                std::cout << "zoid: " << zoid.num << " timestep: " << timestep << " bin idx: " << bin_idx << " curr dt? " << curr_dt << " curr idx: " << i << " existing idx: " << zoid.bin_to_idx[timestep][bin_idx] << " existng size: " << zoid.bin_to_size[timestep][bin_idx] << std::endl;
                for (int k = 0; k < atom_->nlocal + atom_->nghost; k++) {
                    double* tmp_pos = atom_->x[k];
                    auto tmp_bin = get_bin(bounds, tmp_pos, domain->boxlo, domain->boxhi);
                    std::cout << "idx: " << k << " nlocal: " << atom_->nlocal << " bin: " << std::get<0>(tmp_bin) << " " << std::get<1>(tmp_bin) << " " << std::get<2>(tmp_bin)
                        << " pos: " << tmp_pos[0] << " " << tmp_pos[1] << " " << tmp_pos[2] << std::endl;
                }
            }
            assert(zoid.bin_to_idx[timestep][bin_idx] == -1);
            assert(zoid.bin_to_size[timestep][bin_idx] == -1);
            zoid.bin_to_idx[timestep][bin_idx] = prev_idx;
            zoid.bin_to_size[timestep][bin_idx] = num_in_prev_bin;

            if (all_bins.find(prev_bin) != all_bins.end()) {
                std::cout << "curr_dt? " << curr_dt << " zoid: " << zoid.num << " t: " << timestep << " bin already found" << std::endl;
                assert(false);
            }

            all_bins.insert(prev_bin);

            prev_bin = curr_bin;
            prev_idx = i;
        }
    }

    auto last_bin = get_bin(bounds, atom_->x[atom_->nlocal + atom_->nghost - 1], domain->boxlo, domain->boxhi);
    int last_bin_idx = get_bin_idx(last_bin);
    zoid.bin_to_idx[timestep][last_bin_idx] = prev_idx;
    zoid.bin_to_size[timestep][last_bin_idx] = atom_->nlocal + atom_->nghost - prev_idx;
}

void Verlet::construct_bin_to_send_zoids(bool curr_dt, Atom* atom_, queue_info& zoid, int timestep) {
    int zoid_num = zoid.num;
    auto& bounds = stencilMD->GET_BOUNDS(curr_dt, timestep);

    assert(atom_->nlocal + atom_->nghost > 0);

    auto& send_to = curr_dt ? lmp->send_to_neighbors[zoid_num] : lmp->send_to_neighbors_next_dt[zoid_num];

    std::map<IDX_3D, std::set<int>> bin_to_send_zoids;

    for (int i = 0; i < atom_->nlocal + atom_->nghost; i++) {
        auto bin = get_bin(bounds, atom_->x[i], domain->boxlo, domain->boxhi);
        auto bin_idx = get_bin_idx(bin);

        for (int j = 0; j < send_to.size(); j++) {
            int recv_zoid_num = send_to[j];
            auto& recv_zoid = curr_dt ? lmp->zoid_num_to_zoid[recv_zoid_num] : lmp->zoid_num_to_zoid_next_dt[recv_zoid_num];
            Atom* recv_atom = curr_dt ? lmp->atom_stencil_md[recv_zoid_num][timestep] : lmp->atom_stencil_md[recv_zoid_num][NUM_TIMESTEPS_IN_PARALLEL - timestep];

            bool bin_is_local_bin = recv_atom->bin_to_local_idxs.count(bin);
            if (bin_is_local_bin) {
                bin_to_send_zoids[bin].insert(recv_zoid_num);
            }
        }
    }

    for (auto& [bin, send_zoids] : bin_to_send_zoids) {
        auto bin_idx = get_bin_idx(bin);
        zoid.bin_to_num_send_zoids[timestep][bin_idx] = send_zoids.size();
        zoid.bin_to_send_zoids[timestep][bin_idx] = new int[send_zoids.size()];
        int idx = 0;
        for (auto& send_zoid : send_zoids) {
            zoid.bin_to_send_zoids[timestep][bin_idx][idx++] = send_zoid;
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

                bool in_bounds = (pos[dim] >= lo && pos[dim] < hi);

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

                bool in_bounds = (pos[dim] >= lo && pos[dim] < hi);

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

                bool in_bounds = (pos[dim] >= lo && pos[dim] < hi);

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

                bool in_bounds = (pos[dim] >= lo && pos[dim] < hi);

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
    stencilMD->ATOM_SETTINGS();
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

    stencilMD->INIT_DOMAIN_BOUNDS();

    stencilMD->INIT_ALL();

    stencilMD->SETUP();

    MPI_Barrier(world);

    stencilMD->GET_LOCAL_ATOMS_ZOID();

    // check to make sure each timestep has all of the local atoms needed
    for (int t = 0; t < NUM_TIMESTEPS_IN_PARALLEL + 1; t++) {
        int total = 0;
        for (int dep = 0; dep < NUM_DEPS; dep++) {
            for (int j = 0; j < lmp->queues[dep].size(); j++) {
                queue_info& zoid = lmp->queues[dep][j];
                if (zoid.num % comm->nprocs == comm->me) {
                    Atom* atom_ = lmp->atom_stencil_md[zoid.num][t];
                    total += atom_->nlocal;
                    std::set<int> tags;
                    for (int i = 0; i < atom_->nlocal; i++) {
                        assert(atom_->tag[i] >= 0 && atom_->tag[i] <= atom->natoms);
                        tags.insert(atom_->tag[i]);
                    }
                    assert(tags.size() == atom_->nlocal);
                }
            }
        }

        int* atoms_tags = new int[atom->natoms + 1];
        for (int i = 0; i < atom->natoms + 1; i++) {
            atoms_tags[i] = 0;
        }

        for (int dep = 0; dep < NUM_DEPS; dep++) {
            for (int j = 0; j < lmp->queues[dep].size(); j++) {
                queue_info& zoid = lmp->queues[dep][j];
                if (zoid.num % comm->nprocs == comm->me) {
                    Atom* atom_ = lmp->atom_stencil_md[zoid.num][t];
                    for (int k = 0; k < atom_->nlocal; k++) {
                        // TODO: multiple zoids have the same set of local atoms. ok what the fuck.
                        assert(atoms_tags[atom_->tag[k]] == 0);
                        atoms_tags[atom_->tag[k]] = 1;
                    }
                }
            }
        }

        MPI_Allreduce(MPI_IN_PLACE, atoms_tags, atom->natoms + 1, MPI_INT, MPI_SUM, world);

        if (comm->me == 0) {
            int num_nonzero = 0;
            for (int tmp_idx = 0; tmp_idx < atom->natoms + 1; tmp_idx++) {
                if (atoms_tags[tmp_idx] == 0) {
                    num_nonzero++;
                } else {
                    assert(atoms_tags[tmp_idx] == 1);
                }
            }
        }

        delete[] atoms_tags;

        MPI_Allreduce(MPI_IN_PLACE, &total, 1, MPI_INT, MPI_SUM, world);
        if (total != atom->natoms) {
            std::cout << "timestep: " << t << " num atoms I have: " << total
                      << " num atoms: " << atom->natoms << std::endl;
        }
        assert(total == atom->natoms);
    }

    MPI_Barrier(world);

    stencilMD->GET_GHOST_ATOMS_ZOID();

    stencilMD->SORT_LOCAL_ATOMS_BINS();

    // check atom map is correct
    /*
    for (int t = 0; t < NUM_TIMESTEPS_IN_PARALLEL + 1; t++) {
        for (int dep = 0; dep < NUM_DEPS; dep++) {
            for (int j = 0; j < lmp->queues[dep].size(); j++) {
                queue_info& zoid = lmp->queues[dep][j];
                if (zoid.num % comm->nprocs == comm->me) {
                    Atom* atom_ = lmp->atom_stencil_md[zoid.num][t];
                    auto map_arr = atom_->get_map_array();
                    for (int i = 0; i < atom_->nlocal + atom_->nghost; i++) {
                        assert(atom_->map(atom_->tag[i]) == i);
                    }
                }
            }
        }
    }
    */

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
        std::cout << YELLOW
                  << "--------------------------------------------------------------------------------------------------"
                  << RESET_COLOR << std::endl;
    }

    for (int t = 0; t < NUM_TIMESTEPS_IN_PARALLEL + 1; t++) {
        for (int dep = 0; dep < NUM_DEPS; dep++) {
            for (int j = 0; j < lmp->queues[dep].size(); j++) {
                queue_info& zoid = lmp->queues[dep][j];
                int zoid_num = zoid.num;
                assert(zoid_num >= 0 && zoid_num < NUM_ZOIDS);
                // receive only if the zoid belongs to me
                if (zoid_num % comm->nprocs == comm->me) {
                    Atom* atom_ = lmp->atom_stencil_md[zoid_num][t];
                    atom_->map_init_stencil_md();
                    atom_->map_set();
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
                    if (comm->nprocs == 1) {
                        // currently this is only tested for local
                        Atom* prev;
                        if (t == 0) {
                            prev = lmp->atom_stencil_md[zoid_num][t + 1];
                        } else {
                            prev = lmp->atom_stencil_md[zoid_num][t - 1];
                        }
                        stencilMD->sort_ghost_bins(zoid, t, atom_, prev);
                        sort_ghost_atoms_stencil_md_bins(atom_, zoid, t);
                    } else {
                        sort_ghost_atoms_stencil_md(atom_, NULL, zoid, t);
                    }
                }
            }
        }
    }

    for (int t = 0; t < NUM_TIMESTEPS_IN_PARALLEL + 1; t++) {
        for (int dep = 0; dep < NUM_DEPS; dep++) {
            for (int j = 0; j < lmp->queues[dep].size(); j++) {
                queue_info& zoid = lmp->queues[dep][j];
                int zoid_num = zoid.num;
                assert(zoid_num >= 0 && zoid_num < NUM_ZOIDS);
                // receive only if the zoid belongs to me
                if (zoid_num % comm->nprocs == comm->me) {
                    Atom* atom_ = lmp->atom_stencil_md[zoid_num][t];
                    atom_->map_init_stencil_md();
                    atom_->map_set();
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

    stencilMD->CREATE_ATOM_IDX_MAPPING();

    stencilMD->SET_INUM_PER_TIMESTEP();
    stencilMD->SET_INUM_PER_TIMESTEP_NEXT_DT();

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
                    zoid.send_segment_types[t] = new bool*[num_send_neighbors];
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

    std::cout << "send list stencilmd" << std::endl;

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

    std::cout << "second send list stencilmd" << std::endl;
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
                    zoid.send_segment_types[t] = new bool*[num_send_neighbors];
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

    for (int dep = 0; dep < NUM_DEPS; dep++) {
        for (int j = 0; j < lmp->queues[dep].size(); j++) {
            queue_info& zoid = lmp->queues[dep][j];
            int zoid_num = zoid.num;
            if (zoid_num % comm->nprocs == comm->me) {
                if (comm->nprocs == 1) {
                    for (int t = 0; t < NUM_TIMESTEPS_IN_PARALLEL + 1; t++) {
                        Atom* atom_ = lmp->atom_stencil_md[zoid_num][t];
                        construct_send_force_bins(true, atom_, zoid, t);
                        construct_send_pos_bins(true, atom_, zoid, t);
                        construct_send_vel_bins(true, atom_, zoid, t);

                        construct_recv_force_bins(true, atom_, zoid, t);
                        construct_recv_pos_bins(true, atom_, zoid, t);
                        construct_recv_vel_bins(true, atom_, zoid, t);

                        construct_bin_to_idx(true, atom_, zoid, t);

                        construct_no_comm_bins(true, atom_, zoid, t);

                        construct_dtfm_cache(atom_);
                    }
                } else {
                    for (int t = 0; t < NUM_TIMESTEPS_IN_PARALLEL + 1; t++) {
                        Atom* atom_ = lmp->atom_stencil_md[zoid_num][t];
                        construct_dtfm_cache(atom_);
                    }
                }
            }
        }
    }

    for (int dep = 0; dep < NUM_DEPS; dep++) {
        for (int j = 0; j < lmp->queues_next_dt[dep].size(); j++) {
            queue_info& zoid = lmp->queues_next_dt[dep][j];
            int zoid_num = zoid.num;
            if (zoid_num % comm->nprocs == comm->me) {
                if (comm->nprocs == 1) {
                    for (int t = 0; t < NUM_TIMESTEPS_IN_PARALLEL + 1; t++) {
                        Atom* atom_ = lmp->atom_stencil_md[zoid_num][NUM_TIMESTEPS_IN_PARALLEL - t];
                        construct_send_force_bins(false, atom_, zoid, t);
                        construct_send_pos_bins(false, atom_, zoid, t);
                        construct_send_vel_bins(false, atom_, zoid, t);

                        construct_recv_force_bins(false, atom_, zoid, t);
                        construct_recv_pos_bins(false, atom_, zoid, t);
                        construct_recv_vel_bins(false, atom_, zoid, t);

                        construct_bin_to_idx(false, atom_, zoid, t);

                        construct_no_comm_bins(false, atom_, zoid, t);
                    }
                }
            }
        }
    }

    MPI_Barrier(world);

    for (int dep = 0; dep < NUM_DEPS; dep++) {
        for (int j = 0; j < lmp->queues[dep].size(); j++) {
            queue_info& zoid = lmp->queues[dep][j];
            int zoid_num = zoid.num;
            if (zoid_num % comm->nprocs == comm->me) {
                if (comm->nprocs == 1) {
                    for (int t = 0; t < NUM_TIMESTEPS_IN_PARALLEL + 1; t++) {
                        Atom* atom_ = lmp->atom_stencil_md[zoid_num][t];
                        construct_bin_to_send_zoids(true, atom_, zoid, t);
                    }
                }
            }
        }
    }

    for (int dep = 0; dep < NUM_DEPS; dep++) {
        for (int j = 0; j < lmp->queues_next_dt[dep].size(); j++) {
            queue_info& zoid = lmp->queues_next_dt[dep][j];
            int zoid_num = zoid.num;
            if (zoid_num % comm->nprocs == comm->me) {
                if (comm->nprocs == 1) {
                    for (int t = 0; t < NUM_TIMESTEPS_IN_PARALLEL + 1; t++) {
                        Atom* atom_ = lmp->atom_stencil_md[zoid_num][NUM_TIMESTEPS_IN_PARALLEL - t];
                        construct_bin_to_send_zoids(false, atom_, zoid, t);
                    }
                }
            }
        }
    }

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
                            bool segment_type =
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
                            bool segment_type =
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
                            bool segment_type = zoid.send_segment_types[t][i][k];
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
                                bool segment_type =
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
                                    }
                                } else {
                                    int local_size =
                                        zoid.send_segment_sizes[t][i][k];
                                    for (int h = 0; h < local_size; h++) {
                                        int local_idx =
                                            local_list[local_list_idx++];

                                        auto& vec = local_idxs_sent_to_procs
                                            [neighbor_proc];

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
                                    bool segment_type =
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
                                    bool segment_type =
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
                                    bool segment_type =
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
                            new bool[num_local_segments + num_ghost_segments];
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
                        new bool*[recv_from.size()];
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
                        for (int k = 0; k < zoid.recv_ghost_num_segments[t][i]; k++) {
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
                            new bool[num_local_segments_buf +
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
                            bool segment_type = zoid.send_segment_types[t][i][k];
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
                                bool segment_type =
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
                                    }
                                } else {
                                    int local_size =
                                        zoid.send_segment_sizes[t][i][k];
                                    for (int h = 0; h < local_size; h++) {
                                        int local_idx =
                                            local_list[local_list_idx++];

                                        auto& vec = local_idxs_sent_to_procs
                                            [neighbor_proc];

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
                                    bool segment_type =
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
                                    bool segment_type =
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
                                    bool segment_type =
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
                            new bool[num_local_segments + num_ghost_segments];
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
                        new bool*[recv_from.size()];
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
                            new bool[num_local_segments_buf +
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

    /*
    if (DO_WARMUP_PAIR_CALC) {
        std::cout << "Prepping forces for each timestep" << std::endl;
        for (int t = 0; t < NUM_TIMESTEPS_IN_PARALLEL + 1; t++) {
            for (int dep = 0; dep < NUM_DEPS; dep++) {
                for (int j = 0; j < lmp->queues[dep].size(); j++) {
                    queue_info& zoid = lmp->queues[dep][j];
                    int zoid_num = zoid.num;
                    if (zoid_num % comm->nprocs == comm->me) {
                        Force* force_ = lmp->force_stencil_md[zoid_num][t];
                        // Warm up?
                        int curr_dt_flag = 0;
                        force_->pair->compute_stencil_md(
                            eflag, vflag, lmp->atom_stencil_md[zoid_num][t],
                            zoid.can_eval_center[t],
                            lmp->zoid_num_to_zoid[zoid_num], &curr_dt_flag);
                        force_clear_stencil_md(
                            lmp->atom_stencil_md[zoid_num][t], force_,
                            lmp->neighbor_stencil_md[zoid_num][t]);
                    }
                }
            }
        }

        std::cout << "done prepping forces" << std::endl;
    }
    */

    // doing a round of next_dt communication to prep the buffers to have enough size
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
            comm->receive_data_process_stencil_md(false, 1, NUM_TIMESTEPS_IN_PARALLEL + 1,
                                                  &receive_requests_next_dt[i], recv_zoid_num);
        }
    }

    for (int dep = 0; dep < NUM_DEPS; dep++) {
        if (dep > 0) {
            auto begin = std::chrono::high_resolution_clock::now();
            for (int idx: dep_to_wait_idxs_next_dt[dep]) {
                int recv_zoid_num = lmp->recv_from_neighbors_procs_next_dt[idx];
                MPI_Wait(&receive_requests_next_dt[idx], MPI_STATUS_IGNORE);
                comm->unpack_data_process_stencil_md(false, 1, NUM_TIMESTEPS_IN_PARALLEL + 1, recv_zoid_num);
            }
        }

        for (int j = 0; j < lmp->queues_next_dt[dep].size(); j++) {
            queue_info& zoid = lmp->queues_next_dt[dep][j];
            int zoid_num = zoid.num;
            if (zoid_num % comm->nprocs == comm->me) {
                if (dep < NUM_DEPS - 1) {
                    auto comm_ = lmp->comm_stencil_md[zoid_num];
                    auto &atom_arr = lmp->atom_stencil_md[zoid_num];

                    int vec_idx = 0;
                    for (int proc = 0; proc < comm->nprocs; proc++) {
                        bool sent = comm_->send_data_to_process_stencil_md(false,
                                                                           atom_arr,
                                                                           lmp->zoid_num_to_zoid_next_dt[zoid_num],
                                                                           &send_requests_next_dt[zoid_num][vec_idx],
                                                                           proc, false);
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
            comm->receive_data_process_stencil_md(true, 0, NUM_TIMESTEPS_IN_PARALLEL + 1,
                                                  &receive_requests[i], recv_zoid_num);
        }
    }

    // map dependency levels to number of zoids to wait on
    std::vector<int> dep_to_wait_idxs[NUM_DEPS];
    std::set<int> zoids_already_waiting_on;

    std::vector<int> zoid_to_wait_idxs[NUM_ZOIDS];

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

                            zoid_to_wait_idxs[zoid_num].push_back(i);
                            break;
                        }
                    }
                }
            }
        }
    }

    for (int i = 0; i < NUM_ZOIDS; i++) {
        if (i % comm->nprocs == comm->me) {
            std::stringstream s;
            for (auto& x : zoid_to_wait_idxs[i]) {
                s << x << " ";
            }
            std::cout << BOLDCYAN << "zoid: " << i << " wait on: " << s.str() << RESET_COLOR << std::endl;
        }

        MPI_Barrier(world);
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
        if (comm->nprocs != 1) {
            if (dep > 0) {
                for (int idx : dep_to_wait_idxs[dep]) {
                    int recv_zoid_num = lmp->recv_from_neighbors_procs[idx];
                    MPI_Wait(&receive_requests[idx], MPI_STATUS_IGNORE);
                    comm->unpack_data_process_stencil_md(true, 0, NUM_TIMESTEPS_IN_PARALLEL + 1, recv_zoid_num);
                }
            }
        }

        for (int j = 0; j < lmp->queues[dep].size(); j++) {
            queue_info& zoid = lmp->queues[dep][j];
            int zoid_num = zoid.num;
            if (zoid_num % comm->nprocs == comm->me) {
                Atom* atom_ = lmp->atom_stencil_md[zoid_num][0];
                Force* force_ = lmp->force_stencil_md[zoid_num][0];
                Neighbor* neighbor_ = lmp->neighbor_stencil_md[zoid_num][0];
                Comm *comm_ = lmp->comm_stencil_md[zoid_num];
#ifdef LMP_OPENMP
                Modify* modify_ = lmp->modify_stencil_md_omp[zoid_num][0];
#else
                Modify* modify_ = lmp->modify_stencil_md[zoid_num][0];
#endif

                if (comm->nprocs == 1) {
                    comm_->recv_data_bins_stencil_md(true, lmp->atom_stencil_md[zoid_num],
                                                     lmp->zoid_num_to_zoid[zoid_num],
                                                     0, NUM_TIMESTEPS_IN_PARALLEL + 1);
                }

                // todo: eflag and vflag might cause some issues
                // TODO: compute force for each pair in parallel

                stencilMD->fuse_force_computation_atomics<true>(zoid, 0, lmp->atom_stencil_md[zoid_num][0], neighbor_, force_, modify_);
                /*
                int curr_dt_flag = 0;
                force_->pair->compute_stencil_md(
                    eflag, vflag, lmp->atom_stencil_md[zoid_num][0],
                    zoid.can_eval_center[0], lmp->zoid_num_to_zoid[zoid_num],
                    &curr_dt_flag);
                */

                if (atom->molecular != Atom::ATOMIC) {
                    if (force->bond) {
                        /*
                        force_->bond->compute_stencil_md(eflag, vflag, atom_,
                                                         zoid.can_eval_center[0],
                                                         lmp->zoid_num_to_zoid[zoid_num],
                                                         &curr_dt_flag, neighbor_);
                        */
                    }
                    if (force->angle) {
                        assert(false);
                        force->angle->compute(eflag, vflag);
                    }
                    if (force->dihedral) {
                        assert(false);
                        force->dihedral->compute(eflag, vflag);
                    }
                    if (force->improper) {
                        assert(false);
                        force->improper->compute(eflag, vflag);
                    }
                }

                modify_->setup_stencil_md(vflag, atom_);


                if (dep < NUM_DEPS - 1) {
                    if (comm->nprocs == 1) {
                        /*
                        comm_->send_data_bins_stencil_md(true, lmp->atom_stencil_md[zoid_num],
                                                         lmp->zoid_num_to_zoid[zoid_num],
                                                         0, NUM_TIMESTEPS_IN_PARALLEL + 1);
                        */
                    } else {
                        int vec_idx = 0;

                        for (int proc = 0; proc < comm->nprocs; proc++) {
                            bool sent = comm_->send_data_to_process_stencil_md(true,
                                                                               lmp->atom_stencil_md[zoid_num],
                                                                               lmp->zoid_num_to_zoid[zoid_num],
                                                                               &send_requests[zoid_num][vec_idx], proc,
                                                                               true);
                            if (sent) {
                                vec_idx++;
                            }
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

    // modify setup after comm of forces
    // stencilMD->MODIFY_SETUP(vflag);

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

    if (TEST_AGAINST_LAMMPS_LOCAL) {
        for (int dep = 0; dep < NUM_DEPS; dep++) {
            for (int j = 0; j < lmp->queues[dep].size(); j++) {
                queue_info &zoid = lmp->queues[dep][j];
                int zoid_num = zoid.num;
                if (zoid_num % comm->nprocs == comm->me) {
                    Atom *atom_ = lmp->atom_stencil_md[zoid_num][0];
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
        std::cout << GREEN << "-------- SETUP STENCIL MD PASSED ---------" << RESET_COLOR << std::endl;
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

    // stencilMD->MODIFY_SETUP(vflag);
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

void parallel_memset(void* buf, int num_bytes) {
    constexpr int chunk_size = 8192;

    int num_chunks_f = 1 + num_bytes / chunk_size;

    #pragma cilk grainsize 1
    cilk_for (int tid = 0; tid < num_chunks_f; tid++) {
        int ifrom = tid * chunk_size;
        int ito = ((ifrom + chunk_size) > num_bytes) ? num_bytes : ifrom + chunk_size;
        memset((char*) buf + ifrom, 0, ito - ifrom);
    }
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
    double* test_v[test_num_timesteps];
    double* send_f;
    double* send_x;
    double* send_v;

    if (TEST_AGAINST_LAMMPS_LOCAL) {
        for (int i = 0; i < test_num_timesteps; i++) {
            test_f[i] = new double[3 * (atom->natoms + 1)];
            test_x[i] = new double[3 * (atom->natoms + 1)];
            test_v[i] = new double[3 * (atom->natoms + 1)];
            for (int j = 0; j < 3 * (atom->natoms + 1); j++) {
                test_f[i][j] = 0.0;
                test_x[i][j] = 0.0;
                test_v[i][j] = 0.0;
            }
        }

        send_f = new double[3 * (atom->natoms + 1)];
        for (int i = 0; i < 3 * (atom->natoms + 1); i++) {
            send_f[i] = 0;
        }

        send_x = new double[3 * (atom->natoms + 1)];
        for (int i = 0; i < 3 * (atom->natoms + 1); i++) {
            send_x[i] = 0;
        }

        send_v = new double[3 * (atom->natoms + 1)];
        for (int i = 0; i < 3 * (atom->natoms + 1); i++) {
            send_v[i] = 0;
        }
    }

    int64_t lammps_pair_duration = 0;
    int64_t lammps_bond_duration = 0;
    int64_t lammps_comm_duration = 0;
    int64_t lammps_forward_comm_duration = 0;
    int64_t lammps_reverse_comm_duration = 0;
    int64_t lammps_modify_initial_integrate_duration = 0;
    int64_t lammps_modify_final_integrate_duration = 0;
    int64_t lammps_modify_pre_force_duration = 0;
    int64_t lammps_modify_post_force_duration = 0;
    int64_t lammps_num_atoms = 0;

    // for (int i = 0; i < n; i++) {
    auto begin_lammps = std::chrono::high_resolution_clock::now();
    // cilk_scope {
            for (int i = 0; i < n + 1; i++) {
                if (ONLY_RUN_STENCIL_MD) {
                    break;
                }
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
                    for (int j = 0; j < 3 * (atom->natoms + 1); j++) {
                        send_v[j] = 0;
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

                        send_v[tag * 3 + 0] = atom->v[j][0];
                        send_v[tag * 3 + 1] = atom->v[j][1];
                        send_v[tag * 3 + 2] = atom->v[j][2];
                    }

                    MPI_Allreduce(send_f, test_f[i], (atom->natoms + 1) * 3, MPI_DOUBLE,
                                  MPI_SUM, world);

                    MPI_Allreduce(send_x, test_x[i], (atom->natoms + 1) * 3, MPI_DOUBLE,
                                  MPI_SUM, world);

                    MPI_Allreduce(send_v, test_v[i], (atom->natoms + 1) * 3, MPI_DOUBLE,
                                  MPI_SUM, world);
                }

                if (i == n) {
                    break;
                }
                // end stencil md code

                // auto begin_m = std::chrono::high_resolution_clock::now();
                modify->initial_integrate(vflag);
                // auto end_m = std::chrono::high_resolution_clock::now();
                // auto duration_m = std::chrono::duration_cast<std::chrono::microseconds>(end_m - begin_m).count();
                // lammps_modify_initial_integrate_duration += duration_m;
                if (n_post_integrate) {
                    assert(false);
                    modify->post_integrate();
                }
                timer->stamp(Timer::MODIFY);

                // regular communication vs neighbor list rebuild

                nflag = neighbor->decide();

                if (nflag == 0) {
                    timer->stamp();
                    // auto begin = std::chrono::high_resolution_clock::now();
                    comm->forward_comm();
                    // auto end = std::chrono::high_resolution_clock::now();
                    // auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count();
                    // lammps_comm_duration += duration;
                    // lammps_forward_comm_duration += duration;
                    // lammps_forward_comm_times.push_back(duration);
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
                    if (sortflag && ntimestep >= atom->nextsort) {
                        atom->sort();
                    }
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
                    // auto begin = std::chrono::high_resolution_clock::now();
                    modify->pre_force(vflag);
                    // auto end = std::chrono::high_resolution_clock::now();
                    // auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count();
                    // lammps_modify_pre_force_duration += duration;
                    timer->stamp(Timer::MODIFY);
                }

                if (pair_compute_flag) {
                    // auto begin = std::chrono::high_resolution_clock::now();
                    if (!LAMMPS_USE_BINS) {
                        force->pair->compute(eflag, vflag);
                    } else {
                        assert(false);
                        // stencilMD->lammps_fuse_reduce();
                        stencilMD->lammps_fuse_reduce2();
                        // stencilMD->lammps_fuse_force_compute_lammps_bins_split();
                        // stencilMD->lammps_fuse_force_compute_lammps_bins();
                        // stencilMD->lammps_fuse_force_compute();
                        // stencilMD->lammps_fuse_force_compute2();
                        // stencilMD->lammps_fuse_force_compute_atomics();
                    }
                    // auto end = std::chrono::high_resolution_clock::now();
                    // auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count();
                    // lammps_pair_duration += duration;
                    // lammps_num_atoms += atom->nlocal;
                    timer->stamp(Timer::PAIR);
                }

                if (atom->molecular != Atom::ATOMIC) {
                    if (force->bond) {
                        // auto begin = std::chrono::high_resolution_clock::now();
                        if (!LAMMPS_USE_BINS) {
                            force->bond->compute(eflag, vflag);
                        } else {
                            assert(false);
                        }
                        // auto end = std::chrono::high_resolution_clock::now();
                        // auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count();
                        // lammps_bond_duration += duration;
                    }
                    if (force->angle) {
                        assert(false);
                        force->angle->compute(eflag, vflag);
                    }
                    if (force->dihedral) {
                        assert(false);
                        force->dihedral->compute(eflag, vflag);
                    }
                    if (force->improper) {
                        assert(false);
                        force->improper->compute(eflag, vflag);
                    }
                    timer->stamp(Timer::BOND);
                }

                if (kspace_compute_flag) {
                    assert(false);
                    force->kspace->compute(eflag, vflag);
                    timer->stamp(Timer::KSPACE);
                }

                if (n_pre_reverse) {
                    assert(false);
                    modify->pre_reverse(eflag, vflag);
                    timer->stamp(Timer::MODIFY);
                }

                // reverse communication of forces
                if (force->newton) {
                    // auto begin = std::chrono::high_resolution_clock::now();
                    comm->reverse_comm();
                    // auto end = std::chrono::high_resolution_clock::now();
                    // auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count();
                    // lammps_comm_duration += duration;
                    // lammps_reverse_comm_duration += duration;
                    // lammps_reverse_comm_times.push_back(duration);
                    timer->stamp(Timer::COMM);
                }

                // force modifications, final time integration, diagnostics
                if (n_post_force_any) {
                    // auto begin = std::chrono::high_resolution_clock::now();
                    modify->post_force(vflag);
                    // auto end = std::chrono::high_resolution_clock::now();
                    // auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count();
                    // lammps_modify_post_force_duration += duration;
                }

                // auto begin_m2 = std::chrono::high_resolution_clock::now();
                modify->final_integrate();
                // auto end_m2 = std::chrono::high_resolution_clock::now();
                // auto duration_m2 = std::chrono::duration_cast<std::chrono::microseconds>(end_m2 - begin_m2).count();
                // lammps_modify_final_integrate_duration += duration_m2;
                if (n_end_of_step) {
                    // modify->end_of_step();
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
    // }

    auto end_lammps = std::chrono::high_resolution_clock::now();
    auto duration_lammps = std::chrono::duration_cast<std::chrono::microseconds>(end_lammps - begin_lammps).count();

    int64_t total_duration_lammps = 0;
    MPI_Allreduce(&duration_lammps, &total_duration_lammps, 1, MPI_INT64_T, MPI_SUM, world);

    std::cout << "lammps total just running the thing: " << duration_lammps << " microseconds. " << " total duration: " << total_duration_lammps << std::endl;

    int64_t total_comm_duration = 0;
    int64_t total_forward_comm_duration = 0;
    int64_t total_reverse_comm_duration = 0;

    int64_t total_pair_duration = 0;
    int64_t total_bond_duration = 0;
    int64_t total_modify_initial_integrate_duration = 0;
    int64_t total_modify_final_integrate_duration = 0;
    int64_t total_modify_pre_force_duration = 0;
    int64_t total_modify_post_force_duration = 0;

    MPI_Allreduce(&lammps_comm_duration, &total_comm_duration, 1, MPI_INT64_T, MPI_SUM, world);
    MPI_Allreduce(&lammps_forward_comm_duration, &total_forward_comm_duration, 1, MPI_INT64_T, MPI_SUM, world);
    MPI_Allreduce(&lammps_reverse_comm_duration, &total_reverse_comm_duration, 1, MPI_INT64_T, MPI_SUM, world);

    MPI_Allreduce(&lammps_pair_duration, &total_pair_duration, 1, MPI_INT64_T, MPI_SUM, world);
    MPI_Allreduce(&lammps_bond_duration, &total_bond_duration, 1, MPI_INT64_T, MPI_SUM, world);
    MPI_Allreduce(&lammps_modify_initial_integrate_duration, &total_modify_initial_integrate_duration, 1, MPI_INT64_T, MPI_SUM, world);
    MPI_Allreduce(&lammps_modify_final_integrate_duration, &total_modify_final_integrate_duration, 1, MPI_INT64_T, MPI_SUM, world);
    MPI_Allreduce(&lammps_modify_pre_force_duration, &total_modify_pre_force_duration, 1, MPI_INT64_T, MPI_SUM, world);
    MPI_Allreduce(&lammps_modify_post_force_duration, &total_modify_post_force_duration, 1, MPI_INT64_T, MPI_SUM, world);

    if (comm->me == 0) {
        std::cout << GREEN << "process: " << comm->me
                  << " LAMMPS COMM DURATION: " << lammps_comm_duration << " forward: " << lammps_forward_comm_duration
                  << " reverse: " << lammps_reverse_comm_duration
                  << " microseconds. " << " total comm duration: " << total_comm_duration
                  << " total forward comm: " << total_forward_comm_duration << " total reverse comm: "
                  << total_reverse_comm_duration << RESET_COLOR << std::endl;

        std::cout << YELLOW
                  << "lammps pair duration: " << lammps_pair_duration << " bond duration: " << lammps_bond_duration << " ratio: " << (double) lammps_num_atoms / lammps_pair_duration
                  << " " << " total pair duration: " << total_pair_duration << " total bond duration: " << total_bond_duration << RESET_COLOR
                  << std::endl;

        std::cout << YELLOW
                  << " lammps modify initial integrate duration: " << lammps_modify_initial_integrate_duration
                  << " lammps modify final integrate duration: " << lammps_modify_final_integrate_duration
                  << " pre force duration: " << lammps_modify_pre_force_duration
                  << " post force duration: " << lammps_modify_post_force_duration
                  << " total initial integrate: " << total_modify_initial_integrate_duration
                  << " total final integrate: " << total_modify_final_integrate_duration
                  << " total modify pre force duration: " << total_modify_pre_force_duration
                  << " total modify post force duration: " << total_modify_post_force_duration
                  << RESET_COLOR << std::endl;
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

    if (TEST_AGAINST_LAMMPS_LOCAL) {
        delete[] send_f;
        delete[] send_x;
        delete[] send_v;
    }

    MPI_Barrier(world);
    if (ONLY_RUN_LAMMPS) {
        return;
    }

    // setup data structures to run stencil md
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
                                      recv_zoid_num) != recv_from.end()
                                      && zoids_already_waiting_on.find(recv_zoid_num) == zoids_already_waiting_on.end()) {
                            dep_to_wait_idxs[dep].push_back(i);
                            zoids_already_waiting_on.insert(recv_zoid_num);
                            break;
                        }
                    }
                }
            }
        }
    }

    if (comm->me == 0) {
        for (int dep = 1; dep < NUM_DEPS; dep++) {
            int num_directly_wait_on = 0;
            for (int idx: dep_to_wait_idxs[dep]) {
                int recv_zoid_num = lmp->recv_from_neighbors_procs[idx];
                if (get_zoid_dep(recv_zoid_num) == dep - 1 && recv_zoid_num % comm->nprocs != comm->me) {
                    num_directly_wait_on++;
                }
            }

            std::cout << "dep: " << dep << " directly wait on: " << num_directly_wait_on << std::endl;
        }
    }

    std::cout << "------------------------------------------------------------------------" << std::endl;

    MPI_Barrier(world);

    // map dependency levels to number of zoids to wait on
    std::vector<int> dep_to_wait_idxs_next_dt[NUM_DEPS];
    std::set<int> zoids_already_waiting_on_next_dt;

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
                            break;
                        }
                    }
                }
            }
        }
    }

    MPI_Barrier(world);

    if (comm->me == 0) {
        std::cout << BOLDYELLOW << "------ RUN STENCILMD -------" << RESET_COLOR << std::endl;
    }

    auto begin = std::chrono::high_resolution_clock::now();
    cilk_scope {
    run_stencil_md_pipelined(n, dep_to_wait_idxs, dep_to_wait_idxs_next_dt, test_f, test_x, test_v);
    }
    // run_stencil_md_no_cilk_for(n, test_f, test_x, test_v);
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count();

    int64_t total_duration_stencil_md = 0;
    MPI_Allreduce(&duration, &total_duration_stencil_md, 1, MPI_INT64_T, MPI_SUM, world);

    std::cout << "me: " << comm->me << " stencil md total just running the thing: " << duration << " microseconds. " << " unpack duration? " << unpack_duration << " total duration: " << total_duration_stencil_md << std::endl;

    MPI_Barrier(world);

    int64_t stencil_md_total_compute_time_curr_dt_dep[NUM_DEPS] = {0};
    int64_t stencil_md_total_compute_time_next_dt_dep[NUM_DEPS] = {0};

    int64_t stencil_md_total_num_atoms_curr_dt_dep[NUM_DEPS] = {0};
    int64_t stencil_md_total_num_atoms_next_dt_dep[NUM_DEPS] = {0};
    /*
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

        stencil_md_total_num_atoms_curr_dt_dep[dep] = total_num_atoms_curr_dt_dep;
        stencil_md_total_num_atoms_next_dt_dep[dep] = total_num_atoms_next_dt_dep;
    }
    */

    if (TIME_STENCIL_MD) {
        /*
        for (int zoid_num = 0; zoid_num < NUM_ZOIDS; zoid_num++) {
            if (zoid_num % comm->nprocs == comm->me) {
                for (int t = 1; t < NUM_TIMESTEPS_IN_PARALLEL + 1; t++) {
                    std::cout << GREEN << "curr dt zoid: " << zoid_num << " timestep: " << t << " running time: " << curr_dt_compute_dep_time[zoid_num][t]
                              << " num edges: " << curr_dt_num_edges[zoid_num][t]
                              << " ratio: " << (double)curr_dt_num_edges[zoid_num][t] / curr_dt_compute_dep_time[zoid_num][t] << RESET_COLOR << std::endl;

                }

                for (int t = 1; t < NUM_TIMESTEPS_IN_PARALLEL + 1; t++) {
                    std::cout << GREEN << "next dt zoid: " << zoid_num << " timestep: " << t << " running time: " << next_dt_compute_dep_time[zoid_num][t]
                              << " num edges: " << next_dt_num_edges[zoid_num][t]
                              << " ratio: " << (double)next_dt_num_edges[zoid_num][t] / next_dt_compute_dep_time[zoid_num][t] << RESET_COLOR << std::endl;
                }
            }

            MPI_Barrier(world);
        }
        */
    }

    int64_t stencil_md_total_send_comm_duration = 0;
    int64_t stencil_md_total_recv_comm_duration = 0;

    int64_t stencil_md_total_pair_duration = 0;
    int64_t stencil_md_total_bond_duration = 0;

    int64_t stencil_md_total_modify_initial_duration = 0;
    int64_t stencil_md_total_modify_final_duration = 0;
    int64_t stencil_md_total_modify_pre_force_duration = 0;
    int64_t stencil_md_total_modify_post_force_duration = 0;

    int64_t stencil_md_total_mpi_duration = 0;
    int64_t stencil_md_total_curr_dt_comm_duration = 0;
    int64_t stencil_md_total_next_dt_comm_duration = 0;
    int64_t stencil_md_total_send_pack_duration = 0;
    int64_t stencil_md_total_misc_duration = 0;
    int64_t stencil_md_total_pre_recv_time = 0;
    int64_t stencil_md_total_unpack_self_time = 0;

    // int64_t my_compute_duration = curr_dt_compute_duration + next_dt_compute_duration;

    MPI_Allreduce(&pair_duration, &stencil_md_total_pair_duration, 1, MPI_INT64_T, MPI_SUM, world);
    MPI_Allreduce(&bond_duration, &stencil_md_total_bond_duration, 1, MPI_INT64_T, MPI_SUM, world);
    MPI_Allreduce(&modify_initial_duration, &stencil_md_total_modify_initial_duration, 1, MPI_INT64_T, MPI_SUM, world);
    MPI_Allreduce(&modify_final_duration, &stencil_md_total_modify_final_duration, 1, MPI_INT64_T, MPI_SUM, world);

    MPI_Allreduce(&modify_pre_force_duration, &stencil_md_total_modify_pre_force_duration, 1, MPI_INT64_T, MPI_SUM, world);
    MPI_Allreduce(&modify_post_force_duration, &stencil_md_total_modify_post_force_duration, 1, MPI_INT64_T, MPI_SUM, world);

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
                  << " LOCAL PAIR: " << pair_duration << " LOCAL BOND: " << bond_duration
                  << " TOTAL PAIR: " << stencil_md_total_pair_duration << " TOTAL BOND: " << stencil_md_total_bond_duration
                  << " TOTAL COMM: " << stencil_md_total_send_comm_duration + stencil_md_total_recv_comm_duration
                  << " TOTAL SEND COMM: " << stencil_md_total_send_comm_duration << " TOTAL RECV COMM: " << stencil_md_total_recv_comm_duration
                  << " TOTAL MODIFY INITIAL: " << stencil_md_total_modify_initial_duration
                  << " TOTAL MODIFY FINAL DURATION: " << stencil_md_total_modify_final_duration
                  << " TOTAL MODIFY PRE FORCE: " << stencil_md_total_modify_pre_force_duration
                  << " TOTAL MODIFY POST FORCE: " << stencil_md_total_modify_post_force_duration
                  << " TOTAL MPI DURATION: " << stencil_md_total_mpi_duration
                  << " TOTAL SEND PACK DURATION: " << stencil_md_total_send_pack_duration
                  << " TOTAL MISC DURATION: " << stencil_md_total_misc_duration
                  << " TOTAL PRE-RECV TIME: " << stencil_md_total_pre_recv_time
                  << " TOTAL UNPACK SELF TIME: " << stencil_md_total_unpack_self_time << RESET_COLOR << std::endl;

        std::cout << YELLOW << "CURR DT TOTAL COMM DURATION: " << stencil_md_total_curr_dt_comm_duration
                  << " NEXT DT COMM DURATION: " << stencil_md_total_next_dt_comm_duration << RESET_COLOR << std::endl;
    }

    if (TEST_AGAINST_LAMMPS) {
        for (int i = 0; i < test_num_timesteps; i++) {
            delete[] test_f[i];
            delete[] test_x[i];
            delete[] test_v[i];
        }
    }
}

// assume already have all the data necessary to run the zoid
template <bool curr_dt>
void Verlet::run_stencil_md_zoid(int starting_timestep, int start_eval, int end_eval, int zoid_num, double** test_f, double** test_x, double** test_v) {
    int n_pre_force = modify->n_pre_force;
    int n_post_force_any = modify->n_post_force_any;
    int n_end_of_step = modify->n_end_of_step;

    queue_info& zoid = curr_dt ? lmp->zoid_num_to_zoid[zoid_num] : lmp->zoid_num_to_zoid_next_dt[zoid_num];
    auto& atom_arr = lmp->atom_stencil_md[zoid_num];
    int** atom_idx_mapping = zoid.atom_idx_mapping;

    for (int t = start_eval; t < end_eval; t++) {
        Atom* atom_ = curr_dt ? atom_arr[t] : atom_arr[NUM_TIMESTEPS_IN_PARALLEL - t];
        Atom* atom_next_timestep = curr_dt ? atom_arr[t + 1] : atom_arr[NUM_TIMESTEPS_IN_PARALLEL - t - 1];
        Neighbor* neigh_next_timestep = curr_dt ? lmp->neighbor_stencil_md[zoid_num][t + 1] : lmp->neighbor_stencil_md_next_dt[zoid_num][t + 1];

#ifdef LMP_OPENMP
        Modify* modify_ = curr_dt ? lmp->modify_stencil_md_omp[zoid_num][t + 1] : lmp->modify_stencil_md_omp[zoid_num][NUM_TIMESTEPS_IN_PARALLEL - t - 1];
#else
        Modify* modify_ = lmp->modify_stencil_md[zoid_num];
#endif

        if (TEST_AGAINST_LAMMPS_LOCAL) {
            int timestep_to_compare_against = curr_dt ? starting_timestep + t : starting_timestep + NUM_TIMESTEPS_IN_PARALLEL + t;

            stencilMD->COMPARE_FORCE_AGAINST_LAMMPS(curr_dt, timestep_to_compare_against, atom_, zoid, test_f);
            stencilMD->COMPARE_POS_AGAINST_LAMMPS(curr_dt, timestep_to_compare_against, atom_, zoid, test_x);
            stencilMD->COMPARE_VEL_AGAINST_LAMMPS(curr_dt, timestep_to_compare_against, atom_, zoid, test_v);
        }

        // updates positions in atom_next_timestep
        // modify_->initial_integrate_stencil_md(vflag, atom_, atom_next_timestep, atom_idx_mapping[t], nullptr);

        int f_total = (atom_->nghost) * sizeof(double) * 3;
        // int num_chunks_f = 1 + f_total / chunk_size;
        double* f_ = &(atom_->f[atom_->nlocal][0]);

        int eval_f_total = (atom_next_timestep->nghost) * sizeof(double) * 3;
        // int num_chunks_eval_f = 1 + eval_f_total / chunk_size;
        double* eval_f_ = &(atom_next_timestep->eval_f_stencil_md[atom_next_timestep->nlocal][0]);

        auto begin_initial_integrate = std::chrono::high_resolution_clock::now();
        cilk_scope {
            cilk_spawn stencilMD->initial_integrate_stencil_md(zoid, t + 1, atom_, atom_next_timestep, atom_idx_mapping[t]);

            cilk_spawn parallel_memset(f_, f_total);
            parallel_memset(eval_f_, eval_f_total);
        }

        auto end_initial_integrate = std::chrono::high_resolution_clock::now();
        auto duration_initial_integrate = std::chrono::duration_cast<std::chrono::microseconds>(end_initial_integrate - begin_initial_integrate).count();
        modify_initial_duration += duration_initial_integrate;

        if (n_pre_force) {
            // modify_->pre_force_stencil_md(vflag, atom_next_timestep);
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

            int timestep_flag = t + 1;

            auto begin_compute = std::chrono::high_resolution_clock::now();
            stencilMD->stencil_md_fuse_force_computation_atomics(zoid, t + 1, atom_next_timestep, neigh_next_timestep, next_force, modify_);
            auto end_compute = std::chrono::high_resolution_clock::now();
            auto duration_compute = std::chrono::duration_cast<std::chrono::microseconds>(end_compute - begin_compute).count();
            pair_duration += duration_compute;

            /*
            next_force->pair->compute_stencil_md(
                    eflag, vflag, atom_next_timestep,
                    zoid.can_eval_center[t + 1],
                    zoid, &timestep_flag);
            */

            if (atom->molecular != Atom::ATOMIC) {
                if (force->bond) {
                    /*
                    next_force->bond->compute_stencil_md(eflag, vflag, atom_next_timestep,
                                                         zoid.can_eval_center[t + 1],
                                                         zoid, &timestep_flag, neigh_next_timestep);
                    */
                }
                if (force->angle) {
                    assert(false);
                    force->angle->compute(eflag, vflag);
                }
                if (force->dihedral) {
                    assert(false);
                    force->dihedral->compute(eflag, vflag);
                }
                if (force->improper) {
                    assert(false);
                    force->improper->compute(eflag, vflag);
                }
            }
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

        auto begin_post_force = std::chrono::high_resolution_clock::now();
        if (n_post_force_any) {
            // modify_->post_force_stencil_md(vflag, atom_next_timestep);
            // modify->post_force(vflag);
            // stencilMD->fuse_post_force_stencil_md<curr_dt>(zoid, t + 1, atom_next_timestep, modify_);
            stencilMD->post_force_stencil_md_(atom_next_timestep, modify_);
        }

        auto end_post_force = std::chrono::high_resolution_clock::now();
        auto duration_post_force = std::chrono::duration_cast<std::chrono::microseconds>(end_post_force - begin_post_force).count();
        modify_post_force_duration += duration_post_force;

        /*
        modify_->final_integrate_stencil_md(
                atom_, atom_next_timestep, neighbor, atom_idx_mapping[t], nullptr);
        */

        auto begin_final_integrate = std::chrono::high_resolution_clock::now();
        stencilMD->final_integrate_stencil_md_(atom_next_timestep);
        auto end_final_integrate = std::chrono::high_resolution_clock::now();
        auto duration_final_integrate = std::chrono::duration_cast<std::chrono::microseconds>(end_final_integrate - begin_final_integrate).count();
        modify_final_duration += duration_final_integrate;

        if (n_end_of_step) {
            // this doesn't actually do anything
            // modify->end_of_step();
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

template <bool curr_dt>
void Verlet::run_stencil_md_zoid_no_cilk_for(int starting_timestep, int zoid_num,
                                             double** test_f, double** test_x, double** test_v,
                                             std::array<std::atomic<int>, NUM_ZOIDS>& counters, const std::array<int, NUM_ZOIDS>& cache) {
    assert(counters[zoid_num].load() == 0);
    counters[zoid_num].store(cache[zoid_num], std::memory_order_relaxed);

    int n_pre_force = modify->n_pre_force;
    int n_post_force_any = modify->n_post_force_any;
    int n_end_of_step = modify->n_end_of_step;

    queue_info& zoid = curr_dt ? lmp->zoid_num_to_zoid[zoid_num] : lmp->zoid_num_to_zoid_next_dt[zoid_num];
    auto& atom_arr = lmp->atom_stencil_md[zoid_num];
    int** atom_idx_mapping = zoid.atom_idx_mapping;

    constexpr int t = 0;
    Atom* atom_ = curr_dt ? atom_arr[t] : atom_arr[NUM_TIMESTEPS_IN_PARALLEL - t];
    Atom* atom_next_timestep = curr_dt ? atom_arr[t + 1] : atom_arr[NUM_TIMESTEPS_IN_PARALLEL - t - 1];
    Neighbor* neigh_next_timestep = curr_dt ? lmp->neighbor_stencil_md[zoid_num][t + 1] : lmp->neighbor_stencil_md_next_dt[zoid_num][t + 1];

#ifdef LMP_OPENMP
    Modify* modify_ = curr_dt ? lmp->modify_stencil_md_omp[zoid_num][t + 1] : lmp->modify_stencil_md_omp[zoid_num][NUM_TIMESTEPS_IN_PARALLEL - t - 1];
#else
    Modify* modify_ = lmp->modify_stencil_md[zoid_num];
#endif

    if (TEST_AGAINST_LAMMPS_LOCAL) {
        int timestep_to_compare_against = curr_dt ? starting_timestep + t : starting_timestep + NUM_TIMESTEPS_IN_PARALLEL + t;

        stencilMD->COMPARE_FORCE_AGAINST_LAMMPS(curr_dt, timestep_to_compare_against, atom_, zoid, test_f);
        stencilMD->COMPARE_POS_AGAINST_LAMMPS(curr_dt, timestep_to_compare_against, atom_, zoid, test_x);
        stencilMD->COMPARE_VEL_AGAINST_LAMMPS(curr_dt, timestep_to_compare_against, atom_, zoid, test_v);
    }

    stencilMD->fuse_initial_integrate_stencil_md<curr_dt>(zoid, t, atom_, atom_next_timestep, atom_idx_mapping[t]);

    if (pair_compute_flag) {
        Force* next_force;
        if (!PURELY_LOCAL_POTENTIAL) {
            next_force = curr_dt ? lmp->force_stencil_md[zoid_num][t + 1] : lmp->force_stencil_md[zoid_num][NUM_TIMESTEPS_IN_PARALLEL - t - 1];
        } else {
            next_force = curr_dt ? lmp->force_stencil_md[zoid_num][t + 1] : lmp->force_stencil_md_next_dt[zoid_num][t + 1];
        }

        stencilMD->fuse_force_computation_reduce(zoid, t + 1, atom_next_timestep, neigh_next_timestep, next_force, modify_);
        // stencilMD->fuse_force_computation_reduce_updates<curr_dt>(zoid, t + 1, atom_next_timestep, neigh_next_timestep, next_force, modify_);
    }

    if (n_post_force_any) {
        stencilMD->fuse_post_force_stencil_md<curr_dt>(zoid, t + 1, atom_next_timestep, modify_);
    }

    if (n_end_of_step) {
        // this doesn't actually do anything
        // modify->end_of_step();
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

    int dep = curr_dt ? get_zoid_dep(zoid_num) : get_zoid_dep_next_dt(zoid_num);
    auto& neighbors = curr_dt ? lmp->send_to_neighbors[zoid_num] : lmp->send_to_neighbors_next_dt[zoid_num];
    for (auto& neighbor : neighbors) {
        int dep_neighbor = curr_dt ? get_zoid_dep(neighbor) : get_zoid_dep_next_dt(neighbor);
        if (dep_neighbor == dep + 1) {
            // counters[neighbor]--;
            if (--counters[neighbor] == 0) {
                cilk_spawn run_stencil_md_zoid_no_cilk_for<curr_dt>(starting_timestep, neighbor, test_f, test_x, test_v, counters, cache);
            }
        }
    }
}

template <bool curr_dt>
void Verlet::run_stencil_md_dep_templated(int dep, int start_timestep, int start_t, int end_t, int* dep_to_recv_idx,
                                          std::vector<MPI_Request> *send_requests, std::vector<MPI_Request>& receive_requests,
                                          std::vector<int> *dep_to_wait_idxs, std::vector<int> *dep_to_wait_idxs_next_dt,
                                          double **test_f, double **test_x, double** test_v, int pipeline_stage) {
    constexpr bool SKIP_COMM = false;

    if (comm->nprocs != 1 && !SKIP_COMM) {
        if (dep > 0) {
            auto &wait_idxs = curr_dt ? dep_to_wait_idxs[dep] : dep_to_wait_idxs_next_dt[dep];
            int recv_idx = dep_to_recv_idx[dep];

            auto begin_mpi = std::chrono::high_resolution_clock::now();
            MPI_Waitall(wait_idxs.size(), &receive_requests[recv_idx], MPI_STATUSES_IGNORE);
            auto end_mpi = std::chrono::high_resolution_clock::now();
            auto duration_mpi = std::chrono::duration_cast<std::chrono::microseconds>(end_mpi - begin_mpi).count();
            mpi_duration += duration_mpi;
        }
    }

    // auto &zoid_queue = curr_dt ? lmp->queues[dep] : lmp->queues_next_dt[dep];
    auto &zoid_queue = curr_dt ? lmp->my_queues[dep] : lmp->my_queues_next_dt[dep];
    cilk_for (int j = 0; j < zoid_queue.size(); j++) {
        queue_info &zoid = zoid_queue[j];
        int zoid_num = zoid.num;
        if (zoid_num % comm->nprocs != comm->me) {
            continue;
        }

        auto comm_ = lmp->comm_stencil_md[zoid_num];
        if (SKIP_COMM || comm->nprocs == 1) {

        } else {
            comm_->unpack_data_process_zoid_stencil_md(curr_dt, zoid, start_t, end_t, pipeline_stage);
        }

        run_stencil_md_zoid<curr_dt>(start_timestep, start_t - 1, end_t - 1, zoid_num, test_f, test_x, test_v);

        if (comm->nprocs != 1) {
            if (dep < NUM_DEPS - 1) {
                auto begin = std::chrono::high_resolution_clock::now();
                auto &atom_arr = lmp->atom_stencil_md[zoid_num];

                int vec_idx = 0;
                cilk_for(int
                proc = 0;
                proc < comm->nprocs;
                proc++) {
                    comm_->pack_data_to_process_stencil_md(curr_dt, start_t, end_t,
                                                           atom_arr, zoid, proc, pipeline_stage);
                }
                auto end = std::chrono::high_resolution_clock::now();
                auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count();
                send_pack_duration += duration;
            }
        }
    }

    if (SKIP_COMM) {
        return;
    }

    if (comm->nprocs != 1) {
        if (dep < NUM_DEPS - 1) {
            for (int j = 0; j < zoid_queue.size(); j++) {
                queue_info &zoid = zoid_queue[j];
                int zoid_num = zoid.num;
                if (zoid_num % comm->nprocs != comm->me) {
                    continue;
                }

                Comm *comm_ = lmp->comm_stencil_md[zoid_num];
                auto &send_to_neighbors_procs = curr_dt ? lmp->send_to_neighbors_procs[zoid_num]
                                                        : lmp->send_to_neighbors_procs_next_dt[zoid_num];

                // auto begin = std::chrono::high_resolution_clock::now();
                int vec_idx = 0;
                // TODO: parallelize
                for (int proc = 0; proc < comm->nprocs; proc++) {
                    if (proc != comm->me
                        && send_to_neighbors_procs.find(proc) !=
                           send_to_neighbors_procs.end()) {
                        bool sent = comm_->send_packed_data_to_process_stencil_md(curr_dt, start_t,
                                                                                  end_t,
                                                                                  zoid,
                                                                                  &send_requests[zoid_num][proc],
                                                                                  proc, pipeline_stage);
                    }
                }

                // auto end = std::chrono::high_resolution_clock::now();
                // auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count();
                // send_comm_duration += duration;

                // auto begin2 = std::chrono::high_resolution_clock::now();
                if (send_to_neighbors_procs.find(comm->me) != send_to_neighbors_procs.end()) {
                    comm_->send_packed_data_to_process_stencil_md(curr_dt, start_t, end_t, zoid,
                                                                  nullptr, comm->me, pipeline_stage);
                }
                // auto end2 = std::chrono::high_resolution_clock::now();
                // auto duration2 = std::chrono::duration_cast<std::chrono::microseconds>(end2 - begin2).count();
                // unpack_self_time += duration2;
            }
        }
    }
}

template <bool curr_dt>
void Verlet::run_stencil_md_pipelined_helper(int starting_timestep,
                                             std::vector<int>* dep_to_wait_idxs, std::vector<int>* dep_to_wait_idxs_next_dt,
                                             double** test_f, double** test_x, double** test_v) {
    constexpr int start_t = 1;
    constexpr int mid_t = NUM_TIMESTEPS_IN_PARALLEL / 2 + 1;
    constexpr int end_t = NUM_TIMESTEPS_IN_PARALLEL + 1;

    auto& recv_neighbor_procs = curr_dt ? lmp->recv_from_neighbors_procs : lmp->recv_from_neighbors_procs_next_dt;
    int num_zoids_recv_from = recv_neighbor_procs.size();

    // hardcode to 2 stages
    std::vector<MPI_Request> receive_requests(num_zoids_recv_from, MPI_REQUEST_NULL);
    std::vector<MPI_Request> receive_requests2(num_zoids_recv_from, MPI_REQUEST_NULL);

    int dep_to_idx[NUM_DEPS] = {0};
    for (int dep = 2; dep < NUM_DEPS; dep++) {
        auto& wait_idxs = curr_dt ? dep_to_wait_idxs[dep - 1] : dep_to_wait_idxs_next_dt[dep - 1];
        dep_to_idx[dep] = wait_idxs.size() + dep_to_idx[dep - 1];
    }

    constexpr bool PIPELINE = true;

    if (comm->nprocs != 1) {
        int recv_idx = 0;
        for (int dep = 0; dep < NUM_DEPS; dep++) {
            auto& wait_idxs = curr_dt ? dep_to_wait_idxs[dep] : dep_to_wait_idxs_next_dt[dep];
            for (int idx: wait_idxs) {
                int recv_zoid_num = recv_neighbor_procs[idx];
                if (!PIPELINE) {
                    comm->receive_data_process_stencil_md(curr_dt, start_t, end_t, &receive_requests[recv_idx], recv_zoid_num, 0);
                } else {
                    comm->receive_data_process_stencil_md(curr_dt, start_t, mid_t,
                                                          &receive_requests[recv_idx], recv_zoid_num, 0);
                    comm->receive_data_process_stencil_md(curr_dt, mid_t, end_t,
                                                          &receive_requests2[recv_idx], recv_zoid_num, 1);
                }
                recv_idx++;
            }
        }
    }

    std::vector<MPI_Request> send_requests[NUM_ZOIDS];
    std::vector<MPI_Request> send_requests2[NUM_ZOIDS];

    if (comm->nprocs != 1) {
        for (int zoid_num = comm->me; zoid_num < NUM_ZOIDS; zoid_num += comm->nprocs) {
            send_requests[zoid_num] = std::move(std::vector<MPI_Request>(comm->nprocs, MPI_REQUEST_NULL));
            send_requests2[zoid_num] = std::move(std::vector<MPI_Request>(comm->nprocs, MPI_REQUEST_NULL));
        }
    }

    if (!PIPELINE) {
        for (int dep = 0; dep < NUM_DEPS; dep++) {
            run_stencil_md_dep_templated<curr_dt>(dep, starting_timestep, start_t, end_t, dep_to_idx,
                                                  send_requests, receive_requests,
                                                  dep_to_wait_idxs, dep_to_wait_idxs_next_dt, test_f, test_x,
                                                  test_v, 0);
        }
    } else {
        run_stencil_md_dep_templated<curr_dt>(0, starting_timestep, start_t, mid_t, dep_to_idx,
                                              send_requests, receive_requests,
                                              dep_to_wait_idxs, dep_to_wait_idxs_next_dt, test_f, test_x, test_v, 0);

        cilk_scope {
                cilk_spawn run_stencil_md_dep_templated<curr_dt>(0, starting_timestep, mid_t, end_t, dep_to_idx,
                send_requests2, receive_requests2,
                dep_to_wait_idxs, dep_to_wait_idxs_next_dt, test_f, test_x, test_v, 1);

                run_stencil_md_dep_templated<curr_dt>(1, starting_timestep, start_t, mid_t, dep_to_idx,
                send_requests, receive_requests,
                dep_to_wait_idxs, dep_to_wait_idxs_next_dt, test_f, test_x, test_v, 0);
        }

        cilk_scope {
                cilk_spawn run_stencil_md_dep_templated<curr_dt>(1, starting_timestep, mid_t, end_t, dep_to_idx,
                send_requests2, receive_requests2,
                dep_to_wait_idxs, dep_to_wait_idxs_next_dt, test_f, test_x, test_v, 1);

                run_stencil_md_dep_templated<curr_dt>(2, starting_timestep, start_t, mid_t, dep_to_idx,
                send_requests, receive_requests,
                dep_to_wait_idxs, dep_to_wait_idxs_next_dt, test_f, test_x, test_v, 0);
        }

        cilk_scope {
                cilk_spawn run_stencil_md_dep_templated<curr_dt>(2, starting_timestep, mid_t, end_t, dep_to_idx,
                send_requests2, receive_requests2,
                dep_to_wait_idxs, dep_to_wait_idxs_next_dt, test_f, test_x, test_v, 1);

                run_stencil_md_dep_templated<curr_dt>(3, starting_timestep, start_t, mid_t, dep_to_idx,
                send_requests, receive_requests,
                dep_to_wait_idxs, dep_to_wait_idxs_next_dt, test_f, test_x, test_v, 0);
        }

        run_stencil_md_dep_templated<curr_dt>(3, starting_timestep, mid_t, end_t, dep_to_idx,
                                              send_requests2, receive_requests2,
                                              dep_to_wait_idxs, dep_to_wait_idxs_next_dt, test_f, test_x, test_v, 1);
    }

    if (comm->nprocs != 1) {
        for (int i = comm->me; i < NUM_ZOIDS; i += comm->nprocs) {
            if (send_requests[i].size() > 0) {
                MPI_Waitall(send_requests[i].size(), send_requests[i].data(), MPI_STATUSES_IGNORE);
            }
            if (send_requests2[i].size() > 0) {
                MPI_Waitall(send_requests2[i].size(), send_requests2[i].data(), MPI_STATUSES_IGNORE);
            }
        }
    }

    /*
    int memset_start = 0;
    int memset_end = NUM_TIMESTEPS_IN_PARALLEL;

    cilk_for (int i = comm->me; i < NUM_ZOIDS; i += comm->nprocs) {
        cilk_for(int t = memset_start; t < memset_end; t++) {
            Atom *atom_ = curr_dt ? lmp->atom_stencil_md[i][t] : lmp->atom_stencil_md[i][NUM_TIMESTEPS_IN_PARALLEL - t];
            int nall = atom_->nlocal + atom_->nghost;
            memset(&atom_->f[0][0], 0, (nall) * 3 * sizeof(double));
            memset(&atom_->eval_f_stencil_md[0][0], 0, (nall) * 3 * sizeof(double));
        }
    }
    */
}

template <bool curr_dt>
void Verlet::run_stencil_md_no_cilk_for_helper(int starting_timestep, double** test_f, double** test_x, double** test_v,
                                               std::array<std::atomic<int>, NUM_ZOIDS>& counters, const std::array<int, NUM_ZOIDS>& cache) {
    // right now focus on 1 proc implementation, and dt = 1
    assert(comm->nprocs == 1);

    auto& queues = curr_dt ? lmp->queues[0] : lmp->queues_next_dt[0];
    cilk_scope {
            for (int i = 0; i < queues.size(); i++) {
                cilk_spawn run_stencil_md_zoid_no_cilk_for<curr_dt>(starting_timestep, queues[i].num, test_f, test_x, test_v,
                                                         counters, cache);
            }
    }
}

void Verlet::run_stencil_md_no_cilk_for(int num_timesteps, double **test_f, double **test_x, double** test_v) {
    std::array<std::atomic<int>, NUM_ZOIDS> counters;
    std::array<std::atomic<int>, NUM_ZOIDS> counters_next_dt;

    std::array<int, NUM_ZOIDS> cache;
    std::array<int, NUM_ZOIDS> cache_next_dt;

    for (int i = 0; i < NUM_ZOIDS; i++) {
        int num_neighbors_recv_from = 0;
        for (auto& recv_from : lmp->recv_from_neighbors[i]) {
            if (get_zoid_dep(recv_from) == get_zoid_dep(i) - 1) {
                num_neighbors_recv_from++;
            }
        }
        counters[i] = num_neighbors_recv_from;
        cache[i] = num_neighbors_recv_from;

    }

    for (int i = 0; i < NUM_ZOIDS; i++) {
        int num_neighbors_recv_from = 0;
        for (auto& recv_from : lmp->recv_from_neighbors_next_dt[i]) {
            if (get_zoid_dep_next_dt(recv_from) == get_zoid_dep_next_dt(i) - 1) {
                num_neighbors_recv_from++;
            }
        }
        counters_next_dt[i] = num_neighbors_recv_from;
        cache_next_dt[i] = num_neighbors_recv_from;
    }

    for (int t = 0; t < num_timesteps; t += 2 * NUM_TIMESTEPS_IN_PARALLEL) {
        // curr dt
        run_stencil_md_no_cilk_for_helper<true>(t, test_f, test_x, test_v, counters, cache);
        run_stencil_md_no_cilk_for_helper<false>(t, test_f, test_x, test_v, counters_next_dt, cache_next_dt);
    }
}

void Verlet::run_stencil_md_pipelined(int num_timesteps, std::vector<int> *dep_to_wait_idxs, std::vector<int> *dep_to_wait_idxs_next_dt,
                                      double **test_f, double **test_x, double** test_v) {

    for (int t = 0; t < num_timesteps; t += 2 * NUM_TIMESTEPS_IN_PARALLEL) {
        // curr dt
        run_stencil_md_pipelined_helper<true>(t, dep_to_wait_idxs, dep_to_wait_idxs_next_dt, test_f, test_x, test_v);
        run_stencil_md_pipelined_helper<false>(t, dep_to_wait_idxs, dep_to_wait_idxs_next_dt, test_f, test_x, test_v);
    }
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
