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

#include "verlet.h"

#include "angle.h"
#include "atom.h"
#include "atom_vec.h"
#include "bond.h"
#include "comm.h"
#include "dihedral.h"
#include "domain.h"
#include "error.h"
#include "fix.h"
#include "force.h"
#include "improper.h"
#include "kspace.h"
#include "modify.h"
#include "neighbor.h"
#include "output.h"
#include "pair.h"
#include "timer.h"
#include "update.h"
#include "stencil_md.h"
#include "accelerator_kokkos.h"
#include "accelerator_omp.h"
#include "atom.h"
#include "citeme.h"
#include "comm.h"
#include "comm_brick.h"
#include "domain.h"
#include "error.h"
#include "force.h"
#include "group.h"
#include "info.h"
#include "input.h"
#include "lmppython.h"
#include "memory.h"
#include "modify.h"
#include "neighbor.h"
#include "output.h"
#include "timer.h"
#include "universe.h"
#include "update.h"
#include "variable.h"
#include "version.h"

#include <cstring>
#include <mpi.h>
#include <cmath>
#include <map>


using namespace LAMMPS_NS;

/* ---------------------------------------------------------------------- */

Verlet::Verlet(LAMMPS *lmp, int narg, char **arg) :
  Integrate(lmp, narg, arg) {}

/* ----------------------------------------------------------------------
   initialization before run
------------------------------------------------------------------------- */

void Verlet::init()
{
  Integrate::init();

  // warn if no fixes doing time integration

  bool do_time_integrate = false;
  for (const auto &fix : modify->get_fix_list())
    if (fix->time_integrate) do_time_integrate = true;

  if (!do_time_integrate && (comm->me == 0))
    error->warning(FLERR,"No fixes with time integration, atoms won't move");

  // virial_style:
  // VIRIAL_PAIR if computed explicitly in pair via sum over pair interactions
  // VIRIAL_FDOTR if computed implicitly in pair by
  //   virial_fdotr_compute() via sum over ghosts

  if (force->newton_pair) virial_style = VIRIAL_FDOTR;
  else virial_style = VIRIAL_PAIR;

  // setup lists of computes for global and per-atom PE and pressure

  ev_setup();

  // detect if fix omp is present for clearing force arrays

  if (modify->get_fix_by_id("package_omp")) external_force_clear = 1;

  // set flags for arrays to clear in force_clear()

  torqueflag = extraflag = 0;
  if (atom->torque_flag) torqueflag = 1;
  if (atom->avec->forceclearflag) extraflag = 1;

  // orthogonal vs triclinic simulation box

  triclinic = domain->triclinic;
}

/* ----------------------------------------------------------------------
   setup before run
------------------------------------------------------------------------- */

void Verlet::setup(int flag)
{
  if (comm->me == 0 && screen) {
    fputs("Setting up Verlet run ...\n",screen);
    if (flag) {
      fmt::print(screen,"  Unit style    : {}\n"
                        "  Current step  : {}\n"
                        "  Time step     : {}\n",
                 update->unit_style,update->ntimestep,update->dt);
      timer->print_timeout(screen);
    }
  }

  if (lmp->kokkos) {
      error->all(FLERR, "KOKKOS package requires run_style verlet/kk");
  } else {
      std::cout << "No kokkos?????" << std::endl;
  }

  update->setupflag = 1;

  // setup domain, communication and neighboring
  // acquire ghosts
  // build neighbor lists

  atom->setup();
  modify->setup_pre_exchange();
  if (triclinic) domain->x2lamda(atom->nlocal);
  domain->pbc();
  domain->reset_box();
  comm->setup();
  if (neighbor->style) neighbor->setup_bins();
  comm->exchange();
  if (atom->sortfreq > 0) {
      atom->sort();
  }
  comm->borders();
  if (triclinic) domain->lamda2x(atom->nlocal+atom->nghost);
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

  if (pair_compute_flag) force->pair->compute(eflag,vflag);
  else if (force->pair) force->pair->compute_dummy(eflag,vflag);

  if (atom->molecular != Atom::ATOMIC) {
    if (force->bond) force->bond->compute(eflag,vflag);
    if (force->angle) force->angle->compute(eflag,vflag);
    if (force->dihedral) force->dihedral->compute(eflag,vflag);
    if (force->improper) force->improper->compute(eflag,vflag);
  }

  if (force->kspace) {
    force->kspace->setup();
    if (kspace_compute_flag) force->kspace->compute(eflag,vflag);
    else force->kspace->compute_dummy(eflag,vflag);
  }

  modify->setup_pre_reverse(eflag,vflag);
  if (force->newton) comm->reverse_comm();

  modify->setup(vflag);
  output->setup(flag);
  update->setupflag = 0;

  setup_stencil_md();
}

void atom_reorder_stencil_md(Atom* atom_, int* current, int* permute, int start, int end) {
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

void atom_reorder_ghost_stencil_md(Atom* atom_, int* current, int* permute, int start, int end, int offset) {
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

template <typename T, std::size_t... Indices>
auto vectorToTupleHelper(const std::vector<T>& v, std::index_sequence<Indices...>) {
    return std::make_tuple(v[Indices]...);
}

template <std::size_t N, typename T>
auto vectorToTuple(const std::vector<T>& v) {
    assert(v.size() >= N);
    return vectorToTupleHelper(v, std::make_index_sequence<N>());
}

int above_below_or_same(int my_where, int recv_where) {
    int same = 0;
    int above = 1;
    int below = 2;
    if (my_where == LEFT) {
        if (recv_where == LEFT) {
            return same;
        } else if (recv_where == MIDDLE) {
            return above;
        } else if (recv_where == PBC) {
            return below;
        }
    }

    if (my_where == RIGHT) {
        if (recv_where == RIGHT) {
            return same;
        } else if (recv_where == PBC) {
            return above;
        } else if (recv_where == MIDDLE) {
            return below;
        }
    }

    if (my_where == MIDDLE) {
        if (recv_where == MIDDLE) {
            return same;
        } else if (recv_where == RIGHT) {
            return above;
        } else if (recv_where == LEFT) {
            return below;
        }
    }

    if (my_where == PBC) {
        if (recv_where == PBC) {
            return same;
        } else if (recv_where == LEFT) {
            return above;
        } else if (recv_where == RIGHT) {
            return below;
        }
    }
    assert(false);
}

void Verlet::setup_bins_stencil_md(Atom* atom_, queue_info& zoid, int timestep) {
    // binsize:
    // user setting if explicitly set
    // default = 1/2 of neighbor cutoff
    // check if neighbor cutoff = 0.0
    // and in that case, disable sorting

    double binsize = ALLEGRO_SLOPE / 2;
    double bininv = 1.0/binsize;

    int nbinx = static_cast<int> ((domain->prd[0]) * bininv);
    int nbiny = static_cast<int> ((domain->prd[1]) * bininv);
    int nbinz = static_cast<int> ((domain->prd[2]) * bininv);

    int nbins = nbinx*nbiny*nbinz;

    // std::cout << "nbins: " << nbins << std::endl;

    std::map<std::tuple<int, int, int>, std::vector<int>> bin_to_idxs;

    for (int i = atom_->nlocal; i < atom_->nlocal + atom_->nghost; i++) {
        double new_pos[3] = {atom_->x[i][0], atom_->x[i][1], atom_->x[i][2]};
        for (int dim = 0; dim < 3; dim++) {
            if (new_pos[dim] < 0) {
                new_pos[dim] += domain->prd[dim];
            }
            if (new_pos[dim] > domain->prd[dim]) {
                new_pos[dim] -= domain->prd[dim];
            }
        }

        int ix = static_cast<int> ((new_pos[0])*bininv);
        int iy = static_cast<int> ((new_pos[1])*bininv);
        int iz = static_cast<int> ((new_pos[2])*bininv);

        auto key = std::make_tuple(ix, iy, iz);

        bin_to_idxs[key].push_back(i);

        if (atom_->tag[i] == 52734 || atom_->tag[i] == 52771 || atom_->tag[i] == 32627 || atom_->tag[i] == 7960) {
            if (zoid.num == 60 && timestep == 1) {
                std::cout << RED << "RYAN tag: " << atom_->tag[i] << " bin? " << ix << " " << iy << " " << iz << " pos: " << atom_->x[i][0] << " " << atom_->x[i][1] << " " << atom_->x[i][2] << RESET_COLOR << std::endl;
            }
        }
    }

    /*
    std::vector<int> send_to = lmp->send_to_neighbors[zoid.num];
    for (int i = 0; i < send_to.size(); i++) {
        int send_zoid_num = send_to[i];
        queue_info& send_zoid = lmp->zoid_num_to_zoid[send_zoid_num];
        for (auto& [bin_idx, idxs] : bin_to_idxs) {
            int bin_idxs[3] = {std::get<0>(bin_idx), std::get<1>(bin_idx), std::get<2>(bin_idx)};
            if (idxs.size() > 0) {
                bool bin_borders_zoid = true;
                bool bin_in_zoid = true;
                int idx = idxs[0];
                for (int dim = 0; dim < 3; dim++) {
                    double zoid_lo = send_zoid.zoid.cuts[dim].lower + send_zoid.zoid.cuts[dim].slope_lower;
                    double zoid_hi = send_zoid.zoid.cuts[dim].upper + send_zoid.zoid.cuts[dim].slope_upper;
                    double bin_lo = bin_idxs[dim] * binsize;
                    double bin_hi = bin_lo + binsize;

                    bin_in_zoid = bin_in_zoid && bin_lo >= zoid_lo && bin_hi <= zoid_hi;

                    double borders_zoid_lo = zoid_lo - ALLEGRO_SLOPE;
                    double borders_zoid_hi = zoid_hi + ALLEGRO_SLOPE;

                    bin_borders_zoid = bin_borders_zoid && bin_lo >= borders_zoid_lo && bin_hi <= borders_zoid_hi;
                }

                bool bin_send_to = bin_borders_zoid;
                if (bin_send_to) {
                    // std::cout << " bin idx: " << bin_idxs[0] << " " << bin_idxs[1] << " " << bin_idxs[2] << " zoid: " << zoid.num << " send to: " << send_zoid_num << " timestep: " << timestep << std::endl;
                }
            }
        }
    }
    */

    std::vector<int> recv_from = lmp->recv_from_neighbors[zoid.num];
    std::set<std::tuple<int, int, int>> bins;
    for (int i = 0; i < recv_from.size(); i++) {
        int recv_zoid_num = recv_from[i];
        queue_info& recv_zoid = lmp->zoid_num_to_zoid[recv_zoid_num];
        for (auto& [bin_idx, idxs] : bin_to_idxs) {
            if (idxs.size() > 0) {
                int bin_idxs[3] = {std::get<0>(bin_idx), std::get<1>(bin_idx), std::get<2>(bin_idx)};
                bool bin_borders_zoid = true;

                bool bin_in_zoid = true;
                bool bin_in_zoid_prev = true;

                int idx = idxs[0];

                double bin_lo_debug[3] = {0};
                double bin_hi_debug[3] = {0};
                double recv_lo_debug[3] = {0};
                double recv_hi_debug[3] = {0};

                for (int dim = 0; dim < 3; dim++) {
                    double zoid_lo = recv_zoid.zoid.cuts[dim].lower + recv_zoid.zoid.cuts[dim].slope_lower;
                    double zoid_hi = recv_zoid.zoid.cuts[dim].upper + recv_zoid.zoid.cuts[dim].slope_upper;
                    double bin_lo = bin_idxs[dim] * binsize;
                    double bin_hi = bin_lo + binsize;

                    bin_lo_debug[dim] = bin_lo;
                    bin_hi_debug[dim] = bin_hi;

                    bin_in_zoid = bin_in_zoid && bin_lo >= zoid_lo && bin_hi <= zoid_hi;

                    double borders_zoid_lo = zoid_lo - ALLEGRO_SLOPE;
                    double borders_zoid_hi = zoid_hi + ALLEGRO_SLOPE;

                    bin_borders_zoid = bin_borders_zoid && bin_lo >= borders_zoid_lo && bin_hi <= borders_zoid_hi;

                    double lo_prev = recv_zoid.zoid.cuts[dim].lower + (timestep - 1) * recv_zoid.zoid.cuts[dim].slope_lower;
                    double hi_prev = recv_zoid.zoid.cuts[dim].upper + (timestep - 1) * recv_zoid.zoid.cuts[dim].slope_upper;

                    recv_lo_debug[dim] = lo_prev;
                    recv_hi_debug[dim] = hi_prev;

                    // if expanding zoid AND other zoid shrinking, use current instead of past, only need values for actual shrinking?
                    // must be a ghost atom that wasn't local last timestep somehow
                    if (timestep > 0) {
                        // in_zoid_prev = in_zoid_prev && ((atom_pos_shifted >= lo_prev && atom_pos_shifted <= hi_prev));
                        bool in_zoid = bin_lo >= lo_prev && bin_hi <= hi_prev;
                        bool partially_in_zoid = (bin_lo > lo_prev && fabs(hi_prev - bin_hi) <= binsize) || (bin_hi > lo_prev && fabs(bin_lo - lo_prev) <= binsize);
                        // partially_in_zoid = false;
                        bin_in_zoid_prev = bin_in_zoid_prev && (in_zoid || partially_in_zoid);
                    } else {
                        bin_in_zoid_prev = false;
                    }
                }

                bool print = zoid.num == 60 && recv_zoid_num == 1 && timestep == 1;
                bool bin_recv_from = bin_borders_zoid;
                if (bin_in_zoid_prev && print) {
                    /*
                    bins.insert(bin_idx);
                    std::cout << MAGENTA << "in zoid prev? " << bin_in_zoid_prev << " bin: " << std::get<0>(bin_idx) << " " << std::get<1>(bin_idx) << " " << std::get<2>(bin_idx) << " bin lo debug: " << bin_lo_debug[0] << " " << bin_lo_debug[1] << " " << bin_lo_debug[2]
                              << " bin hi debug: " << bin_hi_debug[0] << " " << bin_hi_debug[1] << " " << bin_hi_debug[2]
                              << " lo prev: " << recv_lo_debug[0] << " " << recv_lo_debug[1] << " " << recv_lo_debug[2]
                              << " hi prev: " << recv_hi_debug[0] << " " << recv_hi_debug[1] << " " << recv_hi_debug[2] << std::endl;
                    for (int idx_ : idxs) {
                        std::cout << "idx: " << idx_ << " tag: " << atom_->tag[idx_] << std::endl;
                    }
                    */
                }
            }
        }
    }

    if (zoid.num == 60 && timestep == 1) {
        for (auto& bin : bins) {
            std::cout << YELLOW << "setup bins: " << std::get<0>(bin) << " " << std::get<1>(bin) << " " << std::get<2>(bin) << RESET_COLOR << std::endl;
        }
    }
}

// TODO: Sort ghost atoms by zoid in previous timestep and zoid in next timestep
// Relay this information to zoids for their sendlists, we only need to ensure ghosts are contiguous. Local atoms we can try to make some compromises since
// there are so few local atoms compared to ghost. TBD though.
// this would mean that on the next dt, there will be segments, hopefully not too many, but we shall see I guess
void Verlet::group_ghost_atoms_stencil_md(Atom* atom_, Atom* prev, queue_info& zoid, int timestep) {
    int zoid_num = zoid.num;
    bool print = (zoid.num == 8 && timestep == 1);

    int* current = new int[atom_->nghost];
    int* permute = new int[atom_->nghost];

    for (int i = 0; i < atom_->nghost; i++) {
        current[i] = i;
    }

    std::map<int, std::vector<int>> idx_to_zoids;

    std::map<int, std::vector<int>> idx_to_borders_zoids;

    std::map<int, std::tuple<int, int, int>> idx_to_bin;
    double binsize = ALLEGRO_SLOPE / 2;
    double bininv = 1.0/binsize;

    int nbinx = static_cast<int> ((domain->prd[0]) * bininv);
    int nbiny = static_cast<int> ((domain->prd[1]) * bininv);
    int nbinz = static_cast<int> ((domain->prd[2]) * bininv);

    for (int i = 0; i < atom_->nghost; i++) {
        int idx = atom_->nlocal + i;
        double new_pos[3] = {atom_->x[idx][0], atom_->x[idx][1], atom_->x[idx][2]};
        for (int dim = 0; dim < 3; dim++) {
            if (new_pos[dim] < 0) {
                new_pos[dim] += domain->prd[dim];
            }
            if (new_pos[dim] > domain->prd[dim]) {
                new_pos[dim] -= domain->prd[dim];
            }
        }

        int ix = static_cast<int> ((new_pos[0])*bininv);
        int iy = static_cast<int> ((new_pos[1])*bininv);
        int iz = static_cast<int> ((new_pos[2])*bininv);

        auto key = std::make_tuple(ix, iy, iz);
        idx_to_bin[i] = key;

        int target_zoid_prev = -1;
        int target_zoid_next = -1;

        int target_zoid_curr = -1;
        int target_zoid_curr_next_dt = -1;

        for (int k = 0; k < NUM_ZOIDS; k++) {
            bool in_zoid_curr = true;

            queue_info& zoid_tmp = lmp->zoid_num_to_zoid_next_dt[k];
            for (int dim = 0; dim < domain->dimension; dim++) {
                double value = atom_->x[idx][dim];

                double lo_curr = zoid_tmp.zoid.cuts[dim].lower + (timestep) * zoid_tmp.zoid.cuts[dim].slope_lower;
                double hi_curr = zoid_tmp.zoid.cuts[dim].upper + (timestep) * zoid_tmp.zoid.cuts[dim].slope_upper;

                double sub = value - domain->prd[dim];
                double add = value + domain->prd[dim];

                bool at_least_one_curr = (sub >= lo_curr && sub <= hi_curr)
                                         || (add >= lo_curr && add <= hi_curr)
                                         || (value >= lo_curr && value <= hi_curr);

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

                double lo_prev = zoid_tmp.zoid.cuts[dim].lower + (timestep - 1) * zoid_tmp.zoid.cuts[dim].slope_lower;
                double hi_prev = zoid_tmp.zoid.cuts[dim].upper + (timestep - 1) * zoid_tmp.zoid.cuts[dim].slope_upper;

                // TODO: check that this somehow works?
                if (timestep == 0) {
                    queue_info& zoid_other_dt = lmp->zoid_num_to_zoid_next_dt[k];
                    lo_prev = zoid_other_dt.zoid.cuts[dim].lower + (NUM_TIMESTEPS_IN_PARALLEL - 1) * zoid_other_dt.zoid.cuts[dim].slope_lower;
                    hi_prev = zoid_other_dt.zoid.cuts[dim].upper + (NUM_TIMESTEPS_IN_PARALLEL - 1) * zoid_other_dt.zoid.cuts[dim].slope_upper;
                }

                double lo_next = zoid_tmp.zoid.cuts[dim].lower + (timestep + 1) * zoid_tmp.zoid.cuts[dim].slope_lower;
                double hi_next = zoid_tmp.zoid.cuts[dim].upper + (timestep + 1) * zoid_tmp.zoid.cuts[dim].slope_upper;

                // TODO: debug
                if (timestep == NUM_TIMESTEPS_IN_PARALLEL) {
                    queue_info& zoid_other_dt = lmp->zoid_num_to_zoid_next_dt[k];
                    lo_next = zoid_other_dt.zoid.cuts[dim].lower + (1) * zoid_other_dt.zoid.cuts[dim].slope_lower;
                    hi_next = zoid_other_dt.zoid.cuts[dim].upper + (1) * zoid_other_dt.zoid.cuts[dim].slope_upper;
                }

                double lo_curr = zoid_tmp.zoid.cuts[dim].lower + (timestep) * zoid_tmp.zoid.cuts[dim].slope_lower;
                double hi_curr = zoid_tmp.zoid.cuts[dim].upper + (timestep) * zoid_tmp.zoid.cuts[dim].slope_upper;

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

                bool at_least_one_prev = (sub >= lo_prev && sub <= hi_prev)
                                            || (add >= lo_prev && add <= hi_prev)
                                            || (value >= lo_prev && value <= hi_prev);

                bool at_least_one_curr = (sub >= lo_curr && sub <= hi_curr)
                                            || (add >= lo_curr && add <= hi_curr)
                                            || (value >= lo_curr && value <= hi_curr);

                bool at_least_one_next = (sub >= lo_next && sub <= hi_next)
                                            || (add >= lo_next && add <= hi_next)
                                            || (value >= lo_next && value <= hi_next);

                /*
                in_zoid_prev = in_zoid_prev && ((atom_pos_shifted >= lo_prev && atom_pos_shifted <= hi_prev));
                in_zoid_next = in_zoid_next && ((atom_pos_shifted >= lo_next && atom_pos_shifted <= hi_next));
                in_zoid_curr = in_zoid_curr && ((atom_pos_shifted >= lo_curr && atom_pos_shifted <= hi_curr));
                */

                in_zoid_prev = in_zoid_prev && at_least_one_prev;
                in_zoid_curr = in_zoid_curr && at_least_one_curr;
                in_zoid_next = in_zoid_next && at_least_one_next;

                double lo_borders = lo_curr - ALLEGRO_SLOPE;
                double hi_borders = hi_curr + ALLEGRO_SLOPE;

                bool at_least_one_borders = (sub >= lo_borders && sub <= hi_borders)
                        || (add >= lo_borders && add <= hi_borders)
                        || (value >= lo_borders && value <= hi_borders);
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

        if (timestep > 0) {
            idx_to_zoids[i].push_back(target_zoid_prev);
        }

        if (timestep < NUM_TIMESTEPS_IN_PARALLEL) {
            idx_to_zoids[i].push_back(target_zoid_next);
        }

        idx_to_zoids[i].push_back(target_zoid_curr);
        idx_to_zoids[i].push_back(target_zoid_curr_next_dt);

        /*
        if ((timestep > 0 && target_zoid_prev == target_zoid_curr)) {
            idx_to_zoids[i].insert(idx_to_zoids[i].begin(), -2);
        } else if (timestep < NUM_TIMESTEPS_IN_PARALLEL && target_zoid_next == target_zoid_curr_next_dt){
            idx_to_zoids[i].insert(idx_to_zoids[i].begin(), -1);
        }
        */
    }

    std::vector<int> ghost_idxs;
    for (int i = 0; i < atom_->nghost; i++) {
        ghost_idxs.push_back(i);
    }

    std::stable_sort(ghost_idxs.begin(), ghost_idxs.end(), [&](const int& a, const int& b) {
        std::vector<int>& vec_a = idx_to_zoids[a];
        std::vector<int>& vec_b = idx_to_zoids[b];

        int min_vec_size = std::min(vec_a.size(), vec_b.size());

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

        auto& bin_a = idx_to_bin[a];
        auto& bin_b = idx_to_bin[b];
        if (std::get<0>(bin_a) < std::get<0>(bin_b)) {
            return true;
        }
        if (std::get<0>(bin_a) > std::get<0>(bin_b)) {
            return false;
        }
        if (std::get<1>(bin_a) < std::get<1>(bin_b)) {
            return true;
        }
        if (std::get<1>(bin_a) > std::get<1>(bin_b)) {
            return false;
        }
        if (std::get<2>(bin_a) < std::get<2>(bin_b)) {
            return true;
        }
        if (std::get<2>(bin_a) > std::get<2>(bin_b)) {
            return false;
        }

        return atom_->tag[a + atom_->nlocal] < atom_->tag[b + atom_->nlocal];

        return false;
    });

    for (int i = 0; i < atom_->nghost; i++) {
        permute[i] = ghost_idxs[i];
    }

    atom_reorder_ghost_stencil_md(atom_, current, permute, 0, atom_->nghost, atom_->nlocal);
    bool print_tmp = (zoid_num == 24 && timestep == 1);

    if (print_tmp) {
        // std::cout << "what the fuck? " << std::endl;
        for (int i = 0; i < atom_->nghost; i++) {
            int idx = atom_->nlocal + i;
        }
    }

    delete[] current;
    delete[] permute;

    std::map<int, std::vector<int>> neighbor_to_idxs;

    auto& recv_from = lmp->recv_from_neighbors[zoid.num];

    std::set<std::tuple<int, int, int>> bins;
    bool print_buckets = zoid_num == 35 && timestep == 1;


    assert(ghost_idxs.size() == atom_->nghost);

    std::vector<std::tuple<int, int, int>> ghost_idx_bins;

    for (int i = 0; i < ghost_idxs.size(); i++) {
        int actual_idx = i + atom_->nlocal;
        double *pos = atom_->x[actual_idx];

        double new_pos[3] = {pos[0], pos[1], pos[2]};
        for (int dim = 0; dim < 3; dim++) {
            if (new_pos[dim] < 0) {
                new_pos[dim] += domain->prd[dim];
            }
            if (new_pos[dim] > domain->prd[dim]) {
                new_pos[dim] -= domain->prd[dim];
            }
        }

        int ix = static_cast<int> ((new_pos[0])*bininv);
        int iy = static_cast<int> ((new_pos[1])*bininv);
        int iz = static_cast<int> ((new_pos[2])*bininv);

        auto key = std::make_tuple(ix, iy, iz);
        ghost_idx_bins.push_back(key);

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
                if (zoid.where[dim] == RIGHT && recv_from_zoid.where[dim] == PBC) {
                    pbc_ = -1;
                }
                if (zoid.where[dim] == PBC && recv_from_zoid.where[dim] == RIGHT) {
                    pbc_ = 1;
                }

                double atom_pos_shifted = value + pbc_ * domain->prd[dim];

                double lo_prev = recv_from_zoid.zoid.cuts[dim].lower + (timestep - 1) * recv_from_zoid.zoid.cuts[dim].slope_lower;
                double hi_prev = recv_from_zoid.zoid.cuts[dim].upper + (timestep - 1) * recv_from_zoid.zoid.cuts[dim].slope_upper;

                double sub = value - domain->prd[dim];
                double add = value + domain->prd[dim];

                bool at_least_one_prev = (sub >= lo_prev && sub <= hi_prev)
                                         || (add >= lo_prev && add <= hi_prev)
                                         || (value >= lo_prev && value <= hi_prev);

                // if expanding zoid AND other zoid shrinking, use current instead of past, only need values for actual shrinking?
                // must be a ghost atom that wasn't local last timestep somehow
                if (timestep > 0) {
                    // in_zoid_prev = in_zoid_prev && ((atom_pos_shifted >= lo_prev && atom_pos_shifted <= hi_prev));
                    in_zoid_prev = in_zoid_prev && at_least_one_prev;
                } else {
                    in_zoid_prev = false;
                }
            }

            if (in_zoid_prev) {
                neighbor_to_idxs[recv_zoid_num].push_back(i + atom_->nlocal);
            }
        }
    }

    if (print_buckets) {
        for (int i = 0; i < ghost_idxs.size(); i++) {
            auto& bin = ghost_idx_bins[i];
            int lookup_idx = ghost_idxs[i];
            std::cout << "group ghost idx: " << i << " bin: " << std::get<0>(bin) << " " << std::get<1>(bin) << " " << std::get<2>(bin) << " idx to zoids: " << idx_to_zoids[lookup_idx] << " idx to borders zoids: " << idx_to_borders_zoids[lookup_idx] << std::endl;
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

            int num_segments = get_segments(idx_vec, segment_idxs, segment_lengths);

            zoid.recv_ghost_idxs[timestep][i] = new int[num_segments];
            zoid.recv_ghost_sizes[timestep][i] = new int[num_segments];
            zoid.recv_ghost_num_segments[timestep][i] = num_segments;

            for (int j = 0; j < segment_idxs.size(); j++) {
                zoid.recv_ghost_idxs[timestep][i][j] = segment_idxs[j];
                zoid.recv_ghost_sizes[timestep][i][j] = segment_lengths[j];
            }
        }
    }

    // delete[] idx_to_recv_neighbors;
}

// does the same thing as group_ghost but without the reordering as that has been done already
void Verlet::group_ghost_atoms_stencil_md_next_dt(Atom* atom_, Atom* prev, queue_info& zoid, int timestep) {
    std::map<int, std::vector<int>> neighbor_to_idxs;

    auto& recv_from = lmp->recv_from_neighbors_next_dt[zoid.num];

    for (int i = 0; i < atom_->nghost; i++) {
        int actual_idx = i + atom_->nlocal;
        double *pos = atom_->x[actual_idx];

        for (int j = 0; j < recv_from.size(); j++) {
            int recv_zoid_num = recv_from[j];
            queue_info& recv_from_zoid = lmp->zoid_num_to_zoid_next_dt[recv_zoid_num];

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
                if (zoid.where[dim] == RIGHT && recv_from_zoid.where[dim] == PBC) {
                    pbc_ = -1;
                }
                if (zoid.where[dim] == PBC && recv_from_zoid.where[dim] == RIGHT) {
                    pbc_ = 1;
                }

                double atom_pos_shifted = value + pbc_ * domain->prd[dim];

                double lo_prev = recv_from_zoid.zoid.cuts[dim].lower + (timestep - 1) * recv_from_zoid.zoid.cuts[dim].slope_lower;
                double hi_prev = recv_from_zoid.zoid.cuts[dim].upper + (timestep - 1) * recv_from_zoid.zoid.cuts[dim].slope_upper;

                double sub = value - domain->prd[dim];
                double add = value + domain->prd[dim];

                bool at_least_one_prev = (sub >= lo_prev && sub <= hi_prev)
                                         || (add >= lo_prev && add <= hi_prev)
                                         || (value >= lo_prev && value <= hi_prev);

                // if expanding zoid AND other zoid shrinking, use current instead of past, only need values for actual shrinking?
                // must be a ghost atom that wasn't local last timestep somehow
                if (timestep > 0) {
                    // in_zoid_prev = in_zoid_prev && ((atom_pos_shifted >= lo_prev && atom_pos_shifted <= hi_prev));
                    in_zoid_prev = in_zoid_prev && at_least_one_prev;
                } else {
                    in_zoid_prev = false;
                }
            }

            if (in_zoid_prev) {
                neighbor_to_idxs[recv_zoid_num].push_back(i + atom_->nlocal);
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

            int num_segments = get_segments(idx_vec, segment_idxs, segment_lengths);

            zoid.recv_ghost_idxs[timestep][i] = new int[num_segments];
            zoid.recv_ghost_sizes[timestep][i] = new int[num_segments];
            zoid.recv_ghost_num_segments[timestep][i] = num_segments;

            bool print_tmp = (zoid.num == 33 && recv_from_zoid_num == 57 && timestep == 1);
            if (print_tmp && false) {
                std::cout << "segment idxs: " << segment_idxs << std::endl;
                for (int idx : segment_idxs) {
                    std::cout << "segment idx: " << idx << " tag: " << atom_->tag[idx] << std::endl;
                }
                assert(false);
            }

            for (int j = 0; j < num_segments; j++) {
                zoid.recv_ghost_idxs[timestep][i][j] = segment_idxs[j];
                zoid.recv_ghost_sizes[timestep][i][j] = segment_lengths[j];
            }
        }
    }

    /*
    assert(false);
    int zoid_num = zoid.num;
    auto& recv_from = lmp->recv_from_neighbors_next_dt[zoid.num];
    assert(std::is_sorted(recv_from.begin(), recv_from.end()));

    std::vector<int> neighbors;
    for (int i = 0; i < NUM_ZOIDS; i++) {
        if (i != zoid_num && is_close(zoid.where, lmp->zoid_num_to_zoid_next_dt[i].where)) {
            neighbors.push_back(i);
        }
    }

    std::vector<int>* idx_to_recv_neighbors = new std::vector<int>[atom_->nghost];
    std::vector<int>* idx_to_recv_neighbors_test = new std::vector<int>[atom_->nghost];
    std::vector<int>* idx_to_recv_neighbors_after_swap = new std::vector<int>[atom_->nghost];
    std::vector<int>* idx_to_recv_neighbors_test_after_swap = new std::vector<int>[atom_->nghost];

    std::map<int, std::vector<int>> neighbor_to_idxs;
    std::map<int, std::vector<int>> neighbor_to_idxs_prev;

    std::map<int, std::set<int>> recv_zoid_to_intersection;

    std::set<int> received_atom_idxs;

    bool print = true;

    for (int i = atom_->nlocal; i < atom_->nlocal + atom_->nghost; i++) {
        double *pos = atom_->x[i];
        // TODO: need to bucket the local ones first, and then bucket the 2-hop ghosts
        for (int j = 0; j < recv_from.size(); j++) {
            int recv_zoid_num = recv_from[recv_from.size() - 1 - j];
            queue_info& recv_from_zoid = lmp->zoid_num_to_zoid_next_dt[recv_zoid_num];

            double lo_track[3] = {0};
            double hi_track[3] = {0};

            double lo_prev_track[3] = {0};
            double hi_prev_track[3] = {0};

            // recv_from is neighbor
            bool pbc_flags[3];
            // technically check if borders zoid
            bool in_zoid = true;
            bool in_zoid_prev = true;

            for (int dim = 0; dim < domain->dimension; dim++) {
                bool in_zoid_curr_dim = true;

                double value = pos[dim];
                double lo = recv_from_zoid.zoid.cuts[dim].lower + (timestep) * recv_from_zoid.zoid.cuts[dim].slope_lower;
                double hi = recv_from_zoid.zoid.cuts[dim].upper + (timestep) * recv_from_zoid.zoid.cuts[dim].slope_upper;

                lo -= ALLEGRO_SLOPE;
                hi += ALLEGRO_SLOPE;

                lo_track[dim] = lo;
                hi_track[dim] = hi;

                int pbc_ = 0;
                if (zoid.where[dim] == RIGHT && recv_from_zoid.where[dim] == PBC) {
                    pbc_ = -1;
                }
                if (zoid.where[dim] == PBC && recv_from_zoid.where[dim] == RIGHT) {
                    pbc_ = 1;
                }

                double atom_pos_shifted = value + pbc_ * domain->prd[dim];

                pbc_flags[dim] = pbc_;
                in_zoid = in_zoid && ((atom_pos_shifted >= lo && atom_pos_shifted <= hi));

                // double lo_prev = lo;
                // double hi_prev = hi;
                double lo_prev = recv_from_zoid.zoid.cuts[dim].lower + (timestep - 1) * recv_from_zoid.zoid.cuts[dim].slope_lower;
                double hi_prev = recv_from_zoid.zoid.cuts[dim].upper + (timestep - 1) * recv_from_zoid.zoid.cuts[dim].slope_upper;

                lo_prev_track[dim] = lo_prev;
                hi_prev_track[dim] = hi_prev;

                // if expanding zoid AND other zoid shrinking, use current instead of past, only need values for actual shrinking?
                // must be a ghost atom that wasn't local last timestep somehow
                if (timestep > 0) {
                    in_zoid_prev = in_zoid_prev && ((atom_pos_shifted >= lo_prev && atom_pos_shifted <= hi_prev));
                } else {
                    in_zoid_prev = false;
                    in_zoid = false;
                }
            }

            int target_zoid = -1;
            int target_zoid_prev = -1;

            for (int k = 0; k < NUM_ZOIDS; k++) {
                bool in_zoid_tmp = true;
                queue_info& zoid_tmp = lmp->zoid_num_to_zoid_next_dt[k];

                for (int dim = 0; dim < domain->dimension; dim++) {
                    double value = atom_->x[i][dim];
                    double lo = zoid_tmp.zoid.cuts[dim].lower + (timestep) * zoid_tmp.zoid.cuts[dim].slope_lower;
                    double hi = zoid_tmp.zoid.cuts[dim].upper + (timestep) * zoid_tmp.zoid.cuts[dim].slope_upper;

                    int pbc_ = 0;
                    if (zoid.where[dim] == RIGHT && zoid_tmp.where[dim] == PBC) {
                        pbc_ = -1;
                    }

                    if (zoid.where[dim] == PBC && zoid_tmp.where[dim] == RIGHT) {
                        pbc_ = 1;
                    }

                    double atom_pos_shifted = value + pbc_ * domain->prd[dim];

                    in_zoid_tmp = in_zoid_tmp && ((atom_pos_shifted >= lo && atom_pos_shifted <= hi));
                }

                if (in_zoid_tmp) {
                    target_zoid = k;
                    break;
                }
            }

            for (int k = 0; k < NUM_ZOIDS; k++) {
                bool in_zoid_tmp = true;
                queue_info& zoid_tmp = lmp->zoid_num_to_zoid_next_dt[k];

                for (int dim = 0; dim < domain->dimension; dim++) {
                    double value = atom_->x[i][dim];
                    double lo = zoid_tmp.zoid.cuts[dim].lower + (timestep - 1) * zoid_tmp.zoid.cuts[dim].slope_lower;
                    double hi = zoid_tmp.zoid.cuts[dim].upper + (timestep - 1) * zoid_tmp.zoid.cuts[dim].slope_upper;

                    int pbc_ = 0;
                    if (zoid.where[dim] == RIGHT && zoid_tmp.where[dim] == PBC) {
                        pbc_ = -1;
                    }

                    if (zoid.where[dim] == PBC && zoid_tmp.where[dim] == RIGHT) {
                        pbc_ = 1;
                    }

                    double atom_pos_shifted = value + pbc_ * domain->prd[dim];

                    in_zoid_tmp = in_zoid_tmp && ((atom_pos_shifted >= lo && atom_pos_shifted <= hi));
                }

                if (in_zoid_tmp) {
                    target_zoid_prev = k;
                    break;
                }
            }

            bool print_shit = true;
            if (in_zoid_prev) {
                idx_to_recv_neighbors[i - atom_->nlocal].push_back(recv_zoid_num);
                neighbor_to_idxs[recv_zoid_num].push_back(i);
                received_atom_idxs.insert(i);

                if (zoid_num == 32 && recv_from_zoid.num == 56 && timestep == 1 && print_shit) {
                    std::cout << CYAN << "next dt group ghost idx " << i << " out of: " << atom_->nlocal << " tag: " << atom_->tag[i]
                              << " pos: " << atom_->x[i][0]  << " " << atom_->x[i][1] << " " << atom_->x[i][2]
                              << " target zoid: " << target_zoid << " target zoid prev: " << target_zoid_prev << RESET_COLOR << std::endl;
                }
            } else {
                if (zoid_num == 32 && recv_from_zoid.num == 56 && timestep == 1 && print_shit) {
                    std::cout << MAGENTA << "next dt group ghost not in zoid idx " << i << " out of: " << atom_->nlocal << " tag: " << atom_->tag[i]
                              << " pos: " << atom_->x[i][0]  << " " << atom_->x[i][1] << " " << atom_->x[i][2]
                              << " target zoid: " << target_zoid << " target zoid prev: " << target_zoid_prev << RESET_COLOR << std::endl;
                }
            }
        }
    }

    for (int i = atom_->nlocal; i < atom_->nghost + atom_->nlocal; i++) {
        int target_zoid_prev = -1;
        bool in_zoid_prev_ = true;

        for (int dim = 0; dim < domain->dimension; dim++) {
            double value = atom_->x[i][dim];
            double lo = zoid.zoid.cuts[dim].lower + (timestep - 1) * zoid.zoid.cuts[dim].slope_lower;
            double hi = zoid.zoid.cuts[dim].upper + (timestep - 1) * zoid.zoid.cuts[dim].slope_upper;

            in_zoid_prev_ = in_zoid_prev_ && ((value >= lo && value <= hi));
        }

        if (received_atom_idxs.find(i) == received_atom_idxs.end() && timestep > 0 && !in_zoid_prev_) {
            std::cout << "zoid num: " << zoid_num << " idx: " << i << " tag: " << atom_->tag[i] << " timestep: " << timestep
                      << " pos: " << atom_->x[i][0] << " " << atom_->x[i][1] << " " << atom_->x[i][2] << std::endl;
            assert(false);
        }
    }

    int total = 0;
    std::set<int> test;
    for (int recv_zoid_num: recv_from) {
        int num_idxs_for_neighbor = neighbor_to_idxs[recv_zoid_num].size() + neighbor_to_idxs_prev[recv_zoid_num].size();
        total += num_idxs_for_neighbor;
        for (int idx : neighbor_to_idxs[recv_zoid_num]) {
            test.insert(idx);
        }
        for (int idx : neighbor_to_idxs_prev[recv_zoid_num]) {
            test.insert(idx);
        }
    }

    for (int i = 0; i < recv_from.size(); i++) {
        int recv_from_zoid_num = recv_from[i];

        if (neighbor_to_idxs[recv_from_zoid_num].size() == 0) {
            zoid.recv_ghost_idxs[timestep][i] = new int[1];
            zoid.recv_ghost_idxs[timestep][i][0] = 0;
            zoid.recv_ghost_sizes[timestep][i] = new int[1];
            zoid.recv_ghost_sizes[timestep][i][0] = 0;
        } else {
            std::vector<int> segment_idxs;
            std::vector<int> segment_lengths;

            auto& idx_vec = neighbor_to_idxs[recv_from_zoid_num];
            assert(idx_vec.size() > 0);
            int start = idx_vec[0];
            for (int j = 1; j < idx_vec.size(); j++) {
                if (idx_vec[j] - idx_vec[j - 1] > 1) {
                    if (zoid_num == 32 && recv_from_zoid_num == 56 && timestep == 1) {
                        std::cout << "next dt j: " << j << " idx vec: " << idx_vec[j] << " " << idx_vec[j - 1] << std::endl;
                    }

                    segment_idxs.push_back(start);
                    int segment_length = (idx_vec[j - 1] - start + 1);
                    segment_lengths.push_back(segment_length);
                    start = idx_vec[j];
                }
            }

            int last_segment_length = idx_vec[idx_vec.size() - 1] - start + 1;
            segment_idxs.push_back(start);
            segment_lengths.push_back(last_segment_length);

            int total_size = 0;
            for (int segment_size : segment_lengths) {
                total_size += segment_size;
            }

            zoid.recv_ghost_idxs[timestep][i] = new int[segment_idxs.size()];
            zoid.recv_ghost_sizes[timestep][i] = new int[segment_idxs.size()];
            zoid.recv_ghost_num_segments[timestep][i] = segment_idxs.size();

            for (int j = 0; j < segment_idxs.size(); j++) {
                zoid.recv_ghost_idxs[timestep][i][j] = segment_idxs[j];
                zoid.recv_ghost_sizes[timestep][i][j] = segment_lengths[j];
            }

            if (zoid_num == 32 && recv_from_zoid_num == 56 && timestep == 1) {
                for (int j = 0; j < segment_idxs.size(); j++) {
                    std::cout << "next dt get segment: " << j << " idx: " << segment_idxs[j] << " segment size: " << segment_lengths[j] << std::endl;
                }
                std::cout << "next dt group ghost total segment size: " << total_size << std::endl;
            }
        }
    }

    delete[] idx_to_recv_neighbors;
    delete[] idx_to_recv_neighbors_test;
    delete[] idx_to_recv_neighbors_test_after_swap;
    delete[] idx_to_recv_neighbors_after_swap;
    */
}

void Verlet::group_local_atoms_stencil_md(Atom* atom_, queue_info& zoid, int timestep) {
    assert(false);
    /*
    int zoid_num = zoid.num;
    auto& recv_from = lmp->recv_from_neighbors[zoid.num];

    int nlocal = atom_->nlocal;

    std::vector<int>* idx_to_recv_neighbors = new std::vector<int>[nlocal];
    std::vector<int> idxs_per_recv_neighbor[recv_from.size()];

    for (int i = 0; i < atom_->nlocal; i++) {
        double *pos = atom_->x[i];
        // TODO: need to bucket the local ones first, and then bucket the 2-hop ghosts
        for (int j = 0; j < recv_from.size(); j++) {
            int recv_zoid_num = recv_from[j];
            queue_info& recv_from_zoid = lmp->zoid_num_to_zoid[recv_zoid_num];

            double track_lo[3] = {0};
            double track_hi[3] = {0};
            double track_shifted[3] = {0};

            // recv_from is neighbor
            bool pbc_flags[3];
            // technically check if borders zoid
            bool in_zoid = true;
            bool in_zoid_prev = true;
            for (int dim = 0; dim < domain->dimension; dim++) {
                bool in_zoid_curr_dim = true;

                double value = pos[dim];
                double lo = recv_from_zoid.zoid.cuts[dim].lower + (timestep) * recv_from_zoid.zoid.cuts[dim].slope_lower;
                double hi = recv_from_zoid.zoid.cuts[dim].upper + (timestep) * recv_from_zoid.zoid.cuts[dim].slope_upper;

                // if my zoid is expanding, need neighbors

                // if expanding zoid AND other zoid shrinking, use current instead of past, only need values for actual shrinking?
                if (zoid.zoid.cuts[dim].slope_lower < 0 && recv_from_zoid.zoid.cuts[dim].slope_lower > 0) {
                    lo -= ALLEGRO_SLOPE;
                    hi += ALLEGRO_SLOPE;
                }

                int pbc_ = 0;
                if (zoid.where[dim] == RIGHT && recv_from_zoid.where[dim] == PBC) {
                    pbc_ = -1;
                }
                if (zoid.where[dim] == PBC && recv_from_zoid.where[dim] == RIGHT) {
                    pbc_ = 1;
                }

                double atom_pos_shifted = value + pbc_ * domain->prd[dim];

                pbc_flags[dim] = pbc_;
                in_zoid = in_zoid && ((atom_pos_shifted >= lo && atom_pos_shifted <= hi));

                track_lo[dim] = lo;
                track_hi[dim] = hi;
                track_shifted[dim] = atom_pos_shifted;
            }

            if (in_zoid_prev) {
                idx_to_recv_neighbors[i].push_back(recv_zoid_num);
                idxs_per_recv_neighbor[j].push_back(i);
            }
        }
    }

    for (int i = 0; i < recv_from.size(); i++) {
        int num_recv_from = idxs_per_recv_neighbor[i].size();
        zoid.recv_list_local[timestep][i] = new int[num_recv_from];
        for (int j = 0; j < num_recv_from; j++) {
            zoid.recv_list_local[timestep][i][j] = idxs_per_recv_neighbor[i][j];
        }
        zoid.recv_list_local_size[timestep][i] = num_recv_from;
        bool print_local = true;
        if (zoid_num == 35 && recv_from[i] == 10 && timestep == 2 && print_local) {
            int target_zoid_prev = -1;

            for (int k = 0; k < NUM_ZOIDS; k++) {
                bool in_zoid_tmp = true;
                queue_info& zoid_tmp = lmp->zoid_num_to_zoid[k];

                for (int dim = 0; dim < domain->dimension; dim++) {
                    double value = atom_->x[i][dim];
                    double lo = zoid_tmp.zoid.cuts[dim].lower + (timestep - 1) * zoid_tmp.zoid.cuts[dim].slope_lower;
                    double hi = zoid_tmp.zoid.cuts[dim].upper + (timestep - 1) * zoid_tmp.zoid.cuts[dim].slope_upper;

                    int pbc_ = 0;
                    if (zoid.where[dim] == RIGHT && zoid_tmp.where[dim] == PBC) {
                        pbc_ = -1;
                    }

                    if (zoid.where[dim] == PBC && zoid_tmp.where[dim] == RIGHT) {
                        pbc_ = 1;
                    }

                    double atom_pos_shifted = value + pbc_ * domain->prd[dim];

                    in_zoid_tmp = in_zoid_tmp && ((atom_pos_shifted >= lo && atom_pos_shifted <= hi));
                }

                if (in_zoid_tmp) {
                    target_zoid_prev = k;
                    break;
                }
            }
          for (int k = 0; k < num_recv_from; k++) {
              int idx = zoid.recv_list_local[timestep][i][k];
          }
        }
    }

    delete[] idx_to_recv_neighbors;
    // assert(test_unique.size() == total_size);
    */
}

void Verlet::group_local_atoms_stencil_md_next_dt(Atom* atom_, queue_info& zoid, int timestep) {
    assert(false);
    /*
    int zoid_num = zoid.num;
    auto& recv_from = lmp->recv_from_neighbors_next_dt[zoid.num];

    int nlocal = atom_->nlocal;

    std::vector<int>* idx_to_recv_neighbors = new std::vector<int>[nlocal];
    std::vector<int> idxs_per_recv_neighbor[recv_from.size()];

    bool print_local = false;

    for (int i = 0; i < atom_->nlocal; i++) {
        double *pos = atom_->x[i];
        for (int j = 0; j < recv_from.size(); j++) {
            int recv_zoid_num = recv_from[j];
            queue_info& recv_from_zoid = lmp->zoid_num_to_zoid_next_dt[recv_zoid_num];

            double track_lo[3] = {0};
            double track_hi[3] = {0};
            double track_shifted[3] = {0};

            // recv_from is neighbor
            bool pbc_flags[3];
            // technically check if borders zoid
            bool in_zoid = true;
            for (int dim = 0; dim < domain->dimension; dim++) {
                bool in_zoid_curr_dim = true;

                double value = pos[dim];
                double lo = recv_from_zoid.zoid.cuts[dim].lower + (timestep) * recv_from_zoid.zoid.cuts[dim].slope_lower;
                double hi = recv_from_zoid.zoid.cuts[dim].upper + (timestep) * recv_from_zoid.zoid.cuts[dim].slope_upper;

                // if my zoid is expanding, need neighbors

                // if expanding zoid AND other zoid shrinking, use current instead of past, only need values for actual shrinking?
                if (zoid.zoid.cuts[dim].slope_lower < 0 && recv_from_zoid.zoid.cuts[dim].slope_lower > 0) {
                    lo -= ALLEGRO_SLOPE;
                    hi += ALLEGRO_SLOPE;
                }

                int pbc_ = 0;
                if (zoid.where[dim] == RIGHT && recv_from_zoid.where[dim] == PBC) {
                    pbc_ = -1;
                }
                if (zoid.where[dim] == PBC && recv_from_zoid.where[dim] == RIGHT) {
                    pbc_ = 1;
                }

                double atom_pos_shifted = value + pbc_ * domain->prd[dim];

                pbc_flags[dim] = pbc_;
                in_zoid = in_zoid && ((atom_pos_shifted >= lo && atom_pos_shifted <= hi));

                track_lo[dim] = lo;
                track_hi[dim] = hi;
                track_shifted[dim] = atom_pos_shifted;
            }

            if (in_zoid) {
                idx_to_recv_neighbors[i].push_back(recv_zoid_num);
                idxs_per_recv_neighbor[j].push_back(i);
                if (zoid_num == 32 && recv_from_zoid.num == 56 && timestep == 1 && false) {
                    std::cout << CYAN << "next dt group local idx " << i << " tag: " << atom_->tag[i]
                    << " pos: " << atom_->x[i][0]  << " " << atom_->x[i][1] << " " << atom_->x[i][2] << RESET_COLOR << std::endl;
                }
            }
        }
    }

    for (int i = 0; i < recv_from.size(); i++) {
        int num_recv_from = idxs_per_recv_neighbor[i].size();
        zoid.recv_list_local[timestep][i] = new int[num_recv_from];
        for (int j = 0; j < num_recv_from; j++) {
            zoid.recv_list_local[timestep][i][j] = idxs_per_recv_neighbor[i][j];
        }
        zoid.recv_list_local_size[timestep][i] = num_recv_from;
        if (zoid_num == 32 && recv_from[i] == 56 && timestep == 1) {
            std::cout << "i: " << i << " num recv from? " << num_recv_from << " huh? "
            << " local? " << &zoid.recv_list_local_size[timestep][i] << " addr of zoid? " << &zoid << std::endl;
        }
    }

    delete[] idx_to_recv_neighbors;
    */
}

void setup_can_eval_center_mapping_stencil_md(std::array<Atom*, NUM_TIMESTEPS_IN_PARALLEL + 1>& atom_arr,
                                              queue_info& zoid) {
    // TODO: attempt to have can_eval_center also be a flag to replace exclude_eval_tags

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

            for (int dim = 0; dim < 3; dim++) {
                bool shrinking_dim = zoid.zoid.cuts[dim].slope_lower > 0;
                double lo = zoid.zoid.cuts[dim].lower + t * zoid.zoid.cuts[dim].slope_lower;
                double hi = zoid.zoid.cuts[dim].upper + t * zoid.zoid.cuts[dim].slope_upper;

                bool in_bounds = (pos[dim] >= lo && pos[dim] <= hi);

                double diff = std::min(fabs(pos[dim] - lo), fabs(pos[dim] - hi));

                if (shrinking_dim) {
                    can_eval_center = can_eval_center && (in_bounds || (diff <= ALLEGRO_CUTOFF_RADIUS));
                } else {
                    can_eval_center = can_eval_center && in_bounds && (diff > ALLEGRO_CUTOFF_RADIUS);
                }
            }
            zoid.can_eval_center[t][i] = can_eval_center;
        }

        /*
        // TODO: no more eval ghost bullshit
        for (int i = atom_->nlocal; i < atom_->nlocal + atom_->nghost; i++) {
            double* pos = atom_->x[i];

            bool can_eval_center = true;

            int diffs[3] = {0};

            // check each dimension
            for (int dim = 0; dim < 3; dim++) {
                bool shrinking_dim = zoid.zoid.cuts[dim].slope_lower > 0;
                double lo = zoid.zoid.cuts[dim].lower + t * zoid.zoid.cuts[dim].slope_lower;
                double hi = zoid.zoid.cuts[dim].upper + t * zoid.zoid.cuts[dim].slope_upper;
                double diff;

                if (pos[dim] >= lo && pos[dim] <= hi) {
                    diff = 0;
                } else {
                    diff = std::min(fabs(pos[dim] - lo), fabs(pos[dim] - hi));
                }

                if (shrinking_dim) {
                    can_eval_center = can_eval_center && (diff <= ALLEGRO_CUTOFF_RADIUS);
                } else {
                    can_eval_center = can_eval_center && (pos[dim] >= lo && pos[dim] <= hi);
                }
            }
            zoid.can_eval_center[t][i] = can_eval_center;
        }
        */
    }
}

void setup_can_eval_center_mapping_stencil_md_next_dt(std::array<Atom*, NUM_TIMESTEPS_IN_PARALLEL + 1>& atom_arr,
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
                double lo = zoid.zoid.cuts[dim].lower + t * zoid.zoid.cuts[dim].slope_lower;
                double hi = zoid.zoid.cuts[dim].upper + t * zoid.zoid.cuts[dim].slope_upper;

                bool in_bounds = (pos[dim] >= lo && pos[dim] <= hi);

                double diff = std::min(fabs(pos[dim] - lo), fabs(pos[dim] - hi));

                if (shrinking_dim) {
                    can_eval_center = can_eval_center && (in_bounds || (diff <= ALLEGRO_CUTOFF_RADIUS));
                } else {
                    can_eval_center = can_eval_center && in_bounds && (diff > ALLEGRO_CUTOFF_RADIUS);
                }
            }
            zoid.can_eval_center[t][i] = can_eval_center;
        }
    /*
    for (int t = 0; t < NUM_TIMESTEPS_IN_PARALLEL + 1; t++) {
        int idx_obj_timestep = NUM_TIMESTEPS_IN_PARALLEL - t;
        Atom* atom_ = atom_arr[idx_obj_timestep];

        int total = atom_->nlocal + atom_->nghost;
        zoid.can_eval_center[t] = new bool[total];
        zoid.can_eval_pos[t] = new bool[total];

        for (int i = 0; i < total; i++) {
            zoid.can_eval_center[t][i] = false;
            zoid.can_eval_pos[t][i] = false;
        }

        for (int i = 0; i < atom_->nlocal; i++) {
            zoid.can_eval_center[t][i] = true;
            zoid.can_eval_pos[t][i] = true;
        }

        // TODO: no more eval ghost bullshit
        for (int i = atom_->nlocal; i < atom_->nlocal + atom_->nghost; i++) {
            double* pos = atom_->x[i];

            bool can_eval_center = true;

            int diffs[3] = {0};

            // check each dimension
            for (int dim = 0; dim < 3; dim++) {
                bool shrinking_dim = zoid.zoid.cuts[dim].slope_lower > 0;
                double lo = zoid.zoid.cuts[dim].lower + t * zoid.zoid.cuts[dim].slope_lower;
                double hi = zoid.zoid.cuts[dim].upper + t * zoid.zoid.cuts[dim].slope_upper;
                double diff;

                if (pos[dim] >= lo && pos[dim] <= hi) {
                    diff = 0;
                } else {
                    diff = std::min(fabs(pos[dim] - lo), fabs(pos[dim] - hi));
                }

                if (shrinking_dim) {
                    can_eval_center = can_eval_center && (diff <= ALLEGRO_CUTOFF_RADIUS);
                } else {
                    can_eval_center = can_eval_center && (pos[dim] >= lo && pos[dim] <= hi);
                }
            }
            zoid.can_eval_center[t][i] = can_eval_center;

        }
        */
    }
}

// create an arr, permute_esque
void setup_atom_pos_mapping_stencil_md(std::array<Atom*, NUM_TIMESTEPS_IN_PARALLEL + 1>& atom_arr, queue_info& zoid) {
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
void setup_atom_pos_mapping_stencil_md_next_dt(std::array<Atom*, NUM_TIMESTEPS_IN_PARALLEL + 1>& atom_arr, queue_info& zoid) {
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
    // USE_STENCIL_MD = true;
    std::cout << "RYAN setup stencil md start 2" << std::endl;
    std::cout << "domain boxlo: " << domain->boxlo[0] << " " << domain->boxlo[1] << " " << domain->boxlo[2] << std::endl;
    std::cout << "domain boxhi: " << domain->boxhi[0] << " " << domain->boxhi[1] << " " << domain->boxhi[2] << std::endl;

    // assert(3 * 2 * ALLEGRO_SLOPE * NUM_TIMESTEPS_IN_PARALLEL <= domain->prd[0]);

    get_zoids(ALLEGRO_SLOPE, domain->boxlo, domain->boxhi, lmp->queues);

    // modify zoids
    // used to be 8, 24, 24, 8
    // 8, 8, 8, 8, 24, 8
    if (NUM_DEPS == 8) {
        lmp->queues[5] = lmp->queues[3];
        lmp->queues[4] = lmp->queues[2];

        lmp->queues[7] = lmp->queues[3];

        std::deque<queue_info> split_dep_1[3];

        int dep_to_split = 1;
        for (int i = 0; i < lmp->queues[dep_to_split].size(); i++) {
            queue_info zoid = lmp->queues[dep_to_split][i];
            int new_dep = -1;
            if (zoid.where[0] == PBC || zoid.where[0] == MIDDLE) {
                assert(zoid.where[1] != PBC && zoid.where[1] != MIDDLE);
                assert(zoid.where[2] != PBC && zoid.where[2] != MIDDLE);
                new_dep = 0;
            } else if (zoid.where[1] == PBC || zoid.where[1] == MIDDLE) {
                assert(zoid.where[0] != PBC && zoid.where[0] != MIDDLE);
                assert(zoid.where[2] != PBC && zoid.where[2] != MIDDLE);
                new_dep = 1;
            } else if (zoid.where[2] == PBC || zoid.where[2] == MIDDLE) {
                assert(zoid.where[0] != PBC && zoid.where[0] != MIDDLE);
                assert(zoid.where[1] != PBC && zoid.where[1] != MIDDLE);
                new_dep = 2;
            } else {
                assert(false);
            }

            split_dep_1[new_dep].push_back(zoid);
        }

        std::deque<queue_info> split_dep_2[3];
        int dep_to_split_2 = 2;
        for (int i = 0; i < lmp->queues[dep_to_split_2].size(); i++) {
            queue_info zoid = lmp->queues[dep_to_split_2][i];
            int new_dep = -1;
            if (zoid.where[0] == LEFT || zoid.where[0] == RIGHT) {
                assert(zoid.where[1] != LEFT && zoid.where[1] != RIGHT);
                assert(zoid.where[2] != LEFT && zoid.where[2] != RIGHT);
                new_dep = 0;
            } else if (zoid.where[1] == LEFT || zoid.where[1] == RIGHT) {
                assert(zoid.where[0] != LEFT && zoid.where[0] != RIGHT);
                assert(zoid.where[2] != LEFT && zoid.where[2] != RIGHT);
                new_dep = 1;
            } else if (zoid.where[2] == LEFT || zoid.where[2] == RIGHT) {
                assert(zoid.where[0] != LEFT && zoid.where[0] != RIGHT);
                assert(zoid.where[1] != LEFT && zoid.where[1] != RIGHT);
                new_dep = 2;
            } else {
                assert(false);
            }

            split_dep_2[new_dep].push_back(zoid);
        }

        lmp->queues[1].clear();
        lmp->queues[2].clear();
        lmp->queues[3].clear();
        lmp->queues[4].clear();
        lmp->queues[5].clear();
        lmp->queues[6].clear();

        for (int i = 0; i < split_dep_1[0].size(); i++) {
            lmp->queues[1].push_back(split_dep_1[0][i]);
        }

        for (int i = 0; i < split_dep_1[1].size(); i++) {
            lmp->queues[2].push_back(split_dep_1[1][i]);
        }

        for (int i = 0; i < split_dep_1[2].size(); i++) {
            lmp->queues[3].push_back(split_dep_1[2][i]);
        }

        for (int i = 0; i < split_dep_2[0].size(); i++) {
            lmp->queues[4].push_back(split_dep_2[0][i]);
        }

        for (int i = 0; i < split_dep_2[1].size(); i++) {
            lmp->queues[5].push_back(split_dep_2[1][i]);
        }

        for (int i = 0; i < split_dep_2[2].size(); i++) {
            lmp->queues[6].push_back(split_dep_2[2][i]);
        }
    }

    for (int dep = 0; dep < NUM_DEPS; dep++) {
        for (int j = 0; j < lmp->queues[dep].size(); j++) {
            queue_info& zoid = lmp->queues[dep][j];
            if (zoid.num % comm->nprocs == comm->me) {
                zoid.can_eval_center = new bool*[NUM_TIMESTEPS_IN_PARALLEL + 1];
                zoid.can_eval_pos = new bool*[NUM_TIMESTEPS_IN_PARALLEL + 1];

                zoid.recv_list_local = new int**[NUM_TIMESTEPS_IN_PARALLEL + 1];
                zoid.recv_list_local_size = new int*[NUM_TIMESTEPS_IN_PARALLEL + 1];

                // for send list
                zoid.send_force_idxs = new int**[NUM_TIMESTEPS_IN_PARALLEL + 1];
                zoid.send_force_sizes = new int**[NUM_TIMESTEPS_IN_PARALLEL + 1];
                zoid.send_force_num_segments = new int*[NUM_TIMESTEPS_IN_PARALLEL + 1];

                zoid.send_pos_idxs = new int**[NUM_TIMESTEPS_IN_PARALLEL + 1];
                zoid.send_pos_sizes = new int**[NUM_TIMESTEPS_IN_PARALLEL + 1];
                zoid.send_pos_num_segments = new int*[NUM_TIMESTEPS_IN_PARALLEL + 1];

                zoid.recv_list_local_force_only = new int**[NUM_TIMESTEPS_IN_PARALLEL + 1];
                zoid.recv_list_local_num_force_only = new int*[NUM_TIMESTEPS_IN_PARALLEL + 1];

                zoid.recv_list_local_force_pos = new int**[NUM_TIMESTEPS_IN_PARALLEL + 1];
                zoid.recv_list_local_num_force_pos = new int*[NUM_TIMESTEPS_IN_PARALLEL + 1];

                // for second send list
                zoid.send_segment_sizes = new int**[NUM_TIMESTEPS_IN_PARALLEL + 1];
                zoid.send_segment_types = new int**[NUM_TIMESTEPS_IN_PARALLEL + 1];
                zoid.send_segment_idxs = new int**[NUM_TIMESTEPS_IN_PARALLEL + 1];
                zoid.send_num_segments = new int*[NUM_TIMESTEPS_IN_PARALLEL + 1];

                zoid.send_local_list = new int**[NUM_TIMESTEPS_IN_PARALLEL + 1];
                /*
                zoid.send_local_sizes = new int**[NUM_TIMESTEPS_IN_PARALLEL + 1];
                zoid.send_local_num_segments = new int*[NUM_TIMESTEPS_IN_PARALLEL + 1];
                zoid.send_local_ghost_segments_mapping = new int**[NUM_TIMESTEPS_IN_PARALLEL + 1];

                zoid.send_ghost_idxs = new int**[NUM_TIMESTEPS_IN_PARALLEL + 1];
                zoid.send_ghost_sizes = new int**[NUM_TIMESTEPS_IN_PARALLEL + 1];
                zoid.send_ghost_num_segments = new int*[NUM_TIMESTEPS_IN_PARALLEL + 1];
                */

                zoid.recv_ghost_idxs = new int**[NUM_TIMESTEPS_IN_PARALLEL + 1];
                zoid.recv_ghost_sizes = new int**[NUM_TIMESTEPS_IN_PARALLEL + 1];
                zoid.recv_ghost_num_segments = new int*[NUM_TIMESTEPS_IN_PARALLEL + 1];

                // need to init this so that "copies" can be made
                zoid.num_elems_send = new int*[NUM_TIMESTEPS_IN_PARALLEL + 1];
                zoid.num_elems_recv = new int*[NUM_TIMESTEPS_IN_PARALLEL + 1];

                zoid.atom_idx_mapping = new int*[NUM_TIMESTEPS_IN_PARALLEL + 1];
            }
        }
    }

    // renumber the zoids
    int zoid_numbering_idx = 0;
    for (int dep = 0; dep < NUM_DEPS; dep++) {
        for (int j = 0; j < lmp->queues[dep].size(); j++) {
            lmp->queues[dep][j].num = zoid_numbering_idx++;
        }
    }

    lmp->zoid_num_to_idx = new int[NUM_ZOIDS];
    lmp->zoid_num_to_zoid = new queue_info[NUM_ZOIDS];
    for (int i = 0; i < NUM_ZOIDS; i++) {
        lmp->zoid_num_to_idx[i] = -1;
    }

    int zoid_idx = 0;
    for (int dep = 0; dep < NUM_DEPS; dep++) {
        for (int j = 0; j < lmp->queues[dep].size(); j++) {
            int zoid_num = lmp->queues[dep][j].num;
            lmp->zoid_num_to_idx[zoid_num] = zoid_idx++;
            assert(zoid_num == zoid_idx - 1);
            lmp->zoid_num_to_zoid[zoid_num] = lmp->queues[dep][j];
        }
    }

    for (int dep = 0; dep < NUM_DEPS; dep++) {
        for (int j = 0; j < lmp->queues[dep].size(); j++) {
            const queue_info& zoid = lmp->queues[dep][j];
            int new_dep = NUM_DEPS - 1 - dep;
            queue_info new_zoid;
            cuts_t new_cuts_t;
            for (int dim = 0; dim < 3; dim++) {
                double new_start = zoid.zoid.cuts[dim].lower + NUM_TIMESTEPS_IN_PARALLEL * zoid.zoid.cuts[dim].slope_lower;
                double new_end = zoid.zoid.cuts[dim].upper + NUM_TIMESTEPS_IN_PARALLEL * zoid.zoid.cuts[dim].slope_upper;
                new_cuts_t.cuts[dim].lower = new_start;
                new_cuts_t.cuts[dim].upper = new_end;
                new_cuts_t.cuts[dim].slope_lower = -1 * zoid.zoid.cuts[dim].slope_lower;
                new_cuts_t.cuts[dim].slope_upper = -1 * zoid.zoid.cuts[dim].slope_upper;
            }
            new_zoid.zoid = new_cuts_t;
            new_zoid.num = zoid.num;
            for (int i = 0; i < 3; i++) {
                new_zoid.where[i] = zoid.where[i];
            }
            lmp->queues_next_dt[new_dep].push_back(new_zoid);
        }
    }

    for (int dep = 0; dep < NUM_DEPS; dep++) {
        for (int j = 0; j < lmp->queues_next_dt[dep].size(); j++) {
            queue_info& zoid = lmp->queues_next_dt[dep][j];
            if (zoid.num % comm->nprocs == comm->me) {
                zoid.can_eval_center = new bool*[NUM_TIMESTEPS_IN_PARALLEL + 1];
                zoid.can_eval_pos = new bool*[NUM_TIMESTEPS_IN_PARALLEL + 1];

                zoid.recv_list_local = new int**[NUM_TIMESTEPS_IN_PARALLEL + 1];
                zoid.recv_list_local_size = new int*[NUM_TIMESTEPS_IN_PARALLEL + 1];

                // for send list
                zoid.send_force_idxs = new int**[NUM_TIMESTEPS_IN_PARALLEL + 1];
                zoid.send_force_sizes = new int**[NUM_TIMESTEPS_IN_PARALLEL + 1];
                zoid.send_force_num_segments = new int*[NUM_TIMESTEPS_IN_PARALLEL + 1];

                zoid.send_pos_idxs = new int**[NUM_TIMESTEPS_IN_PARALLEL + 1];
                zoid.send_pos_sizes = new int**[NUM_TIMESTEPS_IN_PARALLEL + 1];
                zoid.send_pos_num_segments = new int*[NUM_TIMESTEPS_IN_PARALLEL + 1];

                zoid.recv_list_local_force_only = new int**[NUM_TIMESTEPS_IN_PARALLEL + 1];
                zoid.recv_list_local_force_pos = new int**[NUM_TIMESTEPS_IN_PARALLEL + 1];

                zoid.recv_list_local_num_force_only = new int*[NUM_TIMESTEPS_IN_PARALLEL + 1];
                zoid.recv_list_local_num_force_pos = new int*[NUM_TIMESTEPS_IN_PARALLEL + 1];

                // for second send list
                zoid.send_segment_sizes = new int**[NUM_TIMESTEPS_IN_PARALLEL + 1];
                zoid.send_segment_types = new int**[NUM_TIMESTEPS_IN_PARALLEL + 1];
                zoid.send_segment_idxs = new int**[NUM_TIMESTEPS_IN_PARALLEL + 1];
                zoid.send_num_segments = new int*[NUM_TIMESTEPS_IN_PARALLEL + 1];

                zoid.send_local_list = new int**[NUM_TIMESTEPS_IN_PARALLEL + 1];
                /*
                zoid.send_local_sizes = new int**[NUM_TIMESTEPS_IN_PARALLEL + 1];
                zoid.send_local_num_segments = new int*[NUM_TIMESTEPS_IN_PARALLEL + 1];
                zoid.send_local_ghost_segments_mapping = new int**[NUM_TIMESTEPS_IN_PARALLEL + 1];

                zoid.send_ghost_idxs = new int**[NUM_TIMESTEPS_IN_PARALLEL + 1];
                zoid.send_ghost_sizes = new int**[NUM_TIMESTEPS_IN_PARALLEL + 1];
                zoid.send_ghost_num_segments = new int*[NUM_TIMESTEPS_IN_PARALLEL + 1];
                */

                zoid.recv_ghost_idxs = new int**[NUM_TIMESTEPS_IN_PARALLEL + 1];
                zoid.recv_ghost_sizes = new int**[NUM_TIMESTEPS_IN_PARALLEL + 1];
                zoid.recv_ghost_num_segments = new int*[NUM_TIMESTEPS_IN_PARALLEL + 1];

                zoid.num_elems_send = new int*[NUM_TIMESTEPS_IN_PARALLEL + 1];
                zoid.num_elems_recv = new int*[NUM_TIMESTEPS_IN_PARALLEL + 1];

                zoid.atom_idx_mapping = new int*[NUM_TIMESTEPS_IN_PARALLEL + 1];
            }
        }
    }

    lmp->zoid_num_to_zoid_next_dt = new queue_info[NUM_ZOIDS];

    for (int dep = 0; dep < NUM_DEPS; dep++) {
        for (int j = 0; j < lmp->queues_next_dt[dep].size(); j++) {
            int zoid_num = lmp->queues_next_dt[dep][j].num;
            lmp->zoid_num_to_zoid_next_dt[zoid_num] = lmp->queues_next_dt[dep][j];
        }
    }

    // for neighbors
    lmp->send_to_neighbors = new std::vector<int>[NUM_ZOIDS];
    lmp->recv_from_neighbors = new std::vector<int>[NUM_ZOIDS];

    for (int i = 0; i < NUM_ZOIDS; i++) {
        auto &zoid = lmp->zoid_num_to_zoid[i];
        int zoid_dep = get_zoid_dep(zoid.num);
        for (int j = 0; j < NUM_ZOIDS; j++) {
            int zoid_dep_neighbor = get_zoid_dep(j);
            if (zoid_dep_neighbor > zoid_dep && is_close_test(zoid.where, lmp->zoid_num_to_zoid[j].where)) {
                // if (zoid_dep_neighbor > zoid_dep && is_dep(zoid.where, lmp->zoid_num_to_zoid[j].where)) {
                // TODO: test this extra condition
                if (zoid_dep_neighbor == zoid_dep + 1 || true) {
                    lmp->send_to_neighbors[i].push_back(j);
                }
            }

            // if (zoid_dep_neighbor < zoid_dep && is_close(zoid.where, lmp->zoid_num_to_zoid[j].where)) {
            if (zoid_dep_neighbor < zoid_dep && is_close_test(lmp->zoid_num_to_zoid[j].where, zoid.where)) {
                // if (zoid_dep_neighbor < zoid_dep && is_dep(zoid.where, lmp->zoid_num_to_zoid[j].where)) {
                if (zoid_dep_neighbor == zoid_dep - 1 || true) {
                    lmp->recv_from_neighbors[i].push_back(j);
                }
            }
        }
    }

    //  for next dt
    lmp->send_to_neighbors_next_dt = new std::vector<int>[NUM_ZOIDS];
    lmp->recv_from_neighbors_next_dt = new std::vector<int>[NUM_ZOIDS];

    for (int i = 0; i < NUM_ZOIDS; i++) {
        auto &zoid = lmp->zoid_num_to_zoid_next_dt[i];
        int zoid_dep = get_zoid_dep(zoid.num);
        zoid_dep = NUM_DEPS - 1 - zoid_dep;
        for (int j = 0; j < NUM_ZOIDS; j++) {
            int zoid_dep_neighbor = get_zoid_dep(j);
            zoid_dep_neighbor = NUM_DEPS - 1 - zoid_dep_neighbor;
            if (zoid_dep_neighbor > zoid_dep && is_close_test_next_dt(zoid.where, lmp->zoid_num_to_zoid_next_dt[j].where)) {
                // if (zoid_dep_neighbor > zoid_dep && is_dep(zoid.where, lmp->zoid_num_to_zoid[j].where)) {
                // TODO: test this extra condition
                if (zoid_dep_neighbor == zoid_dep + 1 || true) {
                    lmp->send_to_neighbors_next_dt[i].push_back(j);
                }
            }

            // if (zoid_dep_neighbor < zoid_dep && is_close(zoid.where, lmp->zoid_num_to_zoid[j].where)) {
            if (zoid_dep_neighbor < zoid_dep && is_close_test_next_dt(lmp->zoid_num_to_zoid_next_dt[j].where, zoid.where)) {
                // if (zoid_dep_neighbor < zoid_dep && is_dep(zoid.where, lmp->zoid_num_to_zoid[j].where)) {
                if (zoid_dep_neighbor == zoid_dep - 1 || true) {
                    lmp->recv_from_neighbors_next_dt[i].push_back(j);
                }
            }
        }
    }

    if (comm->me == 0) {
        std::map<int, std::set<int>> test;
        for (int dep = 0; dep < NUM_DEPS; dep++) {
            std::cout << "num zoids in dep: " << dep << " is: " << lmp->queues[dep].size() << std::endl;
        }

        for (int i = 0; i < NUM_ZOIDS; i++) {
            for (int recipient : lmp->send_to_neighbors[i]) {
                test[recipient].insert(i);
            }
        }

        for (int i = 0; i < NUM_ZOIDS; i++) {
            if (test[i].size() != lmp->recv_from_neighbors[i].size()) {
                std::cout << "ZOID NUM: " << i << " sent: " << test[i] << " recv: " << lmp->recv_from_neighbors[i] << std::endl;
            }
            assert(test[i].size() == lmp->recv_from_neighbors[i].size());
        }

        // test
        std::map<int, std::set<int>> test_next_dt;
        for (int dep = 0; dep < NUM_DEPS; dep++) {
            std::cout << "num zoids in dep: " << dep << " is: " << lmp->queues_next_dt[dep].size() << std::endl;
        }

        for (int i = 0; i < NUM_ZOIDS; i++) {
            for (int recipient : lmp->send_to_neighbors_next_dt[i]) {
                test_next_dt[recipient].insert(i);
            }
        }

        for (int i = 0; i < NUM_ZOIDS; i++) {
            if (test_next_dt[i].size() != lmp->recv_from_neighbors_next_dt[i].size()) {
                std::cout << "ZOID NUM: " << i << " sent: " << test[i] << " recv: " << lmp->recv_from_neighbors_next_dt[i] << std::endl;
            }
            assert(test_next_dt[i].size() == lmp->recv_from_neighbors_next_dt[i].size());
        }
    }

    // print out send_to_recv info
    if (comm->me == 0) {
        for (int i = 0; i < NUM_ZOIDS; i++) {
            queue_info& zoid = lmp->zoid_num_to_zoid[i];
            std::cout << "zoid: " << i << " lo: " << zoid.zoid.cuts[0].lower << " " << zoid.zoid.cuts[1].lower << " " << zoid.zoid.cuts[2].lower << std::endl;
            std::cout << "zoid: " << i << " hi: " << zoid.zoid.cuts[0].upper << " " << zoid.zoid.cuts[1].upper << " " << zoid.zoid.cuts[2].upper << std::endl;
        }
        for (int i = 0; i < NUM_ZOIDS; i++) {
            std::cout << "zoid: " << i << " send_to: " << lmp->send_to_neighbors[i] << " recv from: " << lmp->recv_from_neighbors[i] << std::endl;
        }
        std::cout << "----------------------------------------------" << std::endl;
        for (int i = 0; i < NUM_ZOIDS; i++) {
            std::cout << "zoid: " << i << " send_to next dt: " << lmp->send_to_neighbors_next_dt[i] << " recv from next_dt: " << lmp->recv_from_neighbors_next_dt[i] << std::endl;
        }
    }

    // create objects
    for (int i = 0; i < zoid_idx; i++) {
        continue;
        Comm* comm;
        if (lmp->kokkos) {
            comm = new CommKokkos(lmp);
        } else {
            comm = new CommBrick(lmp);
        }

        /*
        Domain* domain;
        if (lmp->kokkos) {
            domain = new DomainKokkos(lmp);
        }
#ifdef LMP_OPENMP
            else {
            domain = new DomainOMP(lmp);
        }
#else
        else {
            domain = new Domain(lmp);
        }
#endif


        Neighbor* neighbor;
        if (lmp->kokkos) {
            neighbor = new NeighborKokkos(lmp);
        } else {
            neighbor = new Neighbor(lmp);
        }
        */

        /*
        Modify* modify;
        if (lmp->kokkos) {
            modify = new ModifyKokkos(lmp);
        } else {
            modify = new Modify(lmp);
        }
        */

        /*
        std::array<Atom*, NUM_TIMESTEPS_IN_PARALLEL + 1> atom_arr;
        for (int j = 0; j < atom_arr.size(); j++) {
            Atom* atom;
            if (lmp->kokkos) {
                atom = new AtomKokkos(lmp);
            } else {
                atom = new Atom(lmp);
            }

            // invoked from atom_style command
            // atom->create_avec(input->atom_style_args[0],input->narg_atom_style-1,&input->atom_style_args[1],1);
            // atom->create_avec_stencil_md(input->atom_style_args[0],input->narg_atom_style-1,&input->atom_style_args[1],1);

            if (lmp->kokkos) {
                atom->create_avec_stencil_md("atomic/kk",0,nullptr,1);
            } else {
                atom->create_avec_stencil_md("atomic", 0, nullptr, 1);
            }

            // atom->init();
            // copy over some settings of the current atom class
            // atom->settings(lmp->atom);
            atom_arr[j] = atom;
        }
        */

        /*
        Force* force = lmp->force_stencil_md[i];
        force->init_stencil_md(neighbor);
        domain->init();
        for (int j = 0; j < atom_arr.size(); j++) {
            Atom* atom = atom_arr[j];
            atom->init();
        }
        modify->init();
        // neighbor->init();
        neighbor->init_stencil_md(domain);
        comm->init();
        */

        // lmp->atom_stencil_md.push_back(atom_arr);
        lmp->comm_stencil_md.push_back(comm);
        // lmp->domain_stencil_md.push_back(domain);
        // lmp->neighbor_stencil_md.push_back(neighbor);
        // lmp->modify_stencil_md.push_back(modify);
    }

    // set the domains for each zoid
    // TODO: Only the first 4 dep levels are currently used
    // for (int dep = 0; dep < (3 + 1); dep++) {
    for (int dep = 0; dep < NUM_DEPS; dep++) {
        for (int j = 0; j < lmp->queues[dep].size(); j++) {
            queue_info& zoid = lmp->queues[dep][j];
            int zoid_num = zoid.num;
            int idx_ = lmp->zoid_num_to_idx[zoid_num];
            for (int k = 0; k < lmp->domain_stencil_md[j].size(); k++) {
                Domain* domain_ = lmp->domain_stencil_md[idx_][k];
                for (int dim = 0; dim < 3; dim++) {
                    domain_->sublo[dim] = zoid.zoid.cuts[dim].lower + k * zoid.zoid.cuts[dim].slope_lower;
                    domain_->subhi[dim] = zoid.zoid.cuts[dim].upper + k * zoid.zoid.cuts[dim].slope_upper;
                    domain_->boxlo[dim] = zoid.zoid.cuts[dim].lower + k * zoid.zoid.cuts[dim].slope_lower;
                    domain_->boxhi[dim] = zoid.zoid.cuts[dim].upper + k * zoid.zoid.cuts[dim].slope_upper;
                    domain_->prd[dim] = domain->prd[dim];
                }
            }
        }
    }

    // init
    for (int i = 0; i < NUM_ZOIDS; i++) {
        for (int j = 0; j < lmp->force_stencil_md[i].size(); j++) {
            if (i % comm->nprocs == comm->me) {
                Force* force_ = lmp->force_stencil_md[i][j];
                Neighbor* neighbor_ = lmp->neighbor_stencil_md[i][j];
                Domain* domain_ = lmp->domain_stencil_md[i][j];
                force_->init_stencil_md(neighbor_);
                domain_->init();
            }
        }

        // Force* force_ = lmp->force_stencil_md[i];
        // Neighbor* neighbor_ = lmp->neighbor_stencil_md[i];
        // Domain* domain_ = lmp->domain_stencil_md[i];
        Modify* modify_ = lmp->modify_stencil_md[i];
        Comm* comm_ = lmp->comm_stencil_md[i];

        // force_->init_stencil_md(neighbor_);
        // domain_->init();
        for (int j = 0; j < lmp->atom_stencil_md[i].size(); j++) {
            Atom* atom_ = lmp->atom_stencil_md[i][j];
            atom_->init();
        }
        modify_->init_stencil_md(lmp->atom_stencil_md[i][0]);
        for (int j = 0; j < lmp->force_stencil_md[i].size(); j++) {
            if (i % comm->nprocs == comm->me) {
                Neighbor* neighbor_ = lmp->neighbor_stencil_md[i][j];
                Domain* domain_ = lmp->domain_stencil_md[i][j];
                neighbor_->init_stencil_md(domain_);
            }
        }
        // neighbor_->init_stencil_md(domain_);
        comm_->init();
    }

    // atom setup
    for (int dep = 0; dep < NUM_DEPS; dep++) {
        for (int j = 0; j < lmp->queues[dep].size(); j++) {
            queue_info& zoid = lmp->queues[dep][j];
            int zoid_num = zoid.num;
            int idx_ = lmp->zoid_num_to_idx[zoid_num];
            if (zoid_num % comm->nprocs == comm->me) {
                for (int k = 0; k < lmp->atom_stencil_md[idx_].size(); k++) {
                    lmp->atom_stencil_md[idx_][k]->setup_stencil_md(lmp->domain_stencil_md[idx_][k]);
                }
            }
        }
    }

    std::cout << "NUM ATOMS: " << atom->natoms << std::endl;

    // get local atoms for each zoid
    for (int t = 0; t < NUM_TIMESTEPS_IN_PARALLEL + 1; t++) {
        comm->exchange_stencil_md_initial_send();
        for (int dep = 0; dep < NUM_DEPS; dep++) {
            for (int j = 0; j < lmp->queues[dep].size(); j++) {
                queue_info &zoid = lmp->queues[dep][j];
                int zoid_num = zoid.num;
                // receive only if the zoid belongs to me
                if (zoid_num % comm->nprocs == comm->me) {
                    Atom *first = lmp->atom_stencil_md[zoid_num][t];
                    lmp->comm_stencil_md[zoid_num]->exchange_stencil_md_initial_receive(first,
                                                                                        lmp->domain_stencil_md[zoid_num][t],
                                                                                        zoid);
                    // first->sort_stencil_md();
                }
            }
        }

        MPI_Barrier(world);
    }

    std::cout << "got the num atoms? " << std::endl;

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

    // only receive from lammps for time 0
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

    /*
    for (int t = 1; t < NUM_TIMESTEPS_IN_PARALLEL + 1; t++) {
        // borders
        // fulfill zoid by zoid
        for (int dep = 0; dep < NUM_DEPS; dep++) {
            for (int j = 0; j < lmp->queues[dep].size(); j++) {
                // TODO: fix, somehow link zoid_num_to_zoid and queues
                queue_info& tmp = lmp->queues[dep][j];
                int zoid_num = tmp.num;
                queue_info& zoid = lmp->zoid_num_to_zoid[zoid_num];
                int idx_ = lmp->zoid_num_to_idx[zoid_num];

                std::vector<int> neighbors;
                // neighbors is size 26, or 3^3 - 1

                for (int recv_from : lmp->recv_from_neighbors[zoid_num]) {
                    neighbors.push_back(recv_from);
                }

                for (int send_to : lmp->send_to_neighbors[zoid_num]) {
                    neighbors.push_back(send_to);
                }

                for (int neighbor : neighbors) {
                    if (neighbor % comm->nprocs == comm->me) {
                        queue_info& neighbor_zoid = lmp->zoid_num_to_zoid[neighbor];
                        lmp->comm_stencil_md[neighbor]->borders_stencil_md_initial_send_to_zoid(lmp->atom_stencil_md[neighbor][t - 1],
                                                                                                lmp->domain_stencil_md[neighbor][t],
                                                                                                neighbor_zoid, t, zoid_num);
                    }
                }

                if (zoid_num % comm->nprocs == comm->me) {
                    for (int neighbor : neighbors) {
                        // std::cout << "zoid num: " << zoid_num << " receiving from: " << neighbor << " for time: " << t << std::endl;
                        Atom* atom_ = lmp->atom_stencil_md[idx_][t];
                        int prev_ghost = atom_->nghost;
                        lmp->comm_stencil_md[idx_]->borders_stencil_md_initial_receive_from_zoid(lmp->atom_stencil_md[idx_][t],
                                                                                                 lmp->domain_stencil_md[idx_][t],
                                                                                                 zoid, t, neighbor);
                    }

                    Atom* atom_ = lmp->atom_stencil_md[idx_][t];
                    for (int k = 0; k < atom_->nlocal + atom_->nghost; k++) {
                        atom_->eval_mask_stencil_md[k] = 1;
                    }

                    for (int k = 0; k < atom_->nlocal + atom_->nghost; k++) {
                        atom_->actually_eval_mask_stencil_md[k] = 0;
                    }
                }

                MPI_Barrier(world);
            }
        }

        MPI_Barrier(world);
        std::cout << "me: " << comm->me << " finished sending for time: " << t << std::endl;
    }
    */

    /*
    // ghost send atoms to local?, then have one zoid promote its own local atoms from t - 1?
    // fill out local atoms first?
    for (int t = 1; t < NUM_TIMESTEPS_IN_PARALLEL + 1; t++) {
        for (int dep = 0; dep < NUM_DEPS; dep++) {
            for (int j = 0; j < lmp->queues[dep].size(); j++) {
                queue_info &zoid = lmp->queues[dep][j];
                int zoid_num = zoid.num;

                if (zoid_num % comm->nprocs == comm->me) {
                    for (int neighbor: lmp->send_to_neighbors[zoid_num]) {
                        queue_info &neighbor_zoid = lmp->zoid_num_to_zoid[neighbor];
                        lmp->comm_stencil_md[zoid_num]->exchange_stencil_md_initial_send_to_zoid(
                                lmp->atom_stencil_md[zoid_num][t - 1],
                                zoid, neighbor_zoid, t);
                    }

                    Atom *first = lmp->atom_stencil_md[zoid_num][t - 1];
                    Atom *curr = lmp->atom_stencil_md[zoid_num][t];

                    // receive local from other zoid's ghosts
                    for (int neighbor: lmp->recv_from_neighbors[zoid_num]) {
                        Atom *atom_ = lmp->atom_stencil_md[zoid_num][t];
                        int prev_nlocal = atom_->nlocal;
                        lmp->comm_stencil_md[zoid_num]->exchange_stencil_md_initial_receive_from_zoid(
                                lmp->atom_stencil_md[zoid_num][t],
                                zoid, lmp->zoid_num_to_zoid[neighbor], t);
                    }

                    int prev_nlocal = curr->nlocal;
                    for (int i = 0; i < first->nlocal; i++) {
                        bool in_zoid = true;
                        for (int dim = 0; dim < 3; dim++) {
                            double pos = first->x[i][dim];
                            double lo = zoid.zoid.cuts[dim].lower + (t) * zoid.zoid.cuts[dim].slope_lower;
                            double hi = zoid.zoid.cuts[dim].upper + (t) * zoid.zoid.cuts[dim].slope_upper;
                            in_zoid = in_zoid && (pos >= lo && pos <= hi);
                        }

                        // add to local
                        if (in_zoid) {
                            double buf[11];
                            int m = 0;
                            buf[m++] = 0;
                            buf[m++] = first->x[i][0];
                            buf[m++] = first->x[i][1];
                            buf[m++] = first->x[i][2];
                            buf[m++] = first->v[i][0];
                            buf[m++] = first->v[i][1];
                            buf[m++] = first->v[i][2];
                            buf[m++] = ubuf(first->tag[i]).d;
                            buf[m++] = ubuf(first->type[i]).d;
                            buf[m++] = ubuf(first->mask[i]).d;
                            buf[m++] = ubuf(first->image[i]).d;
                            int y = curr->avec->unpack_exchange_stencil_md(buf, curr, domain, LAMMPS_SEND_LOCAL);
                        }
                    }
                }
            }
        }

        MPI_Barrier(world);

        int total = 0;
        for (int dep = 0; dep < NUM_DEPS; dep++) {
            for (int j = 0; j < lmp->queues[dep].size(); j++) {
                queue_info& zoid = lmp->queues[dep][j];
                if (zoid.num % comm->nprocs == comm->me) {
                    total += lmp->atom_stencil_md[zoid.num][t]->nlocal;
                }
            }
        }

        MPI_Allreduce(&total, &total, 1, MPI_INT, MPI_SUM, world);
        if (total != atom->natoms) {
            std::cout << "timestep: " << t << " num atoms I have: " << total << " num atoms: " << atom->natoms << std::endl;
        }
        assert(total == atom->natoms);
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
            std::cout << "timestep: " << t << " num atoms I have: " << total << " num atoms: " << atom->natoms << std::endl;
        }
        assert(total == atom->natoms);
    }

    MPI_Barrier(world);

    std::cout << GREEN << "local atoms for all zoids satisfied I think" << RESET_COLOR << std::endl;

    if (comm->me == 0) {
        for (int dep = 0; dep < NUM_DEPS; dep++) {
            for (int j = 0; j < lmp->queues[dep].size(); j++) {
                queue_info &zoid = lmp->queues[dep][j];
                int zoid_num = zoid.num;
                // receive only if the zoid belongs to me
                std::cout << "zoid num: " << zoid.num << " bounds" << std::endl;
                for (int dim = 0; dim < 3; dim++) {
                    std::cout << "lo: " << zoid.zoid.cuts[dim].lower << " hi: " << zoid.zoid.cuts[dim].upper << std::endl;
                }
            }
        }

        for (int dep = 0; dep < NUM_DEPS; dep++) {
            for (int j = 0; j < lmp->queues_next_dt[dep].size(); j++) {
                queue_info &zoid = lmp->queues_next_dt[dep][j];
                int zoid_num = zoid.num;
                // receive only if the zoid belongs to me
                std::cout << "next dt zoid num: " << zoid.num << " bounds" << std::endl;
                for (int dim = 0; dim < 3; dim++) {
                    std::cout << "lo: " << zoid.zoid.cuts[dim].lower << " hi: " << zoid.zoid.cuts[dim].upper << std::endl;
                }
            }
        }
    }

    for (int t = 0; t < NUM_TIMESTEPS_IN_PARALLEL + 1; t++) {
        for (int dep = 0; dep < NUM_DEPS; dep++) {
            for (int j = 0; j < lmp->queues[dep].size(); j++) {
                queue_info& zoid = lmp->queues[dep][j];
                int zoid_num = zoid.num;
                int idx_ = lmp->zoid_num_to_idx[zoid_num];
                // receive only if the zoid belongs to me
                if (zoid_num % comm->nprocs == comm->me) {
                    Atom* atom_ = lmp->atom_stencil_md[idx_][t];

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
        for (int i = 0; i < NUM_ZOIDS; i++) {
            std::cout << "send to for zoid: " << i << " is: " << lmp->send_to_neighbors[i] << std::endl;
            std::cout << "recv from for zoid: " << i << " is: " << lmp->recv_from_neighbors[i] << std::endl;
        }
    }

    // group ghost, sort ghost atoms
    for (int t = 0; t < NUM_TIMESTEPS_IN_PARALLEL + 1; t++) {
        for (int dep = 0; dep < NUM_DEPS; dep++) {
            for (int j = 0; j < lmp->queues[dep].size(); j++) {
                queue_info &zoid = lmp->queues[dep][j];
                int zoid_num = zoid.num;
                // receive only if the zoid belongs to me
                if (zoid_num % comm->nprocs == comm->me) {
                    Atom *atom_ = lmp->atom_stencil_md[zoid_num][t];

                    int num_recv_from = lmp->recv_from_neighbors[zoid_num].size();
                    zoid.recv_ghost_idxs[t] = new int*[num_recv_from];
                    zoid.recv_ghost_sizes[t] = new int*[num_recv_from];
                    zoid.recv_ghost_num_segments[t] = new int[num_recv_from];
                    group_ghost_atoms_stencil_md(atom_, NULL, zoid, t);

                    for (int k = 0; k < atom_->nlocal + atom_->nghost; k++) {
                        atom_->eval_mask_stencil_md[k] = 1;
                    }

                    for (int k = 0; k < atom_->nlocal + atom_->nghost; k++) {
                        atom_->actually_eval_mask_stencil_md[k] = 0;
                    }

                    zoid.recv_list_local[t] = new int *[num_recv_from];
                    zoid.recv_list_local_size[t] = new int[num_recv_from];

                    zoid.recv_list_local_force_only[t] = new int *[num_recv_from];
                    zoid.recv_list_local_force_pos[t] = new int *[num_recv_from];
                    zoid.recv_list_local_num_force_only[t] = new int[num_recv_from];
                    zoid.recv_list_local_num_force_pos[t] = new int[num_recv_from];

                    for (int idx = 0; idx < atom_->nlocal + atom_->nghost; idx++) {
                        if (atom_->tag_to_idx.count(atom_->tag[idx])) {
                            std::cout << "repeat tag. zoid: " << zoid_num << std::endl;
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
                queue_info &zoid = lmp->queues_next_dt[dep][j];
                int zoid_num = zoid.num;
                // receive only if the zoid belongs to me
                if (zoid_num % comm->nprocs == comm->me) {
                    Atom *atom_ = lmp->atom_stencil_md[zoid_num][NUM_TIMESTEPS_IN_PARALLEL - t];

                    int num_recv_from = lmp->recv_from_neighbors_next_dt[zoid_num].size();

                    zoid.recv_ghost_idxs[t] = new int*[num_recv_from];
                    zoid.recv_ghost_sizes[t] = new int*[num_recv_from];
                    zoid.recv_ghost_num_segments[t] = new int[num_recv_from];
                    group_ghost_atoms_stencil_md_next_dt(atom_, NULL, zoid, t);

                    zoid.recv_list_local[t] = new int *[num_recv_from];
                    zoid.recv_list_local_size[t] = new int[num_recv_from];

                    zoid.recv_list_local_force_only[t] = new int*[num_recv_from];
                    zoid.recv_list_local_force_pos[t] = new int *[num_recv_from];
                    zoid.recv_list_local_num_force_only[t] = new int[num_recv_from];
                    zoid.recv_list_local_num_force_pos[t] = new int[num_recv_from];
                }
            }
        }
    }

    for (int t = 0; t < NUM_TIMESTEPS_IN_PARALLEL + 1; t++) {
        for (int dep = 0; dep < NUM_DEPS; dep++) {
            for (int j = 0; j < lmp->queues[dep].size(); j++) {
                queue_info &zoid = lmp->queues[dep][j];
                int zoid_num = zoid.num;
                int idx_ = lmp->zoid_num_to_idx[zoid_num];
                if (zoid_num % comm->nprocs == comm->me) {
                    lmp->neighbor_stencil_md[idx_][t]->setup_bins_stencil_md(lmp->atom_stencil_md[idx_][t],
                                                                             lmp->domain_stencil_md[idx_][t],
                                                                             lmp->comm_stencil_md[idx_]);

                    lmp->neighbor_stencil_md[idx_][t]->build_stencil_md(1, lmp->atom_stencil_md[idx_][t],
                                                                        lmp->domain_stencil_md[idx_][t], lmp->comm_stencil_md[idx_]);
                    lmp->neighbor_stencil_md[idx_][t]->ncalls = 0;

                    AtomKokkos* atomKK_ = (AtomKokkos*) lmp->atom_stencil_md[idx_][t];
                    Force* force_ = lmp->force_stencil_md[idx_][t];
                    force_->setup();
                    atomKK_->sync_stencil_md(force->pair->execution_space,force->pair->datamask_read, lmp->atom_stencil_md[idx_][t]);
                    force_clear_stencil_md(lmp->atom_stencil_md[idx_][t], force_, lmp->neighbor_stencil_md[idx_][t]);
                    atomKK_->modified_stencil_md(force_->pair->execution_space, force_->pair->datamask_modify, lmp->atom_stencil_md[idx_][t]);
                }
            }
        }
    }

    MPI_Barrier(world);

    for (int t = 0; t < NUM_TIMESTEPS_IN_PARALLEL + 1; t++) {
        for (int dep = 0; dep < NUM_DEPS; dep++) {
            for (int j = 0; j < lmp->queues[dep].size(); j++) {
                queue_info& zoid = lmp->queues[dep][j];
                int zoid_num = zoid.num;
                if (zoid_num % comm->nprocs == comm->me) {
                    Atom* atom_ = lmp->atom_stencil_md[zoid_num][t];
                    // setup_bins_stencil_md(atom_, zoid, t);
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
                setup_atom_pos_mapping_stencil_md(lmp->atom_stencil_md[zoid_num], lmp->zoid_num_to_zoid[zoid_num]);
                setup_can_eval_center_mapping_stencil_md(lmp->atom_stencil_md[zoid_num], lmp->zoid_num_to_zoid[zoid_num]);
            }
        }
    }

    // setup same things for next_dt
    for (int dep = 0; dep < NUM_DEPS; dep++) {
        for (int j = 0; j < lmp->queues_next_dt[dep].size(); j++) {
            queue_info& zoid = lmp->queues_next_dt[dep][j];
            int zoid_num = zoid.num;
            if (zoid_num % comm->nprocs == comm->me) {
                setup_atom_pos_mapping_stencil_md_next_dt(lmp->atom_stencil_md[zoid_num], zoid);
                setup_can_eval_center_mapping_stencil_md_next_dt(lmp->atom_stencil_md[zoid_num], zoid);
            }
        }
    }

    // Now need to communicate with ther zoids to construct send_list and second_send_list
    for (int dep = 0; dep < NUM_DEPS; dep++) {
        for (int j = 0; j < lmp->queues[dep].size(); j++) {
            queue_info &zoid = lmp->queues[dep][j];
            int zoid_num = zoid.num;
            // receive only if the zoid belongs to me
            if (zoid_num % comm->nprocs == comm->me) {
                lmp->comm_stencil_md[zoid_num]->construct_second_send_list_stencil_md_send(lmp->atom_stencil_md[zoid_num], zoid);
            }
        }
    }

    for (int dep = 0; dep < NUM_DEPS; dep++) {
        for (int j = 0; j < lmp->queues[dep].size(); j++) {
            queue_info &zoid = lmp->queues[dep][j];
            int zoid_num = zoid.num;
            // receive only if the zoid belongs to me
            if (zoid_num % comm->nprocs == comm->me) {
                int num_send_neighbors = lmp->send_to_neighbors[zoid_num].size();
                for (int t = 0; t < NUM_TIMESTEPS_IN_PARALLEL + 1; t++) {
                    zoid.send_segment_sizes[t] = new int*[num_send_neighbors];
                    zoid.send_segment_types[t] = new int*[num_send_neighbors];
                    zoid.send_segment_idxs[t] = new int*[num_send_neighbors];
                    zoid.send_num_segments[t] = new int[num_send_neighbors];

                    zoid.send_local_list[t] = new int*[num_send_neighbors];
                    /*
                    zoid.send_local_sizes[t] = new int*[num_send_neighbors];
                    zoid.send_local_num_segments[t] = new int[num_send_neighbors];
                    zoid.send_local_ghost_segments_mapping[t] = new int*[num_send_neighbors];

                    zoid.send_ghost_idxs[t] = new int*[num_send_neighbors];
                    zoid.send_ghost_sizes[t] = new int*[num_send_neighbors];
                    zoid.send_ghost_num_segments[t] = new int[num_send_neighbors];
                    */
                }
                lmp->comm_stencil_md[zoid_num]->construct_second_send_list_stencil_md(lmp->atom_stencil_md[zoid_num], zoid);
            }
        }
    }

    MPI_Barrier(world);
    if (comm->me == 0) {
        std::cout << "constructed second sendlist for everyone" << std::endl;
    }

    // construct local list now, ghost to local?
    for (int dep = 0; dep < NUM_DEPS; dep++) {
        for (int j = 0; j < lmp->queues[dep].size(); j++) {
            queue_info& zoid = lmp->queues[dep][j];
            int zoid_num = zoid.num;
            if (zoid_num % comm->nprocs == comm->me) {
                int num_send_neighbors = lmp->send_to_neighbors[zoid_num].size();
                for (int t = 0; t < NUM_TIMESTEPS_IN_PARALLEL + 1; t++) {
                    zoid.send_force_idxs[t] = new int*[num_send_neighbors];
                    zoid.send_force_sizes[t] = new int*[num_send_neighbors];
                    zoid.send_force_num_segments[t] = new int[num_send_neighbors];

                    zoid.send_pos_idxs[t] = new int*[num_send_neighbors];
                    zoid.send_pos_sizes[t] = new int*[num_send_neighbors];
                    zoid.send_pos_num_segments[t] = new int[num_send_neighbors];
                }
                lmp->comm_stencil_md[zoid_num]->construct_send_list_stencil_md_send(lmp->atom_stencil_md[zoid_num], zoid);
            }
        }
    }

    for (int dep = 0; dep < NUM_DEPS; dep++) {
        for (int j = 0; j < lmp->queues[dep].size(); j++) {
            queue_info& zoid = lmp->queues[dep][j];
            int zoid_num = zoid.num;
            if (zoid_num % comm->nprocs == comm->me) {
                lmp->comm_stencil_md[zoid_num]->construct_send_list_stencil_md(lmp->atom_stencil_md[zoid_num], zoid);
            }
        }
    }

    MPI_Barrier(world);
    if (comm->me == 0) {
        std::cout << "constructed sendlist" << std::endl;
    }

    // next dt construct lists
    for (int dep = 0; dep < NUM_DEPS; dep++) {
        for (int j = 0; j < lmp->queues_next_dt[dep].size(); j++) {
            queue_info& zoid = lmp->queues_next_dt[dep][j];
            int zoid_num = zoid.num;
            if (zoid_num % comm->nprocs == comm->me) {
                lmp->comm_stencil_md[zoid_num]->construct_second_send_list_stencil_md_next_dt_send(lmp->atom_stencil_md[zoid_num], zoid);
            }
        }
    }

    for (int dep = 0; dep < NUM_DEPS; dep++) {
        for (int j = 0; j < lmp->queues_next_dt[dep].size(); j++) {
            queue_info &zoid = lmp->queues_next_dt[dep][j];
            int zoid_num = zoid.num;
            // receive only if the zoid belongs to me
            if (zoid_num % comm->nprocs == comm->me) {
                int num_send_neighbors = lmp->send_to_neighbors_next_dt[zoid_num].size();
                for (int t = 0; t < NUM_TIMESTEPS_IN_PARALLEL + 1; t++) {
                    zoid.send_segment_sizes[t] = new int*[num_send_neighbors];
                    zoid.send_segment_types[t] = new int*[num_send_neighbors];
                    zoid.send_segment_idxs[t] = new int*[num_send_neighbors];
                    zoid.send_num_segments[t] = new int[num_send_neighbors];

                    zoid.send_local_list[t] = new int*[num_send_neighbors];
                    /*
                    zoid.send_local_sizes[t] = new int*[num_send_neighbors];
                    zoid.send_local_num_segments[t] = new int[num_send_neighbors];
                    zoid.send_local_ghost_segments_mapping[t] = new int*[num_send_neighbors];

                    zoid.send_ghost_idxs[t] = new int*[num_send_neighbors];
                    zoid.send_ghost_sizes[t] = new int*[num_send_neighbors];
                    zoid.send_ghost_num_segments[t] = new int[num_send_neighbors];
                    */
                }
                lmp->comm_stencil_md[zoid_num]->construct_second_send_list_stencil_md_next_dt(lmp->atom_stencil_md[zoid_num], zoid);
            }
        }
    }

    MPI_Barrier(world);
    if (comm->me == 0) {
        std::cout << "nextdt constructed second sendlist" << std::endl;
    }

    for (int dep = 0; dep < NUM_DEPS; dep++) {
        for (int j = 0; j < lmp->queues_next_dt[dep].size(); j++) {
            queue_info& zoid = lmp->queues_next_dt[dep][j];
            int zoid_num = zoid.num;
            if (zoid_num % comm->nprocs == comm->me) {
                int num_send_neighbors = lmp->send_to_neighbors_next_dt[zoid_num].size();
                for (int t = 0; t < NUM_TIMESTEPS_IN_PARALLEL + 1; t++) {
                    zoid.send_force_idxs[t] = new int*[num_send_neighbors];
                    zoid.send_force_sizes[t] = new int*[num_send_neighbors];
                    zoid.send_force_num_segments[t] = new int[num_send_neighbors];

                    zoid.send_pos_idxs[t] = new int*[num_send_neighbors];
                    zoid.send_pos_sizes[t] = new int*[num_send_neighbors];
                    zoid.send_pos_num_segments[t] = new int[num_send_neighbors];
                }
                lmp->comm_stencil_md[zoid_num]->construct_send_list_stencil_md_next_dt_send(lmp->atom_stencil_md[zoid_num], zoid);
            }
        }
    }

    for (int dep = 0; dep < NUM_DEPS; dep++) {
        for (int j = 0; j < lmp->queues_next_dt[dep].size(); j++) {
            queue_info &zoid = lmp->queues_next_dt[dep][j];
            int zoid_num = zoid.num;
            // receive only if the zoid belongs to me
            if (zoid_num % comm->nprocs == comm->me) {
                lmp->comm_stencil_md[zoid_num]->construct_send_list_stencil_md_next_dt(lmp->atom_stencil_md[zoid_num], zoid);
            }
        }
    }

    if (comm->me == 0) {
        std::cout << "constructed sendlist for next dt" << std::endl;
    }

    int total_recv = 0;
    for (int dep = 0; dep < NUM_DEPS; dep++) {
        for (int j = 0; j < lmp->queues[dep].size(); j++) {
            queue_info &zoid = lmp->queues[dep][j];
            int zoid_num = zoid.num;
            // receive only if the zoid belongs to me
            if (zoid_num % comm->nprocs == comm->me) {
                int zoid_recv = 0;
                std::vector<int>& send_to = lmp->send_to_neighbors[zoid_num];
                std::vector<int>& recv_from = lmp->recv_from_neighbors[zoid_num];

                for (int t = 0; t < NUM_TIMESTEPS_IN_PARALLEL + 1; t++) {
                    zoid.num_elems_send[t] = new int[send_to.size()];
                    zoid.num_elems_recv[t] = new int[recv_from.size()];
                }

                for (int i = 0; i < send_to.size(); i++) {
                    int send_zoid_num = send_to[i];

                    for (int t = 0; t < NUM_TIMESTEPS_IN_PARALLEL + 1; t++) {
                        int num_elems_timestep = 0;
                        if (DEBUG_SEND_RECV_DATA) {
                            // 3 elems for data, 1 for the debug tag
                            int num_force_sizes = 0;
                            for (int k = 0; k < zoid.send_force_num_segments[t][i]; k++) {
                                num_elems_timestep += zoid.send_force_sizes[t][i][k] * (3 + 1);
                                num_force_sizes += zoid.send_force_sizes[t][i][k];
                            }

                            int num_pos_sizes = 0;
                            for (int k = 0; k < zoid.send_pos_num_segments[t][i]; k++) {
                                num_elems_timestep += zoid.send_pos_sizes[t][i][k] * 2 * (3 + 1);
                                num_pos_sizes += zoid.send_pos_sizes[t][i][k];
                            }

                            int num_local_sizes = 0;
                            int num_ghost_sizes = 0;

                            int num_ghost_segments = 0;

                            for (int k = 0; k < zoid.send_num_segments[t][i]; k++) {
                                int segment_type = zoid.send_segment_types[t][i][k];
                                if (segment_type == LOCAL_SEGMENT_TYPE) {
                                    num_local_sizes += zoid.send_segment_sizes[t][i][k];
                                } else {
                                    assert(segment_type == GHOST_SEGMENT_TYPE);
                                    num_ghost_sizes += zoid.send_segment_sizes[t][i][k];
                                    num_ghost_segments++;
                                }
                            }

                            num_elems_timestep += (num_local_sizes + num_ghost_sizes) * (3 + 1);

                            zoid.num_elems_send[t][i] = num_elems_timestep;
                            if (num_ghost_segments >= 20) {
                                std::cout << GREEN << "zoid num: " << zoid_num << " send to: " << send_zoid_num << " time: " << t << " debug num ghost segments: " << num_ghost_segments << std::endl;
                            }
                            if (zoid_num == 0 && t == 3 && send_zoid_num == 16) {
                                if (comm->me == 0) {
                                    Atom* atom_ = lmp->atom_stencil_md[zoid_num][t];
                                    std::map<int, std::vector<int>> idx_to_borders_zoids;
                                    std::map<int, std::vector<int>> idx_to_zoids;

                                    for (int h = 0; h < atom_->nlocal + atom_->nghost; h++) {
                                        int target_zoid_prev = -1;
                                        int target_zoid_next = -1;

                                        int target_zoid_curr = -1;
                                        int target_zoid_curr_next_dt = -1;

                                        for (int k = 0; k < NUM_ZOIDS; k++) {
                                            bool in_zoid_curr = true;

                                            queue_info &zoid_tmp = lmp->zoid_num_to_zoid_next_dt[k];
                                            for (int dim = 0; dim < domain->dimension; dim++) {
                                                double value = atom_->x[h][dim];

                                                double lo_curr = zoid_tmp.zoid.cuts[dim].lower +
                                                                 (t) * zoid_tmp.zoid.cuts[dim].slope_lower;
                                                double hi_curr = zoid_tmp.zoid.cuts[dim].upper +
                                                                 (t) * zoid_tmp.zoid.cuts[dim].slope_upper;

                                                double sub = value - domain->prd[dim];
                                                double add = value + domain->prd[dim];

                                                bool at_least_one_curr = (sub >= lo_curr && sub <= hi_curr)
                                                                         || (add >= lo_curr && add <= hi_curr)
                                                                         || (value >= lo_curr && value <= hi_curr);

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

                                            queue_info &zoid_tmp = lmp->zoid_num_to_zoid[k];
                                            for (int dim = 0; dim < domain->dimension; dim++) {
                                                double value = atom_->x[h][dim];

                                                double lo_prev = zoid_tmp.zoid.cuts[dim].lower +
                                                                 (t - 1) * zoid_tmp.zoid.cuts[dim].slope_lower;
                                                double hi_prev = zoid_tmp.zoid.cuts[dim].upper +
                                                                 (t - 1) * zoid_tmp.zoid.cuts[dim].slope_upper;

                                                // TODO: check that this somehow works?
                                                if (t == 0) {
                                                    queue_info &zoid_other_dt = lmp->zoid_num_to_zoid_next_dt[k];
                                                    lo_prev = zoid_other_dt.zoid.cuts[dim].lower +
                                                              (NUM_TIMESTEPS_IN_PARALLEL - 1) *
                                                              zoid_other_dt.zoid.cuts[dim].slope_lower;
                                                    hi_prev = zoid_other_dt.zoid.cuts[dim].upper +
                                                              (NUM_TIMESTEPS_IN_PARALLEL - 1) *
                                                              zoid_other_dt.zoid.cuts[dim].slope_upper;
                                                }

                                                double lo_next = zoid_tmp.zoid.cuts[dim].lower +
                                                                 (t + 1) * zoid_tmp.zoid.cuts[dim].slope_lower;
                                                double hi_next = zoid_tmp.zoid.cuts[dim].upper +
                                                                 (t + 1) * zoid_tmp.zoid.cuts[dim].slope_upper;

                                                // TODO: debug
                                                if (t == NUM_TIMESTEPS_IN_PARALLEL) {
                                                    queue_info &zoid_other_dt = lmp->zoid_num_to_zoid_next_dt[k];
                                                    lo_next = zoid_other_dt.zoid.cuts[dim].lower +
                                                              (1) * zoid_other_dt.zoid.cuts[dim].slope_lower;
                                                    hi_next = zoid_other_dt.zoid.cuts[dim].upper +
                                                              (1) * zoid_other_dt.zoid.cuts[dim].slope_upper;
                                                }

                                                double lo_curr = zoid_tmp.zoid.cuts[dim].lower +
                                                                 (t) * zoid_tmp.zoid.cuts[dim].slope_lower;
                                                double hi_curr = zoid_tmp.zoid.cuts[dim].upper +
                                                                 (t) * zoid_tmp.zoid.cuts[dim].slope_upper;

                                                double sub = value - domain->prd[dim];
                                                double add = value + domain->prd[dim];

                                                bool at_least_one_prev = (sub >= lo_prev && sub <= hi_prev)
                                                                         || (add >= lo_prev && add <= hi_prev)
                                                                         || (value >= lo_prev && value <= hi_prev);

                                                bool at_least_one_curr = (sub >= lo_curr && sub <= hi_curr)
                                                                         || (add >= lo_curr && add <= hi_curr)
                                                                         || (value >= lo_curr && value <= hi_curr);

                                                bool at_least_one_next = (sub >= lo_next && sub <= hi_next)
                                                                         || (add >= lo_next && add <= hi_next)
                                                                         || (value >= lo_next && value <= hi_next);

                                                in_zoid_prev = in_zoid_prev && at_least_one_prev;
                                                in_zoid_curr = in_zoid_curr && at_least_one_curr;
                                                in_zoid_next = in_zoid_next && at_least_one_next;

                                                double lo_borders = lo_curr - ALLEGRO_SLOPE;
                                                double hi_borders = hi_curr + ALLEGRO_SLOPE;

                                                bool at_least_one_borders = (sub >= lo_borders && sub <= hi_borders)
                                                                            || (add >= lo_borders && add <= hi_borders)
                                                                            || (value >= lo_borders &&
                                                                                value <= hi_borders);
                                                borders_zoid = borders_zoid && at_least_one_borders;
                                            }

                                            borders_zoid = borders_zoid && !in_zoid_curr;

                                            if (borders_zoid) {
                                                idx_to_borders_zoids[h].push_back(k);
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

                                        idx_to_zoids[h].push_back(target_zoid_prev);
                                        idx_to_zoids[h].push_back(target_zoid_curr);
                                        idx_to_zoids[h].push_back(target_zoid_next);
                                    }


                                    std::cout << YELLOW << "zoid: " << zoid.num << " send to: " << send_zoid_num << " time: " << t << " num send force: "
                                              << zoid.send_force_num_segments[t][i] << " num send pos: " << zoid.send_pos_num_segments[t][i]
                                              << " send ghost num segments: " << num_ghost_segments << RESET_COLOR << std::endl;
                                    int* local_list = zoid.send_local_list[t][i];
                                    int local_list_idx = 0;
                                    std::set<int> ghost_idxs;
                                    for (int k = 0; k < zoid.send_num_segments[t][i]; k++) {
                                        int segment_type = zoid.send_segment_types[t][i][k];
                                        int sz = zoid.send_segment_sizes[t][i][k];
                                        if (segment_type == LOCAL_SEGMENT_TYPE) {
                                            for (int h = 0; h < sz; h++) {
                                                int idx = local_list[local_list_idx++];
                                                double* pos = atom_->x[idx];
                                                double new_pos[3] = {pos[0], pos[1], pos[2]};
                                                double binsize = ALLEGRO_SLOPE / 2;
                                                double bininv = 1.0/binsize;
                                                for (int dim = 0; dim < 3; dim++) {
                                                    if (new_pos[dim] < 0) {
                                                        new_pos[dim] += domain->prd[dim];
                                                    }
                                                    if (new_pos[dim] > domain->prd[dim]) {
                                                        new_pos[dim] -= domain->prd[dim];
                                                    }
                                                }

                                                int ix = static_cast<int> ((new_pos[0])*bininv);
                                                int iy = static_cast<int> ((new_pos[1])*bininv);
                                                int iz = static_cast<int> ((new_pos[2])*bininv);

                                                std::cout << BLUE << "local idx: " << idx << " pos: " << pos[0] << " " << pos[1] << " " << pos[2]
                                                    << " bin: " << ix << " " << iy << " " << iz << " zoids: " << idx_to_zoids[idx] << " borders: " << idx_to_borders_zoids[idx] << RESET_COLOR << std::endl;
                                            }
                                        } else {
                                            int tmp_idx = zoid.send_segment_idxs[t][i][k];
                                            for (int h = 0; h < sz; h++) {
                                                int idx = tmp_idx + h;
                                                ghost_idxs.insert(idx);
                                            }
                                        }
                                    }

                                    for (int idx = atom_->nlocal; idx < atom_->nlocal + atom_->nghost; idx++) {
                                        double* pos = atom_->x[idx];
                                        double new_pos[3] = {pos[0], pos[1], pos[2]};
                                        double binsize = ALLEGRO_SLOPE / 2;
                                        double bininv = 1.0/binsize;
                                        for (int dim = 0; dim < 3; dim++) {
                                            if (new_pos[dim] < 0) {
                                                new_pos[dim] += domain->prd[dim];
                                            }
                                            if (new_pos[dim] > domain->prd[dim]) {
                                                new_pos[dim] -= domain->prd[dim];
                                            }
                                        }

                                        int ix = static_cast<int> ((new_pos[0])*bininv);
                                        int iy = static_cast<int> ((new_pos[1])*bininv);
                                        int iz = static_cast<int> ((new_pos[2])*bininv);

                                        /*
                                        if (ghost_idxs.find(idx) != ghost_idxs.end()) {
                                            std::cout << GREEN << "SEGMENT ghost idx: " << idx << " pos: " << pos[0] << " " << pos[1] << " " << pos[2]
                                                      << " bin: " << ix << " " << iy << " " << iz << " zoids: " << idx_to_zoids[idx] << " borders: " << idx_to_borders_zoids[idx] << RESET_COLOR << std::endl;
                                        } else {
                                            std::cout << YELLOW << "NOT IN SEGMENT ghost idx: " << idx << " pos: " << pos[0] << " " << pos[1] << " " << pos[2]
                                                      << " bin: " << ix << " " << iy << " " << iz << " zoids: " << idx_to_zoids[idx] << " borders: " << idx_to_borders_zoids[idx] << RESET_COLOR << std::endl;
                                        }
                                        */
                                    }
                                }
                            }
                        }
                    }
                }

                for (int i = 0; i < recv_from.size(); i++) {
                    int recv_zoid_num = recv_from[i];

                    for (int t = 0; t < NUM_TIMESTEPS_IN_PARALLEL + 1; t++) {
                        int num_elems_timestep = 0;
                        if (DEBUG_SEND_RECV_DATA) {
                            // 3 elems for data, 1 for the debug tag
                            num_elems_timestep += zoid.recv_list_local_num_force_only[t][i] * (3 + 1);
                            num_elems_timestep += zoid.recv_list_local_num_force_pos[t][i] * 2 * (3 + 1);

                            int total_ghost_idxs = 0;
                            for (int k = 0; k < zoid.recv_ghost_num_segments[t][i]; k++) {
                                total_ghost_idxs += zoid.recv_ghost_sizes[t][i][k];
                            }

                            num_elems_timestep += total_ghost_idxs * (3 + 1);

                            if (recv_zoid_num % comm->nprocs != comm->me) {
                                /*
                                std::cout << YELLOW << "zoid: " << zoid_num << " receving from: " << recv_zoid_num << " time: " << t
                                          << " num force: " << zoid.recv_list_local_num_force_only[t][i] * 3 << " num local pos + force: " << zoid.recv_list_local_num_force_pos[t][i] * 3
                                          << " num ghost pos + vel: " << total_ghost_idxs * 3 << RESET_COLOR << std::endl;
                                */
                            }
                        }

                        zoid.num_elems_recv[t][i] = num_elems_timestep;
                        if (recv_zoid_num % comm->nprocs != comm->me) {
                            total_recv += zoid.num_elems_recv[t][i];
                            zoid_recv += zoid.num_elems_recv[t][i];
                        }

                        /*
                        std::cout << BLUE << "zoid: " << zoid.num << " recv from: " << recv_zoid_num << " time: " << t << " num recv: "
                                  << zoid.recv_ghost_num_segments[t][i] << RESET_COLOR << std::endl;
                        */
                    }
                }
                // std::cout << BLUE << "zoid: " << zoid_num << " nrecv: " << zoid_recv << " not accounting for tag: " << zoid_recv * 0.75 << RESET_COLOR << std::endl;
            }
        }
    }


    // std::cout << CYAN << "Me total recv: " << total_recv << " not accounting for tag: " << total_recv * 0.75 << RESET_COLOR << std::endl;

    for (int dep = 0; dep < NUM_DEPS; dep++) {
        for (int j = 0; j < lmp->queues_next_dt[dep].size(); j++) {
            queue_info &zoid = lmp->queues_next_dt[dep][j];
            int zoid_num = zoid.num;
            // receive only if the zoid belongs to me
            if (zoid_num % comm->nprocs == comm->me) {
                std::vector<int>& send_to = lmp->send_to_neighbors_next_dt[zoid_num];
                std::vector<int>& recv_from = lmp->recv_from_neighbors_next_dt[zoid_num];

                for (int t = 0; t < NUM_TIMESTEPS_IN_PARALLEL + 1; t++) {
                    zoid.num_elems_send[t] = new int[send_to.size()];
                    zoid.num_elems_recv[t] = new int[recv_from.size()];
                }

                for (int i = 0; i < send_to.size(); i++) {
                    int send_zoid_num = send_to[i];

                    for (int t = 0; t < NUM_TIMESTEPS_IN_PARALLEL + 1; t++) {
                        int num_elems_timestep = 0;
                        if (DEBUG_SEND_RECV_DATA) {
                            // 3 elems for data, 1 for the debug tag
                            for (int k = 0; k < zoid.send_force_num_segments[t][i]; k++) {
                                num_elems_timestep += zoid.send_force_sizes[t][i][k] * (3 + 1);
                            }

                            for (int k = 0; k < zoid.send_pos_num_segments[t][i]; k++) {
                                num_elems_timestep += zoid.send_pos_sizes[t][i][k] * 2 * (3 + 1);
                            }

                            int num_local_sizes = 0;
                            int num_ghost_sizes = 0;

                            int num_ghost_segments = 0;

                            for (int k = 0; k < zoid.send_num_segments[t][i]; k++) {
                                int segment_type = zoid.send_segment_types[t][i][k];
                                if (segment_type == LOCAL_SEGMENT_TYPE) {
                                    num_local_sizes += zoid.send_segment_sizes[t][i][k];
                                } else {
                                    assert(segment_type == GHOST_SEGMENT_TYPE);
                                    num_ghost_sizes += zoid.send_segment_sizes[t][i][k];
                                    num_ghost_segments++;
                                }
                            }

                            num_elems_timestep += (num_local_sizes + num_ghost_sizes) * (3 + 1);

                            zoid.num_elems_send[t][i] = num_elems_timestep;
                            std::cout << MAGENTA << "NEXT DT zoid: " << zoid.num << " send to: " << send_zoid_num << " time: " << t << " num send force: "
                                      << zoid.send_force_num_segments[t][i] << " num send pos: " << zoid.send_pos_num_segments[t][i]
                                      << " send ghost num segments: " << num_ghost_segments << RESET_COLOR << std::endl;
                        }
                    }
                }

                for (int i = 0; i < recv_from.size(); i++) {
                    int recv_zoid_num = recv_from[i];

                    for (int t = 0; t < NUM_TIMESTEPS_IN_PARALLEL + 1; t++) {
                        int num_elems_timestep = 0;
                        if (DEBUG_SEND_RECV_DATA) {
                            // 3 elems for data, 1 for the debug tag
                            num_elems_timestep += zoid.recv_list_local_num_force_only[t][i] * (3 + 1);
                            num_elems_timestep += zoid.recv_list_local_num_force_pos[t][i] * 2 * (3 + 1);

                            int total_ghost_idxs = 0;
                            for (int k = 0; k < zoid.recv_ghost_num_segments[t][i]; k++) {
                                total_ghost_idxs += zoid.recv_ghost_sizes[t][i][k];
                            }

                            num_elems_timestep += total_ghost_idxs * (3 + 1);
                        }

                        zoid.num_elems_recv[t][i] = num_elems_timestep;

                        std::cout << CYAN << "NEXT DT zoid: " << zoid.num << " recv from: " << recv_zoid_num << " time: " << t << " num recv: "
                                  << zoid.recv_ghost_num_segments[t][i] << RESET_COLOR << std::endl;
                    }
                }
            }
        }
    }

    for (int t = 0; t < NUM_TIMESTEPS_IN_PARALLEL + 1; t++) {
        for (int dep = 0; dep < 1; dep++) {
            for (int j = 0; j < lmp->queues[dep].size(); j++) {
                queue_info& zoid = lmp->queues[dep][j];
                int zoid_num = zoid.num;
                if (zoid.num % comm->nprocs == comm->me) {
                    Atom* atom_ = lmp->atom_stencil_md[zoid_num][t];
                    auto& send_to = lmp->send_to_neighbors[zoid_num];
                    std::map<int, std::vector<int>> idx_to_send_zoids;

                    for (int i = 0; i < send_to.size(); i++) {
                        int* local_list = zoid.send_local_list[t][i];
                        int local_list_idx = 0;
                        for (int k = 0; k < zoid.send_num_segments[t][i]; k++) {
                            int segment_type = zoid.send_segment_types[t][i][k];
                            if (segment_type == GHOST_SEGMENT_TYPE) {
                                int ghost_size = zoid.send_segment_sizes[t][i][k];
                                int ghost_idx = zoid.send_segment_idxs[t][i][k];
                                for (int h = 0; h < ghost_size; h++) {
                                    idx_to_send_zoids[ghost_idx + h].push_back(send_to[i]);
                                }
                            } else {
                                int local_size = zoid.send_segment_sizes[t][i][k];
                                for (int h = 0; h < local_size; h++) {
                                    int local_idx = local_list[local_list_idx++];
                                    idx_to_send_zoids[local_idx].push_back(send_to[i]);
                                }
                            }
                        }
                    }

                    if (zoid_num == 0 && t == 2) {
                        std::set<std::tuple<int, int, int, int, int, int, int>> send_zoid_keys;
                        std::map<std::tuple<int, int, int, int, int, int, int>, int> send_zoid_keys_count;

                        std::map<std::tuple<int, int, int, int, int, int, int, int>, int> send_proc_keys_count;
                        for (auto& [idx, send_zoids] : idx_to_send_zoids) {
                            std::tuple<int, int, int, int, int, int, int, int> k_proc = {-1, -1, -1, -1, -1, -1, -1, -1};
                            std::set<int> procs;
                            for (int tmp : send_zoids) {
                                procs.insert(tmp % comm->nprocs);
                            }

                            if (procs.find(0) != procs.end()) {
                                std::get<0>(k_proc) = 1;
                            }
                            if (procs.find(1) != procs.end()) {
                                std::get<1>(k_proc) = 1;
                            }
                            if (procs.find(2) != procs.end()) {
                                std::get<2>(k_proc) = 1;
                            }
                            if (procs.find(3) != procs.end()) {
                                std::get<3>(k_proc) = 1;
                            }
                            if (procs.find(4) != procs.end()) {
                                std::get<4>(k_proc) = 1;
                            }
                            if (procs.find(5) != procs.end()) {
                                std::get<5>(k_proc) = 1;
                            }
                            if (procs.find(6) != procs.end()) {
                                std::get<6>(k_proc) = 1;
                            }
                            if (procs.find(7) != procs.end()) {
                                std::get<7>(k_proc) = 1;
                            }
                            send_proc_keys_count[k_proc]++;

                            /*

                            std::tuple<int, int, int, int, int, int, int> k = {-1, -1, -1, -1, -1, -1, -1};
                            if (send_zoids.size() > 0) {
                                std::get<0>(k) = send_zoids[0];
                            }
                            if (send_zoids.size() > 1) {
                                std::get<1>(k) = send_zoids[1];
                            }
                            if (send_zoids.size() > 2) {
                                std::get<2>(k) = send_zoids[2];
                            }
                            if (send_zoids.size() > 3) {
                                std::get<3>(k) = send_zoids[3];
                            }
                            if (send_zoids.size() > 4) {
                                std::get<4>(k) = send_zoids[4];
                            }
                            if (send_zoids.size() > 5) {
                                std::get<5>(k) = send_zoids[5];
                            }
                            if (send_zoids.size() > 6) {
                                std::get<6>(k) = send_zoids[6];
                            }
                            send_zoid_keys.insert(k);
                            send_zoid_keys_count[k]++;
                            */
                        }

                        for (auto& [key, count] : send_proc_keys_count) {
                            std::set<int> procs;
                            if (std::get<0>(key) == 1) {
                                procs.insert(0);
                            }
                            if (std::get<1>(key) == 1) {
                                procs.insert(1);
                            }
                            if (std::get<2>(key) == 1) {
                                procs.insert(2);
                            }
                            if (std::get<3>(key) == 1) {
                                procs.insert(3);
                            }
                            if (std::get<4>(key) == 1) {
                                procs.insert(4);
                            }
                            if (std::get<5>(key) == 1) {
                                procs.insert(5);
                            }
                            if (std::get<6>(key) == 1) {
                                procs.insert(6);
                            }
                            if (std::get<7>(key) == 1) {
                                procs.insert(7);
                            }
                            std::cout << "procs: " << procs << " count: " << count << std::endl;
                        }
                        // assert(false);
                    }
                }
            }
        }
    }

    MPI_Barrier(world);

    /*
    std::cout << GREEN << "TESTING NEXT DT SEND " << RESET_COLOR << std::endl;
    for (int dep = 0; dep < NUM_DEPS; dep++) {
        for (int j = 0; j < lmp->queues_next_dt[dep].size(); j++) {
            queue_info& zoid = lmp->queues_next_dt[dep][j];
            int zoid_num = zoid.num;
            if (zoid_num % comm->nprocs == comm->me) {
                if (dep > 0) {
                    auto &atom_arr = lmp->atom_stencil_md[zoid_num];
                    Comm *comm_ = lmp->comm_stencil_md[zoid_num];
                    comm_->receive_data_stencil_md_next_dt(atom_arr, lmp->zoid_num_to_zoid_next_dt[zoid_num]);
                }

                if (dep < NUM_DEPS - 1) {
                    auto &atom_arr = lmp->atom_stencil_md[zoid_num];
                    Comm *comm_ = lmp->comm_stencil_md[zoid_num];
                    comm_->send_data_stencil_md_next_dt(atom_arr, lmp->zoid_num_to_zoid_next_dt[zoid_num]);
                }
            }
        }
    }

    MPI_Barrier(world);

    std::cout << GREEN << "TESTING NEXT DT SEND SEEMS TO WORK? " << RESET_COLOR << std::endl;
    */

    // compute force but only for the first timestep

    std::vector<MPI_Request> receive_requests[NUM_ZOIDS];
    std::thread receive_request_threads[NUM_ZOIDS];

    for (int zoid_num = 0; zoid_num < NUM_ZOIDS; zoid_num++) {
        if (zoid_num % comm->nprocs == comm->me) {
            receive_requests[zoid_num].reserve(lmp->recv_from_neighbors[zoid_num].size());
        }
    }

    for (int dep = 0; dep < NUM_DEPS; dep++) {
        for (int j = 0; j < lmp->queues[dep].size(); j++) {
            queue_info &zoid = lmp->queues[dep][j];
            int zoid_num = zoid.num;
            int idx_ = lmp->zoid_num_to_idx[zoid_num];
            assert(zoid_num == idx_);
            if (zoid_num % comm->nprocs == comm->me) {
                if (dep > 0) {
                    auto &atom_arr = lmp->atom_stencil_md[zoid_num];
                    Comm *comm_ = lmp->comm_stencil_md[zoid_num];
                    comm_->receive_data_stencil_md(atom_arr, lmp->zoid_num_to_zoid[zoid_num], receive_requests[zoid_num]);
                    receive_request_threads[zoid_num] =
                            std::move(std::thread([&](int zoid_num_) {
                                /*
                                auto begin = std::chrono::high_resolution_clock::now();
                                int wait_status = MPI_Waitall(receive_requests[zoid_num].size(),
                                                              receive_requests[zoid_num].data(), MPI_STATUSES_IGNORE);
                                assert(wait_status == MPI_SUCCESS);
                                auto end = std::chrono::high_resolution_clock::now();
                                auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count();
                                std::cout << MAGENTA << "zoid: " << zoid_num << " receive wait duration: " << duration << RESET_COLOR << std::endl;
                                */
                                if (receive_requests[zoid_num_].size() > 0) {
                                    auto begin = std::chrono::high_resolution_clock::now();
                                    int wait_status = MPI_Waitall(receive_requests[zoid_num_].size(),
                                                                  receive_requests[zoid_num_].data(), MPI_STATUSES_IGNORE);
                                    assert(wait_status == MPI_SUCCESS);
                                    auto end = std::chrono::high_resolution_clock::now();
                                    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count();
                                }
                            }, zoid_num));
                }
            }
        }
    }

    std::cout << "ok after this shit" << std::endl;

    std::vector<MPI_Request> send_requests[NUM_ZOIDS];
    std::vector<std::thread> send_request_threads;

    for (int dep = 0; dep < NUM_DEPS; dep++) {
        for (int j = 0; j < lmp->queues[dep].size(); j++) {
            queue_info& zoid = lmp->queues[dep][j];
            int zoid_num = zoid.num;
            int idx_ = lmp->zoid_num_to_idx[zoid_num];
            assert(zoid_num == idx_);
            if (zoid_num % comm->nprocs == comm->me) {
                Atom* atom_ = lmp->atom_stencil_md[idx_][0];
                AtomKokkos* atomKK_ = (AtomKokkos*) lmp->atom_stencil_md[idx_][0];
                Force* force_ = lmp->force_stencil_md[idx_][0];
                // todo: eflag and vflag might cause some issues
                // TODO: compute force for each pair in parallel
                // std::cout << "Me: " << comm->me << " setup force compute for zoid: " << zoid_num << std::endl;
                if (dep > 0) {
                    auto &atom_arr = lmp->atom_stencil_md[zoid_num];
                    Comm *comm_ = lmp->comm_stencil_md[zoid_num];
                    auto begin = std::chrono::high_resolution_clock::now();
                    receive_request_threads[zoid_num].join();
                    comm_->unpack_data_stencil_md(atom_arr, lmp->zoid_num_to_zoid[zoid_num], receive_requests[zoid_num]);
                    auto end = std::chrono::high_resolution_clock::now();
                    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end-begin).count();
                    std::cout << BLUE << " zoid: " << zoid_num << " unpack+receive data duration: " << duration << " microseconds " << RESET_COLOR << std::endl;
                }

                atomKK_->sync_stencil_md(force_->pair->execution_space,force_->pair->datamask_read, lmp->atom_stencil_md[idx_][0]);
                force_->pair->compute_stencil_md(eflag, vflag, lmp->atom_stencil_md[idx_][0], lmp->atom_stencil_md[idx_][1],
                                                 zoid.can_eval_center[0], lmp->zoid_num_to_zoid[zoid_num], 0);
                atomKK_->modified_stencil_md(force_->pair->execution_space, force_->pair->datamask_modify, lmp->atom_stencil_md[idx_][0]);

                if (dep < NUM_DEPS - 1) {
                    queue_info& zoid = lmp->zoid_num_to_zoid[zoid_num];
                    auto &atom_arr = lmp->atom_stencil_md[zoid_num];
                    Comm *comm_ = lmp->comm_stencil_md[zoid_num];
                    comm_->send_data_stencil_md(atom_arr, lmp->zoid_num_to_zoid[zoid_num], send_requests[zoid_num]);
                    // int* buf = comm_->send_exclude_eval_tags(atom_arr, lmp->zoid_num_to_zoid[zoid_num]);
                    // send_bufs.push_back(buf);

                    if (send_requests[zoid_num].size() > 0) {
                        send_request_threads.push_back(
                                std::thread([&](int zoid_num_) {
                                    int wait_status = MPI_Waitall(send_requests[zoid_num_].size(),
                                                                  send_requests[zoid_num_].data(), MPI_STATUSES_IGNORE);
                                    assert(wait_status == MPI_SUCCESS);
                                }, zoid_num)
                        );
                    }
                }
            }
        }
    }

    auto join_start = std::chrono::high_resolution_clock::now();
    for (auto& t : send_request_threads) {
        t.join();
    }
    auto join_end = std::chrono::high_resolution_clock::now();
    auto duration_join = std::chrono::duration_cast<std::chrono::microseconds>(join_end - join_start).count();
    std::cout << CYAN << "me: " << comm->me << " duration join: " << duration_join << RESET_COLOR << std::endl;

    MPI_Barrier(world);

    double* send_f = new double[(atom->natoms + 1) * 3];
    // somehow memset did not work
    for (int i = 0; i < (atom->natoms + 1) * 3; i++) {
        send_f[i] = 0.0;
    }

    for (int i = 0; i < atom->nlocal; i++) {
        int tag = atom->tag[i];
        if (!(tag >= 0 && tag <= atom->natoms)) {
            std::cout << "ERROR. " << " idx: " << i << " out of nlocal: " << atom->nlocal << " tag: " << atom->tag[i] << std::endl;
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

    MPI_Allreduce(
            send_f,
            recv_f,
            (atom->natoms + 1) * 3,
            MPI_DOUBLE,
            MPI_SUM,
            world);

    for (int i = 0; i < atom->nlocal; i++) {
        int tag = atom->tag[i];
        for (int dim = 0; dim < 3; dim++) {
            if (fabs(recv_f[tag * 3 + dim] - atom->f[i][dim]) > 5e-5) {
                std::cout << "error in recv tag: " << tag << " idx: " << i << " dim: " << dim << " what I have: " << atom->f[i][dim] << " what I got: " << recv_f[tag * 3 + dim] << " diff: " << fabs(recv_f[tag * 3 + dim] - atom->f[i][dim]) << std::endl;
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
            int idx_ = lmp->zoid_num_to_idx[zoid_num];
            assert(zoid_num == idx_);
            if (zoid_num % comm->nprocs == comm->me) {
                Atom* atom_ = lmp->atom_stencil_md[idx_][0];
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
        std::cout << "STENCIL MD Sum x: " << total_x << " Sum y: " << total_y << " Sum z: " << total_z << std::endl;
    }

    int total_evaled = 0;
    for (int dep = 0; dep < NUM_DEPS; dep++) {
        for (int j = 0; j < lmp->queues[dep].size(); j++) {
            queue_info &zoid = lmp->queues[dep][j];
            int zoid_num = zoid.num;
            if (zoid_num % comm->nprocs == comm->me) {
                Atom *atom_ = lmp->atom_stencil_md[zoid_num][0];
                // Atom *next = lmp->atom_stencil_md[zoid_num][1];
                for (int idx = 0; idx < atom_->nlocal + atom_->nghost; idx++) {
                    if (zoid.can_eval_pos[0][idx]) {
                        int tag = atom_->tag[idx];
                        for (int dim = 0; dim < 3; dim++) {
                            double my_force = atom_->f[idx][dim] + atom_->eval_f_stencil_md[idx][dim];
                            if (fabs(recv_f[tag * 3 + dim] - my_force) > 5e-5) {
                                std::cout << "zoid: " << zoid_num << " idx: " << idx << " tag: " << atom_->tag[idx]
                                          << " dim: " << dim << " what I have: " << atom_->f[idx][dim] << " what lammps has: "
                                          << recv_f[tag * 3 + dim] << " diff: "
                                          << fabs(recv_f[tag * 3 + dim] - atom_->f[idx][dim]) << std::endl;
                                std::cout << "tag: " << atom_->tag[idx] << " pos: " << atom_->x[idx][0] << " " << atom_->x[idx][1] << " " << atom_->x[idx][2] << std::endl;

                                std::cout << "recv force: " << atom_->f[idx][dim] << " eval force: " << atom_->eval_f_stencil_md[idx][dim] << " my force: " << my_force << std::endl;
                            }
                            assert(fabs(my_force - recv_f[tag * 3 + dim]) <= 5e-5);
                        }
                        total_evaled++;
                    }
                }
            }
        }
    }

    int total_atoms_evaled = 0;
    MPI_Allreduce(&total_evaled, &total_atoms_evaled, 1, MPI_INT, MPI_SUM, world);

    std::cout << "atoms evaled: " << total_atoms_evaled << " total number of atoms: " << atom->natoms << std::endl;

    MPI_Barrier(world);

    delete[] send_f;
    delete[] recv_f;

    double total_temp_for_me = 0;
    for (int dep = 0; dep < NUM_DEPS; dep++) {
        for (int j = 0; j < lmp->queues[dep].size(); j++) {
            queue_info& zoid = lmp->queues[dep][j];
            int zoid_num = zoid.num;
            int idx_ = lmp->zoid_num_to_idx[zoid_num];
            if (zoid_num % comm->nprocs == comm->me) {
                lmp->modify_stencil_md[idx_]->setup_stencil_md(&total_temp_for_me, lmp->atom_stencil_md[idx_][0]);
            }
        }
    }

    double total = 0;
    MPI_Allreduce(&total_temp_for_me, &total, 1, MPI_DOUBLE, MPI_SUM, world);
    std::cout << "total temp: " << total << std::endl;

    std::cout << GREEN << "-------- SETUP STENCIL MD PASSED ---------" << RESET_COLOR << std::endl;

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

void Verlet::setup_minimal(int flag)
{
  update->setupflag = 1;

  // setup domain, communication and neighboring
  // acquire ghosts
  // build neighbor lists

  if (flag) {
    modify->setup_pre_exchange();
    if (triclinic) domain->x2lamda(atom->nlocal);
    domain->pbc();
    domain->reset_box();
    comm->setup();
    if (neighbor->style) neighbor->setup_bins();
    comm->exchange();
    comm->borders();
    if (triclinic) domain->lamda2x(atom->nlocal+atom->nghost);
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

  if (pair_compute_flag) force->pair->compute(eflag,vflag);
  else if (force->pair) force->pair->compute_dummy(eflag,vflag);

  if (atom->molecular != Atom::ATOMIC) {
    if (force->bond) force->bond->compute(eflag,vflag);
    if (force->angle) force->angle->compute(eflag,vflag);
    if (force->dihedral) force->dihedral->compute(eflag,vflag);
    if (force->improper) force->improper->compute(eflag,vflag);
  }

  if (force->kspace) {
    force->kspace->setup();
    if (kspace_compute_flag) force->kspace->compute(eflag,vflag);
    else force->kspace->compute_dummy(eflag,vflag);
  }

  modify->setup_pre_reverse(eflag,vflag);
  if (force->newton) comm->reverse_comm();

  modify->setup(vflag);
  update->setupflag = 0;
}

/* ----------------------------------------------------------------------
   run for N steps
------------------------------------------------------------------------- */

void Verlet::run(int n)
{
  bigint ntimestep;
  int nflag,sortflag;

  int n_post_integrate = modify->n_post_integrate;
  int n_pre_exchange = modify->n_pre_exchange;
  int n_pre_neighbor = modify->n_pre_neighbor;
  int n_post_neighbor = modify->n_post_neighbor;
  int n_pre_force = modify->n_pre_force;
  int n_pre_reverse = modify->n_pre_reverse;
  int n_post_force_any = modify->n_post_force_any;
  int n_end_of_step = modify->n_end_of_step;

  if (atom->sortfreq > 0) sortflag = 1;
  else sortflag = 0;

  for (int i = 0; i < n; i++) {
    if (timer->check_timeout(i)) {
      update->nsteps = i;
      break;
    }

    ntimestep = ++update->ntimestep;
    ev_set(ntimestep);

    // initial time integration

    timer->stamp();
    modify->initial_integrate(vflag);
    if (n_post_integrate) modify->post_integrate();
    timer->stamp(Timer::MODIFY);

    // regular communication vs neighbor list rebuild

    nflag = neighbor->decide();

    if (nflag == 0) {
      timer->stamp();
      comm->forward_comm();
      timer->stamp(Timer::COMM);
    } else {
      if (n_pre_exchange) {
        timer->stamp();
        modify->pre_exchange();
        timer->stamp(Timer::MODIFY);
      }
      if (triclinic) domain->x2lamda(atom->nlocal);
      domain->pbc();
      if (domain->box_change) {
        domain->reset_box();
        comm->setup();
        if (neighbor->style) neighbor->setup_bins();
      }
      timer->stamp();
      comm->exchange();
      if (sortflag && ntimestep >= atom->nextsort) atom->sort();
      comm->borders();
      if (triclinic) domain->lamda2x(atom->nlocal+atom->nghost);
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
      modify->pre_force(vflag);
      timer->stamp(Timer::MODIFY);
    }

    if (pair_compute_flag) {
      force->pair->compute(eflag,vflag);
      timer->stamp(Timer::PAIR);
    }

    if (atom->molecular != Atom::ATOMIC) {
      if (force->bond) force->bond->compute(eflag,vflag);
      if (force->angle) force->angle->compute(eflag,vflag);
      if (force->dihedral) force->dihedral->compute(eflag,vflag);
      if (force->improper) force->improper->compute(eflag,vflag);
      timer->stamp(Timer::BOND);
    }

    if (kspace_compute_flag) {
      force->kspace->compute(eflag,vflag);
      timer->stamp(Timer::KSPACE);
    }

    if (n_pre_reverse) {
      modify->pre_reverse(eflag,vflag);
      timer->stamp(Timer::MODIFY);
    }

    // reverse communication of forces

    if (force->newton) {
      comm->reverse_comm();
      timer->stamp(Timer::COMM);
    }

    // force modifications, final time integration, diagnostics

    if (n_post_force_any) modify->post_force(vflag);
    modify->final_integrate();
    if (n_end_of_step) modify->end_of_step();
    timer->stamp(Timer::MODIFY);

    // all output

    if (ntimestep == output->next) {
      timer->stamp();
      output->write(ntimestep);
      timer->stamp(Timer::OUTPUT);
    }
  }
}

/* ---------------------------------------------------------------------- */

void Verlet::cleanup()
{
  modify->post_run();
  domain->box_too_small_check();
  update->update_time();
}

/* ----------------------------------------------------------------------
   clear force on own & ghost atoms
   clear other arrays as needed
------------------------------------------------------------------------- */

void Verlet::force_clear()
{
  size_t nbytes;

  if (external_force_clear) return;

  // clear force on all particles
  // if either newton flag is set, also include ghosts
  // when using threads always clear all forces.

  int nlocal = atom->nlocal;

  if (neighbor->includegroup == 0) {
    nbytes = sizeof(double) * nlocal;
    if (force->newton) nbytes += sizeof(double) * atom->nghost;

    if (nbytes) {
      memset(&atom->f[0][0],0,3*nbytes);
      if (torqueflag) memset(&atom->torque[0][0],0,3*nbytes);
      if (extraflag) atom->avec->force_clear(0,nbytes);
    }

  // neighbor includegroup flag is set
  // clear force only on initial nfirst particles
  // if either newton flag is set, also include ghosts

  } else {
    nbytes = sizeof(double) * atom->nfirst;

    if (nbytes) {
      memset(&atom->f[0][0],0,3*nbytes);
      if (torqueflag) memset(&atom->torque[0][0],0,3*nbytes);
      if (extraflag) atom->avec->force_clear(0,nbytes);
    }

    if (force->newton) {
      nbytes = sizeof(double) * atom->nghost;

      if (nbytes) {
        memset(&atom->f[nlocal][0],0,3*nbytes);
        if (torqueflag) memset(&atom->torque[nlocal][0],0,3*nbytes);
        if (extraflag) atom->avec->force_clear(nlocal,nbytes);
      }
    }
  }
}

void Verlet::force_clear_stencil_md(Atom* atom_, Force* force_, Neighbor* neighbor_) {
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

    for (int i = 0; i < atom_->nlocal + atom_->nghost; i++) {
        for (int j = 0; j < 3; j++) {
            atom_->f[i][j] = 0.0;
            atom_->eval_f_stencil_md[i][j] = 0.0;
        }
    }
}
