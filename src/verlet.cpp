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

  if (pair_compute_flag) {
      force->pair->compute(eflag,vflag);
  } else if (force->pair) {
      force->pair->compute_dummy(eflag,vflag);
  }

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

  std::cout << "LAMMPS LOW: " << domain->sublo[0] << " " << domain->sublo[1] << " " << domain->sublo[2]
    << "LAMMPS HI: " << domain->subhi[0] << " " << domain->subhi[1] << " " << domain->subhi[2] << std::endl;

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

                bool bin_recv_from = bin_borders_zoid;
                if (bin_in_zoid_prev) {
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
}

// TODO: Sort ghost atoms by zoid in previous timestep and zoid in next timestep
// Relay this information to zoids for their sendlists, we only need to ensure ghosts are contiguous. Local atoms we can try to make some compromises since
// there are so few local atoms compared to ghost. TBD though.
// this would mean that on the next dt, there will be segments, hopefully not too many, but we shall see I guess
void Verlet::group_ghost_atoms_stencil_md(Atom* atom_, Atom* prev, queue_info& zoid, int timestep) {
    int zoid_num = zoid.num;

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

    std::stable_sort(ghost_idxs.begin(), ghost_idxs.end(), [&](const int& a, const int& b) {
        std::vector<int>& vec_a = idx_to_zoids[a];
        std::vector<int>& vec_b = idx_to_zoids[b];
        assert(vec_a.size() == vec_b.size());

        int min_vec_size = vec_a.size();

        /*
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
        */

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
    });

    for (int i = 0; i < atom_->nghost; i++) {
        permute[i] = ghost_idxs[i];
    }

    atom_reorder_ghost_stencil_md(atom_, current, permute, 0, atom_->nghost, atom_->nlocal);

    if (zoid_num == 35 && timestep == 2) {
        for (int i = 0; i < ghost_idxs.size(); i++) {
            int lookup_idx = ghost_idxs[i];
            int actual_idx = i + atom_->nlocal;
            auto& in_zoids = idx_to_zoids[lookup_idx];
            auto& borders_zoids = idx_to_borders_zoids[lookup_idx];

            std::set<int> in_zoid_procs;
            for (int zoid_ : in_zoids) {
                in_zoid_procs.insert(zoid_ % comm->nprocs);
            }
            std::set<int> borders_zoid_procs;
            for (int zoid_ : borders_zoids) {
                borders_zoid_procs.insert(zoid_ % comm->nprocs);
            }

            /*
            std::cout << MAGENTA << "i: " << i << " group ghost idx: " << actual_idx
                      << " tag: " << atom_->tag[actual_idx]
                      << " in zoid? " << in_zoids
                      << " borders zoids? " << borders_zoids
                      << " pos: " << atom_->x[actual_idx][0] << " " << atom_->x[actual_idx][1] << " " << atom_->x[actual_idx][2] << RESET_COLOR << std::endl;
            */
        }
    }

    delete[] current;
    delete[] permute;

    std::map<int, std::vector<int>> neighbor_to_idxs;

    auto& recv_from = lmp->recv_from_neighbors[zoid.num];

    assert(ghost_idxs.size() == atom_->nghost);

    std::set<int> ghost_idxs_affected;

    for (int i = 0; i < ghost_idxs.size(); i++) {
        int actual_idx = i + atom_->nlocal;
        double *pos = atom_->x[actual_idx];

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

                double lo_curr = zoid.zoid.cuts[dim].lower + (timestep) * zoid.zoid.cuts[dim].slope_lower;
                double hi_curr = zoid.zoid.cuts[dim].upper + (timestep) * zoid.zoid.cuts[dim].slope_upper;

                // if expanding zoid AND other zoid shrinking, use current instead of past, only need values for actual shrinking?
                // must be a ghost atom that wasn't local last timestep somehow
                if (timestep > 0) {
                    // in_zoid_prev = in_zoid_prev && ((atom_pos_shifted >= lo_prev && atom_pos_shifted <= hi_prev));
                    in_zoid_prev = in_zoid_prev && at_least_one_prev;
                } else {
                    in_zoid_prev = false;
                }

                // if (!(value >= lo_curr && value <= hi_curr) && zoid.zoid.cuts[dim].slope_lower < 0 && timestep < NUM_TIMESTEPS_IN_PARALLEL) {
                if (!(value >= lo_curr && value <= hi_curr) && zoid.zoid.cuts[dim].slope_lower < 0) {
                    in_zoid_prev = false;
                }
            }

            if (in_zoid_prev) {
                neighbor_to_idxs[recv_zoid_num].push_back(i + atom_->nlocal);
            }
        }
    }

    // std::cout << RED << "RYAN zoid: " << zoid.num << " timestep: " << timestep << " num ghost affected? " << ghost_idxs_affected.size() << RESET_COLOR << std::endl;

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

                double lo_curr = zoid.zoid.cuts[dim].lower + (timestep) * zoid.zoid.cuts[dim].slope_lower;
                double hi_curr = zoid.zoid.cuts[dim].upper + (timestep) * zoid.zoid.cuts[dim].slope_upper;

                // if (!(value >= lo_curr && value <= hi_curr) && zoid.zoid.cuts[dim].slope_lower < 0 && timestep < NUM_TIMESTEPS_IN_PARALLEL) {
                if (!(value >= lo_curr && value <= hi_curr) && zoid.zoid.cuts[dim].slope_lower < 0) {
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

            for (int j = 0; j < num_segments; j++) {
                zoid.recv_ghost_idxs[timestep][i][j] = segment_idxs[j];
                zoid.recv_ghost_sizes[timestep][i][j] = segment_lengths[j];
            }
        }
    }
}

void setup_can_eval_center_mapping_stencil_md(std::array<Atom*, NUM_TIMESTEPS_IN_PARALLEL + 1>& atom_arr,
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
                double lo = zoid.zoid.cuts[dim].lower + t * zoid.zoid.cuts[dim].slope_lower;
                double hi = zoid.zoid.cuts[dim].upper + t * zoid.zoid.cuts[dim].slope_upper;

                bool in_bounds = (pos[dim] >= lo && pos[dim] <= hi);

                double diff = std::min(fabs(pos[dim] - lo), fabs(pos[dim] - hi));

                if (shrinking_dim) {
                    can_eval_center = can_eval_center && (in_bounds || (diff <= ALLEGRO_CUTOFF_RADIUS));
                    can_eval_center_debug[dim] = (in_bounds || (diff <= ALLEGRO_CUTOFF_RADIUS));
                } else {
                    can_eval_center = can_eval_center && in_bounds && (diff > ALLEGRO_CUTOFF_RADIUS);
                    can_eval_center_debug[dim] = (in_bounds && diff > ALLEGRO_CUTOFF_RADIUS);
                }
                lo_debug[dim] = lo;
                hi_debug[dim] = hi;
            }

            zoid.can_eval_center[t][i] = can_eval_center;
        }
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
    // assert(3 * 2 * ALLEGRO_SLOPE * NUM_TIMESTEPS_IN_PARALLEL <= domain->prd[0]);

    get_zoids(ALLEGRO_SLOPE, domain->boxlo, domain->boxhi, lmp->queues);

    assert(zoid_to_num_map.size() == NUM_ZOIDS);
    std::set<int> zoid_nums;
    for (auto& [k, v] : zoid_to_num_map) {
        zoid_nums.insert(v);
    }
    assert(zoid_nums.size() == NUM_ZOIDS);

    for (int dep = 0; dep < NUM_DEPS; dep++) {
        for (int j = 0; j < lmp->queues[dep].size(); j++) {
            queue_info& zoid = lmp->queues[dep][j];
            auto key = std::make_tuple(zoid.where[0], zoid.where[1], zoid.where[2]);
            if (zoid_to_num_map.find(key) == zoid_to_num_map.end()) {
                std::cout << "error. key: " << std::get<0>(key) << " " << std::get<1>(key) << " " << std::get<2>(key) << std::endl;
                assert(false);
            }
            lmp->queues[dep][j].num = zoid_to_num_map.at(key);
            assert(zoid.num >= 0 && zoid.num < NUM_ZOIDS);
        }
    }

    if (comm->me == 0) {
        for (int dep = 0; dep < NUM_DEPS; dep++) {
            std::vector<int> nums;
            for (int j = 0; j < lmp->queues[dep].size(); j++) {
                nums.push_back(lmp->queues[dep][j].num);
            }

            std::cout << "DEP: " << dep << " zoid nums: " << nums << std::endl;
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

                // debugging
                zoid.debug_atom_pos = new double*[NUM_TIMESTEPS_IN_PARALLEL + 1];

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

                zoid.recv_ghost_idxs = new int**[NUM_TIMESTEPS_IN_PARALLEL + 1];
                zoid.recv_ghost_sizes = new int**[NUM_TIMESTEPS_IN_PARALLEL + 1];
                zoid.recv_ghost_num_segments = new int*[NUM_TIMESTEPS_IN_PARALLEL + 1];

                // need to init this so that "copies" can be made
                zoid.num_elems_send = new int*[NUM_TIMESTEPS_IN_PARALLEL + 1];
                zoid.num_elems_recv = new int*[NUM_TIMESTEPS_IN_PARALLEL + 1];

                zoid.atom_idx_mapping = new int*[NUM_TIMESTEPS_IN_PARALLEL + 1];

                zoid.recv_process_segment_types = new int**[NUM_TIMESTEPS_IN_PARALLEL + 1];
                zoid.recv_process_segment_idxs = new int**[NUM_TIMESTEPS_IN_PARALLEL + 1];
                zoid.recv_process_segment_sizes = new int**[NUM_TIMESTEPS_IN_PARALLEL + 1];
                zoid.recv_process_num_segments = new int*[NUM_TIMESTEPS_IN_PARALLEL + 1];

                zoid.send_process_segment_sizes = new int**[NUM_TIMESTEPS_IN_PARALLEL + 1];
                zoid.send_process_segment_idxs = new int**[NUM_TIMESTEPS_IN_PARALLEL + 1];
                zoid.send_process_segment_types = new int**[NUM_TIMESTEPS_IN_PARALLEL + 1];
                zoid.send_process_num_segments = new int*[NUM_TIMESTEPS_IN_PARALLEL + 1];
                zoid.send_process_local_list = new int**[NUM_TIMESTEPS_IN_PARALLEL + 1];

                for (int t = 0; t < NUM_TIMESTEPS_IN_PARALLEL + 1; t++) {
                    zoid.send_process_segment_sizes[t] = new int*[comm->nprocs];
                    zoid.send_process_segment_idxs[t] = new int*[comm->nprocs];
                    zoid.send_process_segment_types[t] = new int*[comm->nprocs];
                    zoid.send_process_num_segments[t] = new int[comm->nprocs];
                    zoid.send_process_local_list[t] = new int*[comm->nprocs];
                }

                zoid.num_elems_send_process = new int*[NUM_TIMESTEPS_IN_PARALLEL + 1];
                zoid.num_elems_recv_process = new int*[NUM_TIMESTEPS_IN_PARALLEL + 1];

                for (int t = 0; t < NUM_TIMESTEPS_IN_PARALLEL + 1; t++) {
                    zoid.num_elems_send_process[t] = new int[comm->nprocs];
                    zoid.num_elems_recv_process[t] = new int[comm->nprocs];
                }

                zoid.recv_process_force_offset = new int*[NUM_TIMESTEPS_IN_PARALLEL + 1];
                zoid.recv_process_vel_offset = new int*[NUM_TIMESTEPS_IN_PARALLEL + 1];
                zoid.recv_process_pos_offset = new int*[NUM_TIMESTEPS_IN_PARALLEL + 1];
            }
        }
    }

    lmp->zoid_num_to_zoid = new queue_info[NUM_ZOIDS];
    for (int dep = 0; dep < NUM_DEPS; dep++) {
        for (int j = 0; j < lmp->queues[dep].size(); j++) {
            int zoid_num = lmp->queues[dep][j].num;
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

                zoid.recv_ghost_idxs = new int**[NUM_TIMESTEPS_IN_PARALLEL + 1];
                zoid.recv_ghost_sizes = new int**[NUM_TIMESTEPS_IN_PARALLEL + 1];
                zoid.recv_ghost_num_segments = new int*[NUM_TIMESTEPS_IN_PARALLEL + 1];

                zoid.num_elems_send = new int*[NUM_TIMESTEPS_IN_PARALLEL + 1];
                zoid.num_elems_recv = new int*[NUM_TIMESTEPS_IN_PARALLEL + 1];

                zoid.atom_idx_mapping = new int*[NUM_TIMESTEPS_IN_PARALLEL + 1];

                zoid.recv_process_segment_types = new int**[NUM_TIMESTEPS_IN_PARALLEL + 1];
                zoid.recv_process_segment_idxs = new int**[NUM_TIMESTEPS_IN_PARALLEL + 1];
                zoid.recv_process_segment_sizes = new int**[NUM_TIMESTEPS_IN_PARALLEL + 1];
                zoid.recv_process_num_segments = new int*[NUM_TIMESTEPS_IN_PARALLEL + 1];

                zoid.send_process_segment_sizes = new int**[NUM_TIMESTEPS_IN_PARALLEL + 1];
                zoid.send_process_segment_idxs = new int**[NUM_TIMESTEPS_IN_PARALLEL + 1];
                zoid.send_process_segment_types = new int**[NUM_TIMESTEPS_IN_PARALLEL + 1];
                zoid.send_process_num_segments = new int*[NUM_TIMESTEPS_IN_PARALLEL + 1];
                zoid.send_process_local_list = new int**[NUM_TIMESTEPS_IN_PARALLEL + 1];

                for (int t = 0; t < NUM_TIMESTEPS_IN_PARALLEL + 1; t++) {
                    zoid.send_process_segment_sizes[t] = new int*[comm->nprocs];
                    zoid.send_process_segment_idxs[t] = new int*[comm->nprocs];
                    zoid.send_process_segment_types[t] = new int*[comm->nprocs];
                    zoid.send_process_num_segments[t] = new int[comm->nprocs];
                    zoid.send_process_local_list[t] = new int*[comm->nprocs];
                }

                zoid.num_elems_send_process = new int*[NUM_TIMESTEPS_IN_PARALLEL + 1];
                zoid.num_elems_recv_process = new int*[NUM_TIMESTEPS_IN_PARALLEL + 1];

                for (int t = 0; t < NUM_TIMESTEPS_IN_PARALLEL + 1; t++) {
                    zoid.num_elems_send_process[t] = new int[comm->nprocs];
                    zoid.num_elems_recv_process[t] = new int[comm->nprocs];
                }

                zoid.recv_process_force_offset = new int*[NUM_TIMESTEPS_IN_PARALLEL + 1];
                zoid.recv_process_vel_offset = new int*[NUM_TIMESTEPS_IN_PARALLEL + 1];
                zoid.recv_process_pos_offset = new int*[NUM_TIMESTEPS_IN_PARALLEL + 1];
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

    lmp->send_to_neighbors_procs = new std::vector<int>[NUM_ZOIDS];
    for (int dep = 0; dep < NUM_DEPS; dep++) {
        for (int j = 0; j < lmp->queues[dep].size(); j++) {
            int zoid_num = lmp->queues[dep][j].num;

            auto &send_to = lmp->send_to_neighbors[zoid_num];
            std::set<int> send_procs;
            for (int send_zoid: send_to) {
                send_procs.insert(send_zoid % comm->nprocs);
            }

            for (int proc: send_procs) {
                lmp->send_to_neighbors_procs[zoid_num].push_back(proc);

                if (proc == comm->me &&
                    std::find(lmp->recv_from_neighbors_procs.begin(),
                              lmp->recv_from_neighbors_procs.end(), zoid_num) == lmp->recv_from_neighbors_procs.end()) {
                    lmp->recv_from_neighbors_procs.push_back(zoid_num);
                }
            }
        }
    }

    lmp->send_to_neighbors_procs_next_dt = new std::vector<int>[NUM_ZOIDS];
    for (int i = 0; i < NUM_ZOIDS; i++) {
        auto& send_to = lmp->send_to_neighbors_next_dt[i];
        std::set<int> send_procs;
        for (int send_zoid : send_to) {
            send_procs.insert(send_zoid % comm->nprocs);
        }

        for (int proc : send_procs) {
            lmp->send_to_neighbors_procs_next_dt[i].push_back(proc);

            if (proc == comm->me &&
                std::find(lmp->recv_from_neighbors_procs_next_dt.begin(),
                          lmp->recv_from_neighbors_procs_next_dt.end(), i) == lmp->recv_from_neighbors_procs_next_dt.end()) {
                lmp->recv_from_neighbors_procs_next_dt.push_back(i);
            }
        }
    }

    std::vector<int> zoids_not_same_proc;
    for (int zoid : lmp->recv_from_neighbors_procs) {
        if (zoid % comm->nprocs != comm->me) {
            zoids_not_same_proc.push_back(zoid);
        }
    }

    // number of mpi calls
    int num_naive = 0;
    for (int zoid = 0; zoid < NUM_ZOIDS; zoid++) {
        if (zoid % comm->nprocs == comm->me) {
            for (int other_zoid : lmp->recv_from_neighbors[zoid]) {
                if (other_zoid % comm->nprocs != comm->me) {
                    num_naive++;
                }
            }
        }
    }

    std::cout << "me: " << comm->me << " recv from zoids: " << zoids_not_same_proc << " size? " << zoids_not_same_proc.size() << " num naive? " << num_naive << std::endl;

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

        for (int i = 0; i < NUM_ZOIDS; i++) {
            if (get_zoid_dep(i) == 3) {
                std::vector<int> recv_vec;
                for (int r : lmp->recv_from_neighbors[i]) {
                    if (get_zoid_dep(r) == 2) {
                        recv_vec.push_back(r % comm->nprocs);
                    }
                }

                std::cout << "recv from for zoid: " << i << " is: " << lmp->recv_from_neighbors[i]
                          << " recv procs: " << recv_vec << std::endl;
            }
        }
    }

    // create objects
    for (int i = 0; i < NUM_ZOIDS; i++) {
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
    for (int dep = 0; dep < NUM_DEPS; dep++) {
        for (int j = 0; j < lmp->queues[dep].size(); j++) {
            queue_info& zoid = lmp->queues[dep][j];
            int zoid_num = zoid.num;
            for (int k = 0; k < lmp->domain_stencil_md[j].size(); k++) {
                Domain* domain_ = lmp->domain_stencil_md[zoid_num][k];
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
            if (zoid_num % comm->nprocs == comm->me) {
                for (int k = 0; k < lmp->atom_stencil_md[zoid_num].size(); k++) {
                    lmp->atom_stencil_md[zoid_num][k]->setup_stencil_md(lmp->domain_stencil_md[zoid_num][k]);
                }
            }
        }
    }

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
                }
            }
        }

        MPI_Barrier(world);
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
        for (int i = 0; i < NUM_ZOIDS; i++) {
            std::cout << "send to for zoid: " << i << " is: " << lmp->send_to_neighbors[i] << std::endl;
            std::cout << "recv from for zoid: " << i << " is: " << lmp->recv_from_neighbors[i] << std::endl;
            // std::cout << "send to next dt for zoid: " << i << " is: " << lmp->send_to_neighbors_next_dt[i] << std::endl;
            // std::cout << "recv from next dt for zoid: " << i << " is: " << lmp->recv_from_neighbors_next_dt[i] << std::endl;
        }
    }

    // group ghost, sort ghost atoms
    for (int t = 0; t < NUM_TIMESTEPS_IN_PARALLEL + 1; t++) {
        for (int dep = 0; dep < NUM_DEPS; dep++) {
            for (int j = 0; j < lmp->queues[dep].size(); j++) {
                queue_info &zoid = lmp->queues[dep][j];
                int zoid_num = zoid.num;
                assert(zoid_num >= 0 && zoid_num < NUM_ZOIDS);
                // receive only if the zoid belongs to me
                if (zoid_num % comm->nprocs == comm->me) {
                    Atom *atom_ = lmp->atom_stencil_md[zoid_num][t];

                    int num_recv_from = lmp->recv_from_neighbors[zoid_num].size();
                    assert(num_recv_from >= 0 && num_recv_from <= NUM_ZOIDS);
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
                if (zoid_num % comm->nprocs == comm->me) {
                    lmp->neighbor_stencil_md[zoid_num][t]->setup_bins_stencil_md(lmp->atom_stencil_md[zoid_num][t],
                                                                             lmp->domain_stencil_md[zoid_num][t],
                                                                             lmp->comm_stencil_md[zoid_num]);

                    lmp->neighbor_stencil_md[zoid_num][t]->build_stencil_md(1, lmp->atom_stencil_md[zoid_num][t],
                                                                        lmp->domain_stencil_md[zoid_num][t], lmp->comm_stencil_md[zoid_num]);
                    lmp->neighbor_stencil_md[zoid_num][t]->ncalls = 0;

                    AtomKokkos* atomKK_ = (AtomKokkos*) lmp->atom_stencil_md[zoid_num][t];
                    Force* force_ = lmp->force_stencil_md[zoid_num][t];
                    force_->setup();
                    atomKK_->sync_stencil_md(force->pair->execution_space,force->pair->datamask_read, lmp->atom_stencil_md[zoid_num][t]);
                    force_clear_stencil_md(lmp->atom_stencil_md[zoid_num][t], force_, lmp->neighbor_stencil_md[zoid_num][t]);
                    atomKK_->modified_stencil_md(force_->pair->execution_space, force_->pair->datamask_modify, lmp->atom_stencil_md[zoid_num][t]);
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

                /*
                std::cout << GREEN << "how many can eval through time? zoid: " << zoid.num
                    << " eval? " << count << " local: " << nlocal << " nghost: " << nghost
                    << " num eval per timestep: " << num_eval_timesteps << RESET_COLOR << std::endl;
                */
            }
        }
    }

    /*
    for (int t = 0; t < NUM_TIMESTEPS_IN_PARALLEL + 1; t++) {
        for (int zoid_num = 0; zoid_num < NUM_ZOIDS; zoid_num++) {
            if (zoid_num % comm->nprocs == comm->me) {
                queue_info& zoid = lmp->zoid_num_to_zoid[zoid_num];
                Atom* atom_ = lmp->atom_stencil_md[zoid_num][t];
                for (int k = 0; k < atom_->nlocal + atom_->nghost; k++) {
                    bool can_eval_center = zoid.can_eval_center[t][k];
                    if (atom_->tag[k] == 182315 || atom_->tag[k] == 182347) {
                        if (k < atom_->nlocal) {
                            std::cout << "tag: " << atom_->tag[k] << " time: " << t << " zoid: " << zoid_num << " is nlocal. can eval? " << can_eval_center << std::endl;
                        } else {
                            std::cout << "tag: " << atom_->tag[k] << " time: " << t << " zoid: " << zoid_num << " is ghost. can eval? " << can_eval_center << std::endl;
                        }
                    }
                }
            }
        }
    }
    */

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

    int stencil_md_nghost = 0;
    for (int t = 0; t < NUM_TIMESTEPS_IN_PARALLEL + 1; t++) {
        for (int zoid_num = 0; zoid_num < NUM_ZOIDS; zoid_num++) {
            if (zoid_num % comm->nprocs == comm->me) {
                Atom* atom_ = lmp->atom_stencil_md[zoid_num][t];
                std::cout << CYAN << "zoid num: " << zoid_num << " time: " << t << " nlocal: " << atom_->nlocal << " nghost: " << atom_->nghost << RESET_COLOR << std::endl;
                stencil_md_nghost += atom_->nghost;
            }
        }
    }

    std::cout << "proc: " << comm->me << " LAMMPS nlocal: " << atom->nlocal << " LAMMPS nghost: " << atom->nghost << " stencil md nghost: " << stencil_md_nghost << std::endl;

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
                }
                lmp->comm_stencil_md[zoid_num]->construct_second_send_list_stencil_md(lmp->atom_stencil_md[zoid_num], zoid);
            }
        }
    }

    MPI_Barrier(world);

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
                }
                lmp->comm_stencil_md[zoid_num]->construct_second_send_list_stencil_md_next_dt(lmp->atom_stencil_md[zoid_num], zoid);
            }
        }
    }

    MPI_Barrier(world);

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

    int total_recv = 0;
    for (int dep = 0; dep < NUM_DEPS; dep++) {
        for (int j = 0; j < lmp->queues[dep].size(); j++) {
            queue_info &zoid = lmp->queues[dep][j];
            int zoid_num = zoid.num;
            // receive only if the zoid belongs to me
            if (zoid_num % comm->nprocs == comm->me) {
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
                                // std::cout << GREEN << "zoid num: " << zoid_num << " send to: " << send_zoid_num << " time: " << t << " debug num ghost segments: " << num_ghost_segments << std::endl;
                            }
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
                        if (DEBUG_SEND_RECV_DATA) {
                            // 3 elems for data, 1 for the debug tag
                            num_elems_timestep += zoid.recv_list_local_num_force_only[t][i] * (3 + 1);
                            num_elems_timestep += zoid.recv_list_local_num_force_pos[t][i] * 2 * (3 + 1);

                            int total_ghost_idxs = 0;
                            for (int k = 0; k < zoid.recv_ghost_num_segments[t][i]; k++) {
                                total_ghost_idxs += zoid.recv_ghost_sizes[t][i][k];
                            }

                            num_elems_timestep += total_ghost_idxs * (3 + 1);

                            zoid_recv_ghost_pos += total_ghost_idxs * (3 + 1);
                            zoid_recv_local_force += zoid.recv_list_local_num_force_only[t][i] * (3 + 1);
                            zoid_recv_local_pos += zoid.recv_list_local_num_force_pos[t][i] * 2 * (3 + 1);
                        }

                        zoid.num_elems_recv[t][i] = num_elems_timestep;
                        if (recv_zoid_num % comm->nprocs != comm->me) {
                            total_recv += zoid.num_elems_recv[t][i];
                        }
                    }
                }

                std::cout << "zoid: " << zoid.num << " recv ghost pos: " << zoid_recv_ghost_pos
                    << " recv local pos: " << zoid_recv_local_pos << " recv local force: " << zoid_recv_local_force << std::endl;
            }
        }
    }

    // std::cout << CYAN << "Me total recv: " << total_recv << " not accounting for tag: " << total_recv * 0.75 << RESET_COLOR << std::endl;

    // compute number of elements recv, used just to get a sense of the communication volume induced by stencil md
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
                            /*
                            std::cout << MAGENTA << "NEXT DT zoid: " << zoid.num << " send to: " << send_zoid_num << " time: " << t << " num send force: "
                                      << zoid.send_force_num_segments[t][i] << " num send pos: " << zoid.send_pos_num_segments[t][i]
                                      << " send ghost num segments: " << num_ghost_segments << RESET_COLOR << std::endl;
                            */
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
    std::map<int, std::map<int, std::vector<int>>> zoid_to_idx_to_send_zoids[NUM_TIMESTEPS_IN_PARALLEL + 1];

    for (int t = 0; t < NUM_TIMESTEPS_IN_PARALLEL + 1; t++) {
        for (int dep = 0; dep < NUM_DEPS; dep++) {
            for (int j = 0; j < lmp->queues[dep].size(); j++) {
                queue_info& zoid = lmp->queues[dep][j];
                int zoid_num = zoid.num;
                if (zoid.num % comm->nprocs == comm->me) {
                    // map segment_idx + size to zoids
                    std::map<std::pair<int, int>, std::vector<int>> segment_mapping_to_zoids;

                    Atom* atom_ = lmp->atom_stencil_md[zoid_num][t];
                    auto& send_to = lmp->send_to_neighbors[zoid_num];

                    for (int i = 0; i < send_to.size(); i++) {
                        // ghost to local
                        int num_send_pos_segments = zoid.send_pos_num_segments[t][i];
                        for (int k = 0; k < num_send_pos_segments; k++) {
                            int size = zoid.send_pos_sizes[t][i][k];
                            int start_idx = zoid.send_pos_idxs[t][i][k];
                            segment_mapping_to_zoids[{start_idx, size}].push_back(send_to[i]);
                            for (int h = 0; h < size; h++) {
                                int idx = start_idx + h;
                                zoid_to_idx_to_send_zoids[t][zoid_num][idx].push_back(send_to[i]);
                            }
                        }

                        int* local_list = zoid.send_local_list[t][i];
                        int local_list_idx = 0;
                        for (int k = 0; k < zoid.send_num_segments[t][i]; k++) {
                            int segment_type = zoid.send_segment_types[t][i][k];
                            if (segment_type == GHOST_SEGMENT_TYPE) {
                                int ghost_size = zoid.send_segment_sizes[t][i][k];
                                int ghost_idx = zoid.send_segment_idxs[t][i][k];
                                segment_mapping_to_zoids[{ghost_idx, ghost_size}].push_back(send_to[i]);
                                for (int h = 0; h < ghost_size; h++) {
                                    int idx = ghost_idx + h;
                                    zoid_to_idx_to_send_zoids[t][zoid_num][idx].push_back(send_to[i]);
                                }
                            } else {
                                int local_size = zoid.send_segment_sizes[t][i][k];
                                for (int h = 0; h < local_size; h++) {
                                    int local_idx = local_list[local_list_idx++];
                                    zoid_to_idx_to_send_zoids[t][zoid_num][local_idx].push_back(send_to[i]);
                                }
                            }
                        }
                    }

                    bool debug = false;

                    if (debug) {
                        std::map<int, std::set<int>> ghost_idxs_sent_to_procs;
                        std::map<int, std::vector<int>> local_idxs_sent_to_procs;

                        std::map<int, std::set<int>> ghost_to_local_idxs_sent_to_procs;

                        bool print = (zoid_num == 4 && t == 1);

                        for (int i = 0; i < send_to.size(); i++) {
                            bool print2 = print && send_to[i] % comm->nprocs == 4;
                            int neighbor_proc = send_to[i] % comm->nprocs;
                            int* local_list = zoid.send_local_list[t][i];
                            int num_segments = zoid.send_num_segments[t][i];
                            int local_list_idx = 0;

                            for (int k = 0; k < num_segments; k++) {
                                int segment_type = zoid.send_segment_types[t][i][k];
                                if (segment_type == GHOST_SEGMENT_TYPE) {
                                    int ghost_size = zoid.send_segment_sizes[t][i][k];
                                    int ghost_idx = zoid.send_segment_idxs[t][i][k];
                                    for (int h = 0; h < ghost_size; h++) {
                                        int idx = ghost_idx + h;
                                        ghost_idxs_sent_to_procs[neighbor_proc].insert(idx);
                                        if (print && print2) {
                                            std::cout << YELLOW << "zoid: " << zoid_num
                                                      << " send to: " << send_to[i] << " ghost segment number: " << k
                                                      << " idx: " << idx
                                                      << " send to: " << zoid_to_idx_to_send_zoids[t][zoid_num][idx]
                                                      << " pos: " << atom_->x[idx][0] << " " << atom_->x[idx][1] << " " << atom_->x[idx][2] << " tag: " << atom_->tag[idx] << RESET_COLOR << std::endl;
                                        }
                                    }
                                } else {
                                    int local_size = zoid.send_segment_sizes[t][i][k];
                                    for (int h = 0; h < local_size; h++) {
                                        int local_idx = local_list[local_list_idx++];

                                        auto& vec = local_idxs_sent_to_procs[neighbor_proc];

                                        if (print && print2) {
                                            std::cout << CYAN << "zoid: " << zoid_num << " send to: " << send_to[i]
                                                      << " real local segment number: " << k
                                                      << " idx: " << local_idx
                                                      << " send to: " << zoid_to_idx_to_send_zoids[t][zoid_num][local_idx]
                                                      << " pos: " << atom_->x[local_idx][0] << " " << atom_->x[local_idx][1] << " " << atom_->x[local_idx][2] << " tag: " << atom_->tag[local_idx] << RESET_COLOR << std::endl;
                                        }

                                        if (std::find(vec.begin(), vec.end(), local_idx) == vec.end()) {
                                            local_idxs_sent_to_procs[neighbor_proc].push_back(local_idx);
                                        }
                                    }
                                }
                            }

                            int num_send_pos_segments = zoid.send_pos_num_segments[t][i];
                            for (int k = 0; k < num_send_pos_segments; k++) {
                                int size = zoid.send_pos_sizes[t][i][k];
                                int start_idx = zoid.send_pos_idxs[t][i][k];
                                for (int h = 0; h < size; h++) {
                                    int idx = start_idx + h;
                                    ghost_idxs_sent_to_procs[neighbor_proc].insert(idx);
                                    ghost_to_local_idxs_sent_to_procs[neighbor_proc].insert(idx);
                                    if (print && print2) {
                                        std::cout << YELLOW << "idx to send zids not size 1 wtfel lzoid: " << zoid_num
                                                  << " send to: " << send_to[i] << " send ghost to local segment number: " << k
                                                  << " idx: " << idx
                                                  << " send to: " << zoid_to_idx_to_send_zoids[t][zoid_num][idx]
                                                  << " pos: " << atom_->x[idx][0] << " " << atom_->x[idx][1] << " " << atom_->x[idx][2] << " tag: " << atom_->tag[idx] << RESET_COLOR << std::endl;
                                    }
                                }
                            }
                        }

                        for (int k = 0; k < comm->nprocs; k++) {
                            std::vector<int> vec_idxs;

                            if (ghost_idxs_sent_to_procs.find(k) != ghost_idxs_sent_to_procs.end()) {
                                for (int idx: local_idxs_sent_to_procs[k]) {
                                    vec_idxs.push_back(idx);
                                }
                                for (int idx: ghost_idxs_sent_to_procs[k]) {
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

                                int tmp_num_segments = get_segments(ghost_vec_idxs, tmp_idxs, tmp_lengths, false);
                                if (k != comm->me) {
                                    if (tmp_num_segments >= 15) {
                                        std::cout << "zoid: " << zoid_num << " send to proc: " << k
                                            << " time: " << t
                                            << " tmp num segments for all the ghosts sent in one batch: " << tmp_num_segments
                                            << " idxs: " << tmp_idxs << " lengths: " << tmp_lengths << std::endl;
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    MPI_Barrier(world);

    std::map<std::pair<int, int>, int> send_buf_num_tags[NUM_TIMESTEPS_IN_PARALLEL + 1];
    std::map<std::pair<int, int>, std::vector<int>> send_buf_tags[NUM_TIMESTEPS_IN_PARALLEL + 1];
    std::vector<MPI_Request> send_buf_requests;

    std::map<std::pair<int, int>, std::vector<int>> force_offset_idxs[NUM_TIMESTEPS_IN_PARALLEL + 1];
    std::map<std::pair<int, int>, int> pos_offsets[NUM_TIMESTEPS_IN_PARALLEL + 1];

    std::map<std::pair<int, int>, std::vector<int>> vel_offset_idxs[NUM_TIMESTEPS_IN_PARALLEL + 1];

    // send ghost idx to send buf idx per process to the zoids

    for (int t = 0; t < NUM_TIMESTEPS_IN_PARALLEL + 1; t++) {
        lmp->num_recv_force_from_zoid[t] = new int[lmp->recv_from_neighbors_procs.size()];
        memset(lmp->num_recv_force_from_zoid[t], -1, lmp->recv_from_neighbors_procs.size());
        lmp->num_recv_pos_from_zoid[t] = new int[lmp->recv_from_neighbors_procs.size()];
        memset(lmp->num_recv_pos_from_zoid[t], -1, lmp->recv_from_neighbors_procs.size());
        lmp->num_recv_vel_from_zoid[t] = new int[lmp->recv_from_neighbors_procs.size()];
        memset(lmp->num_recv_vel_from_zoid[t], -1, lmp->recv_from_neighbors_procs.size());
    }

    for (int t = 0; t < NUM_TIMESTEPS_IN_PARALLEL + 1; t++) {
        for (int dep = 0; dep < NUM_DEPS; dep++) {
            for (int j = 0; j < lmp->queues[dep].size(); j++) {
                queue_info& zoid = lmp->queues[dep][j];
                int zoid_num = zoid.num;
                std::vector<int>& send_to = lmp->send_to_neighbors[zoid_num];

                if (zoid_num % comm->nprocs == comm->me) {
                    Atom *atom_ = lmp->atom_stencil_md[zoid_num][t];

                    for (int proc = 0; proc < comm->nprocs; proc++) {
                        std::vector<int> send_process_local_idxs;
                        std::vector<int> send_process_local_sizes;

                        for (int i = 0; i < send_to.size(); i++) {
                            int send_zoid_num = send_to[i];
                            if (send_zoid_num % comm->nprocs == proc) {
                                int *local_list = zoid.send_local_list[t][i];
                                int num_segments = zoid.send_num_segments[t][i];
                                int local_list_idx = 0;
                                for (int k = 0; k < num_segments; k++) {
                                    int segment_type = zoid.send_segment_types[t][i][k];
                                    if (segment_type == LOCAL_SEGMENT_TYPE) {
                                        int local_size = zoid.send_segment_sizes[t][i][k];

                                        int actual_size = 0;
                                        for (int h = 0; h < local_size; h++) {
                                            int local_idx = local_list[local_list_idx++];
                                            if (zoid_to_idx_to_send_zoids[t][zoid_num][local_idx].size() == 1) {
                                                if (std::find(send_process_local_idxs.begin(), send_process_local_idxs.end(), local_idx)
                                                    == send_process_local_idxs.end()) {
                                                    send_process_local_idxs.push_back(local_idx);
                                                    actual_size++;
                                                } else {
                                                    assert(false);
                                                }
                                            }
                                        }

                                        // send_process_local_sizes.push_back(local_size);
                                        if (actual_size > 0) {
                                            send_process_local_sizes.push_back(actual_size);
                                        }
                                    }
                                }
                            }
                        }

                        for (int i = 0; i < send_to.size(); i++) {
                            int send_zoid_num = send_to[i];
                            if (send_zoid_num % comm->nprocs == proc) {
                                int *local_list = zoid.send_local_list[t][i];
                                int num_segments = zoid.send_num_segments[t][i];
                                int local_list_idx = 0;
                                for (int k = 0; k < num_segments; k++) {
                                    int segment_type = zoid.send_segment_types[t][i][k];
                                    if (segment_type == LOCAL_SEGMENT_TYPE) {
                                        int local_size = zoid.send_segment_sizes[t][i][k];

                                        int actual_size = 0;
                                        for (int h = 0; h < local_size; h++) {
                                            int local_idx = local_list[local_list_idx++];

                                            if (std::find(send_process_local_idxs.begin(), send_process_local_idxs.end(), local_idx)
                                                == send_process_local_idxs.end()) {
                                                send_process_local_idxs.push_back(local_idx);
                                                actual_size++;
                                            }
                                        }

                                        // send_process_local_sizes.push_back(local_size);
                                        if (actual_size > 0) {
                                            send_process_local_sizes.push_back(actual_size);
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
                                    int segment_type = zoid.send_segment_types[t][i][k];
                                    if (segment_type == GHOST_SEGMENT_TYPE) {
                                        int ghost_size = zoid.send_segment_sizes[t][i][k];
                                        int ghost_idx = zoid.send_segment_idxs[t][i][k];
                                        for (int h = 0; h < ghost_size; h++) {
                                            int idx = ghost_idx + h;
                                            assert(idx < atom_->nlocal + atom_->nghost);
                                            send_process_ghost_idxs_set.insert(idx);
                                        }
                                    }
                                }

                                int num_send_pos_segments = zoid.send_pos_num_segments[t][i];
                                for (int k = 0; k < num_send_pos_segments; k++) {
                                    int size = zoid.send_pos_sizes[t][i][k];
                                    int start_idx = zoid.send_pos_idxs[t][i][k];
                                    for (int h = 0; h < size; h++) {
                                        int idx = start_idx + h;
                                        assert(idx < atom_->nlocal + atom_->nghost);
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
                        int num_ghost_segments = get_segments(send_process_ghost_idxs_vec, send_process_ghost_segment_idxs, send_process_ghost_segment_sizes);

                        int num_local_segments = send_process_local_sizes.size();

                        // construct send process information
                        zoid.send_process_num_segments[t][proc] = num_local_segments + num_ghost_segments;
                        zoid.send_process_segment_types[t][proc] = new int[num_local_segments + num_ghost_segments];
                        zoid.send_process_segment_idxs[t][proc] = new int[num_local_segments + num_ghost_segments];
                        zoid.send_process_segment_sizes[t][proc] = new int[num_local_segments + num_ghost_segments];

                        zoid.send_process_local_list[t][proc] = new int[send_process_local_idxs.size()];
                        for (int k = 0; k < send_process_local_idxs.size(); k++) {
                            zoid.send_process_local_list[t][proc][k] = send_process_local_idxs[k];
                        }

                        int num_elems_send = 0;
;
                        for (int k = 0; k < num_local_segments; k++) {
                            zoid.send_process_segment_types[t][proc][k] = SEND_DATA_PROCESS_LOCAL;
                            zoid.send_process_segment_idxs[t][proc][k] = -2;
                            zoid.send_process_segment_sizes[t][proc][k] = send_process_local_sizes[k];

                            num_elems_send += send_process_local_sizes[k];
                        }

                        for (int k = 0; k < num_ghost_segments; k++) {
                            int segment_num = k + num_local_segments;
                            zoid.send_process_segment_types[t][proc][segment_num] = SEND_DATA_PROCESS_GHOST;
                            zoid.send_process_segment_idxs[t][proc][segment_num] = send_process_ghost_segment_idxs[k];
                            zoid.send_process_segment_sizes[t][proc][segment_num] = send_process_ghost_segment_sizes[k];

                            if (send_process_ghost_segment_idxs[k] + send_process_ghost_segment_sizes[k] > atom_->nlocal + atom_->nghost) {
                                std::cout << RED << "zoid: " << zoid_num << " send to proc: " << proc << " segment num: " << segment_num << " time: " << t <<
                                    " segment idx: " << send_process_ghost_segment_idxs[k] << " size: " << send_process_ghost_segment_sizes[k] <<
                                    " num local: " << atom_->nlocal << " total: " << atom_->nlocal + atom_->nghost << RESET_COLOR << std::endl;
                                for (int h = 0; h < num_ghost_segments; h++) {
                                    std::cout << "ghost segment: " << h << " idx: " << send_process_ghost_segment_idxs[h] << " size: " << send_process_ghost_segment_sizes[h] << std::endl;
                                }
                            }
                            assert(send_process_ghost_segment_idxs[k] + send_process_ghost_segment_sizes[k] <= atom_->nlocal + atom_->nghost);

                            num_elems_send += send_process_ghost_segment_sizes[k];
                        }

                        zoid.num_elems_send_process[t][proc] = num_elems_send;

                        for (int idx : send_process_local_idxs) {
                            send_buf_tags[t][{zoid_num, proc}].push_back(atom_->tag[idx]);
                        }

                        for (int idx : send_process_ghost_idxs_vec) {
                            send_buf_tags[t][{zoid_num, proc}].push_back(atom_->tag[idx]);
                        }

                        send_buf_num_tags[t][{zoid_num, proc}] = send_process_local_idxs.size() + send_process_ghost_idxs_vec.size();

                        // 0 offset for the first force sent
                        force_offset_idxs[t][{zoid_num, proc}].push_back(0);
                        int num_send_force = 0;
                        for (int i = 0; i < send_to.size(); i++) {
                            if (send_to[i] % comm->nprocs == proc) {
                                int num_force_segments = zoid.send_force_num_segments[t][i];
                                int send_force_size = 0;
                                for (int k = 0; k < num_force_segments; k++) {
                                    int segment_size = zoid.send_force_sizes[t][i][k];
                                    send_force_size += segment_size;
                                }

                                int prev_size = force_offset_idxs[t][{zoid_num, proc}][force_offset_idxs[t][{zoid_num, proc}].size() - 1];
                                force_offset_idxs[t][{zoid_num, proc}].push_back(send_force_size + prev_size);

                                num_send_force += send_force_size;
                            }
                        }

                        int num_send_vel = 0;
                        vel_offset_idxs[t][{zoid_num, proc}].push_back(0);
                        for (int i = 0; i < send_to.size(); i++) {
                            if (send_to[i] % comm->nprocs == proc) {
                                int num_vel_segments = zoid.send_pos_num_segments[t][i];
                                int send_vel_size = 0;
                                for (int k = 0; k < num_vel_segments; k++) {
                                    int segment_size = zoid.send_pos_sizes[t][i][k];
                                    send_vel_size += segment_size;
                                }

                                int prev_size = vel_offset_idxs[t][{zoid_num, proc}][vel_offset_idxs[t][{zoid_num, proc}].size() - 1];
                                vel_offset_idxs[t][{zoid_num, proc}].push_back(send_vel_size + prev_size);

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

                                MPI_Isend(&send_buf_num_tags[t][{zoid_num, proc}], 1, MPI_INT, proc, mpi_tag, world, &r1);

                                MPI_Isend(send_buf_tags[t][{zoid_num, proc}].data(),
                                          send_buf_num_tags[t][{zoid_num, proc}], MPI_INT, proc, mpi_tag, world, &r2);

                                MPI_Isend(&force_offset_idxs[t][{zoid_num, proc}][force_offset_vec_idx++], 1, MPI_INT, proc, mpi_tag, world, &r3);

                                MPI_Isend(&pos_offsets[t][{zoid_num, proc}], 1, MPI_INT, proc, mpi_tag, world, &r4);

                                MPI_Isend(&force_offset_idxs[t][{zoid_num, proc}][force_offset_idxs[t][{zoid_num, proc}].size() - 1],
                                          1, MPI_INT, proc, mpi_tag, world, &r5);

                                MPI_Isend(&zoid.num_elems_send_process[t][proc],
                                          1, MPI_INT, proc, mpi_tag, world, &r6);

                                MPI_Isend(&vel_offset_idxs[t][{zoid_num, proc}][vel_offset_vec_idx++],
                                          1, MPI_INT, proc, mpi_tag, world, &r7);

                                MPI_Isend(&vel_offset_idxs[t][{zoid_num, proc}][vel_offset_idxs[t][{zoid_num, proc}].size() - 1],
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
                queue_info &zoid = lmp->queues[dep][j];
                int zoid_num = zoid.num;
                std::vector<int> &recv_from = lmp->recv_from_neighbors[zoid_num];

                if (zoid_num % comm->nprocs == comm->me) {
                    Atom* atom_ = lmp->atom_stencil_md[zoid_num][t];
                    zoid.recv_process_segment_types[t] = new int*[recv_from.size()];
                    zoid.recv_process_segment_idxs[t] = new int*[recv_from.size()];
                    zoid.recv_process_segment_sizes[t] = new int*[recv_from.size()];
                    zoid.recv_process_num_segments[t] = new int[recv_from.size()];

                    // send idxs belonging to that zoid?
                    zoid.recv_process_force_offset[t] = new int[recv_from.size()];
                    zoid.recv_process_pos_offset[t] = new int[recv_from.size()];
                    zoid.recv_process_vel_offset[t] = new int[recv_from.size()];

                    for (int i = 0; i < recv_from.size(); i++) {
                        int recv_zoid_num = recv_from[i];
                        int recv_proc = recv_zoid_num % comm->nprocs;

                        int mpi_tag = (zoid_num << 16 | recv_zoid_num);
                        int num_tags = 0;
                        MPI_Recv(&num_tags, 1, MPI_INT, recv_proc, mpi_tag, world,
                                 MPI_STATUS_IGNORE);

                        int *recv_tags = new int[num_tags];
                        MPI_Recv(recv_tags, num_tags, MPI_INT, recv_proc, mpi_tag, world,
                                 MPI_STATUS_IGNORE);

                        int force_offset_idx = 0;
                        MPI_Recv(&force_offset_idx, 1, MPI_INT, recv_proc, mpi_tag, world,
                                 MPI_STATUS_IGNORE);

                        int num_force_recv = 0;
                        MPI_Recv(&num_force_recv, 1, MPI_INT, recv_proc, mpi_tag, world,
                                 MPI_STATUS_IGNORE);

                        // offset within buffer of forces, which one belongs to this zoid
                        int force_offset = 0;
                        MPI_Recv(&force_offset, 1, MPI_INT, recv_proc, mpi_tag, world,
                                 MPI_STATUS_IGNORE);

                        int num_pos_recv = 0;
                        MPI_Recv(&num_pos_recv, 1, MPI_INT, recv_proc, mpi_tag, world,
                                 MPI_STATUS_IGNORE);

                        int vel_offset = 0;
                        MPI_Recv(&vel_offset, 1, MPI_INT, recv_proc, mpi_tag, world,
                                 MPI_STATUS_IGNORE);

                        int num_vel_recv = 0;
                        MPI_Recv(&num_vel_recv, 1, MPI_INT, recv_proc, mpi_tag, world,
                                 MPI_STATUS_IGNORE);

                        for (int k = 0; k < lmp->recv_from_neighbors_procs.size(); k++) {
                            if (lmp->recv_from_neighbors_procs[k] == recv_from[i]) {
                                lmp->num_recv_force_from_zoid[t][k] = num_force_recv;
                                lmp->num_recv_pos_from_zoid[t][k] = num_pos_recv;
                                lmp->num_recv_vel_from_zoid[t][k] = num_vel_recv;
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
                        for (int k = 0; k < zoid.recv_list_local_num_force_pos[t][i]; k++) {
                            int idx = zoid.recv_list_local_force_pos[t][i][k];
                            int buf_idx = tag_to_idx_in_buf[atom_->tag[idx]];
                            local_buf_idxs.push_back(buf_idx);
                        }

                        std::vector<int> local_segment_idxs_buf;
                        std::vector<int> local_segment_sizes_buf;

                        int num_local_segments_buf = get_segments(local_buf_idxs, local_segment_idxs_buf, local_segment_sizes_buf);

                        if (print) {
                            std::cout << "DEBUG local buf idxs: " << local_buf_idxs << std::endl;
                            std::cout << "segment idxs: " << local_segment_idxs_buf << std::endl;
                            std::cout << "segment sizes: " << local_segment_sizes_buf << std::endl;

                            for (int k = 0; k < zoid.recv_list_local_num_force_only[t][i]; k++) {
                                int idx = zoid.recv_list_local_force_only[t][i][k];
                            }

                            for (int k = 0; k < zoid.recv_list_local_num_force_pos[t][i]; k++) {
                                int idx = zoid.recv_list_local_force_pos[t][i][k];
                            }
                        }

                        std::vector<int> ghost_buf_idxs;

                        std::set<int> ghost_buf_idxs_set;
                        for (int k = 0; k < zoid.recv_ghost_num_segments[t][i]; k++) {
                            int segment_size = zoid.recv_ghost_sizes[t][i][k];
                            int segment_idx = zoid.recv_ghost_idxs[t][i][k];
                            for (int h = 0; h < segment_size; h++) {
                                int idx = segment_idx + h;
                                int buf_idx = tag_to_idx_in_buf[atom_->tag[idx]];
                                ghost_buf_idxs.push_back(buf_idx);

                                ghost_buf_idxs_set.insert(buf_idx);
                                if (print) {
                                    std::cout << CYAN << "debug recv ghost segment: " << k << " idx: " << idx << " buf idx: " << buf_idx << " tag: " << atom_->tag[idx] << RESET_COLOR << std::endl;
                                }
                            }
                        }

                        assert(ghost_buf_idxs.size() == ghost_buf_idxs_set.size());

                        std::vector<int> ghost_segment_idxs_buf;
                        std::vector<int> ghost_segment_sizes_buf;

                        int num_ghost_segments_buf = get_segments(ghost_buf_idxs, ghost_segment_idxs_buf, ghost_segment_sizes_buf);

                        zoid.recv_process_num_segments[t][i] = num_local_segments_buf + num_ghost_segments_buf;
                        zoid.recv_process_segment_types[t][i] = new int[num_local_segments_buf + num_ghost_segments_buf];
                        zoid.recv_process_segment_idxs[t][i] = new int[num_local_segments_buf + num_ghost_segments_buf];
                        zoid.recv_process_segment_sizes[t][i] = new int[num_local_segments_buf + num_ghost_segments_buf];

                        for (int k = 0; k < num_local_segments_buf; k++) {
                            zoid.recv_process_segment_types[t][i][k] = RECV_DATA_PROCESS_LOCAL;
                            zoid.recv_process_segment_idxs[t][i][k] = local_segment_idxs_buf[k];
                            zoid.recv_process_segment_sizes[t][i][k] = local_segment_sizes_buf[k];
                        }

                        for (int k = 0; k < num_ghost_segments_buf; k++) {
                            int segment_num = k + num_local_segments_buf;
                            zoid.recv_process_segment_types[t][i][segment_num] = RECV_DATA_PROCESS_GHOST;
                            zoid.recv_process_segment_idxs[t][i][segment_num] = ghost_segment_idxs_buf[k];
                            zoid.recv_process_segment_sizes[t][i][segment_num] = ghost_segment_sizes_buf[k];
                        }

                        if (print) {
                            std::cout << "num local segments: " << num_local_segments_buf << " num ghost segments: " << num_ghost_segments_buf << std::endl;
                            for (int k = 0; k < ghost_buf_idxs.size(); k++) {
                                std::cout << "debug ghost buf idx: " << k << " buf idx: " << ghost_buf_idxs[k] << " tag: " << recv_tags[ghost_buf_idxs[k]] << std::endl;
                            }

                            for (int k = 0; k < num_local_segments_buf; k++) {
                                std::cout << "debug local buf segment number: " << k
                                          << " segment idx: " << local_segment_idxs_buf[k]
                                          << " size: " << local_segment_sizes_buf[k] << std::endl;
                            }

                            for (int k = 0; k < num_ghost_segments_buf; k++) {
                                std::cout << "debug ghost buf segment number: " << k + num_local_segments_buf
                                    << " segment idx: " << ghost_segment_idxs_buf[k]
                                    << " size: " << ghost_segment_sizes_buf[k] << std::endl;
                            }
                        }

                        delete[] recv_tags;
                    }
                }
            }
        }

        MPI_Barrier(world);
    }

    MPI_Waitall(send_buf_requests.size(), send_buf_requests.data(), MPI_STATUSES_IGNORE);

    // begin next dt stuff,desperately needs cleanup
    std::map<int, std::map<int, std::vector<int>>> zoid_to_idx_to_send_zoids_next_dt[NUM_TIMESTEPS_IN_PARALLEL + 1];

    for (int t = 0; t < NUM_TIMESTEPS_IN_PARALLEL + 1; t++) {
        for (int dep = 0; dep < NUM_DEPS; dep++) {
            for (int j = 0; j < lmp->queues_next_dt[dep].size(); j++) {
                queue_info& zoid = lmp->queues_next_dt[dep][j];
                int zoid_num = zoid.num;
                if (zoid.num % comm->nprocs == comm->me) {
                    // map segment_idx + size to zoids
                    std::map<std::pair<int, int>, std::vector<int>> segment_mapping_to_zoids;

                    Atom* atom_ = lmp->atom_stencil_md[zoid_num][NUM_TIMESTEPS_IN_PARALLEL - t];
                    auto& send_to = lmp->send_to_neighbors_next_dt[zoid_num];

                    for (int i = 0; i < send_to.size(); i++) {
                        // ghost to local
                        int num_send_pos_segments = zoid.send_pos_num_segments[t][i];
                        for (int k = 0; k < num_send_pos_segments; k++) {
                            int size = zoid.send_pos_sizes[t][i][k];
                            int start_idx = zoid.send_pos_idxs[t][i][k];
                            segment_mapping_to_zoids[{start_idx, size}].push_back(send_to[i]);
                            for (int h = 0; h < size; h++) {
                                int idx = start_idx + h;
                                zoid_to_idx_to_send_zoids_next_dt[t][zoid_num][idx].push_back(send_to[i]);
                            }
                        }

                        int* local_list = zoid.send_local_list[t][i];
                        int local_list_idx = 0;
                        for (int k = 0; k < zoid.send_num_segments[t][i]; k++) {
                            int segment_type = zoid.send_segment_types[t][i][k];
                            if (segment_type == GHOST_SEGMENT_TYPE) {
                                int ghost_size = zoid.send_segment_sizes[t][i][k];
                                int ghost_idx = zoid.send_segment_idxs[t][i][k];
                                segment_mapping_to_zoids[{ghost_idx, ghost_size}].push_back(send_to[i]);
                                for (int h = 0; h < ghost_size; h++) {
                                    int idx = ghost_idx + h;
                                    zoid_to_idx_to_send_zoids_next_dt[t][zoid_num][idx].push_back(send_to[i]);
                                }
                            } else {
                                int local_size = zoid.send_segment_sizes[t][i][k];
                                for (int h = 0; h < local_size; h++) {
                                    int local_idx = local_list[local_list_idx++];
                                    zoid_to_idx_to_send_zoids_next_dt[t][zoid_num][local_idx].push_back(send_to[i]);
                                }
                            }
                        }
                    }

                    bool debug = true;

                    if (debug) {
                        std::map<int, std::set<int>> ghost_idxs_sent_to_procs;
                        std::map<int, std::vector<int>> local_idxs_sent_to_procs;

                        bool print = (zoid_num == 32 && t == 0);

                        for (int i = 0; i < send_to.size(); i++) {
                            bool print2 = print && send_to[i] % comm->nprocs == 6;
                            int neighbor_proc = send_to[i] % comm->nprocs;
                            int* local_list = zoid.send_local_list[t][i];
                            int num_segments = zoid.send_num_segments[t][i];
                            int local_list_idx = 0;

                            for (int k = 0; k < num_segments; k++) {
                                int segment_type = zoid.send_segment_types[t][i][k];
                                if (segment_type == GHOST_SEGMENT_TYPE) {
                                    int ghost_size = zoid.send_segment_sizes[t][i][k];
                                    int ghost_idx = zoid.send_segment_idxs[t][i][k];
                                    for (int h = 0; h < ghost_size; h++) {
                                        int idx = ghost_idx + h;
                                        ghost_idxs_sent_to_procs[neighbor_proc].insert(idx);
                                        if (print && print2) {
                                            std::cout << YELLOW << "zoid: " << zoid_num
                                                      << " send to: " << send_to[i] << " ghost segment number: " << k
                                                      << " idx: " << idx
                                                      << " send to: " << zoid_to_idx_to_send_zoids[t][zoid_num][idx]
                                                      << " pos: " << atom_->x[idx][0] << " " << atom_->x[idx][1] << " " << atom_->x[idx][2] << " tag: " << atom_->tag[idx] << RESET_COLOR << std::endl;
                                        }
                                    }
                                } else {
                                    int local_size = zoid.send_segment_sizes[t][i][k];
                                    for (int h = 0; h < local_size; h++) {
                                        int local_idx = local_list[local_list_idx++];

                                        auto& vec = local_idxs_sent_to_procs[neighbor_proc];

                                        if (print && print2) {
                                            std::cout << CYAN << "zoid: " << zoid_num << " send to: " << send_to[i]
                                                      << " real local segment number: " << k
                                                      << " idx: " << local_idx
                                                      << " send to: " << zoid_to_idx_to_send_zoids[t][zoid_num][local_idx]
                                                      << " pos: " << atom_->x[local_idx][0] << " " << atom_->x[local_idx][1] << " " << atom_->x[local_idx][2] << " tag: " << atom_->tag[local_idx] << RESET_COLOR << std::endl;
                                        }

                                        if (std::find(vec.begin(), vec.end(), local_idx) == vec.end()) {
                                            local_idxs_sent_to_procs[neighbor_proc].push_back(local_idx);
                                        }
                                    }
                                }
                            }

                            int num_send_pos_segments = zoid.send_pos_num_segments[t][i];
                            for (int k = 0; k < num_send_pos_segments; k++) {
                                int size = zoid.send_pos_sizes[t][i][k];
                                int start_idx = zoid.send_pos_idxs[t][i][k];
                                for (int h = 0; h < size; h++) {
                                    int idx = start_idx + h;
                                    ghost_idxs_sent_to_procs[neighbor_proc].insert(idx);
                                    if (print && print2) {
                                        std::cout << YELLOW << "idx to send zids not size 1 wtfel lzoid: " << zoid_num
                                                  << " send to: " << send_to[i] << " send ghost to local segment number: " << k
                                                  << " idx: " << idx
                                                  << " send to: " << zoid_to_idx_to_send_zoids[t][zoid_num][idx]
                                                  << " pos: " << atom_->x[idx][0] << " " << atom_->x[idx][1] << " " << atom_->x[idx][2] << " tag: " << atom_->tag[idx] << RESET_COLOR << std::endl;
                                    }
                                }
                            }
                        }

                        for (int k = 0; k < comm->nprocs; k++) {
                            std::vector<int> vec_idxs;

                            if (ghost_idxs_sent_to_procs.find(k) != ghost_idxs_sent_to_procs.end()) {
                                for (int idx: local_idxs_sent_to_procs[k]) {
                                    vec_idxs.push_back(idx);
                                }
                                for (int idx: ghost_idxs_sent_to_procs[k]) {
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

                                int tmp_num_segments = get_segments(ghost_vec_idxs, tmp_idxs, tmp_lengths, false);
                                if (k != comm->me) {
                                    if (tmp_num_segments >= 15) {
                                        std::cout << "zoid: " << zoid_num << " send to proc: " << k
                                                  << " time: " << t
                                                  << " tmp num segments for all the ghosts sent in one batch: " << tmp_num_segments
                                                  << " idxs: " << tmp_idxs << " lengths: " << tmp_lengths << std::endl;
                                    }
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

    std::map<std::pair<int, int>, int> send_buf_num_tags_next_dt[NUM_TIMESTEPS_IN_PARALLEL + 1];
    std::map<std::pair<int, int>, std::vector<int>> send_buf_tags_next_dt[NUM_TIMESTEPS_IN_PARALLEL + 1];
    std::vector<MPI_Request> send_buf_requests_next_dt;

    std::map<std::pair<int, int>, std::vector<int>> force_offset_idxs_next_dt[NUM_TIMESTEPS_IN_PARALLEL + 1];
    std::map<std::pair<int, int>, int> pos_offsets_next_dt[NUM_TIMESTEPS_IN_PARALLEL + 1];

    std::map<std::pair<int, int>, std::vector<int>> vel_offset_idxs_next_dt[NUM_TIMESTEPS_IN_PARALLEL + 1];

    // send ghost idx to send buf idx per process to the zoids

    for (int t = 0; t < NUM_TIMESTEPS_IN_PARALLEL + 1; t++) {
        lmp->num_recv_force_from_zoid_next_dt[t] = new int[lmp->recv_from_neighbors_procs_next_dt.size()];
        memset(lmp->num_recv_force_from_zoid_next_dt[t], -1, lmp->recv_from_neighbors_procs_next_dt.size());
        lmp->num_recv_pos_from_zoid_next_dt[t] = new int[lmp->recv_from_neighbors_procs_next_dt.size()];
        memset(lmp->num_recv_pos_from_zoid_next_dt[t], -1, lmp->recv_from_neighbors_procs_next_dt.size());
        lmp->num_recv_vel_from_zoid_next_dt[t] = new int[lmp->recv_from_neighbors_procs_next_dt.size()];
        memset(lmp->num_recv_vel_from_zoid_next_dt[t], -1, lmp->recv_from_neighbors_procs_next_dt.size());
    }

    for (int t = 0; t < NUM_TIMESTEPS_IN_PARALLEL + 1; t++) {
        for (int dep = 0; dep < NUM_DEPS; dep++) {
            for (int j = 0; j < lmp->queues_next_dt[dep].size(); j++) {
                queue_info& zoid = lmp->queues_next_dt[dep][j];
                int zoid_num = zoid.num;
                std::vector<int>& send_to = lmp->send_to_neighbors_next_dt[zoid_num];

                if (zoid_num % comm->nprocs == comm->me) {
                    Atom *atom_ = lmp->atom_stencil_md[zoid_num][NUM_TIMESTEPS_IN_PARALLEL - t];

                    for (int proc = 0; proc < comm->nprocs; proc++) {
                        std::vector<int> send_process_local_idxs;
                        std::vector<int> send_process_local_sizes;

                        for (int i = 0; i < send_to.size(); i++) {
                            int send_zoid_num = send_to[i];
                            if (send_zoid_num % comm->nprocs == proc) {
                                int *local_list = zoid.send_local_list[t][i];
                                int num_segments = zoid.send_num_segments[t][i];
                                int local_list_idx = 0;
                                for (int k = 0; k < num_segments; k++) {
                                    int segment_type = zoid.send_segment_types[t][i][k];
                                    if (segment_type == LOCAL_SEGMENT_TYPE) {
                                        int local_size = zoid.send_segment_sizes[t][i][k];

                                        int actual_size = 0;
                                        for (int h = 0; h < local_size; h++) {
                                            int local_idx = local_list[local_list_idx++];
                                            if (zoid_to_idx_to_send_zoids_next_dt[t][zoid_num][local_idx].size() == 1) {
                                                if (std::find(send_process_local_idxs.begin(), send_process_local_idxs.end(), local_idx)
                                                    == send_process_local_idxs.end()) {
                                                    send_process_local_idxs.push_back(local_idx);
                                                    actual_size++;
                                                } else {
                                                    assert(false);
                                                }
                                            }
                                        }

                                        // send_process_local_sizes.push_back(local_size);
                                        if (actual_size > 0) {
                                            send_process_local_sizes.push_back(actual_size);
                                        }
                                    }
                                }
                            }
                        }

                        for (int i = 0; i < send_to.size(); i++) {
                            int send_zoid_num = send_to[i];
                            if (send_zoid_num % comm->nprocs == proc) {
                                int *local_list = zoid.send_local_list[t][i];
                                int num_segments = zoid.send_num_segments[t][i];
                                int local_list_idx = 0;
                                for (int k = 0; k < num_segments; k++) {
                                    int segment_type = zoid.send_segment_types[t][i][k];
                                    if (segment_type == LOCAL_SEGMENT_TYPE) {
                                        int local_size = zoid.send_segment_sizes[t][i][k];

                                        int actual_size = 0;
                                        for (int h = 0; h < local_size; h++) {
                                            int local_idx = local_list[local_list_idx++];

                                            if (std::find(send_process_local_idxs.begin(), send_process_local_idxs.end(), local_idx)
                                                == send_process_local_idxs.end()) {
                                                send_process_local_idxs.push_back(local_idx);
                                                actual_size++;
                                            }
                                        }

                                        // send_process_local_sizes.push_back(local_size);
                                        if (actual_size > 0) {
                                            send_process_local_sizes.push_back(actual_size);
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
                                    int segment_type = zoid.send_segment_types[t][i][k];
                                    if (segment_type == GHOST_SEGMENT_TYPE) {
                                        int ghost_size = zoid.send_segment_sizes[t][i][k];
                                        int ghost_idx = zoid.send_segment_idxs[t][i][k];
                                        for (int h = 0; h < ghost_size; h++) {
                                            int idx = ghost_idx + h;
                                            assert(idx < atom_->nlocal + atom_->nghost);
                                            send_process_ghost_idxs_set.insert(idx);
                                        }
                                    }
                                }

                                int num_send_pos_segments = zoid.send_pos_num_segments[t][i];
                                for (int k = 0; k < num_send_pos_segments; k++) {
                                    int size = zoid.send_pos_sizes[t][i][k];
                                    int start_idx = zoid.send_pos_idxs[t][i][k];
                                    for (int h = 0; h < size; h++) {
                                        int idx = start_idx + h;
                                        assert(idx < atom_->nlocal + atom_->nghost);
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
                        int num_ghost_segments = get_segments(send_process_ghost_idxs_vec, send_process_ghost_segment_idxs, send_process_ghost_segment_sizes);

                        int num_local_segments = send_process_local_sizes.size();

                        // construct send process information
                        zoid.send_process_num_segments[t][proc] = num_local_segments + num_ghost_segments;
                        zoid.send_process_segment_types[t][proc] = new int[num_local_segments + num_ghost_segments];
                        zoid.send_process_segment_idxs[t][proc] = new int[num_local_segments + num_ghost_segments];
                        zoid.send_process_segment_sizes[t][proc] = new int[num_local_segments + num_ghost_segments];

                        zoid.send_process_local_list[t][proc] = new int[send_process_local_idxs.size()];
                        for (int k = 0; k < send_process_local_idxs.size(); k++) {
                            zoid.send_process_local_list[t][proc][k] = send_process_local_idxs[k];
                        }

                        int num_elems_send = 0;
                        ;
                        for (int k = 0; k < num_local_segments; k++) {
                            zoid.send_process_segment_types[t][proc][k] = SEND_DATA_PROCESS_LOCAL;
                            zoid.send_process_segment_idxs[t][proc][k] = -2;
                            zoid.send_process_segment_sizes[t][proc][k] = send_process_local_sizes[k];

                            num_elems_send += send_process_local_sizes[k];
                        }

                        for (int k = 0; k < num_ghost_segments; k++) {
                            int segment_num = k + num_local_segments;
                            zoid.send_process_segment_types[t][proc][segment_num] = SEND_DATA_PROCESS_GHOST;
                            zoid.send_process_segment_idxs[t][proc][segment_num] = send_process_ghost_segment_idxs[k];
                            zoid.send_process_segment_sizes[t][proc][segment_num] = send_process_ghost_segment_sizes[k];

                            if (send_process_ghost_segment_idxs[k] + send_process_ghost_segment_sizes[k] > atom_->nlocal + atom_->nghost) {
                                std::cout << RED << "NEXT DT zoid: " << zoid_num << " send to proc: " << proc << " segment num: " << segment_num << " time: " << t <<
                                          " segment idx: " << send_process_ghost_segment_idxs[k] << " size: " << send_process_ghost_segment_sizes[k] <<
                                          " num local: " << atom_->nlocal << " total: " << atom_->nlocal + atom_->nghost << RESET_COLOR << std::endl;
                                for (int h = 0; h < num_ghost_segments; h++) {
                                    std::cout << "ghost segment: " << h << " idx: " << send_process_ghost_segment_idxs[h] << " size: " << send_process_ghost_segment_sizes[h] << std::endl;
                                }
                            }
                            assert(send_process_ghost_segment_idxs[k] + send_process_ghost_segment_sizes[k] <= atom_->nlocal + atom_->nghost);

                            num_elems_send += send_process_ghost_segment_sizes[k];
                        }

                        zoid.num_elems_send_process[t][proc] = num_elems_send;

                        for (int idx : send_process_local_idxs) {
                            send_buf_tags_next_dt[t][{zoid_num, proc}].push_back(atom_->tag[idx]);
                        }

                        for (int idx : send_process_ghost_idxs_vec) {
                            send_buf_tags_next_dt[t][{zoid_num, proc}].push_back(atom_->tag[idx]);
                        }

                        send_buf_num_tags_next_dt[t][{zoid_num, proc}] = send_process_local_idxs.size() + send_process_ghost_idxs_vec.size();

                        // 0 offset for the first force sent
                        force_offset_idxs_next_dt[t][{zoid_num, proc}].push_back(0);
                        int num_send_force = 0;
                        for (int i = 0; i < send_to.size(); i++) {
                            if (send_to[i] % comm->nprocs == proc) {
                                int num_force_segments = zoid.send_force_num_segments[t][i];
                                int send_force_size = 0;
                                std::vector<int> force_sizes;
                                for (int k = 0; k < num_force_segments; k++) {
                                    int segment_size = zoid.send_force_sizes[t][i][k];
                                    send_force_size += segment_size;
                                    force_sizes.push_back(segment_size);
                                }

                                int prev_size = force_offset_idxs_next_dt[t][{zoid_num, proc}][force_offset_idxs_next_dt[t][{zoid_num, proc}].size() - 1];
                                force_offset_idxs_next_dt[t][{zoid_num, proc}].push_back(send_force_size + prev_size);

                                num_send_force += send_force_size;
                            }
                        }

                        int num_send_vel = 0;
                        vel_offset_idxs_next_dt[t][{zoid_num, proc}].push_back(0);
                        for (int i = 0; i < send_to.size(); i++) {
                            if (send_to[i] % comm->nprocs == proc) {
                                int num_vel_segments = zoid.send_pos_num_segments[t][i];
                                int send_vel_size = 0;
                                for (int k = 0; k < num_vel_segments; k++) {
                                    int segment_size = zoid.send_pos_sizes[t][i][k];
                                    send_vel_size += segment_size;
                                }

                                int prev_size = vel_offset_idxs_next_dt[t][{zoid_num, proc}][vel_offset_idxs_next_dt[t][{zoid_num, proc}].size() - 1];
                                vel_offset_idxs_next_dt[t][{zoid_num, proc}].push_back(send_vel_size + prev_size);

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

                                MPI_Isend(&send_buf_num_tags_next_dt[t][{zoid_num, proc}], 1, MPI_INT, proc, mpi_tag, world, &r1);

                                MPI_Isend(send_buf_tags_next_dt[t][{zoid_num, proc}].data(),
                                          send_buf_num_tags_next_dt[t][{zoid_num, proc}], MPI_INT, proc, mpi_tag, world, &r2);

                                MPI_Isend(&force_offset_idxs_next_dt[t][{zoid_num, proc}][force_offset_vec_idx++], 1, MPI_INT, proc, mpi_tag, world, &r3);

                                MPI_Isend(&pos_offsets_next_dt[t][{zoid_num, proc}], 1, MPI_INT, proc, mpi_tag, world, &r4);

                                MPI_Isend(&force_offset_idxs_next_dt[t][{zoid_num, proc}][force_offset_idxs_next_dt[t][{zoid_num, proc}].size() - 1],
                                          1, MPI_INT, proc, mpi_tag, world, &r5);

                                MPI_Isend(&zoid.num_elems_send_process[t][proc],
                                          1, MPI_INT, proc, mpi_tag, world, &r6);

                                MPI_Isend(&vel_offset_idxs_next_dt[t][{zoid_num, proc}][vel_offset_vec_idx++],
                                          1, MPI_INT, proc, mpi_tag, world, &r7);

                                MPI_Isend(&vel_offset_idxs_next_dt[t][{zoid_num, proc}][vel_offset_idxs_next_dt[t][{zoid_num, proc}].size() - 1],
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
                queue_info &zoid = lmp->queues_next_dt[dep][j];
                int zoid_num = zoid.num;
                std::vector<int> &recv_from = lmp->recv_from_neighbors_next_dt[zoid_num];

                if (zoid_num % comm->nprocs == comm->me) {
                    Atom* atom_ = lmp->atom_stencil_md[zoid_num][NUM_TIMESTEPS_IN_PARALLEL - t];
                    zoid.recv_process_segment_types[t] = new int*[recv_from.size()];
                    zoid.recv_process_segment_idxs[t] = new int*[recv_from.size()];
                    zoid.recv_process_segment_sizes[t] = new int*[recv_from.size()];
                    zoid.recv_process_num_segments[t] = new int[recv_from.size()];

                    // send idxs belonging to that zoid?
                    zoid.recv_process_force_offset[t] = new int[recv_from.size()];
                    zoid.recv_process_pos_offset[t] = new int[recv_from.size()];
                    zoid.recv_process_vel_offset[t] = new int[recv_from.size()];

                    for (int i = 0; i < recv_from.size(); i++) {
                        int recv_zoid_num = recv_from[i];
                        int recv_proc = recv_zoid_num % comm->nprocs;

                        int mpi_tag = (zoid_num << 16 | recv_zoid_num);
                        int num_tags = 0;
                        MPI_Recv(&num_tags, 1, MPI_INT, recv_proc, mpi_tag, world,
                                 MPI_STATUS_IGNORE);

                        int *recv_tags = new int[num_tags];
                        MPI_Recv(recv_tags, num_tags, MPI_INT, recv_proc, mpi_tag, world,
                                 MPI_STATUS_IGNORE);

                        int force_offset_idx = 0;
                        MPI_Recv(&force_offset_idx, 1, MPI_INT, recv_proc, mpi_tag, world,
                                 MPI_STATUS_IGNORE);

                        int num_force_recv = 0;
                        MPI_Recv(&num_force_recv, 1, MPI_INT, recv_proc, mpi_tag, world,
                                 MPI_STATUS_IGNORE);

                        // offset within buffer of forces, which one belongs to this zoid
                        int force_offset = 0;
                        MPI_Recv(&force_offset, 1, MPI_INT, recv_proc, mpi_tag, world,
                                 MPI_STATUS_IGNORE);

                        int num_pos_recv = 0;
                        MPI_Recv(&num_pos_recv, 1, MPI_INT, recv_proc, mpi_tag, world,
                                 MPI_STATUS_IGNORE);

                        int vel_offset = 0;
                        MPI_Recv(&vel_offset, 1, MPI_INT, recv_proc, mpi_tag, world,
                                 MPI_STATUS_IGNORE);

                        int num_vel_recv = 0;
                        MPI_Recv(&num_vel_recv, 1, MPI_INT, recv_proc, mpi_tag, world,
                                 MPI_STATUS_IGNORE);

                        for (int k = 0; k < lmp->recv_from_neighbors_procs_next_dt.size(); k++) {
                            if (lmp->recv_from_neighbors_procs_next_dt[k] == recv_from[i]) {
                                lmp->num_recv_force_from_zoid_next_dt[t][k] = num_force_recv;
                                lmp->num_recv_pos_from_zoid_next_dt[t][k] = num_pos_recv;
                                lmp->num_recv_vel_from_zoid_next_dt[t][k] = num_vel_recv;
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
                        for (int k = 0; k < zoid.recv_list_local_num_force_pos[t][i]; k++) {
                            int idx = zoid.recv_list_local_force_pos[t][i][k];
                            int buf_idx = tag_to_idx_in_buf[atom_->tag[idx]];
                            local_buf_idxs.push_back(buf_idx);
                        }

                        std::vector<int> local_segment_idxs_buf;
                        std::vector<int> local_segment_sizes_buf;

                        int num_local_segments_buf = get_segments(local_buf_idxs, local_segment_idxs_buf, local_segment_sizes_buf);

                        std::vector<int> ghost_buf_idxs;

                        std::set<int> ghost_buf_idxs_set;
                        for (int k = 0; k < zoid.recv_ghost_num_segments[t][i]; k++) {
                            int segment_size = zoid.recv_ghost_sizes[t][i][k];
                            int segment_idx = zoid.recv_ghost_idxs[t][i][k];
                            for (int h = 0; h < segment_size; h++) {
                                int idx = segment_idx + h;
                                int buf_idx = tag_to_idx_in_buf[atom_->tag[idx]];
                                ghost_buf_idxs.push_back(buf_idx);

                                ghost_buf_idxs_set.insert(buf_idx);
                            }
                        }

                        assert(ghost_buf_idxs.size() == ghost_buf_idxs_set.size());

                        std::vector<int> ghost_segment_idxs_buf;
                        std::vector<int> ghost_segment_sizes_buf;

                        int num_ghost_segments_buf = get_segments(ghost_buf_idxs, ghost_segment_idxs_buf, ghost_segment_sizes_buf);

                        zoid.recv_process_num_segments[t][i] = num_local_segments_buf + num_ghost_segments_buf;
                        zoid.recv_process_segment_types[t][i] = new int[num_local_segments_buf + num_ghost_segments_buf];
                        zoid.recv_process_segment_idxs[t][i] = new int[num_local_segments_buf + num_ghost_segments_buf];
                        zoid.recv_process_segment_sizes[t][i] = new int[num_local_segments_buf + num_ghost_segments_buf];

                        for (int k = 0; k < num_local_segments_buf; k++) {
                            zoid.recv_process_segment_types[t][i][k] = RECV_DATA_PROCESS_LOCAL;
                            zoid.recv_process_segment_idxs[t][i][k] = local_segment_idxs_buf[k];
                            zoid.recv_process_segment_sizes[t][i][k] = local_segment_sizes_buf[k];
                        }

                        for (int k = 0; k < num_ghost_segments_buf; k++) {
                            int segment_num = k + num_local_segments_buf;
                            zoid.recv_process_segment_types[t][i][segment_num] = RECV_DATA_PROCESS_GHOST;
                            zoid.recv_process_segment_idxs[t][i][segment_num] = ghost_segment_idxs_buf[k];
                            zoid.recv_process_segment_sizes[t][i][segment_num] = ghost_segment_sizes_buf[k];
                        }

                        delete[] recv_tags;
                    }
                }
            }
        }

        MPI_Barrier(world);
    }

    MPI_Waitall(send_buf_requests_next_dt.size(), send_buf_requests_next_dt.data(), MPI_STATUSES_IGNORE);
    MPI_Barrier(world);

    // compute individual zoid numbers
    int zoid_nrecv_force = 0;
    int zoid_nrecv_vel = 0;
    int zoid_nrecv_pos = 0;
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
                            int num_force = zoid.recv_list_local_num_force_only[t][i];
                            int num_pos = zoid.recv_list_local_num_force_pos[t][i];
                            int num_vel = num_pos;

                            int ghost_pos = 0;
                            for (int k = 0; k < zoid.recv_ghost_num_segments[t][i]; k++) {
                                ghost_pos += zoid.recv_ghost_sizes[t][i][k];
                            }

                            zoid_nrecv_force += num_force;
                            zoid_nrecv_vel += num_vel;
                            zoid_nrecv_pos += num_pos + ghost_pos;

                            std::cout << MAGENTA << "zoid: " << zoid.num << " recv from: " << recv_zoid_num << " time: " << t
                                << " num force: " << num_force << " num pos: " << num_pos << " num vel: " << num_vel << RESET_COLOR << std::endl;
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
                if (DEBUG_SEND_RECV_DATA) {
                    nrecv_force += lmp->num_recv_force_from_zoid[t][k];
                    nrecv_pos += lmp->num_recv_pos_from_zoid[t][k];
                    nrecv_vel += lmp->num_recv_vel_from_zoid[t][k];
                }
            }
        }
    }

    int nrecv = nrecv_force * 3 + nrecv_pos * 3 + nrecv_vel * 3;

    std::cout << "process: " << comm->me << " nrecv force: " << nrecv_force * (3) << " nrecv pos: " << nrecv_pos * (3) << " nrecv vel: " << nrecv_vel * 3 << " nrecv total: " << nrecv
        << " zoid calc force. " << zoid_nrecv_force * 3 << " zoid calc pos: " << zoid_nrecv_pos * 3 << " zoid calc vel: " << zoid_nrecv_vel * 3 << std::endl;

    int res = 0;
    MPI_Allreduce(&nrecv, &res, 1, MPI_INT, MPI_SUM, world);

    if (comm->me == 0) {
        std::cout << "num total recv across all processes: " << res << std::endl;
    }

    MPI_Barrier(world);

    // compute force and then clear everything
    constexpr bool DO_WARMUP_PAIR_CALC = false;

    if (DO_WARMUP_PAIR_CALC) {
        std::cout << "Prepping forces for each timestep" << std::endl;
        for (int t = 0; t < NUM_TIMESTEPS_IN_PARALLEL + 1; t++) {
            for (int dep = 0; dep < NUM_DEPS; dep++) {
                for (int j = 0; j < lmp->queues[dep].size(); j++) {
                    queue_info &zoid = lmp->queues[dep][j];
                    int zoid_num = zoid.num;
                    if (zoid_num % comm->nprocs == comm->me) {
                        AtomKokkos* atomKK_ = (AtomKokkos*) lmp->atom_stencil_md[zoid_num][t];
                        Force* force_ = lmp->force_stencil_md[zoid_num][t];
                        atomKK_->sync_stencil_md(force->pair->execution_space,force->pair->datamask_read, lmp->atom_stencil_md[zoid_num][t]);
                        // Warm up?
                        force_->pair->compute_stencil_md(eflag, vflag, lmp->atom_stencil_md[zoid_num][t],
                                                         zoid.can_eval_center[t], lmp->zoid_num_to_zoid[zoid_num], NULL);
                        force_clear_stencil_md(lmp->atom_stencil_md[zoid_num][t], force_, lmp->neighbor_stencil_md[zoid_num][t]);
                        atomKK_->modified_stencil_md(force_->pair->execution_space, force_->pair->datamask_modify, lmp->atom_stencil_md[zoid_num][t]);
                    }
                }
            }
        }

        std::cout << "done prepping forces" << std::endl;
    }


    // compute force but only for the first timestep
    int num_zoids_recv_from = lmp->recv_from_neighbors_procs.size();
    std::thread receive_request_threads[num_zoids_recv_from];

    // map dependency levels to number of zoids to wait on
    std::map<int, std::vector<int>> dep_to_wait_idxs;

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
                        if (std::find(recv_from.begin(), recv_from.end(), recv_zoid_num) != recv_from.end()
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

    for (auto& [dep, wait_idxs] : dep_to_wait_idxs) {
        std::vector<int> zoids;
        for (int idx : wait_idxs) {
            zoids.push_back(lmp->recv_from_neighbors_procs[idx]);
        }
        std::cout << YELLOW << "process: " << comm->me << " dep: " << dep << " wait zoids: " << zoids << RESET_COLOR << std::endl;
    }

    int num_to_wait_on = 0;
    for (auto& [k, v] : dep_to_wait_idxs) {
        num_to_wait_on += v.size();
    }

    int num_zoids_not_mine = 0;
    for (int zoid : lmp->recv_from_neighbors_procs) {
        if (zoid % comm->nprocs != comm->me) {
            num_zoids_not_mine++;
        }
    }

    assert(num_zoids_not_mine == num_to_wait_on);

    for (int i = 0; i < lmp->recv_from_neighbors_procs.size(); i++) {
        int recv_zoid_num = lmp->recv_from_neighbors_procs[i];
        if (recv_zoid_num % comm->nprocs != comm->me) {
            receive_request_threads[i] =
                std::move(std::thread([&](int recv_zoid_num_) {
                    MPI_Request r;
                    comm->receive_data_process_stencil_md(&r, recv_zoid_num_, true);
                    int wait_status = MPI_Wait(&r, MPI_STATUS_IGNORE);
                    assert(wait_status == MPI_SUCCESS);
                }, recv_zoid_num));
            // comm->receive_data_process_stencil_md(&receive_request_vec[i], recv_zoid_num);
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
                for (int i = 0; i < lmp->send_to_neighbors[zoid_num].size(); i++) {
                    if (lmp->send_to_neighbors[zoid_num][i] % comm->nprocs == proc) {
                        num_procs++;
                        break;
                    }
                }
            }
            assert(num_procs >= 0 && num_procs < comm->nprocs);
            if (num_procs > 0) {
                send_requests[zoid_num] = std::vector<MPI_Request>(num_procs, MPI_REQUEST_NULL);
            }
        }
    }

    for (int dep = 0; dep < NUM_DEPS; dep++) {
        if (dep > 0) {
            for (int idx : dep_to_wait_idxs[dep]) {
                int recv_zoid_num = lmp->recv_from_neighbors_procs[idx];
                receive_request_threads[idx].join();
                comm->unpack_data_process_stencil_md(recv_zoid_num, true);

                std::cout << "process: " << comm->me << " waiting for: " << recv_zoid_num << std::endl;
            }
        }

        for (int j = 0; j < lmp->queues[dep].size(); j++) {
            queue_info& zoid = lmp->queues[dep][j];
            int zoid_num = zoid.num;
            if (zoid_num % comm->nprocs == comm->me) {
                Atom* atom_ = lmp->atom_stencil_md[zoid_num][0];
                AtomKokkos* atomKK_ = (AtomKokkos*) lmp->atom_stencil_md[zoid_num][0];
                Force* force_ = lmp->force_stencil_md[zoid_num][0];
                // todo: eflag and vflag might cause some issues
                // TODO: compute force for each pair in parallel
                atomKK_->sync_stencil_md(force_->pair->execution_space,force_->pair->datamask_read, lmp->atom_stencil_md[zoid_num][0]);
                force_->pair->compute_stencil_md(eflag, vflag, lmp->atom_stencil_md[zoid_num][0],
                                                 zoid.can_eval_center[0], lmp->zoid_num_to_zoid[zoid_num], NULL);
                atomKK_->modified_stencil_md(force_->pair->execution_space, force_->pair->datamask_modify, lmp->atom_stencil_md[zoid_num][0]);

                if (dep < NUM_DEPS - 1) {
                    queue_info& zoid = lmp->zoid_num_to_zoid[zoid_num];
                    auto &atom_arr = lmp->atom_stencil_md[zoid_num];
                    Comm *comm_ = lmp->comm_stencil_md[zoid_num];

                    int vec_idx = 0;
                    for (int proc = 0; proc < comm->nprocs; proc++) {
                        bool sent = comm_->send_data_to_process_stencil_md(atom_arr, lmp->zoid_num_to_zoid[zoid_num], &send_requests[zoid_num][vec_idx], proc, true);
                        if (sent) {
                            send_request_threads.push_back(
                                    std::move(std::thread([&](int idx, int zoid_num_) {
                                                  int wait_status = MPI_Wait(&send_requests[zoid_num_][idx], MPI_STATUSES_IGNORE);
                                                  assert(wait_status == MPI_SUCCESS);
                                              }, vec_idx, zoid_num)
                                    ));
                            vec_idx++;
                        }
                    }
                }
            }
        }
    }

    // auto join_start = std::chrono::high_resolution_clock::now();
    for (auto& t : send_request_threads) {
        t.join();
    }

    // auto join_end = std::chrono::high_resolution_clock::now();
    // auto duration_join = std::chrono::duration_cast<std::chrono::microseconds>(join_end - join_start).count();
    // std::cout << CYAN << "me: " << comm->me << " duration join: " << duration_join << RESET_COLOR << std::endl;

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
                for (int idx = 0; idx < atom_->nlocal; idx++) {
                    int tag = atom_->tag[idx];
                    for (int dim = 0; dim < 3; dim++) {
                        double my_force = atom_->f[idx][dim] + atom_->eval_f_stencil_md[idx][dim];
                        if (fabs(recv_f[tag * 3 + dim] - my_force) > 5e-5) {
                            std::cout << "zoid: " << zoid_num << " idx: " << idx << " tag: " << atom_->tag[idx]
                                      << " dim: " << dim << " what I have: " << my_force << " what lammps has: "
                                      << recv_f[tag * 3 + dim] << " diff: "
                                      << fabs(recv_f[tag * 3 + dim] - my_force) << std::endl;
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
            if (zoid_num % comm->nprocs == comm->me) {
                lmp->modify_stencil_md[zoid_num]->setup_stencil_md(&total_temp_for_me, lmp->atom_stencil_md[zoid_num][0]);
            }
        }
    }

    double total = 0;
    MPI_Allreduce(&total_temp_for_me, &total, 1, MPI_DOUBLE, MPI_SUM, world);
    std::cout << "total temp: " << total << std::endl;


    std::cout << GREEN << "-------- SETUP STENCIL MD PASSED ---------" << RESET_COLOR << std::endl;

    MPI_Barrier(world);

    // change the positions of atoms for debugging purposes
    for (int zoid_num = 0; zoid_num < NUM_ZOIDS; zoid_num++) {
        if (zoid_num % comm->nprocs == comm->me) {
            queue_info& zoid = lmp->zoid_num_to_zoid[zoid_num];
            for (int t = 1; t < NUM_TIMESTEPS_IN_PARALLEL + 1; t++) {
                Atom* atom_ = lmp->atom_stencil_md[zoid_num][t];
                zoid.debug_atom_pos[t] = new double[(atom_->nlocal + atom_->nghost) * 3];
                for (int k = 0; k < atom_->nlocal + atom_->nghost; k++) {
                    for (int dim = 0; dim < 3; dim++) {
                        zoid.debug_atom_pos[t][k * 3 + dim] = atom_->x[k][dim];
                        atom_->x[k][dim] = -1000;
                    }
                }
            }
        }
    }

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

    for (int i = 0; i < atom_->nlocal + atom_->nghost; i++) {
        for (int j = 0; j < 3; j++) {
            atom_->f[i][j] = 0.0;
            atom_->eval_f_stencil_md[i][j] = 0.0;
        }
    }
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