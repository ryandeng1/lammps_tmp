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
  if (atom->sortfreq > 0) atom->sort();
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
        int r;
        MPI_Comm_rank(MPI_COMM_WORLD, &r);
        // std::cout << "me: " << r << " i: " << i << " end: " << end << " nlocal: " << atom_->nlocal << " " << std::endl;
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
        int r;
        MPI_Comm_rank(MPI_COMM_WORLD, &r);
        // std::cout << "me: " << r << " i: " << i << " end: " << end << " nlocal: " << atom_->nlocal << " " << std::endl;
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

void Verlet::regroup_shared_local_toms_stencil_md(Atom* atom_, queue_info& zoid, int timestep) {
    int zoid_num = zoid.num;
    std::vector<int>& send_to = lmp->send_to_neighbors[zoid_num];
    std::vector<int>& recv_from = lmp->recv_from_neighbors[zoid_num];

    std::vector<int> buckets_send[send_to.size()];
    std::vector<int> buckets_recv[recv_from.size()];

    int* current = new int[atom_->nlocal];
    for (int i = 0; i < atom_->nlocal; i++) {
        current[i] = i;
    }

    for (int i = 0; i < atom_->nlocal; i++) {
        double* pos = atom_->x[i];
        for (int j = 0; j < send_to.size(); j++) {
            int other_zoid_num = send_to[j];
            queue_info& other_zoid = lmp->zoid_num_to_zoid[other_zoid_num];
            bool in_zoid = true;
            for (int dim = 0; dim < domain->dimension; dim++) {
                double value = pos[dim];
                double lo = other_zoid.zoid.cuts[dim].lower + timestep * other_zoid.zoid.cuts[dim].slope_lower;
                double hi = other_zoid.zoid.cuts[dim].upper + timestep * other_zoid.zoid.cuts[dim].slope_upper;

                int pbc_ = 0;
                if (zoid.where[dim] == RIGHT && other_zoid.where[dim] == PBC) {
                    pbc_ = -1;
                }
                if (zoid.where[dim] == PBC && other_zoid.where[dim] == RIGHT) {
                    pbc_ = 1;
                }

                double atom_pos_shifted = value + pbc_ * domain->prd[dim];

                in_zoid = in_zoid && ((atom_pos_shifted >= lo && atom_pos_shifted <= hi));
            }

            if (in_zoid) {
                buckets_send[j].push_back(i);
            }
        }

        for (int j = 0; j < recv_from.size(); j++) {
            int other_zoid_num = recv_from[j];
            queue_info& other_zoid = lmp->zoid_num_to_zoid[other_zoid_num];
            bool in_zoid = true;
            for (int dim = 0; dim < domain->dimension; dim++) {
                double value = pos[dim];
                double lo = other_zoid.zoid.cuts[dim].lower + timestep * other_zoid.zoid.cuts[dim].slope_lower;
                double hi = other_zoid.zoid.cuts[dim].upper + timestep * other_zoid.zoid.cuts[dim].slope_upper;

                int pbc_ = 0;
                if (zoid.where[dim] == RIGHT && other_zoid.where[dim] == PBC) {
                    pbc_ = -1;
                }
                if (zoid.where[dim] == PBC && other_zoid.where[dim] == RIGHT) {
                    pbc_ = 1;
                }

                double atom_pos_shifted = value + pbc_ * domain->prd[dim];

                in_zoid = in_zoid && ((atom_pos_shifted >= lo && atom_pos_shifted <= hi));
            }

            if (in_zoid) {
                buckets_recv[j].push_back(i);
            }
        }
    }

    int* permute = new int[atom_->nlocal];
    int permute_idx = 0;

    std::set<int> seen_idxs;
    std::map<int, std::set<int>> idx_to_zoids;

    for (int i = 0; i < send_to.size(); i++) {
        for (int j = 0; j < buckets_send[i].size(); j++) {
            int idx = buckets_send[i][j];
            idx_to_zoids[idx].insert(send_to[i]);
            if (seen_idxs.find(idx) != seen_idxs.end()) {
                std::cout << "zoid: " << zoid_num << " sending to: " << send_to[i] << std::endl;
                std::cout << "idx: " << idx << " out of nlocal: " << atom_->nlocal << " zoids:" << idx_to_zoids[idx] << " i: " << i << " out of: " << send_to.size() << std::endl;
            }
            assert(seen_idxs.find(idx) == seen_idxs.end());
            seen_idxs.insert(idx);
            permute[permute_idx++] = idx;
        }
    }

    for (int i = 0; i < recv_from.size(); i++) {
        for (int j = 0; j < buckets_recv[i].size(); j++) {
            int idx = buckets_recv[i][j];
            assert(seen_idxs.find(idx) == seen_idxs.end());
            seen_idxs.insert(idx);
            permute[permute_idx++] = idx;
        }
    }

    for (int i = 0; i < atom_->nlocal; i++) {
        if (seen_idxs.find(i) == seen_idxs.end()) {
            permute[permute_idx++] = i;
        }
    }

    assert(permute_idx == atom_->nlocal);
}

// TODO: fix for the new deps
void Verlet::group_local_atoms_stencil_md(Atom* atom_, queue_info& zoid, int timestep) {
    int zoid_num = zoid.num;
    auto& recv_from = lmp->recv_from_neighbors[zoid.num];

    int nlocal = atom_->nlocal;

    std::set<int> in_bucket_atoms;
    std::map<int, int> atom_idx_to_recv_zoid_num;
    std::map<int, int> tag_to_recv_zoid_num;

    std::map<int, std::set<int>> atom_idx_to_recv_zoids;

    int sz = lmp->recv_from_neighbors[zoid_num].size();
    std::vector<int> buckets[sz + 1];

    bool all_in_bucket = false;

    // TODO: need to bucket the local ones first, and then bucket the 2-hop ghosts
    for (int i = 0; i < lmp->recv_from_neighbors[zoid_num].size(); i++) {
        int recv_zoid_num = lmp->recv_from_neighbors[zoid_num][i];
        queue_info& recv_from_zoid = lmp->zoid_num_to_zoid[recv_zoid_num];
        // recv_from is neighbor
        bool pbc_flags[3];
        for (int j = 0; j < nlocal; j++) {
            double *pos = atom_->x[j];
            // technically check if borders zoid
            bool in_zoid = true;
            for (int dim = 0; dim < domain->dimension; dim++) {
                bool in_zoid_curr_dim = true;

                double value = pos[dim];
                double lo = recv_from_zoid.zoid.cuts[dim].lower + timestep * recv_from_zoid.zoid.cuts[dim].slope_lower;
                double hi = recv_from_zoid.zoid.cuts[dim].upper + timestep * recv_from_zoid.zoid.cuts[dim].slope_upper;

                // lo -= 2 * ALLEGRO_SLOPE;
                // hi += 2 * ALLEGRO_SLOPE;

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
            }

            if (in_zoid) {
                buckets[i].push_back(j);
                in_bucket_atoms.insert(j);

                if (atom_idx_to_recv_zoid_num.count(j)) {
                    all_in_bucket = true;
                }

                atom_idx_to_recv_zoid_num[j] = recv_zoid_num;

                atom_idx_to_recv_zoids[j].insert(recv_zoid_num);
            }
        }
    }

    for (int i = 0; i < lmp->recv_from_neighbors[zoid_num].size(); i++) {
        int recv_zoid_num = lmp->recv_from_neighbors[zoid_num][i];
        queue_info& recv_from_zoid = lmp->zoid_num_to_zoid[recv_zoid_num];
        // recv_from is neighbor
        bool pbc_flags[3];
        for (int j = 0; j < nlocal; j++) {
            double *pos = atom_->x[j];
            // technically check if borders zoid
            bool in_zoid = true;
            for (int dim = 0; dim < domain->dimension; dim++) {
                bool in_zoid_curr_dim = true;

                double value = pos[dim];
                double lo = recv_from_zoid.zoid.cuts[dim].lower + timestep * recv_from_zoid.zoid.cuts[dim].slope_lower;
                double hi = recv_from_zoid.zoid.cuts[dim].upper + timestep * recv_from_zoid.zoid.cuts[dim].slope_upper;

                lo -= 3 * ALLEGRO_SLOPE;
                hi += 3 * ALLEGRO_SLOPE;

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
            }

            if (zoid.num == 34 && lmp->recv_from_neighbors[zoid_num][i] == 2 && timestep == 0) {
                if (atom_->tag[j] == 385) {
                    std::cout << "idx: " << j << std::endl;
                    std::cout << "pos: " << atom_->x[j][0] << " " << atom_->x[j][1] << " " << atom_->x[j][2] << std::endl;
                    std::cout << "pbs flags? " << pbc_flags[0] << " " << pbc_flags[1] << " " << pbc_flags[2] << std::endl;
                    std::cout << "in zoid? " << in_zoid << std::endl;
                    std::cout << "buckets: " << buckets[i] << std::endl;
                    bool tmp = std::find(buckets[i].begin(), buckets[i].end(), j) == buckets[i].end();
                    std::cout << "tmp: " << tmp << std::endl;
                }
            }

            if (in_zoid) {
                if (std::find(buckets[i].begin(), buckets[i].end(), j) == buckets[i].end()) {
                    buckets[i].push_back(j);
                    in_bucket_atoms.insert(j);
                    if (atom_idx_to_recv_zoid_num.count(j)) {
                        all_in_bucket = true;
                    }

                    atom_idx_to_recv_zoid_num[j] = recv_zoid_num;
                    atom_idx_to_recv_zoids[j].insert(recv_zoid_num);
                }
            }
        }
    }

    // send bucket info
    for (int i = 0; i < lmp->recv_from_neighbors[zoid_num].size(); i++) {
        std::vector<int>& bucket = buckets[i];
        zoid.first_recv_sz_stencil_md2[timestep][i] = bucket.size();
        zoid.first_recv_stencil_md2[timestep][i] = new int[bucket.size()];

        if (zoid.num == 34 && lmp->recv_from_neighbors[zoid_num][i] == 2 && timestep == 0) {
            /*
            std::cout << "IDX: " << i << " OUT OF: " << lmp->recv_from_neighbors[zoid_num].size() << std::endl;
            std::cout << "AYO bucket size? " << bucket.size() << std::endl;
            std::cout << "BUCKET??? " << bucket << std::endl;
            std::cout << "NLOCAL: " << atom_->nlocal << std::endl;

            std::cout << "----------- Zoid 34 ------------" << std::endl;
            for (int dim = 0; dim < 3; dim++) {
                std::cout << "lo: " << zoid.zoid.cuts[dim].lower + timestep * zoid.zoid.cuts[dim].slope_lower
                << " hi: " << zoid.zoid.cuts[dim].upper + timestep * zoid.zoid.cuts[dim].slope_upper << std::endl;
            }
            std::cout << "----------- Zoid 2 ------------" << std::endl;
            queue_info& zoid1 = lmp->zoid_num_to_zoid[2];
            for (int dim = 0; dim < 3; dim++) {
                std::cout << "lo: " << zoid1.zoid.cuts[dim].lower + timestep * zoid1.zoid.cuts[dim].slope_lower
                << " hi: " << zoid1.zoid.cuts[dim].upper + timestep * zoid1.zoid.cuts[dim].slope_upper << std::endl;
            }

            for (int h = 0; h < bucket.size(); h++) {
                std::cout << "BUCKET MAN num: " << h << " bucket idx val: " << bucket[h]
                << " tag? " << atom_->tag[bucket[h]]
                << " pos: " << atom_->x[bucket[h]][0] << " " << atom_->x[bucket[h]][1] << " " << atom_->x[bucket[h]][2] << std::endl;
            }

            for (int h = 0; h < atom_->nlocal; h++) {
                if (atom_->tag[h] == 385) {
                    std::cout << "IDX: " << h << " FOUND MAN RYAN" << std::endl;
                }
            }
            */
        }

        for (int j = 0; j < bucket.size(); j++) {
            zoid.first_recv_stencil_md2[timestep][i][j] = bucket[j];
        }
    }

    for (int i = 0; i < atom_->nlocal; i++) {
        if (in_bucket_atoms.find(i) != in_bucket_atoms.end()) {
            atom_->eval_mask_stencil_md[i] = 1;

        } else {
            atom_->eval_mask_stencil_md[i] = 1;
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

            /*
            int num_timesteps_slack;
            if (t == 0) {
                num_timesteps_slack = NUM_TIMESTEPS_IN_PARALLEL;
            } else {
                num_timesteps_slack = NUM_TIMESTEPS_IN_PARALLEL - t;
            }

            bool can_eval_pos = true;

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
                int diff_int = (int) std::ceil(diff / ALLEGRO_SLOPE);

                if (shrinking_dim) {
                    can_eval_center = can_eval_center && (diff_int <= num_timesteps_slack);
                    can_eval_pos = can_eval_pos && (diff_int <= num_timesteps_slack - 1);
                    diffs[dim] = diff_int;
                } else {
                    if (pos[dim] >= lo && pos[dim] <= hi) {
                        // can_eval_center = can_evaltrue;
                        // can_eval_pos = true;
                        double diff_middle = std::min(fabs(pos[dim] - lo), fabs(pos[dim] - hi));
                        int diff_int_middle = (int) std::ceil(diff_middle / ALLEGRO_SLOPE);
                        // how many hops away from border in middle do I have to be to be "safe"
                        int num_hops_check_can_eval_middle = std::max(0, (NUM_TIMESTEPS_IN_PARALLEL - t - 1));
                        can_eval_center = can_eval_center && (diff_int_middle > num_hops_check_can_eval_middle);
                        can_eval_pos = can_eval_pos && can_eval_center;
                    } else {
                        if (t == 0) {
                            // accept this for right zoids but not left zoids, since they actually sent us the data, so can't eval
                            // can remove if we make the middle/pbc zoids wider
                            // complete hack
                            // can_eval_center = can_eval_center && (diff_int <= 1);
                        } else {
                            can_eval_center = false;
                        }
                        can_eval_center = false;
                        can_eval_pos = false;
                    }
                }
            }
            zoid.can_eval_pos[t][i] = can_eval_pos;
            */

        }
    }
}

void setup_can_eval_center_mapping_stencil_md_next_dt(std::array<Atom*, NUM_TIMESTEPS_IN_PARALLEL + 1>& atom_arr,
                                              queue_info& zoid) {
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

            /*
            int num_timesteps_slack;
            if (t == 0) {
                num_timesteps_slack = NUM_TIMESTEPS_IN_PARALLEL;
            } else {
                num_timesteps_slack = NUM_TIMESTEPS_IN_PARALLEL - t;
            }

            bool can_eval_pos = true;

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
                int diff_int = (int) std::ceil(diff / ALLEGRO_SLOPE);

                if (shrinking_dim) {
                    can_eval_center = can_eval_center && (diff_int <= num_timesteps_slack);
                    can_eval_pos = can_eval_pos && (diff_int <= num_timesteps_slack - 1);
                    diffs[dim] = diff_int;
                } else {
                    if (pos[dim] >= lo && pos[dim] <= hi) {
                        // can_eval_center = can_evaltrue;
                        // can_eval_pos = true;
                        double diff_middle = std::min(fabs(pos[dim] - lo), fabs(pos[dim] - hi));
                        int diff_int_middle = (int) std::ceil(diff_middle / ALLEGRO_SLOPE);
                        // how many hops away from border in middle do I have to be to be "safe"
                        int num_hops_check_can_eval_middle = std::max(0, (NUM_TIMESTEPS_IN_PARALLEL - t - 1));
                        can_eval_center = can_eval_center && (diff_int_middle > num_hops_check_can_eval_middle);
                        can_eval_pos = can_eval_pos && can_eval_center;
                    } else {
                        if (t == 0) {
                            // accept this for right zoids but not left zoids, since they actually sent us the data, so can't eval
                            // can remove if we make the middle/pbc zoids wider
                            // complete hack
                            // can_eval_center = can_eval_center && (diff_int <= 1);
                        } else {
                            can_eval_center = false;
                        }
                        can_eval_center = false;
                        can_eval_pos = false;
                    }
                }
            }
            zoid.can_eval_pos[t][i] = can_eval_pos;
            */

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

    int** mapping = new int*[NUM_TIMESTEPS_IN_PARALLEL + 1];
    for (int i = 0; i < NUM_TIMESTEPS_IN_PARALLEL + 1; i++) {
        mapping[i] = new int[max_atoms];
    }

    for (int i = 0; i < NUM_TIMESTEPS_IN_PARALLEL + 1; i++) {
        for (int j = 0; j < max_atoms; j++) {
            mapping[i][j] = -1;
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

            mapping[t][i] = next_idx;
        }
    }

    zoid.atom_idx_mapping = mapping;
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

    int** mapping = new int*[NUM_TIMESTEPS_IN_PARALLEL + 1];
    for (int i = 0; i < NUM_TIMESTEPS_IN_PARALLEL + 1; i++) {
        mapping[i] = new int[max_atoms];
    }

    for (int i = 0; i < NUM_TIMESTEPS_IN_PARALLEL + 1; i++) {
        for (int j = 0; j < max_atoms; j++) {
            mapping[i][j] = -1;
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

            mapping[t][i] = next_idx;
        }
    }

    zoid.atom_idx_mapping = mapping;
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
        std::cout << "zoid dep: " << zoid_dep << " for zoid: " << zoid.num << std::endl;
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

    // comm exchange
    // MODIFY, since we have it so that there are overlapping zoids, middle zoids share some local atoms with left/right, send the initial atoms to
    // dep0 zoids. Then have them send to their neighbors
    for (int i = 0; i < NUM_TIMESTEPS_IN_PARALLEL + 1; i++) {
        comm->exchange_stencil_md_initial_send();
        for (int dep = 0; dep < NUM_DEPS; dep++) {
            for (int j = 0; j < lmp->queues[dep].size(); j++) {
                queue_info& zoid = lmp->queues[dep][j];
                int zoid_num = zoid.num;
                int idx_ = lmp->zoid_num_to_idx[zoid_num];
                // receive only if the zoid belongs to me
                if (zoid_num % comm->nprocs == comm->me) {
                    // std::cout << "exchange stencil md initial receive Zoid num: " << zoid_num << " for time: " << i << std::endl;
                    lmp->comm_stencil_md[idx_]->exchange_stencil_md_initial_receive(lmp->atom_stencil_md[idx_][i],
                                                                                    lmp->domain_stencil_md[idx_][i], zoid);

                    if (i == 0) {
                        std::cout << "zoid num: " << zoid_num << " has nlocal: " << lmp->atom_stencil_md[idx_][i]->nlocal << std::endl;

                        for (int dim = 0; dim < 3; dim++) {
                            std::cout << "lo: " << zoid.zoid.cuts[dim].lower << " " << zoid.zoid.cuts[dim].upper << std::endl;
                        }
                    }
                }
            }
        }

        MPI_Barrier(world);
    }

    MPI_Barrier(MPI_COMM_WORLD);

    for (int t = 0; t < NUM_TIMESTEPS_IN_PARALLEL + 1; t++) {
        for (int dep = 0; dep < NUM_DEPS; dep++) {
            for (int j = 0; j < lmp->queues[dep].size(); j++) {
                queue_info& zoid = lmp->queues[dep][j];
                int zoid_num = zoid.num;
                int idx_ = lmp->zoid_num_to_idx[zoid_num];
                // receive only if the zoid belongs to me
                if (zoid_num % comm->nprocs == comm->me) {
                    Atom* atom_ = lmp->atom_stencil_md[idx_][t];
                    // atom_->sort_stencil_md();

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


    // order shared local atoms in the same way

    if (comm->me == 0) {
        for (int i = 0; i < NUM_ZOIDS; i++) {
            std::cout << "send to for zoid: " << i << " is: " << lmp->send_to_neighbors[i] << std::endl;
            std::cout << "recv from for zoid: " << i << " is: " << lmp->recv_from_neighbors[i] << std::endl;
        }
    }

    for (int i = 0; i < NUM_TIMESTEPS_IN_PARALLEL + 1; i++) {
        int num_atoms_owned = 0;
        for (int dep = 0; dep < NUM_DEPS; dep++) {
            for (int j = 0; j < lmp->queues[dep].size(); j++) {
                queue_info& zoid = lmp->queues[dep][j];
                int zoid_num = zoid.num;
                // receive only if the zoid belongs to me
                if (zoid_num % comm->nprocs == comm->me) {
                    int idx_ = lmp->zoid_num_to_idx[zoid_num];
                    Atom* atom_ = lmp->atom_stencil_md[idx_][i];
                    num_atoms_owned += atom_->nlocal;

                    std::set<int> tags;
                    for (int k = 0; k < atom_->nlocal; k++) {
                        tags.insert(atom_->tag[k]);
                    }
                    if (tags.size() != atom_->nlocal) {
                        std::cout << "zoid_num: " << zoid_num << " before borders tags size: " << tags.size() << " atom nlocal: " << atom_->nlocal << " time: " << i << std::endl;
                    }
                    assert(tags.size() == atom_->nlocal);
                }
            }
        }

        for (int dep = 0; dep < NUM_DEPS; dep++) {
            for (int j = 0; j < lmp->queues[dep].size(); j++) {
                int zoid_num = lmp->queues[dep][j].num;
                if (zoid_num % comm->nprocs == comm->me) {
                    int idx_ = lmp->zoid_num_to_idx[zoid_num];
                    lmp->domain_stencil_md[idx_][i]->pbc_stencil_md(lmp->atom_stencil_md[idx_][i]);
                }
            }
        }

        for (int dep = 0; dep < NUM_DEPS; dep++) {
            for (int j = 0; j < lmp->queues[dep].size(); j++) {
                queue_info& zoid = lmp->queues[dep][j];
                int zoid_num = zoid.num;
                int idx_ = lmp->zoid_num_to_idx[zoid_num];
                if (zoid_num % comm->nprocs == comm->me) {
                    lmp->neighbor_stencil_md[idx_][i]->setup_bins_stencil_md(lmp->atom_stencil_md[idx_][i],
                                                                             lmp->domain_stencil_md[idx_][i],
                                                                             lmp->comm_stencil_md[idx_]);
                }
            }
        }

        for (int dep = 0; dep < NUM_DEPS; dep++) {
            for (int j = 0; j < lmp->queues[dep].size(); j++) {
                // TODO: fix, somehow link zoid_num_to_zoid and queues
                queue_info& tmp = lmp->queues[dep][j];
                int zoid_num = tmp.num;
                queue_info& zoid = lmp->zoid_num_to_zoid[zoid_num];

                if (zoid_num % comm->nprocs == comm->me) {
                    // todo: fix this 6 must mean some kind of weird error here
                    int num_recv_from = lmp->recv_from_neighbors[zoid_num].size();
                    zoid.first_recv_stencil_md[i] = new int[num_recv_from];
                    zoid.init_first_recv = true;

                    zoid.first_recv_sz_stencil_md2[i] = new int[num_recv_from];
                    zoid.first_recv_stencil_md2[i] = new int*[num_recv_from];
                    group_local_atoms_stencil_md(lmp->atom_stencil_md[zoid_num][i], zoid, i);
                }
            }
        }

        // borders
        for (int dep = 0; dep < NUM_DEPS; dep++) {
            for (int j = 0; j < lmp->queues[dep].size(); j++) {
                // TODO: fix, somehow link zoid_num_to_zoid and queues
                queue_info& tmp = lmp->queues[dep][j];
                int zoid_num = tmp.num;
                queue_info& zoid = lmp->zoid_num_to_zoid[zoid_num];
                int idx_ = lmp->zoid_num_to_idx[zoid_num];
                if (zoid_num % comm->nprocs == comm->me) {
                    // todo: fix this 6 must mean some kind of weird error here
                    lmp->comm_stencil_md[idx_]->borders_stencil_md_initial_send(lmp->atom_stencil_md[idx_][i], lmp->domain_stencil_md[idx_][i],
                                                                                zoid, i);
                }
            }
        }

        MPI_Barrier(world);

        std::cout << "me: " << comm->me << " finished sending" << std::endl;

        for (int dep = 0; dep < NUM_DEPS; dep++) {
            for (int j = 0; j < lmp->queues[dep].size(); j++) {
                // TODO: fix, somehow link zoid_num_to_zoid and queues
                queue_info& tmp = lmp->queues[dep][j];
                int zoid_num = tmp.num;
                queue_info& zoid = lmp->zoid_num_to_zoid[zoid_num];
                int idx_ = lmp->zoid_num_to_idx[zoid_num];
                if (zoid_num % comm->nprocs == comm->me) {
                    lmp->comm_stencil_md[idx_]->borders_stencil_md_initial_receive(lmp->atom_stencil_md[idx_][i], lmp->domain_stencil_md[idx_][i],
                                                                                zoid);
                    Atom* atom_ = lmp->atom_stencil_md[idx_][i];
                    for (int k = atom_->nlocal; k < atom_->nlocal + atom_->nghost; k++) {
                        atom_->eval_mask_stencil_md[k] = 1;
                    }

                    for (int k = 0; k < atom_->nlocal + atom_->nghost; k++) {
                        atom_->actually_eval_mask_stencil_md[k] = 0;
                    }
                }
            }
        }

        MPI_Barrier(world);

        // create tag to idx
        for (int dep = 0; dep < NUM_DEPS; dep++) {
            for (int j = 0; j < lmp->queues[dep].size(); j++) {
                queue_info& zoid = lmp->queues[dep][j];
                int zoid_num = zoid.num;
                int idx_ = lmp->zoid_num_to_idx[zoid_num];
                if (zoid_num % comm->nprocs == comm->me) {
                    Atom* atom_ = lmp->atom_stencil_md[zoid_num][i];
                    for (int idx = 0; idx < atom_->nlocal + atom_->nghost; idx++) {
                        atom_->tag_to_idx[atom_->tag[idx]] = idx;
                    }
                }
            }
        }

        for (int dep = 0; dep < NUM_DEPS; dep++) {
            for (int j = 0; j < lmp->queues[dep].size(); j++) {
                queue_info& zoid = lmp->queues[dep][j];
                int zoid_num = zoid.num;
                int idx_ = lmp->zoid_num_to_idx[zoid_num];
                if (zoid_num % comm->nprocs == comm->me) {
                    lmp->neighbor_stencil_md[idx_][i]->build_stencil_md(1, lmp->atom_stencil_md[idx_][i], lmp->domain_stencil_md[idx_][i], lmp->comm_stencil_md[idx_]);
                    lmp->neighbor_stencil_md[idx_][i]->ncalls = 0;

                    AtomKokkos* atomKK_ = (AtomKokkos*) lmp->atom_stencil_md[idx_][i];
                    Force* force_ = lmp->force_stencil_md[idx_][i];
                    force_->setup();
                    atomKK_->sync_stencil_md(force->pair->execution_space,force->pair->datamask_read, lmp->atom_stencil_md[idx_][i]);
                    force_clear_stencil_md(lmp->atom_stencil_md[idx_][i], force_, lmp->neighbor_stencil_md[idx_][i]);
                    atomKK_->modified_stencil_md(force_->pair->execution_space, force_->pair->datamask_modify, lmp->atom_stencil_md[idx_][i]);
                }
            }
        }
    }

    MPI_Barrier(world);

    for (int dep = 0; dep < NUM_DEPS; dep++) {
        for (int j = 0; j < lmp->queues[dep].size(); j++) {
            queue_info& zoid = lmp->queues[dep][j];
            int zoid_num = zoid.num;
            int idx_ = lmp->zoid_num_to_idx[zoid_num];
            if (zoid_num % comm->nprocs == comm->me) {
                setup_atom_pos_mapping_stencil_md(lmp->atom_stencil_md[idx_], zoid);
                setup_can_eval_center_mapping_stencil_md(lmp->atom_stencil_md[idx_], zoid);
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

    // construct first send_list, sending ghost->local
    for (int dep = 0; dep < NUM_DEPS; dep++) {
        for (int j = 0; j < lmp->queues[dep].size(); j++) {
            queue_info& zoid = lmp->queues[dep][j];
            int zoid_num = zoid.num;
            if (zoid_num % comm->nprocs == comm->me) {
                lmp->comm_stencil_md[zoid_num]->construct_send_list_stencil_md(lmp->atom_stencil_md[zoid_num], zoid);
            }
        }
    }

    for (int dep = 0; dep < NUM_DEPS; dep++) {
        for (int j = 0; j < lmp->queues_next_dt[dep].size(); j++) {
            queue_info& zoid = lmp->queues_next_dt[dep][j];
            int zoid_num = zoid.num;
            if (zoid_num % comm->nprocs == comm->me) {
                lmp->comm_stencil_md[zoid_num]->construct_send_list_stencil_md_next_dt(lmp->atom_stencil_md[zoid_num], zoid);
            }
        }
    }

    std::vector<int*> send_bufs;
    // compute force but only for the first timestep
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
                std::cout << "Me: " << comm->me << " setup force compute for zoid: " << zoid_num << std::endl;

                if (dep > 0) {
                    auto &atom_arr = lmp->atom_stencil_md[zoid_num];
                    Comm *comm_ = lmp->comm_stencil_md[zoid_num];
                    comm_->receive_data_stencil_md(atom_arr, lmp->zoid_num_to_zoid[zoid_num]);
                    comm_->receive_exclude_eval_tags(atom_arr, lmp->zoid_num_to_zoid[zoid_num]);
                }

                atomKK_->sync_stencil_md(force_->pair->execution_space,force_->pair->datamask_read, lmp->atom_stencil_md[idx_][0]);
                force_->pair->compute_stencil_md(eflag, vflag, lmp->atom_stencil_md[idx_][0], lmp->atom_stencil_md[idx_][1],
                                                 zoid.can_eval_center[0], lmp->zoid_num_to_zoid[zoid_num]);
                atomKK_->modified_stencil_md(force_->pair->execution_space, force_->pair->datamask_modify, lmp->atom_stencil_md[idx_][0]);

                if (dep < NUM_DEPS - 1) {
                    queue_info& zoid = lmp->zoid_num_to_zoid[zoid_num];
                    auto &atom_arr = lmp->atom_stencil_md[zoid_num];
                    Comm *comm_ = lmp->comm_stencil_md[zoid_num];
                    comm_->send_data_stencil_md(atom_arr, zoid);
                    int* buf = comm_->send_exclude_eval_tags(atom_arr, lmp->zoid_num_to_zoid[zoid_num]);
                    send_bufs.push_back(buf);
                }
            }
        }
    }

    MPI_Barrier(world);

    for (int i = 0; i < send_bufs.size(); i++) {
        delete[] send_bufs[i];
    }

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
                std::cout << "tag: " << tag << " idx: " << i << " dim: " << dim << " what I have: " << atom->f[i][dim] << " what I got: " << recv_f[tag * 3 + dim] << " diff: " << fabs(recv_f[tag * 3 + dim] - atom->f[i][dim]) << std::endl;
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
                    x_ += atom_->f[k][0];
                    y_ += atom_->f[k][1];
                    z_ += atom_->f[k][2];
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
