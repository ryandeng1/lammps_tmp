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
  assert(false);
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

                lo -= 2 * ALLEGRO_SLOPE;
                hi += 2 * ALLEGRO_SLOPE;

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

// create an arr, permute_esque
void setup_atom_pos_mapping_stencil_md(std::array<Atom*, NUM_TIMESTEPS_IN_PARALLEL + 1> atom_arr, queue_info& zoid) {
    int max_atoms = -1;
    for (int i = 0; i < atom_arr.size(); i++) {
        Atom* atom_ = atom_arr[i];
        int num_atoms = atom_->nlocal + atom_->nghost;
        if (num_atoms > max_atoms) {
            max_atoms = num_atoms;
        }
    }

    int** mapping = new int*[NUM_TIMESTEPS_IN_PARALLEL + 1];
    for (int i = 0; i < NUM_TIMESTEPS_IN_PARALLEL; i++) {
        mapping[i] = new int[max_atoms];
    }
    for (int i = 0; i < NUM_TIMESTEPS_IN_PARALLEL; i++) {
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

void Verlet::setup_stencil_md() {
    // USE_STENCIL_MD = true;
    std::cout << "RYAN setup stencil md start 2" << std::endl;
    std::cout << "domain boxlo: " << domain->boxlo[0] << " " << domain->boxlo[1] << " " << domain->boxlo[2] << std::endl;
    std::cout << "domain boxhi: " << domain->boxhi[0] << " " << domain->boxhi[1] << " " << domain->boxhi[2] << std::endl;

    get_zoids(ALLEGRO_SLOPE, domain->boxlo, domain->boxhi, lmp->queues);

    // modify zoids
    // used to be 8, 24, 24, 8
    // 8, 8, 8, 8, 24, 8
    // lmp->queues[5] = lmp->queues[3];
    // lmp->queues[4] = lmp->queues[2];

    /*
    lmp->queues[7] = lmp->queues[3];

    std::deque<queue_info> split_dep_1[3];

    int dep_to_split = 1;
    for (int i = 0; i < lmp->queues[dep_to_split].size(); i++) {
        queue_info& zoid = lmp->queues[dep_to_split][i];
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
        queue_info& zoid = lmp->queues[dep_to_split_2][i];
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

    */


//    lmp->queues[1] = split_dep_1[0];
//    lmp->queues[2] = split_dep_1[1];
//    lmp->queues[3] = split_dep_1[2];

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
    // for (int dep = 0; dep < 3 + 1; dep++) {
    for (int dep = 0; dep < NUM_DEPS; dep++) {
        for (int j = 0; j < lmp->queues[dep].size(); j++) {
            int zoid_num = lmp->queues[dep][j].num;
            lmp->zoid_num_to_idx[zoid_num] = zoid_idx++;
            assert(zoid_num == zoid_idx - 1);
            lmp->zoid_num_to_zoid[zoid_num] = lmp->queues[dep][j];
        }
    }

    // lmp->send_to = new std::vector<int>[NUM_ZOIDS];
    // lmp->recv_from = new std::vector<int>[NUM_ZOIDS];
    lmp->send_to_next_dt = new std::vector<int>[NUM_ZOIDS];
    lmp->recv_from_next_dt = new std::vector<int>[NUM_ZOIDS];

    lmp->recv_from_neighbors = new std::vector<int>[NUM_ZOIDS];
    lmp->send_to_neighbors = new std::vector<int>[NUM_ZOIDS];

    lmp->send_to_shared_ghost = new std::vector<int>[NUM_ZOIDS];
    lmp->recv_from_shared_ghost = new std::vector<int>[NUM_ZOIDS];

    for (int i = 0; i < NUM_ZOIDS; i++) {
        auto& zoid = lmp->zoid_num_to_zoid[i];
        int zoid_dep = get_zoid_dep(zoid.num);
        for (int j = 0; j < NUM_ZOIDS; j++) {
            int zoid_dep_neighbor = get_zoid_dep(j);
            if (zoid_dep_neighbor > zoid_dep && is_close(zoid.where, lmp->zoid_num_to_zoid[j].where)) {
                // TODO: test this extra condition
                if (zoid_dep_neighbor == zoid_dep + 1) {
                    lmp->send_to_neighbors[i].push_back(j);
                }
            }

            if (zoid_dep_neighbor < zoid_dep && is_close(zoid.where, lmp->zoid_num_to_zoid[j].where)) {
                if (zoid_dep_neighbor == zoid_dep - 1) {
                    lmp->recv_from_neighbors[i].push_back(j);
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
            // assert(test[i] == lmp->recv_from_neighbors[i]);
        }
    }


    // TODO: lmp->send_to and lmp->recv_from are deprecated
    /*
    for (int dep = 0; dep < 3 + 1; dep++) {
        for (int j = 0; j < lmp->queues[dep].size(); j++) {
            auto &subzoid = lmp->queues[dep][j];
            if (dep > 0) {
                int prev_dep_size = lmp->queues[dep - 1].size();
                for (int i = 0; i < prev_dep_size; i++) {
                    auto &q_info = lmp->queues[dep - 1][i];
                    if (is_dep(q_info.where, subzoid.where)) {
                        lmp->recv_from[subzoid.num].push_back(q_info.num);
                    }
                }
            }

            // for current dt send to zoids in the next dependency level
            if (dep < 3) {
                int next_dep_size = lmp->queues[dep + 1].size();
                for (int i = 0; i < next_dep_size; i++) {
                    auto &q_info = lmp->queues[dep + 1][i];
                    if (is_dep(subzoid.where, q_info.where)) {
                        lmp->send_to[subzoid.num].push_back(q_info.num);
                    }
                }
            }
        }
    }
    */

    // TODO: fill in next_dt later
    /*
    for (int dep = 3 + 1; dep < 2 * (3 + 1); dep++) {
        for (int j = 0; j < lmp->queues[dep].size(); j++) {
            auto &subzoid = lmp->queues[dep][j];
            if (dep > 3 + 1) {
                int prev_dep_size = lmp->queues[dep - 1].size();
                for (int i = 0; i < prev_dep_size; i++) {
                    auto &q_info = lmp->queues[dep - 1][i];
                    if (is_dep_inverted(q_info.where, subzoid.where)) {
                        lmp->recv_from_next_dt[subzoid.num].push_back(q_info.num);
                    }
                }
            }

            // for current dt send to zoids in the next dependency level
            if (dep < 2 * (3 + 1) - 1) {
                int next_dep_size = lmp->queues[dep + 1].size();
                for (int i = 0; i < next_dep_size; i++) {
                    auto &q_info = lmp->queues[dep + 1][i];
                    if (is_dep_inverted(subzoid.where, q_info.where)) {
                        lmp->send_to_next_dt[subzoid.num].push_back(q_info.num);
                    }
                }
            }
        }
    }
    */

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
                    }
                }
            }
        }
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
                    atom_->sort_stencil_md();

//                    if (t == 1 && (zoid_num == 2 || zoid_num == 34)) {
//                        for (int h = 0; h < atom_->nlocal; h++) {
//                            std::cout << "Zoid num: " << zoid_num << " h: " << h << " tag: " << atom_->tag[h]
//                            << " pos: " << atom_->x[h][0] << " " << atom_->x[h][1] << " " << atom_->x[h][2] << std::endl;
//                        }
//                    }

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
                    for (int t = 0; t < NUM_TIMESTEPS_IN_PARALLEL + 1; t++) {
                        Atom* atom_ = lmp->atom_stencil_md[zoid_num][t];
                        for (int idx = 0; idx < atom_->nlocal + atom_->nghost; idx++) {
                            atom_->tag_to_idx[atom_->tag[idx]] = idx;
                        }
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

    // construct first send_list, sending ghost->local
    for (int dep = 0; dep < NUM_DEPS; dep++) {
        for (int j = 0; j < lmp->queues[dep].size(); j++) {
            queue_info& tmp = lmp->queues[dep][j];
            int zoid_num = tmp.num;
            queue_info& zoid = lmp->zoid_num_to_zoid[zoid_num];
            if (zoid_num % comm->nprocs == comm->me) {
                lmp->comm_stencil_md[zoid_num]->construct_send_list_stencil_md(lmp->atom_stencil_md[zoid_num], zoid);
            }
        }
    }

    for (int dep = 0; dep < NUM_DEPS; dep++) {
        for (int j = 0; j < lmp->queues[dep].size(); j++) {
            queue_info& zoid = lmp->queues[dep][j];
            int zoid_num = zoid.num;
            int idx_ = lmp->zoid_num_to_idx[zoid_num];
            if (zoid_num % comm->nprocs == comm->me) {
                setup_atom_pos_mapping_stencil_md(lmp->atom_stencil_md[idx_], zoid);

                Atom* atom_ = lmp->atom_stencil_md[zoid_num][0];
                Atom* next = lmp->atom_stencil_md[zoid_num][1];
                for (int i = 0; i < atom_->nlocal; i++) {
                    if (atom_->tag[i] == 10152) {
                        std::cout << "Zoid num: " << zoid_num << " has atom tag: " << atom_->tag[i] << " with pos: " << atom_->x[i][0] << " " << atom_->x[i][1] << " " << atom_->x[i][2] << std::endl;
                        std::cout << "-----------Zoid: " << zoid_num << " ---------------------" << std::endl;
                        for (int dim = 0; dim < 3; dim++) {
                            std::cout << "lo: " << zoid.zoid.cuts[dim].lower << " hi: " << zoid.zoid.cuts[dim].upper << std::endl;
                        }
                        bool s = (zoid.atom_idx_mapping[0][i] >= next->nlocal);
                        std::cout << "zoid_num: " << zoid_num << " atom pos mapping: " << s << std::endl;

                    }
                }
            }
            MPI_Barrier(world);
        }
    }

    // assert(false);


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
                force_->pair->compute_stencil_md(eflag, vflag, lmp->atom_stencil_md[idx_][0], lmp->atom_stencil_md[idx_][1], zoid.atom_idx_mapping[0], lmp->zoid_num_to_zoid[zoid_num]);
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

    /*
    std::vector<int*> send_bufs3;
    // get the fully shared eval tags, what was actually eval'ed
    for (int dep = 0; dep < NUM_DEPS; dep++) {
        for (int j = 0; j < lmp->queues[dep].size(); j++) {
            queue_info &zoid = lmp->queues[dep][j];
            int zoid_num = zoid.num;
            int idx_ = lmp->zoid_num_to_idx[zoid_num];
            assert(zoid_num == idx_);
            if (zoid_num % comm->nprocs == comm->me) {
                Atom* atom_ = lmp->atom_stencil_md[zoid_num][0];

                int* send_buf = new int[atom_->fully_eval_ghost_tags.size()];
                int send_buf_idx = 0;
                for (int ghost_tag : atom_->fully_eval_ghost_tags) {
                    send_buf[send_buf_idx++] = ghost_tag;
                }
                for (int k = 0; k < lmp->queues[dep].size(); k++) {
                    if (lmp->queues[dep][k].num != zoid_num) {
                        int sz = atom_->fully_eval_ghost_tags.size();
                        int send_zoid_num = lmp->queues[dep][k].num;
                        MPI_Request r1;
                        MPI_Request r2;
                        MPI_Isend(&sz, 1, MPI_INT, send_zoid_num % comm->nprocs, send_zoid_num, world, &r1);
                        if (sz > 0) {
                            MPI_Isend(send_buf, sz, MPI_INT, send_zoid_num % comm->nprocs, send_zoid_num, world, &r2);
                        }
                    }
                }
                send_bufs3.push_back(send_buf);
            }
        }
    }

    MPI_Barrier(world);

    std::map<int, std::set<int>> ghost_to_zoids_eval_final;
    for (int dep = 0; dep < NUM_DEPS; dep++) {
        for (int j = 0; j < lmp->queues[dep].size(); j++) {
            queue_info& tmp = lmp->queues[dep][j];
            int zoid_num = tmp.num;
            queue_info& zoid = lmp->zoid_num_to_zoid[zoid_num];
            int idx_ = lmp->zoid_num_to_idx[zoid_num];
            if (zoid_num % comm->nprocs == comm->me) {
                Atom* atom_ = lmp->atom_stencil_md[zoid_num][0];
                Force* force_ = lmp->force_stencil_md[zoid_num][0];

                for (int k = 0; k < lmp->queues[dep].size(); k++) {
                    if (lmp->queues[dep][k].num != zoid_num) {
                        int nrecv;
                        MPI_Recv(&nrecv, 1, MPI_INT, lmp->queues[dep][k].num % comm->nprocs,
                                 zoid_num, world, MPI_STATUS_IGNORE);

                        int *data = new int[nrecv];

                        if (nrecv) {
                            MPI_Recv(data, nrecv, MPI_INT,
                                     lmp->queues[dep][k].num % comm->nprocs, zoid_num, world, MPI_STATUS_IGNORE);
                        }

                        for (int h = 0; h < nrecv; h++) {
                            ghost_to_zoids_eval_final[data[h]].insert(lmp->queues[dep][k].num);
                        }

                        delete[] data;
                    }
                }
            }
        }
    }

    MPI_Barrier(world);

    for (int i = 0; i < send_bufs3.size(); i++) {
        delete[] send_bufs3[i];
    }

    std::map<int, int> tag_to_local_zoid;
    for (int dep = 0; dep < NUM_DEPS; dep++) {
        for (int j = 0; j < lmp->queues[dep].size(); j++) {
            queue_info& tmp = lmp->queues[dep][j];
            int zoid_num = tmp.num;
            queue_info& zoid = lmp->zoid_num_to_zoid[zoid_num];
            if (zoid_num % comm->nprocs == comm->me) {
                // lmp->comm_stencil_md[zoid_num]->construct_send_list_stencil_md(lmp->atom_stencil_md[zoid_num], zoid);
                Atom* atom_ = lmp->atom_stencil_md[zoid_num][0];
                for (int idx = 0; idx < atom_->nlocal; idx++) {
                    tag_to_local_zoid[atom_->tag[idx]] = zoid_num;
                }
            }
        }
    }

    for (auto&[k, v]: ghost_to_zoids_eval_final) {
        if (v.size() >= 2) {
            if (tag_to_local_zoid.count(k) && tag_to_local_zoid[k] == 63 && false) {
                std::cout << "LOCAL ZOID: " << tag_to_local_zoid[k] << " for tag: " << k << std::endl;
                Atom* atom_ = lmp->atom_stencil_md[tag_to_local_zoid[k]][0];
                for (int idx = 0; idx < atom_->nlocal; idx++) {
                    if (atom_->tag[idx] == k) {
                        std::cout << "Pos: " << atom_->x[idx][0] << " " << atom_->x[idx][1] << " " << atom_->x[idx][2] << std::endl;
                    }
                }
            }
            MPI_Barrier(world);
            if (comm->me == 0 && false) {
                std::cout << "OK HERE. GHOST TAG: " << k << " WAS GHOST EVALED BY ZOIDS: " << v <<  " NUM SHARED: " << v.size() << std::endl;
                for (int zoid_num : v) {
                    std::cout << "------ ZOID: " << zoid_num << " ------ZOID DIMS ------" << std::endl;
                    for (int dim = 0; dim < 3; dim++) {
                        std::cout << "lo: " << lmp->zoid_num_to_zoid[zoid_num].zoid.cuts[dim].lower << " hi: "
                                  << lmp->zoid_num_to_zoid[zoid_num].zoid.cuts[dim].upper << std::endl;
                    }

                }
            }
            MPI_Barrier(world);
        }
    }
    */

    MPI_Barrier(world);

    std::cout << "natoms: " << atom->natoms << std::endl;
    double* send_f = new double[(atom->natoms + 1) * 3];

    memset(send_f, 0, sizeof(send_f));

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

    memset(recv_f, 0, sizeof(recv_f));

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
            if (fabs(recv_f[tag * 3 + dim] - atom->f[i][dim]) > 1e-5) {
                std::cout << "idx: " << i << " dim: " << dim << " what I have: " << atom->f[i][dim] << " what I got: " << recv_f[tag * 3 + dim] << " diff: " << fabs(recv_f[tag * 3 + dim] - atom->f[i][dim]) << std::endl;
            }
            assert(fabs(recv_f[tag * 3 + dim] - atom->f[i][dim]) <= 1e-5);
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
                for (int k = 0; k < atom_->nlocal + atom_->nghost; k++) {
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
            queue_info& zoid = lmp->queues[dep][j];
            int zoid_num = zoid.num;
            if (zoid_num % comm->nprocs == comm->me) {
                Atom* atom_ = lmp->atom_stencil_md[zoid_num][0];
                Atom* next = lmp->atom_stencil_md[zoid_num][1];
                for (int idx = 0; idx < atom_->nlocal; idx++) {
                    if (zoid.atom_idx_mapping[0][idx] < next->nlocal) {
                        int tag = atom_->tag[idx];
                        for (int dim = 0; dim < 3; dim++) {
                            if (fabs(recv_f[tag * 3 + dim] - atom_->f[idx][dim]) > 1e-5) {
                                std::cout << "zoid: " << zoid_num << " idx: " << idx << " tag: " << atom_->tag[idx] << " dim: " << dim << " what I have: " << atom_->f[idx][dim] << " what I got: " << recv_f[tag * 3 + dim] << " diff: " << fabs(recv_f[tag * 3 + dim] - atom_->f[idx][dim]) << std::endl;
                            }
                            assert(fabs(atom_->f[idx][dim] - recv_f[tag * 3 + dim]) <= 1e-5);
                        }
                        total_evaled++;
                    }
                }
            }
        }
    }

    int total_atoms_evaled = 0;
    MPI_Allreduce(&total_atoms_evaled, &total_evaled, 1, MPI_INT, MPI_SUM, world);
    if (total_atoms_evaled != atom->natoms) {
        std::cout << "atoms evaled: " << total_atoms_evaled << " total number of atoms: " << atom->natoms << std::endl;
    }

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
            if (torqueflag) memset(&atom_->torque[0][0],0,3*nbytes);
            if (extraflag) atom_->avec->force_clear(0,nbytes);
        }

        if (force_->newton) {
            nbytes = sizeof(double) * atom_->nghost;

            if (nbytes) {
                memset(&atom_->f[nlocal][0],0,3*nbytes);
                if (torqueflag) memset(&atom_->torque[nlocal][0],0,3*nbytes);
                if (extraflag) atom_->avec->force_clear(nlocal,nbytes);
            }
        }
    }
}
