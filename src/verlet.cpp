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

void Verlet::group_local_atoms_stencil_md(Atom* atom_, queue_info& zoid, int timestep) {
    int zoid_num = zoid.num;
    auto& recv_from = lmp->recv_from[zoid.num];

    int nlocal = atom_->nlocal;
    int* current = new int[nlocal];
    int* permute = new int[nlocal];
    for (int i = 0; i < nlocal; i++) {
        current[i] = i;
    }
    std::vector<int> buckets[recv_from.size() + 1];
    std::set<int> in_bucket_atoms;
    std::map<int, int> atom_idx_to_recv_zoid_num;
    std::map<int, int> tag_to_recv_zoid_num;
    for (int i = 0; i < recv_from.size(); i++) {
        int recv_zoid_num = recv_from[i];
        queue_info& recv_from_zoid = lmp->zoid_num_to_zoid[recv_zoid_num];
        // recv_from is neighbor
        for (int j = 0; j < nlocal; j++) {
            double *pos = atom_->x[j];
            // technically check if borders zoid
            bool in_zoid = true;
            for (int dim = 0; dim < domain->dimension; dim++) {
                double value = pos[dim];
                double lo = recv_from_zoid.zoid.cuts[dim].lower + (timestep - 1) * recv_from_zoid.zoid.cuts[dim].slope_lower;
                double hi = recv_from_zoid.zoid.cuts[dim].upper + (timestep - 1) * recv_from_zoid.zoid.cuts[dim].slope_upper;
                if (lo < 0) {
                    in_zoid = in_zoid && ((value >= domain->prd[dim] + lo) || (value < hi));
                } else {
                    in_zoid = in_zoid && (value >= lo) && (value < hi);
                }
            }
            if (in_zoid) {
                buckets[i].push_back(j);
                in_bucket_atoms.insert(j);
                if (atom_idx_to_recv_zoid_num.count(j)) {
                    int found_zoid = atom_idx_to_recv_zoid_num[j];
                    std::cout << "Recv zoid num: " << found_zoid << " timestep: " << timestep << std::endl;
                    std::cout << "Atom pos: " << atom_->x[j][0] << " " << atom_->x[j][1] << " " << atom_->x[j][2] << std::endl;
                    for (int dim = 0; dim < 3; dim++) {
                        std::cout << "This zoid right now: lo: " << recv_from_zoid.zoid.cuts[dim].lower + timestep * recv_from_zoid.zoid.cuts[dim].slope_lower << " hi: "
                            << recv_from_zoid.zoid.cuts[dim].upper + timestep * recv_from_zoid.zoid.cuts[dim].slope_upper << std::endl;
                    }

                    queue_info& other_zoid = lmp->zoid_num_to_zoid[found_zoid];
                    for (int dim = 0; dim < 3; dim++) {
                        std::cout << "other zound before: lo: " << other_zoid.zoid.cuts[dim].lower + timestep * other_zoid.zoid.cuts[dim].slope_lower << " hi: "
                                  << other_zoid.zoid.cuts[dim].upper + timestep * other_zoid.zoid.cuts[dim].slope_upper << std::endl;
                    }

                    std::cout << "recv from for zoid: " << zoid.num << " is: " << recv_from << std::endl;

                    assert(false);
                }
                atom_idx_to_recv_zoid_num[j] = recv_zoid_num;
                tag_to_recv_zoid_num[atom_->tag[j]] = recv_zoid_num;
            }
        }
    }

    for (int i = 0; i < nlocal; i++) {
        if (in_bucket_atoms.find(i) == in_bucket_atoms.end()) {
            buckets[recv_from.size()].push_back(i);
        }
    }

    int permute_idx = 0;
    std::vector<int>& last_bucket = buckets[recv_from.size()];
    for (int i = 0; i < last_bucket.size(); i++) {
        permute[permute_idx++] = last_bucket[i];
    }

    for (int i = 0; i < recv_from.size(); i++) {
        std::sort(buckets[i].begin(), buckets[i].end(), [&](const int & a, const int & b) -> bool
        {
            // return a.mProperty > b.mProperty;
            return atom_->tag[a] < atom_->tag[b];
        });
        for (int j = 0; j < buckets[i].size(); j++) {
            permute[permute_idx++] = buckets[i][j];
        }
    }

    assert(permute_idx == nlocal);

    atom_reorder_stencil_md(atom_, current, permute, 0, nlocal);

    // create first_recv?
    for (int i = 0; i < recv_from.size(); i++) {
        int recv_zoid_num = recv_from[i];
        int idx = last_bucket.size();

        for (int j = 0; j < i; j++) {
            idx += buckets[j].size();
        }

        zoid.first_recv_stencil_md[timestep][i] = idx;
        // std::cout << "me: " << comm->me << " i: " << i << " idx: " << idx << " timestep: " << timestep << std::endl;

        for (int k = idx; k < idx + buckets[i].size(); k++) {
            assert(tag_to_recv_zoid_num[atom_->tag[k]] == recv_zoid_num);
        }
    }
}

void Verlet::group_ghost_atoms_stencil_md(Atom* atom_, queue_info& zoid, int timestep) {
    int zoid_num = zoid.num;
    auto& recv_from = lmp->recv_from[zoid.num];

    int nghost = atom_->nghost;
    int* current = new int[nghost];
    int* permute = new int[nghost];
    for (int i = 0; i < atom_->nghost; i++) {
        current[i] = i;
    }
    std::vector<int> buckets[recv_from.size() + 1];
    std::set<int> in_bucket_atoms;
    std::map<int, int> atom_idx_to_recv_zoid_num;

    std::map<int, std::set<int>> atom_idx_to_recv_zoid_num2;
    std::map<int, int> tag_to_recv_zoid_num;
    for (int i = 0; i < recv_from.size(); i++) {
        int recv_zoid_num = recv_from[i];
        queue_info& recv_from_zoid = lmp->zoid_num_to_zoid[recv_zoid_num];
        // recv_from is neighbor
        for (int j = 0; j < nghost; j++) {
            int idx = atom_->nlocal + j;
            double *pos = atom_->x[idx];
            // check if borders zoid
            bool borders_zoid = true;
            for (int dim = 0; dim < domain->dimension; dim++) {
                double value = pos[dim];
                double lo = recv_from_zoid.zoid.cuts[dim].lower + (timestep) * recv_from_zoid.zoid.cuts[dim].slope_lower;
                double hi = recv_from_zoid.zoid.cuts[dim].upper + (timestep) * recv_from_zoid.zoid.cuts[dim].slope_upper;
                lo -= 5.5;
                hi += 5.5;
                if (lo < 0) {
                    borders_zoid = borders_zoid && ((value >= domain->prd[dim] + lo) || (value < hi));
                } else {
                    borders_zoid = borders_zoid && (value >= lo) && (value < hi);
                }
            }

            if (borders_zoid) {
                // buckets[i].push_back(idx);
                in_bucket_atoms.insert(idx);
                /*
                if (atom_idx_to_recv_zoid_num.count(idx)) {
                    int found_zoid = atom_idx_to_recv_zoid_num[idx];
                    std::cout << "Recv zoid num: " << found_zoid << " timestep: " << timestep << std::endl;
                    std::cout << "Atom pos: " << pos[0] << " " << pos[1] << " " << pos[2] << std::endl;
                    for (int dim = 0; dim < 3; dim++) {
                        std::cout << "This zoid right now: lo: " << recv_from_zoid.zoid.cuts[dim].lower + timestep * recv_from_zoid.zoid.cuts[dim].slope_lower << " hi: "
                                  << recv_from_zoid.zoid.cuts[dim].upper + timestep * recv_from_zoid.zoid.cuts[dim].slope_upper << std::endl;
                    }

                    queue_info& other_zoid = lmp->zoid_num_to_zoid[found_zoid];
                    for (int dim = 0; dim < 3; dim++) {
                        std::cout << "other zound before: lo: " << other_zoid.zoid.cuts[dim].lower + timestep * other_zoid.zoid.cuts[dim].slope_lower << " hi: "
                                  << other_zoid.zoid.cuts[dim].upper + timestep * other_zoid.zoid.cuts[dim].slope_upper << std::endl;
                    }

                    std::cout << "recv from for zoid: " << zoid.num << " is: " << recv_from << std::endl;

                    assert(false);
                }
                atom_idx_to_recv_zoid_num[idx] = recv_zoid_num;
                */
                // atom_idx_to_recv_zoid_num2[idx].insert(recv_zoid_num);
                atom_idx_to_recv_zoid_num2[idx].insert(i);
                tag_to_recv_zoid_num[atom_->tag[idx]] = recv_zoid_num;
            }
        }
    }

    // construct buckets
    for (auto& [atom_idx, recv_from_idxs] : atom_idx_to_recv_zoid_num2) {
        std::vector<int> actual_vals;
        for (int x: recv_from_idxs) {
            actual_vals.push_back(recv_from[x]);
        }
        if (recv_from_idxs.size() > 1) {
            std::cout << "Recv from size > 1: " << recv_from_idxs.size() << " zoid_num: " << zoid_num << " actual vals: " << actual_vals << std::endl;
        }
        int min_idx = *recv_from_idxs.begin();
        buckets[min_idx].push_back(atom_idx);
    }

    for (int i = 0; i < nghost; i++) {
        if (in_bucket_atoms.find(i + atom_->nlocal) == in_bucket_atoms.end()) {
            buckets[recv_from.size()].push_back(i + atom_->nlocal);
        }
    }

    int total = 0;
    for (int k = 0; k < recv_from.size() + 1; k++) {
        total += buckets[k].size();
    }
    std::cout << "Total: " << total << " nghost: " << nghost << std::endl;
    assert(total == nghost);

    int permute_idx = 0;
    std::vector<int>& last_bucket = buckets[recv_from.size()];
    for (int i = 0; i < last_bucket.size(); i++) {
        permute[permute_idx++] = last_bucket[i] - atom_->nlocal;
    }

    for (int i = 0; i < recv_from.size(); i++) {
        std::sort(buckets[i].begin(), buckets[i].end(), [&](const int & a, const int & b) -> bool
        {
            // return a.mProperty > b.mProperty;
            return atom_->tag[a] < atom_->tag[b];
        });
        for (int j = 0; j < buckets[i].size(); j++) {
            permute[permute_idx++] = buckets[i][j] - atom_->nlocal;
        }
    }

    assert(permute_idx == nghost);

    atom_reorder_ghost_stencil_md(atom_, current, permute, 0, nghost, atom_->nlocal);

    // create second recv?
    for (int i = 0; i < recv_from.size(); i++) {
        int recv_zoid_num = recv_from[i];
        int idx = last_bucket.size();

        for (int j = 0; j < i; j++) {
            idx += buckets[j].size();
        }

        zoid.second_recv_stencil_md[timestep][i] = idx + atom_->nlocal;
        // std::cout << "me: " << comm->me << " i: " << i << " idx: " << idx << " timestep: " << timestep << std::endl;

        for (int k = idx; k < idx + buckets[i].size(); k++) {
            // int real_idx = k + atom_->nlocal;
            // assert(tag_to_recv_zoid_num[atom_->tag[real_idx]] == recv_zoid_num);
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

    int** mapping = new int*[NUM_TIMESTEPS_IN_PARALLEL];
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
    // TODO: RYAN FIX
    double cutoff = 5.5 + ADDITIONAL_CUTOFF;
    get_zoids(cutoff, domain->boxlo, domain->boxhi, lmp->queues);

    lmp->zoid_num_to_idx = new int[NUM_ZOIDS];
    lmp->zoid_num_to_zoid = new queue_info[NUM_ZOIDS];
    for (int i = 0; i < NUM_ZOIDS; i++) {
        lmp->zoid_num_to_idx[i] = -1;
    }
    int zoid_idx = 0;
    for (int dep = 0; dep < 3 + 1; dep++) {
        for (int j = 0; j < lmp->queues[dep].size(); j++) {
            int zoid_num = lmp->queues[dep][j].num;
            lmp->zoid_num_to_idx[zoid_num] = zoid_idx++;
            lmp->zoid_num_to_zoid[zoid_num] = lmp->queues[dep][j];
        }
    }

    lmp->send_to = new std::vector<int>[NUM_ZOIDS];
    lmp->recv_from = new std::vector<int>[NUM_ZOIDS];
    lmp->send_to_next_dt = new std::vector<int>[NUM_ZOIDS];
    lmp->recv_from_next_dt = new std::vector<int>[NUM_ZOIDS];

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
    for (int dep = 0; dep < (3 + 1); dep++) {
        for (int j = 0; j < lmp->queues[dep].size(); j++) {
            queue_info& zoid = lmp->queues[dep][j];
            int zoid_num = zoid.num;
            int idx_ = lmp->zoid_num_to_idx[zoid_num];
            for (int k = 0; k < lmp->domain_stencil_md[j].size(); k++) {
                Domain* domain_ = lmp->domain_stencil_md[idx_][k];
                for (int dim = 0; dim < domain->dimension; dim++) {
                    domain_->sublo[dim] = zoid.zoid.cuts[dim].lower + k * zoid.zoid.cuts[dim].slope_lower;
                    domain_->subhi[dim] = zoid.zoid.cuts[dim].upper + k * zoid.zoid.cuts[dim].slope_upper;
                    domain_->boxlo[dim] = zoid.zoid.cuts[dim].lower + k * zoid.zoid.cuts[dim].slope_lower;
                    domain_->boxhi[dim] = zoid.zoid.cuts[dim].upper + k * zoid.zoid.cuts[dim].slope_upper;
                    domain_->prd[dim] = domain->prd[dim];
                    // domain_->boxlo[dim] = domain->boxlo[dim];
                    // domain_->boxhi[dim] = domain->boxhi[dim];
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
    for (int dep = 0; dep < 3 + 1; dep++) {
        for (int j = 0; j < lmp->queues[dep].size(); j++) {
            queue_info& zoid = lmp->queues[dep][j];
            int zoid_num = zoid.num;
            int idx_ = lmp->zoid_num_to_idx[zoid_num];
            if (zoid_num % comm->nprocs == comm->me) {
                for (int k = 0; k < lmp->atom_stencil_md[idx_].size(); k++) {
                    lmp->atom_stencil_md[idx_][k]->setup_stencil_md(lmp->domain_stencil_md[idx_][k]);
                }
                // lmp->atom_stencil_md[idx_][0]->setup_stencil_md(lmp->domain_stencil_md[idx_]);
            }
        }
    }

    // comm exchange
    for (int i = 0; i < NUM_TIMESTEPS_IN_PARALLEL + 1; i++) {
        comm->exchange_stencil_md_initial_send();
        for (int dep = 0; dep < 3 + 1; dep++) {
            for (int j = 0; j < lmp->queues[dep].size(); j++) {
                queue_info& zoid = lmp->queues[dep][j];
                int zoid_num = zoid.num;
                int idx_ = lmp->zoid_num_to_idx[zoid_num];
                // receive only if the zoid belongs to me
                if (zoid_num % comm->nprocs == comm->me) {
                    // std::cout << "exchange stencil md initial receive Zoid num: " << zoid_num << " for time: " << i << std::endl;
                    lmp->comm_stencil_md[idx_]->exchange_stencil_md_initial_receive(lmp->atom_stencil_md[idx_][i], lmp->domain_stencil_md[idx_][i], zoid);
                }
            }
        }
    }

    /*
    comm->exchange_stencil_md_initial_send();
    for (int dep = 0; dep < 3 + 1; dep++) {
        for (int j = 0; j < lmp->queues[dep].size(); j++) {
            queue_info& zoid = lmp->queues[dep][j];
            int zoid_num = zoid.num;
            int idx_ = lmp->zoid_num_to_idx[zoid_num];
            // receive only if the zoid belongs to me
            if (zoid_num % comm->nprocs == comm->me) {
                // lmp->comm_stencil_md[idx_]->exchange_stencil_md_initial_receive(lmp->atom_stencil_md[idx_][0], lmp->domain_stencil_md[idx_], zoid);
            }
        }
    }
    */

    // look for atom 31979
    /*
    for (int dep = 0; dep < 1; dep++) {
        for (int j = 0; j < lmp->queues[dep].size(); j++) {
            queue_info& zoid = lmp->queues[dep][j];
            int zoid_num = zoid.num;
            // receive only if the zoid belongs to me
            if (zoid_num % comm->nprocs == comm->me) {
                int idx_ = lmp->zoid_num_to_idx[zoid_num];
                for (int t = 0; t < NUM_TIMESTEPS_IN_PARALLEL + 1; t++) {
                    Atom* atom_ = lmp->atom_stencil_md[idx_][t];
                    for (int i = 0; i < atom_->nlocal; i++) {
                        if (atom_->tag[i] == 31979) {
                            std::cout << "SETUP STENCIL MD FOUND ATOM. Time: " << t << " Zoid num: " << zoid_num << " idx: " << i << " pos: " << atom_->x[i][0] << " " << atom_->x[i][1] << " " << atom_->x[i][2] << " nlocal: " << atom_->nlocal << std::endl;
                        }
                    }
                }

            }
        }
    }
    */

    for (int i = 0; i < NUM_TIMESTEPS_IN_PARALLEL + 1; i++) {
        int num_atoms_owned = 0;
        for (int dep = 0; dep < 3 + 1; dep++) {
            for (int j = 0; j < lmp->queues[dep].size(); j++) {
                queue_info& zoid = lmp->queues[dep][j];
                int zoid_num = zoid.num;
                // receive only if the zoid belongs to me
                if (zoid_num % comm->nprocs == comm->me) {
                    int idx_ = lmp->zoid_num_to_idx[zoid_num];
                    Atom* atom_ = lmp->atom_stencil_md[idx_][i];
                    num_atoms_owned += atom_->nlocal;
                    // std::cout << "Zoid idx: " << idx_ << " at time: " << i << " has num atoms: " << atom_->nlocal << std::endl;
                }
            }
        }

        // std::cout << "Me: " << comm->me << " Num atoms owned: " << num_atoms_owned << " for timestep: " << i << std::endl;
        int total_num_atoms = 0;
        MPI_Reduce(&num_atoms_owned, &total_num_atoms, 1, MPI_INT, MPI_SUM, 0, world);
        if (comm-> me == 0) {
            // std::cout << "total atoms owned across all zoids: " << total_num_atoms << " for timestep: " << i << std::endl;
        }

        for (int dep = 0; dep < 3 + 1; dep++) {
            for (int j = 0; j < lmp->queues[dep].size(); j++) {
                int zoid_num = lmp->queues[dep][j].num;
                if (zoid_num % comm->nprocs == comm->me) {
                    int idx_ = lmp->zoid_num_to_idx[zoid_num];
                    lmp->domain_stencil_md[idx_][i]->pbc_stencil_md(lmp->atom_stencil_md[idx_][i]);
                }
            }
        }

        for (int dep = 0; dep < 3 + 1; dep++) {
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

        // atom sort
        if (atom->sortfreq > 0 && false) {
            for (int dep = 0; dep < 3 + 1; dep++) {
                for (int j = 0; j < lmp->queues[dep].size(); j++) {
                    int zoid_num = lmp->queues[dep][j].num;
                    if (zoid_num % comm->nprocs == comm->me) {
                        int idx_ = lmp->zoid_num_to_idx[zoid_num];
                        Atom* atom_ = lmp->atom_stencil_md[idx_][i];
                        atom_->sort_stencil_md();
                    }
                }
            }
        }

        // sort the nlocal for firstrecv
        for (int dep = 0; dep < 3 + 1; dep++) {
            for (int j = 0; j < lmp->queues[dep].size(); j++) {
                // TODO: fix, somehow link zoid_num_to_zoid and queues
                queue_info& tmp = lmp->queues[dep][j];
                int zoid_num = tmp.num;
                queue_info& zoid = lmp->zoid_num_to_zoid[zoid_num];
                int idx_ = lmp->zoid_num_to_idx[zoid_num];
                int num_recv_from = lmp->recv_from[idx_].size();
                if (zoid_num % comm->nprocs == comm->me) {
                    // todo: fix this 6 must mean some kind of weird error here
                    zoid.first_recv_stencil_md[i] = new int[num_recv_from];
                    zoid.init_first_recv = true;
                    group_local_atoms_stencil_md(lmp->atom_stencil_md[idx_][i], zoid, i);
                }
            }
        }

        // borders
        for (int dep = 0; dep < 3 + 1; dep++) {
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

        for (int dep = 0; dep < 3 + 1; dep++) {
            for (int j = 0; j < lmp->queues[dep].size(); j++) {
                // TODO: fix, somehow link zoid_num_to_zoid and queues
                queue_info& tmp = lmp->queues[dep][j];
                int zoid_num = tmp.num;
                queue_info& zoid = lmp->zoid_num_to_zoid[zoid_num];
                int idx_ = lmp->zoid_num_to_idx[zoid_num];
                if (zoid_num % comm->nprocs == comm->me) {
                    lmp->comm_stencil_md[idx_]->borders_stencil_md_initial_receive(lmp->atom_stencil_md[idx_][i], lmp->domain_stencil_md[idx_][i],
                                                                                zoid);
                }
            }
        }

        MPI_Barrier(world);

        for (int dep = 0; dep < 3 + 1; dep++) {
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

        for (int dep = 0; dep < 3 + 1; dep++) {
            for (int j = 0; j < lmp->queues[dep].size(); j++) {
                queue_info& tmp = lmp->queues[dep][j];
                int zoid_num = tmp.num;
                queue_info& zoid = lmp->zoid_num_to_zoid[zoid_num];
                if (zoid_num % comm->nprocs == comm->me) {
                    Atom* atom_ = lmp->atom_stencil_md[zoid_num][i];
                    // second_recv contains ghost atoms of a particular atom (not time 0, and not the last timestep)
                    if (i > 0 && i < NUM_TIMESTEPS_IN_PARALLEL) {
                        int num_recv_from = lmp->recv_from[zoid_num].size();
                        zoid.second_recv_stencil_md[i] = new int[num_recv_from];
                        group_ghost_atoms_stencil_md(lmp->atom_stencil_md[zoid_num][i], zoid, i);
                    }
                }
            }
        }
    }

    MPI_Barrier(world);

    for (int dep = 0; dep < 3 + 1; dep++) {
        for (int j = 0; j < lmp->queues[dep].size(); j++) {
            queue_info& tmp = lmp->queues[dep][j];
            int zoid_num = tmp.num;
            queue_info& zoid = lmp->zoid_num_to_zoid[zoid_num];
            if (zoid_num % comm->nprocs == comm->me) {
                int num_recv_from = lmp->recv_from[zoid_num].size();
                for (int t = 1; t < NUM_TIMESTEPS_IN_PARALLEL; t++) {
                    for (int k = 0; k < num_recv_from; k++) {
                        std::cout << "Dep: " << dep << " Zoid second recv stencil md: " << zoid.second_recv_stencil_md[t][k]
                            << " for t: " << t << " k: " << k << " out of: " << num_recv_from << std::endl;
                    }
                }
            }
        }
    }

    // construct second_send list, sending ghost->ghost
    for (int dep = 0; dep < 3 + 1; dep++) {
        for (int j = 0; j < lmp->queues[dep].size(); j++) {
            queue_info& tmp = lmp->queues[dep][j];
            int zoid_num = tmp.num;
            queue_info& zoid = lmp->zoid_num_to_zoid[zoid_num];
            if (zoid_num % comm->nprocs == comm->me) {
                // lmp->comm_stencil_md[zoid_num]->construct_send_list_stencil_md(lmp->atom_stencil_md[zoid_num], zoid);
                lmp->comm_stencil_md[zoid_num]->construct_second_send_list_stencil_md_send(lmp->atom_stencil_md[zoid_num], zoid);
            }
        }
    }

    for (int dep = 0; dep < 3 + 1; dep++) {
        for (int j = 0; j < lmp->queues[dep].size(); j++) {
            queue_info& tmp = lmp->queues[dep][j];
            int zoid_num = tmp.num;
            queue_info& zoid = lmp->zoid_num_to_zoid[zoid_num];
            if (zoid_num % comm->nprocs == comm->me) {
                // lmp->comm_stencil_md[zoid_num]->construct_send_list_stencil_md(lmp->atom_stencil_md[zoid_num], zoid);
                lmp->comm_stencil_md[zoid_num]->construct_second_send_list_stencil_md_receive(lmp->atom_stencil_md[zoid_num], zoid);
            }
        }
    }

    // construct first send_list, sending ghost->local
    for (int dep = 0; dep < 3 + 1; dep++) {
        for (int j = 0; j < lmp->queues[dep].size(); j++) {
            queue_info& tmp = lmp->queues[dep][j];
            int zoid_num = tmp.num;
            queue_info& zoid = lmp->zoid_num_to_zoid[zoid_num];
            if (zoid_num % comm->nprocs == comm->me) {
                // lmp->comm_stencil_md[zoid_num]->construct_send_list_stencil_md(lmp->atom_stencil_md[zoid_num], zoid);
                lmp->comm_stencil_md[zoid_num]->construct_send_list_stencil_md(lmp->atom_stencil_md[zoid_num], zoid);
            }
        }
    }

    for (int dep = 0; dep < 3 + 1; dep++) {
        for (int j = 0; j < lmp->queues[dep].size(); j++) {
            queue_info& zoid = lmp->queues[dep][j];
            int zoid_num = zoid.num;
            int idx_ = lmp->zoid_num_to_idx[zoid_num];
            if (zoid_num % comm->nprocs == comm->me) {
                setup_atom_pos_mapping_stencil_md(lmp->atom_stencil_md[idx_], zoid);
            }
        }
    }

    // compute force but only for the first timestep
    for (int dep = 0; dep < 3 + 1; dep++) {
        for (int j = 0; j < lmp->queues[dep].size(); j++) {
            queue_info& zoid = lmp->queues[dep][j];
            int zoid_num = zoid.num;
            int idx_ = lmp->zoid_num_to_idx[zoid_num];
            assert(zoid_num == idx_);
            if (zoid_num % comm->nprocs == comm->me) {
                AtomKokkos* atomKK_ = (AtomKokkos*) lmp->atom_stencil_md[idx_][0];
                Force* force_ = lmp->force_stencil_md[idx_][0];
                // todo: eflag and vflag might cause some issues
                // TODO: compute force for each pair in parallel

                // std::cout << "Me: " << comm->me << " setup force compute for zoid: " << zoid_num << std::endl;
                /*
                Domain* domain_ = lmp->domain_stencil_md[idx_][0];
                for (int dim = 0; dim < 3; dim++) {
                    std::cout << "Zoid: " << zoid_num << " dim: " << dim << " lower: " << domain_->boxlo[dim] << " upper: " << domain_->boxhi[dim] << std::endl;
                }
                */
                bool tmp[3] = {false
                               , false, false};
                for (int dim = 0; dim < 3; dim++) {
                    tmp[dim] = (zoid.where[dim] == PBC);
                }
                force_->pair->compute_stencil_md(eflag, vflag, lmp->atom_stencil_md[idx_][0], tmp);
                atomKK_->modified_stencil_md(force_->pair->execution_space, force_->pair->datamask_modify, lmp->atom_stencil_md[idx_][0]);
            }
        }
    }

    /*
    for (int dep = 0; dep < 3 + 1; dep++) {
        for (int j = 0; j < lmp->queues[dep].size(); j++) {
            queue_info& zoid = lmp->queues[dep][j];
            int zoid_num = zoid.num;
            int idx_ = lmp->zoid_num_to_idx[zoid_num];
            if (zoid_num % comm->nprocs == comm->me) {
                lmp->neighbor_stencil_md[idx_]->build_stencil_md(1, lmp->atom_stencil_md[idx_][0], lmp->domain_stencil_md[idx_], lmp->comm_stencil_md[idx_]);
                lmp->neighbor_stencil_md[idx_]->ncalls = 0;

                AtomKokkos* atomKK_ = (AtomKokkos*) lmp->atom_stencil_md[idx_][0];
                Force* force_ = lmp->force_stencil_md[idx_];
                lmp->force_stencil_md[idx_]->setup();
                atomKK_->sync_stencil_md(force->pair->execution_space,force->pair->datamask_read, lmp->atom_stencil_md[idx_][0]);
                force_clear_stencil_md(lmp->atom_stencil_md[idx_][0], lmp->force_stencil_md[idx_], lmp->neighbor_stencil_md[idx_]);
                // todo: eflag and vflag might cause some issues
                // TODO: compute force for each pair in parallel
                force_->pair->compute_stencil_md(eflag, vflag, lmp->atom_stencil_md[idx_][0]);
                atomKK_->modified_stencil_md(force_->pair->execution_space, force_->pair->datamask_modify, lmp->atom_stencil_md[idx_][0]);
            }
        }
    }
    */

    double total_temp_for_me = 0;
    for (int dep = 0; dep < 3 + 1; dep++) {
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
