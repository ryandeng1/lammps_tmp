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

#include "verlet_kokkos.h"
#include "neighbor.h"
#include "domain.h"
#include "comm.h"
#include "atom_kokkos.h"
#include "atom_masks.h"
#include "force.h"
#include "pair.h"
#include "bond.h"
#include "angle.h"
#include "dihedral.h"
#include "improper.h"
#include "kspace.h"
#include "output.h"
#include "update.h"
#include "modify.h"
#include "timer.h"
#include "memory_kokkos.h"
#include "kokkos.h"
#include <chrono>
#include <thread>
#include <unistd.h>

using namespace LAMMPS_NS;

template<class ViewA, class ViewB>
struct ForceAdder {
  ViewA a;
  ViewB b;
  ForceAdder(const ViewA& a_, const ViewB& b_):a(a_),b(b_) {}
  KOKKOS_INLINE_FUNCTION
  void operator() (const int& i) const {
    a(i,0) += b(i,0);
    a(i,1) += b(i,1);
    a(i,2) += b(i,2);
  }
};

/* ---------------------------------------------------------------------- */

template<class View>
struct Zero {
  View v;
  Zero(const View &v_):v(v_) {}
  KOKKOS_INLINE_FUNCTION
  void operator()(const int &i) const {
    v(i,0) = 0;
    v(i,1) = 0;
    v(i,2) = 0;
  }
};

/* ---------------------------------------------------------------------- */

VerletKokkos::VerletKokkos(LAMMPS *lmp, int narg, char **arg) :
  Verlet(lmp, narg, arg)
{
  atomKK = (AtomKokkos *) atom;
}

/* ----------------------------------------------------------------------
   setup before run
------------------------------------------------------------------------- */

void VerletKokkos::setup(int flag)
{
  std::cout << "VERLEX KOKKOS SETUP: " << flag << std::endl;
  if (comm->me == 0 && screen) {
    fputs("KOKKOS Setting up Verlet run ...\n",screen);
    if (flag) {
      fmt::print(screen,"  Unit style    : {}\n"
                        "  Current step  : {}\n"
                        "  Time step     : {}\n",
                 update->unit_style,update->ntimestep,update->dt);
      timer->print_timeout(screen);
    }
  }

  update->setupflag = 1;

  // setup domain, communication and neighboring
  // acquire ghosts
  // build neighbor lists

  lmp->kokkos->auto_sync = 1;

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

  std::cout << "Domain check. " << " num atoms: " << atom->natoms << std::endl;
  for (int dim = 0; dim < 3; dim++) {
    std::cout << "Domain sublo: " << domain->sublo[dim] << " subhi: " << domain->subhi[dim] << std::endl;
  }
  if (pair_compute_flag) {
    atomKK->sync(force->pair->execution_space,force->pair->datamask_read);
    std::cout << "REGULAR MD START force compute Me: " << comm->me << std::endl;
    force->pair->compute(eflag,vflag);
    std::cout << "REGULAR MD END force compute. Me: " << comm->me << std::endl;
    atomKK->modified(force->pair->execution_space,force->pair->datamask_modify);
  }
  else if (force->pair) force->pair->compute_dummy(eflag,vflag);

  // sum up forces


  if (atom->molecular != Atom::ATOMIC) {
    if (force->bond) {
      atomKK->sync(force->bond->execution_space,force->bond->datamask_read);
      force->bond->compute(eflag,vflag);
      atomKK->modified(force->bond->execution_space,force->bond->datamask_modify);
    }
    if (force->angle) {
      atomKK->sync(force->angle->execution_space,force->angle->datamask_read);
      force->angle->compute(eflag,vflag);
      atomKK->modified(force->angle->execution_space,force->angle->datamask_modify);
    }
    if (force->dihedral) {
      atomKK->sync(force->dihedral->execution_space,force->dihedral->datamask_read);
      force->dihedral->compute(eflag,vflag);
      atomKK->modified(force->dihedral->execution_space,force->dihedral->datamask_modify);
    }
    if (force->improper) {
      atomKK->sync(force->improper->execution_space,force->improper->datamask_read);
      force->improper->compute(eflag,vflag);
      atomKK->modified(force->improper->execution_space,force->improper->datamask_modify);
    }
  }

  if (force->kspace) {
    force->kspace->setup();
    if (kspace_compute_flag) {
      atomKK->sync(force->kspace->execution_space,force->kspace->datamask_read);
      force->kspace->compute(eflag,vflag);
      atomKK->modified(force->kspace->execution_space,force->kspace->datamask_modify);
    } else force->kspace->compute_dummy(eflag,vflag);
  }

  modify->setup_pre_reverse(eflag,vflag);
  if (force->newton) {
      auto begin = std::chrono::high_resolution_clock::now();
      comm->reverse_comm();
      auto end = std::chrono::high_resolution_clock::now();
      auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end-begin).count();
      std::cout << GREEN << "me: " << comm->me << " nprocs: " << comm->nprocs << " lammps reverse comm duration: " << duration << RESET_COLOR << std::endl;
  }

  double neigh_cutoff = force->pair->cutforce  + neighbor->skin;
  std::cout << "neigh cutoff: " << neigh_cutoff << std::endl;

  double x_ = 0.0;
  double y_ = 0.0;
  double z_ = 0.0;
  for (int i = 0; i < atom->nlocal; i++) {
    x_ += atom->f[i][0];
    y_ += atom->f[i][1];
    z_ += atom->f[i][2];
  }

  double total_x = 0;
  double total_y = 0;
  double total_z = 0;
  MPI_Reduce(&x_, &total_x, 1, MPI_DOUBLE, MPI_SUM, 0, world);
  MPI_Reduce(&y_, &total_y, 1, MPI_DOUBLE, MPI_SUM, 0, world);
  MPI_Reduce(&z_, &total_z, 1, MPI_DOUBLE, MPI_SUM, 0, world);

  if (comm->me == 0) {
      std::cout << "Sum x: " << total_x << " Sum y: " << total_y << " Sum z: " << total_z << std::endl;
  }

  lmp->kokkos->auto_sync = 0;
  modify->setup(vflag);
  output->setup(flag);
  lmp->kokkos->auto_sync = 1;
  update->setupflag = 0;

  std::cout << "ME: " << comm->me << " START SETUP STENCIL MD" << " nthreads: " << std::thread::hardware_concurrency() << std::endl;
  setup_stencil_md();
  std::cout << "ME: " << comm->me << " DONE SETUP STENCIL MD" << std::endl;
}

/* ----------------------------------------------------------------------
   setup without output
   flag = 0 = just force calculation
   flag = 1 = reneighbor and force calculation
------------------------------------------------------------------------- */


void VerletKokkos::setup_minimal(int flag)
{
  update->setupflag = 1;

  // setup domain, communication and neighboring
  // acquire ghosts
  // build neighbor lists

  lmp->kokkos->auto_sync = 1;

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

  if (pair_compute_flag) {
    atomKK->sync(force->pair->execution_space,force->pair->datamask_read);
    force->pair->compute(eflag,vflag);
    atomKK->modified(force->pair->execution_space,force->pair->datamask_modify);
  }
  else if (force->pair) force->pair->compute_dummy(eflag,vflag);

  if (atom->molecular != Atom::ATOMIC) {
    if (force->bond) {
      atomKK->sync(force->bond->execution_space,force->bond->datamask_read);
      force->bond->compute(eflag,vflag);
      atomKK->modified(force->bond->execution_space,force->bond->datamask_modify);
    }
    if (force->angle) {
      atomKK->sync(force->angle->execution_space,force->angle->datamask_read);
      force->angle->compute(eflag,vflag);
      atomKK->modified(force->angle->execution_space,force->angle->datamask_modify);
    }
    if (force->dihedral) {
      atomKK->sync(force->dihedral->execution_space,force->dihedral->datamask_read);
      force->dihedral->compute(eflag,vflag);
      atomKK->modified(force->dihedral->execution_space,force->dihedral->datamask_modify);
    }
    if (force->improper) {
      atomKK->sync(force->improper->execution_space,force->improper->datamask_read);
      force->improper->compute(eflag,vflag);
      atomKK->modified(force->improper->execution_space,force->improper->datamask_modify);
    }
  }

  if (force->kspace) {
    force->kspace->setup();
    if (kspace_compute_flag) {
      atomKK->sync(force->kspace->execution_space,force->kspace->datamask_read);
      force->kspace->compute(eflag,vflag);
      atomKK->modified(force->kspace->execution_space,force->kspace->datamask_modify);
    } else force->kspace->compute_dummy(eflag,vflag);
  }

  modify->setup_pre_reverse(eflag,vflag);
  if (force->newton) comm->reverse_comm();

  lmp->kokkos->auto_sync = 0;
  modify->setup(vflag);
  lmp->kokkos->auto_sync = 1;
  update->setupflag = 0;
}

/* ----------------------------------------------------------------------
   run for N steps
------------------------------------------------------------------------- */

void VerletKokkos::run(int n) {
    assert(false);
    std::cout << "atom nlocal: " << atom->nlocal << " nghost: " << atom->nghost << std::endl;
    constexpr auto max_precision{std::numeric_limits<long double>::digits10 + 1};
    std::cout << std::setprecision(max_precision);
    bigint ntimestep;
    int nflag, sortflag;

    int n_post_integrate = modify->n_post_integrate;
    int n_pre_exchange = modify->n_pre_exchange;
    int n_pre_neighbor = modify->n_pre_neighbor;
    int n_post_neighbor = modify->n_post_neighbor;
    int n_pre_force = modify->n_pre_force;
    int n_pre_reverse = modify->n_pre_reverse;
    int n_post_force = modify->n_post_force_any;
    int n_end_of_step = modify->n_end_of_step;

    /*
    std::cout << "n_post_integrate: " << n_post_integrate << " n_pre_exchange: " << n_pre_exchange << " n_pre_neighbor: " << n_pre_neighbor << " n_post_neighbor: " << n_post_neighbor << std::endl;
    std::cout << "n_pre_force: " << n_pre_force << " n_pre_reverse: " << n_pre_reverse << " n_post_force: " << n_post_force << " n_end_of_step: " << n_end_of_step << std::endl;
    */

    lmp->kokkos->auto_sync = 0;

    if (atomKK->sortfreq > 0) sortflag = 1;
    else sortflag = 0;

    f_merge_copy = DAT::t_f_array("VerletKokkos::f_merge_copy", atomKK->k_f.extent(0));

    atomKK->sync(Device, ALL_MASK);

    int test_num_timesteps = 1 * (NUM_TIMESTEPS_IN_PARALLEL + 1);
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

    timer->init_timeout();

    int64_t lammps_comm_duration = 0;

    for (int i = 0; i < test_num_timesteps; i++) {
        if (timer->check_timeout(i)) {
            update->nsteps = i;
            break;
        }

        ntimestep = ++update->ntimestep;
        ev_set(ntimestep);

        // initial time integration
        timer->stamp();
        if (TEST_AGAINST_LAMMPS) {
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

            MPI_Allreduce(
                    send_f,
                    test_f[i],
                    (atom->natoms + 1) * 3,
                    MPI_DOUBLE,
                    MPI_SUM,
                    world);

            MPI_Allreduce(
                    send_x,
                    test_x[i],
                    (atom->natoms + 1) * 3,
                    MPI_DOUBLE,
                    MPI_SUM,
                    world);
        }

        if (comm->me == 0) {
            std::cout << BLUE << "REGULAR MD INITIAL INTEGRATE" << " for time: " << i << RESET_COLOR << std::endl;
        }
        modify->initial_integrate(vflag);
        if (n_post_integrate) modify->post_integrate();
        timer->stamp(Timer::MODIFY);

        // regular communication vs neighbor list rebuild
        nflag = neighbor->decide();

        if (nflag == 0) {
            timer->stamp();
            auto begin = std::chrono::high_resolution_clock::now();
            comm->forward_comm();
            auto end = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end-begin).count();
            std::cout << BLUE << "Me: " << comm->me << " forward comm duration: " << duration << RESET_COLOR << std::endl;
            timer->stamp(Timer::COMM);
        } else {
            assert(false);
            // added debug
            //atomKK->sync(Host,ALL_MASK);
            //atomKK->modified(Host,ALL_MASK);

            if (n_pre_exchange) {
                timer->stamp();
                modify->pre_exchange();
                timer->stamp(Timer::MODIFY);
            }
            // debug
            //atomKK->sync(Host,ALL_MASK);
            //atomKK->modified(Host,ALL_MASK);
            if (triclinic) domain->x2lamda(atomKK->nlocal);
            domain->pbc();
            if (domain->box_change) {
                domain->reset_box();
                comm->setup();
                if (neighbor->style) neighbor->setup_bins();
            }
            timer->stamp();

            // added debug
            //atomKK->sync(Device,ALL_MASK);
            //atomKK->modified(Device,ALL_MASK);

            comm->exchange();
            if (sortflag && ntimestep >= atomKK->nextsort) atomKK->sort();
            comm->borders();

            // added debug
            //atomKK->sync(Host,ALL_MASK);
            //atomKK->modified(Host,ALL_MASK);

            if (triclinic) domain->lamda2x(atomKK->nlocal + atomKK->nghost);

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

        bool execute_on_host = false;
        unsigned int datamask_read_host = 0;
        unsigned int datamask_exclude = 0;
        int allow_overlap = lmp->kokkos->allow_overlap;

        if (allow_overlap && atomKK->k_f.h_view.data() != atomKK->k_f.d_view.data()) {

            datamask_exclude = (F_MASK | ENERGY_MASK | VIRIAL_MASK);

            if (pair_compute_flag) {
                if (force->pair->execution_space == Host) {
                    execute_on_host = true;
                    datamask_read_host |= force->pair->datamask_read;
                }
            }
            if (atomKK->molecular && force->bond) {
                if (force->bond->execution_space == Host) {
                    execute_on_host = true;
                    datamask_read_host |= force->bond->datamask_read;
                }
            }
            if (atomKK->molecular && force->angle) {
                if (force->angle->execution_space == Host) {
                    execute_on_host = true;
                    datamask_read_host |= force->angle->datamask_read;
                }
            }
            if (atomKK->molecular && force->dihedral) {
                if (force->dihedral->execution_space == Host) {
                    execute_on_host = true;
                    datamask_read_host |= force->dihedral->datamask_read;
                }
            }
            if (atomKK->molecular && force->improper) {
                if (force->improper->execution_space == Host) {
                    execute_on_host = true;
                    datamask_read_host |= force->improper->datamask_read;
                }
            }
            if (kspace_compute_flag) {
                if (force->kspace->execution_space == Host) {
                    execute_on_host = true;
                    datamask_read_host |= force->kspace->datamask_read;
                }
            }
        }

        if (pair_compute_flag) {
            atomKK->sync(force->pair->execution_space, force->pair->datamask_read);
            atomKK->sync(force->pair->execution_space, ~(~force->pair->datamask_read | datamask_exclude));
            auto begin = std::chrono::high_resolution_clock::now();
            force->pair->compute(eflag, vflag);
            auto end = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end-begin).count();
            std::cout << BLUE << "Process: " << comm->me << " lammps compute duration: " << std::chrono::duration_cast<std::chrono::milliseconds>(end-begin).count() << " milliseconds " << RESET_COLOR << std::endl;
            lammps_compute_duration += duration;
            atomKK->modified(force->pair->execution_space, force->pair->datamask_modify);
            atomKK->modified(force->pair->execution_space, ~(~force->pair->datamask_modify | datamask_exclude));
            timer->stamp(Timer::PAIR);
        }

        if (execute_on_host) {
            if (pair_compute_flag && force->pair->datamask_modify != datamask_exclude)
                Kokkos::fence();
            atomKK->sync_overlapping_device(Host, ~(~datamask_read_host | datamask_exclude));
            if (pair_compute_flag && force->pair->execution_space != Host) {
                Kokkos::deep_copy(LMPHostType(), atomKK->k_f.h_view, 0.0);
            }
        }

        if (atomKK->molecular) {
            if (force->bond) {
                atomKK->sync(force->bond->execution_space, ~(~force->bond->datamask_read | datamask_exclude));
                force->bond->compute(eflag, vflag);
                atomKK->modified(force->bond->execution_space, ~(~force->bond->datamask_modify | datamask_exclude));
            }
            if (force->angle) {
                atomKK->sync(force->angle->execution_space, ~(~force->angle->datamask_read | datamask_exclude));
                force->angle->compute(eflag, vflag);
                atomKK->modified(force->angle->execution_space, ~(~force->angle->datamask_modify | datamask_exclude));
            }
            if (force->dihedral) {
                atomKK->sync(force->dihedral->execution_space, ~(~force->dihedral->datamask_read | datamask_exclude));
                force->dihedral->compute(eflag, vflag);
                atomKK->modified(force->dihedral->execution_space,
                                 ~(~force->dihedral->datamask_modify | datamask_exclude));
            }
            if (force->improper) {
                atomKK->sync(force->improper->execution_space, ~(~force->improper->datamask_read | datamask_exclude));
                force->improper->compute(eflag, vflag);
                atomKK->modified(force->improper->execution_space,
                                 ~(~force->improper->datamask_modify | datamask_exclude));
            }
            timer->stamp(Timer::BOND);
        }

        if (kspace_compute_flag) {
            atomKK->sync(force->kspace->execution_space, ~(~force->kspace->datamask_read | datamask_exclude));
            force->kspace->compute(eflag, vflag);
            atomKK->modified(force->kspace->execution_space, ~(~force->kspace->datamask_modify | datamask_exclude));
            timer->stamp(Timer::KSPACE);
        }

        if (execute_on_host) {
            if (f_merge_copy.extent(0) < atomKK->k_f.extent(0))
                f_merge_copy = DAT::t_f_array("VerletKokkos::f_merge_copy", atomKK->k_f.extent(0));
            f = atomKK->k_f.d_view;
            Kokkos::deep_copy(LMPHostType(), f_merge_copy, atomKK->k_f.h_view);
            Kokkos::parallel_for(atomKK->k_f.extent(0),
                                 ForceAdder<DAT::t_f_array, DAT::t_f_array>(atomKK->k_f.d_view, f_merge_copy));
            atomKK->k_f.clear_sync_state(); // special case
            atomKK->k_f.modify<LMPDeviceType>();
        }

        if (n_pre_reverse) {
            modify->pre_reverse(eflag, vflag);
            timer->stamp(Timer::MODIFY);
        }

        // reverse communication of forces

        if (force->newton) {
            Kokkos::fence();
            auto begin = std::chrono::high_resolution_clock::now();
            comm->reverse_comm();
            auto end = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end-begin).count();
            std::cout << BLUE << "me: " << comm->me << " nprocs: " << comm->nprocs << " REVERSE COMM DURATION: " << duration << RESET_COLOR << std::endl;
            timer->stamp(Timer::COMM);
            lammps_comm_duration += duration;
        }

        // force modifications, final time integration, diagnostics

        if (n_post_force) modify->post_force(vflag);

        if (comm->me == 0) {
            std::cout << BLUE << "me: " << comm->me << " REGULAR MD FINAL INTEGRATE for time: " << i << RESET_COLOR << std::endl;
        }

        modify->final_integrate();

        if (n_end_of_step) modify->end_of_step();
        timer->stamp(Timer::MODIFY);

        // all output

        if (ntimestep == output->next) {
            atomKK->sync(Host, ALL_MASK);

            timer->stamp();
            output->write(ntimestep);
            timer->stamp(Timer::OUTPUT);
        }
    }

    delete[] send_f;
    delete[] send_x;

    atomKK->sync(Host, ALL_MASK);
    lmp->kokkos->auto_sync = 1;

    MPI_Barrier(world);
    std::cout << GREEN << "process: " << comm->me << " LAMMPS COMM DURATION: " << lammps_comm_duration << " microseconds. " << RESET_COLOR << std::endl;
    std::cout << "---------------Start STENCIL MD test run----------------" << std::endl;

    int num_zoids_recv_from = lmp->recv_from_neighbors_procs.size();
    std::thread receive_request_threads[num_zoids_recv_from];

    // map dependency levels to number of zoids to wait on
    std::map<int, std::vector<int>> dep_to_wait_idxs;

    std::set<int> zoids_already_waiting_on;
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
                        if (std::find(recv_from.begin(), recv_from.end(), recv_zoid_num) != recv_from.end()
                            && zoids_already_waiting_on.find(recv_zoid_num) == zoids_already_waiting_on.end()) {
                            dep_to_wait_idxs[dep].push_back(i);
                            zoids_already_waiting_on.insert(recv_zoid_num);
                            dep_recv_zoids.push_back(recv_zoid_num);
                            break;
                        }
                    }
                }
            }
        }
        std::cout << YELLOW << "me: " << comm->me << " dep: " << dep << " recv zoid: " << dep_recv_zoids << RESET_COLOR << std::endl;
    }

    /*
    std::vector<int> zoid_nums;
    for (int dep = 0; dep < NUM_DEPS; dep++) {
        for (int j = 0; j < lmp->queues[dep].size(); j++) {
            zoid_nums.push_back(lmp->queues[dep][j].num);
        }
    }

    for (int dep = 1; dep < NUM_DEPS; dep++) {
        auto &vec = dep_to_wait_idxs[dep];
        std::sort(vec.begin(), vec.end(), [&](const int &idx_a, const int &idx_b) {
            int recv_zoid_a = lmp->recv_from_neighbors_procs[idx_a];
            int recv_zoid_b = lmp->recv_from_neighbors_procs[idx_b];

            int zoid_num_idx_a = std::find(zoid_nums.begin(), zoid_nums.end(), recv_zoid_a) - zoid_nums.begin();
            int zoid_num_idx_b = std::find(zoid_nums.begin(), zoid_nums.end(), recv_zoid_b) - zoid_nums.begin();

            assert(zoid_num_idx_a >= 0 && zoid_num_idx_a < zoid_nums.size());
            assert(zoid_num_idx_b >= 0 && zoid_num_idx_b < zoid_nums.size());

            return zoid_num_idx_a < zoid_num_idx_b;
        });
    }
    */

    MPI_Barrier(world);

    std::vector<MPI_Request> receive_requests(lmp->recv_from_neighbors_procs.size(), MPI_REQUEST_NULL);
    for (int i = 0; i < lmp->recv_from_neighbors_procs.size(); i++) {
        int recv_zoid_num = lmp->recv_from_neighbors_procs[i];
        if (recv_zoid_num % comm->nprocs != comm->me) {
            comm->receive_data_process_stencil_md(&receive_requests[i], recv_zoid_num, false);
            /*
            receive_request_threads[i] =
                    std::move(std::thread([&](int recv_zoid_num_) {
                        MPI_Request r;
                        auto begin_mpi = std::chrono::high_resolution_clock::now();
                        comm->receive_data_process_stencil_md(&r, recv_zoid_num_);
                        int wait_status = MPI_Wait(&r, MPI_STATUS_IGNORE);
                        assert(wait_status == MPI_SUCCESS);
                        auto end_mpi = std::chrono::high_resolution_clock::now();
                        auto duration_mpi = std::chrono::duration_cast<std::chrono::microseconds>(end_mpi-begin_mpi).count();
                        sleep(60);
                    }, recv_zoid_num));
            */
        }
    }

    /*
    int num_threads_recv = 0;
    for (int i = 0; i < lmp->recv_from_neighbors_procs.size(); i++) {
        int recv_zoid_num = lmp->recv_from_neighbors_procs[i];
        if (recv_zoid_num % comm->nprocs != comm->me) {
            num_threads_recv++;
            receive_request_threads[i] =
                    std::move(std::thread([&](int recv_zoid_num_) {
                        MPI_Request r;
                        auto begin_mpi = std::chrono::high_resolution_clock::now();
                        comm->receive_data_process_stencil_md(&r, recv_zoid_num_);
                        int wait_status = MPI_Wait(&r, MPI_STATUS_IGNORE);
                        assert(wait_status == MPI_SUCCESS);
                        auto end_mpi = std::chrono::high_resolution_clock::now();
                        auto duration_mpi = std::chrono::duration_cast<std::chrono::microseconds>(end_mpi-begin_mpi).count();

                    }, recv_zoid_num));
        }
    }
    */

    // start compute

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
            send_requests[zoid_num] = std::vector<MPI_Request>(num_procs, MPI_REQUEST_NULL);
        }
    }

    int64_t send_comm_duration = 0;
    int64_t recv_comm_duration = 0;
    int64_t stencil_md_compute_duration = 0;

    for (int dep = 0; dep < NUM_DEPS; dep++) {
        if (dep > 0) {
            auto begin = std::chrono::high_resolution_clock::now();

            for (int idx : dep_to_wait_idxs[dep]) {
                auto begin_mpi = std::chrono::high_resolution_clock::now();
                int recv_zoid_num = lmp->recv_from_neighbors_procs[idx];

                auto time_before = timeSinceEpochMillisec();

                // receive_request_threads[idx].join();
                MPI_Wait(&receive_requests[idx], MPI_STATUS_IGNORE);
                auto end_mpi = std::chrono::high_resolution_clock::now();
                auto duration_mpi = std::chrono::duration_cast<std::chrono::microseconds>(end_mpi-begin_mpi).count();

                auto time_after = timeSinceEpochMillisec();

                auto begin_pack = std::chrono::high_resolution_clock::now();
                comm->unpack_data_process_stencil_md(recv_zoid_num, false);
                auto end_pack = std::chrono::high_resolution_clock::now();
                auto duration_pack = std::chrono::duration_cast<std::chrono::microseconds>(end_pack-begin_pack).count();

                if (duration_mpi > 100000) {
                    std::cout << YELLOW << "process: " << comm->me << " dep: " << dep << " recv time: " << duration_mpi << " from zoid: " << lmp->recv_from_neighbors_procs[idx] << " unpack time: " << duration_pack
                              << " time before join: " << time_before << " time after join: " << time_after << RESET_COLOR << std::endl;
                }
            }
            auto end = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end-begin).count();
            recv_comm_duration += duration;
        }

        for (int j = 0; j < lmp->queues[dep].size(); j++) {
            int zoid_num = lmp->queues[dep][j].num;
            if (zoid_num % comm->nprocs != comm->me) {
                continue;
            }

            int zoid_pairs_evaled = 0;
            int zoid_atoms_evaled = 0;
            int zoid_atoms_evaled_timesteps[NUM_TIMESTEPS_IN_PARALLEL + 1] = {0};
            int64_t zoid_compute_duration = 0;
            std::vector<int> zoid_compute_duration_vec;
            std::vector<int> zoid_pairs_evaled_vec;
            std::vector<int> zoid_atoms_evaled_vec;

            auto& atom_arr = lmp->atom_stencil_md[zoid_num];

            auto time_before_first_compute = timeSinceEpochMillisec();

            int **atom_idx_mapping = lmp->queues[dep][j].atom_idx_mapping;
            for (int t = 0; t < NUM_TIMESTEPS_IN_PARALLEL; t++) {
                queue_info& zoid = lmp->queues[dep][j];
                Atom *atom_ = lmp->atom_stencil_md[zoid_num][t];
                Atom *atom_next_timestep = lmp->atom_stencil_md[zoid_num][t + 1];
                AtomKokkos *atomKK_ = (AtomKokkos *) atom_;
                Neighbor *neighbor_ = lmp->neighbor_stencil_md[zoid_num][t];
                Force *force_ = lmp->force_stencil_md[zoid_num][t];
                Modify *modify_ = lmp->modify_stencil_md[zoid_num];

                if (TEST_AGAINST_LAMMPS) {
                    for (int k = 0; k < atom_->nlocal; k++) {
                        int tag = atom_->tag[k];
                        double *x_ = atom_->x[k];
                        for (int dim = 0; dim < 3; dim++) {
                            double val = x_[dim];
                            if (val < 0) {
                                val += domain->prd[dim];
                            } else if (val >= domain->prd[dim]) {
                                val -= domain->prd[dim];
                            }

                            double test_val = test_x[t][tag * 3 + dim];
                            if (test_val < 0) {
                                test_val += domain->prd[dim];
                            } else if (test_val >= domain->prd[dim]) {
                                test_val -= domain->prd[dim];
                            }

                            if (fabs(val - test_val) > 1e-6) {
                                std::cout << "-------POS DIFF--------" << std::endl;
                                std::cout << "idx: " << k << " out of: " << atom_->nlocal << std::endl;
                                std::cout << "Dim: " << dim << " Zoid: " << zoid_num << " timestep: " << t << " tag: "
                                          << tag << " different. " << std::endl;
                                std::cout << "What I have: " << x_[0] << " " << x_[1] << " " << x_[2] << std::endl;
                                std::cout << "What does LAMMPS have? " << test_x[t][tag * 3 + 0] << " "
                                          << test_x[t][tag * 3 + 1] << " " << test_x[t][tag * 3 + 2] << std::endl;
                                std::cout << "Diff: " << fabs(x_[dim] - test_x[t][tag * 3 + dim]) << std::endl;
                                std::cout << "pos: " << atom_->x[k][0] << " " << atom_->x[k][1] << " " << atom_->x[k][2]
                                          << std::endl;

                                for (int tmp = 0; tmp < 3; tmp++) {
                                    std::cout << "lo: "
                                              << zoid.zoid.cuts[tmp].lower + zoid.zoid.cuts[tmp].slope_lower * t
                                              << std::endl;
                                    std::cout << "hi: "
                                              << zoid.zoid.cuts[tmp].upper + zoid.zoid.cuts[tmp].slope_upper * t
                                              << std::endl;
                                }
                                assert(false);
                            } else {
                                /*
                                if (x_[dim] < 0) {
                                    x_[dim] = test_val - domain->prd[dim];
                                } else if (x_[dim] >= domain->prd[dim]) {
                                    x_[dim] = test_val + domain->prd[dim];
                                } else {
                                    x_[dim] = test_val;
                                }
                                */
                            }
                        }
                    }

                    for (int k = 0; k < atom_->nlocal; k++) {
                        // compare forces only on evaluatable atoms
                        int tag = atom_->tag[k];
                        if (!zoid.can_eval_pos[t][k]) {
                            continue;
                        }
                        for (int dim = 0; dim < 3; dim++) {
                            double my_force = atom_->f[k][dim] + atom_->eval_f_stencil_md[k][dim];
                            if (fabs(my_force - test_f[t][tag * 3 + dim]) > 1e-6) {
                                std::cout << "------FORCE DIFF--------" << std::endl;
                                std::cout << "idx: " << k << " out of: " << atom_->nlocal << std::endl;
                                std::cout << "Dim: " << dim << " Zoid: " << zoid_num << " timestep: " << t << " tag: "
                                          << tag << " different. " << std::endl;
                                std::cout << "what I have f: " << atom_->f[k][0] << " " << atom_->f[k][1] << " "
                                          << atom_->f[k][2] << std::endl;
                                std::cout << "what I have eval " << atom_->eval_f_stencil_md[k][0] << " "
                                          << atom_->eval_f_stencil_md[k][1] << " " << atom_->eval_f_stencil_md[k][2]
                                          << std::endl;
                                // std::cout << "What I have: " << [0] << " " << f_[1] << " " << f_[2] << std::endl;
                                std::cout << "what I have: " << atom_->f[k][0] + atom_->eval_f_stencil_md[k][0] << " "
                                          <<
                                          atom_->f[k][1] + atom_->eval_f_stencil_md[k][1] << " "
                                          << atom_->f[k][2] + atom_->eval_f_stencil_md[k][2] << std::endl;
                                std::cout << "What does LAMMPS have? " << test_f[t][tag * 3 + 0] << " "
                                          << test_f[t][tag * 3 + 1] << " " << test_f[t][tag * 3 + 2] << std::endl;
                                std::cout << "Diff: " << fabs(my_force - test_f[t][tag * 3 + dim]) << std::endl;
                                std::cout << "pos: " << atom_->x[k][0] << " " << atom_->x[k][1] << " " << atom_->x[k][2]
                                          << std::endl;

                                for (int tmp = 0; tmp < 3; tmp++) {
                                    std::cout << "lo: "
                                              << zoid.zoid.cuts[tmp].lower + zoid.zoid.cuts[tmp].slope_lower * t
                                              << std::endl;
                                    std::cout << "hi: "
                                              << zoid.zoid.cuts[tmp].upper + zoid.zoid.cuts[tmp].slope_upper * t
                                              << std::endl;
                                }

                                assert(false);
                            }
                        }
                    }
                }

                // updates positions in atom_next_timestep
                modify_->initial_integrate_stencil_md(vflag, atom_, atom_next_timestep, atom_idx_mapping[t], zoid.can_eval_pos[t]);

                if (TEST_AGAINST_LAMMPS) {
                    for (int k = 0; k < atom_next_timestep->nlocal; k++) {
                        int tag = atom_next_timestep->tag[k];
                        double *x_ = atom_next_timestep->x[k];
                        for (int dim = 0; dim < 3; dim++) {
                            double val = x_[dim];
                            if (val < 0) {
                                val += domain->prd[dim];
                            } else if (val >= domain->prd[dim]) {
                                val -= domain->prd[dim];
                            }

                            double test_val = test_x[t + 1][tag * 3 + dim];
                            if (test_val < 0) {
                                test_val += domain->prd[dim];
                            } else if (test_val >= domain->prd[dim]) {
                                test_val -= domain->prd[dim];
                            }

                            if (fabs(val - test_val) > 1e-6) {
                                std::cout << "-------POS DIFF NEXT--------" << std::endl;
                                std::cout << "idx: " << k << " out of: " << atom_next_timestep->nlocal << std::endl;
                                std::cout << "Dim: " << dim << " Zoid: " << zoid_num << " timestep: " << t + 1
                                          << " tag: " << tag << " different. " << std::endl;
                                std::cout << "What I have: " << x_[0] << " " << x_[1] << " " << x_[2] << std::endl;
                                std::cout << "What does LAMMPS have? " << test_x[t + 1][tag * 3 + 0] << " "
                                          << test_x[t + 1][tag * 3 + 1] << " " << test_x[t + 1][tag * 3 + 2]
                                          << std::endl;
                                std::cout << "Diff: " << fabs(val - test_val) << std::endl;
                                std::cout << "me val: " << val << " test_val: " << test_val << std::endl;
                                std::cout << "pos: " << atom_next_timestep->x[k][0] << " "
                                          << atom_next_timestep->x[k][1] << " " << atom_next_timestep->x[k][2]
                                          << std::endl;
                                std::cout << "can eval center? " << zoid.can_eval_center[t + 1][k]
                                          << " relevant? " << (zoid.relevant_atom_idxs[t + 1].find(k) != zoid.relevant_atom_idxs[t + 1].end()) << std::endl;
                                if (atom_->tag_to_idx.count(atom_next_timestep->tag[k])) {
                                    int prev_idx = atom_->tag_to_idx[atom_next_timestep->tag[k]];
                                    std::cout << "prev nlocal: " << atom_->nlocal << " prev idx: " << prev_idx << " prev can eval center? " << zoid.can_eval_center[t][prev_idx]
                                              << " relevant? " << (zoid.relevant_atom_idxs[t].find(prev_idx) != zoid.relevant_atom_idxs[t].end()) << std::endl;
                                } else {
                                    std::cout << "could not find prev tag" << std::endl;
                                }

                                for (int tmp = 0; tmp < 3; tmp++) {
                                    std::cout << "lo: "
                                              << zoid.zoid.cuts[tmp].lower + zoid.zoid.cuts[tmp].slope_lower * (t + 1)
                                              << std::endl;
                                    std::cout << "hi: "
                                              << zoid.zoid.cuts[tmp].upper + zoid.zoid.cuts[tmp].slope_upper * (t + 1)
                                              << std::endl;
                                }
                                assert(false);
                            } else {
                                //                            if (x_[dim] < 0) {
                                //                                x_[dim] = test_val - domain->prd[dim];
                                //                            } else if (x_[dim] >= domain->prd[dim]) {
                                //                                x_[dim] = test_val + domain->prd[dim];
                                //                            } else {
                                //                                x_[dim] = test_val;
                                //                            }
                            }
                        }
                    }
                }

                if (n_pre_force) {
                    assert(false);
                    modify->pre_force(vflag);
                    timer->stamp(Timer::MODIFY);
                }

                bool execute_on_host = false;
                unsigned int datamask_read_host = 0;
                unsigned int datamask_exclude = 0;
                int allow_overlap = lmp->kokkos->allow_overlap;

                if (allow_overlap && atomKK_->k_f.h_view.data() != atomKK_->k_f.d_view.data()) {

                    datamask_exclude = (F_MASK | ENERGY_MASK | VIRIAL_MASK);

                    if (pair_compute_flag) {
                        if (force->pair->execution_space == Host) {
                            execute_on_host = true;
                            datamask_read_host |= force->pair->datamask_read;
                        }
                    }
                    if (atomKK->molecular && force->bond) {
                        if (force->bond->execution_space == Host) {
                            execute_on_host = true;
                            datamask_read_host |= force->bond->datamask_read;
                        }
                    }
                    if (atomKK->molecular && force->angle) {
                        if (force->angle->execution_space == Host) {
                            execute_on_host = true;
                            datamask_read_host |= force->angle->datamask_read;
                        }
                    }
                    if (atomKK->molecular && force->dihedral) {
                        if (force->dihedral->execution_space == Host) {
                            execute_on_host = true;
                            datamask_read_host |= force->dihedral->datamask_read;
                        }
                    }
                    if (atomKK->molecular && force->improper) {
                        if (force->improper->execution_space == Host) {
                            execute_on_host = true;
                            datamask_read_host |= force->improper->datamask_read;
                        }
                    }
                    if (kspace_compute_flag) {
                        if (force->kspace->execution_space == Host) {
                            execute_on_host = true;
                            datamask_read_host |= force->kspace->datamask_read;
                        }
                    }
                }

                if (pair_compute_flag) {
                    atomKK_->sync_stencil_md(force->pair->execution_space, force->pair->datamask_read, atom_);
                    atomKK_->sync_stencil_md(force->pair->execution_space,
                                             ~(~force->pair->datamask_read | datamask_exclude), atom_);
                    Force* next_force = lmp->force_stencil_md[zoid_num][t + 1];
                    int * atom_idx_mapping_ = lmp->queues[dep][j].atom_idx_mapping[t + 1];
                    // force_clear_stencil_md(atom_next_timestep, next_force, neighbor_);
                    auto begin = std::chrono::high_resolution_clock::now();
                    int zoid_pairs_evaled_tmp = 0;
                    next_force->pair->compute_stencil_md(eflag, vflag, atom_next_timestep,
                                                         zoid.can_eval_center[t + 1], lmp->zoid_num_to_zoid[zoid_num], &zoid_pairs_evaled_tmp);
                    auto end = std::chrono::high_resolution_clock::now();
                    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end-begin).count();
                    zoid_compute_duration += duration;
                    zoid_compute_duration_vec.push_back(duration);
                    zoid_pairs_evaled_vec.push_back(zoid_pairs_evaled_tmp);
                    /*
                    std::cout << YELLOW << "zoid: " << zoid_num << " time: " << t + 1 << " before compute: " << time_before_compute << " time after compute: " << time_after_compute
                        << " duration: " << time_after_compute - time_before_compute << RESET_COLOR << std::endl;
                    */
                    stencil_md_compute_duration += duration;
                    atomKK_->modified_stencil_md(force->pair->execution_space, force->pair->datamask_modify, atom_);
                    atomKK_->modified_stencil_md(force->pair->execution_space,
                                                 ~(~force->pair->datamask_modify | datamask_exclude), atom_);
                    timer->stamp(Timer::PAIR);

                    int num_eval = 0;
                    for (int k = 0; k < atom_next_timestep->nlocal + atom_next_timestep->nghost; k++) {
                        if (zoid.can_eval_center[t + 1][k]) {
                            zoid_atoms_evaled++;
                            zoid_atoms_evaled_timesteps[t + 1]++;
                            num_eval++;
                        }
                    }

                    zoid_atoms_evaled_vec.push_back(num_eval);
                }

                if (execute_on_host) {
                    assert(false);
                    std::cout << "Execute on host" << std::endl;
                    if (pair_compute_flag && force->pair->datamask_modify != datamask_exclude)
                        Kokkos::fence();
                    atomKK_->sync_overlapping_device(Host, ~(~datamask_read_host | datamask_exclude));
                    if (pair_compute_flag && force->pair->execution_space != Host) {
                        Kokkos::deep_copy(LMPHostType(), atomKK->k_f.h_view, 0.0);
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

                if (n_post_force) {
                    modify->post_force(vflag);
                }

                if (comm->me == 0) {
//                    std::cout << "STENCIL MD FINAL INTEGRATE for t: " << time_idx << " and next t: " << next_time_idx
//                              << std::endl;
                }

                // std::cout << "-------- ZOID NUM: " << zoid.num << " FINAL INTEGRATE WRITING INTO: ---------" << t + 1 << std::endl;

                modify_->final_integrate_stencil_md(atom_, atom_next_timestep, neighbor_, atom_idx_mapping[t], zoid.can_eval_pos[t + 1]);

                if (n_end_of_step) {
                    assert(false);
                    modify->end_of_step();
                }
                timer->stamp(Timer::MODIFY);

                // all output

                if (ntimestep == output->next) {
                    assert(false);
                    atomKK_->sync(Host, ALL_MASK);
                    timer->stamp();
                    output->write(ntimestep);
                    timer->stamp(Timer::OUTPUT);
                }
            }

            if (dep < NUM_DEPS - 1) {
                auto begin = std::chrono::high_resolution_clock::now();
                queue_info& zoid = lmp->zoid_num_to_zoid[zoid_num];
                auto &atom_arr = lmp->atom_stencil_md[zoid_num];
                Comm *comm_ = lmp->comm_stencil_md[zoid_num];
                // comm_->send_data_stencil_md(atom_arr, lmp->zoid_num_to_zoid[zoid_num], send_requests[zoid_num]);
                int vec_idx = 0;
                for (int proc = 0; proc < comm->nprocs; proc++) {
                    auto begin_send = std::chrono::high_resolution_clock::now();
                    auto before_send_long = timeSinceEpochMillisec();
                    bool sent = comm_->send_data_to_process_stencil_md(atom_arr, lmp->zoid_num_to_zoid[zoid_num], &send_requests[zoid_num][vec_idx], proc, false);
                    auto end_send = std::chrono::high_resolution_clock::now();
                    auto duration_send = std::chrono::duration_cast<std::chrono::microseconds>(end_send-begin_send).count();
                    auto after_send_long = timeSinceEpochMillisec();
                    if (sent) {
                        auto begin_thread = std::chrono::high_resolution_clock::now();
                        send_request_threads.push_back(
                                std::move(std::thread([&](int idx, int zoid_num_) {
                                              auto time_before = timeSinceEpochMillisec();
                                              int wait_status = MPI_Wait(&send_requests[zoid_num_][idx], MPI_STATUS_IGNORE);
                                              assert(wait_status == MPI_SUCCESS);
                                              auto time_t = timeSinceEpochMillisec();
                                          }, vec_idx, zoid_num)
                                ));
                        vec_idx++;
                        auto end_thread = std::chrono::high_resolution_clock::now();
                        auto duration_thread = std::chrono::duration_cast<std::chrono::microseconds>(end_thread - begin_thread).count();
                        int64_t total_compute_duration = 0;
                        for (auto& d : zoid_compute_duration_vec) {
                            total_compute_duration += d;
                        }

                        /*
                        std::cout << MAGENTA << "zoid num: " << zoid_num << " send to proc: " << proc << " duration: " << duration_send
                                  << " before send ms: " << before_send_long << " after send long: " << after_send_long
                                  << " compute duration: " << zoid_compute_duration / 1000 << " ms "
                                  << " atoms eval'ed: " << zoid_atoms_evaled_vec << " pairs eval'ed vec: " << zoid_pairs_evaled_vec << " time before compute: " << time_before_first_compute
                                  << " compute duration vec: " << zoid_compute_duration_vec << " total duration: " << total_compute_duration
                                  << " time thread: " << duration_thread << RESET_COLOR << std::endl;
                        */
                    }
                }

                auto end = std::chrono::high_resolution_clock::now();
                auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end-begin).count();
                send_comm_duration += duration;
            }

            /*
            auto& send_to = lmp->send_to_neighbors[zoid_num];
            queue_info& zoid = lmp->queues[dep][j];
            for (int i = 0; i < send_to.size(); i++) {
                if (send_to[i] != 22) {
                    continue;
                }
                for (int t = 0; t < NUM_TIMESTEPS_IN_PARALLEL + 1; t++) {
                    Atom *atom_ = lmp->atom_stencil_md[zoid_num][t];
                    int num_segments = zoid.send_force_num_segments[t][i];
                    for (int k = 0; k < num_segments; k++) {
                        int segment_idx = zoid.send_force_idxs[t][i][k];
                        int segment_size = zoid.send_force_sizes[t][i][k];
                        for (int h = 0; h < segment_size; h++) {
                            int idx = segment_idx + h;
                            if (fabs(atom_->eval_f_stencil_md[idx][0]) <= 1e-6) {
                                std::cout << CYAN << "SEND EMPTY FORCE zoid num: " << zoid.num << " send to: " << send_to[i] << " time: " << t
                                          << " tag: " << atom_->tag[idx] << " can eval? " << zoid.can_eval_center[t][idx]
                                          << " pos: " << atom_->x[idx][0] << " " << atom_->x[idx][1] << " " << atom_->x[idx][2]
                                          << " force: " << atom_->eval_f_stencil_md[idx][0] << " " << atom_->eval_f_stencil_md[idx][1] << " " << atom_->eval_f_stencil_md[idx][2] << RESET_COLOR << std::endl;
                            }
                        }
                    }

                    int num_pos_segments = zoid.send_pos_num_segments[t][i];
                    for (int k = 0; k < num_pos_segments; k++) {
                        int segment_idx = zoid.send_pos_idxs[t][i][k];
                        int segment_size = zoid.send_pos_sizes[t][i][k];
                        for (int h = 0; h < segment_size; h++) {
                            int idx = segment_idx + h;
                            std::cout << YELLOW << "SEND POS zoid num: " << zoid.num << " send to: " << send_to[i] << " time: " << t
                                      << " tag: " << atom_->tag[idx] << " can eval? " << zoid.can_eval_center[t][idx]
                                      << " pos: " << atom_->x[idx][0] << " " << atom_->x[idx][1] << " " << atom_->x[idx][2] << RESET_COLOR << std::endl;
                        }
                    }
                }
            }
            */
        }
    }

    for (auto& t : send_request_threads) {
        t.join();
    }

    // int wait_all_status = MPI_Waitall(send_requests.size(), send_requests.data(), MPI_STATUSES_IGNORE);
    // assert(wait_all_status == MPI_SUCCESS);
    std::cout << GREEN << "--------------- STENCIL MD INITIAL DT PASSED -------------------" << RESET_COLOR << std::endl;
    std::cout << BLUE << "recv comm total duration: " << recv_comm_duration << " microseconds. send comm duration: " << send_comm_duration << " microseconds " << RESET_COLOR << std::endl;
    std::cout << YELLOW << "lammps compute duration: " << lammps_compute_duration << " microseconds. " << " stencil md compute duration: " << stencil_md_compute_duration << " microseconds. " << RESET_COLOR << std::endl;

    MPI_Barrier(world);

    // clear force on everything except last timestep of initial,
    for (int i = 0; i < NUM_ZOIDS; i++) {
        if (i % comm->nprocs == comm->me) {
            for (int t = 0; t < lmp->atom_stencil_md[i].size() - 1; t++) {
                Atom* atom_ = lmp->atom_stencil_md[i][t];
                for (int j = 0; j < atom_->nlocal + atom_->nghost; j++) {
                    for (int dim = 0; dim < 3; dim++) {
                        atom_->f[j][dim] = 0;
                        atom_->eval_f_stencil_md[j][dim] = 0;
                    }
                }
            }
        }
    }

    std::cout << BLUE << "--------- START STENCIL MD NEXT DT -------------- " << RESET_COLOR << std::endl;

    int num_zoids_recv_from_next_dt = lmp->recv_from_neighbors_procs_next_dt.size();
    std::thread receive_request_threads_next_dt[num_zoids_recv_from_next_dt];
    std::vector<MPI_Request> receive_requests_next_dt(num_zoids_recv_from_next_dt, MPI_REQUEST_NULL);

    // map dependency levels to number of zoids to wait on
    std::map<int, std::vector<int>> dep_to_wait_idxs_next_dt;

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
                        auto& recv_from = lmp->recv_from_neighbors_next_dt[zoid_num];
                        if (std::find(recv_from.begin(), recv_from.end(), recv_zoid_num) != recv_from.end()
                            && zoids_already_waiting_on_next_dt.find(recv_zoid_num) == zoids_already_waiting_on_next_dt.end()) {
                            dep_to_wait_idxs_next_dt[dep].push_back(i);
                            zoids_already_waiting_on_next_dt.insert(recv_zoid_num);
                            dep_recv_zoids.push_back(recv_zoid_num);
                            break;
                        }
                    }
                }
            }
        }
        // std::cout << YELLOW << "NEXT DT me: " << comm->me << " dep: " << dep << " recv zoid: " << dep_recv_zoids << RESET_COLOR << std::endl;
    }

    /*

    for (int dep = 1; dep < NUM_DEPS; dep++) {
        for (int i = 0; i < lmp->recv_from_neighbors_procs_next_dt.size(); i++) {
            int recv_zoid_num = lmp->recv_from_neighbors_procs_next_dt[i];
            if (recv_zoid_num % comm->nprocs != comm->me
                    && get_zoid_dep_next_dt(recv_zoid_num) == dep - 1) {
                dep_to_wait_idxs_next_dt[dep].push_back(i);
            }
        }

        auto& v = dep_to_wait_idxs_next_dt[dep];
        for (int idx : v) {
            std::cout << "me: " << comm->me << " next dt dep: " << dep << " wait on zoid: " << lmp->recv_from_neighbors_procs_next_dt[idx] << std::endl;
        }
    }
    */

    int num_to_wait_on_next_dt = 0;
    for (auto& [k, v] : dep_to_wait_idxs_next_dt) {
        num_to_wait_on_next_dt += v.size();
    }

    int num_zoids_not_mine_next_dt = 0;
    for (int zoid : lmp->recv_from_neighbors_procs_next_dt) {
        if (zoid % comm->nprocs != comm->me) {
            num_zoids_not_mine_next_dt++;
        }
    }

    assert(num_zoids_not_mine_next_dt == num_to_wait_on_next_dt);

    // std::cout << CYAN << "ME: " << comm->me << " RECV ZOIDS: " << lmp->recv_from_neighbors_procs_next_dt << RESET_COLOR << std::endl;

    /*
    int num_threads = 0;
    for (int i = 0; i < lmp->recv_from_neighbors_procs_next_dt.size(); i++) {
        int recv_zoid_num = lmp->recv_from_neighbors_procs_next_dt[i];
        if (recv_zoid_num % comm->nprocs != comm->me) {
            num_threads++;
            receive_request_threads_next_dt[i] =
                    std::move(std::thread([&](int idx, int recv_zoid_num_) {
                        MPI_Request r;
                        comm->receive_data_process_stencil_md_next_dt(&r, recv_zoid_num_);
                        int wait_status = MPI_Wait(&r, MPI_STATUS_IGNORE);
                        assert(wait_status == MPI_SUCCESS);
                    }, i, recv_zoid_num));
        }
    }

    assert(num_threads == num_to_wait_on_next_dt);
    */

    for (int i = 0; i < lmp->recv_from_neighbors_procs_next_dt.size(); i++) {
        int recv_zoid_num = lmp->recv_from_neighbors_procs_next_dt[i];
        if (recv_zoid_num % comm->nprocs != comm->me) {
            comm->receive_data_process_stencil_md_next_dt(&receive_requests_next_dt[i], recv_zoid_num);
        }
    }

    std::vector<MPI_Request> send_requests_next_dt[NUM_ZOIDS];
    std::vector<std::thread> send_request_threads_next_dt;

    for (int zoid_num = 0; zoid_num < NUM_ZOIDS; zoid_num++) {
        if (zoid_num % comm->nprocs == comm->me) {
            int num_procs = 0;
            for (int proc = 0; proc < comm->nprocs; proc++) {
                if (proc == comm->me) {
                    continue;
                }
                for (int i = 0; i < lmp->send_to_neighbors_next_dt[zoid_num].size(); i++) {
                    if (lmp->send_to_neighbors_next_dt[zoid_num][i] % comm->nprocs == proc) {
                        num_procs++;
                        break;
                    }
                }
            }
            assert(num_procs >= 0 && num_procs < comm->nprocs);
            send_requests_next_dt[zoid_num] = std::vector<MPI_Request>(num_procs, MPI_REQUEST_NULL);
        }
    }

    // start compute
    for (int dep = 0; dep < NUM_DEPS; dep++) {
        if (dep > 0) {
            auto begin = std::chrono::high_resolution_clock::now();
            for (int idx : dep_to_wait_idxs_next_dt[dep]) {
                int recv_zoid_num = lmp->recv_from_neighbors_procs_next_dt[idx];
                // receive_request_threads_next_dt[idx].join();
                MPI_Wait(&receive_requests_next_dt[idx], MPI_STATUS_IGNORE);
                comm->unpack_data_process_stencil_md_next_dt(recv_zoid_num);
            }
            auto end = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end-begin).count();
            if (duration > 100000) {
                std::cout << YELLOW << "NEXT DT process: " << comm->me << " dep: " << dep << " recv time: " << duration << RESET_COLOR << std::endl;
            }
            recv_comm_duration += duration;
        }

        for (int j = 0; j < lmp->queues_next_dt[dep].size(); j++) {
            queue_info& zoid = lmp->queues_next_dt[dep][j];
            int zoid_num = lmp->queues_next_dt[dep][j].num;
            if (zoid_num % comm->nprocs != comm->me) {
                continue;
            }

            auto &atom_arr = lmp->atom_stencil_md[zoid_num];

            /*
            if (dep > 0) {
                comm_->receive_data_stencil_md_next_dt(atom_arr, lmp->zoid_num_to_zoid_next_dt[zoid_num]);
                // comm_->receive_exclude_eval_tags_next_dt(atom_arr, lmp->zoid_num_to_zoid_next_dt[zoid_num]);
            }
            */

            int **atom_idx_mapping = lmp->queues_next_dt[dep][j].atom_idx_mapping;
            for (int t = 0; t < NUM_TIMESTEPS_IN_PARALLEL; t++) {
                int idx_obj_timestep = NUM_TIMESTEPS_IN_PARALLEL - t;
                int idx_obj_timestep_next = NUM_TIMESTEPS_IN_PARALLEL - (t + 1);

                Atom *atom_ = atom_arr[idx_obj_timestep];
                Atom *atom_next_timestep = atom_arr[idx_obj_timestep_next];

                AtomKokkos* atomKK_ = (AtomKokkos*) atom_;
                Neighbor *neighbor_ = lmp->neighbor_stencil_md[zoid_num][idx_obj_timestep];
                Force *force_ = lmp->force_stencil_md[zoid_num][idx_obj_timestep];
                Modify *modify_ = lmp->modify_stencil_md[zoid_num];

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

                        int timestep_to_compare = NUM_TIMESTEPS_IN_PARALLEL + t;

                        double test_val = test_x[timestep_to_compare][tag * 3 + dim];
                        if (test_val < 0) {
                            test_val += domain->prd[dim];
                        } else if (test_val >= domain->prd[dim]) {
                            test_val -= domain->prd[dim];
                        }

                        if (fabs(val - test_val) > 1e-6) {
                            std::cout << "-------POS DIFF--------" << std::endl;
                            std::cout << "idx: " << k << " out of: " << atom_->nlocal << std::endl;
                            std::cout << "Dim: " << dim << " Zoid: " << zoid_num << " timestep: " << t << " tag: " << tag << " different. " << std::endl;
                            std::cout << "What I have: " << x_[0] << " " << x_[1] << " " << x_[2] << std::endl;
                            std::cout << "What does LAMMPS have? " << test_x[timestep_to_compare][tag * 3 + 0] << " "
                                << test_x[timestep_to_compare][tag * 3 + 1] << " " << test_x[timestep_to_compare][tag * 3 + 2] << std::endl;
                            std::cout << "Diff: " << fabs(x_[dim] - test_x[timestep_to_compare][tag * 3 + dim]) << std::endl;
                            std::cout << "pos: " << atom_->x[k][0] << " " << atom_->x[k][1] << " " << atom_->x[k][2] << std::endl;
                            std::cout << "can eval center? " << zoid.can_eval_center[t][k] << " relevant? " << (zoid.relevant_atom_idxs[t].find(k) != zoid.relevant_atom_idxs[t].end()) << std::endl;

                            for (int tmp = 0; tmp < 3; tmp++) {
                                std::cout << "lo: " << zoid.zoid.cuts[tmp].lower + zoid.zoid.cuts[tmp].slope_lower * t << std::endl;
                                std::cout << "hi: " << zoid.zoid.cuts[tmp].upper + zoid.zoid.cuts[tmp].slope_upper * t << std::endl;
                                std::cout << "slope lower: " << zoid.zoid.cuts[tmp].slope_lower << " slope upper: " << zoid.zoid.cuts[tmp].slope_upper << std::endl;
                            }
                            assert(false);
                        } else {
                            /*
                            if (x_[dim] < 0) {
                                x_[dim] = test_val - domain->prd[dim];
                            } else if (x_[dim] >= domain->prd[dim]) {
                                x_[dim] = test_val + domain->prd[dim];
                            } else {
                                x_[dim] = test_val;
                            }
                            */
                        }
                    }
                }

                for (int k = 0; k < atom_->nlocal; k++) {
                    // compare forces only on evaluatable atoms
                    int tag = atom_->tag[k];
                    if (!zoid.can_eval_pos[t][k]) {
                        continue;
                    }
                    for (int dim = 0; dim < 3; dim++) {
                        double my_force = atom_->f[k][dim] + atom_->eval_f_stencil_md[k][dim];
                        int timestep_to_compare = NUM_TIMESTEPS_IN_PARALLEL + t;
                        if (fabs(my_force - test_f[timestep_to_compare][tag * 3 + dim]) > 1e-6) {
                            std::cout << "my idx: " << idx_obj_timestep << std::endl;
                            std::cout << "------FORCE DIFF--------" << std::endl;
                            std::cout << "idx: " << k << " out of: " << atom_->nlocal << std::endl;
                            std::cout << "Dim: " << dim << " Zoid: " << zoid_num << " timestep: " << t << " tag: " << tag << " different. " << std::endl;
                            std::cout << "what I have f: " << atom_->f[k][0] << " " << atom_->f[k][1] << " " << atom_->f[k][2] << std::endl;
                            std::cout << "what I have eval " << atom_->eval_f_stencil_md[k][0] << " " << atom_->eval_f_stencil_md[k][1] << " " << atom_->eval_f_stencil_md[k][2] << std::endl;
                            // std::cout << "What I have: " << [0] << " " << f_[1] << " " << f_[2] << std::endl;
                            std::cout << "what I have: " << atom_->f[k][0] + atom_->eval_f_stencil_md[k][0] << " " <<
                                      atom_->f[k][1] + atom_->eval_f_stencil_md[k][1] << " " << atom_->f[k][2] + atom_->eval_f_stencil_md[k][2] << std::endl;
                            std::cout << "What does LAMMPS have? "
                                << test_f[timestep_to_compare][tag * 3 + 0] << " " << test_f[timestep_to_compare][tag * 3 + 1] << " "
                                << test_f[timestep_to_compare][tag * 3 + 2] << std::endl;
                            std::cout << "Diff: " << fabs(my_force - test_f[timestep_to_compare][tag * 3 + dim]) << std::endl;
                            std::cout << "pos: " << atom_->x[k][0] << " " << atom_->x[k][1] << " " << atom_->x[k][2] << std::endl;

                            for (int tmp = 0; tmp < 3; tmp++) {
                                std::cout << "lo: " << zoid.zoid.cuts[tmp].lower + zoid.zoid.cuts[tmp].slope_lower * t << std::endl;
                                std::cout << "hi: " << zoid.zoid.cuts[tmp].upper + zoid.zoid.cuts[tmp].slope_upper * t << std::endl;
                                std::cout << "slope lower: " << zoid.zoid.cuts[tmp].slope_lower << " slope upper: " << zoid.zoid.cuts[tmp].slope_upper << std::endl;
                            }

                            assert(false);
                        }
                    }
                }

                // updates positions in atom_next_timestep
                if (comm->me == 0) {
                    // std::cout << "-------- ZOID NUM: " << zoid.num << " NEXT DT INITIAL INTEGRATE WRITING INTO: ---------" << t + 1 << std::endl;
                }

                modify_->initial_integrate_stencil_md(vflag, atom_, atom_next_timestep, atom_idx_mapping[t], zoid.can_eval_pos[t]);

                if (TEST_AGAINST_LAMMPS) {
                    for (int k = 0; k < atom_next_timestep->nlocal; k++) {
                        int tag = atom_next_timestep->tag[k];
                        double *x_ = atom_next_timestep->x[k];
                        for (int dim = 0; dim < 3; dim++) {
                            double val = x_[dim];
                            if (val < 0) {
                                val += domain->prd[dim];
                            } else if (val >= domain->prd[dim]) {
                                val -= domain->prd[dim];
                            }

                            int timestep_to_compare = NUM_TIMESTEPS_IN_PARALLEL + t + 1;
                            double test_val = test_x[timestep_to_compare][tag * 3 + dim];
                            if (test_val < 0) {
                                test_val += domain->prd[dim];
                            } else if (test_val >= domain->prd[dim]) {
                                test_val -= domain->prd[dim];
                            }

                            if (fabs(val - test_val) > 1e-6) {
                                std::cout << "-------POS DIFF NEXT--------" << std::endl;
                                std::cout << "idx: " << k << " out of: " << atom_next_timestep->nlocal << std::endl;
                                std::cout << "Dim: " << dim << " Zoid: " << zoid_num << " timestep: " << t + 1
                                          << " tag: " << tag << " different. " << std::endl;
                                std::cout << "What I have: " << x_[0] << " " << x_[1] << " " << x_[2] << std::endl;
                                std::cout << "What does LAMMPS have? "
                                          << test_x[timestep_to_compare][tag * 3 + 0] << " "
                                          << test_x[timestep_to_compare][tag * 3 + 1] << " "
                                          << test_x[timestep_to_compare][tag * 3 + 2] << std::endl;
                                std::cout << "Diff: " << fabs(val - test_val) << std::endl;
                                std::cout << "me val: " << val << " test_val: " << test_val << std::endl;
                                std::cout << "pos: " << atom_next_timestep->x[k][0] << " "
                                          << atom_next_timestep->x[k][1] << " " << atom_next_timestep->x[k][2]
                                          << std::endl;
                                std::cout << "can eval center? " << zoid.can_eval_center[t + 1][k]
                                    << " relevant? " << (zoid.relevant_atom_idxs[t + 1].find(k) != zoid.relevant_atom_idxs[t + 1].end()) << std::endl;
                                if (atom_->tag_to_idx.count(atom_next_timestep->tag[k])) {
                                    int prev_idx = atom_->tag_to_idx[atom_next_timestep->tag[k]];
                                    std::cout << "prev can eval center? " << zoid.can_eval_center[t][prev_idx]
                                        << " relevant? " << (zoid.relevant_atom_idxs[t].find(prev_idx) != zoid.relevant_atom_idxs[t].end()) << std::endl;
                                } else {
                                    std::cout << "could not find prev tag" << std::endl;
                                }

                                for (int tmp = 0; tmp < 3; tmp++) {
                                    std::cout << "lo: "
                                              << zoid.zoid.cuts[tmp].lower + zoid.zoid.cuts[tmp].slope_lower * (t + 1)
                                              << std::endl;
                                    std::cout << "hi: "
                                              << zoid.zoid.cuts[tmp].upper + zoid.zoid.cuts[tmp].slope_upper * (t + 1)
                                              << std::endl;
                                }
                                assert(false);
                            } else {
                                //                            if (x_[dim] < 0) {
                                //                                x_[dim] = test_val - domain->prd[dim];
                                //                            } else if (x_[dim] >= domain->prd[dim]) {
                                //                                x_[dim] = test_val + domain->prd[dim];
                                //                            } else {
                                //                                x_[dim] = test_val;
                                //                            }
                            }
                        }
                    }
                }


                if (n_pre_force) {
                    assert(false);
                    modify->pre_force(vflag);
                    timer->stamp(Timer::MODIFY);
                }

                bool execute_on_host = false;
                unsigned int datamask_read_host = 0;
                unsigned int datamask_exclude = 0;
                int allow_overlap = lmp->kokkos->allow_overlap;

                if (allow_overlap && atomKK_->k_f.h_view.data() != atomKK_->k_f.d_view.data()) {

                    datamask_exclude = (F_MASK | ENERGY_MASK | VIRIAL_MASK);

                    if (pair_compute_flag) {
                        if (force->pair->execution_space == Host) {
                            execute_on_host = true;
                            datamask_read_host |= force->pair->datamask_read;
                        }
                    }
                    if (atomKK->molecular && force->bond) {
                        if (force->bond->execution_space == Host) {
                            execute_on_host = true;
                            datamask_read_host |= force->bond->datamask_read;
                        }
                    }
                    if (atomKK->molecular && force->angle) {
                        if (force->angle->execution_space == Host) {
                            execute_on_host = true;
                            datamask_read_host |= force->angle->datamask_read;
                        }
                    }
                    if (atomKK->molecular && force->dihedral) {
                        if (force->dihedral->execution_space == Host) {
                            execute_on_host = true;
                            datamask_read_host |= force->dihedral->datamask_read;
                        }
                    }
                    if (atomKK->molecular && force->improper) {
                        if (force->improper->execution_space == Host) {
                            execute_on_host = true;
                            datamask_read_host |= force->improper->datamask_read;
                        }
                    }
                    if (kspace_compute_flag) {
                        if (force->kspace->execution_space == Host) {
                            execute_on_host = true;
                            datamask_read_host |= force->kspace->datamask_read;
                        }
                    }
                }

                if (pair_compute_flag) {
                    atomKK_->sync_stencil_md(force->pair->execution_space, force->pair->datamask_read, atom_);
                    atomKK_->sync_stencil_md(force->pair->execution_space,
                                             ~(~force->pair->datamask_read | datamask_exclude), atom_);
                    Atom* next_next_timestep = NULL;
                    Force* next_force = lmp->force_stencil_md[zoid_num][idx_obj_timestep_next];
                    int * atom_idx_mapping_ = lmp->queues_next_dt[dep][j].atom_idx_mapping[t + 1];
                    // force_clear_stencil_md(atom_next_timestep, next_force, neighbor_);
                    next_force->pair->compute_stencil_md(eflag, vflag, atom_next_timestep,
                                                         zoid.can_eval_center[t + 1], lmp->zoid_num_to_zoid_next_dt[zoid_num], NULL);
                    atomKK_->modified_stencil_md(force->pair->execution_space, force->pair->datamask_modify, atom_);
                    atomKK_->modified_stencil_md(force->pair->execution_space,
                                                 ~(~force->pair->datamask_modify | datamask_exclude), atom_);
                    timer->stamp(Timer::PAIR);
                }

                if (execute_on_host) {
                    assert(false);
                    std::cout << "Execute on host" << std::endl;
                    if (pair_compute_flag && force->pair->datamask_modify != datamask_exclude)
                        Kokkos::fence();
                    atomKK_->sync_overlapping_device(Host, ~(~datamask_read_host | datamask_exclude));
                    if (pair_compute_flag && force->pair->execution_space != Host) {
                        Kokkos::deep_copy(LMPHostType(), atomKK->k_f.h_view, 0.0);
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

                if (n_post_force) {
                    modify->post_force(vflag);
                }

                if (comm->me == 0) {
                    // std::cout << "-------- NEXT DT ZOID NUM: " << zoid.num << " FINAL INTEGRATE WRITING INTO: ---------" << t + 1 << std::endl;
                }
                modify_->final_integrate_stencil_md(atom_, atom_next_timestep, neighbor_, atom_idx_mapping[t], zoid.can_eval_pos[t + 1]);

                if (n_end_of_step) {
                    assert(false);
                    modify->end_of_step();
                }
                timer->stamp(Timer::MODIFY);

                // all output

                if (ntimestep == output->next) {
                    assert(false);
                    atomKK_->sync(Host, ALL_MASK);
                    timer->stamp();
                    output->write(ntimestep);
                    timer->stamp(Timer::OUTPUT);
                }
            }

            // send data
            if (dep < NUM_DEPS - 1) {
                Comm *comm_ = lmp->comm_stencil_md[zoid_num];
                comm_->send_data_to_process_stencil_md_next_dt(atom_arr, lmp->zoid_num_to_zoid_next_dt[zoid_num], send_requests_next_dt[zoid_num]);

                if (send_requests_next_dt[zoid_num].size() > 0) {
                    send_request_threads_next_dt.push_back(
                            std::move(std::thread([&](int zoid_num_) {
                                          int wait_status = MPI_Waitall(send_requests_next_dt[zoid_num_].size(),
                                                                        send_requests_next_dt[zoid_num_].data(), MPI_STATUSES_IGNORE);
                                          assert(wait_status == MPI_SUCCESS);
                                      }, zoid_num)
                            ));
                }

                /*
                queue_info& zoid = lmp->zoid_num_to_zoid_next_dt[zoid_num];
                auto &atom_arr = lmp->atom_stencil_md[zoid_num];
                Comm *comm_ = lmp->comm_stencil_md[zoid_num];

                comm_->send_data_stencil_md_next_dt(atom_arr, zoid);
                */
                // int* buf = comm_->send_exclude_eval_tags_next_dt(atom_arr, lmp->zoid_num_to_zoid[zoid_num]);
            }
        }
    }

    for (auto& t : send_request_threads_next_dt) {
        t.join();
    }

    std::cout << GREEN << " ------------- STENCIL MD NEXT DT PASSED ----------------- " << RESET_COLOR << std::endl;

    for (int i = 0; i < NUM_TIMESTEPS_IN_PARALLEL + 1; i++) {
        delete[] test_f[i];
        delete[] test_x[i];
    }
}

/* ----------------------------------------------------------------------
   clear force on own & ghost atoms
   clear other arrays as needed
------------------------------------------------------------------------- */

void VerletKokkos::force_clear()
{
  if (external_force_clear) return;

  atomKK->k_f.clear_sync_state(); // ignore host forces/torques since device views
  atomKK->k_torque.clear_sync_state(); //   will be cleared below

  // clear force on all particles
  // if either newton flag is set, also include ghosts
  // when using threads always clear all forces.

  if (neighbor->includegroup == 0) {
    int nall = atomKK->nlocal;
    if (force->newton) nall += atomKK->nghost;

    Kokkos::parallel_for(nall, Zero<typename ArrayTypes<LMPDeviceType>::t_f_array>(atomKK->k_f.view<LMPDeviceType>()));
    atomKK->modified(Device,F_MASK);

    if (torqueflag) {
      Kokkos::parallel_for(nall, Zero<typename ArrayTypes<LMPDeviceType>::t_f_array>(atomKK->k_torque.view<LMPDeviceType>()));
      atomKK->modified(Device,TORQUE_MASK);
    }

    // reset SPIN forces

    if (extraflag) {
      Kokkos::parallel_for(nall, Zero<typename ArrayTypes<LMPDeviceType>::t_fm_array>(atomKK->k_fm.view<LMPDeviceType>()));
      atomKK->modified(Device,FM_MASK);
      Kokkos::parallel_for(nall, Zero<typename ArrayTypes<LMPDeviceType>::t_fm_array>(atomKK->k_fm_long.view<LMPDeviceType>()));
      atomKK->modified(Device,FML_MASK);
    }

  // neighbor includegroup flag is set
  // clear force only on initial nfirst particles
  // if either newton flag is set, also include ghosts

  } else {
    Kokkos::parallel_for(atomKK->nfirst, Zero<typename ArrayTypes<LMPDeviceType>::t_f_array>(atomKK->k_f.view<LMPDeviceType>()));
    atomKK->modified(Device,F_MASK);

    if (torqueflag) {
      Kokkos::parallel_for(atomKK->nfirst, Zero<typename ArrayTypes<LMPDeviceType>::t_f_array>(atomKK->k_torque.view<LMPDeviceType>()));
      atomKK->modified(Device,TORQUE_MASK);
    }

    // reset SPIN forces

    if (extraflag) {
      Kokkos::parallel_for(atomKK->nfirst, Zero<typename ArrayTypes<LMPDeviceType>::t_fm_array>(atomKK->k_fm.view<LMPDeviceType>()));
      atomKK->modified(Device,FM_MASK);
      Kokkos::parallel_for(atomKK->nfirst, Zero<typename ArrayTypes<LMPDeviceType>::t_fm_array>(atomKK->k_fm_long.view<LMPDeviceType>()));
      atomKK->modified(Device,FML_MASK);
    }

    if (force->newton) {
      auto range = Kokkos::RangePolicy<LMPDeviceType>(atomKK->nlocal, atomKK->nlocal + atomKK->nghost);
      Kokkos::parallel_for(range, Zero<typename ArrayTypes<LMPDeviceType>::t_f_array>(atomKK->k_f.view<LMPDeviceType>()));
      atomKK->modified(Device,F_MASK);

      if (torqueflag) {
        Kokkos::parallel_for(range, Zero<typename ArrayTypes<LMPDeviceType>::t_f_array>(atomKK->k_torque.view<LMPDeviceType>()));
        atomKK->modified(Device,TORQUE_MASK);
      }

      // reset SPIN forces

      if (extraflag) {
        Kokkos::parallel_for(range, Zero<typename ArrayTypes<LMPDeviceType>::t_fm_array>(atomKK->k_fm.view<LMPDeviceType>()));
        atomKK->modified(Device,FM_MASK);
        Kokkos::parallel_for(range, Zero<typename ArrayTypes<LMPDeviceType>::t_fm_array>(atomKK->k_fm_long.view<LMPDeviceType>()));
        atomKK->modified(Device,FML_MASK);
      }
    }
  }
}

void VerletKokkos::force_clear_stencil_md(Atom* atom_, Force* force_, Neighbor* neighbor_) {
    // TODO: might need to Kokkos-ify this shit as well
    return Verlet::force_clear_stencil_md(atom_, force_, neighbor_);

    if (external_force_clear) return;

    AtomKokkos* atomKK_ = (AtomKokkos*) atom_;
    atomKK_->k_f.clear_sync_state(); // ignore host forces/torques since device views
    atomKK_->k_torque.clear_sync_state(); //   will be cleared below

    // clear force on all particles
    // if either newton flag is set, also include ghosts
    // when using threads always clear all forces.

    if (neighbor_->includegroup == 0) {
        int nall = atomKK_->nlocal;
        if (force->newton) nall += atomKK_->nghost;

        Kokkos::parallel_for(nall, Zero<typename ArrayTypes<LMPDeviceType>::t_f_array>(atomKK_->k_f.view<LMPDeviceType>()));
        atomKK_->modified(Device,F_MASK);

        if (torqueflag) {
            Kokkos::parallel_for(nall, Zero<typename ArrayTypes<LMPDeviceType>::t_f_array>(atomKK_->k_torque.view<LMPDeviceType>()));
            atomKK_->modified(Device,TORQUE_MASK);
        }

        // reset SPIN forces

        if (extraflag) {
            Kokkos::parallel_for(nall, Zero<typename ArrayTypes<LMPDeviceType>::t_fm_array>(atomKK_->k_fm.view<LMPDeviceType>()));
            atomKK_->modified(Device,FM_MASK);
            Kokkos::parallel_for(nall, Zero<typename ArrayTypes<LMPDeviceType>::t_fm_array>(atomKK_->k_fm_long.view<LMPDeviceType>()));
            atomKK_->modified(Device,FML_MASK);
        }

        // neighbor includegroup flag is set
        // clear force only on initial nfirst particles
        // if either newton flag is set, also include ghosts

    } else {
        Kokkos::parallel_for(atomKK_->nfirst, Zero<typename ArrayTypes<LMPDeviceType>::t_f_array>(atomKK_->k_f.view<LMPDeviceType>()));
        atomKK_->modified(Device,F_MASK);

        if (torqueflag) {
            Kokkos::parallel_for(atomKK_->nfirst, Zero<typename ArrayTypes<LMPDeviceType>::t_f_array>(atomKK_->k_torque.view<LMPDeviceType>()));
            atomKK_->modified(Device,TORQUE_MASK);
        }

        // reset SPIN forces

        if (extraflag) {
            Kokkos::parallel_for(atomKK_->nfirst, Zero<typename ArrayTypes<LMPDeviceType>::t_fm_array>(atomKK_->k_fm.view<LMPDeviceType>()));
            atomKK_->modified(Device,FM_MASK);
            Kokkos::parallel_for(atomKK_->nfirst, Zero<typename ArrayTypes<LMPDeviceType>::t_fm_array>(atomKK_->k_fm_long.view<LMPDeviceType>()));
            atomKK_->modified(Device,FML_MASK);
        }

        if (force->newton) {
            auto range = Kokkos::RangePolicy<LMPDeviceType>(atomKK_->nlocal, atomKK_->nlocal + atomKK_->nghost);
            Kokkos::parallel_for(range, Zero<typename ArrayTypes<LMPDeviceType>::t_f_array>(atomKK_->k_f.view<LMPDeviceType>()));
            atomKK_->modified(Device,F_MASK);

            if (torqueflag) {
                Kokkos::parallel_for(range, Zero<typename ArrayTypes<LMPDeviceType>::t_f_array>(atomKK_->k_torque.view<LMPDeviceType>()));
                atomKK_->modified(Device,TORQUE_MASK);
            }

            // reset SPIN forces

            if (extraflag) {
                Kokkos::parallel_for(range, Zero<typename ArrayTypes<LMPDeviceType>::t_fm_array>(atomKK_->k_fm.view<LMPDeviceType>()));
                atomKK_->modified(Device,FM_MASK);
                Kokkos::parallel_for(range, Zero<typename ArrayTypes<LMPDeviceType>::t_fm_array>(atomKK_->k_fm_long.view<LMPDeviceType>()));
                atomKK_->modified(Device,FML_MASK);
            }
        }
    }
}
