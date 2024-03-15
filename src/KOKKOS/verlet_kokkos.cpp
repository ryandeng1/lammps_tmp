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
