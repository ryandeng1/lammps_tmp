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

#include "accelerator_omp.h"
#include "angle.h"
#include "atom.h"
#include "atom_vec.h"
#include "bond.h"
#include "comm.h"
#include "dihedral.h"
#include "error.h"
#include "fix.h"
#include "force.h"
#include "improper.h"
#include "info.h"
#include "kspace.h"
#include "modify.h"
#include "neighbor.h"
#include "output.h"
#include "pointers.h"
#include "stencil_md_utils.h"
#include "timer.h"
#include "update.h"
#include "version.h"

#include <atomic>
#include <chrono>
#include <memory>
#include <mpi.h>
#include <cmath>
#include <cstring>
#include <map>
#include <algorithm>
#include "pair_lj_cut.h"

#include "stencil_md.h"

#include <iostream>
#include <cilk/cilk.h>
#include <cilk/cilk_api.h>
// #include <mpi_proto.h>
#include <ostream>
#include <unordered_map>
#include <sstream>
#include <cilk/opadd_reducer.h>
#include <sys/time.h>
#include "time_stencil_md.h"

using namespace LAMMPS_NS;

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

constexpr bool USE_DOUBLE_BUFFERING = true;

static std::vector<std::tuple<std::string, std::string, int, int64_t>> stencil_md_timings;
static std::vector<std::tuple<std::string, std::string, int, int64_t>> lammps_timings;
static spinlock m;
static constexpr bool TIME_STENCILMD_STATES = false;
static constexpr bool TIME_LAMMPS_STATES = false;
constexpr int64_t MICROSECOND_FACTOR = 1000000;
constexpr int NUM_RECV_NEIGHBORS[NUM_DEPS] = {0, 2, 8, 26};

static cilk::opadd_reducer<double> v_comm_time = 0;
static double other_time = 0;

/* ---------------------------------------------------------------------- */

Verlet::Verlet(LAMMPS* lmp, int narg, char** arg) : Integrate(lmp, narg, arg) {}

/* ----------------------------------------------------------------------
   initialization before run
------------------------------------------------------------------------- */

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
        // assert(false);
    }

    modify->setup(vflag);
    output->setup(flag);
    update->setupflag = 0;

    std::cout << GREEN << "------------------- LAMMPS SETUP DONE -------------------------" << RESET_COLOR << std::endl;

    if (!ONLY_RUN_LAMMPS) {
        setup_stencil_md_many_zoids();
    }
}

void Verlet::setup_stencil_md_many_zoids() {
    // setup MPI stuff?
    bool enable_striping = true;
    bool enable_hashing = true;
    MPI_Info comm_info;
    MPI_Info_create(&comm_info);
    MPI_Info_set(comm_info, "mpi_assert_no_any_source", "true");
    MPI_Info_set(comm_info, "mpi_assert_no_any_tag", "true");
    MPI_Info_set(comm_info, "enable_multi_nic_striping", enable_striping ? "true" : "false");
    MPI_Info_set(comm_info, "enable_multi_nic_hashing", enable_hashing ? "true" : "false");
    MPI_Comm_set_info(world, comm_info);

    /* LAMMPS TESTING CODE */
    double* send_f;
    double* recv_f;
    if (TEST_AGAINST_LAMMPS) {
        send_f = new double[(atom->natoms + 1) * 3];
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

        recv_f = new double[(atom->natoms + 1) * 3];

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
    }

    MPI_Barrier(world);

    // stencilMD->INIT_ZOIDS_NUMBERING();
    stencilMD->INIT_ZOIDS_NUMBERING_BALANCED_IMPROVED();
    stencilMD->INIT_ZOID_MANY_CUTS();
    stencilMD->INIT_ZOID_DATA_MANY_CUTS();
    stencilMD->INIT_MY_ZOIDS();
    stencilMD->INIT_ZOID_MANY_CUTS_NEIGHBORS();

    stencilMD->SORT_MY_ZOIDS<true>();
    stencilMD->SORT_MY_ZOIDS<false>();
    auto begin = std::chrono::high_resolution_clock::now();
    stencilMD->GET_ATOMS_ZOID_MANY_CUTS();
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::seconds>(end - begin).count();
    if (comm->me == 0) {
        std::cout << BOLDMAGENTA << "GET ATOMS: " << duration << " seconds." << RESET_COLOR << std::endl;
    }

    begin = std::chrono::high_resolution_clock::now();
    stencilMD->SORT_LOCAL_ATOMS_ZOID_MANY_CUTS();
    end = std::chrono::high_resolution_clock::now();
    duration = std::chrono::duration_cast<std::chrono::seconds>(end - begin).count();

    if (comm->me == 0) {
        std::cout << BOLDMAGENTA << "SORT ATOMS: " << duration << " seconds." << RESET_COLOR << std::endl;
    }

    begin = std::chrono::high_resolution_clock::now();
    stencilMD->CREATE_NEIGHBOR_LIST<USE_NEWTON>();
    end = std::chrono::high_resolution_clock::now();
    duration = std::chrono::duration_cast<std::chrono::seconds>(end - begin).count();
    if (comm->me == 0) {
        std::cout << BOLDMAGENTA << "GET NEIGHBOR LIST: " << duration << " seconds." << RESET_COLOR << std::endl;
    }

    begin = std::chrono::high_resolution_clock::now();
    if (EXPERIMENT == BOND_FENE) {
        stencilMD->CREATE_BOND_LIST<USE_NEWTON>();
    }
    end = std::chrono::high_resolution_clock::now();
    duration = std::chrono::duration_cast<std::chrono::seconds>(end - begin).count();
    if (comm->me == 0) {
        std::cout << BOLDMAGENTA << "GET BOND LIST: " << duration << " seconds." << RESET_COLOR << std::endl;
    }

    stencilMD->INIT_PER_WORKER_ARRAYS<true>();
    stencilMD->INIT_PER_WORKER_ARRAYS<false>();

    stencilMD->INIT_AFFINITY_AND_LOCKS();

    stencilMD->CONSTRUCT_SEND_FORCE_IDXS_ZOID_MANY_CUTS<true, USE_NEWTON>();
    stencilMD->CONSTRUCT_SEND_FORCE_IDXS_ZOID_MANY_CUTS<false, USE_NEWTON>();

    stencilMD->CONSTRUCT_SEND_VEL_IDXS_ZOID_MANY_CUTS<true>();
    stencilMD->CONSTRUCT_SEND_VEL_IDXS_ZOID_MANY_CUTS<false>();

    stencilMD->CONSTRUCT_RECV_POS_IDXS_ZOID_MANY_CUTS<true>();
    stencilMD->CONSTRUCT_RECV_POS_IDXS_ZOID_MANY_CUTS<false>();

    stencilMD->CONSTRUCT_SEND_POS_IDXS_ZOID_MANY_CUTS<true>();
    stencilMD->CONSTRUCT_SEND_POS_IDXS_ZOID_MANY_CUTS<false>();

    if constexpr (EXPERIMENT == EAM) {
        stencilMD->CONSTRUCT_RECV_RHO_IDXS_ZOID_MANY_CUTS<true, false>();
        stencilMD->CONSTRUCT_SEND_RHO_IDXS_ZOID_MANY_CUTS<true, false>();

        stencilMD->CONSTRUCT_RECV_RHO_IDXS_ZOID_MANY_CUTS<false, false>();
        stencilMD->CONSTRUCT_SEND_RHO_IDXS_ZOID_MANY_CUTS<false, false>();

        stencilMD->CONSTRUCT_RECV_FP_IDXS_ZOID_MANY_CUTS<true>();
        stencilMD->CONSTRUCT_SEND_FP_IDXS_ZOID_MANY_CUTS<true>();

        stencilMD->CONSTRUCT_RECV_FP_IDXS_ZOID_MANY_CUTS<false>();
        stencilMD->CONSTRUCT_SEND_FP_IDXS_ZOID_MANY_CUTS<false>();
    }

    stencilMD->CONSTRUCT_RECV_VEL_IDXS_ZOID_MANY_CUTS<true>();
    stencilMD->CONSTRUCT_RECV_VEL_IDXS_ZOID_MANY_CUTS<false>();

    stencilMD->CONSTRUCT_RECV_FORCE_IDXS_ZOID_MANY_CUTS<true, USE_NEWTON>();
    stencilMD->CONSTRUCT_RECV_FORCE_IDXS_ZOID_MANY_CUTS<false, USE_NEWTON>();

    stencilMD->INIT_SEND_RECV_BUFFERS_MANY_CUTS();

    stencilMD->CONSTRUCT_SEND_ZOID_TO_ZOID_SIZES<true>();
    stencilMD->CONSTRUCT_SEND_ZOID_TO_ZOID_SIZES<false>();
    stencilMD->CONSTRUCT_RECV_ZOID_TO_ZOID_SIZES<true>();
    stencilMD->CONSTRUCT_RECV_ZOID_TO_ZOID_SIZES<false>();

    stencilMD->CONSTRUCT_SEND_ZOID_TO_ZOID_SIZES_PIPELINED<true>();
    stencilMD->CONSTRUCT_SEND_ZOID_TO_ZOID_SIZES_PIPELINED<false>();
    stencilMD->CONSTRUCT_RECV_ZOID_TO_ZOID_SIZES_PIPELINED<true>();
    stencilMD->CONSTRUCT_RECV_ZOID_TO_ZOID_SIZES_PIPELINED<false>();

    stencilMD->CONSTRUCT_SEND_PROC_TO_PROC_OFFSETS_PIPELINED<true>();
    stencilMD->CONSTRUCT_SEND_PROC_TO_PROC_OFFSETS_PIPELINED<false>();
    stencilMD->CONSTRUCT_RECV_PROC_TO_PROC_OFFSETS_PIPELINED<true>();
    stencilMD->CONSTRUCT_RECV_PROC_TO_PROC_OFFSETS_PIPELINED<false>();

    stencilMD->CONSTRUCT_RECV_PROC_TO_PROC_AND_ZOID_TO_ZOID_RECEIVE_REQUEST_IDXS<true>();
    stencilMD->CONSTRUCT_RECV_PROC_TO_PROC_AND_ZOID_TO_ZOID_RECEIVE_REQUEST_IDXS<false>();

    stencilMD->CONSTRUCT_PER_ZOID_RECV_REQUEST_IDXS<true>();
    stencilMD->CONSTRUCT_PER_ZOID_RECV_REQUEST_IDXS<false>();

    stencilMD->CONSTRUCT_RECV_ZOID_PAIR_AND_PROC_PAIR_IDXS<true>();
    stencilMD->CONSTRUCT_RECV_ZOID_PAIR_AND_PROC_PAIR_IDXS<false>();

    stencilMD->INIT_MPIX_STREAM_DATA<true>();
    stencilMD->INIT_MPIX_STREAM_DATA<false>();

    stencilMD->GET_SEND_STATISTICS<true>();

    stencilMD->GET_RECV_STATISTICS<true>();
    stencilMD->GET_RECV_STATISTICS<false>();

    stencilMD->INIT_DEP_PROC_RECV_ZOID_DATA<true>();
    stencilMD->INIT_DEP_PROC_RECV_ZOID_DATA<false>();
    stencilMD->INIT_DEP_TO_SEND_ZOIDS<true>();
    stencilMD->INIT_DEP_TO_SEND_ZOIDS<false>();

    std::vector<MPI_Request> send_r[NUM_DEPS];
    for (int dep = 0; dep < NUM_DEPS - 1; dep++) {
        send_r[dep].reserve(comm->nprocs);
    }

    std::vector<MPI_Request> recv_r[NUM_DEPS];
    for (int dep = 1; dep < NUM_DEPS; dep++) {
        recv_r[dep].resize(comm->nprocs);
    }

    int recv_r_idxs[NUM_DEPS] = {0};

    std::map<std::pair<int, int>, bool> did_recv_map;
    std::map<int, int> dep_to_nproc_send;

    std::vector<MPI_Request> all_recv_requests[stencilMD->NUM_ZOIDS_MANY_CUTS];
    std::vector<MPI_Request> all_send_requests[stencilMD->NUM_ZOIDS_MANY_CUTS];

    for (int dep = 1; dep < NUM_DEPS; dep++) {
        for (int j = 0; j < stencilMD->my_queues_many_cuts[dep].size(); j++) {
            auto& zoid = stencilMD->my_queues_many_cuts[dep][j];
            int zoid_num = zoid.num;
            assert(zoid_num % comm->nprocs == comm->me);

            all_recv_requests[zoid_num].reserve(20);
            stencilMD->RECEIVE_DATA_ZOID_TO_ZOID_SETUP(zoid.num, all_recv_requests[zoid_num]);
        }
    }

    for (int dep = 0; dep < NUM_DEPS; dep++) {
        for (int j = 0; j < stencilMD->my_queues_many_cuts[dep].size(); j++) {
            auto& zoid = stencilMD->my_queues_many_cuts[dep][j];
            int zoid_num = zoid.num;
            assert(zoid_num % comm->nprocs == comm->me);
            stencilMD->UNPACK_DATA_MANY_CUTS_ZOID_SETUP(zoid, all_recv_requests[zoid_num]);
            if constexpr (EXPERIMENT == BOND_FENE) {
                stencilMD->BOND_FENE_FORCE_COMPUTE_ZOID_MANY_CUTS(zoid, dep, 0);
                // stencilMD->BOND_FENE_FORCE_COMPUTE_ZOID_MANY_CUTS_NEUTRAL_TERRITORY_ESQUE(zoid, dep, 0);
            } else if constexpr (EXPERIMENT == LJ) {
                stencilMD->LJ_FORCE_COMPUTE_ZOID_MANY_CUTS(zoid, dep, 0);
            } else if constexpr (EXPERIMENT == DPD) {
                stencilMD->DPD_FORCE_COMPUTE_ZOID_MANY_CUTS(zoid, dep, 0);
            } else if constexpr (EXPERIMENT == SW) {
                stencilMD->SW_FORCE_COMPUTE_ZOID_MANY_CUTS(zoid, dep, 0);
            } else if constexpr (EXPERIMENT == TERSOFF) {
                stencilMD->TERSOFF_FORCE_COMPUTE_ZOID_MANY_CUTS(zoid, dep, 0);
            } else if constexpr (EXPERIMENT == EAM) {
                stencilMD->EAM_FORCE_COMPUTE_ZOID_MANY_CUTS(zoid, dep, 0);
            }

            if constexpr (EXPERIMENT == BOND_FENE) {
                stencilMD->post_force_stencil_md_zoid_many_cuts_setup(zoid, 0);
            }

            if (TEST_AGAINST_LAMMPS) {
                stencilMD->TEST_AGAINST_LAMMPS_FORCE_DOUBLE_BUFFERING_SETUP(recv_f, zoid, 0);
            }
            stencilMD->PACK_AND_SEND_DATA_ZOID_TO_ZOID_SETUP(zoid, dep, all_send_requests[zoid_num]);
        }
    }

    for (int dep = 0; dep < NUM_DEPS; dep++) {
        for (int j = 0; j < stencilMD->my_queues_many_cuts[dep].size(); j++) {
            auto& zoid = stencilMD->my_queues_many_cuts[dep][j];
            int zoid_num = zoid.num;
            MPI_Waitall(all_send_requests[zoid_num].size(), all_send_requests[zoid_num].data(), MPI_STATUSES_IGNORE);
        }
    }

    int dep = 0;
    for (int j = 0; j < stencilMD->my_queues_many_cuts[dep].size(); j++) {
        auto& zoid = stencilMD->my_queues_many_cuts[dep][j];
        int zoid_num = zoid.num;
        auto& top = zoid.local_idxs_per_timestep[NUM_TIMESTEPS_IN_PARALLEL];
        std::cout << "dep 0 zoid top num: " << top.size() << std::endl;
        int total_beneath = 0;
        int total_num_pairs = 0;
        for (int t = 1; t < NUM_TIMESTEPS_IN_PARALLEL; t++) {
            const auto& local_idxs = zoid.local_idxs_per_timestep[t];
            for (auto& idx: local_idxs) {
                if (std::find(top.begin(), top.end(), idx) == top.end()) {
                    total_beneath++;
                    total_num_pairs += zoid.neighbor_list[t][idx].size();
                }
            }
        }

        std::cout << "total beneath: " << total_beneath << " npairs: " << total_num_pairs << std::endl;
    }

    // int64_t total_num_pairs = 0;
    // int64_t arr[NUM_TIMESTEPS_IN_PARALLEL + 1] = {0};
    // for (int dep = 0; dep < NUM_DEPS; dep++) {
    //     for (int j = 0; j < stencilMD->my_queues_many_cuts[dep].size(); j++) {
    //         auto& zoid = stencilMD->my_queues_many_cuts[dep][j];
    //         for (int t = 1; t < NUM_TIMESTEPS_IN_PARALLEL + 1; t++) {
    //             const auto& local_idxs = zoid.local_idxs_per_timestep[t];
    //             for (auto& local_idx : local_idxs) {
    //                 total_num_pairs += zoid.neighbor_list[t][local_idx].size();
    //                 arr[t] += zoid.neighbor_list[t][local_idx].size();;
    //             }
    //         }
    //     }
    // }
    
    // for (int dep = 0; dep < NUM_DEPS; dep++) {
    //     for (int j = 0; j < stencilMD->my_queues_many_cuts_next_dt[dep].size(); j++) {
    //         auto& zoid = stencilMD->my_queues_many_cuts_next_dt[dep][j];
    //         for (int t = 1; t < NUM_TIMESTEPS_IN_PARALLEL + 1; t++) {
    //             const auto& local_idxs = zoid.local_idxs_per_timestep[t];
    //             for (auto& local_idx : local_idxs) {
    //                 total_num_pairs += zoid.neighbor_list[t][local_idx].size();
    //                 arr[t] += zoid.neighbor_list[t][local_idx].size();;
    //             }
    //         }
    //     }
    // }

    // MPI_Allreduce(MPI_IN_PLACE, &total_num_pairs, 1, MPI_LONG, MPI_SUM, world);
    // MPI_Allreduce(MPI_IN_PLACE, arr, NUM_TIMESTEPS_IN_PARALLEL + 1, MPI_LONG, MPI_SUM, world);
    // if (comm->me == 0) {
    //     std::stringstream s1;
    //     s1 << "total num pairs: " << total_num_pairs << " across 2dt: " << 2 * NUM_TIMESTEPS_IN_PARALLEL << " per timestep: " << total_num_pairs * 1.0 / NUM_TIMESTEPS_IN_PARALLEL << std::endl;
    //     std::cout << s1.str();

    //     for (int t = 0; t < NUM_TIMESTEPS_IN_PARALLEL + 1; t++) {
    //         std::cout << "time: " << t << " num pairs: " << arr[t] << std::endl;
    //     }
    // }

    /*
    for (int dep = 1; dep < NUM_DEPS; dep++) {
        for (int proc = 0; proc < comm->nprocs; proc++) {
            bool did_recv = stencilMD->RECEIVE_DATA_MANY_CUTS<true>(dep, proc, &recv_r[dep][recv_r_idxs[dep]],
                                                                    setup_start_t, setup_end_t);
            if (did_recv) {
                recv_r_idxs[dep]++;
            }
            did_recv_map[{dep, proc}] = did_recv;
        }
    }

    for (int dep = 0; dep < NUM_DEPS; dep++) {
        if (dep > 0) {
            MPI_Waitall(recv_r_idxs[dep], recv_r[dep].data(), MPI_STATUSES_IGNORE);
            for (int proc = 0; proc < comm->nprocs; proc++) {
                if (did_recv_map[{dep, proc}] || proc == comm->me) {
                    stencilMD->UNPACK_DATA_MANY_CUTS<true, true>(dep, proc, setup_start_t, setup_end_t);
                }
            }
        }

        for (int j = 0; j < stencilMD->queues_many_cuts[dep].size(); j++) {
            auto& zoid = stencilMD->queues_many_cuts[dep][j];
            if (zoid.num % comm->nprocs != comm->me) {
                continue;
            }
            stencilMD->FORCE_COMPUTE_ZOID_MANY_CUTS(zoid, dep, 0);
            stencilMD->post_force_stencil_md_zoid_many_cuts_setup(zoid, 0);
            if (TEST_AGAINST_LAMMPS) {
                stencilMD->TEST_AGAINST_LAMMPS_FORCE_DOUBLE_BUFFERING_SETUP(recv_f, zoid, 0);
            }
            stencilMD->PACK_DATA_MANY_CUTS_ZOID<true, true>(zoid, dep, setup_start_t, setup_end_t);
        }

        if (dep < NUM_DEPS - 1) {
            int nproc_send = stencilMD->SEND_DATA_MANY_CUTS<true>(dep, send_r[dep]);
            dep_to_nproc_send[dep] = nproc_send;
        }
    }

    for (int dep = 0; dep < NUM_DEPS - 1; dep++) {
        MPI_Waitall(dep_to_nproc_send[dep], send_r[dep].data(), MPI_STATUSES_IGNORE);
    }
    */

    std::cout << BOLDGREEN << "Initial Force computation passed" << RESET_COLOR << std::endl;

    if (TEST_AGAINST_LAMMPS) {
        delete[] send_f;
        delete[] recv_f;
    }

    MPI_Barrier(world);
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
    if (TIME_STENCILMD_STATES) {
        stencil_md_timings.reserve(1024 * 10);
    }

    if (TIME_LAMMPS_STATES) {
	lammps_timings.reserve(3 * n);
    }

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

    if (TEST_AGAINST_LAMMPS) {
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
        if (TEST_AGAINST_LAMMPS) {
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
	if (TIME_LAMMPS_STATES) {
	    struct timeval tv_compute_start;
	    gettimeofday(&tv_compute_start, NULL);
	    lammps_timings.push_back(std::make_tuple("COMPUTE", "START", comm->me, tv_compute_start.tv_sec * MICROSECOND_FACTOR + tv_compute_start.tv_usec));
	}

        modify->initial_integrate(vflag);

	if (TIME_LAMMPS_STATES) {
	    struct timeval tv_compute_end;
	    gettimeofday(&tv_compute_end, NULL);
	    lammps_timings.push_back(std::make_tuple("COMPUTE", "END", comm->me, tv_compute_end.tv_sec * MICROSECOND_FACTOR + tv_compute_end.tv_usec));
	    lammps_timings.push_back(std::make_tuple("COMM", "START", comm->me, tv_compute_end.tv_sec * MICROSECOND_FACTOR + tv_compute_end.tv_usec));
	}

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
            auto begin = std::chrono::high_resolution_clock::now();
            comm->forward_comm();
            auto end = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count();
            lammps_comm_duration += duration;
            // lammps_forward_comm_duration += duration;
            // lammps_forward_comm_times.push_back(duration);
            timer->stamp(Timer::COMM);
	    if (TIME_LAMMPS_STATES) {
		struct timeval tv_comm_end;
		gettimeofday(&tv_comm_end, NULL);
		lammps_timings.push_back(std::make_tuple("COMM", "END", comm->me, tv_comm_end.tv_sec * MICROSECOND_FACTOR + tv_comm_end.tv_usec));
		lammps_timings.push_back(std::make_tuple("COMPUTE", "START", comm->me, tv_comm_end.tv_sec * MICROSECOND_FACTOR + tv_comm_end.tv_usec));
	    }
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
            int64_t total_num_pairs = 0;
            for (int k = 0; k < atom->nlocal; k++) {
                total_num_pairs += force->pair->list->numneigh[k];
            }
            auto begin = std::chrono::high_resolution_clock::now();
            force->pair->compute(eflag, vflag);
            auto end = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count();

            double throughput = total_num_pairs * 1.0 / duration;
            // std::stringstream s1;
            // s1 << "throughput: " << total_num_pairs * 1.0 / duration << std::endl;
            // std::cout << s1.str();
            // MPI_Allreduce(MPI_IN_PLACE, &total_num_pairs, 1, MPI_LONG, MPI_SUM, world);
            // MPI_Allreduce(MPI_IN_PLACE, &duration, 1, MPI_LONG, MPI_SUM, world);
            MPI_Allreduce(MPI_IN_PLACE, &throughput, 1, MPI_DOUBLE, MPI_SUM, world);

            if (comm->me == 0) {
                int world_size;
                MPI_Comm_size(world, &world_size);
                std::stringstream s1;
                std::cout << "average throughput per process: " << throughput / world_size << std::endl;
            }

            // lammps_pair_duration += duration;
            // lammps_num_atoms += atom->nlocal;
            // MPI_Allreduce(MPI_IN_PLACE, &total_num_pairs, 1, MPI_INT, MPI_SUM, world);
            // if (comm->me == 0) {
            //     std::stringstream s1;
            //     s1 << "total num pairs: " << total_num_pairs << std::endl;
            //     std::cout << s1.str();
            // }
            timer->stamp(Timer::PAIR);
        }

        if (atom->molecular != Atom::ATOMIC) {
            if (force->bond) {
                // auto begin = std::chrono::high_resolution_clock::now();
                force->bond->compute(eflag, vflag);
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

	if (TIME_LAMMPS_STATES) {
	    struct timeval tv_compute_end;
	    gettimeofday(&tv_compute_end, NULL);
	    lammps_timings.push_back(std::make_tuple("COMPUTE", "END", comm->me, tv_compute_end.tv_sec * MICROSECOND_FACTOR + tv_compute_end.tv_usec));
	    lammps_timings.push_back(std::make_tuple("COMM", "START", comm->me, tv_compute_end.tv_sec * MICROSECOND_FACTOR + tv_compute_end.tv_usec));
	}

        // reverse communication of forces
        if (force->newton) {
            auto begin = std::chrono::high_resolution_clock::now();
            comm->reverse_comm();
            auto end = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count();
            lammps_comm_duration += duration;
            // lammps_reverse_comm_duration += duration;
            // lammps_reverse_comm_times.push_back(duration);
            timer->stamp(Timer::COMM);
	    if (TIME_LAMMPS_STATES) {
		struct timeval tv_comm_end;
		gettimeofday(&tv_comm_end, NULL);
		lammps_timings.push_back(std::make_tuple("COMM", "END", comm->me, tv_comm_end.tv_sec * MICROSECOND_FACTOR + tv_comm_end.tv_usec));
		lammps_timings.push_back(std::make_tuple("COMPUTE", "START", comm->me, tv_comm_end.tv_sec * MICROSECOND_FACTOR + tv_comm_end.tv_usec));
	    }
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
	if (TIME_LAMMPS_STATES) {
	    struct timeval tv_compute_end;
	    gettimeofday(&tv_compute_end, NULL);
	    lammps_timings.push_back(std::make_tuple("COMPUTE", "END", comm->me, tv_compute_end.tv_sec * MICROSECOND_FACTOR + tv_compute_end.tv_usec));
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
                  << total_reverse_comm_duration
                  << " percent comm: " << total_comm_duration * 1.0 / total_duration_lammps
                  << RESET_COLOR << std::endl;

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

    if (comm->me == 0) {
        for (auto& tup : lammps_timings) {
	    std::stringstream s_w;
	    s_w << std::get<0>(tup) << "," << std::get<1>(tup) << "," << std::get<2>(tup) << "," << std::get<3>(tup) << std::endl;
	    std::cout << s_w.str();
        }
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

    if (TEST_AGAINST_LAMMPS) {
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

    std::vector<std::atomic_flag> zoid_claimed(stencilMD->NUM_ZOIDS_MANY_CUTS);
    std::vector<std::atomic_flag*> zoid_unpack_self_claimed(stencilMD->NUM_ZOIDS_MANY_CUTS);

    for (int dep = 0; dep < NUM_DEPS; dep++) {
        for (int j = 0; j < stencilMD->my_queues_many_cuts[dep].size(); j++) {
            int zoid_num = stencilMD->my_queues_many_cuts[dep][j].num;
            int max_recv_neighbors = std::max(stencilMD->recv_from_neighbors_many_cuts[zoid_num].size(), stencilMD->recv_from_neighbors_many_cuts_next_dt[zoid_num].size());
            zoid_unpack_self_claimed[zoid_num] = new std::atomic_flag[max_recv_neighbors];
        }
    }

    for (int i = 0; i < stencilMD->NUM_ZOIDS_MANY_CUTS; i++) {
        zoid_claimed[i].clear();
    }

    // run_stencil_md_many_cuts(n, test_f, test_x, test_v, zoid_claimed);
    // run_stencil_md_many_cuts(n, test_f, test_x, test_v, zoid_claimed);
    // run_stencil_md_many_cuts_pipelined(n, test_f, test_x, test_v, zoid_claimed, zoid_claimed2);
    run_stencil_md_many_cuts(2 * NUM_TIMESTEPS_IN_PARALLEL, test_f, test_x, test_v, zoid_claimed, zoid_unpack_self_claimed);

    total_time = 0;
    total_num_pairs = 0;
    v_comm_time = 0;
    stencilMD->reset_timers();

    MPI_Barrier(world);
    if (comm->me == 0) {
        std::cout << "----- END WARMUP -----" << std::endl;
    }

    // comm_time = 0;
    // compute_time = 0;

    int64_t duration;

    cilk_scope {
        auto begin = std::chrono::high_resolution_clock::now();
        run_stencil_md_many_cuts(n, test_f, test_x, test_v, zoid_claimed, zoid_unpack_self_claimed);
        // run_stencil_md_many_cuts_pipelined(n, test_f, test_x, test_v, zoid_claimed, zoid_claimed2);
        auto end = std::chrono::high_resolution_clock::now();
        duration = std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count();
    }

    for (int dep = 0; dep < NUM_DEPS; dep++) {
        for (int j = 0; j < stencilMD->my_queues_many_cuts[dep].size(); j++) {
            int zoid_num = stencilMD->my_queues_many_cuts[dep][j].num;
            int max_recv_neighbors = std::max(stencilMD->recv_from_neighbors_many_cuts[zoid_num].size(), stencilMD->recv_from_neighbors_many_cuts_next_dt[zoid_num].size());
            delete[] zoid_unpack_self_claimed[zoid_num];
        }
    }

    double throughput = n * 1.0 / (duration * 1.0 / 1e6);

    int64_t total_duration_stencil_md = 0;
    MPI_Allreduce(&duration, &total_duration_stencil_md, 1, MPI_INT64_T, MPI_SUM, world);

    MPI_Barrier(world);

    std::stringstream output_stream;
    output_stream << "me: " << comm->me 
    << " stencil md total just running the thing: " << duration << " microseconds. "
    << " throughput (timesteps/s) : " << throughput
    << " total duration: " << total_duration_stencil_md << std::endl;
    std::cout << output_stream.str();

    double total_compute_time = 0;
    double total_comm_time = 0;
    for (int w = 0; w < 24; w++) {
        total_compute_time += s_compute_time[w];
        total_comm_time += s_comm_time[w];

        // if (comm->me == 0) {
        //     std::cout << "worker w: " << w << " comm time: " << s_comm_time[w] * 1e6 << " " << " compute time: " << s_compute_time[w] * 1e6 << std::endl;
        // }
    }

    if (comm->me == 0) {
        std::cout << "total v comm time: " << v_comm_time * 1e6 << std::endl;
    }

    total_comm_time += v_comm_time;

    double all_reduce_compute;
    double all_reduce_comm;
    double all_reduce_other;

    MPI_Allreduce(&total_compute_time, &all_reduce_compute, 1, MPI_DOUBLE, MPI_SUM, world);
    MPI_Allreduce(&total_comm_time, &all_reduce_comm, 1, MPI_DOUBLE, MPI_SUM, world);
    MPI_Allreduce(&other_time, &all_reduce_other, 1, MPI_DOUBLE, MPI_SUM, world);

    if (comm->me == 0) {
        std::cout << "total compute time: " << total_compute_time * 1e6 << " total comm time: " << total_comm_time * 1e6  << " total other time: " << other_time * 1e6
        << " all reduce compute time: " << all_reduce_compute * 1e6  << " all reduce comm: " << all_reduce_comm * 1e6
        << " percentage comm: " << all_reduce_comm / (all_reduce_comm + all_reduce_compute + all_reduce_other)
        << " percentage compute: " << all_reduce_compute / (all_reduce_comm + all_reduce_compute + all_reduce_other)
        << " percentage other: " << all_reduce_other / (all_reduce_comm + all_reduce_compute + all_reduce_other)
        << std::endl;
    }

    int64_t num_pairs_total = total_num_pairs;
    double time_total = total_time;

    MPI_Allreduce(MPI_IN_PLACE, &num_pairs_total, 1, MPI_LONG, MPI_SUM, world);
    MPI_Allreduce(MPI_IN_PLACE, &time_total, 1, MPI_DOUBLE, MPI_SUM, world);

    if (comm->me == 0) {
        std::cout << "average throughput: " << num_pairs_total * 1.0 / time_total << std::endl;
    }

    if (comm->me == 0) {
        for (auto& tup : stencil_md_timings) {
	    std::stringstream s_w;
	    s_w << std::get<0>(tup) << "," << std::get<1>(tup) << "," << std::get<2>(tup) << "," << std::get<3>(tup) << std::endl;
	    std::cout << s_w.str();
        }
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
    int64_t stencil_md_total_unpack_duration = 0;
    int64_t stencil_md_total_misc_duration = 0;
    int64_t stencil_md_total_pre_recv_time = 0;
    int64_t stencil_md_total_unpack_self_time = 0;

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
    MPI_Allreduce(&unpack_duration, &stencil_md_total_unpack_duration, 1, MPI_INT64_T, MPI_SUM, world);

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
                  << " TOTAL UNPACK SELF TIME: " << stencil_md_total_unpack_self_time
                  << " TOTAL UNPACK DURATION: " << stencil_md_total_unpack_duration
                  << RESET_COLOR << std::endl;

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

template <bool curr_dt>
void Verlet::run_stencil_md_zoid_many_cuts(int starting_timestep, int dep, queue_info& zoid, int start_t, int end_t,
                                           double** test_f, double** test_x, double** test_v) {
    for (int t = start_t; t < end_t; t++) {
        if (TEST_AGAINST_LAMMPS) {
            int timestep_to_compare_against = curr_dt ? starting_timestep + t
                    : starting_timestep + NUM_TIMESTEPS_IN_PARALLEL + t;

            stencilMD->TEST_AGAINST_LAMMPS_FORCE_DOUBLE_BUFFERING(curr_dt, timestep_to_compare_against,
                                                                  test_f[timestep_to_compare_against],
                                                                  zoid, t);

            stencilMD->TEST_AGAINST_LAMMPS_POS_DOUBLE_BUFFERING(curr_dt, timestep_to_compare_against,
                                                                test_x[timestep_to_compare_against],
                                                                zoid, t);

            stencilMD->TEST_AGAINST_LAMMPS_VEL_DOUBLE_BUFFERING(curr_dt, timestep_to_compare_against,
                                                                test_v[timestep_to_compare_against],
                                                                zoid, t);
        }

        stencilMD->NVE_INITIAL_INTEGRATE_ZOID_MANY_CUTS(zoid, dep, t);
        if constexpr (EXPERIMENT == BOND_FENE) {
            stencilMD->BOND_FENE_FORCE_COMPUTE_ZOID_MANY_CUTS(zoid, dep, t + 1);
            stencilMD->FUSE_POST_FORCE_FINAL_INTEGRATE_ZOID_MANY_CUTS(zoid, dep, t + 1);
        } else if constexpr (EXPERIMENT == LJ) {
            stencilMD->LJ_FORCE_COMPUTE_ZOID_MANY_CUTS(zoid, dep, t + 1);
            stencilMD->NVE_FINAL_INTEGRATE_ZOID_MANY_CUTS(zoid, dep, t + 1);
        } else if constexpr (EXPERIMENT == DPD) {
            stencilMD->DPD_FORCE_COMPUTE_ZOID_MANY_CUTS(zoid, dep, t + 1);
            stencilMD->NVE_FINAL_INTEGRATE_ZOID_MANY_CUTS(zoid, dep, t + 1);
        } else if constexpr (EXPERIMENT == SW) {
            stencilMD->SW_FORCE_COMPUTE_ZOID_MANY_CUTS(zoid, dep, t + 1);
            stencilMD->NVE_FINAL_INTEGRATE_ZOID_MANY_CUTS(zoid, dep, t + 1);
        } else if constexpr (EXPERIMENT == TERSOFF) {
            stencilMD->TERSOFF_FORCE_COMPUTE_ZOID_MANY_CUTS(zoid, dep, t + 1);
            stencilMD->NVE_FINAL_INTEGRATE_ZOID_MANY_CUTS(zoid, dep, t + 1);
        } else if constexpr (EXPERIMENT == EAM) {
            stencilMD->EAM_FORCE_COMPUTE_ZOID_MANY_CUTS(zoid, dep, t + 1);
            stencilMD->NVE_FINAL_INTEGRATE_ZOID_MANY_CUTS(zoid, dep, t + 1);
        }
    }
}

template <bool curr_dt>
void Verlet::run_stencil_md_many_cuts_helper_dep(int starting_timestep, int dep,
                                                 std::vector<MPI_Request>* recv_requests,
                                                 std::vector<MPI_Request>* send_requests,
                                                 int start_t, int end_t, int pipeline_stage,
                                                 double** test_f, double** test_x, double** test_v) {
    /*
    auto& queues = curr_dt ? stencilMD->my_queues_many_cuts
            : stencilMD->my_queues_many_cuts_next_dt;

    cilk_for (int j = 0; j < queues[dep].size(); j++) {
        auto& zoid = queues[dep][j];
        int zoid_num = zoid.num;
        stencilMD->UNPACK_DATA_MANY_CUTS_ZOID_PIPELINED<curr_dt, false>(zoid, recv_requests[zoid_num],
                                                                        start_t, end_t, pipeline_stage);
        run_stencil_md_zoid_many_cuts<curr_dt>(starting_timestep, dep, zoid,
                                               start_t - 1, end_t - 1,
                                               test_f, test_x, test_v);

        stencilMD->PACK_AND_SEND_DATA_ZOID_TO_ZOID_PIPELINED<curr_dt, false>(zoid, dep,
                                                                             start_t, end_t, pipeline_stage,
                                                                             send_requests[zoid_num]);
    }
    */

    assert(false);
}

// Assume no comm needed
template <bool curr_dt>
void Verlet::run_stencil_md_zoid_many_cuts_no_comm(int starting_timestep, int dep, queue_info& zoid, int start_t, int end_t,
                                                   std::vector<MPI_Request>* send_r,
                                                   double** test_f, double** test_x, double** test_v) {
    int zoid_num = zoid.num;
    int num_neighbors_receive_self = stencilMD->UNPACK_DATA_MANY_CUTS_ZOID_SELF_ONLY<curr_dt>(zoid, start_t, end_t);

    if (TIME_STENCILMD_STATES) {
        struct timeval tv_compute_begin;
        gettimeofday(&tv_compute_begin, NULL);
        m.lock();
        stencil_md_timings.push_back(std::make_tuple("COMPUTE", "START", zoid.num, tv_compute_begin.tv_sec * MICROSECOND_FACTOR + tv_compute_begin.tv_usec));
        m.unlock();
    }

    run_stencil_md_zoid_many_cuts<curr_dt>(starting_timestep, dep, zoid,
                                           start_t - 1, end_t - 1,
                                           test_f, test_x, test_v);

    if (TIME_STENCILMD_STATES) {
        struct timeval tv_compute_end;
        gettimeofday(&tv_compute_end, NULL);
        m.lock();
        stencil_md_timings.push_back(std::make_tuple("COMPUTE", "END", zoid.num, tv_compute_end.tv_sec * MICROSECOND_FACTOR + tv_compute_end.tv_usec));
        stencil_md_timings.push_back(std::make_tuple("SEND", "START", zoid.num, tv_compute_end.tv_sec * MICROSECOND_FACTOR + tv_compute_end.tv_usec));
        m.unlock();
    }

    /*
    stencilMD->PACK_AND_SEND_DATA_ZOID_TO_ZOID<curr_dt>(zoid, dep,
                                                        start_t, end_t,
                                                        send_r[zoid_num]);
    */
    stencilMD->PACK_AND_SEND_DATA_ZOID_TO_ZOID_REVISED<curr_dt>(zoid, dep,
                                                                start_t, end_t,
                                                                send_r[zoid_num]);

    if (TIME_STENCILMD_STATES) {
        struct timeval tv_send_end;
        gettimeofday(&tv_send_end, NULL);
        m.lock();
        stencil_md_timings.push_back(std::make_tuple("SEND", "END", zoid.num, tv_send_end.tv_sec * MICROSECOND_FACTOR + tv_send_end.tv_usec));
        m.unlock();
    }
}

// Assume no comm needed
template <bool curr_dt>
void Verlet::run_stencil_md_zoid_many_cuts_no_comm_pipelined(int starting_timestep, int dep, queue_info& zoid,
                                                             int start_t, int end_t, int pipeline_stage,
                                                             std::vector<MPI_Request>* send_r,
                                                             double** test_f, double** test_x, double** test_v) {
    int zoid_num = zoid.num;
    int num_neighbors_receive_self = stencilMD->UNPACK_DATA_MANY_CUTS_ZOID_SELF_ONLY_PIPELINED<curr_dt>(zoid, start_t, end_t, pipeline_stage);

    run_stencil_md_zoid_many_cuts<curr_dt>(starting_timestep, dep, zoid,
                                           start_t - 1, end_t - 1,
                                           test_f, test_x, test_v);


    stencilMD->PACK_AND_SEND_DATA_ZOID_TO_ZOID_PIPELINED<curr_dt>(zoid, dep, start_t, end_t, pipeline_stage, send_r[zoid.num]);
}

// Assume no comm needed
template <bool curr_dt>
void Verlet::run_stencil_md_zoid_many_cuts_no_comm_pipelined_only_next_dep(int starting_timestep, int dep, queue_info& zoid,
                                                                           int start_t, int end_t, int pipeline_stage,
                                                                           std::vector<std::vector<MPI_Request>>& send_r,
                                                                           double** test_f, double** test_x, double** test_v) {
    int zoid_num = zoid.num;
    int num_neighbors_receive_self = stencilMD->UNPACK_DATA_MANY_CUTS_ZOID_SELF_ONLY_PIPELINED<curr_dt>(zoid, start_t, end_t, pipeline_stage);

    run_stencil_md_zoid_many_cuts<curr_dt>(starting_timestep, dep, zoid,
                                           start_t - 1, end_t - 1,
                                           test_f, test_x, test_v);

    stencilMD->PACK_AND_SEND_DATA_PIPELINED_ONLY_NEXT_DEP<curr_dt>(zoid, dep, start_t, end_t, pipeline_stage, send_r[zoid.num]);
}

template <bool curr_dt>
void Verlet::unpack_self_wrapper(int starting_timestep, int dep, queue_info& zoid,
                                 int start_t, int end_t, std::atomic<int>& counter,
                                 std::vector<MPI_Request>* send_r,
                                 double** test_f, double** test_x, double** test_v,
                                 std::vector<std::atomic_flag>& claimed) {

    int num_neighbors_receive_self = stencilMD->UNPACK_DATA_MANY_CUTS_ZOID_SELF_ONLY<curr_dt>(zoid, start_t, end_t);
    counter -= num_neighbors_receive_self;
    if (counter == 0) {
        if (!claimed[zoid.num].test(std::memory_order_relaxed)) {
            if (!claimed[zoid.num].test_and_set(std::memory_order_relaxed)) {
                stencilMD->UNPACK_FORCE_MANY_CUTS_ZOID<curr_dt>(zoid, start_t, end_t);
                if (TIME_STENCILMD_STATES) {
                    struct timeval tv_compute_begin;
                    gettimeofday(&tv_compute_begin, NULL);
                    m.lock();
                    stencil_md_timings.push_back(std::make_tuple("COMPUTE", "START", zoid.num, tv_compute_begin.tv_sec * MICROSECOND_FACTOR + tv_compute_begin.tv_usec));
                    m.unlock();
                }
                run_stencil_md_zoid_many_cuts<curr_dt>(starting_timestep, dep, zoid, start_t - 1, end_t - 1,
                                                       test_f, test_x, test_v);
                if (TIME_STENCILMD_STATES) {
                    struct timeval tv_compute_end;
                    gettimeofday(&tv_compute_end, NULL);
                    m.lock();
                    stencil_md_timings.push_back(std::make_tuple("COMPUTE", "END", zoid.num, tv_compute_end.tv_sec * MICROSECOND_FACTOR + tv_compute_end.tv_usec));
                    stencil_md_timings.push_back(std::make_tuple("SEND", "START", zoid.num, tv_compute_end.tv_sec * MICROSECOND_FACTOR + tv_compute_end.tv_usec));
                    m.unlock();
                }

                /*
                stencilMD->PACK_AND_SEND_DATA_ZOID_TO_ZOID<curr_dt>(zoid, dep,
                                                                    start_t, end_t, send_r[zoid.num]);
                */
                stencilMD->PACK_AND_SEND_DATA_ZOID_TO_ZOID_REVISED<curr_dt>(zoid, dep,
                                                                            start_t, end_t, send_r[zoid.num]);

                if (TIME_STENCILMD_STATES) {
                    struct timeval tv_send_end;
                    gettimeofday(&tv_send_end, NULL);
                    m.lock();
                    stencil_md_timings.push_back(std::make_tuple("SEND", "END", zoid.num, tv_send_end.tv_sec * MICROSECOND_FACTOR + tv_send_end.tv_usec));
                    m.unlock();
                }
            }
        }
    }
}

template <bool curr_dt>
void Verlet::unpack_self_wrapper_pipelined(int starting_timestep, int dep, queue_info& zoid,
                                           int start_t, int end_t, int pipeline_stage,
                                           std::atomic<int>& counter,
                                           std::vector<MPI_Request>* send_r,
                                           double** test_f, double** test_x, double** test_v,
                                           std::vector<std::atomic_flag>& claimed) {

    int num_neighbors_receive_self = stencilMD->UNPACK_DATA_MANY_CUTS_ZOID_SELF_ONLY_PIPELINED<curr_dt>(zoid, start_t, end_t, pipeline_stage);
    counter -= num_neighbors_receive_self;
    if (counter == 0) {
        if (!claimed[zoid.num].test(std::memory_order_relaxed)) {
            if (!claimed[zoid.num].test_and_set(std::memory_order_relaxed)) {
                stencilMD->UNPACK_FORCE_MANY_CUTS_ZOID_PIPELINED<curr_dt>(zoid, start_t, end_t, pipeline_stage);
                run_stencil_md_zoid_many_cuts<curr_dt>(starting_timestep, dep, zoid, start_t - 1, end_t - 1,
                                                       test_f, test_x, test_v);
                stencilMD->PACK_AND_SEND_DATA_ZOID_TO_ZOID_PIPELINED<curr_dt>(zoid, dep, start_t, end_t, pipeline_stage, send_r[zoid.num]);
            }
        }
    }
}

template <bool curr_dt>
void Verlet::unpack_self_wrapper_pipelined_only_next_dep(int starting_timestep, int dep, queue_info& zoid,
                                                         int start_t, int end_t, int pipeline_stage,
                                                         std::atomic<int>& counter,
                                                         std::vector<std::vector<MPI_Request>>& send_r,
                                                         double** test_f, double** test_x, double** test_v,
                                                         std::vector<std::atomic_flag>& claimed) {

    int num_neighbors_receive_self = stencilMD->UNPACK_DATA_MANY_CUTS_ZOID_SELF_ONLY_PIPELINED<curr_dt>(zoid, start_t, end_t, pipeline_stage);
    counter -= num_neighbors_receive_self;
    if (counter == 0) {
        if (!claimed[zoid.num].test(std::memory_order_relaxed)) {
            if (!claimed[zoid.num].test_and_set(std::memory_order_relaxed)) {
                stencilMD->UNPACK_FORCE_MANY_CUTS_ZOID_PIPELINED_ONLY_NEXT_DEP<curr_dt>(zoid, dep, start_t, end_t, pipeline_stage);
                run_stencil_md_zoid_many_cuts<curr_dt>(starting_timestep, dep, zoid, start_t - 1, end_t - 1,
                                                       test_f, test_x, test_v);
                stencilMD->PACK_AND_SEND_DATA_PIPELINED_ONLY_NEXT_DEP<curr_dt>(zoid, dep, start_t, end_t, pipeline_stage, send_r[zoid.num]);
            }
        }
    }
}

template <bool curr_dt>
void Verlet::unpack_other_wrapper(int starting_timestep, int dep, queue_info& zoid,
                                  int recv_zoid_num,
                                  int start_t, int end_t, std::atomic<int>& counter,
                                  std::vector<MPI_Request>* send_r,
                                  double** test_f, double** test_x, double** test_v,
                                  std::vector<std::atomic_flag>& claimed) {

    stencilMD->UNPACK_POS_VEL_MANY_CUTS_ZOID<curr_dt>(zoid, recv_zoid_num, start_t, end_t);
    counter--;
    if (counter == 0) {
        if (!claimed[zoid.num].test(std::memory_order_relaxed)) {
            if (!claimed[zoid.num].test_and_set(std::memory_order_relaxed)) {
                stencilMD->UNPACK_FORCE_MANY_CUTS_ZOID<curr_dt>(zoid, start_t, end_t);

                if (TIME_STENCILMD_STATES) {
                    struct timeval tv_compute_begin;
                    gettimeofday(&tv_compute_begin, NULL);
                    m.lock();
                    stencil_md_timings.push_back(std::make_tuple("COMPUTE", "START", zoid.num, tv_compute_begin.tv_sec * MICROSECOND_FACTOR + tv_compute_begin.tv_usec));
                    m.unlock();
                }

                run_stencil_md_zoid_many_cuts<curr_dt>(starting_timestep, dep, zoid, start_t - 1, end_t - 1,
                                                       test_f, test_x, test_v);

                if (TIME_STENCILMD_STATES) {
                    struct timeval tv_compute_end;
                    gettimeofday(&tv_compute_end, NULL);
                    m.lock();
                    stencil_md_timings.push_back(std::make_tuple("COMPUTE", "END", zoid.num, tv_compute_end.tv_sec * MICROSECOND_FACTOR + tv_compute_end.tv_usec));
                    stencil_md_timings.push_back(std::make_tuple("SEND", "START", zoid.num, tv_compute_end.tv_sec * MICROSECOND_FACTOR + tv_compute_end.tv_usec));
                    m.unlock();
                }

                /*
                stencilMD->PACK_AND_SEND_DATA_ZOID_TO_ZOID<curr_dt>(zoid, dep,
                                                                            start_t, end_t, send_r[zoid.num]);
                */

                stencilMD->PACK_AND_SEND_DATA_ZOID_TO_ZOID_REVISED<curr_dt>(zoid, dep,
                                                                            start_t, end_t, send_r[zoid.num]);

                if (TIME_STENCILMD_STATES) {
                    struct timeval tv_send_end;
                    gettimeofday(&tv_send_end, NULL);
                    m.lock();
                    stencil_md_timings.push_back(std::make_tuple("SEND", "END", zoid.num, tv_send_end.tv_sec * MICROSECOND_FACTOR + tv_send_end.tv_usec));
                    m.unlock();
                }
            }
        }
    }
}

template <bool curr_dt>
void Verlet::unpack_other_wrapper_pipelined(int starting_timestep, int dep, queue_info& zoid,
                                            int recv_zoid_num,
                                            int start_t, int end_t, int pipeline_stage, std::atomic<int>& counter,
                                            std::vector<MPI_Request>* send_r,
                                            double** test_f, double** test_x, double** test_v,
                                            std::vector<std::atomic_flag>& claimed) {

    stencilMD->UNPACK_POS_VEL_MANY_CUTS_ZOID_PIPELINED<curr_dt>(zoid, recv_zoid_num, start_t, end_t, pipeline_stage);
    counter--;
    if (counter == 0) {
        if (!claimed[zoid.num].test(std::memory_order_relaxed)) {
            if (!claimed[zoid.num].test_and_set(std::memory_order_relaxed)) {
                stencilMD->UNPACK_FORCE_MANY_CUTS_ZOID_PIPELINED<curr_dt>(zoid, start_t, end_t, pipeline_stage);

                run_stencil_md_zoid_many_cuts<curr_dt>(starting_timestep, dep, zoid, start_t - 1, end_t - 1,
                                                       test_f, test_x, test_v);

                stencilMD->PACK_AND_SEND_DATA_ZOID_TO_ZOID_PIPELINED<curr_dt>(zoid, dep, start_t, end_t, pipeline_stage, send_r[zoid.num]);
            }
        }
    }
}

template <bool curr_dt>
void Verlet::unpack_other_wrapper_pipelined_only_next_dep(int starting_timestep, int dep, queue_info& zoid,
                                                          int recv_zoid_num,
                                                          int start_t, int end_t, int pipeline_stage, std::atomic<int>& counter,
                                                          std::vector<std::vector<MPI_Request>>& send_r,
                                                          double** test_f, double** test_x, double** test_v,
                                                          std::vector<std::atomic_flag>& claimed) {

    stencilMD->UNPACK_POS_VEL_MANY_CUTS_ZOID_PIPELINED<curr_dt>(zoid, recv_zoid_num, start_t, end_t, pipeline_stage);
    counter--;
    if (counter == 0) {
        if (!claimed[zoid.num].test(std::memory_order_relaxed)) {
            if (!claimed[zoid.num].test_and_set(std::memory_order_relaxed)) {
                stencilMD->UNPACK_FORCE_MANY_CUTS_ZOID_PIPELINED_ONLY_NEXT_DEP<curr_dt>(zoid, dep, start_t, end_t, pipeline_stage);
                run_stencil_md_zoid_many_cuts<curr_dt>(starting_timestep, dep, zoid, start_t - 1, end_t - 1,
                                                       test_f, test_x, test_v);
                stencilMD->PACK_AND_SEND_DATA_PIPELINED_ONLY_NEXT_DEP<curr_dt>(zoid, dep, start_t, end_t, pipeline_stage, send_r[zoid.num]);
            }
        }
    }
}

template <bool curr_dt>
void Verlet::unpack_data_pipelined_proc_to_proc_helper(int starting_timestep, int dep, queue_info& zoid,
                                                       int start_t, int end_t, int pipeline_stage,
                                                       std::vector<std::vector<MPI_Request>>& send_r,
                                                       double** test_f, double** test_x, double** test_v) {
    stencilMD->UNPACK_FORCE_MANY_CUTS_ZOID_PIPELINED_ONLY_NEXT_DEP<curr_dt>(zoid, dep, start_t, end_t, pipeline_stage);
    run_stencil_md_zoid_many_cuts<curr_dt>(starting_timestep, dep, zoid, start_t - 1, end_t - 1,
                                           test_f, test_x, test_v);
    stencilMD->PACK_AND_SEND_DATA_PIPELINED_ONLY_NEXT_DEP<curr_dt>(zoid, dep, start_t, end_t, pipeline_stage, send_r[zoid.num]);
}

template <bool curr_dt>
void Verlet::unpack_data_pipelined_proc_to_proc_wrapper(int starting_timestep, int dep,
                                                queue_info& zoid, int recv_zoid_num, int find_idx, int send_dep, int proc, std::atomic<int>& counter,
                                                int start_t, int end_t, int pipeline_stage,
                                                std::vector<std::vector<MPI_Request>>& send_r,
                                                std::atomic_flag& claimed,
                                                double** test_f, double** test_x, double** test_v) {

    stencilMD->UNPACK_POS_VEL_MANY_CUTS_ZOID_PIPELINED_PROC_TO_PROC<curr_dt>(zoid,
                                                                            send_dep,
                                                                            proc,
                                                                            recv_zoid_num,
                                                                            find_idx,
                                                                            start_t,
                                                                            end_t,
                                                                            pipeline_stage);
    counter--;
    if (counter == 0) {
        if (!claimed.test(std::memory_order_relaxed)) {
            if (!claimed.test_and_set(std::memory_order_relaxed)) {
                unpack_data_pipelined_proc_to_proc_helper<curr_dt>(starting_timestep,
                                                                dep,zoid,
                                                                start_t, end_t,
                                                                pipeline_stage,
                                                                send_r,
                                                                test_f, test_x, test_v);
            }
        }
    }
}



template <bool curr_dt>
void Verlet::unpack_data_pipelined_proc_to_proc(int starting_timestep, int dep,
                                                int proc, int send_dep,
                                                int start_t, int end_t, int pipeline_stage, std::vector<std::atomic<int>>& counters,
                                                std::vector<std::vector<MPI_Request>>& send_r,
                                                double** test_f, double** test_x, double** test_v,
                                                std::vector<std::atomic_flag>& claimed) {

    constexpr int curr_dt_idx = static_cast<int>(curr_dt);

    auto& lst_zoids_from_proc = stencilMD->recv_zoids_from_proc[curr_dt_idx][pipeline_stage][send_dep][proc];
    for (auto& lst_info : lst_zoids_from_proc) {
        int zoid_num = lst_info[0];
        int zoid_dep = curr_dt ? stencilMD->zoid_num_to_dep[zoid_num] : stencilMD->zoid_num_to_dep_next_dt[zoid_num];
        int recv_zoid_num = lst_info[1];
        int find_idx = lst_info[2];
        auto &zoid = curr_dt ? stencilMD->zoid_num_to_zoid_many_cuts[zoid_num]
                             : stencilMD->zoid_num_to_zoid_many_cuts_next_dt[zoid_num];

        cilk_spawn unpack_data_pipelined_proc_to_proc_wrapper<curr_dt>(starting_timestep, dep, zoid, recv_zoid_num, find_idx, send_dep, proc, counters[zoid.num],
                                                    start_t, end_t, pipeline_stage, send_r, claimed[zoid.num], test_f, test_x, test_v);
        
        /*
        stencilMD->UNPACK_POS_VEL_MANY_CUTS_ZOID_PIPELINED_PROC_TO_PROC<curr_dt>(zoid,
                                                                                 send_dep,
                                                                                 proc,
                                                                                 recv_zoid_num,
                                                                                 find_idx,
                                                                                 start_t,
                                                                                 end_t,
                                                                                 pipeline_stage);
        counters[zoid_num]--;
        // if (zoid_dep == dep) {
            if (counters[zoid_num] == 0) {
                if (!claimed[zoid_num].test(std::memory_order_relaxed)) {
                    if (!claimed[zoid_num].test_and_set(std::memory_order_relaxed)) {
                        cilk_spawn unpack_data_pipelined_proc_to_proc_helper<curr_dt>(starting_timestep,
                                                                                      dep,
                                                                                      zoid,
                                                                                      start_t,
                                                                                      end_t,
                                                                                      pipeline_stage,
                                                                                      send_r,
                                                                                      test_f,
                                                                                      test_x,
                                                                                      test_v);
                    }
                }
            }
        // }
        */
    }

    /*
    if (counter == 0) {
        if (!claimed[zoid.num].test(std::memory_order_relaxed)) {
            if (!claimed[zoid.num].test_and_set(std::memory_order_relaxed)) {
                stencilMD->UNPACK_FORCE_MANY_CUTS_ZOID_PIPELINED_ONLY_NEXT_DEP<curr_dt>(zoid, dep, start_t, end_t, pipeline_stage);
                run_stencil_md_zoid_many_cuts<curr_dt>(starting_timestep, dep, zoid, start_t - 1, end_t - 1,
                                                       test_f, test_x, test_v);
                stencilMD->PACK_AND_SEND_DATA_PIPELINED_ONLY_NEXT_DEP<curr_dt>(zoid, dep, start_t, end_t, pipeline_stage, send_r[zoid.num]);
            }
        }
    }
    */
}

template <bool curr_dt>
void Verlet::unpack_data_proc_to_proc_wrapper_better_work_queue(int starting_timestep, int dep,
                                    queue_info& zoid, int recv_zoid_num, int find_idx,
                                    int proc, int send_dep,
                                    int start_timestep, int end_timestep, int pipeline_stage,
                                    std::vector<std::atomic<int>>& zoid_recv_neighbor_counters,
                                    std::vector<std::atomic<int>>& dep_counters,
                                    std::vector<std::vector<MPI_Request>>& send_r_zoid_to_zoid,
                                    std::vector<std::vector<MPI_Request>>& send_r_proc_to_proc,
                                    double** test_f, double** test_x, double** test_v,
                                    std::vector<std::atomic_flag>& zoid_claimed,
                                    std::vector<std::atomic_flag>& dep_claimed,
                                    MPIX_Stream_Manager* stream_manager,
                                    std::vector<std::atomic<bool>>& zoid_done) {

    constexpr int curr_dt_idx = static_cast<int>(curr_dt);
    int zoid_num = zoid.num;
    stencilMD->UNPACK_POS_VEL_MANY_CUTS_ZOID_PIPELINED_PROC_TO_PROC<curr_dt>(zoid,
                                                                    send_dep,
                                                                    proc,
                                                                    recv_zoid_num,
                                                                    find_idx,
                                                                    start_timestep,
                                                                    end_timestep,
                                                                    pipeline_stage);
    auto& counter = zoid_recv_neighbor_counters[zoid_num];
    auto& claimed = zoid_claimed[zoid_num];
    counter.fetch_sub(1, std::memory_order_relaxed);
    if (counter.load(std::memory_order_relaxed) == 0 && !claimed.test(std::memory_order_relaxed) && !claimed.test_and_set(std::memory_order_relaxed)) {
        cilk_spawn stencil_md_run_zoid_wrapper_better_work_queue<curr_dt>(starting_timestep, dep, zoid, start_timestep, end_timestep,
            zoid_recv_neighbor_counters, dep_counters, 
            send_r_zoid_to_zoid, send_r_proc_to_proc,
            test_f, test_x, test_v,
            zoid_claimed, dep_claimed, stream_manager, zoid_done);
    }
}

template <bool curr_dt>
void Verlet::unpack_data_proc_to_proc_better_work_queue(int starting_timestep, int dep,
                                    int proc, int send_dep,
                                    int start_timestep, int end_timestep, int pipeline_stage,
                                    std::vector<std::atomic<int>>& zoid_recv_neighbor_counters,
                                    std::vector<std::atomic<int>>& dep_counters,
                                    std::vector<std::vector<MPI_Request>>& send_r_zoid_to_zoid,
                                    std::vector<std::vector<MPI_Request>>& send_r_proc_to_proc,
                                    double** test_f, double** test_x, double** test_v,
                                    std::vector<std::atomic_flag>& zoid_claimed,
                                    std::vector<std::atomic_flag>& dep_claimed,
                                    MPIX_Stream_Manager* stream_manager, std::vector<std::atomic<bool>>& zoid_done) {

    constexpr int curr_dt_idx = static_cast<int>(curr_dt);

    auto& lst_zoids_from_proc = stencilMD->recv_zoids_from_proc[curr_dt_idx][pipeline_stage][send_dep][proc];
    for (auto& lst_info : lst_zoids_from_proc) {
        int zoid_num = lst_info[0];
        int zoid_dep = curr_dt ? stencilMD->zoid_num_to_dep[zoid_num] : stencilMD->zoid_num_to_dep_next_dt[zoid_num];
        int recv_zoid_num = lst_info[1];
        int find_idx = lst_info[2];
        auto &zoid = curr_dt ? stencilMD->zoid_num_to_zoid_many_cuts[zoid_num]
                             : stencilMD->zoid_num_to_zoid_many_cuts_next_dt[zoid_num];

        cilk_spawn unpack_data_proc_to_proc_wrapper_better_work_queue<curr_dt>(starting_timestep, dep, zoid, recv_zoid_num, find_idx, proc, send_dep,
            start_timestep, end_timestep, pipeline_stage,
            zoid_recv_neighbor_counters, dep_counters,
            send_r_zoid_to_zoid, send_r_proc_to_proc,
            test_f, test_x, test_v,
            zoid_claimed, dep_claimed, stream_manager, zoid_done);
    }
}

template <bool curr_dt>
void Verlet::unpack_data_proc_to_proc_wrapper(int starting_timestep, int dep,
                                    queue_info& zoid, int recv_zoid_num, int find_idx,
                                    int proc, int send_dep,
                                    int start_timestep, int end_timestep, int pipeline_stage,
                                    std::vector<std::atomic<int>>& zoid_recv_neighbor_counters,
                                    std::vector<std::atomic<int>>& dep_counters,
                                    std::vector<std::vector<MPI_Request>>& send_r_zoid_to_zoid,
                                    std::vector<std::vector<MPI_Request>>& send_r_proc_to_proc,
                                    double** test_f, double** test_x, double** test_v,
                                    std::vector<std::atomic_flag>& zoid_claimed,
                                    std::vector<std::atomic_flag>& dep_claimed,
                                    MPIX_Stream_Manager* stream_manager) {

    constexpr int curr_dt_idx = static_cast<int>(curr_dt);
    int zoid_num = zoid.num;
    stencilMD->UNPACK_POS_VEL_MANY_CUTS_ZOID_PIPELINED_PROC_TO_PROC<curr_dt>(zoid,
                                                                    send_dep,
                                                                    proc,
                                                                    recv_zoid_num,
                                                                    find_idx,
                                                                    start_timestep,
                                                                    end_timestep,
                                                                    pipeline_stage);
    auto& counter = zoid_recv_neighbor_counters[zoid_num];
    auto& claimed = zoid_claimed[zoid_num];
    counter--;
    if (counter == 0 && !claimed.test(std::memory_order_relaxed) && !claimed.test_and_set(std::memory_order_relaxed)) {
        cilk_spawn stencil_md_run_zoid_wrapper<curr_dt>(starting_timestep, dep, zoid, start_timestep, end_timestep,
            zoid_recv_neighbor_counters, dep_counters, 
            send_r_zoid_to_zoid, send_r_proc_to_proc,
            test_f, test_x, test_v,
            zoid_claimed, dep_claimed, stream_manager);
    }
}

template <bool curr_dt>
void Verlet::unpack_data_proc_to_proc(int starting_timestep, int dep,
                                    int proc, int send_dep,
                                    int start_timestep, int end_timestep, int pipeline_stage,
                                    std::vector<std::atomic<int>>& zoid_recv_neighbor_counters,
                                    std::vector<std::atomic<int>>& dep_counters,
                                    std::vector<std::vector<MPI_Request>>& send_r_zoid_to_zoid,
                                    std::vector<std::vector<MPI_Request>>& send_r_proc_to_proc,
                                    double** test_f, double** test_x, double** test_v,
                                    std::vector<std::atomic_flag>& zoid_claimed,
                                    std::vector<std::atomic_flag>& dep_claimed,
                                    MPIX_Stream_Manager* stream_manager) {

    constexpr int curr_dt_idx = static_cast<int>(curr_dt);

    auto& lst_zoids_from_proc = stencilMD->recv_zoids_from_proc[curr_dt_idx][pipeline_stage][send_dep][proc];
    for (auto& lst_info : lst_zoids_from_proc) {
        int zoid_num = lst_info[0];
        int zoid_dep = curr_dt ? stencilMD->zoid_num_to_dep[zoid_num] : stencilMD->zoid_num_to_dep_next_dt[zoid_num];
        int recv_zoid_num = lst_info[1];
        int find_idx = lst_info[2];
        auto &zoid = curr_dt ? stencilMD->zoid_num_to_zoid_many_cuts[zoid_num]
                             : stencilMD->zoid_num_to_zoid_many_cuts_next_dt[zoid_num];

        cilk_spawn unpack_data_proc_to_proc_wrapper<curr_dt>(starting_timestep, dep, zoid, recv_zoid_num, find_idx, proc, send_dep,
            start_timestep, end_timestep, pipeline_stage,
            zoid_recv_neighbor_counters, dep_counters,
            send_r_zoid_to_zoid, send_r_proc_to_proc,
            test_f, test_x, test_v,
            zoid_claimed, dep_claimed, stream_manager);
    }
}

// wrapper function around running a zoid.
//
template <bool curr_dt>
void Verlet::stencil_md_run_zoid_wrapper(int starting_timestep, int dep, queue_info& zoid,
                                    int start_timestep, int end_timestep,
                                    std::vector<std::atomic<int>>& zoid_recv_neighbor_counters, std::vector<std::atomic<int>>& dep_counters,
                                    std::vector<std::vector<MPI_Request>>& send_r_zoid_to_zoid,
                                    std::vector<std::vector<MPI_Request>>& send_r_proc_to_proc,
                                    double** test_f, double** test_x, double** test_v,
                                    std::vector<std::atomic_flag>& zoid_claimed,
                                    std::vector<std::atomic_flag>& dep_claimed,
                                    MPIX_Stream_Manager* stream_manager) noexcept {

    if (dep > 0) {
        stencilMD->UNPACK_FORCE_MANY_CUTS_ZOID_PIPELINED_ONLY_NEXT_DEP<curr_dt>(zoid, dep, start_timestep, end_timestep, DEFAULT_PIPELINE_STAGE);
    }

    run_stencil_md_zoid_many_cuts<curr_dt>(starting_timestep, dep, zoid, start_timestep - 1, end_timestep - 1,
                                        test_f, test_x, test_v);
    
    // std::vector<int> eval_zoids;

    stencilMD->PACK_DATA_WITH_PROC_TO_PROC<curr_dt>(zoid, dep, start_timestep, end_timestep, DEFAULT_PIPELINE_STAGE, stream_manager, send_r_zoid_to_zoid[zoid.num]);

    dep_counters[dep]--;
    if (dep_counters[dep] == 0 && !dep_claimed[dep].test(std::memory_order_relaxed) && !dep_claimed[dep].test_and_set(std::memory_order_relaxed)) {
        stencilMD->SEND_DATA_PROC_TO_PROC<curr_dt>(DEFAULT_PIPELINE_STAGE, dep, send_r_proc_to_proc[dep], stream_manager);
    }

    auto& send_neighbors = curr_dt ? stencilMD->send_to_neighbors_many_cuts[zoid.num] : stencilMD->send_to_neighbors_many_cuts_next_dt[zoid.num];

    /*
    for (int i = 0; i < send_neighbors.size(); i++) {
        int send_zoid_num = send_neighbors[i];
        if (send_zoid_num % comm->nprocs != comm->me) {
            continue;
        }

        auto& send_zoid = curr_dt ? stencilMD->zoid_num_to_zoid_many_cuts[send_zoid_num] : stencilMD->zoid_num_to_zoid_many_cuts_next_dt[send_zoid_num];
        int send_zoid_dep = curr_dt ? stencilMD->zoid_num_to_dep[send_zoid_num] : stencilMD->zoid_num_to_dep_next_dt[send_zoid_num];
        auto& recv_neighbors = curr_dt ? stencilMD->recv_from_neighbors_many_cuts[send_zoid_num] : stencilMD->recv_from_neighbors_many_cuts_next_dt[send_zoid_num];
        auto find_it = std::find(recv_neighbors.begin(), recv_neighbors.end(), zoid.num);
        assert(find_it != recv_neighbors.end());
        int find_idx = std::distance(recv_neighbors.begin(), find_it);
        stencilMD->UNPACK_DATA_MANY_CUTS_HELPER_SELF_PIPELINED<curr_dt>(send_zoid, find_idx, zoid.num, i, default_start_t, default_end_t, DEFAULT_PIPELINE_STAGE);
        zoid_recv_neighbor_counters[send_zoid_num]--;
        auto& claimed = zoid_claimed[send_zoid_num];
        if (zoid_recv_neighbor_counters[send_zoid_num] == 0
            && !claimed.test(std::memory_order_relaxed)
            && !claimed.test_and_set(std::memory_order_relaxed)) {
            // eval_zoids.push_back(send_zoid_num);
        }
    }
    */

    // stencilMD->SEND_DATA_ZOID_TO_ZOID<curr_dt>(zoid, dep, DEFAULT_PIPELINE_STAGE, send_r_zoid_to_zoid[zoid.num], stream_manager);
    /*
    for (int eval_zoid_num : eval_zoids) {
        int eval_zoid_dep = curr_dt ? stencilMD->zoid_num_to_dep[eval_zoid_num] : stencilMD->zoid_num_to_dep_next_dt[eval_zoid_num];
        auto& eval_zoid = curr_dt ? stencilMD->zoid_num_to_zoid_many_cuts[eval_zoid_num] : stencilMD->zoid_num_to_zoid_many_cuts_next_dt[eval_zoid_num];
        cilk_spawn stencil_md_run_zoid_wrapper<curr_dt>(starting_timestep, eval_zoid_dep, eval_zoid, default_start_t, default_end_t,
            zoid_recv_neighbor_counters, dep_counters, send_r_zoid_to_zoid, send_r_proc_to_proc,
            test_f, test_x, test_v, 
            zoid_claimed, dep_claimed, stream_manager);
    }
    */
}

template <bool curr_dt>
void Verlet::stencil_md_run_zoid_wrapper_better_work_queue(int starting_timestep, int dep, queue_info& zoid,
                                    int start_timestep, int end_timestep,
                                    std::vector<std::atomic<int>>& zoid_recv_neighbor_counters, std::vector<std::atomic<int>>& dep_counters,
                                    std::vector<std::vector<MPI_Request>>& send_r_zoid_to_zoid,
                                    std::vector<std::vector<MPI_Request>>& send_r_proc_to_proc,
                                    double** test_f, double** test_x, double** test_v,
                                    std::vector<std::atomic_flag>& zoid_claimed,
                                    std::vector<std::atomic_flag>& dep_claimed,
                                    MPIX_Stream_Manager* stream_manager,
                                    std::vector<std::atomic<bool>>& zoid_done) noexcept {

    if (dep > 0) {
        stencilMD->UNPACK_FORCE_MANY_CUTS_ZOID_PIPELINED_ONLY_NEXT_DEP<curr_dt>(zoid, dep, start_timestep, end_timestep, DEFAULT_PIPELINE_STAGE, true);
    }

    run_stencil_md_zoid_many_cuts<curr_dt>(starting_timestep, dep, zoid, start_timestep - 1, end_timestep - 1,
                                        test_f, test_x, test_v);
    zoid_done[zoid.num].store(true, std::memory_order_relaxed);
    
    stencilMD->PACK_DATA_WITH_PROC_TO_PROC<curr_dt>(zoid, dep, start_timestep, end_timestep, DEFAULT_PIPELINE_STAGE, stream_manager, send_r_zoid_to_zoid[zoid.num]);

    /*
    dep_counters[dep].fetch_sub(1, std::memory_order_relaxed);
    if (dep_counters[dep].load(std::memory_order_relaxed) == 0 && !dep_claimed[dep].test(std::memory_order_relaxed) && !dep_claimed[dep].test_and_set(std::memory_order_relaxed)) {
        stencilMD->SEND_DATA_PROC_TO_PROC<curr_dt>(DEFAULT_PIPELINE_STAGE, dep, send_r_proc_to_proc[dep], stream_manager);
    }
    */
}

template <bool curr_dt>
void Verlet::run_stencil_md_many_cuts_waitany_pipelined_helper(int starting_timestep, int dep, int pipeline_stage,
                                                               int start_t, int end_t,
                                                               double **test_f, double **test_x, double **test_v,
                                                               std::vector<MPI_Request>* send_r,
                                                               std::vector<MPI_Request>* recv_r,
                                                               std::vector<std::atomic_flag>& claimed) {
    constexpr int MAX_NEIGHBORS = 26;

    auto& my_queues = curr_dt ? stencilMD->my_queues_many_cuts
                              : stencilMD->my_queues_many_cuts_next_dt;

    std::vector<std::atomic<int>> recv_neighbor_counts(my_queues[dep].size());
    std::unordered_map<int, int> zoid_to_my_queue_idx;

    for (int j = 0; j < my_queues[dep].size(); j++) {
        int zoid_num = my_queues[dep][j].num;
        int num_recv_neighbors = curr_dt ? stencilMD->recv_from_neighbors_many_cuts[zoid_num].size()
                : stencilMD->recv_from_neighbors_many_cuts_next_dt[zoid_num].size();
        recv_neighbor_counts[j] = num_recv_neighbors;
        zoid_to_my_queue_idx[zoid_num] = j;
    }

    cilk_scope {
        if (dep < NUM_DEPS - 1) {
            for (int j = 0; j < my_queues[dep + 1].size(); j++) {
                auto &zoid = my_queues[dep + 1][j];
                int zoid_num = zoid.num;
                cilk_spawn stencilMD->RECEIVE_DATA_ZOID_TO_ZOID_WAITANY_PIPELINED<curr_dt>(dep + 1, zoid_num, pipeline_stage,
                                                                                           recv_r[dep + 1]);
            }

            auto& zoids_to_send_data = curr_dt ? stencilMD->dep_to_send_zoids[dep + 1]
                                               : stencilMD->dep_to_send_zoids_next_dt[dep + 1];

            for (int i = 0; i < zoids_to_send_data.size(); i++) {
                int zoid_num = zoids_to_send_data[i];
                auto& zoid = curr_dt ? stencilMD->zoid_num_to_zoid_many_cuts[zoid_num]
                                     : stencilMD->zoid_num_to_zoid_many_cuts_next_dt[zoid_num];

                cilk_spawn stencilMD->SEND_DATA_ZOID_TO_ZOID_TO_DEP_REVISED_PIPELINED<curr_dt>(zoid, dep + 1,
                                                                                               start_t, end_t, pipeline_stage,
                                                                                               send_r[zoid.num]);
            }
        }

        for (int j = 0; j < my_queues[dep].size(); j++) {
            auto& zoid = my_queues[dep][j];
            if (zoid.no_comm_needed) {
                cilk_spawn run_stencil_md_zoid_many_cuts_no_comm_pipelined<curr_dt>(
                        starting_timestep, dep, zoid, start_t, end_t, pipeline_stage,
                        send_r, test_f, test_x, test_v);
            } else {
                cilk_spawn unpack_self_wrapper_pipelined<curr_dt>(starting_timestep, dep, zoid,
                                                                  start_t, end_t, pipeline_stage,
                                                                  recv_neighbor_counts[j],
                                                                  send_r, test_f, test_x, test_v, claimed);
            }
        }

        int num_wait = 0;
        auto& recv_request_map = curr_dt ? stencilMD->recv_request_idx_to_zoid[dep]
                : stencilMD->recv_request_idx_to_zoid_next_dt[dep];

        constexpr bool USE_WAIT_ANY = true;

        if (USE_WAIT_ANY) {
            while (num_wait < recv_request_map.size()) {
                int idx;
                MPI_Waitany(recv_r[dep].size(), recv_r[dep].data(), &idx, MPI_STATUSES_IGNORE);

                auto [recv_zoid_num, zoid_num] = curr_dt ? stencilMD->recv_request_idx_to_zoid[dep].at(idx)
                        : stencilMD->recv_request_idx_to_zoid_next_dt[dep].at(idx);

                auto& zoid = curr_dt ? stencilMD->zoid_num_to_zoid_many_cuts[zoid_num]
                        : stencilMD->zoid_num_to_zoid_many_cuts_next_dt[zoid_num];

                int my_queue_idx = zoid_to_my_queue_idx.at(zoid_num);

                cilk_spawn unpack_other_wrapper_pipelined<curr_dt>(starting_timestep, dep, zoid,
                                                                recv_zoid_num,
                                                                start_t, end_t, pipeline_stage,
                                                                recv_neighbor_counts[my_queue_idx],
                                                                send_r, test_f, test_x, test_v, claimed);
                num_wait++;
            }
        } else {
            std::vector<int> waitsome_idxs(recv_request_map.size(), 0);
            while (num_wait < recv_request_map.size()) {
                int count;
                MPI_Waitsome(recv_r[dep].size(), recv_r[dep].data(), &count, waitsome_idxs.data(), MPI_STATUSES_IGNORE);

                for (int i = 0; i < count; i++) {
                    int idx = waitsome_idxs[i];
                    assert(idx >= 0 && idx < recv_request_map.size());
                    assert(recv_request_map.count(idx));
                    auto [recv_zoid_num, zoid_num] = curr_dt ? stencilMD->recv_request_idx_to_zoid[dep].at(idx)
                            : stencilMD->recv_request_idx_to_zoid_next_dt[dep].at(idx);

                    auto& zoid = curr_dt ? stencilMD->zoid_num_to_zoid_many_cuts[zoid_num]
                            : stencilMD->zoid_num_to_zoid_many_cuts_next_dt[zoid_num];

                    int my_queue_idx = zoid_to_my_queue_idx.at(zoid_num);

                    cilk_spawn unpack_other_wrapper_pipelined<curr_dt>(starting_timestep, dep, zoid,
                                                                    recv_zoid_num,
                                                                    start_t, end_t, pipeline_stage,
                                                                    recv_neighbor_counts[my_queue_idx],
                                                                    send_r, test_f, test_x, test_v, claimed);
                    num_wait++;
                }
            }
        }
    }

    for (int j = 0; j < my_queues[dep].size(); j++) {
        int zoid_num = my_queues[dep][j].num;
        claimed[zoid_num].clear();
    }
}

template <bool curr_dt>
void Verlet::run_stencil_md_many_cuts_waitany_pipelined_helper_with_proc_to_proc(int starting_timestep, int dep, int pipeline_stage,
                                                               int start_t, int end_t,
                                                               double **test_f, double **test_x, double **test_v,
                                                               std::vector<std::atomic<int>>& recv_neighbor_counters,
                                                               std::vector<std::vector<MPI_Request>>& send_r,
                                                               std::vector<MPI_Request>& send_r_proc_to_proc,
                                                               std::vector<MPI_Request>* recv_r,
                                                               std::vector<std::atomic_flag>& claimed) {

    constexpr int curr_dt_idx = static_cast<int>(curr_dt);

    auto& my_queues = curr_dt ? stencilMD->my_queues_many_cuts
                              : stencilMD->my_queues_many_cuts_next_dt;

    cilk_scope {
        if (dep < NUM_DEPS - 1) {
            for (int j = 0; j < my_queues[dep + 1].size(); j++) {
                auto &zoid = my_queues[dep + 1][j];
                cilk_spawn stencilMD->RECEIVE_DATA_ZOID_TO_ZOID_WAITANY_PIPELINED_ONLY_NEXT_DEP<curr_dt>(dep + 1, zoid.num, pipeline_stage,
                                                                                                         recv_r[dep + 1]);
            }

            cilk_spawn stencilMD->RECEIVE_DATA_PROC_TO_PROC_PIPELINED<curr_dt>(pipeline_stage, dep + 1, recv_r[dep + 1]);
        }

        for (int j = 0; j < my_queues[dep].size(); j++) {
            auto& zoid = my_queues[dep][j];

            if (zoid.no_comm_needed) {
                cilk_spawn run_stencil_md_zoid_many_cuts_no_comm_pipelined_only_next_dep<curr_dt>(
                    starting_timestep, dep, zoid, start_t, end_t, pipeline_stage,
                    send_r, test_f, test_x, test_v);
            } else {
                cilk_spawn unpack_self_wrapper_pipelined_only_next_dep<curr_dt>(starting_timestep, dep, zoid,
                                                                                start_t, end_t, pipeline_stage,
                                                                                recv_neighbor_counters[zoid.num],
                                                                                send_r, test_f, test_x, test_v, claimed);
            }
        }

        int num_wait = 0;
        auto& recv_request_map = stencilMD->recv_request_idx_to_zoid_with_proc_to_proc[curr_dt_idx][pipeline_stage][dep];

        int nrecv_zoid_to_zoid = stencilMD->nrecv_zoid_to_zoid[curr_dt_idx][pipeline_stage][dep];

        constexpr bool USE_WAIT_ANY = false;

        if (USE_WAIT_ANY) {
            while (num_wait < recv_request_map.size()) {
                int idx;
                MPI_Waitany(recv_r[dep].size(), recv_r[dep].data(), &idx, MPI_STATUSES_IGNORE);

                assert(recv_request_map.count(idx));
                auto [recv_zoid_num, zoid_num] = recv_request_map.at(idx);

                if (idx >= nrecv_zoid_to_zoid) {
                    int proc = recv_zoid_num;
                    int send_dep = zoid_num;

                    cilk_spawn unpack_data_pipelined_proc_to_proc<curr_dt>(starting_timestep, dep,
                                                                proc, send_dep,
                                                                start_t, end_t, pipeline_stage, recv_neighbor_counters,
                                                                send_r,
                                                                test_f, test_x, test_v,
                                                                claimed);

                } else {
                    auto& zoid = curr_dt ? stencilMD->zoid_num_to_zoid_many_cuts[zoid_num]
                                        : stencilMD->zoid_num_to_zoid_many_cuts_next_dt[zoid_num];

                    cilk_spawn unpack_other_wrapper_pipelined_only_next_dep<curr_dt>(starting_timestep, dep, zoid,
                                                                                    recv_zoid_num,
                                                                                    start_t, end_t, pipeline_stage,
                                                                                    recv_neighbor_counters[zoid_num],
                                                                                    send_r, test_f, test_x, test_v, claimed);
                }

                num_wait++;
            }
        } else {
            std::vector<int> waitsome_idxs(recv_request_map.size(), 0);
            while (num_wait < recv_request_map.size()) {
                int count;
                MPI_Waitsome(recv_r[dep].size(), recv_r[dep].data(), &count, waitsome_idxs.data(), MPI_STATUSES_IGNORE);

                for (int i = 0; i < count; i++) {
                    int idx = waitsome_idxs[i];
                    assert(idx >= 0 && idx < recv_request_map.size());
                    assert(recv_request_map.count(idx));
                    auto [recv_zoid_num, zoid_num] = recv_request_map.at(idx);

                    if (idx >= nrecv_zoid_to_zoid) {
                        int proc = recv_zoid_num;
                        int send_dep = zoid_num;

                        cilk_spawn unpack_data_pipelined_proc_to_proc<curr_dt>(starting_timestep, dep,
                                                                    proc, send_dep,
                                                                    start_t, end_t, pipeline_stage, recv_neighbor_counters,
                                                                    send_r,
                                                                    test_f, test_x, test_v,
                                                                    claimed);

                    } else {
                        auto& zoid = curr_dt ? stencilMD->zoid_num_to_zoid_many_cuts[zoid_num]
                                            : stencilMD->zoid_num_to_zoid_many_cuts_next_dt[zoid_num];

                        cilk_spawn unpack_other_wrapper_pipelined_only_next_dep<curr_dt>(starting_timestep, dep, zoid,
                                                                                        recv_zoid_num,
                                                                                        start_t, end_t, pipeline_stage,
                                                                                        recv_neighbor_counters[zoid_num],
                                                                                        send_r, test_f, test_x, test_v, claimed);
                    }
                    num_wait++;
                }
            }
        }
    }

    for (int j = 0; j < my_queues[dep].size(); j++) {
        int zoid_num = my_queues[dep][j].num;
        claimed[zoid_num].clear();
    }

    cilk_spawn stencilMD->SEND_DATA_PROC_TO_PROC_PIPELINED<curr_dt>(pipeline_stage, dep, send_r_proc_to_proc);
}

template <bool curr_dt>
void Verlet::run_stencil_md_many_cuts_waitany(int starting_timestep, double **test_f, double **test_x, double **test_v,
                                              std::vector<std::atomic_flag>& claimed) {
    constexpr int MAX_NEIGHBORS = 26;

    auto& my_queues = curr_dt ? stencilMD->my_queues_many_cuts
                              : stencilMD->my_queues_many_cuts_next_dt;

    std::vector<MPI_Request> send_r[stencilMD->NUM_ZOIDS_MANY_CUTS];
    std::vector<MPI_Request> recv_r[NUM_DEPS];
    for (int dep = 1; dep < NUM_DEPS; dep++) {
        if (curr_dt) {
            recv_r[dep].resize(stencilMD->recv_request_idx_to_zoid[dep].size());
        } else {
            recv_r[dep].resize(stencilMD->recv_request_idx_to_zoid_next_dt[dep].size());
        }
    }

    for (int dep = 0; dep < NUM_DEPS - 1; dep++) {
        for (int j = 0; j < my_queues[dep].size(); j++) {
            int zoid_num = my_queues[dep][j].num;
            assert(zoid_num % comm->nprocs == comm->me);
            if (curr_dt) {
                send_r[zoid_num].resize(stencilMD->send_to_neighbors_num_not_in_proc[zoid_num]);
            } else {
                send_r[zoid_num].resize(stencilMD->send_to_neighbors_num_not_in_proc_next_dt[zoid_num]);
            }
        }
    }

    for (int dep = 1; dep < NUM_DEPS; dep++) {
        for (int j = 0; j < my_queues[dep].size(); j++) {
            int zoid_num = my_queues[dep][j].num;
            assert(zoid_num % comm->nprocs == comm->me);
            // stencilMD->RECEIVE_DATA_ZOID_TO_ZOID_WAITANY<curr_dt>(dep, zoid_num, recv_r[dep]);
        }
    }

    int tmp_start_t = 1;
    int tmp_end_t = NUM_TIMESTEPS_IN_PARALLEL + 1;

    for (int dep = 0; dep < NUM_DEPS; dep++) {
        std::vector<std::atomic<int>> recv_neighbor_counts(my_queues[dep].size());
        std::unordered_map<int, int> zoid_to_my_queue_idx;

        for (int j = 0; j < my_queues[dep].size(); j++) {
            int zoid_num = my_queues[dep][j].num;
            int num_recv_neighbors = curr_dt ? stencilMD->recv_from_neighbors_many_cuts[zoid_num].size()
                    : stencilMD->recv_from_neighbors_many_cuts_next_dt[zoid_num].size();
            recv_neighbor_counts[j] = num_recv_neighbors;
            zoid_to_my_queue_idx[zoid_num] = j;
        }

        cilk_scope {
            if (dep < NUM_DEPS - 1) {
                for (int j = 0; j < my_queues[dep + 1].size(); j++) {
                    auto& zoid = my_queues[dep + 1][j];
                    int zoid_num = zoid.num;
                    cilk_spawn stencilMD->RECEIVE_DATA_ZOID_TO_ZOID_WAITANY<curr_dt>(dep + 1, zoid_num, recv_r[dep + 1]);
                }

                auto& zoids_to_send_data = curr_dt ? stencilMD->dep_to_send_zoids[dep + 1]
                                                   : stencilMD->dep_to_send_zoids_next_dt[dep + 1];

                for (int i = 0; i < zoids_to_send_data.size(); i++) {
                    int zoid_num = zoids_to_send_data[i];
                    auto& zoid = curr_dt ? stencilMD->zoid_num_to_zoid_many_cuts[zoid_num]
                                         : stencilMD->zoid_num_to_zoid_many_cuts_next_dt[zoid_num];

                    cilk_spawn stencilMD->SEND_DATA_ZOID_TO_ZOID_TO_DEP_REVISED<curr_dt>(zoid, dep + 1,
                                                                                         tmp_start_t, tmp_end_t,
                                                                                         send_r[zoid.num]);
                }
            }

            for (int j = 0; j < my_queues[dep].size(); j++) {
                auto& zoid = my_queues[dep][j];
                if (zoid.no_comm_needed) {
                    cilk_spawn run_stencil_md_zoid_many_cuts_no_comm<curr_dt>(
                            starting_timestep, dep, zoid, tmp_start_t, tmp_end_t,
                            send_r,
                            test_f, test_x, test_v);
                } else {
                    cilk_spawn unpack_self_wrapper<curr_dt>(starting_timestep, dep, zoid,
                                                            tmp_start_t, tmp_end_t, recv_neighbor_counts[j],
                                                            send_r, test_f, test_x, test_v, claimed);
                }
            }

            int num_wait = 0;
            auto& recv_request_map = curr_dt ? stencilMD->recv_request_idx_to_zoid[dep]
                    : stencilMD->recv_request_idx_to_zoid_next_dt[dep];
            while (num_wait < recv_request_map.size()) {

                if (TIME_STENCILMD_STATES) {
                    struct timeval tv_wait_begin;
                    gettimeofday(&tv_wait_begin, NULL);
                    m.lock();
                    stencil_md_timings.push_back(std::make_tuple("WAIT", "START", 0, tv_wait_begin.tv_sec * MICROSECOND_FACTOR + tv_wait_begin.tv_usec));
                    m.unlock();
                }

                int idx;
                MPI_Waitany(recv_r[dep].size(), recv_r[dep].data(), &idx, MPI_STATUSES_IGNORE);

                if (TIME_STENCILMD_STATES) {
                    struct timeval tv_wait_end;
                    gettimeofday(&tv_wait_end, NULL);
                    m.lock();
                    stencil_md_timings.push_back(std::make_tuple("WAIT", "END", 0, tv_wait_end.tv_sec * MICROSECOND_FACTOR + tv_wait_end.tv_usec));
                    m.unlock();
                }

                auto [recv_zoid_num, zoid_num] = curr_dt ? stencilMD->recv_request_idx_to_zoid[dep].at(idx)
                        : stencilMD->recv_request_idx_to_zoid_next_dt[dep].at(idx);

                auto& zoid = curr_dt ? stencilMD->zoid_num_to_zoid_many_cuts[zoid_num]
                        : stencilMD->zoid_num_to_zoid_many_cuts_next_dt[zoid_num];

                int my_queue_idx = zoid_to_my_queue_idx.at(zoid_num);

                cilk_spawn unpack_other_wrapper<curr_dt>(starting_timestep, dep, zoid,
                                                         recv_zoid_num,
                                                         tmp_start_t, tmp_end_t, recv_neighbor_counts[my_queue_idx],
                                                         send_r, test_f, test_x, test_v, claimed);
                num_wait++;
            }
        }

        for (int j = 0; j < my_queues[dep].size(); j++) {
            int zoid_num = my_queues[dep][j].num;
            claimed[zoid_num].clear();
        }
    }

    for (int dep = 0; dep < NUM_DEPS - 1; dep++) {
        for (int j = 0; j < my_queues[dep].size(); j++) {
            int zoid_num = my_queues[dep][j].num;
            MPI_Waitall(send_r[zoid_num].size(), send_r[zoid_num].data(), MPI_STATUSES_IGNORE);
        }
    }
}

template <bool curr_dt>
void Verlet::run_stencil_md_many_cuts_waitany_with_proc_to_proc(int starting_timestep, double **test_f, double **test_x, double **test_v,
                                                                std::vector<std::atomic<int>>& recv_neighbor_counters,
                                                                std::vector<std::vector<MPI_Request>>& send_r,
                                                                std::vector<std::vector<MPI_Request>>& send_r_proc_to_proc,
                                                                std::vector<std::vector<MPI_Request>>& recv_r, 
                                                                std::vector<std::vector<MPI_Request>>& recv_r_proc_to_proc, 
                                                                std::vector<std::atomic_flag>& claimed) {

    constexpr int curr_dt_idx = static_cast<int>(curr_dt);

    auto& my_queues = curr_dt ? stencilMD->my_queues_many_cuts : stencilMD->my_queues_many_cuts_next_dt;

    for (int dep = 1; dep < NUM_DEPS; dep++) {
        for (int j = 0; j < my_queues[dep].size(); j++) {
            int zoid_num = my_queues[dep][j].num;
            recv_neighbor_counters[zoid_num] = NUM_RECV_NEIGHBORS[dep];
        }
    }

    for (int dep = 0; dep < NUM_DEPS; dep++) {
        for (int j = 0; j < my_queues[dep].size(); j++) {
            int zoid_num = my_queues[dep][j].num;
            claimed[zoid_num].clear();
        }
    }

    for (int dep = 1; dep < NUM_DEPS; dep++) {
        // cilk_spawn stencilMD->RECEIVE_DATA_PROC_TO_PROC_AND_ZOID_TO_ZOID<curr_dt>(dep, recv_r[dep], recv_r_proc_to_proc[dep]);
        /*
        for (int j = 0; j < my_queues[dep].size(); j++) {
            auto &zoid = my_queues[dep][j];
            cilk_spawn stencilMD->RECEIVE_DATA_ZOID_TO_ZOID_WAITANY_PIPELINED_ONLY_NEXT_DEP<curr_dt>(dep, zoid.num, DEFAULT_PIPELINE_STAGE,
                                                                                                         recv_r[dep]);
        }
        */
        cilk_spawn stencilMD->RECEIVE_DATA_PROC_TO_PROC_PIPELINED<curr_dt>(DEFAULT_PIPELINE_STAGE, dep, recv_r[dep]);
    }

    for (int dep = 0; dep < NUM_DEPS; dep++) {
        int64_t total_wait_duration = 0;
        auto begin = std::chrono::high_resolution_clock::now();

        cilk_scope {
            for (int j = 0; j < my_queues[dep].size(); j++) {
                stencilMD->RECEIVE_DATA_ZOID_TO_ZOID_WAITANY_PIPELINED_ONLY_NEXT_DEP<curr_dt>(dep, my_queues[dep][j].num, 
                        DEFAULT_PIPELINE_STAGE, recv_r[dep]);
                auto& zoid = my_queues[dep][j];
                if (zoid.no_comm_needed) {
                    cilk_spawn run_stencil_md_zoid_many_cuts_no_comm_pipelined_only_next_dep<curr_dt>(
                            starting_timestep, dep, zoid, default_start_t, default_end_t,
                            DEFAULT_PIPELINE_STAGE, send_r,
                            test_f, test_x, test_v);
                } else {
                    cilk_spawn unpack_self_wrapper_pipelined_only_next_dep<curr_dt>(starting_timestep, dep, zoid,
                                                                                default_start_t, default_end_t, DEFAULT_PIPELINE_STAGE,
                                                                                recv_neighbor_counters[zoid.num],
                                                                                send_r, test_f, test_x, test_v, claimed);
                }
            }

            int num_wait = 0;
            auto& recv_request_map = stencilMD->recv_request_idx_to_zoid_with_proc_to_proc[curr_dt_idx][DEFAULT_PIPELINE_STAGE][dep];
            int nrecv_zoid_to_zoid = stencilMD->nrecv_zoid_to_zoid[curr_dt_idx][DEFAULT_PIPELINE_STAGE][dep];

            int total_num_wait_on = recv_request_map.size();
            std::vector<int> waitsome_idxs(total_num_wait_on, 0);

            while (num_wait < total_num_wait_on) {
                int count;
                auto begin = std::chrono::high_resolution_clock::now();
                MPI_Waitsome(total_num_wait_on, recv_r[dep].data(), &count, waitsome_idxs.data(), MPI_STATUSES_IGNORE);
                auto end = std::chrono::high_resolution_clock::now();
                auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count();
                total_wait_duration += duration;

                for (int i = 0; i < count; i++) {
                    int idx = waitsome_idxs[i];
                    assert(idx >= 0 && idx < recv_request_map.size());
                    assert(recv_request_map.count(idx));
                    auto [recv_zoid_num, zoid_num] = recv_request_map.at(idx);

                    if (idx >= nrecv_zoid_to_zoid) {
                        int proc = recv_zoid_num;
                        int send_dep = zoid_num;

                        cilk_spawn unpack_data_pipelined_proc_to_proc<curr_dt>(starting_timestep, dep,
                                                                proc, send_dep,
                                                                default_start_t, default_end_t, DEFAULT_PIPELINE_STAGE, 
                                                                recv_neighbor_counters,
                                                                send_r,
                                                                test_f, test_x, test_v,
                                                                claimed);
                    } else {
                        auto& zoid = curr_dt ? stencilMD->zoid_num_to_zoid_many_cuts[zoid_num]
                                            : stencilMD->zoid_num_to_zoid_many_cuts_next_dt[zoid_num];

                        cilk_spawn unpack_other_wrapper_pipelined_only_next_dep<curr_dt>(starting_timestep, dep, zoid,
                                                                                        recv_zoid_num,
                                                                                        default_start_t, default_end_t, DEFAULT_PIPELINE_STAGE,
                                                                                        recv_neighbor_counters[zoid_num],
                                                                                        send_r, test_f, test_x, test_v, claimed);
                    }
                    num_wait++;
                }
            }
            
            /*

            if (dep > 1) {
                int num_wait_proc_to_proc = 0;
                auto& recv_request_map_proc_to_proc = stencilMD->recv_request_idx_to_proc_pair[curr_dt_idx][dep];

                int total_num_wait_proc_to_proc = recv_request_map_proc_to_proc.size();

                // assert(recv_request_map_proc_to_proc.size() == recv_r_proc_to_proc[dep].size());

                std::vector<int> wait_idxs(recv_request_map_proc_to_proc.size(), 0);

                while (num_wait_proc_to_proc < total_num_wait_proc_to_proc) {
                    int num_wait_idxs;
                    MPI_Waitsome(total_num_wait_proc_to_proc, recv_r_proc_to_proc[dep].data(), &num_wait_idxs, wait_idxs.data(), MPI_STATUSES_IGNORE);

                    for (int i = 0; i < num_wait_idxs; i++) {
                        int idx = wait_idxs[i];

                        assert(idx != MPI_UNDEFINED);

                        assert(recv_request_map_proc_to_proc.count(idx));
                        auto& [proc, send_dep] = recv_request_map_proc_to_proc.at(idx);
                        cilk_spawn unpack_data_pipelined_proc_to_proc<curr_dt>(starting_timestep, dep,
                                                                proc, send_dep,
                                                                default_start_t, default_end_t, DEFAULT_PIPELINE_STAGE, 
                                                                recv_neighbor_counters,
                                                                send_r,
                                                                test_f, test_x, test_v,
                                                                claimed);
                    }

                    num_wait_proc_to_proc += num_wait_idxs;
                }
            }

            int num_wait_zoid_to_zoid = 0;
            auto& recv_request_map_zoid_to_zoid = stencilMD->recv_request_idx_to_zoid_pair[curr_dt_idx][dep];
            int total_num_wait_zoid_to_zoid = recv_request_map_zoid_to_zoid.size();

            while (num_wait_zoid_to_zoid < total_num_wait_zoid_to_zoid) {
                int idx;
                MPI_Waitany(total_num_wait_zoid_to_zoid, recv_r[dep].data(), &idx, MPI_STATUSES_IGNORE);

                assert(idx != MPI_UNDEFINED);

                assert(recv_request_map_zoid_to_zoid.count(idx));
                auto [recv_zoid_num, zoid_num] = recv_request_map_zoid_to_zoid.at(idx);

                auto& zoid = curr_dt ? stencilMD->zoid_num_to_zoid_many_cuts[zoid_num]
                        : stencilMD->zoid_num_to_zoid_many_cuts_next_dt[zoid_num];

                cilk_spawn unpack_other_wrapper_pipelined_only_next_dep<curr_dt>(starting_timestep, dep, zoid,
                                                                                recv_zoid_num,
                                                                                default_start_t, default_end_t, DEFAULT_PIPELINE_STAGE,
                                                                                recv_neighbor_counters[zoid_num],
                                                                                send_r, test_f, test_x, test_v, claimed);
                num_wait_zoid_to_zoid++;
            }
            */
        }
        cilk_spawn stencilMD->SEND_DATA_PROC_TO_PROC_PIPELINED<curr_dt>(DEFAULT_PIPELINE_STAGE, dep, send_r_proc_to_proc[dep]);
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count();
        if (comm->me == 0) {
        std::stringstream s1;
        s1 << BOLDMAGENTA << "me: " << comm->me << " dep: " << dep << " total duration: " << duration
        << " wait duration: " << total_wait_duration
        << RESET_COLOR << std::endl;
        std::cout << s1.str();
        }
    }

    for (int dep = 0; dep < NUM_DEPS - 1; dep++) {
        for (int j = 0; j < my_queues[dep].size(); j++) {
            int zoid_num = my_queues[dep][j].num;
            int num_wait_zoid = stencilMD->send_to_neighbors_num_not_in_proc_only_next_dep[curr_dt_idx][zoid_num];
            auto res = MPI_Waitall(send_r[zoid_num].size(), send_r[zoid_num].data(), MPI_STATUSES_IGNORE);
        }
        if (dep < 2) {
            int num_wait_dep = stencilMD->send_dep_to_procs[curr_dt_idx][DEFAULT_PIPELINE_STAGE][dep].size();
            MPI_Waitall(send_r_proc_to_proc[dep].size(), send_r_proc_to_proc[dep].data(), MPI_STATUSES_IGNORE);
        }
    }
}

template <bool curr_dt>
void Verlet::run_stencil_md_many_cuts_unpack_self_wrapper(int starting_timestep, int dep, queue_info& zoid,
    double** test_f, double** test_x, double** test_v,
    std::vector<std::atomic<int>>& zoid_recv_neighbor_counters,
    std::vector<std::atomic<int>>& dep_counters,
    std::vector<std::vector<MPI_Request>>& send_r_zoid_to_zoid,
    std::vector<std::vector<MPI_Request>>& send_r_proc_to_proc,
    std::vector<std::atomic_flag>& zoid_claimed,
    std::vector<std::atomic_flag>& dep_claimed,
    MPIX_Stream_Manager* stream_manager) noexcept {

    int num_neighbors_receive_self = stencilMD->UNPACK_DATA_MANY_CUTS_ZOID_SELF_ONLY_PIPELINED<curr_dt>(zoid, default_start_t, default_end_t, DEFAULT_PIPELINE_STAGE);
    zoid_recv_neighbor_counters[zoid.num] -= num_neighbors_receive_self;
    auto& claimed = zoid_claimed[zoid.num];
    if (zoid_recv_neighbor_counters[zoid.num] == 0) {
        if (!claimed.test(std::memory_order_relaxed) && !claimed.test_and_set(std::memory_order_relaxed)) {
            stencil_md_run_zoid_wrapper<curr_dt>(starting_timestep, dep, zoid, default_start_t, default_end_t,
                zoid_recv_neighbor_counters, dep_counters, send_r_zoid_to_zoid, send_r_proc_to_proc,
                test_f, test_x, test_v, 
                zoid_claimed, dep_claimed, stream_manager);
        }
    }
}


template <bool curr_dt>
void Verlet::run_stencil_md_many_cuts_process_stream(int starting_timestep, int dep, int stream_num,
    double** test_f, double** test_x, double** test_v,
    std::vector<std::atomic<int>>& zoid_recv_neighbor_counters,
    std::vector<std::atomic<int>>& dep_counters,
    std::vector<std::vector<MPI_Request>>& send_r_zoid_to_zoid,
    std::vector<std::vector<MPI_Request>>& send_r_proc_to_proc,
    std::vector<std::vector<std::vector<MPI_Request>>>& recv_r_zoid_to_zoid_streams,
    std::vector<std::atomic_flag>& zoid_claimed,
    std::vector<std::atomic_flag>& dep_claimed,
    MPIX_Stream_Manager* stream_manager,
    std::vector<std::atomic_flag>& zoid_unpack_claimed) noexcept {

    auto& my_queues = curr_dt ? stencilMD->my_queues_many_cuts : stencilMD->my_queues_many_cuts_next_dt;
    constexpr int curr_dt_idx = static_cast<int>(curr_dt);
    auto& zoid_pairs_at_stream = stencilMD->stream_num_to_zoid_pairs[curr_dt_idx][dep][stream_num];
    auto& send_dep_proc_pairs_at_stream = stencilMD->stream_num_to_dep_proc_pairs[curr_dt_idx][dep][stream_num];
    if (dep < NUM_DEPS - 1 && zoid_pairs_at_stream.size() + send_dep_proc_pairs_at_stream.size() == 0) {
        stencilMD->RECEIVE_DATA_PROC_TO_PROC_AND_ZOID_TO_ZOID_STREAMS<curr_dt>(dep + 1, stream_num, DEFAULT_PIPELINE_STAGE, 
            recv_r_zoid_to_zoid_streams[dep + 1][stream_num], stream_manager);
    } else {
        int total_num_wait = zoid_pairs_at_stream.size() + send_dep_proc_pairs_at_stream.size();
        int num_wait = 0;

        auto& all_requests_at_stream = recv_r_zoid_to_zoid_streams[dep][stream_num];
        std::vector<bool> requests_completed(total_num_wait, false);

        while (true) {
            bool all_true = (std::find(requests_completed.cbegin(), requests_completed.cend(), false) == requests_completed.cend());
            if (all_true) {
                break;
            }

            for (int idx = 0; idx < total_num_wait; idx++) {
                if (!requests_completed[idx] && MPIX_Request_is_complete(all_requests_at_stream[idx])) {
                    if (all_requests_at_stream[idx] != MPI_REQUEST_NULL) {
                        MPI_Request_free(&all_requests_at_stream[idx]);
                    }
                    requests_completed[idx] = true;
                    if (idx < zoid_pairs_at_stream.size()) {
                        auto [src_zoid_num, dst_zoid_num] = zoid_pairs_at_stream[idx];
                        auto& zoid = curr_dt ? stencilMD->zoid_num_to_zoid_many_cuts[dst_zoid_num] : stencilMD->zoid_num_to_zoid_many_cuts_next_dt[dst_zoid_num];
                        stencilMD->UNPACK_POS_VEL_MANY_CUTS_ZOID_PIPELINED<curr_dt>(zoid, src_zoid_num, default_start_t, default_end_t, DEFAULT_PIPELINE_STAGE);
                        zoid_recv_neighbor_counters[zoid.num]--;
                        auto& claimed = zoid_claimed[zoid.num];
                        if (zoid_recv_neighbor_counters[zoid.num] == 0
                            && !claimed.test(std::memory_order_relaxed)
                            && !claimed.test_and_set(std::memory_order_relaxed)) {
                                cilk_spawn stencil_md_run_zoid_wrapper<curr_dt>(starting_timestep, dep, zoid, default_start_t, default_end_t,
                                zoid_recv_neighbor_counters, dep_counters, send_r_zoid_to_zoid, send_r_proc_to_proc,
                                test_f, test_x, test_v, 
                            zoid_claimed, dep_claimed, stream_manager);
                        }
                    } else {
                        assert(idx - zoid_pairs_at_stream.size() < send_dep_proc_pairs_at_stream.size());
                        auto& [send_dep, proc] = send_dep_proc_pairs_at_stream[idx - zoid_pairs_at_stream.size()];
                        cilk_spawn unpack_data_proc_to_proc<curr_dt>(starting_timestep, dep,
                                                            proc, send_dep,
                                                            default_start_t, default_end_t, DEFAULT_PIPELINE_STAGE, 
                                                            zoid_recv_neighbor_counters,
                                                            dep_counters,
                                                            send_r_zoid_to_zoid,
                                                            send_r_proc_to_proc,
                                                            test_f, test_x, test_v,
                                                            zoid_claimed, dep_claimed, stream_manager);
                    }
                }
            }

            stream_manager->m[stream_num].lock();
            MPIX_Stream_progress(stream_manager->streams[stream_num]);
            stream_manager->m[stream_num].unlock();

            // check for work to do
            for (int j = 0; j < my_queues[dep].size(); j++) {
                int zoid_num = my_queues[dep][j].num;
                auto& claimed = zoid_unpack_claimed[zoid_num];
                if (!claimed.test(std::memory_order_relaxed) && !claimed.test_and_set(std::memory_order_relaxed)) {
                    auto& zoid = my_queues[dep][j];
                    cilk_spawn run_stencil_md_many_cuts_unpack_self_wrapper<curr_dt>(starting_timestep, dep, zoid,
                        test_f, test_x, test_v,
                        zoid_recv_neighbor_counters, dep_counters,
                        send_r_zoid_to_zoid, send_r_proc_to_proc,
                        zoid_claimed, dep_claimed, stream_manager);
                }
            }
        }

        if (dep < NUM_DEPS - 1) {
            stencilMD->RECEIVE_DATA_PROC_TO_PROC_AND_ZOID_TO_ZOID_STREAMS<curr_dt>(dep + 1, stream_num, DEFAULT_PIPELINE_STAGE,
                recv_r_zoid_to_zoid_streams[dep + 1][stream_num], stream_manager);
        }
    }
}

template <bool curr_dt>
void Verlet::run_stencil_md_many_cuts_process_stream_better_work_queue(int starting_timestep, int dep, int stream_num,
    double** test_f, double** test_x, double** test_v,
    std::vector<std::atomic<int>>& zoid_recv_neighbor_counters,
    std::vector<std::atomic<int>>& dep_counters,
    std::vector<std::vector<MPI_Request>>& send_r_zoid_to_zoid,
    std::vector<std::vector<MPI_Request>>& send_r_proc_to_proc,
    std::vector<std::vector<std::vector<MPI_Request>>>& recv_r_zoid_to_zoid_streams,
    std::vector<std::atomic_flag>& zoid_claimed,
    std::vector<std::atomic_flag>& dep_claimed,
    MPIX_Stream_Manager* stream_manager,
    std::vector<std::atomic_flag*>& zoid_unpack_self_claimed,
    std::vector<std::atomic<bool>>& zoid_done,
    std::vector<std::atomic_flag>& zoid_unpack_claimed) noexcept {

    auto& my_queues = curr_dt ? stencilMD->my_queues_many_cuts : stencilMD->my_queues_many_cuts_next_dt;
    constexpr int curr_dt_idx = static_cast<int>(curr_dt);
    auto& zoid_pairs_at_stream = stencilMD->stream_num_to_zoid_pairs[curr_dt_idx][dep][stream_num];
    auto& send_dep_proc_pairs_at_stream = stencilMD->stream_num_to_dep_proc_pairs[curr_dt_idx][dep][stream_num];
    int total_num_wait = zoid_pairs_at_stream.size() + send_dep_proc_pairs_at_stream.size();
    if (dep < NUM_DEPS - 1 && total_num_wait == 0) {
        stencilMD->RECEIVE_DATA_PROC_TO_PROC_AND_ZOID_TO_ZOID_STREAMS<curr_dt>(dep + 1, stream_num, DEFAULT_PIPELINE_STAGE, 
            recv_r_zoid_to_zoid_streams[dep + 1][stream_num], stream_manager);
    } else {
        int num_wait = 0;
        int num_iter = 0;

        auto& all_requests_at_stream = recv_r_zoid_to_zoid_streams[dep][stream_num];
        std::vector<bool> requests_completed(total_num_wait, false);

        auto loop_begin = MPI_Wtime();

        double stream_time = 0;

        while (true) {
            bool all_true = (std::find(requests_completed.cbegin(), requests_completed.cend(), false) == requests_completed.cend());
            if (all_true) {
                break;
            }

            for (int idx = 0; idx < total_num_wait; idx++) {
                if (!requests_completed[idx] && MPIX_Request_is_complete(all_requests_at_stream[idx])) {
                    if (all_requests_at_stream[idx] != MPI_REQUEST_NULL) {
                        MPI_Request_free(&all_requests_at_stream[idx]);
                    }
                    requests_completed[idx] = true;
                    if (idx < zoid_pairs_at_stream.size()) {
                        auto [src_zoid_num, dst_zoid_num] = zoid_pairs_at_stream[idx];
                        auto& zoid = curr_dt ? stencilMD->zoid_num_to_zoid_many_cuts[dst_zoid_num] : stencilMD->zoid_num_to_zoid_many_cuts_next_dt[dst_zoid_num];
                        stencilMD->UNPACK_POS_VEL_MANY_CUTS_ZOID_PIPELINED<curr_dt>(zoid, src_zoid_num, default_start_t, default_end_t, DEFAULT_PIPELINE_STAGE);
                        zoid_recv_neighbor_counters[zoid.num].fetch_sub(1, std::memory_order_relaxed);
                        auto& claimed = zoid_claimed[zoid.num];
                        if (zoid_recv_neighbor_counters[zoid.num].load(std::memory_order_relaxed) == 0
                            && !claimed.test(std::memory_order_relaxed)
                            && !claimed.test_and_set(std::memory_order_relaxed)) {
                                cilk_spawn stencil_md_run_zoid_wrapper_better_work_queue<curr_dt>(starting_timestep, dep, zoid, default_start_t, default_end_t,
                                zoid_recv_neighbor_counters, dep_counters, send_r_zoid_to_zoid, send_r_proc_to_proc,
                                test_f, test_x, test_v, 
                            zoid_claimed, dep_claimed, stream_manager, zoid_done);
                        }
                    } else {
                        assert(idx - zoid_pairs_at_stream.size() < send_dep_proc_pairs_at_stream.size());
                        auto& [send_dep, proc] = send_dep_proc_pairs_at_stream[idx - zoid_pairs_at_stream.size()];
                        cilk_spawn unpack_data_proc_to_proc_better_work_queue<curr_dt>(starting_timestep, dep,
                                                            proc, send_dep,
                                                            default_start_t, default_end_t, DEFAULT_PIPELINE_STAGE, 
                                                            zoid_recv_neighbor_counters,
                                                            dep_counters,
                                                            send_r_zoid_to_zoid,
                                                            send_r_proc_to_proc,
                                                            test_f, test_x, test_v,
                                                            zoid_claimed, dep_claimed, stream_manager, zoid_done);
                    }
                }
            }

            auto comm_begin = MPI_Wtime();

            if (stream_manager->m[stream_num].try_lock()) {
                MPIX_Stream_progress(stream_manager->streams[stream_num]);
                stream_manager->m[stream_num].unlock();
            }

            auto comm_end = MPI_Wtime();
            stream_time += (comm_end - comm_begin);

            for (int d = dep; d < NUM_DEPS; d++) {
                for (int j = 0; j < my_queues[d].size(); j++) {
                    auto& zoid = my_queues[d][j];
                    int zoid_num = zoid.num;
                    auto& recv_neighbors = curr_dt ? stencilMD->recv_from_neighbors_many_cuts[zoid_num] : stencilMD->recv_from_neighbors_many_cuts_next_dt[zoid_num];
                    for (int i = 0; i < recv_neighbors.size(); i++) {
                        int recv_zoid_num = recv_neighbors[i];
                        if (recv_zoid_num % comm->nprocs == comm->me && zoid_done[recv_zoid_num].load(std::memory_order_relaxed)) {
                            auto& zoid_unpack_self_claimed_flag = zoid_unpack_self_claimed[zoid.num][i];
                            // unpack
                            if (!zoid_unpack_self_claimed_flag.test(std::memory_order_relaxed) && !zoid_unpack_self_claimed_flag.test_and_set(std::memory_order_relaxed)) {
                                cilk_spawn [this](int starting_timestep, int dep, queue_info& zoid, int recv_zoid_num, int recv_idx,
                                    double** test_f, double** test_x, double** test_v,
                                    std::vector<std::atomic<int>>& zoid_recv_neighbor_counters,
                                    std::vector<std::atomic<int>>& dep_counters,
                                    std::vector<std::vector<MPI_Request>>& send_r_zoid_to_zoid,
                                    std::vector<std::vector<MPI_Request>>& send_r_proc_to_proc,
                                    std::vector<std::atomic_flag>& zoid_claimed,
                                    std::vector<std::atomic_flag>& dep_claimed,
                                    MPIX_Stream_Manager* stream_manager,
                                    std::vector<std::atomic_flag*>& zoid_unpack_self_claimed,
                                    std::vector<std::atomic<bool>>& zoid_done) {

                                    int zoid_num = zoid.num;
                                    auto &send_neighbors = curr_dt ? stencilMD->send_to_neighbors_many_cuts[recv_zoid_num]
                                                                : stencilMD->send_to_neighbors_many_cuts_next_dt[recv_zoid_num];
                                    auto find_it = std::find(send_neighbors.begin(), send_neighbors.end(), zoid_num);
                                    assert(find_it != send_neighbors.end());
                                    int find_idx = std::distance(send_neighbors.begin(), find_it);
                                    // DO NOT unpack force
                                    stencilMD->UNPACK_DATA_MANY_CUTS_HELPER_SELF_PIPELINED<curr_dt>(zoid, recv_idx, recv_zoid_num, find_idx, default_start_t, default_end_t, DEFAULT_PIPELINE_STAGE, false);
                                    zoid_recv_neighbor_counters[zoid.num].fetch_sub(1, std::memory_order_relaxed);
                                    auto& claimed = zoid_claimed[zoid.num];
                                    if (zoid_recv_neighbor_counters[zoid.num].load(std::memory_order_relaxed) == 0) {
                                        if (!claimed.test(std::memory_order_relaxed) && !claimed.test_and_set(std::memory_order_relaxed)) {
                                            stencil_md_run_zoid_wrapper_better_work_queue<curr_dt>(starting_timestep, dep, zoid, default_start_t, default_end_t,
                                                zoid_recv_neighbor_counters, dep_counters, send_r_zoid_to_zoid, send_r_proc_to_proc,
                                                test_f, test_x, test_v, 
                                                zoid_claimed, dep_claimed, stream_manager, zoid_done);
                                        }
                                    }
                                }(starting_timestep, d, zoid, recv_zoid_num, i,
                                test_f, test_x, test_v,
                                zoid_recv_neighbor_counters, dep_counters,
                                send_r_zoid_to_zoid, send_r_proc_to_proc,
                                zoid_claimed, dep_claimed,
                                stream_manager, zoid_unpack_self_claimed, zoid_done);
                            }
                        }
                    }
                }
            }

            /*
            // check for work to do
            for (int j = 0; j < my_queues[dep].size(); j++) {
                int zoid_num = my_queues[dep][j].num;
                auto& claimed = zoid_unpack_claimed[zoid_num];
                if (!claimed.test(std::memory_order_relaxed) && !claimed.test_and_set(std::memory_order_relaxed)) {
                    auto& zoid = my_queues[dep][j];
                    cilk_spawn run_stencil_md_many_cuts_unpack_self_wrapper<curr_dt>(starting_timestep, dep, zoid,
                        test_f, test_x, test_v,
                        zoid_recv_neighbor_counters, dep_counters,
                        send_r_zoid_to_zoid, send_r_proc_to_proc,
                        zoid_claimed, dep_claimed, stream_manager);
                }
            }
            */

            num_iter++;
        }

        auto loop_end = MPI_Wtime();
        v_comm_time += (loop_end - loop_begin);
        // std::stringstream s1;
        // s1 << "total loop time: " << (loop_end - loop_begin) << " stream time: " << stream_time << std::endl;
        // std::cout << s1.str();

        // auto comm_end = MPI_Wtime();
        // comm_time += (comm_end - comm_begin);

        if (dep < NUM_DEPS - 1) {
            stencilMD->RECEIVE_DATA_PROC_TO_PROC_AND_ZOID_TO_ZOID_STREAMS<curr_dt>(dep + 1, stream_num, DEFAULT_PIPELINE_STAGE,
                recv_r_zoid_to_zoid_streams[dep + 1][stream_num], stream_manager);
        }
    }
}

__attribute__((noinline)) void start_progress_thread(MPIX_Stream_Manager* manager, std::atomic<bool>* done) noexcept {
    // auto comm_begin = MPI_Wtime();
    // auto w = __cilkrts_get_worker_number();

    while (!done->load(std::memory_order_acquire)) {
        if (manager->global_lock.try_lock()) {
            // cilk_for (int stream_num = 0; stream_num < NUM_STREAMS; stream_num++) {
            for (int stream_num = 0; stream_num < NUM_STREAMS; stream_num++) {
                if (manager->m[stream_num].try_lock()) {
                    MPIX_Stream_progress(manager->streams[stream_num]);
                    manager->m[stream_num].unlock();
                }
            }
            manager->global_lock.unlock();
        }
    }

    // auto comm_end = MPI_Wtime();
    // s_comm_time[w] += (comm_end - comm_begin);
}

template <bool curr_dt>
void Verlet::run_stencil_md_many_cuts_proc_to_proc(int starting_timestep, double **test_f, double **test_x, double **test_v,
                                                    std::vector<std::atomic<int>>& zoid_recv_neighbor_counters,
                                                    std::vector<std::atomic<int>>& dep_counters,
                                                    std::vector<std::vector<MPI_Request>>& send_r_zoid_to_zoid,
                                                    std::vector<std::vector<MPI_Request>>& send_r_proc_to_proc,
                                                    std::vector<std::vector<std::vector<MPI_Request>>>& recv_r_zoid_to_zoid_streams, 
                                                    std::vector<std::vector<MPI_Request>>& recv_r_proc_to_proc, 
                                                    std::vector<std::atomic_flag>& zoid_claimed,
                                                    std::vector<std::atomic_flag>& dep_claimed,
                                                    MPIX_Stream_Manager* stream_manager,
                                                    std::vector<std::atomic_flag*>& zoid_unpack_self_claimed) noexcept {

    constexpr int curr_dt_idx = static_cast<int>(curr_dt);
    auto& my_queues = curr_dt ? stencilMD->my_queues_many_cuts : stencilMD->my_queues_many_cuts_next_dt;

    auto other_begin = MPI_Wtime();

    for (int dep = 0; dep < NUM_DEPS; dep++) {
        for (int j = 0; j < my_queues[dep].size(); j++) {
            int zoid_num = my_queues[dep][j].num;
            zoid_recv_neighbor_counters[zoid_num] = NUM_RECV_NEIGHBORS[dep];
        }
        dep_counters[dep] = my_queues[dep].size();
    }

    for (int dep = 0; dep < NUM_DEPS; dep++) {
        for (int j = 0; j < my_queues[dep].size(); j++) {
            int zoid_num = my_queues[dep][j].num;
            zoid_claimed[zoid_num].clear(std::memory_order_relaxed);
        }
        dep_claimed[dep].clear(std::memory_order_relaxed);
    }

    std::vector<std::vector<int>> dep_to_active_streams(NUM_DEPS);
    for (int dep = 1; dep < NUM_DEPS; dep++) {
        for (int stream_num = 0; stream_num < NUM_STREAMS; stream_num++) {
            auto& zoid_pairs_at_stream = stencilMD->stream_num_to_zoid_pairs[curr_dt_idx][dep][stream_num];
            auto& send_dep_proc_pairs_at_stream = stencilMD->stream_num_to_dep_proc_pairs[curr_dt_idx][dep][stream_num];
            int nrecv = zoid_pairs_at_stream.size() + send_dep_proc_pairs_at_stream.size();
            if (nrecv > 0) {
                dep_to_active_streams[dep].push_back(stream_num);
            }
        }
    }

    std::vector<std::atomic_flag> zoid_unpack_claimed(stencilMD->NUM_ZOIDS_MANY_CUTS);

    auto other_end = MPI_Wtime();
    other_time += (other_end - other_begin);

    constexpr bool USE_BETTER_WORK_QUEUE = true;
    if (USE_BETTER_WORK_QUEUE) {
        cilk_scope {
            auto other_begin = MPI_Wtime();

            std::vector<std::atomic<bool>> zoid_done(stencilMD->NUM_ZOIDS_MANY_CUTS);
            for (int dep = 0; dep < NUM_DEPS; dep++) {
                for (int j = 0; j < my_queues[dep].size(); j++) {
                    auto& zoid = my_queues[dep][j];
                    zoid_done[zoid.num].store(false, std::memory_order_relaxed);

                    int num_recv_neighbors = std::max(stencilMD->recv_from_neighbors_many_cuts[zoid.num].size(), stencilMD->recv_from_neighbors_many_cuts_next_dt[zoid.num].size());
                    for (int i = 0; i < num_recv_neighbors; i++) {
                        zoid_unpack_self_claimed[zoid.num][i].clear(std::memory_order_relaxed);
                    }
                }
            }

            auto other_end = MPI_Wtime();
            other_time += (other_end - other_begin);

            std::atomic<bool> progress_thread_done = false;

            cilk_spawn start_progress_thread(stream_manager, &progress_thread_done);

            for (int dep = 0; dep < NUM_DEPS; dep++) {
                if (dep == 0) {
                    cilk_for (int stream_num = 0; stream_num < NUM_STREAMS; stream_num++) {
                        auto& zoid_pairs_at_stream = stencilMD->stream_num_to_zoid_pairs[curr_dt_idx][dep + 1][stream_num];
                        auto& send_dep_proc_pairs_at_stream = stencilMD->stream_num_to_dep_proc_pairs[curr_dt_idx][dep + 1][stream_num];
                        if (zoid_pairs_at_stream.size() + send_dep_proc_pairs_at_stream.size() > 0) {
                            stencilMD->RECEIVE_DATA_PROC_TO_PROC_AND_ZOID_TO_ZOID_STREAMS<curr_dt>(dep + 1, stream_num, DEFAULT_PIPELINE_STAGE, 
                                recv_r_zoid_to_zoid_streams[dep + 1][stream_num], stream_manager);
                        }
                    }
                }

                cilk_scope {
                    if (dep == 0) {
                        for (int j = 0; j < my_queues[dep].size(); j++) {
                            auto& zoid = my_queues[dep][j];
                            cilk_spawn stencil_md_run_zoid_wrapper_better_work_queue<curr_dt>(starting_timestep, dep, zoid, default_start_t, default_end_t,
                                zoid_recv_neighbor_counters, dep_counters, send_r_zoid_to_zoid, send_r_proc_to_proc,
                                test_f, test_x, test_v, 
                                zoid_claimed, dep_claimed, stream_manager, zoid_done);
                        }
                    } else {
                        for (int stream_num = 0; stream_num < NUM_STREAMS; stream_num++) {
                            cilk_spawn run_stencil_md_many_cuts_process_stream_better_work_queue<curr_dt>(starting_timestep, dep, stream_num,
                                test_f, test_x, test_v,
                                zoid_recv_neighbor_counters, dep_counters,
                                send_r_zoid_to_zoid, send_r_proc_to_proc, 
                                recv_r_zoid_to_zoid_streams, zoid_claimed, dep_claimed, stream_manager, zoid_unpack_self_claimed, zoid_done, zoid_unpack_claimed);
                        }
                    }
                }

                cilk_spawn stencilMD->SEND_DATA_PROC_TO_PROC<curr_dt>(DEFAULT_PIPELINE_STAGE, dep, send_r_proc_to_proc[dep], stream_manager);
            }

            progress_thread_done.store(true, std::memory_order_release);

            auto comm_begin = MPI_Wtime();
            auto w = __cilkrts_get_worker_number();

            for (int dep = 0; dep < NUM_DEPS - 1; dep++) {
                for (int j = 0; j < my_queues[dep].size(); j++) {
                    int zoid_num = my_queues[dep][j].num;
                    stream_manager->global_lock.lock();
                    MPI_Waitall(send_r_zoid_to_zoid[zoid_num].size(), send_r_zoid_to_zoid[zoid_num].data(), MPI_STATUSES_IGNORE);
                    stream_manager->global_lock.unlock();
                }
                if (dep < 2) {
                    stream_manager->global_lock.lock();
                    MPI_Waitall(send_r_proc_to_proc[dep].size(), send_r_proc_to_proc[dep].data(), MPI_STATUSES_IGNORE);
                    stream_manager->global_lock.unlock();
                }
            }
            auto comm_end = MPI_Wtime();
            v_comm_time += (comm_end - comm_begin);
        }

        return;
    }

    for (int dep = 0; dep < NUM_DEPS; dep++) {
        if (dep == 0) {
            cilk_for (int stream_num = 0; stream_num < NUM_STREAMS; stream_num++) {
                auto& zoid_pairs_at_stream = stencilMD->stream_num_to_zoid_pairs[curr_dt_idx][dep + 1][stream_num];
                auto& send_dep_proc_pairs_at_stream = stencilMD->stream_num_to_dep_proc_pairs[curr_dt_idx][dep + 1][stream_num];
                int total_num_wait = zoid_pairs_at_stream.size() + send_dep_proc_pairs_at_stream.size();
                if (total_num_wait > 0) {
                    stencilMD->RECEIVE_DATA_PROC_TO_PROC_AND_ZOID_TO_ZOID_STREAMS<curr_dt>(dep + 1, stream_num, DEFAULT_PIPELINE_STAGE, 
                        recv_r_zoid_to_zoid_streams[dep + 1][stream_num], stream_manager);
                }
            }
        }

        if (dep < NUM_DEPS - 1) {
            cilk_spawn [this](MPIX_Stream_Manager* manager, std::vector<std::atomic<int>>& recv_neighbor_counters, int dep, std::vector<int>& active_streams) {
                while (true) {
                    bool done = true;
                    auto& my_queues = curr_dt ? stencilMD->my_queues_many_cuts[dep] : stencilMD->my_queues_many_cuts_next_dt[dep];
                    int num_wait = 0;
                    for (int j = 0; j < my_queues.size(); j++) {
                        int zoid_num = my_queues[j].num;
                        if (recv_neighbor_counters[zoid_num] > 0) {
                            done = false;
                            break;
                        }
                    }

                    if (done) {
                        break;
                    }

                    if (manager->global_lock.try_lock()) {
                        // for (int i = 0; i < manager->num_streams; i++) {
                        // for (auto& active_stream_num : active_streams) {
                        for (int i = 0; i < active_streams.size(); i++) {
                            int active_stream_num = active_streams[i];
                            if (manager->m[active_stream_num].try_lock()) {
                                MPIX_Stream_progress(manager->streams[active_stream_num]);
                                manager->m[active_stream_num].unlock();
                            }
                        }
                        // }
                        manager->global_lock.unlock();
                    }

                    // std::this_thread::sleep_for(std::chrono::nanoseconds(1));
                }
            }(stream_manager, zoid_recv_neighbor_counters, dep + 1, dep_to_active_streams[dep + 1]);
        }

        cilk_scope {
            if (dep == 0) {
                for (int j = 0; j < my_queues[dep].size(); j++) {
                    auto& zoid = my_queues[dep][j];
                    cilk_spawn stencil_md_run_zoid_wrapper<curr_dt>(starting_timestep, dep, zoid, default_start_t, default_end_t,
                        zoid_recv_neighbor_counters, dep_counters, send_r_zoid_to_zoid, send_r_proc_to_proc,
                        test_f, test_x, test_v, 
                        zoid_claimed, dep_claimed, stream_manager);
                }
            } else {
                for (int stream_num = 0; stream_num < NUM_STREAMS; stream_num++) {
                    cilk_spawn run_stencil_md_many_cuts_process_stream<curr_dt>(starting_timestep, dep, stream_num,
                        test_f, test_x, test_v,
                        zoid_recv_neighbor_counters, dep_counters,
                        send_r_zoid_to_zoid, send_r_proc_to_proc, 
                        recv_r_zoid_to_zoid_streams, zoid_claimed, dep_claimed, stream_manager, zoid_unpack_claimed);
                }
            }
        }
    }

    for (int dep = 0; dep < NUM_DEPS - 1; dep++) {
        for (int j = 0; j < my_queues[dep].size(); j++) {
            int zoid_num = my_queues[dep][j].num;
            stream_manager->global_lock.lock();
            MPI_Waitall(send_r_zoid_to_zoid[zoid_num].size(), send_r_zoid_to_zoid[zoid_num].data(), MPI_STATUSES_IGNORE);
            stream_manager->global_lock.unlock();
        }
        if (dep < 2) {
            stream_manager->global_lock.lock();
            MPI_Waitall(send_r_proc_to_proc[dep].size(), send_r_proc_to_proc[dep].data(), MPI_STATUSES_IGNORE);
            stream_manager->global_lock.unlock();
        }
    }
}


template <bool curr_dt>
void Verlet::run_stencil_md_many_cuts_waitany_pipelined(int starting_timestep, double **test_f, double **test_x, double **test_v,
                                                        std::vector<std::atomic_flag>& claimed, std::vector<std::atomic_flag>& claimed2) {
    constexpr int MAX_NEIGHBORS = 26;

    auto& my_queues = curr_dt ? stencilMD->my_queues_many_cuts
                              : stencilMD->my_queues_many_cuts_next_dt;

    std::vector<MPI_Request> send_r[stencilMD->NUM_ZOIDS_MANY_CUTS];
    std::vector<MPI_Request> send_r2[stencilMD->NUM_ZOIDS_MANY_CUTS];

    std::vector<MPI_Request> recv_r[NUM_DEPS];
    std::vector<MPI_Request> recv_r2[NUM_DEPS];
    for (int dep = 1; dep < NUM_DEPS; dep++) {
        if (curr_dt) {
            recv_r[dep].resize(stencilMD->recv_request_idx_to_zoid[dep].size(), MPI_REQUEST_NULL);
            recv_r2[dep].resize(stencilMD->recv_request_idx_to_zoid[dep].size(), MPI_REQUEST_NULL);
        } else {
            recv_r[dep].resize(stencilMD->recv_request_idx_to_zoid_next_dt[dep].size(), MPI_REQUEST_NULL);
            recv_r2[dep].resize(stencilMD->recv_request_idx_to_zoid_next_dt[dep].size(), MPI_REQUEST_NULL);
        }
    }

    for (int dep = 0; dep < NUM_DEPS - 1; dep++) {
        for (int j = 0; j < my_queues[dep].size(); j++) {
            int zoid_num = my_queues[dep][j].num;
            assert(zoid_num % comm->nprocs == comm->me);
            if (curr_dt) {
                send_r[zoid_num].resize(stencilMD->send_to_neighbors_num_not_in_proc[zoid_num], MPI_REQUEST_NULL);
                send_r2[zoid_num].resize(stencilMD->send_to_neighbors_num_not_in_proc[zoid_num], MPI_REQUEST_NULL);
            } else {
                send_r[zoid_num].resize(stencilMD->send_to_neighbors_num_not_in_proc_next_dt[zoid_num], MPI_REQUEST_NULL);
                send_r2[zoid_num].resize(stencilMD->send_to_neighbors_num_not_in_proc_next_dt[zoid_num], MPI_REQUEST_NULL);
            }
        }
    }

    run_stencil_md_many_cuts_waitany_pipelined_helper<curr_dt>(starting_timestep, 0, 0, start_t[0], end_t[0],
                                                               test_f, test_x, test_v,
                                                               send_r, recv_r, claimed);

    cilk_scope {
            cilk_spawn run_stencil_md_many_cuts_waitany_pipelined_helper<curr_dt>(starting_timestep, 0, 1,
                                                                                  start_t[1], end_t[1],
                                                                                  test_f, test_x, test_v,
                                                                                  send_r2, recv_r2, claimed2);

            run_stencil_md_many_cuts_waitany_pipelined_helper<curr_dt>(starting_timestep, 1, 0,
                                                                       start_t[0], end_t[0],
                                                                       test_f, test_x, test_v,
                                                                       send_r, recv_r, claimed);
    }

    cilk_scope {
            cilk_spawn run_stencil_md_many_cuts_waitany_pipelined_helper<curr_dt>(starting_timestep, 1, 1,
                                                                                  start_t[1], end_t[1],
                                                                                  test_f, test_x, test_v,
                                                                                  send_r2, recv_r2, claimed2);

            run_stencil_md_many_cuts_waitany_pipelined_helper<curr_dt>(starting_timestep, 2, 0,
                                                                       start_t[0], end_t[0],
                                                                       test_f, test_x, test_v,
                                                                       send_r, recv_r, claimed);
    }

    cilk_scope {
            cilk_spawn run_stencil_md_many_cuts_waitany_pipelined_helper<curr_dt>(starting_timestep, 2, 1,
                                                                                  start_t[1], end_t[1],
                                                                                  test_f, test_x, test_v,
                                                                                  send_r2, recv_r2, claimed2);

            run_stencil_md_many_cuts_waitany_pipelined_helper<curr_dt>(starting_timestep, 3, 0,
                                                                       start_t[0], end_t[0],
                                                                       test_f, test_x, test_v,
                                                                       send_r, recv_r, claimed);
    }

    run_stencil_md_many_cuts_waitany_pipelined_helper<curr_dt>(starting_timestep, 3, 1,
                                                               start_t[1], end_t[1],
                                                               test_f, test_x, test_v,
                                                               send_r2, recv_r2, claimed2);

    for (int dep = 0; dep < NUM_DEPS - 1; dep++) {
        for (int j = 0; j < my_queues[dep].size(); j++) {
            int zoid_num = my_queues[dep][j].num;
            MPI_Waitall(send_r[zoid_num].size(), send_r[zoid_num].data(), MPI_STATUSES_IGNORE);
            MPI_Waitall(send_r2[zoid_num].size(), send_r2[zoid_num].data(), MPI_STATUSES_IGNORE);
        }
    }
}

template <bool curr_dt>
void Verlet::run_stencil_md_many_cuts_waitany_pipelined_with_proc_to_proc(int starting_timestep, double **test_f, double **test_x, double **test_v,
                                                                          std::vector<std::vector<MPI_Request>>* send_r,
                                                                          std::vector<std::vector<MPI_Request>>* send_r_proc_to_proc,
                                                                          std::vector<std::atomic<int>>* recv_neighbor_counters,
                                                                          std::vector<std::atomic_flag>& claimed, std::vector<std::atomic_flag>& claimed2) {
    constexpr int curr_dt_idx = static_cast<int>(curr_dt);

    auto& my_queues = curr_dt ? stencilMD->my_queues_many_cuts
                              : stencilMD->my_queues_many_cuts_next_dt;

    // std::vector<std::atomic<int>> recv_neighbors_counters(stencilMD->NUM_ZOIDS_MANY_CUTS);
    // std::vector<std::atomic<int>> recv_neighbors_counters2(stencilMD->NUM_ZOIDS_MANY_CUTS);

    for (int dep = 0; dep < NUM_DEPS - 1; dep++) {
        for (int j = 0; j < my_queues[dep].size(); j++) {
            int zoid_num = my_queues[dep][j].num;
            assert(zoid_num % comm->nprocs == comm->me);
            send_r[0][zoid_num].resize(stencilMD->send_to_neighbors_num_not_in_proc_only_next_dep[curr_dt_idx][zoid_num], MPI_REQUEST_NULL);
            send_r[1][zoid_num].resize(stencilMD->send_to_neighbors_num_not_in_proc_only_next_dep[curr_dt_idx][zoid_num], MPI_REQUEST_NULL);
        }
    }

    for (int dep = 1; dep < NUM_DEPS; dep++) {
        for (int j = 0; j < my_queues[dep].size(); j++) {
            int zoid_num = my_queues[dep][j].num;
            assert(zoid_num % comm->nprocs == comm->me);
            recv_neighbor_counters[0][zoid_num] = curr_dt ? stencilMD->recv_from_neighbors_many_cuts[zoid_num].size()
                : stencilMD->recv_from_neighbors_many_cuts_next_dt[zoid_num].size();
            recv_neighbor_counters[1][zoid_num] = curr_dt ? stencilMD->recv_from_neighbors_many_cuts[zoid_num].size()
                : stencilMD->recv_from_neighbors_many_cuts_next_dt[zoid_num].size();
        }
    }

    for (int dep = 0; dep < NUM_DEPS - 1; dep++) {
        if (curr_dt) {
            send_r_proc_to_proc[0][dep].resize(stencilMD->send_dep_to_procs[curr_dt_idx][0][dep].size(), MPI_REQUEST_NULL);
            send_r_proc_to_proc[1][dep].resize(stencilMD->send_dep_to_procs[curr_dt_idx][1][dep].size(), MPI_REQUEST_NULL);
        } else {
            send_r_proc_to_proc[0][dep].resize(stencilMD->send_dep_to_procs[curr_dt_idx][0][dep].size(), MPI_REQUEST_NULL);
            send_r_proc_to_proc[1][dep].resize(stencilMD->send_dep_to_procs[curr_dt_idx][1][dep].size(), MPI_REQUEST_NULL);
        }
    }

    std::vector<MPI_Request> recv_r[NUM_DEPS];
    std::vector<MPI_Request> recv_r2[NUM_DEPS];

    for (int dep = 1; dep < NUM_DEPS; dep++) {
        recv_r[dep].resize(stencilMD->recv_request_idx_to_zoid_with_proc_to_proc[curr_dt_idx][0][dep].size(), MPI_REQUEST_NULL);
        recv_r2[dep].resize(stencilMD->recv_request_idx_to_zoid_with_proc_to_proc[curr_dt_idx][1][dep].size(), MPI_REQUEST_NULL);
    }

    run_stencil_md_many_cuts_waitany_pipelined_helper_with_proc_to_proc<curr_dt>(starting_timestep, 0, 0,
                                                                                 start_t[0], end_t[0],
                                                                                 test_f, test_x, test_v,
                                                                                 recv_neighbor_counters[0],
                                                                                 send_r[0], send_r_proc_to_proc[0][0], recv_r,
                                                                                 claimed);

    cilk_scope {
        cilk_spawn run_stencil_md_many_cuts_waitany_pipelined_helper_with_proc_to_proc<curr_dt>(starting_timestep, 0, 1,
                                                                                                start_t[1], end_t[1],
                                                                                                test_f, test_x, test_v,
                                                                                                recv_neighbor_counters[1],
                                                                                                send_r[1], send_r_proc_to_proc[1][0], recv_r2,
                                                                                                claimed2);

        run_stencil_md_many_cuts_waitany_pipelined_helper_with_proc_to_proc<curr_dt>(starting_timestep, 1, 0,
                                                                                     start_t[0], end_t[0],
                                                                                     test_f, test_x, test_v,
                                                                                     recv_neighbor_counters[0],
                                                                                     send_r[0], send_r_proc_to_proc[0][1], recv_r,
                                                                                     claimed);
    }

    cilk_scope {
        cilk_spawn run_stencil_md_many_cuts_waitany_pipelined_helper_with_proc_to_proc<curr_dt>(starting_timestep, 1, 1,
        start_t[1], end_t[1],
        test_f, test_x, test_v,
        recv_neighbor_counters[1],
        send_r[1], send_r_proc_to_proc[1][1], recv_r2,
        claimed2);

        run_stencil_md_many_cuts_waitany_pipelined_helper_with_proc_to_proc<curr_dt>(starting_timestep, 2, 0,
        start_t[0], end_t[0],
        test_f, test_x, test_v,
        recv_neighbor_counters[0],
        send_r[0], send_r_proc_to_proc[0][2], recv_r,
        claimed);
    }

    cilk_scope {
        cilk_spawn run_stencil_md_many_cuts_waitany_pipelined_helper_with_proc_to_proc<curr_dt>(starting_timestep, 2, 1,
        start_t[1], end_t[1],
        test_f, test_x, test_v,
        recv_neighbor_counters[1],
        send_r[1], send_r_proc_to_proc[1][2], recv_r2,
        claimed2);

        run_stencil_md_many_cuts_waitany_pipelined_helper_with_proc_to_proc<curr_dt>(starting_timestep, 3, 0,
        start_t[0], end_t[0],
        test_f, test_x, test_v,
        recv_neighbor_counters[0],
        send_r[0], send_r_proc_to_proc[0][3], recv_r,
        claimed);
    }

    run_stencil_md_many_cuts_waitany_pipelined_helper_with_proc_to_proc<curr_dt>(starting_timestep, 3, 1,
                                                                                 start_t[1], end_t[1],
                                                                                 test_f, test_x, test_v,
                                                                                 recv_neighbor_counters[1],
                                                                                 send_r[1], send_r_proc_to_proc[1][3], recv_r2,
                                                                                 claimed2);

    for (int dep = 0; dep < NUM_DEPS - 1; dep++) {
        for (int j = 0; j < my_queues[dep].size(); j++) {
            int zoid_num = my_queues[dep][j].num;
            MPI_Waitall(send_r[0][zoid_num].size(), send_r[0][zoid_num].data(), MPI_STATUSES_IGNORE);
            MPI_Waitall(send_r[1][zoid_num].size(), send_r[1][zoid_num].data(), MPI_STATUSES_IGNORE);
        }

        if (dep < 2) {
            MPI_Waitall(send_r_proc_to_proc[0][dep].size(), send_r_proc_to_proc[0][dep].data(), MPI_STATUSES_IGNORE);
            MPI_Waitall(send_r_proc_to_proc[1][dep].size(), send_r_proc_to_proc[1][dep].data(), MPI_STATUSES_IGNORE);
        }
    }
}

void Verlet::run_stencil_md_many_cuts(int num_timesteps, double** test_f, double** test_x, double** test_v,
                                      std::vector<std::atomic_flag>& claimed, std::vector<std::atomic_flag*>& zoid_unpack_self_claimed) {
    auto other_begin = MPI_Wtime();

    std::vector<std::atomic<int>> zoid_recv_neighbor_counters(stencilMD->NUM_ZOIDS_MANY_CUTS);
    std::vector<std::atomic<int>> dep_counters(NUM_DEPS);
    std::vector<std::vector<MPI_Request>> send_r_zoid_to_zoid(stencilMD->NUM_ZOIDS_MANY_CUTS);
    std::vector<std::vector<MPI_Request>> send_r_proc_to_proc(NUM_DEPS);

    MPIX_Stream_Manager* stream_manager =  new MPIX_Stream_Manager(NUM_STREAMS);

    // std::vector<std::vector<MPI_Request>> recv_r_zoid_to_zoid(NUM_DEPS);
    std::vector<std::vector<MPI_Request>> recv_r_zoid_to_zoid(stencilMD->NUM_ZOIDS_MANY_CUTS);
    std::vector<std::vector<MPI_Request>> recv_r_proc_to_proc(NUM_DEPS);

    std::vector<std::vector<std::vector<MPI_Request>>> recv_r_zoid_to_zoid_streams(NUM_DEPS);

    std::vector<std::atomic_flag> dep_claimed(NUM_DEPS);

    int max_zoids_per_dep = (stencilMD->NUM_ZOIDS_MANY_CUTS / 8) * 3 / comm->nprocs;
    if (comm->me == 0) {
        std::cout << "max zoids per dep: " << max_zoids_per_dep << std::endl;
    }

    constexpr int MAX_NEIGHBORS = 6;

    for (int dep = 0; dep < NUM_DEPS; dep++) {
        for (int j = 0; j < stencilMD->my_queues_many_cuts[dep].size(); j++) {
            int zoid_num = stencilMD->my_queues_many_cuts[dep][j].num;
            send_r_zoid_to_zoid[zoid_num].resize(MAX_NEIGHBORS, MPI_REQUEST_NULL);
            recv_r_zoid_to_zoid[zoid_num].resize(MAX_NEIGHBORS, MPI_REQUEST_NULL);
        }

        send_r_proc_to_proc[dep].resize(comm->nprocs, MPI_REQUEST_NULL);
        recv_r_proc_to_proc[dep].resize(comm->nprocs, MPI_REQUEST_NULL);

        recv_r_zoid_to_zoid_streams[dep].resize(NUM_STREAMS);
        for (int i = 0; i < NUM_STREAMS; i++) {
            recv_r_zoid_to_zoid_streams[dep][i].resize(max_zoids_per_dep * MAX_NEIGHBORS, MPI_REQUEST_NULL);
        }
    }

    auto other_end = MPI_Wtime();
    other_time += (other_end - other_begin);

    for (int t = 0; t < num_timesteps; t += 2 * NUM_TIMESTEPS_IN_PARALLEL) {
        // run_stencil_md_many_cuts_helper<true>(t, test_f, test_x, test_v);
        // run_stencil_md_many_cuts_helper<false>(t, test_f, test_x, test_v);
        // run_stencil_md_many_cuts_waitany<true>(t, test_f, test_x, test_v, claimed);
        // run_stencil_md_many_cuts_waitany<false>(t, test_f, test_x, test_v, claimed);
        // run_stencil_md_many_cuts_waitany_spawn_wait_loop<true>(t, test_f, test_x, test_v, claimed);
        // run_stencil_md_many_cuts_waitany_spawn_wait_loop<false>(t, test_f, test_x, test_v, claimed);
        // run_stencil_md_many_cuts_waitany_with_proc_to_proc<true>(t, test_f, test_x, test_v, 
        //     recv_neighbor_counters, send_r, send_r_proc_to_proc,
        //     recv_r, recv_r_proc_to_proc, claimed);
        // run_stencil_md_many_cuts_waitany_with_proc_to_proc<false>(t, test_f, test_x, test_v, 
        //     recv_neighbor_counters, send_r, send_r_proc_to_proc,
        //     recv_r, recv_r_proc_to_proc, claimed);
        run_stencil_md_many_cuts_proc_to_proc<true>(t, test_f, test_x, test_v,
            zoid_recv_neighbor_counters, dep_counters,
            // send_r_zoid_to_zoid, send_r_proc_to_proc,
            send_r_zoid_to_zoid, send_r_proc_to_proc,
            recv_r_zoid_to_zoid_streams, recv_r_proc_to_proc,
            claimed, dep_claimed, stream_manager, zoid_unpack_self_claimed);
        run_stencil_md_many_cuts_proc_to_proc<false>(t, test_f, test_x, test_v,
            zoid_recv_neighbor_counters, dep_counters,
            send_r_zoid_to_zoid, send_r_proc_to_proc,
            recv_r_zoid_to_zoid_streams, recv_r_proc_to_proc,
            claimed, dep_claimed, stream_manager, zoid_unpack_self_claimed);
    }

    delete stream_manager;
}

void Verlet::run_stencil_md_many_cuts_pipelined(int num_timesteps, double** test_f, double** test_x, double** test_v,
                                                std::vector<std::atomic_flag>& claimed, std::vector<std::atomic_flag>& claimed2) {

    constexpr bool WITH_PROC_TO_PROC = true;

    if (WITH_PROC_TO_PROC) {
        std::vector<std::vector<MPI_Request>> send_r[NUM_PIPELINE_STAGES] = {
            std::vector<std::vector<MPI_Request>>(stencilMD->NUM_ZOIDS_MANY_CUTS),
            std::vector<std::vector<MPI_Request>>(stencilMD->NUM_ZOIDS_MANY_CUTS)
        };

        std::vector<std::vector<MPI_Request>> send_r_proc_to_proc[NUM_PIPELINE_STAGES] = {
            std::vector<std::vector<MPI_Request>>(NUM_DEPS),
            std::vector<std::vector<MPI_Request>>(NUM_DEPS)
        };

        // send_r[0].resize(stencilMD->NUM_ZOIDS_MANY_CUTS); send_r[1].resize(stencilMD->NUM_ZOIDS_MANY_CUTS);
        // send_r_proc_to_proc[0].resize(NUM_DEPS); send_r_proc_to_proc[1].resize(NUM_DEPS);

        std::vector<std::atomic<int>> recv_neighbor_counters[NUM_PIPELINE_STAGES] = {
            std::vector<std::atomic<int>>(stencilMD->NUM_ZOIDS_MANY_CUTS),
            std::vector<std::atomic<int>>(stencilMD->NUM_ZOIDS_MANY_CUTS)
        };

        for (int t = 0; t < num_timesteps; t += 2 * NUM_TIMESTEPS_IN_PARALLEL) {
            // run_stencil_md_many_cuts_waitany_pipelined<true>(t, test_f, test_x, test_v, claimed, claimed2);
            // run_stencil_md_many_cuts_waitany_pipelined<false>(t, test_f, test_x, test_v, claimed, claimed2);
            run_stencil_md_many_cuts_waitany_pipelined_with_proc_to_proc<true>(t, test_f, test_x, test_v,
                                                                            send_r, send_r_proc_to_proc,
                                                                            recv_neighbor_counters,
                                                                            claimed, claimed2);
            run_stencil_md_many_cuts_waitany_pipelined_with_proc_to_proc<false>(t, test_f, test_x, test_v,
                                                                                send_r, send_r_proc_to_proc,
                                                                                recv_neighbor_counters,
                                                                                claimed, claimed2);
        }
    } else {
        for (int t = 0; t < num_timesteps; t += 2 * NUM_TIMESTEPS_IN_PARALLEL) {
            run_stencil_md_many_cuts_waitany_pipelined<true>(t, test_f, test_x, test_v, claimed, claimed2);
            run_stencil_md_many_cuts_waitany_pipelined<false>(t, test_f, test_x, test_v, claimed, claimed2);
        }
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
