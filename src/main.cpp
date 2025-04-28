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

#include "lammps.h"

#include "accelerator_kokkos.h"
#include "input.h"
#include "lmppython.h"
#include "pthread.h"
#include <cilk/cilk_api.h>
#include <sstream>
#include "CLI11.hpp"
#include "stencil_md_utils.h"

#ifdef __linux__
#include "stencil_md_utils.h"
#include <cilk/cilk_set_affinity.h>
#endif

#if defined(LAMMPS_EXCEPTIONS)
#include "exceptions.h"
#endif

#include <cstdlib>
#include <mpi.h>

#if defined(LAMMPS_TRAP_FPE) && defined(_GNU_SOURCE)
#include <fenv.h>
#endif

// import MolSSI Driver Interface library
#if defined(LMP_MDI)
#include <mdi.h>
#endif
#include <unistd.h>
#include <csignal>

using namespace LAMMPS_NS;

/* ----------------------------------------------------------------------
   main program to drive LAMMPS
------------------------------------------------------------------------- */

int main(int argc, char **argv)
{
  // MPI_Init(&argc, &argv);
  // TODO: Ryan stencil md, change this back if needed?
  #undef _OPENMP
  int provided;
  MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &provided);
  assert(provided >= MPI_THREAD_MULTIPLE);
  if (provided < MPI_THREAD_MULTIPLE) {
    std::cout << "COULD NOT PROVIDE MPI_THREAD_MULTIPLE" << std::endl;
    return 0;
  }

  CLI::App app{"StencilMD YAML file"};
  argv = app.ensure_utf8(argv);

  std::string config_filename = "default";
  app.add_option("-c,--config", config_filename, "config file for StencilMD");

  CLI11_PARSE(app, argc, argv);

#ifdef __linux__
    if (!ONLY_RUN_LAMMPS) {
        constexpr bool USE_MULTI_SOCKET = true;
        if (USE_MULTI_SOCKET) {
            auto calling_thread = pthread_self();

            int world_size;
            MPI_Comm_size(MPI_COMM_WORLD, &world_size);

            int rank;
            MPI_Comm_rank(MPI_COMM_WORLD, &rank);

            cilk_for (int i = 0; i < 1e6; i++) {
                if (rand() == 0) {
                    std::cout << "i: " << i << std::endl;
                }
            }

            int nworkers = __cilkrts_get_nworkers();

            // assume each process gets 1 progress thread

            auto* cpusets = new cpu_set_t[nworkers];
            constexpr int NUM_CORES_PER_SOCKET = 24;
            constexpr int NUM_CORES_PER_NODE = 24 * 2;

            int num_processes_per_node = NUM_CORES_PER_NODE / (nworkers + 1);
            int num_nodes = world_size / num_processes_per_node;
            int num_processes_per_socket = num_processes_per_node / 2;

            int rank_within_node = rank % num_processes_per_node;

            int start;
            if (rank_within_node >= num_processes_per_socket) {
                start = (rank_within_node - num_processes_per_socket) * (nworkers + 1) + NUM_CORES_PER_SOCKET;
            } else {
                start = rank_within_node * (nworkers + 1);
            }

            std::stringstream workers_str;
            for (int w = 0; w < nworkers; w++) {
                CPU_ZERO(&cpusets[w]);
                CPU_SET(start + w, &cpusets[w]);
                workers_str << start + w << " ";
            }

            std::cout << "rank: " << rank << " workers: " << workers_str.str() << std::endl;

            set_worker_affinity(nworkers, cpusets, calling_thread);
            delete[] cpusets;
        }

            /*
            if (world_size == 16) {
                auto* cpusets = new cpu_set_t[nworkers];

                constexpr int NUM_CORES_PER_SOCKET = 24;

                if (rank % 2 == 0) {
                    for (int i = 0; i < nworkers; i++) {
                        CPU_ZERO(&cpusets[i]);
                        CPU_SET(i, &cpusets[i]);
                    }
                } else {
                    for (int i = 0; i < nworkers; i++) {
                        CPU_ZERO(&cpusets[i]);
                        CPU_SET(i + NUM_CORES_PER_SOCKET, &cpusets[i]);
                    }
                }

                set_worker_affinity(nworkers, cpusets, calling_thread);

                delete[] cpusets;
            } else if (world_size == 32) {
                auto* cpusets = new cpu_set_t[nworkers];

                constexpr int NUM_CORES_PER_SOCKET = 24;

                if (rank % 4 == 0) {
                    for (int i = 0; i < nworkers; i++) {
                        CPU_ZERO(&cpusets[i]);
                        CPU_SET(i, &cpusets[i]);
                    }
                } else if (rank % 4 == 1) {
                    for (int i = 0; i < nworkers; i++) {
                        CPU_ZERO(&cpusets[i]);
                        CPU_SET(i + nworkers + 1, &cpusets[i]);
                    }
                } else if (rank % 4 == 2) {
                    for (int i = 0; i < nworkers; i++) {
                        CPU_ZERO(&cpusets[i]);
                        CPU_SET(i + NUM_CORES_PER_SOCKET, &cpusets[i]);
                    }
                } else if (rank % 4 == 3) {
                    for (int i = 0; i < nworkers; i++) {
                        CPU_ZERO(&cpusets[i]);
                        CPU_SET(i + NUM_CORES_PER_SOCKET + nworkers + 1, &cpusets[i]);
                    }
                }

                set_worker_affinity(nworkers, cpusets, calling_thread);

                delete[] cpusets;
            } else if (world_size == 64) {
                auto* cpusets = new cpu_set_t[nworkers];

                constexpr int NUM_CORES_PER_SOCKET = 24;

                constexpr bool USE_HYPERTHREADING = true;

                if (USE_HYPERTHREADING) {
                    constexpr int HYPERTHREAD = 48;
                    assert(nworkers % 2 == 0);
                    int half = nworkers / 2;

                    if (rank % 8 == 0) {
                        for (int i = 0; i < half; i++) {
                            CPU_ZERO(&cpusets[i]);
                            CPU_SET(i, &cpusets[i]);

                            CPU_ZERO(&cpusets[i + half]);
                            CPU_SET(i + HYPERTHREAD, &cpusets[i + half]);

                            if (rank < 8) {
                                std::stringstream s1;
                                s1 << "assign thread: " << i << " to worker: " << i << " thread: " << i + HYPERTHREAD << " to worker: " << i + nworkers / 2 << std::endl;
                                std::cout << s1.str();
                            }
                        }
                    } else if (rank % 8 == 1) {
                        for (int i = 0; i < half; i++) {
                            CPU_ZERO(&cpusets[i]);
                            CPU_SET(i + half + 1, &cpusets[i]);

                            CPU_ZERO(&cpusets[i + half]);
                            CPU_SET(i + half + 1 + HYPERTHREAD, &cpusets[i + half]);

                            if (rank < 8) {
                                std::stringstream s1;
                                s1 << "assign thread: " << i + half + 1 << " to worker: " << i
                                << " thread: " << i + half + 1 + HYPERTHREAD << " to worker: " << i + half << std::endl;
                                std::cout << s1.str();
                            }
                        }
                    } else if (rank % 8 == 2) {
                        for (int i = 0; i < half; i++) {
                            CPU_ZERO(&cpusets[i]);
                            CPU_SET(i + 2 * (half + 1), &cpusets[i]);

                            CPU_ZERO(&cpusets[i + nworkers / 2]);
                            CPU_SET(i + 2 * (half + 1) + HYPERTHREAD, &cpusets[i + half]);

                            if (rank < 8) {
                                std::stringstream s1;
                                s1 << "assign thread: " << i + 2 * (half + 1) << " to worker: " << i
                                << " thread: " << i + 2 * (half + 1) + HYPERTHREAD << " to worker: " << i + half << std::endl;
                                std::cout << s1.str();
                            }
                        }
                    } else if (rank % 8 == 3) {
                        for (int i = 0; i < half; i++) {
                            CPU_ZERO(&cpusets[i]);
                            CPU_SET(i + 3 * (half + 1), &cpusets[i]);

                            CPU_ZERO(&cpusets[i + nworkers / 2]);
                            CPU_SET(i + 3 * (half + 1) + HYPERTHREAD, &cpusets[i + nworkers / 2]);

                            if (rank < 8) {
                                std::stringstream s1;
                                s1 << "assign thread: " << i + 3 * (half + 1) << " to worker: " << i
                                << " thread: " << i + 3 * (half + 1) + HYPERTHREAD << " to worker: " << i + half << std::endl;
                                std::cout << s1.str();
                            }
                        }
                    } else if (rank % 8 == 4) {
                        for (int i = 0; i < half; i++) {
                            CPU_ZERO(&cpusets[i]);
                            CPU_SET(i + NUM_CORES_PER_SOCKET, &cpusets[i]);

                            CPU_ZERO(&cpusets[i + half]);
                            CPU_SET(i + NUM_CORES_PER_SOCKET + HYPERTHREAD, &cpusets[i + half]);

                            if (rank < 8) {
                                std::stringstream s1;
                                s1 << "assign thread: " << i + NUM_CORES_PER_SOCKET << " to worker: " << i
                                << " thread: " << i + NUM_CORES_PER_SOCKET + HYPERTHREAD << " to worker: " << i + half << std::endl;
                                std::cout << s1.str();
                            }
                        }
                    } else if (rank % 8 == 5) {
                        for (int i = 0; i < half; i++) {
                            CPU_ZERO(&cpusets[i]);
                            CPU_SET(i + NUM_CORES_PER_SOCKET + half + 1, &cpusets[i]);

                            CPU_ZERO(&cpusets[i + half]);
                            CPU_SET(i + NUM_CORES_PER_SOCKET + half + 1 + HYPERTHREAD, &cpusets[i + half]);

                            if (rank < 8) {
                                std::stringstream s1;
                                s1 << "assign thread: " << i + NUM_CORES_PER_SOCKET + half + 1 << " to worker: " << i
                                << " thread: " << i + NUM_CORES_PER_SOCKET + half + 1 + HYPERTHREAD << " to worker: " << i + half << std::endl;
                                std::cout << s1.str();
                            }
                        }
                    } else if (rank % 8 == 6) {
                        for (int i = 0; i < half; i++) {
                            CPU_ZERO(&cpusets[i]);
                            CPU_SET(i + NUM_CORES_PER_SOCKET + 2 * (half + 1), &cpusets[i]);

                            CPU_ZERO(&cpusets[i + half]);
                            CPU_SET(i + NUM_CORES_PER_SOCKET + 2 * (half + 1) + HYPERTHREAD, &cpusets[i + half]);

                            if (rank < 8) {
                                std::stringstream s1;
                                s1 << "assign thread: " << i + NUM_CORES_PER_SOCKET + 2 * (half + 1) << " to worker: " << i
                                << " thread: " << i + NUM_CORES_PER_SOCKET + 2 * (half + 1) + HYPERTHREAD << " to worker: " << i + half << std::endl;
                                std::cout << s1.str();
                            }
                        }
                    } else if (rank % 8 == 7) {
                        for (int i = 0; i < half; i++) {
                            CPU_ZERO(&cpusets[i]);
                            CPU_SET(i + NUM_CORES_PER_SOCKET + 3 * (half + 1), &cpusets[i]);

                            CPU_ZERO(&cpusets[i + half]);
                            CPU_SET(i + NUM_CORES_PER_SOCKET + 3 * (half + 1) + HYPERTHREAD, &cpusets[i + half]);

                            if (rank < 8) {
                                std::stringstream s1;
                                s1 << "assign thread: " << i + NUM_CORES_PER_SOCKET + 3 * (half + 1) << " to worker: " << i
                                << " thread: " << i + NUM_CORES_PER_SOCKET + 3 * (half + 1) + HYPERTHREAD << " to worker: " << i + half << std::endl;
                                std::cout << s1.str();
                            }
                        }
                    }
                } else {
                    if (rank % 8 == 0) {
                        for (int i = 0; i < nworkers; i++) {
                            CPU_ZERO(&cpusets[i]);
                            CPU_SET(i, &cpusets[i]);
                        }
                    } else if (rank % 8 == 1) {
                        for (int i = 0; i < nworkers; i++) {
                            CPU_ZERO(&cpusets[i]);
                            CPU_SET(i + nworkers + 1, &cpusets[i]);
                        }
                    } else if (rank % 8 == 2) {
                        for (int i = 0; i < nworkers; i++) {
                            CPU_ZERO(&cpusets[i]);
                            CPU_SET(i + 2 * (nworkers + 1), &cpusets[i]);
                        }
                    } else if (rank % 8 == 3) {
                        for (int i = 0; i < nworkers; i++) {
                            CPU_ZERO(&cpusets[i]);
                            CPU_SET(i + 3 * (nworkers + 1), &cpusets[i]);
                        }
                    } else if (rank % 8 == 4) {
                        for (int i = 0; i < nworkers; i++) {
                            CPU_ZERO(&cpusets[i]);
                            CPU_SET(i + NUM_CORES_PER_SOCKET, &cpusets[i]);
                        }
                    } else if (rank % 8 == 5) {
                        for (int i = 0; i < nworkers; i++) {
                            CPU_ZERO(&cpusets[i]);
                            CPU_SET(i + NUM_CORES_PER_SOCKET + nworkers + 1, &cpusets[i]);
                        }
                    } else if (rank % 8 == 6) {
                        for (int i = 0; i < nworkers; i++) {
                            CPU_ZERO(&cpusets[i]);
                            CPU_SET(i + NUM_CORES_PER_SOCKET + 2 * (nworkers + 1), &cpusets[i]);
                        }
                    } else if (rank % 8 == 7) {
                        for (int i = 0; i < nworkers; i++) {
                            CPU_ZERO(&cpusets[i]);
                            CPU_SET(i + NUM_CORES_PER_SOCKET + 3 * (nworkers + 1), &cpusets[i]);
                        }
                    }
                }

                set_worker_affinity(nworkers, cpusets, calling_thread);

                delete[] cpusets;
            }
        } else {
            auto calling_thread = pthread_self();

            cilk_for (int i = 0; i < 1e6; i++) {
                if (rand() == 0) {
                    std::cout << "i: " << i << std::endl;
                }
            }

            int nworkers = __cilkrts_get_nworkers();

            auto* cpusets = new cpu_set_t[nworkers];

            constexpr int NUM_CORES_PER_SOCKET = 24;

            for (int i = 0; i < nworkers; i++) {
                CPU_ZERO(&cpusets[i]);
                CPU_SET(i + NUM_CORES_PER_SOCKET, &cpusets[i]);
            }

            set_worker_affinity(nworkers, cpusets, calling_thread);

            delete[] cpusets;
        }
        */
    }
#endif

  MPI_Comm lammps_comm = MPI_COMM_WORLD;

#if defined(LMP_MDI)
  // initialize MDI interface, if compiled in

  int mdi_flag;
  if (MDI_Init(&argc, &argv)) MPI_Abort(MPI_COMM_WORLD, 1);
  if (MDI_Initialized(&mdi_flag)) MPI_Abort(MPI_COMM_WORLD, 1);

  // get the MPI communicator that spans all ranks running LAMMPS
  // when using MDI, this may be a subset of MPI_COMM_WORLD

  if (mdi_flag)
    if (MDI_MPI_get_world_comm(&lammps_comm)) MPI_Abort(MPI_COMM_WORLD, 1);
#endif

#if defined(LAMMPS_TRAP_FPE) && defined(_GNU_SOURCE)
  // enable trapping selected floating point exceptions.
  // this uses GNU extensions and is only tested on Linux
  // therefore we make it depend on -D_GNU_SOURCE, too.
  fesetenv(FE_NOMASK_ENV);
  fedisableexcept(FE_ALL_EXCEPT);
  feenableexcept(FE_DIVBYZERO);
  feenableexcept(FE_INVALID);
  feenableexcept(FE_OVERFLOW);
#endif

#ifdef LAMMPS_EXCEPTIONS
  try {
    auto lammps = new LAMMPS(argc, argv, lammps_comm);
    lammps->input->file();
    delete lammps;
  } catch (LAMMPSAbortException &ae) {
    KokkosLMP::finalize();
    Python::finalize();
    MPI_Abort(ae.universe, 1);
  } catch (LAMMPSException &) {
    KokkosLMP::finalize();
    Python::finalize();
    MPI_Barrier(lammps_comm);
    MPI_Finalize();
    exit(1);
  } catch (fmt::format_error &fe) {
    fprintf(stderr, "fmt::format_error: %s\n", fe.what());
    KokkosLMP::finalize();
    Python::finalize();
    MPI_Abort(MPI_COMM_WORLD, 1);
    exit(1);
  }
#else
  try {
    auto lammps = new LAMMPS(argc, argv, lammps_comm);
    lammps->input->file();
    delete lammps;
  } catch (fmt::format_error &fe) {
    fprintf(stderr, "fmt::format_error: %s\n", fe.what());
    KokkosLMP::finalize();
    Python::finalize();
    MPI_Abort(MPI_COMM_WORLD, 1);
    exit(1);
  }
#endif
  KokkosLMP::finalize();
  Python::finalize();
  MPI_Barrier(lammps_comm);
  MPI_Finalize();
}
