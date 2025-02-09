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

#ifdef __linux__
    constexpr bool USE_MULTI_SOCKET = false;
    if (USE_MULTI_SOCKET) {
        auto calling_thread = pthread_self();

        int rank;
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);

        cilk_for (int i = 0; i < 1e6; i++) {
            if (rand() == 0) {
                std::cout << "i: " << i << std::endl;
            }
        }

        int nworkers = __cilkrts_get_nworkers();

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
