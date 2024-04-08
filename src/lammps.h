/* -*- c++ -*- ----------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   https://www.lammps.org/, Sandia National Laboratories
   LAMMPS development team: developers@lammps.org

   Copyright (2003) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under
   the GNU General Public License.

   See the README file in the top-level LAMMPS directory.
------------------------------------------------------------------------- */

#ifndef LMP_LAMMPS_H
#define LMP_LAMMPS_H

#include <cstdio>
#include <mpi.h>
#include <vector>
#include <array>
#include <set>
#include <unordered_set>
#include <unordered_map>
#include "stencil_md_utils.h"
// #include <torch/torch.h>

namespace LAMMPS_NS {

class LAMMPS {
 public:
  // ptrs to fundamental LAMMPS classes
  class Memory *memory;            // memory allocation functions
  class Error *error;              // error handling
  class Universe *universe;        // universe of processors
  class Input *input;              // input script processing
                                   // ptrs to top-level LAMMPS-specific classes
  class Atom *atom;                // atom-based quantities
  class Update *update;            // integrators/minimizers
  class Neighbor *neighbor;        // neighbor lists
  class Comm *comm;                // inter-processor communication
  class Domain *domain;            // simulation box
  class Force *force;              // inter-particle forces
  class Modify *modify;            // fixes and computes
  class Group *group;              // groups of atoms
  class Output *output;            // thermo/dump/restart
  class Timer *timer;              // CPU timing info
                                   //
  class KokkosLMP *kokkos;         // KOKKOS accelerator class
  class AtomKokkos *atomKK;        // KOKKOS version of Atom class
  class MemoryKokkos *memoryKK;    // KOKKOS version of Memory class
  class Python *python;            // Python interface
  class CiteMe *citeme;            // handle citation info

  class StencilMD *stencilMD;

  // Stencil MD classes
  // std::vector<class Neighbor*> neighbor_stencil_md;
  // std::vector<class Domain*> domain_stencil_md;
  std::vector<std::array<class Atom*, NUM_TIMESTEPS_IN_PARALLEL + 1>> atom_stencil_md;
  std::vector<std::array<class AtomKokkos*, NUM_TIMESTEPS_IN_PARALLEL + 1>> atom_kokkos_stencil_md;
  std::vector<class Comm*> comm_stencil_md;

  std::vector<std::array<class Domain*, NUM_TIMESTEPS_IN_PARALLEL + 1>> domain_stencil_md;
  std::vector<std::array<class Neighbor*, NUM_TIMESTEPS_IN_PARALLEL + 1>> neighbor_stencil_md;
  std::vector<std::array<class Force*, NUM_TIMESTEPS_IN_PARALLEL + 1>> force_stencil_md;

  std::vector<std::array<class Domain*, NUM_TIMESTEPS_IN_PARALLEL + 1>> domain_stencil_md_next_dt;
  std::vector<std::array<class Neighbor*, NUM_TIMESTEPS_IN_PARALLEL + 1>> neighbor_stencil_md_next_dt;
  std::vector<std::array<class Force*, NUM_TIMESTEPS_IN_PARALLEL + 1>> force_stencil_md_next_dt;

  std::vector<std::array<class Modify*, NUM_TIMESTEPS_IN_PARALLEL + 1>> modify_stencil_md_omp;

  // need a separate comm, but can reuse neighbor lists
  // need a separate comm for the send_lists, as sending to "next" zoid is different compared to sending to "prev" zoid as I walk down the
  // array
  std::vector<class Comm*> comm_stencil_md_next_dt;

  std::vector<class Modify*> modify_stencil_md;
  std::vector<class Update*> update_stencil_md;
  std::deque<queue_info> queues[NUM_DEPS];

  std::deque<queue_info> queues_next_dt[NUM_DEPS];

  // TODO: use this, replace send_to_next_dt and recv_from_next_dt with this as well
  std::vector<int>* send_to_neighbors;
  std::vector<int>* recv_from_neighbors;

  std::unordered_map<int, std::vector<std::pair<int, int>>> recv_zoid_to_my_zoids;
  std::unordered_map<int, std::vector<std::pair<int, int>>> recv_zoid_to_my_zoids_next_dt;

  std::vector<int>* send_to_neighbors_next_dt;
  std::vector<int>* recv_from_neighbors_next_dt;

  std::unordered_set<int>* send_to_neighbors_procs;
  std::vector<int> recv_from_neighbors_procs;

  std::unordered_set<int>* send_to_neighbors_procs_next_dt;
  std::vector<int> recv_from_neighbors_procs_next_dt;

  int* num_recv_force_from_zoid[NUM_TIMESTEPS_IN_PARALLEL + 1];
  int* num_recv_pos_from_zoid[NUM_TIMESTEPS_IN_PARALLEL + 1];
  int* num_recv_vel_from_zoid[NUM_TIMESTEPS_IN_PARALLEL + 1];

  int* num_recv_force_from_zoid_next_dt[NUM_TIMESTEPS_IN_PARALLEL + 1];
  int* num_recv_pos_from_zoid_next_dt[NUM_TIMESTEPS_IN_PARALLEL + 1];
  int* num_recv_vel_from_zoid_next_dt[NUM_TIMESTEPS_IN_PARALLEL + 1];

  int* zoid_num_to_idx;
  queue_info* zoid_num_to_zoid;
  queue_info* zoid_num_to_zoid_next_dt;

  /*
  torch::jit::Module lmp_model;
  std::unordered_map<std::string, std::string> lmp_model_metadata = {
        {"config", ""},
        {"nequip_version", ""},
        {"r_max", ""},
        {"n_species", ""},
        {"type_names", ""},
        {"_jit_bailout_depth", ""},
        {"_jit_fusion_strategy", ""},
        {"allow_tf32", ""}
  };
  */

  const char *version;    // LAMMPS version string = date
  int num_ver;            // numeric version id derived from *version*
                          // that is constructed so that will be greater
                          // for newer versions in numeric or string
                          // value comparisons
                          //
  MPI_Comm world;         // MPI communicator
  FILE *infile;           // infile
  FILE *screen;           // screen output
  FILE *logfile;          // logfile
                          //
  double initclock;       // wall clock at instantiation
  int skiprunflag;        // 1 inserts timer command to skip run and minimize loops

  char *suffix, *suffix2, *suffixp;    // suffixes to add to input script style names
  int suffix_enable;                   // 1 if suffixes are enabled, 0 if disabled
  char *exename;                       // pointer to argv[0]
                                       //
  char ***packargs;                    // arguments for cmdline package commands
  int num_package;                     // number of cmdline package commands
                                       //
  int clientserver;                    // 0 = neither, 1 = client, 2 = server
  void *cslib;                         // client/server messaging via CSlib
  MPI_Comm cscomm;                     // MPI comm for client+server in mpi/one mode

  const char *match_style(const char *style, const char *name);
  static const char *installed_packages[];
  static bool is_installed_pkg(const char *pkg);

  static bool has_git_info();
  static const char *git_commit();
  static const char *git_branch();
  static const char *git_descriptor();

  LAMMPS(int, char **, MPI_Comm);
  ~LAMMPS();
  void create();
  void post_create();
  void init();
  void destroy();
  void print_config(FILE *);    // print compile time settings

  void read_model() { assert(false); }

 private:
  struct package_styles_lists *pkg_lists;
  void init_pkg_lists();
  void help();
  /// Default constructor. Declared private to prohibit its use
  LAMMPS(){};
  /// Copy constructor. Declared private to prohibit its use
  LAMMPS(const LAMMPS &){};
};

}    // namespace LAMMPS_NS

#endif
