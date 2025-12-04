//
// Created by Ryan Deng on 5/7/23.
//

#pragma once

#include <array>
#include <atomic>
#include <mpi.h>
#include <map>
#include <cassert>
#include <chrono>
#include <set>
#include <deque>
#include <iostream>

//the following are UBUNTU/LINUX, and MacOS ONLY terminal color codes.
#define RESET_COLOR   "\033[0m"
#define BLACK   "\033[30m"      /* Black */
#define RED     "\033[31m"      /* Red */
#define GREEN   "\033[32m"      /* Green */
#define YELLOW  "\033[33m"      /* Yellow */
#define BLUE    "\033[34m"      /* Blue */
#define MAGENTA "\033[35m"      /* Magenta */
#define CYAN    "\033[36m"      /* Cyan */
#define WHITE   "\033[37m"      /* White */
#define BOLDBLACK   "\033[1m\033[30m"      /* Bold Black */
#define BOLDRED     "\033[1m\033[31m"      /* Bold Red */
#define BOLDGREEN   "\033[1m\033[32m"      /* Bold Green */
#define BOLDYELLOW  "\033[1m\033[33m"      /* Bold Yellow */
#define BOLDBLUE    "\033[1m\033[34m"      /* Bold Blue */
#define BOLDMAGENTA "\033[1m\033[35m"      /* Bold Magenta */
#define BOLDCYAN    "\033[1m\033[36m"      /* Bold Cyan */
#define BOLDWHITE   "\033[1m\033[37m"      /* Bold White */

constexpr int LAMMPS_SEND_LOCAL = -1;
constexpr int LAMMPS_SEND_GHOST = -2;

constexpr int LEFT = 0;
constexpr int RIGHT = 1;
constexpr int MIDDLE = 2;
constexpr int PBC = 3;

constexpr int BOND_FENE = 0;
constexpr int LJ = 1;
constexpr int DPD = 2;
constexpr int SW = 3;
constexpr int TERSOFF = 4;
constexpr int EAM = 5;

// constexpr int EXPERIMENT = BOND_FENE;
// constexpr int EXPERIMENT = LJ;
constexpr int EXPERIMENT = DPD;
// constexpr int EXPERIMENT = SW;
// constexpr int EXPERIMENT = TERSOFF;

constexpr int NUM_DEPS = 4;

constexpr int NUM_ZOIDS = 4 * 4 * 4;

constexpr int NUM_TIMESTEPS_IN_PARALLEL = 4;
// constexpr double ADDITIONAL_CUTOFF = 0.4 + 1e-10;
// constexpr double ADDITIONAL_CUTOFF = 0.3 + 1e-10;
constexpr double ADDITIONAL_CUTOFF = 0.5 + 1e-10;
// constexpr double ADDITIONAL_CUTOFF = 2.0 + 1e-10;

// this is for the potential, used when there is a multi-body potential
// constexpr double CUTOFF = 3.77;
// constexpr double CUTOFF = 3.2;
// constexpr double ALLEGRO_CUTOFF_RADIUS = 2 * CUTOFF;
// constexpr double ALLEGRO_CUTOFF_RADIUS = 1.12;
//  constexpr double ALLEGRO_CUTOFF_RADIUS = 2.5;
constexpr double ALLEGRO_CUTOFF_RADIUS = 1.0;

constexpr bool DEBUG_SEND_RECV_DATA = false;

constexpr bool TEST_AGAINST_LAMMPS = false;

constexpr double ALLEGRO_SLOPE = ALLEGRO_CUTOFF_RADIUS + ADDITIONAL_CUTOFF;

constexpr bool ONLY_RUN_LAMMPS = false;

constexpr bool ONLY_RUN_STENCIL_MD = true;

constexpr bool USE_BOND = true;

constexpr bool USE_PIPELINE = false;
constexpr int DEFAULT_PIPELINE_STAGE = 0;
constexpr int default_start_t = 1;
constexpr int default_end_t = NUM_TIMESTEPS_IN_PARALLEL + 1;
constexpr int NUM_PIPELINE_STAGES = 2;
constexpr int start_t[NUM_PIPELINE_STAGES] = {1, NUM_TIMESTEPS_IN_PARALLEL / 2 + 1};
constexpr int end_t[NUM_PIPELINE_STAGES] = {NUM_TIMESTEPS_IN_PARALLEL / 2 + 1, NUM_TIMESTEPS_IN_PARALLEL + 1};

using dbl3_t_stencil_md = struct { double x,y,z; };

using IDX_3D = std::array<int, 3>;

constexpr int MODIFY_GRAINSIZE = 1024;

constexpr bool USE_NEWTON = true;

const std::map<IDX_3D, int> zoid_to_num_map = {
        {{LEFT,  LEFT,  LEFT},  0},
        {{RIGHT, RIGHT, RIGHT}, 1},
        {{LEFT,  LEFT,  RIGHT}, 2},
        {{RIGHT, RIGHT, LEFT},     3},
        {{LEFT, RIGHT, LEFT},      4},
        {{RIGHT, LEFT, LEFT},      5},
        {{LEFT, RIGHT, RIGHT},     6},
        {{RIGHT, LEFT, RIGHT},     7},

        // begin dep 1
        // group 0
        {{LEFT, LEFT, MIDDLE},     8},
        {{LEFT, MIDDLE, LEFT},     16},
        {{MIDDLE, LEFT, LEFT},     24},

        // group 1
        {{RIGHT, RIGHT, PBC},      9},
        {{RIGHT, PBC, RIGHT},      17},
        {{PBC, RIGHT, RIGHT},      25},

        // group 2
        {{LEFT, LEFT, PBC},        10},
        {{LEFT, MIDDLE, RIGHT},    18},
        {{MIDDLE, LEFT, RIGHT},    26},

        // group 3
        {{RIGHT, RIGHT, MIDDLE},   11},
        {{RIGHT, PBC, LEFT},       19},
        {{PBC, RIGHT, LEFT},       27},

        // group 4
        {{LEFT, RIGHT, MIDDLE},    12},
        {{LEFT, PBC, LEFT},        20},
        {{MIDDLE, RIGHT, LEFT},    28},

        // group 5
        {{RIGHT, LEFT, MIDDLE},    13},
        {{RIGHT, MIDDLE, LEFT},    21},
        {{PBC, LEFT, LEFT},        29},

        // group 6
        {{LEFT, RIGHT, PBC},       14},
        {{LEFT, PBC, RIGHT},       22},
        {{MIDDLE, RIGHT, RIGHT},   30},

        // group 7
        {{RIGHT, LEFT, PBC},       15},
        {{RIGHT, MIDDLE, RIGHT},   23},
        {{PBC, LEFT, RIGHT},       31},

        // begin dep 2
        // group 0
        {{LEFT, MIDDLE, MIDDLE},   32},
        {{MIDDLE, LEFT, MIDDLE},   40},
        {{MIDDLE, MIDDLE, LEFT},   48},

        // group 1
        {{RIGHT, PBC, PBC},        33},
        {{PBC, RIGHT, PBC},        41},
        {{PBC, PBC, RIGHT},        49},

        // group 2
        {{LEFT, MIDDLE, PBC},      34},
        {{MIDDLE, LEFT, PBC},      42},
        {{MIDDLE, MIDDLE, RIGHT},  50},

        // group 3
        {{RIGHT, PBC, MIDDLE},     35},
        {{PBC, RIGHT, MIDDLE},     43},
        {{PBC, PBC, LEFT},         51},

        // group 4
        {{LEFT, PBC, MIDDLE},      36},
        {{MIDDLE, RIGHT, MIDDLE},  44},
        {{MIDDLE, PBC, LEFT},      52},

        // group 5
        {{RIGHT, MIDDLE, MIDDLE},  37},
        {{PBC, LEFT, MIDDLE},      45},
        {{PBC, MIDDLE, LEFT},      53},

        // group 6
        {{LEFT, PBC, PBC},         38},
        {{MIDDLE, RIGHT, PBC},     46},
        {{MIDDLE, PBC, RIGHT},     54},


        // group 7
        {{RIGHT, MIDDLE, PBC},     39},
        {{PBC, LEFT, PBC},         47},
        {{PBC, MIDDLE, RIGHT},     55},

        // begin dep3
        {{MIDDLE, MIDDLE, MIDDLE}, 56},

        {{PBC, PBC, PBC},          57},

        {{MIDDLE, MIDDLE, PBC},    58},

        {{PBC, PBC, MIDDLE},       59},

        {{MIDDLE, PBC, MIDDLE},    60},

        {{PBC, MIDDLE, MIDDLE},    61},

        {{MIDDLE, PBC, PBC},       62},

        {{PBC, MIDDLE, PBC},       63},
};

struct cut_info {
  double lower;
  double upper;
  double slope_lower;
  double slope_upper;
};

struct cuts {
  cut_info cuts[3];
};

struct spinlock {
    std::atomic<bool> lock_ = {0};

    void lock() noexcept {
        for (;;) {
            // Optimistically assume the lock is free on the first try
            if (!lock_.exchange(true, std::memory_order_acquire)) {
                return;
            }
            // Wait for lock to be released without generating cache misses
            while (lock_.load(std::memory_order_relaxed)) {
                // Issue X86 PAUSE or ARM YIELD instruction to reduce contention between
                // hyper-threads
#ifdef __SSE__
                __builtin_ia32_pause();
#endif
#ifdef __aarch64__
                __builtin_arm_yield();
#endif
            }
        }
    }

    bool try_lock() noexcept {
        // First do a relaxed load to check if lock is free in order to prevent
        // unnecessary cache misses if someone does while(!try_lock())
        return !lock_.load(std::memory_order_relaxed) &&
               !lock_.exchange(true, std::memory_order_acquire);
    }

    void unlock() noexcept {
        lock_.store(false, std::memory_order_release);
    }
};

typedef struct cuts cuts_t;

constexpr int DOUBLE_BUFFERING = 2;
constexpr int NUM_DIMENSIONS = 3;

// struct that holds information for queue
struct queue_info {
  /* Start stuff for 2 timesteps */
  std::array<double, 3>* lo;
  std::array<double, 3>* hi;

  bool no_comm_needed;

  std::vector<dbl3_t_stencil_md>* x_stencil_md;
  std::vector<dbl3_t_stencil_md>* v_stencil_md;
  std::vector<dbl3_t_stencil_md>* eval_f_stencil_md;
  std::vector<dbl3_t_stencil_md>* f_stencil_md;

  std::vector<int>* tag_stencil_md;
  std::vector<int>* type_stencil_md;
  std::vector<int>* mask_stencil_md;
  std::vector<int>* image_stencil_md;
  spinlock** spinlocks_stencil_md;
  std::atomic_flag** claimed_flags_stencil_md;

  std::vector<int>* local_idxs_per_timestep;

  std::vector<std::vector<int>>* neighbor_list;
  std::vector<std::vector<std::pair<int, int>>>* bond_list;
  std::vector<std::tuple<int, int, int>>* bond_list_modified;

  // TODO:
  std::vector<int>* send_force_idxs_double_buffering_flattened;
  std::vector<int>** send_pos_idxs_double_buffering_flattened;
  std::vector<int>* send_vel_idxs_double_buffering_flattened;

  std::vector<int>* recv_force_idxs_double_buffering_flattened;
  std::vector<int>** recv_pos_idxs_double_buffering_flattened;
  std::vector<int>* recv_vel_idxs_double_buffering_flattened;

  std::vector<int>** send_force_idxs_double_buffering_flattened_pipelined;
  std::vector<int>*** send_pos_idxs_double_buffering_flattened_pipelined;
  std::vector<int>*** send_vel_idxs_double_buffering_flattened_pipelined;

  std::vector<int>** recv_force_idxs_double_buffering_flattened_pipelined;
  std::vector<int>*** recv_pos_idxs_double_buffering_flattened_pipelined;
  std::vector<int>*** recv_vel_idxs_double_buffering_flattened_pipelined;

  std::vector<int>** send_force_idxs_double_buffering;
  std::vector<int>** recv_force_idxs_double_buffering;

  std::vector<int>** send_pos_idxs_double_buffering;
  std::vector<int>** recv_pos_idxs_double_buffering;

  std::vector<int>** send_vel_idxs_double_buffering;
  std::vector<int>** recv_vel_idxs_double_buffering;

  std::vector<int>** send_rho_idxs_double_buffering;
  std::vector<int>** recv_rho_idxs_double_buffering;
  std::vector<int>** send_fp_idxs_double_buffering;
  std::vector<int>** recv_fp_idxs_double_buffering;
  /* end for two timesteps */

  // Many-body potentials
  std::vector<std::vector<int>>* neigh_short;
  std::vector<int>* local_and_one_hop_ghost_idxs_per_timestep;

  // EAM
  std::vector<double>* rho_stencil_md;
  std::vector<double>* fp_stencil_md;

  // Use extra-memory
  dbl3_t_stencil_md** per_worker_force_updates;
  std::vector<int>* global_to_local_idx;
  std::vector<int>* local_to_global_idx;

  int t0;
  int t1;
  int dim;
  cuts_t zoid;
  int num;
  int where[3];
};

int get_zoid_dep(int);

int get_zoid_dep_next_dt(int);

bool is_close(int *, int *);

bool is_close_test(int *, int *);

bool is_close_test_next_dt(int *, int *);

bool is_dep(int *, int *);

bool is_dep_inverted(int *, int *);

void get_zoids(double slope, double *lo, double *hi, std::deque<queue_info> *queues);

void print_cuts(const cuts_t &);

int get_segments(const std::vector<int>&, std::vector<int>&, std::vector<int>&, bool print=false);

int get_mpi_tag(int dst, int src, int start_timestep=0, int end_timestep=0);


