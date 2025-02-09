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

// Used for spatial sorting
/*
#include <CGAL/spatial_sort.h>
#include <CGAL/Exact_predicates_inexact_constructions_kernel.h>
#include <CGAL/point_generators_3.h>
#include <CGAL/hilbert_sort.h>
#include <CGAL/Spatial_sort_traits_adapter_3.h>

typedef CGAL::Simple_cartesian<double> K;
typedef K::Point_3                                          Point;
typedef CGAL::Spatial_sort_traits_adapter_3<K,
        CGAL::Pointer_property_map<Point>::type> Search_traits;
typedef std::pair<Point,int>              Point_with_info;
typedef std::vector<Point_with_info>      Data_vector;

typedef CGAL::Spatial_sort_traits_adapter_3<K,
        CGAL::First_of_pair_property_map<Point_with_info>
> Search_traits_pair;
*/

/*
//property map and get as friend
// to be allowed to use private member
class Vect_ppmap{
    const Data_vector& points;
public:
    //classical typedefs
    typedef Data_vector::size_type key_type;
    typedef Point_d value_type;
    typedef const value_type& reference;
    typedef boost::readable_property_map_tag category;
    Vect_ppmap(const Data_vector& points_):points(points_){}
    friend reference get(const Vect_ppmap& vmap, key_type i) {
        return vmap.points[i].first;
    }
};

typedef CGAL::Spatial_sort_traits_adapter_3<K,Vect_ppmap>   Search_traits_pair;
*/

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

constexpr bool LOCAL_SEGMENT_TYPE = true;
constexpr bool GHOST_SEGMENT_TYPE = false;

constexpr bool SEND_DATA_PROCESS_LOCAL = true;
constexpr bool SEND_DATA_PROCESS_GHOST = false;

constexpr bool RECV_DATA_PROCESS_LOCAL = true;
constexpr bool RECV_DATA_PROCESS_GHOST = false;

constexpr int NUM_DEPS = 4;

constexpr int NUM_DEPS_BINS = 8;

constexpr int NUM_ZOIDS = 4 * 4 * 4;

constexpr int NUM_TIMESTEPS_IN_PARALLEL = 4;
// constexpr double ADDITIONAL_CUTOFF = 0.4001;
constexpr double ADDITIONAL_CUTOFF = 0.4;

constexpr double ALLEGRO_CUTOFF_RADIUS = 1.12;

constexpr double MIDDLE_ZOID_WIDTH_RATIO = 0.5;

constexpr bool DEBUG_SEND_RECV_DATA = true;

constexpr bool TEST_AGAINST_LAMMPS = true;

constexpr bool PURELY_LOCAL_POTENTIAL = true;

constexpr bool USE_FAKE_COMPUTE_TEMP = true;

constexpr double ALLEGRO_SLOPE = ALLEGRO_CUTOFF_RADIUS + ADDITIONAL_CUTOFF;

constexpr bool TRY_PRECOMPUTE_RELEVANT_ATOM_IDX = false;

constexpr bool DEBUG = true;

constexpr int NUM_WORKERS_PER_THREAD = 512;

constexpr int NUM_ATOMS_PER_WORKER = 128;

constexpr bool ONLY_RUN_LAMMPS = false;

constexpr bool ONLY_RUN_STENCIL_MD = false;

constexpr bool LAMMPS_USE_CILK = false;

constexpr bool LAMMPS_USE_BINS = false;

constexpr bool TIME_STENCIL_MD = true;

constexpr bool USE_BOND = true;

constexpr int NUM_PIPELINE_STAGES = 2;

constexpr bool USE_ATOMICS = false;

constexpr int NUM_BINS = 1;

constexpr int LAMMPS_NUM_REGIONS = 4;

constexpr bool PAIR_USE_BINS = true;

constexpr bool SORT_BINS_BASED_ON_LOCAL_IDX = false;

using dbl3_t_stencil_md = struct { double x,y,z; };

using IDX_3D = std::array<int, 3>;

constexpr int MODIFY_GRAINSIZE = 1024;
constexpr int MAX_NEIGHBORS_PER_ATOM = 20;

const std::map<IDX_3D, int> partition_to_dep = {
        {{LEFT,   LEFT,   LEFT},   0},
        {{LEFT,   LEFT,   RIGHT},  0},
        {{LEFT,   RIGHT,  LEFT},   0},
        {{RIGHT,  LEFT,   LEFT},   0},
        {{RIGHT,  RIGHT,  RIGHT},  0},
        {{RIGHT,  RIGHT,  LEFT},   0},
        {{RIGHT,  LEFT,   RIGHT},  0},
        {{LEFT,   RIGHT,  RIGHT},  0},

        {{LEFT,   LEFT,   MIDDLE}, 1},
        {{LEFT,   RIGHT,  MIDDLE}, 1},
        {{RIGHT,  LEFT,   MIDDLE}, 1},
        {{RIGHT,  RIGHT,  MIDDLE}, 1},

        {{LEFT,   MIDDLE, LEFT},   2},
        {{LEFT,   MIDDLE, RIGHT},  2},
        {{RIGHT,  MIDDLE, LEFT},   2},
        {{RIGHT,  MIDDLE, RIGHT},  2},

        {{MIDDLE, LEFT,   LEFT},   3},
        {{MIDDLE, LEFT,   RIGHT},  3},
        {{MIDDLE, RIGHT,  LEFT},   3},
        {{MIDDLE, RIGHT,  RIGHT},  3},

        {{MIDDLE, MIDDLE, LEFT},   4},
        {{MIDDLE, MIDDLE, RIGHT},  4},

        {{MIDDLE, LEFT,   MIDDLE}, 5},
        {{MIDDLE, RIGHT,  MIDDLE}, 5},

        {{LEFT,   MIDDLE, MIDDLE}, 6},
        {{RIGHT,  MIDDLE, MIDDLE}, 6},

        {{MIDDLE, MIDDLE, MIDDLE}, 7},
};

const std::map<IDX_3D, int> lammps_partition_to_dep = {
        {{LEFT,   LEFT,   LEFT},   0},
        {{LEFT,   LEFT,   RIGHT},  0},
        {{LEFT,   RIGHT,  LEFT},   0},
        {{RIGHT,  LEFT,   LEFT},   0},
        {{RIGHT,  RIGHT,  RIGHT},  0},
        {{RIGHT,  RIGHT,  LEFT},   0},
        {{RIGHT,  LEFT,   RIGHT},  0},
        {{LEFT,   RIGHT,  RIGHT},  0},

        {{LEFT,   LEFT,   MIDDLE}, 1},
        {{LEFT,   RIGHT,  MIDDLE}, 1},
        {{RIGHT,  LEFT,   MIDDLE}, 1},
        {{RIGHT,  RIGHT,  MIDDLE}, 1},
        {{LEFT,   LEFT,   PBC}, 1},
        {{LEFT,   RIGHT,  PBC}, 1},
        {{RIGHT,  LEFT,   PBC}, 1},
        {{RIGHT,  RIGHT,  PBC}, 1},

        {{LEFT,   MIDDLE, LEFT},   2},
        {{LEFT,   MIDDLE, RIGHT},  2},
        {{RIGHT,  MIDDLE, LEFT},   2},
        {{RIGHT,  MIDDLE, RIGHT},  2},
        {{LEFT,   PBC, LEFT},   2},
        {{LEFT,   PBC, RIGHT},  2},
        {{RIGHT,  PBC, LEFT},   2},
        {{RIGHT,  PBC, RIGHT},  2},

        {{MIDDLE, LEFT,   LEFT},   3},
        {{MIDDLE, LEFT,   RIGHT},  3},
        {{MIDDLE, RIGHT,  LEFT},   3},
        {{MIDDLE, RIGHT,  RIGHT},  3},
        {{PBC, LEFT,   LEFT},   3},
        {{PBC, LEFT,   RIGHT},  3},
        {{PBC, RIGHT,  LEFT},   3},
        {{PBC, RIGHT,  RIGHT},  3},

        {{MIDDLE, MIDDLE, LEFT},   4},
        {{MIDDLE, MIDDLE, RIGHT},  4},
        {{PBC, PBC, LEFT},   4},
        {{PBC, PBC, RIGHT},  4},
        {{MIDDLE, PBC, LEFT},   4},
        {{MIDDLE, PBC, RIGHT},  4},
        {{PBC, MIDDLE, LEFT},   4},
        {{PBC, MIDDLE, RIGHT},  4},

        {{MIDDLE, LEFT,   MIDDLE}, 5},
        {{MIDDLE, RIGHT,  MIDDLE}, 5},
        {{MIDDLE, LEFT,   PBC}, 5},
        {{MIDDLE, RIGHT,  PBC}, 5},
        {{PBC, LEFT,   MIDDLE}, 5},
        {{PBC, RIGHT,  MIDDLE}, 5},
        {{PBC, LEFT,   PBC}, 5},
        {{PBC, RIGHT,  PBC}, 5},

        {{LEFT,   MIDDLE, MIDDLE}, 6},
        {{RIGHT,  MIDDLE, MIDDLE}, 6},
        {{LEFT,   PBC, MIDDLE}, 6},
        {{RIGHT,  PBC, MIDDLE}, 6},
        {{LEFT,   MIDDLE, PBC}, 6},
        {{RIGHT,  MIDDLE, PBC}, 6},
        {{LEFT,   PBC, PBC}, 6},
        {{RIGHT,  PBC, PBC}, 6},

        {{MIDDLE, MIDDLE, MIDDLE}, 7},
        {{MIDDLE, MIDDLE, PBC}, 7},
        {{MIDDLE, PBC, MIDDLE}, 7},
        {{PBC, MIDDLE, MIDDLE}, 7},
        {{PBC, PBC, MIDDLE}, 7},
        {{PBC, MIDDLE, PBC}, 7},
        {{MIDDLE, PBC, PBC}, 7},
        {{PBC, PBC, PBC}, 7},
};

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

/*
const std::map<std::tuple<int, int, int>, int> zoid_to_num_map = {
        {std::make_tuple(LEFT, LEFT, LEFT), 0},
        {std::make_tuple(RIGHT, RIGHT, RIGHT), 1},
        {std::make_tuple(LEFT, LEFT, RIGHT), 2},
        {std::make_tuple(RIGHT, RIGHT, LEFT), 3},
        {std::make_tuple(LEFT, RIGHT, LEFT), 4},
        {std::make_tuple(RIGHT, LEFT, LEFT), 5},
        {std::make_tuple(LEFT, RIGHT, RIGHT), 6},
        {std::make_tuple(RIGHT, LEFT, RIGHT), 7},

        // begin dep 1
        // group 0
        {std::make_tuple(LEFT, LEFT, MIDDLE), 8},
        {std::make_tuple(LEFT, MIDDLE, LEFT), 16},
        {std::make_tuple(MIDDLE, LEFT, LEFT), 24},

        // group 1
        {std::make_tuple(RIGHT, RIGHT, PBC), 9},
        {std::make_tuple(RIGHT, PBC, RIGHT), 17},
        {std::make_tuple(PBC, RIGHT, RIGHT), 25},

        // group 2
        {std::make_tuple(LEFT, LEFT, PBC), 10},
        {std::make_tuple(LEFT, MIDDLE, RIGHT), 18},
        {std::make_tuple(MIDDLE, LEFT, RIGHT), 26},

        // group 3
        {std::make_tuple(RIGHT, RIGHT, MIDDLE), 11},
        {std::make_tuple(RIGHT, PBC, LEFT), 19},
        {std::make_tuple(PBC, RIGHT, LEFT), 27},

        // group 4
        {std::make_tuple(LEFT, RIGHT, MIDDLE), 12},
        {std::make_tuple(LEFT, PBC, LEFT), 20},
        {std::make_tuple(MIDDLE, RIGHT, LEFT), 28},

        // group 5
        {std::make_tuple(RIGHT, LEFT, MIDDLE), 13},
        {std::make_tuple(RIGHT, MIDDLE, LEFT), 21},
        {std::make_tuple(PBC, LEFT, LEFT), 29},

        // group 6
        {std::make_tuple(LEFT, RIGHT, PBC), 14},
        {std::make_tuple(LEFT, PBC, RIGHT), 22},
        {std::make_tuple(MIDDLE, RIGHT, RIGHT), 30},

        // group 7
        {std::make_tuple(RIGHT, LEFT, PBC), 15},
        {std::make_tuple(RIGHT, MIDDLE, RIGHT), 23},
        {std::make_tuple(PBC, LEFT, RIGHT), 31},

        // begin dep 2
        // group 0
        {std::make_tuple(LEFT, MIDDLE, MIDDLE), 32},
        {std::make_tuple(MIDDLE, LEFT, MIDDLE), 40},
        {std::make_tuple(MIDDLE, MIDDLE, LEFT), 48},

        // group 1
        {std::make_tuple(RIGHT, PBC, PBC), 33},
        {std::make_tuple(PBC, RIGHT, PBC), 41},
        {std::make_tuple(PBC, PBC, RIGHT), 49},

        // group 2
        {std::make_tuple(LEFT, MIDDLE, PBC), 34},
        {std::make_tuple(MIDDLE, LEFT, PBC), 42},
        {std::make_tuple(MIDDLE, MIDDLE, RIGHT), 50},

        // group 3
        {std::make_tuple(RIGHT, PBC, MIDDLE), 35},
        {std::make_tuple(PBC, RIGHT, MIDDLE), 43},
        {std::make_tuple(PBC, PBC, LEFT), 51},

        // group 4
        {std::make_tuple(LEFT, PBC, MIDDLE), 36},
        {std::make_tuple(MIDDLE, RIGHT, MIDDLE), 44},
        {std::make_tuple(MIDDLE, PBC, LEFT), 52},

        // group 5
        {std::make_tuple(RIGHT, MIDDLE, MIDDLE), 37},
        {std::make_tuple(PBC, LEFT, MIDDLE), 45},
        {std::make_tuple(PBC, MIDDLE, LEFT), 53},

        // group 6
        {std::make_tuple(LEFT, PBC, PBC), 38},
        {std::make_tuple(MIDDLE, RIGHT, PBC), 46},
        {std::make_tuple(MIDDLE, PBC, RIGHT), 54},


        // group 7
        {std::make_tuple(RIGHT, MIDDLE, PBC), 39},
        {std::make_tuple(PBC, LEFT, PBC), 47},
        {std::make_tuple(PBC, MIDDLE, RIGHT), 55},

        // begin dep3
        {std::make_tuple(MIDDLE, MIDDLE, MIDDLE), 56},

        {std::make_tuple(PBC, PBC, PBC), 57},

        {std::make_tuple(MIDDLE, MIDDLE, PBC), 58},

        {std::make_tuple(PBC, PBC, MIDDLE), 59},

        {std::make_tuple(MIDDLE, PBC, MIDDLE), 60},

        {std::make_tuple(PBC, MIDDLE, MIDDLE), 61},

        {std::make_tuple(MIDDLE, PBC, PBC), 62},

        {std::make_tuple(PBC, MIDDLE, PBC), 63},
        // end dep 3
};
*/

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

  std::pair<int, dbl3_t_stencil_md>** per_worker_force_updates;
  std::vector<int>** space_cut_idxs;

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

  // TODO:
  std::vector<int>* send_force_idxs_double_buffering_flattened;
  std::vector<int>** send_pos_idxs_double_buffering_flattened;
  std::vector<int>* send_vel_idxs_double_buffering_flattened;

  std::vector<int>* recv_force_idxs_double_buffering_flattened;
  std::vector<int>** recv_pos_idxs_double_buffering_flattened;
  std::vector<int>* recv_vel_idxs_double_buffering_flattened;

  std::vector<int>** send_force_idxs_double_buffering;
  std::vector<int>** recv_force_idxs_double_buffering;

  std::vector<int>** send_pos_idxs_double_buffering;
  std::vector<int>** recv_pos_idxs_double_buffering;

  std::vector<int>** recv_pos_local_idxs_double_buffering;
  std::vector<int>** recv_pos_ghost_idxs_double_buffering;

  std::vector<int>** send_vel_idxs_double_buffering;
  std::vector<int>** recv_vel_idxs_double_buffering;
  /* end for two timesteps */

  std::vector<bool>* local_bins_comm;
  std::vector<IDX_3D>* no_comm_local_bins;
  std::vector<IDX_3D>* comm_local_bins;
  std::vector<int>** bin_to_force_comm;
  int** bin_to_pos_vel_comm;

  int** bin_to_idx;
  int** bin_to_size;

  int** bin_to_num_send_zoids;
  int*** bin_to_send_zoids;

  int** send_force_num_bins;
  // std::tuple<int, int, int>*** send_force_bins;
  IDX_3D*** send_force_bins;

  int** send_pos_num_bins;
  // std::tuple<int, int, int>*** send_pos_bins;
  IDX_3D*** send_pos_bins;

  int** send_vel_num_bins;
  // std::tuple<int, int, int>*** send_vel_bins;
  IDX_3D*** send_vel_bins;

  int** recv_force_num_bins;
  // std::tuple<int, int, int>*** recv_force_bins;
  IDX_3D*** recv_force_bins;

  int** recv_pos_num_bins;
  // std::tuple<int, int, int>*** recv_pos_bins;
  IDX_3D*** recv_pos_bins;

  int** recv_vel_num_bins;
  // std::tuple<int, int, int>*** recv_vel_bins;
  IDX_3D*** recv_vel_bins;

  int* inum_per_timestep;
  int debug_int;
  int t0;
  int t1;
  int dim;
  cuts_t zoid;
  int num;
  int where[3];
  int **atom_idx_mapping;
  int **reverse_atom_idx_mapping;
  std::vector<int>* reverse_atom_idx_mapping_idxs;

  // debugging
  double** debug_atom_pos;

  // use pointers since any copies, the pointers will be copied over rather than arrays it seems
  bool **can_eval_center;
  bool **can_eval_pos;

  int*** recv_list_local;
  int** recv_list_local_size;

  // for send list
  int*** send_force_idxs;
  int*** send_force_sizes;
  int** send_force_num_segments;
  int** send_force_total_num_elems;

  int*** send_pos_idxs;
  int*** send_pos_sizes;
  int** send_pos_num_segments;
  int** send_pos_total_num_elems;

  int*** recv_list_local_force_only;
  int** recv_list_local_num_force_pos;

  int*** recv_list_local_force_pos;
  int** recv_list_local_num_force_only;

  // for second sendlist
  int*** send_local_list;
  int*** send_segment_sizes;
  bool*** send_segment_types;
  int*** send_segment_idxs;
  int** send_num_segments;

  int*** recv_ghost_idxs;
  int*** recv_ghost_sizes;
  int** recv_ghost_num_segments;

  int** num_elems_send;
  int** num_elems_recv;

  // info used for sending data from zoid to a process

  // helper variables to help with having a temporary place to store data when doing MPI_Isend
  int** num_elems_send_process;
  int** num_elems_recv_process;

  int*** send_process_segment_sizes;
  int*** send_process_segment_idxs;
  bool*** send_process_segment_types;
  int** send_process_num_segments;
  int*** send_process_local_list;

  int** recv_process_force_offset;
  int** recv_process_vel_offset;
  int** recv_process_pos_offset;

  int*** recv_process_segment_sizes;
  bool*** recv_process_segment_types;
  int*** recv_process_segment_idxs;
  int** recv_process_num_segments;

  std::set<int>* relevant_atom_idxs;
  std::set<int>* relevant_atom_tags;
  std::set<int>* can_eval_center_tags;

  // num elems send to process across ALL timesteps
  int** num_send_process_timestep;
  int* num_send_process;
  int* num_recv_process;
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

IDX_3D get_bin(std::vector<double>& bounds, double* pos, double* lo, double* hi);

inline __attribute__((always_inline)) int lammps_get_bin_idx(const IDX_3D& bin) {
    int x = bin[0];
    int y = bin[1];
    int z = bin[2];
    if (x >= LAMMPS_NUM_REGIONS || x < 0) {
        std::cout << "bin x: " << x << std::endl;
    }
    if (y >= LAMMPS_NUM_REGIONS || y < 0) {
        std::cout << "bin y: " << y << std::endl;
    }
    if (z >= LAMMPS_NUM_REGIONS || z < 0) {
        std::cout << "bin z: " << z << std::endl;
    }
    assert(x < LAMMPS_NUM_REGIONS && x >= 0);
    assert(y < LAMMPS_NUM_REGIONS && y >= 0);
    assert(z < LAMMPS_NUM_REGIONS && z >= 0);
    return z * LAMMPS_NUM_REGIONS * LAMMPS_NUM_REGIONS + y * LAMMPS_NUM_REGIONS + x;
}

inline __attribute__((always_inline)) int get_bin_idx(const IDX_3D& bin) {
    // int x = std::get<0>(bin);
    // int y = std::get<1>(bin);
    // int z = std::get<2>(bin);
    int x = bin[0];
    int y = bin[1];
    int z = bin[2];
    /*
    if (x >= NUM_BINS || x < 0) {
        std::cout << "bin x: " << x << std::endl;
    }
    if (y >= NUM_BINS || y < 0) {
        std::cout << "bin y: " << y << std::endl;
    }
    if (z >= NUM_BINS || z < 0) {
        std::cout << "bin z: " << z << std::endl;
    }
    */
    assert(x < NUM_BINS && x >= 0);
    assert(y < NUM_BINS && y >= 0);
    assert(z < NUM_BINS && z >= 0);
    return z * NUM_BINS * NUM_BINS + y * NUM_BINS + x;
}

inline __attribute__((always_inline)) double min_dist_to_boundary(const std::array<std::vector<double>, 3>& bounds, dbl3_t_stencil_md pos) {
    double dist = 10000000;

    for (auto& x_bound : bounds[0]) {
        for (auto& y_bound : bounds[1]) {
            for (auto& z_bound : bounds[2]) {
                double delx = pos.x - x_bound;
                double dely = pos.y - y_bound;
                double delz = pos.z - z_bound;
                dist = std::min(delx * delx + dely * dely + delz * delz, dist);
            }
        }
    }

    assert(dist != 10000000);

    return dist;
}



