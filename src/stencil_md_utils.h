//
// Created by Ryan Deng on 5/7/23.
//

#pragma once

#include <mpi.h>
#include <map>
#include <cassert>
#include <chrono>
#include <set>

#include <deque>
#include <iostream>

// Used for spatial sorting
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

constexpr int NUM_ZOIDS = 4 * 4 * 4;

constexpr int NUM_TIMESTEPS_IN_PARALLEL = 10;
constexpr double ADDITIONAL_CUTOFF = 0.401;

constexpr double ALLEGRO_CUTOFF_RADIUS = 1.12;

constexpr double MIDDLE_ZOID_WIDTH_RATIO = 0.5;

constexpr bool DEBUG_SEND_RECV_DATA = true;

constexpr bool TEST_AGAINST_LAMMPS = true;

constexpr bool PURELY_LOCAL_POTENTIAL = true;

constexpr bool USE_FAKE_COMPUTE_TEMP = true;

constexpr double ALLEGRO_SLOPE = ALLEGRO_CUTOFF_RADIUS + ADDITIONAL_CUTOFF;

constexpr bool TRY_PRECOMPUTE_RELEVANT_ATOM_IDX = false;

constexpr bool DEBUG = true;

constexpr int NUM_WORKERS_PER_THREAD = 64;

constexpr int NUM_ATOMS_PER_WORKER = 128;

constexpr bool ONLY_RUN_LAMMPS = false;

constexpr bool ONLY_RUN_STENCIL_MD = false;

constexpr bool LAMMPS_USE_CILK = false;

constexpr bool TIME_STENCIL_MD = true;

constexpr bool USE_BOND = true;

constexpr int NUM_PIPELINE_STAGES = 2;

constexpr bool USE_ATOMICS = false;

constexpr int NUM_BINS = 48;

constexpr bool PAIR_USE_BINS = true;

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

        /*
        // group 0
        {std::make_tuple(LEFT, LEFT, MIDDLE), 8},
        {std::make_tuple(LEFT, MIDDLE, LEFT), 16},
        {std::make_tuple(MIDDLE, LEFT, LEFT), 24},

        // group 1
        {std::make_tuple(RIGHT, RIGHT, MIDDLE), 9},
        {std::make_tuple(RIGHT, MIDDLE, RIGHT), 17},
        {std::make_tuple(MIDDLE, RIGHT, RIGHT), 25},

        // group 2
        {std::make_tuple(LEFT, LEFT, PBC), 10},
        {std::make_tuple(LEFT, PBC, LEFT), 18},
        {std::make_tuple(PBC, LEFT, LEFT), 26},

        // group 3
        {std::make_tuple(RIGHT, RIGHT, PBC), 11},
        {std::make_tuple(RIGHT, PBC, RIGHT), 19},
        {std::make_tuple(PBC, RIGHT, RIGHT), 27},

        // group 4
        {std::make_tuple(LEFT, RIGHT, MIDDLE), 12},
        {std::make_tuple(LEFT, MIDDLE, RIGHT), 20},
        {std::make_tuple(MIDDLE, LEFT, RIGHT), 28},

        // group 5
        {std::make_tuple(RIGHT, LEFT, MIDDLE), 13},
        {std::make_tuple(RIGHT, MIDDLE, LEFT), 21},
        {std::make_tuple(MIDDLE, RIGHT, LEFT), 29},

        // group 6
        {std::make_tuple(LEFT, RIGHT, PBC), 14},
        {std::make_tuple(LEFT, PBC, RIGHT), 22},
        {std::make_tuple(PBC, LEFT, RIGHT), 30},

        // group 7
        {std::make_tuple(RIGHT, LEFT, PBC), 15},
        {std::make_tuple(RIGHT, PBC, LEFT), 23},
        {std::make_tuple(PBC, RIGHT, LEFT), 31},

        // end dep 1

        // begin dep 2
        {std::make_tuple(LEFT, MIDDLE, MIDDLE), 32},
        {std::make_tuple(RIGHT, MIDDLE, MIDDLE), 40},
        {std::make_tuple(MIDDLE, LEFT, MIDDLE), 48},

        // group 1
        {std::make_tuple(MIDDLE, MIDDLE, LEFT), 33},
        {std::make_tuple(MIDDLE, MIDDLE, RIGHT), 41},
        {std::make_tuple(MIDDLE, RIGHT, MIDDLE), 49},

        // group 2
        {std::make_tuple(LEFT, PBC, PBC), 34},
        {std::make_tuple(RIGHT, PBC, PBC), 42},
        {std::make_tuple(PBC, LEFT, PBC), 50},

        // group 3
        {std::make_tuple(PBC, PBC, RIGHT), 35},
        {std::make_tuple(PBC, PBC, LEFT), 43},
        {std::make_tuple(PBC, RIGHT, PBC), 51},

        // group 4
        {std::make_tuple(MIDDLE, PBC, LEFT), 36},
        {std::make_tuple(MIDDLE, PBC, RIGHT), 44},
        {std::make_tuple(MIDDLE, RIGHT, PBC), 52},

        // group 5
        {std::make_tuple(PBC, LEFT, MIDDLE), 37},
        {std::make_tuple(PBC, RIGHT, MIDDLE), 45},
        {std::make_tuple(PBC, MIDDLE, LEFT), 53},

        // group 6
        {std::make_tuple(RIGHT, MIDDLE, PBC), 38},
        {std::make_tuple(LEFT, MIDDLE, PBC), 46},
        {std::make_tuple(PBC, MIDDLE, RIGHT), 54},


        // group 7
        {std::make_tuple(LEFT, PBC, MIDDLE), 39},
        {std::make_tuple(RIGHT, PBC, MIDDLE), 47},
        {std::make_tuple(MIDDLE, LEFT, PBC), 55},

        // end dep 2

        // begin dep 3
        // good
        {std::make_tuple(MIDDLE, MIDDLE, MIDDLE), 56},

        {std::make_tuple(MIDDLE, MIDDLE, PBC), 57},

        {std::make_tuple(PBC, PBC, PBC), 58},

        {std::make_tuple(PBC, PBC, MIDDLE), 59},

        {std::make_tuple(MIDDLE, PBC, MIDDLE), 60},

        {std::make_tuple(PBC, MIDDLE, MIDDLE), 61},

        {std::make_tuple(PBC, MIDDLE, PBC), 62},

        {std::make_tuple(MIDDLE, PBC, PBC), 63},
        */

        // end dep 3
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

typedef struct cuts cuts_t;

// struct that holds information for queue
struct queue_info {
  int** bin_to_idx;
  int** bin_to_size;

  int** send_force_num_bins;
  std::tuple<int, int, int>*** send_force_bins;

  int** send_pos_num_bins;
  std::tuple<int, int, int>*** send_pos_bins;

  int** send_vel_num_bins;
  std::tuple<int, int, int>*** send_vel_bins;

  int** recv_force_num_bins;
  std::tuple<int, int, int>*** recv_force_bins;

  int** recv_pos_num_bins;
  std::tuple<int, int, int>*** recv_pos_bins;

  int** recv_vel_num_bins;
  std::tuple<int, int, int>*** recv_vel_bins;

  int* inum_per_timestep;
  int debug_int;
  int t0;
  int t1;
  int dim;
  cuts_t zoid;
  int num;
  int where[3];
  int **atom_idx_mapping;

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

uint64_t timeSinceEpochMillisec();

int get_mpi_tag(int dst, int src, int start_timestep=0, int end_timestep=0);

std::tuple<int, int, int> get_bin(std::vector<double>& bounds, double* pos, double* lo, double* hi);

inline __attribute__((always_inline)) int get_bin_idx(const std::tuple<int, int, int>& bin) {
    int x = std::get<0>(bin);
    int y = std::get<1>(bin);
    int z = std::get<2>(bin);
    if (x >= NUM_BINS || x < 0) {
        std::cout << "bin x: " << x << std::endl;
    }
    if (y >= NUM_BINS || y < 0) {
        std::cout << "bin y: " << y << std::endl;
    }
    if (z >= NUM_BINS || z < 0) {
        std::cout << "bin z: " << z << std::endl;
    }
    assert(x < NUM_BINS && x >= 0);
    assert(y < NUM_BINS && y >= 0);
    assert(z < NUM_BINS && z >= 0);
    return z * NUM_BINS * NUM_BINS + y * NUM_BINS + x;
}


