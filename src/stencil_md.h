//
// Created by Ryan Deng on 5/7/23.
//

#ifndef LAMMPS_STENCIL_MD_H
#define LAMMPS_STENCIL_MD_H

#include <mpi.h>
#include <map>

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

constexpr int LEFT = -1;
constexpr int RIGHT = -2;
constexpr int MIDDLE = -3;
constexpr int PBC = -4;

constexpr int LOCAL_SEGMENT_TYPE = 0;
constexpr int GHOST_SEGMENT_TYPE = -1;

constexpr int NUM_DEPS = 4;

constexpr int NUM_ZOIDS = 4 * 4 * 4;

constexpr int NUM_TIMESTEPS_IN_PARALLEL = 3;
constexpr double ADDITIONAL_CUTOFF = 0.2;

// constexpr double ALLEGRO_SLOPE = 5.0 + ADDITIONAL_CUTOFF;
constexpr double ALLEGRO_CUTOFF_RADIUS = 5.0;
constexpr double ALLEGRO_SLOPE = 2 * ALLEGRO_CUTOFF_RADIUS + ADDITIONAL_CUTOFF;

// constexpr double MIDDLE_ZOID_WIDTH_RATIO = 6.2 / 2;
constexpr double MIDDLE_ZOID_WIDTH_RATIO = 0.5;

constexpr bool DEBUG_SEND_RECV_DATA = true;

/*
std::map<std::tuple<int, int, int>, int> zoid_to_num_map {
        {std::make_tuple(LEFT, LEFT, LEFT), 0},
        {std::make_tuple(LEFT, LEFT, RIGHT), 0},
        {std::make_tuple(LEFT, RIGHT, LEFT), 0},
        {std::make_tuple(RIGHT, LEFT, LEFT), 0},
        {std::make_tuple(RIGHT, RIGHT, RIGHT), 0},
        {std::make_tuple(RIGHT, RIGHT, LEFT), 0},
        {std::make_tuple(RIGHT, LEFT, RIGHT), 0},
        {std::make_tuple(LEFT, RIGHT, RIGHT), 0},



        {std::make_tuple(MIDDLE, MIDDLE, MIDDLE), 0},
        {std::make_tuple(MIDDLE, MIDDLE, PBC), 0},
        {std::make_tuple(MIDDLE, PBC, MIDDLE), 0},
        {std::make_tuple(PBC, MIDDLE, MIDDLE), 0},
        {std::make_tuple(RIGHT, LEFT, LEFT), 0},
        {std::make_tuple(RIGHT, RIGHT, RIGHT), 0},
        {std::make_tuple(RIGHT, RIGHT, LEFT), 0},
        {std::make_tuple(RIGHT, LEFT, RIGHT), 0},
        {std::make_tuple(LEFT, RIGHT, RIGHT), 0},

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

typedef struct cuts cuts_t;

// struct that holds information for queue
struct queue_info {
  int t0;
  int t1;
  int dim;
  cuts_t zoid;
  int num;
  int where[3];
  int **atom_idx_mapping;

  // use pointers since any copies, the pointers will be copied over rather than arrays it seems
  bool **can_eval_center;
  bool **can_eval_pos;

  int*** recv_list_local;
  int** recv_list_local_size;

  // for send list
  int*** send_force_idxs;
  int*** send_force_sizes;
  int** send_force_num_segments;

  int*** send_pos_idxs;
  int*** send_pos_sizes;
  int** send_pos_num_segments;

  int*** recv_list_local_force_only;
  int** recv_list_local_num_force_pos;

  int*** recv_list_local_force_pos;
  int** recv_list_local_num_force_only;

  // for second sendlist
  int*** send_local_list;
  int*** send_segment_sizes;
  int*** send_segment_types;
  int*** send_segment_idxs;
  int** send_num_segments;

  /*
  int*** send_local_sizes;
  int*** send_local_ghost_segments_mapping;
  int** send_local_num_segments;

  int*** send_ghost_idxs;
  int*** send_ghost_sizes;
  int** send_ghost_num_segments;
  */

  int*** recv_ghost_idxs;
  int*** recv_ghost_sizes;
  int** recv_ghost_num_segments;

  int** num_elems_send;
  int** num_elems_recv;
};

int get_zoid_dep(int);

bool is_close(int *, int *);

bool is_close_test(int *, int *);

bool is_close_test_next_dt(int *, int *);

bool is_dep(int *, int *);

bool is_dep_inverted(int *, int *);

void get_zoids(double slope, double *lo, double *hi, std::deque<queue_info> *queues);

void print_cuts(const cuts_t &);

int get_segments(const std::vector<int>&, std::vector<int>&, std::vector<int>&, bool print=false);

#endif    //LAMMPS_STENCIL_MD_H
