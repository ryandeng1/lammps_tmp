//
// Created by Ryan Deng on 5/7/23.
//

#ifndef LAMMPS_STENCIL_MD_H
#define LAMMPS_STENCIL_MD_H

#include <mpi.h>

#include <deque>
#include <iostream>

constexpr int LEFT = -1;
constexpr int RIGHT = -2;
constexpr int MIDDLE = -3;
constexpr int PBC = -4;

constexpr int NUM_DEPS = 8;

constexpr int NUM_ZOIDS = 4 * 4 * 4;

constexpr int NUM_TIMESTEPS_IN_PARALLEL = 2;
constexpr double ADDITIONAL_CUTOFF = 0.2;

constexpr double ALLEGRO_SLOPE = 2 * (3.0 + ADDITIONAL_CUTOFF);

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
    int** atom_idx_mapping;
    int* first_recv_stencil_md[NUM_TIMESTEPS_IN_PARALLEL + 1];
    bool init_first_recv = false;

    int** first_recv_stencil_md2[NUM_TIMESTEPS_IN_PARALLEL + 1];
    int* first_recv_sz_stencil_md2[NUM_TIMESTEPS_IN_PARALLEL + 1];
};

int get_zoid_dep(int);

bool is_close(int*, int*);

bool is_dep(int *, int *);

bool is_dep_inverted(int *, int *);

void get_zoids(double slope, double *lo, double *hi, std::deque<queue_info> *queues);

void print_cuts(const cuts_t&);

#endif //LAMMPS_STENCIL_MD_H
