//
// Created by Ryan Deng on 5/7/23.
//

#include <mpi.h>
#include <cmath>
#include <cstring>
#include <deque>
#include <iostream>
#include <vector>

#include "stencil_md_utils.h"

int get_zoid_dep(int zoid_num)
{
  int my_zoid_dep = -1;
  if (zoid_num < 8) {
    my_zoid_dep = 0;
  } else if (zoid_num < 32) {
    my_zoid_dep = 1;
  } else if (zoid_num < 56) {
    my_zoid_dep = 2;
  } else {
    assert(zoid_num < 64);
    my_zoid_dep = 3;
  }
  assert(my_zoid_dep != -1);
  return my_zoid_dep;
}

int get_zoid_dep_next_dt(int zoid_num)
{
    int my_zoid_dep = -1;
    if (zoid_num < 8) {
        my_zoid_dep = 3;
    } else if (zoid_num < 32) {
        my_zoid_dep = 2;
    } else if (zoid_num < 56) {
        my_zoid_dep = 1;
    } else {
        assert(zoid_num < 64);
        my_zoid_dep = 0;
    }
    assert(my_zoid_dep != -1);
    return my_zoid_dep;
}

// use this if guarantee don't want shared ghosts
/*
int get_zoid_dep(int zoid_num) {
  int my_zoid_dep = -1;
  if (zoid_num < 8) {
    my_zoid_dep = 0;
  } else if (zoid_num < 16) {
    my_zoid_dep = 1;
  } else if (zoid_num < 24) {
    my_zoid_dep = 2;
  } else if (zoid_num < 32) {
    my_zoid_dep = 3;
  } else if (zoid_num < 40) {
    my_zoid_dep = 4;
  } else if (zoid_num < 48) {
    my_zoid_dep = 5;
  } else if (zoid_num < 56) {
    my_zoid_dep = 6;
  } else if (zoid_num < 64) {
    my_zoid_dep = 7;
  }

  assert(my_zoid_dep != -1);
  return my_zoid_dep;
}
*/

bool is_close(int *pos1, int *pos2)
{
  for (int i = 0; i < 3; i++) {
    if (pos1[i] == pos2[i]) { continue; }

    if (pos1[i] == LEFT || pos1[i] == RIGHT) {
      if (pos2[i] == MIDDLE || pos2[i] == PBC) { continue; }
    }

    if (pos1[i] == MIDDLE || pos1[i] == PBC) {
      if (pos2[i] == LEFT || pos2[i] == RIGHT) { continue; }
    }

    return false;
  }
  return true;
}

bool is_close_test(int *pos1, int *pos2) {
    for (int i = 0; i < 3; i++) {
        if (pos1[i] == pos2[i]) { continue; }

        if (pos1[i] == LEFT || pos1[i] == RIGHT) {
            if (pos2[i] == MIDDLE || pos2[i] == PBC) { continue; }
        }

        return false;
    }
    return true;
}

bool is_close_test_next_dt(int *pos1, int *pos2) {
    for (int i = 0; i < 3; i++) {
        if (pos1[i] == pos2[i]) { continue; }

        if (pos1[i] == MIDDLE || pos1[i] == PBC) {
            if (pos2[i] == LEFT || pos2[i] == RIGHT) { continue; }
        }

        return false;
    }
    return true;
}

bool is_dep(int *pos1, int *pos2)
{
  bool is_diff_in_one_spot = false;
  for (int i = 0; i < 3; i++) {
    if (pos1[i] != pos2[i]) {
      if (is_diff_in_one_spot) { return false; }
      is_diff_in_one_spot = true;
      if (pos1[i] == LEFT || pos1[i] == RIGHT) {
        if (pos2[i] != MIDDLE && pos2[i] != PBC) { return false; }
      }
    }
  }
  return true;
}

bool is_dep_inverted(int *pos1, int *pos2)
{
  bool is_diff_in_one_spot = false;
  for (int i = 0; i < 3; i++) {
    if (pos1[i] != pos2[i]) {
      if (is_diff_in_one_spot) { return false; }
      is_diff_in_one_spot = true;
      if (pos1[i] == MIDDLE || pos1[i] == PBC) {
        if (pos2[i] != LEFT && pos2[i] != RIGHT) { return false; }
      }
    }
  }
  return true;
}

int get_segments(const std::vector<int>& idxs, std::vector<int>& segment_idxs, std::vector<int>& segment_lengths, bool print) {
    if (idxs.size() == 0) {
        return 0;
    }

    assert(segment_idxs.size() == 0);
    assert(segment_lengths.size() == 0);

    int start = 0;

    for (int j = 1; j < idxs.size(); j++) {
        // if (idxs[j] - idxs[j - 1] > 1) {
        if (idxs[j] - idxs[j - 1] != 1) {
            segment_idxs.push_back(idxs[start]);
            int segment_length = (j - 1 - start + 1);
            segment_lengths.push_back(segment_length);
            start = j;
        }
    }

    int last_segment_length = idxs.size() - 1 - start + 1;
    segment_idxs.push_back(idxs[start]);
    segment_lengths.push_back(last_segment_length);

    if (print) {
        for (int i = 0; i < segment_idxs.size(); i++) {
            std::cout << RED << "segment: " << i << " segment idx: " << segment_idxs[i] << " size: " << segment_lengths[i] << RESET_COLOR << std::endl;
        }
    }

    return segment_lengths.size();
}

void get_zoids(double slope, double *lo, double *hi, std::deque<queue_info> *queues)
{
  int num_dims = 3;
  int initial_dep = 0;
  queue_info initial_zoid;
  cuts_t curr_cuts_t;

  double lattice[3] = {hi[0] - lo[0], hi[1] - lo[0], hi[2] - lo[2]};

  for (int i = 0; i < 3; i++) {
    curr_cuts_t.cuts[i].lower = lo[i];
    curr_cuts_t.cuts[i].upper = hi[i];
    curr_cuts_t.cuts[i].slope_lower = 0.0;
    curr_cuts_t.cuts[i].slope_upper = 0.0;
  }
  initial_zoid.t0 = 0;
  initial_zoid.t1 = NUM_TIMESTEPS_IN_PARALLEL;
  initial_zoid.dim = num_dims - 1;
  initial_zoid.zoid = curr_cuts_t;
  initial_zoid.num = 0;
  // double slope = lmp->neighbor->cutneighmax + ADDITIONAL_CUTOFF;
  for (int dim = 0; dim < 3; dim++) {
    initial_zoid.zoid.cuts[dim].slope_lower = 0.0;
    initial_zoid.zoid.cuts[dim].slope_upper = 0.0;
  }

  int world_rank;
  int world_size;
  MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
  MPI_Comm_size(MPI_COMM_WORLD, &world_size);

  queues[initial_dep].push_back(initial_zoid);
  // 3 dimensions means 4 dependency levels
  for (int dep = 0; dep < num_dims + 1; dep++) {
    const std::deque<queue_info> &queue = queues[dep];
    while (queue.size() > 0) {
      queue_info q_info = queue.front();
      bool done_cutting = true;
      bool is_neg = false;
      for (int i = 0; i < queue.size(); i++) {
        if (queue[i].dim >= 0) { done_cutting = false; }
        if (queue[i].dim < 0) { is_neg = true; }
      }
      if (done_cutting && is_neg) {
        break;
      } else {
        queues[dep].pop_front();
        const int dim = q_info.dim;
        if (dim < 0) {
          queues[dep].push_back(q_info);
          continue;
        }
        const int dt = q_info.t1 - q_info.t0;
        // TODO: 2 * thresh needs to be changed to 3 * thresh for parallel
        // receive
        const double thresh = 2 * slope * dt + 2 * slope;
        const double lb = q_info.zoid.cuts[dim].upper - q_info.zoid.cuts[dim].lower;
        // const bool can_cut = lb >= 2 * thresh;
        const bool can_cut = lb >= thresh;
        if (!can_cut) {
          queue_info next = q_info;
          next.t0 = 0;
          next.t1 = NUM_TIMESTEPS_IN_PARALLEL;
          next.dim = dim - 1;
          next.zoid = q_info.zoid;
          queues[dep].push_back(next);
        } else {
          const double mid = lb / 2;
          const double start = q_info.zoid.cuts[dim].lower;
          const double end = q_info.zoid.cuts[dim].upper;

          // std::cout << "lb: " << lb << " mid: " << lb / 2 << " start: " << start << " end: " << end << std::endl;

          // bool initial_cut = std::abs(lb - args.lattice[dim * 3 + dim]) <=
          // 1e-8;
          bool initial_cut = std::abs(lb - lattice[dim]) <= 1e-8;

          cuts_t left_zoid = q_info.zoid;
          if (initial_cut) {
            left_zoid.cuts[dim].lower = start + MIDDLE_ZOID_WIDTH_RATIO * slope;
            // left_zoid.cuts[dim].lower = start;
          } else {
            left_zoid.cuts[dim].lower = start;
          }
          left_zoid.cuts[dim].upper = start + mid - MIDDLE_ZOID_WIDTH_RATIO * slope;
          // left_zoid.cuts[dim].upper = start + mid;
          left_zoid.cuts[dim].slope_lower = slope;
          left_zoid.cuts[dim].slope_upper = -slope;

          queue_info left_zoid_info = q_info;
          left_zoid_info.t0 = q_info.t0;
          left_zoid_info.t1 = q_info.t1;
          left_zoid_info.zoid = left_zoid;
          left_zoid_info.dim = dim - 1;

          left_zoid_info.where[dim] = LEFT;
          if (!(left_zoid.cuts[dim].lower <= left_zoid.cuts[dim].upper)) {
            std::cout << "ERROR ALERT" << std::endl;
            assert(false);
            // print_cuts_t(left_zoid);
          }

          queues[dep].push_back(left_zoid_info);
          cuts_t right_zoid = q_info.zoid;
          // right_zoid.cuts[dim].lower = start + mid;
          right_zoid.cuts[dim].lower = start + mid + MIDDLE_ZOID_WIDTH_RATIO * slope;
          // right_zoid.cuts_t[dim].upper = end;
          if (initial_cut) {
            right_zoid.cuts[dim].upper = end - MIDDLE_ZOID_WIDTH_RATIO * slope;
            // right_zoid.cuts[dim].upper = end;
          } else {
            right_zoid.cuts[dim].upper = end;
          }
          right_zoid.cuts[dim].slope_lower = slope;
          right_zoid.cuts[dim].slope_upper = -slope;

          queue_info right_zoid_info = q_info;
          right_zoid_info.t0 = q_info.t0;
          right_zoid_info.t1 = q_info.t1;
          right_zoid_info.zoid = right_zoid;
          right_zoid_info.dim = dim - 1;

          right_zoid_info.where[dim] = RIGHT;
          queues[dep].push_back(right_zoid_info);

          int next_dep = dep + 1;
          cuts_t middle_zoid = q_info.zoid;
          // middle_zoid.cuts[dim].lower = start + mid;
          // middle_zoid.cuts[dim].upper = start + mid;
          middle_zoid.cuts[dim].lower = start + mid - MIDDLE_ZOID_WIDTH_RATIO * slope;
          middle_zoid.cuts[dim].upper = start + mid + MIDDLE_ZOID_WIDTH_RATIO * slope;
          middle_zoid.cuts[dim].slope_lower = -slope;
          middle_zoid.cuts[dim].slope_upper = slope;

          queue_info middle_zoid_info = q_info;
          middle_zoid_info.t0 = q_info.t0;
          middle_zoid_info.t1 = q_info.t1;
          middle_zoid_info.zoid = middle_zoid;
          middle_zoid_info.dim = dim - 1;

          middle_zoid_info.where[dim] = MIDDLE;
          queues[next_dep].push_back(middle_zoid_info);

          if (std::abs(lb - lattice[dim]) <= 1e-8) {
            // initial cut
            cuts_t pbc_zoid = q_info.zoid;
            // pbc_zoid.cuts[dim].lower = -2 * slope;
            // pbc_zoid.cuts[dim].upper = 2 * slope;
            // pbc_zoid.cuts[dim].lower = start;
            // pbc_zoid.cuts[dim].upper = start;
            pbc_zoid.cuts[dim].lower = lo[dim] - MIDDLE_ZOID_WIDTH_RATIO * slope;
            pbc_zoid.cuts[dim].upper = lo[dim] + MIDDLE_ZOID_WIDTH_RATIO * slope;

            pbc_zoid.cuts[dim].slope_lower = -slope;
            pbc_zoid.cuts[dim].slope_upper = slope;

            queue_info pbc_zoid_info = q_info;
            pbc_zoid_info.t0 = q_info.t0;
            pbc_zoid_info.t1 = q_info.t1;
            pbc_zoid_info.zoid = pbc_zoid;
            pbc_zoid_info.dim = dim - 1;

            pbc_zoid_info.where[dim] = PBC;
            queues[next_dep].push_back(pbc_zoid_info);
          } else {
            if (std::abs(q_info.zoid.cuts[dim].slope_lower - slope) > 1e-5) {
              assert(false);
              cuts_t left_inverted_zoid = q_info.zoid;
              left_inverted_zoid.cuts[dim].lower = start;
              left_inverted_zoid.cuts[dim].slope_lower = q_info.zoid.cuts[dim].slope_lower;
              left_inverted_zoid.cuts[dim].upper = start;
              left_inverted_zoid.cuts[dim].slope_upper = slope;

              queue_info left_inverted_zoid_info;
              left_inverted_zoid_info.t0 = q_info.t0;
              left_inverted_zoid_info.t1 = q_info.t1;
              left_inverted_zoid_info.zoid = left_inverted_zoid;
              left_inverted_zoid_info.dim = dim - 1;
              queues[next_dep].push_back(left_inverted_zoid_info);
            }
            if (std::abs(q_info.zoid.cuts[dim].slope_upper - (-slope)) > 1e-5) {
              assert(false);
              cuts_t right_inverted_zoid = q_info.zoid;
              right_inverted_zoid.cuts[dim].lower = start;
              right_inverted_zoid.cuts[dim].slope_lower = -slope;
              right_inverted_zoid.cuts[dim].upper = start;
              right_inverted_zoid.cuts[dim].slope_upper = q_info.zoid.cuts[dim].slope_upper;

              queue_info right_inverted_zoid_info;
              right_inverted_zoid_info.t0 = q_info.t0;
              right_inverted_zoid_info.t1 = q_info.t1;
              right_inverted_zoid_info.zoid = right_inverted_zoid;
              right_inverted_zoid_info.dim = dim - 1;
              queues[next_dep].push_back(right_inverted_zoid_info);
            }
          }
        }
      }
    }
  }

  int num = 0;
  for (int dep = 0; dep < 3 + 1; dep++) {
    for (int j = 0; j < queues[dep].size(); j++) { queues[dep][j].num = num++; }
  }

  // std::deque<queue_info> queues_next_dt[num_dims + 1];
  for (int dep = 0; dep < 3 + 1; dep++) {
    for (int j = 0; j < queues[dep].size(); j++) {
      const queue_info &info = queues[dep][j];
      int new_dep = 3 - dep;
      queue_info initial;
      initial.t0 = info.t1;
      initial.t1 = info.t1 + NUM_TIMESTEPS_IN_PARALLEL;
      // irrelevant
      initial.dim = num_dims - 1;
      cuts_t new_cuts_t;
      for (int i = 0; i < 3; i++) {
        int dt = (info.t1 - info.t0);
        double new_start = info.zoid.cuts[i].lower + dt * info.zoid.cuts[i].slope_lower;
        double new_end = info.zoid.cuts[i].upper + dt * info.zoid.cuts[i].slope_upper;
        new_cuts_t.cuts[i].lower = new_start;
        new_cuts_t.cuts[i].upper = new_end;
        new_cuts_t.cuts[i].slope_lower = -1 * info.zoid.cuts[i].slope_lower;
        new_cuts_t.cuts[i].slope_upper = -1 * info.zoid.cuts[i].slope_upper;
      }
      initial.zoid = new_cuts_t;
      initial.num = info.num;
      for (int i = 0; i < 3; i++) { initial.where[i] = info.where[i]; }
      // queues_next_dt[new_dep].push_back(initial);
      // queues[new_dep + 3 + 1].push_back(initial);
    }
  }
}

void print_cuts(const cuts_t &c)
{
  for (int i = 0; i < 3; i++) {
    std::cout << "cuts_t: " << i << " lower: " << c.cuts[i].lower << " upper: " << c.cuts[i].upper
              << std::endl;
    std::cout << "Slope lower: " << c.cuts[i].slope_lower
              << " Slope upper: " << c.cuts[i].slope_upper << std::endl;
  }
}

uint64_t timeSinceEpochMillisec() {
    using namespace std::chrono;
    return duration_cast<milliseconds>(system_clock::now().time_since_epoch()).count();
}
