#include <cilk/opadd_reducer.h>

static cilk::opadd_reducer<double> compute_time;
static cilk::opadd_reducer<double> comm_time;
