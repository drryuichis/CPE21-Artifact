#ifndef PML_H
#define PML_H

#include "constants.h"
#include "grid.h"

void init_eta(grid_t grid, float dt_sch, float *eta);

#endif
