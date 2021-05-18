#ifndef GRID_H
#define GRID_H

#include <sys/types.h>
#include "constants.h"

#ifdef __cplusplus
extern "C" {
#endif

#define IDX3(i,j,k)((llint)((i + lx) * ldimy + j + ly) * (llint)ldimz + k + lz)

#define IDX3_grid(i, j, k, grid) (((i + grid.lx) * grid.ldimy + j + grid.ly) * grid.ldimz + k + grid.lz)

typedef struct grid_t {
    llint ntaperx, ntapery, ntaperz;
    llint ndampx, ndampy, ndampz;
    llint nx, ny, nz;
    llint ldimx, ldimy, ldimz;
    llint dx, dy, dz;
    llint x1, x2, x3, x4, x5, x6;
    llint y1, y2, y3, y4, y5, y6;
    llint z1, z2, z3, z4, z5, z6;
    llint lx, ly, lz;
    // These parameters are used only for tasks
    llint ntx, nty, tsx, tsy;

    // Used for multi-gpu targets
    llint ngpu;
} grid_t;

grid_t init_grid(llint nx, llint ny, llint nz, llint tsx, llint tsy, llint ngpu);

// Allocate or release a grid on GPU
float * allocateDeviceGrid (grid_t grid);
void freeDeviceGrid (float *ptr, grid_t grid);

// Allocate or release a grid on host
float * allocateHostGrid (grid_t grid);
void freeHostGrid (float *ptr, grid_t grid);

// Useful size of the grid, in bytes
size_t gridSize (grid_t grid);


#ifdef __cplusplus
} // extern "C"
#endif

#endif
