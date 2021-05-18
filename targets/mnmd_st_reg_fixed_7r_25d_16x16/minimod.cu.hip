#include <hip/hip_runtime.h>

#include <stdio.h>
#include <float.h>
#include <math.h>
#include <time.h>

#include "../../constants.h"
#include "../../grid.h"

#define N_RADIUS 4
#define N_THREADS_X_DIM 16
#define N_THREADS_Y_DIM 16
#define N_THREADS_Z_DIM 0

// Constant memory coefficients
__constant__ float c_coef0;
__constant__ float c_coefx[N_RADIUS+1];
__constant__ float c_coefy[N_RADIUS+1];
__constant__ float c_coefz[N_RADIUS+1];

#define INNER_STENCIL(R0, R1, R2, R3, R4, R5, R6, R7, R8) i++; if (i >= x4) { return; } \
    zrs##R8 = u[IDX3(i+N_RADIUS,j,k)]; \
    __syncthreads(); \
    if (threadIdx.y < 2 * N_RADIUS) { \
        s_u[threadIdx.y + (threadIdx.y/N_RADIUS)*N_THREADS_Y_DIM][suk] = \
            u[IDX3(i,j0+threadIdx.y+(threadIdx.y/N_RADIUS)*N_THREADS_Y_DIM-N_RADIUS,k)]; \
    } \
    if (threadIdx.x < 2 * N_RADIUS) { \
        s_u[suj][threadIdx.x + (threadIdx.x/N_RADIUS)*N_THREADS_X_DIM] = \
            u[IDX3(i,j,k0+threadIdx.x+(threadIdx.x/N_RADIUS)*N_THREADS_X_DIM-N_RADIUS)]; \
    } \
    s_u[suj][suk] = u[IDX3(i,j,k)]; \
    __syncthreads(); \
    if (j < y4 && k < z4) { \
        float lap = __fmaf_rn(c_coef0, zrs##R4 \
                    , __fmaf_rn(c_coefx[1], __fadd_rn(zrs##R5,zrs##R3) \
                    , __fmaf_rn(c_coefy[1], __fadd_rn(s_u[suj+1][suk],s_u[suj-1][suk]) \
                    , __fmaf_rn(c_coefz[1], __fadd_rn(s_u[suj][suk+1],s_u[suj][suk-1]) \
                    , __fmaf_rn(c_coefx[2], __fadd_rn(zrs##R6,zrs##R2) \
                    , __fmaf_rn(c_coefy[2], __fadd_rn(s_u[suj+2][suk],s_u[suj-2][suk]) \
                    , __fmaf_rn(c_coefz[2], __fadd_rn(s_u[suj][suk+2],s_u[suj][suk-2]) \
                    , __fmaf_rn(c_coefx[3], __fadd_rn(zrs##R7,zrs##R1) \
                    , __fmaf_rn(c_coefy[3], __fadd_rn(s_u[suj+3][suk],s_u[suj-3][suk]) \
                    , __fmaf_rn(c_coefz[3], __fadd_rn(s_u[suj][suk+3],s_u[suj][suk-3]) \
                    , __fmaf_rn(c_coefx[4], __fadd_rn(zrs##R8,zrs##R0) \
                    , __fmaf_rn(c_coefy[4], __fadd_rn(s_u[suj+4][suk],s_u[suj-4][suk]) \
                    , __fmul_rn(c_coefz[4], __fadd_rn(s_u[suj][suk+4],s_u[suj][suk-4]) \
        ))))))))))))); \
        v[IDX3(i,j,k)] = __fmaf_rn(2.f, zrs##R4, \
            __fmaf_rn(vp[IDX3(i,j,k)], lap, -v[IDX3(i,j,k)]) \
        ); \
    }

#define PML_STENCIL(R0, R1, R2, R3, R4, R5, R6, R7, R8) i++; if (i >= x4) { return; } \
    zrs##R8 = u[IDX3(i+N_RADIUS,j,k)]; \
    __syncthreads(); \
    if (threadIdx.y < 2 * N_RADIUS) { \
        s_u[threadIdx.y + (threadIdx.y/N_RADIUS)*N_THREADS_Y_DIM][suk] = \
            u[IDX3(i,j0+threadIdx.y+(threadIdx.y/N_RADIUS)*N_THREADS_Y_DIM-N_RADIUS,k)]; \
    } \
    if (threadIdx.x < 2 * N_RADIUS) { \
        s_u[suj][threadIdx.x + (threadIdx.x/N_RADIUS)*N_THREADS_X_DIM] = \
            u[IDX3(i,j,k0+threadIdx.x+(threadIdx.x/N_RADIUS)*N_THREADS_X_DIM-N_RADIUS)]; \
    } \
    s_u[suj][suk] = u[IDX3(i,j,k)]; \
    __syncthreads(); \
    if (j < y4 && k < z4) { \
        float lap = __fmaf_rn(c_coef0, zrs##R4 \
                    , __fmaf_rn(c_coefx[1], __fadd_rn(zrs##R5,zrs##R3) \
                    , __fmaf_rn(c_coefy[1], __fadd_rn(s_u[suj+1][suk],s_u[suj-1][suk]) \
                    , __fmaf_rn(c_coefz[1], __fadd_rn(s_u[suj][suk+1],s_u[suj][suk-1]) \
                    , __fmaf_rn(c_coefx[2], __fadd_rn(zrs##R6,zrs##R2) \
                    , __fmaf_rn(c_coefy[2], __fadd_rn(s_u[suj+2][suk],s_u[suj-2][suk]) \
                    , __fmaf_rn(c_coefz[2], __fadd_rn(s_u[suj][suk+2],s_u[suj][suk-2]) \
                    , __fmaf_rn(c_coefx[3], __fadd_rn(zrs##R7,zrs##R1) \
                    , __fmaf_rn(c_coefy[3], __fadd_rn(s_u[suj+3][suk],s_u[suj-3][suk]) \
                    , __fmaf_rn(c_coefz[3], __fadd_rn(s_u[suj][suk+3],s_u[suj][suk-3]) \
                    , __fmaf_rn(c_coefx[4], __fadd_rn(zrs##R8,zrs##R0) \
                    , __fmaf_rn(c_coefy[4], __fadd_rn(s_u[suj+4][suk],s_u[suj-4][suk]) \
                    , __fmul_rn(c_coefz[4], __fadd_rn(s_u[suj][suk+4],s_u[suj][suk-4]) \
        ))))))))))))); \
        const float s_eta_c = eta[IDX3(i,j,k)]; \
        v[IDX3(i,j,k)] = __fdiv_rn( \
            __fmaf_rn( \
                __fmaf_rn(2.f, s_eta_c, \
                    __fsub_rn(2.f, \
                        __fmul_rn(s_eta_c, s_eta_c) \
                    ) \
                ), \
                zrs##R4, \
                __fmaf_rn( \
                    vp[IDX3(i,j,k)], \
                    __fadd_rn(lap, phi[IDX3(i,j,k)]), \
                    -v[IDX3(i,j,k)] \
                ) \
            ), \
            __fmaf_rn(2.f, s_eta_c, 1.f) \
        ); \
        phi[IDX3(i,j,k)] = __fdiv_rn( \
                __fsub_rn( \
                    phi[IDX3(i,j,k)], \
                    __fmaf_rn( \
                    __fmul_rn( \
                        __fsub_rn(eta[IDX3(i+1,j,k)], eta[IDX3(i-1,j,k)]), \
                        __fsub_rn(zrs##R5,zrs##R3) \
                    ), hdx_2, \
                    __fmaf_rn( \
                    __fmul_rn( \
                        __fsub_rn(eta[IDX3(i,j+1,k)], eta[IDX3(i,j-1,k)]), \
                        __fsub_rn(s_u[suj+1][suk], s_u[suj-1][suk]) \
                    ), hdy_2, \
                    __fmul_rn( \
                        __fmul_rn( \
                            __fsub_rn(eta[IDX3(i,j,k+1)], eta[IDX3(i,j,k-1)]), \
                            __fsub_rn(s_u[suj][suk+1], s_u[suj][suk-1]) \
                        ), \
                    hdz_2) \
                    )) \
                ) \
            , \
            __fadd_rn(1.f, s_eta_c) \
        ); \
    }

__global__ void __launch_bounds__(1024) kernel_7r_25d_inner(
    llint nx, llint ny, llint nz, int ldimx, int ldimy, int ldimz,
    llint x3, llint x4, llint y3, llint y4, llint z3, llint z4,
    llint lx, llint ly, llint lz,
    float hdx_2, float hdy_2, float hdz_2,
    const float *__restrict__ u, float *__restrict__ v, const float *__restrict__ vp,
    const float *__restrict__ phi, const float *__restrict__ eta
) {
    __shared__ float s_u[N_THREADS_Y_DIM+2*N_RADIUS][N_THREADS_X_DIM+2*N_RADIUS];

    const llint j0 = y3 + blockIdx.y * blockDim.y;
    const llint k0 = z3 + blockIdx.x * blockDim.x;

    const llint j = j0 + threadIdx.y;
    const llint k = k0 + threadIdx.x;

    const llint suj = threadIdx.y + N_RADIUS;
    const llint suk = threadIdx.x + N_RADIUS;

    float zrs0, zrs1, zrs2, zrs3, zrs4, zrs5, zrs6, zrs7, zrs8;

    // Preparation
    zrs0 = u[IDX3(x3-4,j,k)];
    zrs1 = u[IDX3(x3-3,j,k)];
    zrs2 = u[IDX3(x3-2,j,k)];
    zrs3 = u[IDX3(x3-1,j,k)];
    zrs4 = u[IDX3(x3+0,j,k)];
    zrs5 = u[IDX3(x3+1,j,k)];
    zrs6 = u[IDX3(x3+2,j,k)];
    zrs7 = u[IDX3(x3+3,j,k)];

    llint i = x3-1;

    while (true) {
        INNER_STENCIL(0, 1, 2, 3, 4, 5, 6, 7, 8);
        INNER_STENCIL(1, 2, 3, 4, 5, 6, 7, 8, 0);
        INNER_STENCIL(2, 3, 4, 5, 6, 7, 8, 0, 1);
        INNER_STENCIL(3, 4, 5, 6, 7, 8, 0, 1, 2);
        INNER_STENCIL(4, 5, 6, 7, 8, 0, 1, 2, 3);
        INNER_STENCIL(5, 6, 7, 8, 0, 1, 2, 3, 4);
        INNER_STENCIL(6, 7, 8, 0, 1, 2, 3, 4, 5);
        INNER_STENCIL(7, 8, 0, 1, 2, 3, 4, 5, 6);
        INNER_STENCIL(8, 0, 1, 2, 3, 4, 5, 6, 7);
    }
}

__global__ void __launch_bounds__(1024) kernel_7r_25d_pml(
    llint nx, llint ny, llint nz, int ldimx, int ldimy, int ldimz,
    llint x3, llint x4, llint y3, llint y4, llint z3, llint z4,
    llint lx, llint ly, llint lz,
    float hdx_2, float hdy_2, float hdz_2,
    const float *__restrict__ u, float *__restrict__ v, const float *__restrict__ vp,
    float *__restrict__ phi, const float *__restrict__ eta
) {
    __shared__ float s_u[N_THREADS_Y_DIM+2*N_RADIUS][N_THREADS_X_DIM+2*N_RADIUS];

    const llint j0 = y3 + blockIdx.y * blockDim.y;
    const llint k0 = z3 + blockIdx.x * blockDim.x;

    const llint j = j0 + threadIdx.y;
    const llint k = k0 + threadIdx.x;

    const llint suj = threadIdx.y + N_RADIUS;
    const llint suk = threadIdx.x + N_RADIUS;

    float zrs0, zrs1, zrs2, zrs3, zrs4, zrs5, zrs6, zrs7, zrs8;

    // Preparation
    zrs0 = u[IDX3(x3-4,j,k)];
    zrs1 = u[IDX3(x3-3,j,k)];
    zrs2 = u[IDX3(x3-2,j,k)];
    zrs3 = u[IDX3(x3-1,j,k)];
    zrs4 = u[IDX3(x3+0,j,k)];
    zrs5 = u[IDX3(x3+1,j,k)];
    zrs6 = u[IDX3(x3+2,j,k)];
    zrs7 = u[IDX3(x3+3,j,k)];

    llint i = x3-1;

    while (true) {
        PML_STENCIL(0, 1, 2, 3, 4, 5, 6, 7, 8);
        PML_STENCIL(1, 2, 3, 4, 5, 6, 7, 8, 0);
        PML_STENCIL(2, 3, 4, 5, 6, 7, 8, 0, 1);
        PML_STENCIL(3, 4, 5, 6, 7, 8, 0, 1, 2);
        PML_STENCIL(4, 5, 6, 7, 8, 0, 1, 2, 3);
        PML_STENCIL(5, 6, 7, 8, 0, 1, 2, 3, 4);
        PML_STENCIL(6, 7, 8, 0, 1, 2, 3, 4, 5);
        PML_STENCIL(7, 8, 0, 1, 2, 3, 4, 5, 6);
        PML_STENCIL(8, 0, 1, 2, 3, 4, 5, 6, 7);
    }
}

__global__ void kernel_add_source_kernel(float *g_u, llint idx, float source) {
    g_u[idx] += source;
}

extern "C" void target(
    uint nsteps, double *time_kernel,
    const grid_t grid,
    llint sx, llint sy, llint sz,
    float hdx_2, float hdy_2, float hdz_2,
    const float *__restrict__ coefx, const float *__restrict__ coefy, const float *__restrict__ coefz,
    float *__restrict__ u, const float *__restrict__ v, const float *__restrict__ vp,
    const float *__restrict__ phi, const float *__restrict__ eta, const float *__restrict__ source
) {
    struct timespec start, end;

    float *d_u = allocateDeviceGrid(grid);
    float *d_v = allocateDeviceGrid(grid);
    float *d_phi = allocateDeviceGrid(grid);
    float *d_eta = allocateDeviceGrid(grid);
    float *d_vp = allocateDeviceGrid(grid);

    hipMemset (d_u, 0, gridSize(grid));
    hipMemset (d_v, 0, gridSize(grid));
    hipMemcpy(d_vp, vp, gridSize(grid), hipMemcpyDefault);
    hipMemcpy(d_phi, phi, gridSize(grid), hipMemcpyDefault);
    hipMemcpy(d_eta, eta, gridSize(grid), hipMemcpyDefault);

    float coef0 = coefx[0] + coefy[0] + coefz[0];
    hipMemcpyToSymbol (HIP_SYMBOL(c_coef0), &coef0, sizeof (float));
    hipMemcpyToSymbol (HIP_SYMBOL(c_coefx), coefx, (N_RADIUS + 1) * sizeof (float));
    hipMemcpyToSymbol (HIP_SYMBOL(c_coefy), coefy, (N_RADIUS + 1) * sizeof (float));
    hipMemcpyToSymbol (HIP_SYMBOL(c_coefz), coefz, (N_RADIUS + 1) * sizeof (float));

    const llint xmin = 0; const llint xmax = grid.nx;
    const llint ymin = 0; const llint ymax = grid.ny;

    dim3 threadsPerBlock(N_THREADS_X_DIM, N_THREADS_Y_DIM, 1);

    int num_streams = 7;
    hipStream_t streams[num_streams];
    for (int i = 0; i < num_streams; i++) {
        hipStreamCreateWithFlags(&(streams[i]), hipStreamNonBlocking);
    }

    const uint npo = 100;
    for (uint istep = 1; istep <= nsteps; ++istep) {
        clock_gettime(CLOCK_REALTIME, &start);

        dim3 n_block_front(
            (grid.z2-grid.z1+N_THREADS_X_DIM-1) / N_THREADS_X_DIM,
            (grid.ny+N_THREADS_Y_DIM-1) / N_THREADS_Y_DIM,
            1);
        hipLaunchKernelGGL(kernel_7r_25d_pml, dim3(n_block_front), dim3(threadsPerBlock), 0, streams[1], grid.nx, grid.ny, grid.nz,
            grid.ldimx, grid.ldimy, grid.ldimz,
            xmin, xmax, ymin, ymax, grid.z1, grid.z2,
            grid.lx, grid.ly, grid.lz,
            hdx_2, hdy_2, hdz_2,
            d_u, d_v, d_vp,
            d_phi, d_eta);

        dim3 n_block_top(
            (grid.z4-grid.z3+N_THREADS_X_DIM-1) / N_THREADS_X_DIM,
            (grid.y2-grid.y1+N_THREADS_Y_DIM-1) / N_THREADS_Y_DIM,
            1);
        hipLaunchKernelGGL(kernel_7r_25d_pml, dim3(n_block_top), dim3(threadsPerBlock), 0, streams[2], grid.nx, grid.ny, grid.nz,
            grid.ldimx, grid.ldimy, grid.ldimz,
            xmin,xmax,grid.y1,grid.y2,grid.z3,grid.z4,
            grid.lx, grid.ly, grid.lz,
            hdx_2, hdy_2, hdz_2,
            d_u, d_v, d_vp,
            d_phi, d_eta);

        dim3 n_block_left(
            (grid.z4-grid.z3+N_THREADS_X_DIM-1) / N_THREADS_X_DIM,
            (grid.y4-grid.y3+N_THREADS_Y_DIM-1) / N_THREADS_Y_DIM,
            1);
        hipLaunchKernelGGL(kernel_7r_25d_pml, dim3(n_block_left), dim3(threadsPerBlock), 0, streams[3], grid.nx, grid.ny, grid.nz,
            grid.ldimx, grid.ldimy, grid.ldimz,
            grid.x1,grid.x2,grid.y3,grid.y4,grid.z3,grid.z4,
            grid.lx, grid.ly, grid.lz,
            hdx_2, hdy_2, hdz_2,
            d_u, d_v, d_vp,
            d_phi, d_eta);

        dim3 n_block_center(
            (grid.z4-grid.z3+N_THREADS_X_DIM-1) / N_THREADS_X_DIM,
            (grid.y4-grid.y3+N_THREADS_Y_DIM-1) / N_THREADS_Y_DIM,
            1);
        hipLaunchKernelGGL(kernel_7r_25d_inner, dim3(n_block_center), dim3(threadsPerBlock), 0, streams[0], grid.nx, grid.ny, grid.nz,
            grid.ldimx, grid.ldimy, grid.ldimz,
            grid.x3,grid.x4,grid.y3,grid.y4,grid.z3,grid.z4,
            grid.lx, grid.ly, grid.lz,
            hdx_2, hdy_2, hdz_2,
            d_u, d_v, d_vp,
            d_phi, d_eta);

        dim3 n_block_right(
            (grid.z4-grid.z3+N_THREADS_X_DIM-1) / N_THREADS_X_DIM,
            (grid.y4-grid.y3+N_THREADS_Y_DIM-1) / N_THREADS_Y_DIM,
            1);
        hipLaunchKernelGGL(kernel_7r_25d_pml, dim3(n_block_right), dim3(threadsPerBlock), 0, streams[4], grid.nx, grid.ny, grid.nz,
            grid.ldimx, grid.ldimy, grid.ldimz,
            grid.x5,grid.x6,grid.y3,grid.y4,grid.z3,grid.z4,
            grid.lx, grid.ly, grid.lz,
            hdx_2, hdy_2, hdz_2,
            d_u, d_v, d_vp,
            d_phi, d_eta);

        dim3 n_block_bottom(
            (grid.z4-grid.z3+N_THREADS_X_DIM-1) / N_THREADS_X_DIM,
            (grid.y6-grid.y5+N_THREADS_Y_DIM-1) / N_THREADS_Y_DIM,
            1);
        hipLaunchKernelGGL(kernel_7r_25d_pml, dim3(n_block_bottom), dim3(threadsPerBlock), 0, streams[5], grid.nx, grid.ny, grid.nz,
            grid.ldimx, grid.ldimy, grid.ldimz,
            xmin,xmax,grid.y5,grid.y6,grid.z3,grid.z4,
            grid.lx, grid.ly, grid.lz,
            hdx_2, hdy_2, hdz_2,
            d_u, d_v, d_vp,
            d_phi, d_eta);

        dim3 n_block_back(
            (grid.z6-grid.z5+N_THREADS_X_DIM-1) / N_THREADS_X_DIM,
            (grid.ny+N_THREADS_Y_DIM-1) / N_THREADS_Y_DIM,
            1);
        hipLaunchKernelGGL(kernel_7r_25d_pml, dim3(n_block_back), dim3(threadsPerBlock), 0, streams[6], grid.nx, grid.ny, grid.nz,
            grid.ldimx, grid.ldimy, grid.ldimz,
            xmin,xmax,ymin,ymax,grid.z5,grid.z6,
            grid.lx, grid.ly, grid.lz,
            hdx_2, hdy_2, hdz_2,
            d_u, d_v, d_vp,
            d_phi, d_eta);

        for (int i = 0; i < num_streams; i++) {
            hipStreamSynchronize(streams[i]);
        }

        hipLaunchKernelGGL(kernel_add_source_kernel, dim3(1), dim3(1), 0, 0, d_v, IDX3_grid(sx,sy,sz,grid), source[istep-1]);
        clock_gettime(CLOCK_REALTIME, &end);
        *time_kernel += (end.tv_sec  - start.tv_sec) +
                        (double)(end.tv_nsec - start.tv_nsec) / 1.0e9;

        float *t = d_u;
        d_u = d_v;
        d_v = t;

        // Print out
        if (istep % npo == 0) {
            printf("time step %u / %u\n", istep, nsteps);
        }
    }


    for (int i = 0; i < num_streams; i++) {
        hipStreamDestroy(streams[i]);
    }


    hipMemcpy(u, d_u, gridSize(grid), hipMemcpyDeviceToHost);

    // for (int i = 0; i < grid.nx; i++) {
    //     for (int j = 0; j < grid.ny; j++) {
    //         for (int k = 0; k < grid.nz; k++) {
    //             float f_n = u[IDX3_grid(i,j,k,grid)];
    //             // float f_n = eta[IDX3_grid(i,j,k,grid)];
    //             printf("(%3d,%3d,%3d):\t%x\n", i, j, k, *(unsigned int*)&f_n);
    //         }
    //         printf("\n");
    //     }
    //     printf("\n");
    // }
    // printf("post-computation: %f %f %f\n", hdx_2, hdy_2, hdz_2);
    // printf("post-computation: %3d %3d %3d\n", *(unsigned int*)&hdx_2, *(unsigned int*)&hdy_2, *(unsigned int*)&hdz_2);

    freeDeviceGrid(d_u, grid);
    freeDeviceGrid(d_v, grid);
    freeDeviceGrid(d_vp, grid);
    freeDeviceGrid(d_phi, grid);
    freeDeviceGrid(d_eta, grid);
}
