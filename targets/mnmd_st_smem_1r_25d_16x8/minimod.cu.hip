#include <hip/hip_runtime.h>

#include <stdio.h>
#include <float.h>
#include <math.h>
#include <time.h>

#include "../../constants.h"
#include "../../grid.h"

#define N_RADIUS 4
#define N_THREADS_X_DIM 16
#define N_THREADS_Y_DIM 8
#define N_THREADS_Z_DIM 0

// Constant memory coefficients
__constant__ float c_coef0;
__constant__ float c_coefx[N_RADIUS+1];
__constant__ float c_coefy[N_RADIUS+1];
__constant__ float c_coefz[N_RADIUS+1];

__global__ void __launch_bounds__(256) kernel_1r_25d(
    llint nx, llint ny, llint nz, int ldimx, int ldimy, int ldimz,
    llint x3, llint x4, llint y3, llint y4, llint z3, llint z4,
    llint lx, llint ly, llint lz,
    float hdx_2, float hdy_2, float hdz_2,
    const float *__restrict__ u, float *__restrict__ v, const float *__restrict__ vp,
    float *__restrict__ phi, const float *__restrict__ eta
) {
    const int nz_s_u = 1+2*N_RADIUS;
    __shared__ float s_u[nz_s_u][N_THREADS_Y_DIM+2*N_RADIUS][N_THREADS_X_DIM+2*N_RADIUS];

    const llint j0 = blockIdx.y * blockDim.y;
    const llint k0 = blockIdx.x * blockDim.x;

    const llint je = min(j0 + N_THREADS_Y_DIM, ny);
    const llint ke = min(k0 + N_THREADS_X_DIM, nz);

    const llint j = j0 + threadIdx.y;
    const llint k = k0 + threadIdx.x;

    const llint sje = (j0+N_THREADS_Y_DIM<ny) ? N_THREADS_Y_DIM : ((ny-1)%N_THREADS_Y_DIM+1);
    const llint ske = (k0+N_THREADS_X_DIM<nz) ? N_THREADS_X_DIM : ((nz-1)%N_THREADS_X_DIM+1);

    const llint suj = threadIdx.y + N_RADIUS;
    const llint suk = threadIdx.x + N_RADIUS;

    // Preparation
    for (llint i = -N_RADIUS; i < N_RADIUS; i++) {
        int z = (i + nz_s_u) % nz_s_u;
        s_u[z][suj][suk] = u[IDX3(i,j,k)];
        if (threadIdx.y < N_RADIUS) {
            s_u[z][threadIdx.y][suk] = u[IDX3(i, j - N_RADIUS, k)];
            s_u[z][threadIdx.y+sje+N_RADIUS][suk] = u[IDX3(i, threadIdx.y+je, k)];
        }
        if (threadIdx.x < N_RADIUS) {
            s_u[z][suj][threadIdx.x] = u[IDX3(i,j,k - N_RADIUS)];
            s_u[z][suj][threadIdx.x+ske+N_RADIUS] = u[IDX3(i,j,threadIdx.x+ke)];
        }
    }

    // Streaming
    for (llint i = 0; i < nx; i++) {
        int z = (i + nz_s_u) % nz_s_u;
        int z_R = (i + N_RADIUS + nz_s_u) % nz_s_u;

        s_u[z_R][suj][suk] = u[IDX3(i+N_RADIUS,j,k)];
        if (threadIdx.y < N_RADIUS) {
            s_u[z_R][threadIdx.y][suk] = u[IDX3(i+N_RADIUS, j - N_RADIUS, k)];
            s_u[z_R][threadIdx.y+sje+N_RADIUS][suk] = u[IDX3(i+N_RADIUS, threadIdx.y+je, k)];
        }
        if (threadIdx.x < N_RADIUS) {
            s_u[z_R][suj][threadIdx.x] = u[IDX3(i+N_RADIUS,j,k - N_RADIUS)];
            s_u[z_R][suj][threadIdx.x+ske+N_RADIUS] = u[IDX3(i+N_RADIUS,j,threadIdx.x+ke)];
        }
        __syncthreads();

        if (j < ny && k < nz) {
            float lap = __fmaf_rn(c_coef0, s_u[z][suj][suk]
                    , __fmaf_rn(c_coefx[1], __fadd_rn(s_u[(z+1)%nz_s_u][suj][suk],s_u[(z-1+nz_s_u)%nz_s_u][suj][suk])
                    , __fmaf_rn(c_coefy[1], __fadd_rn(s_u[z][suj+1][suk],s_u[z][suj-1][suk])
                    , __fmaf_rn(c_coefz[1], __fadd_rn(s_u[z][suj][suk+1],s_u[z][suj][suk-1])
                    , __fmaf_rn(c_coefx[2], __fadd_rn(s_u[(z+2)%nz_s_u][suj][suk],s_u[(z-2+nz_s_u)%nz_s_u][suj][suk])
                    , __fmaf_rn(c_coefy[2], __fadd_rn(s_u[z][suj+2][suk],s_u[z][suj-2][suk])
                    , __fmaf_rn(c_coefz[2], __fadd_rn(s_u[z][suj][suk+2],s_u[z][suj][suk-2])
                    , __fmaf_rn(c_coefx[3], __fadd_rn(s_u[(z+3)%nz_s_u][suj][suk],s_u[(z-3+nz_s_u)%nz_s_u][suj][suk])
                    , __fmaf_rn(c_coefy[3], __fadd_rn(s_u[z][suj+3][suk],s_u[z][suj-3][suk])
                    , __fmaf_rn(c_coefz[3], __fadd_rn(s_u[z][suj][suk+3],s_u[z][suj][suk-3])
                    , __fmaf_rn(c_coefx[4], __fadd_rn(s_u[(z+4)%nz_s_u][suj][suk],s_u[(z-4+nz_s_u)%nz_s_u][suj][suk])
                    , __fmaf_rn(c_coefy[4], __fadd_rn(s_u[z][suj+4][suk],s_u[z][suj-4][suk])
                    , __fmul_rn(c_coefz[4], __fadd_rn(s_u[z][suj][suk+4],s_u[z][suj][suk-4])
            )))))))))))));

            if (i >= x3 && i < x4 && j >= y3 && j < y4 && k >= z3 && k < z4) {
                v[IDX3(i,j,k)] = __fmaf_rn(2.f, s_u[z][suj][suk],
                    __fmaf_rn(vp[IDX3(i,j,k)], lap, -v[IDX3(i,j,k)])
                );
            } else {
                const float s_eta_c = eta[IDX3(i,j,k)];

                v[IDX3(i,j,k)] = __fdiv_rn(
                    __fmaf_rn(
                        __fmaf_rn(2.f, s_eta_c,
                            __fsub_rn(2.f,
                                __fmul_rn(s_eta_c, s_eta_c)
                            )
                        ),
                        s_u[z][suj][suk],
                        __fmaf_rn(
                            vp[IDX3(i,j,k)],
                            __fadd_rn(lap, phi[IDX3(i,j,k)]),
                            -v[IDX3(i,j,k)]
                        )
                    ),
                    __fmaf_rn(2.f, s_eta_c, 1.f)
                );

                phi[IDX3(i,j,k)] = __fdiv_rn(
                        __fsub_rn(
                            phi[IDX3(i,j,k)],
                            __fmaf_rn(
                            __fmul_rn(
                                __fsub_rn(eta[IDX3(i+1,j,k)], eta[IDX3(i-1,j,k)]),
                                __fsub_rn(s_u[(z+1)%nz_s_u][suj][suk], s_u[(z-1+nz_s_u)%nz_s_u][suj][suk])
                            ), hdx_2,
                            __fmaf_rn(
                            __fmul_rn(
                                __fsub_rn(eta[IDX3(i,j+1,k)], eta[IDX3(i,j-1,k)]),
                                __fsub_rn(s_u[z][suj+1][suk], s_u[z][suj-1][suk])
                            ), hdy_2,
                            __fmul_rn(
                                __fmul_rn(
                                    __fsub_rn(eta[IDX3(i,j,k+1)], eta[IDX3(i,j,k-1)]),
                                    __fsub_rn(s_u[z][suj][suk+1], s_u[z][suj][suk-1])
                                ),
                            hdz_2)
                            ))
                        )
                    ,
                    __fadd_rn(1.f, s_eta_c)
                );
            }
        }
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

    dim3 threadsPerBlock(N_THREADS_X_DIM, N_THREADS_Y_DIM, 1);

    const uint npo = 100;
    for (uint istep = 1; istep <= nsteps; ++istep) {
        clock_gettime(CLOCK_REALTIME, &start);

        dim3 n_block_front(
            (grid.nz+N_THREADS_X_DIM-1) / N_THREADS_X_DIM,
            (grid.ny+N_THREADS_Y_DIM-1) / N_THREADS_Y_DIM);
        hipLaunchKernelGGL(kernel_1r_25d, dim3(n_block_front), dim3(threadsPerBlock), 0, 0, grid.nx,grid.ny,grid.nz,
            grid.ldimx, grid.ldimy, grid.ldimz,
            grid.x3,grid.x4,grid.y3,grid.y4,grid.z3,grid.z4,
            grid.lx,grid.ly,grid.lz,
            hdx_2, hdy_2, hdz_2,
            d_u, d_v, d_vp,
            d_phi, d_eta);

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

    hipMemcpy(u, d_u, gridSize(grid), hipMemcpyDeviceToHost);

    freeDeviceGrid(d_u, grid);
    freeDeviceGrid(d_v, grid);
    freeDeviceGrid(d_vp, grid);
    freeDeviceGrid(d_phi, grid);
    freeDeviceGrid(d_eta, grid);
}
