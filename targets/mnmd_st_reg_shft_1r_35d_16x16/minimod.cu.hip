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

__constant__ float c_source[1000];
__constant__ llint c_sidx;

__global__ void __launch_bounds__(576) kernel_1r_35d(
    uint istep,
    llint nx, llint ny, llint nz, int ldimx, int ldimy, int ldimz,
    llint x3, llint x4, llint y3, llint y4, llint z3, llint z4,
    llint lx, llint ly, llint lz,
    float hdx_2, float hdy_2, float hdz_2,
    const float *__restrict__ u, const float *__restrict__ v,
    float *__restrict__ u1, float *__restrict__ v1,
    const float *__restrict__ vp, const float *__restrict__ phi, float *__restrict__ phi1, const float *__restrict__ eta
) {
    const int nz_t0u = 1+2*N_RADIUS;
    const int nz_t0phi = 1+N_RADIUS;
    __shared__ float s_u[N_THREADS_Y_DIM+2*N_RADIUS][N_THREADS_X_DIM+2*N_RADIUS];
    __shared__ float s_t0u[nz_t0u][N_THREADS_Y_DIM][N_THREADS_X_DIM];
    __shared__ float s_t0phi[nz_t0phi][N_THREADS_Y_DIM][N_THREADS_X_DIM];

    const llint j0 = (llint)blockIdx.y * (N_THREADS_Y_DIM - 2 * N_RADIUS) - N_RADIUS;
    const llint k0 = (llint)blockIdx.x * (N_THREADS_X_DIM - 2 * N_RADIUS) - N_RADIUS;

    const llint je = min(j0 + N_THREADS_Y_DIM, ny + N_RADIUS);
    const llint ke = min(k0 + N_THREADS_X_DIM, nz + N_RADIUS);

    const llint j = j0 + threadIdx.y;
    const llint k = k0 + threadIdx.x;

    const llint sje = (j0+N_THREADS_Y_DIM<=ny) ? N_THREADS_Y_DIM : ((ny-1)%N_THREADS_Y_DIM+1);
    const llint ske = (k0+N_THREADS_X_DIM<=nz) ? N_THREADS_X_DIM : ((nz-1)%N_THREADS_X_DIM+1);
                                    //    ^ should be <=

    const llint suj = threadIdx.y + N_RADIUS;
    const llint suk = threadIdx.x + N_RADIUS;

    float infront1, infront2, infront3, infront4; // variables for input "in front of" the current slice
    float behind1, behind2, behind3, behind4; // variables for input "behind" the current slice
    float current; // input value in the current slice

    float lap, lap2;

    /* load four plane above the top pml layer */
    if (j < ny + N_RADIUS && k < nz + N_RADIUS) {
        behind3 = u[IDX3(-4,j,k)];
        behind2 = u[IDX3(-3,j,k)];
        behind1 = u[IDX3(-2,j,k)];
        current  = u[IDX3(-1,j,k)];
        infront1 = u[IDX3(0,j,k)];
        infront2 = u[IDX3(1,j,k)];
        infront3 = u[IDX3(2,j,k)];
        infront4 = u[IDX3(3,j,k)];

        s_t0u[nz_t0u-4][threadIdx.y][threadIdx.x] = u[IDX3(-4,j,k)];
        s_t0u[nz_t0u-3][threadIdx.y][threadIdx.x] = u[IDX3(-3,j,k)];
        s_t0u[nz_t0u-2][threadIdx.y][threadIdx.x] = u[IDX3(-2,j,k)];
        s_t0u[nz_t0u-1][threadIdx.y][threadIdx.x] = u[IDX3(-1,j,k)];
    }

    /* enter the streaming from 0 to nx, and then four more planes below the bottom pml layer */
    for (llint i = 0; i < nx + N_RADIUS; i++) {
        /* calculate the buffer index for the current active plan */
        const int z = i % nz_t0u;

        __syncthreads();

        /* first time step */
        if (i < nx) {

            if (j < ny + N_RADIUS && k < nz + N_RADIUS) {
                behind4  = behind3;
                behind3  = behind2;
                behind2  = behind1;
                behind1  = current;
                current  = infront1;
                infront1 = infront2;
                infront2 = infront3;
                infront3 = infront4;
                infront4 = u[IDX3(i+N_RADIUS,j,k)];
            }

            if (threadIdx.y < N_RADIUS) {
                s_u[threadIdx.y][suk] = u[IDX3(i, j-N_RADIUS, k)];
                s_u[threadIdx.y+sje+N_RADIUS][suk] = u[IDX3(i, threadIdx.y+je, k)];
            }
            if (threadIdx.x < N_RADIUS) {
                s_u[suj][threadIdx.x] = u[IDX3(i,j,k-N_RADIUS)];
                s_u[suj][threadIdx.x+ske+N_RADIUS] = u[IDX3(i,j,threadIdx.x+ke)];
            }
            s_u[suj][suk] = u[IDX3(i,j,k)];

            s_t0u[z][threadIdx.y][threadIdx.x] = 0.0;
            s_t0phi[i%nz_t0phi][threadIdx.y][threadIdx.x] = 0.0;
        }

        __syncthreads();

        /* first time step */
        if (i < nx) {

            if (j < 0 || j >= ny || k < 0 || k >= nz) {
                /* if the position fall into the thin boundary beyond the data grid, just load it into s_t0u */
                if (j < ny + N_RADIUS && k < nz + N_RADIUS) {
                    s_t0u[z][threadIdx.y][threadIdx.x] = u[IDX3(i,j,k)];
                }
            } else {
                /* otherwise, do the actual stencil logic */
                /* for the first time step, we read directly from u. */
                lap = __fmaf_rn(c_coef0, current
                        , __fmaf_rn(c_coefx[1], __fadd_rn(infront1,behind1)
                        , __fmaf_rn(c_coefy[1], __fadd_rn(s_u[suj+1][suk],s_u[suj-1][suk])
                        , __fmaf_rn(c_coefz[1], __fadd_rn(s_u[suj][suk+1],s_u[suj][suk-1])
                        , __fmaf_rn(c_coefx[2], __fadd_rn(infront2,behind2)
                        , __fmaf_rn(c_coefy[2], __fadd_rn(s_u[suj+2][suk],s_u[suj-2][suk])
                        , __fmaf_rn(c_coefz[2], __fadd_rn(s_u[suj][suk+2],s_u[suj][suk-2])
                        , __fmaf_rn(c_coefx[3], __fadd_rn(infront3,behind3)
                        , __fmaf_rn(c_coefy[3], __fadd_rn(s_u[suj+3][suk],s_u[suj-3][suk])
                        , __fmaf_rn(c_coefz[3], __fadd_rn(s_u[suj][suk+3],s_u[suj][suk-3])
                        , __fmaf_rn(c_coefx[4], __fadd_rn(infront4,behind4)
                        , __fmaf_rn(c_coefy[4], __fadd_rn(s_u[suj+4][suk],s_u[suj-4][suk])
                        , __fmul_rn(c_coefz[4], __fadd_rn(s_u[suj][suk+4],s_u[suj][suk-4])
                )))))))))))));

                if (i >= x3 && i < x4 && j >= y3 && j < y4 && k >= z3 && k < z4) {
                    /* if this is the inner layer, compute as normal, but instead of uploading v, store the result into s_t0u */
                    s_t0u[z][threadIdx.y][threadIdx.x] = __fmaf_rn(2.f, current,
                        __fmaf_rn(vp[IDX3(i,j,k)], lap, -v[IDX3(i,j,k)])
                    );
                } else {
                    /* if this is the boundary layer */
                    const float s_eta_c = eta[IDX3(i,j,k)];

                    s_t0u[z][threadIdx.y][threadIdx.x] = __fdiv_rn(
                        __fmaf_rn(
                            __fmaf_rn(2.f, s_eta_c,
                                __fsub_rn(2.f,
                                    __fmul_rn(s_eta_c, s_eta_c)
                                )
                            ),
                            current,
                            __fmaf_rn(
                                vp[IDX3(i,j,k)],
                                __fadd_rn(lap, phi[IDX3(i,j,k)]),
                                -v[IDX3(i,j,k)]
                            )
                        ),
                        __fmaf_rn(2.f, s_eta_c, 1.f)
                    );

                    s_t0phi[i%nz_t0phi][threadIdx.y][threadIdx.x] = __fdiv_rn(
                            __fsub_rn(
                                phi[IDX3(i,j,k)],
                                __fmaf_rn(
                                __fmul_rn(
                                    __fsub_rn(eta[IDX3(i+1,j,k)], eta[IDX3(i-1,j,k)]),
                                    __fsub_rn(infront1,behind1)
                                ), hdx_2,
                                __fmaf_rn(
                                __fmul_rn(
                                    __fsub_rn(eta[IDX3(i,j+1,k)], eta[IDX3(i,j-1,k)]),
                                    __fsub_rn(s_u[suj+1][suk], s_u[suj-1][suk])
                                ), hdy_2,
                                __fmul_rn(
                                    __fmul_rn(
                                        __fsub_rn(eta[IDX3(i,j,k+1)], eta[IDX3(i,j,k-1)]),
                                        __fsub_rn(s_u[suj][suk+1], s_u[suj][suk-1])
                                    ),
                                hdz_2)
                                ))
                            )
                        ,
                        __fadd_rn(1.f, s_eta_c)
                    );
                }
            }

            /* if this is *the* point, then update it using the `source` */
            if (IDX3(i,j,k) == c_sidx) {
                s_t0u[z][threadIdx.y][threadIdx.x] += c_source[istep-1];
            }

        } else {
            /* this is for the planes below the bottom pml layer (nx to nx + N_RADIUS), so just load the value */
            if (j < ny + N_RADIUS && k < nz + N_RADIUS) {
                s_t0u[z][threadIdx.y][threadIdx.x] = u[IDX3(i,j,k)];
            } else {
                s_t0u[z][threadIdx.y][threadIdx.x] = 0.0;
            }
            s_t0phi[i%nz_t0phi][threadIdx.y][threadIdx.x] = 0.0;
        }

        __syncthreads();

        /*
            enter second time step

            i >= N_RADIUS : for second time step, do it from N_RADIUS to nx + N_RADIUS

            !((j < j0 + N_RADIUS ||
            j >= je - N_RADIUS ||
            k < k0 + N_RADIUS ||
            k >= ke - N_RADIUS)) : back away from boundary
         */

        if (i >= N_RADIUS && !(j < j0 + N_RADIUS ||
            j >= je - N_RADIUS ||
            k < k0 + N_RADIUS ||
            k >= ke - N_RADIUS))
        {
            /* streaming index */
            llint i2 = i - N_RADIUS;
            /* buffer index */
            const int z2 = i2%nz_t0u;

            lap2 = __fmaf_rn(c_coef0, s_t0u[z2][threadIdx.y][threadIdx.x]
                        , __fmaf_rn(c_coefx[1], __fadd_rn(s_t0u[(z2+1)%nz_t0u][threadIdx.y][threadIdx.x],
                                                            s_t0u[(z2-1+nz_t0u)%nz_t0u][threadIdx.y][threadIdx.x])
                        , __fmaf_rn(c_coefy[1], __fadd_rn(s_t0u[z2][threadIdx.y+1][threadIdx.x],
                                                            s_t0u[z2][threadIdx.y-1][threadIdx.x])
                        , __fmaf_rn(c_coefz[1], __fadd_rn(s_t0u[z2][threadIdx.y][threadIdx.x+1],
                                                            s_t0u[z2][threadIdx.y][threadIdx.x-1])
                        , __fmaf_rn(c_coefx[2], __fadd_rn(s_t0u[(z2+2)%nz_t0u][threadIdx.y][threadIdx.x],
                                                            s_t0u[(z2-2+nz_t0u)%nz_t0u][threadIdx.y][threadIdx.x])
                        , __fmaf_rn(c_coefy[2], __fadd_rn(s_t0u[z2][threadIdx.y+2][threadIdx.x],
                                                            s_t0u[z2][threadIdx.y-2][threadIdx.x])
                        , __fmaf_rn(c_coefz[2], __fadd_rn(s_t0u[z2][threadIdx.y][threadIdx.x+2],
                                                            s_t0u[z2][threadIdx.y][threadIdx.x-2])
                        , __fmaf_rn(c_coefx[3], __fadd_rn(s_t0u[(z2+3)%nz_t0u][threadIdx.y][threadIdx.x],
                                                            s_t0u[(z2-3+nz_t0u)%nz_t0u][threadIdx.y][threadIdx.x])
                        , __fmaf_rn(c_coefy[3], __fadd_rn(s_t0u[z2][threadIdx.y+3][threadIdx.x],
                                                            s_t0u[z2][threadIdx.y-3][threadIdx.x])
                        , __fmaf_rn(c_coefz[3], __fadd_rn(s_t0u[z2][threadIdx.y][threadIdx.x+3],
                                                            s_t0u[z2][threadIdx.y][threadIdx.x-3])
                        , __fmaf_rn(c_coefx[4], __fadd_rn(s_t0u[(z2+4)%nz_t0u][threadIdx.y][threadIdx.x],
                                                            s_t0u[(z2-4+nz_t0u)%nz_t0u][threadIdx.y][threadIdx.x])
                        , __fmaf_rn(c_coefy[4], __fadd_rn(s_t0u[z2][threadIdx.y+4][threadIdx.x],
                                                            s_t0u[z2][threadIdx.y-4][threadIdx.x])
                        , __fmul_rn(c_coefz[4], __fadd_rn(s_t0u[z2][threadIdx.y][threadIdx.x+4],
                                                            s_t0u[z2][threadIdx.y][threadIdx.x-4])
                )))))))))))));

            if (i2 >= x3 && i2 < x4 && j >= y3 && j < y4 && k >= z3 && k < z4) {
                /* if this is the inner region */
                u1[IDX3(i2,j,k)] = __fmaf_rn(2.f, s_t0u[z2][threadIdx.y][threadIdx.x],
                // ^ store to v                   ^ using the first-time-step result
                    // __fmaf_rn(vp[IDX3(i2,j,k)], lap2, -u[IDX3(i2,j,k)])
                    __fmaf_rn(vp[IDX3(i2,j,k)], lap2, -behind4)
                                            //  ^ lap2 ^ data in u stores two steps before
                );
            } else {
                const float s_eta_c = eta[IDX3(i2,j,k)];

                u1[IDX3(i2,j,k)] = __fdiv_rn(
                    __fmaf_rn(
                        __fmaf_rn(2.f, s_eta_c,
                            __fsub_rn(2.f,
                                __fmul_rn(s_eta_c, s_eta_c)
                            )
                        ),
                        s_t0u[z2][threadIdx.y][threadIdx.x],
                        __fmaf_rn(
                            vp[IDX3(i2,j,k)],
                            __fadd_rn(lap2, s_t0phi[i2%nz_t0phi][threadIdx.y][threadIdx.x]),
                            -u[IDX3(i2,j,k)]
                        )
                    ),
                    __fmaf_rn(2.f, s_eta_c, 1.f)
                );

                phi1[IDX3(i2,j,k)] = __fdiv_rn(
                        __fsub_rn(
                            s_t0phi[i2%nz_t0phi][threadIdx.y][threadIdx.x],
                            __fmaf_rn(
                            __fmul_rn(
                                __fsub_rn(eta[IDX3(i2+1,j,k)], eta[IDX3(i2-1,j,k)]),
                                __fsub_rn(s_t0u[(z2+1)%nz_t0u][threadIdx.y][threadIdx.x],s_t0u[(z2-1+nz_t0u)%nz_t0u][threadIdx.y][threadIdx.x])
                            ), hdx_2,
                            __fmaf_rn(
                            __fmul_rn(
                                __fsub_rn(eta[IDX3(i2,j+1,k)], eta[IDX3(i2,j-1,k)]),
                                __fsub_rn(s_t0u[z2][threadIdx.y+1][threadIdx.x],s_t0u[z2][threadIdx.y-1][threadIdx.x])
                            ), hdy_2,
                            __fmul_rn(
                                __fmul_rn(
                                    __fsub_rn(eta[IDX3(i2,j,k+1)], eta[IDX3(i2,j,k-1)]),
                                    __fsub_rn(s_t0u[z2][threadIdx.y][threadIdx.x+1],s_t0u[z2][threadIdx.y][threadIdx.x-1])
                                ),
                            hdz_2)
                            ))
                        )
                    ,
                    __fadd_rn(1.f, s_eta_c)
                );
            }

            if (IDX3(i2,j,k) == c_sidx) {
                /* if this is the place, increase v */
                u1[c_sidx] += c_source[istep];
            }

            /* store partial result to w */
            v1[IDX3(i2,j,k)] = s_t0u[z2][threadIdx.y][threadIdx.x];
        }

        // __syncthreads();

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
    float *d_u1 = allocateDeviceGrid(grid);
    float *d_v1 = allocateDeviceGrid(grid);
    float *d_phi = allocateDeviceGrid(grid);
    float *d_phi1 = allocateDeviceGrid(grid);
    float *d_eta = allocateDeviceGrid(grid);
    float *d_vp = allocateDeviceGrid(grid);

    hipMemset (d_u, 0, gridSize(grid));
    hipMemset (d_v, 0, gridSize(grid));
    hipMemset (d_u1, 0, gridSize(grid));
    hipMemset (d_v1, 0, gridSize(grid));
    hipMemset (d_phi1, 0, gridSize(grid));
    hipMemcpy(d_vp, vp, gridSize(grid), hipMemcpyDefault);
    hipMemcpy(d_phi, phi, gridSize(grid), hipMemcpyDefault);
    hipMemcpy(d_eta, eta, gridSize(grid), hipMemcpyDefault);

    float coef0 = coefx[0] + coefy[0] + coefz[0];
    hipMemcpyToSymbol (HIP_SYMBOL(c_coef0), &coef0, sizeof (float));
    hipMemcpyToSymbol (HIP_SYMBOL(c_coefx), coefx, (N_RADIUS + 1) * sizeof (float));
    hipMemcpyToSymbol (HIP_SYMBOL(c_coefy), coefy, (N_RADIUS + 1) * sizeof (float));
    hipMemcpyToSymbol (HIP_SYMBOL(c_coefz), coefz, (N_RADIUS + 1) * sizeof (float));

    llint sidx = IDX3_grid(sx,sy,sz,grid);
    hipMemcpyToSymbol (HIP_SYMBOL(c_source), source, nsteps * sizeof (float));
    hipMemcpyToSymbol (HIP_SYMBOL(c_sidx), &sidx, sizeof (llint));

    dim3 threadsPerBlock(N_THREADS_X_DIM, N_THREADS_Y_DIM, 1);
    int result_dim_x = N_THREADS_X_DIM - N_RADIUS * 2;
    int result_dim_y = N_THREADS_Y_DIM - N_RADIUS * 2;
    dim3 n_block(
        (grid.nz+result_dim_x-1) / result_dim_x,
        (grid.ny+result_dim_y-1) / result_dim_y);

    const uint npo = 100;
    for (uint istep = 1; istep <= nsteps; ++istep) {
        clock_gettime(CLOCK_REALTIME, &start);

        hipLaunchKernelGGL(kernel_1r_35d, dim3(n_block), dim3(threadsPerBlock), 0, 0, istep,
            grid.nx,grid.ny,grid.nz,
            grid.ldimx, grid.ldimy, grid.ldimz,
            grid.x3,grid.x4,grid.y3,grid.y4,grid.z3,grid.z4,
            grid.lx,grid.ly,grid.lz,
            hdx_2, hdy_2, hdz_2,
            d_u, d_v, d_u1, d_v1,
            d_vp, d_phi, d_phi1, d_eta);

        // kernel_add_source_kernel<<<1, 1>>>(d_v, IDX3_grid(sx,sy,sz,grid), source[istep]);
        clock_gettime(CLOCK_REALTIME, &end);
        *time_kernel += (end.tv_sec  - start.tv_sec) +
                        (double)(end.tv_nsec - start.tv_nsec) / 1.0e9;

        float *t;
        t = d_u;
        d_u = d_u1;
        d_u1 = t;
        t = d_v;
        d_v = d_v1;
        d_v1 = t;
        t = d_phi;
        d_phi = d_phi1;
        d_phi1 = t;

        istep++;

        // Print out
        if (istep % npo == 0) {
            printf("time step %u / %u\n", istep, nsteps);
        }
    }

    hipMemcpy(u, d_u, gridSize(grid), hipMemcpyDeviceToHost);

    // for (int i = 0; i < grid.nx; i++) {
    //     for (int j = 0; j < grid.ny; j++) {
    //         for (int k = 0; k < grid.nz; k++) {
    //             float f_n = u[IDX3_grid(i,j,k,grid)];
    //             printf("(%3d,%3d,%3d):\t%x\n", i, j, k, *(unsigned int*)&f_n);
    //         }
    //         printf("\n");
    //     }
    //     printf("\n");
    // }

    freeDeviceGrid(d_u, grid);
    freeDeviceGrid(d_v, grid);
    freeDeviceGrid(d_u1, grid);
    freeDeviceGrid(d_v1, grid);
    freeDeviceGrid(d_phi, grid);
    freeDeviceGrid(d_phi1, grid);
    freeDeviceGrid(d_vp, grid);
    freeDeviceGrid(d_eta, grid);
}
