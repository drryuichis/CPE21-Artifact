**[2021] Total E&P RT USA, LLC. and Rice University**
**All Rights Reserved.**

**NOTICE**:  All information contained herein is, and remains
the property of Total E&P RT USA, LLC. and Rice University.
The intellectual and technical concepts contained
herein are proprietary to Total E&P RT USA, LLC. and Rice University
and may be covered by U.S. and Foreign Patents,
patents in process, and are protected by trade secret or copyright law.

----

The code is provided as it is, use it at your own risk and expect no support, the authors are not liable under any circumstances.
Feedback is welcome.

contact: ryuichi@rice.edu

# CPE21-Artifact

## Compile and Run AMD Kernels

### MI200 gmem 7r 3d 32 8 4

```
HIPCCFLAGS="--amdgpu-target=gfx90a" TARGET=mnmd_gmem_7r_3d_32x8x4 COMPILER=hipcc make clean
HIPCCFLAGS="--amdgpu-target=gfx90a" TARGET=mnmd_gmem_7r_3d_32x8x4 COMPILER=hipcc make all
./main_mnmd_gmem_7r_3d_32x8x4_hipcc --grid 1000 --nsteps 1000 --niters 5 --warm-up
```

### MI200 gmem 7r 3d 64 4 4
```
HIPCCFLAGS="--amdgpu-target=gfx90a" TARGET=mnmd_gmem_7r_3d_64x4x4 COMPILER=hipcc make clean
HIPCCFLAGS="--amdgpu-target=gfx90a" TARGET=mnmd_gmem_7r_3d_64x4x4 COMPILER=hipcc make all
./main_mnmd_gmem_7r_3d_64x4x4_hipcc --grid 1000 --nsteps 1000 --niters 5 --warm-up
```

### MI200 smem_u 7r 3d 32 8 4
```
HIPCCFLAGS="--amdgpu-target=gfx90a" TARGET=mnmd_smem_u_7r_3d_32x8x4 COMPILER=hipcc make clean
HIPCCFLAGS="--amdgpu-target=gfx90a" TARGET=mnmd_smem_u_7r_3d_32x8x4 COMPILER=hipcc make all
./main_mnmd_smem_u_7r_3d_32x8x4_hipcc --grid 1000 --nsteps 1000 --niters 5 --warm-up
```

### MI200 smem_u 7r 3d 64 4 4
```
HIPCCFLAGS="--amdgpu-target=gfx90a" TARGET=mnmd_smem_u_7r_3d_64x4x4 COMPILER=hipcc make clean
HIPCCFLAGS="--amdgpu-target=gfx90a" TARGET=mnmd_smem_u_7r_3d_64x4x4 COMPILER=hipcc make all
./main_mnmd_smem_u_7r_3d_64x4x4_hipcc --grid 1000 --nsteps 1000 --niters 5 --warm-up
```

### MI200 st_smem 7r 25d 16 8 0
```
HIPCCFLAGS="--amdgpu-target=gfx90a" TARGET=mnmd_st_smem_7r_25d_16x8 COMPILER=hipcc make clean
HIPCCFLAGS="--amdgpu-target=gfx90a" TARGET=mnmd_st_smem_7r_25d_16x8 COMPILER=hipcc make all
./main_mnmd_st_smem_7r_25d_16x8_hipcc --grid 1000 --nsteps 1000 --niters 5 --warm-up
```

### MI200 st_smem 7r 25d 16 16 0
```
HIPCCFLAGS="--amdgpu-target=gfx90a" TARGET=mnmd_st_smem_7r_25d_16x16 COMPILER=hipcc make clean
HIPCCFLAGS="--amdgpu-target=gfx90a" TARGET=mnmd_st_smem_7r_25d_16x16 COMPILER=hipcc make all
./main_mnmd_st_smem_7r_25d_16x16_hipcc --grid 1000 --nsteps 1000 --niters 5 --warm-up
```

### MI200 st_reg_shft 7r 25d 16 16 0
```
HIPCCFLAGS="--amdgpu-target=gfx90a" TARGET=mnmd_st_reg_shft_7r_25d_16x16 COMPILER=hipcc make clean
HIPCCFLAGS="--amdgpu-target=gfx90a" TARGET=mnmd_st_reg_shft_7r_25d_16x16 COMPILER=hipcc make all
./main_mnmd_st_reg_shft_7r_25d_16x16_hipcc --grid 1000 --nsteps 1000 --niters 5 --warm-up
```

### MI200 st_reg_shft 7r 25d 32 16 0
```
HIPCCFLAGS="--amdgpu-target=gfx90a" TARGET=mnmd_st_reg_shft_7r_25d_32x16 COMPILER=hipcc make clean
HIPCCFLAGS="--amdgpu-target=gfx90a" TARGET=mnmd_st_reg_shft_7r_25d_32x16 COMPILER=hipcc make all
./main_mnmd_st_reg_shft_7r_25d_32x16_hipcc --grid 1000 --nsteps 1000 --niters 5 --warm-up
```

### MI200 st_reg_shft 7r 25d 32 32 0
```
HIPCCFLAGS="--amdgpu-target=gfx90a" TARGET=mnmd_st_reg_shft_7r_25d_32x32 COMPILER=hipcc make clean
HIPCCFLAGS="--amdgpu-target=gfx90a" TARGET=mnmd_st_reg_shft_7r_25d_32x32 COMPILER=hipcc make all
./main_mnmd_st_reg_shft_7r_25d_32x32_hipcc --grid 1000 --nsteps 1000 --niters 5 --warm-up
```

### MI200 st_reg_shft_db 7r 25d 16 16 0
```
HIPCCFLAGS="--amdgpu-target=gfx90a" TARGET=mnmd_st_reg_shft_db_7r_25d_16x16 COMPILER=hipcc make clean
HIPCCFLAGS="--amdgpu-target=gfx90a" TARGET=mnmd_st_reg_shft_db_7r_25d_16x16 COMPILER=hipcc make all
./main_mnmd_st_reg_shft_db_7r_25d_16x16_hipcc --grid 1000 --nsteps 1000 --niters 5 --warm-up
```

### MI200 st_reg_shft_db 7r 25d 32 16 0
```
HIPCCFLAGS="--amdgpu-target=gfx90a" TARGET=mnmd_st_reg_shft_db_7r_25d_32x16 COMPILER=hipcc make clean
HIPCCFLAGS="--amdgpu-target=gfx90a" TARGET=mnmd_st_reg_shft_db_7r_25d_32x16 COMPILER=hipcc make all
./main_mnmd_st_reg_shft_db_7r_25d_32x16_hipcc --grid 1000 --nsteps 1000 --niters 5 --warm-up
```

### MI200 st_reg_shft_db 7r 25d 32 32 0
```
HIPCCFLAGS="--amdgpu-target=gfx90a" TARGET=mnmd_st_reg_shft_db_7r_25d_32x32 COMPILER=hipcc make clean
HIPCCFLAGS="--amdgpu-target=gfx90a" TARGET=mnmd_st_reg_shft_db_7r_25d_32x32 COMPILER=hipcc make all
./main_mnmd_st_reg_shft_db_7r_25d_32x32_hipcc --grid 1000 --nsteps 1000 --niters 5 --warm-up
```

### MI200 st_reg_fixed 7r 25d 16 16 0
```
HIPCCFLAGS="--amdgpu-target=gfx90a" TARGET=mnmd_st_reg_fixed_7r_25d_16x16 COMPILER=hipcc make clean
HIPCCFLAGS="--amdgpu-target=gfx90a" TARGET=mnmd_st_reg_fixed_7r_25d_16x16 COMPILER=hipcc make all
./main_mnmd_st_reg_fixed_7r_25d_16x16_hipcc --grid 1000 --nsteps 1000 --niters 5 --warm-up
```

### MI200 st_reg_fixed 7r 25d 32 16 0
```
HIPCCFLAGS="--amdgpu-target=gfx90a" TARGET=mnmd_st_reg_fixed_7r_25d_32x16 COMPILER=hipcc make clean
HIPCCFLAGS="--amdgpu-target=gfx90a" TARGET=mnmd_st_reg_fixed_7r_25d_32x16 COMPILER=hipcc make all
./main_mnmd_st_reg_fixed_7r_25d_32x16_hipcc --grid 1000 --nsteps 1000 --niters 5 --warm-up
```

### MI200 st_reg_fixed 7r 25d 32 32 0
```
HIPCCFLAGS="--amdgpu-target=gfx90a" TARGET=mnmd_st_reg_fixed_7r_25d_32x32 COMPILER=hipcc make clean
HIPCCFLAGS="--amdgpu-target=gfx90a" TARGET=mnmd_st_reg_fixed_7r_25d_32x32 COMPILER=hipcc make all
./main_mnmd_st_reg_fixed_7r_25d_32x32_hipcc --grid 1000 --nsteps 1000 --niters 5 --warm-up
```

### MI200 st_reg_fixed_db 7r 25d 16 16 0
```
HIPCCFLAGS="--amdgpu-target=gfx90a" TARGET=mnmd_st_reg_fixed_db_7r_25d_16x16 COMPILER=hipcc make clean
HIPCCFLAGS="--amdgpu-target=gfx90a" TARGET=mnmd_st_reg_fixed_db_7r_25d_16x16 COMPILER=hipcc make all
./main_mnmd_st_reg_fixed_db_7r_25d_16x16_hipcc --grid 1000 --nsteps 1000 --niters 5 --warm-up
```

### MI200 st_reg_fixed_db 7r 25d 32 16 0
```
HIPCCFLAGS="--amdgpu-target=gfx90a" TARGET=mnmd_st_reg_fixed_db_7r_25d_32x16 COMPILER=hipcc make clean
HIPCCFLAGS="--amdgpu-target=gfx90a" TARGET=mnmd_st_reg_fixed_db_7r_25d_32x16 COMPILER=hipcc make all
./main_mnmd_st_reg_fixed_db_7r_25d_32x16_hipcc --grid 1000 --nsteps 1000 --niters 5 --warm-up
```

### MI200 st_reg_fixed_db 7r 25d 32 32 0
```
HIPCCFLAGS="--amdgpu-target=gfx90a" TARGET=mnmd_st_reg_fixed_db_7r_25d_32x32 COMPILER=hipcc make clean
HIPCCFLAGS="--amdgpu-target=gfx90a" TARGET=mnmd_st_reg_fixed_db_7r_25d_32x32 COMPILER=hipcc make all
./main_mnmd_st_reg_fixed_db_7r_25d_32x32_hipcc --grid 1000 --nsteps 1000 --niters 5 --warm-up
```

### MI200 st_semi 7r 25d 16 16 0
```
HIPCCFLAGS="--amdgpu-target=gfx90a" TARGET=mnmd_st_semi_7r_25d_16x16 COMPILER=hipcc make clean
HIPCCFLAGS="--amdgpu-target=gfx90a" TARGET=mnmd_st_semi_7r_25d_16x16 COMPILER=hipcc make all
./main_mnmd_st_semi_7r_25d_16x16_hipcc --grid 1000 --nsteps 1000 --niters 5 --warm-up
```

### MI200 st_semi 7r 25d 32 16 0
```
HIPCCFLAGS="--amdgpu-target=gfx90a" TARGET=mnmd_st_semi_7r_25d_32x16 COMPILER=hipcc make clean
HIPCCFLAGS="--amdgpu-target=gfx90a" TARGET=mnmd_st_semi_7r_25d_32x16 COMPILER=hipcc make all
./main_mnmd_st_semi_7r_25d_32x16_hipcc --grid 1000 --nsteps 1000 --niters 5 --warm-up
```

### MI200 st_semi 7r 25d 32 32 0
```
HIPCCFLAGS="--amdgpu-target=gfx90a" TARGET=mnmd_st_semi_7r_25d_32x32 COMPILER=hipcc make clean
HIPCCFLAGS="--amdgpu-target=gfx90a" TARGET=mnmd_st_semi_7r_25d_32x32 COMPILER=hipcc make all
./main_mnmd_st_semi_7r_25d_32x32_hipcc --grid 1000 --nsteps 1000 --niters 5 --warm-up
```

### MI200 st_semi_db 7r 25d 16 16 0
```
HIPCCFLAGS="--amdgpu-target=gfx90a" TARGET=mnmd_st_semi_db_7r_25d_16x16 COMPILER=hipcc make clean
HIPCCFLAGS="--amdgpu-target=gfx90a" TARGET=mnmd_st_semi_db_7r_25d_16x16 COMPILER=hipcc make all
./main_mnmd_st_semi_db_7r_25d_16x16_hipcc --grid 1000 --nsteps 1000 --niters 5 --warm-up
```

### MI200 st_semi_db 7r 25d 32 16 0
```
HIPCCFLAGS="--amdgpu-target=gfx90a" TARGET=mnmd_st_semi_db_7r_25d_32x16 COMPILER=hipcc make clean
HIPCCFLAGS="--amdgpu-target=gfx90a" TARGET=mnmd_st_semi_db_7r_25d_32x16 COMPILER=hipcc make all
./main_mnmd_st_semi_db_7r_25d_32x16_hipcc --grid 1000 --nsteps 1000 --niters 5 --warm-up
```

### MI200 st_semi_db 7r 25d 32 32 0
```
HIPCCFLAGS="--amdgpu-target=gfx90a" TARGET=mnmd_st_semi_db_7r_25d_32x32 COMPILER=hipcc make clean
HIPCCFLAGS="--amdgpu-target=gfx90a" TARGET=mnmd_st_semi_db_7r_25d_32x32 COMPILER=hipcc make all
./main_mnmd_st_semi_db_7r_25d_32x32_hipcc --grid 1000 --nsteps 1000 --niters 5 --warm-up
```

### MI200 st_reg_shft 7r 25d 64 16 0
```
HIPCCFLAGS="--amdgpu-target=gfx90a" TARGET=mnmd_st_reg_shft_7r_25d_64x16 COMPILER=hipcc make clean
HIPCCFLAGS="--amdgpu-target=gfx90a" TARGET=mnmd_st_reg_shft_7r_25d_64x16 COMPILER=hipcc make all
./main_mnmd_st_reg_shft_7r_25d_64x16_hipcc --grid 1000 --nsteps 1000 --niters 5 --warm-up
```

### MI200 st_reg_shft_db 7r 25d 64 16 0
```
HIPCCFLAGS="--amdgpu-target=gfx90a" TARGET=mnmd_st_reg_shft_db_7r_25d_64x16 COMPILER=hipcc make clean
HIPCCFLAGS="--amdgpu-target=gfx90a" TARGET=mnmd_st_reg_shft_db_7r_25d_64x16 COMPILER=hipcc make all
./main_mnmd_st_reg_shft_db_7r_25d_64x16_hipcc --grid 1000 --nsteps 1000 --niters 5 --warm-up
```

### MI200 st_reg_fixed 7r 25d 64 16 0
```
HIPCCFLAGS="--amdgpu-target=gfx90a" TARGET=mnmd_st_reg_fixed_7r_25d_64x16 COMPILER=hipcc make clean
HIPCCFLAGS="--amdgpu-target=gfx90a" TARGET=mnmd_st_reg_fixed_7r_25d_64x16 COMPILER=hipcc make all
./main_mnmd_st_reg_fixed_7r_25d_64x16_hipcc --grid 1000 --nsteps 1000 --niters 5 --warm-up
```

### MI200 st_reg_fixed_db 7r 25d 64 16 0
```
HIPCCFLAGS="--amdgpu-target=gfx90a" TARGET=mnmd_st_reg_fixed_db_7r_25d_64x16 COMPILER=hipcc make clean
HIPCCFLAGS="--amdgpu-target=gfx90a" TARGET=mnmd_st_reg_fixed_db_7r_25d_64x16 COMPILER=hipcc make all
./main_mnmd_st_reg_fixed_db_7r_25d_64x16_hipcc --grid 1000 --nsteps 1000 --niters 5 --warm-up
```

### MI200 st_semi 7r 25d 64 16 0
```
HIPCCFLAGS="--amdgpu-target=gfx90a" TARGET=mnmd_st_semi_7r_25d_64x16 COMPILER=hipcc make clean
HIPCCFLAGS="--amdgpu-target=gfx90a" TARGET=mnmd_st_semi_7r_25d_64x16 COMPILER=hipcc make all
./main_mnmd_st_semi_7r_25d_64x16_hipcc --grid 1000 --nsteps 1000 --niters 5 --warm-up
```

### MI200 st_semi_db 7r 25d 64 16 0
```
HIPCCFLAGS="--amdgpu-target=gfx90a" TARGET=mnmd_st_semi_db_7r_25d_64x16 COMPILER=hipcc make clean
HIPCCFLAGS="--amdgpu-target=gfx90a" TARGET=mnmd_st_semi_db_7r_25d_64x16 COMPILER=hipcc make all
./main_mnmd_st_semi_db_7r_25d_64x16_hipcc --grid 1000 --nsteps 1000 --niters 5 --warm-up
```
