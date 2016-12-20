/*
 * Copyright 2011-2013 Blender Foundation
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

/* CUDA kernel entry points */

/* device data taken from CUDA occupancy calculator */

#ifdef __CUDA_ARCH__

/* 2.0 and 2.1 */
#if __CUDA_ARCH__ == 200 || __CUDA_ARCH__ == 210
#  define CUDA_MULTIPRESSOR_MAX_REGISTERS 32768
#  define CUDA_MULTIPROCESSOR_MAX_BLOCKS 8
#  define CUDA_BLOCK_MAX_THREADS 1024
#  define CUDA_THREAD_MAX_REGISTERS 63

/* tunable parameters */
#  define CUDA_THREADS_BLOCK_WIDTH 16
#  define CUDA_KERNEL_MAX_REGISTERS 32
#  define CUDA_KERNEL_BRANCHED_MAX_REGISTERS 40

/* 3.0 and 3.5 */
#elif __CUDA_ARCH__ == 300 || __CUDA_ARCH__ == 350
#  define CUDA_MULTIPRESSOR_MAX_REGISTERS 65536
#  define CUDA_MULTIPROCESSOR_MAX_BLOCKS 16
#  define CUDA_BLOCK_MAX_THREADS 1024
#  define CUDA_THREAD_MAX_REGISTERS 63

/* tunable parameters */
#  define CUDA_THREADS_BLOCK_WIDTH 16
#  define CUDA_KERNEL_MAX_REGISTERS 63
#  define CUDA_KERNEL_BRANCHED_MAX_REGISTERS 63

/* 3.2 */
#elif __CUDA_ARCH__ == 320
#  define CUDA_MULTIPRESSOR_MAX_REGISTERS 32768
#  define CUDA_MULTIPROCESSOR_MAX_BLOCKS 16
#  define CUDA_BLOCK_MAX_THREADS 1024
#  define CUDA_THREAD_MAX_REGISTERS 63

/* tunable parameters */
#  define CUDA_THREADS_BLOCK_WIDTH 16
#  define CUDA_KERNEL_MAX_REGISTERS 63
#  define CUDA_KERNEL_BRANCHED_MAX_REGISTERS 63

/* 3.7 */
#elif __CUDA_ARCH__ == 370
#  define CUDA_MULTIPRESSOR_MAX_REGISTERS 65536
#  define CUDA_MULTIPROCESSOR_MAX_BLOCKS 16
#  define CUDA_BLOCK_MAX_THREADS 1024
#  define CUDA_THREAD_MAX_REGISTERS 255

/* tunable parameters */
#  define CUDA_THREADS_BLOCK_WIDTH 16
#  define CUDA_KERNEL_MAX_REGISTERS 63
#  define CUDA_KERNEL_BRANCHED_MAX_REGISTERS 63

/* 5.0, 5.2, 5.3, 6.0, 6.1 */
#elif __CUDA_ARCH__ >= 500
#  define CUDA_MULTIPRESSOR_MAX_REGISTERS 65536
#  define CUDA_MULTIPROCESSOR_MAX_BLOCKS 32
#  define CUDA_BLOCK_MAX_THREADS 1024
#  define CUDA_THREAD_MAX_REGISTERS 255

/* tunable parameters */
#  define CUDA_THREADS_BLOCK_WIDTH 16
#  define CUDA_KERNEL_MAX_REGISTERS 48
#  define CUDA_KERNEL_BRANCHED_MAX_REGISTERS 63

/* unknown architecture */
#else
#  error "Unknown or unsupported CUDA architecture, can't determine launch bounds"
#endif

#include "../../kernel_compat_cuda.h"
#include "../../kernel_math.h"
#include "../../kernel_types.h"
#include "../../kernel_globals.h"
#include "../../kernel_film.h"
#include "../../kernel_path.h"
#include "../../kernel_path_branched.h"
#include "../../kernel_bake.h"

#include "../../filter/filter.h"

/* compute number of threads per block and minimum blocks per multiprocessor
 * given the maximum number of registers per thread */

#define CUDA_LAUNCH_BOUNDS(threads_block_width, thread_num_registers) \
	__launch_bounds__( \
		threads_block_width*threads_block_width, \
		CUDA_MULTIPRESSOR_MAX_REGISTERS/(threads_block_width*threads_block_width*thread_num_registers) \
		)

/* sanity checks */

#if CUDA_THREADS_BLOCK_WIDTH*CUDA_THREADS_BLOCK_WIDTH > CUDA_BLOCK_MAX_THREADS
#  error "Maximum number of threads per block exceeded"
#endif

#if CUDA_MULTIPRESSOR_MAX_REGISTERS/(CUDA_THREADS_BLOCK_WIDTH*CUDA_THREADS_BLOCK_WIDTH*CUDA_KERNEL_MAX_REGISTERS) > CUDA_MULTIPROCESSOR_MAX_BLOCKS
#  error "Maximum number of blocks per multiprocessor exceeded"
#endif

#if CUDA_KERNEL_MAX_REGISTERS > CUDA_THREAD_MAX_REGISTERS
#  error "Maximum number of registers per thread exceeded"
#endif

#if CUDA_KERNEL_BRANCHED_MAX_REGISTERS > CUDA_THREAD_MAX_REGISTERS
#  error "Maximum number of registers per thread exceeded"
#endif

/* kernels */

extern "C" __global__ void
CUDA_LAUNCH_BOUNDS(CUDA_THREADS_BLOCK_WIDTH, CUDA_KERNEL_MAX_REGISTERS)
kernel_cuda_path_trace(float *buffer, uint *rng_state, int sample, int sx, int sy, int sw, int sh, int offset, int stride)
{
	int x = sx + blockDim.x*blockIdx.x + threadIdx.x;
	int y = sy + blockDim.y*blockIdx.y + threadIdx.y;

	if(x < sx + sw && y < sy + sh)
		kernel_path_trace(NULL, buffer, rng_state, sample, x, y, offset, stride);
}

#ifdef __BRANCHED_PATH__
extern "C" __global__ void
CUDA_LAUNCH_BOUNDS(CUDA_THREADS_BLOCK_WIDTH, CUDA_KERNEL_BRANCHED_MAX_REGISTERS)
kernel_cuda_branched_path_trace(float *buffer, uint *rng_state, int sample, int sx, int sy, int sw, int sh, int offset, int stride)
{
	int x = sx + blockDim.x*blockIdx.x + threadIdx.x;
	int y = sy + blockDim.y*blockIdx.y + threadIdx.y;

	if(x < sx + sw && y < sy + sh)
		kernel_branched_path_trace(NULL, buffer, rng_state, sample, x, y, offset, stride);
}
#endif

extern "C" __global__ void
CUDA_LAUNCH_BOUNDS(CUDA_THREADS_BLOCK_WIDTH, CUDA_KERNEL_MAX_REGISTERS)
kernel_cuda_convert_to_byte(uchar4 *rgba, float *buffer, float sample_scale, int sx, int sy, int sw, int sh, int offset, int stride)
{
	int x = sx + blockDim.x*blockIdx.x + threadIdx.x;
	int y = sy + blockDim.y*blockIdx.y + threadIdx.y;

	if(x < sx + sw && y < sy + sh)
		kernel_film_convert_to_byte(NULL, rgba, buffer, sample_scale, x, y, offset, stride);
}

extern "C" __global__ void
CUDA_LAUNCH_BOUNDS(CUDA_THREADS_BLOCK_WIDTH, CUDA_KERNEL_MAX_REGISTERS)
kernel_cuda_convert_to_half_float(uchar4 *rgba, float *buffer, float sample_scale, int sx, int sy, int sw, int sh, int offset, int stride)
{
	int x = sx + blockDim.x*blockIdx.x + threadIdx.x;
	int y = sy + blockDim.y*blockIdx.y + threadIdx.y;

	if(x < sx + sw && y < sy + sh)
		kernel_film_convert_to_half_float(NULL, rgba, buffer, sample_scale, x, y, offset, stride);
}

extern "C" __global__ void
CUDA_LAUNCH_BOUNDS(CUDA_THREADS_BLOCK_WIDTH, CUDA_KERNEL_MAX_REGISTERS)
kernel_cuda_shader(uint4 *input,
                   float4 *output,
                   float *output_luma,
                   int type,
                   int sx,
                   int sw,
                   int offset,
                   int sample)
{
	int x = sx + blockDim.x*blockIdx.x + threadIdx.x;

	if(x < sx + sw) {
		kernel_shader_evaluate(NULL,
		                       input,
		                       output,
		                       output_luma,
		                       (ShaderEvalType)type, 
		                       x,
		                       sample);
	}
}

#ifdef __BAKING__
extern "C" __global__ void
CUDA_LAUNCH_BOUNDS(CUDA_THREADS_BLOCK_WIDTH, CUDA_KERNEL_MAX_REGISTERS)
kernel_cuda_bake(uint4 *input, float4 *output, int type, int filter, int sx, int sw, int offset, int sample)
{
	int x = sx + blockDim.x*blockIdx.x + threadIdx.x;

	if(x < sx + sw)
		kernel_bake_evaluate(NULL, input, output, (ShaderEvalType)type, filter, x, offset, sample);
}
#endif

extern "C" __global__ void
CUDA_LAUNCH_BOUNDS(CUDA_THREADS_BLOCK_WIDTH, CUDA_KERNEL_MAX_REGISTERS)
kernel_cuda_filter_divide_shadow(int sample, float* buffers, int4 buffer_rect, int offset, int stride, float *unfiltered, float *sampleVariance, float *sampleVarianceV, float *bufferVariance, int4 prefilter_rect)
{
	int x = prefilter_rect.x + blockDim.x*blockIdx.x + threadIdx.x;
	int y = prefilter_rect.y + blockDim.y*blockIdx.y + threadIdx.y;
	if(x < prefilter_rect.z && y < prefilter_rect.w) {
		int tile_x[4] = {buffer_rect.x, buffer_rect.x, buffer_rect.x+buffer_rect.z, buffer_rect.x+buffer_rect.z};
		int tile_y[4] = {buffer_rect.y, buffer_rect.y, buffer_rect.y+buffer_rect.w, buffer_rect.y+buffer_rect.w};
		float *tile_buffers[9] = {NULL, NULL, NULL, NULL, buffers, NULL, NULL, NULL, NULL};
		int tile_offset[9] = {0, 0, 0, 0, offset, 0, 0, 0, 0};
		int tile_stride[9] = {0, 0, 0, 0, stride, 0, 0, 0, 0};
		kernel_filter_divide_shadow(NULL, sample, tile_buffers, x, y, tile_x, tile_y, tile_offset, tile_stride, unfiltered, sampleVariance, sampleVarianceV, bufferVariance, prefilter_rect);
	}
}

extern "C" __global__ void
CUDA_LAUNCH_BOUNDS(CUDA_THREADS_BLOCK_WIDTH, CUDA_KERNEL_MAX_REGISTERS)
kernel_cuda_filter_get_feature(int sample, float* buffers, int m_offset, int v_offset, int4 buffer_rect, int offset, int stride, float *mean, float *variance, int4 prefilter_rect)
{
	int x = prefilter_rect.x + blockDim.x*blockIdx.x + threadIdx.x;
	int y = prefilter_rect.y + blockDim.y*blockIdx.y + threadIdx.y;
	if(x < prefilter_rect.z && y < prefilter_rect.w) {
		int tile_x[4] = {buffer_rect.x, buffer_rect.x, buffer_rect.x+buffer_rect.z, buffer_rect.x+buffer_rect.z};
		int tile_y[4] = {buffer_rect.y, buffer_rect.y, buffer_rect.y+buffer_rect.w, buffer_rect.y+buffer_rect.w};
		float *tile_buffers[9] = {NULL, NULL, NULL, NULL, buffers, NULL, NULL, NULL, NULL};
		int tile_offset[9] = {0, 0, 0, 0, offset, 0, 0, 0, 0};
		int tile_stride[9] = {0, 0, 0, 0, stride, 0, 0, 0, 0};
		kernel_filter_get_feature(NULL, sample, tile_buffers, m_offset, v_offset, x, y, tile_x, tile_y, tile_offset, tile_stride, mean, variance, prefilter_rect);
	}
}

extern "C" __global__ void
CUDA_LAUNCH_BOUNDS(CUDA_THREADS_BLOCK_WIDTH, CUDA_KERNEL_MAX_REGISTERS)
kernel_cuda_filter_non_local_means(float *noisyImage, float *weightImage, float *variance, float *filteredImage, int4 prefilter_rect, int r, int f, float a, float k_2)
{
	int x = prefilter_rect.x + blockDim.x*blockIdx.x + threadIdx.x;
	int y = prefilter_rect.y + blockDim.y*blockIdx.y + threadIdx.y;
	if(x < prefilter_rect.z && y < prefilter_rect.w) {
		kernel_filter_non_local_means(x, y, noisyImage, weightImage, variance, filteredImage, prefilter_rect, r, f, a, k_2);
	}
}

extern "C" __global__ void
CUDA_LAUNCH_BOUNDS(CUDA_THREADS_BLOCK_WIDTH, CUDA_KERNEL_MAX_REGISTERS)
kernel_cuda_filter_combine_halves(float *mean, float *variance, float *a, float *b, int4 prefilter_rect, int r)
{
	int x = prefilter_rect.x + blockDim.x*blockIdx.x + threadIdx.x;
	int y = prefilter_rect.y + blockDim.y*blockIdx.y + threadIdx.y;
	if(x < prefilter_rect.z && y < prefilter_rect.w) {
		kernel_filter_combine_halves(x, y, mean, variance, a, b, prefilter_rect, r);
	}
}

extern "C" __global__ void
CUDA_LAUNCH_BOUNDS(CUDA_THREADS_BLOCK_WIDTH, CUDA_KERNEL_MAX_REGISTERS)
kernel_cuda_filter_construct_transform(int sample, float const* __restrict__ buffer, float *transform, void *storage, int4 filter_area, int4 rect)
{
	int x = blockDim.x*blockIdx.x + threadIdx.x;
	int y = blockDim.y*blockIdx.y + threadIdx.y;
	if(x < filter_area.z && y < filter_area.w) {
		CUDAFilterStorage *l_storage = ((CUDAFilterStorage*) storage) + y*filter_area.z + x;
		float *l_transform = transform + y*filter_area.z + x;
		kernel_filter_construct_transform(NULL, sample, buffer, x + filter_area.x, y + filter_area.y, l_transform, l_storage, rect, filter_area.z*filter_area.w, threadIdx.y*blockDim.x + threadIdx.x);
	}
}

extern "C" __global__ void
CUDA_LAUNCH_BOUNDS(CUDA_THREADS_BLOCK_WIDTH, CUDA_KERNEL_MAX_REGISTERS)
kernel_cuda_filter_estimate_bandwidths(int sample, float const* __restrict__ buffer, float const* __restrict__ transform, void *storage, int4 filter_area, int4 rect)
{
	int x = blockDim.x*blockIdx.x + threadIdx.x;
	int y = blockDim.y*blockIdx.y + threadIdx.y;
	if(x < filter_area.z && y < filter_area.w) {
		CUDAFilterStorage *l_storage = ((CUDAFilterStorage*) storage) + y*filter_area.z + x;
		float const* __restrict__ l_transform = transform + y*filter_area.z + x;
		kernel_filter_estimate_bandwidths(NULL, sample, buffer, x + filter_area.x, y + filter_area.y, l_transform, l_storage, rect, filter_area.z*filter_area.w, threadIdx.y*blockDim.x + threadIdx.x);
	}
}

extern "C" __global__ void
CUDA_LAUNCH_BOUNDS(CUDA_THREADS_BLOCK_WIDTH, CUDA_KERNEL_MAX_REGISTERS)
kernel_cuda_filter_estimate_bias_variance(int sample, float const* __restrict__ buffer, float const* __restrict__ transform, void *storage, int4 filter_area, int4 rect, int candidate)
{
	int x = blockDim.x*blockIdx.x + threadIdx.x;
	int y = blockDim.y*blockIdx.y + threadIdx.y;
	if(x < filter_area.z && y < filter_area.w) {
		CUDAFilterStorage *l_storage = ((CUDAFilterStorage*) storage) + y*filter_area.z + x;
		float const* __restrict__ l_transform = transform + y*filter_area.z + x;
		kernel_filter_estimate_bias_variance(NULL, sample, buffer, x + filter_area.x, y + filter_area.y, l_transform, l_storage, rect, candidate, filter_area.z*filter_area.w, threadIdx.y*blockDim.x + threadIdx.x);
	}
}

extern "C" __global__ void
CUDA_LAUNCH_BOUNDS(CUDA_THREADS_BLOCK_WIDTH, CUDA_KERNEL_MAX_REGISTERS)
kernel_cuda_filter_calculate_bandwidth(int sample, void *storage, int4 filter_area)
{
	int x = blockDim.x*blockIdx.x + threadIdx.x;
	int y = blockDim.y*blockIdx.y + threadIdx.y;
	if(x < filter_area.z && y < filter_area.w) {
		CUDAFilterStorage *l_storage = ((CUDAFilterStorage*) storage) + y*filter_area.z + x;
		kernel_filter_calculate_bandwidth(NULL, sample, l_storage);
	}
}

extern "C" __global__ void
CUDA_LAUNCH_BOUNDS(CUDA_THREADS_BLOCK_WIDTH, CUDA_KERNEL_MAX_REGISTERS)
kernel_cuda_filter_final_pass_wlr(int sample, float* buffer, int offset, int stride, float const* __restrict__ transform, void *storage, float *buffers, int4 filter_area, int4 rect)
{
	int x = blockDim.x*blockIdx.x + threadIdx.x;
	int y = blockDim.y*blockIdx.y + threadIdx.y;
	if(x < filter_area.z && y < filter_area.w) {
		CUDAFilterStorage *l_storage = ((CUDAFilterStorage*) storage) + y*filter_area.z + x;
		float const* __restrict__ l_transform = transform + y*filter_area.z + x;
		float weight_cache[CUDA_WEIGHT_CACHE_SIZE];
		kernel_filter_final_pass_wlr(NULL, sample, buffer, x + filter_area.x, y + filter_area.y, offset, stride, buffers, 0, make_int2(0, 0), l_storage, weight_cache, l_transform, filter_area.z*filter_area.w, filter_area, rect);
	}
}

extern "C" __global__ void
CUDA_LAUNCH_BOUNDS(CUDA_THREADS_BLOCK_WIDTH, CUDA_KERNEL_MAX_REGISTERS)
kernel_cuda_filter_final_pass_nlm(int sample, float* buffer, int offset, int stride, float const* __restrict__ transform, void *storage, float *buffers, int4 filter_area, int4 rect)
{
	int x = blockDim.x*blockIdx.x + threadIdx.x;
	int y = blockDim.y*blockIdx.y + threadIdx.y;
	if(x < filter_area.z && y < filter_area.w) {
		CUDAFilterStorage *l_storage = ((CUDAFilterStorage*) storage) + y*filter_area.z + x;
		float const* __restrict__ l_transform = transform + y*filter_area.z + x;
		float weight_cache[CUDA_WEIGHT_CACHE_SIZE];
		if(kernel_data.film.denoise_cross) {
			kernel_filter_final_pass_nlm(NULL, sample, buffer, x + filter_area.x, y + filter_area.y, offset, stride, buffers, 0, make_int2(0, 6), l_storage, weight_cache, l_transform, filter_area.z*filter_area.w, filter_area, rect);
			kernel_filter_final_pass_nlm(NULL, sample, buffer, x + filter_area.x, y + filter_area.y, offset, stride, buffers, 0, make_int2(6, 0), l_storage, weight_cache, l_transform, filter_area.z*filter_area.w, filter_area, rect);
		}
		else {
			kernel_filter_final_pass_nlm(NULL, sample, buffer, x + filter_area.x, y + filter_area.y, offset, stride, buffers, 0, make_int2(0, 0), l_storage, weight_cache, l_transform, filter_area.z*filter_area.w, filter_area, rect);
		}
	}
}

extern "C" __global__ void
CUDA_LAUNCH_BOUNDS(CUDA_THREADS_BLOCK_WIDTH, CUDA_KERNEL_MAX_REGISTERS)
kernel_cuda_filter_divide_combined(float *buffers, int sample, int offset, int stride, int4 filter_area)
{
	int x = blockDim.x*blockIdx.x + threadIdx.x;
	int y = blockDim.y*blockIdx.y + threadIdx.y;
	if(x < filter_area.z && y < filter_area.w) {
		kernel_filter_divide_combined(NULL, x + filter_area.x, y + filter_area.y, sample, buffers, offset, stride);
	}
}

#endif

