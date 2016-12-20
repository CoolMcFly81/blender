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

/* Templated common implementation part of all CPU kernels.
 *
 * The idea is that particular .cpp files sets needed optimization flags and
 * simply includes this file without worry of copying actual implementation over.
 */

#include "kernel_compat_cpu.h"
#include "kernel_math.h"
#include "kernel_types.h"
#include "kernel_globals.h"
#include "kernel_cpu_image.h"
#include "kernel_film.h"
#include "kernel_path.h"
#include "kernel_path_branched.h"
#include "kernel_bake.h"

#include "filter/filter.h"

#ifdef KERNEL_STUB
#  include "util_debug.h"
#  define STUB_ASSERT(arch, name) assert(!(#name " kernel stub for architecture " #arch " was called!"))
#endif

CCL_NAMESPACE_BEGIN


/* Path Tracing */

void KERNEL_FUNCTION_FULL_NAME(path_trace)(KernelGlobals *kg,
                                           float *buffer,
                                           unsigned int *rng_state,
                                           int sample,
                                           int x, int y,
                                           int offset,
                                           int stride)
{
#ifdef KERNEL_STUB
	STUB_ASSERT(KERNEL_ARCH, path_trace);
#else
#  ifdef __BRANCHED_PATH__
	if(kernel_data.integrator.branched) {
		kernel_branched_path_trace(kg,
		                           buffer,
		                           rng_state,
		                           sample,
		                           x, y,
		                           offset,
		                           stride);
	}
	else
#  endif
	{
		kernel_path_trace(kg, buffer, rng_state, sample, x, y, offset, stride);
	}
#endif /* KERNEL_STUB */
}

/* Film */

void KERNEL_FUNCTION_FULL_NAME(convert_to_byte)(KernelGlobals *kg,
                                                uchar4 *rgba,
                                                float *buffer,
                                                float sample_scale,
                                                int x, int y,
                                                int offset,
                                                int stride)
{
#ifdef KERNEL_STUB
	STUB_ASSERT(KERNEL_ARCH, convert_to_byte);
#else
	kernel_film_convert_to_byte(kg,
	                            rgba,
	                            buffer,
	                            sample_scale,
	                            x, y,
	                            offset,
	                            stride);
#endif /* KERNEL_STUB */
}

void KERNEL_FUNCTION_FULL_NAME(convert_to_half_float)(KernelGlobals *kg,
                                                      uchar4 *rgba,
                                                      float *buffer,
                                                      float sample_scale,
                                                      int x, int y,
                                                      int offset,
                                                      int stride)
{
#ifdef KERNEL_STUB
	STUB_ASSERT(KERNEL_ARCH, convert_to_half_float);
#else
	kernel_film_convert_to_half_float(kg,
	                                  rgba,
	                                  buffer,
	                                  sample_scale,
	                                  x, y,
	                                  offset,
	                                  stride);
#endif /* KERNEL_STUB */
}

/* Shader Evaluate */

void KERNEL_FUNCTION_FULL_NAME(shader)(KernelGlobals *kg,
                                       uint4 *input,
                                       float4 *output,
                                       float *output_luma,
                                       int type,
                                       int filter,
                                       int i,
                                       int offset,
                                       int sample)
{
#ifdef KERNEL_STUB
	STUB_ASSERT(KERNEL_ARCH, shader);
#else
	if(type >= SHADER_EVAL_BAKE) {
		kernel_assert(output_luma == NULL);
#  ifdef __BAKING__
		kernel_bake_evaluate(kg,
		                     input,
		                     output,
		                     (ShaderEvalType)type,
		                     filter,
		                     i,
		                     offset,
		                     sample);
#  endif
	}
	else {
		kernel_shader_evaluate(kg,
		                       input,
		                       output,
		                       output_luma,
		                       (ShaderEvalType)type,
		                       i,
		                       sample);
	}
#endif /* KERNEL_STUB */
}

/* Denoise filter */

void KERNEL_FUNCTION_FULL_NAME(filter_divide_shadow)(KernelGlobals *kg,
                                                     int sample,
                                                     float** buffers,
                                                     int x,
                                                     int y,
                                                     int *tile_x,
                                                     int *tile_y,
                                                     int *offset,
                                                     int *stride,
                                                     float *unfiltered, float *sampleVariance, float *sampleVarianceV, float *bufferVariance,
                                                     int* prefilter_rect)
{
#ifdef KERNEL_STUB
	STUB_ASSERT(KERNEL_ARCH, filter_divide_shadow);
#else
	int4 rect = make_int4(prefilter_rect[0], prefilter_rect[1], prefilter_rect[2], prefilter_rect[3]);
	kernel_filter_divide_shadow(kg, sample, buffers, x, y, tile_x, tile_y, offset, stride, unfiltered, sampleVariance, sampleVarianceV, bufferVariance, rect);
#endif
}

void KERNEL_FUNCTION_FULL_NAME(filter_get_feature)(KernelGlobals *kg,
                                                   int sample,
                                                   float** buffers,
                                                   int m_offset,
                                                   int v_offset,
                                                   int x,
                                                   int y,
                                                   int *tile_x,
                                                   int *tile_y,
                                                   int *offset,
                                                   int *stride,
                                                   float *mean, float *variance,
                                                   int* prefilter_rect)
{
#ifdef KERNEL_STUB
	STUB_ASSERT(KERNEL_ARCH, filter_get_feature);
#else
	int4 rect = make_int4(prefilter_rect[0], prefilter_rect[1], prefilter_rect[2], prefilter_rect[3]);
	kernel_filter_get_feature(kg, sample, buffers, m_offset, v_offset, x, y, tile_x, tile_y, offset, stride, mean, variance, rect);
#endif
}

void KERNEL_FUNCTION_FULL_NAME(filter_non_local_means)(int x, int y,
                                                       float *noisyImage,
                                                       float *weightImage,
                                                       float *variance,
                                                       float *filteredImage,
                                                       int* filter_rect,
                                                       int r, int f,
                                                       float a, float k_2)
{
#ifdef KERNEL_STUB
	STUB_ASSERT(KERNEL_ARCH, filter_non_local_means);
#else
	int4 rect = make_int4(filter_rect[0], filter_rect[1], filter_rect[2], filter_rect[3]);
	kernel_filter_non_local_means(x, y, noisyImage, weightImage, variance, filteredImage, rect, r, f, a, k_2);
#endif
}

void KERNEL_FUNCTION_FULL_NAME(filter_non_local_means_3)(int x, int y,
                                                         float *noisyImage[3],
                                                         float *weightImage[3],
                                                         float *variance[3],
                                                         float *filteredImage[3],
                                                         int* filter_rect,
                                                         int r, int f,
                                                         float a, float k_2)
{
#ifdef KERNEL_STUB
	STUB_ASSERT(KERNEL_ARCH, filter_non_local_means_3);
#else
	int4 rect = make_int4(filter_rect[0], filter_rect[1], filter_rect[2], filter_rect[3]);
	kernel_filter_non_local_means_3(x, y,
	                                (float ccl_readonly_ptr*) noisyImage,
	                                (float ccl_readonly_ptr*) weightImage,
	                                (float ccl_readonly_ptr*) variance,
	                                filteredImage, rect, r, f, a, k_2);
#endif
}

void KERNEL_FUNCTION_FULL_NAME(filter_combine_halves)(int x, int y,
                                                      float *mean,
                                                      float *variance,
                                                      float *a,
                                                      float *b,
                                                      int* prefilter_rect,
                                                      int r)
{
#ifdef KERNEL_STUB
	STUB_ASSERT(KERNEL_ARCH, filter_combine_halves);
#else
	int4 rect = make_int4(prefilter_rect[0], prefilter_rect[1], prefilter_rect[2], prefilter_rect[3]);
	kernel_filter_combine_halves(x, y, mean, variance, a, b, rect, r);
#endif
}

void KERNEL_FUNCTION_FULL_NAME(filter_construct_transform)(KernelGlobals *kg,
                                                           int sample,
                                                           float* buffer,
                                                           int x,
                                                           int y,
                                                           void *storage,
                                                           int* prefilter_rect)
{
#ifdef KERNEL_STUB
	STUB_ASSERT(KERNEL_ARCH, filter_construct_transform);
#else
	int4 rect = make_int4(prefilter_rect[0], prefilter_rect[1], prefilter_rect[2], prefilter_rect[3]);
	kernel_filter_construct_transform(kg, sample, buffer, x, y, (FilterStorage*) storage, rect);
#endif
}

void KERNEL_FUNCTION_FULL_NAME(filter_estimate_wlr_params)(KernelGlobals *kg,
                                                           int sample,
                                                           float* buffer,
                                                           int x,
                                                           int y,
                                                           void *storage,
                                                           int* prefilter_rect)
{
#ifdef KERNEL_STUB
	STUB_ASSERT(KERNEL_ARCH, filter_estimate_wlr_params);
#else
	int4 rect = make_int4(prefilter_rect[0], prefilter_rect[1], prefilter_rect[2], prefilter_rect[3]);
	kernel_filter_estimate_wlr_params(kg, sample, buffer, x, y, (FilterStorage*) storage, rect);
#endif
}

void KERNEL_FUNCTION_FULL_NAME(filter_final_pass_wlr)(KernelGlobals *kg,
                                                      int sample,
                                                      float* buffer,
                                                      int x,
                                                      int y,
                                                      int offset,
                                                      int stride,
                                                      float *buffers,
                                                      void *storage_ptr,
                                                      float *weight_cache,
                                                      int* filter_area,
                                                      int* prefilter_rect)
{
#ifdef KERNEL_STUB
	STUB_ASSERT(KERNEL_ARCH, filter_final_pass_wlr);
#else
	int4 rect = make_int4(prefilter_rect[0], prefilter_rect[1], prefilter_rect[2], prefilter_rect[3]);
	int4 area = make_int4(filter_area[0], filter_area[1], filter_area[2], filter_area[3]);
	FilterStorage *storage = (FilterStorage*) storage_ptr;
	kernel_filter_final_pass_wlr(kg, sample, buffer, x, y, offset, stride, buffers, 0, make_int2(0, 0), storage, weight_cache, storage->transform, 1, area, rect);
#endif
}

void KERNEL_FUNCTION_FULL_NAME(filter_final_pass_nlm)(KernelGlobals *kg,
                                                      int sample,
                                                      float* buffer,
                                                      int x,
                                                      int y,
                                                      int offset,
                                                      int stride,
                                                      float *buffers,
                                                      void *storage_ptr,
                                                      float *weight_cache,
                                                      int* filter_area,
                                                      int* prefilter_rect)
{
#ifdef KERNEL_STUB
	STUB_ASSERT(KERNEL_ARCH, filter_final_pass_nlm);
#else
	int4 rect = make_int4(prefilter_rect[0], prefilter_rect[1], prefilter_rect[2], prefilter_rect[3]);
	int4 area = make_int4(filter_area[0], filter_area[1], filter_area[2], filter_area[3]);
	FilterStorage *storage = (FilterStorage*) storage_ptr;
	if(kernel_data.film.denoise_cross) {
		kernel_filter_final_pass_nlm(kg, sample, buffer, x, y, offset, stride, buffers, 0, make_int2(0, 6), storage, weight_cache, storage->transform, 1, area, rect);
		kernel_filter_final_pass_nlm(kg, sample, buffer, x, y, offset, stride, buffers, 0, make_int2(6, 0), storage, weight_cache, storage->transform, 1, area, rect);
	}
	else {
		kernel_filter_final_pass_nlm(kg, sample, buffer, x, y, offset, stride, buffers, 0, make_int2(0, 0), storage, weight_cache, storage->transform, 1, area, rect);
	}
#endif
}

void KERNEL_FUNCTION_FULL_NAME(filter_divide_combined)(KernelGlobals *kg,
                                                       int x, int y,
                                                       int sample,
                                                       float *buffers,
                                                       int offset,
                                                       int stride)
{
#ifdef KERNEL_STUB
	STUB_ASSERT(KERNEL_ARCH, filter_divide_combined);
#else
	kernel_filter_divide_combined(kg, x, y, sample, buffers, offset, stride);
#endif
}

#undef KERNEL_STUB
#undef STUB_ASSERT
#undef KERNEL_ARCH

CCL_NAMESPACE_END
