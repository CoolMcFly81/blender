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

#ifndef __UTIL_SYSTEM_H__
#define __UTIL_SYSTEM_H__

#include "util_string.h"

CCL_NAMESPACE_BEGIN

/* Get number of available CPU groups. */
int system_cpu_group_count();

/* Get number of threads/processors in the specified group. */
int system_cpu_group_thread_count(int group);

/* Get total number of threads in all groups. */
int system_cpu_thread_count();

/* Get current process groups. */
unsigned short system_cpu_process_groups(unsigned short max_groups,
                                         unsigned short *grpups);

string system_cpu_brand_string();
int system_cpu_bits();
bool system_cpu_support_sse2();
bool system_cpu_support_sse3();
bool system_cpu_support_sse41();
bool system_cpu_support_avx();
bool system_cpu_support_avx2();

void system_enable_ftz();

#ifdef WITH_CYCLES_DEBUG_FPE
typedef enum FPEState
{
	FPE_ENABLED,
	FPE_DISABLED,
} FPEState;

class scoped_fpe
{
public:
	explicit scoped_fpe(FPEState state);
	void restore();
	~scoped_fpe();
private:
	bool was_enabled;
};
#endif

CCL_NAMESPACE_END

#endif /* __UTIL_SYSTEM_H__ */

