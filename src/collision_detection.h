/*******************************************************
 * Copyright (C) 2012-2013 Hammad Mazhar <hammad@hamelot.co.uk>, Simulation Based Engineering Lab <sbel.wisc.edu>
 * Some rights reserved. See LICENSE, AUTHORS.
 * This file is part of Chrono-Collide.
 *******************************************************/
#ifndef SSCD_H
#define SSCD_H

#undef _GLIBCXX_ATOMIC_BUILTINS
#undef _GLIBCXX_USE_INT128

#pragma once
//====================================INCLUDES=================================//
#include <cuda.h>
#include <vector>
#include <numeric>
#include <algorithm>
#include <iostream>
#include <string>
#include <sstream>
#include <fstream>
#include <omp.h>
#include <glm/glm.hpp>
#include <glm/gtc/quaternion.hpp>
#include <glm/gtx/quaternion.hpp>
// thrust and cuda includes
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/generate.h>
#include <thrust/sort.h>
#include <thrust/reduce.h>
#include <thrust/inner_product.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/remove.h>
#include <thrust/binary_search.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/random.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/set_operations.h>
#include <thrust/functional.h>
#include <thrust/unique.h>
#include <thrust/scan.h>
#include <thrust/sequence.h>
#include <thrust/count.h>
//#include <vector_types.h>
#include <thrust/execution_policy.h>
#include <thrust/system/omp/execution_policy.h>
#include <thrust/system/cuda/execution_policy.h>
#include <thrust/system/cpp/execution_policy.h>
using namespace std;
using namespace glm;
//====================================DEFINES=================================//
// takes care of some GCC issues
#undef _GLIBCXX_ATOMIC_BUILTINS
#undef _GLIBCXX_USE_INT128

#define MIN_ZERO_EPSILON 1.1754943508222875E-38
#define EPS FLT_EPSILON
#define kCollideEpsilon  1e-5f

enum shape_type {
	SPHERE, ELLIPSOID, BOX, CYLINDER, RECT, CONE, TRIANGLEMESH
};
#define PRINT_DEBUG_GPU
//===================================USE CUDA?=================================//
//#define SIM_ENABLE_GPU_MODE
#ifdef SIM_ENABLE_GPU_MODE
//#define THRUST_DEVICE_SYSTEM THRUST_DEVICE_SYSTEM_CUDA
#define custom_vector thrust::device_vector
#define EXEC_POLICY thrust::device
#else
//#define THRUST_HOST_SYSTEM THRUST_HOST_SYSTEM_OMP
#define custom_vector thrust::host_vector
#define EXEC_POLICY thrust::omp::par
#endif
//====================================UNITTEST===============================//
#define UNIT_TEST

//===================================ECLIPSE=================================//
#ifdef __CDT_PARSER__
#define __host__
#define __device__
#define __global__
#define __constant__
#define __shared__
#define __KERNEL__(...) ()
#else
#define __KERNEL__(...)  <<< __VA_ARGS__ >>>
#endif

//===================================REAL=================================//
typedef float real;
typedef glm::vec2 real2;
typedef glm::vec3 real3;
typedef glm::vec4 real4;

#define R3  real3
#define R4  real4
#define R2  real2
#define I4  ivec4
#define I3  ivec3
#define I2  ivec2
#define U3  uvec3
#define ZERO_EPSILON 1e-8
typedef unsigned int uint;

#define ZERO_VECTOR R3(0)

static __host__ __device__ bool IsZero(const real &a) {
	return abs(a) < ZERO_EPSILON;
}

static __host__ __device__ bool IsZero(const real3 &a) {
	return IsZero(a.x) && IsZero(a.y) && IsZero(a.z);
}

static __host__ __device__ bool isEqual(const real &_a, const real &_b) {
	real ab;
	ab = fabs(_a - _b);
	if (fabs(ab) < ZERO_EPSILON)
		return 1;
	real a, b;
	a = fabs(_a);
	b = fabs(_b);
	if (b > a) {
		return ab < ZERO_EPSILON * b;
	} else {
		return ab < ZERO_EPSILON * a;
	}
}

static __host__ __device__ bool isEqual(const real3 &a, const real3 &b) {
	return isEqual(a.x, b.x) && isEqual(a.y, b.y) && isEqual(a.z, b.z);
}

template<class T>
static __host__ __device__ inline void Swap(T &a, T &b) {
	T tmp = a;
	a = b;
	b = tmp;
}

//===============================RAW POINTER CASTS============================//
#define CASTC1(x) (char*)thrust::raw_pointer_cast(&x[0])
#define CASTU1(x) (uint*)thrust::raw_pointer_cast(&x[0])
#define CASTU2(x) (uvec2*)thrust::raw_pointer_cast(&x[0])
#define CASTU3(x) (uvec3*)thrust::raw_pointer_cast(&x[0])
#define CASTI1(x) (int*)thrust::raw_pointer_cast(&x[0])
#define CASTI2(x) (ivec2*)thrust::raw_pointer_cast(&x[0])
#define CASTI3(x) (ivec3*)thrust::raw_pointer_cast(&x[0])
#define CASTI4(x) (ivec4*)thrust::raw_pointer_cast(&x[0])
#define CASTR1(x) (real*)thrust::raw_pointer_cast(&x[0])
#define CASTR2(x) (real2*)thrust::raw_pointer_cast(&x[0])
#define CASTR3(x) (real3*)thrust::raw_pointer_cast(&x[0])
#define CASTR4(x) (real4*)thrust::raw_pointer_cast(&x[0])
#define CASTQ(x)  (quat*)thrust::raw_pointer_cast(&x[0])
#define CASTD1(x) (double*)thrust::raw_pointer_cast(&x[0])
#define CASTD2(x) (double2*)thrust::raw_pointer_cast(&x[0])
#define CASTD3(x) (double3*)thrust::raw_pointer_cast(&x[0])
#define CASTD4(x) (double4*)thrust::raw_pointer_cast(&x[0])
#define CASTB1(x) (bool*)thrust::raw_pointer_cast(&x[0])
#define CASTLL(x) (long long*)thrust::raw_pointer_cast(&x[0])
#define CASTS(x) (shape_type*)thrust::raw_pointer_cast(&x[0])

//====================================THRUST=================================//
#define Thrust_Inclusive_Scan_Sum(x,y)  thrust::inclusive_scan(x.begin(),x.end(), x.begin()); y=x.back();
#define Thrust_Sort_By_Key(x,y)         thrust::sort_by_key(x.begin(),x.end(),y.begin())
#define Thrust_Reduce_By_KeyA(x,y,z)x=  thrust::reduce_by_key(y.begin(),y.end(),thrust::constant_iterator<uint>(1),y.begin(),z.begin()).first-y.begin()
#define Thrust_Reduce_By_KeyB(x,y,z,w)x=thrust::reduce_by_key(y.begin(),y.end(),thrust::constant_iterator<uint>(1),z.begin(),w.begin()).first-z.begin()
#define Thrust_Inclusive_Scan(x)        thrust::inclusive_scan(x.begin(), x.end(), x.begin())
#define Thrust_Fill(x,y)                thrust::fill(x.begin(),x.end(),y)
#define Thrust_Sort(x)                  thrust::sort(x.begin(),x.end())
#define Thrust_Count(x,y)               thrust::count(x.begin(),x.end(),y)
#define Thrust_Sequence(x)              thrust::sequence(x.begin(),x.end())
#define Thrust_Equal(x,y)               thrust::equal(x.begin(),x.end(), y.begin())
#define Thrust_Max(x)                   x[thrust::max_element(x.begin(),x.end())-x.begin()]
#define Thrust_Min(x)                   x[thrust::max_element(x.begin(),x.end())-x.begin()]
#define Thrust_Total(x)                 thrust::reduce(x.begin(),x.end())

#define DBG(x)                          cudaThreadSynchronize();printf(x);CUT_CHECK_ERROR(x);fflush(stdout);
//====================================CUDA=================================//
#define THREADS                         128
#define MAXBLOCK                        65535
#define BLOCKS(x)                       max((int)ceil(x/(float)THREADS),1)
#define BLOCKS_T(x,y)                   max((int)ceil(x/(float)y),1)
#define BLOCKS2D(x)                     dim3(min(MAXBLOCK,BLOCKS(x)),ceil(BLOCKS(x)/(float)MAXBLOCK),1)
#define COPY_TO_CONST_MEM(x)            cudaMemcpyToSymbolAsync(x##_const,  &x, sizeof(x),0,cudaMemcpyHostToDevice)

#define INDEX1D (blockIdx.x * blockDim.x + threadIdx.x)
#define INDEX3D (threadIdx.x + blockDim.x * threadIdx.y + (blockIdx.x * blockDim.x * blockDim.y) + (blockIdx.y * blockDim.x * blockDim.y))

#define INIT_CHECK_THREAD_BOUNDED(x,y)  uint index = x; if (index >= y) { return;}

#endif

