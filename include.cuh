#ifndef INCLUDE_H
#define INCLUDE_H

#pragma once

#include <cuda.h>
#include <helper_math.h>
#include <vector>
#include <numeric>
#include <algorithm>
#include <iostream>
#include <string>
#include <sstream>
#include <fstream>
#include <GL/glut.h>
#include <omp.h>

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
#include <vector_types.h>

using namespace std;

// takes care of some GCC issues
#undef _GLIBCXX_ATOMIC_BUILTINS
#undef _GLIBCXX_USE_INT128

// defines to cast thrust vectors as raw pointers
#define CASTC1(x) (char*)thrust::raw_pointer_cast(&x[0])
#define CASTU1(x) (uint*)thrust::raw_pointer_cast(&x[0])
#define CASTU2(x) (uint2*)thrust::raw_pointer_cast(&x[0])
#define CASTU3(x) (uint3*)thrust::raw_pointer_cast(&x[0])
#define CASTI1(x) (int*)thrust::raw_pointer_cast(&x[0])
#define CASTI2(x) (int2*)thrust::raw_pointer_cast(&x[0])
#define CASTI3(x) (int3*)thrust::raw_pointer_cast(&x[0])
#define CASTI4(x) (int4*)thrust::raw_pointer_cast(&x[0])
#define CASTF1(x) (float*)thrust::raw_pointer_cast(&x[0])
#define CASTF2(x) (float2*)thrust::raw_pointer_cast(&x[0])
#define CASTF3(x) (float3*)thrust::raw_pointer_cast(&x[0])
#define CASTF4(x) (float4*)thrust::raw_pointer_cast(&x[0])
#define CASTD1(x) (double*)thrust::raw_pointer_cast(&x[0])
#define CASTD2(x) (double2*)thrust::raw_pointer_cast(&x[0])
#define CASTD3(x) (double3*)thrust::raw_pointer_cast(&x[0])
#define CASTD4(x) (double4*)thrust::raw_pointer_cast(&x[0])
#define CASTB1(x) (bool*)thrust::raw_pointer_cast(&x[0])
#define CASTLL(x) (long long*)thrust::raw_pointer_cast(&x[0])

// thrust helper functions
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

#define THREADS                         128
#define MAXBLOCK                        65535
#define BLOCKS(x)                       max((int)ceil(x/(float)THREADS),1)
#define BLOCKS_T(x,y)                   max((int)ceil(x/(float)y),1)
#define BLOCKS2D(x)                     dim3(min(MAXBLOCK,BLOCKS(x)),ceil(BLOCKS(x)/(float)MAXBLOCK),1)
#define COPY_TO_CONST_MEM(x)            cudaMemcpyToSymbolAsync(x##_const,  &x, sizeof(x),0,cudaMemcpyHostToDevice)

#define INDEX1D (blockIdx.x * blockDim.x + threadIdx.x)
#define INDEX3D (threadIdx.x + blockDim.x * threadIdx.y + (blockIdx.x * blockDim.x * blockDim.y) + (blockIdx.y * blockDim.x * blockDim.y))

#define INIT_CHECK_THREAD_BOUNDED(x,y)  uint index = x; if (index >= y) { return;}

#define F3  make_float3
#define F4  make_float4
#define F2  make_float2
#define I4  make_int4
#define I3  make_int3
#define I2  make_int2
#define U3  make_uint3
#define I3F make_int3f

#define OGL 1
#define SCALE 1

typedef float real;
typedef float3 realV;
typedef unsigned int uint;

#define SIM_ENABLE_GPU_MODE // Controls whether or not you are using CUDA
#ifdef SIM_ENABLE_GPU_MODE
	#define THRUST_DEVICE_SYSTEM THRUST_DEVICE_SYSTEM_CUDA
	#define custom_vector thrust::device_vector
#else
	#define THRUST_DEVICE_SYSTEM THRUST_DEVICE_SYSTEM_OMP
	#define custom_vector thrust::host_vector
#endif

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

inline float4 make_float4(
        float w,
        float3 a) {
    return make_float4(w, a.x, a.y, a.z);
}

inline float4 mult(
        const float4 &a,
        const float4 &b) {
    float w0 = a.x;
    float w1 = b.x;
    float3 v0 = make_float3(a.y, a.z, a.w);
    float3 v1 = make_float3(b.y, b.z, b.w);
    float4 quat = F4(w0 * w1 - dot(v0, v1), w0 * v1 + w1 * v0 + cross(v0, v1));

    //quat.x = a.x * b.x - a.y * b.y - a.z * b.z - a.w * b.w;
    //quat.y = a.x * b.y + a.y * b.x - a.w * b.z + a.z * b.w;
    //quat.z = a.x * b.z + a.z * b.x + a.w * b.y - a.y * b.w;
    //quat.w = a.x * b.w + a.w * b.x - a.z * b.y + a.y * b.z;
    return quat;
}
inline float4 inv(
        const float4& a) {
    return (1.0f / (dot(a, a))) * F4(a.x, -a.y, -a.z, -a.w);
}

inline float3 quatRotate(
        const float3 &v,
        const float4 &q) {
    float4 r = mult(mult(q, F4(0, v.x, v.y, v.z)), inv(q));
    return make_float3(r.y, r.z, r.w);
}

////////////////////////Quaternion and Vector Code////////////////////////
typedef double camreal;

struct camreal3 {

	camreal3(camreal a = 0, camreal b = 0, camreal c = 0) :
			x(a), y(b), z(c) {
	}

	camreal x, y, z;
};
struct camreal4 {

	camreal4(camreal d = 0, camreal a = 0, camreal b = 0, camreal c = 0) :
			w(d), x(a), y(b), z(c) {
	}

	camreal w, x, y, z;
};

static camreal3 operator +(const camreal3 rhs, const camreal3 lhs) {
	camreal3 temp;
	temp.x = rhs.x + lhs.x;
	temp.y = rhs.y + lhs.y;
	temp.z = rhs.z + lhs.z;
	return temp;
}
static camreal3 operator -(const camreal3 rhs, const camreal3 lhs) {
	camreal3 temp;
	temp.x = rhs.x - lhs.x;
	temp.y = rhs.y - lhs.y;
	temp.z = rhs.z - lhs.z;
	return temp;
}
static void operator +=(camreal3 &rhs, const camreal3 lhs) {
	rhs = rhs + lhs;
}

static void operator -=(camreal3 &rhs, const camreal3 lhs) {
	rhs = rhs - lhs;
}

static camreal3 operator *(const camreal3 rhs, const camreal3 lhs) {
	camreal3 temp;
	temp.x = rhs.x * lhs.x;
	temp.y = rhs.y * lhs.y;
	temp.z = rhs.z * lhs.z;
	return temp;
}

static camreal3 operator *(const camreal3 rhs, const camreal lhs) {
	camreal3 temp;
	temp.x = rhs.x * lhs;
	temp.y = rhs.y * lhs;
	temp.z = rhs.z * lhs;
	return temp;
}

static inline camreal3 cross(camreal3 a, camreal3 b) {
	return camreal3(a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z,
			a.x * b.y - a.y * b.x);
}

static camreal4 Q_from_AngAxis(camreal angle, camreal3 axis) {
	camreal4 quat;
	camreal halfang;
	camreal sinhalf;
	halfang = (angle * 0.5);
	sinhalf = sin(halfang);
	quat.w = cos(halfang);
	quat.x = axis.x * sinhalf;
	quat.y = axis.y * sinhalf;
	quat.z = axis.z * sinhalf;
	return (quat);
}

static camreal4 normalize(const camreal4 &a) {
	camreal length = 1.0 / sqrt(a.w * a.w + a.x * a.x + a.y * a.y + a.z * a.z);
	return camreal4(a.w * length, a.x * length, a.y * length, a.z * length);
}

static inline camreal4 inv(camreal4 a) {
//return (1.0f / (dot(a, a))) * F4(a.x, -a.y, -a.z, -a.w);
	camreal4 temp;
	camreal t1 = a.w * a.w + a.x * a.x + a.y * a.y + a.z * a.z;
	t1 = 1.0 / t1;
	temp.w = t1 * a.w;
	temp.x = -t1 * a.x;
	temp.y = -t1 * a.y;
	temp.z = -t1 * a.z;
	return temp;
}

static inline camreal4 mult(const camreal4 &a, const camreal4 &b) {
	camreal4 temp;
	temp.w = a.w * b.w - a.x * b.x - a.y * b.y - a.z * b.z;
	temp.x = a.w * b.x + b.w * a.x + a.y * b.z - a.z * b.y;
	temp.y = a.w * b.y + b.w * a.y + a.z * b.x - a.x * b.z;
	temp.z = a.w * b.z + b.w * a.z + a.x * b.y - a.y * b.x;
	return temp;
}

static inline camreal3 quatRotate(const camreal3 &v, const camreal4 &q) {
	camreal4 r = mult(mult(q, camreal4(0, v.x, v.y, v.z)), inv(q));
	return camreal3(r.x, r.y, r.z);
}

static camreal4 operator %(const camreal4 rhs, const camreal4 lhs) {
	return mult(rhs, lhs);
}
////////////////////////END Quaternion and Vector Code////////////////////////

class OpenGLCamera {
public:
	OpenGLCamera(camreal3 pos, camreal3 lookat, camreal3 up,
			camreal viewscale) {
		max_pitch_rate = 5;
		max_heading_rate = 5;
		camera_pos = pos;
		look_at = lookat;
		camera_up = up;
		camera_heading = 0;
		camera_pitch = 0;
		dir = camreal3(0, 0, 1);
		mouse_pos = camreal3(0, 0, 0);
		camera_pos_delta = camreal3(0, 0, 0);
		scale = viewscale;
	}
	void ChangePitch(GLfloat degrees) {
		if (fabs(degrees) < fabs(max_pitch_rate)) {
			camera_pitch += degrees;
		} else {
			if (degrees < 0) {
				camera_pitch -= max_pitch_rate;
			} else {
				camera_pitch += max_pitch_rate;
			}
		}

		if (camera_pitch > 360.0f) {
			camera_pitch -= 360.0f;
		} else if (camera_pitch < -360.0f) {
			camera_pitch += 360.0f;
		}
	}
	void ChangeHeading(GLfloat degrees) {
		if (fabs(degrees) < fabs(max_heading_rate)) {
			if ((camera_pitch > 90 && camera_pitch < 270)
					|| (camera_pitch < -90 && camera_pitch > -270)) {
				camera_heading -= degrees;
			} else {
				camera_heading += degrees;
			}
		} else {
			if (degrees < 0) {
				if ((camera_pitch > 90 && camera_pitch < 270)
						|| (camera_pitch < -90 && camera_pitch > -270)) {
					camera_heading += max_heading_rate;
				} else {
					camera_heading -= max_heading_rate;
				}
			} else {
				if ((camera_pitch > 90 && camera_pitch < 270)
						|| (camera_pitch < -90 && camera_pitch > -270)) {
					camera_heading -= max_heading_rate;
				} else {
					camera_heading += max_heading_rate;
				}
			}
		}

		if (camera_heading > 360.0f) {
			camera_heading -= 360.0f;
		} else if (camera_heading < -360.0f) {
			camera_heading += 360.0f;
		}
	}
	void Move2D(int x, int y) {
		camreal3 mouse_delta = mouse_pos - camreal3(x, y, 0);
		ChangeHeading(.02 * mouse_delta.x);
		ChangePitch(.02 * mouse_delta.y);
		mouse_pos = camreal3(x, y, 0);
	}
	void SetPos(int button, int state, int x, int y) {
		mouse_pos = camreal3(x, y, 0);
	}
	void Update() {
		camreal4 pitch_quat, heading_quat;
		camreal3 angle;
		angle = cross(dir, camera_up);
		pitch_quat = Q_from_AngAxis(camera_pitch, angle);
		heading_quat = Q_from_AngAxis(camera_heading, camera_up);
		camreal4 temp = (pitch_quat % heading_quat);
		temp = normalize(temp);
		dir = quatRotate(dir, temp);
		camera_pos += camera_pos_delta;
		look_at = camera_pos + dir * 1;
		camera_heading *= .5;
		camera_pitch *= .5;
		camera_pos_delta = camera_pos_delta * .5;
		gluLookAt(camera_pos.x, camera_pos.y, camera_pos.z, look_at.x,
				look_at.y, look_at.z, camera_up.x, camera_up.y, camera_up.z);
	}
	void Forward() {
		camera_pos_delta += dir * scale;
	}
	void Back() {
		camera_pos_delta -= dir * scale;
	}
	void Right() {
		camera_pos_delta += cross(dir, camera_up) * scale;
	}
	void Left() {
		camera_pos_delta -= cross(dir, camera_up) * scale;
	}
	void Up() {
		camera_pos_delta -= camera_up * scale;
	}
	void Down() {
		camera_pos_delta += camera_up * scale;
	}

	camreal max_pitch_rate, max_heading_rate;
	camreal3 camera_pos, look_at, camera_up;
	camreal camera_heading, camera_pitch, scale;
	camreal3 dir, mouse_pos, camera_pos_delta;
};

// collision detection structures
struct AABB {
    realV min, max;
};

typedef thrust::pair<realV, realV> bbox;

// reduce a pair of bounding boxes (a,b) to a bounding box containing a and b
struct bbox_reduction: public thrust::binary_function<bbox, bbox, bbox> {
    bbox __host__ __device__ operator()(
            bbox a,
            bbox b) {
        realV ll = F3(fmin(a.first.x, b.first.x), fmin(a.first.y, b.first.y), fmin(a.first.z, b.first.z)); // lower left corner
        realV ur = F3(fmax(a.second.x, b.second.x), fmax(a.second.y, b.second.y), fmax(a.second.z, b.second.z)); // upper right corner
        return bbox(ll, ur);
    }
};

// convert a point to a bbox containing that point, (point) -> (point, point)
struct bbox_transformation: public thrust::unary_function<realV, bbox> {
    bbox __host__ __device__ operator()(
            realV point) {
        return bbox(point, point);
    }
};
// end collision detection structures

class CollisionDetector {
private:
	// variables
    realV min_bounding_point;
    realV max_bounding_point;
    realV global_origin;
    realV bin_size_vec;
    realV bins_per_axis;

    //uint number_of_particles; // UNNECESSARY, should be number of AABBs
    uint numAABB;
    uint last_active_bin, number_of_bin_intersections, number_of_contacts_possible;//, number_of_contacts;
    uint val;

    custom_vector<realV> aabb_data;
    custom_vector<uint> Bins_Intersected;
    custom_vector<uint> bin_number;
    custom_vector<uint> body_number;
    custom_vector<uint> bin_start_index;
    custom_vector<uint> Num_ContactD;

	// functions
    //void host_Generate_AABB(const realV* pos, const real* radius, realV* aabb_data); //UNNECESSARY, user passes aabb_data to collision manager
    void host_Count_AABB_BIN_Intersection(
            const realV* aabb_data,
            uint* Bins_Intersected);
    void host_Store_AABB_BIN_Intersection(
            const realV* aabb_data,
            const uint* Bins_Intersected,
            uint * bin_number,
            uint * body_number);
    void host_Count_AABB_AABB_Intersection(
            const realV* aabb_data,
            const uint * bin_number,
            const uint * body_number,
            const uint * bin_start_index,
            //const bool* active, // UNNECESSARY, if user did not want the body to collide, they wouldn't have passed the bounding box in
            uint* Num_ContactD);
    void host_Store_AABB_AABB_Intersection(
            const float3* aabb_data,
            const uint * bin_number,
            const uint * body_number,
            const uint * bin_start_index,
            const uint* Num_ContactD,
            //const bool* active, // UNNECESSARY, if user did not want the body to collide, they wouldn't have passed the bounding box in
            long long* potential_contacts);
//    void host_Store_Contact( //(UNNECESSARY, user will process their own contacts)
//            const long long * potential_contacts,
//            const float3* pos,
//            const float* radius,
//            uint* id_a,
//            uint* id_b,
//            float3* Norm,
//            float * c_dist,
//            float* rest_len,
//            uint* counter);

public:
	// variables
    custom_vector<long long> potential_contacts;

	// functions
	CollisionDetector(custom_vector<realV> aabb_data);		// constructor
	int detectPossibleCollisions();
};

#endif
