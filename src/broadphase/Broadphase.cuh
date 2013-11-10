/*******************************************************
 * Copyright (C) 2012-2013 Hammad Mazhar <hammad@hamelot.co.uk>, Simulation Based Engineering Lab <sbel.wisc.edu>
 * Some rights reserved. See LICENSE, AUTHORS.
 * This file is part of Chrono-Collide.
 *******************************************************/
#ifndef BROADPHASE_H
#define BROADPHASE_H

#pragma once
#include "collision_detection.h"
#include "includes/gpu_math.h"
#include "AABBGenerator.cuh"

class Broadphase {
	public:
		Broadphase();
		int detectPossibleCollisions(custom_vector<real3> &aabb_data, custom_vector<long long> &potentialCollisions);
		int setBinsPerAxis(int3 binsPerAxis);
		int3 getBinsPerAxis();
		int setBodyPerBin(int max, int min);
		int getMaxBodyPerBin();
		int getMinBodyPerBin();
		uint getNumPossibleContacts();
		void setParallelConfiguration(
				bool parallel_transform_reduce,
				bool parallel_transform,
				bool parallel_inclusive_scan1,
				bool parallel_sort_by_key,
				bool parallel_reduce_by_key,
				bool parallel_max_element,
				bool parallel_inclusive_scan2,
				bool parallel_inclusive_scan3,
				bool parallel_sort,
				bool parallel_unique
		);
		void getParallelConfiguration();
		private:
		// variables
				real3 min_bounding_point;
				real3 max_bounding_point;
				real3 global_origin;
				real3 bin_size_vec;
				int3 grid_size;
				uint numAABB;
				uint last_active_bin, number_of_bin_intersections, number_of_contacts_possible;
				uint val;
				int min_body_per_bin,max_body_per_bin;

				bool par_transform_reduce;
				bool par_transform;
				bool par_inclusive_scan1;
				bool par_sort_by_key;
				bool par_reduce_by_key;
				bool par_max_element;
				bool par_inclusive_scan2;
				bool par_inclusive_scan3;
				bool par_sort;
				bool par_unique;

				// functions
				void host_Count_AABB_BIN_Intersection(
						const real3 *aabb_data,
						uint *Bins_Intersected);
				void host_Store_AABB_BIN_Intersection(
						const real3 *aabb_data,
						const uint *Bins_Intersected,
						uint *bin_number,
						uint *body_number);
				void host_Count_AABB_AABB_Intersection(
						const real3 *aabb_data,
						const uint *bin_number,
						const uint *body_number,
						const uint *bin_start_index,
						uint *Num_ContactD);
				void host_Store_AABB_AABB_Intersection(
						const real3 *aabb_data,
						const uint *bin_number,
						const uint *body_number,
						const uint *bin_start_index,
						const uint *Num_ContactD,
						long long *potentialCollisions);
			};

#endif

