/*******************************************************
 * Copyright (C) 2012-2013 Hammad Mazhar <hammad@hamelot.co.uk>, Simulation Based Engineering Lab <sbel.wisc.edu>
 * Some rights reserved. See LICENSE, AUTHORS.
 * This file is part of Chrono-Collide.
 *******************************************************/

#ifndef AABBGENERATOR_H
#define AABBGENERATOR_H

#pragma once
#include "collision_detection.h"
//#include "includes/gpu_math.h"
class AABBGenerator {

        typedef thrust::pair<real3, real3> bbox;

    public:

        // functions
        AABBGenerator();
                void GenerateAABB(
                    const custom_vector<shape_type> &obj_data_T, //Shape Type
                    const custom_vector<real3> &obj_data_A, //Data A
                    const custom_vector<real3> &obj_data_B, //Data B
                    const custom_vector<real3> &obj_data_C, //Data C
                    const custom_vector<quat> &obj_data_R, //Data D
                    const custom_vector<uint> &obj_data_ID, //Body ID
                    const custom_vector<real3> &body_pos,   //Position global
                    const custom_vector<quat> &body_rot,   //Rotation global
                    custom_vector<real3> &aabb_data);

            private:

                void host_ComputeAABB(
                    const shape_type *obj_data_T,    
                    const real3 *obj_data_A,
                    const real3 *obj_data_B,
                    const real3 *obj_data_C,
                    const quat *obj_data_R,
                    const uint  *obj_data_ID,
                    const real3 *body_pos,
                    const quat *body_rot,
                    real3 *aabb_data
                );

                // variables
                uint numAABB;

        };

#endif

//Sphere
//[x y z]       -   Local Position
//[r 0 0]       -   Radius
//[0 0 0]       -   NA
//[w x y z]     -   Rotation

//Ellipsoid
//[x y z]       -   Local Position
//[rx ry rz]    -   Radius
//[0 0 0]       -   NA
//[w x y z]     -   Rotation

//Box
//[x y z]       -   Local Position
//[rx ry rz]    -   Radius
//[0 0 0]       -   NA
//[w x y z]     -   Rotation

//Cylinder
//[x y z]       -   Local Position
//[rx ry rz]    -   Radius
//[0 0 0]       -   NA
//[w x y z]     -   Rotation

