#ifndef AABBGENERATOR_H
#define AABBGENERATOR_H

#pragma once
#include "sscd.h"
#include "includes/gpu_math.h"
class AABBGenerator {

        typedef thrust::pair<real3, real3> bbox;

    public:
        // variables
        custom_vector<real3> aabb_data;
        // functions
        AABBGenerator();
        void GenerateAABB(
            const custom_vector<SHAPE> &obj_data_T,
            const custom_vector<real3> &obj_data_A,
            const custom_vector<real3> &obj_data_B,
            const custom_vector<real3> &obj_data_C,
            const custom_vector<real4> &obj_data_R,
            const custom_vector<uint> &obj_data_ID,
            const custom_vector<real3> &body_pos,
            const custom_vector<real4> &body_rot);
        int activateDebugMode();
        int deactivateDebugMode();

    private:

        void host_ComputeAABB(
            const SHAPE *obj_data_T,
            const real3 *obj_data_A,
            const real3 *obj_data_B,
            const real3 *obj_data_C,
            const real4 *obj_data_R,
            const uint   *obj_data_ID,
            const real3 *body_pos,
            const real4 *body_rot,
            real3 *aabb_data
        );

        // variables
        uint numAABB;
        bool debugMode;
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
