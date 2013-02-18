#ifndef NARROWPHASE_H
#define NARROWPHASE_H

#pragma once
#include "sscd.h"
#include "sscdMath.h"
#include "SupportFunctions.cuh"
class Narrowphase {
    public:
        // variables

        // functions
        Narrowphase();
        void DoNarrowphase(const SHAPE *obj_data_T,
                           const real3 *obj_data_A,
                           const real3 *obj_data_B,
                           const real3 *obj_data_C,
                           const real4 *obj_data_R,
                           const uint   *obj_data_ID,
                           const real3 *body_pos,
                           const real4 *body_rot,
                           custom_vector<long long> &potentialCollisions,
                           custom_vector<real3> &norm_data,
                           custom_vector<real3> &cpta_data,
                           custom_vector<real3> &cptb_data,
                           custom_vector<real> &dpth_data,
                           custom_vector<uint2> &bids_data);

        void host_MPR_Store(
            const SHAPE *obj_data_T,
            const  real3 *obj_data_A,
            const  real3 *obj_data_B,
            const  real3 *obj_data_C,
            const  real4 *obj_data_R,
            const  uint *obj_data_ID,
            const real3 *body_pos,
            const real4 *body_rot,
            long long *contact_pair,
            uint *contact_active,
            real3 *norm,
            real3 *ptA,
            real3 *ptB,
            real *contactDepth,
            int2 *ids);


    private:
        uint total_possible_contacts;

        bool debugMode;


};

#endif

