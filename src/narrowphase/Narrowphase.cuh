/*******************************************************
 * Copyright (C) 2012-2013 Hammad Mazhar <hammad@hamelot.co.uk>, Simulation Based Engineering Lab <sbel.wisc.edu>
 * Some rights reserved. See LICENSE, AUTHORS.
 * This file is part of Chrono-Collide.
 *******************************************************/
#ifndef NARROWPHASE_H
#define NARROWPHASE_H

#pragma once
#include "collision_detection.h"
//#include "includes/gpu_math.h"
#include "SupportFunctions.cuh"
class Narrowphase {
    public:
        // variables

        // functions
        Narrowphase();
void DoNarrowphase(const custom_vector<shape_type> &obj_data_T,
        const custom_vector<real3> &obj_data_A,
        const custom_vector<real3> &obj_data_B,
        const custom_vector<real3> &obj_data_C,
        const custom_vector<quat> &obj_data_R,
        const custom_vector<uint> &obj_data_ID,
        const custom_vector<bool> & obj_active,
        const custom_vector<real3> &body_pos,
        const custom_vector<quat> &body_rot,
        custom_vector<long long> &potentialCollisions,
        custom_vector<real3> &norm_data,
        custom_vector<real3> &cpta_data,
        custom_vector<real3> &cptb_data,
        custom_vector<real> &dpth_data,
        custom_vector<ivec2> &bids_data,
        uint & number_of_contacts);

        void host_MPR_Store(
            const shape_type *obj_data_T,
            const real3 *obj_data_A,
            const real3 *obj_data_B,
            const real3 *obj_data_C,
            const quat *obj_data_R,
            const uint *obj_data_ID,
            const bool * obj_active,
            const real3 *body_pos,
            const quat *body_rot,
            long long *contact_pair,
            uint *contact_active,
            real3 *norm,
            real3 *ptA,
            real3 *ptB,
            real *contactDepth,
            ivec2 *ids);
        void SetCollisionEnvelope(const real &envelope) {
          collision_envelope = envelope;
        }
        real GetCollisionEnvelope() {
          return collision_envelope;
        }
        private:
        uint total_possible_contacts;
                real collision_envelope;

};

#endif


