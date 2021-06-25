/*
 * OpenVINS: An Open Platform for Visual-Inertial Research
 * Copyright (C) 2019 Patrick Geneva
 * Copyright (C) 2019 Kevin Eckenhoff
 * Copyright (C) 2019 Guoquan Huang
 * Copyright (C) 2019 OpenVINS Contributors
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <https://www.gnu.org/licenses/>.
 */
#include "UpdaterHelper.h"
#include <iomanip>

using namespace ov_core;
using namespace ov_msckf;


void UpdaterHelper::get_feature_jacobian_representation(std::shared_ptr<State> state, UpdaterHelperFeature &feature, Eigen::MatrixXd &H_f,
                                                        std::vector<Eigen::MatrixXd> &H_x, std::vector<std::shared_ptr<Type>> &x_order) {

    // Global XYZ representation
    if (feature.feat_representation == LandmarkRepresentation::Representation::GLOBAL_3D) {
        H_f.resize(3,3);
        H_f.setIdentity();
        return;
    }

    // Global inverse depth representation
    if (feature.feat_representation == LandmarkRepresentation::Representation::GLOBAL_FULL_INVERSE_DEPTH) {

        // Get the feature linearization point
        Eigen::Matrix<double,3,1> p_FinG = (state->_options.do_fej)? feature.p_FinG_fej : feature.p_FinG;

        // Get inverse depth representation (should match what is in Landmark.cpp)
        double g_rho = 1/p_FinG.norm();
        double g_phi = std::acos(g_rho*p_FinG(2));
        //double g_theta = std::asin(g_rho*p_FinG(1)/std::sin(g_phi));
        double g_theta = std::atan2(p_FinG(1),p_FinG(0));
        Eigen::Matrix<double,3,1> p_invFinG;
        p_invFinG(0) = g_theta;
        p_invFinG(1) = g_phi;
        p_invFinG(2) = g_rho;

        // Get inverse depth bearings
        double sin_th = std::sin(p_invFinG(0,0));
        double cos_th = std::cos(p_invFinG(0,0));
        double sin_phi = std::sin(p_invFinG(1,0));
        double cos_phi = std::cos(p_invFinG(1,0));
        double rho = p_invFinG(2,0);

        // Construct the Jacobian
        H_f.resize(3,3);
        H_f << -(1.0/rho)*sin_th*sin_phi, (1.0/rho)*cos_th*cos_phi, -(1.0/(rho*rho))*cos_th*sin_phi,
                (1.0/rho)*cos_th*sin_phi, (1.0/rho)*sin_th*cos_phi, -(1.0/(rho*rho))*sin_th*sin_phi,
                0.0, -(1.0/rho)*sin_phi, -(1.0/(rho*rho))*cos_phi;
        return;
    }


    //======================================================================
    //======================================================================
    //======================================================================


    // Assert that we have an anchor pose for this feature
    assert(feature.anchor_cam_id!=-1);

    // Anchor pose orientation and position, and camera calibration for our anchor camera
    Eigen::Matrix3d R_ItoC = state->_calib_IMUtoCAM.at(feature.anchor_cam_id)->Rot();
    Eigen::Vector3d p_IinC = state->_calib_IMUtoCAM.at(feature.anchor_cam_id)->pos();
    Eigen::Matrix3d R_GtoI = state->_clones_IMU.at(feature.anchor_clone_timestamp)->Rot();
    Eigen::Vector3d p_IinG = state->_clones_IMU.at(feature.anchor_clone_timestamp)->pos();
    Eigen::Vector3d p_FinA = feature.p_FinA;

    // If I am doing FEJ, I should FEJ the anchor states (should we fej calibration???)
    // Also get the FEJ position of the feature if we are
    if(state->_options.do_fej) {
        // "Best" feature in the global frame
        Eigen::Vector3d p_FinG_best = R_GtoI.transpose() * R_ItoC.transpose()*(feature.p_FinA - p_IinC) + p_IinG;
        // Transform the best into our anchor frame using FEJ
        R_GtoI = state->_clones_IMU.at(feature.anchor_clone_timestamp)->Rot_fej();
        p_IinG = state->_clones_IMU.at(feature.anchor_clone_timestamp)->pos_fej();
        p_FinA = (R_GtoI.transpose()*R_ItoC.transpose()).transpose()*(p_FinG_best - p_IinG) + p_IinC;
    }
    Eigen::Matrix3d R_CtoG = R_GtoI.transpose()*R_ItoC.transpose();

    // Jacobian for our anchor pose
    Eigen::Matrix<double,3,6> H_anc;
    H_anc.block(0,0,3,3).noalias() = -R_GtoI.transpose()*skew_x(R_ItoC.transpose()*(p_FinA-p_IinC));
    H_anc.block(0,3,3,3).setIdentity();

    // Add anchor Jacobians to our return vector
    x_order.push_back(state->_clones_IMU.at(feature.anchor_clone_timestamp));
    H_x.push_back(H_anc);

    // Get calibration Jacobians (for anchor clone)
    if (state->_options.do_calib_camera_pose) {
        Eigen::Matrix<double,3,6> H_calib;
        H_calib.block(0,0,3,3).noalias() = -R_CtoG*skew_x(p_FinA-p_IinC);
        H_calib.block(0,3,3,3) = -R_CtoG;
        x_order.push_back(state->_calib_IMUtoCAM.at(feature.anchor_cam_id));
        H_x.push_back(H_calib);
    }

    // If we are doing anchored XYZ feature
    if (feature.feat_representation == LandmarkRepresentation::Representation::ANCHORED_3D) {
        H_f = R_CtoG;
        return;
    }

    // If we are doing full inverse depth
    if (feature.feat_representation == LandmarkRepresentation::Representation::ANCHORED_FULL_INVERSE_DEPTH) {

        // Get inverse depth representation (should match what is in Landmark.cpp)
        double a_rho = 1/p_FinA.norm();
        double a_phi = std::acos(a_rho*p_FinA(2));
        double a_theta = std::atan2(p_FinA(1),p_FinA(0));
        Eigen::Matrix<double,3,1> p_invFinA;
        p_invFinA(0) = a_theta;
        p_invFinA(1) = a_phi;
        p_invFinA(2) = a_rho;

        // Using anchored inverse depth
        double sin_th = std::sin(p_invFinA(0,0));
        double cos_th = std::cos(p_invFinA(0,0));
        double sin_phi = std::sin(p_invFinA(1,0));
        double cos_phi = std::cos(p_invFinA(1,0));
        double rho = p_invFinA(2,0);
        //assert(p_invFinA(2,0)>=0.0);

        // Jacobian of anchored 3D position wrt inverse depth parameters
        Eigen::Matrix<double,3,3> d_pfinA_dpinv;
        d_pfinA_dpinv << -(1.0/rho)*sin_th*sin_phi, (1.0/rho)*cos_th*cos_phi, -(1.0/(rho*rho))*cos_th*sin_phi,
                (1.0/rho)*cos_th*sin_phi, (1.0/rho)*sin_th*cos_phi, -(1.0/(rho*rho))*sin_th*sin_phi,
                0.0, -(1.0/rho)*sin_phi, -(1.0/(rho*rho))*cos_phi;
        H_f = R_CtoG*d_pfinA_dpinv;
        return;
    }

    // If we are doing the MSCKF version of inverse depth
    if (feature.feat_representation == LandmarkRepresentation::Representation::ANCHORED_MSCKF_INVERSE_DEPTH) {

        // Get inverse depth representation (should match what is in Landmark.cpp)
        Eigen::Matrix<double,3,1> p_invFinA_MSCKF;
        p_invFinA_MSCKF(0) = p_FinA(0)/p_FinA(2);
        p_invFinA_MSCKF(1) = p_FinA(1)/p_FinA(2);
        p_invFinA_MSCKF(2) = 1/p_FinA(2);

        // Using the MSCKF version of inverse depth
        double alpha = p_invFinA_MSCKF(0,0);
        double beta = p_invFinA_MSCKF(1,0);
        double rho = p_invFinA_MSCKF(2,0);

        // Jacobian of anchored 3D position wrt inverse depth parameters
        Eigen::Matrix<double,3,3> d_pfinA_dpinv;
        d_pfinA_dpinv << (1.0/rho), 0.0, -(1.0/(rho*rho))*alpha,
                0.0, (1.0/rho), -(1.0/(rho*rho))*beta,
                0.0, 0.0, -(1.0/(rho*rho));
        H_f = R_CtoG*d_pfinA_dpinv;
        return;
    }

    /// CASE: Estimate single depth of the feature using the initial bearing
    if (feature.feat_representation == LandmarkRepresentation::Representation::ANCHORED_INVERSE_DEPTH_SINGLE) {

        // Get inverse depth representation (should match what is in Landmark.cpp)
        double rho = 1.0/p_FinA(2);
        Eigen::Vector3d bearing = rho*p_FinA;

        // Jacobian of anchored 3D position wrt inverse depth parameters
        Eigen::Vector3d d_pfinA_drho;
        d_pfinA_drho << -(1.0/(rho*rho))*bearing;
        H_f = R_CtoG*d_pfinA_drho;
        return;

    }

    // Failure, invalid representation that is not programmed
    assert(false);

}



void UpdaterHelper::get_feature_jacobian_intrinsics(std::shared_ptr<State> state, const Eigen::Vector2d &uv_norm, bool isfisheye,
                                                    Eigen::Matrix<double,8,1> cam_d, Eigen::Matrix<double,2,2> &dz_dzn, Eigen::Matrix<double,2,8> &dz_dzeta) {

    // Calculate distortion uv and jacobian
    if(isfisheye) {

        // Calculate distorted coordinates for fisheye
        double r = std::sqrt(uv_norm(0)*uv_norm(0)+uv_norm(1)*uv_norm(1));
        double theta = std::atan(r);
        double theta_d = theta+cam_d(4)*std::pow(theta,3)+cam_d(5)*std::pow(theta,5)+cam_d(6)*std::pow(theta,7)+cam_d(7)*std::pow(theta,9);

        // Handle when r is small (meaning our xy is near the camera center)
        double inv_r = (r > 1e-8)? 1.0/r : 1.0;
        double cdist = (r > 1e-8)? theta_d * inv_r : 1.0;

        // Jacobian of distorted pixel to "normalized" pixel
        Eigen::Matrix<double,2,2> duv_dxy = Eigen::Matrix<double,2,2>::Zero();
        duv_dxy << cam_d(0), 0, 0, cam_d(1);

        // Jacobian of "normalized" pixel to normalized pixel
        Eigen::Matrix<double,2,2> dxy_dxyn = Eigen::Matrix<double,2,2>::Zero();
        dxy_dxyn << theta_d*inv_r, 0, 0, theta_d*inv_r;

        // Jacobian of "normalized" pixel to r
        Eigen::Matrix<double,2,1> dxy_dr = Eigen::Matrix<double,2,1>::Zero();
        dxy_dr << -uv_norm(0)*theta_d*inv_r*inv_r, -uv_norm(1)*theta_d*inv_r*inv_r;

        // Jacobian of r pixel to normalized xy
        Eigen::Matrix<double,1,2> dr_dxyn = Eigen::Matrix<double,1,2>::Zero();
        dr_dxyn << uv_norm(0)*inv_r, uv_norm(1)*inv_r;

        // Jacobian of "normalized" pixel to theta_d
        Eigen::Matrix<double,2,1> dxy_dthd = Eigen::Matrix<double,2,1>::Zero();
        dxy_dthd << uv_norm(0)*inv_r, uv_norm(1)*inv_r;

        // Jacobian of theta_d to theta
        double dthd_dth = 1+3*cam_d(4)*std::pow(theta,2)+5*cam_d(5)*std::pow(theta,4)+7*cam_d(6)*std::pow(theta,6)+9*cam_d(7)*std::pow(theta,8);

        // Jacobian of theta to r
        double dth_dr = 1/(r*r+1);

        // Total Jacobian wrt normalized pixel coordinates
        dz_dzn = duv_dxy*(dxy_dxyn+(dxy_dr+dxy_dthd*dthd_dth*dth_dr)*dr_dxyn);

        // Compute the Jacobian in respect to the intrinsics if we are calibrating
        if(state->_options.do_calib_camera_intrinsics) {

            // Calculate distorted coordinates for fisheye
            double x1 = uv_norm(0)*cdist;
            double y1 = uv_norm(1)*cdist;

            // Jacobian
            dz_dzeta(0,0) = x1;
            dz_dzeta(0,2) = 1;
            dz_dzeta(0,4) = cam_d(0)*uv_norm(0)*inv_r*std::pow(theta,3);
            dz_dzeta(0,5) = cam_d(0)*uv_norm(0)*inv_r*std::pow(theta,5);
            dz_dzeta(0,6) = cam_d(0)*uv_norm(0)*inv_r*std::pow(theta,7);
            dz_dzeta(0,7) = cam_d(0)*uv_norm(0)*inv_r*std::pow(theta,9);
            dz_dzeta(1,1) = y1;
            dz_dzeta(1,3) = 1;
            dz_dzeta(1,4) = cam_d(1)*uv_norm(1)*inv_r*std::pow(theta,3);
            dz_dzeta(1,5) = cam_d(1)*uv_norm(1)*inv_r*std::pow(theta,5);
            dz_dzeta(1,6) = cam_d(1)*uv_norm(1)*inv_r*std::pow(theta,7);
            dz_dzeta(1,7) = cam_d(1)*uv_norm(1)*inv_r*std::pow(theta,9);
        }


    } else {

        // Calculate distorted coordinates for radial
        double r = std::sqrt(uv_norm(0)*uv_norm(0)+uv_norm(1)*uv_norm(1));
        double r_2 = r*r;
        double r_4 = r_2*r_2;

        // Jacobian of distorted pixel to normalized pixel
        double x = uv_norm(0);
        double y = uv_norm(1);
        double x_2 = uv_norm(0)*uv_norm(0);
        double y_2 = uv_norm(1)*uv_norm(1);
        double x_y = uv_norm(0)*uv_norm(1);
        dz_dzn(0,0) = cam_d(0)*((1+cam_d(4)*r_2+cam_d(5)*r_4)+(2*cam_d(4)*x_2+4*cam_d(5)*x_2*r)+2*cam_d(6)*y+(2*cam_d(7)*x+4*cam_d(7)*x));
        dz_dzn(0,1) = cam_d(0)*(2*cam_d(4)*x_y+4*cam_d(5)*x_y*r+2*cam_d(6)*x+2*cam_d(7)*y);
        dz_dzn(1,0) = cam_d(1)*(2*cam_d(4)*x_y+4*cam_d(5)*x_y*r+2*cam_d(6)*x+2*cam_d(7)*y);
        dz_dzn(1,1) = cam_d(1)*((1+cam_d(4)*r_2+cam_d(5)*r_4)+(2*cam_d(4)*y_2+4*cam_d(5)*y_2*r)+2*cam_d(7)*x+(2*cam_d(6)*y+4*cam_d(6)*y));

        // Compute the Jacobian in respect to the intrinsics if we are calibrating
        if(state->_options.do_calib_camera_intrinsics) {

            // Calculate distorted coordinates for radtan
            double x1 = uv_norm(0)*(1+cam_d(4)*r_2+cam_d(5)*r_4)+2*cam_d(6)*uv_norm(0)*uv_norm(1)+cam_d(7)*(r_2+2*uv_norm(0)*uv_norm(0));
            double y1 = uv_norm(1)*(1+cam_d(4)*r_2+cam_d(5)*r_4)+cam_d(6)*(r_2+2*uv_norm(1)*uv_norm(1))+2*cam_d(7)*uv_norm(0)*uv_norm(1);

            // Jacobian
            dz_dzeta(0,0) = x1;
            dz_dzeta(0,2) = 1;
            dz_dzeta(0,4) = cam_d(0)*uv_norm(0)*r_2;
            dz_dzeta(0,5) = cam_d(0)*uv_norm(0)*r_4;
            dz_dzeta(0,6) = 2*cam_d(0)*uv_norm(0)*uv_norm(1);
            dz_dzeta(0,7) = cam_d(0)*(r_2+2*uv_norm(0)*uv_norm(0));
            dz_dzeta(1,1) = y1;
            dz_dzeta(1,3) = 1;
            dz_dzeta(1,4) = cam_d(1)*uv_norm(1)*r_2;
            dz_dzeta(1,5) = cam_d(1)*uv_norm(1)*r_4;
            dz_dzeta(1,6) = cam_d(1)*(r_2+2*uv_norm(1)*uv_norm(1));
            dz_dzeta(1,7) = 2*cam_d(1)*uv_norm(0)*uv_norm(1);
        }

    }


}



void UpdaterHelper::get_feature_jacobian_full(std::shared_ptr<State> state, UpdaterHelperFeature &feature, Eigen::MatrixXd &H_f, Eigen::MatrixXd &H_x, Eigen::VectorXd &res, std::vector<std::shared_ptr<Type>> &x_order) {

    // Total number of measurements for this feature
    int total_meas = 0;
    for (auto const& pair : feature.timestamps) {
        total_meas += (int)pair.second.size();
    }

    // Compute the size of the states involved with this feature
    int total_hx = 0;
    std::unordered_map<std::shared_ptr<Type>,size_t> map_hx;
    for (auto const& pair : feature.timestamps) {

        // Our extrinsics and intrinsics
        std::shared_ptr<PoseJPL> calibration = state->_calib_IMUtoCAM.at(pair.first);
        std::shared_ptr<Vec> distortion = state->_cam_intrinsics.at(pair.first);

        // If doing calibration extrinsics
        if(state->_options.do_calib_camera_pose) {
            map_hx.insert({calibration,total_hx});
            x_order.push_back(calibration);
            total_hx += calibration->size();
        }

        // If doing calibration intrinsics
        if(state->_options.do_calib_camera_intrinsics) {
            map_hx.insert({distortion,total_hx});
            x_order.push_back(distortion);
            total_hx += distortion->size();
        }

        // Loop through all measurements for this specific camera
        for (size_t m = 0; m < feature.timestamps[pair.first].size(); m++) {

            // Add this clone if it is not added already
            std::shared_ptr<PoseJPL> clone_Ci = state->_clones_IMU.at(feature.timestamps[pair.first].at(m));
            if(map_hx.find(clone_Ci) == map_hx.end()) {
                map_hx.insert({clone_Ci,total_hx});
                x_order.push_back(clone_Ci);
                total_hx += clone_Ci->size();
            }

        }

    }

    // If we are using an anchored representation, make sure that the anchor is also added
    if (LandmarkRepresentation::is_relative_representation(feature.feat_representation)) {

        // Assert we have a clone
        assert(feature.anchor_cam_id != -1);

        // Add this anchor if it is not added already
        std::shared_ptr<PoseJPL> clone_Ai = state->_clones_IMU.at(feature.anchor_clone_timestamp);
        if(map_hx.find(clone_Ai) == map_hx.end()) {
            map_hx.insert({clone_Ai,total_hx});
            x_order.push_back(clone_Ai);
            total_hx += clone_Ai->size();
        }

        // Also add its calibration if we are doing calibration
        if(state->_options.do_calib_camera_pose) {
            // Add this anchor if it is not added already
            std::shared_ptr<PoseJPL> clone_calib = state->_calib_IMUtoCAM.at(feature.anchor_cam_id);
            if(map_hx.find(clone_calib) == map_hx.end()) {
                map_hx.insert({clone_calib,total_hx});
                x_order.push_back(clone_calib);
                total_hx += clone_calib->size();
            }
        }

    }

    //=========================================================================
    //=========================================================================

    // Calculate the position of this feature in the global frame
    // If anchored, then we need to calculate the position of the feature in the global
    Eigen::Vector3d p_FinG = feature.p_FinG;
    if(LandmarkRepresentation::is_relative_representation(feature.feat_representation)) {
        // Assert that we have an anchor pose for this feature
        assert(feature.anchor_cam_id!=-1);
        // Get calibration for our anchor camera
        Eigen::Matrix<double, 3, 3> R_ItoC = state->_calib_IMUtoCAM.at(feature.anchor_cam_id)->Rot();
        Eigen::Matrix<double, 3, 1> p_IinC = state->_calib_IMUtoCAM.at(feature.anchor_cam_id)->pos();
        // Anchor pose orientation and position
        Eigen::Matrix<double,3,3> R_GtoI = state->_clones_IMU.at(feature.anchor_clone_timestamp)->Rot();
        Eigen::Matrix<double,3,1> p_IinG = state->_clones_IMU.at(feature.anchor_clone_timestamp)->pos();
        // Feature in the global frame
        p_FinG = R_GtoI.transpose() * R_ItoC.transpose()*(feature.p_FinA - p_IinC) + p_IinG;
    }

    // Calculate the position of this feature in the global frame FEJ
    // If anchored, then we can use the "best" p_FinG since the value of p_FinA does not matter
    Eigen::Vector3d p_FinG_fej = feature.p_FinG_fej;
    if(LandmarkRepresentation::is_relative_representation(feature.feat_representation)) {
        p_FinG_fej = p_FinG;
    }

    //=========================================================================
    //=========================================================================

    // Allocate our residual and Jacobians
    int c = 0;
    int jacobsize = (feature.feat_representation!=LandmarkRepresentation::Representation::ANCHORED_INVERSE_DEPTH_SINGLE) ? 3 : 1;
    res = Eigen::VectorXd::Zero(2*total_meas);
    H_f = Eigen::MatrixXd::Zero(2*total_meas,jacobsize);
    H_x = Eigen::MatrixXd::Zero(2*total_meas,total_hx);

    // Derivative of p_FinG in respect to feature representation.
    // This only needs to be computed once and thus we pull it out of the loop
    Eigen::MatrixXd dpfg_dlambda;
    std::vector<Eigen::MatrixXd> dpfg_dx;
    std::vector<std::shared_ptr<Type>> dpfg_dx_order;
    UpdaterHelper::get_feature_jacobian_representation(state, feature, dpfg_dlambda, dpfg_dx, dpfg_dx_order);

    // Assert that all the ones in our order are already in our local jacobian mapping
    for(auto &type : dpfg_dx_order) {
        assert(map_hx.find(type)!=map_hx.end());
    }
    // Loop through each camera for this feature
    for (auto const& pair : feature.timestamps) { ////////////this for loop is redundent cuz it happens once

        // Our calibration between the IMU and CAMi frames
        std::shared_ptr<Vec> distortion = state->_cam_intrinsics.at(pair.first);
        std::shared_ptr<PoseJPL> calibration = state->_calib_IMUtoCAM.at(pair.first); ////////////// Camera <-> IMU
        // std::cout << " feature.timestamps.size() \n" <<   feature.timestamps.size() << std::endl;
        // std::cout << " state->_calib_IMUtoCAM.size() \n" <<  state->_calib_IMUtoCAM.size() << std::endl;
        // std::cin.get();
        Eigen::Matrix<double,3,3> R_ItoC = calibration->Rot();   /// this is pose between imu and camera
        Eigen::Matrix<double,3,1> p_IinC = calibration->pos();
        Eigen::Matrix<double,8,1> cam_d = distortion->value();

        // Loop through all measurements for this specific camera
        for (size_t m = 0; m < feature.timestamps[pair.first].size(); m++) { /// this is actually for all cameras seeing that feature
            
            //=========================================================================
            //=========================================================================

            // Get current IMU clone state
            std::shared_ptr<PoseJPL> clone_Ii = state->_clones_IMU.at(feature.timestamps[pair.first].at(m));
            Eigen::Matrix<double,3,3> R_GtoIi = clone_Ii->Rot();
            Eigen::Matrix<double,3,1> p_IiinG = clone_Ii->pos();

            // Get current feature in the IMU
            Eigen::Matrix<double,3,1> p_FinIi = R_GtoIi*(p_FinG-p_IiinG);
            // std::cin.get();
            // if(m==1){

            //     std::cout << " p_IiinG \n" <<  p_IiinG << std::endl;
            //     std::cout << "p_FinG = " << p_FinG << std::endl;
            //     std::cin.get();
            // }

            // Project the current feature into the current frame of reference
            Eigen::Matrix<double,3,1> p_FinCi = R_ItoC*p_FinIi+p_IinC;    //// MSCKF line 1102
            Eigen::Matrix<double,2,1> uv_norm;
            uv_norm << p_FinCi(0)/p_FinCi(2),p_FinCi(1)/p_FinCi(2); ///////z_hat_i_(j)

            // Distort the normalized coordinates (false=radtan, true=fisheye)
            Eigen::Matrix<double,2,1> uv_dist;

            // Calculate distortion uv and jacobian
            if(state->_cam_intrinsics_model.at(pair.first)) {
                // std::cout << "fisheye \n" << std::endl;

                // Calculate distorted coordinates for fisheye
                double r = std::sqrt(uv_norm(0)*uv_norm(0)+uv_norm(1)*uv_norm(1));
                double theta = std::atan(r);
                double theta_d = theta+cam_d(4)*std::pow(theta,3)+cam_d(5)*std::pow(theta,5)+cam_d(6)*std::pow(theta,7)+cam_d(7)*std::pow(theta,9);

                // Handle when r is small (meaning our xy is near the camera center)
                double inv_r = (r > 1e-8)? 1.0/r : 1.0;
                double cdist = (r > 1e-8)? theta_d * inv_r : 1.0;

                // Calculate distorted coordinates for fisheye
                double x1 = uv_norm(0)*cdist;
                double y1 = uv_norm(1)*cdist;
                uv_dist(0) = cam_d(0)*x1 + cam_d(2);
                uv_dist(1) = cam_d(1)*y1 + cam_d(3);

            } else {
                // std::cout << "radial \n" << std::endl;

                // Calculate distorted coordinates for radial
                double r = std::sqrt(uv_norm(0)*uv_norm(0)+uv_norm(1)*uv_norm(1));
                double r_2 = r*r;
                double r_4 = r_2*r_2;
                double x1 = uv_norm(0)*(1+cam_d(4)*r_2+cam_d(5)*r_4)+2*cam_d(6)*uv_norm(0)*uv_norm(1)+cam_d(7)*(r_2+2*uv_norm(0)*uv_norm(0));
                double y1 = uv_norm(1)*(1+cam_d(4)*r_2+cam_d(5)*r_4)+cam_d(6)*(r_2+2*uv_norm(1)*uv_norm(1))+2*cam_d(7)*uv_norm(0)*uv_norm(1);
                uv_dist(0) = cam_d(0)*x1 + cam_d(2);
                uv_dist(1) = cam_d(1)*y1 + cam_d(3);

            }

            // Our residual
            Eigen::Matrix<double,2,1> uv_m;      //////////////// measurement imp
            uv_m << (double)feature.uvs[pair.first].at(m)(0), (double)feature.uvs[pair.first].at(m)(1);
            // std::cout << "uv_norm \n" << uv_norm << std::endl;
            // std::cout << "uv_dist \n" << uv_dist << std::endl;
            // std::cin.get();
            res.block(2*c,0,2,1) = uv_m - uv_dist;  /// eq 20 


            //=========================================================================
            //=========================================================================

            // If we are doing first estimate Jacobians, then overwrite with the first estimates
            if(state->_options.do_fej) {
                R_GtoIi = clone_Ii->Rot_fej();
                p_IiinG = clone_Ii->pos_fej();
                //R_ItoC = calibration->Rot_fej();
                //p_IinC = calibration->pos_fej();
                p_FinIi = R_GtoIi*(p_FinG_fej-p_IiinG);
                p_FinCi = R_ItoC*p_FinIi+p_IinC;
                //uv_norm << p_FinCi(0)/p_FinCi(2),p_FinCi(1)/p_FinCi(2);
                //cam_d = state->get_intrinsics_CAM(pair.first)->fej();
            }

            // Compute Jacobians in respect to normalized image coordinates and possibly the camera intrinsics
            Eigen::Matrix<double,2,2> dz_dzn = Eigen::Matrix<double,2,2>::Zero();
            Eigen::Matrix<double,2,8> dz_dzeta = Eigen::Matrix<double,2,8>::Zero();
            UpdaterHelper::get_feature_jacobian_intrinsics(state, uv_norm, state->_cam_intrinsics_model.at(pair.first), cam_d, dz_dzn, dz_dzeta);

            // Normalized coordinates in respect to projection function
            Eigen::Matrix<double,2,3> dzn_dpfc = Eigen::Matrix<double,2,3>::Zero();
            dzn_dpfc << 1/p_FinCi(2),0,-p_FinCi(0)/(p_FinCi(2)*p_FinCi(2)),
                    0, 1/p_FinCi(2),-p_FinCi(1)/(p_FinCi(2)*p_FinCi(2));

            // Derivative of p_FinCi in respect to p_FinIi
            Eigen::Matrix<double,3,3> dpfc_dpfg = R_ItoC*R_GtoIi;

            // Derivative of p_FinCi in respect to camera clone state
            Eigen::Matrix<double,3,6> dpfc_dclone = Eigen::Matrix<double,3,6>::Zero();
            dpfc_dclone.block(0,0,3,3).noalias() = R_ItoC*skew_x(p_FinIi);
            dpfc_dclone.block(0,3,3,3) = -dpfc_dpfg;

            //=========================================================================
            //=========================================================================


            // Precompute some matrices
            Eigen::Matrix<double,2,3> dz_dpfc = dz_dzn*dzn_dpfc;
            Eigen::Matrix<double,2,3> dz_dpfg = dz_dpfc*dpfc_dpfg;

            // CHAINRULE: get the total feature Jacobian
            H_f.block(2*c,0,2,H_f.cols()).noalias() = dz_dpfg*dpfg_dlambda;

            // CHAINRULE: get state clone Jacobian
            H_x.block(2*c,map_hx[clone_Ii],2,clone_Ii->size()).noalias() = dz_dpfc*dpfc_dclone;

            // CHAINRULE: loop through all extra states and add their
            // NOTE: we add the Jacobian here as we might be in the anchoring pose for this measurement
            for(size_t i=0; i<dpfg_dx_order.size(); i++) {
                H_x.block(2*c,map_hx[dpfg_dx_order.at(i)],2,dpfg_dx_order.at(i)->size()).noalias() += dz_dpfg*dpfg_dx.at(i);
            }

            //=========================================================================
            //=========================================================================

            // Derivative of p_FinCi in respect to camera calibration (R_ItoC, p_IinC)
            if(state->_options.do_calib_camera_pose) {

                // Calculate the Jacobian
                Eigen::Matrix<double,3,6> dpfc_dcalib = Eigen::Matrix<double,3,6>::Zero();
                dpfc_dcalib.block(0,0,3,3) = skew_x(p_FinCi-p_IinC);
                dpfc_dcalib.block(0,3,3,3) = Eigen::Matrix<double,3,3>::Identity();

                // Chainrule it and add it to the big jacobian
                H_x.block(2*c,map_hx[calibration],2,calibration->size()).noalias() += dz_dpfc*dpfc_dcalib;

            }

            // Derivative of measurement in respect to distortion parameters
            if(state->_options.do_calib_camera_intrinsics) {
                H_x.block(2*c,map_hx[distortion],2,distortion->size()) = dz_dzeta;
            }

            // Move the Jacobian and residual index forward
            c++;

        }

    }


}

double UpdaterHelper::compute_error(std::shared_ptr<State> state, 
                                    UpdaterHelperFeature &feature, 
                                    Eigen::MatrixXd &sigma_cam_states_, 
                                    double alpha, double beta, double rho){
    
    // Total error
    double err = 0;

    std::shared_ptr<PoseJPL> calibration = state->_calib_IMUtoCAM.at(0); ////////////// Camera <-> IMU
    Eigen::Matrix<double,3,3> R_ItoC = calibration->Rot();   /// this is pose between imu and camera
    Eigen::Matrix<double,3,1> p_IinC = calibration->pos();

    // Get the position of the anchor pose
    Eigen::Matrix<double, 4, 1> q_anchor; 
    q_anchor = sigma_cam_states_.block(0,0,1,4).transpose();
    Eigen::Matrix<double,3,3> R_GtoIi = ov_core::quat_2_Rot(q_anchor);
    Eigen::Matrix<double,3,1> p_IiinG = sigma_cam_states_.block(0,4,1,3).transpose();
    Eigen::Matrix<double,3,3> R_GtoA =  R_ItoC * R_GtoIi;
    Eigen::Matrix<double,3,1> p_AinG =  p_IiinG - R_GtoA.transpose()*p_IinC;

    auto first = state->_clones_IMU.begin();
    // Loop through each camera for this feature
    for (auto const& pair : feature.timestamps) {
        // Add CAM_I features
        for (size_t m = 0; m < feature.timestamps[pair.first].size(); m++) {

            //=====================================================================================
            //=====================================================================================

            auto search = state->_clones_IMU.find(feature.timestamps[pair.first].at(m));
            int index = distance(first, search);

            // Get current IMU clone state
            Eigen::Matrix<double, 4, 1> q_sigma;
            q_sigma = sigma_cam_states_.block(0,7*index,1,4).transpose();
            R_GtoIi = ov_core::quat_2_Rot(q_sigma);
            p_IiinG = sigma_cam_states_.block(0,4+7*index ,1,3).transpose();

            // Calculate Cam clone in the global
            Eigen::Matrix<double,3,3> R_GtoCi =  R_ItoC * R_GtoIi;
            Eigen::Matrix<double,3,1> p_CiinG =  p_IiinG - R_GtoCi.transpose()*p_IinC;

            // Convert current position relative to anchor
            Eigen::Matrix<double,3,3> R_AtoCi;
            R_AtoCi.noalias() = R_GtoCi*R_GtoA.transpose();
            Eigen::Matrix<double,3,1> p_CiinA;
            p_CiinA.noalias() = R_GtoA*(p_CiinG-p_AinG);
            Eigen::Matrix<double,3,1> p_AinCi;
            p_AinCi.noalias() = -R_AtoCi*p_CiinA;

            //=====================================================================================
            //=====================================================================================

            // Middle variables of the system
            double hi1 = R_AtoCi(0, 0) * alpha + R_AtoCi(0, 1) * beta + R_AtoCi(0, 2) + rho * p_AinCi(0, 0);
            double hi2 = R_AtoCi(1, 0) * alpha + R_AtoCi(1, 1) * beta + R_AtoCi(1, 2) + rho * p_AinCi(1, 0);
            double hi3 = R_AtoCi(2, 0) * alpha + R_AtoCi(2, 1) * beta + R_AtoCi(2, 2) + rho * p_AinCi(2, 0);
            // Calculate residual
            Eigen::Matrix<float, 2, 1> z;
            z << hi1 / hi3, hi2 / hi3;
            Eigen::Matrix<float, 2, 1> res = feature.uvs_norm[pair.first].at(m) - z;
            // Append to our summation variables
            err += pow(res.norm(), 2);
        }
    }

    return err;

}


bool UpdaterHelper::single_gaussnewton(std::shared_ptr<State> state, 
                                        UpdaterHelperFeature &feature,
                                        Eigen::MatrixXd &sigma_cam_states_,
                                        Eigen::Vector3d &p_FinA,
                                        Eigen::Vector3d &p_FinG) {

    //Get into inverse depth
    double rho = 1/p_FinA(2);
    double alpha = p_FinA(0)/p_FinA(2);
    double beta = p_FinA(1)/p_FinA(2);

    // Optimization parameters
    /// Init lambda for Levenberg-Marquardt optimization
    double init_lamda = 1e-3;
    /// Max runs for Levenberg-Marquardt
    int max_runs = 5;
    /// Max lambda for Levenberg-Marquardt optimization
    double max_lamda = 1e10;
    /// Cutoff for dx increment to consider as converged
    double min_dx = 1e-6;
    /// Cutoff for cost decrement to consider as converged
    double min_dcost = 1e-6;
    /// Multiplier to increase/decrease lambda
    double lam_mult = 10;
    /// Minimum distance to accept triangulated features
    double min_dist = 0.10;
    /// Minimum distance to accept triangulated features
    double max_dist = 60;
    /// Max baseline ratio to accept triangulated features
    double max_baseline = 40;
    //
    double lam = init_lamda;
    double eps = 10000;
    int runs = 0;

    // Variables used in the optimization
    bool recompute = true;
    Eigen::Matrix<double,3,3> Hess = Eigen::Matrix<double,3,3>::Zero();
    Eigen::Matrix<double,3,1> grad = Eigen::Matrix<double,3,1>::Zero(); 

     // Cost at the last iteration
    double cost_old = compute_error(state, feature, sigma_cam_states_, alpha, beta, rho);

    std::shared_ptr<PoseJPL> calibration = state->_calib_IMUtoCAM.at(0); ////////////// Camera <-> IMU
    Eigen::Matrix<double,3,3> R_ItoC = calibration->Rot();   /// this is pose between imu and camera
    Eigen::Matrix<double,3,1> p_IinC = calibration->pos();

    // Get the position of the anchor pose
    Eigen::Matrix<double, 4, 1> q_anchor; 
    q_anchor = sigma_cam_states_.block(0,0,1,4).transpose();
    Eigen::Matrix<double,3,3> R_GtoIi = ov_core::quat_2_Rot(q_anchor);
    Eigen::Matrix<double,3,1> p_IiinG = sigma_cam_states_.block(0,4,1,3).transpose();
    Eigen::Matrix<double,3,3> R_GtoA =  R_ItoC * R_GtoIi;
    Eigen::Matrix<double,3,1> p_AinG =  p_IiinG - R_GtoA.transpose()*p_IinC;

    // Loop till we have either
    // 1. Reached our max iteration count
    // 2. System is unstable
    // 3. System has converged
    while (runs < max_runs && lam < max_lamda && eps > min_dx) {

        // Triggers a recomputation of jacobians/information/gradients
        if (recompute) {

            Hess.setZero();
            grad.setZero();

            double err = 0;

            auto first = state->_clones_IMU.begin();
            // Loop through each camera for this feature
            for (auto const& pair : feature.timestamps) {

                // Add CAM_I features
                for (size_t m = 0; m < feature.timestamps[pair.first].size(); m++) {

                    //=====================================================================================
                    //=====================================================================================

                    auto search = state->_clones_IMU.find(feature.timestamps[pair.first].at(m));
                    int index = distance(first, search);

                    // Get current IMU clone state
                    Eigen::Matrix<double, 4, 1> q_sigma;
                    q_sigma = sigma_cam_states_.block(0,7*index,1,4).transpose();
                    R_GtoIi = ov_core::quat_2_Rot(q_sigma);
                    p_IiinG = sigma_cam_states_.block(0,4+7*index ,1,3).transpose();

                    // Calculate Cam clone in the global
                    Eigen::Matrix<double,3,3> R_GtoCi =  R_ItoC * R_GtoIi;
                    Eigen::Matrix<double,3,1> p_CiinG =  p_IiinG - R_GtoCi.transpose()*p_IinC;

                    // Convert current position relative to anchor
                    Eigen::Matrix<double,3,3> R_AtoCi;
                    R_AtoCi.noalias() = R_GtoCi*R_GtoA.transpose();
                    Eigen::Matrix<double,3,1> p_CiinA;
                    p_CiinA.noalias() = R_GtoA*(p_CiinG-p_AinG);
                    Eigen::Matrix<double,3,1> p_AinCi;
                    p_AinCi.noalias() = -R_AtoCi*p_CiinA;

                    //=====================================================================================
                    //=====================================================================================

                    // Middle variables of the system
                    double hi1 = R_AtoCi(0, 0) * alpha + R_AtoCi(0, 1) * beta + R_AtoCi(0, 2) + rho * p_AinCi(0, 0);
                    double hi2 = R_AtoCi(1, 0) * alpha + R_AtoCi(1, 1) * beta + R_AtoCi(1, 2) + rho * p_AinCi(1, 0);
                    double hi3 = R_AtoCi(2, 0) * alpha + R_AtoCi(2, 1) * beta + R_AtoCi(2, 2) + rho * p_AinCi(2, 0);
                    // Calculate jacobian
                    double d_z1_d_alpha = (R_AtoCi(0, 0) * hi3 - hi1 * R_AtoCi(2, 0)) / (pow(hi3, 2));
                    double d_z1_d_beta = (R_AtoCi(0, 1) * hi3 - hi1 * R_AtoCi(2, 1)) / (pow(hi3, 2));
                    double d_z1_d_rho = (p_AinCi(0, 0) * hi3 - hi1 * p_AinCi(2, 0)) / (pow(hi3, 2));
                    double d_z2_d_alpha = (R_AtoCi(1, 0) * hi3 - hi2 * R_AtoCi(2, 0)) / (pow(hi3, 2));
                    double d_z2_d_beta = (R_AtoCi(1, 1) * hi3 - hi2 * R_AtoCi(2, 1)) / (pow(hi3, 2));
                    double d_z2_d_rho = (p_AinCi(1, 0) * hi3 - hi2 * p_AinCi(2, 0)) / (pow(hi3, 2));
                    Eigen::Matrix<double, 2, 3> H;
                    H << d_z1_d_alpha, d_z1_d_beta, d_z1_d_rho, d_z2_d_alpha, d_z2_d_beta, d_z2_d_rho;
                    // Calculate residual
                    Eigen::Matrix<float, 2, 1> z;
                    z << hi1 / hi3, hi2 / hi3;
                    Eigen::Matrix<float, 2, 1> res = feature.uvs_norm[pair.first].at(m) - z;

                    //=====================================================================================
                    //=====================================================================================

                    // Append to our summation variables
                    err += std::pow(res.norm(), 2);
                    grad.noalias() += H.transpose() * res.cast<double>();
                    Hess.noalias() += H.transpose() * H;
                }

            }

        }

        // Solve Levenberg iteration
        Eigen::Matrix<double,3,3> Hess_l = Hess;
        for (size_t r=0; r < (size_t)Hess.rows(); r++) {
            Hess_l(r,r) *= (1.0+lam);
        }

        Eigen::Matrix<double,3,1> dx = Hess_l.colPivHouseholderQr().solve(grad);
        //Eigen::Matrix<double,3,1> dx = (Hess+lam*Eigen::MatrixXd::Identity(Hess.rows(), Hess.rows())).colPivHouseholderQr().solve(grad);

        // Check if error has gone down
        double cost = compute_error(state, feature, sigma_cam_states_, alpha+dx(0,0), beta+dx(1,0), rho+dx(2,0));

        // Debug print
        //cout << "run = " << runs << " | cost = " << dx.norm() << " | lamda = " << lam << " | depth = " << 1/rho << endl;

        // Check if converged
        if (cost <= cost_old && (cost_old-cost)/cost_old < min_dcost) {
            alpha += dx(0, 0);
            beta += dx(1, 0);
            rho += dx(2, 0);
            eps = 0;
            break;
        }

        // If cost is lowered, accept step
        // Else inflate lambda (try to make more stable)
        if (cost <= cost_old) {
            recompute = true;
            cost_old = cost;
            alpha += dx(0, 0);
            beta += dx(1, 0);
            rho += dx(2, 0);
            runs++;
            lam = lam/lam_mult;
            eps = dx.norm();
        } else {
            recompute = false;
            lam = lam*lam_mult;
            continue;
        }
    }

    // Revert to standard, and set to all
    p_FinA(0) = alpha/rho;
    p_FinA(1) = beta/rho;
    p_FinA(2) = 1/rho;

    // Get tangent plane to x_hat
    Eigen::HouseholderQR<Eigen::MatrixXd> qr(p_FinA);
    Eigen::MatrixXd Q = qr.householderQ();

    // Max baseline we have between poses
    double base_line_max = 0.0;

    // Check maximum baseline
    // Loop through each camera for this feature
    auto first = state->_clones_IMU.begin();
    for (auto const& pair : feature.timestamps) {
        // Loop through the other clones to see what the max baseline is
        for (size_t m = 0; m < feature.timestamps[pair.first].size(); m++) {

            auto search = state->_clones_IMU.find(feature.timestamps[pair.first].at(m));
            int index = distance(first, search);

            // Get current IMU clone state
            Eigen::Matrix<double, 4, 1> q_sigma;
            q_sigma = sigma_cam_states_.block(0,7*index,1,4).transpose();
            R_GtoIi = ov_core::quat_2_Rot(q_sigma);
            p_IiinG = sigma_cam_states_.block(0,4+7*index ,1,3).transpose();

            // Calculate Cam clone in the global
            Eigen::Matrix<double,3,3> R_GtoCi =  R_ItoC * R_GtoIi;
            Eigen::Matrix<double,3,1> p_CiinG =  p_IiinG - R_GtoCi.transpose()*p_IinC;

            // Convert current position relative to anchor
            Eigen::Matrix<double,3,1> p_CiinA = R_GtoA*(p_CiinG-p_AinG);
            // Dot product camera pose and nullspace
            double base_line = ((Q.block(0,1,3,2)).transpose() * p_CiinA).norm();
            if (base_line > base_line_max) base_line_max = base_line;
        }
    }

    // Check if this feature is bad or not
    // 1. If the feature is too close
    // 2. If the feature is invalid
    // 3. If the baseline ratio is large
    if(p_FinA(2) < min_dist
        || p_FinA(2) > max_dist
        || (p_FinA.norm() / base_line_max) > max_baseline
        || std::isnan(p_FinA.norm())) {
        return false;
    }

    // Finally get position in global frame
    p_FinG = R_GtoA.transpose()*p_FinA + p_AinG;
    return true;
}

bool UpdaterHelper::single_triangulation_and_gaussnewton(std::shared_ptr<State> state,
                                        UpdaterHelperFeature &feature,
                                        Eigen::MatrixXd &sigma_cam_states_,
                                        Eigen::Vector3d &p_FinG){
    bool refine_features = true;
    
    // Total number of measurements
    // Also set the first measurement to be the anchor frame
    int total_meas = 0;
    size_t anchor_most_meas = 0;
    size_t most_meas = 0;
    for (auto const& pair : feature.timestamps) {
        total_meas += (int)pair.second.size();
        if(pair.second.size() > most_meas) {
            anchor_most_meas = pair.first;
            most_meas = pair.second.size();
        }
    }

    // Our linear system matrices
    Eigen::Matrix3d A = Eigen::Matrix3d::Zero();
    Eigen::Vector3d b = Eigen::Vector3d::Zero();

    std::shared_ptr<PoseJPL> calibration = state->_calib_IMUtoCAM.at(0); ////////////// Camera <-> IMU
    Eigen::Matrix<double,3,3> R_ItoC = calibration->Rot();   /// this is pose between imu and camera
    Eigen::Matrix<double,3,1> p_IinC = calibration->pos();

    // Get the position of the anchor pose
    Eigen::Matrix<double, 4, 1> q_anchor; 
    q_anchor = sigma_cam_states_.block(0,0,1,4).transpose();
    Eigen::Matrix<double,3,3> R_GtoIi = ov_core::quat_2_Rot(q_anchor);
    Eigen::Matrix<double,3,1> p_IiinG = sigma_cam_states_.block(0,4,1,3).transpose();
    Eigen::Matrix<double,3,3> R_GtoA =  R_ItoC * R_GtoIi;
    Eigen::Matrix<double,3,1> p_AinG =  p_IiinG - R_GtoA.transpose()*p_IinC;

    auto first = state->_clones_IMU.begin();
    // Loop through each camera for this feature
    for (auto const& pair : feature.timestamps) {
        
        // Add CAM_I features
        for (size_t m = 0; m < feature.timestamps[pair.first].size(); m++) {

            auto search = state->_clones_IMU.find(feature.timestamps[pair.first].at(m));
            int index = distance(first, search);

            // Get current IMU clone state
            Eigen::Matrix<double, 4, 1> q_sigma;
            q_sigma = sigma_cam_states_.block(0,7*index,1,4).transpose();
            R_GtoIi = ov_core::quat_2_Rot(q_sigma);
            p_IiinG = sigma_cam_states_.block(0,4+7*index ,1,3).transpose();

            // Calculate Cam clone in the global
            Eigen::Matrix<double,3,3> R_GtoCi =  R_ItoC * R_GtoIi;
            Eigen::Matrix<double,3,1> p_CiinG =  p_IiinG - R_GtoCi.transpose()*p_IinC;

            // Convert current position relative to anchor
            Eigen::Matrix<double,3,3> R_AtoCi;
            R_AtoCi.noalias() = R_GtoCi*R_GtoA.transpose();
            Eigen::Matrix<double,3,1> p_CiinA;
            p_CiinA.noalias() = R_GtoA*(p_CiinG-p_AinG);

            // Get the UV coordinate normal
            Eigen::Matrix<double, 3, 1> b_i;
            b_i << (double)feature.uvs_norm[pair.first].at(m)(0), (double)feature.uvs_norm[pair.first].at(m)(1), 1;
            b_i = R_AtoCi.transpose() * b_i;
            b_i = b_i / b_i.norm();
            Eigen::Matrix3d Bperp = skew_x(b_i);

            // Append to our linear system
            Eigen::Matrix3d Ai = Bperp.transpose() * Bperp;
            A += Ai;
            b += Ai * p_CiinA;

        }
    }

    // Solve the linear system
    Eigen::MatrixXd p_f = A.colPivHouseholderQr().solve(b);

    // Check A and p_f
    Eigen::JacobiSVD<Eigen::MatrixXd> svd(A, Eigen::ComputeThinU | Eigen::ComputeThinV);
    Eigen::MatrixXd singularValues;
    singularValues.resize(svd.singularValues().rows(), 1);
    singularValues = svd.singularValues();
    double condA = singularValues(0, 0) / singularValues(singularValues.rows() - 1, 0);

    // If we have a bad condition number, or it is too close
    // Then set the flag for bad (i.e. set z-axis to nan)

    // if (std::abs(condA) > _options.max_cond_number || p_f(2,0) < _options.min_dist || p_f(2,0) > _options.max_dist || std::isnan(p_f.norm())) {
    if (std::abs(condA) > 10000 || p_f(2,0) < 0.10 || p_f(2,0) > 60 || std::isnan(p_f.norm())) {
        return false;
    }

    // Store it in our feature object
    Eigen::Vector3d p_FinA = p_f;
    p_FinG = R_GtoA.transpose()*p_FinA + p_AinG;

    // Refine feature
    bool success_refine = true;
    if(refine_features){
        success_refine = single_gaussnewton(state, feature, sigma_cam_states_, p_FinA, p_FinG);
    }

    return success_refine;
}

int UpdaterHelper::pass_sigma_points(std::shared_ptr<State> state,
                                     UpdaterHelperFeature &feature,
                                     Eigen::MatrixXd &sigma_points_,
                                     Eigen::MatrixXd &P_zz,
                                     Eigen::MatrixXd &z_sigmas,
                                     Eigen::VectorXd &z_measured,
                                     Eigen::VectorXd &res,
                                     Eigen::MatrixXd &R){
    
    // std::cout << "sigma_points_ = \n" << sigma_points_ << std::endl;
    // std::cin.get();
    int n = sigma_points_.rows()/2;
    int n_state = sigma_points_.cols();
    int n_cams = (n-15)/6;//(n_state-16)/7;
    // std::cout << "n_cams = " << n_cams << std::endl;
    int problem_checker = 0;
    
    Eigen::MatrixXd sigma_cam_states_ = sigma_points_.block(0 , 16, 2*n , n_cams*7);
    // std::cout << "sigma_points_ = " << sigma_points_.rows() << " X " << sigma_points_.cols() << std::endl;
    // std::cout << "sigma_cam_states_ = " << sigma_cam_states_.rows() << " X " << sigma_cam_states_.cols() << std::endl;
    // std::cout << "sigma_points_ = " << sigma_points_.row(149) << std::endl;
    // std::cout << "sigma_cam_states_ = " << sigma_cam_states_.row(76) << std::endl;
    // std::cin.get();
    Eigen::Vector3d p_FinG = feature.p_FinG;
    // std::cout << "p_FinG = " << p_FinG << std::endl;
    std::list<int> storing_index_of_max;
    
    Eigen::Matrix<double,2,1> maxOfUV;
    maxOfUV << 0,0;

    for (auto const& pair : feature.timestamps) {
        int n_cams_active = feature.timestamps[pair.first].size();
        // std::cout << "n_cams_active = " << n_cams_active << std::endl;
        std::shared_ptr<Vec> distortion = state->_cam_intrinsics.at(pair.first);
        std::shared_ptr<PoseJPL> calibration = state->_calib_IMUtoCAM.at(pair.first); ////////////// Camera <-> IMU
        
        Eigen::Matrix<double,3,3> R_ItoC = calibration->Rot();   /// this is pose between imu and camera
        Eigen::Matrix<double,3,1> p_IinC = calibration->pos();
        Eigen::Matrix<double,8,1> cam_d = distortion->value();
        // std::cout << "R_ItoC = " << R_ItoC << std::endl;
        // std::cout << "p_IinC = " << p_IinC.transpose() << std::endl;
        // std::cout << "cam_d = " << cam_d.transpose() << std::endl;

        // Two for each camera pose seeing the feature
        Eigen::VectorXd z_i = Eigen::VectorXd::Zero(2 * n_cams_active);
        Eigen::MatrixXd z_stacked(2*n,2*n_cams_active);
        z_stacked = Eigen::MatrixXd::Zero(2*n,2*n_cams_active);

        // std::cout << "sigma_points_ 10x10 \n" << sigma_points_.block(0,0,10,10) <<std::endl;

        bool perform_check = true;
        // Passing cubature point through the measurement model
        auto first = state->_clones_IMU.begin();    
        // std::cout << " start passing sigma points through model" << 2*n << std::endl;

        for (int y=0; y< 2*n ; y++){ // for each sigma point
        
            bool success = true;
            Eigen::MatrixXd sig_imu_clone;
            sig_imu_clone = sigma_cam_states_.row(y);
            success = single_triangulation_and_gaussnewton(state, feature, sig_imu_clone, p_FinG);
            // if(abs(p_FinG(0))>5 or abs(p_FinG(1))>5 or abs(p_FinG(2))>5){
            //     std::cout << "p_FinG = \n" << p_FinG << std::endl;
            //     std::cin.get();
            // }

            int iter = 0;
            for (size_t m = 0; m < feature.timestamps[pair.first].size(); m++) { //for each IMU clone
                auto search = state->_clones_IMU.find(feature.timestamps[pair.first].at(m));
                int index = distance(first, search);
                // std::cout << " index = \n" << index << std::endl;
                // std::cin.get();

                // auto clone_access = state->_clones_IMU.at(feature.timestamps[pair.first].at(m));
                // std::cout << " imu clone_id =  " << clone_access->id() <<std::endl;
                // std::cout << " index =  " << index*6 + 15 <<std::endl;
                // std::cin.get();


                // Get current IMU clone state
                Eigen::Matrix<double, 4, 1> q_sigma;
                q_sigma = sigma_cam_states_.block(y,7*index,1,4).transpose();
                Eigen::Matrix<double,3,3> R_GtoIi = ov_core::quat_2_Rot(q_sigma);
                Eigen::Matrix<double,3,1> p_IiinG = sigma_cam_states_.block(y,4+7*index ,1,3).transpose();

                // Get current feature in the IMU
                Eigen::Matrix<double,3,1> p_FinIi = R_GtoIi*(p_FinG-p_IiinG);

                // Project the current feature into the current frame of reference
                Eigen::Matrix<double,3,1> p_FinCi = R_ItoC*p_FinIi+p_IinC;    //// MSCKF line 1102
                Eigen::Matrix<double,2,1> uv_norm;
                uv_norm << p_FinCi(0)/p_FinCi(2),p_FinCi(1)/p_FinCi(2); ///////z_hat_i_(j)
                // if((y==76 and m==8) or (y==77 and m==8) or (y==1 and m==8)){

                //     std::cout << " ==================================== " << std::endl;
                //     std::cout << "y = " << y << std::endl;
                //     std::cout << "p_IiinG = " << p_IiinG << std::endl;
                //     std::cout << "R_GtoIi = " << R_GtoIi << std::endl;
                //     std::cout << "p_FinIi = " << p_FinIi << std::endl;
                //     std::cout << "p_FinCi = " << p_FinCi << std::endl;
                //     std::cout << "p_FinG = " << p_FinG << std::endl;
                //     std::cin.get(); 
                // }
                // if(m==8){

                //     std::cout << " ==================================== " << std::endl;
                //     std::cout << "y = " << y << std::endl;
                //     std::cout << "p_IiinG = " << p_IiinG << std::endl;
                //     std::cin.get(); 
                // }

                // Distort the normalized coordinates (false=radtan, true=fisheye)
                Eigen::Matrix<double,2,1> uv_dist;

                // Calculate distortion uv and jacobian
                if(state->_cam_intrinsics_model.at(pair.first)) {
                    // std::cout << "fisheye \n" << std::endl;

                    // Calculate distorted coordinates for fisheye
                    double r = std::sqrt(uv_norm(0)*uv_norm(0)+uv_norm(1)*uv_norm(1));
                    double theta = std::atan(r);
                    double theta_d = theta+cam_d(4)*std::pow(theta,3)+cam_d(5)*std::pow(theta,5)+cam_d(6)*std::pow(theta,7)+cam_d(7)*std::pow(theta,9);

                    // Handle when r is small (meaning our xy is near the camera center)
                    double inv_r = (r > 1e-8)? 1.0/r : 1.0;
                    double cdist = (r > 1e-8)? theta_d * inv_r : 1.0;

                    // Calculate distorted coordinates for fisheye
                    double x1 = uv_norm(0)*cdist;
                    double y1 = uv_norm(1)*cdist;
                    uv_dist(0) = cam_d(0)*x1 + cam_d(2);
                    uv_dist(1) = cam_d(1)*y1 + cam_d(3);

                } else {
                    // std::cout << "radial \n" << std::endl;

                    // Calculate distorted coordinates for radial
                    double r = std::sqrt(uv_norm(0)*uv_norm(0)+uv_norm(1)*uv_norm(1));
                    double r_2 = r*r;
                    double r_4 = r_2*r_2;
                    double x1 = uv_norm(0)*(1+cam_d(4)*r_2+cam_d(5)*r_4)+2*cam_d(6)*uv_norm(0)*uv_norm(1)+cam_d(7)*(r_2+2*uv_norm(0)*uv_norm(0));
                    double y1 = uv_norm(1)*(1+cam_d(4)*r_2+cam_d(5)*r_4)+cam_d(6)*(r_2+2*uv_norm(1)*uv_norm(1))+2*cam_d(7)*uv_norm(0)*uv_norm(1);
                    uv_dist(0) = cam_d(0)*x1 + cam_d(2);
                    uv_dist(1) = cam_d(1)*y1 + cam_d(3);
                }
                
                if(perform_check == true){
                    if(z_i.norm()>0 and 1==1){
                        if(abs(uv_dist(0)) > abs(z_i.maxCoeff())+300 or abs(uv_dist(1)) > abs(z_i.maxCoeff())+300){
                            // dont do update
                            storing_index_of_max.push_back(iter);
                            // storing zero if value is large measurement
                            uv_dist(0) = 0;
                            uv_dist(1) = 0;
                            z_i.template segment<2>(2 * iter) = uv_dist;
                        }else{
                            z_i.template segment<2>(2 * iter) = uv_dist;  //// Stacking z_i elements
                        }
                    }else if (uv_dist.norm()>1500){
                        storing_index_of_max.push_back(iter);
                        // storing zero if value is large measurement
                        uv_dist(0) = 0;
                        uv_dist(1) = 0;
                        z_i.template segment<2>(2 * iter) = uv_dist;
                        
                    } else{
                        z_i.template segment<2>(2 * iter) = uv_dist;  //// Stacking z_i elements
                    }

                    if(maxOfUV.squaredNorm() < uv_dist.squaredNorm()){
                        maxOfUV = uv_dist;
                    }
                }else{

                    z_i.template segment<2>(2 * iter) = uv_dist;  //// Stacking z_i elements
                }
                
                iter++;

            }

            if(perform_check == true){
                if( storing_index_of_max.size() >= 1){
                    std::cout << "num of cams total = " << iter << std::endl;
                    std::cout << "num of cams wrong = " << storing_index_of_max.size()<< std::endl;
                    // std::cin.get();
                    problem_checker += 1;
                }
                
                for (int n : storing_index_of_max) { 
                    z_i.template segment<2>(2*n) = maxOfUV;
                }

                // Empty list
                while (!storing_index_of_max.empty())
                {
                    storing_index_of_max.pop_front();
                }
                maxOfUV << 0,0;
            }



            z_stacked.row(y) = z_i.transpose();
    
        }
        std::cout << " " << std::endl;

        // Change wrong measurement to average
        // double avg=0;
        // std::list<int> storing_index_of_maxs;
        // for (size_t y = 0; y < feature.timestamps[pair.first].size(); y++) { //for each IMU clone
        //     for (int i=0; i< 2 ; i++){
        //         for (int x=0; x< 2*n ; x++){ // for each sigma point
        //             if(z_stacked(n,2*y+i) == 0 ){
        //                 // if val = 0 store index
        //                 storing_index_of_maxs.push_back(x);
        //             }else{ // else put in average
        //                 if(x==0){
        //                     avg = z_stacked(n,2*y+i);
        //                 }else{
        //                     avg = ( avg + z_stacked(n,2*y+i) ) / 2;
        //                 }
        //             }
        //         }

        //         // put average whereever there is a 0
        //         for (int n : storing_index_of_maxs) { 
        //             z_stacked(n,2*y+i) = avg;
        //         }
        //         avg = 0;

        //         // Empty list
        //         while (!storing_index_of_maxs.empty())
        //         {
        //             storing_index_of_maxs.pop_front();
        //         }

        //     }
            
        // }
        // double max=0;
        // for (size_t y = 0; y < feature.timestamps[pair.first].size(); y++) { //for each IMU clone
        //     for (int i=0; i< 2 ; i++){
        //         for (int x=0; x< 2*n ; x++){ // for each sigma point
        //             if(z_stacked(n,2*y+i) == 0 ){
        //                 // if val = 0 store index
        //                 storing_index_of_maxs.push_back(x);
        //             }else{ // else put in average
        //                 if(z_stacked(n,2*y+i) > max){
        //                     max = z_stacked(n,2*y+i) ;
        //                 }
        //             }
        //         }

        //         // put average whereever there is a 0
        //         for (int n : storing_index_of_maxs) { 
        //             z_stacked(n,2*y+i) = max;
        //         }
        //         max = 0;

        //         // Empty list
        //         while (!storing_index_of_maxs.empty())
        //         {
        //             storing_index_of_maxs.pop_front();
        //         }

        //     }
        // }

        // 
        // std::cout << " z_stacked.row(76) = " << z_stacked.row(76) << std::endl;
        // std::cout << " sigma_cam_states_.row(75) before = " << sigma_cam_states_.row(75) << std::endl;
        // std::cout << " sigma_cam_states_.row(76) = " << sigma_cam_states_.row(76) << std::endl;
        // std::cout << " sigma_cam_states_.row(76) after = " << sigma_cam_states_.row(77) << std::endl;
        // std::cout << "z_stacked.colwise().sum() / (2*n) = " << z_stacked.colwise().sum() / (2*n) << std::endl;



        // Observations vector
        Eigen::VectorXd z = Eigen::VectorXd::Constant(2 * n_cams_active,
                                                std::numeric_limits<double>::quiet_NaN());

        for (size_t m = 0; m < feature.timestamps[pair.first].size(); m++) { //for each IMU clone
            Eigen::Matrix<double,2,1> uv_m;   
            uv_m << (double)feature.uvs[pair.first].at(m)(0), (double)feature.uvs[pair.first].at(m)(1);
            z.template segment<2>(2*m) = uv_m;

        }

        z_sigmas = z_stacked;
        z_measured = z;

        // std::cout << "z = \n" << z << std::endl;
        // std::cin.get();

        Eigen::MatrixXd z_hat = z_stacked.colwise().sum() / (2*n); // zhat = [ . . .]

        // std::cout << "z_stacked cols= \n" << z_stacked.row(n+5) << std::endl;
        // std::cout << " z_hat = \n" << z_hat << std::endl;
        // std::cout << " z = \n" << z << std::endl;
        // std::cin.get();

        P_zz = Eigen::MatrixXd::Zero(2*n_cams_active,2*n_cams_active);

        for (int i = 0 ; i<2*n ; i++){
            P_zz.noalias() += z_stacked.row(i).transpose() * z_stacked.row(i);
            Eigen::MatrixXd z_h_temp = z_hat.transpose() * z_hat;  // zhat.T * zhat
            P_zz -= z_h_temp; // changed , moved it out of the loop
        }
        P_zz = P_zz/(2*n);
        P_zz = P_zz + R; 

        // Enforce symmetry
        Eigen::MatrixXd Pzz_corrected_transpose = P_zz.transpose();
        Eigen::MatrixXd Pzz_corrected;
        Pzz_corrected = P_zz + Pzz_corrected_transpose;
        Pzz_corrected /= 2;
        P_zz = Pzz_corrected;

        Eigen::MatrixXd z_h_trans = z_hat.transpose();
        res = z - z_h_trans; // column vector

        /////////////////////////////////////////////////////////////////////////////////////
        // std::cout << " Pzz_corrected = \n" << Pzz_corrected.row(1) << std::endl;
        // std::cin.get();
    
        //compute cross covariance
        // // Compute the cross-covariance of the state and measurements
        // P_xz = Eigen::MatrixXd::Zero(n,2*n_cams_active);

        // Eigen::MatrixXd dx (1, n);
        // dx = Eigen::MatrixXd::Zero(1,n);
        // Eigen::MatrixXd dx_trans (n, 1);
        // Eigen::MatrixXd dz (1, n_cams_active*2);

        // for(int i=0; i < 2*n; i++)
        // {
        //     // implemented this => dx = sigma_points_.row(i) - x_f;  
        //     Eigen::Matrix<double, 4, 1> IMU_sigma_q = sigma_points_.block(i,0,1,4).transpose();
        //     Eigen::Matrix<double, 4, 1> IMU_q_inv = ov_core::Inv(state->_imu->quat());
        //     Eigen::Matrix<double, 4, 1> del_q = ov_core::quat_multiply_my_my(IMU_sigma_q, IMU_q_inv);

        //     dx.block(0,0,1,3) = del_q.block(0,0,3,1).transpose()/del_q(3,0) * 2; // check            
        //     dx.block(0,3,1,3) = sigma_points_.block(i,4,1,3) - state->_imu->pos().transpose();
        //     dx.block(0,6,1,3) = sigma_points_.block(i,7,1,3) - state->_imu->vel().transpose(); 
        //     dx.block(0,9,1,3) = sigma_points_.block(i,10,1,3) - state->_imu->bias_g().transpose(); 
        //     dx.block(0,12,1,3) = sigma_points_.block(i,13,1,3) - state->_imu->bias_a().transpose(); 

        //     int ci = 0;
        //     for (const auto& [key,value]:state->_clones_IMU){ //for each IMU clone
        //         std::shared_ptr<PoseJPL> clone_Ii = value;
        //         Eigen::Matrix<double, 4, 1> IMU_clone_sigma_q = sigma_points_.block(i,16+ci*7,1,4).transpose();
        //         Eigen::Matrix<double, 4, 1> IMU_clone_q_inv = ov_core::Inv(clone_Ii->quat());
        //         Eigen::Matrix<double, 4, 1> del_q = ov_core::quat_multiply_my_my(IMU_clone_sigma_q, IMU_clone_q_inv);

        //         dx.block(0,15+6*ci,1,3) = del_q.block(0,0,3,1).transpose()/del_q(3,0) * 2; // check    
        //         dx.block(0,18+6*ci,1,3) = sigma_points_.block(i,20+ci*7,1,3) - clone_Ii->pos().transpose();
        //         ci += 1;
        //     }

        //     dz = z_stacked.row(i) - z_hat;
        //     dx_trans = dx.transpose();
        //     P_xz.noalias() += dx_trans * dz; // state_dim X 2*cam_meas_dim
            
        // }
        // P_xz /= (2*n); 


        // std::cout << " z = " << z << std::endl;
        // std::cout << " z_h_trans = " << z_h_trans << std::endl;
        // std::cin.get();
        // std::cout << " res = " << res << std::endl;
        // std::cin.get();


        // std::cout << " res size = " <<  res.rows() << " X " << res.cols() << std::endl;
        // std::cin.get();

        // Eigen::VectorXd diags_xz = P_xz.diagonal();

        // for(int i=0; i<diags_xz.rows(); i++) {
        //     if(diags_xz(i) < 0.0) {
        //         printf(RED "P_xz - diagonal at %d is %.2f\n" RESET,i,diags_xz(i));
        //     }
        // }
        // std::cout << " P_xz(2,2) = " <<  P_xz(2,2) << std::endl;


        // assert(false);
    }

    return problem_checker;
}

void UpdaterHelper::calculate_cubature_points(std::shared_ptr<State> state, Eigen::MatrixXd P, Eigen::MatrixXd &sigma_points_){


    int n = P.rows();  // Dimension of state vector
    int k = 2*n;       // <number of cubature points
    int n_states = n+1+(n-15)/6;
    // int n_camera = n - 15; //Dimension of camera vector
    sigma_points_.resize(k,n_states);
    sigma_points_ = Eigen::MatrixXd::Zero(k,n_states);

    Eigen::LLT<Eigen::MatrixXd> lltOfA(P);
    Eigen::MatrixXd P_chol = lltOfA.matrixU(); // Cholesky decomposition
    Eigen::MatrixXd U; 
    U = P_chol* sqrt(n);

    // std::cout << " P_chol 10x10 = \n" << P_chol.block(0,0,10,10) << std::endl;
    // std::cin.get();

    
    // std::shared_ptr<Vec> distortion = state->_cam_intrinsics.at(0);
    // std::shared_ptr<PoseJPL> calibration = state->_calib_IMUtoCAM.at(0); ////////////// Camera <-> IMU
    
    // Eigen::Matrix<double,3,3> R_ItoC = calibration->Rot();   /// this is pose between imu and camera
    // Eigen::Matrix<double,3,1> p_IinC = calibration->pos();
    // Eigen::Matrix<double,8,1> cam_d = distortion->value();    

    // Sigma matrix
    Eigen::Matrix<double, 4, 1> dq;

    for (int i=0 ; i<n ; i++){
        dq << .5 * U.block(i,0,1,3).transpose(), 1.0;
        dq = ov_core::quatnorm(dq);
        sigma_points_.block(i,0,1,4) = ov_core::quat_multiply(dq, state->_imu->quat()).transpose();

        // if(i==2){
        //     Eigen::Quaterniond quat;
        //     quat.vec() = .5 * U.block(i,0,1,3).transpose();
        //     quat.w() = 1;
        //     quat.normalize();
        //     Eigen::Quaterniond imu_quat;
        //     imu_quat.vec() = state->_imu->quat().block(0,0,3,1);
        //     imu_quat.w() = state->_imu->quat()(3,0);

        //     // std::cout << " U.block(i,0,1,3) = \n" << U.block(i,0,1,3) << std::endl;
        //     std::cout << " dq = \n" << dq << std::endl;
        //     std::cout << " state->_imu->quat() = \n" << state->_imu->quat() << std::endl;
        //     std::cout << " sigma_points_.block(i,0,1,4) = \n" << ov_core::quat_multiply_my(dq, state->_imu->quat()) << std::endl;
        //     std::cout << " quat.vec() = \n" << quat.vec() << std::endl;
        //     std::cout << " quat.w() = \n" << quat.w() << std::endl;
        //     std::cout << " imu_quat.vec() = \n" << imu_quat.vec() << std::endl;
        //     std::cout << " imu_quat.w() = \n" << imu_quat.w() << std::endl;
        //     std::cout << "(quat*imu_quat).vec() = \n" << (quat*imu_quat).vec() << std::endl;
        //     std::cout << "(quat*imu_quat).w() = \n" << (quat*imu_quat).w() << std::endl;

        // }
        sigma_points_.block(i,4,1,3) = state->_imu->pos().transpose() + U.block(i,3,1,3);
        sigma_points_.block(i,7,1,3) = state->_imu->vel().transpose() + U.block(i,6,1,3);
        sigma_points_.block(i,10,1,3) = state->_imu->bias_g().transpose() + U.block(i,9,1,3);
        sigma_points_.block(i,13,1,3) = state->_imu->bias_a().transpose() + U.block(i,12,1,3);
        
        dq << .5 * -U.block(i,0,1,3).transpose(), 1.0;
        dq = ov_core::quatnorm(dq);
        sigma_points_.block(n+i,0,1,4) = ov_core::quat_multiply(dq, state->_imu->quat()).transpose();
        sigma_points_.block(n+i,4,1,3) = state->_imu->pos().transpose() - U.block(i,3,1,3);
        sigma_points_.block(n+i,7,1,3) = state->_imu->vel().transpose() - U.block(i,6,1,3);
        sigma_points_.block(n+i,10,1,3) = state->_imu->bias_g().transpose() - U.block(i,9,1,3);
        sigma_points_.block(n+i,13,1,3) = state->_imu->bias_a().transpose() - U.block(i,12,1,3);

        //because we are taking selected camaera states
        int ci = 0;
        for (const auto& [key,value]:state->_clones_IMU){
            std::shared_ptr<PoseJPL> clone_Ii = value;

            dq << .5 * U.block(i,15+ci*6,1,3).transpose(), 1.0;
            dq = ov_core::quatnorm(dq);
            sigma_points_.block(i,16+ci*7,1,4) = ov_core::quat_multiply(dq, clone_Ii->quat()).transpose();
            sigma_points_.block(i,20+ci*7, 1,3) = clone_Ii->pos().transpose() + U.block(i,18+ci*6, 1,3) ;

            dq << .5 * -U.block(i,15+ci*6,1,3).transpose(), 1.0;
            dq = ov_core::quatnorm(dq);
            sigma_points_.block(n+i,16+ci*7,1,4) = ov_core::quat_multiply(dq, clone_Ii->quat()).transpose();
            sigma_points_.block(n+i,20+ci*7, 1,3) = clone_Ii->pos().transpose() - U.block(i,18+ci*6, 1,3) ;

            ci += 1;
        }
        // std::cout << "sigma_points_ = \n" << sigma_points_.row(n) << std::endl; 
        // std::cin.get();
    }
    // std::cout << "sigma_points_ = \n" << sigma_points_.row(0) << std::endl;
    // std::cout << " sigma_points_ 10x10 = \n" << sigma_points_.block(0,0,10,10) << std::endl;
    // std::cin.get();
}

void UpdaterHelper::nullspace_project_inplace(Eigen::MatrixXd &H_f, Eigen::MatrixXd &H_x, Eigen::VectorXd &res) {

    // Apply the left nullspace of H_f to all variables
    // Based on "Matrix Computations 4th Edition by Golub and Van Loan"
    // See page 252, Algorithm 5.2.4 for how these two loops work
    // They use "matlab" index notation, thus we need to subtract 1 from all index
    Eigen::JacobiRotation<double> tempHo_GR;
    for (int n = 0; n < H_f.cols(); ++n) {
        for (int m = (int) H_f.rows() - 1; m > n; m--) {
            // Givens matrix G
            tempHo_GR.makeGivens(H_f(m - 1, n), H_f(m, n));
            // Multiply G to the corresponding lines (m-1,m) in each matrix
            // Note: we only apply G to the nonzero cols [n:Ho.cols()-n-1], while
            //       it is equivalent to applying G to the entire cols [0:Ho.cols()-1].
            (H_f.block(m - 1, n, 2, H_f.cols() - n)).applyOnTheLeft(0, 1, tempHo_GR.adjoint());
            (H_x.block(m - 1, 0, 2, H_x.cols())).applyOnTheLeft(0, 1, tempHo_GR.adjoint());
            (res.block(m - 1, 0, 2, 1)).applyOnTheLeft(0, 1, tempHo_GR.adjoint());
        }
    }

    // The H_f jacobian max rank is 3 if it is a 3d position, thus size of the left nullspace is Hf.rows()-3
    // NOTE: need to eigen3 eval here since this experiences aliasing!
    //H_f = H_f.block(H_f.cols(),0,H_f.rows()-H_f.cols(),H_f.cols()).eval();
    H_x = H_x.block(H_f.cols(),0,H_x.rows()-H_f.cols(),H_x.cols()).eval();
    res = res.block(H_f.cols(),0,res.rows()-H_f.cols(),res.cols()).eval();

    // Sanity check
    assert(H_x.rows()==res.rows());
}




void UpdaterHelper::measurement_compress_inplace(Eigen::MatrixXd &H_x, Eigen::VectorXd &res) {

    // Return if H_x is a fat matrix (there is no need to compress in this case)
    if(H_x.rows() <= H_x.cols())
        return;

    // Do measurement compression through givens rotations
    // Based on "Matrix Computations 4th Edition by Golub and Van Loan"
    // See page 252, Algorithm 5.2.4 for how these two loops work
    // They use "matlab" index notation, thus we need to subtract 1 from all index
    Eigen::JacobiRotation<double> tempHo_GR;
    for (int n=0; n<H_x.cols(); n++) {
        for (int m=(int)H_x.rows()-1; m>n; m--) {
            // Givens matrix G
            tempHo_GR.makeGivens(H_x(m-1,n), H_x(m,n));
            // Multiply G to the corresponding lines (m-1,m) in each matrix
            // Note: we only apply G to the nonzero cols [n:Ho.cols()-n-1], while
            //       it is equivalent to applying G to the entire cols [0:Ho.cols()-1].
            (H_x.block(m-1,n,2,H_x.cols()-n)).applyOnTheLeft(0,1,tempHo_GR.adjoint());
            (res.block(m-1,0,2,1)).applyOnTheLeft(0,1,tempHo_GR.adjoint());
        }
    }

    // If H is a fat matrix, then use the rows
    // Else it should be same size as our state
    int r = std::min(H_x.rows(),H_x.cols());

    // Construct the smaller jacobian and residual after measurement compression
    assert(r<=H_x.rows());
    H_x.conservativeResize(r, H_x.cols());
    res.conservativeResize(r, res.cols());

}



