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
#include "UpdaterMSCKF.h"
#include <iostream> /////////////////////////
#include <fstream> ///////////////////
#include <string>


using namespace ov_core;
using namespace ov_msckf;





void UpdaterMSCKF::update(std::shared_ptr<State> state, std::vector<std::shared_ptr<Feature>>& feature_vec) {

    // Return if no features
    if(feature_vec.empty())
        return;

    // Start timing
    boost::posix_time::ptime rT0, rT1, rT2, rT3, rT4, rT5, rT6, rT7;
    rT0 =  boost::posix_time::microsec_clock::local_time();

    // 0. Get all timestamps our clones are at (and thus valid measurement times)
    std::vector<double> clonetimes;
    for(const auto& clone_imu : state->_clones_IMU) {
        clonetimes.emplace_back(clone_imu.first);
    }

    // 1. Clean all feature measurements and make sure they all have valid clone times
    auto it0 = feature_vec.begin();
    while(it0 != feature_vec.end()) {

        // Clean the feature
        (*it0)->clean_old_measurements(clonetimes);

        // Count how many measurements
        int ct_meas = 0;
        for(const auto &pair : (*it0)->timestamps) {
            ct_meas += (*it0)->timestamps[pair.first].size();
        }

        // Remove if we don't have enough
        if(ct_meas < 2) {
            (*it0)->to_delete = true;
            it0 = feature_vec.erase(it0);
        } else {
            it0++;
        }

    }
    rT1 =  boost::posix_time::microsec_clock::local_time();

    // 2. Create vector of cloned *CAMERA* poses at each of our clone timesteps
    std::unordered_map<size_t, std::unordered_map<double, FeatureInitializer::ClonePose>> clones_cam;
    for(const auto &clone_calib : state->_calib_IMUtoCAM) {

        // For this camera, create the vector of camera poses
        std::unordered_map<double, FeatureInitializer::ClonePose> clones_cami;
        for(const auto &clone_imu : state->_clones_IMU) {

            // Get current camera pose
            Eigen::Matrix<double,3,3> R_GtoCi = clone_calib.second->Rot()*clone_imu.second->Rot();
            Eigen::Matrix<double,3,1> p_CioinG = clone_imu.second->pos() - R_GtoCi.transpose()*clone_calib.second->pos();

            // Append to our map
            clones_cami.insert({clone_imu.first,FeatureInitializer::ClonePose(R_GtoCi,p_CioinG)});

        }

        // Append to our map
        clones_cam.insert({clone_calib.first,clones_cami});

    }

    // 3. Try to triangulate all MSCKF or new SLAM features that have measurements
    auto it1 = feature_vec.begin();
    while(it1 != feature_vec.end()) {

        // Triangulate the feature and remove if it fails
        bool success_tri = true;
        if(initializer_feat->config().triangulate_1d) {
            success_tri = initializer_feat->single_triangulation_1d(it1->get(), clones_cam);
        } else {
            success_tri = initializer_feat->single_triangulation(it1->get(), clones_cam);
        }

        // Gauss-newton refine the feature
        bool success_refine = true;
        if(initializer_feat->config().refine_features) {
            success_refine = initializer_feat->single_gaussnewton(it1->get(), clones_cam);
        }

        // Remove the feature if not a success
        if(!success_tri || !success_refine) {
            (*it1)->to_delete = true;
            it1 = feature_vec.erase(it1);
            continue;
        }
        it1++;

    }
    rT2 =  boost::posix_time::microsec_clock::local_time();

    // Calculate the max possible measurement size
    size_t max_meas_size = 0;
    for(size_t i=0; i<feature_vec.size(); i++) {
        for (const auto &pair : feature_vec.at(i)->timestamps) {
            max_meas_size += 2*feature_vec.at(i)->timestamps[pair.first].size();
        }
    }

    // Calculate max possible state size (i.e. the size of our covariance)
    // NOTE: that when we have the single inverse depth representations, those are only 1dof in size
    size_t max_hx_size = state->max_covariance_size();
    for(auto &landmark : state->_features_SLAM) {
        max_hx_size -= landmark.second->size();
    }


    // Large Jacobian and residual of *all* features for this update
    Eigen::VectorXd res_big = Eigen::VectorXd::Zero(max_meas_size);
    Eigen::MatrixXd Hx_big = Eigen::MatrixXd::Zero(max_meas_size, max_hx_size);
    std::unordered_map<std::shared_ptr<Type>,size_t> Hx_mapping;
    std::vector<std::shared_ptr<Type>> Hx_order_big;
    size_t ct_jacob = 0;
    size_t ct_meas = 0;
    size_t ct_meas_ckf = 0;

    // Calculate Cubature Points
    Eigen::MatrixXd sigma_points_;
    Eigen::MatrixXd P = StateHelper::get_full_covariance(state); //get covariance
    // std::cout << std::fixed;
    // std::cout << std::setprecision(20);
    // std::cout << " P det  = " << P.determinant() << std::endl;
    // std::cin.get();
    // std::cout << " P   = " << P << std::endl;
    // std::cin.get();
    UpdaterHelper::calculate_cubature_points(state, P, sigma_points_); // or we can calculates this outside the loop
    
    Eigen::VectorXd z_measured_big = Eigen::VectorXd::Zero(max_meas_size);
    Eigen::MatrixXd z_sigmas_big = Eigen::MatrixXd::Zero(sigma_points_.rows(), max_meas_size);
    // Eigen::MatrixXd Pxz_big = Eigen::MatrixXd::Zero(sigma_points_.rows()/2, max_meas_size);
    
    // 4. Compute linear system for each feature, nullspace project, and reject
    auto it2 = feature_vec.begin();
    // std::cout << " max_meas_size = " << max_meas_size <<std::endl;
    // std::cout << " feature_vec = " << feature_vec.size() <<std::endl;
    bool mode_ckf = true;
    // std::cout << "Imu vel = " << state->_imu->vel().norm() << std::endl;
    // std::cin.get();
    while(it2 != feature_vec.end()) {    ///////for each feature that cant be seen anymore //MSCKF line 451

        // Convert our feature into our current format
        UpdaterHelper::UpdaterHelperFeature feat;
        feat.featid = (*it2)->featid;
        feat.uvs = (*it2)->uvs;
        feat.uvs_norm = (*it2)->uvs_norm;
        feat.timestamps = (*it2)->timestamps;

        // If we are using single inverse depth, then it is equivalent to using the msckf inverse depth
        feat.feat_representation = state->_options.feat_rep_msckf;
        if(state->_options.feat_rep_msckf==LandmarkRepresentation::Representation::ANCHORED_INVERSE_DEPTH_SINGLE) {
            feat.feat_representation = LandmarkRepresentation::Representation::ANCHORED_MSCKF_INVERSE_DEPTH;
        }

        // Save the position and its fej value
        if(LandmarkRepresentation::is_relative_representation(feat.feat_representation)) {
            feat.anchor_cam_id = (*it2)->anchor_cam_id;
            feat.anchor_clone_timestamp = (*it2)->anchor_clone_timestamp;
            feat.p_FinA = (*it2)->p_FinA;
            feat.p_FinA_fej = (*it2)->p_FinA;  //// if here then p_f_g is calculated in get_feature_jacobian_full
        } else {
            feat.p_FinG = (*it2)->p_FinG; /////////// p_f_g
            feat.p_FinG_fej = (*it2)->p_FinG; /////////// p_f_g
        }

        // Our return values (feature jacobian, state jacobian, residual, and order of state jacobian)
        Eigen::MatrixXd H_f;
        Eigen::MatrixXd H_x;
        Eigen::VectorXd res;
        std::vector<std::shared_ptr<Type>> Hx_order;

        // Get the Jacobian for this feature
        UpdaterHelper::get_feature_jacobian_full(state, feat, H_f, H_x, res, Hx_order);  /// also calculates the residual and p_f_g
        // std::cout << " res size = " <<  res.rows() << " X " << res.cols() << std::endl;
        // std::cin.get();
            //CKF
        
        if (mode_ckf == true){
            // Eigen::MatrixXd P_xz;
            Eigen::MatrixXd z_sigmas;
            Eigen::VectorXd z_measured;
            Eigen::MatrixXd P_zz;  /// used for Chi2 distance check // not used elsewhere
            Eigen::VectorXd resedual_error;

            // Covarience of measurements
            int R_dimension = 2*feat.timestamps[0].size();
            Eigen::MatrixXd R = _options.sigma_pix_sq*Eigen::MatrixXd::Identity(R_dimension,R_dimension); 
            int problem = UpdaterHelper::pass_sigma_points(state, feat, sigma_points_, P_zz, z_sigmas, z_measured, resedual_error, R);
            // UpdaterHelper::calc_residual_CKF(state, feat, sigma_points_, P_xz, P_zz, resedual_error, R);
            /// Chi2 distance check CKF
            P_zz.diagonal() += _options.sigma_pix_sq*Eigen::VectorXd::Ones(P_zz.rows()); // MSCKF line 1494
            if ( 1==1 ){ // I dont think gating test is correct here
                double chi2_CKF = resedual_error.dot(P_zz.llt().solve(resedual_error)); // MSCKF line 1495 (gamma)
                // std::cout << "chi2_CKF = " << chi2_CKF << std::endl;
                // Get our threshold (we precompute up to 500 but handle the case that it is more)
                double chi2_check_CKF;
                if(resedual_error.rows() < 500) {
                    chi2_check_CKF = chi_squared_table[resedual_error.rows()];
                } else {
                    boost::math::chi_squared chi_squared_dist(resedual_error.rows());
                    chi2_check_CKF = boost::math::quantile(chi_squared_dist, 0.95);
                    printf(YELLOW "chi2_check_CKF over the residual limit - %d\n" RESET, (int)resedual_error.rows());
                }
                // Check if we should delete or not
                if(chi2_CKF > _options.chi2_multipler*chi2_check_CKF || problem>=5) {
                    (*it2)->to_delete = true;
                    it2 = feature_vec.erase(it2);
                    std::cout << " chi2_check_CKF failed " << std::endl;
                    //cout << "featid = " << feat.featid << endl;
                    //cout << "chi2 = " << chi2 << " > " << _options.chi2_multipler*chi2_check << endl;
                    //cout << "res = " << endl << res.transpose() << endl;
                    continue;
                }
            }

            // We are good!! Append to our large Pxx and Pzz matrix and resedual
            // Pzz_big.block(ct_meas_ckf,ct_meas_ckf,P_zz.rows(),P_zz.rows()) = P_zz;
            z_sigmas_big.block(0,ct_meas_ckf,z_sigmas.rows(),z_sigmas.cols()) = z_sigmas;
            z_measured_big.block(ct_meas_ckf,0,z_measured.rows(),1) = z_measured;    
            ct_meas_ckf += z_measured.rows();

            // std::cout << std::fixed;
            // std::cout << std::setprecision(1);
            // std::cout << " resedual_error.rows() = \n" << resedual_error.rows() << std::endl;
            // std::cout << " P_xz = \n" << P_xz << std::endl;
            // std::cout << " Pxz_big = \n" << Pxz_big << std::endl;
            // std::cin.get();

            // StateHelper::CKF_update(state, P_xz, P_zz, resedual_error);
        }
            //CKF

        // Nullspace project
        UpdaterHelper::nullspace_project_inplace(H_f, H_x, res);

        if (mode_ckf == false){

            /// Chi2 distance check                ////////////// gatingTest
            Eigen::MatrixXd P_marg = StateHelper::get_marginal_covariance(state, Hx_order); // this is the covariance
            Eigen::MatrixXd S = H_x*P_marg*H_x.transpose(); // MSCKF line 1492
            S.diagonal() += _options.sigma_pix_sq*Eigen::VectorXd::Ones(S.rows()); // MSCKF line 1494
            double chi2 = res.dot(S.llt().solve(res)); // MSCKF line 1495 (gamma)

            // Get our threshold (we precompute up to 500 but handle the case that it is more)
            double chi2_check;
            if(res.rows() < 500) {
                chi2_check = chi_squared_table[res.rows()];
            } else {
                boost::math::chi_squared chi_squared_dist(res.rows());
                chi2_check = boost::math::quantile(chi_squared_dist, 0.95);
                printf(YELLOW "chi2_check over the residual limit - %d\n" RESET, (int)res.rows());
            }


            // Check if we should delete or not
            if(chi2 > _options.chi2_multipler*chi2_check) {
                (*it2)->to_delete = true;
                it2 = feature_vec.erase(it2);
                std::cout << " chi2_check failed " << std::endl;
                //cout << "featid = " << feat.featid << endl;
                //cout << "chi2 = " << chi2 << " > " << _options.chi2_multipler*chi2_check << endl;
                //cout << "res = " << endl << res.transpose() << endl;
                continue;
            }
        }

        // We are good!!! Append to our large H vector
        size_t ct_hx = 0;
        for(const auto &var : Hx_order) {

            // Ensure that this variable is in our Jacobian
            if(Hx_mapping.find(var)==Hx_mapping.end()) {
                Hx_mapping.insert({var,ct_jacob});
                Hx_order_big.push_back(var);
                ct_jacob += var->size();
            }

            // Append to our large Jacobian
            Hx_big.block(ct_meas,Hx_mapping[var],H_x.rows(),var->size()) = H_x.block(0,ct_hx,H_x.rows(),var->size());
            ct_hx += var->size();

        }

        // Append our residual and move forward
        res_big.block(ct_meas,0,res.rows(),1) = res;       /////////// residual
        ct_meas += res.rows();
        it2++;

    }
    rT3 =  boost::posix_time::microsec_clock::local_time();

    // We have appended all features to our Hx_big, res_big
    // Delete it so we do not reuse information
    for (size_t f=0; f < feature_vec.size(); f++) {
        feature_vec[f]->to_delete = true;
    }

    // std::cin.get();
    if (mode_ckf == false){
        
        // Return if we don't have anything and resize our matrices
        if(ct_meas < 1) {
            return;
        }
        assert(ct_meas<=max_meas_size);
        assert(ct_jacob<=max_hx_size);
        res_big.conservativeResize(ct_meas,1);
        Hx_big.conservativeResize(ct_meas,ct_jacob);

        // 5. Perform measurement compression
        UpdaterHelper::measurement_compress_inplace(Hx_big, res_big);
        if(Hx_big.rows() < 1) {
            return;
        }
        rT4 =  boost::posix_time::microsec_clock::local_time();

        // Our noise is isotropic, so make it here after our compression
        // Will be used ((((((((((((((( Covarience of resedial pixels ))))))))))))))))))))))
        Eigen::MatrixXd R_big = _options.sigma_pix_sq*Eigen::MatrixXd::Identity(res_big.rows(),res_big.rows()); 

        // 6. With all good features update the state
        StateHelper::EKFUpdate(state, Hx_order_big, Hx_big, res_big, R_big);
        rT5 =  boost::posix_time::microsec_clock::local_time();

        // Text file
        std::ofstream myfile ("/home/zahid/example.txt", std::ios::app);
        if (myfile.is_open())
        {
            
          myfile << " Meas Covar \n";
          myfile << P(6,6) << "," << P(7,7) << "," << P(8,8) << "\n";

        //   myfile << " marginalize ^ \n";
        //   myfile << "\n";
          myfile.close();
            
        }
        else std::cout << "Unable to open file";

        // Text file
        std::ofstream myfile2 ("/home/zahid/example2.txt", std::ios::app);
        if (myfile2.is_open())
        {
            
          myfile2 << " State IMU vel \n";
          myfile2 << state->_imu->vel().x() << "," << state->_imu->vel().y() << "," << state->_imu->vel().z() << "\n";

        //   myfile << " marginalize ^ \n";
        //   myfile << "\n";
          myfile2.close();
            
        }
        else std::cout << "Unable to open file";

        std::cin.get();


    }else{

        if(ct_meas_ckf < 1) {
            return;
        }
        assert( (int)ct_meas_ckf <= (int)max_meas_size);
        z_sigmas_big.conservativeResize(sigma_points_.rows(),ct_meas_ckf);
        z_measured_big.conservativeResize(ct_meas_ckf,1);

        Eigen::MatrixXd R_big_ckf = _options.sigma_pix_sq*Eigen::MatrixXd::Identity(z_measured_big.rows(),z_measured_big.rows()); 
        // std::cout << " Propagation :: " << std::endl;
        // std::cout << "State IMU vel = " << state->_imu->vel().norm() << std::endl;
        StateHelper::CKF_update(state, sigma_points_, z_sigmas_big, z_measured_big, P, R_big_ckf);
        // std::cout << " Update :: " << std::endl;
        // std::cout << "State IMU vel = " << state->_imu->vel().norm() << std::endl;
        
        Eigen::MatrixXd P = StateHelper::get_full_covariance(state); //get covariance
        // std::cout << "State Covar = " << P(6,6) << ", " << P(7,7) << ", " << P(8,8) << std::endl; 

        // Text file
        std::ofstream myfile ("/home/zahid/example.txt", std::ios::app);
        if (myfile.is_open())
        {
            
          myfile << " Meas Covar \n";
          myfile << P(6,6) << "," << P(7,7) << "," << P(8,8) << "\n";

        //   myfile << " marginalize ^ \n";
        //   myfile << "\n";
          myfile.close();
            
        }
        else std::cout << "Unable to open file";

        // Text file
        std::ofstream myfile2 ("/home/zahid/example2.txt", std::ios::app);
        if (myfile2.is_open())
        {
            
          myfile2 << " State IMU vel \n";
          myfile2 << state->_imu->vel().x() << "," << state->_imu->vel().y() << "," << state->_imu->vel().z() << "\n";

        //   myfile << " marginalize ^ \n";
        //   myfile << "\n";
          myfile2.close();
            
        }
        else std::cout << "Unable to open file";

        std::cin.get();

    }

    // Debug print timing information
    //printf("[MSCKF-UP]: %.4f seconds to clean\n",(rT1-rT0).total_microseconds() * 1e-6);
    //printf("[MSCKF-UP]: %.4f seconds to triangulate\n",(rT2-rT1).total_microseconds() * 1e-6);
    //printf("[MSCKF-UP]: %.4f seconds create system (%d features)\n",(rT3-rT2).total_microseconds() * 1e-6, (int)feature_vec.size());
    //printf("[MSCKF-UP]: %.4f seconds compress system\n",(rT4-rT3).total_microseconds() * 1e-6);
    //printf("[MSCKF-UP]: %.4f seconds update state (%d size)\n",(rT5-rT4).total_microseconds() * 1e-6, (int)res_big.rows());
    //printf("[MSCKF-UP]: %.4f seconds total\n",(rT5-rT1).total_microseconds() * 1e-6);

}










