//
// Created by heinj on 23.01.21.
//

#include <iostream>
#include <boost/program_options.hpp>
#include <eigen3/Eigen/Dense>
#include <k4a/k4a.hpp>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <pcl/registration/icp.h>
#include <pcl/registration/correspondence_rejection_trimmed.h>
#include <pcl/visualization/cloud_viewer.h>
#include <pcl/visualization/keyboard_event.h>

#include "MultiPlayback.hpp"
#include "utils/io.hpp"
#include "utils/point_cloud.hpp"

namespace po = boost::program_options;

// Global pointers used for viewer callback
MultiPlayback *pPlayback = nullptr;
pcl::PointCloud<pcl::PointXYZRGB>::Ptr full_pc_ptr(new pcl::PointCloud<pcl::PointXYZRGB>);
pcl::PointCloud<pcl::PointXYZRGB>::Ptr combined_pc_ptr(new pcl::PointCloud<pcl::PointXYZRGB>);

po::variables_map parse_arguments(int argc, char **argv, bool print = true) {
    po::options_description generic_options("Generic options");
    generic_options.add_options()
            ("help", "print this help message")
            ("config", po::value<std::string>(), "config file path");
    po::options_description config_options("Configuration");
    config_options.add_options()
            ("main", po::value<std::string>()->required(), "main camera mkv recording")
            ("sub", po::value<std::string>()->required(), "subordinate camera mkv recording")
            ("extrinsics", po::value<std::string>()->required(),
             "extrinsic parameter file for the subordinate camera\\'s color sensor")
            ("labels", po::value<std::string>(), "ground truth labels file path")
            ("object", po::value<std::string>(), "3d object model file path")
            ("hand", po::value<std::string>(), "hand point cloud file path")
            ("start_time", po::value<int>()->default_value(0), "timestamp to start recordings")
            ("overwrite", po::bool_switch(), "overwrite all labels after start_time")
            ("show_pc", po::bool_switch(), "show 3d point cloud visualization")
            ("show_rgb", po::bool_switch(), "show rgb image");
    po::options_description all_options;
    all_options.add(generic_options).add(config_options);

    po::variables_map arguments;
    po::store(po::parse_command_line(argc, argv, all_options), arguments);
    if (arguments.count("help")) {
        std::cout << all_options << std::endl;
    }
    if (arguments.count("config")) {
        po::store(po::parse_config_file(arguments["config"].as<std::string>().c_str(), config_options), arguments);
    }
    po::notify(arguments);

    std::cout << "Running with arguments: " << std::endl;
    for (auto arg : arguments) {
        std::cout << arg.first.c_str() << ": ";
        try {
            std::cout << (arg.second.as<bool>() ? "true" : "false") << std::endl;
        } catch (...) {/* do nothing */ }
        try {
            std::cout << arg.second.as<int>() << std::endl;
        } catch (...) {/* do nothing */ }
        try {
            std::cout << arg.second.as<double>() << std::endl;
        } catch (...) {/* do nothing */ }
        try {
            std::cout << arg.second.as<std::string>() << std::endl;
        } catch (...) {/* do nothing */ }
    }

    return arguments;
}

void on_key_pressed(const pcl::visualization::KeyboardEvent &event, void *viewer) {
    if (event.keyDown() && event.getKeySym() == "s") {
        std::string timestamp = "";
        if (pPlayback != nullptr) {
            timestamp = "_" + std::to_string(pPlayback->get_master_timestamp());
        }
        save_pc("pointcloud_raw" + timestamp + ".ply", combined_pc_ptr);
        if (full_pc_ptr != nullptr) {
            save_pc("pointcloud_handobject" + timestamp + ".ply", full_pc_ptr);
        }
    }
}

int main(int argc, char **argv) {
    po::variables_map args = parse_arguments(argc, argv);
    int start_timestamp = args["start_time"].as<int>();
    bool overwrite_labels = args["overwrite"].as<bool>();
    bool show_pc = args["show_pc"].as<bool>();
    bool show_rgb = args["show_rgb"].as<bool>();

    // Initialize MultiPlayback
    std::vector<std::string> recording_files = {
            args["main"].as<std::string>(),
            args["sub"].as<std::string>()
    };
    MultiPlayback playback(recording_files);
    pPlayback = &playback;

    // Load subordinate camera's extrinsics
    std::string calibration_file = args["extrinsics"].as<std::string>();
    Eigen::Matrix4f sub_extrinsics = Eigen::Matrix4f::Zero();
    load_Matrix4f(calibration_file, sub_extrinsics);

    // Load pose labels
    std::map<uint64_t, Eigen::Matrix4f> poses;
    bool has_labels = false;
    std::string labels_file;
    if (args.count("labels")) {
        labels_file = args["labels"].as<std::string>();
        poses = load_pose_labels(labels_file);
        has_labels = poses.size() > 0;
    }
    if (!has_labels) {
        std::cerr << "No labels found. Will continue without registration." << std::endl;
    }

    pcl::PointCloud<pcl::PointXYZRGB> handobject;
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr handobject_ptr = nullptr;

    // Load object model
    bool has_object = args.count("object");
    if (has_object) {
        std::string object_file = args["object"].as<std::string>();
        pcl::PointCloud<pcl::PointXYZRGB> model = load_ply(object_file, true, 96, 32, 32);
        handobject += model;
    }
    // Load hand model
    bool has_hand = args.count("hand");
    if (has_hand) {
        std::string hand_file = args["hand"].as<std::string>();
        pcl::PointCloud<pcl::PointXYZRGB> model = load_ply(hand_file, true, 96, 32, 32);
        handobject += model;
    }

    if (has_object || has_hand) {
        handobject_ptr = handobject.makeShared();
        handobject_ptr->is_dense = false;
    } else {
        full_pc_ptr = nullptr;
        std::cout << "Neither hand nor object model given to register in point cloud!" << std::endl;
    }

    // Get calibrations and transformations of devices
    k4a::calibration main_calibration;
    k4a::calibration sub_calibration;
    playback.read_calibration(0, &main_calibration);
    playback.read_calibration(1, &sub_calibration);
    k4a::transformation main_transform(main_calibration);
    k4a::transformation sub_transform(sub_calibration);
    k4a_transformation_t main_transformation = k4a_transformation_create(&main_calibration);
    k4a_transformation_t sub_transformation = k4a_transformation_create(&sub_calibration);

    // Init ICP
    pcl::IterativeClosestPoint<pcl::PointXYZRGB, pcl::PointXYZRGB, float> icp;
    icp.setMaximumIterations(25);
    icp.setMaxCorrespondenceDistance(0.05);
    icp.setTransformationEpsilon(1e-9);
    pcl::registration::CorrespondenceRejectorTrimmed::Ptr trim(new pcl::registration::CorrespondenceRejectorTrimmed());
    trim->setOverlapRatio(0.66);
    trim->setMinCorrespondences(40);
    icp.addCorrespondenceRejector(trim);

    // Variables to store current frame
    int idx = 0;
    std::vector<k4a_capture_t> captures;
    Eigen::Matrix4f pose;
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr main_pc_ptr(new pcl::PointCloud<pcl::PointXYZRGB>);
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr sub_pc_ptr(new pcl::PointCloud<pcl::PointXYZRGB>);
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr registered_handobject_ptr(new pcl::PointCloud<pcl::PointXYZRGB>);
    pcl::visualization::CloudViewer viewer("Aligned Point Clouds");

    playback.open();
    bool has_next_capture = playback.seek_timestamp(start_timestamp, &captures);
    while (has_next_capture) {
        uint64_t master_timestamp = playback.get_master_timestamp();
        std::cout << "Timestamp: " << master_timestamp << std::endl;

        // Check if there's a predefined pose for the current timestamp
        bool has_label = poses.count(master_timestamp);
        if (has_label) {
            pose = poses[master_timestamp];
            std::cout << "Prior pose: " << std::endl << pose << std::endl;
        }

        // Get colored point clouds of each camera
        main_pc_ptr->clear();
        sub_pc_ptr->clear();
        capture_to_point_cloud(captures[0], main_transformation, main_pc_ptr);
        capture_to_point_cloud(captures[1], sub_transformation, sub_pc_ptr);

        // Transform secondary point cloud into main camera frame
        combined_pc_ptr->clear();
        pcl::transformPointCloud<pcl::PointXYZRGB, float>(*sub_pc_ptr, *combined_pc_ptr, sub_extrinsics);
        pcl::PointCloud<pcl::PointXYZRGB>::concatenate(*combined_pc_ptr, *main_pc_ptr);
        combined_pc_ptr->is_dense = false;

        if (handobject_ptr != nullptr) {
            if (!has_label || overwrite_labels) {
                // Generate new label for this frame

                // Set ICP point clouds
                icp.setInputSource(handobject_ptr);
                icp.setInputTarget(combined_pc_ptr);
                icp.align(*registered_handobject_ptr, pose);

                // Check and store ICP result
                if (icp.hasConverged()) {
                    has_label = true;
                    std::cout << "ICP has converged, score is " << icp.getFitnessScore() << std::endl;
                    pose = icp.getFinalTransformation();
                    poses[master_timestamp] = pose;
                    std::cout << "Posterior pose: " << std::endl << pose << std::endl;
                } else {
                    std::cerr << "ICP has not converged!" << std::endl;
                }
            } else if (has_label) {
                // Transform handobject point cloud according to existing label
                registered_handobject_ptr->clear();
                pcl::transformPointCloud<pcl::PointXYZRGB, float>(*handobject_ptr, *registered_handobject_ptr, pose);
            }

            // Store in-between results
            idx++;
            if (has_labels && idx % 30 == 0) {
                save_pose_labels(labels_file, poses);
            }
        }

        // Show point cloud
        if (show_pc) {
            if (handobject_ptr != nullptr) {
                full_pc_ptr->clear();
                pcl::PointCloud<pcl::PointXYZRGB>::concatenate(*full_pc_ptr, *combined_pc_ptr);
                pcl::PointCloud<pcl::PointXYZRGB>::concatenate(*full_pc_ptr, *registered_handobject_ptr);
            }

            // Update Cloudviewer
            try {
                viewer.showCloud(full_pc_ptr);
            } catch (const std::exception &e) {
                cerr << e.what() << endl;
            }
        }

        // Show main camera RGB
        if (show_rgb) {
            k4a::image main_rgb = k4a::image(k4a_capture_get_color_image(captures[0]));
            cv::Mat cv_main_rgb = color_to_opencv(main_rgb);

            if (handobject_ptr != nullptr) {
                for (auto it = registered_handobject_ptr->begin(); it != registered_handobject_ptr->end(); ++it) {
                    k4a_float3_t src_pt;
                    src_pt.xyz.x = it->x;
                    src_pt.xyz.y = it->y;
                    src_pt.xyz.z = it->z;

                    k4a_float2_t target;
                    bool valid = main_calibration.convert_3d_to_2d(src_pt,
                                                                   K4A_CALIBRATION_TYPE_COLOR,
                                                                   K4A_CALIBRATION_TYPE_COLOR,
                                                                   &target);
                    if (valid && target.xy.x >= 0 && target.xy.x < cv_main_rgb.cols && target.xy.y >= 0 &&
                        target.xy.y < cv_main_rgb.rows) {
                        cv_main_rgb.at<float>(target.xy.y, target.xy.x, 0) = (uint8_t) 32; // G
                        cv_main_rgb.at<float>(target.xy.y, target.xy.x, 1) = (uint8_t) 32; // B
                        cv_main_rgb.at<float>(target.xy.y, target.xy.x, 2) = (uint8_t) 96; // R
                    }
                }
            }

            cv::imshow("Main RGB", cv_main_rgb);
            main_rgb.~image();
        }

        // Load next capture
        has_next_capture = playback.next_synchronized_captures(&captures);
    }

    // Finish up
    playback.close();
    if (has_labels) save_pose_labels(labels_file, poses);
    cv::destroyAllWindows();
    std::cout << "Exiting..." << std::endl;
    cv::waitKey(1000);
    return 0;
}