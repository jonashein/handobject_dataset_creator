#pragma once
#include <eigen3/Eigen/Dense>
#include <k4a/k4a.hpp>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <pcl/io/ply_io.h>

void store_calibration_extrinsics(const std::string &output_file, const k4a_calibration_extrinsics_t &calib)
{
    std::ofstream file;
    file.open(output_file, std::ofstream::trunc);
    file << calib.rotation[0] << " " << calib.rotation[1] << " " << calib.rotation[2] << " " << calib.translation[0] << std::endl;
    file << calib.rotation[3] << " " << calib.rotation[4] << " " << calib.rotation[5] << " " << calib.translation[1] << std::endl;
    file << calib.rotation[6] << " " << calib.rotation[7] << " " << calib.rotation[8] << " " << calib.translation[2] << std::endl;
    file.close();
}

void store_calibration_intrinsics(const std::string &output_file, const k4a_calibration_intrinsics_t &calib)
{
    std::ofstream file;
    file.open(output_file, std::ofstream::trunc);
    file << calib.parameters.param.fx << " " << 0.0 << " " << calib.parameters.param.cx << std::endl;
    file << 0.0 << " " << calib.parameters.param.fy << " " << calib.parameters.param.cy << std::endl;
    file << 0.0 << " " << 0.0 << " " << 1.0 << std::endl;
    file.close();
}

void load_Matrix4f(const std::string& file_path, Eigen::Matrix4f& mat) {
    mat = Eigen::Matrix4f::Zero();
    std::ifstream file(file_path);

    for(unsigned int row = 0; row < 4; row++) {
        for (unsigned int col = 0; col < 4; col++) {
            file >> mat(row, col);
        }
    }
    file.close();
}

pcl::PointCloud<pcl::PointXYZRGB> load_ply(const std::string &filename, bool override_rgb = false,
                                           uint8_t r = 0, uint8_t g = 0, uint8_t b = 0) {
    pcl::PointCloud<pcl::PointXYZRGB> mesh;
    pcl::PLYReader fileReader;
    fileReader.read(filename, mesh);
    if (override_rgb) {
        for (auto pt : mesh) {
            pt.r = r;
            pt.g = g;
            pt.b = b;
        }
    }
    return mesh;
}

void save_pc(const std::string &filename, const pcl::PointCloud<pcl::PointXYZRGB>::ConstPtr pc)
{
    pcl::PLYWriter file;
    file.write(filename, *pc);
}

std::map<uint64_t, Eigen::Matrix4f> load_pose_labels(const std::string &label_file)
{
    std::map<uint64_t, Eigen::Matrix4f> labels;
    std::ifstream file(label_file);

    std::string line;
    while (std::getline(file, line))
    {
        uint64_t timestamp = std::stoull(line);
        Eigen::Matrix4f mat;
        for (int row = 0; row < 4; ++row)
        {
            for (int col = 0; col < 4; ++col)
            {
                file >> mat(row, col);
            }
        }
        labels[timestamp] = mat;

        // Read line break
        std::getline(file, line);
    }

    file.close();
    return labels;
}

void save_pose_labels(const std::string &output_file, const std::map<uint64_t, Eigen::Matrix4f> &labels) {
    std::ofstream file(output_file, std::ofstream::trunc);

    for (auto it = labels.begin(); it != labels.end(); ++it)
    {
        file << it->first << std::endl;
        file << it->second << std::endl;
    }
    std::cout << "Stored " << labels.size() << " labels." << std::endl;
    file.close();
}
