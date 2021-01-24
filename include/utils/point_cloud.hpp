#pragma once
#include <eigen3/Eigen/Dense>
#include <k4a/k4a.hpp>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <pcl/io/ply_io.h>

cv::Mat color_to_opencv(const k4a::image &im)
{
    cv::Mat cv_image_with_alpha(im.get_height_pixels(), im.get_width_pixels(), CV_8UC4, (void *)im.get_buffer());
    cv::Mat cv_image_no_alpha;
    cv::cvtColor(cv_image_with_alpha, cv_image_no_alpha, cv::COLOR_BGRA2BGR);
    return cv_image_no_alpha;
}

cv::Mat depth_to_opencv(const k4a::image &im)
{
    cv::Mat cv_image(im.get_height_pixels(),
                     im.get_width_pixels(),
                     CV_16UC1,
                     (void *)im.get_buffer());
    return cv_image;
}

void fill_point_cloud(const k4a_image_t point_cloud_image,
                      const k4a_image_t color_image,
                      pcl::PointCloud<pcl::PointXYZRGB>::Ptr point_cloud_ptr,
                      uint8_t r,
                      uint8_t g,
                      uint8_t b)
{
    /*
     *  Function: Get from a Mat to pcl pointcloud datatype
     *  In: cv::Mat
     *  Out: pcl::PointCloud
     *
     *  Based on https://stackoverflow.com/a/32521044
     */
    const float POINT_CLOUD_SCALE_FACTOR = 0.001f; // 1.0f;

    int width = k4a_image_get_width_pixels(point_cloud_image);
    int height = k4a_image_get_height_pixels(color_image);

    int16_t *point_cloud_image_data = (int16_t *)(void *)k4a_image_get_buffer(point_cloud_image);
    uint8_t *color_image_data = k4a_image_get_buffer(color_image);

    for (int i = 0; i < width * height; i++)
    {
        pcl::PointXYZRGB point(color_image_data[4 * i + 2],
                               color_image_data[4 * i + 1],
                               color_image_data[4 * i + 0]);
        point.x = (float)point_cloud_image_data[3 * i + 0] * POINT_CLOUD_SCALE_FACTOR;
        point.y = (float)point_cloud_image_data[3 * i + 1] * POINT_CLOUD_SCALE_FACTOR;
        point.z = (float)point_cloud_image_data[3 * i + 2] * POINT_CLOUD_SCALE_FACTOR;
        uint8_t alpha = color_image_data[4 * i + 3];
        if (point.z != 0 && (point.r != 0 || point.g != 0 || point.b != 0 || alpha != 0)
            && sqrt(point.x * point.x + point.y * point.y + point.z * point.z) <= 1.5) // remove all points which are too far away from the camera (e.g. walls)
        {
            point_cloud_ptr->points.push_back(point);
        }
    }

    point_cloud_ptr->width = (int)point_cloud_ptr->points.size();
    point_cloud_ptr->height = 1;
    point_cloud_ptr->is_dense = false;
}

bool capture_to_point_cloud(const k4a_capture_t &capture,
                            const k4a_transformation_t &transformation_handle,
                            pcl::PointCloud<pcl::PointXYZRGB>::Ptr point_cloud_ptr)
{

    k4a_image_t color_image = k4a_capture_get_color_image(capture);
    k4a_image_t depth_image = k4a_capture_get_depth_image(capture);

    int color_image_width_pixels = k4a_image_get_width_pixels(color_image);
    int color_image_height_pixels = k4a_image_get_height_pixels(color_image);
    k4a_image_t transformed_depth_image = NULL;
    if (K4A_RESULT_SUCCEEDED != k4a_image_create(K4A_IMAGE_FORMAT_DEPTH16,
                                                 color_image_width_pixels,
                                                 color_image_height_pixels,
                                                 color_image_width_pixels * (int)sizeof(uint16_t),
                                                 &transformed_depth_image))
    {
        printf("Failed to create transformed depth image\n");
        return false;
    }

    k4a_image_t point_cloud_image = NULL;
    if (K4A_RESULT_SUCCEEDED != k4a_image_create(K4A_IMAGE_FORMAT_CUSTOM,
                                                 color_image_width_pixels,
                                                 color_image_height_pixels,
                                                 color_image_width_pixels * 3 * (int)sizeof(int16_t),
                                                 &point_cloud_image))
    {
        printf("Failed to create point cloud image\n");
        return false;
    }

    if (K4A_RESULT_SUCCEEDED !=
        k4a_transformation_depth_image_to_color_camera(transformation_handle, depth_image, transformed_depth_image))
    {
        printf("Failed to compute transformed depth image\n");
        return false;
    }

    if (K4A_RESULT_SUCCEEDED != k4a_transformation_depth_image_to_point_cloud(transformation_handle,
                                                                              transformed_depth_image,
                                                                              K4A_CALIBRATION_TYPE_COLOR,
                                                                              point_cloud_image))
    {
        printf("Failed to compute point cloud\n");
        return false;
    }

    fill_point_cloud(point_cloud_image, color_image, point_cloud_ptr, 0, 0, 0);

    k4a_image_release(transformed_depth_image);
    k4a_image_release(point_cloud_image);
    k4a_image_release(color_image);
    k4a_image_release(depth_image);
    return true;
}