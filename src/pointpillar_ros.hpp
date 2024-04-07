#ifndef __POINTPILLAR_ROS_HPP__
#define __POINTPILLAR_ROS_HPP__

#include <ros/ros.h>
#include <sensor_msgs/PointCloud2.h>
#include <pcl/point_types.h>
#include <pcl_conversions/pcl_conversions.h> // 包含转换函数
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/filters/filter.h>
#include <pcl/filters/crop_box.h>

#include <cuda_runtime.h>
#include <string.h>
#include <iostream>
#include <sstream>
#include <fstream>
#include <dirent.h>
#include <deque>
#include <mutex>
#include <string.h>
#include "pointpillar.hpp"
#include "check.hpp"
#include "dtype.hpp"

#include <visualization_msgs/Marker.h>
#include <visualization_msgs/MarkerArray.h>
#include <geometry_msgs/Pose.h>
#include <geometry_msgs/Vector3.h>
#include <geometry_msgs/Point.h>
#include <geometry_msgs/Quaternion.h>
#include <tf/tf.h> 
#include <yaml-cpp/yaml.h>

#include <jsk_recognition_msgs/BoundingBox.h>
#include <jsk_recognition_msgs/BoundingBoxArray.h>
#include <visualization_msgs/Marker.h>
#include <visualization_msgs/MarkerArray.h>

using namespace pointpillar::lidar;
using namespace nvtype;
class PointPillar{
public:
    PointPillar(ros::NodeHandle &nh, const std::string& parameter_file_dir);
    ~PointPillar();
    void PclToArray(const pcl::PointCloud<pcl::PointXYZI> & cloud, float *out_points_array);
    void GetDeviceInfo();
    void StreamDestroy();
    void Predict();
    void DetectObjPub(const pcl::PointCloud<pcl::PointXYZI>::Ptr& cloud,const std::vector<BoundingBox> boxes);
    
private:
    void PointCloudCallBack(const sensor_msgs::PointCloud2ConstPtr& cloud);
    std::shared_ptr<pointpillar::lidar::Core> CreatePointPillarCore(); 
    void SaveBoxPred(std::vector<pointpillar::lidar::BoundingBox> boxes, std::string file_name);
    void GetParameter();
    void PrintParamter() const;
    void DrawBox(std::vector<pointpillar::lidar::BoundingBox> boxes);
    void DrawBox(std::vector<pointpillar::lidar::BoundingBox> boxes, const pcl::PointCloud<pcl::PointXYZI>::Ptr& cloud);

private:
    ros::NodeHandle nh_;
    pcl::PointCloud<pcl::PointXYZI>::Ptr predict_cloud_;
    ros::Subscriber lidar_sub_;
    ros::Publisher aerohead_pub_;
    ros::Publisher aeroengine_pub_;
    ros::Publisher backwheel_pub_;

    ros::Publisher pub_bbox_;
    ros::Publisher pub_bbox_label_;
    mutable std::mutex data_mutex_;
    double boxes_thresh_;

    // cuda
    std::deque<pcl::PointCloud<pcl::PointXYZI>::Ptr> cloud_;
    std::shared_ptr<pointpillar::lidar::Core> predict_;
    cudaStream_t stream_;
    std::string file_name_;

    // pointpillar parmater
    VoxelizationParameter vp_;
    PostProcessParameter pp_;
    std::string model_addr_;
    std::string parameter_file_dir_;

    double aerohead_thresh_;
    double aeroengine_thresh_;
    double wheel_thresh_;


};














#endif