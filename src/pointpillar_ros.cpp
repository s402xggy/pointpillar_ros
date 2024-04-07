#include "pointpillar_ros.hpp"
#include <cuda_device_runtime_api.h>
#include <filesystem>

std::string kLABEL[4] = {"AeroHead", "AeroEngine", "AeroWheel"};

PointPillar::PointPillar(ros::NodeHandle &nh, const std::string& parameter_file_dir): nh_(nh), 
parameter_file_dir_(parameter_file_dir) {
    // lidar_sub_ = nh_.subscribe("/complete_cloud", 1, &PointPillar::PointCloudCallBack,this);
    lidar_sub_ = nh_.subscribe("/ground_segmentation/nonground", 1, &PointPillar::PointCloudCallBack,this);

    predict_cloud_ = boost::make_shared<pcl::PointCloud<pcl::PointXYZI>>();
    aerohead_pub_ = nh_.advertise<sensor_msgs::PointCloud2>("/aerohead_cloud", 1);
    aeroengine_pub_ = nh_.advertise<sensor_msgs::PointCloud2>("/aeroengine_cloud", 1);
    backwheel_pub_ = nh_.advertise<sensor_msgs::PointCloud2>("/backwheel_cloud", 1);
    pub_bbox_ = nh_.advertise<jsk_recognition_msgs::BoundingBoxArray>("/boxes", 1);
    pub_bbox_label_ = nh_.advertise<visualization_msgs::MarkerArray>("/boxes_label", 1);
    GetParameter();
    PrintParamter();
    GetDeviceInfo();
    cudaStreamCreate(&stream_);

    predict_ = CreatePointPillarCore();
    if (predict_ == nullptr) {
      printf("Core has been failed.\n");
      return ;
    }
    else {
      predict_->print();
      predict_->set_timer(true);
    }

    std::filesystem::remove_all(file_name_);
    std::filesystem::create_directory(file_name_);
}

PointPillar::~PointPillar() {
 
}



void PointPillar::PointCloudCallBack(const sensor_msgs::PointCloud2ConstPtr& cloud_msg) {
    if (cloud_msg->data.empty())
      return ;
    pcl::PointCloud<pcl::PointXYZI>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZI>);
    pcl::PointCloud<pcl::PointXYZI>::Ptr cloud_filtered(new pcl::PointCloud<pcl::PointXYZI>);
    pcl::fromROSMsg(*cloud_msg, *cloud);
    std::vector<int> indices;
    pcl::removeNaNFromPointCloud(*cloud, *cloud_filtered, indices);
    data_mutex_.lock();
    if (cloud_.size() > 10) 
      cloud_.pop_front();
    cloud_.push_back(cloud_filtered);
    data_mutex_.unlock();
}

void PointPillar::GetDeviceInfo(void)
{
  cudaDeviceProp prop;
  int count = 0;
  cudaGetDeviceCount(&count);
  printf("\nGPU has cuda devices: %d\n", count);
  for (int i = 0; i < count; ++i) {
    cudaGetDeviceProperties(&prop, i);
    printf("----device id: %d info----\n", i);
    printf("  GPU : %s \n", prop.name);
    printf("  Capbility: %d.%d\n", prop.major, prop.minor);
    printf("  Global memory: %luMB\n", prop.totalGlobalMem >> 20);
    printf("  Const memory: %luKB\n", prop.totalConstMem  >> 10);
    printf("  SM in a block: %luKB\n", prop.sharedMemPerBlock >> 10);
    printf("  warp size: %d\n", prop.warpSize);
    printf("  threads in a block: %d\n", prop.maxThreadsPerBlock);
    printf("  block dim: (%d,%d,%d)\n", prop.maxThreadsDim[0], prop.maxThreadsDim[1], prop.maxThreadsDim[2]);
    printf("  grid dim: (%d,%d,%d)\n", prop.maxGridSize[0], prop.maxGridSize[1], prop.maxGridSize[2]);
  }
  printf("\n");
}

std::shared_ptr<pointpillar::lidar::Core> PointPillar::CreatePointPillarCore() {
    vp_.grid_size =
        vp_.compute_grid_size(vp_.max_range, vp_.min_range, vp_.voxel_size);
    pp_.min_range = vp_.min_range;
    pp_.max_range = vp_.max_range;
    pp_.feature_size = nvtype::Int2(vp_.grid_size.x/2, vp_.grid_size.y/2);
    pointpillar::lidar::CoreParameter param;
    param.voxelization = vp_;
    param.lidar_model = model_addr_;
    param.lidar_post = pp_;
    return pointpillar::lidar::create_core(param);
}


void PointPillar::PclToArray(const pcl::PointCloud<pcl::PointXYZI> & cloud, float *out_points_array) {
    size_t len = cloud.size();
    for(size_t i = 0; i < len; i++)
    {
      pcl::PointXYZI point = cloud.at(i);
      out_points_array[i * 4] = point.x;
      out_points_array[i * 4 + 1] = point.y;
      out_points_array[i * 4 + 2] = point.z;
      out_points_array[i * 4 + 3] = float(point.intensity);
    }
}

void PointPillar::Predict() {
    static int i = 0;
    if (cloud_.size() == 0) {
      return;
    }
    data_mutex_.lock();
    predict_cloud_ = cloud_.front();
    cloud_.pop_front();
    data_mutex_.unlock();
    float *out_points_array = (new float[predict_cloud_->points.size() * 4]);
    PclToArray(*predict_cloud_, out_points_array);
    auto bboxes = predict_->forward(out_points_array, predict_cloud_->points.size(), stream_);
    std::cout<<"Detections after NMS: "<< bboxes.size()<<std::endl;
    std::cout << "=============================================" << std::endl;
    if (bboxes.size() != 0) {
      std::string name = file_name_  + std::to_string(i) + ".txt";
      SaveBoxPred(bboxes, name);
      DrawBox(bboxes, predict_cloud_);
      i++;
    }
    delete out_points_array;

}

void PointPillar::StreamDestroy() {
  checkRuntime(cudaStreamDestroy(stream_));

}


void PointPillar::DetectObjPub(const pcl::PointCloud<pcl::PointXYZI>::Ptr& cloud,const std::vector<BoundingBox> boxes) {
    pcl::PointCloud<pcl::PointXYZI>::Ptr aerohead_cloud(new pcl::PointCloud<pcl::PointXYZI>);
    pcl::PointCloud<pcl::PointXYZI>::Ptr aeroengine_cloud(new pcl::PointCloud<pcl::PointXYZI>);
    pcl::PointCloud<pcl::PointXYZI>::Ptr wheel_cloud(new pcl::PointCloud<pcl::PointXYZI>);
    sensor_msgs::PointCloud2 aerohead_cloud_msg;
    sensor_msgs::PointCloud2 aeroengine_cloud_msg;
    sensor_msgs::PointCloud2 wheel_cloud_msg;

    pcl::CropBox<pcl::PointXYZI> crop_filter;
    crop_filter.setInputCloud(cloud);
    for (const auto& box : boxes) {
      if (box.id == 0 && box.score < aerohead_thresh_)
        continue;
      else if (box.id == 1 && box.score < aeroengine_thresh_)
        continue;
      else if (box.id == 2 && box.score < wheel_thresh_)
        continue;
      Eigen::Vector4f min_point(((-box.w) / 2), ((-box.l) / 2), ((-box.h) / 2), 1);
      Eigen::Vector4f max_point((box.w / 2), (box.l / 2), (box.h / 2), 1);
      crop_filter.setMin(min_point);
      crop_filter.setMax(max_point);
      Eigen::Affine3f transform = pcl::getTransformation(box.x, box.y, box.z, 0, 0, box.rt);
      crop_filter.setTransform(transform.inverse());
      pcl::PointCloud<pcl::PointXYZI>::Ptr filtered_cloud(new pcl::PointCloud<pcl::PointXYZI>);
      crop_filter.filter(*filtered_cloud);
      for (auto & point: filtered_cloud->points) {
        point.intensity = box.id;
      }
      if (box.id == 0)
        *aerohead_cloud += *filtered_cloud;
      else if (box.id == 1) 
        *aeroengine_cloud += *filtered_cloud;
      else if (box.id == 2)
        *wheel_cloud += *filtered_cloud;
    }
    pcl::toROSMsg(*aerohead_cloud, aerohead_cloud_msg);
    pcl::toROSMsg(*aeroengine_cloud, aeroengine_cloud_msg);
    pcl::toROSMsg(*wheel_cloud, wheel_cloud_msg);

    aerohead_cloud_msg.header.stamp = ros::Time::now();                 // 设置当前时间为时间戳
    aerohead_cloud_msg.header.frame_id = "rslidar";
    aeroengine_cloud_msg.header.stamp = aerohead_cloud_msg.header.stamp;  
    aeroengine_cloud_msg.header.frame_id = aerohead_cloud_msg.header.frame_id;
    wheel_cloud_msg.header.stamp = aerohead_cloud_msg.header.stamp;  
    wheel_cloud_msg.header.frame_id = aerohead_cloud_msg.header.frame_id;

    aerohead_pub_.publish(aerohead_cloud_msg);
    aeroengine_pub_.publish(aeroengine_cloud_msg);
    backwheel_pub_.publish(wheel_cloud_msg);

}


void PointPillar::SaveBoxPred(std::vector<pointpillar::lidar::BoundingBox> boxes, std::string file_name)
{
    std::ofstream ofs;
    ofs.open(file_name, std::ios::out);
    if (ofs.is_open()) {
        for (const auto box : boxes) {
          ofs << box.x << " ";
          ofs << box.y << " ";
          ofs << box.z << " ";
          ofs << box.w << " ";
          ofs << box.l << " ";
          ofs << box.h << " ";
          ofs << box.rt << " ";
          ofs << box.id << " ";
          ofs << box.score << "\n";
        }
    }
    else {
      std::cerr << "Output file cannot be opened!" << std::endl;
    }
    ofs.close();
    std::cout << "Saved prediction in: " << file_name << std::endl;
    return;
};

void PointPillar::DrawBox(std::vector<pointpillar::lidar::BoundingBox> boxes) {
  jsk_recognition_msgs::BoundingBoxArray arr_bbox;
  visualization_msgs::MarkerArray marker_array;
  int i = 0;
  for (const auto box : boxes) {
    if (box.id == 0 && box.score < aerohead_thresh_)
      continue;
    else if (box.id == 1 && box.score < aeroengine_thresh_)
      continue;
    else if (box.id == 2 && box.score < wheel_thresh_)
      continue;
    jsk_recognition_msgs::BoundingBox bbox;
    visualization_msgs::Marker marker;
    bbox.header.frame_id = "rslidar";  // Replace with your frame_id
    bbox.header.stamp = ros::Time::now();
    bbox.pose.position.x =  box.x;
    bbox.pose.position.y =  box.y;
    bbox.pose.position.z = box.z ;
    bbox.dimensions.x = box.w;  // width
    bbox.dimensions.y = box.l;  // length
    bbox.dimensions.z = box.h;  // height
    // Using tf::Quaternion for quaternion from roll, pitch, yaw
    tf::Quaternion q = tf::createQuaternionFromRPY(0, 0, box.rt);
    bbox.pose.orientation.x = q.x();
    bbox.pose.orientation.y = q.y();
    bbox.pose.orientation.z = q.z();
    bbox.pose.orientation.w = q.w();
    bbox.value = box.score;
    bbox.label = box.id;

    marker.header.frame_id = "rslidar";
    marker.header.stamp = bbox.header.stamp;
    marker.ns = "bbox_labels";
    marker.id = i;
    marker.type = visualization_msgs::Marker::TEXT_VIEW_FACING;
    marker.text = kLABEL[box.id] + "\nscore: " + std::to_string(box.score);; // Replace with actual label
    marker.pose.position.x = boxes[i].x;
    marker.pose.position.y = boxes[i].y;
    marker.pose.position.z = boxes[i].z + boxes[i].h;  // Position the text above the box
    marker.scale.z = 1;  // Text size
    marker.color.a = 1;  // Don't forget to set the alpha!
    marker.color.r = 0;
    marker.color.g = 1;
    marker.color.b = 0;
    marker.lifetime = ros::Duration(0.5);
    marker_array.markers.push_back(marker);
    arr_bbox.boxes.push_back(bbox);
    i++;
  }
  arr_bbox.header.frame_id = "rslidar";
  arr_bbox.header.stamp = ros::Time::now();
  pub_bbox_.publish(arr_bbox);
  pub_bbox_label_.publish(marker_array);
}

void PointPillar::DrawBox(std::vector<pointpillar::lidar::BoundingBox> boxes, const pcl::PointCloud<pcl::PointXYZI>::Ptr& cloud) {
  jsk_recognition_msgs::BoundingBoxArray arr_bbox;
  visualization_msgs::MarkerArray marker_array;
  int i = 0;
  for (const auto box : boxes) {
    if (box.id == 0 && box.score < aerohead_thresh_)
      continue;
    else if (box.id == 1 && box.score < aeroengine_thresh_)
      continue;
    else if (box.id == 2 && box.score < wheel_thresh_)
      continue;
    jsk_recognition_msgs::BoundingBox bbox;
    visualization_msgs::Marker marker;

    bbox.header.frame_id = "rslidar";  // Replace with your frame_id
    bbox.header.stamp = ros::Time::now();
    bbox.pose.position.x =  box.x;
    bbox.pose.position.y =  box.y;
    bbox.pose.position.z = box.z ;
    bbox.dimensions.x = box.w;  // width
    bbox.dimensions.y = box.l;  // length
    bbox.dimensions.z = box.h;  // height
    // Using tf::Quaternion for quaternion from roll, pitch, yaw
    tf::Quaternion q = tf::createQuaternionFromRPY(0, 0, box.rt);
    bbox.pose.orientation.x = q.x();
    bbox.pose.orientation.y = q.y();
    bbox.pose.orientation.z = q.z();
    bbox.pose.orientation.w = q.w();
    bbox.value = box.score;
    bbox.label = box.id;

    marker.header.frame_id = "rslidar";
    marker.header.stamp = bbox.header.stamp;
    marker.ns = "bbox_labels";
    marker.id = i;
    marker.type = visualization_msgs::Marker::TEXT_VIEW_FACING;
    marker.text = kLABEL[box.id] + "\nscore: " + std::to_string(box.score);; // Replace with actual label
    marker.pose.position.x = boxes[i].x;
    marker.pose.position.y = boxes[i].y;
    marker.pose.position.z = boxes[i].z + boxes[i].h;  // Position the text above the box
    marker.scale.z = 1;  // Text size
    marker.color.a = 1; // Don't forget to set the alpha!
    marker.color.r = 0;
    marker.color.g = 1;
    marker.color.b = 0;
    marker.lifetime = ros::Duration(0.5);
    marker_array.markers.push_back(marker);
    arr_bbox.boxes.push_back(bbox);
    i++;
  }
 
  DetectObjPub(cloud, boxes);
  arr_bbox.header.frame_id = "rslidar";
  arr_bbox.header.stamp = ros::Time::now();
  pub_bbox_.publish(arr_bbox);
  pub_bbox_label_.publish(marker_array);
}


void PointPillar::GetParameter() {
  YAML::Node config = YAML::LoadFile(parameter_file_dir_);
  vp_.min_range.x = config["VoxelizationParameter"]["min_range"][0].as<float>();
  vp_.min_range.y = config["VoxelizationParameter"]["min_range"][1].as<float>();
  vp_.min_range.z = config["VoxelizationParameter"]["min_range"][2].as<float>();

  vp_.max_range.x = config["VoxelizationParameter"]["max_range"][0].as<float>();
  vp_.max_range.y = config["VoxelizationParameter"]["max_range"][1].as<float>();
  vp_.max_range.z = config["VoxelizationParameter"]["max_range"][2].as<float>();

  vp_.voxel_size.x = config["VoxelizationParameter"]["voxel_size"][0].as<float>();
  vp_.voxel_size.y = config["VoxelizationParameter"]["voxel_size"][1].as<float>();
  vp_.voxel_size.z = config["VoxelizationParameter"]["voxel_size"][2].as<float>();

  vp_.max_voxels = config["VoxelizationParameter"]["max_voxels"].as<int>();
  vp_.max_points_per_voxel = config["VoxelizationParameter"]["max_points_per_voxel"].as<int>();
  vp_.max_points = config["VoxelizationParameter"]["max_points"].as<int>();
  vp_.num_feature = config["VoxelizationParameter"]["num_feature"].as<int>();
  // postprocessparameter
  pp_.min_range.x = config["PostProcessParameter"]["min_range"][0].as<float>();
  pp_.min_range.y = config["PostProcessParameter"]["min_range"][1].as<float>();
  pp_.min_range.z = config["PostProcessParameter"]["min_range"][2].as<float>();

  pp_.max_range.x = config["PostProcessParameter"]["max_range"][0].as<float>();
  pp_.max_range.y = config["PostProcessParameter"]["max_range"][1].as<float>();
  pp_.max_range.z = config["PostProcessParameter"]["max_range"][2].as<float>();

  pp_.num_classes = config["PostProcessParameter"]["num_classes"].as<int>();
  pp_.num_anchors = config["PostProcessParameter"]["num_anchors"].as<int>();
  pp_.len_per_anchor = config["PostProcessParameter"]["len_per_anchor"].as<int>();
  if (config["PostProcessParameter"]["anchors"] && config["PostProcessParameter"]["anchors"].IsSequence()) {
          // 确保数组大小正确
          size_t num = config["PostProcessParameter"]["anchors"].size();
        // pp_.anchors = new float[num];
          // 遍历数组，将每个元素赋值给anchors数组
          for (size_t i = 0; i < num; ++i) {
              pp_.anchors[i] = config["PostProcessParameter"]["anchors"][i].as<float>();
          }
          std::cout << std::endl;
  }
  if (config["PostProcessParameter"]["anchor_bottom_heights"] && config["PostProcessParameter"]["anchor_bottom_heights"].IsSequence()) {
          // 确保数组大小正确
          size_t num = config["PostProcessParameter"]["anchor_bottom_heights"].size();
        // pp_.anchor_bottom_heights = new float[num];
          // 遍历数组，将每个元素赋值给anchors数组
          for (size_t i = 0; i < num; ++i) {
              pp_.anchor_bottom_heights[i] = config["PostProcessParameter"]["anchor_bottom_heights"][i].as<float>();
          }
          std::cout << std::endl;
  }
  aerohead_thresh_ = config["PostProcessParameter"]["aerohead_thresh"].as<double>();
  aeroengine_thresh_ = config["PostProcessParameter"]["aeroengine_thresh"].as<double>();
  wheel_thresh_ = config["PostProcessParameter"]["wheel_thresh"].as<double>();

  pp_.num_box_values = config["PostProcessParameter"]["num_box_values"].as<int>();
  pp_.score_thresh = config["PostProcessParameter"]["score_thresh"].as<float>();
  pp_.dir_offset = config["PostProcessParameter"]["dir_offset"].as<float>();
  pp_.nms_thresh = config["PostProcessParameter"]["nms_thresh"].as<float>();
  model_addr_ = config["LidarModelParameter"]["lidar_model"].as<std::string>();
  file_name_ = config["LidarModelParameter"]["save_txt_dir"].as<std::string>();

}



void PointPillar::PrintParamter() const {
  std::cout << "==================VoxelizationParameter==================" << std::endl;
  std::cout << "min_range: " << vp_.min_range.x << " " << vp_.min_range.y << " " << vp_.min_range.z << std::endl;
  std::cout << "max_range: " << vp_.max_range.x << " " << vp_.max_range.y << " " << vp_.max_range.z << std::endl;
  std::cout << "voxel_size: " << vp_.voxel_size.x << " " << vp_.voxel_size.y << " " << vp_.voxel_size.z << std::endl;
  std::cout << "max_voxels: " << vp_.max_voxels << std::endl;
  std::cout << "max_points_per_voxel: " << vp_.max_points_per_voxel << std::endl;
  std::cout << "max_points: " << vp_.max_points << std::endl;
  std::cout << "num_feature: " << vp_.num_feature << std::endl;

  std::cout << "===================PostProcessParameter===================" << std::endl;
  std::cout << "min_range: " << pp_.min_range.x << " " << pp_.min_range.y << " " << pp_.min_range.z << std::endl;
  std::cout << "max_range: " << pp_.max_range.x << " " << pp_.max_range.y << " " << pp_.max_range.z << std::endl;
  std::cout << "num_classes: " << pp_.num_classes << std::endl;
  std::cout << "num_anchors: " << pp_.num_anchors << std::endl;
  std::cout << "len_per_anchor: " << pp_.len_per_anchor << std::endl;

  std::cout << std::endl;
  std::cout << "num_box_values: " << pp_.num_box_values << std::endl;
  std::cout << "score_thresh: " << pp_.score_thresh << std::endl;
  std::cout << "dir_offset: " << pp_.dir_offset << std::endl;
  std::cout << "nms_thresh: " << pp_.nms_thresh << std::endl;
  std::cout << "aerohead_thresh: " << aerohead_thresh_ << std::endl;
  std::cout << "aeroengine_thresh: " << aeroengine_thresh_ << std::endl;
  std::cout << "wheel_thresh: " << wheel_thresh_ << std::endl;

  // std::cout << "boxes_thresh: " << boxes_thresh_ << std::endl;
  std::cout << "===================LidarModelParameter===================" << std::endl;
  std::cout << "lidar_model: " << model_addr_<< std::endl;
  std::cout << "save_txt_dir: " << file_name_<< std::endl;
}