#include "pointpillar_ros.hpp"
#include <ros/ros.h>
#include <cstdlib>

int main(int argc, char** argv) {

    ros::init(argc, argv, "pointpillar_node");
    ros::NodeHandle nh; //创建ros句柄
    std::string pathValue = std::getenv("SEED_HOME");
    std::string parame_file_dir = pathValue + "/data/pointpillar/yaml/pointpillarconfig.yaml";
    PointPillar point_pillar(nh, parame_file_dir);
    ros::Rate rate(20);
    while(ros::ok())
    {   
        rate.sleep();
        ros::spinOnce();
        point_pillar.Predict();
    }
    point_pillar.StreamDestroy();
    return 0;
}