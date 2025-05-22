// #include <cv2>
// #include <Eigen3>
#include <rclcpp.hpp>
#include <visualization_msgs/msg/Marker.hpp>
#include <eigen3/Eigen/Geometry>
#include <vector>

namespace  cones_detector 
{
 class ConesDetector : rclcpp::Node
 {

    ConesDetector();
    std::vector<visualization_msgs::msg::Marker> boundingBoxesCallback();

    private:
    ~ConesDetector();
    
 }
} // cones detector