#ifndef POINTCLOUD_FILTER_HPP
#define POINTCLOUD_FILTER_HPP

// #include <cv2>
// #include <Eigen3>
#include <rclcpp.hpp>
#include <visualization_msgs/msg/Marker.hpp>
#include "yolo_msgs/msg/detection_array.hpp"

class ConesDetector : public rclcpp::Node
{
public:
    ConesDetector();

private:
    // callbacks
    void boundingBoxesCallback(const yolo_msgs::msg::DetectionArray::SharedPtr msg);

    // methods
    void initialize();
    void load_parameters();

    // pubs and subs
    rclcpp::Subscription<const yolo_msgs::msg::DetectionArray>::SharedPtr m_input_sub;
    rclcpp::Publisher<sensor_msgs::msg::MarkerArray>::SharedPtr m_output_pub;

    //topics
    std::string m_input_topic;
    std::string m_output_topic;

};

#endif // CAMERA_CONES_DETECTOR_HPP