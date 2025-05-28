#ifndef CAMERA_CONES_DETECTOR_HPP
#define CAMERA_CONES_DETECTOR_HPP

#include <rclcpp/rclcpp.hpp>
#include <visualization_msgs/msg/marker.hpp>
#include <yolo_msgs/msg/detection.hpp>
#include <opencv2/opencv.hpp>
#include <Eigen/Dense>

class ConesDetector : public rclcpp::Node
{
public:
    ConesDetector();

private:
    // callbacks
    void boundingBoxesCallback(const yolo_msgs::DetectionArray::SharedPtr msg);

    // methods
    void initialize();
    void load_parameters();

    // pubs and subs
    rclcpp::Subscription<const yolo_msgs::DetectionArray>::SharedPtr m_input_sub;
    rclcpp::Publisher<sensor_msgs::msg::MarkerArray>::SharedPtr m_output_pub;

    //topics
    std::string m_input_topic;
    std::string m_output_topic;

};

#endif // CAMERA_CONES_DETECTOR_HPP