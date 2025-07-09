#ifndef CAMERA_CONES_DETECTOR_HPP
#define CAMERA_CONES_DETECTOR_HPP

#include <rclcpp/rclcpp.hpp>
#include <vision_msgs/msg/detection2_d_array.hpp>
#include <visualization_msgs/msg/marker_array.hpp>
#include <opencv2/opencv.hpp>
#include <Eigen/Dense>

class ConesDetector : public rclcpp::Node
{
public:
    ConesDetector();

private:
    // Callback
    void boundingBoxesCallback(const vision_msgs::msg::Detection2DArray::SharedPtr msg);

    // Methods
    void initialize();
    void load_parameters();

    // Publishers and subscribers
    rclcpp::Subscription<vision_msgs::msg::Detection2DArray>::SharedPtr m_input_sub;
    rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr m_output_pub;

    // Topics
    std::string m_input_topic;
    std::string m_output_topic;

    // Camera parameters (presumibilmente definiti altrove)
    cv::Mat K, distCoeffs;
    Eigen::Matrix3d R;
    Eigen::Vector3d t;
};

#endif // CAMERA_CONES_DETECTOR_HPP
