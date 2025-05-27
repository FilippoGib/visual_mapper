#include "camera_cones_detector/camera_cones_detector.cpp"
#include "yolo_msgs/msg/detection_array.hpp"
#include "rclcpp/rclcpp.hpp"

void ConesDetector::load_parameters()
{
    this->declare_parameter<std::string>("input_topic", "");
    m_input_topic = this->get_parameter("input_topic").get_value<std::string>();

    this->declare_parameter<std::string>("output_topic", "");
    m_output_topic = this->get_parameter("output_topic").get_value<std::string>();
}

void ConesDetector::initialize()
{
    // Load parameters
    this->load_parameters();

    rclcpp::QoS qos_rel(rclcpp::KeepLast(1));
    qos_rel.reliable();

    // Initialize publishers and subscribers
    m_input_sub = this->create_subscription<yolo_msgs::msg::DetectionArray>(
        m_input_topic, qos_rel,
        [this](const yolo_msgs::msg::DetectionArray::SharedPtr msg) {
            this->boundingBoxesCallback(msg);
        });
    
    m_output_pub = this->create_publisher<sensor_msgs::msg::MarkerArray>(m_output_topic, 10);
}

ConesDetector::ConesDetector() : Node("camera_cones_detector_node") 
{
    this->initialize();
}

void ConesDetector::boundingBoxesCallback(const yolo_msgs::msg::DetectionArray::SharedPtr msg)
{
    //callback
    
}