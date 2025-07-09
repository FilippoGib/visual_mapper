#include "camera_cones_detector.hpp"
#include "vision_msgs/msg/bounding_box2_d.hpp"
#include "vision_msgs/msg/detection2_d.hpp"
#include "vision_msgs/msg/detection2_d_array.hpp"
#include "geometry_msgs/msg/pose2_d.hpp"
#include <rclcpp/rclcpp.hpp>
#include <Eigen/Dense>

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
    m_input_sub = this->create_subscription<vision_msgs::msg::Detection2DArray>(
        m_input_topic, 10,
        std::bind(&ConesDetector::boundingBoxesCallback, this, std::placeholders::_1)
    );

    m_output_pub = this->create_publisher<visualization_msgs::msg::MarkerArray>(m_output_topic, 10);
}

ConesDetector::ConesDetector() : Node("camera_cones_detector_node") 
{
    this->initialize();
}

Eigen::Vector3d backprojectPixelToGroundWithDistortion(
    double u, double v,
    const cv::Mat& K, const cv::Mat& distCoeffs,
    const Eigen::Matrix3d& R, const Eigen::Vector3d& t,
    double Z)
{
    std::vector<cv::Point2f> distortedPoints = {cv::Point2f(u, v)};
    std::vector<cv::Point2f> undistortedPoints;
    cv::undistortPoints(distortedPoints, undistortedPoints, K, distCoeffs);

    double x_n = undistortedPoints[0].x;
    double y_n = undistortedPoints[0].y;

    Eigen::Vector3d ray_cam(x_n, y_n, 1.0);
    Eigen::Vector3d n_world(0, 0, 1);
    double d_world = -Z;

    Eigen::Vector3d n_cam = R.transpose() * n_world;
    double d_cam = n_world.dot(t) + d_world;

    double lam = -d_cam / n_cam.dot(ray_cam);
    Eigen::Vector3d X_cam = lam * ray_cam;
    Eigen::Vector3d X_world = R * X_cam + t;

    return X_world;
}


void ConesDetector::boundingBoxesCallback(const vision_msgs::msg::Detection2DArray::SharedPtr msg)
{
    visualization_msgs::msg::MarkerArray marker_array;

    int id = 0;
    for (const auto& detection : msg->detections)
    {
        // Bounding box center and size
        double center_x = detection.bbox.center.position.x;
        double center_y = detection.bbox.center.position.y;
        double width = detection.bbox.size_x;
        double height = detection.bbox.size_y;

        // Convert to corners
        double xmin = center_x - width / 2.0;
        double xmax = center_x + width / 2.0;
        double ymin = center_y - height / 2.0;
        double ymax = center_y + height / 2.0;

        // Compute bottom center and top center pixel coordinates
        double base_u = (xmin + xmax) / 2.0;
        double base_v = ymax;

        double tip_u = (xmin + xmax) / 2.0;
        double tip_v = ymin;

        // Backproject to 3D
        double ground_z = 0.0;
        double cone_height = 0.35;

        Eigen::Vector3d base_3d = backprojectPixelToGroundWithDistortion(
            base_u, base_v, K, distCoeffs, R, t, ground_z);

        Eigen::Vector3d tip_3d = backprojectPixelToGroundWithDistortion(
            tip_u, tip_v, K, distCoeffs, R, t, cone_height);

        // Create marker
        visualization_msgs::msg::Marker marker;
        marker.header = msg->header;
        marker.ns = "cones";
        marker.id = id++;
        marker.type = visualization_msgs::msg::Marker::LINE_LIST;
        marker.action = visualization_msgs::msg::Marker::ADD;
        marker.scale.x = 0.02;

        marker.color.r = 1.0f;
        marker.color.g = 0.5f;
        marker.color.b = 0.0f;
        marker.color.a = 1.0f;

        geometry_msgs::msg::Point base_pt, tip_pt;
        base_pt.x = base_3d.x(); base_pt.y = base_3d.y(); base_pt.z = base_3d.z();
        tip_pt.x = tip_3d.x();   tip_pt.y = tip_3d.y();   tip_pt.z = tip_3d.z();

        marker.points.push_back(base_pt);
        marker.points.push_back(tip_pt);

        marker_array.markers.push_back(marker);
    }


    m_output_pub->publish(marker_array);
}
