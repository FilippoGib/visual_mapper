#include "camera_cones_detector.hpp"
#include "yolo_msgs/msg/detection_array.hpp"
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
    m_input_sub = this->create_subscription<yolo_msgs::DetectionArray>(
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


void ConesDetector::boundingBoxesCallback(const yolo_msgs::DetectionArray::SharedPtr msg)
{
    visualization_msgs::msg::MarkerArray marker_array;

    int id = 0;
    for (const auto& detection : msg->detections)
    {
        // Extract bounding box (assuming detection has bbox: xmin, ymin, xmax, ymax)
        // If your detection message format differs, adapt accordingly
        double xmin = detection.xmin;
        double ymin = detection.ymin;
        double xmax = detection.xmax;
        double ymax = detection.ymax;

        // Compute bottom center and top center pixel coordinates
        double base_u = (xmin + xmax) / 2.0;
        double base_v = ymax;  // bottom of bounding box

        double tip_u = (xmin + xmax) / 2.0;
        double tip_v = ymin;   // top of bounding box

        // Backproject pixels to 3D points (assume ground plane Z=0 for base, Z=cone_height for tip)
        // Use your actual camera intrinsics, distortion, rotation R, and translation t here
        double ground_z = 0.0;
        double cone_height = 0.35;  // example cone height in meters

        Eigen::Vector3d base_3d = backprojectPixelToGroundWithDistortion(
            base_u, base_v, K, distCoeffs, R, t, ground_z);

        Eigen::Vector3d tip_3d = backprojectPixelToGroundWithDistortion(
            tip_u, tip_v, K, distCoeffs, R, t, cone_height);

        // Create a marker representing the cone (using LINE_LIST for simplicity)
        visualization_msgs::msg::Marker marker;
        marker.header = msg->header;
        marker.ns = "cones";
        marker.id = id++;
        marker.type = visualization_msgs::msg::Marker::LINE_LIST;
        marker.action = visualization_msgs::msg::Marker::ADD;
        marker.scale.x = 0.02;  // line width

        marker.color.r = 1.0f;
        marker.color.g = 0.5f;
        marker.color.b = 0.0f;
        marker.color.a = 1.0f;

        // Base point
        geometry_msgs::msg::Point base_pt;
        base_pt.x = base_3d.x();
        base_pt.y = base_3d.y();
        base_pt.z = base_3d.z();

        // Tip point
        geometry_msgs::msg::Point tip_pt;
        tip_pt.x = tip_3d.x();
        tip_pt.y = tip_3d.y();
        tip_pt.z = tip_3d.z();

        // Add lines for the cone sides (simplified as a line from base to tip)
        marker.points.push_back(base_pt);
        marker.points.push_back(tip_pt);

        marker_array.markers.push_back(marker);
    }

    m_output_pub->publish(marker_array);
}
