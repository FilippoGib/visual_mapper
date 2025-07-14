// birdeye_view.cpp
#include "birdeye_view.hpp"

BirdEyeView::BirdEyeView(double fx, double fy, double cx, double cy,
                         int output_width, int output_height,
                         std::vector<cv::Point2f> src_points,
                         std::vector<cv::Point2f> dst_points)
    : output_width_(output_width), output_height_(output_height) {
  homography_ = cv::getPerspectiveTransform(src_points, dst_points);
}

cv::Mat BirdEyeView::transform(const cv::Mat& input_image) const {
  cv::Mat output_image;
  cv::warpPerspective(input_image, output_image, homography_, cv::Size(output_width_, output_height_));
  return output_image;
}
