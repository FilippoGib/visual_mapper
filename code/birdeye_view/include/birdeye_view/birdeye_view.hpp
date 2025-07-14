// birdeye_view.hpp
#pragma once

#include <opencv2/opencv.hpp>

class BirdEyeView {
public:
  BirdEyeView(double fx, double fy, double cx, double cy,
              int output_width, int output_height,
              std::vector<cv::Point2f> src_points,
              std::vector<cv::Point2f> dst_points);

  cv::Mat transform(const cv::Mat& input_image) const;

private:
  cv::Mat homography_;
  int output_width_;
  int output_height_;
};
