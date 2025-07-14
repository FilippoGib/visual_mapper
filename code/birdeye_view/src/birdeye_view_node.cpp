#include "birdeye_view/birdeye_view.hpp"
#include <rclcpp/rclcpp.hpp>

int main(int argc, char **argv) {
  rclcpp::init(argc, argv);
  auto node = std::make_shared<birdeye_view::Birdeye>();
  rclcpp::spin(node);
  rclcpp::shutdown();
  return 0;
}