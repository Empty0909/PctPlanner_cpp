#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl_conversions/pcl_conversions.h>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <string>

// 文件作用：读取示例 map.pcd，周期性发布到 /global_points（intensity 置 0）
class PcdPublisher : public rclcpp::Node {
public:
  PcdPublisher() : rclcpp::Node("pcd_publisher") {
    publisher_ = this->create_publisher<sensor_msgs::msg::PointCloud2>(
        "/global_points", 10);
    timer_ = this->create_wall_timer(std::chrono::seconds(10),
                                     std::bind(&PcdPublisher::onTimer, this));

    // 通过源文件路径推断包根目录，便于二进制/源码两种运行方式找到 PCD
    std::string pkg_root = __FILE__;
    const std::string marker = "/cpp_port/src";
    auto pos = pkg_root.find(marker);
    if (pos != std::string::npos) {
      pkg_root = pkg_root.substr(0, pos);
    } else {
      pkg_root = "."; // fallback
    }
    const std::string pcd_path = pkg_root + "/rsc/pcd/map.pcd";
    RCLCPP_INFO(this->get_logger(), "Loading PCD file from: %s",
                pcd_path.c_str());

    pcl::PointCloud<pcl::PointXYZ> cloud;
    if (pcl::io::loadPCDFile<pcl::PointXYZ>(pcd_path, cloud) != 0) {
      RCLCPP_ERROR(this->get_logger(), "Failed to read PCD file: %s",
                   pcd_path.c_str());
      return;
    }
    RCLCPP_INFO(this->get_logger(), "Loaded %zu points from PCD file.",
                cloud.size());

    pcl::PointCloud<pcl::PointXYZI> cloud_i;
    cloud_i.reserve(cloud.size());
    for (const auto &pt : cloud.points) {
      pcl::PointXYZI q;
      q.x = pt.x;
      q.y = pt.y;
      q.z = pt.z;
      q.intensity = 0.0f;
      cloud_i.push_back(q);
    }

    sensor_msgs::msg::PointCloud2 msg;
    pcl::toROSMsg(cloud_i, msg);
    msg.header.frame_id = "map";

    point_cloud_msg_ = msg;
    has_msg_ = true;
  }

private:
  void onTimer() {
    if (!has_msg_)
      return;
    point_cloud_msg_.header.stamp = this->get_clock()->now();
    publisher_->publish(point_cloud_msg_);
    RCLCPP_INFO(this->get_logger(), "Publishing point cloud");
  }

  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr publisher_;
  rclcpp::TimerBase::SharedPtr timer_;
  sensor_msgs::msg::PointCloud2 point_cloud_msg_;
  bool has_msg_ = false;
};

int main(int argc, char **argv) {
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<PcdPublisher>());
  rclcpp::shutdown();
  return 0;
}
