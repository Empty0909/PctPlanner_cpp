// semantic_io.hpp
// 语义地图读写模块 - 与 Python ziquan-explore 分支的 semantic_io.py 对齐
#pragma once

#include <Eigen/Core>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <nav_msgs/msg/path.hpp>

#include <cmath>
#include <string>
#include <unordered_map>
#include <vector>

namespace pctplanner {

// ---------- Trajectory ----------
struct Trajectory {
  std::vector<std::vector<double>> waypoints;
  int num_waypoints = 0;
  double total_distance = 0.0;

  Trajectory() = default;

  void append_point(double x, double y, double z) {
    waypoints.push_back({x, y, z});
    num_waypoints++;
    if (num_waypoints > 1) {
      const auto &p0 = waypoints[num_waypoints - 2];
      const auto &p1 = waypoints[num_waypoints - 1];
      double dx = p1[0] - p0[0];
      double dy = p1[1] - p0[1];
      double dz = p1[2] - p0[2];
      total_distance += std::sqrt(dx * dx + dy * dy + dz * dz);
    }
  }

  void clear() {
    waypoints.clear();
    num_waypoints = 0;
    total_distance = 0.0;
  }

  nav_msgs::msg::Path to_ros() const {
    nav_msgs::msg::Path path_msg;
    path_msg.header.frame_id = "map";
    for (const auto &wp : waypoints) {
      geometry_msgs::msg::PoseStamped pose;
      pose.header.frame_id = "map";
      pose.pose.position.x = wp[0];
      pose.pose.position.y = wp[1];
      pose.pose.position.z = wp[2];
      pose.pose.orientation.w = 1.0;
      path_msg.poses.push_back(pose);
    }
    return path_msg;
  }
};

// ---------- JsonNode ----------
struct JsonNode {
  int id = 0;
  std::string label;
  std::vector<double> position; // [x, y, z]
  int point_index = 0;
  std::string timestamp;

  Eigen::Vector3f pos_array() const {
    return Eigen::Vector3f(static_cast<float>(position[0]),
                           static_cast<float>(position[1]),
                           static_cast<float>(position[2]));
  }
};

// ---------- Edge ----------
struct Edge {
  int source = 0;
  int target = 0;
  Trajectory trajectory;

  // 起点和终点位置（在加载后填充）
  Eigen::Vector3f start_pos = Eigen::Vector3f::Zero();
  Eigen::Vector3f end_pos = Eigen::Vector3f::Zero();

  void append_point(double x, double y, double z) {
    trajectory.append_point(x, y, z);
  }

  void clear() { trajectory.clear(); }

  nav_msgs::msg::Path traj2ros() const { return trajectory.to_ros(); }
};

// ---------- SemanticMap ----------
struct SemanticMap {
  std::string model_file;
  std::vector<JsonNode> nodes;
  std::vector<Edge> edges;

  // 在加载后调用，将 edge 的 start_pos/end_pos 关联到 nodes
  void wire_edges() {
    std::unordered_map<int, size_t> id2idx;
    for (size_t i = 0; i < nodes.size(); ++i) {
      id2idx[nodes[i].id] = i;
    }
    for (auto &e : edges) {
      if (id2idx.count(e.source))
        e.start_pos = nodes[id2idx[e.source]].pos_array();
      if (id2idx.count(e.target))
        e.end_pos = nodes[id2idx[e.target]].pos_array();
    }
  }

  void clear_edge_trajectory_by_idx(size_t edge_idx) {
    if (edge_idx < edges.size()) {
      edges[edge_idx].clear();
    }
  }

  void append_point_by_edge_idx(size_t edge_idx, double x, double y, double z) {
    if (edge_idx < edges.size()) {
      edges[edge_idx].append_point(x, y, z);
    }
  }
};

// ---------- I/O 函数 ----------
// 从 JSON 文件加载语义地图
SemanticMap load_semantic_json(const std::string &path);

// 将语义地图写入 JSON 文件
void write_semantic_json(const SemanticMap &graph, const std::string &path);

} // namespace pctplanner
