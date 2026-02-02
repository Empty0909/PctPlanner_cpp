// semantic_io.cpp
// 语义地图读写模块实现
#include "semantic_io.hpp"

#include <fstream>
#include <iomanip>
#include <nlohmann/json.hpp>
#include <stdexcept>

namespace pctplanner {

using json = nlohmann::json;

// 从 JSON 对象解析 Trajectory
static Trajectory parse_trajectory(const json &obj) {
  Trajectory traj;
  if (obj.contains("waypoints")) {
    for (const auto &wp : obj["waypoints"]) {
      if (wp.is_array() && wp.size() >= 3) {
        traj.waypoints.push_back(
            {wp[0].get<double>(), wp[1].get<double>(), wp[2].get<double>()});
      }
    }
  }
  traj.num_waypoints =
      obj.contains("num_waypoints") ? obj["num_waypoints"].get<int>() : 0;
  traj.total_distance = obj.contains("total_distance")
                            ? obj["total_distance"].get<double>()
                            : 0.0;
  return traj;
}

// 从 JSON 对象解析 JsonNode
static JsonNode parse_node(const json &obj) {
  JsonNode node;
  node.id = obj.value("id", 0);
  node.label = obj.value("label", "");
  if (obj.contains("position") && obj["position"].is_array()) {
    for (const auto &v : obj["position"]) {
      node.position.push_back(v.get<double>());
    }
  }
  node.point_index = obj.value("point_index", 0);
  node.timestamp = obj.value("timestamp", "");
  return node;
}

// 从 JSON 对象解析 Edge
static Edge parse_edge(const json &obj) {
  Edge edge;
  edge.source = obj.value("source", 0);
  edge.target = obj.value("target", 0);
  if (obj.contains("trajectory")) {
    edge.trajectory = parse_trajectory(obj["trajectory"]);
  }
  return edge;
}

SemanticMap load_semantic_json(const std::string &path) {
  std::ifstream file(path);
  if (!file.is_open()) {
    throw std::runtime_error("Failed to open semantic JSON file: " + path);
  }

  json raw;
  file >> raw;

  SemanticMap map;
  map.model_file = raw.value("model_file", "");

  if (raw.contains("nodes") && raw["nodes"].is_array()) {
    for (const auto &n : raw["nodes"]) {
      map.nodes.push_back(parse_node(n));
    }
  }

  if (raw.contains("edges") && raw["edges"].is_array()) {
    for (const auto &e : raw["edges"]) {
      map.edges.push_back(parse_edge(e));
    }
  }

  // 关联边的起点和终点位置
  map.wire_edges();

  return map;
}

// Trajectory 序列化为 JSON
static json trajectory_to_json(const Trajectory &traj) {
  json obj;
  json waypoints_arr = json::array();
  for (const auto &wp : traj.waypoints) {
    waypoints_arr.push_back(wp);
  }
  obj["waypoints"] = waypoints_arr;
  obj["num_waypoints"] = traj.num_waypoints;
  obj["total_distance"] = traj.total_distance;
  return obj;
}

// JsonNode 序列化为 JSON
static json node_to_json(const JsonNode &node) {
  json obj;
  obj["id"] = node.id;
  obj["label"] = node.label;
  obj["position"] = node.position;
  obj["point_index"] = node.point_index;
  obj["timestamp"] = node.timestamp;
  return obj;
}

// Edge 序列化为 JSON
static json edge_to_json(const Edge &edge) {
  json obj;
  obj["source"] = edge.source;
  obj["target"] = edge.target;
  obj["trajectory"] = trajectory_to_json(edge.trajectory);
  return obj;
}

void write_semantic_json(const SemanticMap &graph, const std::string &path) {
  json payload;
  payload["model_file"] = graph.model_file;

  json nodes_arr = json::array();
  for (const auto &n : graph.nodes) {
    nodes_arr.push_back(node_to_json(n));
  }
  payload["nodes"] = nodes_arr;

  json edges_arr = json::array();
  for (const auto &e : graph.edges) {
    edges_arr.push_back(edge_to_json(e));
  }
  payload["edges"] = edges_arr;

  std::ofstream file(path);
  if (!file.is_open()) {
    throw std::runtime_error("Failed to write semantic JSON file: " + path);
  }
  file << std::setw(2) << payload << std::endl;
}

} // namespace pctplanner
