#!/bin/bash
# PctPlanner C++ 版本构建脚本
# 使用方法: ./build.sh [clean|release|debug]

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD_DIR="${SCRIPT_DIR}/build"
BUILD_TYPE="${1:-Release}"

# 处理参数
case "$1" in
    clean)
        echo "清理构建目录..."
        rm -rf "${BUILD_DIR}/cmake_build"
        echo "清理完成"
        exit 0
        ;;
    release)
        BUILD_TYPE="Release"
        ;;
    debug)
        BUILD_TYPE="Debug"
        ;;
    "")
        BUILD_TYPE="Release"
        ;;
    *)
        echo "用法: $0 [clean|release|debug]"
        exit 1
        ;;
esac

# 创建 cmake 构建目录（与 colcon 分开）
CMAKE_BUILD_DIR="${BUILD_DIR}/cmake_build"
mkdir -p "${CMAKE_BUILD_DIR}"

echo "=================================================="
echo "PctPlanner C++ 构建"
echo "=================================================="
echo "构建目录: ${CMAKE_BUILD_DIR}"
echo "构建类型: ${BUILD_TYPE}"
echo ""

cd "${CMAKE_BUILD_DIR}"

# 配置
echo ">>> 配置 CMake..."
cmake "${SCRIPT_DIR}" \
    -DCMAKE_BUILD_TYPE="${BUILD_TYPE}" \
    -DCMAKE_EXPORT_COMPILE_COMMANDS=ON

# 构建
echo ""
echo ">>> 编译..."
make -j$(nproc)

echo ""
echo "=================================================="
echo "构建完成！"
echo "=================================================="
echo ""
echo "可执行文件位置:"
echo "  - tomography_node:      ${CMAKE_BUILD_DIR}/tomography_node"
echo "  - planner_node:         ${CMAKE_BUILD_DIR}/planner_node"
echo "  - planner_direct_node:  ${CMAKE_BUILD_DIR}/planner_direct_node"
echo "  - pcd_publisher:        ${CMAKE_BUILD_DIR}/pcd_publisher"
echo "  - tomography_benchmark: ${CMAKE_BUILD_DIR}/tomography_benchmark"
echo ""
echo "运行测试:"
echo "  cd ${CMAKE_BUILD_DIR} && ctest"
