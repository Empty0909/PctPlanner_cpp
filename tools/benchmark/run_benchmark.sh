#!/bin/bash
#
# PctPlanner C++ vs Python 版本对比测试一键执行脚本
#
# 用法:
#   ./run_benchmark.sh [选项]
#
# 选项:
#   --limit N       限制测试用例数量（默认全部）
#   --generate N    重新生成 N 个测试用例
#   --clean         清理旧结果文件
#   --help          显示帮助信息
#
# 示例:
#   ./run_benchmark.sh --limit 100          # 只测试前 100 个用例
#   ./run_benchmark.sh --generate 1000      # 重新生成 1000 个测试用例并测试
#   ./run_benchmark.sh --clean              # 清理后重新测试
#

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# 目录结构
DATA_DIR="$SCRIPT_DIR/data"
RESULTS_DIR="$SCRIPT_DIR/results"

# 默认测试用例文件
TEST_CASES_CSV="$DATA_DIR/test_cases_10k.csv"
TEST_CASES_JSON="$DATA_DIR/test_cases_10k.json"

# 库路径
PYTHON_LIB_PATH="/home/lzy/PctPlanner/PctPlanner_py/planner/lib"
PYTHON_GTSAM_PATH="/home/lzy/PctPlanner/PctPlanner_py/planner/lib/3rdparty/gtsam-4.1.1/install/lib"
CPP_LIB_PATH="/home/lzy/PctPlanner/PctPlanner_Cpp/planner_lib"
CPP_GTSAM_PATH="/home/lzy/PctPlanner/PctPlanner_Cpp/planner_lib/3rdparty/gtsam-4.1.1/install/lib"

# 颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 显示帮助
show_help() {
    head -20 "$0" | tail -18
    exit 0
}

# 解析参数
LIMIT=""
GENERATE=""
CLEAN=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --limit)
            LIMIT="--limit $2"
            shift 2
            ;;
        --generate)
            GENERATE="$2"
            shift 2
            ;;
        --clean)
            CLEAN=true
            shift
            ;;
        --help|-h)
            show_help
            ;;
        *)
            echo -e "${RED}未知参数: $1${NC}"
            echo "使用 --help 查看帮助"
            exit 1
            ;;
    esac
done

echo ""
echo -e "${BLUE}════════════════════════════════════════════════════════════${NC}"
echo -e "${BLUE}         PctPlanner C++ vs Python 版本对比测试${NC}"
echo -e "${BLUE}════════════════════════════════════════════════════════════${NC}"
echo ""

# 创建目录结构
mkdir -p "$DATA_DIR" "$RESULTS_DIR"

# 清理旧结果
if [ "$CLEAN" = true ]; then
    echo -e "${YELLOW}[清理] 删除旧结果文件...${NC}"
    rm -f "$RESULTS_DIR"/*.json "$RESULTS_DIR"/*.md
    echo -e "${GREEN}清理完成 ✓${NC}"
    echo ""
fi

# ========== Step 1: 检查环境 ==========
echo -e "${YELLOW}[1/5] 检查环境...${NC}"

# 检查 Python 依赖
for pkg in numpy scipy; do
    if ! python3 -c "import $pkg" 2>/dev/null; then
        echo -e "${RED}错误: 未找到 $pkg，请安装: pip3 install $pkg${NC}"
        exit 1
    fi
done

# 检查库文件
if [ ! -d "$PYTHON_LIB_PATH" ]; then
    echo -e "${RED}错误: Python 库路径不存在: $PYTHON_LIB_PATH${NC}"
    exit 1
fi

if [ ! -d "$CPP_LIB_PATH" ]; then
    echo -e "${RED}错误: C++ 库路径不存在: $CPP_LIB_PATH${NC}"
    exit 1
fi

echo -e "${GREEN}  Python 库: $PYTHON_LIB_PATH ✓${NC}"
echo -e "${GREEN}  C++ 库:    $CPP_LIB_PATH ✓${NC}"
echo ""

# ========== Step 2: 生成测试数据 ==========
echo -e "${YELLOW}[2/5] 准备测试数据...${NC}"

if [ -n "$GENERATE" ]; then
    echo "  生成 $GENERATE 个新测试用例..."
    python3 generate_test_cases.py \
        --num-cases "$GENERATE" \
        --output-csv "$TEST_CASES_CSV" \
        --output-json "$TEST_CASES_JSON"
elif [ ! -f "$TEST_CASES_CSV" ]; then
    echo "  未找到测试数据，生成默认 10000 个用例..."
    python3 generate_test_cases.py \
        --num-cases 10000 \
        --output-csv "$TEST_CASES_CSV" \
        --output-json "$TEST_CASES_JSON"
else
    CASE_COUNT=$(tail -n +2 "$TEST_CASES_CSV" | wc -l)
    echo -e "${GREEN}  使用现有测试数据: $CASE_COUNT 个用例 ✓${NC}"
fi
echo ""

# ========== Step 3: 运行 Python 版本测试 ==========
echo -e "${YELLOW}[3/5] 运行 Python 版本测试...${NC}"

# 使用批量模式避免内存泄漏
export LD_LIBRARY_PATH="$PYTHON_LIB_PATH:$PYTHON_GTSAM_PATH:$LD_LIBRARY_PATH"
python3 run_batch.py \
    --version python \
    --input "$TEST_CASES_CSV" \
    --output "$RESULTS_DIR/python_version_results.json" \
    --batch-size 50 \
    $LIMIT
PYTHON_EXIT=$?

if [ $PYTHON_EXIT -ne 0 ]; then
    echo -e "${YELLOW}Python 版本测试遇到问题 (exit code: $PYTHON_EXIT)，继续执行...${NC}"
fi

echo ""

# ========== Step 4: 运行 C++ 版本测试 ==========
echo -e "${YELLOW}[4/5] 运行 C++ 版本测试...${NC}"

# 使用批量模式避免内存泄漏
export LD_LIBRARY_PATH="$CPP_LIB_PATH:$CPP_GTSAM_PATH:$LD_LIBRARY_PATH"
python3 run_batch.py \
    --version cpp \
    --input "$TEST_CASES_CSV" \
    --output "$RESULTS_DIR/cpp_version_results.json" \
    --batch-size 50 \
    $LIMIT
CPP_EXIT=$?

if [ $CPP_EXIT -ne 0 ]; then
    echo -e "${YELLOW}C++ 版本测试遇到问题 (exit code: $CPP_EXIT)，继续执行...${NC}"
fi

echo ""

# ========== Step 5: 对比分析 ==========
echo -e "${YELLOW}[5/5] 对比分析...${NC}"

python3 compare_versions.py \
    --python "$RESULTS_DIR/python_version_results.json" \
    --cpp "$RESULTS_DIR/cpp_version_results.json" \
    --output "$RESULTS_DIR/version_comparison_report.md"

echo ""
echo -e "${BLUE}════════════════════════════════════════════════════════════${NC}"
echo -e "${GREEN}测试完成！${NC}"
echo ""
echo "结果文件:"
echo "  - Python 结果: $RESULTS_DIR/python_version_results.json"
echo "  - C++ 结果:    $RESULTS_DIR/cpp_version_results.json"  
echo "  - 对比报告:    $RESULTS_DIR/version_comparison_report.md"
echo ""
echo -e "${BLUE}════════════════════════════════════════════════════════════${NC}"
