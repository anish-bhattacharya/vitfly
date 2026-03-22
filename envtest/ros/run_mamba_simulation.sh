#!/bin/bash
#===============================================================================
# DroneMamba 完整仿真评估流程
#
# 用法：./run_mamba_simulation.sh [选项]
#
# 选项:
#   --model PATH      模型权重路径 (必需)
#   --episodes N      评估次数 (默认：50)
#   --timeout T       超时时间 (默认：60 秒)
#   --env ENV         环境名称 (默认：spheres)
#   --help            显示帮助
#===============================================================================

set -e

# 默认参数
MODEL_PATH=""
EPISODES=50
TIMEOUT=60
ENV_NAME="spheres"
OUTPUT_DIR="results/mamba_eval_$(date +%Y%m%d_%H%M%S)"

# 解析命令行参数
while [[ $# -gt 0 ]]; do
    case $1 in
        --model)
            MODEL_PATH="$2"
            shift 2
            ;;
        --episodes)
            EPISODES="$2"
            shift 2
            ;;
        --timeout)
            TIMEOUT="$2"
            shift 2
            ;;
        --env)
            ENV_NAME="$2"
            shift 2
            ;;
        --help)
            echo "用法：$0 --model PATH [选项]"
            echo ""
            echo "选项:"
            echo "  --model PATH      模型权重路径 (必需)"
            echo "  --episodes N      评估次数 (默认：50)"
            echo "  --timeout T       超时时间 (默认：60 秒)"
            echo "  --env ENV         环境名称 (默认：spheres)"
            echo "  --help            显示帮助"
            exit 0
            ;;
        *)
            echo "未知选项：$1"
            exit 1
            ;;
    esac
done

# 检查必需参数
if [ -z "$MODEL_PATH" ]; then
    echo "错误：必须指定模型路径 (--model PATH)"
    exit 1
fi

if [ ! -f "$MODEL_PATH" ]; then
    echo "错误：模型文件不存在：$MODEL_PATH"
    exit 1
fi

# 颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

print_header() {
    echo -e "\n${BLUE}========================================${NC}"
    echo -e "${BLUE}$1${NC}"
    echo -e "${BLUE}========================================${NC}\n"
}

print_success() { echo -e "${GREEN}✓ $1${NC}"; }
print_warning() { echo -e "${YELLOW}⚠ $1${NC}"; }
print_error() { echo -e "${RED}✗ $1${NC}"; }

#===============================================================================
# 开始执行
#===============================================================================

print_header "DroneMamba 完整仿真评估流程"

echo "配置参数:"
echo "  模型路径：   $MODEL_PATH"
echo "  评估次数：   $EPISODES"
echo "  超时时间：   ${TIMEOUT}s"
echo "  环境名称：   $ENV_NAME"
echo "  输出目录：   $OUTPUT_DIR"
echo ""

#-------------------------------------------------------------------------------
# 步骤 1: 检查 ROS 环境
#-------------------------------------------------------------------------------
print_header "步骤 1/5: 检查 ROS 环境"

if ! command -v roscore &> /dev/null; then
    print_error "ROS 未安装或未配置"
    echo "请确保已安装 ROS 并执行：source /opt/ros/noetic/setup.bash"
    exit 1
fi

print_success "ROS 已安装"

# 检查是否已经在运行
if pgrep -x "roscore" > /dev/null; then
    print_success "roscore 已在运行"
else
    print_warning "roscore 未运行，正在启动..."
    roscore &
    sleep 3
    print_success "roscore 已启动"
fi

#-------------------------------------------------------------------------------
# 步骤 2: 检查仿真环境
#-------------------------------------------------------------------------------
print_header "步骤 2/5: 检查仿真环境"

cd "$(dirname "$0")/../.."
source devel/setup.bash 2>/dev/null || true

# 检查环境 launch 文件
ENV_LAUNCH="envsim/launch/envsim.launch"
if [ -f "$ENV_LAUNCH" ]; then
    print_success "仿真环境 launch 文件存在"
else
    print_error "仿真环境 launch 文件不存在"
    exit 1
fi

#-------------------------------------------------------------------------------
# 步骤 3: 启动仿真环境
#-------------------------------------------------------------------------------
print_header "步骤 3/5: 启动仿真环境"

print_warning "将在 5 秒后启动仿真环境..."
sleep 5

# 启动仿真（后台）
print_success "启动仿真环境..."
roslaunch envsim envsim.launch world_name:=$ENV_NAME &
SIM_PID=$!

print_warning "等待仿真初始化 (10 秒)..."
sleep 10

# 检查仿真节点
if rostopic list | grep -q "/kingfisher/state"; then
    print_success "仿真节点已就绪"
else
    print_error "仿真节点未就绪，请手动检查"
    kill $SIM_PID 2>/dev/null
    exit 1
fi

#-------------------------------------------------------------------------------
# 步骤 4: 运行 DroneMamba 评估
#-------------------------------------------------------------------------------
print_header "步骤 4/5: 运行 DroneMamba 评估"

print_success "开始仿真评估..."
echo ""

# 运行评估脚本
python3 envtest/ros/run_mamba_competition.py \
    --model_path "$MODEL_PATH" \
    --num_episodes $EPISODES \
    --timeout $TIMEOUT \
    --output_dir "$OUTPUT_DIR"

EVAL_STATUS=$?

#-------------------------------------------------------------------------------
# 步骤 5: 清理和生成报告
#-------------------------------------------------------------------------------
print_header "步骤 5/5: 清理和生成报告"

# 停止仿真
print_warning "停止仿真环境..."
kill $SIM_PID 2>/dev/null || true
sleep 2
print_success "仿真已停止"

# 检查评估结果
if [ $EVAL_STATUS -eq 0 ]; then
    print_success "评估完成！"

    # 显示统计结果
    if [ -f "$OUTPUT_DIR/statistics.json" ]; then
        echo ""
        print_header "评估统计"
        cat "$OUTPUT_DIR/statistics.json"
        echo ""
    fi

    # 生成对比图表
    if [ -f "compare_models.py" ]; then
        print_success "生成对比图表..."
        python3 compare_models.py --output_dir "$OUTPUT_DIR/plots"
    fi

    print_success "所有结果已保存至：$OUTPUT_DIR"
else
    print_error "评估失败 (状态码：$EVAL_STATUS)"
fi

#-------------------------------------------------------------------------------
# 完成
#-------------------------------------------------------------------------------
print_header "仿真评估完成！"

echo "输出文件:"
echo "  统计数据：   $OUTPUT_DIR/statistics.json"
echo "  详细日志：   $OUTPUT_DIR/episode_*.csv"
echo "  评估报告：   $OUTPUT_DIR/evaluation_report.txt"
echo ""

print_success "所有步骤完成！"
