#!/bin/bash

# 检查是否提供了足够的参数
if [ "$#" -ne 2 ]; then
    echo "Usage: $0 <Target Class> <Seed>"
    exit 1
fi

# 获取测试类作为参数
TARGET_CLASS=$1

# 获取种子作为第二个参数
SEED=$2

# 定义EvoSuite的其他参数
EVOSUITE_JAR_PATH="./evosuite-1.0.6.jar"  # EvoSuite JAR文件的路径
PROJECT_CP="./target/classes"     # 项目的类路径

# 获取脚本所在的绝对路径
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

# 切换到脚本所在的目录
cd "$DIR" || exit

# 使用EvoSuite生成测试用例
java -jar "$EVOSUITE_JAR_PATH" -projectCP "$PROJECT_CP" -class="$TARGET_CLASS" -Dsearch_budget=60 -Dminimize=false -Dassertion_strategy=all -Drandom_seed="$SEED"
#java -jar "$EVOSUITE_JAR_PATH" -projectCP "$PROJECT_CP" -class="$TARGET_CLASS" -Dsearch_budget=60 -Dminimize=false -Dassertion_strategy=all


# 检查EvoSuite的退出状态
if [ $? -ne 0 ]; then
    echo "EvoSuite执行失败，请检查日志和参数配置。"
    exit 1
fi

echo "EvoSuite执行成功。"