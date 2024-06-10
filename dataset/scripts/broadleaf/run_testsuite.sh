#!/bin/bash

# 检查是否提供了足够的参数
if [ "$#" -ne 2 ]; then
    echo "Usage: $0 <path> <Target Class>"
    exit 1
fi

CLASS_PATH=$1

# 获取测试类作为参数
TARGET_CLASS=$2

PROJECT_CP="./$CLASS_PATH/target/classes/"     # 项目的类路径

# 获取脚本所在的绝对路径
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

# 切换到脚本所在的目录
cd "$DIR" || exit

# 读取jar_paths.txt文件并设置CLASSPATH
jar_paths=$(cat jar_paths.txt | tr '\n' ':')

jar_paths="$jar_paths":evosuite-standalone-runtime-1.0.6.jar:hamcrest-core-1.3.jar:junit-4.12.jar:evosuite-tests/:"$PROJECT_CP"

# 使用EvoSuite生成测试用例
#java -jar "$EVOSUITE_JAR_PATH" -projectCP "$PROJECT_CP" -class="$TARGET_CLASS" -Dsearch_budget=60 -Dminimize=false -Dassertion_strategy=all -Drandom_seed="$SEED"
#java -jar "$EVOSUITE_JAR_PATH" -projectCP "$PROJECT_CP" -class="$TARGET_CLASS" -Dsearch_budget=10 -Dminimize=false -Dassertion_strategy=all -libraryPath="$jar_paths"

#echo "java -cp "$jar_paths"  org.junit.runner.JUnitCore "$TARGET_CLASS""

java -cp "$jar_paths"  org.junit.runner.JUnitCore "$TARGET_CLASS"

#-libraryPath=admin:BroadleafCommerce:BroadleafCommerce:common:core/broadleaf-framework/#target/:intergration


# 检查EvoSuite的退出状态
if [ $? -ne 0 ]; then
    echo "EvoSuite执行失败，请检查日志和参数配置。"
    exit 1
fi

echo "EvoSuite执行成功。"
