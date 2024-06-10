#!/bin/bash

# 检查是否提供了足够的参数
if [ "$#" -ne 3 ]; then
    echo "Usage: $0 <path> <Target Class> <seed>"
    exit 1
fi

CLASS_PATH=$1

# 获取测试类作为参数
TARGET_CLASS=$2

SEED=$3


# 获取种子作为第二个参数
#SEED=$2

# 定义EvoSuite的其他参数
EVOSUITE_JAR_PATH="./evosuite-1.0.6.jar"  # EvoSuite JAR文件的路径
PROJECT_CP="./$CLASS_PATH/target/classes/"     # 项目的类路径


# 获取脚本所在的绝对路径
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

# 切换到脚本所在的目录
cd "$DIR" || exit

# 读取jar_paths.txt文件并设置CLASSPATH
jar_paths=$(cat jar_paths.txt | tr '\n' ':')

# 使用EvoSuite生成测试用例
#java -jar "$EVOSUITE_JAR_PATH" -projectCP "$PROJECT_CP" -class="$TARGET_CLASS" -Dsearch_budget=60 -Dminimize=false -Dassertion_strategy=all -Drandom_seed="$SEED"
#java -jar "$EVOSUITE_JAR_PATH" -projectCP "$PROJECT_CP" -class="$TARGET_CLASS" -Dsearch_budget=10 -Dminimize=false -Dassertion_strategy=all -libraryPath="$jar_paths"

#echo "java -jar "$EVOSUITE_JAR_PATH" -projectCP "$jar_paths" -class "$TARGET_CLASS" -Dsearch_budget=10 -Dminimize=false -Dassertion_strategy=all "

java -jar "$EVOSUITE_JAR_PATH" -projectCP "$PROJECT_CP":"$jar_paths" -class "$TARGET_CLASS" -Dsearch_budget=60 -Dminimize=false -Dassertion_strategy=all -Drandom_seed="$SEED"

#java -jar "$EVOSUITE_JAR_PATH" -projectCP "$PROJECT_CP" -class "$TARGET_CLASS" -Dsearch_budget=60 -Dminimize=false -Dassertion_strategy=all -Drandom_seed=1

NEW_TARGET_CLASS="${TARGET_CLASS//./\/}"  # 替换所有的.为/  
NEW_TARGET_CLASS="${NEW_TARGET_CLASS}_ESTest.java"  # 在末尾添加_ESTest.java

#echo "javac  -cp "$PROJECT_CP":evosuite-standalone-runtime-1.0.6.jar:hamcrest-core-1.3.jar:junit-4.12.jar:evosuite-tests/:"$jar_paths" "$NEW_TARGET_CLASS""

javac  -cp "$PROJECT_CP":evosuite-standalone-runtime-1.0.6.jar:hamcrest-core-1.3.jar:junit-4.12.jar:evosuite-tests/:"$jar_paths" "evosuite-tests/$NEW_TARGET_CLASS"

#javac  -cp "$PROJECT_CP":evosuite-standalone-runtime-1.0.6.jar:hamcrest-core-1.3.jar:junit-4.12.jar:evosuite-tests/ "evosuite-tests/$NEW_TARGET_CLASS"

#-libraryPath=admin:BroadleafCommerce:BroadleafCommerce:common:core/broadleaf-framework/#target/:intergration


# 检查EvoSuite的退出状态
if [ $? -ne 0 ]; then
    echo "EvoSuite执行失败，请检查日志和参数配置。"
    exit 1
fi

echo "EvoSuite执行成功。"
