#!/bin/bash

# 检查是否提供了足够的参数
# if [ "$#" -ne 1 ]; then
#     echo "Usage: $0 <Target Class>"
#     exit 1
# fi

# Check if a file path is provided
# 检查是否提供了足够的参数
if [ "$#" -ne 2 ]; then
    echo "Usage: $0 <Target Class> <Seed>"
    exit 1
fi

find . -type f -name "*.class" -exec rm -f {} \;

# 获取测试类作为参数
TARGET_CLASS=$1

SEED=$2
# 移除文件扩展名，只保留包路径  
PACKAGE_PATH="${TARGET_CLASS%.java}"  
  
# 初始化空字符串来保存路径  
PATHS=""  
  
# 使用IFS（内部字段分隔符）将包路径按'/'分割  
IFS='/' read -ra ADDR <<< "$PACKAGE_PATH"  
  
# 遍历数组，构建所有父目录路径并拼接到PATHS变量中  
for ((i=0; i<${#ADDR[@]}-1; i++)); do  
    # 拼接当前索引之前的所有元素来构建路径  
    CURRENT_PATH=$(IFS=:/; echo "${ADDR[@]:0:$i+1}" | xargs printf "%s/")  
    # 如果PATHS非空，则添加冒号和路径  
    if [[ -n "$PATHS" ]]; then  
        PATHS+=":$CURRENT_PATH"  
    else  
        # 如果是第一个路径，则直接赋值  
        PATHS="$CURRENT_PATH"  
    fi  
done 

# echo "$PATHS"
# 获取种子作为第二个参数
#SEED=$2

# 定义EvoSuite的其他参数
EVOSUITE_JAR_PATH="./evosuite-1.0.6.jar"  # EvoSuite JAR文件的路径


# 获取脚本所在的绝对路径
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

# 切换到脚本所在的目录
cd "$DIR" || exit

repo="."
declare -A seen_jars  # Declare an associative array to store seen jar file names
jar_paths=""

# Use find command to traverse all .jar files in the Maven repository and record their paths
while IFS= read -r -d '' jar_file; do
    # Check if the jar file path contains a space
    if [[ "$jar_file" =~ \  ]]; then
        continue  # Skip adding paths with spaces
    fi

    jar_basename=$(basename "$jar_file")  # Extract the base name of the jar file

    # If this jar file name has not been seen before, add its path
    if [[ -z "${seen_jars[$jar_basename]}" ]]; then
        seen_jars[$jar_basename]=1  # Mark this jar file name as seen
        jar_paths="$jar_paths:$jar_file"
    fi
done < <(find "$repo" -type f -name "*.jar" -print0)

# Remove the leading colon
jar_paths=${jar_paths#:}

echo "$jar_paths":"$PATHS" > jar_paths.txt

# Export the jar paths to jar_paths.txt
# echo "$jar_paths"

# echo "javac -cp "$jar_paths":"$PATHS" "$TARGET_CLASS""

javac -cp "$jar_paths":"$PATHS":classes/ "$TARGET_CLASS" -d classes


# 初始化空字符串来保存路径  
DOT_PATHS=""  
  
# 去除开头的"java/"（如果存在）  
if [[ "$PACKAGE_PATH" == java/* ]]; then  
    PACKAGE_PATH="${PACKAGE_PATH#java/}"  
fi  
  
# 将斜杠（/）替换为点（.）  
DOT_PACKAGE_PATH="${PACKAGE_PATH//\//.}" 

#echo "$DOT_PACKAGE_PATH"

#echo "java -jar "$EVOSUITE_JAR_PATH" -projectCP "$PATHS" -class "$DOT_PACKAGE_PATH" -Dsearch_budget=60 -Dminimize=false -Dassertion_strategy=all -Drandom_seed=1"

java -jar "$EVOSUITE_JAR_PATH" -projectCP classes -class "$DOT_PACKAGE_PATH" -Dsearch_budget=60 -Dminimize=false -Dassertion_strategy=all -Drandom_seed="$SEED"

DOT_PACKAGE_PATH="${PACKAGE_PATH}_ESTest.java"  # 在末尾添加_ESTest.java

#echo "$DOT_PACKAGE_PATH"

#echo "javac  -cp "$jar_paths":"$PATHS":evosuite-standalone-runtime-1.0.6.jar:hamcrest-core-1.3.jar:junit-4.12.jar:evosuite-tests/ "evosuite-tests/$DOT_PACKAGE_PATH""

TEST_PATH=evosuite-tests/"$DOT_PACKAGE_PATH"

echo "$TEST_PATH"

if [ -f "$TEST_PATH" ]; then  
    javac -cp "$jar_paths":classes:evosuite-standalone-runtime-1.0.6.jar:hamcrest-core-1.3.jar:junit-4.12.jar:evosuite-tests/ "$TEST_PATH"
else  
    echo "build fail"
    exit 1  
fi

#javac  -cp "$jar_paths":classes:evosuite-standalone-runtime-1.0.6.jar:hamcrest-core-1.3.jar:junit-4.12.jar:evosuite-tests/ "evosuite-tests/$DOT_PACKAGE_PATH"

echo "EvoSuite run pass"
