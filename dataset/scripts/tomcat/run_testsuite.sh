#!/bin/bash

# 检查是否提供了足够的参数
if [ "$#" -ne 1 ]; then
    echo "Usage: $0 <Target Class>"
    exit 1
fi


# 获取测试类作为参数
TARGET_CLASS=$1

PACKAGE_PATH="${TARGET_CLASS%.java}"  

# 去除开头的"java/"（如果存在）  
if [[ "$PACKAGE_PATH" == java/* ]]; then  
    PACKAGE_PATH="${PACKAGE_PATH#java/}"  
fi  

TEST_PATH="evosuite-tests/${PACKAGE_PATH}"_ESTest.class

echo "$TEST_PATH"

if [ -f "$TEST_PATH" ]; then
  # 将斜杠（/）替换为点（.）
  DOT_PACKAGE_PATH="${PACKAGE_PATH//\//.}"

  PROJECT_CP="java/"     # 项目的类路径

  # 获取脚本所在的绝对路径
  DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

  # 切换到脚本所在的目录
  cd "$DIR" || exit

  # 读取jar_paths.txt文件并设置CLASSPATH
  jar_paths=$(cat jar_paths.txt | tr '\n' ':')

  javac -cp "$jar_paths":classes/ "$TARGET_CLASS" -d classes

  jar_paths=evosuite-standalone-runtime-1.0.6.jar:hamcrest-core-1.3.jar:junit-4.12.jar:evosuite-tests/:classes/:"$jar_paths"

  #echo "$jar_paths"

  #javac -cp "$jar_paths" "$TARGET_CLASS" -d classes

  if [ $? -ne 0 ]; then
    echo "build fail"
    exit 1
  fi


  java -cp "$jar_paths":"$PROJECT_CP"  org.junit.runner.JUnitCore "$DOT_PACKAGE_PATH"_ESTest

  echo "EvoSuite run pass"
else   
  echo "build fail"
  exit 1
fi


