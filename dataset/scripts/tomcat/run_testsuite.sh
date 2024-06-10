#!/bin/bash


if [ "$#" -ne 1 ]; then
    echo "Usage: $0 <Target Class>"
    exit 1
fi



TARGET_CLASS=$1

PACKAGE_PATH="${TARGET_CLASS%.java}"  

 
if [[ "$PACKAGE_PATH" == java/* ]]; then  
    PACKAGE_PATH="${PACKAGE_PATH#java/}"  
fi  

TEST_PATH="evosuite-tests/${PACKAGE_PATH}"_ESTest.class

echo "$TEST_PATH"

if [ -f "$TEST_PATH" ]; then

  DOT_PACKAGE_PATH="${PACKAGE_PATH//\//.}"

  PROJECT_CP="java/"


  DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"


  cd "$DIR" || exit


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


