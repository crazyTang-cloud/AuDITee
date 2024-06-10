#!/bin/bash

# check parameters
if [ "$#" -ne 3 ]; then
    echo "Usage: $0 <path> <Target Class> <seed>"
    exit 1
fi

CLASS_PATH=$1

# get target class
TARGET_CLASS=$2

SEED=$3



# define EvoSuite
EVOSUITE_JAR_PATH="./evosuite-1.0.6.jar"
PROJECT_CP="./$CLASS_PATH/target/classes/"


DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

cd "$DIR" || exit

# get CLASSPATH
jar_paths=$(cat jar_paths.txt | tr '\n' ':')

# EvoSuite generate tase cases
#java -jar "$EVOSUITE_JAR_PATH" -projectCP "$PROJECT_CP" -class="$TARGET_CLASS" -Dsearch_budget=60 -Dminimize=false -Dassertion_strategy=all -Drandom_seed="$SEED"
#java -jar "$EVOSUITE_JAR_PATH" -projectCP "$PROJECT_CP" -class="$TARGET_CLASS" -Dsearch_budget=10 -Dminimize=false -Dassertion_strategy=all -libraryPath="$jar_paths"

#echo "java -jar "$EVOSUITE_JAR_PATH" -projectCP "$jar_paths" -class "$TARGET_CLASS" -Dsearch_budget=10 -Dminimize=false -Dassertion_strategy=all "

java -jar "$EVOSUITE_JAR_PATH" -projectCP "$PROJECT_CP":"$jar_paths" -class "$TARGET_CLASS" -Dsearch_budget=60 -Dminimize=false -Dassertion_strategy=all -Drandom_seed="$SEED"

#java -jar "$EVOSUITE_JAR_PATH" -projectCP "$PROJECT_CP" -class "$TARGET_CLASS" -Dsearch_budget=60 -Dminimize=false -Dassertion_strategy=all -Drandom_seed=1

NEW_TARGET_CLASS="${TARGET_CLASS//./\/}" 
NEW_TARGET_CLASS="${NEW_TARGET_CLASS}_ESTest.java"

#echo "javac  -cp "$PROJECT_CP":evosuite-standalone-runtime-1.0.6.jar:hamcrest-core-1.3.jar:junit-4.12.jar:evosuite-tests/:"$jar_paths" "$NEW_TARGET_CLASS""

javac  -cp "$PROJECT_CP":evosuite-standalone-runtime-1.0.6.jar:hamcrest-core-1.3.jar:junit-4.12.jar:evosuite-tests/:"$jar_paths" "evosuite-tests/$NEW_TARGET_CLASS"

#javac  -cp "$PROJECT_CP":evosuite-standalone-runtime-1.0.6.jar:hamcrest-core-1.3.jar:junit-4.12.jar:evosuite-tests/ "evosuite-tests/$NEW_TARGET_CLASS"

#-libraryPath=admin:BroadleafCommerce:BroadleafCommerce:common:core/broadleaf-framework/#target/:intergration


if [ $? -ne 0 ]; then
    echo "EvoSuite generates error"
    exit 1
fi

echo "EvoSuite generate successful"
