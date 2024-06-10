#!/bin/bash


if [ "$#" -ne 2 ]; then
    echo "Usage: $0 <Target Class> <Seed>"
    exit 1
fi


TARGET_CLASS=$1


SEED=$2

# define EvoSuite
EVOSUITE_JAR_PATH="./evosuite-1.0.6.jar"  # EvoSuite JAR path
PROJECT_CP="./target/classes"     # project class path


DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"


cd "$DIR" || exit

# EvoSuite generates test cases
java -jar "$EVOSUITE_JAR_PATH" -projectCP "$PROJECT_CP" -class="$TARGET_CLASS" -Dsearch_budget=60 -Dminimize=false -Dassertion_strategy=all -Drandom_seed="$SEED"
#java -jar "$EVOSUITE_JAR_PATH" -projectCP "$PROJECT_CP" -class="$TARGET_CLASS" -Dsearch_budget=60 -Dminimize=false -Dassertion_strategy=all



if [ $? -ne 0 ]; then
    echo "EvoSuite generate error"
    exit 1
fi

echo "EvoSuite generate successfully"
