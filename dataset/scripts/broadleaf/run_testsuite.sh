#!/bin/bash


if [ "$#" -ne 2 ]; then
    echo "Usage: $0 <path> <Target Class>"
    exit 1
fi

CLASS_PATH=$1


TARGET_CLASS=$2

PROJECT_CP="./$CLASS_PATH/target/classes/"


DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

cd "$DIR" || exit


jar_paths=$(cat jar_paths.txt | tr '\n' ':')

jar_paths="$jar_paths":evosuite-standalone-runtime-1.0.6.jar:hamcrest-core-1.3.jar:junit-4.12.jar:evosuite-tests/:"$PROJECT_CP"

#java -jar "$EVOSUITE_JAR_PATH" -projectCP "$PROJECT_CP" -class="$TARGET_CLASS" -Dsearch_budget=60 -Dminimize=false -Dassertion_strategy=all -Drandom_seed="$SEED"
#java -jar "$EVOSUITE_JAR_PATH" -projectCP "$PROJECT_CP" -class="$TARGET_CLASS" -Dsearch_budget=10 -Dminimize=false -Dassertion_strategy=all -libraryPath="$jar_paths"

#echo "java -cp "$jar_paths"  org.junit.runner.JUnitCore "$TARGET_CLASS""

java -cp "$jar_paths"  org.junit.runner.JUnitCore "$TARGET_CLASS"

#-libraryPath=admin:BroadleafCommerce:BroadleafCommerce:common:core/broadleaf-framework/#target/:intergration


if [ $? -ne 0 ]; then
    echo "test case run error"
    exit 1
fi

echo "test case run successfully"
