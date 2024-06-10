#!/bin/bash


# if [ "$#" -ne 1 ]; then
#     echo "Usage: $0 <Target Class>"
#     exit 1
# fi

# Check if a file path is provided

if [ "$#" -ne 2 ]; then
    echo "Usage: $0 <Target Class> <Seed>"
    exit 1
fi

find . -type f -name "*.class" -exec rm -f {} \;


TARGET_CLASS=$1

SEED=$2

PACKAGE_PATH="${TARGET_CLASS%.java}"  
  

PATHS=""  
  

IFS='/' read -ra ADDR <<< "$PACKAGE_PATH"  
  

for ((i=0; i<${#ADDR[@]}-1; i++)); do  

    CURRENT_PATH=$(IFS=:/; echo "${ADDR[@]:0:$i+1}" | xargs printf "%s/")  

    if [[ -n "$PATHS" ]]; then  
        PATHS+=":$CURRENT_PATH"  
    else  

        PATHS="$CURRENT_PATH"  
    fi  
done 

# echo "$PATHS"



EVOSUITE_JAR_PATH="./evosuite-1.0.6.jar" 



DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"


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


 
DOT_PATHS=""  
  
 
if [[ "$PACKAGE_PATH" == java/* ]]; then  
    PACKAGE_PATH="${PACKAGE_PATH#java/}"  
fi  
  

DOT_PACKAGE_PATH="${PACKAGE_PATH//\//.}" 

#echo "$DOT_PACKAGE_PATH"

#echo "java -jar "$EVOSUITE_JAR_PATH" -projectCP "$PATHS" -class "$DOT_PACKAGE_PATH" -Dsearch_budget=60 -Dminimize=false -Dassertion_strategy=all -Drandom_seed=1"

java -jar "$EVOSUITE_JAR_PATH" -projectCP classes -class "$DOT_PACKAGE_PATH" -Dsearch_budget=60 -Dminimize=false -Dassertion_strategy=all -Drandom_seed="$SEED"

DOT_PACKAGE_PATH="${PACKAGE_PATH}_ESTest.java"

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
