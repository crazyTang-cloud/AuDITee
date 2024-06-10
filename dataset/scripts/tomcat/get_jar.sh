#!/bin/bash

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

# Export the jar paths to jar_paths.txt
echo "$jar_paths" > jar_paths.txt
