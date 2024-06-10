#!/bin/bash  
  
repo="."  
declare -A seen_jars 
jar_paths=""  
  

while IFS= read -r -d '' jar_file; do  
    jar_basename=$(basename "$jar_file")
  
 
    if [[ -z "${seen_jars[$jar_basename]}" ]]; then  
        seen_jars[$jar_basename]=1  
        jar_paths="$jar_paths:$jar_file"  
    fi  
done < <(find "$repo" -type f -name "*.jar" -print0)  
  

jar_paths=${jar_paths#:}  
 
#export CLASSPATH="$jar_paths"

echo "$jar_paths" > jar_paths.txt
