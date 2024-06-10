#!/bin/bash  
  
repo="."  
declare -A seen_jars  # 声明一个关联数组来存储已经看到的jar包名  
jar_paths=""  
  
# 使用 find 命令遍历 Maven 仓库中的所有 .jar 文件，并将路径记录下来  
while IFS= read -r -d '' jar_file; do  
    jar_basename=$(basename "$jar_file")  # 提取jar包的基本名称（不包含路径）  
  
    # 如果这个jar包名还没有被看到，则添加它  
    if [[ -z "${seen_jars[$jar_basename]}" ]]; then  
        seen_jars[$jar_basename]=1  # 标记这个jar包名已经被看到  
        jar_paths="$jar_paths:$jar_file"  
    fi  
done < <(find "$repo" -type f -name "*.jar" -print0)  
  
# 去除开头的冒号  
jar_paths=${jar_paths#:}  
 
#export CLASSPATH="$jar_paths"
# 将最终的 jar_paths 写入到 jar_paths.txt 文件中  
echo "$jar_paths" > jar_paths.txt
