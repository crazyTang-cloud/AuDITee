#!/bin/bash

# 检查是否提供了两个参数
if [ "$#" -ne 2 ]; then
    echo "Usage: $0 pre_commit_id commit_id"
    exit 1
fi

# 获取脚本所在的绝对路径
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

# 切换到脚本所在的目录
cd "$DIR" || exit

# 获取参数
pre_commit_id=$1
commit_id=$2

# 改变目录到指定的位置
cd ../dataset/data.inuse/JGroups || { echo "Failed to change directory"; exit 1; }

# 调用Python脚本并传递参数
python test_process.py "$pre_commit_id" "$commit_id"
