import ast

import pandas as pd
import re  
from collections import defaultdict  

def parse_line(line):
    # 假设行格式为：some_string,element1,element2,...
    # 去除行尾的换行符（如果有的话）
    diff_line = line.strip()
    # 使用ast.literal_eval来安全地解析字符串为元组
    tuple_data = ast.literal_eval(diff_line)
    # 第一个元素是字符串，第二个元素是列表（已经是列表了，不需要进一步处理）
    commit_id, diff_list = tuple_data
    return commit_id, diff_list

# 读取diff_files.txt并构建commit_id到数组长度的映射
commit_to_array_length = defaultdict(int)
with open('diff_files.txt', 'r') as file:
    for line in file:
        string_part, elements = parse_line(line)
        commit_to_array_length[string_part] = len(elements)

total_commit_radio = 0
total_class_radio = 0

for i in range(1,4):
    # 读取CSV文件并获取commit_id列表
    df_sdp = pd.read_csv(f'jgroups_sdp_commit_seed{i}.csv')
    sdp_commit_ids = df_sdp['commit_id'].tolist()
    df_pre = pd.read_csv(f'jgroups_test_pre_seed{i}.csv')
    pre_commit_ids = df_pre['commit_id'].tolist()

    # 计算jgroups_sdp_test_commit_seed1.csv中commit_id在diff_files.txt中对应的数组元素数量总和
    sdp_total_array_length = sum(commit_to_array_length[commit_id] for commit_id in sdp_commit_ids if commit_id in commit_to_array_length)

    # 计算jgroups_test_pre_seed1.csv中commit_id在diff_files.txt中对应的数组元素数量总和
    pre_total_array_length = sum(commit_to_array_length[commit_id] for commit_id in pre_commit_ids if commit_id in commit_to_array_length)

    # 如果其中一个总和为0，则不能计算比例，因为会导致除以零的错误
    if pre_total_array_length == 0:
        ratio = 0
    else:
        ratio = sdp_total_array_length / pre_total_array_length

    total_commit_radio += len(sdp_commit_ids) / len(pre_commit_ids)
    total_class_radio += ratio

    # 输出结果
    print(f"jgroups_sdp_commit_seed{i}.csv中总的数据量占jgroups_test_pre_seed{i}.csv总的数据量的比例为: {len(sdp_commit_ids) / len(pre_commit_ids):.2%}")
    print(f"jgroups_sdp_commit_seed{i}.csv中commit_id在diff_files.txt中对应的数组元素数量总和为: {sdp_total_array_length}")
    print(f"jgroups_test_pre_seed{i}.csv中commit_id在diff_files.txt中对应的数组元素数量总和为: {pre_total_array_length}")
    print(f"前者相对后者的比例为: {ratio:.2%}")

print(f"commit: {total_commit_radio/3}")
print(f"class: {total_class_radio/3}")
