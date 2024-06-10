import ast

import pandas as pd
import re  
from collections import defaultdict  

def parse_line(line):
    diff_line = line.strip()
    tuple_data = ast.literal_eval(diff_line)
    commit_id, diff_list = tuple_data
    return commit_id, diff_list

commit_to_array_length = defaultdict(int)
with open('diff_files.txt', 'r') as file:
    for line in file:
        string_part, elements = parse_line(line)
        commit_to_array_length[string_part] = len(elements)

total_commit_radio = 0
total_class_radio = 0

for i in range(1,4):
    df_sdp = pd.read_csv(f'jgroups_sdp_commit_seed{i}.csv')
    sdp_commit_ids = df_sdp['commit_id'].tolist()
    df_pre = pd.read_csv(f'jgroups_test_pre_seed{i}.csv')
    pre_commit_ids = df_pre['commit_id'].tolist()

    sdp_total_array_length = sum(commit_to_array_length[commit_id] for commit_id in sdp_commit_ids if commit_id in commit_to_array_length)

    pre_total_array_length = sum(commit_to_array_length[commit_id] for commit_id in pre_commit_ids if commit_id in commit_to_array_length)

    if pre_total_array_length == 0:
        ratio = 0
    else:
        ratio = sdp_total_array_length / pre_total_array_length

    total_commit_radio += len(sdp_commit_ids) / len(pre_commit_ids)
    total_class_radio += ratio



print(f"radio of code changes: {total_commit_radio/3}")
print(f"radio of classes: {total_class_radio/3}")
