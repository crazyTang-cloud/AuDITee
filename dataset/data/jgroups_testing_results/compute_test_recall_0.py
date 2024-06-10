import pandas as pd  

total_r0 = 0

for i in range(1, 4):
    # 读取CSV文件
    df_sdp = pd.read_csv(f'jgroups_sdp_test_commit_seed{i}.csv')
    df_pre = pd.read_csv(f'jgroups_test_pre_seed{i}.csv')

    # 筛选出status为pass的行
    df_pass = df_pre[df_pre['status'] == 'pass']

    # 找出这些行的commit_id是否不在jgroups_sdp_test_commit_seed1.csv中
    # 使用~操作符取反isin方法的结果
    not_matching_commit_ids = ~df_pass['commit_id'].isin(df_sdp['commit_id'])

    # 计算不在jgroups_sdp_test_commit_seed1.csv中的commit_id的比例
    proportion = not_matching_commit_ids.sum() / len(df_pass)

    total_r0 += proportion
    print(f"当jgroups_test_pre_seed{i}.csv中status为pass时，commit_id不在jgroups_sdp_test_commit_seed{i}.csv中的比例为: {proportion:.2%}")

print(total_r0/3)
