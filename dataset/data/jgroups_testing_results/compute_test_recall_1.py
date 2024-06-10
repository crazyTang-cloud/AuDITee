import pandas as pd  

total_r1 = 0

for i in range(1, 4):
    # 读取CSV文件
    df_sdp = pd.read_csv(f'jgroups_sdp_test_commit_seed{i}.csv')
    df_pre = pd.read_csv(f'jgroups_test_pre_seed{i}.csv')

    # 筛选出status为fail的行
    df_fail = df_pre[df_pre['status'] == 'fail']

    # 找出这些行在jgroups_sdp_test_commit_seed1.csv中的对应项
    # 使用isin方法检查commit_id是否存在于df_sdp中
    matching_commit_ids = df_fail['commit_id'].isin(df_sdp['commit_id'])

    # 计算存在对应commit_id的比例
    proportion = matching_commit_ids.sum() / len(matching_commit_ids)

    total_r1 += proportion

    print(f"当jgroups_test_pre_seed{i}.csv中status为fail时，jgroups_sdp_test_commit_seed{i}.csv存在对应commit_id的比例为: {proportion:.2%}")

print(total_r1/3)
