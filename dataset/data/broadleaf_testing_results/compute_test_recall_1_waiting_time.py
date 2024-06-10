import pandas as pd  

total_r1 = 0

for i in range(1, 4):
    df_sdp = pd.read_csv(f'broadleaf_sdp_commit_seed{i}.csv')
    df_pre = pd.read_csv(f'broadleaf_test_pre_seed{i}.csv')

    df_fail = df_pre[df_pre['status'] == 1]

    matching_commit_ids = df_fail['commit_id'].isin(df_sdp['commit_id'])

    proportion = matching_commit_ids.sum() / len(matching_commit_ids)

    total_r1 += proportion


print("average testing recall 1:" + str(total_r1/3))
