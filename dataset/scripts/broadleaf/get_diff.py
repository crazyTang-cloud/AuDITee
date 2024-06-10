import git
import pandas as pd

repo = git.Repo('./')
# 遍历仓库的所有commit
repo.git.checkout('master')

#获得差异文件
def get_commit_diff_file(commit_id):
    # 使用git diff-tree命令来获取commit之间的差异
    # --name-status参数会输出文件名和状态（M表示修改，A表示添加）
    # --no-renames参数排除重命名的情况
    # -r参数是递归处理子目录
    # --diff-filter=ACM参数仅包括添加(A)、修改(M)和复制(C)的文件
        # cmd = ["git diff HEAD^ --name-status -- '*.java'"]
        # result = subprocess.run(cmd, stdout=subprocess.PIPE, universal_newlines=True, shell=True, check=True)
        #
        # # 解析输出，'\0'是分隔符
        # files = result.stdout.split('\0')[:-1]  # 移除最后一个空字符串（由于'\0'结尾）
        #
        # result_paths = []
        #
        # for file in files:
        #     if file[0] == 'M':
        #         result_paths.append(file)
    # 获取指定提交
    commit = repo.commit(commit_id)

    # 获取提交与父提交的差异
    diff = commit.diff(commit.parents[0]) if commit.parents else commit.diff(None)

    modified_java_files = []

    # 遍历差异获取修改过的.java文件
    for file_diff in diff:
        if file_diff.change_type == 'M' and file_diff.a_path.endswith('.java'):
            if 'test' not in file_diff.a_path.lower():
                modified_java_files.append(file_diff.a_path)

    return modified_java_files


# files现在包含了所有真正内容发生变化的文件名
    # return result_paths
    # result_paths = []
    #
    # commit = repo.commit(commit_id)
    # # # 获取commit修改的文件列表
    # # modified_files = commit.stats.files
    # # 获取commit中的修改文件列表，不包括添加或删除的文件  
    # modified_files = [item.a_path for item in commit.stats.files if item.change_type == 'M']
    # for file_path, stats in modified_files.items():
    #     files = file_path.split('/')
    #     if 'test' not in files and 'tests' not in files and '.java' in files[-1]:
    #         result_path = '.'.join(files[1:])
    #         result_path = result_path[:-5]
    #         result_paths.append(result_path)
    #
    # return result_paths

df = pd.read_csv('broadleaf_modified.csv')

#diff_files = {}
f = open("diff_files.txt","w")
for i in range(len(df)):
    commit_id = df.iloc[i]["commit_id"]
    diff_file = get_commit_diff_file(commit_id)
    line_data = (commit_id, str(diff_file))
    #diff_files[commit_id] = diff_file
    f.write(str(line_data) + '\n')


