import git
import pandas as pd

repo = git.Repo('./')

repo.git.checkout('master')


def get_commit_diff_file(commit_id):
   
    commit = repo.commit(commit_id)


    diff = commit.diff(commit.parents[0]) if commit.parents else commit.diff(None)

    modified_java_files = []


    for file_diff in diff:
        if file_diff.change_type == 'M' and file_diff.a_path.endswith('.java'):
            if 'test' not in file_diff.a_path.lower() and 'tests' not in file_diff.a_path.lower():
                modified_java_files.append(file_diff.a_path)

    return modified_java_files




df = pd.read_csv('jgroups_final.csv')

#diff_files = {}
f = open("diff_files.txt","w")
for i in range(len(df)):
    commit_id = df.iloc[i]["commit_id"]
    diff_file = get_commit_diff_file(commit_id)
    line_data = (commit_id, str(diff_file))
    #diff_files[commit_id] = diff_file
    f.write(str(line_data) + '\n')


