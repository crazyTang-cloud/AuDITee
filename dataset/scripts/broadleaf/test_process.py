import csv
import multiprocessing
import os
import subprocess
import sys

import git
import pandas as pd
from pathlib import Path
import ast
import re

import pandas as pd

df = pd.read_csv('../broadleaf_final.csv')
df_map = {}
for i in range(len(df)):
    df_map[df.iloc[i]["commit_id"]] = (df.iloc[i]["pre_commit_id"], df.iloc[i]["fix_commit_id"])


script_path = Path(__file__).resolve().parent

subprocess.run(["cd", str(script_path)], shell=True)

repo = git.Repo('./')

repo.git.checkout('master', force=True)

def handle_diff_file(diff_file):
    project_path, class_path = extract_project_and_class(diff_file)

    #a.b.c
    file_path = re.sub(r'\.java$', '', class_path)
    file_path = file_path.replace('/', '.')

    return project_path[:-1], file_path


def extract_project_and_class(path):

    if 'src/main/java/' in path:
        parts = path.split('src/main/java/')
        return parts[0], parts[1]

    else:
        parts = path.split('org/')
        return parts[0], "org/" + parts[1]

def regular_path(path):
    
    parts = path.split(".")
   
    directory_path = "/".join(parts[:-1]).replace(".", "/")
    file_name = parts[-1]
    
    return f"evosuite-tests/{directory_path}/{file_name}"

def testing(generate_commit_id,commit_id, diff_list, seed):
        repo.index.reset()
        repo.git.checkout(generate_commit_id, force=True)
        # diff_list = ast.literal_eval(diff_list)
        if len(diff_list) > 0:
            print("start pre commit install" + generate_commit_id)
            try:
                subprocess.run("mvn clean install -q -DskipTests", shell=True, check=True)
                subprocess.run("mvn dependency:copy-dependencies -q -DincludeScope=runtime", shell=True, check=True)
                subprocess.run("get_jar.sh", shell=True)
            except subprocess.CalledProcessError as e:
                return -1
            diff_map = {}
            length =10 if len(diff_list) > 10 else len(diff_list)
            for diff_file in diff_list[:length]:
                project_path, file_path = handle_diff_file(diff_file)
                if project_path in diff_map.keys():
                    file_paths = diff_map[project_path]
                    file_paths.append(file_path)
                    diff_map[project_path] = file_paths
                else:
                    diff_map[project_path] = [file_path]

            for key in diff_map.keys():
                file_paths = diff_map[key]
                # subprocess.run("cd "+key, shell=True)
                # depth = len(key.split('/')) + 1
                for file_path in file_paths:
                    subprocess.run(f"evosuite_1.0.6_mvn.sh {key} {file_path} {seed}", shell=True)

                # subprocess.run("cd "+"../"*depth, shell=True)

            repo.index.reset()
            repo.git.checkout(commit_id, force=True)
            print("start this commit install" + commit_id)
            try:
                subprocess.run("mvn clean install -q -DskipTests", shell=True, check=True)
                subprocess.run("mvn dependency:copy-dependencies -q -DincludeScope=runtime", shell=True, check=True)
                subprocess.run("get_jar.sh", shell=True)
            except  subprocess.CalledProcessError as e:
                return -1

            for key in diff_map.keys():
                file_paths = diff_map[key]
                # subprocess.run("cd "+key, shell=True)
                # depth = len(key.split('/')) + 1

                for file_path in file_paths:
                    # regular_result_path = regular_path(file_path) + '_ESTest.class'
                    regular_result_path = file_path + '_ESTest'
                    # rrp = script_path / regular_result_path
                    result = subprocess.run(f"run_testsuite.sh {key} {regular_result_path}", capture_output=True, shell=True, text=True)
                    print("result:" + str(result))
                    print("stdout:" + str(result.stdout))
                    if "AssertionError" in str(result):
                        # if "Time" in str(result):
                        print(commit_id + "is fail-----------------------------------------------------------------")
                        return 1

            print(commit_id + "finished++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
            return 0

            # subprocess.run("cd " + "../" * depth, shell=True)
        else:
            return 0

def get_commit_diff_file(commit_id):

    commit = repo.commit(commit_id)


    diff = commit.diff(commit.parents[0]) if commit.parents else commit.diff(None)

    modified_java_files = []

    for file_diff in diff:
        if file_diff.change_type == 'M' and file_diff.a_path.endswith('.java'):
            if 'test' not in file_diff.a_path.lower():
                modified_java_files.append(file_diff.a_path)

    return modified_java_files

def test_commit(pre_commit_id, commit_id, seed=2):
    diff_list = get_commit_diff_file(commit_id)
    test_status = testing(pre_commit_id, commit_id, diff_list, seed)
    return test_status

pre_commit_id = sys.argv[1]
commit_id = sys.argv[2]
seed = sys.argv[3]
status = test_commit(pre_commit_id, commit_id, int(seed))
# f.write(status)

if status == 1:
    print("status is:  ->fail")
    print(str(status))
elif status == 0:
    print("status is:  ->pass")
    print(str(status))
else:
    print("status is:  ->test error")
    print(str(status))
sys.exit(status)

# if len(sys.argv) != 3:
#     sys.exit("usage error")
#     # print("Usage: python test_process.py pre_commit_id commit_id")
#     # f.write('usage error')
#
# start = int(sys.argv[1])
# end = int(sys.argv[2])
#
# main(start, end)
