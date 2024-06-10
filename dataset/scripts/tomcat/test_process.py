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

df = pd.read_csv('../tomcat_final.csv')
df_map = {}
for i in range(len(df)):
    df_map[df.iloc[i]["commit_id"]] = (df.iloc[i]["pre_commit_id"], df.iloc[i]["fix_commit_id"])


script_path = Path(__file__).resolve().parent

subprocess.run(["cd", str(script_path)], shell=True)

repo = git.Repo('./')

repo.git.checkout('master', force=True)


def testing(generate_commit_id, commit_id, diff_list, seed=1):
        print(" pre commit : " + generate_commit_id)
        print(" this commit : " + commit_id)
        # diff_list = ast.literal_eval(diff_list)
        if len(diff_list) > 0:
            diff_map = {}
            length =10 if len(diff_list) > 10 else len(diff_list)
            build_success = [0] * length
            i = -1
            for diff_file in diff_list[:length]:
                i += 1
                repo.index.reset()
                repo.git.checkout(generate_commit_id, force=True)
                result = subprocess.run(f"./evosuite_1.0.6_tomcat.sh {diff_file} {seed}", capture_output=True, shell=True, text=True)
                # print(result)
                if "build fail" in str(result):
                    continue

                build_success[i] += 1


                repo.index.reset()
                repo.git.checkout(commit_id, force=True)

                result = subprocess.run(f"./run_testsuite.sh {diff_file}", capture_output=True, shell=True, text=True, encoding='utf-8', errors='ignore')
                # print(result)

                if "AssertionError" in str(result):
                    # if "Time" in str(result):
                    print(commit_id + "is fail-----------------------------------------------------------------")
                    return 1
                elif "build fail" in str(result):
                    continue
                else:
                    build_success[i] += 1

            if 2 not in build_success:
                print(commit_id + "build error!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!1")
                return -1

            print(commit_id + "finished++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
            return 0
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
