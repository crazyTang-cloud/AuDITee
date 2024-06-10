import csv
import multiprocessing
import os
import subprocess
import sys

import git
import pandas as pd
from pathlib import Path

# #获取脚本的绝对路径
# script_dir = os.path.dirname(os.path.abspath(__file__))
#
# # 切换到脚本所在的目录
# os.chdir(script_dir)


# 获取当前脚本所在目录的Path对象
script_path = Path(__file__).resolve().parent

# os.system("cd " + str(script_path))
subprocess.run(["cd", str(script_path)], shell=True)

# 构建完整的文件路径
file_path = script_path / 'build.sh'
# 获取commit信息 # {"commit":"abcde12345","date":"20230414","summary":"xxx"}
repo = git.Repo('./')
# 遍历仓库的所有commit
repo.git.checkout('master', force=True)



# build_sh_exist = 1
# mvn_exist = 0

#获得差异文件
def get_commit_diff_file(commit_id):
    result_paths = []

    commit = repo.commit(commit_id)
    # 获取commit修改的文件列表
    modified_files = commit.stats.files

    for file_path, stats in modified_files.items():
        files = file_path.split('/')
        if 'test' not in files and 'tests' not in files and '.java' in files[-1]:
            result_path = '.'.join(files[1:])
            result_path = result_path[:-5]
            result_paths.append(result_path)

    return result_paths

def checkout_commit(commit_id):
    command = "rm build.properties"
    # 使用subprocess执行命令
    subprocess.run(command, shell=True)
    # os.system(command)
    subprocess.run(['git', 'checkout', '--', '.'], shell=True)
    # os.system("git checkout -- .")
    repo.git.checkout(commit_id, force=True)

def build(mvn=False):
    try:
        if not mvn:
            command = "rm -r classes"
            # 使用subprocess执行命令
            subprocess.run(command, shell=True)
            # os.system(command)
            subprocess.run(["build.sh"], shell=True, check=True)
            # os.system('build.sh')
        else:
            command = "rm -r target"
            # 使用subprocess执行命令
            subprocess.run(command, shell=True)
            # os.system(command)
            subprocess.run(["mvn", "clean", "install", "-Dmaven.test.skip=true"], shell=True, check=True)
            # os.system("mvn clean install -Dmaven.test.skip=true")
    except subprocess.CalledProcessError as e:
        raise subprocess.CalledProcessError("build error")

def regular_path(path):
    # 将包名和类名转化为文件路径
    # 分割包名和类名
    parts = path.split(".")
    # 将包名部分转化为目录结构，并加上"evosuite-tests/"前缀
    directory_path = "/".join(parts[:-1]).replace(".", "/")
    file_name = parts[-1]
    # 组合完整的文件路径
    return f"evosuite-tests/{directory_path}/{file_name}"

def generate_testsuite(result_paths, seed=1, mvn=False):
    command = "rm -r evosuite-report"
    # 使用subprocess执行命令
    subprocess.run(command, shell=True)
    # os.system(command)
    command = "rm -r evosuite-tests"
    # 使用subprocess执行命令
    subprocess.run(command, shell=True)
    # os.system(command)
    print("start generate_testsuite......................................................................................")

    # 生成测试用例
    if not mvn:
        print(
            "start generate_testsuite without mvn----@@@@@@@@@2......................................................................................")
        for result_path in result_paths:
            print(result_path)
            subprocess.run(f"evosuite_1.0.6_bash.sh {result_path} {seed}", shell=True)
            # os.system("evosuite_1.2.0_bash.sh {}".format(result_path))
            regular_result_path = regular_path(result_path) + '_ESTest.java'
            print(regular_result_path)
            rrp = script_path / regular_result_path
            # if os.path.exists(regular_result_path):
            print(str(rrp.is_file())+'......................................................................+++++_____-------=======')
            if rrp.is_file():
            # if os.path.exists(regular_result_path):
                print("this path:---------------------------------------------")
                # os.system("pwd")
                subprocess.run(["pwd"], shell=True)
                print("------------------------------------------------------------------------")
                print("testing generation------------------------------------------------------")
                command = 'javac -cp classes/:lib/:evosuite-standalone-runtime-1.0.6.jar:hamcrest-core-1.3.jar:junit-4.12.jar:evosuite-tests/'+ ' ' + regular_result_path
                subprocess.run(command, shell=True)
                # os.system(command + ' ' + regular_result_path)
    else:
        print(
            "start generate_testsuite with mvn----@@@@@@@@@2......................................................................................")
        for result_path in result_paths:
            print(result_path)
            subprocess.run(f"evosuite_1.0.6_mvn.sh {result_path} {seed}", shell=True)
            # os.system("evosuite_1.2.0_mvn.sh {}".format(result_path))
            regular_result_path = regular_path(result_path) + '_ESTest.java'
            print(regular_result_path)
            rrp = script_path / regular_result_path
            # if os.path.exists(regular_result_path):
            if rrp.is_file():
            # if os.path.exists(regular_result_path):
                print("testing generation------------------------------------------------------")
                command = 'javac -cp target/classes/:evosuite-standalone-runtime-1.0.6.jar:hamcrest-core-1.3.jar:junit-4.12.jar:evosuite-tests/'+ ' ' + regular_result_path
                subprocess.run(command, shell=True)
                # os.system(command + ' ' + regular_result_path)

def run_testsuite(result_paths, mvn=False):
    if not mvn:
        print(
            "start run_testsuite without mvn----@@@@@@@@@2......................................................................................")
        for result_path in result_paths:
            regular_result_path = regular_path(result_path) + '_ESTest.class'
            rrp = script_path / regular_result_path
            # if os.path.exists(regular_result_path):
            if rrp.is_file():
            # if os.path.exists(regular_result_path):
                print("start testing------------------------------------------------------")
                command = 'java -cp classes/:lib/:evosuite-standalone-runtime-1.0.6.jar:hamcrest-core-1.3.jar:junit-4.12.jar:evosuite-tests/ org.junit.runner.JUnitCore'
                command_ = command + ' ' + result_path + '_ESTest'
                # result = os.popen(command_)
                print("test running output --------------------------------------------------------------------------")
                result = subprocess.run(command_, capture_output=True, shell=True, text=True)
                print("result:" + str(result))
                print("stdout:" + str(result.stdout))
                if "AssertionError" in str(result):
                # if "Time" in str(result):
                    return "fail"
    else:
        print(
            "start run_testsuite with mvn----@@@@@@@@@2......................................................................................")
        for result_path in result_paths:
            regular_result_path = regular_path(result_path) + '_ESTest.class'
            rrp = script_path / regular_result_path
            # if os.path.exists(regular_result_path):
            if rrp.is_file():
                print("start testing------------------------------------------------------")
                command = 'java -cp target/classes/:evosuite-standalone-runtime-1.0.6.jar:hamcrest-core-1.3.jar:junit-4.12.jar:evosuite-tests/ org.junit.runner.JUnitCore'
                command_ = command + ' ' + result_path + '_ESTest'
                # result = os.popen(command_)

                result = subprocess.run(command_, shell=True, capture_output=True, text=True)
                print("test running output --------------------------------------------------------------------------")
                # print(str(result))
                print("result:" + str(result))
                print("stdout:" + str(result.stdout))
                if "AssertionError" in str(result):
                    return "fail"

    return "pass"

def test_commit(pre_commit_id,commit_id, seed=1):
    result_paths = get_commit_diff_file(commit_id)

    if len(result_paths) <= 0:
        #print("pass")
        return "pass"

    # print(result_paths)
    checkout_commit(pre_commit_id)
    # command = "git checkout -b branch_" + pre_commit_id[:10] + " " + pre_commit_id
    # os.system(command)
    # checkout_commit(commit_id)

    # if os.path.exists('build.sh'):
    if file_path.is_file():
        try:
            print("start build pre commit......................................................................................")
            build()
        except subprocess.CalledProcessError as e:
            # command = "git checkout -D branch_" + pre_commit_id[:10]
            #print("build error")
            return "build error"

        generate_testsuite(result_paths,seed)

        # command = "git checkout -D branch_" + pre_commit_id[:10]
        # os.system(command)

        # return run_testsuite(result_paths)

        # command = "git checkout -b branch_" + commit_id[:10] + " " + commit_id
        # os.system(command)
        checkout_commit(commit_id)

        # if os.path.exists('build.sh'):
        if file_path.is_file():
            try:
                print(
                    "start build commit......................................................................................")
                build()
            except subprocess.CalledProcessError as e:
                # command = "git checkout -D branch_" + commit_id[:10]
                #print("build error")
                return "build error"

            status = run_testsuite(result_paths)
            # command = "git checkout -D branch_" + commit_id[:10]
            # os.system(command)
            #print(status)
            return status
        else:
            try:
                print(
                    "start build commit......................................................................................")
                build(mvn=True)
            except subprocess.CalledProcessError as e:
                # command = "git checkout -D branch_" + commit_id[:10]
                #print("build error")
                return "build error"

            status = run_testsuite(result_paths,mvn=True)
            # command = "git checkout -D branch_" + commit_id[:10]
            # os.system(command)
            #print(status)
            return status
    else:
        try:
            build(mvn=True)
        except subprocess.CalledProcessError as e:
            # command = "git checkout -D branch_" + pre_commit_id[:10]
            # print("build error")
            return "build error"

        generate_testsuite(result_paths, seed, mvn=True)

        # command = "git checkout -b branch_" + commit_id[:10] + " " + commit_id
        # os.system(command)
        checkout_commit(commit_id)

        try:
            build(mvn=True)
        except subprocess.CalledProcessError as e:
            # command = "git checkout -D branch_" + commit_id[:10]
            # print("build error")
            return "build error"

        status = run_testsuite(result_paths, mvn=True)
        # command = "git checkout -D branch_" + commit_id[:10]
        # os.system(command)
        # print(status)
        return status


import sys

# f = open('./data_temp.txt','w')

# 检查是否提供了足够的参数
if len(sys.argv) != 4:
    sys.exit("usage error")
    # print("Usage: python test_process.py pre_commit_id commit_id")
    # f.write('usage error')



# print("start testing")
# 获取参数
pre_commit_id = sys.argv[1]
commit_id = sys.argv[2]
seed = sys.argv[3]
status = test_commit(pre_commit_id, commit_id, int(seed))
# f.write(status)

print("status is:  ->"+str(status))
print(str(status))
sys.exit(status)




# # 读取CSV文件
# def read_csv(file_path):
#     return pd.read_csv(file_path)
#
#
# # 你的脚本函数，这里只是一个示例，你需要替换为你自己的脚本逻辑
# def process_data(start_index, end_index, data):
#     chunk = data[start_index:end_index]
#     # 在这里处理你的数据...
#     print(f"Processing data from index {start_index} to {end_index}")
#     statuses = []
#     for i in range(0, len(chunk)):
#         commit_id = chunk.iloc[i, -1]
#         pre_commit_id = chunk.iloc[i, -2]
#         print(commit_id)
#         print(pre_commit_id)
#         status = test_commit(pre_commit_id, commit_id)
#         statuses.append(status)
#         print(statuses)
#     with open("test_result.txt", 'a') as f:
#         for status in statuses:
#             f.write(commit_id + ' ' + str(status) + '\n')
#     # 返回处理结果或进行其他操作...
#
#
# # 主函数
# def main(file_path):
#     # 读取CSV文件
#     data = read_csv(file_path)
#
#     # 计算每组数据的索引
#     chunk_size = 100
#     num_chunks = (len(data) + chunk_size - 1) // chunk_size
#
#     # 使用multiprocessing并行处理数据
#     with multiprocessing.Pool() as pool:
#         for i in range(num_chunks):
#             start_index = i * chunk_size
#             end_index = min((i + 1) * chunk_size, len(data))
#             pool.apply_async(process_data, args=(start_index, end_index, data))
#
#             # 等待所有进程完成
#         pool.close()
#         pool.join()
#
#
# if __name__ == "__main__":
#     file_path = '../jgroups_final.csv'  # 替换为你的CSV文件路径
#     main(file_path)




# if __name__ == '__main__':
#     df = pd.read_csv('../jgroups_final.csv')
#     statuses = []
#     for i in range(0,len(df)):
#         commit_id = df.iloc[i, -1]
#         pre_commit_id = df.iloc[i, -2]
#         print(commit_id)
#         print(pre_commit_id)
#         status = test_commit(pre_commit_id, commit_id)
#         statuses.append((commit_id, status))
#         print(statuses)
#     with open('test_result.csv', 'w', newline='', encoding='utf-8') as csvfile:
#         writer = csv.writer(csvfile)
#         writer.writerow(['commit_id', 'time', 'has_pom'])  # 写入CSV头部
#         # 将commit信息写入CSV文件
#         for status in statuses:
#             writer.writerow([status[0], status[1]])
    # with open("test_result.txt",'w') as f:
    #     for status in statuses:
    #         f.write(str(status) + '\n')
