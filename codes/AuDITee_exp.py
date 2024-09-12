import csv
import os
import subprocess

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.decomposition import PCA
from sklearn import preprocessing
from evaluation_online import compute_online_PF
from data_stream.real_data_stream import data_id_2name, set_test_stream, set_train_stream
from data_stream.real_data_stream import class_data_ind_org, class_data_ind_reset
from data_stream.real_data_preprocess import real_data_preprocess
from DenStream.DenStream import DenStream
from skmultiflow.trees import HoeffdingTreeClassifier
from codes.bagging_ooc import OzaBaggingClassifier_OOC
import warnings
from data_stream.real_data_stream import lookup_best_para
from utility import load_para_csv, check_random_state

"""the settings for systematic experiments"""
N_PARA_TUNE, WAIT_DAYS = 500, 15
CLF_NAME = "our".lower()
PARA_CSV = load_para_csv()  # preliminary experimental results
INVALID_VALUE, LABELS = -1, [0, 1]
dir_rslt_save = "../results/rslt.save/"


def run_AuDITee(project_id=0, seeds=range(0, 10), selector=1, verbose_int=0, tau=0.001, is_plot=False):
    # prepare
    project_name, n_test = data_id_2name(project_id)
    if is_plot:
        x_lim, y_lim = None, None

    """prepare test data_stream stream"""
    test_stream, commit_stream = set_test_stream(project_name)
    X_org = test_stream.X[class_data_ind_org().id_X_np]
    # convert #fea14 to #fea12 and the test data_stream stream
    XX_aug, use_data = real_data_preprocess(X_org)  # X_aug = [X_trans, churn_np]
    yy = test_stream.y[use_data]
    time = test_stream.X[use_data, class_data_ind_org().id_time][:, np.newaxis]
    vl = test_stream.X[use_data, class_data_ind_org().id_vl][:, np.newaxis]
    test_commit = commit_stream[use_data]

    # handle negative #test
    n_data_all, n_fea = XX_aug.shape[0], XX_aug.shape[1] - 1  # X_aug = [X_trans, churn_np]
    assert n_fea == 12, "# transformed #fea should be 12. Sth. must be wrong."
    # if N_TEST < 0:
    #     N_TEST += n_data_all
    #     if verbose_int >= 2:
    #         print("actual N_TEST=%d" % N_TEST)
    # assert N_PARA_TUNE < N_TEST, "nb_pre=%d should be smaller than nb_data=%d" % (N_PARA_TUNE, N_TEST)

    # fea normalizer based on all test data_stream, which is used for DenStream
    norm_scaler = my_norm_scaler(n_fea=n_fea, norm_name="z_score")
    norm_scaler.my_fit(XX_aug[:, :-1])  # the last col is churn_np
    # print('std:', np.std(norm_scaler.my_transform(XX), axis=0))

    # prepare all test samples
    test_data_all = np.hstack((time, XX_aug, vl, yy))  # col=3+1+12 ~ (time, #fea12, churn, vl, yy)
    data_ind_reset = class_data_ind_reset(
        id_time=0, id_y=-1, id_vl=-2, id_churn=-3, id_X_np=np.s_[:, 1:1 + n_fea], n_fea=n_fea)

    # data_stream pre-train
    data_ptrn = test_data_all[:N_PARA_TUNE]  # (time, #fea12, churn, vl, y)
    X_ptrn, y_ptrn = data_ptrn[data_ind_reset.id_X_np], data_ptrn[:, data_ind_reset.id_y]
    # churn_ptrn = data_ptrn[:, data_ind_reset.id_churn]

    """auto-para DenStream~(lambd, eps, beta, mu)"""
    X_ptrn_norm = norm_scaler.my_transform(X_ptrn)
    eps, mu, beta, lambd = 1.47, 1.57, 0.78, 0.26  # preliminary exp; better pf may be achieved by careful para tune

    """pre-train DenStream"""
    cluster = DenStream(theta_cl=None, lambd=lambd, eps=eps, beta=beta, mu=mu)
    cluster.partial_fit(X_ptrn_norm, y_ptrn)
    # plot 2d-pca
    if is_plot:
        all_yy = test_data_all[:, data_ind_reset.id_y]
        all_X_org = test_data_all[data_ind_reset.id_X_np]
        all_X_norm = norm_scaler.my_transform(all_X_org)
        # conduct pca on all points
        pca_hd = PCA(n_components=2)
        all_X_norm_pca = pca_hd.fit_transform(all_X_norm)
        print("\t pca 2d output:")
        print("\t\tpca.explained_var_ratios are ", pca_hd.explained_variance_ratio_)
        print("\t\tpca.singular_values are ", pca_hd.singular_values_)
        # 2d pca scatter of all points
        plt.scatter(all_X_norm_pca[:, 0], all_X_norm_pca[:, 1], c=all_yy)
        plt.title("%s: 2D scatter of normed pca points" % project_name)
        plt.xlim(x_lim)
        plt.ylim(y_lim)
        plt.grid(True)
        plt.show()
        # 2d micro-clusters on pre-train samples
        cluster.plot_cluster(X_ptrn_norm, y_ptrn, pca_hd, "%s: data_stream pre-train" % project_name, x_lim, y_lim)

    # look up the best classifier para~(n_tree, theta_imb, theta_cl)
    n_tree, theta_imb, theta_cl = lookup_best_para(project_name, WAIT_DAYS, CLF_NAME, PARA_CSV)
    # update DenStream para
    cluster.theta_cl = theta_cl

    """main parts across seeds"""
    nb_train_delay_np, nb_train_human_np = np.empty((len(seeds))), np.empty((len(seeds)))
    acc_train_churn_np = np.empty((len(seeds)))  # init
    nb_pred_y_np = np.empty((len(seeds), 2))
    for ss, seed in enumerate(seeds):
        s4_threshold = 0.5
        s5_threshold = 0.0
        my_rng = check_random_state(seed)
        if is_plot:
            x_lim, y_lim = None, None

        """pre-train classifier"""
        classifier = OzaBaggingClassifier_OOC(HoeffdingTreeClassifier(), n_tree, seed, theta_imb, theta_cl)
        cl_ptrn = comp_cl_upper(y_ptrn, y_ptrn)  # suppose noise-free
        classifier.partial_fit(X_ptrn, y_ptrn, cl_ptrn, LABELS)

        """[core] test-then-train:
        at each test step, only 1 test data arrives; none or several training data can become available.
        """
        nb_test_act = n_test - N_PARA_TUNE
        # init, test stream related. Note: we can know the size in advance.
        test_time, test_y_tru, test_y_pre = np.empty(nb_test_act), np.empty(nb_test_act), np.empty(nb_test_act)
        # init: train stream related. Note that we cannot know the size in advance.
        cmt_time_train_lst, use_time_train_lst, y_train_tru_lst, y_train_obv_lst = [], [], [], []
        code_churn_lst, cl_train_lst, use_cluster_lst = [], [], []
        # init: the test process
        prev_test_time = data_ptrn[-1, data_ind_reset.id_time]  # vip
        data_buffer, nb_train_, nb_train_human_ = np.empty((0, data_ptrn.shape[1])), 0, 0

        # test_results = {}

        test_result = pd.read_csv(
            f'../dataset/data/{project_name}_testing_results/{project_name}_test_pre_seed{seed}.csv')
        test_result_ = {}
        for tr in range(len(test_result)):
            test_result_[test_result.iloc[tr][0]] = test_result.iloc[tr][2]

        if selector == 0:
            file_path = f'../dataset/data/results/{project_name}/AuDITee_S1/seed{seed}/rst.csv'
            # 确保目录存在，如果不存在则创建
            directory = os.path.dirname(file_path)
            if not os.path.exists(directory):
                os.makedirs(directory)
            commit_result = open(file_path, 'w', newline='', encoding='utf-8')
            writer = csv.writer(commit_result)
            writer.writerow(['commit_id', 'SDP_predict', 'commit_guru_result', 'test_result'])  # 写入CSV头部
        elif selector == 1:
            file_path = f'../dataset/data/results/{project_name}/AuDITee_S2/seed{seed}/rst.csv'
            # 确保目录存在，如果不存在则创建
            directory = os.path.dirname(file_path)
            if not os.path.exists(directory):
                os.makedirs(directory)
            commit_result = open(file_path, 'w', newline='', encoding='utf-8')
            writer = csv.writer(commit_result)
            writer.writerow(['commit_id', 'predict_prob_1', 'SDP_predict', 'commit_guru_result', 'test_result',
                             'random_variable'])  # 写入CSV头部
        elif selector == 2:
            file_path = f'../dataset/data/results/{project_name}/AuDITee_S3/seed{seed}/rst.csv'
            # 确保目录存在，如果不存在则创建
            directory = os.path.dirname(file_path)
            if not os.path.exists(directory):
                os.makedirs(directory)
            commit_result = open(file_path, 'w', newline='', encoding='utf-8')
            writer = csv.writer(commit_result)
            writer.writerow(
                ['commit_id', 'predict_prob_1', 'predict_prob_0', 'SDP_predict', 'commit_guru_result', 'test_result',
                 'random_variable'])  # 写入CSV头部
        elif selector == 3:
            file_path = f'../dataset/data/results/{project_name}/AuDITee_S4/tau{tau}/seed{seed}/rst.csv'
            # 确保目录存在，如果不存在则创建
            directory = os.path.dirname(file_path)
            if not os.path.exists(directory):
                os.makedirs(directory)
            commit_result = open(file_path, 'w', newline='', encoding='utf-8')
            writer = csv.writer(commit_result)
            writer.writerow(['commit_id', 'predict_prob_1', 'SDP_predict', 'commit_guru_result', 'test_result',
                             'threshold'])  # 写入CSV头部
        elif selector == 4:
            file_path = f'../dataset/data/results/{project_name}/AuDITee_S5/tau{tau}/seed{seed}/rst.csv'
            # 确保目录存在，如果不存在则创建
            directory = os.path.dirname(file_path)
            if not os.path.exists(directory):
                os.makedirs(directory)
            commit_result = open(file_path, 'w', newline='', encoding='utf-8')
            writer = csv.writer(commit_result)
            writer.writerow(
                ['commit_id', 'predict_prob_1', 'predict_prob_0', 'SDP_predict', 'commit_guru_result', 'test_result',
                 'threshold'])  # 写入CSV头部
        else:
            assert False, "can not find such selector"

        # throughout each test time step
        for tt in range(nb_test_act):
            # print('start test----------------------------------------------------------------------')
            # get the test data_stream
            test_step = tt + N_PARA_TUNE

            test_1data = test_data_all[test_step, :].reshape((1, -1))  # 16~(time, #fea12, churn, vl, yy)
            test_X, test_churn = test_1data[data_ind_reset.id_X_np], test_1data[0, data_ind_reset.id_churn]
            test_time[tt] = test_1data[:, data_ind_reset.id_time]
            test_y_tru[tt] = test_1data[:, data_ind_reset.id_y]

            """test: predict with classifiers"""
            # test_y_pre[tt] = classifier.predict(test_X)[0]
            y_pre_prob_ = classifier.predict_proba(test_X)[0]
            y_pre_diff = y_pre_prob_[1] - y_pre_prob_[0]

            new_1data = test_1data  # overwritten if testing label, vip
            sota_X, sota_churn, sota_time, sota_y_obv, sota_y_tru = \
                np.empty((0, n_fea)), np.empty(0), np.empty(0), np.empty(0), np.empty(0)  # init empty, required
            commit_id = test_commit.iloc[test_step]['commit_id']
            test_label = test_result_[commit_id]
            random_variable = my_rng.uniform(0, 1)

            if selector == 0:
                SDP_predict = 0
                if y_pre_prob_[1] >= 0.5:
                    if test_label >= 0:
                        sota_X, sota_churn, sota_time, sota_y_tru = \
                            test_X, np.array([test_churn]), np.array([test_time[tt]]), np.array([test_y_tru[tt]])
                        sota_y_obv = np.array([test_label])  # note np.array([...])
                        new_1data = np.empty((0, data_ptrn.shape[1]))  # overwritten, vip
                    SDP_predict = 1
                writer.writerow([commit_id, SDP_predict, int(test_1data[0][-1]), test_label])

            elif selector == 1:
                if random_variable <= y_pre_prob_[1]:
                    if test_label >= 0:
                        sota_X, sota_churn, sota_time, sota_y_tru = \
                            test_X, np.array([test_churn]), np.array([test_time[tt]]), np.array([test_y_tru[tt]])
                        sota_y_obv = np.array([test_label])  # note np.array([...])
                        new_1data = np.empty((0, data_ptrn.shape[1]))  # overwritten, vip
                SDP_predict = 1 if y_pre_prob_[1] >= 0.5 else 0
                writer.writerow(
                    [commit_id, y_pre_prob_[1], SDP_predict, int(test_1data[0][-1]), test_label, random_variable])

            elif selector == 2:
                if random_variable <= y_pre_diff:
                    if test_label >= 0:
                        sota_X, sota_churn, sota_time, sota_y_tru = \
                            test_X, np.array([test_churn]), np.array([test_time[tt]]), np.array([test_y_tru[tt]])
                        sota_y_obv = np.array([test_label])  # note np.array([...])
                        new_1data = np.empty((0, data_ptrn.shape[1]))  # overwritten, vip

                SDP_predict = 1 if y_pre_prob_[1] >= y_pre_prob_[0] else 0

                writer.writerow(
                    [commit_id, y_pre_prob_[1], y_pre_prob_[0], SDP_predict, int(test_1data[0][-1]), test_label,
                     random_variable])

            elif selector == 3:
                if y_pre_prob_[1] >= s4_threshold:
                    if test_label >= 0:
                        sota_X, sota_churn, sota_time, sota_y_tru = \
                            test_X, np.array([test_churn]), np.array([test_time[tt]]), np.array([test_y_tru[tt]])
                        sota_y_obv = np.array([test_label])  # note np.array([...])
                        new_1data = np.empty((0, data_ptrn.shape[1]))  # overwritten, vip

                    s4_threshold = change_s4_threshold(test_label, s4_threshold)

                SDP_predict = 1 if y_pre_prob_[1] > 0.5 else 0

                writer.writerow(
                    [commit_id, y_pre_prob_[1], SDP_predict, int(test_1data[0][-1]), test_label, s4_threshold])

            elif selector == 4:
                if y_pre_diff >= s5_threshold:
                    if test_label >= 0:
                        sota_X, sota_churn, sota_time, sota_y_tru = \
                            test_X, np.array([test_churn]), np.array([test_time[tt]]), np.array([test_y_tru[tt]])
                        sota_y_obv = np.array([test_label])  # note np.array([...])
                        new_1data = np.empty((0, data_ptrn.shape[1]))  # overwritten, vip

                    s5_threshold = change_s5_threshold(test_label, s5_threshold)

                SDP_predict = 1 if y_pre_prob_[1] >= y_pre_prob_[0] else 0

                writer.writerow(
                    [commit_id, y_pre_prob_[1], y_pre_prob_[0], SDP_predict, int(test_1data[0][-1]), test_label,
                     s5_threshold])

            # if test_y_pre[tt] == 1:
            #     # overwrite empty when correct human labeling; note np.array([...]), vip
            #     # todo test to get test_y_tru
            #     commit_id = test_commit.iloc[test_step]['commit_id']
            #
            #     writer.writerow([str(commit_id)])
            #     # print(commit_id)
            #     test_label = test_result_[commit_id]
            #
            #     if test_label >= 0:
            #         sota_X, sota_churn, sota_time, sota_y_tru = \
            #             test_X, np.array([test_churn]), np.array([test_time[tt]]), np.array([test_y_tru[tt]])
            #         sota_y_obv = np.array([test_label])  # note np.array([...])
            #         new_1data = np.empty((0, data_ptrn.shape[1]))  # overwritten, vip

            """get the new train data_stream batch"""
            data_buffer, new_train_def, new_train_cln, new_train_unl = set_train_stream(
                prev_test_time, test_time[tt], new_1data, data_ind_reset, data_buffer, WAIT_DAYS)
            # note the order (clean, defect, sota)
            cmt_time_train = np.concatenate((
                new_train_cln[:, data_ind_reset.id_time], new_train_def[:, data_ind_reset.id_time], sota_time))
            use_time_train = test_time[tt] * np.ones(cmt_time_train.shape)
            X_train = np.concatenate((
                new_train_cln[data_ind_reset.id_X_np], new_train_def[data_ind_reset.id_X_np], sota_X))
            y_train_obv = np.concatenate((
                np.zeros(new_train_cln.shape[0]), np.ones(new_train_def.shape[0]), sota_y_obv))
            y_train_tru = np.concatenate((
                new_train_cln[:, data_ind_reset.id_y], new_train_def[:, data_ind_reset.id_y], sota_y_tru))
            churn_train = np.concatenate((
                new_train_cln[:, data_ind_reset.id_churn], new_train_def[:, data_ind_reset.id_churn], sota_churn))
            # counter for training data statistics
            nb_train_ += y_train_obv.shape[0]
            nb_train_human_ += sota_y_tru.shape[0]

            # assign
            cmt_time_train_lst.extend(cmt_time_train.tolist())
            code_churn_lst.extend(churn_train)
            use_time_train_lst.extend(use_time_train.tolist())
            y_train_obv_lst.extend(y_train_obv.tolist())
            y_train_tru_lst.extend(y_train_tru.tolist())
            if verbose_int >= 2:
                print("\ttest_step=%d, y_true=%d, y_pre=%d: %s"
                      % (test_step, test_y_tru[tt], y_pre_prob_[1], CLF_NAME))
                print("\t\tnew_train: y_true=%s, y_obv=%s" % (str(y_train_tru), str(y_train_obv)))
                print("\t\t#acc_train_all=%d, #acc_train_human=%d" % (nb_train_, nb_train_human_))

            """then update: update classifiers and DenStream given new labelled training data_stream"""
            if y_train_obv.shape[0] > 0:
                X_train_norm = norm_scaler.my_transform(X_train)
                cl_train, cl_c1_refine, use_cluster_train = cluster.compute_CLs(X_train_norm, y_train_obv)
                # update classifier
                classifier.partial_fit(X_train, y_train_obv, cl_train, LABELS)
                # update micro-cluster
                cluster.partial_fit(X_train_norm, y_train_obv)
                cluster.revise_cluster_info(X_train_norm, y_train_obv, cl_train)
                # assign
                cl_train_lst.extend(cl_train.tolist())
                use_cluster_lst.extend(use_cluster_train.tolist())
                # print
                if verbose_int >= 2:
                    for y_tru_, y_obv_, cl_ in zip(y_train_tru, y_train_obv, cl_train):
                        print("\t\t\ty_trn_tru=%d, y_trn_obv=%d, cl_est=%.2f" % (y_tru_, y_obv_, cl_))
                if is_plot and False:  # manual control
                    info = "test-step=%d, train X_org with y_true" % test_step
                    cluster.plot_cluster(X_train_norm, y_train_tru, pca_hd, info, x_lim, y_lim, True)
            prev_test_time = test_time[tt]  # update VIP


def change_s4_threshold(test_result, threshold):
    # test 1
    if test_result == 1:
        threshold -= tau
        threshold = threshold if threshold >= 0 else 0
        assert threshold >= 0
    # test 0
    elif test_result == 0:
        threshold += tau
        threshold = threshold if threshold <= 0.5 else 0.5
        assert threshold <= 0.5

    return threshold


def change_s5_threshold(test_result, threshold):
    if test_result == 1:
        threshold -= tau
        threshold = threshold if threshold >= 0 else 0
        assert threshold >= 0
    # test 0
    elif test_result == 0:
        threshold += tau
        threshold = threshold if threshold <= 1 else 1
        assert threshold <= 1
    return threshold


def comp_cl_upper(y_true, y_obv):
    """compute CLs for the upper bound, also the benchmark CLs.
    """
    assert y_true.shape == y_obv.shape, "the shape of y_obv should equal to that of y_true"
    upper_conf_levels = np.ones(np.size(y_true))
    upper_conf_levels[np.where(y_obv != y_true)] = 0
    # upper_conf_levels = np.where(y_true == y_true, 1, 0)
    return upper_conf_levels


class my_norm_scaler:
    """set my normaliser for DenStream in JIT-SDP.
    Note that the 1st fea "fix_bug" does not get involved & we should have 12 fea-s.

    Anonymous
    created on 2021-12-4, lasted major updated on 2022-7-28
    """

    def __init__(self, n_fea, norm_name="z_score"):
        self.n_fea = n_fea
        self.norm_name = norm_name  # by default z-score
        if self.norm_name.lower() == "min_max".lower():
            self.norm_scaler = preprocessing.MinMaxScaler()
        elif self.norm_name.lower() == "z_score".lower():
            self.norm_scaler = preprocessing.StandardScaler()

    def check_feature(self, XX):
        assert XX.shape[1] == self.n_fea, "wrong fea number. It should be 13 for transformed jit-sdp."

    def my_fit(self, XX):
        self.check_feature(XX)
        """see comments in my_transform() below"""
        if self.n_fea == 12:  # for jit-sdp: the 1st fea "fix_bug" should NOT be normalised.
            my_norm = self.norm_scaler.fit(XX[:, 1:])
        else:  # for synthetic
            my_norm = self.norm_scaler.fit(XX)
        return my_norm

    def my_transform(self, xx):
        if xx.ndim == 1:  # if xx contains only 1 data_stream sample
            xx = xx.reshape((-1, self.n_fea))
        """the real jit-sdp vs synthetic. 
        This is roughly decided based on #fea: for jit-sdp: n_fea=12; for syn: probably NOT 13. 
        """
        if self.n_fea == 12:  # for jit-sdp: the 1st fea "fix_bug" should remain unchanged.
            xx_trans = np.hstack((xx[:, 0].reshape(-1, 1), self.norm_scaler.transform(xx[:, 1:])))
        else:  # for synthetic
            xx_trans = self.norm_scaler.transform(xx)
        return xx_trans


def rslt_dir(clf_name, project_id, n_trees, theta_imb, theta_cl):
    """
    Set the result directory that store the training and test result for each seed.
    Anonymous on 2022/7
    """

    # handle other inputs
    clf_name = clf_name.lower()
    project_name, _ = data_id_2name(project_id)
    pre_to_dir = dir_rslt_save + project_name + "/" + clf_name

    pre_to_dir += "/testing"  # note to use str()
    # para-s for each classifier
    to_dir = pre_to_dir + "/n_trees" + str(n_trees)
    if clf_name != "oza":
        to_dir += "-theta_imb" + str(theta_imb)
    if clf_name != "oza" and clf_name != "oob" and clf_name != "oob_filter":
        to_dir += "-theta_cl" + str(theta_cl)
    return to_dir


def plot_on_1pf(pf_tt, clf_lst, title_info, my_ylim=None):
    """
    Plot ctn pf-s of multiple cases in a single plot.
    @pf_tt: shape (nb_test, nb_classifier)

    Anonymous
    Created on 2022-8-1, last updated on 2023/1
    """
    if not (isinstance(clf_lst, list) or isinstance(clf_lst, tuple)):
        raise Exception("Error: clf_lst should be a list")
    if np.ndim(pf_tt) == 1:
        pf_tt = pf_tt[:, np.newaxis]
    if pf_tt.shape[1] != len(clf_lst):
        raise Exception("Error: # classifier NOT matches column size of pf_tt")
    # plot
    xx = np.array(range(pf_tt.shape[0]))  # shape (nb_test, )
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    for cc, clf_name in enumerate(clf_lst):
        ax.plot(xx, pf_tt[:, cc], label=clf_name)
    # setup
    ax.set_title(title_info)
    if my_ylim is None:
        plt.ylim((0, 1))
    else:
        plt.ylim(my_ylim)
    ax.grid(True)
    ax.legend(loc="best")
    plt.show()


def print_pf(rslt_test, rslt_train):
    """
    rslt_test ~ (test_time, y_true, y_pred)
    rslt_train ~ (commit_time, use_time, yy, y_obv, cl, use_cluster)
    Anonymous created on 2022-8-2
    """
    # extract data_stream info
    test_y_tru, test_y_pre = rslt_test[:, 1], rslt_test[:, 2]
    y_train_tru_all, y_train_obv_all, cl_pre = rslt_train[:, 2], rslt_train[:, 3], rslt_train[:, 4]

    # training label noise
    nb_train, nb_train_noise = y_train_obv_all.shape[0], len(np.where(y_train_tru_all != y_train_obv_all)[0])
    trn_label_noise = nb_train_noise / nb_train  # vip
    print("\t training: label_noise=%f" % trn_label_noise)

    # training 1-sided label noise
    nb_defect = np.sum(y_train_obv_all == 1)
    trn_1side_noise = nb_train_noise / nb_defect
    print("\t training: 1sided_noise=%f" % trn_1side_noise)

    # c1% of test data_stream, i.e., true c1%
    nb_test_c1, nb_test = len(np.where(test_y_tru == 1)[0]), test_y_tru.shape[0]
    tst_c1_percent = nb_test_c1 / nb_test
    print("\t testing: class imbalance c1%%=%f" % tst_c1_percent)

    # pf: online prediction
    eval_pfs(test_y_tru, test_y_pre, True)


def eval_pfs(test_y_tru, test_y_pre, verbose=False):
    """evaluate PFs in terms of g-mean, recall-1, recall-0..
    2022/6/2        Separate this func.
    2022/11/28      insert mcc and other pf metrics
    """
    # ave PFs across test steps
    theta_eval = 0.99
    pfs_tt_dct = compute_online_PF(test_y_tru, test_y_pre, theta_eval)
    gmean_ave_tt = np.nanmean(pfs_tt_dct["gmean_tt"])
    r1_ave_tt, r0_ave_tt = np.nanmean(pfs_tt_dct["recall1_tt"]), np.nanmean(pfs_tt_dct["recall0_tt"])
    if verbose:
        print("\t ave online gmean=%.4f, r1=%.4f, r0=%.4f" % (gmean_ave_tt, r1_ave_tt, r0_ave_tt))
    return pfs_tt_dct


if __name__ == "__main__":
    selector = 4
    project = 0
    taus = [0.001,0.002,0.003,0.004,0.005,0.006,0.007,0.008,0.009,0.01]
    if selector == 3 or selector == 4:
        for tau in taus:
            run_AuDITee(project_id=project, selector=selector,tau=tau)
    else:
        run_AuDITee(project_id=project, selector=selector)
