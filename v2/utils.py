'''
    Copyright 2018 - The OPRECOMP Project Consortium, Alma Mater Studiorum
    Università di Bologna. All rights reserved.

    Licensed under the Apache License, Version 2.0 (the "License");
    you may not use this file except in compliance with the License.
    You may obtain a copy of the License at

        http://www.apache.org/licenses/LICENSE-2.0

    Unless required by applicable law or agreed to in writing, software
    distributed under the License is distributed on an "AS IS" BASIS,
    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
    See the License for the specific language governing permissions and
    limitations under the License.
'''


import os
import sys
import math
from decimal import *
import collections 
from collections import OrderedDict
import operator
import json
import pickle
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, LabelBinarizer, MinMaxScaler
from sklearn.metrics import mean_squared_error, median_absolute_error
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.metrics import explained_variance_score
from sklearn.metrics import precision_recall_fscore_support
from sklearn.manifold import TSNE
import pandas as pd
from sklearn_pandas import DataFrameMapper
import time
import re
import subprocess
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.dates as mdates
import numpy as np
import scipy.stats as stats
import ast
import numpy as np
np.seterr(all='warn')
import warnings
import tensorflow as tf
from keras import backend as K 
import yaml
from sklearn.neighbors import KDTree
import random

seed = 42

very_large_error = 1000.00

def run_greedy_search(binary_name, err_rate):
    cmds = ['mpirun', '-np', str(n_mpi_procs), greedy_search_script,
            str(seed), binary_name, str(refine), err_rate]
    p = subprocess.Popen(cmds, stdout=subprocess.PIPE)
    output, err = p.communicate()
    final_config = ast.literal_eval((output.split('\n')[-2]).split(';')[0])
    min_config = ast.literal_eval((output.split('\n')[-2]).split(';')[1])
    return final_config, min_config

def run_greedy_search_opt(binary_name, err_rate, benchmark):
    cmds = ['mpirun', '-np', str(n_mpi_procs), greedy_search_opt_script,
            str(seed), binary_name, str(refine), err_rate, benchmark]
    p = subprocess.Popen(cmds, stdout=subprocess.PIPE)
    output, err = p.communicate()
    final_config = ast.literal_eval((output.split('\n')[-2]).split(';')[0])
    min_config = ast.literal_eval((output.split('\n')[-2]).split(';')[1])
    return final_config, min_config

def read_file(filename):
    with open(filename, 'r') as rf:
        lines = rf.readlines()
    error_ratios_exps = map(int, lines[0].split(',')[:-1])

    exps_res = {}
    lines = lines[1:]
    for i in range(len(lines)):
        exps_res[i] = map(int, lines[i].split(',')[:-1])
        
    return error_ratios_exps, exps_res

def read_multiple_result_file(basedir, benchmark):
    multi_exp_res = {}
    for filename in os.listdir(basedir):
        if filename.startswith('exp_results_' + benchmark
                ) and filename.endswith('.pickle'):
            with open(basedir + filename, 'rb') as handle:
                exp_res = pickle.load(handle)
            multi_exp_res.update(exp_res)
    return multi_exp_res

def read_result_dict(filename):
    with open(filename, 'r') as rf:
        lines = rf.readlines()
    error_ratios_exps = []
    error_ratios = []
    exps_res = {}
    exps_res_dicts = {}
    configs = []
    times = []
    error_actuals = []
    for l in lines:
        er_exp = int(l.split(';')[0])
        er = l.split(';')[1]
        exp_dict = ast.literal_eval(l.split(';')[2])
        exps_res_dicts[er_exp] = exp_dict
        error_ratios_exps.append(er_exp)
        error_ratios.append(er)
        for i in range(len(exp_dict['var_bits'])):
            if i not in exps_res:
                exps_res[i] = [exp_dict['var_bits'][i]]
            else:
                exps_res[i].append(exp_dict['var_bits'][i])
        if len(l.split(';')) > 3:
            error_actual = float(l.split(';')[3])
        else:
            error_actual = -1.0
        error_actuals.append(error_actual)
        configs.append(exp_dict['var_bits'])
        times.append(exp_dict['time'])
    return (error_ratios_exps, error_ratios, exps_res_dicts, 
            exps_res, configs, times, error_actuals)

def run_program_stoch(program, dataset_idx, target_result):
    output = subprocess.Popen([program, '%s'%(seed), '%s'%(dataset_idx)],
                              stdout=subprocess.PIPE).communicate()[0]
    floating_result = parse_output(output.decode('utf-8'))
    return check_output(floating_result, target_result)

def run_program(program, target_result):
    output = subprocess.Popen([program, '%s'%(seed)],
                              stdout=subprocess.PIPE).communicate()[0]
    floating_result = parse_output(output.decode('utf-8'))
    return check_output(floating_result, target_result)

def check_all_zeros(result):
    for i in range(len(result)):
        if result[i] != 0.00:
            return False
    return True

def check_output(floating_result, target_result):
    if len(floating_result) != len(target_result):
        print('Error: floating result len %s while target_result len %s'
                % (len(floating_result), len(target_result)))
        return very_large_error

    if check_all_zeros(floating_result) != check_all_zeros(target_result):
        return very_large_error

    signal_sqr = 0.00
    error_sqr = 0.00
    sqnr = 0.00
    for i in range(len(floating_result)):
        # if floating_result[i] == 0, check_output returns 1: this is an 
        # unwanted behaviour
        if floating_result[i] == 0.00:
            continue    # mmmhhh, TODO: fix this in a smarter way

        # if there is even just one inf in the result list, we assume that
        # for the given configuration the program did not run properly
        if str(floating_result[i]) == 'inf':
            return 'Nan'

        signal_sqr = target_result[i] ** 2
        error_sqr = (floating_result[i] - target_result[i]) ** 2
        temp = 0.00
        if error_sqr != 0.00:
            temp = signal_sqr / error_sqr
        if temp != 0:
            temp = 1.0 / temp
        if temp > sqnr:
            sqnr = temp;

    return sqnr

def check_output_MSE(floating_result, target_result):
    res1 = np.asarray(floating_result, dtype=np.float64)
    res2 = np.asarray(target_result, dtype=np.float64)
    error = np.square(res1 - res2).mean()
    return error

def check_output_GS(floating_result, target_result):

    # TODO: modify this func to return checksum error, instead of true and false 
    #       feed the checksum error to greedy decision func

    if len(floating_result) != len(target_result):
        return 0.00
    signal_sqr = 0.00
    error_sqr = 0.00
    sqnr = 0.00
    for i in range(len(floating_result)):
        signal_sqr = target_result[i] ** 2
        error_sqr = (floating_result[i] - target_result[i]) ** 2
        temp = 0.00
        if error_sqr != 0.00:
            temp = signal_sqr / error_sqr
        if temp != 0:
            temp = 1.0 / temp
        if temp > sqnr:
            sqnr = temp;

    return sqnr

def parse_output(line):
    list_target = []
    line.replace(' ', '')
    line.replace('\n', '')

    # remove unexpected space
    array = line.split(',')

    for target in array:
        try:
            if len(target) > 0 and target != '\n':
                list_target.append(float(target))
        except:
            continue

    return list_target

def write_conf(conf_file, config):
    conf_string = ''
    for i in config:
        conf_string += str(i) + ','
    with open(conf_file, 'w') as write_file:
        write_file.write(conf_string)

def read_target(target_file):
    # format a1,a2,a3...
    list_target = []
    with open(target_file) as conf_file:
        for line in conf_file:
            line.replace(' ', '')

            # remove unexpected space
            array = line.split(',')
            for target in array:
                try:
                    if len(target) > 0 and target != '\n':
                        list_target.append(float(target))
                except:
                    print('Failed to parse target file')
    return list_target

def read_conf(conf_file_name):
    # format a1,a2,a3,...
    list_argument = []
    with open(conf_file_name) as conf_file:
        for line in conf_file:
            line.replace(' ', '')

            # remove unexpected space
            array = line.split(',')
            for argument in array:
                try:
                    if len(argument) > 0 and argument != '\n':
                        list_argument.append(int(argument))
                except:
                    print('Failed to parse target file')
    return list_argument

'''
Drop unused features and rows with NaN
'''
def drop_stuff(df, features_to_be_dropped):
    for fd in features_to_be_dropped:
        if fd in df:
            del df[fd]
    new_df = df.dropna(axis=0, how='all')
    new_df = new_df.dropna(axis=1, how='all')
    new_df = new_df.fillna(0)
    return new_df

'''
Pre-process input data.
Encode the categorical features
'''
def preprocess_noScaling(df, categorical_features, continuous_features):
    for c in categorical_features:
        df = encode_category(df, c)
    return df

'''
Pre-process input data.
Scale continuous features and encode the categorical ones
'''
def preprocess(df, categorical_features, continuous_features, scaler=None):
    if scaler == None:
        scaler = MinMaxScaler(feature_range=(0, 1))
    df[continuous_features] = scaler.fit_transform(df[continuous_features])

    for c in categorical_features:
        df = encode_category(df, c)

    return df, scaler

'''
Encode categorical values into with one-hot encoding
'''
def encode_category(data_set_in, c):
    if c not in data_set_in:
        return data_set_in
    dummy_cols =  pd.get_dummies(data_set_in[c], dummy_na=False)
    data_set = pd.concat([data_set_in, dummy_cols], axis=1)
    del data_set[c]
    return data_set

'''
Evaluate prediction
'''
def evaluate_predictions(predicted, actual):
    abs_errors = []
    p_abs_errors = []
    sp_abs_errors = []
    squared_errors = []
    underest_count = 0
    overest_count = 0
    errors = []

    for i in range(len(predicted)):
        abs_errors.append(abs(predicted[i] - actual[i]))
        errors.append(predicted[i] - actual[i])
        squared_errors.append((predicted[i] - actual[i])*
            (predicted[i] - actual[i]))
        if actual[i] != 0:
            p_abs_errors.append((abs(predicted[i]-actual[i]))* 
                    100 / abs(actual[i]))
        sp_abs_errors.append((abs(predicted[i]-actual[i])) * 100 / 
            abs(predicted[i] + actual[i]))
        if predicted[i] - actual[i] > 0:
            overest_count += 1
        elif predicted[i] - actual[i] < 0:
            underest_count += 1

    MAE = Decimal(np.mean(np.asarray(abs_errors)))
    MAPE = Decimal(np.mean(np.asarray(p_abs_errors)))
    SMAPE = Decimal(np.nanmean(np.asarray(sp_abs_errors)))
    MSE = Decimal(np.mean(np.asarray(squared_errors)))
    RMSE = Decimal(math.sqrt(MSE))
    R2 = r2_score(actual, predicted)
    SK_MAE = mean_absolute_error(actual, predicted)
    MedAE = median_absolute_error(actual, predicted)
    SK_MSE = mean_squared_error(actual, predicted)
    EV = explained_variance_score(actual, predicted)

    stats_res = {}
    stats_res["MAE"] = MAE
    stats_res["MSE"] = MSE
    stats_res["ERRORS"] = errors
    stats_res["RMSE"] = RMSE
    stats_res["MAPE"] = MAPE
    stats_res["SMAPE"] = SMAPE
    stats_res["ABS_ERRORS"] = abs_errors
    stats_res["P_ABS_ERRORS"] = p_abs_errors
    stats_res["SP_ABS_ERRORS"] = sp_abs_errors
    stats_res["SQUARED_ERRORS"] = squared_errors
    stats_res["R2"] = R2
    stats_res["MedAE"] = MedAE
    stats_res["EV"] = EV
    stats_res["abs_errs"] = abs_errors
    stats_res["p_abs_errs"] = p_abs_errors
    stats_res["sp_abs_errs"] = sp_abs_errors
    stats_res["squared_errs"] = squared_errors
    stats_res["accuracy"] = 100 - abs(MAPE)
    stats_res["underest_count"] = underest_count
    stats_res["overest_count"] = overest_count
    stats_res["underest_ratio"] = underest_count / len(predicted)
    stats_res["overest_ratio"] = overest_count / len(predicted)

    return stats_res

'''
Evaluate binary classification results
- works with DT output
'''
def evaluate_binary_classification(predicted, actual):
    pred_classes = []
    for i in range(len(predicted)):
        if predicted[i] < 0.5:
            pred_classes.append(0)
        else:
            pred_classes.append(1)

    precision_S, recall_S, fscore_S, xyz = precision_recall_fscore_support(
            actual, pred_classes, average='binary', pos_label=0)
    precision_L, recall_L, fscore_L, xyz = precision_recall_fscore_support(
            actual, pred_classes, average='binary', pos_label=1)
    precision_W, recall_W, fscore_W, xyz = precision_recall_fscore_support(
            actual, pred_classes, average='weighted')

    stats_res = {}
    stats_res['small_error_precision'] = precision_S
    stats_res['small_error_recall'] = recall_S
    stats_res['small_error_fscore'] = fscore_S
    stats_res['large_error_precision'] = precision_L
    stats_res['large_error_recall'] = recall_L
    stats_res['large_error_fscore'] = fscore_L
    stats_res['precision'] = precision_W
    stats_res['recall'] = recall_W
    stats_res['fscore'] = fscore_W

    return stats_res

'''
Analyse features importance
- works only with RF
'''
def eval_feature_importance(model, X):
    print("Features importance: %s " % model.feature_importances_)
    importances = model.feature_importances_
    std = np.std([tree.feature_importances_ for tree in model.estimators_],
            axis=0)
    indices = np.argsort(importances)[::-1]
    print("Feature ranking:")
    for f in range(X.shape[1]):
        print("%d. feature %d (%f)" % (f + 1, indices[f], 
            importances[indices[f]]))
    plt.figure()
    plt.title("Feature importances")
    plt.bar(range(X.shape[1]), importances[indices],
                   color="r", yerr=std[indices], align="center")
    plt.xticks(range(X.shape[1]), indices)
    plt.xlim([-1, X.shape[1]])
    plt.show()

'''
Load data needed in order to create a prediction model
'''
def load_data_for_prediction(benchmark, benchmarks_home, data_set_dir, 
        data_set_dir_davide, benchmark_nVar, min_nbit, max_nbit):
    benchmark_dir = benchmarks_home + '/' + benchmark + '/'

    nVar = benchmark_nVar[benchmark]
    var_ranges = {}
    for i in range(nVar):
        var_ranges[i] = [x for x in range(min_nbit, max_nbit+1)]

    original_config = [max_nbit for i in range(nVar)]

    if benchmark == 'BlackScholes' or benchmark == 'Jacobi':
        single_file = False
    else:
        single_file = True

    if single_file:    # all experiments results are in the same file
        nSamples = 10000
        if benchmark == 'FWT':
            nSamples = 20000

        pickle_file = data_set_dir + "exp_results_" + benchmark + "_" + str(
                nSamples) + ".pickle"

        if os.path.isfile(pickle_file):
            with open(pickle_file, 'rb') as handle:
                exp_res = pickle.load(handle)
        else:          
            print('Data set not available for benchmark %s' % benchmark)
            return  [], [], [], []

    else:              # results are split in different files
        exp_res = read_multiple_result_file(data_set_dir_davide, 
                benchmark)

    errors = []
    configs = []
    vars_vals = {}
    data_dict = {}
    j = 0
    for k in exp_res.keys():
        config, error = exp_res[k]
        configs.append(config)
        errors.append(error)
        data_dict[j] = {'error': error}
        for i in range(len(config)):
            data_dict[j]['var_%s' % i] = config[i]
            if i in vars_vals:
                vars_vals[i].append(config[i])
            else:
                vars_vals[i] = [config[i]]
        j += 1
    return errors, configs, vars_vals, data_dict

'''
Load data needed in order to create a prediction model
- load input-set specific experiments
'''
def load_saved_data_inputSpecific(benchmark, benchmarks_home, data_set_dir, 
        data_set_dir_davide, benchmark_nVar, min_nbit, max_nbit, set_size,
        input_set_idx):
    benchmark_dir = benchmarks_home + '/' + benchmark + '/'

    nVar = benchmark_nVar[benchmark]
    var_ranges = {}
    for i in range(nVar):
        var_ranges[i] = [x for x in range(min_nbit, max_nbit+1)]

    original_config = [max_nbit for i in range(nVar)]

    if benchmark == 'BlackScholes' or benchmark == 'Jacobi':
        single_file = False
    else:
        single_file = True

    if single_file:    # all experiments results are in the same file
        nSamples = set_size

        pickle_file = "{}{}/exp_results_{}_{}.pickle".format(data_set_dir,
                input_set_idx, benchmark, nSamples)
        print(pickle_file)
        if os.path.isfile(pickle_file):
            with open(pickle_file, 'rb') as handle:
                exp_res = pickle.load(handle)
        else:          
            print('Data set not available for benchmark %s' % benchmark)
            return  [], [], [], []

    else:              # results are split in different files
        exp_res = read_multiple_result_file(data_set_dir_davide, 
                benchmark)

    errors = []
    configs = []
    vars_vals = {}
    data_dict = {}
    j = 0
    for k in exp_res.keys():
        config, error = exp_res[k]
        configs.append(config)
        errors.append(error)
        data_dict[j] = {'error': error}
        for i in range(len(config)):
            data_dict[j]['var_%s' % i] = config[i]
            if i in vars_vals:
                vars_vals[i].append(config[i])
            else:
                vars_vals[i] = [config[i]]
        j += 1
    return errors, configs, vars_vals, data_dict

'''
Evaluate solution found by optimizer.
- run the actual program
- check whether the generated error is smaller than the desired one
'''
def check_solution(benchmark, opt_config, trgt_error_ratio, benchmarks_home, 
        binary_map, large_error_threshold, benchmark_nVar, max_nbit, min_nbit,
        input_set_idx=-1):
    nVar = benchmark_nVar[benchmark]
    original_config = [max_nbit for i in range(nVar)]
    benchmark_dir = benchmarks_home + '/' + benchmark + '/'
    config_file = benchmark_dir + 'config_file.txt'
    write_conf(config_file, opt_config)
    if input_set_idx == -1:
        program = benchmark_dir + binary_map[benchmark] + '.sh'
        target_file = benchmark_dir + 'target.txt'
        target_result = read_target(target_file)
        error = run_program(program, target_result)
    else:
        program = benchmark_dir + binary_map[benchmark] + '_multiDataSet.sh'
        target_file = '{}targets/target_{}.txt'.format(benchmark_dir,
                input_set_idx)
        target_result = read_target(target_file)
        error = run_program_stoch(program, input_set_idx, target_result)

    # write back original config
    write_conf(config_file, original_config)
    is_error_se_trgt = (error <= trgt_error_ratio)
    if error < large_error_threshold:
        error_class = 0
    else:
        error_class = 1
    return error, is_error_se_trgt, error_class

'''
Evaluate solution found by optimizer.
- run the actual program
'''
def run_program_withConf(benchmark, opt_config, benchmarks_home, 
        binary_map, benchmark_nVar, max_nbit, min_nbit):
    benchmark_dir = benchmarks_home + '/' + benchmark + '/'
    program = benchmark_dir + binary_map[benchmark] + '.sh'
    config_file = benchmark_dir + 'config_file.txt'
    target_file = benchmark_dir + 'target.txt'
    target_result = read_target(target_file)
    nVar = benchmark_nVar[benchmark]
    original_config = [max_nbit for i in range(nVar)]
    write_conf(config_file, opt_config)
    with warnings.catch_warnings():
        warnings.filterwarnings('error')
        try:
            error = run_program(program, target_result)
        #except eWarning as Exception:
        except Warning:
            print("<<<<<<<<<<<<<<<<<<<< EXCEPPPPPP")
    # write back original config
    write_conf(config_file, original_config)
    return error

'''
Trace statistics
'''
def trace_stats(sol_stats, tracefile):
    if sol_stats == None:
        stats_str = '=======================================================\n'
        with open(tracefile, 'w+') as write_file:
            write_file.write(stats_str)
    else:
        stats_str = ('{0};{1};{2};{3};{4};{5};{6};{7};{8};{9};{10};{11};'
                '{12};{13};{14};{15};{16};{17};{18}\n'.format(
                sol_stats['config'], sol_stats['delta_config'], 
                sol_stats['error'], sol_stats['delta_error'],
                sol_stats['error_capped'], sol_stats['error_pred'], 
                sol_stats['delta_error_pred'], sol_stats['error_log'],
                sol_stats['delta_error_log'], sol_stats['error_pred_log'],
                sol_stats['delta_error_pred_log'], sol_stats['cost'], 
                sol_stats['C_Acc'], sol_stats['R_MAE'], sol_stats['R_R2'], 
                sol_stats['R_EV'], sol_stats['C_FI'], sol_stats['R_FI'],
                sol_stats['iteration_time']))
        with open(tracefile, 'a') as write_file:
            write_file.write(stats_str)

'''
Trace statistics
- compact format (human readable)
'''
def trace_stats_compact(sol_stats, tracefile):
    if sol_stats == None:
        stats_str = '=======================================================\n'
        with open(tracefile, 'w+') as write_file:
            write_file.write(stats_str)
    else:
        #sstr = ('Conf {0}; Conf delta {1}; Error {2:.3f}; Error Pred {3:.3f}; '
        #    'Error log {4:.3f}; Error Pred Log {5:.3f}; Sol Cost {6}; C_Acc'
        #    ' {7:.3f}; R_MAE {8:.3f}; R_R2 {9:.3f}; R_EV {10:.3f}; '
        #    'Iter time {11:.3f}\n'.format(
        #    sol_stats['config'], sol_stats['delta_config'], sol_stats['error'], 
        #    sol_stats['error_pred'], sol_stats['error_log'], 
        #    sol_stats['error_pred_log'], sol_stats['cost'], sol_stats['C_Acc'],
        #    sol_stats['R_MAE'], sol_stats['R_R2'], sol_stats['R_EV'],
        #    sol_stats['iteration_time']))
        sstr = ('Conf {0}; Err {1:.3f}; ErrPred {2:.3f}; '
            'ErrLog {3:.5f}; ErrPredLog {4:.3f}; Sol Cost {5}; Acc'
            ' {6:.3f}; MAE {7:.3f}; R2 {8:.3f}; EV {9:.3f}; FI_C {10};'
            ' FI_R {11}; Iter time {12:.3f}\n'.format(
            sol_stats['config'], sol_stats['error'], 
            sol_stats['error_pred'], sol_stats['error_log'], 
            sol_stats['error_pred_log'], sol_stats['cost'], sol_stats['C_Acc'],
            sol_stats['R_MAE'], sol_stats['R_R2'], sol_stats['R_EV'],
            sol_stats['C_FI'], sol_stats['R_FI'], sol_stats['iteration_time']))
        with open(tracefile, 'a') as write_file:
            write_file.write(sstr)

'''
Trace statistics
- bare minimum
'''
def trace_stats_bare(sol_stats, tracefile):
    if sol_stats == None:
        stats_str = '=======================================================\n'
        with open(tracefile, 'w+') as write_file:
            write_file.write(stats_str)
    else:
        sstr = ('Conf {0}; Err {1:.3f}; ErrPred {2:.3f}; Sol Cost {3}; Acc'
            ' {4:.3f}; MAE {5:.3f}; Iter time {6:.3f}\n'.format(
            sol_stats['config'], sol_stats['error'], 
            sol_stats['error_pred'], sol_stats['cost'], sol_stats['C_Acc'],
            sol_stats['R_MAE'], sol_stats['iteration_time']))
        with open(tracefile, 'a') as write_file:
            write_file.write(sstr)

'''
Pretty print trace statistics
'''
def print_trace_stats(sol_stats):
    print('[sol_stats] Conf {0}; Conf delta {1}; Error {2}; Error Pred {3}; '
            'Error log {4:.5f}; Error Pred Log {5:.3f}; Sol Cost {6}'.format(
            sol_stats['config'], sol_stats['delta_config'], sol_stats['error'], 
            sol_stats['error_pred'], sol_stats['error_log'], 
            sol_stats['error_pred_log'], sol_stats['cost']))

'''
Extract a subset of the whole data set 
- Ideally we want to obtain a subset that resembles the whole data set
'''
def select_data_subset(df_full, size, large_error_threshold, err_str='error'):
    # large errors are a big problem; we want to have them in our training set
    # at least with the same proportion they appear in the whol data set
    # TODO: the creation of the initial training set can be refined
    cnt_large_error_whole_data = len(df_full[
            df_full[err_str] >= large_error_threshold])
    large_error_ratio_whole_data = cnt_large_error_whole_data / len(df_full)

    # no large errors -- i.e. saxpy or convolution
    if large_error_ratio_whole_data == 0:
        df = df_full.sample(size)
        return df

    large_error_ratio_train_set = 0
    while large_error_ratio_train_set < large_error_ratio_whole_data:
        if size > len(df_full):
            size = len(df_full) - 1
        df = df_full.sample(size)
        large_error_ratio_train_set = len(df[
                df[err_str] >= large_error_threshold]) / len(df)

    return df

'''
Create training and test set for both the classification and the regression
task. This is useful in the active learning settings where we want  both
classifier and regressor to start from the same starting point
- the function take as input the data set size to be drawn and the size of the
  training set; i.e. data set size = 120 and train set size = 100
- the training set is extracted from the larger data set via a fixed split (in
  this way both classifier and regressor have the same training set and test
  set)
''' 
def create_train_test_sets(benchmark, benchmarks_home, data_set_dir,
        data_set_dir_davide, benchmark_nVar, min_nbit, max_nbit,
        value_inPlace_of_inf, errors_close_to_0_threshold,
        large_error_threshold, set_size, train_set_size):
    errors, configs, vars_vals, data_dict = load_data_for_prediction(
            benchmark, benchmarks_home, data_set_dir, data_set_dir_davide, 
            benchmark_nVar, min_nbit, max_nbit)

    df_full = pd.DataFrame.from_dict(data_dict)
    df_full = df_full.transpose()

    '''
    There could be some issues in giving np.inf value to configs with error
    equal to zero.
    '''
    df_error_zero = df_full[(df_full['error'] == 0)]
    df_full = df_full[(df_full['error'] != 0)]

    '''
    The error values are very small (i.e. 1e-10, 1e-40..) and the regressor
    struggles to distinguish between the targets (even after scaling). Since we
    are not interested in the prediction of the error _per se_ but only in the 
    relationships between number of bits assigned to variables and error, we
    can use the -log(error), in order to magnify the distance between target
    values.
    '''
    df_full['log_error'] = -np.log(df_full['error'])

    # artificially set config with no error to large error logs
    df_error_zero['log_error'] = value_inPlace_of_inf
    frames = [df_full, df_error_zero]
    result = pd.concat(frames)
    df_full = result
    df_full = df_full.sample(frac=1).reset_index(drop=True)

    #'''
    #error values extremely close to zero (both negative and positive create 
    #lots of problems. I pretend they do not exist for this test
    #'''
    if benchmark != 'Jacobi':
        df_full = df_full[(df_full['log_error'] <= 
            -errors_close_to_0_threshold) | 
                (df_full['log_error'] >= errors_close_to_0_threshold)]

    if set_size == 'FULL':
        size = len(df_full)
        df = df_full
    else:
        size = set_size

    df = select_data_subset(df_full, size, large_error_threshold)

    df = df[(df != 0).all(1)]

    def error_class(row):
        if row['error'] >= large_error_threshold:
            return 1
        else:
            return 0
    df['error_class'] = df.apply(error_class, axis=1)
   
    target_regr = df['log_error']
    target_classr = df['error_class']

    # compute variance / std
    df_varsOnly = df.copy()
    del df_varsOnly['error']
    del df_varsOnly['log_error']
    del df_varsOnly['error_class']
    df['var'] =  df_varsOnly.std(axis=1)

    #print(df['error'])
    #print(target_regr)
    #print(target_classr)
    #sys.exit()

    del df['error']
    del df['log_error']
    del df['error_class']

    train_data_regr = df[:train_set_size]
    train_data_classr = df[:train_set_size]

    train_target_regr = target_regr[:train_set_size]
    train_target_classr = target_classr[:train_set_size]

    test_data_regr = df[train_set_size:]
    test_data_classr = df[train_set_size:]

    test_target_regr = target_regr[train_set_size:]
    test_target_classr = target_classr[train_set_size:]

    return (train_data_regr, test_data_regr, train_target_regr,
            test_target_regr, train_data_classr, test_data_classr,
            train_target_classr, test_target_classr)

'''
Create training and test set for both the classification and the regression
task. This is useful in the active learning settings where we want  both
classifier and regressor to start from the same starting point
- the function take as input the data set size to be drawn and the size of the
  training set; i.e. data set size = 120 and train set size = 100
- the data is extracted via random sampling from the set of experiments made
  with LHS at ~9k samples
- the training set is extracted from the larger data set via a fixed split (in
  this way both classifier and regressor have the same training set and test
  set)
''' 
def create_train_test_sets_randomSample10kLHS(benchmark, benchmarks_home,
        data_set_dir, data_set_dir_davide, benchmark_nVar, min_nbit, max_nbit,
        value_inPlace_of_inf, errors_close_to_0_threshold,
        large_error_threshold, set_size, train_set_size,
        trgt_error_ratio_log_exp, load_initial=False, saved_df='', 
        weights_based_on_target=False, input_set_idx=0,
        max_train_set_size=9000):

    df_full = prepare_df_stoch_meanStd(benchmark, 9000,
            large_error_threshold, 0.000001, data_set_dir)

    good_cols = []
    to_del_cols = []
    for c in df_full.columns:
        str_ok = 'ds_{}'.format(input_set_idx)
        if 'var_' in c:
            good_cols.append(c)
        elif str_ok in c:
            good_cols.append(c)
        else:
            to_del_cols.append(c)
    for c in to_del_cols:
        del df_full[c]

    err_str = 'err_ds_{}'.format(input_set_idx)
    log_err_str = 'log_err_ds_{}'.format(input_set_idx)
    class_err_str = 'err_class_ds_{}'.format(input_set_idx)

    df_error_zero = df_full[(df_full[err_str] == 0)]
    df_full = df_full[(df_full[err_str] != 0)]

    # artificially set config with no error to large error logs
    df_error_zero[log_err_str] = value_inPlace_of_inf
    frames = [df_full, df_error_zero]
    result = pd.concat(frames)
    df_full = result
    df_full = df_full.sample(frac=1).reset_index(drop=True)

    #'''
    #error values extremely close to zero (both negative and positive create 
    #lots of problems. I pretend they do not exist for this test
    #'''
    if benchmark != 'Jacobi':
        df_full = df_full[(df_full[log_err_str] <= 
            -errors_close_to_0_threshold) | 
                (df_full[log_err_str] >= errors_close_to_0_threshold)]

    if set_size == 'FULL':
        size = len(df_full)
        df = df_full
    else:
        size = set_size

    df = select_data_subset(df_full, size, large_error_threshold, err_str)

    df = df[(df != 0).all(1)]

    def error_class(row):
        if row[err_str] >= large_error_threshold:
            return 1
        else:
            return 0
    df[class_err_str] = df.apply(error_class, axis=1)
   
    target_regr = df[log_err_str]
    target_classr = df[class_err_str]

    if len(set(target_classr.tolist())) > 1:
        classr_needed = True
    else:
        classr_needed = False
    del df[err_str]
    del df[log_err_str]
    del df[class_err_str]

    if len(df) < train_set_size:
        train_set_size = len(df) - 2

    train_data_regr = df[:train_set_size]
    train_data_classr = df[:train_set_size]

    train_target_regr = target_regr[:train_set_size]
    train_target_classr = target_classr[:train_set_size]

    test_data_regr = df[train_set_size:]
    test_data_classr = df[train_set_size:]

    test_target_regr = target_regr[train_set_size:]
    test_target_classr = target_classr[train_set_size:]

    weights_regr = []

    return (train_data_regr, test_data_regr, train_target_regr,
            test_target_regr, train_data_classr, test_data_classr,
            train_target_classr, test_target_classr, classr_needed, 
            weights_regr)

'''
Create training and test set for both the classification and the regression
task. This is useful in the active learning settings where we want  both
classifier and regressor to start from the same starting point
- the function take as input the data set size to be drawn and the size of the
  training set; i.e. data set size = 120 and train set size = 100
- the training data is obtained by selecting the LHS experiments performed for
  the target training set size
- the test set is extracted via random sampling from the set of experiments made
  with LHS at ~9k samples
- the training set is extracted from the larger data set via a fixed split (in
  this way both classifier and regressor have the same training set and test
  set)
''' 
def create_train_test_sets_directLHS(benchmark, benchmarks_home,
        data_set_dir, data_set_dir_davide, benchmark_nVar, min_nbit, max_nbit,
        value_inPlace_of_inf, errors_close_to_0_threshold,
        large_error_threshold, set_size, train_set_size,
        trgt_error_ratio_log_exp, load_initial=False, saved_df='', 
        weights_based_on_target=False, input_set_idx=0, 
        max_train_set_size=9000):

    def error_class(row):
        if row[err_str] >= large_error_threshold:
            return 1
        else:
            return 0
    err_str = 'err_ds_{}'.format(input_set_idx)
    log_err_str = 'log_err_ds_{}'.format(input_set_idx)
    class_err_str = 'err_class_ds_{}'.format(input_set_idx)
    str_ok = 'ds_{}'.format(input_set_idx)

    # ------------ prepare train set -----------------------------
    df = prepare_df_stoch_meanStd(benchmark, train_set_size,
            0.9, 0.000001, data_set_dir, True, True)

    good_cols = []
    to_del_cols = []
    for c in df.columns:
        if 'var_' in c:
            good_cols.append(c)
        elif str_ok in c:
            good_cols.append(c)
        else:
            to_del_cols.append(c)
    for c in to_del_cols:
        del df[c]
    df_error_zero_train = df[(df[err_str] == 0)]
    df = df[(df[err_str] != 0)]
    # artificially set config with no error to large error logs
    df_error_zero_train[log_err_str] = value_inPlace_of_inf
    frames = [df, df_error_zero_train]
    df = pd.concat(frames)
    df = df[(df != 0).all(1)]
    df[class_err_str] = df.apply(error_class, axis=1)
    seriesObj = df.apply(
            lambda x: True if x[class_err_str] == 1 else False, axis=1)
    num_large_errs = len(seriesObj[seriesObj == True].index)
    ratio_large_errs = num_large_errs / len(df)
    # we want a "balanced" training set, a set that contains a large enough
    # number of examples with an error smaller than the error target (to improve
    #  the accuracy of the initial regressor). For this goal, if the initial
    # training sets contain to many large errors, we extend it and add new
    # examples (until a certain max size is reached)
    df_ext = df
    if ratio_large_errs > 0.5:
        lb_lhs = math.ceil(min_nbit + (max_nbit - min_nbit) / 2)
        ub_lhs = max_nbit
        while ratio_large_errs > 0.5 and len(df_ext) < max_train_set_size:
            for c in to_del_cols:
                del conf_add[c]
            added_zero_train = conf_add[(conf_add[err_str] == 0)]
            conf_add = conf_add[(conf_add[err_str] != 0)]
            # artificially set config with no error to large error logs
            added_zero_train[log_err_str] = value_inPlace_of_inf
            frames = [conf_add, added_zero_train]
            conf_add = pd.concat(frames)
            #conf_add = conf_add[(conf_add != 0).all(1)]
            conf_add[class_err_str] = conf_add.apply(error_class, axis=1)
            dfs = [df, conf_add]
            df_ext = pd.concat(dfs)
            df = df_ext
            seriesObj = df.apply(
                    lambda x: True if x[class_err_str] == 1 else False, axis=1)
            num_large_errs = len(seriesObj[seriesObj == True].index)
            ratio_large_errs = num_large_errs / len(df)
            lb_lhs = math.ceil(lb_lhs + (ub_lhs - lb_lhs) / 2)
            if lb_lhs > 50:
                lb_lhs = 50 
    df_train = df_ext
    df_train = df_train.drop_duplicates()
    #print(df_train)
    #print('Ratio large errs {}'.format(ratio_large_errs))
    #sys.exit()

    target_regr = df_train[log_err_str]
    target_classr = df_train[class_err_str]
    if len(set(target_classr.tolist())) > 1:
        classr_needed = True
    else:
        classr_needed = False

    if weights_based_on_target:
        weights_regr = compute_sample_weights_regr(df_train, err_str,
                log_err_str, large_error_threshold)
    else:
        weights_regr = []
    del df_train[err_str]
    del df_train[log_err_str]
    del df_train[class_err_str]
    train_data_regr = df_train
    train_data_classr = df_train
    train_target_regr = target_regr
    train_target_classr = target_classr
    # ------------ end prepare train set ---------------------------

    # ------------ prepare test set ---------------------------
    df_full_test = prepare_df_stoch_meanStd(benchmark, 9000,
            large_error_threshold, 0.000001, data_set_dir)
    for c in to_del_cols:
        del df_full_test[c]
    df_error_zero_test = df_full_test[(df_full_test[err_str] == 0)]
    df_full_test = df_full_test[(df_full_test[err_str] != 0)]
    # artificially set config with no error to large error logs
    df_error_zero_test[log_err_str] = value_inPlace_of_inf
    frames = [df_full_test, df_error_zero_test]
    result = pd.concat(frames)
    df_full_test = result
    df_full_test = df_full_test.sample(frac=1).reset_index(drop=True)
    if set_size == 'FULL':
        size = len(df_full_test)
        df_test = df_full_test
    else:
        size = set_size
    df_test = select_data_subset(df_full_test, size, large_error_threshold, err_str)
    df_test = df_test[(df_test != 0).all(1)]
    df_test[class_err_str] = df_test.apply(error_class, axis=1)
    target_regr = df_test[log_err_str]
    target_classr = df_test[class_err_str]
    if len(set(target_classr.tolist())) > 1:
        classr_needed = True
    else:
        classr_needed = False
    del df_test[err_str]
    del df_test[log_err_str]
    del df_test[class_err_str]
    test_data_regr = df_test[train_set_size:]
    test_data_classr = df_test[train_set_size:]
    test_target_regr = target_regr[train_set_size:]
    test_target_classr = target_classr[train_set_size:]
    # ------------ end prepare test set ---------------------------

    return (train_data_regr, test_data_regr, train_target_regr,
            test_target_regr, train_data_classr, test_data_classr,
            train_target_classr, test_target_classr, classr_needed, 
            weights_regr)

'''
Retrieve experiments results from text file and produce the data frame for
learning
- mean & std computed over all data sets (regression_meanStd_standardLoss)
'''
def prepare_df_stoch_meanStd(benchmark, data_set_size, 
        large_error_value, small_std_value, exp_res_dir, 
        directLHS=False, flatten_large_errors=True):
    if directLHS:
        datafile = exp_res_dir + 'exp_results_{}_{}.csv'.format(benchmark,
                data_set_size)
    else:
        datafile = exp_res_dir + 'exp_results_{}.csv'.format(benchmark)
        #datafile=exp_res_dir+'exp_results_{}_UNCERTAIN.csv'.format(benchmark)
        #datafile = exp_res_dir + 'exp_results_{}_260.csv'.format(benchmark)

    df = pd.read_csv(datafile, sep=';')
    del df['err_mean']
    del df['err_std']

    if flatten_large_errors:
        # errors larger than a threshold are flattened
        for c in df.columns:
            if 'err_ds_' in c:
                df.loc[df[c] > large_error_value, c] = large_error_value

    log_errs_cols = []
    errs_cols = []
    vars_cols = []
    for c in df.columns:
        if 'var_' in c:
            vars_cols.append(c)
        if 'err_' in c:
            new_name = 'log_{}'.format(c)
            df[new_name] = -np.log(df[c])
            if 'std' not in new_name and 'mean' not in new_name:
                log_errs_cols.append(new_name)
                errs_cols.append(c)

    df['log_err_mean'] = np.mean(df[log_errs_cols], axis=1)
    df['log_err_std'] = np.std(df[log_errs_cols], axis=1)
    df['err_mean'] = np.mean(df[errs_cols], axis=1)
    df['err_std'] = np.std(df[errs_cols], axis=1)

    # in some case the std is 0 (i.e for all data sets no output was produced)
    # we cannot take the log of zero, hence conversion to a small number != 0
    df.loc[df['err_std'] == 0, 'err_std'] = small_std_value

    if not directLHS:
        if data_set_size != 'FULL' and data_set_size > len(df):
            data_set_size = len(df)
        if data_set_size != 'FULL':
            df = df.sample(data_set_size)
    return df

'''
Retrieve experiments results from text file and produce the data frame for
learning
- work only for the regressor (predicting the error)
- the mean is going to be the value used by the MP to predict the error
  associated to a configuration while the standard deviation can be thought as
  the confidence of the ML model (regression_confDistr_mle)
- the df need to be reshaped: 
    config_0 err_ds_0
    config_0 err_ds_1
    ...
    config_0 err_ds_29
    config_1 err_ds_0
    config_1 err_ds_1
    ...
    config_1 err_ds_29
    ...
    config_N err_ds_0
    config_N err_ds_1
    ...
    config_N err_ds_29
'''
def prepare_df_stoch_confDistr(benchmark, data_set_size, 
        flatten_large_error_value, small_std_value, exp_res_dir):
    # errors larger than a threshold are dropped

    # we use the inverted notation (see above) -- if not present we compute it
    datafile_rev = exp_res_dir + 'exp_results_rev_{}.csv'.format(benchmark)
    if os.path.isfile(datafile_rev):
        df = pd.read_csv(datafile_rev, sep=';', index_col=0)
    else:
        datafile = exp_res_dir + 'exp_results_{}.csv'.format(benchmark)

        df = pd.read_csv(datafile, sep=';')
        del df['err_mean']
        del df['err_std']

        errs_cols = []
        vars_cols = []
        for c in df.columns:
            if 'var_' in c:
                vars_cols.append(c)
            if 'err_' in c:
                errs_cols.append(c)

        new_df = []
        for index, row in df.iterrows():
            for ec in errs_cols:
                d = {}
                for vc in vars_cols:
                    d[vc] = row[vc]
                if row[ec] > flatten_large_error_value:
                    d['err'] = flatten_large_error_value
                    d['log_err'] = -np.log(flatten_large_error_value)
                else:
                    d['err'] = row[ec]
                    d['log_err'] = -np.log(row[ec])
                new_df.append(d)
        df = pd.DataFrame(new_df)
        df.to_csv(datafile_rev, sep=';')

    return df

'''
Split data set in training set and test set (stochastic approach)
- work only for the regressor (predicting the error)
- mean & std computed over all data sets (regression_meanStd_standardLoss)
'''
def split_data_set_stoch_regr_meanStd(df, set_split, flatten_large_error_value, 
        scaled=False):
    # errors larger than a threshold are dropped
    df = df[df['err_mean'] != flatten_large_error_value]
    df = df[df != flatten_large_error_value]
    df = df[df != np.log(flatten_large_error_value)]
    target_cols = []
    for c in df.columns:
        if 'ds' in c:
            del df[c]
        if c == 'log_err_mean' or c == 'log_err_std':
            target_cols.append(c)
        if c == 'err_mean' or c == 'err_std':
            del df[c]
    target = df[target_cols]
    #target['log_err_mean'] = target['log_err_mean'].mul(-1)
    target.loc[:,'log_err_mean'] *= -1
    for c in target_cols:
        del df[c]

    if scaled:
        scaler_data = StandardScaler()
        df = scaler_data.fit_transform(df)
        scaler_target = MinMaxScaler()
        target = scaler_target.fit_transform(target)
    else:
        scaler_data = -1
        scaler_target = -1

    msk = np.random.rand(len(df)) < set_split
    train_data = df[msk]
    train_target = target[msk]
    test_data = df[~msk]
    test_target = target[~msk]
    return (train_data, test_data, train_target, test_target, scaler_data,
            scaler_target)

'''
Split data set in training set and test set (stochastic approach)
- work only for the regressor (predicting the error)
- the mean is going to be the value used by the MP to predict the error
  associated to a configuration while the standard deviation can be thought as
  the confidence of the ML model (regression_confDistr_mle)
'''
def split_data_set_stoch_regr_confDistr(df, set_split, 
        threshold_large_error, scaled=False, double_log=False, 
        scaler_type_data='std', scaler_type_target='std'):
    # errors larger than a threshold are dropped
    df = df[df['err'] != threshold_large_error]
    # the ML model use the log error only
    del df['err']

    if double_log:
        df['log_err'] *= -1
        df['log_err'] = -np.log(df['log_err'])

    target = df['log_err']
    target *= -1
    del df['log_err']
    target_array = target.values.astype(float).reshape(-1, 1)
    target_array[~np.isfinite(target_array)] = 0

    if scaled:
        if scaler_type_data == 'std':
            scaler_data = StandardScaler()
        elif scaler_type_data == 'minmax': 
            scaler_data = MinMaxScaler()
        df = scaler_data.fit_transform(df)

        if scaler_type_target == 'std':
            scaler_target = StandardScaler()
        elif scaler_type_target == 'minmax': 
            scaler_target = MinMaxScaler()
        target = scaler_target.fit_transform(target_array)
    else:
        scaler_data = -1
        scaler_target = -1

    msk = np.random.rand(len(df)) < set_split
    train_data = df[msk]
    train_target = target[msk]
    test_data = df[~msk]
    test_target = target[~msk]

    return (train_data, test_data, train_target, test_target, scaler_data,
            scaler_target)

def read_var_ops(benchmark, var_ops_file):
    with open(var_ops_file, 'r') as stream:
        try:
            var_ops_info = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print("Problem loading yaml file {}".format(exc))
    max_use_refs = max(var_ops_info['use_refs'])
    use_refs_norm = [x / max_use_refs for x in var_ops_info['use_refs']]
    var_ops_info['use_refs_norm'] = use_refs_norm
    use_refs_inv = []
    for x in var_ops_info['use_refs']:
        if x != 0:
            use_refs_inv.append(max_use_refs / x)
        else:
            use_refs_inv.append(max_use_refs + 1)
    var_ops_info['use_refs_inv'] = use_refs_inv
    #var_weights = [int(x * 10) for x in use_refs_inv]
    var_weights = [int(x) for x in use_refs_inv]
    var_ops_info['var_weights'] = var_weights
    return var_ops_info

'''
Compute sample weights based on error target
- function to compute weight for the regressor
'''
def compute_sample_weights_regr(df, err_str, log_err_str, target):
    log_target = -np.log(target)
    dif_target = (1/(df[log_err_str]-log_target))
    # large errors (negative values when taking -np.log) have no weight
    ws_no_large_err = dif_target.apply(lambda x: x if x > 0 else 0.05)
    ws = ws_no_large_err.apply(lambda x: 2 if x > 2 else x)
    return np.asarray(ws.tolist())

'''
Increment the precision of the configuration given as input. The precision is
changed according to the number of operation in which a variable is involved.
'''
def var_ops_based_sol_update(conf, var_ops_info, max_nbit, mult):
    new_conf = []
    for i in range(len(conf)):
        p = random.random()
        if p < mult * var_ops_info['use_refs_norm'][i]:
            c = random.randrange(1, 5)
            if conf[i] + c < max_nbit:
                new_conf.append(conf[i] + c)
            else:
                new_conf.append(max_nbit)
        else:
            new_conf.append(conf[i])
    return new_conf

    










