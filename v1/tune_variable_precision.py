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
#!/usr/bin/python3.6

import numpy as np
import sys
import os
import time
import math
import utils
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, median_absolute_error
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.metrics import explained_variance_score, accuracy_score, log_loss
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
eml_path = './'
if not eml_path in sys.path:
    sys.path.insert(1, eml_path)
sys.path.insert(1, eml_path)
from eml import util
from eml.tree import describe as tdescribe
from eml.tree import embed as tembed
from eml.backend import cplex_backend
from eml.tree.reader import sklearn_reader
from eml.net import describe as ndescribe
from eml.net.reader import keras_reader
from eml.net import process as nprocess
from eml.net import embed as nembed
import docplex.mp.model as cpx
import cplex
from docplex.mp.solution import SolveSolution
from docplex.mp.model_reader import ModelReader
import tensorflow as tf
from keras.models import Sequential, Model
from keras import backend as K 
from keras import optimizers, initializers, regularizers
from keras.layers import Dense
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from keras.wrappers.scikit_learn import KerasRegressor
import warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

static_tuner_dir = '../'
benchmarks_home = static_tuner_dir + 'benchmarks'
data_set_dir = static_tuner_dir + 'precision_and_errors/'

DEBUG = False

min_nbit = 4
max_nbit = 53

# max desirable error exponent (larger exp --> smaller error)
max_target_error_exp = 80    # i.e. 1e-50

large_error_threshold = 0.9
errors_close_to_0_threshold =  0.05
classifier_threshold = 0.2
value_inPlace_of_inf = 1000

max_dt_depth = 20
max_depth_class_dt = {'convolution' : 5, 'correlation' : 10, 'saxpy' : 5}

epochs = 100
batch_size = 32
alpha_asymm_loss = 0.75

opt_time_limit = 30
bounds_opt_time_limit = 600

nvar_bounds_tightening = 10

max_refine_iterations = 2000

scaling = True   # apply standardization to data set for ML?
initial_dataset_size = 8000
initial_train_set_size = 7000

# how many times we retry the NN regressor if the maximum predicted error is
# smaller than the log target error (after this value, we increase the size of
# the initial training set)
initial_regr_tries = 20

# new configurations are given different (greater) weight
new_config_weight = 2

increase_step = 5
increase_freq = 10
increase_freq_begin = 2

# constant to be added to constraint y_var >= trgt_error_ratio_log_exp
# in order to make the solution more robust to numerical issues
# (in some cases the optimal solution found by cplex is not a valid
# solution for the real problem - it doesn't respect the desired error)
global_robust_param = 0

max_loss_threshold = 50

binary_map = {'convolution' : 'conv2', 'correlation' : 'corr2',  
    'saxpy' : 'saxpy2'}
benchmark_nVar = {'convolution': 4, 'correlation' : 7, 'saxpy' : 3}

'''
Optimization model with embedded regressor and classifier
1) regressor: used to predict error given bit config (a solution)
2) classifier: used to classify config in 'good' (error lower than threshold) 
    and bad (error larger than threshold)
Decision variables = n.bits assigned to each benchmark variable: b_i for i in 
    number of benchmark vars (nVars)
'''
def solve_opt_model(benchmark, mdl, trgt_error_ratio_log_exp, regr, mae, rmse, 
        underest_ratio, classr, acc, prev_bit_sum, increase_tot_nbits,
        train_set_size, large_err_thresh, n_iter=-1, wrong_config=[], 
        debug=True):
    if mdl == None:
        if debug:
            print("\tMP model needs to be created or loaded")
        trgt_error_ratio_log_exp = math.ceil(trgt_error_ratio_log_exp)
        robust_param = global_robust_param 

        # The requested target error (plus eventual other stuff) can be
        # infeasible, according to the prediction model, because it could be
        # greater than the largest error that can be estimated (we are dealing 
        # with exponents here, hence the emphasis on _greater_ values). The  
        # largest possible error attainable according to prediction model is:
        # max_error = pred_model(max_nbit, max_nbit, max_nbit). If we encounter 
        # this situation we force the optimizer to return the config with max 
        # nbits for each var, since the opt problem would be unsolvable
        max_conf = [max_nbit for i in range(benchmark_nVar[benchmark])]
        max_conf_dict = {}
        for i in range(len(max_conf)):
            max_conf_dict['var_{}'.format(i)] = [max_conf[i]]
        max_conf_df = pd.DataFrame.from_dict(max_conf_dict)
        max_predictable_error = regr.predict(max_conf_df)[0][0]
        if max_predictable_error < trgt_error_ratio_log_exp:
            dif = trgt_error_ratio_log_exp - max_predictable_error
            robust_param = -math.ceil(dif)

        # Build a backend object
        bkd = cplex_backend.CplexBackend()

        # create MP model from scratch
        mdl = create_MP_model(benchmark, bkd, 
                trgt_error_ratio_log_exp, regr, mae, rmse, underest_ratio, 
                classr, acc, robust_param, large_err_thresh, debug)

        # the error constraint changes depending on the program input
        y_var = mdl.get_var_by_name('y')
        mdl.add_constraint(y_var >= trgt_error_ratio_log_exp + robust_param)

        sum_bit_var = mdl.get_var_by_name('sum_bit')
        if increase_tot_nbits:
            new_nbit_sum = math.ceil(prev_bit_sum + increase_step)
            mdl.add_constraint(sum_bit_var >= new_nbit_sum)
        else:
            mdl.add_constraint(sum_bit_var >= prev_bit_sum)
            new_nbit_sum = prev_bit_sum

    '''
    Cut previous solutions
    '''
    if n_iter > 0 and len(wrong_config) > 1:
        if debug:
            print("\t Cut previous solution")
        cut_solution(mdl, wrong_config, n_iter)

    '''
    Solve optimization model
    '''
    mdl.set_time_limit(opt_time_limit)
    if debug:
        print('=== Starting the solution process (Timelimit {}s)'.format(
            opt_time_limit))
    before_solve_time = time.time()

    sol = mdl.solve()
    after_solve_time = time.time()
    if debug:
        print("Time needed to solve MP & EML model {}".format(
            after_solve_time - before_solve_time))

    if sol is None:
        if debug:
            print('No solution found')
        opt_config = None
    else:
        opt_config = []
        if debug:
            print('=== Solution Data')
            print('Solution time: {:.2f} (sec)'.format(
                mdl.solve_details.time))
            print('Solver status: {}'.format(sol.solve_details.status))
        for i in range(benchmark_nVar[benchmark]):
            name='x_{}'.format(i)
            if debug:
                print('\t# Bits for {}: {}'.format(name, sol[name]))
            opt_config.append(int(sol[name]))
        if debug:
            print('\tY value: {}'.format(sol['y']))

    return opt_config, mdl, new_nbit_sum

'''
Create model from scratch
'''
def create_MP_model(benchmark, bkd, trgt_error_ratio_log_exp, regr, mae, rmse,
        underest_ratio, classr, acc, robust_param, large_err_thresh,
        debug=True):

    before_modelEM_time = time.time()
    '''
    Create optimization model
    '''
    # Build a docplex model
    mdl = cpx.Model()

    x_vars = []
    for i in range(benchmark_nVar[benchmark]):
        x_vars.append(mdl.integer_var(
            lb=min_nbit, ub=max_nbit, name='x_{}'.format(i)))

    # output of the NN is a floating number, the negative log of error exp  
    # the upper bound is computed in function of the max target error exp
    ub_y = -np.log(float('1e-' + str(max_target_error_exp)))
    # y might have negative values, in case the predicted error is very large
    # artificially tighten the bound of y_var in order to limit its range: the
    # lower bound corresponds to neg log of the error threshol for large error
    lb_y = -np.log(large_err_thresh)
    y_var = mdl.continuous_var(lb=lb_y, ub=ub_y, name='y')

    lb_sum_var = min_nbit * benchmark_nVar[benchmark]
    ub_sum_var = max_nbit * benchmark_nVar[benchmark]
    sum_bit_var = mdl.integer_var(lb=lb_sum_var, ub=ub_sum_var, name='sum_bit')

    # c_var represents the config class (large or small error)
    if classr != None:
        c_var = mdl.continuous_var(lb=0, ub=1, name='c')

    '''
    EML Regressor
    '''
    # convert Keras NN in EML format
    regr_em = keras_reader.read_keras_sequential(regr)
    # Reset existing bounds (just to ensure idempotence)
    regr_em.reset_bounds()
    # Enforce basic input bounds
    in_layer = regr_em.layer(0)
    for neuron in in_layer.neurons():
        neuron.update_lb(min_nbit)
        neuron.update_ub(max_nbit)

    before_regrNN_propagate_time = time.time()

    # Compute bounds for the hidden neurons via Interval Based Reasoning
    nprocess.ibr_bounds(regr_em)

    # Tighten bounds via MILP (only for benchmarms with many vars)
    if benchmark_nVar[benchmark] > nvar_bounds_tightening:
        bounds_bkd = cplex_backend.CplexBackend()
        nprocess.fwd_bound_tighthening(bounds_bkd, regr_em, 
                timelimit=bounds_opt_time_limit)
        if debug:
            print("- Tighten bounds")

    after_regrNN_propagate_time = time.time()
    if debug:
        print("Time needed to compute bounds of NN regressor {0}".format(
            after_regrNN_propagate_time - before_regrNN_propagate_time))

    nembed.encode(bkd, regr_em, mdl, x_vars, y_var, 'regr_NN')

    '''
    EML Classifier
    '''
    if classr != None:
        classr_em = sklearn_reader.read_sklearn_tree(classr)

        # the bounds for the DT attributes need to be manually specified
        for attr in classr_em.attributes():
            classr_em.update_ub(attr, max_nbit)
            classr_em.update_lb(attr, min_nbit)

        tembed.encode_backward_implications(bkd, classr_em, mdl, x_vars, 
                c_var, 'classr_DT')

        # add constraints on not allowed configurations (the ones found by the 
        # classifier --> the (predicted) class of the config must be the small
        # error class (corresponding to value 0; 1 --> large error)
        #     ==> predicted class < 0.5
        mdl.add_constraint(c_var <= classifier_threshold)
        mdl.add_constraint(sum_bit_var == sum(x_vars))

    # obj: minimize number of bit per variable
    mdl.minimize(mdl.sum(x_vars))

    after_modelEM_time = time.time()
    if debug:
        print("Time needed to create MP model {}".format(
            after_modelEM_time - before_modelEM_time))

    return mdl

'''
Asymmetric loss function: it places more weight to underestimates
than overestimates
alpha parameter controls the relative weight (negative values -> underest.
have more weight, positive values -> overest. have more weight)
'''
def asymmetric_loss(y_true, y_pred):
    alpha_formula = -alpha_asymm_loss
    sq_error = K.square(y_pred - y_true)
    weight = K.square(K.sign(y_pred - y_true) + alpha_formula)
    return sq_error * weight

'''
Create NN model to predict error for a configuration of # bits for each
    variable. Relies on experiments already made as a data set, fails if data
    is absent.
'''
def prediction_model_NN(benchmark, alpha, set_size, train_data,
        test_data, train_target, test_target, large_err_thresh, 
        verbose=True):
    if scaling:
        scaler = MinMaxScaler()
        train_data_tensor = scaler.fit_transform(train_data)
        test_data_tensor = scaler.fit_transform(test_data)
        train_target_tensor = scaler.fit_transform(
                train_target.values.reshape(-1, 1))
        test_target_tensor = scaler.fit_transform(
                test_target.values.reshape(-1, 1))
    else:
        train_data_tensor = train_data.values
        test_data_tensor = test_data.values
        train_target_tensor = train_target.values
        test_target_tensor = test_target.values

    n_samples, n_features = train_data_tensor.shape
    input_shape = (train_data_tensor.shape[1],)

    pred_model = Sequential()
    pred_model.add(Dense(n_features * 2, activation='relu',
        activity_regularizer=regularizers.l1(1e-5),
        input_shape=input_shape))
    pred_model.add(Dense(n_features, activation='relu'))
    pred_model.add(Dense(1, activation='linear'))

    model_creation_time = time.time() 
    early_stopping = EarlyStopping(
            monitor='val_loss', patience=10, min_delta=1e-5) 
    reduce_lr = ReduceLROnPlateau(
            monitor='val_loss', patience=5, min_lr=1e-5, factor=0.2) 

    pred_model.compile(optimizer='adam', loss=asymmetric_loss)
    history = pred_model.fit(train_data_tensor, train_target_tensor,
            epochs=epochs, batch_size=batch_size, shuffle=True,
            validation_split=0.1, verbose=verbose, 
            callbacks=[early_stopping, reduce_lr])

    model_train_time = time.time()
    if verbose:
        print(">>> NN model created in {} s".format(
            model_train_time-model_creation_time))

    predicted = pred_model.predict(test_data_tensor)
    actual = test_target_tensor
    stats_res = utils.evaluate_predictions(predicted, actual)
    test_loss = pred_model.evaluate(test_data_tensor, test_target_tensor, 
            verbose=0)
    stats_res["test_loss"] = test_loss
    if verbose:
        print("MAE: {0:.3f}".format(stats_res["MAE"]))
        print("MSE: {0:.3f}".format(stats_res["MSE"]))
        print("RMSE: {0:.3f}".format(stats_res["RMSE"]))

    max_conf = [max_nbit for i in range(benchmark_nVar[benchmark])]
    max_conf_dict = {}
    for i in range(len(max_conf)):
        max_conf_dict['var_{}'.format(i)] = [max_conf[i]]
    max_conf_df = pd.DataFrame.from_dict(max_conf_dict)
    max_pred_err = pred_model.predict(max_conf_df)[0][0]

    return (pred_model, stats_res['MAE'], stats_res['RMSE'], stats_res['R2'],
            stats_res['EV'], stats_res['accuracy'], 
            stats_res['underest_ratio'], test_loss, max_pred_err)

def classifier_DT(benchmark, data_set_size, train_data,
        test_data, train_target, test_target, large_err_thresh,
        verbose=True):
    classes = train_target.tolist()
    weights = [1] * len(classes)

    classr = DecisionTreeClassifier(max_depth=max_depth_class_dt[benchmark])
    classr.fit(train_data, train_target, sample_weight=weights)
    predicted = classr.predict(test_data)
    stats_res = utils.evaluate_binary_classification(predicted,test_target)
    stats_res["accuracy"] = accuracy_score(test_target,predicted)

    if verbose:
        print("Weigthed F-Score: {0:.3f}".format(stats_res["fscore"]))
        print("Accuracy: {0:.3f}".format(stats_res["accuracy"]))
    return classr, stats_res["accuracy"], stats_res["fscore"]

'''
Return prediction/classification for a given config 
- we suppose only 2 classes: large error (class 1) and small errors (0)
'''
def get_pred_class(config, regr, classr):
    conf_dict = {}
    for i in range(len(config)):
        conf_dict['var_{}'.format(i)] = [config[i]]
    conf_df = pd.DataFrame.from_dict(conf_dict)
    prediction_with_conf = regr.predict(conf_df)[0]
    if classr == None:
        classPred_with_conf = 0
    else:
        classPred_with_conf = classr.predict(conf_df)[0]
    return prediction_with_conf[0], classPred_with_conf

'''
Exclude a single solution from the feasible ones (as a constraint)
- alternative version
'''
def cut_solution(mdl, cut_sol, n_iter):
    bin_vars_cut_vals = []
    for i in range(len(cut_sol)):
        x_var = mdl.get_var_by_name('x_{}'.format(i))
        bin_vars_cut_vals.append(mdl.binary_var(name='bvcv_{}_{}'.format(
            n_iter, i)))
        mdl.add(mdl.if_then(x_var == cut_sol[i], 
            bin_vars_cut_vals[i] == 1))
    # remove given assignment from solution pool
    # assignent is true only if all binary vars are equal to 1
    mdl.add_constraint(sum(bin_vars_cut_vals) <= 2)

'''
Compute statistics
'''
def compute_sol_stats(opt_config, prev_config, prev_sol_stats, error, errPred,
        error_class, acc, mae, r2, ev, sol_time,  print_dbg, first_sol,
        large_err_thresh):
    sol_stats = {}
    sol_stats['config'] = opt_config
    sol_stats['delta_config'] = [i - j for i, j in zip(
        prev_sol_stats['config'], sol_stats['config'])]
    sol_stats['error'] = error
    sol_stats['error_class'] = error_class
    sol_stats['error_log'] = -np.log(error)
    sol_stats['error_pred'] = np.exp(-errPred)
    sol_stats['error_pred_log'] = errPred
    sol_stats['C_Acc'] = acc
    sol_stats['R_MAE'] = mae 
    sol_stats['R_R2'] = r2 
    sol_stats['R_EV'] = ev
    if not first_sol:
        sol_stats['delta_error'] = prev_sol_stats['error'] - sol_stats['error']
        sol_stats['delta_error_log'] = prev_sol_stats[
                'error_log'] - sol_stats['error_log']
        sol_stats['delta_error_pred'] = prev_sol_stats['error_pred'
                ] - sol_stats['error_pred']
        sol_stats['delta_error_pred_log'] = prev_sol_stats['error_pred_log'
                ] - sol_stats['error_pred_log']
    else:
        sol_stats['delta_error'] = 0
        sol_stats['delta_error_log'] = 0
        sol_stats['delta_error_pred'] = 0
        sol_stats['delta_error_pred_log'] = 0
        sol_stats['delta_error'] = 0
    if error_class == 1:
        sol_stats['error_capped'] = large_err_thresh
    else:
        sol_stats['error_capped'] = sol_stats['error']
    sol_stats['cost'] = sum(opt_config)
    for k, v in sol_stats.items():
        prev_sol_stats[k] = v
    sol_stats['iteration_time'] = sol_time
    return sol_stats

'''
Infer new examples to be added to the ML training sets
'''
def infer_new_examples(benchmark, sol_stats):
    new_examples = {}
    nn = 0
    conf_dict = {}
    for i in range(len(sol_stats['config'])):
        conf_dict['var_{}'.format(i)] = [sol_stats['config'][i]]
    new_examples['df'] = pd.DataFrame.from_dict(conf_dict)
    new_examples['error'] = [sol_stats['error']] * (nn+1)
    new_examples['error_log'] = [sol_stats['error_log']] * (nn+1)
    new_examples['error_class'] = [sol_stats['error_class']] * (nn+1)
    return new_examples

'''
Retrain both ML models (classifier and regressor) using the new data provided
- the models are _partially_ retrained: we start from the model previously
  trained (and its original weights) and we apply a new round of training using
  only the new training examples 
'''
def refine_ML(benchmark, regr, classr, initial_df, new_examples,
        classr_needed):
    train_sets_R = [initial_df['train_data_R'], new_examples['df']]
    train_sets_C = [initial_df['train_data_C'], new_examples['df']]
    new_train_data_regr = pd.concat(train_sets_R)
    new_train_data_classr = pd.concat(train_sets_C)
    new_examples_as_pd_series_err_log = pd.Series(v for v in new_examples[
            'error_log'])
    new_train_target_regr = pd.concat(
            [initial_df['train_trgt_R'], new_examples_as_pd_series_err_log], 
            ignore_index=True)
    new_examples_as_pd_series_err_class = pd.Series(v for v in new_examples[
            'error_class'])
    new_train_target_classr = pd.concat(
            [initial_df['train_trgt_C'], new_examples_as_pd_series_err_class], 
            ignore_index=True)

    if scaling:
        scaler_train_data_R = MinMaxScaler()
        scaler_train_data_R.fit(new_train_data_regr)

        scaler_train_data_C = MinMaxScaler()
        scaler_train_data_C.fit(new_train_data_regr)

        scaler_train_trgt_R = MinMaxScaler()
        scaler_train_trgt_R.fit(new_train_target_regr.values.reshape(-1, 1))

        scaler_train_trgt_C = MinMaxScaler()
        scaler_train_trgt_C.fit(new_train_target_classr.values.reshape(-1, 1))

        # scale new examples + tensors
        train_data_tensor_R = scaler_train_data_R.transform(
            new_examples['df'])
        train_target_tensor_R = scaler_train_trgt_R.transform(
            np.asarray(new_examples['error_log']).reshape(-1, 1))

        train_data_tensor_C = scaler_train_trgt_C.transform(
            new_examples['df'])
        train_target_tensor_C = scaler_train_trgt_C.transform(
            np.asarray(new_examples['error_class']).reshape(-1, 1))

        # scale test data + tensors
        scaler = MinMaxScaler()
        test_data_tensor_R = scaler.fit_transform(
            initial_df['test_data_R'])
        test_data_tensor_C = scaler.fit_transform(
            initial_df['test_data_C'])
        test_target_tensor_R = scaler.fit_transform(
            initial_df['test_trgt_R'].values.reshape(-1, 1))
        test_target_tensor_C = initial_df['test_trgt_C']
    else:
        # train data tensors (new examples only)
        train_data_tensor_R = new_examples['df'].values
        train_data_tensor_C = new_examples['df'].values

        # test data tensors
        test_data_tensor_R = initial_df['test_data_R'].values
        test_data_tensor_C = initial_df['test_data_C'].values

        train_target_tensor_R = [new_examples['error_log']]
        train_target_tensor_C = [new_examples['error_class']]
        test_target_tensor_R = initial_df['test_trgt_R']
        test_target_tensor_C = initial_df['test_trgt_C']

    if classr_needed:
        train_data_C = new_train_data_classr
        train_target_C = new_train_target_classr
        test_data_C = initial_df['test_data_C']
        test_target_C = initial_df['test_trgt_C']

    # retrain classifier and compute stats
    if batch_size > len(train_data_tensor_C):
        bsize = len(train_data_tensor_C)
    else:
        bsize = batch_size

    ex_weigths_list = [new_config_weight] * len(train_data_tensor_R)
    ex_weigths = np.asarray(ex_weigths_list)
    if classr_needed:
        classr.fit(train_data_C, train_target_C)
        predicted_C = classr.predict(test_data_C)
        actual_C = test_target_C
        stats_res_C = utils.evaluate_binary_classification(predicted_C,
                actual_C)
        stats_res_C['loss'] = 0.0
        stats_res_C['accuracy'] = accuracy_score(test_target_C, predicted_C)
        stats_res_C['binary_crossentropy'] = log_loss(test_target_C,
                predicted_C)
    else:
        stats_res_C = {}
        stats_res_C['loss'] = 1.0
        stats_res_C['accuracy'] = 1.0
        stats_res_C['binary_crossentropy'] = 1.0
 
    # retrain regressor and compute stats
    if batch_size > len(train_data_tensor_R):
        bsize = len(train_data_tensor_R)
    else:
        bsize = batch_size

    print("\t - Regressor: new train set size {} (test size {})".format(
        len(train_data_tensor_R), len(test_data_tensor_R)))
    history = regr.fit(train_data_tensor_R, train_target_tensor_R, 
            epochs=epochs, batch_size=bsize, shuffle=True, verbose=0,
            sample_weight=ex_weigths)
    predicted_R = regr.predict(test_data_tensor_R)
    if scaling:
        actual_R = test_target_tensor_R
    else:
        actual_R = test_target_tensor_R.values
    stats_res_R = utils.evaluate_predictions(predicted_R, actual_R)
    test_loss_R = regr.evaluate(test_data_tensor_R, test_target_tensor_R)
    stats_res_R["test_loss"] = test_loss_R

    new_df = {}
    new_df = {'train_data_R': new_train_data_regr, 
            'test_data_R': initial_df['test_data_R'], 
            'train_trgt_R': new_train_target_regr,
            'test_trgt_R': initial_df['test_trgt_R'], 
            'train_data_C': new_train_data_classr, 
            'test_data_C': initial_df['test_data_C'],
            'train_trgt_C': new_train_target_classr, 
            'test_trgt_C': initial_df['test_trgt_C']}

    max_conf = [max_nbit for i in range(benchmark_nVar[benchmark])]
    max_conf_dict = {}
    for i in range(len(max_conf)):
        max_conf_dict['var_{}'.format(i)] = [max_conf[i]]
    max_conf_df = pd.DataFrame.from_dict(max_conf_dict)
    max_pred_err = regr.predict(max_conf_df)[0][0]

    return (regr, stats_res_R['MAE'], stats_res_R['RMSE'], stats_res_R['R2'],
            stats_res_R['EV'], stats_res_R['underest_ratio'], classr, 
            stats_res_C['accuracy'], max_pred_err, new_df)

'''
Solve the optimization model and refine it 
- active learning scheme
- the classifier and the regressor are updated as well with new examples
'''
def opt_model_AL(benchmark, initial_df, trgt_error_ratio,
        trgt_error_ratio_log_exp, regr, mae, rmse, r2, ev, underest_ratio,
        classr, acc, train_set_size, large_err_thresh, input_set_idx=-1, 
        debug=True, initial_train_time=0):
    if classr == None:
        classr_needed = False
    else:
        classr_needed = True
    if debug:
        print("----- Search solution with opt model -----")
    before_firstSol_time = time.time()
    before_firstSol_solve_time = before_firstSol_time
    opt_config, mdl, bit_sum = solve_opt_model(benchmark, None, 
            trgt_error_ratio_log_exp, regr, mae, rmse, underest_ratio, classr, 
            acc, 0, False, train_set_size, large_err_thresh, 0, [], True)
    if mdl == 1:
        return opt_config

    after_firstSol_solve_time = time.time()

    if opt_config == None:
        return opt_config

    before_firstSol_check_time = time.time()
    error, is_error_se_trgt, error_class = utils.check_solution(benchmark, 
            opt_config, trgt_error_ratio, benchmarks_home, binary_map,
            large_err_thresh, benchmark_nVar, max_nbit, min_nbit, input_set_idx)

    after_firstSol_time = time.time()
    after_firstSol_check_time = after_firstSol_time

    errPred, classPred = get_pred_class(opt_config, regr, classr)

    if debug:
        print(" First Solution (found in {0:.3f}s): {1}; error {2} < "
                "error_target {3}? {4}".format(
                    after_firstSol_time - before_firstSol_time, 
            opt_config, error, trgt_error_ratio, is_error_se_trgt))
        print("\tTime to find sol {0:.3f}, time to check sol {1:.3f}".format(
            after_firstSol_solve_time - before_firstSol_solve_time,
            after_firstSol_check_time - before_firstSol_check_time))

    if is_error_se_trgt:
        if debug:
            print(" Solution found satisfies actual program run")
        return opt_config, 1

    if debug:
        print(" Actual error (log) {0:.5f}, predicted error (log) {1:.3f}, "
                "desired error (log) {2:.3f}; actual class {3}, predicted "
                "class {4}".format(-np.log(error), errPred, 
                    -np.log(trgt_error_ratio), error_class, classPred))
        print(" Solution found _does not_ satisfy actual program run")
        print(" Refine model and search for new solution")

    prev_sol_stats = {}
    prev_sol_stats['config'] = opt_config
    prev_sol_stats['delta_config'] = 0
    prev_sol_stats['error'] = 0
    prev_sol_stats['error_class'] = 0
    prev_sol_stats['delta_error'] = 0
    prev_sol_stats['error_log'] = 0
    prev_sol_stats['delta_error_log'] = 0
    prev_sol_stats['error_pred'] = 0
    prev_sol_stats['error_pred_log'] = 0
    prev_sol_stats['delta_error_pred'] = 0
    prev_sol_stats['delta_error_pred_log'] = 0
    prev_sol_stats['error_capped'] = 0
    prev_sol_stats['cost'] = sum(opt_config)
    sol_stats = compute_sol_stats(opt_config, opt_config, prev_sol_stats, error,
            errPred, error_class, acc, mae, r2, ev, 
            after_firstSol_time - before_firstSol_time + initial_train_time, 
            True, True, large_err_thresh)
    prev_sol_stats = sol_stats
    prev_config = opt_config
    prev_bit_sum = sol_stats['cost']

    before_refineSol_time = time.time()
    n_iter = 1
    n_small_error = 0
    n_large_error = 0
    # we retrain the model until a certain number of iterations is performed and
    # until the error is smaller than the desider one
    while not is_error_se_trgt and n_iter < max_refine_iterations:
        print(">>>>>>> Iteration {} <<<<<<<<".format(n_iter))
        before_iterRefSol_time = time.time()
        before_refineSol_solve_time = time.time()

        if debug:
            print("\t Infer new examples")
        new_examples = infer_new_examples(benchmark, sol_stats)
        if debug:
            print("\t Refine ML model")

        # again, we need to assure that the regr NN can predict the target
        # error (err < target --> out NN > log(target))
        iter_refine_net = 0
        max_pred_err = -1
        (regr_ref, mae, rmse, r2, ev, underest_ratio, classr_ref, acc,
                max_pred_err, new_df) = refine_ML(benchmark, regr,
                        classr, initial_df, new_examples, classr_needed)
        if debug:
            print("\tmax_pred_err {0:.3f} (trgt err ratio log exp: {1:.3f}) - "
                    ' #iter needed: {2}) '.format(max_pred_err, 
                        trgt_error_ratio_log_exp, iter_refine_net))

            print("\t - # Examples used for training regr so far: {}".format(
                len(new_df['train_data_R'])))
            print("\t - # Examples used for training clf so far: {}".format(
                len(new_df['train_data_C'])))
            print("\t - Refined stats: regr MAE {0:.3f}, R2 {1:.3f}, EV "
                    "{2:.3f} -- Classr Acc {3:.3f}".format(mae, r2, ev, acc))

        # the refinement can happen even if the previous solution is already
        # fine (we may want to improve it). If the previous solution was
        # infeasible we also want it to be deleted from the solution pool
        if not is_error_se_trgt:
            # if the error predictor is not very robust, we may end up with many 
            # 'small' error; after a while we increase the target
            if n_small_error != 0 and ((n_small_error % 10) == 0):
                error_increase = 1
                trgt_error_ratio_log_exp += error_increase
                if debug:
                    print("\t\t Increase error ({})".format(error_increase))

        # after every N iterations, we increase the minimum number of bits of a
        # solution
        if n_iter % increase_freq == 0 and n_iter > increase_freq_begin:
            increase_tot_nbits = True
        else:
            increase_tot_nbits = False

        wrong_config = opt_config
        if debug:
            print("\t----- Search solution with opt model -----")
        opt_config, mdl_ref, bit_sum = solve_opt_model(benchmark, None, 
                trgt_error_ratio_log_exp, regr, mae, rmse, underest_ratio, 
                classr, acc, prev_bit_sum, increase_tot_nbits, 
                train_set_size, large_err_thresh, n_iter, wrong_config, 
                True)
        prev_bit_sum = bit_sum

        after_refineSol_solve_time = time.time()
        before_refineSol_check_time = time.time()
        error, is_error_se_trgt, error_class = utils.check_solution(benchmark, 
            opt_config, trgt_error_ratio, benchmarks_home, binary_map,
            large_err_thresh, benchmark_nVar, max_nbit, min_nbit, input_set_idx)
        after_refineSol_check_time = time.time()

        errPred, classPred = get_pred_class(opt_config, regr, classr)
        if debug:
            print("\t Refined solution at iter {0}: {1}; error {2} < "
                    "error_target {3}? {4}".format(n_iter, opt_config, 
                        error, trgt_error_ratio, is_error_se_trgt))
            print("\t Actual error (log) {0:.3f}, predicted error (log) {1:.3f},"
                    " desired error (log) {2:.3f}; actual class {3}, predicted "
                    "class {4}".format(-np.log(error), errPred, 
                        -np.log(trgt_error_ratio), error_class, classPred))

            if not is_error_se_trgt:
                if error_class == 0:
                    error_delta = errPred - (-np.log(error))
                    print("\t--> Small Error - error delta (log) {0:.3f}".format(
                        error_delta))
                    n_small_error += 1
                else:
                    print("\t--> Large Error")
                    n_large_error += 1
            print("Time to find refined sol {0:.3f}, time to check sol "
                "{1:.3f}".format(
                    after_refineSol_solve_time - before_refineSol_solve_time,
                    after_refineSol_check_time - before_refineSol_check_time))

        after_iterRefSol_time = time.time()
        sol_stats = compute_sol_stats(opt_config, prev_config, prev_sol_stats,
                error, errPred, error_class, acc, mae, r2, ev,
                after_iterRefSol_time - before_iterRefSol_time, True, False,
                large_err_thresh)
        prev_sol_stats = sol_stats
        prev_config = opt_config

        mdl = mdl_ref

        # update DF with new examples (for next iteration)
        initial_df = new_df

        n_iter += 1

    after_refineSol_time = time.time()
    if debug:
        print(" Refined Sol found in {0:.3f}s and after {1} iterations".format(
            after_refineSol_time - before_refineSol_time, n_iter))
        print("----- Found solution {} -----".format(opt_config))

    return opt_config, n_iter

'''
Create the regressor
'''
def regressor_creation(benchmark, benchmarks_home, data_set_dir, benchmark_nVar,
        min_nbit, max_nbit, value_inPlace_of_inf, errors_close_to_0_threshold,
        large_err_thresh, initial_dataset_size, initial_train_set_size,
        trgt_error_ratio_log_exp, alpha_asymm_loss, input_set_idx=0, 
        verbose=True):
    '''
    Prepare initial data frame
    '''
    print("Prepare initial data frame")
    (train_data_regr, test_data_regr, train_target_regr, test_target_regr,
            train_data_classr, test_data_classr, train_target_classr,
            test_target_classr, classr_needed) = utils.create_train_test_sets(
                    benchmark, benchmarks_home, data_set_dir,
                    benchmark_nVar, min_nbit, max_nbit,
                    value_inPlace_of_inf, errors_close_to_0_threshold,
                    large_err_thresh, initial_dataset_size,
                    initial_train_set_size, input_set_idx)

    initial_df = {'train_data_R': train_data_regr, 'test_data_R':
            test_data_regr, 'train_trgt_R': train_target_regr, 'test_trgt_R':
            test_target_regr, 'train_data_C': train_data_classr, 'test_data_C':
            test_data_classr, 'train_trgt_C': train_target_classr, 'test_trgt_C':
            test_target_classr, 'classr_needed': classr_needed}

    before_nn_time = time.time()
    '''
    Create model to predict error (regression)
    '''
    print("Create Regressor")
    (regr, mae, rmse, r2, ev, acc, underest_ratio, loss, max_pred_err
            ) = prediction_model_NN(benchmark, alpha_asymm_loss,
                    initial_train_set_size, train_data_regr, test_data_regr,
                    train_target_regr, test_target_regr, True, False)
    after_nn_time = time.time()

    return (initial_df, before_nn_time, after_nn_time, regr, mae, rmse, r2, ev,
            acc, underest_ratio, loss, max_pred_err, train_data_regr,
            test_data_regr, train_target_regr, test_target_regr,
            train_data_classr, test_data_classr, train_target_classr,
            test_target_classr, classr_needed)

'''
Create the classifier
'''
def classifier_creation(classr_needed, benchmark, initial_train_set_size,
        train_data_classr, test_data_classr, train_target_classr,
        test_target_classr, large_err_thresh, verbose=True):
    if classr_needed:
        print("Create Classifier")
        classr, acc, fscore = classifier_DT( benchmark,
                initial_train_set_size, train_data_classr, test_data_classr,
                train_target_classr, test_target_classr, large_err_thresh,
                verbose)
    else:
        classr = None
        acc = 1.0
        fscore = 1.0
    return classr, acc, fscore

def main(argv):
    opt_config = -1
    benchmark = argv[0]

    # the desired error ratio is expressed as 1e-exp
    # the input taken from the user is the exp
    trgt_error_ratio_exp = int(argv[1])

    input_set_idx = int(argv[2])

    if trgt_error_ratio_exp > max_target_error_exp:
        print("Desired error ({}) larger than the max allowed ({})".format(
            trgt_error_ratio_exp, max_target_error_exp))
        sys.exit()

    trgt_error_ratio = float('1e-' + str(trgt_error_ratio_exp))
    trgt_error_ratio_log_exp = -np.log(trgt_error_ratio)
    largeET = large_error_threshold 

    before_initial_time = time.time()
    (initial_df, before_nn_time, after_nn_time, regr, mae, rmse, r2, ev, acc,
            underest_ratio, loss, max_pred_err, train_data_regr, test_data_regr,
            train_target_regr, test_target_regr, train_data_classr,
            test_data_classr, train_target_classr, test_target_classr,
            classr_needed) = regressor_creation(benchmark, benchmarks_home,
                    data_set_dir, benchmark_nVar, min_nbit, max_nbit,
                    value_inPlace_of_inf, errors_close_to_0_threshold, largeET,
                    initial_dataset_size, initial_train_set_size,
                    trgt_error_ratio_log_exp, alpha_asymm_loss, input_set_idx, 
                    True)
    print("Initial regressor with MAE {0:.3f} ({1:.3f} sec)".format(
        mae, after_nn_time - before_nn_time))

    before_cl_time = time.time()
    (classr, acc, fscore) = classifier_creation(classr_needed, benchmark,
            initial_train_set_size, train_data_classr, test_data_classr,
            train_target_classr, test_target_classr, largeET, True)
    after_cl_time = time.time()
    print("Initial classifier with accuracy {0:.3f} ({1:.3f} sec)".format(
        acc, after_cl_time - before_cl_time))
    after_initial_time = time.time()
    print("Initial phase (data retrieval, regr & classr train) took {0:.3f} "
            "sec)".format(after_initial_time - before_initial_time))

    after_initial_time = time.time()
    before_opt_time = time.time()

    '''
    Create MP model and solve optimization problem
    - active learning approach
    '''
    opt_config, n_iter = opt_model_AL(benchmark, initial_df, trgt_error_ratio,
            trgt_error_ratio_log_exp, regr, mae, rmse, r2, ev, underest_ratio,
            classr, acc, initial_train_set_size, largeET, input_set_idx, True,
            after_initial_time - before_initial_time)
    after_opt_time = time.time()

    # post process and evaluation
    if opt_config == -1:
        print("Some problem happened with the optimizer")
        sys.exit()
    else:
        print("Solution for {0} with desired max error ratio 1e-{1} found in "
                "{2:.3f}s (ML) + {3:.3f}s (opt) and {4} iterations".format(
                    benchmark, trgt_error_ratio_exp, (after_opt_time - 
                        before_opt_time), (after_initial_time -
                            before_initial_time), n_iter))
    if opt_config == None:
        opt_config = [max_nbit for i in range(benchmark_nVar[benchmark])]
    print("Exp error {0}, error {1}, log exp error {2:.3f}".format(
        trgt_error_ratio_exp, trgt_error_ratio, trgt_error_ratio_log_exp))
    
    # final check: actually run the bechmark with the optimal config
    print("Run benchmark with opt config (%s)" % opt_config) 
    error, is_error_se_trgt, error_class = utils.check_solution(benchmark, 
            opt_config, trgt_error_ratio, benchmarks_home, binary_map, 
            largeET, benchmark_nVar, max_nbit, min_nbit, input_set_idx)

    print("Error {0} (log: {1:.3f}) <= target {2} (log: {3:.3f})? --> {4}".
            format(error, -np.log(error), trgt_error_ratio, 
                -np.log(trgt_error_ratio), is_error_se_trgt))
    if not is_error_se_trgt:
        print("\tDistance from trgt: {0:.3f}".format(1-trgt_error_ratio/error))

if __name__ == '__main__':
    main(sys.argv[1:])
