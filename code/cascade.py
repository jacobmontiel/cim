# -*- coding: utf-8 -*-
"""Cascade Imputation

This is the implementation of the Cascade Imputation algorithm described in:

Jacob Montie, Jesse Read, Albert Bifet and Talel Abdessalem.
“Scalable Model-based Cascaded Imputation of Missing Data”.
ECML PKDD conference. Submitted.
2017.

"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier,\
    ExtraTreesRegressor, ExtraTreesClassifier
from sklearn import dummy
import utilities as utl


def cascade_imputation(data, classifier, num_attr, cat_attr, verbose=False, fp_log=None, target_col=""):
    """ Cascade Imputation
    :param data: DataFrame to be processed
    :param classifier: Classifier/regressor to be used for imputation
    :param num_attr: List of numerical attributes
    :param cat_attr: List of categorical attributes
    :param verbose: Flag to enable debug messages
    :param fp_log: Pointer to log file
    :param target_col: Name of the target column, "target" by default
    :return: DataFrame with imputed values
    """

    vprint = print if verbose else lambda *a, **k: None

    # If the data set has no Index column, create it from the index
    idx_col = "ID"
    if target_col == "":
        target_col = "target"
    keep_ids = True    # Flag to keep ids if they are part of the original data

    if idx_col not in data.columns:
        print("Adding idx_col")
        data[idx_col] = data.index
        keep_ids = False

    if classifier is None:
        classifier = "RF"   # Default classifier/regressor
        vprint("Using RF as default classifier/regressor")

    N, D = data.shape

    # Save original order of indices
    original_order = data[idx_col]

    # Convert 'object' columns
    data = utl.convert_obj_to_num(data)

    # Mark nulls positions
    nulls = pd.isnull(data)
    # plt.spy(nulls, aspect='auto')

    # Weights [0,1]. 1: Instance has no missing values, 0: Instance has missing values for all attributes
    weights = (D - (nulls == 1).astype(int).sum(axis=1)) / D
    # Concatenate weights
    attr_names = list(data.columns.values) + list(["weights"])
    data = pd.concat([data, weights], axis=1)
    data.columns = attr_names
    D += 1

    # Update nulls positions
    nulls = pd.isnull(data)

    for i in range(2):
        # Run the cascade twice in case there are missing values in the first column
        count_nan = nulls.sum().sum()
        if count_nan > 0:
            print(file=fp_log)
            print("Cascade imputation run {}: {} missing values to fill".format(i+1, count_nan), file=fp_log)
            # Sort columns per missing values count. Ascending order from left to right.
            empty_cnt = []
            for col in nulls:
                count = np.count_nonzero(nulls[col])
                empty_cnt.append([col, count])

            sorted_attr = [row[0] for row in (sorted(empty_cnt, key=lambda tup: tup[1]))]
            print(sorted_attr)

            # Move ID, target and weights to the right end of the DataFrame
            sorted_attr.remove(idx_col)
            sorted_attr.remove(target_col)
            sorted_attr.remove('weights')
            sorted_attr.append(idx_col)
            sorted_attr.append(target_col)
            sorted_attr.append('weights')
            data = data.reindex(columns=sorted_attr, fill_value=0)

            # The main loop to traverse columns
            for k in range(1, D - 3):
                print("---------------------", file=fp_log)
                attr = data.columns[k]
                if pd.isnull(data[attr]).sum() > 0:
                    # Process this feature
                    print("Feature ({}): {}".format(k+1, attr), file=fp_log)
                    # Re-order instances, full instances in the top
                    row_idx = np.argsort(pd.isnull(data[attr]))
                    # Reset the index to facilitate tracking of train/test split, we will rely on the idx_col
                    # to reconstruct the original data sets
                    data = data.reindex(index=row_idx, fill_value=0).reset_index(drop=True)
                    n = N - np.count_nonzero(pd.isnull(data[attr]))
                    print("Instances to fill = " + str(N - n), file=fp_log)
                    # Impute missing data
                    ###################################
                    print("Make split at column {}/{} and row {}/{}".format(k+1, D, n+1, N), file=fp_log)
                    K = k + 1  # number of attributes in total (including target column/class)
                    Xy = data.ix[:, range(K)].as_matrix()  # make numpy matrix
                    X = Xy[0:n, 0:k]  # input for training
                    y = Xy[0:n, k]  # target for training
                    # Protect corner case where train set X has missing values
                    X_nulls = np.isnan(X)
                    X_nulls_cnt = X_nulls.sum().sum()
                    if X_nulls_cnt > 0:
                        print("WARNING: found ", X_nulls_cnt, " missing values in train set X, will drop instances")
                        print("WARNING: found ", X_nulls_cnt, " missing values in train set X, will drop instances",
                              file=fp_log)
                        nulls_row_idx = np.where(X_nulls.sum(axis=1) > 0)[0]
                        X = np.delete(X, nulls_row_idx, axis=0)
                        y = np.delete(y, nulls_row_idx, axis=0)
                    Xp = Xy[n:, 0:k]  # input for prediction
                    # Protect corner case where prediction set X has missing values
                    X_nulls = np.isnan(Xp)
                    X_nulls_cnt = X_nulls.sum().sum()
                    if X_nulls_cnt > 0:
                        print("WARNING: found ", X_nulls_cnt, " missing values in train set Xp, will fill value with 0")
                        print("WARNING: found ", X_nulls_cnt, " missing values in train set Xp, will fill value with 0",
                              file=fp_log)
                        Xp[X_nulls] = 0


                    if X.shape[0] == 0:
                        print("Not enough samples for training, skipping feature", file=fp_log)
                        continue
                    else:
                        vprint("We want to build a model on training set", X.shape, "and use it on test examples", Xp.shape)
                        h = None
                        if data.columns[k] in set(num_attr):
                            # REGRESSION
                            vprint("{} data type , using a Regressor".format(data.dtypes[k]), file=fp_log)
                            if classifier == 'LR':
                                h = LinearRegression(n_jobs=-1)
                            elif classifier == 'DT':
                                h = DecisionTreeRegressor(max_depth=5)
                            elif classifier == 'RF':
                                h = RandomForestRegressor(max_depth=4,
                                                          n_estimators=100,
                                                          random_state=1,
                                                          n_jobs=-1)
                            elif classifier == 'ET':
                                h = ExtraTreesRegressor(n_estimators=100,
                                                        max_features="auto",
                                                        criterion='mse',
                                                        min_samples_split=4,
                                                        max_depth=35,
                                                        min_samples_leaf=2,
                                                        n_jobs=-1)
                            else:
                                vprint("No such specification: ", classifier)
                                exit(1)
                        elif data.columns[k] in set(cat_attr):
                            # CLASSIFICATION
                            vprint("{} data type, using a Classifier".format(data.dtypes[k]), file=fp_log)
                            if classifier == 'LR':
                                if len(np.unique(y.astype("int64"))) == 1:
                                    vprint("Only 1 class in training set, will use majority class", file=fp_log)
                                    h = dummy.DummyClassifier(strategy="most_frequent")
                                else:
                                    h = LogisticRegression(n_jobs=-1)
                            elif classifier == 'DT':
                                h = DecisionTreeClassifier(max_depth=5)
                            elif classifier == 'RF':
                                h = RandomForestClassifier(max_depth=4,
                                                           n_estimators=100,
                                                           random_state=1,
                                                           n_jobs=-1)
                            elif classifier == 'ET':
                                h = ExtraTreesClassifier(n_estimators=100,
                                                         max_features="auto",
                                                         criterion='entropy',
                                                         min_samples_split=4,
                                                         max_depth=35,
                                                         min_samples_leaf=2,
                                                         n_jobs=-1)
                            else:
                                vprint("No such specification: ", classifier)
                                exit(1)
                        elif data.dtypes[k] == 'object':
                            vprint("not expecting this!", file=fp_log)
                            exit(1)
                        else:
                            vprint("Unexpected data type!", file=fp_log)
                            exit(1)

                        print(" Training...")
                        if data.columns[k] in set(num_attr):
                            h.fit(X, y)
                        elif data.columns[k] in set(cat_attr):
                            h.fit(X, y.astype("int64"))
                        print(" Predicting...")
                        yp = h.predict(Xp)
                        print(" Filling...")
                        # make ", yp.shape, "fit into", (N-n), "rows"
                        idx_range = range(n, N)
                        data.ix[idx_range, attr] = yp

                else:
                    print("Nothing to do for: " + attr, file=fp_log)

            # Update nulls positions
            nulls = pd.isnull(data)
        else:
            print("Cascade imputation run {}: {} missing values to fill".format(i + 1, count_nan), file=fp_log)

    # Reindexing (back to original order)
    data = data.set_index(idx_col, drop=False)
    data = data.reindex(original_order)

    # Move ID and target columns to the front
    mid = data[target_col]
    data.drop(labels=[target_col], axis=1, inplace=True)
    data.insert(0, target_col, mid)
    mid = data[idx_col]
    data.drop(labels=[idx_col], axis=1, inplace=True)
    if keep_ids:
        data.insert(0, idx_col, mid)

    vprint("Finished")
    return data
