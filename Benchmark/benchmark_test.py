# Portions of this file are adapted from EPF Toolbox:
# https://github.com/jeslago/epftoolbox
# Original license: AGPL-3.0
# Modified by Btissame El Mahtout, 2026.

# select country: "Germany" or "Spain"
country = "Germany"

# -------------------------------
#               Packages                          
#--------------------------------------
import numpy as np
import pandas as pd
from calendar import day_abbr
import locale
import os
import joblib


import time

# import argparse
from epftoolbox.evaluation import MAE, sMAPE
from epftoolbox.models import LEAR

import pickle

import epftoolbox.models._dnn as dnn_mod
import epftoolbox.models._dnn_hyperopt as hyper_mod


from epftoolbox.models import (
    evaluate_lear_in_test_dataset,
    hyperparameter_optimizer,
    evaluate_dnn_in_test_dataset,
)
from hyperopt import hp





#----------------------------------
#  modify original function to be compatible with kera 3
#----------------------------------------------
import tensorflow.keras as kr
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Dense,
    Input,
    Dropout,
    AlphaDropout,
    BatchNormalization,
)
from tensorflow.keras.regularizers import l2, l1
from tensorflow.keras.layers import LeakyReLU, PReLU
import tensorflow.keras.backend as K


def _reg_float32(self, lambda_reg):
    lambda_reg = float(np.float32(lambda_reg))

    if self.regularization == "l2":
        return l2(lambda_reg)
    if self.regularization == "l1":
        return l1(lambda_reg)
    return None


# IMPORTANT: assign, do not call
dnn_mod.DNNModel._reg = _reg_float32


def _init_keras3(
    self,
    neurons,
    n_features,
    outputShape=24,
    dropout=0,
    batch_normalization=False,
    lr=None,
    verbose=False,
    epochs_early_stopping=40,
    scaler=None,
    loss="mae",
    optimizer="adam",
    activation="relu",
    initializer="glorot_uniform",
    regularization=None,
    lambda_reg=0,
):

    self.neurons = neurons
    self.dropout = dropout

    if self.dropout > 1 or self.dropout < 0:
        raise ValueError("Dropout parameter must be between 0 and 1")

    self.batch_normalization = batch_normalization
    self.verbose = verbose
    self.epochs_early_stopping = epochs_early_stopping
    self.n_features = n_features
    self.scaler = scaler
    self.outputShape = outputShape
    self.activation = activation
    self.initializer = initializer
    self.regularization = regularization
    self.lambda_reg = lambda_reg

    self.model = self._build_model()

    if lr is None:
        opt = kr.optimizers.Adam()  # default
    else:
        if optimizer == "adam":
            opt = kr.optimizers.Adam(learning_rate=lr, clipvalue=10000)
        elif optimizer == "RMSprop":
            opt = kr.optimizers.RMSprop(learning_rate=lr, clipvalue=10000)
        elif optimizer == "adagrad":
            opt = kr.optimizers.Adagrad(learning_rate=lr, clipvalue=10000)
        elif optimizer == "adadelta":
            opt = kr.optimizers.Adadelta(learning_rate=lr, clipvalue=10000)
        else:
            raise ValueError(f"Unknown optimizer: {optimizer}")

    self.model.compile(loss=loss, optimizer=opt)


dnn_mod.DNNModel.__init__ = _init_keras3


def _build_model_keras3(self):
    """Internal method that defines the structure of the DNN

    Returns
    -------
    tensorflow.keras.models.Model
        A neural network model using keras and tensorflow
    """
    # inputShape = (None, self.n_features)

    # past_data = Input(batch_shape=inputShape)
    past_data = Input(shape=(self.n_features,))

    past_Dense = past_data
    if self.activation == "selu":
        self.initializer = "lecun_normal"

    for k, neurons in enumerate(self.neurons):

        if self.activation == "LeakyReLU":
            past_Dense = Dense(
                neurons,
                activation="linear",
                # batch_input_shape=inputShape,
                kernel_initializer=self.initializer,
                kernel_regularizer=self._reg(self.lambda_reg),
            )(past_Dense)
            past_Dense = LeakyReLU(alpha=0.001)(past_Dense)

        elif self.activation == "PReLU":
            past_Dense = Dense(
                neurons,
                activation="linear",
                # batch_input_shape=inputShape,
                kernel_initializer=self.initializer,
                kernel_regularizer=self._reg(self.lambda_reg),
            )(past_Dense)
            past_Dense = PReLU()(past_Dense)

        else:
            past_Dense = Dense(
                neurons,
                activation=self.activation,
                # batch_input_shape=inputShape,
                kernel_initializer=self.initializer,
                kernel_regularizer=self._reg(self.lambda_reg),
            )(past_Dense)

        if self.batch_normalization:
            past_Dense = BatchNormalization()(past_Dense)

        if self.dropout > 0:
            if self.activation == "selu":
                past_Dense = AlphaDropout(self.dropout)(past_Dense)
            else:
                past_Dense = Dropout(self.dropout)(past_Dense)

    output_layer = Dense(
        self.outputShape,
        kernel_initializer=self.initializer,
        kernel_regularizer=self._reg(self.lambda_reg),
    )(past_Dense)
    model = Model(inputs=[past_data], outputs=[output_layer])

    return model


dnn_mod.DNNModel._build_model = _build_model_keras3


#---------------------------------------
#                  Add commodity variables to LEAR
#----------------------------------------


def _build_and_split_XYs_fuel(self, df_train, df_test=None, date_test=None):
    """Internal function that generates the X,Y arrays for training and testing based on pandas dataframes

    Parameters
    ----------
    df_train : pandas.DataFrame
        Pandas dataframe containing the training data

    df_test : pandas.DataFrame
        Pandas dataframe containing the test data

    date_test : datetime, optional
        If given, then the test dataset is only built for that date

    Returns
    -------
    list
        [Xtrain, Ytrain, Xtest] as the list containing the (X,Y) input/output pairs for training,
        and the input for testing
    """

    # Checking that the first index in the dataframes corresponds with the hour 00:00
    if df_train.index[0].hour != 0 or df_test.index[0].hour != 0:
        print("Problem with the index")

    #
    # Defining the number of Exogenous inputs
    n_exogenous_inputs = len(df_train.columns) - 1

    # 96 prices + n_exogenous * (24 * 3 exogeneous) + 7 weekday dummies
    # Price lags: D-1, D-2, D-3, D-7
    # DA Exogeneous inputs lags: D, D-1, D-7
    # Fuel Exogeneous inputs lags: D-2
    n_features = 96 + 7 + 2 * 3 * 24 + 4

    # Extracting the predicted dates for testing and training. We leave the first week of data
    # out of the prediction as we the maximum lag can be one week

    # We define the potential time indexes that have to be forecasted in training
    # and testing
    indexTrain = df_train.loc[df_train.index[0] + pd.Timedelta(weeks=1) :].index

    # For testing, the test dataset is different whether depending on whether a specific test
    # dataset is provided
    if date_test is None:
        indexTest = df_test.loc[df_test.index[0] + pd.Timedelta(weeks=1) :].index
    else:
        indexTest = df_test.loc[date_test : date_test + pd.Timedelta(hours=23)].index

    # We extract the prediction dates/days.
    predDatesTrain = indexTrain.round("1h")[::24]
    predDatesTest = indexTest.round("1h")[::24]

    # We create two dataframe to build XY.
    # These dataframes have as indices the first hour of the day (00:00)
    # and the columns represent the 23 possible horizons/dates along a day
    indexTrain = pd.DataFrame(
        index=predDatesTrain, columns=["h" + str(hour) for hour in range(24)]
    )
    indexTest = pd.DataFrame(
        index=predDatesTest, columns=["h" + str(hour) for hour in range(24)]
    )
    for hour in range(24):
        indexTrain.loc[:, "h" + str(hour)] = indexTrain.index + pd.Timedelta(hours=hour)
        indexTest.loc[:, "h" + str(hour)] = indexTest.index + pd.Timedelta(hours=hour)

    # Preallocating in memory the X and Y arrays
    Xtrain = np.zeros([indexTrain.shape[0], n_features])
    Xtest = np.zeros([indexTest.shape[0], n_features])
    Ytrain = np.zeros([indexTrain.shape[0], 24])

    # Index that
    feature_index = 0

    #
    # Adding the historial prices during days D-1, D-2, D-3, and D-7
    #

    # For each hour of a day
    for hour in range(24):
        # For each possible past day where prices can be included
        for past_day in [1, 2, 3, 7]:

            # We define the corresponding past time indexs using the auxiliary dataframses
            pastIndexTrain = pd.to_datetime(
                indexTrain.loc[:, "h" + str(hour)].values
            ) - pd.Timedelta(hours=24 * past_day)
            pastIndexTest = pd.to_datetime(
                indexTest.loc[:, "h" + str(hour)].values
            ) - pd.Timedelta(hours=24 * past_day)

            # We include the historical prices at day D-past_day and hour "h"
            Xtrain[:, feature_index] = df_train.loc[pastIndexTrain, "Price"]
            Xtest[:, feature_index] = df_test.loc[pastIndexTest, "Price"]
            feature_index += 1

    #
    # Adding the exogenous inputs during days D, D-1,  D-7
    #
    # For each hour of a day
    for hour in range(24):
        # For each possible past day where exogenous inputs can be included
        for past_day in [1, 7]:
            # For each of the exogenous input
            # for exog in range(1, n_exogenous_inputs + 1):
            for exog in range(1, 3):  # take DA only

                # Definying the corresponding past time indexs using the auxiliary dataframses
                pastIndexTrain = pd.to_datetime(
                    indexTrain.loc[:, "h" + str(hour)].values
                ) - pd.Timedelta(hours=24 * past_day)
                pastIndexTest = pd.to_datetime(
                    indexTest.loc[:, "h" + str(hour)].values
                ) - pd.Timedelta(hours=24 * past_day)

                # Including the exogenous input at day D-past_day and hour "h"
                Xtrain[:, feature_index] = df_train.loc[
                    pastIndexTrain, "Exogenous " + str(exog)
                ]
                Xtest[:, feature_index] = df_test.loc[
                    pastIndexTest, "Exogenous " + str(exog)
                ]
                feature_index += 1

        # For each of the exogenous inputs we include feature if feature selection indicates it
        # for exog in range(1, n_exogenous_inputs + 1):
        for exog in range(1, 3):  # take DA only

            # Definying the corresponding future time indexs using the auxiliary dataframses
            futureIndexTrain = pd.to_datetime(indexTrain.loc[:, "h" + str(hour)].values)
            futureIndexTest = pd.to_datetime(indexTest.loc[:, "h" + str(hour)].values)

            # Including the exogenous input at day D and hour "h"
            Xtrain[:, feature_index] = df_train.loc[
                futureIndexTrain, "Exogenous " + str(exog)
            ]
            Xtest[:, feature_index] = df_test.loc[
                futureIndexTest, "Exogenous " + str(exog)
            ]
            feature_index += 1

    # ------------------------------------------------------
    # Fuel exogenous: ONLY lag 2, daily constant, no hours
    # ------------------------------------------------------

    fuel_exogs = range(3, n_exogenous_inputs + 1)

    dayIndexTrain = pd.to_datetime(indexTrain.index.values)  # daily 00:00
    dayIndexTest = pd.to_datetime(indexTest.index.values)

    pastDayTrain = dayIndexTrain - pd.Timedelta(days=2)
    pastDayTest = dayIndexTest - pd.Timedelta(days=2)

    for exog in fuel_exogs:
        Xtrain[:, feature_index] = df_train.loc[pastDayTrain, f"Exogenous {exog}"]
        Xtest[:, feature_index] = df_test.loc[pastDayTest, f"Exogenous {exog}"]
        feature_index += 1

    # Adding the dummy variables that depend on the day of the week. Monday is 0 and Sunday is 6
    #
    # For each day of the week
    for dayofweek in range(7):
        Xtrain[indexTrain.index.dayofweek == dayofweek, feature_index] = 1
        Xtest[indexTest.index.dayofweek == dayofweek, feature_index] = 1
        feature_index += 1

    # Extracting the predicted values Y
    for hour in range(24):
        # Defining time index at hour h
        futureIndexTrain = pd.to_datetime(indexTrain.loc[:, "h" + str(hour)].values)
        futureIndexTest = pd.to_datetime(indexTest.loc[:, "h" + str(hour)].values)

        # Extracting Y value based on time indexs
        Ytrain[:, hour] = df_train.loc[futureIndexTrain, "Price"]

    return Xtrain, Ytrain, Xtest


LEAR._build_and_split_XYs = _build_and_split_XYs_fuel

#--------------------------------------------------------------
#               Add commodity variables to dnn
#---------------------------------------------------------------


def _build_space_fuel_dnn(nlayer, data_augmentation, n_exogenous_inputs):
    """Function that generates the hyperparameter/feature search space

    Parameters
    ----------
    nlayer : int
        Number of layers of the DNN model
    data_augmentation : bool
        Boolean that selects whether augmenting data is considered
    n_exogenous_inputs : int
        Number of exogenous inputs in the market under study

    Returns
    -------
    dict
        Dictionary defining the search space
    """

    # Defining the hyperparameter space. First the neural net hyperparameters,
    # later the input features
    space = {
        "batch_normalization": hp.choice("batch_normalization", [False, True]),
        "dropout": hp.uniform("dropout", 0, 1),
        "lr": hp.loguniform("lr", np.log(5e-4), np.log(0.1)),
        "seed": hp.quniform("seed", 1, 1000, 1),
        "neurons1": hp.quniform("neurons1", 50, 500, 1),
        "activation": hp.choice(
            "activation",
            ["relu", "softplus", "tanh", "selu", "LeakyReLU", "PReLU", "sigmoid"],
        ),
        "init": hp.choice(
            "init",
            [
                "Orthogonal",
                "lecun_uniform",
                "glorot_uniform",
                "glorot_normal",
                "he_uniform",
                "he_normal",
            ],
        ),
        "reg": hp.choice(
            "reg",
            [
                {"val": None, "lambda": 0},
                {
                    "val": "l1",
                    "lambda": hp.loguniform("lambdal1", np.log(1e-5), np.log(1)),
                },
            ],
        ),
        "scaleX": hp.choice(
            "scaleX", ["No", "Norm", "Norm1", "Std", "Median", "Invariant"]
        ),
        "scaleY": hp.choice(
            "scaleY", ["No", "Norm", "Norm1", "Std", "Median", "Invariant"]
        ),
    }

    if nlayer >= 2:
        space["neurons2"] = hp.quniform("neurons2", 25, 400, 1)
    if nlayer >= 3:
        space["neurons3"] = hp.quniform("neurons3", 25, 300, 1)
    if nlayer >= 4:
        space["neurons4"] = hp.quniform("neurons4", 25, 200, 1)
    if nlayer >= 5:
        space["neurons5"] = hp.quniform("neurons5", 25, 200, 1)

    # Defining the possible input features as hyperparameters
    space["In: Day"] = hp.choice("In: Day", [False, True])
    space["In: Price D-1"] = hp.choice("In: Price D-1", [False, True])
    space["In: Price D-2"] = hp.choice("In: Price D-2", [False, True])
    space["In: Price D-3"] = hp.choice("In: Price D-3", [False, True])
    space["In: Price D-7"] = hp.choice("In: Price D-7", [False, True])

    for n_ex in range(1, 3):
        space["In: Exog-" + str(n_ex) + " D"] = hp.choice(
            "In: Exog-" + str(n_ex) + " D", [False, True]
        )
        space["In: Exog-" + str(n_ex) + " D-1"] = hp.choice(
            "In: Exog-" + str(n_ex) + " D-1", [False, True]
        )
        space["In: Exog-" + str(n_ex) + " D-7"] = hp.choice(
            "In: Exog-" + str(n_ex) + " D-7", [False, True]
        )

    for n_ex in range(3, n_exogenous_inputs + 1):
        space["In: Exog-" + str(n_ex) + " D-2"] = hp.choice(
            "In: Exog-" + str(n_ex) + " D-2", [False, True]
        )

    return space


hyper_mod._build_space = _build_space_fuel_dnn


def _build_and_split_XYs_fuel_dnn(
    dfTrain,
    features,
    shuffle_train,
    n_exogenous_inputs,
    dfTest=None,
    percentage_val=0.25,
    date_test=None,
    hyperoptimization=False,
    data_augmentation=False,
):
    """Method to buil the X,Y pairs for training/test DNN models using dataframes and a list of
    the selected inputs

    Parameters
    ----------
    dfTrain : pandas.DataFrame
        Pandas dataframe containing the training data
    features : dict
        Dictionary that define the selected input features. The dictionary is based on the results
        of a hyperparameter/feature optimization run using the :class:`hyperparameter_optimizer`function
    shuffle_train : bool
        If true, the validation and training datasets are shuffled
    n_exogenous_inputs : int
        Number of exogenous inputs, i.e. inputs besides historical prices
    dfTest : pandas.DataFrame
        Pandas dataframe containing the test data
    percentage_val : TYPE, optional
        Percentage of data to be used for validation
    date_test : None, optional
        If given, then the test dataset is only built for that date
    hyperoptimization : bool, optional
        Description
    data_augmentation : bool, optional
        Description

    Returns
    -------
    list
        A list ``[Xtrain, Ytrain, Xval, Yval, Xtest, Ytest, indexTest]`` that contains the X, Y pairs
        for training, validation, and testing, as well as the date index of the test dataset
    """

    # Checking that the first index in the dataframes corresponds with the hour 00:00
    if dfTrain.index[0].hour != 0 or dfTest.index[0].hour != 0:
        print("Problem with the index")

    # Calculating the number of input features
    n_features = (
        features["In: Day"]
        + 24 * features["In: Price D-1"]
        + 24 * features["In: Price D-2"]
        + 24 * features["In: Price D-3"]
        + 24 * features["In: Price D-7"]
    )

    # Day ahead features for exogenous 1 and 2
    for n_ex in range(1, 3):

        n_features += (
            24 * features["In: Exog-" + str(n_ex) + " D"]
            + 24 * features["In: Exog-" + str(n_ex) + " D-1"]
            + 24 * features["In: Exog-" + str(n_ex) + " D-7"]
        )

    # -------------------------------------------------
    # Exogenous 3..n (fuel, daily constant, ONLY D-2)
    # -------------------------------------------------
    for n_ex in range(3, n_exogenous_inputs + 1):
        n_features += features["In: Exog-" + str(n_ex) + " D-2"]

    # Extracting the predicted dates for testing and training. We leave the first week of data
    # out of the prediction as we the maximum lag can be one week
    # In addition, if we allow training using all possible predictions within a day, we consider
    # a indexTrain per starting hour of prediction

    # We define the potential time indexes that have to be forecasted in training
    # and testing
    indexTrain = dfTrain.loc[dfTrain.index[0] + pd.Timedelta(weeks=1) :].index

    if date_test is None:
        indexTest = dfTest.loc[dfTest.index[0] + pd.Timedelta(weeks=1) :].index
    else:
        indexTest = dfTest.loc[date_test : date_test + pd.Timedelta(hours=23)].index

    # We extract the prediction dates/days. For the regular case,
    # it is just the index resample to 24 so we have a date per day.
    # For the multiple datapoints per day, we have as many dates as indexs
    if data_augmentation:
        predDatesTrain = indexTrain.round("1h")
    else:
        predDatesTrain = indexTrain.round("1h")[::24]

    predDatesTest = indexTest.round("1h")[::24]

    # We create dataframe where the index is the time where a prediction is made
    # and the columns is the horizons of the prediction
    indexTrain = pd.DataFrame(
        index=predDatesTrain, columns=["h" + str(hour) for hour in range(24)]
    )
    indexTest = pd.DataFrame(
        index=predDatesTest, columns=["h" + str(hour) for hour in range(24)]
    )
    for hour in range(24):
        indexTrain.loc[:, "h" + str(hour)] = indexTrain.index + pd.Timedelta(hours=hour)
        indexTest.loc[:, "h" + str(hour)] = indexTest.index + pd.Timedelta(hours=hour)

    # If we consider 24 predictions per day, the last 23 indexs cannot be used as there is not data
    # for that horizon:
    if data_augmentation:
        indexTrain = indexTrain.iloc[:-23]

    # Preallocating in memory the X and Y arrays
    Xtrain = np.zeros([indexTrain.shape[0], n_features])
    Xtest = np.zeros([indexTest.shape[0], n_features])
    Ytrain = np.zeros([indexTrain.shape[0], 24])
    Ytest = np.zeros([indexTest.shape[0], 24])

    # Adding the day of the week as a feature if needed
    indexFeatures = 0
    if features["In: Day"]:
        # For training, we assume the day of the week is a continuous variable.
        # So monday at 00 is 1. Monday at 1h is 1.04, Tuesday at 2h is 2.08, etc.
        Xtrain[:, 0] = indexTrain.index.dayofweek + indexTrain.index.hour / 24
        Xtest[:, 0] = indexTest.index.dayofweek
        indexFeatures += 1

    # For each possible horizon
    for hour in range(24):
        # For each possible past day where prices can be included
        for past_day in [1, 2, 3, 7]:

            # We define the corresponding past time indexs
            pastIndexTrain = pd.to_datetime(
                indexTrain.loc[:, "h" + str(hour)].values
            ) - pd.Timedelta(hours=24 * past_day)
            pastIndexTest = pd.to_datetime(
                indexTest.loc[:, "h" + str(hour)].values
            ) - pd.Timedelta(hours=24 * past_day)

            # We include feature if feature selection indicates it
            if features["In: Price D-" + str(past_day)]:
                Xtrain[:, indexFeatures] = dfTrain.loc[pastIndexTrain, "Price"]
                Xtest[:, indexFeatures] = dfTest.loc[pastIndexTest, "Price"]
                indexFeatures += 1

    # For each possible horizon
    for hour in range(24):
        # For each possible past day where exogeneous can be included
        for past_day in [1, 7]:

            # We define the corresponding past time indexs
            pastIndexTrain = pd.to_datetime(
                indexTrain.loc[:, "h" + str(hour)].values
            ) - pd.Timedelta(hours=24 * past_day)
            pastIndexTest = pd.to_datetime(
                indexTest.loc[:, "h" + str(hour)].values
            ) - pd.Timedelta(hours=24 * past_day)

            # For each of the exogenous inputs we include feature if feature selection indicates it
            # for exog in range(1, n_exogenous_inputs + 1):
            for exog in range(1, 3):  # take DA only
                if features["In: Exog-" + str(exog) + " D-" + str(past_day)]:
                    Xtrain[:, indexFeatures] = dfTrain.loc[
                        pastIndexTrain, "Exogenous " + str(exog)
                    ]
                    Xtest[:, indexFeatures] = dfTest.loc[
                        pastIndexTest, "Exogenous " + str(exog)
                    ]
                    indexFeatures += 1

        # For each of the exogenous inputs we include feature if feature selection indicates it
        # for exog in range(1, n_exogenous_inputs + 1):
        for exog in range(1, 3):  # take DA only
            # Adding exogenous inputs at day D
            if features["In: Exog-" + str(exog) + " D"]:
                futureIndexTrain = pd.to_datetime(
                    indexTrain.loc[:, "h" + str(hour)].values
                )
                futureIndexTest = pd.to_datetime(
                    indexTest.loc[:, "h" + str(hour)].values
                )

                Xtrain[:, indexFeatures] = dfTrain.loc[
                    futureIndexTrain, "Exogenous " + str(exog)
                ]
                Xtest[:, indexFeatures] = dfTest.loc[
                    futureIndexTest, "Exogenous " + str(exog)
                ]
                indexFeatures += 1
    # ------------------------------------------------------
    # Fuel exogenous: ONLY lag 2, daily constant, no hours
    # (feature-selection aware)
    # ------------------------------------------------------

    fuel_exogs = range(3, n_exogenous_inputs + 1)

    dayIndexTrain = pd.to_datetime(indexTrain.index.values)  # daily 00:00
    dayIndexTest = pd.to_datetime(indexTest.index.values)

    pastDayTrain = dayIndexTrain - pd.Timedelta(days=2)
    pastDayTest = dayIndexTest - pd.Timedelta(days=2)

    for exog in fuel_exogs:
        if features["In: Exog-" + str(exog) + " D-2"]:
            Xtrain[:, indexFeatures] = dfTrain.loc[pastDayTrain, f"Exogenous {exog}"]
            Xtest[:, indexFeatures] = dfTest.loc[pastDayTest, f"Exogenous {exog}"]
            indexFeatures += 1

    # Extracting the predicted values Y
    for hour in range(24):
        futureIndexTrain = pd.to_datetime(indexTrain.loc[:, "h" + str(hour)].values)
        futureIndexTest = pd.to_datetime(indexTest.loc[:, "h" + str(hour)].values)

        Ytrain[:, hour] = dfTrain.loc[futureIndexTrain, "Price"]
        Ytest[:, hour] = dfTest.loc[futureIndexTest, "Price"]

    # Redefining indexTest to return only the dates at which a prediction is made
    indexTest = indexTest.index

    if shuffle_train:
        nVal = int(percentage_val * Xtrain.shape[0])

        if hyperoptimization:
            # We fixed the random shuffle index so that the validation dataset does not change during the
            # hyperparameter optimization process
            np.random.seed(7)

        # We shuffle the data per week to avoid data contamination
        index = np.arange(Xtrain.shape[0])
        index_week = index[::7]
        np.random.shuffle(index_week)
        index_shuffle = [
            ind + i for ind in index_week for i in range(7) if ind + i in index
        ]

        Xtrain = Xtrain[index_shuffle]
        Ytrain = Ytrain[index_shuffle]

    else:
        nVal = int(percentage_val * Xtrain.shape[0])
    nTrain = Xtrain.shape[0] - nVal  # complements nVal

    Xval = Xtrain[nTrain:]  # last nVal obs
    Xtrain = Xtrain[:nTrain]  # first nTrain obs
    Yval = Ytrain[nTrain:]
    Ytrain = Ytrain[:nTrain]

    return Xtrain, Ytrain, Xval, Yval, Xtest, Ytest, indexTest


dnn_mod._build_and_split_XYs = _build_and_split_XYs_fuel_dnn
hyper_mod._build_and_split_XYs = _build_and_split_XYs_fuel_dnn  # <<< ADD THIS


#-----------------------------------------
#                   data Preprocessing
#--------------------------------------------
#  set language setting
locale.getlocale()

#  check the working directory
os.getcwd()

# read the data
data = pd.read_csv(f"../Data/{country}.csv")

#%%
# change to local time
data["Date"] = (
    pd.to_datetime(data["time_utc"], utc=True)  # 1) parse as UTC tz‐aware
    .dt.tz_convert("Europe/Berlin")  # 2) convert to local time
    .dt.tz_localize(None)  # 3) drop tz info, get naive datetimes
)

# Drop the original UTC column
data = data.drop(columns=["time_utc"])

# make changes to accomodate the day saving time
data = (
    data.groupby("Date", as_index=False)
    .mean()
    .set_index("Date")
    .sort_index()
    .asfreq("h")  # Regularize hourly frequency
    .interpolate(method="time")
)  # Fill gaps using time-based linear interpolation)


# Create new column Wind_DA by summing WindOn_DA and WindOff_DA for Germany and WindOn_DA for Spain
if country == "Germany":
    data["Wind_DA"] = data["WindOn_DA"] + data["WindOff_DA"]
elif country == "Spain":
    data["Wind_DA"] = data["WindOn_DA"]


# Agregate Wind_DA and Solar_DA
data["Renewable_DA"] = data["Wind_DA"] + data["Solar_DA"]


# Create a path to store dataset and results of LEAR ad DNN
path_datasets_folder = f"./{country}/datasets"
path_recalibration_folder = f"./{country}/experimental_files"
os.makedirs(path_datasets_folder, exist_ok=True)
os.makedirs(path_recalibration_folder, exist_ok=True)

#
# set the Date index and keep only the columns of interest
data = data[["Price", "Load_DA", "Renewable_DA", "Coal", "NGas", "Oil", "EUA"]]

# Change the name of the columns to fit the expected format
data = data.rename(
    columns={
        "Load_DA": "Exogenous 1",  # keep original style (D, D-1, D-7)
        "Renewable_DA": "Exogenous 2",  # keep original style (D, D-1, D-7)
        "Coal": "Exogenous 3",  # ONLY D-2
        "NGas": "Exogenous 4",  # ONLY D-2
        "Oil": "Exogenous 5",  # ONLY D-2
        "EUA": "Exogenous 6",  # ONLY D-2
    }
)

# write it out—index=True (the default) so Date becomes the first column
data.to_csv(f"{country}/datasets/my_data.csv")


# specify the test period
begin_test = "2023-01-16 00:00"
end_test = "2025-01-14 23:00"


# name of the dataset without .csv
dataset = "my_data"


#-----------------------------------------------------------
#                      LEAR
#-----------------------------------------------------------
# Number of days used in the training dataset for recalibration
calibration_window_LEAR = 2 * 364


# Folders for data and output
path_datasets_folder = os.path.join(".", f"{country}/datasets")
path_recalibration_folder = os.path.join(".", f"{country}/experimental_files")
# run Lear

start_lear = time.time()
evaluate_lear_in_test_dataset(
    path_recalibration_folder=path_recalibration_folder,
    path_datasets_folder=path_datasets_folder,
    dataset=dataset,
    calibration_window=calibration_window_LEAR,
    begin_test_date=begin_test,
    end_test_date=end_test,
)

end_lear = time.time()

execution_time_lear = end_lear - start_lear

# Save execution time
file_execution_time_lear = (
    f"{country}/execution_time_lear.pkl"
)
joblib.dump(execution_time_lear, file_execution_time_lear)


forecast_lear = pd.read_csv(
    f"{country}/experimental_files/LEAR_forecast_datmy_data_YT2_CW728.csv",
    index_col=False,
)



Lear_forcast = forecast_lear.iloc[:, 1:]
forecast_lear_stack = Lear_forcast.stack()


# real prices
test_data = data.loc[begin_test:end_test]
price_test = test_data[["Price"]].reset_index(drop=True)
# Ensure price_test is a Series
price_test_series = price_test.squeeze()



forecast_errors_lear = forecast_lear_stack.reset_index(drop=True) - price_test_series


forcast_lear_num = forecast_errors_lear.to_numpy()


np.mean(forcast_lear_num**2) ** 0.5




#-----------------------------------------------------------
#                     DNN
#-------------------------------------------------------------

# Number of layers in DNN
nlayers = 2

# Boolean that selects whether the validation and training datasets are shuffled
shuffle_train = 0

# Boolean that selects whether a data augmentation technique for DNNs is used
data_augmentation = 0

# Boolean that selects whether we start a new hyperparameter optimization or we restart an existing one
new_hyperopt = 1

# Number of years used in the training dataset for recalibration
calibration_window = 2

# Unique identifier to read the trials file of hyperparameter optimization
experiment_id = 1

# Number of iterations for hyperparameter optimization
max_evals = 500
path_datasets_folder = f"./{country}/datasets/"
path_hyperparameters_folder = f"./{country}/experimental_files/"





new_recalibration = 1






# Set up the paths for saving data (this are the defaults for the library)
path_datasets_folder = os.path.join(".", f"{country}/datasets")
path_recalibration_folder = os.path.join(".", f"{country}/experimental_files")
path_hyperparameter_folder = os.path.join(".", f"{country}/experimental_files")

start_dnn = time.time()
evaluate_dnn_in_test_dataset(
    experiment_id,
    path_hyperparameter_folder=path_hyperparameter_folder,
    path_datasets_folder=path_datasets_folder,
    shuffle_train=shuffle_train,
    path_recalibration_folder=path_recalibration_folder,
    nlayers=nlayers,
    dataset=dataset,
    data_augmentation=data_augmentation,
    calibration_window=calibration_window,
    new_recalibration=new_recalibration,
    begin_test_date=begin_test,
    end_test_date=end_test,
)
end_dnn = time.time()

execution_time_dnn = end_dnn - start_dnn

# Save execution time
file_execution_time_dnn = (
    f"{country}/execution_time_dnn.pkl"
)
joblib.dump(execution_time_dnn, file_execution_time_dnn)


forecast_DNN = pd.read_csv(
    f"{country}/experimental_files/DNN_forecast_nl2_datmy_data_YT2_SFH0_CW2_1.csv",
    index_col=False,
)



DNN_forcast = forecast_DNN.iloc[:, 1:]
forecast_stack_DNN = DNN_forcast.stack()



test_data = data.loc[begin_test:end_test]



forecast_errors_DNN = forecast_stack_DNN.reset_index(drop=True) - price_test_series


forcast_DNN_num = forecast_errors_DNN.to_numpy()


np.mean(forcast_DNN_num**2) ** 0.5
