#!/usr/bin/env python

import os, warnings, sys

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # or any {'0', '1', '2'}
warnings.filterwarnings("ignore")

import algorithm.preprocessing.pipeline as pp_pipe
import algorithm.preprocessing.preprocess_utils as pp_utils
import algorithm.utils as utils
from algorithm.model.clustering import ClusteringModel as Model
from algorithm.utils import get_model_config


# get model configuration parameters
model_cfg = get_model_config()


def get_trained_model(train_data, data_schema, hyper_params):

    # set random seeds
    utils.set_seeds()

    # preprocess data
    print("Pre-processing data...")
    train_X, _, preprocess_pipe = preprocess_data(train_data, None, data_schema)

    num_clusters = data_schema["datasetSpecs"]["suggestedNumClusters"]

    # Create and train model
    print("Fitting model ...")
    model = train_model(train_X, hyper_params, num_clusters)

    return preprocess_pipe, model


def train_model(train_X, hyper_params, num_clusters):
    # get model hyper-paameters parameters
    model_params = {**hyper_params, "K": num_clusters}

    # Create and train model
    model = Model(**model_params)
    model.fit(train_X)
    return model


def preprocess_data(train_data, valid_data, data_schema):
    # print('Preprocessing train_data of shape...', train_data.shape)
    pp_params = pp_utils.get_preprocess_params(train_data, data_schema, model_cfg)

    preprocess_pipe = pp_pipe.get_preprocess_pipeline(pp_params, model_cfg)
    train_data = preprocess_pipe.fit_transform(train_data)

    if valid_data is not None:
        valid_data = preprocess_pipe.transform(valid_data)
    return train_data, valid_data, preprocess_pipe
