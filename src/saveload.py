import numpy as np
import pandas as pd

def load_data(path, header=None):
    return pd.read_csv(path,header=header)

def save_data(data, path, header=None):
    pd.DataFrame(data).to_csv(path, index=False, header=header)

def load_data_bundle():
    training = np.array(load_data("data/preprocessed/train.gz"))
    validation = np.array(load_data("data/preprocessed/validation.gz"))
    test = np.array(load_data("data/preprocessed/test.gz"))

    return training, validation, test