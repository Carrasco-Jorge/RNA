import gzip
import pickle
import numpy as np
import pandas as pd
from src.preprocessing import one_hot_encode

f = gzip.open('data/mnist.pkl.gz', 'rb')
training_data, validation_data, test_data = pickle.load(f, encoding="latin1")
print("Training shape   : ",(training_data[0]).shape, (training_data[1]).shape)
y = one_hot_encode(training_data[1])
train = np.column_stack([training_data[0],
                         y])

print("Validation shape : ",(validation_data[0]).shape, (validation_data[1]).shape)
y = one_hot_encode(validation_data[1])
validation = np.column_stack([validation_data[0],
                              y])

print("Test shape       : ",(test_data[0]).shape, (test_data[1]).shape)
y = one_hot_encode(test_data[1])
test = np.column_stack([test_data[0],
                        y])

f.close()


print("Final shapes : ", train.shape, validation.shape, test.shape)

pd.DataFrame(train).to_csv("data/preprocessed/train.gz", index=False, header=False)
pd.DataFrame(validation).to_csv("data/preprocessed/validation.gz", index=False, header=False)
pd.DataFrame(test).to_csv("data/preprocessed/test.gz", index=False, header=False)
