import numpy as np
import pickle
from src.network import Network
from src.saveload import load_data
from src.plot import plot_cost, plot_accuracy

# Load network
network = Network([784,30,10])

file = open("data/network_data.pkl", "rb")
network = pickle.load(file)
file.close()

# Load test set
test = np.array(load_data("data/preprocessed/test.gz"))

# Load training info
training_info = load_data("data/training_info.csv",1)

# Hyper params
eta = 7.5
mu = 0.9

# Visualization
plot_cost(training_info, plot_training=True, title=f"eta = {eta}, mu = {mu}")
plot_accuracy(training_info, plot_training=True, title=f"eta = {eta}, mu = {mu}")

# Test accuracy
cost, accuracy = network.evaluate(test, 10)
print(f"Test accuracy : {accuracy}")
