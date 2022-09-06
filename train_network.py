import pickle
#from src.network import Network
from src.network2 import Network
from src.saveload import load_data_bundle, save_data

# 28 x 28 pix ==> 784 inputs
training, validation, test = load_data_bundle()

print(f"Training set shape   : {training.shape}")
print(f"Validation set shape : {validation.shape}\n")


# Initialize network object
network = Network([784, 30, 10])

# Begin training
epochs = 30
batch_size = 10
eta = 7.5
mu = 0.9

train_cost, test_cost, train_acc, test_acc = network.SGD(training,
                              epochs=epochs,
                              batch_size=batch_size,
                              eta=eta,
                              mu=mu,
                              outputs=10,
                              test_data=validation,
                              train_monitoring=True)

save_data({"Train cost":train_cost,
            "Test cost":test_cost,
            "Train accuracy":train_acc,
            "Test accuracy":test_acc},
            "data/training_info.csv",
            1)

file = open("data/network_data.pkl", "wb")
pickle.dump(network, file)
file.close()
