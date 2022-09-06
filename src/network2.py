import random
import numpy as np
from src.preprocessing import printProgressBar
from src.timer import Timer

class Network():
    def __init__(self, sizes):
        if not isinstance(sizes, list):
            raise ValueError("sizes must be a list of int. Represents the layer sizes.")

        n = len(sizes)
        self.num_layers = len(sizes)

        self.weights = [np.random.randn(sizes[i], sizes[i-1])/np.sqrt(sizes[i-1]) for i in range(1,n)]
        self.biases = [np.random.randn(size) for size in sizes[1:]]

        self.v_w = [np.zeros((sizes[i], sizes[i-1])) for i in range(1,n)]
        self.v_b = [np.zeros(size) for size in sizes[1:]]

        self.timer = Timer()


    def SGD(self, training_data:np.ndarray, outputs:int=1, epochs:int=10, batch_size:int=10, eta:float=0.1, mu:float=0.5, test_data:np.ndarray=None, train_monitoring=False):
        """
        Trains neural network applying the base Stochastic Gradient Descent algorithm. Receives training data as a matrix X with the inputs at the left and the outputs at the right, indicated by the ``outputs`` parameter.
        """
        print("Begining training...\n")
        n = training_data.shape[0]
        
        self.reset_log()
        self.write_log(f"Training set: {n}\nMnibatch size: {batch_size}\nEta: {eta}\nMu: {mu}\n\n")

        # Initialize cost and accuracy arrays
        test_cost_arr = np.zeros(epochs)
        train_cost_arr = np.zeros(epochs)
        test_acc_arr = np.zeros(epochs)
        train_acc_arr = np.zeros(epochs)

        # Training loop
        for counter in range(epochs):
            self.timer.start()

            random.shuffle(training_data)
            mini_batches = [training_data[k:k+batch_size,:]
                            for k in range(0, n, batch_size)]

            for minibatch in mini_batches:
                # Backpropagation on minibatch
                nabla_w_t = [np.zeros(w.shape) for w in self.weights]
                nabla_b_t = [np.zeros(b.shape) for b in self.biases]
                for X in minibatch:
                    # Split inputs and outputs
                    x = X[:-outputs].copy()
                    y = X[-outputs:].copy()
                    # Backpropagation for obs 'x'
                    nabla_w, nabla_b = self.backpropagation(x, y)
                    # Add gradient to total
                    nabla_w_t = [nw_t+nw for nw_t, nw in zip(nabla_w_t,nabla_w)]
                    nabla_b_t = [nb_t+nb for nb_t, nb in zip(nabla_b_t,nabla_b)]

                # Update weights and biases
                # Backpropagation with momentum
                v_prime = lambda v, nu: (1-mu) * v - nu
                f = lambda u, v: u + (eta/batch_size) * v
                
                weights = []
                v_w = []
                for w, nw, v in zip(self.weights,nabla_w_t,self.v_w):
                    new_v = v_prime(v,nw)
                    v_w.append(new_v)
                    weights.append(f(w,new_v))
                biases = []
                v_b = []
                for b, nb, v in zip(self.biases,nabla_b_t,self.v_b):
                    new_v = v_prime(v,nb)
                    v_b.append(new_v)
                    biases.append(f(b,new_v))
                
                self.weights = weights
                self.biases = biases
                self.v_w = v_w
                self.v_b = v_b

            # Evaluate accuracy
            test_cost, test_accuracy = self.evaluate(test_data, outputs)

            test_cost_arr[counter] = test_cost
            test_acc_arr[counter] = test_accuracy
            # Log info
            string = self.print_epoch_info(counter+1,
                                            test_accuracy)

            if train_monitoring:
                train_cost, train_accuracy = self.evaluate(training_data, outputs)
                train_cost_arr[counter] = train_cost
                train_acc_arr[counter] = train_accuracy

            self.write_log(string)
            self.timer.end()
            # Progress bar
            printProgressBar(counter+1,epochs, prefix=f"Epoch {counter+1}", suffix=f"| {self.timer.time:.2f}", length=50)

        print("\nTraining finished.")
        return train_cost_arr, test_cost_arr, train_acc_arr, test_acc_arr

    def feedforward(self, x, save=True):
        current_a = x
        if save:
            a_s = [current_a]
            z_s = []
        for w, b in zip(self.weights, self.biases):
            if save:
                z_s.append(np.dot(w,current_a) + b)
                current_a = self.sigma(z_s[-1])
                a_s.append(current_a)
            else:
                z = np.dot(w,current_a) + b
                current_a = self.sigma(z)

        if save:
            return z_s, a_s
        return current_a

    def backpropagation(self, x, y):
        # Feedforward (save a's ad z's)
        z_s, a_s = self.feedforward(x)

        # Initialize nw, nb arrays
        nw = [np.zeros(w.shape) for w in self.weights]
        nb = [np.zeros(b.shape) for b in self.biases]

        # Backpropagation procedure
         # Output layer
        delta = self.delta_output(a_s[-1], y, z_s[-1])
        nw[-1] = np.outer(delta, a_s[-2])
        nb[-1] = delta
         # Hidden layers
        for l in range(2,self.num_layers,1):
            delta = np.dot(self.weights[-l+1].transpose(), delta) * self.sigma_prime(z_s[-l])
            nw[-l] = np.outer(delta, a_s[-l-1])
            nb[-l] = delta

        return nw, nb

    def evaluate(self, data, outputs):
        n = data.shape[0]
        score = 0
        cost = 0
        for X in data:
            # Split inputs from outputs
            x = X[:-outputs].copy()
            y = X[-outputs:].copy()
            # Feedforward
            output = self.feedforward(x, save=False)
            y_pred = np.argmax(output)
            y_class = np.argmax(y)

            if y_class == y_pred:
                score += 1

            cost += self.cost_function(output, y)
        """
        test_results = [(np.argmax(self.feedforward(x)), y)
                        for (x, y) in test_data]
        return sum(int(x == y) for (x, y) in test_results)
        """
        return cost/n, score/n

    def print_epoch_info(self, num_epoch, validation_accuracy):
        s = f"Finished epoch {num_epoch}\n  Accuracy: validation -> {validation_accuracy:.4f}\n\n"
        return s

    def reset_log(self):
        file = open("ann_log.txt", "w")
        file.close()

    def write_log(self, string):
        file = open("ann_log.txt", "a")
        file.write(string)
        file.close()

    def cost_function(self, output, y):
        return (np.sum((output - y)**2))/2

    def delta_output(self, output, y, z):
        return (output - y) * self.sigma_prime(z)
    
    @staticmethod
    def sigma(z):
        s = 1.0/(1.0+np.exp(-z))
        return s
    @staticmethod
    def sigma_prime(z):
        s = Network.sigma(z)
        return s*(1-s)
