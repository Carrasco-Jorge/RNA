import matplotlib.pyplot as plt

def plot_cost(training_info, figsize=(10,6), alpha=0.75, plot_training=False, title=None):
    epochs = training_info.shape[0]
    fig, ax1 = plt.subplots(figsize=figsize)
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Validation cost', color="red")
    ax1.plot(range(epochs) ,training_info.iloc[:,1], color="red", alpha=alpha)
    if plot_training:
        ax2 = ax1.twinx()
        ax2.set_ylabel('Train cost', color="blue")
        ax2.plot(range(epochs) ,training_info.iloc[:,0], color="blue", alpha=alpha)
    #fig.tight_layout()
    plt.title(title)
    plt.show()

def plot_accuracy(training_info, figsize=(10,6), alpha=0.75, plot_training=False, title=None):
    epochs = training_info.shape[0]
    fig, ax = plt.subplots(figsize=figsize)
    ax.set_xlabel('Epochs')
    ax.set_ylabel('Accuracy')
    if plot_training: ax.plot(range(epochs) ,training_info.iloc[:,2], color="blue", alpha=alpha, label="Train")
    ax.plot(range(epochs) ,training_info.iloc[:,3], color="red", alpha=alpha, label="Validation")
    ax.set_ylim([min(training_info.iloc[:,3]),1.0])
    #fig.tight_layout()
    plt.legend()
    plt.title(title)
    plt.show()