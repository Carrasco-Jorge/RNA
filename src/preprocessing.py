import numpy as np

def one_hot_encode(column):
    column = np.array(column)
    if len(column.shape) > 1:
        raise ValueError("column must be a 1-D ndarray.")
    
    categories = np.unique(column)
    columns = []

    for i in range(len(categories)):
        columns.append(column==categories[i])

    columns = np.column_stack(columns).astype(int)

    return columns

def split_data(data, test_size = 0.1):
    if not isinstance(test_size, float) or test_size >= 1 or test_size <= 0:
        raise ValueError("test_size must be a number between 0 and 1.")
    
    data = np.array(data)
    np.random.shuffle(data)
    indx = int(data.shape[0] * test_size)
    training_set = data[indx:]
    test_set = data[:indx]

    return training_set, test_set

# Print iterations progress
def printProgressBar (iteration, total, prefix = '', suffix = '', decimals = 1, length = 100, fill = '█', printEnd = "\r"):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
        printEnd    - Optional  : end character (e.g. "\r", "\r\n") (Str)
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print(f'\r{prefix} |{bar}| {percent}% {suffix}', end = printEnd)
    