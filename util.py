import numpy as np
import csv
from sklearn.linear_model import LogisticRegression
from sklearn import metrics


def shuffle(X, y):
    permutation = np.arange(X.shape[0])
    np.random.shuffle(permutation)
    return X[permutation], y[permutation]

def load(filename):
    reader = csv.reader(open(filename, 'r'), delimiter=',')
    data = np.array(list(reader)).astype('float')
    labels = data[:, 0]
    data = data[:, 1:]
    return shuffle(data, labels)

def load_records(filename):
    records = []
    with open(filename, 'r') as file:
        reader = csv.reader(file, delimiter=',')
        for row in reader:
            records.append(list(map(float, row)))
    return records

def load_keystone_data(train_filename, test_filename):
    train_records = load_records(train_filename)
    test_records = load_records(test_filename)

    optimal_train = train_records[-1]
    optimal_test = test_records[-1]
    return np.array(train_records[:-1]), optimal_train, np.array(test_records[:-1]), optimal_test



class Metrics:
    def __init__(self, train_labels, test_labels):
        # track metrics for all runs
        self.total_train_loss = []
        self.total_train_error = []
        self.total_test_loss = []
        self.total_test_error = []
        self.test_labels = test_labels
        self.train_labels = train_labels

    def add_predictions(self, train_predictions, test_predictions):
        self.train_errors.append(metrics.mean_absolute_error(self.train_labels, train_predictions))
        self.train_losses.append(metrics.log_loss(self.train_labels, train_predictions))
        self.test_losses.append(metrics.log_loss(self.test_labels, test_predictions))
        self.test_errors.append(metrics.mean_absolute_error(self.test_labels, test_predictions))

    def start_run(self):
        # track metrics for one runs
        self.train_errors = []
        self.train_losses = []
        self.test_losses = []
        self.test_errors = []

    def finish_run(self):
        self.total_test_error.append(self.test_errors)
        self.total_test_loss.append(self.test_losses)
        self.total_train_error.append(self.train_errors)
        self.total_train_loss.append(self.train_losses)

    def calculate_metrics(self):
        if hasattr(self, 'test_loss_data'):
            return

        all_test_losses = np.array(self.total_test_loss)
        all_test_errors = np.array(self.total_test_error)
        all_train_losses = np.array(self.total_train_loss)
        all_train_errors = np.array(self.total_train_error)

        # calculate averages for each run
        mean_test_losses = np.mean(all_test_losses, axis=0)
        mean_test_errors = np.mean(all_test_errors, axis=0)
        mean_train_losses = np.mean(all_train_losses, axis=0)
        mean_train_errors = np.mean(all_train_errors, axis=0)

        # calculate standard deviations
        std_test_losses = np.std(all_test_losses, axis=0)
        std_test_errors = np.std(all_test_errors, axis=0)
        std_train_losses = np.std(all_train_losses, axis=0)
        std_train_errors = np.std(all_train_errors, axis=0)
        
        self.test_loss_data = (mean_test_losses, std_test_losses)
        self.test_error_data = (mean_test_errors, std_test_errors)
        self.train_loss_data = (mean_train_losses, std_train_losses)
        self.train_error_data = (mean_train_errors, std_train_errors)



def incremental(train_data, train_labels, test_data, test_labels, num_batches, num_runs, reg_param, num_iterations, annealing=False):
    data_batches, label_batches = load_incremental_batches(train_data, train_labels)

    incremental_metrics = Metrics(train_labels, test_labels)
