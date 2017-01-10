import numpy as np
import csv
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
import time

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

# each batch includes all data up to and including current index.
def load_optimal_batches(data, labels, num_batches):
    batch_size = int(data.shape[0] / num_batches)
    data_batches = []
    label_batches = []
    for i in range(num_batches):
        k = (i + 1) * batch_size
        data_batches.append(data[:k])
        label_batches.append(labels[:k])
    return data_batches, label_batches

# each batch includes only that subset's data.
def load_incremental_batches(data, labels, num_batches):
    batch_size = int(data.shape[0] / num_batches)
    data_batches = []
    label_batches = []
    for i in range(num_batches):
        k = i * batch_size
        data_batches.append(data[k:k+batch_size])
        label_batches.append(labels[k:k+batch_size])
    return data_batches, label_batches

def load_incremental_labels(labels, num_batches):
    batch_size = int(len(labels) / num_batches)
    label_batches = []
    for i in range(num_batches):
        k = i * batch_size
        label_batches.append(labels[k:k+batch_size])
    return label_batches

class Metrics:
    def __init__(self, train_label_batches, test_labels):
        # track metrics for all runs
        self.total_train_loss = []
        self.total_train_error = []
        self.total_test_loss = []
        self.total_test_error = []
        self.test_labels = test_labels
        self.train_label_batches = train_label_batches
        self.total_train_times = []

    def start_trial(self):
        self.start_time = time.time()
    
    def end_trial(self):
        end_time = time.time()
        elapsed_time = end_time - self.start_time
        if len(self.cumulative_times) == 0:
            prev_time = 0
        else:
            prev_time = self.cumulative_times[-1]
        self.cumulative_times.append(prev_time + elapsed_time)

    def add_predictions(self, train_predictions, test_predictions, batch_index):
        train_labels = self.train_label_batches[batch_index]
        self.train_errors.append(metrics.mean_absolute_error(train_labels, train_predictions))
        self.train_losses.append(metrics.log_loss(train_labels, train_predictions))
        self.test_losses.append(metrics.log_loss(self.test_labels, test_predictions))
        self.test_errors.append(metrics.mean_absolute_error(self.test_labels, test_predictions))

    def start_run(self):
        # track metrics for one runs
        self.train_errors = []
        self.train_losses = []
        self.test_losses = []
        self.test_errors = []
        self.cumulative_times = []

    def finish_run(self):
        self.total_test_error.append(self.test_errors)
        self.total_test_loss.append(self.test_losses)
        self.total_train_error.append(self.train_errors)
        self.total_train_loss.append(self.train_losses)
        self.total_train_times.append(self.cumulative_times)

    def calculate_metric(self, metric_data):
        all_of_metric = np.array(metric_data)
        mean_data = np.mean(all_of_metric, axis=0)
        std_data = np.std(all_of_metric, axis=0)
        return mean_data, std_data
    
    def calculate_metrics(self):
        if hasattr(self, 'test_loss_data'):
            return
        
        self.test_loss_data = self.calculate_metric(self.total_test_loss)
        self.test_error_data = self.calculate_metric(self.total_test_error)
        self.train_loss_data = self.calculate_metric(self.total_train_loss)
        self.train_error_data = self.calculate_metric(self.total_train_error)
        self.train_time_data = self.calculate_metric(self.total_train_times)

