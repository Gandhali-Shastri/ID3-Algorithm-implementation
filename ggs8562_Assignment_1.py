# Name: Gandhali Shastri
# Student id: 1001548562

# References:
# https://machinelearningmastery.com/implement-decision-tree-algorithm-scratch-python/
# https://machinelearningmastery.com/implement-resampling-methods-scratch-python/
# https://medium.com/machine-learning-guy/an-introduction-to-decision-tree-learning-id3-algorithm-54c74eb2ad55
# Dataset: https://archive.ics.uci.edu/ml/machine-learning-databases/

from random import randrange
from csv import reader
import math
import sys


# Load data samples CSV file
def read_data(filename):
    file = open(filename)
    lines = reader(file)
    dataset = list(lines)
    return dataset


# Splittin the dataset into k folds
def split_dataset_cross_validation(dataset, K):
    dataset_split = list()
    dataset_copy = list(dataset)
    fold_size = int(len(dataset) / K)
    for i in range(K):
        fold = list()
        while len(fold) < fold_size:
            index = randrange(len(dataset_copy))
            fold.append(dataset_copy.pop(index))
        dataset_split.append(fold)
    return dataset_split


# Splitting the dataset into a train and test data
def split_dataset(dataset, split=0.80):
    train = list()
    train_size = split * len(dataset)
    dataset_copy = list(dataset)
    while len(train) < train_size:
        index = randrange(len(dataset_copy))
        train.append(dataset_copy.pop(index))
    return train, dataset_copy


# Evaluate when split into 80-20
def evaluate_algorithm_1(dataset, algorithm, *args):
    training_data, test_data = split_dataset(dataset, 0.80)
    acc = list()
    predicted = algorithm(training_data, test_data, *args)
    actual = [row[-1] for row in test_data]
    accuracy = calculate_accuracy(actual, predicted)
    acc.append(accuracy)
    return acc


# Evaluate when cross validation split
def evaluate_algorithm_2(dataset, algorithm, K, *args):
    folds = split_dataset_cross_validation(dataset, K)
    acc = list()
    for fold in folds:
        training_data = list(folds)
        training_data.remove(fold)
        training_data = sum(training_data, [])
        test_data = list()
        for row in fold:
            row_copy = list(row)
            test_data.append(row_copy)
            row_copy[-1] = None
        predicted = algorithm(training_data, test_data, *args)
        actual = [row[-1] for row in fold]
        accuracy = calculate_accuracy(actual, predicted)
        acc.append(accuracy)
    return acc


# Split a dataset based on an attribute and an attribute value
def test_split(index, value, dataset):
    left, right = list(), list()
    for row in dataset:
        if row[index] < value:
            left.append(row)
        else:
            right.append(row)
    return left, right


# Calculate accuracy percentage
def calculate_accuracy(actual, predicted):
    correct = 0
    for i in range(len(actual)):
        if actual[i] == predicted[i]:
            correct += 1
    return correct / float(len(actual)) * 100.0


# Calculate entropy
def entropy(groups, classes, b_score):
    # count all samples at split point
    n_instances = float(sum([len(group) for group in groups]))

    ent = 0
    for group in groups:
        size = float(len(group))

        if size == 0:
            continue
        score = 0.0

        for class_val in classes:
            p = [row[-1] for row in group].count(class_val) / size
        if p > 0:
            score = (p * math.log(p, 2))
        #  Entrpy gain
        ent -= (score * (size / n_instances))
    return ent


# best split for the dataset
def get_split(dataset):
    class_values = list(set(row[-1] for row in dataset))

    b_index, b_value, b_score, b_groups = 999, 999, 1, None
    for index in range(len(dataset[0]) - 1):
        for row in dataset:
            groups = test_split(index, row[index], dataset)
            ent = entropy(groups, class_values, b_score)
            if ent < b_score:
                b_index, b_value, b_score, b_groups = index, row[index], ent, groups
    return {'index': b_index, 'value': b_value, 'groups': b_groups}


# child splits for a node or make terminal
def split(node, max_depth, min_size, depth):
    left, right = node['groups']
    del (node['groups'])

    if not left or not right:
        node['left'] = node['right'] = terminal_node(left + right)
        return

    if depth >= max_depth:
        node['left'], node['right'] = terminal_node(left), terminal_node(right)
        return

    if len(left) <= min_size:
        node['left'] = terminal_node(left)
    else:
        node['left'] = get_split(left)
        split(node['left'], max_depth, min_size, depth + 1)

    if len(right) <= min_size:
        node['right'] = terminal_node(right)
    else:
        node['right'] = get_split(right)
        split(node['right'], max_depth, min_size, depth + 1)


# Classification and Regression Tree Algorithm
def decision_tree(train, test, max_depth, min_size):
    tree = build_tree(train, max_depth, min_size)
    predictions = list()
    for row in test:
        prediction = predict(tree, row)
        predictions.append(prediction)
    return (predictions)


# Build decision tree
def build_tree(train, max_depth, min_size):
    root = get_split(train)

    split(root, max_depth, min_size, 1)
    return root


#  terminal node value
def terminal_node(group):
    outcomes = [row[-1] for row in group]
    return max(set(outcomes), key=outcomes.count)


# Print decision tree
def print_tree(node, depth=0):
    if isinstance(node, dict):
        print('%s[Attribute[%s] = %.50s]' % (depth * '\t', (node['index'] + 1), node['value']))
        print_tree(node['left'], depth + 1)
        print_tree(node['right'], depth + 1)
    else:
        print('%s[%s]' % (depth * ' ', node))


# Make a prediction with a decision tree
def predict(node, row):
    if row[node['index']] < node['value']:
        if isinstance(node['left'], dict):
            return predict(node['left'], row)
        else:
            return node['left']
    else:
        if isinstance(node['right'], dict):
            return predict(node['right'], row)
        else:
            return node['right']


def display():
    print('Attributes ', attributes)
    print('Textual Representation of tree')
    print_tree(tree)
    Accuracy1 = evaluate_algorithm_1(dataset, decision_tree, max_depth, min_size)
    print('  ')
    print('Splitting data into 80-20%')
    print('Scores: %s' % Accuracy1)
    print('Accuracy: %.3f%%' % (sum(Accuracy1) / float(len(Accuracy1))))
    # Calculating acc for k cross validation by setting K value
    print('  ')
    print('k-cross validation')
    Accuracy2 = evaluate_algorithm_2(dataset, decision_tree, K, max_depth, min_size)
    print('Accuracies: %s' % Accuracy2)
    print('Mean Accuracy: %.3f%%' % (sum(Accuracy2) / float(len(Accuracy2))))


if __name__ == '__main__':
    attribute_file = sys.argv[1]
    data_file = sys.argv[2]

    f = open(attribute_file)
    attributes = f.readline().split(',')
    attributes = attributes[0:len(attributes) - 1]

    f.close()

    dataset = read_data(data_file)

    K = 5
    max_depth = 3
    min_size = 1

    training_data, test_data = split_dataset(dataset, 0.80)
    tree = build_tree(training_data, max_depth, min_size)

    display()