from collections import Counter
import random
import sys
from math import log2

class DecisionNode:
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels
        self.feature_index = None
        self.threshold = None
        self.left = None
        self.right = None
        self.prediction = self.get_highest_class(labels)

    def get_highest_class(self, labels):
        class_counts = Counter(labels)
        majority_class = max(class_counts, key=class_counts.get)
        return majority_class

def get_most_common(lst):
    counts = {}
    for elem in lst:
        if elem in counts:
            counts[elem] += 1
        else:
            counts[elem] = 1
    return max(counts, key=counts.get)

def calculate_entropy(labels):
    class_counts = Counter(labels)
    total_samples = len(labels)

    entropy = -sum((count / total_samples) * (log2(count / total_samples) + 1e-10) for count in class_counts.values())

    return entropy

def calculate_information_gain(parent_labels, left_labels, right_labels):
    parent_entropy = calculate_entropy(parent_labels)
    left_entropy = calculate_entropy(left_labels)
    right_entropy = calculate_entropy(right_labels)

    left_weight = len(left_labels) / len(parent_labels)
    right_weight = len(right_labels) / len(parent_labels)

    gain = parent_entropy - (left_weight * left_entropy + right_weight * right_entropy)

    return gain

def find_best_split(data, labels, option):
    no_of_attributes = len(data[0])
    highest_gain = 0
    best_threshold = None
    best_feature_index = None

    if option == 'randomized':
        feature_index = random.randint(0, no_of_attributes - 1)
        thresholds = list(set(sample[feature_index] for sample in data))
        for threshold in thresholds:
            isLeftTrue = [sample[feature_index] <= threshold for sample in data]
            isRightTrue = [not left for left in isLeftTrue]

            left_labels = [labels[i] for i in range(len(labels)) if isLeftTrue[i]]
            right_labels = [labels[i] for i in range(len(labels)) if isRightTrue[i]]

            gain = calculate_information_gain(labels, left_labels, right_labels)

            if gain > highest_gain:
                highest_gain = gain
                best_threshold = threshold
                best_feature_index = feature_index
    else:
        for feature_index in range(no_of_attributes):
            thresholds = list(set(sample[feature_index] for sample in data))
            for threshold in thresholds:
                isLeftTrue = [sample[feature_index] <= threshold for sample in data]
                isRightTrue = [not left for left in isLeftTrue]

                left_labels = [labels[i] for i in range(len(labels)) if isLeftTrue[i]]
                right_labels = [labels[i] for i in range(len(labels)) if isRightTrue[i]]

                gain = calculate_information_gain(labels, left_labels, right_labels)

                if gain > highest_gain:
                    highest_gain = gain
                    best_threshold = threshold
                    best_feature_index = feature_index

    return best_feature_index, best_threshold

def build_tree(data, labels, depth=0, max_depth=None, option='optimized'):
    unique_classes = list(set(labels))

    if len(unique_classes) == 1 or (max_depth is not None and depth == max_depth):
        return DecisionNode(data, labels)

    best_feature_index, best_threshold = find_best_split(data, labels, option)

    if best_feature_index is None:
        return DecisionNode(data, labels)

    isLeftTrue = [sample[best_feature_index] <= best_threshold for sample in data]
    isRightTrue = [not left for left in isLeftTrue]

    left_data = [data[i] for i in range(len(data)) if isLeftTrue[i]]
    left_labels = [labels[i] for i in range(len(labels)) if isLeftTrue[i]]

    right_data = [data[i] for i in range(len(data)) if isRightTrue[i]]
    right_labels = [labels[i] for i in range(len(labels)) if isRightTrue[i]]

    node = DecisionNode(data, labels)
    node.feature_index = best_feature_index
    node.threshold = best_threshold

    node.left = build_tree(left_data, left_labels, depth + 1, max_depth, option)
    node.right = build_tree(right_data, right_labels, depth + 1, max_depth, option)

    return node

def predict_sample(tree, sample):
    if tree.left is None and tree.right is None:
        return tree.prediction

    if sample[tree.feature_index] <= tree.threshold:
        return predict_sample(tree.left, sample)
    else:
        return predict_sample(tree.right, sample)

def predict(tree, data):
    predictions = [predict_sample(tree, sample) for sample in data]
    return predictions

def build_decision_forest(data, labels, num_trees=3, max_depth=None, option='randomized'):
    forest = []
    for _ in range(num_trees):
        subset_indices = random.sample(range(len(data)), len(data))

        subset_data = [data[i] for i in subset_indices]
        subset_labels = [labels[i] for i in subset_indices]

        tree = build_tree(subset_data, subset_labels, max_depth=max_depth, option=option)
        forest.append(tree)
        
    return forest

def predict_decision_forest(forest, data):
    predictions = [get_most_common(predict(tree, data)) for tree in forest]
    return predictions

def read_dataset(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()
        data = [list(map(float, line.strip().split())) for line in lines]
    features = [sample[:-1] for sample in data]
    labels = [sample[-1] for sample in data]
    return features, labels

def accuracy(predictions, true_labels):
    correct = sum(pred == true for pred, true in zip(predictions, true_labels))
    total = len(true_labels)
    return correct / total

def print_results(index, predicted_class, true_class, accuracy):
    print(f"Index = {index}, Result = {predicted_class}, True Class = {true_class}, Accuracy = {accuracy:.2f}")


def main(training_file, test_file, option):
    training_data, training_labels = read_dataset(training_file)
    test_data, test_labels = read_dataset(test_file)

    predictions = []

    if option == 'forest3' or option == 'forest15':
        num_trees = 3 if option == 'forest3' else 15
        forest = build_decision_forest(training_data, training_labels, num_trees=num_trees, option='randomized')
        for index, (sample, true_label) in enumerate(zip(test_data, test_labels)):
            predictions_per_tree = predict_decision_forest(forest, [sample])
            predicted_class = predicted_class = get_most_common(predictions_per_tree)
            acc = accuracy(predictions_per_tree, [true_label])
            print_results(index, predicted_class, true_label, acc)
            predictions.append(predicted_class)
    else:
        tree = build_tree(training_data, training_labels, option=option)
        for index, (sample, true_label) in enumerate(zip(test_data, test_labels)):
            prediction = predict(tree, [sample])[0]
            acc = accuracy([prediction], [true_label])
            print_results(index, prediction, true_label, acc)
            predictions.append(prediction)

    overall_accuracy = accuracy(predictions, test_labels)
    print(f"\nClassification Accuracy = {overall_accuracy:.2%}")

if __name__ == "__main__":

    if len(sys.argv) != 4:
        print("Incorrect Input")
        sys.exit(1)

    training_file = sys.argv[1]
    test_file = sys.argv[2]
    option = sys.argv[3]

    main(training_file, test_file, option)
