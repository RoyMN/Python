import numpy as np
import pandas as pd
import random
import pprint


def train_test_split(df, test_size):
    """ Splits the data-frame into train and test-samples. Takes as
    input the data-frame and test_size as a fraction (e.g. 0<x<1) """

    test_size = round(test_size * len(df))
    data_indices = df.index.tolist()
    test_indices = random.sample(population=data_indices, k=test_size)
    test_df = df.loc[test_indices]
    train_df = df.drop(test_indices)
    return train_df, test_df


def give_label(data):
    """Heper-function to give a label based on the data inputted. It will use a
    majority-vote to find the label"""

    label_column = data[:, -1]
    unique_labels, counts_unique_labels = np.unique(label_column, return_counts=True)
    vote_index = counts_unique_labels.argmax()
    label = unique_labels[vote_index]
    return label


def purity_check(data):
    """Helper-function determining whether or not a data-set
    is pure, i.e. all classes are the same."""

    unique_labels = np.unique(data[:, -1])
    if len(unique_labels) == 1:
        return True
    else:
        return False


def find_possible_splits(data):
    """Looks through each column of the data and adds potential
    splits to a dictionary"""

    potential_splits = {}
    n_rows, n_columns = data.shape
    for column_index in range(n_columns - 1):
        potential_splits[column_index] = []
        unique_values = np.unique(data[:, column_index])
        for index in range(len(unique_values)):
            potential_splits[column_index].append(unique_values[index])
    return potential_splits


def split(data, split_column, split_value):
    """Takes in a data-set, a column and value to split on. Returns
    a left_split containing all the rows with that value, and a right_split
    containing all the rows that does not have that value."""

    split_column_values = data[:, split_column]
    left_split = data[split_column_values == split_value]
    right_split = data[split_column_values != split_value]
    return left_split, right_split


def calculate_entropy(data):
    """Calculates the entropy on some data by entropy
    formula E = -sum*p(log2 * p)"""

    label_column = data[:, -1]
    _, counts = np.unique(label_column, return_counts=True)
    probabilities = counts / counts.sum()  # This is a list of probabilities
    entropy = -sum(probabilities * np.log2(probabilities))
    return entropy


def calculate_total_entropy(left_split, right_split):
    """Calculates the total entropy of the data-set after having made some
    split (left_split)."""

    n_data_points = len(left_split) + len(right_split)
    p_left_split = len(left_split) / n_data_points
    p_right_split = len(right_split) / n_data_points

    total_entropy = (p_left_split * calculate_entropy(left_split)
                     + p_right_split * calculate_entropy(right_split))
    return total_entropy


def calculate_gini(data):
    """Calculates the gini-impurity on some data
    by formula I(G)=1-sum(f(i,j)**2)"""

    label_column = data[:, -1]
    _, counts = np.unique(label_column, return_counts=True)
    probabilities = counts / counts.sum()
    gini_impurity = 1 - sum(probabilities**2)
    return gini_impurity


def calculate_total_gini(left_split, right_split):
    """Calculates the total gini-impurity of the data-set after
    having made some split (left_split)."""

    n_data_points = len(left_split) + len(right_split)
    cost_left = calculate_gini(left_split)
    cost_right = calculate_gini(right_split)

    total_gini_impurity = (len(left_split)/n_data_points)*cost_left \
                          + (len(right_split)/n_data_points)*cost_right

    return total_gini_impurity


def find_best_split(data, possible_splits, impurity_measure='entropy'):
    """Find the best split based on some data and the possible
    splits on that data."""

    #
    if impurity_measure == 'gini':
        overall_cost = 1000000000   # Starting with an arbitrary high number
        for column_index in possible_splits:
            for value in find_possible_splits[column_index]:
                split_left, split_right = split(data, column_index, value)
                current_cost = calculate_total_gini(split_left, split_right)
                if current_cost <= overall_cost:
                    overall_cost = current_cost
                    split_column, split_value = column_index, value
        return split_column, split_value

    else:
        overall_entropy = 1000000000
        for column_index in possible_splits:
            for value in possible_splits[column_index]:
                split_left, split_right = split(data, column_index, value)
                current_total_entropy = calculate_total_entropy(split_left, split_right)
                if current_total_entropy <= overall_entropy:
                    overall_entropy = current_total_entropy
                    split_column, split_value = column_index, value
        return split_column, split_value


def learn(X, y, impurity_measure='entropy', column_labels=None, prune=False):
    """Learns a ID3-tree based on data-set X and target y. It is assumed
    neither has a header. A pandas data-frame will be constructed, and the headers
    will be 0,1,2,..Class. If you already have a pandas data-frame,
    using it directly would be easier."""

    dx = pd.DataFrame(X)
    dt = pd.DataFrame(y)
    df = pd.concat([dx, dt], axis=1)
    if column_labels is not None:
        df.columns = column_labels
    else:
        df.columns = np.arange(0, (len(df.columns)))

    return learn(df, impurity_measure)


def learn(df, impurity_measure='entropy', counter=0, prune=False):
    """Learns a ID3-tree based on a dataframe. The expected format is that
    the dataframe has a header, and target/label-column as the last column
    and named 'Class' exactly"""

    if counter == 0:
        #   Making sure the column-header is not overwritten
        #   in the recursive calls.
        global feature_names
        feature_names = df.columns
        data = df.values
    else:
        data = df

    #   If the data is pure, set the label
    if purity_check(data):
        label = give_label(data)
        return label

    else:
        counter += 1

        #   Make a split
        possible_splits = find_possible_splits(data)
        split_column, split_value = find_best_split(data, possible_splits)
        split_left, split_right = split(data, split_column, split_value)

        #   Make a tree-stub.
        feature = feature_names[split_column]

        node = "{} = {}".format(feature, split_value)
        tree = {node: []}

        #   Adding to nodes by recursion.
        #   Prune the number of nodes by the rule that if there is no
        #   decision-gain in the splits of the node, then just return it
        #   as a leaf.
        if learn(split_left, impurity_measure, counter) == learn(split_right, impurity_measure, counter):
            tree = learn(split_left, impurity_measure, counter)
        else:
            #   left, or "yes", goes first and right, or "no" goes second
            tree[node].append(learn(split_left, impurity_measure, counter))
            tree[node].append(learn(split_right, impurity_measure, counter))

        return tree


def predict(x, tree):
    """Makes a prediction for data-point x from
    the specified tree"""

    node = list(tree.keys())[0]
    feature, _, value = node.split()
    if x[feature] == value:
        #   Going left, which is the first element
        answer = tree[node][0]
    else:
        answer = tree[node][1]

    if not isinstance(answer, dict):
        #   This means we've reached a leaf
        return answer
    else:
        #   Recursively traverse the tree
        sub_tree = answer
        return predict(x, sub_tree)
    return


def calculate_accuracy(df, tree):
    df["classification"] = df.apply(predict, axis=1, args=(tree,))
    df["correct"] = df.classification == df.Class

    accuracy = df.correct.mean()

    return accuracy


def prune(tree):
    """Prunes a tree by the following logic: if removing
    a node (Not finished)"""
    return

if __name__ == '__main__':
    ''' From here you will find staging of the examples.'''
    #   Reading the data-set and setting all attributes with '?' as NaN-values
    #df = pd.read_csv("agaricus-lepiota.data", na_values=['?'])
    df = pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/mushroom/agaricus-lepiota.data", na_values=['?'])

    #   Swapping first and last column to get target as last column.
    col_list = list(df)
    col_list[0], col_list[-1] = col_list[-1], col_list[0]
    df = df.loc[:, col_list]
    #   Dropping all NaN-values
    df = df.dropna()
    feature_names = df.columns

    #   Seperate train, validation and test-data.
    remaining_df, validation_df = train_test_split(df, 0.2)
    train_df, test_df = train_test_split(remaining_df, 0.1)

    #   Learn the tree (usage of pandas data-frame is simpler
    #   then data-set X and target y)
    tree1 = learn(train_df)
    tree2 = learn(train_df, impurity_measure='gini')
    pp = pprint.PrettyPrinter(indent=4)
    #   Printing a "pretty-format" of the tree's (still not pretty)
    pp.pprint(tree1)
    pp.pprint(tree2)
    #   uncommented as I'm using pd instead.
    # X = df.iloc[:, :-1]
    # y = df.iloc[:, -1]
    # labels = [
    #  'class', 'cap shape', 'cap surface', 'cap color', 'bruised', 'odor',
    #  'gill attachment', 'gill spacing', 'gill size', 'gill color',
    #  'stalk shape', 'stalk root', 'stalk surface above ring',
    #  'stalk surface below ring', 'stalk color above ring',
    #  'stalk color below ring', 'veil type', 'veil color', 'ring number',
    #  'ring type', 'spore print color', 'population', 'habitat'
    # ]
    # tree = learn(X, y, impurity_measure='entropy', class_labels=labels)

    #   Example of how to make a prediction
    print("Object:")
    print(validation_df.iloc[0])
    print("prediction: "+predict(validation_df.iloc[0], tree1))