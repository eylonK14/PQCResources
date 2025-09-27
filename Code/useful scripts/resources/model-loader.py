from joblib import load
import numpy as np
import pandas as pd
from sklearn import tree
from joblib import dump, load
import ast
import argparse
import warnings
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import _tree


warnings.filterwarnings("ignore")

def load_data(target):
    data = pd.read_csv('DataAndModels/NoamData/all-files-ech-noam-10.csv')
    data = data.drop(columns=['Unnamed: 0'])
    labels = data.pop('label')
    data = data.to_numpy()
    labels = labels.iloc[:].values
    
    # Print unique labels counts
    unique, counts = np.unique(labels, return_counts=True)
    with open('tdl/label_counts.txt', 'w') as f:
        print(dict(zip(unique, counts)), file=f)

    new_data = []
    for tdl, label in zip(data, labels):
        # Filter out even labels if target is browser/os
        if target in ['browser', 'os']:
            if label % 2 == 0:
                continue
        new_data.append([ast.literal_eval(j) for j in tdl])

    if target == 'pqc':
        labels = np.array([i % 2 for i in labels])
    elif target == 'browser':
        labels = np.array([((i // 10) % 10) * 10 for i in labels if i % 2 == 1])
    elif target == 'os':
        labels = np.array([(i // 100) * 100 for i in labels if i % 2 == 1])
    elif target == 'all':
        labels = np.array([i for i in labels])

    data = np.array(new_data)
    data = np.array([i.flatten() for i in data])
    return data, labels
    
def tree_to_code(tree_model, feature_names):
    """Helper function to print code-like logic of a decision tree."""
    tree_ = tree_model.tree_
    feature_name = [
        feature_names[i] if i != _tree.TREE_UNDEFINED else "undefined!"
        for i in tree_.feature
    ]
    print(f"def tree({', '.join(feature_names)}):")

    def recurse(node, depth):
        indent = "  " * depth
        if tree_.feature[node] != _tree.TREE_UNDEFINED:
            name = feature_name[node]
            threshold = tree_.threshold[node]
            print(f"{indent}if {name} <= {threshold}:")
            recurse(tree_.children_left[node], depth + 1)
            print(f"{indent}else:  # if {name} > {threshold}")
            recurse(tree_.children_right[node], depth + 1)
        else:
            print(f"{indent}return {np.argmax(tree_.value[node])}")

    recurse(0, 1)
    
def run_models(target: str) -> None:
    """Train multiple models, collect metrics, and store them in one file."""

    data, labels = load_data(target)
    print(f'{labels = }')
    labels = LabelEncoder().fit_transform(labels)
    print(f'{labels = }')
    
    # Load the dictionary of models
    loaded_models = load(f"ech_eylon_10_{target}_trained_models.joblib")  # or "browser_trained_models.joblib", etc.

    # Suppose you want to predict with the Random Forest on new data `X_new`:

    model_names = [
        'Random Forest',
        'XGBoost',
        'Logistic Regression',
        'KNN',
        'Decision Tree',
        'MLP',
        'Naive Bayes',
        'AdaBoost',
        'Gradient Boosting'
    ]
    
    feature_list = []
    for i in range(20):
        feature_list.append(f'packet[{i + 1}].direction')
        feature_list.append(f'packet[{i + 1}].length')
        feature_list.append(f'packet[{i + 1}].time')

    preds = []
    for model in model_names:
        loaded_model = loaded_models[model]
        predictions = loaded_model.predict(data)
        acc = accuracy_score(labels, predictions)
        print(predictions)
        print(acc)
        if model == 'Decision Tree':
                tree.plot_tree(loaded_model, feature_names=feature_list)
                tree_to_code(loaded_model, feature_list)
                plt.show()
        preds.append(f'{model} : {acc}')
        
    print(preds)



def main():
    targets = ['pqc', 'browser', 'all']
    for target in targets:
        run_models(target)


if __name__ == '__main__':
    main()
