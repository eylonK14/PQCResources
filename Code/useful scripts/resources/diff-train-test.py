import ast
import numpy as np
import pandas as pd
import argparse
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder
import warnings
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier
from sklearn import tree
from sklearn.tree import _tree
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

def load_data(file_path, target):
    data = pd.read_csv(file_path)
    if 'Unnamed: 0' in data.columns:
        data = data.drop(columns=['Unnamed: 0'])
    labels = data.pop('label')
    data = data.to_numpy()
    labels = labels.iloc[:].values
    
    new_data = []
    filtered_labels = []
    for tdl, label in zip(data, labels):
        if target in ['browser', 'os']:
            if label % 2 == 0:
                continue
        new_data.append([ast.literal_eval(j) for j in tdl])
        filtered_labels.append(label)
    
    labels = np.array(filtered_labels)
    
    if target == 'pqc':
        labels = np.array([i % 2 for i in labels])
    elif target == 'browser':
        labels = np.array([((i // 10) % 10) * 10 for i in labels])
    elif target == 'os':
        labels = np.array([(i // 100) * 100 for i in labels])
    elif target == 'all':
        pass  # Keep original labels
    
    data = np.array(new_data)
    data = np.array([i.flatten() for i in data])
    return data, labels

def run_and_organize(train_data, train_labels, test_data, test_labels, idx, model, model_names, results):
    model.fit(train_data, train_labels)
    y_pred = model.predict(test_data)
    y_proba = model.predict_proba(test_data) if hasattr(model, 'predict_proba') else None
    
    accuracy = accuracy_score(test_labels, y_pred)
    precision = precision_score(test_labels, y_pred, average='weighted')
    recall = recall_score(test_labels, y_pred, average='weighted')
    f1 = f1_score(test_labels, y_pred, average='weighted')

    if y_proba is not None:
        unique_classes = np.unique(test_labels)
        if len(unique_classes) == 1:
            auc = 'N/A'
        elif len(unique_classes) == 2:
            auc = roc_auc_score(test_labels, y_proba[:, 1])
        else:
            auc = roc_auc_score(test_labels, y_proba, multi_class='ovr', average='weighted')
    else:
        auc = 'N/A'    
    # auc = roc_auc_score(test_labels, y_proba, multi_class='ovr', average='weighted') if y_proba is not None else 'N/A'
    
    acc_res = f'{accuracy:.2f}'
    prec_res = f'{precision:.2f}'
    rec_res = f'{recall:.2f}'
    f1_res = f'{f1:.2f}'
    auc_res = f'{auc:.2f}' if auc != 'N/A' else 'N/A'
    
    results.loc[len(results)] = [model_names[idx], acc_res, prec_res, rec_res, f1_res, auc_res]

def run_models(train_path: str, test_path: str, target: str) -> None:
    train_data, train_labels = load_data(train_path, target)
    test_data, test_labels = load_data(test_path, target)
    
    if len(np.unique(train_labels)) < 2:
        print(f"Training data for target '{target}' contains only one class. Skipping training for this target.")
        return

    # Encode labels based on training data
    le = LabelEncoder()
    le.fit(train_labels)
    train_labels = le.transform(train_labels)
    test_labels = le.transform(test_labels)
    
    models = [
        RandomForestClassifier(),
        XGBClassifier(),
        LogisticRegression(),
        KNeighborsClassifier(),
        DecisionTreeClassifier(),
        MLPClassifier(),
        GaussianNB(),
        AdaBoostClassifier(),
        GradientBoostingClassifier()
    ]
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
    model_names, models = zip(*sorted(zip(model_names, models)))
    
    results = pd.DataFrame(columns=['Model', 'Accuracy', 'Precision', 'Recall', 'F1-score', 'AUC'])
    
    for idx, model in enumerate(models):
        run_and_organize(train_data, train_labels, test_data, test_labels, idx, model, model_names, results)
    
    # Generate output filename based on directories
    train_dir = os.path.basename(os.path.dirname(train_path))
    test_dir = os.path.basename(os.path.dirname(test_path))
    output_file = f'ICC/{train_dir}_train_{test_dir}_test_{target}.csv'
    results.to_csv(output_file)
    
    # Plot Decision Tree
    dt_index = model_names.index('Decision Tree')
    dt_model = models[dt_index]
    dt_model.fit(train_data, train_labels)  # Re-fit to ensure tree is plotted correctly
    
    l = [f'packet[{i//3 +1}].{["direction", "length", "time"][i%3]}' for i in range(90)]
    plt.figure(dpi=500)
    tree.plot_tree(dt_model, feature_names=l) # , filled=True, max_depth=2)
    plt.savefig(f'ICC/decision_tree_{train_dir}_to_{test_dir}_{target}.png')
    tree_to_code(dt_model, l)
    plt.close()

def tree_to_code(tree, feature_names):
    tree_ = tree.tree_
    feature_name = [
        feature_names[i] if i != _tree.TREE_UNDEFINED else "undefined!"
        for i in tree_.feature
    ]
    print(f"def tree({", ".join(feature_names)}):")

    def recurse(node, depth):
        indent = "  " * depth
        if tree_.feature[node] != _tree.TREE_UNDEFINED:
            name = feature_name[node]
            threshold = tree_.threshold[node]
            print (f"{indent}if {name} <= {threshold}:")
            recurse(tree_.children_left[node], depth + 1)
            print (f"{indent}else:  # if {name} > {threshold}")
            recurse(tree_.children_right[node], depth + 1)
        else:
            print (f"{indent}return {np.argmax(tree_.value[node])}")
    recurse(0, 1)

def main():
    parser = argparse.ArgumentParser(description='Train on one dataset and test on another.')
    parser.add_argument('--train-path', required=True, help='Path to training CSV (e.g., multi-location-experiment/noam/combined.csv)')
    parser.add_argument('--test-path', required=True, help='Path to testing CSV (e.g., multi-location-experiment/eylon/combined.csv)')
    # parser.add_argument('--target', required=True, choices=['pqc', 'browser', 'os', 'all'], help='Classification target')
    args = parser.parse_args()
    
    for target in ['pqc', 'browser', 'os', 'all']:
        run_models(args.train_path, args.test_path, target)

if __name__ == '__main__':
    main()
