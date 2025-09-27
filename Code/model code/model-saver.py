import ast
import os
import ntpath
import numpy as np
import pandas as pd
import argparse
import warnings
import matplotlib.pyplot as plt
import filecmp

from sklearn.ensemble import (
    RandomForestClassifier,
    AdaBoostClassifier,
    GradientBoostingClassifier,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.metrics import classification_report, f1_score, precision_score, recall_score, accuracy_score, roc_auc_score
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier, _tree
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn import tree

from joblib import dump

warnings.filterwarnings("ignore")


def path_leaf(path: str):
    head, tail = ntpath.split(path)
    return tail or ntpath.basename(head)


def load_data(file_path, target, stnn, combo):
    """original = pd.read_csv(file_path)
    if 'Unnamed: 0' in original.columns:
        original = original.drop(columns=['Unnamed: 0'])
    labels_list = original['label'].tolist()
    if combo:
        tdl_data = original['TRDL'].tolist()
        stnn_data = original['udps.stnn_image_enh'].tolist()
        combined_features = []
        filtered_labels = []
        for tdl_str, stnn_str, lab in zip(tdl_data, stnn_data, labels_list):
            if target in ['browser', 'os'] and lab % 2 == 0:
                continue
            try:
                tdl_nested = ast.literal_eval(tdl_str)
                tdl_features = []
                for item in tdl_nested:
                    inner = ast.literal_eval(item)
                    tdl_features.extend(inner)
            except Exception as e:
                tdl_features = []
                print(f"Error processing TDL features: {e}")
            try:
                stnn_features = list(ast.literal_eval(stnn_str))
            except Exception as e:
                stnn_features = []
                print(f"Error processing STNN features: {e}")
            combined = tdl_features + stnn_features
            combined_features.append(combined)
            filtered_labels.append(lab)
        data_arr = np.array([np.array(x).flatten() for x in combined_features])
        labels_arr = np.array(filtered_labels)
    else:
        if stnn:
            raw_data = original['udps.stnn_image_enh'].tolist()
        else:
            raw_data = original.drop(columns=['label']).values.tolist()
        filtered_features = []
        filtered_labels = []
        for d, lab in zip(raw_data, labels_list):
            if target in ['browser', 'os'] and lab % 2 == 0:
                continue
            if stnn:
                try:
                    processed = list(ast.literal_eval(d))
                except Exception as e:
                    processed = []
                    print(f"Error processing STNN feature: {e}")
                filtered_features.append(processed)
            else:
                filtered_features.append(d)
            filtered_labels.append(lab)
        data_arr = np.array([np.array(x).flatten() for x in filtered_features])
        labels_arr = np.array(filtered_labels)
    if target == 'algo':
        labels_arr = np.array([i % 2 for i in labels_arr])
    elif target == 'pqc':
        labels_arr = np.array([i % 2 for i in labels_arr])
    elif target == 'browser':
        labels_arr = np.array([((i // 10) % 10) * 10 for i in labels_arr])
    elif target == 'os':
        labels_arr = np.array([(i // 100) * 100 for i in labels_arr])
    return data_arr, labels_arr """
    data = pd.read_csv(file_path)
    if 'Unnamed: 0' in data.columns:
        data = data.drop(columns=['Unnamed: 0'])
    labels = data.pop('label')
    data = data.to_numpy()
    labels = labels.iloc[:].values
    
    new_data = []
    filtered_labels = []
    for tdl, label in zip(data, labels):
        if target in ['algo']: #filter firfox
            if label // 10 == 1:
                continue
        new_data.append([ast.literal_eval(j) for j in tdl])
        filtered_labels.append(label)
    
    labels = np.array(filtered_labels)
    
    if target == 'algo':
        labels = np.array([i % 2 for i in labels])
    if target == 'tuple':
        pass # Keep original labels
    
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


def run_and_organize(y_test, idx, y_pred, model_names, results):
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')
    results.loc[len(results)] = [model_names[idx], f'{accuracy:.2f}', f'{precision:.2f}', f'{recall:.2f}', f'{f1:.2f}']


def tree_to_code(tree_model, feature_names, file_path="tree_code.txt"):
    tree_ = tree_model.tree_
    feature_name = [
        feature_names[i] if i != _tree.TREE_UNDEFINED else "undefined!"
        for i in tree_.feature
    ]
    lines = []
    lines.append(f"def tree({', '.join(feature_names)}):")
    def recurse(node, depth):
        indent = "  " * depth
        if tree_.feature[node] != _tree.TREE_UNDEFINED:
            name = feature_name[node]
            threshold = tree_.threshold[node]
            lines.append(f"{indent}if {name} <= {threshold}:")
            recurse(tree_.children_left[node], depth + 1)
            lines.append(f"{indent}else:  # if {name} > {threshold}")
            recurse(tree_.children_right[node], depth + 1)
        else:
            lines.append(f"{indent}return {np.argmax(tree_.value[node])}")
    recurse(0, 1)
    with open(file_path, "w") as f:
        for line in lines:
            f.write(line + "\n")


def run_models(train_path: str, test_path: str, target: str, stnn: bool, combo: bool) -> None:
    train_filename = path_leaf(train_path).removesuffix('.csv')
    test_filename = path_leaf(test_path).removesuffix('.csv')
    
    print(f'train_filename = {train_filename}')
    print(f'test_filename = {test_filename}')
    
    if filecmp.cmp(train_path, test_path, shallow=False):
        print("Train file and test file are equal!")
        data, labels = load_data(train_path, target, stnn, combo)
        if len(np.unique(labels)) < 2:
            print(f"Training data for target '{target}' contains only one class. Skipping training for this target.")
            return
        labels = LabelEncoder().fit_transform(labels)
        x_train, x_test, y_train, y_test = train_test_split(
            data, labels, test_size=0.1, random_state=42, stratify=labels
        )
        unique, counts = np.unique(labels, return_counts=True)
        print(dict(zip(unique, counts)))
    else:
        print("Train file and test file aren't equal!")
        x_train, y_train = load_data(train_path, target, stnn, combo)
        x_test, y_test = load_data(test_path, target, stnn, combo)
        print(f'Train size: {len(x_train)}, Test size: {len(x_test)}')
        le = LabelEncoder()
        le.fit(y_test)
        #y_test = le.transform(y_test)
        #y_train = le.transform(y_train)
    
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
    
    exp_dir = f'{train_filename}_train_{test_filename}_test_exp'
    os.makedirs(exp_dir, exist_ok=True)
    os.makedirs(os.path.join(exp_dir, 'TreePlots'), exist_ok=True)
    os.makedirs(os.path.join(exp_dir, 'Results'), exist_ok=True)
    os.makedirs(os.path.join(exp_dir, 'Functions'), exist_ok=True)
    os.makedirs(os.path.join(exp_dir, 'Models'), exist_ok=True)
    
    trained_models = {}
    predictions = {}
    
    for idx, model in enumerate(models):
        print(model)
        model.fit(x_train, y_train)
        y_pred = model.predict(x_test)
        print(classification_report(y_test, y_pred))
        predictions[idx] = y_pred
        
        # If the model is a Decision Tree, handle the feature names.
        if model_names[idx] == 'Decision Tree':
            if combo:
                # Two separate lists: one for TDL features and one for STNN features.
                tdl_feature_names = [
                    f'packet[{i//3 + 1}].{["direction", "length", "time"][i % 3]}' 
                    for i in range(60)
                ]
                stnn_feature_names = [
                    'src_size_min', 'src_size_max', 'src_size_mean', 'src_size_stddev', 'src_size_skewness',
                    'src_iat_min', 'src_iat_max', 'src_iat_mean', 'src_iat_stddev', 'src_iat_skewness',
                    'src_total_bytes', 'src_total_packets', 'src_bytes_per_sec', 'src_packets_per_sec',
                    'dst_size_min', 'dst_size_max', 'dst_size_mean', 'dst_size_stddev', 'dst_size_skewness',
                    'dst_iat_min', 'dst_iat_max', 'dst_iat_mean', 'dst_iat_stddev', 'dst_iat_skewness',
                    'dst_total_bytes', 'dst_total_packets', 'dst_bytes_per_sec', 'dst_packets_per_sec',
                    'bi_total_bytes', 'bi_total_packets', 'bi_bytes_per_sec', 'bi_packets_per_sec'
                ]
                feature_names = tdl_feature_names + stnn_feature_names
            elif stnn:
                feature_names = [
                    'src_size_min', 'src_size_max', 'src_size_mean', 'src_size_stddev', 'src_size_skewness',
                    'src_iat_min', 'src_iat_max', 'src_iat_mean', 'src_iat_stddev', 'src_iat_skewness',
                    'src_total_bytes', 'src_total_packets', 'src_bytes_per_sec', 'src_packets_per_sec',
                    'dst_size_min', 'dst_size_max', 'dst_size_mean', 'dst_size_stddev', 'dst_size_skewness',
                    'dst_iat_min', 'dst_iat_max', 'dst_iat_mean', 'dst_iat_stddev', 'dst_iat_skewness',
                    'dst_total_bytes', 'dst_total_packets', 'dst_bytes_per_sec', 'dst_packets_per_sec',
                    'bi_total_bytes', 'bi_total_packets', 'bi_bytes_per_sec', 'bi_packets_per_sec'
                ]
            else:
                feature_names = [
                    f'packet[{i//3 + 1}].{["direction", "length", "time"][i % 3]}' 
                    for i in range(60)
                ]
            # Adjust the length of feature_names to match the number of features in the model.
            n_features = model.n_features_in_
            if len(feature_names) < n_features:
                feature_names += [f"feature_{i}" for i in range(len(feature_names), n_features)]
            elif len(feature_names) > n_features:
                feature_names = feature_names[:n_features]
                
            plt.figure(dpi=500)
            tree.plot_tree(model, feature_names=feature_names)
            tree_plot_path = os.path.join(exp_dir, 'TreePlots', f'decision_tree_{target}.png')
            plt.savefig(tree_plot_path)
            func_path = os.path.join(exp_dir, 'Functions', f'decision_tree_{target}_func.txt')
            tree_to_code(model, feature_names, func_path)
            plt.close()
            
        trained_models[model_names[idx]] = model
    
    results = pd.DataFrame(columns=['Model', 'Accuracy', 'Precision', 'Recall', 'F1-score'])
    for idx, model in enumerate(models):
        run_and_organize(y_test, idx, predictions[idx], model_names, results)
    
    results_path = os.path.join(exp_dir, 'Results', f'{target}.csv')
    results.to_csv(results_path, index=False)
    print(f"Results saved to {results_path}")
    
    models_path = os.path.join(exp_dir, 'Models', f'{target}_trained_models.joblib')
    dump(trained_models, models_path)
    print(f"All trained models for '{target}' saved to {models_path}")


def main():
    parser = argparse.ArgumentParser(description='Train and test for datasets (could be the same)')
    parser.add_argument('--train-path', required=True, help='Path to training CSV (e.g., multi-location-experiment/noam/combined.csv)')
    parser.add_argument('--test-path', required=True, help='Path to testing CSV (e.g., multi-location-experiment/eylon/combined.csv)')
    parser.add_argument('--algo', required=True, help='Use for Kyber and MLKEM classification', action=argparse.BooleanOptionalAction)
    parser.add_argument('--stnn', required=True, help='Use for STNN features', action=argparse.BooleanOptionalAction)
    parser.add_argument('--combo', required=True, help='Enable combo mode (combine TDL and STNN features)', action=argparse.BooleanOptionalAction)
    
    args = parser.parse_args()
    
    if args.algo:
        for target in ['algo', 'tuple']:
            run_models(args.train_path, args.test_path, target, args.stnn, args.combo)
    else:
        for target in ['pqc', 'os','browser', 'all']:
            run_models(args.train_path, args.test_path, target, args.stnn, args.combo)


if __name__ == '__main__':
    main()
