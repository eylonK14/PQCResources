#!/usr/bin/env python3

import ast
import numpy as np
import pandas as pd
import argparse
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.metrics import classification_report
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder
import warnings
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.tree import _tree


warnings.filterwarnings("ignore")


def stoopid_map(x: int) -> int:
    if x == 10:
        return 0
    elif x == 11 or x == 12:
        return 1
    return -1


def load_data(target: str, file: int) -> tuple:
    data = pd.read_csv(f'tdl/data_wo_ch.csv')
    data = data.drop(columns=['Unnamed: 0'])
    labels = data.pop('label')
    data = data.to_numpy()
    labels = labels.iloc[:].values
    # print unique labels and their counts
    unique, counts = np.unique(labels, return_counts=True)
    print(dict(zip(unique, counts)))

    with open('tdl/label_counts.txt', 'w') as f:
        print(dict(zip(unique, counts)), file=f)
    new_data = []
    print(f'{target = }')
    for tdl, label in zip(data, labels):
        print(f'{tdl = }')
        if target == 'browser' and label % 2 == 0:
            continue
    #     if target == '512ec' and not (10 <= label % 100 < 20):
    #         continue
    #     if target == '512type' and not (10 < label % 100 < 20):
    #         continue
    #     if target == '768type' and not (20 < label % 100 < 30):
    #         continue
        new_data.append([ast.literal_eval(j) for j in tdl])
    if target == 'browser':
        labels = np.array([((i // 10) % 10) * 10 for i in labels if i % 2 == 1])
    # if target == 'kem':
    #     labels = np.array([((i // 10) % 10) * 10 for i in labels])
    # elif target == '512ec':
    #     labels = np.array([stoopid_map(i % 100) for i in labels if 10 <= i % 100 < 20])
    # elif target == '512type':
    #     labels = np.array([i % 100 for i in labels if 10 < i % 100 < 20])
    # elif target == '768type':
    #     labels = np.array([i % 100 for i in labels if 20 < i % 100 < 30])
    elif target == 'all':
        labels = np.array([i for i in labels]) #])
    data = np.array(new_data)
    data = np.array([i.flatten() for i in data])
    return data, labels


def run_and_organize(data, idx, labels, model, model_names, results):
    skf = StratifiedKFold(n_splits=10)
    accuracy = cross_val_score(model, data, labels, cv=skf, scoring='accuracy')
    precision = cross_val_score(model, data, labels, cv=skf, scoring='precision_weighted')
    recall = cross_val_score(model, data, labels, cv=skf, scoring='recall_weighted')
    f1 = cross_val_score(model, data, labels, cv=skf, scoring='f1_weighted')
    auc = cross_val_score(model, data, labels, cv=skf, scoring='roc_auc_ovr')
    # 2 decimal places for results
    acc_res = f'{accuracy.mean():.2f} +/- {accuracy.std():.2f}'
    prec_res = f'{precision.mean():.2f} +/- {precision.std():.2f}'
    rec_res = f'{recall.mean():.2f} +/- {recall.std():.2f}'
    f1_res = f'{f1.mean():.2f} +/- {f1.std():.2f}'
    auc_res = f'{auc.mean():.2f} +/- {auc.std():.2f}'
    results.loc[len(results)] = [model_names[idx], acc_res, prec_res, rec_res, f1_res, auc_res]


def run_models(target: str, amount: int) -> None:
    data, labels = load_data(target, amount)
    print(f'{labels = }')
    labels = LabelEncoder().fit_transform(labels)
    print(f'{labels = }')

    x_train, x_test, y_train, y_test = train_test_split(
        data,
        labels,
        test_size=0.1,
        random_state=42,
        stratify=labels
    )
    print(f'{len(x_train)}')
    print(f'{len(x_test)}')
    print(f'{len(y_train)}')
    print(f'{len(x_test)}')
    # random forest
    clf = RandomForestClassifier(n_estimators=1, random_state=0)
    clf.fit(x_train, y_train)
    y_pred = clf.predict(x_test)
    print("Random Forest")
    print(classification_report(y_test, y_pred))

    clf = DecisionTreeClassifier(random_state=0)
    clf.fit(x_train, y_train)
    y_pred = clf.predict(x_test)
    print("Decision Tree")
    print(classification_report(y_test, y_pred))

    # dataframe of results: model, accuracy, precision, recall, f1, auc
    # list 10 classifiers to test, for each use stratified kfold with 10 splits
    # save results to csv
    # use cross_val_score for each classifier
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
    # sort models alphabetically
    model_names, models = zip(*sorted(zip(model_names, models)))

    results = pd.DataFrame(columns=[
        'Model',
        'Accuracy',
        'Precision',
        'Recall',
        'F1-score',
        'AUC'
    ])
    for idx, model in enumerate(models):
        run_and_organize(data, idx, labels, model, model_names, results)
    results.to_csv(f'ICC/new-{target}-with-{amount}-packets-results.csv')
    tree.plot_tree(clf)
    tree_to_code(clf, [str(l) for l in labels])
    plt.show()


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


def main() -> None:
    targets = ['browser']
    # packets = [2,3,4]
    packets = [25]
    for target in targets:
        for amount in packets:
            run_models(target, amount)


if __name__ == "__main__":
    main()

