# Enhanced version with packet-aware feature importance analysis

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import ast
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import (
    AdaBoostClassifier,
    GradientBoostingClassifier,
    RandomForestClassifier
)
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from sklearn.inspection import permutation_importance

warnings.filterwarnings("ignore")

def create_feature_names():
    """
    Create meaningful feature names for packet features.
    60 features total: direction, length, time for 20 packets
    """
    feature_names = []
    for packet_num in range(1, 21):  # 20 packets
        feature_names.extend([
            f'packet_{packet_num}_direction',
            f'packet_{packet_num}_length', 
            f'packet_{packet_num}_time'
        ])
    return feature_names

def load_data(target: str, csv_path: str) -> tuple:
    """
    Load and preprocess data from a CSV file.
    """
    try:
        data = pd.read_csv(csv_path)
    except Exception as e:
        print(f"Error loading CSV at '{csv_path}': {e}")
        return None, None

    if 'Unnamed: 0' in data.columns:
        data = data.drop(columns=['Unnamed: 0'])

    labels = data.pop('label').values

    # Log label counts
    unique, counts = np.unique(labels, return_counts=True)
    print("Label distribution:", dict(zip(unique, counts)))

    rows = data.to_numpy()
    filtered_rows, filtered_labels = [], []
    for row, label in zip(rows, labels):
        if target in ['browser', 'os'] and label % 2 == 0:
            continue
        filtered_rows.append([ast.literal_eval(cell) for cell in row])
        filtered_labels.append(label)

    X = np.array([np.array(r).flatten() for r in filtered_rows])
    y = np.array(filtered_labels)
    print(f"Total samples: {len(y)}")

    # Process labels based on target
    if target == 'algo':
        y_proc = y // 10
    elif target == 'tuple':
        y_proc = y.copy()
    else:
        base = y
        if target == 'pqc':
            y_proc = base
        elif target == 'browser':
            y_proc = ((base // 10) % 10) * 10
        elif target == 'os':
            y_proc = (base // 100) * 100
        elif target == 'all':
            y_proc = base
        else:
            print(f"Unknown target: {target}")
            return None, None

    return X, y_proc

def get_feature_importance(model, model_name, X, y):
    """
    Extract feature importance from a trained model.
    """
    importance_dict = {}
    
    # Fit the model on full dataset
    model.fit(X, y)
    
    # Tree-based models have built-in feature importance
    if hasattr(model, 'feature_importances_'):
        importance_dict['built_in'] = model.feature_importances_
    
    # For linear models, use coefficient magnitudes
    elif hasattr(model, 'coef_'):
        if len(model.coef_.shape) > 1:  # Multi-class
            importance_dict['built_in'] = np.mean(np.abs(model.coef_), axis=0)
        else:  # Binary classification
            importance_dict['built_in'] = np.abs(model.coef_[0])
    
    # For models without built-in importance, use permutation importance
    if model_name in ['KNN', 'MLP', 'Naive Bayes'] or 'built_in' not in importance_dict:
        try:
            perm_importance = permutation_importance(model, X, y, n_repeats=10, random_state=42)
            importance_dict['permutation'] = perm_importance.importances_mean
        except:
            importance_dict['permutation'] = np.zeros(X.shape[1])
    
    return importance_dict

def analyze_packet_feature_importance(feature_importances, feature_names):
    """
    Analyze feature importance by packet position and feature type.
    """
    analysis_results = {}
    
    for model_name, importance_dict in feature_importances.items():
        # Use built-in importance if available, otherwise use permutation importance
        if 'built_in' in importance_dict:
            importance = importance_dict['built_in']
        else:
            importance = importance_dict['permutation']
        
        # Create DataFrame for easier analysis
        df = pd.DataFrame({
            'feature_name': feature_names,
            'importance': importance
        })
        
        # Extract packet number and feature type
        df['packet_num'] = df['feature_name'].str.extract(r'packet_(\d+)_').astype(int)
        df['feature_type'] = df['feature_name'].str.extract(r'packet_\d+_(.+)')
        
        # Group by feature type
        type_importance = df.groupby('feature_type')['importance'].agg(['mean', 'std', 'sum'])
        
        # Group by packet position
        packet_importance = df.groupby('packet_num')['importance'].agg(['mean', 'std', 'sum'])
        
        # Find most important packets for each feature type
        direction_packets = df[df['feature_type'] == 'direction'].nlargest(5, 'importance')
        length_packets = df[df['feature_type'] == 'length'].nlargest(5, 'importance')
        time_packets = df[df['feature_type'] == 'time'].nlargest(5, 'importance')
        
        analysis_results[model_name] = {
            'type_importance': type_importance,
            'packet_importance': packet_importance,
            'top_direction': direction_packets,
            'top_length': length_packets,
            'top_time': time_packets,
            'full_df': df
        }
    
    return analysis_results

def plot_comprehensive_feature_importance(feature_importances, feature_names, top_n=15):
    """
    Create comprehensive visualizations for packet feature importance.
    """
    n_models = len(feature_importances)
    
    # Create subplots
    fig = plt.figure(figsize=(20, 6 * n_models))
    
    for idx, (model_name, importance_dict) in enumerate(feature_importances.items()):
        # Use built-in importance if available, otherwise use permutation importance
        if 'built_in' in importance_dict:
            importance = importance_dict['built_in']
            title_suffix = "(Built-in Importance)"
        else:
            importance = importance_dict['permutation']
            title_suffix = "(Permutation Importance)"
        
        # Create DataFrame for easier plotting
        df = pd.DataFrame({
            'feature_name': feature_names,
            'importance': importance
        })
        df['packet_num'] = df['feature_name'].str.extract(r'packet_(\d+)_').astype(int)
        df['feature_type'] = df['feature_name'].str.extract(r'packet_\d+_(.+)')
        
        # Plot 1: Top N individual features
        plt.subplot(n_models, 3, idx * 3 + 1)
        top_features = df.nlargest(top_n, 'importance')
        colors = ['red' if 'direction' in name else 'blue' if 'length' in name else 'green' 
                 for name in top_features['feature_name']]
        
        plt.barh(range(len(top_features)), top_features['importance'], color=colors)
        plt.yticks(range(len(top_features)), 
                  [f"P{name.split('_')[1]} {name.split('_')[2]}" for name in top_features['feature_name']])
        plt.xlabel('Importance')
        plt.title(f'{model_name} - Top {top_n} Features {title_suffix}')
        plt.gca().invert_yaxis()
        
        # Plot 2: Importance by feature type
        plt.subplot(n_models, 3, idx * 3 + 2)
        type_importance = df.groupby('feature_type')['importance'].sum()
        colors = ['red', 'blue', 'green']
        plt.bar(type_importance.index, type_importance.values, color=colors)
        plt.xlabel('Feature Type')
        plt.ylabel('Total Importance')
        plt.title(f'{model_name} - Importance by Feature Type')
        plt.xticks(rotation=45)
        
        # Plot 3: Importance by packet position
        plt.subplot(n_models, 3, idx * 3 + 3)
        packet_importance = df.groupby('packet_num')['importance'].sum()
        plt.plot(packet_importance.index, packet_importance.values, 'o-', linewidth=2, markersize=4)
        plt.xlabel('Packet Position')
        plt.ylabel('Total Importance')
        plt.title(f'{model_name} - Importance by Packet Position')
        plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

def plot_feature_type_heatmap(analysis_results):
    """
    Create a heatmap showing feature type importance across all models.
    """
    # Prepare data for heatmap
    heatmap_data = []
    model_names = []
    
    for model_name, results in analysis_results.items():
        type_importance = results['type_importance']['sum']
        heatmap_data.append([
            type_importance.get('direction', 0),
            type_importance.get('length', 0), 
            type_importance.get('time', 0)
        ])
        model_names.append(model_name)
    
    # Create heatmap
    plt.figure(figsize=(8, 6))
    sns.heatmap(heatmap_data, 
                xticklabels=['Direction', 'Length', 'Time'],
                yticklabels=model_names,
                annot=True, 
                fmt='.3f',
                cmap='YlOrRd')
    plt.title('Feature Type Importance Across Models')
    plt.tight_layout()
    plt.show()

def save_packet_analysis_to_csv(analysis_results, filename='packet_feature_analysis.csv'):
    """
    Save detailed packet feature analysis to CSV.
    """
    all_results = []
    
    for model_name, results in analysis_results.items():
        df = results['full_df'].copy()
        df['model'] = model_name
        all_results.append(df)
    
    # Combine all results
    combined_df = pd.concat(all_results, ignore_index=True)
    
    # Save to CSV
    combined_df.to_csv(filename, index=False)
    print(f"Detailed packet feature analysis saved to {filename}")
    
    # Also save summary statistics
    summary_file = filename.replace('.csv', '_summary.csv')
    summary_data = []
    
    for model_name, results in analysis_results.items():
        type_importance = results['type_importance']
        for feature_type in ['direction', 'length', 'time']:
            if feature_type in type_importance.index:
                summary_data.append({
                    'model': model_name,
                    'feature_type': feature_type,
                    'mean_importance': type_importance.loc[feature_type, 'mean'],
                    'std_importance': type_importance.loc[feature_type, 'std'],
                    'total_importance': type_importance.loc[feature_type, 'sum']
                })
    
    summary_df = pd.DataFrame(summary_data)
    summary_df.to_csv(summary_file, index=False)
    print(f"Summary statistics saved to {summary_file}")

def run_and_organize_with_importance(data, idx, labels, model, model_names, results, feature_importances):
    """
    Performs stratified 10-fold cross-validation and extracts feature importance.
    """
    skf = StratifiedKFold(n_splits=10)
    accuracy = cross_val_score(model, data, labels, cv=skf, scoring='accuracy')
    precision = cross_val_score(model, data, labels, cv=skf, scoring='precision_weighted')
    recall = cross_val_score(model, data, labels, cv=skf, scoring='recall_weighted')
    f1 = cross_val_score(model, data, labels, cv=skf, scoring='f1_weighted')
    auc = cross_val_score(model, data, labels, cv=skf, scoring='roc_auc_ovr')

    acc_res = f"{accuracy.mean():.2f} +/- {accuracy.std():.2f}"
    prec_res = f"{precision.mean():.2f} +/- {precision.std():.2f}"
    rec_res = f"{recall.mean():.2f} +/- {recall.std():.2f}"
    f1_res = f"{f1.mean():.2f} +/- {f1.std():.2f}"
    auc_res = f"{auc.mean():.2f} +/- {auc.std():.2f}"

    results.loc[len(results)] = [
        model_names[idx], acc_res, prec_res, rec_res, f1_res, auc_res
    ]
    
    # Get feature importance
    model_name = model_names[idx]
    importance_dict = get_feature_importance(model, model_name, data, labels)
    feature_importances[model_name] = importance_dict
    
    print(f"Completed {model_name}")

def run_models_with_packet_analysis(target: str, csv_path: str) -> tuple:
    """
    Runs multiple classification models and provides packet-aware feature importance analysis.
    """
    # Load and encode data
    data, labels = load_data(target, csv_path)
    labels = LabelEncoder().fit_transform(labels)
    
    # Create feature names
    feature_names = create_feature_names()

    # Estimators and matching names
    models = [
        RandomForestClassifier(random_state=42),
        XGBClassifier(random_state=42),
        LogisticRegression(random_state=42, max_iter=1000),
        KNeighborsClassifier(),
        DecisionTreeClassifier(random_state=42),
        MLPClassifier(random_state=42, max_iter=1000),
        GaussianNB(),
        AdaBoostClassifier(random_state=42),
        GradientBoostingClassifier(random_state=42)
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

    # Sort models and names
    model_names, models = zip(*sorted(zip(model_names, models)))

    # Prepare results storage
    results = pd.DataFrame(
        columns=['Model', 'Accuracy', 'Precision', 'Recall', 'F1-score', 'AUC']
    )
    
    # Dictionary to store feature importances
    feature_importances = {}

    # Run each model
    for idx, model in enumerate(models):
        run_and_organize_with_importance(data, idx, labels, model, model_names, results, feature_importances)

    # Save model performance results
    results.to_csv(f'pqc-algo-multi-res.csv')
    print("Model performance results saved to pqc-algo-multi-res.csv")
    
    # Analyze packet feature importance
    analysis_results = analyze_packet_feature_importance(feature_importances, feature_names)
    
    # Save packet analysis
    save_packet_analysis_to_csv(analysis_results, 'packet_feature_analysis.csv')
    
    # Create visualizations
    plot_comprehensive_feature_importance(feature_importances, feature_names)
    plot_feature_type_heatmap(analysis_results)
    
    # Print summary insights
    print("\n" + "="*50)
    print("PACKET FEATURE IMPORTANCE SUMMARY")
    print("="*50)
    
    for model_name, results_data in analysis_results.items():
        print(f"\n{model_name}:")
        type_importance = results_data['type_importance']['sum']
        print(f"  Direction total importance: {type_importance.get('direction', 0):.3f}")
        print(f"  Length total importance: {type_importance.get('length', 0):.3f}")
        print(f"  Time total importance: {type_importance.get('time', 0):.3f}")
        
        # Top 5 individual features overall
        full_df = results_data['full_df']
        top_features = full_df.nlargest(5, 'importance')
        print(f"  Top 5 most important features:")
        for i, (_, row) in enumerate(top_features.iterrows(), 1):
            print(f"    {i}. {row['feature_name']} (importance: {row['importance']:.3f})")
        
        # Top 3 features by type
        print(f"  Top 3 direction features:")
        top_direction = results_data['top_direction'].head(3)
        for i, (_, row) in enumerate(top_direction.iterrows(), 1):
            print(f"    {i}. {row['feature_name']} (importance: {row['importance']:.3f})")
            
        print(f"  Top 3 length features:")
        top_length = results_data['top_length'].head(3)
        for i, (_, row) in enumerate(top_length.iterrows(), 1):
            print(f"    {i}. {row['feature_name']} (importance: {row['importance']:.3f})")
            
        print(f"  Top 3 time features:")
        top_time = results_data['top_time'].head(3)
        for i, (_, row) in enumerate(top_time.iterrows(), 1):
            print(f"    {i}. {row['feature_name']} (importance: {row['importance']:.3f})")
    
    return results, feature_importances, analysis_results

# Example usage:
if __name__ == "__main__":
    target = 'all'  # Choose from: 'pqc', 'algo', 'tuple', 'all', 'browser', 'os'
    csv_path = 'C:\\Users\\Eylon\\PQC\\JournalDatasets\\pqc-pob.csv'
    
    results, feature_importances, analysis_results = run_models_with_packet_analysis(target, csv_path)
    
    # Access specific analysis results:
    # print("Random Forest direction packet importance:", analysis_results['Random Forest']['top_direction'])