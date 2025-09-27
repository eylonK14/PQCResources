# Useful Scripts

This directory contains utility scripts and helper functions for data processing, statistical analysis, and model evaluation in the PQC traffic classification project. The scripts are organized into resource management tools and statistical analysis utilities.

## Directory Structure

### resources/

Core utility scripts for data processing, model execution, and file management.

#### Data Processing & Model Execution

- **main-model-runner.py**: Primary model execution script supporting multiple machine learning algorithms (Random Forest, XGBoost, Logistic Regression, SVM, etc.) with cross-validation and classification reporting
- **model-runner.py**: Secondary model execution script for specific experimental configurations and model comparisons
- **modified_rnd_forest.py**: Enhanced Random Forest implementation with custom parameters and feature selection capabilities

#### Model Management

- **model-loader.py**: Utility for loading pre-trained models from disk and applying them to new datasets for inference and evaluation
- **model-saver.py**: Duplicate of the main model persistence framework (also found in model code/) for saving trained models with metadata

#### File & Data Management

- **count_pcaps.py**: PCAP file analyzer that counts packets in network capture files and bins them by packet count ranges (20-30, 30-40, etc.) for dataset organization
- **create_files_compare.py**: File comparison utility for validating dataset consistency and identifying differences between processed files
- **merge_results.py**: Results aggregation script that combines classification results from multiple experiments into consolidated CSV reports
- **similar_flows.py**: Flow similarity analysis tool for identifying and clustering network flows with similar characteristics

#### Data Cleaning & Preprocessing

- **diff-train-test.py**: Training/testing data comparison utility for ensuring proper data split validation and identifying potential data leakage
- **remove_ch.py**: ClientHello packet filtering script for removing or isolating TLS ClientHello packets from network captures
- **zero_thy_col.py**: Data preprocessing utility for handling zero/null columns in feature matrices (duplicate of tdl/ version)

#### Algorithm-Specific Tools

- **kyber_random_forest.py**: Random Forest classifier specifically tuned for Kyber PQC algorithm detection with optimized hyperparameters

#### PowerShell Utilities

- **RenamePcaps.ps1**: PowerShell script for batch renaming PCAP files according to standardized naming conventions
- **RenamePcapsExtent.ps1**: Extended version of the PCAP renaming script with additional file extension handling and validation

#### Interactive Analysis

- **code.ipynb**: Jupyter notebook for interactive data exploration, model testing, and result visualization

#### Data Files

- **pob-mod.csv**: Modified Proof-of-Breach dataset with processed features for specific experimental configurations
- **pob-similar-stats.csv**: Statistical summary of similar flows analysis results from the POB dataset

### statistics/

Statistical analysis and visualization tools for network traffic characterization.

#### Core Statistics Scripts

- **journal_stats.py**: Comprehensive statistical analysis tool for journal publication, computing packet size distributions, inter-arrival times, ClientHello/ServerHello size statistics
- **journal_processed_stats.py**: Post-processing statistical analysis for journal-ready results with publication-quality metrics
- **journal_bins.py**: Binning analysis for journal publication, creating histogram data for packet size and timing distributions

#### Packet Analysis

- **packet_size.py**: Packet size distribution analysis with visualization capabilities for comparing PQC vs non-PQC traffic patterns
- **size_bins.py**: Packet size binning utility for creating categorical features from continuous packet size data
- **count_acks.py**: TCP acknowledgment packet counter for protocol-level traffic analysis
- **large.py**: Large packet identification and analysis tool for detecting oversized packets that may indicate PQC handshakes

#### TLS-Specific Analysis

- **clienthello_split.py**: ClientHello packet extraction and analysis tool for TLS handshake characterization
- **clienthello_mean_dir.py**: Directory-level ClientHello size averaging for bulk dataset processing
- **clienthello_mean_subdir.py**: Subdirectory-level ClientHello analysis for hierarchical dataset organization
- **cilenthello_split_dir.py**: Directory-based ClientHello length analysis with automated processing

#### Directory & Dataset Analysis

- **dir_stats.py**: Directory-level statistical analysis for batch processing of network capture datasets
- **ch_all.py**: Comprehensive ClientHello analysis across all dataset directories

#### Visualization

- **histogram.py**: Histogram generation for various network traffic metrics with customizable binning and styling
- **xgboost_metrics.py**: XGBoost model performance visualization with feature importance plots and confusion matrices
- **sizes_nopqc.png** & **sizes_pqc.png**: Pre-generated packet size distribution visualizations comparing PQC and non-PQC traffic

#### Output & Results

- **output_barcharts/**: Directory containing generated bar chart visualizations for statistical analysis results
- **tdl/**: Subdirectory containing TDL-specific statistical analysis tools and results
- **1**: Temporary or intermediate results file (likely needs cleanup)

## Usage Patterns

### Model Execution Workflow

1. Use `count_pcaps.py` to analyze dataset packet distributions
2. Run `main-model-runner.py` for comprehensive model evaluation
3. Apply `merge_results.py` to consolidate results across experiments
4. Generate visualizations with `histogram.py` and `xgboost_metrics.py`

### Statistical Analysis Pipeline

1. Process directories with `journal_stats.py` for comprehensive metrics
2. Extract TLS-specific features using `clienthello_split.py`
3. Create publication-ready binned data with `journal_bins.py`
4. Generate visualizations using scripts in the statistics/ directory

### Data Preprocessing Chain

1. Rename files systematically with PowerShell scripts
2. Remove unwanted packets using `remove_ch.py`
3. Handle missing data with `zero_thy_col.py`
4. Validate data integrity with `create_files_compare.py`