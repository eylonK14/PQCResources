# PQC Resources

This repository contains research resources and tools for Post-Quantum Cryptography (PQC) network traffic analysis and classification. The project focuses on machine learning-based detection and classification of PQC vs non-PQC network communications through packet analysis.

## Directory Structure

### Code

Contains the core machine learning implementation and analysis tools for PQC traffic classification.

#### model code

Machine learning models and analysis scripts for PQC detection. **For getting started with classification, use `journal_classification.ipynb` as the preferred entry point.**

- **journal_classification.ipynb**: **[RECOMMENDED]** Jupyter notebook for journal-quality classification analysis and visualization - the preferred starting point for PQC traffic classification
- **feature_importance.py**: Enhanced feature importance analysis with packet-aware algorithms, supporting multiple classifiers (Random Forest, XGBoost, etc.)
- **lstm.ipynb**: LSTM-based deep learning approach for sequential packet analysis
- **model-saver.py**: Model persistence and evaluation framework with comprehensive metrics and cross-validation

#### tdl

Time-Direction-Length (TDL) feature extraction framework using NFStream.

- **TDL.py**: Core NFStream plugin for extracting packet timing, direction, and length features from network flows
- **tdl_runner.py** & **tdl_runner_from_10.py**: Execution scripts for TDL feature extraction with different configurations
- **zero_thy_col.py**: Data preprocessing utility for handling zero/null columns in extracted features

#### useful scripts

Utility scripts and helper functions for data processing and analysis.
Contains additional resources and statistical analysis tools to support the main classification pipeline.

### Docker Dataset

Containerized network capture data with labeled PQC and non-PQC sessions.

- **pqc-paob-docker-20.csv**: Pre-processed dataset with TDL features (600 samples, 20 packets per flow)
- **110/, 111/, 112/**: Network captures from non-PQC sessions (traditional cryptography)
- **120/, 121/, 122/**: Network captures from PQC-enabled sessions
- **Classification Results/**: Model predictions and performance metrics on the docker dataset

### PQBench Resources

Comprehensive benchmarking datasets and results from various PQC algorithm evaluations.

#### Main Dataset Files

- **pqc-algo*.csv**: Algorithm-focused datasets comparing different PQC implementations (Kyber, MLKEM, etc.)
- **pqc-pob*.csv**: Proof-of-Breach datasets with varying packet counts (1, 5, 10, 15, 20, 25, 30 packets)
- **algo-*.csv** & **pob-*.csv**: Specialized datasets for different experimental configurations

#### Organized Results

- **AccuracyToPacketsPOB/**: Performance analysis showing accuracy vs number of packets used for classification
- **Algo Results/** & **POB Results/**: Comprehensive classification results for algorithm and proof-of-breach experiments
- **Algosep/** & **POBsep/**: Separated datasets by algorithm type (Kyber/MLKEM) and PQC presence (PQC/NOPQC)

#### Experimental Data

- **POB-Cyprus/** & **POB-ICC/**: Geographically distributed network captures for generalization testing
- **PQC-Algo/** & **PQC-POB/**: Raw network captures organized by experimental conditions (2XX, 3XX, 4XX session types)

### PQBench Writing Material

Academic publication materials and documentation for the PQBench research project.
Contains cover letters, research papers, and supplementary materials for journal submissions and conference presentations.

## Research Focus

This project investigates machine learning approaches for detecting Post-Quantum Cryptography in network traffic through:

1. **Feature Engineering**: Time-Direction-Length (TDL) analysis of packet sequences
2. **Classification**: Multiple ML algorithms (Random Forest, XGBoost, LSTM) for PQC detection
3. **Benchmarking**: Comprehensive evaluation across different PQC algorithms and network conditions
4. **Scalability**: Analysis of classification accuracy vs number of packets required

## Data Format

Network flows are represented as sequences of `[direction, length, time]` tuples, where:

- **direction**: 0/1 indicating packet flow direction
- **length**: Packet size in bytes  
- **time**: Relative timestamp from flow start

Labels indicate session types (110-122 for different PQC/non-PQC configurations).
