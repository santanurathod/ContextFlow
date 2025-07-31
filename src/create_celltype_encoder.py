import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import anndata as ad
import scanpy as sc
import os
import squidpy as sq
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import classification_report, accuracy_score
import xgboost as xgb
import catboost
import json
import pickle

from tqdm import tqdm
import sys
sys.path.append('/Users/rssantanu/Desktop/codebase/constrained_FM')


import argparse

parser = argparse.ArgumentParser()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train XGB model for celltype classification.")
    parser.add_argument("--h5ad_path", type=str, default="GSE232025_stereoseq.h5ad", help="Path to input h5ad file")
    args = parser.parse_args()

    scRNA = ad.read_h5ad(os.path.join('/Users/rssantanu/Desktop/codebase/constrained_FM', 'datasets/h5ad_processed_datasets', args.h5ad_path))
    X= scRNA.obsm['X_pca']
    y= scRNA.obs['celltype'].to_list()

    le = LabelEncoder()
    y_encoded = le.fit_transform(y)  # Now y_encoded is integer labels

    X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.1, random_state=42)

    # Initialize and train the model
    model = xgb.XGBClassifier(
        objective='multi:softmax',
        num_class=len(set(y_encoded)),
        n_estimators=1000,
        learning_rate=0.1,
        max_depth=8,
        random_state=42
    )
    
    model.fit(X_train, y_train)
    print(f"Accuracy for train set:", accuracy_score(y_train, model.predict(X_train)))
    print(f"Accuracy for test set:", accuracy_score(y_test, model.predict(X_test)))

    pickle.dump(model, open(f'datasets/metadata/cell_label_encoder_{args.h5ad_path.split("_")[0]}/multi_class_clf.pkl', 'wb'))
    pickle.dump(le, open(f'datasets/metadata/cell_label_encoder_{args.h5ad_path.split("_")[0]}/label_encoder.pkl', 'wb'))


