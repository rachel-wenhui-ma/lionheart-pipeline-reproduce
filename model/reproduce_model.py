import numpy as np
import os
import pandas as pd
import csv

from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    roc_auc_score,
    balanced_accuracy_score,
    accuracy_score,
    recall_score,
    f1_score,
    confusion_matrix,
    precision_score,
    roc_curve,
)
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.base import BaseEstimator, TransformerMixin

# Custom modules assumed to be in your local environment
from generalize.model.transformers.pca_by_explained_variance import PCAByExplainedVariance
from generalize.model.utils.weighting import calculate_sample_weight
import joblib
import random

# ---------------------------------------------------------------------
# Set global seed
# ---------------------------------------------------------------------
def set_global_seed(seed):
    """
    Sets the random seed for both numpy and python's random module to ensure reproducibility.
    """
    np.random.seed(seed)
    random.seed(seed)

# ---------------------------------------------------------------------
# Paths / constants
# ---------------------------------------------------------------------
CURRENT_DIR = os.getcwd()
SOURCE_DIR = f"{CURRENT_DIR}/inference_resources_v003"
RESULT_FILE_PATH = f"{CURRENT_DIR}/model/evaluation_score.csv"

# Datasets used for the Leave-One-Dataset-Out (LODO) training phase
TRAIN_FOLDERS = ["Budhraja", "Cristiano", "EndoscopyII", "GECOCA", "Jiang", "Mathios", "MathiosValidation", 
                "NordentoftFrydendahl", "ProstateAarhus"]

# Datasets reserved strictly for external validation
VALIDATION_FOLDER = ["ZhuValidation"]

THRESHOLD_LIST = ["Max. J Threshold", "High Specificity Threshold", "0.5 Threshold"]

# Grid for PCA: retaining 98.7% to 99.7% of variance
PCA_VARIANCE_GRID = np.arange(0.987, 0.998, 0.001)

# Grid for Lasso regularization strength (Inverse of lambda)
C_GRID = [0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1, 0.2]


# ---------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------
def get_features(folders):
    """
    Loads feature arrays and metadata for a list of dataset folders.

    Args:
        folders (list): List of folder names in SOURCE_DIR.

    Returns:
        tuple: (features_all, meta_all) stacked arrays and concatenated dataframe.
    """
    total_features = []
    total_meta = []

    for f in folders:
        base_dir = f"{SOURCE_DIR}/shared_features/{f}"
        feature_dir = f"{base_dir}/feature_dataset.npy"
        meta_dir = f"{base_dir}/meta_data.csv"

        features = np.load(feature_dir)
        meta = pd.read_csv(meta_dir)

        # Track which dataset these samples came from
        meta["dataset"] = f

        # Select Panel 0 only to match the LIONHEART authors' primary model configuration
        features = features[:, 0, :]

        total_features.append(features)
        total_meta.append(meta)

    features_all = np.vstack(total_features)
    meta_all = pd.concat(total_meta, ignore_index=True)

    return features_all, meta_all

# ---------------------------------------------------------------------
# Weighting
# ---------------------------------------------------------------------
def compute_dataset_balanced_weights(y, ds):
    """
    Calculates sample weights to ensure every dataset contributes equally to the loss,
    regardless of its size.

    Args:
        y (array): Target labels.
        ds (array): Dataset identifier for each sample.

    Returns:
        array: Weight for each sample.
    """
    y = np.asarray(y)
    ds = np.asarray(ds)
    w = np.zeros_like(y, dtype=float)

    unique_ds = np.unique(ds)
    total_sample = len(y)
    num_ds = len(unique_ds)

    for d in unique_ds:
        mask_d = (ds == d)
        num_samples = mask_d.sum()

        # Weight = (1 / samples_in_dataset) * (global_scaling_factor)
        w[mask_d] = (1 / num_samples) * (total_sample / num_ds)

    return w


# ---------------------------------------------------------------------
# Preprocessing + model training
# ---------------------------------------------------------------------
def fit_logistic_regression(X_train, y_train, C_value, sample_weight=None, random_state=1):
    """
    Fits a Logistic Regression model with L1 (Lasso) regularization using the SAGA solver.
    """
    lr = LogisticRegression(
        penalty="l1",
        solver="saga",
        max_iter=30000, # High max_iter to ensure convergence with SAGA
        class_weight=None,
        tol=1e-4,
        random_state=random_state,
    )
    lr.set_params(C=C_value)
    lr.fit(X_train, y_train, sample_weight=sample_weight)
    return lr


def train_model(Xtr, ytr, pca_var, C_value, ds_tr=None, seed=1):
    """
    Constructs and trains the full processing pipeline.
    
    Pipeline Steps:
    1. Variance Threshold (remove constants)
    2. Row-wise Scaling (Sample standardization)
    3. Column-wise Scaling (Feature standardization)
    4. PCA (Dimensionality reduction by variance)
    5. PCA Scaling (Standardize components)
    6. Logistic Regression (L1)

    Returns:
        dict: A dictionary containing the trained steps and model parameters.
    """
    # 1) Drop features with 0 variance
    var_thresh = VarianceThreshold(threshold=0.0)
    Xtr_clean = var_thresh.fit_transform(Xtr)

    # 2) Normalize rows first (common in gene/DNA sequencing data)
    row_scaler = StandardScaler()
    Xtr_row_scaled = row_scaler.fit_transform(Xtr_clean.T).T

    # 3) Normalize columns (standard feature scaling)
    col_scaler = StandardScaler()
    Xtr_col_scaled = col_scaler.fit_transform(Xtr_row_scaled)

    # 4) Reduce dimensions while keeping `pca_var` % of variance
    pca = PCAByExplainedVariance(target_variance=pca_var, random_state=seed)
    Xtr_pca = pca.fit_transform(Xtr_col_scaled)

    # 5) Normalize the resulting PCA components
    pca_scaler = StandardScaler()
    Xtr_pca_scaled = pca_scaler.fit_transform(Xtr_pca)

    # 6) Compute weights and train classifier
    if ds_tr is not None:
        sample_weight = compute_dataset_balanced_weights(ytr, ds_tr)
    else:
        sample_weight = None

    lr_model = fit_logistic_regression(Xtr_pca_scaled, ytr, C_value, sample_weight, random_state=seed)

    # Bundle everything into a dictionary to act as a "Model Object"
    model = {
        "var_thresh": var_thresh,
        "col_scaler": col_scaler,
        "pca": pca,
        "pca_scaler": pca_scaler,
        "logistic_regression": lr_model,
        "pca_var": pca_var,
        "C": C_value,
    }
    return model


def apply_model(X, model):
    """
    Transforms raw features X using the trained pipeline components (except the classifier).
    This reproduces the training preprocessing for validation/test sets.
    """
    X_clean = model["var_thresh"].transform(X)
    
    # Row scaling (fit_transform on transpose) is stateless per sample, so we re-fit on the new X
    row_scaler = StandardScaler()
    X_row_scaled = row_scaler.fit_transform(X_clean.T).T
    
    # Apply learned transforms for columns and PCA
    X_col_scaled = model["col_scaler"].transform(X_row_scaled)
    X_pca = model["pca"].transform(X_col_scaled)
    X_pca_scaled = model["pca_scaler"].transform(X_pca)
    return X_pca_scaled


def predict_proba_pos(model, X):
    """
    Orchestrates the pipeline to output probability of Class 1 (Cancer).
    """
    X_proc = apply_model(X, model)
    return model["logistic_regression"].predict_proba(X_proc)[:, 1]


# ---------------------------------------------------------------------
# Threshold helpers
# ---------------------------------------------------------------------
def max_j_threshold(y_true, y_prob):
    """
    Finds the threshold that maximizes Youden's J statistic (Sensitivity + Specificity - 1).
    """
    fpr, tpr, thresholds = roc_curve(y_true, y_prob, pos_label=1)
    J = tpr - fpr
    best_idx = np.argmax(J)
    return thresholds[best_idx]


def pick_high_specificity_threshold(y_true, y_pred_prob, target_spec=0.95):
    """
    Finds a threshold that guarantees at least `target_spec` specificity (default 95%).
    Useful for screening tests where False Positives must be minimized.
    """
    fpr, tpr, thresholds = roc_curve(y_true, y_pred_prob)
    specificity = 1.0 - fpr

    idx = np.where(specificity >= target_spec)[0]
    if len(idx) == 0:
        best_idx = np.argmax(specificity)
    else:
        # Choose the one with best sensitivity among those meeting specificity requirement
        best_idx = idx[np.argmax(tpr[idx])]

    return thresholds[best_idx]


# ---------------------------------------------------------------------
# Hyperparameter tuning (inner CV)
# ---------------------------------------------------------------------
def tune_hyperparameters(X_train, y_train, ds_train=None, seed=1):
    """
    Performs Grid Search for PCA variance and C value.
    
    Strategy:
    - If multiple datasets exist: Uses Leave-One-Dataset-Out (LODO) for the inner loop.
    - Otherwise: Falls back to Repeated Stratified K-Fold.
    
    Returns:
        tuple: (best_pca, best_C, best_score)
    """
    if ds_train is not None:
        ds_train = np.asarray(ds_train)
        unique_ds = np.unique(ds_train)
    else:
        unique_ds = None

    # Fallback CV strategy
    inner_cv = RepeatedStratifiedKFold(
        n_splits=5,
        n_repeats=10,
        random_state=seed,
    )

    results_list = []

    for pca_var in PCA_VARIANCE_GRID:
        for C in C_GRID:            
            scores = []
            
            # 1. Dataset-wise inner CV (Preferred)
            if (ds_train is not None) and (len(unique_ds) >= 2):
                for val_ds in unique_ds:
                    is_val = (ds_train == val_ds)
                    is_tr = ~is_val

                    Xtr_inner, ytr_inner = X_train[is_tr], y_train[is_tr]
                    Xval_inner, yval_inner = X_train[is_val], y_train[is_val]
                    ds_tr_inner = ds_train[is_tr]

                    model = train_model(Xtr_inner, ytr_inner, pca_var, C, ds_tr_inner)
                    yval_prob = predict_proba_pos(model, Xval_inner)

                    thr = max_j_threshold(yval_inner, yval_prob)
                    y_pred = (yval_prob >= thr).astype(int)
                    bal_acc = balanced_accuracy_score(yval_inner, y_pred)
                    scores.append(bal_acc)
            
            # 2. Standard K-Fold (Fallback)
            else:
                for tr_idx, val_idx in inner_cv.split(X_train, y_train):
                    Xtr_inner, ytr_inner = X_train[tr_idx], y_train[tr_idx]
                    Xval_inner, yval_inner = X_train[val_idx], y_train[val_idx]
                    ds_tr_inner = ds_train[tr_idx]

                    model = train_model(Xtr_inner, ytr_inner, pca_var, C, ds_tr_inner)
                    yval_prob = predict_proba_pos(model, Xval_inner)

                    thr = max_j_threshold(yval_inner, yval_prob)
                    y_pred = (yval_prob >= thr).astype(int)
                    bal_acc = balanced_accuracy_score(yval_inner, y_pred)
                    scores.append(bal_acc)

            mean_score = float(np.mean(scores))
            std_score = float(np.std(scores))

            if (ds_train is not None) and (len(unique_ds) >= 2):
                num_folds = len(unique_ds)
            else:
                num_folds = 50
            standard_error = std_score / np.sqrt(num_folds)

            results_list.append({
                'C': C,
                'PCA_Var': pca_var,
                'Mean_Balanced_Accuracy': mean_score,
                'SE': standard_error
                })
            
    result_df = pd.DataFrame(results_list)

    if result_df.empty:
        raise ValueError("Grid Search produced no result")

    print("\nRunning Simplest Model Selection Strategy...")
    final_pca, final_C, final_score = select_simplest_model(
        result_df, 
        main_var='C', 
        other_vars=['PCA_Var']
    )

    return final_pca, final_C, final_score

def select_simplest_model(result_df, main_var, other_vars=None):
    """
    Implements the 'One Standard Error Rule'. 
    It selects the simplest model (lowest C, lowest PCA components) that is within 
    one standard deviation of the absolute best score. Prevents overfitting.
    """
    all_vars = [main_var]
    if other_vars is not None:
        all_vars += other_vars

    # Identify the absolute best performance
    best_ind = result_df['Mean_Balanced_Accuracy'].idxmax()
    best_row = result_df.loc[best_ind]

    best_score = best_row['Mean_Balanced_Accuracy']
    best_se = best_row['SE']

    # Define the "good enough" threshold
    threshold = best_score - best_se
    print(f"Best Score: {best_score:.4f} | SE: {best_se:.4f} | Threshold: {threshold:.4f}")

    # Filter for candidates meeting the threshold
    candidates = result_df[result_df['Mean_Balanced_Accuracy'] >= threshold].copy()

    # Further filter: params must be <= the params of the best score (prefer simplicity)
    for var_name in all_vars:
        limit_val = best_row[var_name]
        candidates = candidates[candidates[var_name] <= limit_val]

    # Sort candidates to find the 'smallest' parameters (simplest model)
    for var_name in reversed(all_vars):
        candidates = candidates.sort_values(
            by=var_name,
            ascending=True,
            kind='stable'
        )
    
    # Pick the winner (first in sorted list)
    if candidates.empty:
        return best_row[other_vars[0]] if other_vars else None, best_row[main_var], best_score
    
    winner = candidates.iloc[0]

    main_val = winner[main_var]
    other_val = winner[other_vars[0]] if other_vars else None

    print(f"Selected: {main_var}={main_val}, {other_vars[0] if other_vars else 'None'}={other_val}")

    return other_val, main_val, winner['Mean_Balanced_Accuracy']


# ---------------------------------------------------------------------
# Nested leave-one-dataset-out CV (outer loop)
# ---------------------------------------------------------------------
def nested_lodo_cv(features, meta, seed=1):
    """
    Runs the Outer CV loop.
    Iterates through each dataset, holding it out as a Test set, while tuning/training 
    on the remaining datasets.
    """
    set_global_seed(seed)

    X_all = features.copy()
    y_all = (meta["Cancer Status"].values == "Cancer").astype(int)
    datasets = meta["dataset"].values

    outer_ds = sorted(np.unique(datasets))
    results = []

    for val_ds in outer_ds:
        # Skip ProstateAarhus as a validation set (per project specific logic)
        if val_ds == "ProstateAarhus":
            continue
        else:
            is_test = (datasets == val_ds)
            is_train = ~is_test

            X_train, y_train = X_all[is_train], y_all[is_train]
            X_test, y_test = X_all[is_test], y_all[is_test]
            ds_train = datasets[is_train]

            print(f"\n[Outer CV] Holding out dataset: {val_ds}")
            
            # Inner Loop: Tune Hyperparameters
            best_pca_var, best_C, best_inner_score = tune_hyperparameters(
                X_train, y_train, ds_train=ds_train, seed=seed
            )

            # Train Final Model for this fold
            model = train_model(X_train, y_train, best_pca_var, best_C, ds_tr=ds_train)
            
            # Evaluation
            y_prob_test = predict_proba_pos(model, X_test)
            auc_test = roc_auc_score(y_test, y_prob_test)
            print(f"{val_ds}: AUC={auc_test:.4f}")

            # Calculate metrics for different thresholds
            for thres_name in THRESHOLD_LIST:
                if thres_name == "Max. J Threshold":
                    threshold = max_j_threshold(y_test, y_prob_test)
                elif thres_name == "High Specificity Threshold":
                    threshold = pick_high_specificity_threshold(y_test, y_prob_test)
                else:
                    threshold = 0.5

                y_pred = (y_prob_test >= threshold).astype(int)

                # Store comprehensive metrics
                results.append(
                    {
                        "validation_dataset": val_ds,
                        "Threshold Version": thres_name,
                        "AUC": auc_test,
                        "Accuracy": accuracy_score(y_test, y_pred),
                        "Balanced Accuracy": balanced_accuracy_score(y_test, y_pred),
                        "F1": f1_score(y_test, y_pred),
                        "Sensitivity": recall_score(y_test, y_pred, pos_label=1),
                        "Specificity": recall_score(y_test, y_pred, pos_label=0),
                        "PPV": precision_score(y_test, y_pred),
                        "NPV": confusion_matrix(y_test, y_pred).ravel()[0] / (confusion_matrix(y_test, y_pred).ravel()[0] + confusion_matrix(y_test, y_pred).ravel()[2]) if (confusion_matrix(y_test, y_pred).ravel()[0] + confusion_matrix(y_test, y_pred).ravel()[2]) > 0 else np.nan,
                        "TP": confusion_matrix(y_test, y_pred).ravel()[3],
                        "FP": confusion_matrix(y_test, y_pred).ravel()[1],
                        "TN": confusion_matrix(y_test, y_pred).ravel()[0],
                        "FN": confusion_matrix(y_test, y_pred).ravel()[2],
                        "Threshold": threshold,
                        "best_pca_var": best_pca_var,
                        "best_C": best_C,
                        "best_inner_bal_acc": best_inner_score,
                        "n_test_samples": len(y_test),
                    }
                )

    result_df = pd.DataFrame(results)
    print("\nPer-dataset results:")
    print(result_df)

    # Weighted Average AUC (weighted by sample size of the validation set)
    unweighted_mean_auc = result_df["AUC"].mean()
    weighted_mean_auc = np.average(
        result_df["AUC"], weights=result_df["n_test_samples"]
    )
    print(f"\nUnweighted mean AUC: {unweighted_mean_auc:.4f}")
    print(f"Weighted mean AUC:   {weighted_mean_auc:.4f}")

    return result_df


# ---------------------------------------------------------------------
# Final model training + validation evaluation
# ---------------------------------------------------------------------
def train_final_model(X_train, y_train, ds_train):
    """
    Trains the final production model on ALL available training data.
    
    Note: Hyperparameter tuning is currently bypassed with hardcoded best values
    (pca_var=0.988, C=0.2) based on previous experiments.
    """
    print("\nTuning hyperparameters on ALL training data...")
    best_pca_var, best_C, best_inner_score = tune_hyperparameters(
        X_train, y_train, ds_train=ds_train, seed=1
    )

    print(f"\nTraining final model with pca_var={best_pca_var}, C={best_C}")
    model = train_model(X_train, y_train, best_pca_var, best_C, ds_tr=ds_train)
    return model


def compute_auc(meta, y_prob):
    y_val = (meta["Cancer Status"].values == "Cancer").astype(int)
    return roc_auc_score(y_val, y_prob)


# ---------------------------------------------------------------------
# Main Execution
# ---------------------------------------------------------------------
if __name__ == "__main__":
    # 1) Load features for the primary training cohort
    X_train_all, meta_train_all = get_features(TRAIN_FOLDERS)
    y_train_all = (meta_train_all["Cancer Status"].values == "Cancer").astype(int)
    ds_train_all = meta_train_all["dataset"].values

    # 2) Run Nested LODO CV to validate performance and reproduce paper metrics
    outer_df = nested_lodo_cv(X_train_all, meta_train_all, seed=1)
    min_auc = outer_df["AUC"].min()
    max_auc = outer_df["AUC"].max()

    print(f"Min AUC: {min_auc}")
    print(f"Max AUC: {max_auc}")

    outer_df.to_csv("model/evaluation_score.csv")


    # 3) Train the definitive model using fixed hyperparameters
    final_model = train_final_model(X_train_all, y_train_all, ds_train_all)
    print("\nFinal trained model:")
    print(final_model)

    # 4) Persist the model to disk
    os.makedirs("model", exist_ok=True)
    joblib.dump(final_model, "model/model.joblib")

    # 5) External Validation: Test on completely unseen data (ZhuValidation)
    print(f"\nEvaluating on validation dataset: {VALIDATION_FOLDER}")
    X_val, meta_val = get_features(VALIDATION_FOLDER)
    y_val_prob = predict_proba_pos(final_model, X_val)
    auc_val = compute_auc(meta_val, y_val_prob)
    print(f"{VALIDATION_FOLDER} ROC AUC: {auc_val:.4f}")

    # 6) Test: Test on hg38 data
    X = np.load(f"{CURRENT_DIR}/model/feature_dataset.npy")
    final_model = joblib.load(f"{CURRENT_DIR}/model/model.joblib")
    prob = predict_proba_pos(final_model, X)[0]
    print(f"P(Cancer) of Reproduced Model + Official Features: {prob: .6f}")
