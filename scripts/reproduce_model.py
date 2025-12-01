import numpy as np
import os
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, balanced_accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_curve
from generalize.model.transformers.pca_by_explained_variance import PCAByExplainedVariance
import joblib

CURRENT_DIR = os.getcwd()

SOURCE_DIR = f"{CURRENT_DIR}/data"

TRAIN_FOLDERS = ["Budhraja", "Cristiano", "EndoscopyII", "GECOCA", "Jiang", "Mathios", "MathiosValidation", 
                "NordentoftFrydendahl", "ProstateAarhus"]
VALIDATION_FOLDER = ["ZhuValidation"]

# Grid search for the minimum number of PCs that explain 98.7% - 99.7% variance
PCA_VARIANCE_GRID = np.arange(0.987, 0.998, 0.001)

# Grid search for L1-penalized logistic regression
C_GRID = np.logspace(-3, 1, 20)

## 1. Get features
def get_features(folders):
    total_features = []
    total_meta = []
    
    for f in folders:
        base_dir = f"{SOURCE_DIR}/shared_features/{f}"
        feature_dir = f"{base_dir}/feature_dataset.npy"
        meta_dir = f"{base_dir}/meta_data.csv"

        features = np.load(feature_dir)
        meta = pd.read_csv(meta_dir)

        # Add dataset name to meta information
        meta["dataset"] = f

        ## Choose panel 0
        features = features[:, 0, :]

        total_features.append(features)
        total_meta.append(meta)

    features_all = np.vstack(total_features)
    meta_all = pd.concat(total_meta, ignore_index=True)

    return features_all, meta_all

def fit_standardizer(Xtr):
    scaler = StandardScaler()
    scaler_model = scaler.fit(Xtr)
    return scaler_model

def apply_standardizer(X, scaler):
    return scaler.transform(X)

def fit_pca(Xtr_norm, variance_fraction):
    pca = PCAByExplainedVariance(target_variance=variance_fraction)
    X_pca = pca.fit_transform(Xtr_norm)
    return pca, X_pca

def apply_pca(X_norm, pca_model):
    return pca_model.transform(X_norm)

def compute_dataset_balanced_weights(y, ds):
    """
    Compute sample weights so that:
      - within each dataset, each class (0/1) gets total weight 1
      - each dataset contributes the same total weight overall
      - scale total weight to match the number of samples
    """
    y = np.asarray(y)
    ds = np.asarray(ds)
    w = np.zeros_like(y, dtype=float)

    unique_ds = np.unique(ds)
    for d in unique_ds:
        mask_d = (ds == d)
        y_d = y[mask_d]

        n_pos = (y_d == 1).sum()
        n_neg = (y_d == 0).sum()

        if n_pos > 0:
            w[mask_d & (y == 1)] = 1.0 / n_pos
        if n_neg > 0:
            w[mask_d & (y == 0)] = 1.0 / n_neg

    # Rescale w so that the sum matches the number of training samples
    w = w * (len(y) / sum(w))

    return w


def fit_logistic_regression(X_train, y_train, C_value, sample_weight=None):
    """
    Fit L1-regularized (LASSO) logistic regression with class_weight='balanced'.
    """
    logistic_regression = LogisticRegression(
        penalty="l1",
        solver="liblinear",
        max_iter=30000,
        class_weight=None,
        tol = 1e-6,
        random_state=1
        )
    logistic_regression.set_params(C=C_value)
    logistic_regression.fit(X_train, y_train, sample_weight=sample_weight)
    return logistic_regression

def train_model(Xtr, ytr, pca_var, C_value, ds_tr=None):
    # Standardize/Normalize features
    scaler_model = fit_standardizer(Xtr)

    # Apply standardizer
    Xtr_norm = apply_standardizer(Xtr, scaler_model)

    # Fit and apply PCA
    pca_model, X_pca = fit_pca(Xtr_norm, pca_var)

    # Standardize PCA model
    pca_norm_model = fit_standardizer(X_pca)
    X_pca_norm = apply_standardizer(X_pca, pca_norm_model)

    if ds_tr is not None:
        sample_weight = compute_dataset_balanced_weights(ytr, ds_tr)
    else:
        sample_weight = None

    # Fit LASSO Logistic Regression
    lr_model = fit_logistic_regression(X_pca_norm, ytr, C_value, sample_weight)

    model = {
        "scaler_model": scaler_model,
        "pca": pca_model,
        "pca_norm_model": pca_norm_model,
        "logistic_regression": lr_model,
        "pca_var": pca_var,
        "C": C_value,
    }

    return model

def apply_model(X, model):
    # Apply standardizer
    X_norm = apply_standardizer(X, model["scaler_model"])

    # Apply PCA
    X_pca = apply_pca(X_norm, model["pca"])

    # Standardize PCA
    X_pca_norm = apply_standardizer(X_pca, model["pca_norm_model"])

    return X_pca_norm

def predict(model, X):
    """
    Predict probabilities for the positive class.
    """
    return model.predict_proba(X)[:, 1]

def tune_hyperparameters(X_train, y_train, ds_train=None):
    """
    Tune PCA variance fraction and C using inner CV.
    If ds_train is provided and has >= 2 unique datasets, use leave-one-dataset-out CV.
    Otherwise, fall back to 5-fold StratifiedKFold.
    """
    if ds_train is not None:
        ds_train = np.asarray(ds_train)
        unique_ds = np.unique(ds_train)
    else:
        unique_ds = None

    inner_cv = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=1,
    )

    best_score = -np.inf
    best_pca_var = None
    best_C = None

    for pca_var in PCA_VARIANCE_GRID:
        for c in C_GRID:
            scores = []

            # ---- dataset-level inner CV if possible ----
            if (ds_train is not None) and (len(unique_ds) >= 2):
                for val_ds in unique_ds:
                    is_val = (ds_train == val_ds)
                    is_tr = ~is_val

                    Xtr_inner, ytr_inner = X_train[is_tr], y_train[is_tr]
                    Xval_inner, yval_inner = X_train[is_val], y_train[is_val]
                    ds_tr_inner = ds_train[is_tr]

                    # Fit the model using current pca_var and c
                    model = train_model(Xtr_inner, ytr_inner, pca_var, c, ds_tr_inner)

                    Xval_pca = apply_model(Xval_inner, model)

                    # Prediction using fixed predict()
                    yval_prob = predict(model["logistic_regression"], Xval_pca)

                    # Max-J threshold on inner validation
                    fpr, tpr, thresholds = roc_curve(yval_inner, yval_prob, pos_label=1)
                    J = tpr - fpr
                    best_idx = np.argmax(J)
                    thr_best = thresholds[best_idx]

                    y_pred = (yval_prob >= thr_best).astype(int)

                    bal_acc = balanced_accuracy_score(yval_inner, y_pred)
                    scores.append(bal_acc)
            else:
                # ---- fallback: standard StratifiedKFold ----
                for train_i, val_i in inner_cv.split(X_train, y_train):
                    Xtr_inner, ytr_inner = X_train[train_i], y_train[train_i]
                    Xval_inner, yval_inner = X_train[val_i], y_train[val_i]

                    model = train_model(Xtr_inner, ytr_inner, pca_var, c, ds_tr=None)
                    Xval_pca = apply_model(Xval_inner, model)

                    yval_prob = predict(model["logistic_regression"], Xval_pca)
                    
                    # Max-J threshold on inner validation
                    fpr, tpr, thresholds = roc_curve(yval_inner, yval_prob, pos_label=1)
                    J = tpr - fpr
                    best_idx = np.argmax(J)
                    thr_best = thresholds[best_idx]

                    y_pred = (yval_prob >= thr_best).astype(int)

                    bal_acc = balanced_accuracy_score(yval_inner, y_pred)
                    scores.append(bal_acc)

            mean_score = np.mean(scores)

            if mean_score > best_score:
                best_score = mean_score
                best_pca_var = pca_var
                best_C = c
    
    print(f"Best inner CV balanced accuracy: {best_score:.4f}")
    print(f"(pca_var={best_pca_var}, C={best_C})")

    return best_pca_var, best_C, best_score


def nested_lodo_cv(features, meta):
    X_all = features.copy()
    y_all = meta["Cancer Status"].values
    y_all = (y_all == "Cancer").astype(int)

    datasets = meta["dataset"].values

    outer_ds = sorted(np.unique(datasets))

    results = []

    ## OUTER LOOP
    for val_ds in outer_ds:
        # only use the Prostate Aarhus dataset for training
        if val_ds == "ProstateAarhus":
            continue
        else:
            is_test = (datasets == val_ds)
            is_train = ~is_test
            X_train, y_train = X_all[is_train], y_all[is_train]
            X_test, y_test = X_all[is_test], y_all[is_test]
            ds_train = datasets[is_train]

            # Inner tuning
            best_pca_var, best_C, best_inner_score = tune_hyperparameters(X_train, y_train, ds_train=ds_train)

            # Fit model on the outer loop using the inner tuning
            model = train_model(X_train, y_train, best_pca_var, best_C, ds_tr=ds_train)

            # Evaluate on the held-out dataset
            X_test_pca= apply_model(X_test, model)

            proba_test = predict(model["logistic_regression"], X_test_pca)

            auc_test = roc_auc_score(y_test, proba_test)

            # Balanced accuracy at 0.5 threshold
            y_pred_05 = (proba_test >= 0.5).astype(int)

            bal_acc_05 = balanced_accuracy_score(y_test, y_pred_05)

            print(f"{val_ds}: AUC={auc_test:.4f}, BA@0.5={bal_acc_05:.4f}")

            results.append({
                "validation_dataset": val_ds,
                "AUC": auc_test,
                "BA_0_5": bal_acc_05,
                "best_pca_var": best_pca_var,
                "best_C": best_C,
                "n_test_samples": int(is_test.sum()),
                "best_inner_bal_acc": best_inner_score,
            })

    result_df = pd.DataFrame(results)
    print("\nPer-dataset results:")
    print(result_df)

    unweighted_mean_auc = result_df["AUC"].mean()
    weighted_mean_auc = np.average(result_df["AUC"], weights=result_df["n_test_samples"])

    print(f"\nUnweighted mean AUC: {unweighted_mean_auc:.4f}")
    print(f"Weighted mean AUC:   {weighted_mean_auc:.4f}")

    return result_df


def train_final_model(X_train, y_train, ds_train):
    """
    Train final model on all training data, with hyperparameter tuning.
    """
    print("\nTuning hyperparameters on ALL training data...")
    best_pca_var, best_C, best_inner_score = tune_hyperparameters(X_train, y_train, ds_train=ds_train)

    print(f"\nTraining final model with pca_var={best_pca_var}, C={best_C}")
    model = train_model(X_train, y_train, best_pca_var, best_C, ds_tr=ds_train)

    print(f"Final model inner balanced acc: {best_inner_score:.4f}")
    return model


def make_prediction(model, feature_data):
    """
    Make prediction using the pre-trained model
    """
    X_val = feature_data.copy()

    # Standardize PCA
    X_val_pca = apply_model(X_val, model)

    # Probability of prediction
    y_prob = predict(model["logistic_regression"], X_val_pca)

    return y_prob


def compute_performance(meta, y_prob):
    y_val = meta["Cancer Status"].values
    y_val = (y_val == "Cancer").astype(int)

    # Compute AUC score
    auc_val = roc_auc_score(y_val, y_prob)
            
    return auc_val


if __name__ == "__main__":
    # 1) load training datasets (exclude validation sets)
    X_train_all, meta_train_all = get_features(TRAIN_FOLDERS)
    y_train_all = meta_train_all["Cancer Status"].values
    y_train_all = (y_train_all == "Cancer").astype(int)
    ds_train_all = meta_train_all["dataset"].values

    # 2) nested leave-one-dataset-out CV
    # outer_df = nested_lodo_cv(X_train_all, meta_train_all)

    # 3) train final model on all training datasets
    final_model = train_final_model(X_train_all, y_train_all, ds_train_all)

    print(f"\nFinal trained model: ")

    print(final_model)

    # 4) export model to joblib
    filename = 'model/model.joblib'
    joblib.dump(final_model, filename)

    # 5) evaluate on validation datasets
    val_aucs = {}

    print(f"\nEvaluating on validation dataset: {VALIDATION_FOLDER}")
    features_val, meta_val = get_features(VALIDATION_FOLDER)

    y_val_prob = make_prediction(final_model, features_val)

    auc_val = compute_performance(meta_val, y_val_prob)
    
    print(f"{VALIDATION_FOLDER} ROC AUC: {auc_val:.4f}")