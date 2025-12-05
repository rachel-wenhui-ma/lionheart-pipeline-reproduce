# Reproducing LIONHEART: A Novel cfDNA-Based Cancer Detection Method

## 1\. Introduction

Cell-free DNA (cfDNA)–based cancer detection is an emerging and rapidly advancing field. LIONHEART, recently accepted by Nature Communications, introduces a pan-cancer detection method that correlates cfDNA fragment coverage with chromatin accessibility across cell types. We reproduced the core LIONHEART pipeline, including feature extraction and classification, and obtained results that closely match the original implementation.

## 2\. Method Summary

The biological principle behind LIONHEART is that cfDNA in blood preserves the chromatin-accessibility patterns of its tissue of origin. Tumor-derived cfDNA therefore, correlates more strongly with cancer-specific chromatin profiles.

LIONHEART computes these correlation patterns across 898 cell types and applies a simple LASSO logistic model to the standardized scores. Despite its simplicity, the method generalizes well and can detect cancer and infer its tissue of origin.

## 3\. Scope of Reproduction

LIONHEART provides a complete end-to-end pipeline. But in this project, we only focused on reproducing its core algorithm components, including feature extraction and assembly, model training, leave-one-dataset-out cross-validation, and prediction using the trained model.

## 4\. Implementation Details

### 4.1 Feature Extraction:

The feature-extraction pipeline consists of a sequence of preprocessing and statistical steps applied to 10bp genome-wide cfDNA coverage. For each sample, the goal is to compute 898 cell-type–specific correlation features, resulting in a final 10 × 898 feature matrix (10 statistical summaries per cell type). 

The reproduction of the feature-extraction pipeline started by understanding the source code and outlining a high-level skeleton of the processing steps. We then filled in the actual computational logic step by step. To debug efficiently, we started with a minimal test case (chr21 \+ one DNase mask) before scaling to all 22 chromosomes and all 898 masks.

To ensure that our reproduced features match the official implementation, we directly reused several core helper functions from the LIONHEART repository for ZIPoisson outlier detection, GC correction, insert-size correction, and megabin normalization, while re-implementing the overall pipeline logic around them.

### 4.2. Machine Learning Model:

A simple LASSO logistic regression was utilized to classify cancerous versus non-cancerous patients. The model was trained on nine LIONHEART feature datasets, validated on one external dataset (named Zhu Validation), and finally tested on the features extracted from hg38.

We reproduced this full modeling pipeline, including all preprocessing, dimension reduction, and hyperparameter-selection steps.

The hyperparameters (i.e., regularization term C and PCA variance) are tuned using the nested leave-one-dataset-out cross-validation strategy. Through a grid search, the inner loop selects the best hyperparameter pair which produces the highest mean balanced accuracy. The chosen hyperparameters are fed into the logistic regression model for training and validation on the “left-out” dataset. In other words, for each iteration, eight datasets were used for tuning and training, and the best model was tested on the remaining dataset.

However, standard grid selection was not sufficient. We replicated the author’s “Simplest Model Refit Strategy”, which relies on the One-Standard-Error rule. Instead of simply picking the model with the highest balanced accuracy, this strategy prioritizes the simplest model (lowest C and PCA variance) with an “acceptable” accuracy, whose performance falls within one standard deviation of the best model. This strategy ensures that the model is not too complex and does not overfit the training samples.

To evaluate accuracy, we employed three different threshold selection strategies, including Max J., high specificity, and the standard 0.5 threshold. Max J. strategy prioritizes a model that balances well between sensitivity and specificity, while a high specificity strategy ensures that false positive results are minimal.

After performing nested cross-validation and confirming that the model generalizes well to unseen cohorts, we trained the final, simplest model on all nine datasets and validated its performance using an external dataset, the Zhu Validation set. 

## 5\. Results and Discussion

### 5.1 Feature Extraction Comparison

When comparing our features and LIONHEART features of the same cfDNA sample, all ten feature types show perfect or near-perfect correlation (≥ 0.99999) between our reproduced implementation and the official LIONHEART outputs. The small absolute deviations in raw statistics do not affect the final model predictions, confirming that our reproduction of the feature-extraction pipeline is highly accurate.

We were only able to validate our feature extraction pipeline using the single cfDNA sample shared in the LIONHEART repository, due to the lack of publicly available data. Although LIONHEART provides extracted features for more than 2,000 samples of 9 datasets, the original cfDNA BAM files are not included, and all the corresponding datasets are not publicly accessible due to patient privacy restrictions.

In terms of computational efficiency, the feature-extraction pipeline is both time-consuming and memory-intensive. Although masks are loaded sequentially, each mask is very large due to the 10bp resolution. In theory, the correlation computations could be parallelized, but parallel loading of multiple masks would exceed the memory capacity of our regular laptop. Therefore, our pipeline ran sequentially for 898 masks, taking over 2 hours. But this is a hardware limitation rather than a limitation of the algorithm.

### 5.2 Model Result Comparison

Our final model selected only 43 active features, as opposed to 78 active features in the reference model. Though there are 45 percent fewer features chosen, our model achieved a comparable Area Under the Curve (AUC) with the paper (89.76 vs. 91.70) for validation set. On test set, we also achieved comparable probability of predicting cancer (90.12 vs. 94.39), suggeting that our simple model is more efficient. We successfully identified the main biological signals.

However, fewer features also come at a cost. Our model misses the fine-grained information needed to classify the “not-too-obvious” cases. This is clearly evident in cross-dataset validation for Nordentoft and GECOCA datasets, which are more noisy and contain low-fraction data. While the reference model maintains performance (AUC: 0.86), our performance dropped significantly to just above 0.6 for both datasets. 

This could be attributed to the fact that the reproduced model selected a different hyperparameters pair \- 0.04 for regularization term and 0.987 for PCA variance, in comparison to the paper’s 0.2 and 0.988 hyperparameters, respectively. The stricter penalty causes the model to be more conservative, skipping out on subtle clues.

To sum up, our reproduced model, though providing comparable final results to the paper, lacks the ability to generalize to new datasets, especially those with hidden and noisy signals. 

## 7\. Conclusion

We successfully reproduced the core computational components of the LIONHEART method, including its feature extraction pipeline and classification model. Our reproduced features matched the official ones with near-perfect agreement, and the model achieved comparable predictive performance. While limited by the availability of raw cfDNA datasets and computational resources, this work provides a validated and accessible reconstruction of the LIONHEART framework.

## 8\. Reference 

\[1\] S. Freese et al., "Cross-dataset pan-cancer detection by correlating cell-free DNA fragment coverage with open chromatin sites across cell types,"   
Nat. Commun., in press, 2024\. \[Online\]. Available: https://www.researchgate.net/publication/397850417. Accessed: Feb. 15, 2025\.

\[2\] S. Freese, “LIONHEART: Pan-cancer detection from cfDNA fragmentomics,” GitHub repository, 2024\. \[Online\]. Available: https://github.com/BesenbacherLab/lionheart. Accessed: Feb. 15, 2025\.

## 9\. AI Usage Acknowledgement

ChatGPT 5.1 and Google Gemini 3 was used for paper reviewing, algorithm interpretation, project planning and pipeline design.  
Cursor, ChatGPT 5.1, and Google Gemini 3 was used for source-code analysis, code implementation, debugging, results comparison and visualization.

## 10\. Code Repository

https://github.com/rachel-wenhui-ma/lionheart-pipeline-reproduce  
