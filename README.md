#  Machine Failure Prediction

An end-to-end Machine Learning project for predicting **machine failures** using industrial sensor and operational data.

The project covers the complete ML workflow, including preprocessing, model comparison, hyperparameter tuning, threshold analysis, evaluation, and model export.

##  Project Objective

Machine failures can lead to production downtime, maintenance costs, and operational losses.

The goal of this project is to build a reliable binary classification model that predicts whether a machine is likely to experience a failure.

Because the dataset is highly imbalanced, **PR-AUC (Precision-Recall AUC)** was used as the primary metric for model selection.

---

##  Final Model

After comparing multiple classification algorithms, **Random Forest** achieved the best PR-AUC and was selected as the champion model.

| Metric    |       Score |
| --------- | ----------: |
| Accuracy  |  **99.80%** |
| Precision | **100.00%** |
| Recall    |  **94.03%** |
| F1-Score  |  **96.92%** |
| ROC-AUC   |  **99.08%** |
| PR-AUC    |  **97.73%** |

### Why PR-AUC?

The dataset contains a strong class imbalance:

* Training samples: **7,964**
* Testing samples: **1,992**
* Positive training cases: **266**
* Positive testing cases: **67**

Because failure cases represent a small portion of the dataset, accuracy alone can be misleading. Therefore, PR-AUC was prioritized during model selection.

---

##  Models Compared

The following models were evaluated:

* Dummy Classifier
* Logistic Regression
* Decision Tree
* Random Forest
* Support Vector Machine (SVM)

| Model               | Accuracy | Precision | Recall | F1-Score | ROC-AUC |     PR-AUC |
| ------------------- | -------: | --------: | -----: | -------: | ------: | ---------: |
|  Random Forest    |   99.80% |   100.00% | 94.03% |   96.92% |  99.08% | **97.73%** |
| SVM                 |   99.80% |   100.00% | 94.03% |   96.92% |  98.41% |     95.83% |
| Logistic Regression |   99.80% |   100.00% | 94.03% |   96.92% |  98.41% |     95.63% |
| Decision Tree       |   99.65% |    94.12% | 95.52% |   94.81% |  97.66% |     90.05% |
| Dummy Classifier    |   96.64% |     0.00% |  0.00% |    0.00% |  50.00% |      3.36% |

---

##  Machine Learning Pipeline

```text
Raw Dataset
     ↓
Data Inspection
     ↓
Data Preprocessing
     ↓
Missing Value Imputation
     ↓
Train / Test Split
     ↓
Baseline Model
     ↓
Model Comparison
     ↓
Random Forest Hyperparameter Tuning
     ↓
Threshold Analysis
     ↓
Final Evaluation
     ↓
Champion Model Selection
     ↓
Model Export
     ↓
Interactive Dashboard
```

---

##  Interactive Dashboard

An interactive Plotly dashboard was created to visualize:

* Machine failure distribution
* Model PR-AUC comparison
* Model accuracy comparison
* F1-score comparison
* Recall comparison
* Champion model performance

Dashboard file:

`machine_failure_dashboard.html`

---

##  Saved Model

The final Random Forest model was exported using Joblib:

`machine_failure_random_forest.pkl`

The exported model can be loaded later for inference without retraining.

---

##  Technologies

* Python
* Pandas
* NumPy
* Scikit-learn
* Plotly
* Matplotlib
* Joblib
* Jupyter Notebook
* Kaggle

---

##  Project Structure

```text
machine-failure-prediction/
│
├── machine_failure_prediction.ipynb
├── machine_failure_random_forest.pkl
├── machine_failure_dashboard.html
└── README.md
```

---

##  Key Takeaways

* **Random Forest** achieved the strongest overall performance.
* **PR-AUC** was prioritized because of the severe class imbalance.
* Hyperparameter tuning was compared against the baseline model rather than automatically replacing it.
* Threshold optimization was evaluated using out-of-fold predictions.
* The optimized threshold did not improve the final test predictions, so the standard **0.50 threshold** was retained.
* The trained model and interactive dashboard were exported for reuse.

---

##  Kaggle Notebook

The complete project notebook is available on Kaggle:

**Kaggle:** https://www.kaggle.com/code/arbhdmoa22/notebook6aa3867377

---

##  Project Status

**Completed — Machine Learning Pipeline Ready for Use **
