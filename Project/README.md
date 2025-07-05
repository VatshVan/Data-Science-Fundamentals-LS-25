# Heart Disease Modeling Project

> **Data Science Fundamentals with Python** – July 2025  
> A complete applied machine learning project using the **UCI Heart Disease Dataset**

---

## Overview

This project applies essential Data Science techniques to analyze and model **heart disease** using clinical data. It includes supervised learning, regression, dimensionality reduction and unsupervised clustering — all within a single pipeline.

---

## Dataset

- **Source:** UCI Heart Disease (Cleveland subset)  
- **Instances:** 303  
- **Features:** 13 clinical attributes + 1 target  
- **Target column (`num`)** was binarized:
  - `0` = No heart disease
  - `1-4` = Presence of heart disease (`1`)

---

## Project Tasks

### 1. Exploratory Data Analysis & Preprocessing
- Handled missing values (`ca`, `thal`) via median imputation
- Converted object columns to numeric
- Standardized features using `StandardScaler`

---

### 2. Heart Disease Classification
- **Models Used:**
  - Logistic Regression
  - Random Forest Classifier
- **Evaluation Metrics:**
  - Accuracy, Precision, Recall, F1-Score
  - Confusion Matrix & Classification Report
- Random Forest outperformed with ~90% accuracy

---

### 3. Cholesterol Level Prediction
- Built a **Multiple Linear Regression** model to predict `chol` (serum cholesterol)
- Applied polynomial feature expansion and feature selection
- Model performed poorly (best R² ≈ 0.15) due to limited correlation of input features with cholesterol

---

### 4. Principal Component Analysis (PCA)
- Reduced dimensionality while retaining **90% variance**
- Final shape: `(303, 11)` components
- Visualized explained variance per component

---

### 5. K-Means Clustering
- Performed unsupervised grouping of patients using PCA-reduced features
- Optimal clusters: **k = 2** (based on silhouette score ≈ 0.19)
- Visualized clusters on 2D PCA space

---

## Libraries Used

- `pandas`, `numpy`, `matplotlib`, `seaborn`
- `scikit-learn`: preprocessing, models, PCA, KMeans, metrics

---

## File Structure

```bash
├── heart_disease.csv         # Dataset (Cleveland subset)
├── Heart_Disease_Model.ipynb # Main Jupyter Notebook
├── README.md                 # This file
