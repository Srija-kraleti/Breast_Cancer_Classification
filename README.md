# 🎗️ Breast Cancer Classification

A machine learning project that classifies breast cancer tumors as **Malignant** or **Benign** using Logistic Regression on the scikit-learn Breast Cancer dataset.

---

## 📌 Overview

This project builds a binary classification model to predict whether a breast cancer tumor is malignant (cancerous) or benign (non-cancerous) based on 30 numeric features computed from digitized images of fine needle aspirate (FNA) of breast masses.

---

## 📂 Project Structure

```
Breast_Cancer_Classification.ipynb   # Main Jupyter Notebook
README.md                            # Project documentation
```

---

## 🗃️ Dataset

- **Source:** `sklearn.datasets.load_breast_cancer()`
- **Samples:** 569
- **Features:** 30 numeric features (mean, standard error, and worst values of cell nuclei measurements)
- **Target:**
  - `1` → Benign
  - `0` → Malignant

---

## 🔧 Tech Stack

| Library | Purpose |
|---|---|
| `numpy` | Numerical computations |
| `pandas` | Data manipulation and analysis |
| `scikit-learn` | Dataset, model training, and evaluation |

---

## 🚀 Workflow

1. **Data Collection** — Load the breast cancer dataset from scikit-learn
2. **Data Processing** — Convert to a Pandas DataFrame, add target labels, check for missing values
3. **Exploratory Analysis** — View shape, data types, statistical summary, and label distribution
4. **Feature/Target Split** — Separate features (`X`) and labels (`Y`)
5. **Train-Test Split** — 80% training / 20% testing (`random_state=2`)
6. **Model Training** — Fit a Logistic Regression model on training data
7. **Model Evaluation** — Compute accuracy on both training and test sets
8. **Predictive System** — Pick a random sample from test data and predict its diagnosis

---

## 📊 Results

| Dataset | Accuracy |
|---|---|
| Training Data | ~94–96% |
| Test Data | ~92–95% |

> Exact values may vary slightly depending on the run.

---

## 🔮 Predictive System

The notebook includes a live prediction system that:
- Randomly selects a sample from the test set
- Predicts whether the tumor is **Malignant** or **Benign**
- Compares the prediction with the actual ground truth label

```python
if prediction[0] == 0:
    print('The Breast cancer is Malignant')
else:
    print('The Breast Cancer is Benign')
```

---

## ▶️ How to Run

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/breast-cancer-classification.git
   cd breast-cancer-classification
   ```

2. Install dependencies:
   ```bash
   pip install numpy pandas scikit-learn
   ```

3. Open the notebook:
   ```bash
   jupyter notebook Breast_Cancer_Classification.ipynb
   ```

4. Run all cells from top to bottom.

---

## 📄 License

This project is open-source and available under the [MIT License](LICENSE).
