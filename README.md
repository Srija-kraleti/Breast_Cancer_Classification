# 🩺 Breast Cancer Classification using Machine Learning

A Machine Learning project that classifies breast cancer cases as **Malignant** or **Benign** using the **Breast Cancer Wisconsin dataset** available through Scikit-learn. The project uses **Logistic Regression** for classification and evaluates the model using training/testing accuracy and a confusion matrix.

---

## 📌 Project Overview

Breast cancer is one of the most common types of cancer. Early and accurate classification of breast tumors can support medical analysis and decision-making.

In this project, machine learning is used to classify breast cancer cases based on various tumor-related features.

The project follows a complete machine learning workflow:

**Data Collection → Data Processing → Exploratory Data Analysis → Feature/Target Separation → Train-Test Split → Model Training → Model Evaluation → Prediction**

---

## 🎯 Objectives

* Load and analyze the Breast Cancer dataset.
* Perform basic data exploration and preprocessing.
* Understand the distribution of malignant and benign cases.
* Analyze relationships between features.
* Train a **Logistic Regression** classification model.
* Evaluate model performance using accuracy.
* Visualize model performance using a confusion matrix.
* Build a simple predictive system for individual samples.

---

## 📊 Dataset

The project uses the built-in **Breast Cancer Wisconsin dataset** from `sklearn.datasets`.

### Dataset Details

* **Source:** Scikit-learn
* **Total Samples:** 569
* **Number of Features:** 30
* **Target Variable:** `label`
* **Classes:** 2

### Target Labels

| Label | Diagnosis |
| ----: | --------- |
|   `0` | Malignant |
|   `1` | Benign    |

The dataset contains numerical features describing characteristics of breast cell nuclei.

Examples include:

* Mean Radius
* Mean Texture
* Mean Perimeter
* Mean Area
* Mean Smoothness
* Mean Compactness
* Mean Concavity
* Mean Symmetry
* Mean Fractal Dimension

and their corresponding standard error and worst-case measurements.

---

## 🛠️ Technologies Used

* **Python**
* **NumPy**
* **Pandas**
* **Matplotlib**
* **Seaborn**
* **Scikit-learn**
* **Jupyter Notebook**

---

## 📚 Machine Learning Algorithm

### Logistic Regression

Logistic Regression is a supervised machine learning algorithm commonly used for binary classification problems.

In this project, Logistic Regression is used to classify a tumor into one of two categories:

```text
0 → Malignant
1 → Benign
```

The model is trained using the training portion of the dataset and then evaluated using unseen test data.

---

## 🔄 Project Workflow

### 1. Import Dependencies

The following Python libraries are used:

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn.datasets

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
```

---

### 2. Data Collection

The Breast Cancer dataset is loaded directly from Scikit-learn:

```python
breast_cancer_dataset = sklearn.datasets.load_breast_cancer()
```

The dataset is then converted into a Pandas DataFrame for easier analysis.

---

### 3. Data Processing

The target variable is added to the DataFrame:

```python
data_frame['label'] = breast_cancer_dataset.target
```

The dataset is examined using:

```python
data_frame.shape
data_frame.info()
data_frame.isnull().sum()
data_frame.describe()
```

This helps understand:

* Dataset dimensions
* Data types
* Missing values
* Statistical characteristics of the features

---

### 4. Exploratory Data Analysis

The project analyzes the distribution of the target variable:

```python
data_frame['label'].value_counts()
```

A count plot is used to visualize the number of benign and malignant cases.

The project also analyzes:

* Feature correlations
* Correlation with the target variable
* Distribution of mean radius
* Average feature values for each diagnosis

A correlation heatmap is generated to understand relationships between features.

---

### 5. Feature and Target Separation

The dataset is divided into:

* **X → Features**
* **Y → Target**

```python
X = data_frame.drop(columns='label')
Y = data_frame['label']
```

---

### 6. Train-Test Split

The dataset is divided into training and testing data.

```python
X_train, X_test, Y_train, Y_test = train_test_split(
    X,
    Y,
    test_size=0.2,
    random_state=2
)
```

The split uses:

* **80% → Training data**
* **20% → Testing data**

---

### 7. Model Training

A Logistic Regression model is created:

```python
model = LogisticRegression(max_iter=10000)
```

The model is trained using:

```python
model.fit(X_train, Y_train)
```

---

## 📈 Model Evaluation

The trained model is evaluated on both training and testing datasets.

### Training Accuracy

```python
X_train_prediction = model.predict(X_train)

training_data_accuracy = accuracy_score(
    Y_train,
    X_train_prediction
)
```

### Testing Accuracy

```python
X_test_prediction = model.predict(X_test)

test_data_accuracy = accuracy_score(
    Y_test,
    X_test_prediction
)
```

The notebook also visualizes the comparison between training and testing accuracy using a bar chart.

> **Note:** The exact accuracy values are generated when the notebook is executed and should be taken directly from the notebook output rather than assumed from the code alone.

---

## 🔍 Predictive System

The project includes a simple predictive system that selects a sample from the test dataset and predicts its diagnosis.

```python
random_sample = X_test.sample(n=1)

prediction = model.predict(random_sample)
```

The prediction is interpreted as:

```text
0 → The Breast cancer is Malignant
1 → The Breast cancer is Benign
```

The actual label is also retrieved so that the prediction can be compared with the true diagnosis.

---

## 📊 Confusion Matrix

A confusion matrix is generated to visualize the classification performance:

```python
from sklearn.metrics import confusion_matrix

cm = confusion_matrix(
    Y_test,
    X_test_prediction
)
```

The confusion matrix shows the relationship between:

* Actual Malignant
* Actual Benign
* Predicted Malignant
* Predicted Benign

This provides more detailed information about the types of predictions made by the model beyond overall accuracy.

---

## 📁 Project Structure

```text
Breast-Cancer-Classification/
│
├── Breast_Cancer_Classification.ipynb
├── README.md
└── requirements.txt
```

---

## 🚀 How to Run the Project

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/Breast-Cancer-Classification.git
```

### 2. Navigate to the Project Directory

```bash
cd Breast-Cancer-Classification
```

### 3. Install Required Libraries

```bash
pip install numpy pandas matplotlib seaborn scikit-learn jupyter
```

### 4. Launch Jupyter Notebook

```bash
jupyter notebook
```

### 5. Open the Notebook

Open:

```text
Breast_Cancer_Classification.ipynb
```

Run the cells sequentially to reproduce the analysis and results.

---

## 📦 Requirements

```text
numpy
pandas
matplotlib
seaborn
scikit-learn
jupyter
```

---

## 📌 Key Concepts Demonstrated

* Data Collection
* Data Preprocessing
* Exploratory Data Analysis
* Data Visualization
* Feature-Target Separation
* Train-Test Split
* Supervised Learning
* Logistic Regression
* Binary Classification
* Model Prediction
* Accuracy Evaluation
* Confusion Matrix
* Correlation Analysis

---

## 🔮 Future Improvements

The current project can be extended by:

* Comparing Logistic Regression with other classification algorithms.
* Adding Precision, Recall and F1-score.
* Generating a complete classification report.
* Applying feature scaling.
* Performing hyperparameter tuning.
* Using cross-validation.
* Comparing multiple models.
* Building an interactive Streamlit application.
* Adding a user interface for entering patient-related feature values.
* Deploying the trained model as a web application.

---

## ⚠️ Disclaimer

This project is intended for **educational and machine learning demonstration purposes only**.

The predictions produced by this model should **not be used as a substitute for professional medical diagnosis or clinical decision-making**.

---

## 👩‍💻 Author

**Kameswari Srija Kraleti**

B.Tech – Computer Science and Engineering

### 🔗 Connect With Me

* GitHub: `https://github.com/Srija-kraleti`
* LinkedIn: 'https://www.linkedin.com/in/kameswari-srija-kraleti'

---

## ⭐ If You Found This Project Useful

If you found this project helpful for learning Machine Learning and classification, consider giving the repository a ⭐ on GitHub!
