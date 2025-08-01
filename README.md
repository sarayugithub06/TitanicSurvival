# 🚢 Titanic Survival Prediction 🔍 | Machine Learning Project

Welcome aboard! 

This project dives into the historical **Titanic shipwreck** using **machine learning** to predict which passengers survived. We leverage a **Random Forest Classifier** and classic data science techniques to solve this Kaggle challenge with clean, effective code and insightful visuals.

---

## 🎯 Objective

> Predict the survival of Titanic passengers using passenger data like age, gender, class, and family relations.

---

## 📦 Tech Stack

- 💻 **Language**: Python
- 📊 **Data Analysis**: Pandas, NumPy
- 📈 **Visualization**: Matplotlib, Seaborn
- 🧠 **ML Model**: Random Forest Classifier (from Scikit-learn)

---

## 📁 Dataset Info

- **Source**: [Kaggle - Titanic: Machine Learning from Disaster](https://www.kaggle.com/c/titanic)
- **Files**:
  - `train.csv` – Contains features + survival labels.
  - `test.csv` – Features only, used for final predictions.

---

## 🔍 Exploratory Data Analysis (EDA)

- 📌 Checked null values and basic stats.
- 🧼 Filled missing `Age` values using **median imputation**.
- 🧠 Engineered a new feature: `FamilySize = SibSp + Parch`.
- 👤 Converted `Sex` to numeric values (male = 0, female = 1).
- 🧹 Removed non-informative features: `PassengerId`, `Name`, `Ticket`, `Cabin`.

📊 **Visualization Samples**:
- Survival distribution (`Survived` count)
- Heatmap of confusion matrix after predictions

---

## 🛠️ Model Building

### ✅ Model Used:
- **RandomForestClassifier** (100 trees, `random_state=42`)

### ⚙️ Steps:
1. Preprocessed the training data.
2. Split into training and validation sets (80/20).
3. Trained a Random Forest model.
4. Evaluated with **accuracy**, **confusion matrix**, and **classification report**.

---

## 📈 Model Evaluation

- **Accuracy Score**
- **Precision / Recall / F1 Score**
- 🔥 Visualization of Confusion Matrix using Seaborn Heatmap

---

## 📤 Submission

After training and evaluation, the model predicts survival on the test dataset and exports results to:

```bash
submission.csv
