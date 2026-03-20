# SCT_ML_1
# 🏠 House Price Prediction using Machine Learning

## 📌 Project Overview

This project focuses on predicting house prices using an advanced machine learning approach based on the Kaggle **House Prices – Advanced Regression Techniques** dataset.

The model leverages data preprocessing, feature engineering, and ensemble learning techniques to achieve high prediction accuracy.

---

## 🚀 Key Features

* 📊 Data preprocessing (handling missing values)
* 🔤 Categorical feature encoding (One-Hot Encoding)
* 📈 Feature engineering using full dataset
* 🌲 Random Forest Regressor (Advanced Model)
* 📉 Model evaluation using R² Score & MSE
* 📦 Kaggle submission file generation

---

## 🧠 Machine Learning Model

* **Algorithm Used:** Random Forest Regressor
* **Why?** Handles non-linear relationships and improves prediction accuracy over linear models

---

## 📊 Results

* ✅ R² Score: **0.89**
* 📉 Mean Squared Error: ~3.8 Billion
* 📈 Strong predictive performance on validation data

---

## 📁 Project Structure

```
house_price_ml_project/
│
├── data/                # Dataset (not uploaded to GitHub)
├── output/              # Generated results
│   ├── advanced_plot.png
│   ├── advanced_submission.csv
│
├── main.py              # Main ML pipeline
├── requirements.txt     # Dependencies
└── README.md            # Project documentation
```

---

## ⚙️ Installation & Setup

### 1️⃣ Clone the repository

```
git clone https://github.com/hasinib729/SCT_ML_1.git
cd house-price-prediction-ml
```

### 2️⃣ Install dependencies

```
pip install -r requirements.txt
```

### 3️⃣ Add dataset

Download dataset from Kaggle and place files in:

```
data/train.csv
data/test.csv
```

---

## ▶️ Run the Project

```
python main.py
```

---

## 📦 Output

After execution:

* 📄 `advanced_submission.csv` → Kaggle submission file
* 📊 `advanced_plot.png` → Visualization of predictions

---

## 🏆 Kaggle Submission

The model predictions are formatted and ready for submission on Kaggle.

---

## 📚 Technologies Used

* Python 🐍
* Pandas
* NumPy
* Scikit-learn
* Matplotlib

---

## 💼 Resume Description

Developed an advanced house price prediction model using Random Forest on Kaggle dataset. Performed data preprocessing including handling missing values and categorical encoding. Achieved high prediction accuracy with R² score of 0.89. Built a complete end-to-end machine learning pipeline.

---

## 🔮 Future Improvements

* Implement XGBoost for better performance
* Hyperparameter tuning (GridSearchCV)
* Feature importance visualization
* Cross-validation for robustness

---

## 🙌 Acknowledgements

* Kaggle Dataset: House Prices – Advanced Regression Techniques
* Scikit-learn documentation

---

## ⭐ If you like this project

Give it a ⭐ on GitHub!
