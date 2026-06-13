# 🏠 House Price Prediction — Linear Regression from Scratch

> An end-to-end Machine Learning pipeline implementing Multiple Linear Regression and all major gradient descent variants from scratch using NumPy only — no scikit-learn for core algorithms. Trained on the Ames Housing dataset.

![Python](https://img.shields.io/badge/Python-3.10%2B-blue) ![NumPy](https://img.shields.io/badge/Built%20with-NumPy%20only-013243) ![License](https://img.shields.io/badge/License-MIT-green)


## 📊 Results

| Model | R2 Score | RMSE | MAE |
| - | - | - | - |
| Batch Gradient Descent | 0.6928 | 48,541 | 27,842 |
| Stochastic Gradient Descent | 0.8511 | 33,796 | 20,043 |
| Mini-batch Gradient Descent | 0.8656 | 32,106 | 20,367 |
| **Ridge Regression** | **0.8715** | **31,391** | **19,971** |
| Lasso Regression | 0.8301 | 36,104 | 20,632 |
| Elastic Net Regression | 0.8585 | 32,942 | 18,858 |


**Best model: Ridge Regression** with R2 = 0.871 on the held-out test set.


## 🎯 What This Project Does

Given 79 features of a house (lot area, neighbourhood, year built, garage size, etc.), the model predicts the sale price. The project implements the full ML pipeline from scratch:

- Data loading and EDA

- Missing value imputation

- One-hot encoding of categorical features

- Feature scaling (StandardScaler from scratch)

- Six regression models from scratch

- Evaluation metrics from scratch

- Loss curve and residual visualizations


## 🧠 Algorithms Implemented from Scratch

### 1. Batch Gradient Descent (BGD)

Uses the full training set to compute the gradient at each step.

$$w := w - \\alpha \\cdot \\frac\{2\}\{n\} X^T(Xw - y)$$

Stable but slow. Converges to a smooth loss curve.

### 2. Stochastic Gradient Descent (SGD)

Updates weights using one random sample at a time.

$$w := w - \\alpha \\cdot 2(x\_i w - y\_i) x\_i$$

Noisier loss curve but can escape local minima — achieved **0.85 R2**, outperforming BGD.

### 3. Mini-batch Gradient Descent

Compromise between BGD and SGD — uses a batch of 32 samples per update.

Best of both worlds: smoother than SGD, faster than BGD. Achieved **0.87 R2**.

### 4. Ridge Regression (L2)

Adds L2 penalty to the weight gradient to prevent overfitting:

$$w := w - \\alpha \\left(\\frac\{-2\}\{n\} X^T(y - \\hat\{y\}) + 2\\lambda w\\right)$$

Shrinks all weights toward zero proportionally. Best overall performer at **0.871 R2**.

### 5. Lasso Regression (L1)

Adds L1 penalty — uses the subgradient since |w| is not differentiable at 0:

$$w := w - \\alpha \\left(\\frac\{-2\}\{n\} X^T(y - \\hat\{y\}) + \\lambda \\cdot \\text\{sign\}(w)\\right)$$

Pushes weights to exactly zero — performs automatic feature selection.

### 6. Elastic Net Regression (L1 + L2)

Combines both penalties with a mixing ratio `l1\_ratio`:

$$w := w - \\alpha \\left(\\frac\{-2\}\{n\} X^T(y - \\hat\{y\}) + \\lambda\_1 \\cdot \\text\{sign\}(w) + 2\\lambda\_2 w\\right)$$

Where `l1\_ratio` controls the balance: `l1\_ratio=1` → pure Lasso, `l1\_ratio=0` → pure Ridge.


## 📁 Project Structure

```
house-price-prediction/  
│  
├── my\_library/                              ← Custom ML library (NumPy only)  
│   ├── \_\_init\_\_.py  
│   ├── preprocessing.py                    ← StandardScaler with inverse\_transform  
│   ├── metrics.py                          ← MSE, RMSE, MAE, R2 from scratch  
│   ├── batch\_gradient\_descent.py           ← BGD  
│   ├── stochastic\_gradient\_descent.py      ← SGD  
│   ├── mini\_batch\_gradient\_descent.py      ← Mini-batch GD  
│   ├── ridge\_regression.py                 ← Ridge (L2)  
│   ├── lasso\_regression.py                 ← Lasso (L1)  
│   └── elastic\_net\_regression.py           ← Elastic Net (L1 + L2)  
│  
├── data/  
│   ├── train.csv                           ← Ames Housing dataset  
│  
├── notebooks/  
│   ├── batch\_gradient\_descent.ipynb  
│   ├── stochastic\_gradient\_descent.ipynb  
│   ├── mini\_batch\_gradient\_descent.ipynb  
│   ├── ridge\_regression.ipynb  
│   ├── lasso\_regression.ipynb  
│   ├── elastic\_net\_regression.ipynb  
│   └── model\_comparison.ipynb             ← All models compared  
│  
├── requirements.txt  
└── README.md
```


## 🔑 Key Implementation Details

### Why feature scaling is mandatory for SGD and Mini-batch but not BGD?

BGD averages gradients over all 1168 training samples — the large feature magnitudes cancel out. SGD updates on a single sample at a time, so a feature like `LotArea` (values in thousands) produces a gradient thousands of times larger than a binary feature, causing immediate explosion without scaling.

### Why does SGD outperform BGD here?

SGD's noisy per-sample updates act as implicit regularization — the randomness helps the optimizer escape suboptimal regions that BGD gets stuck in. This is a well-known phenomenon in ML.

### Why log-space is not needed here but was needed in Naive Bayes?

Linear regression computes products of at most a few terms — no underflow risk. Naive Bayes multiplies 131 probabilities, each \< 1, which requires log-space to prevent float64 underflow.

### Laplace smoothing equivalent in regression — regularization

Just as Laplace smoothing prevents zero-probability collapse in Naive Bayes, L2 regularization in Ridge prevents individual weights from growing unboundedly when features are correlated.


## 🚀 Getting Started

```
git clone https://github.com/sajidansari7/house-price-prediction.git  
cd house-price-prediction  
  
python -m venv venv  
source venv/bin/activate  
pip install -r requirements.txt  
  
jupyter notebook
```

Open any notebook in the `notebooks/` folder and run all cells.


## 💡 Key Interview Talking Points

**Q: Why implement from scratch instead of using sklearn?** Understanding the gradient update rule, why we divide by n\_samples, why we never regularize the bias term, and why log-space matters for Naive Bayes but not regression — these are things you only truly understand by implementing from scratch.

**Q: Why does Ridge outperform plain BGD?** The Ames Housing dataset has 300+ features after one-hot encoding, many of which are correlated. Without regularization, the model overfits by assigning large weights to correlated features. Ridge penalizes large weights, forcing the model to distribute importance more evenly.

**Q: What is the difference between Ridge and Lasso in practice?** Ridge shrinks all weights proportionally toward zero but never reaches exactly zero. Lasso applies a constant push via `sign(w)` which drives small weights to exactly zero — automatic feature selection. That's why Lasso is preferred when you suspect many features are irrelevant.

**Q: Why never regularize the bias term?** The bias captures the global mean of the target — regularizing it would shift all predictions toward zero regardless of the data, causing systematic underfitting.


## 📋 Dataset

- **Ames Housing Dataset** — 1460 training samples, 79 features

- Target variable: `SalePrice` (house sale price in USD)

- After one-hot encoding: ~300 binary and continuous features

- 80/20 train/test split


## 🔗 Related Projects

- [Disease Symptom Classifier](https://github.com/sajidansari7/disease-symptom-classifier-using-naive-bayes) — Naive Bayes from scratch, 99.9% accuracy, Streamlit deployment


**Built by Sajid Ansari** · BTech CSE · NIT Rourkela · ML Portfolio  
[GitHub](https://github.com/sajidansari7) · [LinkedIn](https://linkedin.com/in/sajidansari7) 

