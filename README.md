# 🏥 Medical Insurance Statistical Dashboard

**Live Application:** (https://202618005-ds602-statistical-methods-kb.streamlit.app/)
**Author:** Khushal Vishal Bafna | Roll No: 202618005  
**Academic Context:** M.Sc. Data Science (Semester 1), Dhirubhai Ambani University

---

## 📌 Project Overview
This repository contains an end-to-end applied statistical modeling workflow. The project bridges the gap between static academic analysis and production-ready data science by transitioning exploratory data analysis (EDA) and inferential statistics from a Jupyter Notebook into a fully interactive web application. 

The dashboard enables users to dynamically filter data, execute live hypothesis tests, and generate real-time medical cost predictions backed by Ordinary Least Squares (OLS) regression and Gauss-Markov diagnostic checks.

## 🗄️ Dataset Architecture
The underlying dataset (`data/insurance.csv`) consists of 1,338 distinct patient records containing 7 attributes. It is structured to evaluate how personal demographics and health choices drive medical billing.

| Attribute | Data Type | Description |
| :--- | :--- | :--- |
| **`charges`** | Continuous (Target) | Individual medical costs billed by health insurance ($). |
| **`age`** | Continuous | Age of the primary beneficiary (18-64 years). |
| **`bmi`** | Continuous | Body Mass Index, indicating weight relative to height. |
| **`children`** | Discrete | Number of dependents covered by the insurance plan. |
| **`smoker`** | Categorical | Smoking status of the beneficiary (yes/no). |
| **`sex`** | Categorical | Biological sex of the insurance contractor (male/female). |
| **`region`** | Categorical | Geographic residential area in the US (4 distinct regions). |

---

## 🧮 Statistical Methodology & Mathematical Formulas

The dashboard rigorously applies parametric and non-parametric statistical tests, validating mathematical assumptions before executing inference.

### 1. Descriptive Statistics
Used in the EDA tab to summarize the dataset distributions.
* **Sample Mean:** 
  $$\bar{x}=\frac{1}{n}\sum_{i=1}^{n}x_i$$
* **Sample Standard Deviation:** 
  $$s=\sqrt{\frac{1}{n-1}\sum_{i=1}^{n}(x_i-\bar{x})^2}$$
* **Skewness (Fisher-Pearson):** 
  $$g_1=\frac{\frac{1}{n}\sum_{i=1}^n(x_i-\bar{x})^3}{\left[\frac{1}{n}\sum_{i=1}^n(x_i-\bar{x})^2\right]^{3/2}}$$

### 2. Assumption Checks
Prior to running A/B tests, the application verifies normality and homoscedasticity.
* **Shapiro-Wilk Test (Normality):** Evaluates if a sample comes from a normally distributed population. 
  $$W=\frac{\left(\sum_{i=1}^na_ix_{(i)}\right)^2}{\sum_{i=1}^n(x_i-\bar{x})^2}$$
* **Levene’s Test (Equal Variance):** Assesses if samples have equal variances.
  $$W=\frac{(N-k)}{(k-1)}\frac{\sum_{i=1}^kN_i(\bar{Z}_{i.}-\bar{Z}_{..})^2}{\sum_{i=1}^k\sum_{j=1}^{N_i}(Z_{ij}-\bar{Z}_{i.})^2}$$

### 3. Hypothesis Testing (Inference)
* **Two-Sample t-test (Parametric):** Applied when data passes both Shapiro-Wilk and Levene's tests.
  $$t=\frac{\bar{x}_1-\bar{x}_2}{\sqrt{\frac{s_1^2}{n_1}+\frac{s_2^2}{n_2}}}$$
* **Mann-Whitney U Test (Non-parametric):** The automatic fallback for non-normal or heteroscedastic data. It compares the sum of ranks ($R_1$).
  $$U_1=n_1n_2+\frac{n_1(n_1+1)}{2}-R_1$$
* **One-Way ANOVA:** Compares means across three or more independent groups (e.g., regions).
  $$F=\frac{MS_{between}}{MS_{within}}$$

### 4. Regression & Diagnostics
* **Multiple Linear Regression (OLS):** Models the relationship between the target (`charges`) and independent predictors.
  $$Y=\beta_0+\beta_1X_1+\beta_2X_2+\dots+\beta_kX_k+\epsilon$$
* **Variance Inflation Factor (VIF):** Detects multicollinearity among predictors.
  $$VIF_i=\frac{1}{1-R_i^2}$$

---

## 📈 Key Statistical Findings

* **Primary Cost Drivers:** The OLS model achieved a strong explanatory power with an **$R^2$ of 0.751**. Smoking is the dominant financial driver; holding all else constant, being a smoker increases baseline medical charges by **$23,850** ($p<0.001$).
* **Continuous Predictors:** Age and BMI are highly significant positive drivers. Each additional year of age adds **$256.86**, and each unit increase in BMI adds **$339.19** to expected charges ($p<0.001$).
* **Group Disparities:** Non-parametric Mann-Whitney U testing confirmed massive, statistically significant disparities in costs between smokers and non-smokers. One-Way ANOVA ($p=0.0309$) also highlighted significant geographic pricing variations.
* **Model Diagnostics:** Variance Inflation Factor (VIF) screening confirmed the absence of multicollinearity (all continuous $VIF\approx1.0$). However, Jarque-Bera testing on model residuals ($p<0.001$) flagged non-normality, indicating potential non-linear interactions (e.g., $BMI\times Smoker$) that could be addressed with log-transformations in future iterations.

---

## 💻 Tech Stack
* **Language:** Python 3
* **Frontend:** Streamlit
* **Statistical Computation:** Statsmodels, SciPy
* **Data Manipulation:** Pandas, NumPy
* **Data Visualization:** Plotly, Matplotlib

---

## ⚙️ Local Installation & Setup

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/Khushal0911/Khushal-Bafna_202618005_DS602_Statistical-Methods.git](https://github.com/Khushal0911/Khushal-Bafna_202618005_DS602_Statistical-Methods.git)
   cd Khushal-Bafna_202618005_DS602_Statistical-Methods
