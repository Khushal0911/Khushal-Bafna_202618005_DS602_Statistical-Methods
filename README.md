# 🏥 Medical Insurance Statistical Dashboard

**Live Application:** [Insert Streamlit Community Cloud URL Here]

## Project Overview
This repository contains an end-to-end applied statistical modeling project analyzing a medical insurance dataset. The workflow transitions from static exploratory data analysis and hypothesis testing in a Jupyter Notebook to a fully interactive web dashboard built with Streamlit.

## Dataset Summary
The dataset (`data/insurance.csv`) contains 1,338 patient records and 7 attributes[cite: 3]. It is used to analyze the factors driving medical costs[cite: 4].
* **Demographics:** `age`, `sex`, `region`[cite: 3]
* **Health Metrics:** `bmi`, `smoker`[cite: 3]
* **Dependents:** `children`[cite: 3]
* **Target Variable:** `charges`[cite: 3]

## Key Statistical Findings
1. **Hypothesis Testing:** A Mann-Whitney U test (p < 0.001) confirms a massive, statistically significant difference in medical charges between smokers and non-smokers[cite: 3]. A One-Way ANOVA (p = 0.0309) also indicates significant variations in average charges across different geographic regions[cite: 3].
2. **Regression Modeling:** An Ordinary Least Squares (OLS) multiple regression model achieved an R-squared of 0.751[cite: 3]. 
3. **Primary Cost Drivers:** Smoking status is the most dominant factor, increasing expected medical charges by $23,850[cite: 3]. Age and BMI are also highly significant continuous predictors, adding $256.86 and $339.19 per respective unit increase[cite: 3].
4. **Diagnostic Checks:** Variance Inflation Factor (VIF) analysis confirmed no multicollinearity among continuous predictors (all VIFs ~ 1.0)[cite: 3]. However, the Jarque-Bera test on model residuals (p < 0.001) indicates non-normality, suggesting the potential need for log-transformations in future iterations[cite: 3].

## Running the Application Locally
1. Clone this repository and navigate to the root directory.
2. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use: venv\Scripts\activate
