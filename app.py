import streamlit as st
import pandas as pd
import numpy as np
import scipy.stats as stats
import statsmodels.formula.api as smf
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
import plotly.express as px
import matplotlib.pyplot as plt

st.set_page_config(page_title="Insurance Statistical Dashboard", layout="wide")

# Cache data loading
@st.cache_data
def load_data():
    url = "https://raw.githubusercontent.com/stedy/Machine-Learning-with-R-datasets/master/insurance.csv"
    return pd.read_csv(url)

df = load_data()

st.title("Applied Statistical Modeling & Interactive Web Dashboard")

# Navigation Tabs
tab1, tab2, tab3 = st.tabs([
    "Data Exploration", 
    "Hypothesis Testing Lab", 
    "Live Prediction & Diagnostics"
])

# -----------------------------------------------------------------------------
# TAB 1: DATA EXPLORATION
# -----------------------------------------------------------------------------
with tab1:
    st.header("Exploratory Data Analysis")

    # Sidebar Filters
    st.sidebar.subheader("Filter Data (Tab 1)")
    age_range = st.sidebar.slider("Age Range", int(df['age'].min()), int(df['age'].max()), (20, 60))
    selected_regions = st.sidebar.multiselect("Select Regions", options=df['region'].unique(), default=list(df['region'].unique()))
    selected_smoker = st.sidebar.multiselect("Smoking Status", options=df['smoker'].unique(), default=list(df['smoker'].unique()))

    filtered_df = df[
        (df['age'] >= age_range[0]) & 
        (df['age'] <= age_range[1]) & 
        (df['region'].isin(selected_regions)) & 
        (df['smoker'].isin(selected_smoker))
    ]

    col1, col2 = st.columns([1, 2])
    with col1:
        st.subheader("Summary Metrics")
        num_cols = ['age', 'bmi', 'children', 'charges']
        desc = pd.DataFrame(index=num_cols)
        desc['Mean'] = filtered_df[num_cols].mean()
        desc['Median'] = filtered_df[num_cols].median()
        desc['Std'] = filtered_df[num_cols].std()
        desc['IQR'] = filtered_df[num_cols].apply(lambda x: stats.iqr(x))
        desc['Skewness'] = filtered_df[num_cols].apply(lambda x: stats.skew(x))
        st.dataframe(desc.round(2), use_container_width=True)

    with col2:
        st.subheader("Distribution Plot")
        plot_metric = st.selectbox("Feature to inspect:", num_cols, index=3)
        fig_dist = px.histogram(filtered_df, x=plot_metric, color='smoker', marginal="box", nbins=30, barmode="overlay")
        st.plotly_chart(fig_dist, use_container_width=True)

    st.subheader("Bivariate Correlation Matrix")
    corr = filtered_df[num_cols].corr()
    fig_corr = px.imshow(corr, text_auto=True, aspect="auto", color_continuous_scale="RdBu_r")
    st.plotly_chart(fig_corr, use_container_width=True)

# -----------------------------------------------------------------------------
# TAB 2: HYPOTHESIS TESTING LAB
# -----------------------------------------------------------------------------
with tab2:
    st.header("Interactive Hypothesis Testing Lab")

    test_type = st.radio("Select Hypothesis Test Protocol:", ["Two-Group Comparison", "One-Way ANOVA (Multi-Group)"])
    alpha = 0.05

    if test_type == "Two-Group Comparison":
        col1, col2 = st.columns(2)
        with col1:
            cat_var = st.selectbox("Grouping Variable (2-level):", ["smoker", "sex"])
            val1 = df[cat_var].unique()[0]
            val2 = df[cat_var].unique()[1]
        with col2:
            num_var = st.selectbox("Numerical Metric:", ["charges", "bmi", "age"])

        grp1 = df[df[cat_var] == val1][num_var]
        grp2 = df[df[cat_var] == val2][num_var]

        # Assumption checks
        s1 = stats.shapiro(grp1).pvalue
        s2 = stats.shapiro(grp2).pvalue
        lev = stats.levene(grp1, grp2).pvalue

        st.markdown(f"**Normality p-values:** `{val1}` = {s1:.3e}, `{val2}` = {s2:.3e}")
        st.markdown(f"**Levene's Equal Variance p-value:** {lev:.3e}")

        if s1 > alpha and s2 > alpha and lev > alpha:
            stat, pval = stats.ttest_ind(grp1, grp2)
            applied = "Two-Sample t-test"
        else:
            stat, pval = stats.mannwhitneyu(grp1, grp2)
            applied = "Mann-Whitney U Test (Non-parametric)"

        st.info(f"**Applied Protocol:** {applied}")
        st.write(f"**Test Statistic:** {stat:.4f} | **p-value:** {pval:.4e}")

        if pval < alpha:
            st.error(f"**Conclusion:** Reject H0 at alpha = 0.05. Significant difference found between {val1} and {val2}.")
        else:
            st.success(f"**Conclusion:** Fail to Reject H0 at alpha = 0.05. No significant difference detected.")

    else:
        st.subheader("One-Way ANOVA Test")
        group_var = st.selectbox("Categorical Factor (>=3 groups):", ["region", "children"])
        target_num = st.selectbox("Continuous Target:", ["charges", "bmi"])

        group_names = df[group_var].unique()
        arrays = [df[df[group_var] == g][target_num] for g in group_names]

        f_val, p_val = stats.f_oneway(*arrays)
        st.write(f"**F-Statistic:** {f_val:.4f} | **p-value:** {p_val:.4e}")

        if p_val < alpha:
            st.error("Reject H0: Mean differences across groups are statistically significant.")
        else:
            st.success("Fail to Reject H0: No statistically significant difference across groups.")

# -----------------------------------------------------------------------------
# TAB 3: LIVE PREDICTION & DIAGNOSTICS
# -----------------------------------------------------------------------------
with tab3:
    st.header("Live Prediction & Gauss-Markov Residual Diagnostics")

    # Fit Model
    formula = "charges ~ age + bmi + children + C(sex) + C(smoker) + C(region)"
    model = smf.ols(formula=formula, data=df).fit()

    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("Predict Medical Charges")
        in_age = st.slider("Age", 18, 65, 30)
        in_bmi = st.slider("BMI", 15.0, 50.0, 25.0)
        in_children = st.number_input("Children", min_value=0, max_value=5, value=0)
        in_sex = st.selectbox("Sex", df['sex'].unique())
        in_smoker = st.selectbox("Smoker", df['smoker'].unique())
        in_region = st.selectbox("Region", df['region'].unique())

        input_data = pd.DataFrame({
            'age': [in_age],
            'bmi': [in_bmi],
            'children': [in_children],
            'sex': [in_sex],
            'smoker': [in_smoker],
            'region': [in_region]
        })

        pred_res = model.get_prediction(input_data)
        pred_df = pred_res.summary_frame(alpha=0.05)

        st.markdown(f"### Predicted Charges: **${pred_df['mean'].iloc[0]:,.2f}**")
        st.markdown(f"**95% Confidence Interval:** [${pred_df['mean_ci_lower'].iloc[0]:,.2f}, ${pred_df['mean_ci_upper'].iloc[0]:,.2f}]")
        st.markdown(f"**95% Prediction Interval:** [${pred_df['obs_ci_lower'].iloc[0]:,.2f}, ${pred_df['obs_ci_upper'].iloc[0]:,.2f}]")

    with col2:
        st.subheader("Model Diagnostic Plots")
        fig_diag, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

        ax1.scatter(model.fittedvalues, model.resid, alpha=0.3, edgecolors='k')
        ax1.axhline(0, color='red', linestyle='--')
        ax1.set_title("Residuals vs. Fitted")
        ax1.set_xlabel("Fitted Values")
        ax1.set_ylabel("Residuals")

        sm.qqplot(model.resid, line='45', fit=True, ax=ax2)
        ax2.set_title("Normal Q-Q")

        plt.tight_layout()
        st.pyplot(fig_diag)

        jb_stat, jb_pval, _, _ = sm.stats.stattools.jarque_bera(model.resid)
        st.caption(f"Jarque-Bera p-value: {jb_pval:.4e} (Non-normal residuals indicate potential non-linearities or interaction effects).")
