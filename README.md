# 🚗 Used Car Price Predictor Backend & Web Application:-

An end-to-end Machine Learning regression project built entirely from scratch to analyze, optimize, and forecast vehicle resale values. This repository encompasses everything from exploratory data analysis (EDA) and robust feature engineering to a live interactive multi-model deployment pipeline.
# 🔗 [https://ml-pipeline-of-car-dataset-web-application.streamlit.app/] ***Web App Link***



# 🚀 Key Project Milestones & Upgrades:-
Initial Performance Baseline: Originally achieved an R^2 score of 0.20 due to unhandled high-cardinality columns and severe data skewness.
Advanced Optimization Sprint: By handling outliers, using advanced target encoding, implementing dynamic feature engineering, and tuning hyperparameters, the final model accuracy shot up to 77.11% (R^2 = 0.7711).
Error Reduction: Mean Absolute Error (MAE) was squeezed down from ~180,432 to 100,207 Rupees, meaning predictions are incredibly accurate.


# 🛠️ Data Pipeline & Feature Engineering:-
To move from a baseline academic project to a high-performing script, the data went through a rigorous engineering pipeline:
Logarithmic Price Scaling: Applied np.log1p to the target feature (Price) to address right-skewed price distributions and stabilize regression gradients.
Dynamic Data Cleansing: Removed heavily damaged or scrap entries under 50,000 Rupees and capped luxury anomalies above 6,000,000 Rupees to keep the training pool clean.
Usage Density Feature (kms_per_year): Engineered a ratio capturing annual vehicle stress by dividing total kilometers driven by calculated vehicle age.
Brand Premium Mapping (company_tier): Created a mapping of manufacturers to their dataset median market price, giving tree models a concrete pricing weight for categorical features.

# 📊 Model Comparison & Final Results
The final preprocessed pipeline was evaluated against multiple advanced tree-based ensemble variants on an independent 20% test split:

* 🏆 **XGBoost Regressor (Tuned):** $R²$ Score: **0.7711** | MAE: **100,207.70 Rs.**
* 🌲 **Random Forest (Ensemble):** $R²$ Score: **0.7390** | MAE: **115,554.22 Rs.**
* 📉 **Decision Tree (Baseline):** $R²$ Score: **0.0082** | MAE: **189,683.84 Rs.**

---

## 📩 Contact & Connect
* **Email:** englandengland271@gmail.com
* **LinkedIn:** [Mohamed Nafay Ali](https://www.linkedin.com/in/mohammed-nafay-ali-18519130a)
* 

