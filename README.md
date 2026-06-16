# ML_Pipeline_of_Car_Dataset
Predicting the market value of a pre-owned vehicle is a classic regression challenge in predictive modeling. This project develops an end-to-end machine learning pipeline utilizing a dataset of 816 vehicle records to analyze and forecast resale prices based on key technical attributes. 

Note:-I create this project on Mobile ohone and when i stuck i asked Gemini AI to help and improvised the code..

Used Car Price Prediction 🚗📊
An end-to-end Machine Learning project developed to predict the resale value of used cars based on their features. This repository covers the entire data science pipeline, from initial data visualization fixes to training advanced tree-based ensemble models.


🛠️ Project Workflow & Features
Exploratory Data Analysis (EDA):
Handled heavy data distribution skewness using logarithmic scaling for clearer visualizations.
Analyzed manufacturer trends, identifying market tiers from high-end luxury brands down to economy budget segments.
Investigated the financial impact of different fuel types on resale values.



Feature Engineering:
Calculated a custom feature (car_age) to isolate and study vehicle depreciation patterns.
Constructed a Correlation Matrix Heatmap to identify relationships among features. Discovered that chronological age (correlation of -0.29) has a stronger impact on price drops than raw mileage (0.12).


Data Pipeline & Encoding:
Used scikit-learn's ColumnTransformer and OneHotEncoder to safely transform categorical columns (fuel_type, company) into machine-readable numeric formats while dropping redundant columns to avoid the dummy variable trap.
Segmented data into independent Training (80%) and Testing (20%) datasets to ensure robust model evaluation.


Model Training & Comparison:
Evaluated multiple advanced regression algorithms side-by-side to find the optimal solution:
Decision Tree Regressor
Random Forest Regressor (Achieved the lowest Mean Absolute Error of 180,432 Rupees)
XGBoost Regressor (Achieved the highest R² Score of 0.2023)


📈 Key Insights
Diesel Premium: Diesel vehicles command a significantly higher resale baseline average (509K) compared to Petrol alternatives (324K) in this dataset, driven by engine durability and inclusion in premium SUV segments.
Non-Linear Depreciation: Tree-based ensemble models vastly outperformed simple linear constraints, effectively capturing the nuanced, non-linear ways cars lose value over time.


💻 Tech Stack Used
Language: Python
Libraries: Pandas, NumPy, Matplotlib, Seaborn, Scikit-Learn, XGBoost
Platform: Kaggle Notebooks

Contact:-
Gmail:-englandengland271@gmail.com
LinkedIn:-[https://www.linkedin.com/in/mohammed-nafay-ali-16519138a?utm_source=share&utm_campaign=share_via&utm_content=profile&utm_medium=android_app]

Streamlit web:-[https://mlpipelineofcardataset-6zcwtm2lamlvacdfw5g7iz.streamlit.app/]
