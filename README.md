🌾 **SDG2 – Agricultural Yield Optimizer**
A Machine Learning–based predictive system for estimating crop yield using soil nutrient levels and rainfall data.

**📌 Project Overview**
Agriculture plays a critical role in ensuring food security and economic sustainability. Accurate crop yield prediction enables optimized fertilizer usage, improved resource allocation, and better agricultural planning.
The Agricultural Yield Optimizer is developed under the SDG 2 (Zero Hunger) Initiative as part of the Artificial Intelligence and Machine Learning Internship Program.
This system predicts crop yield (kg per hectare) using:
- Nitrogen (N)
- Phosphorous (P)
- Potassium (K)
- Rainfall
The project integrates data preprocessing, feature scaling, regression modeling, and performance evaluation to build a reliable and data-driven predictive model.

**🎯 Objectives**
- Develop a regression-based crop yield prediction model
- Handle missing values using mean imputation
- Apply feature scaling to normalize input variables
- Compare multiple models and evaluate performance
- Select the best-performing model using error metrics

**🧠 Machine Learning Pipeline**
The system follows a structured ML workflow:
**Dataset (GitHub Repository)**
            ↓
**Data Loading (Pandas)**
            ↓
**Missing Value Handling (Mean Imputation)**
            ↓
**Feature Scaling (Standardization)**
            ↓
**Train-Test Split (80:20)**
            ↓
**Model Training (Regression Models)**
            ↓
**Prediction**
            ↓
**Performance Evaluation (R², MAE, MSE, RMSE)**

**📂 Repository Structure**
SDG2-Agricultural-Yield-Optimizer/
│
├── Dataset/
│
├── Preprocessing/
│
├── Models/
│
├── Evaluation/
│   ├── evaluation_metrics.csv
│   ├── model_comparison.csv
│
├── cleaned_X1.csv
├── y1.csv
│
└── README.md

**⚙️ Technologies Used**
- Python
- Pandas
- NumPy
- Scikit-learn
- Jupyter Notebook

**📊 Model Performance Comparison**
After evaluating multiple regression models, the following results were obtained
| Model                  | MAE    | MSE       | RMSE   | R² Score   |
| ---------------------- | ------ | --------- | ------ | ---------- |
| Linear Regression      | 200.70 | 62423.84  | 249.85 | **0.8277** |
| Random Forest          | 214.96 | 70233.92  | 265.02 | 0.8061     |
| Improved Random Forest | 217.50 | 72486.07  | 269.23 | 0.7999     |
| Advanced Model         | 549.27 | 528872.51 | 727.24 | 0.4476     |

**✅ Final Selected Model: Linear Regression**
The Linear Regression model achieved the highest R² score and the lowest overall prediction error.
- R² Score: 0.8277
- MAE: 200.70
- RMSE: 249.85
This indicates that approximately 82.77% of the variance in crop yield is explained by the model.

**🔍 Key Insights**
- Proper preprocessing significantly improved model performance
- Feature scaling reduced bias during training
- Earlier model versions showed R² ≈ 0.39
- Final optimized model improved R² to ≈ 0.83
- Linear Regression outperformed Random Forest for this dataset

  **🚀 Future Improvements**
- Incorporate temperature, humidity, and soil moisture features
- Implement advanced models (XGBoost, Gradient Boosting)
- Apply hyperparameter tuning and cross-validation
- Develop a web-based or mobile-based deployment interface
- Integrate with IoT soil sensors for real-time prediction

**🌍 Sustainable Development Goal Alignment**
This project contributes to:
**SDG 2 – Zero Hunger**
By promoting data-driven agricultural decision-making and improving crop productivity through machine learning techniques.

**👨‍💻 Internship Details**
Artificial Intelligence and Machine Learning Internship Program
International Institute of Medical Science and Technology Council
Cohort 1 – SDG Initiative

**📜 License**
This project is developed for educational and research purposes under an internship program.


