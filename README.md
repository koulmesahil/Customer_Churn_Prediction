
# Customer Churn Prediction

## 1. **Introduction**
Customer churn prediction is a critical task for businesses aiming to retain customers and improve their service offerings. In this project, I explore the problem of customer churn using two distinct datasets. One dataset has more **categorical attributes**, while the other contains more **numerical attributes**. This allows for a comparative analysis of how different machine learning models perform when dealing with these different data types. 

The goal of this project is to build accurate and effective models for predicting customer churn, while also exploring the performance of various algorithms on different data types. This will provide insights into the strengths and limitations of different models and how to optimize them for different data structures.

## 2. **Datasets**
The project utilizes two datasets found on kaggle, both focused on customer churn prediction:

- **Dataset 1: Categorical-heavy Dataset**
  - This dataset contains primarily categorical attributes (e.g., customer type, payment methods, service types).
  - Requires encoding techniques such as One-Hot Encoding or Label Encoding to convert categorical variables into a format suitable for machine learning models.
  
- **Dataset 2: Numerical-heavy Dataset**
  - This dataset contains more numerical attributes (e.g., customer tenure, total charges, monthly charges).
  - Standardization or normalization is applied to the numerical features to ensure uniformity during model training.

## 3. **Features**
   - **Comparative Analysis**: By working with two different datasets (one categorical-heavy and one numerical-heavy), we can compare how various models, such as decision trees, logistic regression, and gradient boosting, perform on each dataset.
   - **Data Preprocessing**: Appropriate techniques such as label encoding, one-hot encoding, standardization, and normalization were used based on the nature of each dataset.
   - **Model Optimization**: Techniques like hyperparameter tuning (GridSearchCV) and cross-validation were used to ensure the best performance for each model on both datasets.
   - **Evaluation Metrics**: The models were evaluated using several metrics, including accuracy, precision, recall, F1-score, and AUC-ROC curve.

## 4. **Technologies Used**
   - **Python**: Programming language used for model development and analysis.
   - **Pandas**: For data manipulation and preprocessing.
   - **Scikit-learn**: For implementing machine learning algorithms and model evaluation.
   - **XGBoost**: For implementing gradient boosting algorithms.
   - **Matplotlib/Seaborn**: For data visualization and performance evaluation metrics (e.g., confusion matrix, ROC curve).


## 5. **Data Preprocessing**
   - **Categorical-heavy Dataset**:
     - Used **One-Hot Encoding** for categorical features with multiple categories.
     - Applied **Label Encoding** for binary categorical features.
     - Handled missing data using imputation techniques, such as filling with mode or median values.
  ![image](https://github.com/user-attachments/assets/22f7e0b5-fbe6-4d0f-8ba5-dfbc83d7f163)
![image](https://github.com/user-attachments/assets/94baa997-7752-4c57-b43a-21d2411fe130)


   - **Numerical-heavy Dataset**:
     - Applied **Standardization** to ensure all numerical features are on the same scale.
     - **Feature Engineering**: Created new features where applicable (e.g., interaction terms between different numerical attributes).
       ![image](https://github.com/user-attachments/assets/b4872e18-4cd6-4b64-9af8-2a5097617a1f)
       ![image](https://github.com/user-attachments/assets/826dae6f-227c-41ec-b4f2-eb2f18ac8f06)



## 6. **Modeling**
   For both datasets, the following machine learning models were applied and optimized:
   
   - **Logistic Regression**: 
     - Simple yet effective for binary classification problems like churn prediction.
     - Applied regularization techniques such as L2 to prevent overfitting.
   
   - **Decision Trees**:
     - Used to handle both numerical and categorical data effectively.
     - Focused on interpretability by visualizing tree structures.
   
   - **Random Forest**:
     - Built as an ensemble of decision trees to improve robustness and accuracy.
     - Tuned parameters like the number of trees and depth for optimal performance.

   - **XGBoost**:
     - A powerful gradient-boosting algorithm known for handling both categorical and numerical data well.
     - Tuned parameters such as learning rate, tree depth, and number of estimators for improved performance.
![image](https://github.com/user-attachments/assets/86c70423-4a9b-4f68-9d64-e422f651e5be)
![image](https://github.com/user-attachments/assets/8890fa8e-98be-4b8e-b679-8cccf628485f)

## 7. **Results**
   - **Categorical-heavy Dataset**:
     - **Best Model**: Random Forest achieved the best performance on this dataset, showing high accuracy and good interpretability for categorical data.
     - **Evaluation**: The Random Forest model achieved an **accuracy of 85%**, with a strong performance in recall and F1-score, demonstrating its ability to handle categorical features effectively.

   - **Numerical-heavy Dataset**:
     - **Best Model**: XGBoost outperformed other models, achieving the highest **accuracy of 90%** on the numerical dataset.
     - **Evaluation**: The XGBoost model demonstrated excellent performance across all metrics, particularly AUC-ROC, indicating strong predictive power for numerical features.

   - **Model Comparison**:
     - Random Forest performed better on categorical-heavy data, while XGBoost excelled with numerical-heavy data. This highlights the importance of model selection based on data characteristics.

## 8. **Visualization**
   - The project includes a variety of visualizations to illustrate model performance:
     - **Confusion Matrix**: Displays the performance of each model in terms of true positives, false positives, true negatives, and false negatives.
     - **ROC Curve**: Shows the trade-off between true positive rate and false positive rate, giving a visual understanding of model performance.
     - **Feature Importance**: For models like Random Forest and XGBoost, feature importance scores are provided to show which features contributed most to the prediction.

## 9. **Conclusion**
   Through this project, we explored the impact of different data types (categorical vs. numerical) on machine learning models for churn prediction. Random Forest showed its strength in handling categorical-heavy datasets, while XGBoost excelled with numerical-heavy datasets. By optimizing each model and preprocessing the data appropriately, we were able to build high-performing churn prediction models.

## 10. **References**
   - **Scikit-learn Documentation**: https://scikit-learn.org/stable/
   - **XGBoost Documentation**: https://xgboost.readthedocs.io/en/latest/
   - **Kaggle**: Used as a source for the datasets and initial inspiration for the problem-solving approach.

---

