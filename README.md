# Heart Disease Prediction 

Predicting heart disease using multiple ML algorithms with model persistence.

## Models Compared
| Model | Accuracy | F1 Score |
|-------|----------|----------|
| KNN | 88.59% | 0.8986 |
| Logistic Regression | 87.50% | 0.8878 |
| Naive Bayes | 86.96% | 0.8788 |
| SVM (RBF Kernel) | 86.41% | 0.8804 |
| Decision Tree | 76.63% | 0.7795 |

Best Model: KNN with 88.59% accuracy

## Model Saved
Best model (KNN) saved using joblib for deployment:
- `KNN_heart.pkl` — trained model
- `scaler.pkl` — feature scaler
- `columns.pkl` — feature names

## Libraries
![Python](https://img.shields.io/badge/Python-3776AB?style=flat&logo=python&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-F89939?style=flat&logo=scikit-learn&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-150458?style=flat&logo=pandas&logoColor=E70488)
![NumPy](https://img.shields.io/badge/NumPy-013243?style=flat&logo=numpy&logoColor=4DABCF)

## Dataset
UCI Heart Disease Dataset — 303 patients, 80/20 train-test split

## Key Highlights
- 5 ML algorithms compared
- Best model saved with joblib for future deployment
- StandardScaler applied for feature normalization
