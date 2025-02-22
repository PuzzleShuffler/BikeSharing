##### Load libraries #####
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from mlxtend.regressor import StackingCVRegressor
from sklearn import set_config
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import (ElasticNet, ElasticNetCV, Lasso, LassoCV,
                                  Ridge, RidgeCV)
from sklearn.metrics import make_scorer, mean_squared_log_error
from sklearn.model_selection import RepeatedKFold, cross_val_score, train_test_split, RandomizedSearchCV
from sklearn.neighbors import LocalOutlierFactor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, PowerTransformer, RobustScaler
from sklearn.svm import SVC
from xgboost import XGBRegressor

##### Load Data #####
train = pd.read_csv('train.csv', parse_dates=['datetime'], infer_datetime_format=True)
test = pd.read_csv('test.csv', parse_dates=['datetime'], infer_datetime_format=True)
sampleSubmission = pd.read_csv('sampleSubmission.csv')

##### Cleaning Data #####
# Correct Data Types
cat = ['season', 'holiday', 'workingday']
for col in cat:
    train[col] = train[col].astype('category')
    
# Drop Atemp (Correlated with Temp and less relevant)
train = train.drop(['atemp'], axis='columns')

# Date Fields
train['dt_year'] = train['datetime'].dt.year.astype('category')
train['dt_month'] = train['datetime'].dt.month_name().astype('category')
train['dt_hour'] = train['datetime'].dt.hour.astype('category')
train['dt_day'] = train['datetime'].dt.day_name().astype('category')

test['dt_year'] = test['datetime'].dt.year.astype('category')
test['dt_month'] = test['datetime'].dt.month_name().astype('category')
test['dt_hour'] = test['datetime'].dt.hour.astype('category')
test['dt_day'] = test['datetime'].dt.day_name().astype('category')

##### Split-out Validation Dataset #####
X = train.drop(['datetime', 'casual', 'registered', 'count'], axis='columns')
y = train[['count']]

# Train & Validation Sets
X_train, X_validate, y_train, y_validate = train_test_split(X, y, test_size=0.30)

##### Model Set Ups #####
# Numeric Columns
numeric_features = X.select_dtypes(include=['int64', 'float64']).columns

# Categorical Columns
categorical_features = X.select_dtypes(include=['category', 'object']).columns

# Categorical Feature Pipeline (OneHot)
categorical_pipeline_steps = []
categorical_pipeline_steps.append(('onehot', OneHotEncoder(handle_unknown='ignore')))
categorical_pipeline = Pipeline(steps=categorical_pipeline_steps)

# Numeric Feature Pipeline (Impute, Scaler)
numeric_pipeline_steps = []
numeric_pipeline_steps.append(('scaler', RobustScaler()))
numeric_pipeline = Pipeline(steps=numeric_pipeline_steps)

# Preprocessing Transformer
preprocessing_transformer_steps = []
preprocessing_transformer_steps.append(('cat', categorical_pipeline, categorical_features))
preprocessing_transformer_steps.append(('num', numeric_pipeline, numeric_features))
preprocessing_transformer=ColumnTransformer(transformers=preprocessing_transformer_steps)

# Display Transformer
set_config(display='diagram')
preprocessing_transformer

##### Spot-Check Algorithms #####
# Store Results
results = []
names = []
scoring = 'neg_mean_squared_log_error'

# Models
models = []
models.append(('gbr', GradientBoostingRegressor(n_estimators=1000)))
models.append(('lgbm', LGBMRegressor(n_estimators=1000)))
models.append(('xgb', XGBRegressor(n_estimators=1000)))
models.append(('rf', RandomForestRegressor()))

for name, model in models:
    # Pipeline
    model_pipeline_steps = []
    model_pipeline_steps.append(('transformer', preprocessing_transformer))
    model_pipeline_steps.append(('model', model))
    model_pipeline = Pipeline(steps=model_pipeline_steps)
    transformed_model = TransformedTargetRegressor(regressor=model_pipeline, transformer=PowerTransformer(method='box-cox'))
    # CV results
    cv_results = cross_val_score(transformed_model, X_train, y_train, cv=10, scoring=scoring)
    # Changing to Root Mean Squared Logarithmic Error (RMSLE)
    cv_results = np.sqrt(abs(cv_results))
    # Append Results
    results.append(cv_results)
    names.append(name)
    # Results Mean +/- std
    msg = "%s: %f +/- %f" % (name, cv_results.mean(), cv_results.std())
    # Print Results
    print(msg)
    
# Algorithm Comparison Boxplot
fig = plt.figure()
fig.suptitle('Algorithm Comparison')
ax = fig.add_subplot(111)
plt.boxplot(results)
ax.set_xticklabels(names)
plt.show()

##### Tune Best Model, Fit on Validation Set #####
# Model 
model = LGBMRegressor()

# Scoring
scoring = 'neg_mean_squared_log_error'

# Pipelinee Steps
model_pipeline_steps = []
model_pipeline_steps.append(('transformer', preprocessing_transformer))
model_pipeline_steps.append(('model', model))
model_pipeline = Pipeline(steps=model_pipeline_steps)

# Tranform Target
transformed_model = TransformedTargetRegressor(regressor=model_pipeline, transformer=PowerTransformer(method='box-cox'))

# Param Grid
transformed_model.get_params()
param_grid = {'regressor__model__n_estimators':np.linspace(1000, 10000, 10, dtype=int),
              'regressor__model__learning_rate': [0.1, 0.01]}

# RandomizedSearch CV
rscv_model = RandomizedSearchCV(estimator=transformed_model, param_distributions=param_grid, scoring=scoring, n_iter=2, n_jobs=-1)

# Display Model
rscv_model

# Fit Model on Train
rscv_model.fit(X_train, y_train)

# Best
print('best score: %f' % np.sqrt(abs(rscv_model.best_score_)))
print('best params:', str(rscv_model.best_params_))

# Root Mean Squared Logarithmic Error (RMSLE) on Validation Set
print('RMSLE on Validation Set: %f' % np.sqrt(abs(mean_squared_log_error(y_validate, rscv_model.predict(X_validate)))))

##### Standalone Model on Entire Training Set #####
# Model
model = LGBMRegressor()

# Model Pipeline
model_pipeline_steps = []
model_pipeline_steps.append(('transformer', preprocessing_transformer))
model_pipeline_steps.append(('model', model))
model_pipeline = Pipeline(steps=model_pipeline_steps)
final_model = TransformedTargetRegressor(regressor=model_pipeline, transformer=PowerTransformer(method='box-cox')).set_params(**rscv_model.best_params_)

# Display Model
final_model

# Fit Model
final_model.fit(X,y)

# Predictions
predictions = final_model.predict(test.drop(['datetime', 'atemp'], axis='columns'))

# Submission CSV
submission = test[['datetime']]
submission['count'] = predictions
submission.to_csv('submissions/lgbm 2021-05-15.csv', index=False)

##### Create Stacking Model on Entire Training Set #####
# Models to Stack
gbr = GradientBoostingRegressor(n_estimators=5000, learning_rate=0.01)
lgbm = LGBMRegressor(n_estimators=5000, learning_rate=0.01)
xgb = XGBRegressor(n_estimators=5000, learning_rate=0.01)

# Stacking Model
model = StackingCVRegressor(regressors=(gbr, lgbm, xgb), 
                            meta_regressor=xgb, 
                            use_features_in_secondary=True, n_jobs=-1)

# Model Pipeline
model_pipeline_steps = []
model_pipeline_steps.append(('transformer', preprocessing_transformer))
model_pipeline_steps.append(('model', model))
model_pipeline = Pipeline(steps=model_pipeline_steps)

# Transform Target in Model
transformed_model = TransformedTargetRegressor(regressor=model_pipeline, transformer=PowerTransformer(method='box-cox'))

# Display Model
transformed_model

# Fit Model
transformed_model.fit(X,y)

# Predictions
predictions = transformed_model.predict(test.drop(['datetime', 'atemp'], axis='columns'))

# Submission CSV
submission = test[['datetime']]
submission['count'] = predictions
submission.to_csv('submissions/stack 2021-05-15.csv', index=False)