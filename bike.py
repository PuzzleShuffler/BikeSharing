##### Load libraries #####
from sklearn.ensemble import GradientBoostingRegressor
import numpy as np
from sklearn.metrics import make_scorer
from sklearn.model_selection import RepeatedKFold
from sklearn.preprocessing import PowerTransformer
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import LocalOutlierFactor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import Lasso
from sklearn.metrics import mean_squared_log_error
from sklearn.svm import SVC
from skopt import BayesSearchCV

##### Load dataset #####
train = pd.read_csv('train.csv', parse_dates=['datetime'], infer_datetime_format=True)
test = pd.read_csv('test.csv', parse_dates=['datetime'], infer_datetime_format=True)
sampleSubmission = pd.read_csv('sampleSubmission.csv')

##### Summarize Data #####
# Head of Data
train.head()

# Shape of Data
train.shape

# Data Information
train.info()
train.dtypes

# Correct Data Types
cat = ['season', 'holiday', 'workingday']
for col in cat:
    train[col] = train[col].astype('category')

num_col = list(train.select_dtypes(include=['int64', 'float64']).columns)
num_col.remove('casual')
num_col.remove('registered')
num_col.remove('count')

cat_col = list(train.select_dtypes(include=['category', 'object']).columns)

# Distribution of Numeric Columns
for col in num_col:
    print('Distribution of %s is:' % (col))
    print(train[col].describe())
    print("")

# Class Distrubtion of Categorical Columns
for col in cat_col:
    print('Class Distribution of %s is:' % (col))
    print(train.groupby([col]).size())
    print("")

# Describe all Data
train.describe(include='all')

# Correlation of Numeric Columns
train[num_col].corr()

# Skewness of Numeric Columns
train[num_col].skew()

##### Data Visualizations #####
# Histograms of Numeric Columns
for col in num_col:
    sns.histplot(train[col], kde=True)
    plt.show()
print("")

# Count Plots of Categorical Columns
for col in cat_col:
    sns.countplot(train[col])
    plt.show()
print("")

# Correlation Heat Map of Numeric Columns
sns.heatmap(train[num_col].corr(), cmap="coolwarm", annot=True)
plt.show()
print("")

##### Data Cleaning #####
# Number of unique values in each column
print('Columns and their number of unique values:')
print(train.nunique())
print("")

# Outliers
lof = LocalOutlierFactor()
yhat = lof.fit_predict(train[num_col])

mask = yhat == -1  # mask for outliers, 1 is normal, -1 is outlier
print('There are %d outliers according to LocalOutlierFactor.' % len(train[mask]))
print(train[mask])  # Outliers
print("")

mask = yhat == 1  # mask for normal, 1 is normal, -1 is outlier
outlierFreeData = train[mask]

# Missing Values
print('There are %d missing pieces of data.' %
      outlierFreeData.isna().sum().sum())
outlierFreeData.isna().sum()
print("")

# Drop Atemp (Correlated with Temp and less relevant)
outlierFreeData = outlierFreeData.drop(['atemp'], axis='columns')

# Date Fields (Train)
outlierFreeData['dt_year'] = outlierFreeData['datetime'].dt.year.astype('category')
outlierFreeData['dt_month'] = outlierFreeData['datetime'].dt.month_name().astype('category')
outlierFreeData['dt_hour'] = outlierFreeData['datetime'].dt.hour.astype('category')
outlierFreeData['dt_day'] = outlierFreeData['datetime'].dt.day_name().astype('category')

# Date Fields (Test)
test['dt_year'] = test['datetime'].dt.year.astype('category')
test['dt_month'] = test['datetime'].dt.month_name().astype('category')
test['dt_hour'] = test['datetime'].dt.hour.astype('category')
test['dt_day'] = test['datetime'].dt.day_name().astype('category')

# Split X, Y
X = outlierFreeData.drop(['datetime', 'casual', 'registered', 'count'], axis='columns')
y = outlierFreeData[['count']]

# Numeric Columns
num_col = list(X.select_dtypes(include=['int64', 'float64']).columns)

# Categorical Columns
cat_col = list(X.select_dtypes(include=['category', 'object']).columns)

# Model
model = GradientBoostingRegressor()
transformer = ColumnTransformer(transformers=[('cat', OneHotEncoder(), cat_col), ('num', StandardScaler(), num_col)], remainder='passthrough')
pipeline = Pipeline(steps=[('prep', transformer), ('m', model)])
transformed_model = TransformedTargetRegressor(regressor=pipeline, transformer=PowerTransformer())
cv = RepeatedKFold(n_splits=10, n_repeats=2, random_state=1)

param_grid = {'regressor__m__learning_rate': np.linspace(0.01, 0.3, 50), 'regressor__m__n_estimators': np.linspace(50, 1000, 50, dtype='int')}
opt = BayesSearchCV(transformed_model, search_spaces=param_grid, n_iter=10, refit=True, verbose=True, cv=10, n_jobs=-1)
opt.fit(X,y)

# Best
print('val score: ', opt.best_score_)
print('best params: ', str(opt.best_params_))

# Predictions on Test
pred = opt.predict(test.drop(['datetime', 'atemp'], axis='columns'))

# Submission CSV
submission = test[['datetime']]
submission['count'] = pred
submission['count'] = np.where(submission['count'] < 0, 0, submission['count'])
submission.to_csv('submissions/GradientBoostingRegressor 2021-05-07.csv', index=False)
