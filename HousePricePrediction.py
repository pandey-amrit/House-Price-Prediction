import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt

# Step 1: Load the dataset
data_path = 'HousePricePrediction.xlsx'  # Replace with your dataset path
data = pd.read_excel(data_path)

# Step 2: Exploratory Data Analysis (EDA)
print(data.info())
print(data.describe())

# Step 3: Handle missing values
# Drop rows where 'SalePrice' is missing
if 'SalePrice' not in data.columns:
    raise KeyError("The 'SalePrice' column is missing from the dataset.")

data = data.dropna(subset=['SalePrice'])

# Redefine numeric and categorical columns after cleaning
num_cols = data.select_dtypes(include=['float64', 'int64']).columns.difference(['SalePrice'])
cat_cols = data.select_dtypes(include=['object']).columns

# Fill missing values for numerical columns and encode categorical ones
data[num_cols] = data[num_cols].fillna(data[num_cols].median())
data[cat_cols] = data[cat_cols].fillna('Missing')

# Step 4: Define features and target
X = data.drop(columns=['SalePrice'])  # Replace 'SalePrice' with the target column name
y = data['SalePrice']

# Step 5: Preprocessing
numeric_transformer = Pipeline(steps=[
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, num_cols),
        ('cat', categorical_transformer, cat_cols)
    ]
)

# Step 6: Model Selection and Training
models = {
    'RandomForest': RandomForestRegressor(random_state=42),
    'GradientBoosting': GradientBoostingRegressor(random_state=42)
}

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

results = {}
for name, model in models.items():
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('model', model)
    ])
    
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    
    results[name] = {
        'RMSE': np.sqrt(mean_squared_error(y_test, y_pred)),
        'MAE': mean_absolute_error(y_test, y_pred),
        'R2': r2_score(y_test, y_pred)
    }

# Step 7: Hyperparameter Tuning for the Best Model
param_grid = {
    'model__n_estimators': [100, 200, 300],
    'model__max_depth': [None, 10, 20],
    'model__min_samples_split': [2, 5, 10]
}

gb_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('model', GradientBoostingRegressor(random_state=42))
])

grid_search = GridSearchCV(gb_pipeline, param_grid, cv=3, scoring='neg_mean_squared_error', n_jobs=-1)
grid_search.fit(X_train, y_train)

best_model = grid_search.best_estimator_
print("Best Parameters:", grid_search.best_params_)

# Step 8: Final Evaluation
y_pred = best_model.predict(X_test)
print("Final RMSE:", np.sqrt(mean_squared_error(y_test, y_pred)))
print("Final MAE:", mean_absolute_error(y_test, y_pred))
print("Final R2:", r2_score(y_test, y_pred))

# Step 9: Plot Feature Importances (for tree-based models)
if hasattr(best_model.named_steps['model'], 'feature_importances_'):
    feature_importances = best_model.named_steps['model'].feature_importances_
    feature_names = num_cols.tolist() + list(best_model.named_steps['preprocessor'].
                                             transformers_[1][1].named_steps['onehot'].
                                             get_feature_names_out(cat_cols))
    
    importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': feature_importances
    }).sort_values(by='Importance', ascending=False)

    plt.figure(figsize=(10, 6))
    plt.barh(importance_df['Feature'][:10], importance_df['Importance'][:10])
    plt.xlabel('Feature Importance')
    plt.ylabel('Features')
    plt.title('Top 10 Feature Importances')
    plt.gca().invert_yaxis()
    plt.show()