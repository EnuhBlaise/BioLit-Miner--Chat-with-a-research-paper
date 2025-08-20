# Machine Learning Analysis Template
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.svm import SVC, SVR
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import mean_squared_error, r2_score, classification_report, confusion_matrix

def load_and_preprocess_data(file_path, target_column):
    """Load and preprocess data for machine learning"""
    df = pd.read_csv(file_path)
    
    # Handle missing values
    df = df.dropna()
    
    # Separate features and target
    X = df.drop(columns=[target_column])
    y = df[target_column]
    
    # Encode categorical variables
    categorical_cols = X.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col])
    
    return X, y

def classification_pipeline(X, y, test_size=0.2, random_state=42):
    """Complete classification pipeline"""
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Define models
    models = {
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=random_state),
        'Logistic Regression': LogisticRegression(random_state=random_state),
        'SVM': SVC(random_state=random_state)
    }
    
    results = {}
    
    for name, model in models.items():
        # Train model
        if name == 'Random Forest':
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
        else:
            model.fit(X_train_scaled, y_train)
            y_pred = model.predict(X_test_scaled)
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted')
        recall = recall_score(y_test, y_pred, average='weighted')
        f1 = f1_score(y_test, y_pred, average='weighted')
        
        results[name] = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'model': model
        }
        
        print(f"\n{name} Results:")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1 Score: {f1:.4f}")
    
    return results, X_test, y_test, scaler

def regression_pipeline(X, y, test_size=0.2, random_state=42):
    """Complete regression pipeline"""
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Define models
    models = {
        'Random Forest': RandomForestRegressor(n_estimators=100, random_state=random_state),
        'Linear Regression': LinearRegression(),
        'SVR': SVR()
    }
    
    results = {}
    
    for name, model in models.items():
        # Train model
        if name == 'Random Forest':
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
        else:
            model.fit(X_train_scaled, y_train)
            y_pred = model.predict(X_test_scaled)
        
        # Calculate metrics
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test, y_pred)
        
        results[name] = {
            'mse': mse,
            'rmse': rmse,
            'r2_score': r2,
            'model': model
        }
        
        print(f"\n{name} Results:")
        print(f"MSE: {mse:.4f}")
        print(f"RMSE: {rmse:.4f}")
        print(f"R² Score: {r2:.4f}")
    
    return results, X_test, y_test, scaler

def feature_importance_analysis(model, feature_names):
    """Analyze feature importance for tree-based models"""
    if hasattr(model, 'feature_importances_'):
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        plt.figure(figsize=(10, 6))
        sns.barplot(data=importance_df.head(10), x='importance', y='feature')
        plt.title('Top 10 Feature Importances')
        plt.xlabel('Importance')
        plt.show()
        
        return importance_df
    else:
        print("Model does not support feature importance analysis")
        return None

def hyperparameter_tuning(X, y, model, param_grid, cv=5):
    """Perform hyperparameter tuning using GridSearchCV"""
    grid_search = GridSearchCV(
        model, param_grid, cv=cv, scoring='accuracy', n_jobs=-1
    )
    grid_search.fit(X, y)
    
    print(f"Best parameters: {grid_search.best_params_}")
    print(f"Best cross-validation score: {grid_search.best_score_:.4f}")
    
    return grid_search.best_estimator_

def cross_validation_analysis(X, y, models, cv=5):
    """Perform cross-validation analysis"""
    cv_results = {}
    
    for name, model in models.items():
        scores = cross_val_score(model, X, y, cv=cv, scoring='accuracy')
        cv_results[name] = {
            'mean_score': scores.mean(),
            'std_score': scores.std(),
            'scores': scores
        }
        
        print(f"\n{name} Cross-Validation Results:")
        print(f"Mean Score: {scores.mean():.4f} (+/- {scores.std() * 2:.4f})")
    
    return cv_results

# Template usage example
if __name__ == "__main__":
    # Load and preprocess data
    # X, y = load_and_preprocess_data("your_data_file.csv", "target_column")
    
    # For classification
    # results, X_test, y_test, scaler = classification_pipeline(X, y)
    
    # For regression
    # results, X_test, y_test, scaler = regression_pipeline(X, y)
    
    # Feature importance analysis
    # best_model = results['Random Forest']['model']
    # feature_importance_analysis(best_model, X.columns)
    
    # Hyperparameter tuning example
    # param_grid = {'n_estimators': [50, 100, 200], 'max_depth': [3, 5, 7]}
    # best_rf = hyperparameter_tuning(X, y, RandomForestClassifier(), param_grid)
    
    print("Machine learning analysis template ready for customization")