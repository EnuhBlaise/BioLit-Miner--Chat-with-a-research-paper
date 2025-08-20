# Statistical Analysis Template
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

def load_data(file_path):
    """Load data from various formats"""
    if file_path.endswith('.csv'):
        return pd.read_csv(file_path)
    elif file_path.endswith('.xlsx'):
        return pd.read_excel(file_path)
    elif file_path.endswith('.json'):
        return pd.read_json(file_path)
    else:
        raise ValueError("Unsupported file format")

def descriptive_statistics(df, columns=None):
    """Generate descriptive statistics"""
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns
    
    stats_df = df[columns].describe()
    print("Descriptive Statistics:")
    print(stats_df)
    return stats_df

def correlation_analysis(df, method='pearson'):
    """Perform correlation analysis"""
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    corr_matrix = df[numeric_cols].corr(method=method)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0)
    plt.title(f'{method.title()} Correlation Matrix')
    plt.show()
    
    return corr_matrix

def t_test_analysis(df, group_col, value_col, alpha=0.05):
    """Perform independent t-test"""
    groups = df[group_col].unique()
    if len(groups) != 2:
        raise ValueError("T-test requires exactly 2 groups")
    
    group1 = df[df[group_col] == groups[0]][value_col]
    group2 = df[df[group_col] == groups[1]][value_col]
    
    t_stat, p_value = stats.ttest_ind(group1, group2)
    
    print(f"T-test Results:")
    print(f"T-statistic: {t_stat:.4f}")
    print(f"P-value: {p_value:.4f}")
    print(f"Significant at α = {alpha}: {p_value < alpha}")
    
    return t_stat, p_value

def anova_analysis(df, group_col, value_col, alpha=0.05):
    """Perform one-way ANOVA"""
    groups = [group[value_col].values for name, group in df.groupby(group_col)]
    f_stat, p_value = stats.f_oneway(*groups)
    
    print(f"ANOVA Results:")
    print(f"F-statistic: {f_stat:.4f}")
    print(f"P-value: {p_value:.4f}")
    print(f"Significant at α = {alpha}: {p_value < alpha}")
    
    return f_stat, p_value

def linear_regression_analysis(df, target_col, feature_cols):
    """Perform linear regression analysis"""
    from sklearn.linear_model import LinearRegression
    from sklearn.metrics import r2_score, mean_squared_error
    
    X = df[feature_cols]
    y = df[target_col]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    
    print(f"Linear Regression Results:")
    print(f"R² Score: {r2:.4f}")
    print(f"Mean Squared Error: {mse:.4f}")
    print(f"Coefficients: {dict(zip(feature_cols, model.coef_))}")
    print(f"Intercept: {model.intercept_:.4f}")
    
    return model, r2, mse

# Template usage example
if __name__ == "__main__":
    # Load your data
    # df = load_data("your_data_file.csv")
    
    # Perform analyses
    # descriptive_statistics(df)
    # correlation_analysis(df)
    # t_test_analysis(df, 'group_column', 'value_column')
    # anova_analysis(df, 'group_column', 'value_column')
    # linear_regression_analysis(df, 'target_column', ['feature1', 'feature2'])
    
    print("Statistical analysis template ready for customization")