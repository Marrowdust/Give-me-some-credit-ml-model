import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import StandardScaler, RobustScaler, PowerTransformer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, StackingClassifier
from sklearn.metrics import (classification_report, confusion_matrix, roc_auc_score,
                           roc_curve, precision_recall_curve, average_precision_score,
                           accuracy_score, precision_score, recall_score, f1_score)
from imblearn.over_sampling import SMOTENC, ADASYN
from imblearn.combine import SMOTETomek
from imblearn.pipeline import Pipeline
import matplotlib.pyplot as plt
import seaborn as sns

np.random.seed(42)

def load_and_process_data(file_path):
    """Load and process the Give Me Some Credit dataset with enhanced feature engineering."""
    df = pd.read_csv(file_path)
    
    if 'Unnamed: 0' in df.columns:
        df = df.drop('Unnamed: 0', axis=1)
    
    # Enhanced outlier handling
    df['age'] = df['age'].clip(18, 100)
    df['RevolvingUtilizationOfUnsecuredLines'] = df['RevolvingUtilizationOfUnsecuredLines'].clip(0, 10)
    df['DebtRatio'] = df['DebtRatio'].clip(0, 20)
    
    # Handle missing values with more sophisticated approach
    df['MonthlyIncome'] = df.groupby('age')['MonthlyIncome'].transform(
        lambda x: x.fillna(x.median())
    )
    df['NumberOfDependents'] = df['NumberOfDependents'].fillna(
        df.groupby('age')['NumberOfDependents'].transform('median')
    )
    
    # Advanced feature engineering
    df['TotalNumberOfDelinquencies'] = (
        df['NumberOfTime30-59DaysPastDueNotWorse'] +
        df['NumberOfTime60-89DaysPastDueNotWorse'] +
        df['NumberOfTimes90DaysLate']
    )
    
    df['DebtToIncome'] = df['DebtRatio'] * df['MonthlyIncome']
    df['UtilizationToIncome'] = df['RevolvingUtilizationOfUnsecuredLines'] * df['MonthlyIncome']
    df['IncomePerDependent'] = df['MonthlyIncome'] / (df['NumberOfDependents'] + 1)
    
    # Create interaction features
    df['UtilizationDebtRatio'] = df['RevolvingUtilizationOfUnsecuredLines'] * df['DebtRatio']
    df['DelinquencyDebtRatio'] = df['TotalNumberOfDelinquencies'] * df['DebtRatio']
    
    # Create binary flags
    df['HasHighUtilization'] = (df['RevolvingUtilizationOfUnsecuredLines'] > 0.5).astype(int)
    df['HasDelinquency'] = (df['TotalNumberOfDelinquencies'] > 0).astype(int)
    df['HasHighDebtRatio'] = (df['DebtRatio'] > 1).astype(int)
    
    # Log transform highly skewed features
    skewed_features = ['RevolvingUtilizationOfUnsecuredLines', 'DebtRatio', 'MonthlyIncome', 'DebtToIncome']
    for feature in skewed_features:
        df[f'{feature}_Log'] = np.log1p(df[feature])
    
    return df

def train_optimized_model(X_train, y_train, X_test, y_test):
    """Train an optimized stacking model with multiple base learners."""
    
    # Base learners
    dt = DecisionTreeClassifier(random_state=42)
    rf = RandomForestClassifier(random_state=42, n_jobs=-1)
    gb = GradientBoostingClassifier(random_state=42)
    
    # Define parameter grids for each model
    dt_params = {
        'max_depth': [10, 15, 20],
        'min_samples_split': [20, 50, 100],
        'min_samples_leaf': [10, 20, 50],
        'class_weight': ['balanced']
    }
    
    rf_params = {
        'n_estimators': [100, 200],
        'max_depth': [10, 15, 20],
        'min_samples_split': [20, 50],
        'min_samples_leaf': [10, 20],
        'class_weight': ['balanced']
    }
    
    gb_params = {
        'n_estimators': [100, 200],
        'max_depth': [5, 10],
        'learning_rate': [0.05, 0.1],
        'min_samples_split': [20, 50],
        'min_samples_leaf': [10, 20]
    }
    
    # Optimize each base learner
    dt_opt = GridSearchCV(dt, dt_params, cv=5, scoring='roc_auc', n_jobs=-1)
    rf_opt = GridSearchCV(rf, rf_params, cv=5, scoring='roc_auc', n_jobs=-1)
    gb_opt = GridSearchCV(gb, gb_params, cv=5, scoring='roc_auc', n_jobs=-1)
    
    # Train optimized base learners
    dt_opt.fit(X_train, y_train)
    rf_opt.fit(X_train, y_train)
    gb_opt.fit(X_train, y_train)
    
    # Create stacking classifier
    estimators = [
        ('dt', dt_opt.best_estimator_),
        ('rf', rf_opt.best_estimator_),
        ('gb', gb_opt.best_estimator_)
    ]
    
    stack = StackingClassifier(
        estimators=estimators,
        final_estimator=GradientBoostingClassifier(
            n_estimators=100,
            learning_rate=0.05,
            max_depth=5,
            random_state=42
        ),
        cv=5
    )
    
    # Train stacking classifier
    stack.fit(X_train, y_train)
    
    # Make predictions
    y_pred = stack.predict(X_test)
    y_prob = stack.predict_proba(X_test)[:, 1]
    
    return stack, y_pred, y_prob

def main():
    print("Loading and processing data...")
    full_df = load_and_process_data('/kaggle/input/GiveMeSomeCredit/cs-training.csv')
    
    # Split features and target
    X = full_df.drop('SeriousDlqin2yrs', axis=1)
    y = full_df['SeriousDlqin2yrs']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=0.2, 
        random_state=42,
        stratify=y
    )
    
    # Scale features using RobustScaler (better for outliers)
    scaler = RobustScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns)
    X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_test.columns)
    
    # Apply SMOTETomek for better balanced data
    smote_tomek = SMOTETomek(random_state=42)
    X_train_balanced, y_train_balanced = smote_tomek.fit_resample(X_train_scaled, y_train)
    
    print("Training optimized model...")
    model, y_pred, y_prob = train_optimized_model(
        X_train_balanced, y_train_balanced,
        X_test_scaled, y_test
    )
    
    # Calculate and display metrics
    metrics = {
        'Accuracy': accuracy_score(y_test, y_pred),
        'Precision': precision_score(y_test, y_pred),
        'Recall': recall_score(y_test, y_pred),
        'F1 Score': f1_score(y_test, y_pred),
        'ROC AUC': roc_auc_score(y_test, y_prob)
    }
    
    print("\nModel Performance Metrics:")
    print("-" * 50)
    for metric, value in metrics.items():
        print(f"{metric}: {value:.3f}")
    
    print("\nDetailed Classification Report:")
    print("-" * 50)
    print(classification_report(y_test, y_pred))
    
    return model, metrics

if __name__ == "__main__":
    model, metrics = main()