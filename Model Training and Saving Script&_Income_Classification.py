
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
import pickle
import warnings
warnings.filterwarnings('ignore')

# Load and prepare your data (assuming you have the dataset)
def prepare_model():
    # Create sample data similar to your notebook
    np.random.seed(42)
    n_samples = 1000
    
    # Generate base features
    data = {
        'age': np.random.randint(18, 70, n_samples),
        'education': np.random.choice(['High School', 'Bachelor', 'Master', 'PhD'], n_samples),
        'experience_years': np.random.randint(0, 40, n_samples),
        'job_category': np.random.choice(['Tech', 'Healthcare', 'Finance', 'Education', 'Other'], n_samples),
        'credit_score': np.random.randint(300, 850, n_samples)
    }
    
    df = pd.DataFrame(data)
    
    # Apply feature engineering from your notebook
    education_multiplier = {'High School': 0.8, 'Bachelor': 1.0, 'Master': 1.3, 'PhD': 1.6}
    df['education_mult'] = df['education'].map(education_multiplier)
    
    # Generate income based on features
    base_income = 30000 + df['age'] * 500 + df['experience_years'] * 800
    df['income'] = base_income * df['education_mult']
    
    # Add tech bonus
    tech_bonus = np.where(df['job_category'] == 'Tech', 1.2, 1.0)
    df['income'] = df['income'] * tech_bonus
    
    # Create income categories
    df['income_category'] = pd.cut(df['income'], 
                                  bins=[0, 40000, 70000, 100000, float('inf')],
                                  labels=['Low', 'Medium', 'High', 'Very High'])
    
    # Prepare features and target
    features = ['age', 'experience_years', 'credit_score', 'education_mult']
    X = df[features]
    y = df['income_category']
    
    # Handle missing values
    imputer = SimpleImputer(strategy='median')
    X_imputed = imputer.fit_transform(X)
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_imputed)
    
    # Train model
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
    
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Save model and preprocessors
    with open('income_model.pkl', 'wb') as f:
        pickle.dump(model, f)
    
    with open('scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)
        
    with open('imputer.pkl', 'wb') as f:
        pickle.dump(imputer, f)
    
    print("Model trained and saved successfully!")
    print(f"Model accuracy: {model.score(X_test, y_test):.3f}")

if __name__ == "__main__":
    prepare_model()
