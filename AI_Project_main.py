
import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
import os
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from xgboost import XGBRegressor

# Define the base directory of the project to locate files relatively
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Global variables to store trained model and preprocessors
# These will be populated by a function and then used by the Flask app
_xgb_model = None
_lr_model = None
_encoders = None
_feature_columns = None
_scaler = None
_cat_cols = ['Fuel', 'Transmission', 'Assembly', 'BodyType', 'Color', 'City']

def _print_dataset_description():
    print("=" * 60)
    print("DATASET DESCRIPTION & FEATURES")
    print("=" * 60)
    print("""Dataset : PakWheels Used-Car Listings
Source  : Scraped from https://www.pakwheels.com (Pakistan's
          largest online car marketplace).
Domain  : Automotive / E-Commerce

FEATURES
-------------------------------------------------------------
Column           Type         Description
-------------------------------------------------------------
nam              Categorical  Car make, model, variant & year
Price            Numerical*   Asking price (raw text, e.g. 'PKR 23.9 lacs')
Year             Numerical    Manufacturing year (integer)
Millage          Numerical*   Odometer reading (raw text, e.g. '120,000 km')
Fuel             Categorical  Fuel type (Petrol/Diesel/CNG/LPG/Electric)
Transmission     Categorical  Gearbox type (Manual / Automatic)
Province         Categorical  City/Province of the seller
Color            Categorical  Exterior colour
Assembly         Categorical  Local or Imported assembly
EngineCapacity   Numerical*   Engine displacement (raw text, e.g. '660 cc')
BodyType         Categorical  Body style (Hatchback/Sedan/SUV...)
AdReference      Numerical    Unique numeric ad ID (dropped - identifier only)
Features         Text         Comma-separated list of car features
OwnerNam         Categorical  Seller name (dropped - PII)
url              Text         Ad URL (dropped - identifier only)

* = stored as string in raw CSV; converted to float during preprocessing""")


def fix_price_col(p):
    """Convert price strings like 'PKR 23.9 lacs' / '1.2 Crore' to float (lacs)."""
    if not isinstance(p, str):
        return np.nan
    p = p.lower()
    numbers = re.findall(r'[\d.]+', p)
    if not numbers:
        return np.nan

    number = float(numbers[0])
    if 'crore' in p or 'cr' in p:
        return number * 100          # 1 crore = 100 lacs
    return number


def fix_capacity_col(p):
    """Convert engine strings like '660 cc' to float (cc)."""
    if not isinstance(p, str):
        return np.nan
    numbers = re.findall(r'[\d]+', p)
    if not numbers:
        return np.nan
    return float(numbers[0])


def fix_mileage_col(p):
    """Convert mileage strings like '120,000 km' to float (km)."""
    if not isinstance(p, str):
        return np.nan
    numbers = re.findall(r'[\d,]+', p)
    if not numbers:
        return np.nan
    return float(numbers[0].replace(',', ''))


def remove_outliers_iqr(df, col):
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR

    n_out = ((df[col] < lower) | (df[col] > upper)).sum()
    print(f"  {col}: {n_out} outliers | valid range [{lower:.1f}, {upper:.1f}]")

    return df[(df[col] >= lower) & (df[col] <= upper)]


def initialize_model_components(file_path=None):
    """
    Loads, preprocesses, and splits the data, trains models, and populates
    the global variables for the trained models, scaler, encoders, and feature columns.
    """
    global _xgb_model, _lr_model, _encoders, _feature_columns, _scaler

    if file_path is None:
        file_path = os.path.join(BASE_DIR, 'data.csv')

    df = pd.read_csv(file_path)
    # print(f"Original shape : {df.shape}  ({df.shape[0]} rows x {df.shape[1]} columns)")
    df.drop(columns=['url', 'OwnerNam', 'AdReference'], inplace=True)
    df.rename(columns={'nam': 'Name'}, inplace=True)

    # print("\nDatatypes Before conversion:")
    # print(df.dtypes)

    df['Price'] = df['Price'].apply(fix_price_col)
    df['EngineCapacity'] = df['EngineCapacity'].apply(fix_capacity_col)
    df['Millage'] = df['Millage'].apply(fix_mileage_col)

    for c in _cat_cols:
        if c in df.columns:
            df[c] = df[c].astype('category')

    # print("\nDatatypes After conversion:")
    # print(df.dtypes)

    # print("\nMissing values after conversion of datatypes:")
    # print(df.isnull().sum())

    for col in ['BodyType', 'Color', 'Features']: # fixing missing
        mode_val = df[col].mode()[0]
        df[col] = df[col].fillna(mode_val)

    df['Province'] = df['Province'].str.strip().str.title()
    translation_map = {
        'Punjab'      : 'Lahore',
        'Sindh'       : 'Karachi',
        'Kpk'         : 'Peshawar',
        'Balochistan' : 'Quetta',
        'Ict'         : 'Islamabad',
        'Isb'         : 'Islamabad',
        'Khi'         : 'Karachi',
        'Lhr'         : 'Lahore',
        'Fsd'         : 'Faisalabad',
        'Mul'         : 'Multan',
    }
    df['City'] = df['Province'].replace(translation_map)
    df = df[df['City'] != 'Un-Registered'] # dropping records having Un-Registered city
    df['City'] = df['City'].astype('category')
    df.drop(columns=['Province'], inplace=True) # drop province because we have city

    df = df.dropna() # droping null
    df.drop_duplicates(inplace=True) # droping duplicates

    # print("\nMissing values AFTER imputation:")
    # print(df.isnull().sum())

    # filtering dataset
    df = df[df['Millage'] >  100.0] # atleast 100km driven
    df = df[df['Price'] >= 10.0] # at least price is 10 lacs
    df = df[df['EngineCapacity'] >= 300]   # cc must be >=300

    # Outlier removal
    numeric_cols_for_outliers = ['Price', 'Millage', 'EngineCapacity']
    for col in numeric_cols_for_outliers:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
        df = df[(df[col] >= lower) & (df[col] <= upper)]

    # print(f"\nFinal dataset shape : {df.shape}  ({df.shape[0]} rows x {df.shape[1]} columns)")
    # print("\nColumn data types:")
    # print(df.dtypes)

    # Feature Selection and Encoding
    X = df.drop(columns=['Price', 'Name', 'Features'])
    y = df['Price']

    _encoders = {}
    for col in _cat_cols:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str))
        _encoders[col] = le

    _feature_columns = X.columns

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    _scaler = StandardScaler()
    X_train_scaled = _scaler.fit_transform(X_train)
    X_test_scaled = _scaler.transform(X_test)

    # Initialize and train the XGBoost Regressor
    _xgb_model = XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=6, random_state=42)
    _xgb_model.fit(X_train_scaled, y_train)

    # Initialize and train the Linear Regression model
    _lr_model = LinearRegression()
    _lr_model.fit(X_train_scaled, y_train)

    print("Models and preprocessors initialized and trained successfully.")
    return _xgb_model, _lr_model, _scaler, _encoders, _feature_columns, X_test_scaled, y_test, df

def generate_and_save_charts(df, X_test_scaled, y_test, xgb_model):
    """Generates and saves visual analysis charts to the static directory."""
    static_charts_dir = os.path.join(BASE_DIR, 'static', 'images', 'charts')
    os.makedirs(static_charts_dir, exist_ok=True)
    
    # Set dark theme for charts
    plt.style.use('dark_background')
    
    # Chart 1: Actual vs Predicted
    y_pred_xgb = xgb_model.predict(X_test_scaled)
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, y_pred_xgb, alpha=0.5, color='#00f2fe')
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], '#f093fb', lw=2, linestyle='--')
    plt.xlabel('Actual Price (Lacs)')
    plt.ylabel('Predicted Price (Lacs)')
    plt.title('Model Performance: Actual vs Predicted Prices', pad=20, fontsize=14, color='#00f2fe')
    plt.grid(True, alpha=0.1)
    plt.savefig(os.path.join(static_charts_dir, 'actual_vs_predicted.png'), bbox_inches='tight', transparent=True)
    plt.close()

    # Chart 2: Price Distribution
    plt.figure(figsize=(10, 6))
    sns.histplot(df['Price'], kde=True, color='#4facfe', bins=30)
    plt.title('Market Price Distribution', pad=20, fontsize=14, color='#4facfe')
    plt.xlabel('Price (Lacs)')
    plt.ylabel('Count')
    plt.grid(True, alpha=0.1)
    plt.savefig(os.path.join(static_charts_dir, 'price_distribution.png'), bbox_inches='tight', transparent=True)
    plt.close()

    # Chart 3: Average Price by Body Type
    plt.figure(figsize=(12, 6))
    avg_price_body = df.groupby('BodyType')['Price'].mean().sort_values(ascending=False)
    avg_price_body.plot(kind='bar', color=['#00f2fe', '#4facfe', '#f093fb'], alpha=0.8)
    plt.title('Average Price by Body Type', pad=20, fontsize=14, color='#f093fb')
    plt.xlabel('Body Type')
    plt.ylabel('Average Price (Lacs)')
    plt.xticks(rotation=45)
    plt.grid(True, axis='y', alpha=0.1)
    plt.savefig(os.path.join(static_charts_dir, 'body_type_analysis.png'), bbox_inches='tight', transparent=True)
    plt.close()

    # Chart 4: Average Price by Year
    plt.figure(figsize=(12, 6))
    avg_price_year = df.groupby('Year')['Price'].mean()
    plt.plot(avg_price_year.index, avg_price_year.values, marker='o', color='#00f2fe', linewidth=2)
    plt.fill_between(avg_price_year.index, avg_price_year.values, alpha=0.2, color='#00f2fe')
    plt.title('Price Trend by Manufacturing Year', pad=20, fontsize=14, color='#00f2fe')
    plt.xlabel('Year')
    plt.ylabel('Average Price (Lacs)')
    plt.grid(True, alpha=0.1)
    plt.savefig(os.path.join(static_charts_dir, 'year_trend.png'), bbox_inches='tight', transparent=True)
    plt.close()

    print(f"Charts saved successfully in {static_charts_dir}")

def predict_single_car_price(car_data):
    """
    Predicts the price for a single car given its features using the globally loaded XGBoost model.
    car_data should be a dictionary with keys matching feature names.
    """
    if _xgb_model is None or _scaler is None or _encoders is None or _feature_columns is None:
        raise RuntimeError("Models and preprocessors not loaded. Call initialize_model_components() first.")

    sample_df = pd.DataFrame([car_data])

    # Apply Label Encoding to categorical features
    for col in _cat_cols:
        if col in sample_df.columns:
            val = str(sample_df[col].iloc[0])
            try:
                sample_df[col] = _encoders[col].transform([val])
            except ValueError:
                # Handle unseen categories: assign a default value (e.g., -1 or 0)
                # or raise an error. For simplicity, we'll assign -1.
                print(f"Warning: Category '{val}' for column '{col}' not seen during training. Assigning -1.")
                sample_df[col] = -1

    # Ensure column order matches the training data
    # This is crucial for consistent predictions
    processed_sample = sample_df[_feature_columns]

    # Scale numerical features
    scaled_sample = _scaler.transform(processed_sample)

    # Predict using the XGBoost model
    prediction = _xgb_model.predict(scaled_sample)[0]
    return prediction


if __name__ == "__main__":
    _print_dataset_description()

    # Initialize models and preprocessors
    xgb_model, lr_model, scaler, encoders, feature_columns, X_test_scaled, y_test, df = initialize_model_components()

    print("\n" + "="*40)
    print("MODEL TRAINING & EVALUATION (Standalone Script)")
    print("="*40)

    # --- Model 1: XGBoost Regressor ---
    y_pred_xgb = xgb_model.predict(X_test_scaled)
    print("\n[1] XGBOOST REGRESSOR")
    print(f"MAE  : {mean_absolute_error(y_test, y_pred_xgb):.4f}")
    print(f"RMSE : {np.sqrt(mean_squared_error(y_test, y_pred_xgb)):.4f}")
    print(f"R2   : {r2_score(y_test, y_pred_xgb):.4f}")

    # --- Model 2: Linear Regression ---
    y_pred_lr = lr_model.predict(X_test_scaled)
    print("\n[2] LINEAR REGRESSION")
    print(f"MAE  : {mean_absolute_error(y_test, y_pred_lr):.4f}")
    print(f"RMSE : {np.sqrt(mean_squared_error(y_test, y_pred_lr)):.4f}")
    print(f"R2   : {r2_score(y_test, y_pred_lr):.4f}")

    # Visualization of Results: Actual vs Predicted (XGBoost)
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, y_pred_xgb, alpha=0.4, color='darkblue')
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    plt.xlabel('Actual Price (Lacs)')
    plt.ylabel('Predicted Price (Lacs)')
    plt.title('XGBoost Regressor: Actual vs Predicted Prices')
    plt.show() # Keep for standalone script execution

    # --- Testing with Manual Input ---
    print("\n" + "="*30)
    print("MANUAL MODEL TESTING (from script)")
    print("="*30)

    sample_car = {
        'Year': 2015,
        'Millage': 150000,
        'Fuel': 'Petrol',
        'Transmission': 'Automatic',
        'City': 'Lahore',
        'Color': 'Black',
        'Assembly': 'Local',
        'EngineCapacity': 1300,
        'BodyType': 'Sedan'
    }

    try:
        prediction = predict_single_car_price(sample_car)
        print(f"Car Details:")
        for key, value in sample_car.items():
            print(f"  {key}: {value}")
        print(f"Predicted Price: {prediction:.2f} Lacs")
    except RuntimeError as e:
        print(f"Error during manual prediction: {e}")
    except ValueError as e:
        print(f"Prediction Error: {e}")
        print("Check if the input categories match those in the dataset.")

    # Comment out or remove other plotting sections if not needed for script execution
    # plt.hist(df['Price'])
    # plt.title('Price Distribution')
    # plt.xlabel('Price in lacs')
    # plt.ylabel('Frequency')
    # plt.show()
    # ... and so on for other plots
