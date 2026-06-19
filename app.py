from flask import Flask, render_template, request, jsonify
import sys
import os

# Add the directory containing AI_Project_main.py to the Python path
# This assumes app.py is in the same directory as AI_Project_main.py
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import the functions and global variables from AI_Project_main.py
# This will not execute the `if __name__ == "__main__":` block
import AI_Project_main

app = Flask(__name__)

# Global variables to store model components, loaded once when the app starts
# These are separate from AI_Project_main's globals to avoid direct modification
# but AI_Project_main's predict_single_car_price uses its own globals.
# We'll call initialize_model_components once to populate AI_Project_main's globals.

def load_model_components_for_app():
    """Initializes model components by calling the function from AI_Project_main.py."""
    try:
        print("Loading model components for Flask app...")
        # This call populates the global variables within AI_Project_main.py
        xgb_model, lr_model, scaler, encoders, feature_columns, X_test_scaled, y_test, df = AI_Project_main.initialize_model_components()
        
        # Generate and save charts for analytics page
        AI_Project_main.generate_and_save_charts(df, X_test_scaled, y_test, xgb_model)
        
        print("Model components and charts loaded.")
    except FileNotFoundError:
        print("ERROR: data.csv not found! Please ensure it is in the project directory.")
        sys.exit(1)
    except Exception as e:
        print(f"ERROR during initialization: {e}")
        sys.exit(1)

# Load models when the Flask app starts
with app.app_context():
    load_model_components_for_app()

@app.route('/')
def index():
    """Renders the main prediction page."""
    # Pass categorical options to the template for dropdowns
    # These options are derived from the encoders fitted during initialization
    categorical_options = {}
    
    if AI_Project_main._encoders: 
        for col in AI_Project_main._cat_cols:
            if col in AI_Project_main._encoders:
                categorical_options[col] = list(AI_Project_main._encoders[col].classes_)
            else:
                print(f"Warning: Encoder for {col} not found in AI_Project_main._encoders.")
    else:
        print("Warning: AI_Project_main._encoders not loaded.")

    # Provide some default options if encoders are not ready or for testing
    # These are fallback options if the model initialization failed or for demonstration
    if not categorical_options.get('Fuel'):
        categorical_options['Fuel'] = ['Petrol', 'Diesel', 'Hybrid', 'CNG', 'LPG', 'Electric']
    if not categorical_options.get('Transmission'):
        categorical_options['Transmission'] = ['Automatic', 'Manual']
    if not categorical_options.get('City'):
        categorical_options['City'] = ['Lahore', 'Karachi', 'Islamabad', 'Peshawar', 'Quetta', 'Faisalabad', 'Multan']
    if not categorical_options.get('Color'):
        categorical_options['Color'] = ['White', 'Black', 'Silver', 'Grey', 'Blue', 'Red', 'Green', 'Brown', 'Gold', 'Beige', 'Other']
    if not categorical_options.get('Assembly'):
        categorical_options['Assembly'] = ['Local', 'Imported']
    if not categorical_options.get('BodyType'):
        categorical_options['BodyType'] = ['Sedan', 'Hatchback', 'SUV', 'Crossover', 'Mini Van', 'Coupe', 'Pickup', 'Van', 'Convertible', 'Other']

    return render_template('index.html', options=categorical_options)

@app.route('/analytics')
def analytics():
    """Renders the analytics page with market charts."""
    chart_files = [
        'actual_vs_predicted.png',
        'price_distribution.png',
        'body_type_analysis.png',
        'year_trend.png'
        'images/charts/actual_vs_predicted.png',
        'images/charts/price_distribution.png',
        'images/charts/body_type_analysis.png',
        'images/charts/year_trend.png'
    ]
    return render_template('analytics.html', charts=chart_files)

@app.route('/predict', methods=['POST'])
def predict():
    """Handles prediction requests."""
    try:
        data = request.get_json()
        car_data = {
            'Year': int(data['year']),
            'Millage': float(data['millage']),
            'Fuel': data['fuel'],
            'Transmission': data['transmission'],
            'City': data['city'],
            'Color': data['color'],
            'Assembly': data['assembly'],
            'EngineCapacity': float(data['engineCapacity']),
            'BodyType': data['bodytype']
        }

        # Call the prediction function from the AI_Project_main module
        predicted_price = AI_Project_main.predict_single_car_price(car_data)

        return jsonify({'prediction': f"{predicted_price:.2f} Lacs"})

    except Exception as e:
        print(f"Error during prediction: {e}")
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    # Flask will automatically look for templates in 'templates/' and static files in 'static/'
    app.run(debug=False, host='127.0.0.1', port=5000) # Set debug=False for more stable initial testing
