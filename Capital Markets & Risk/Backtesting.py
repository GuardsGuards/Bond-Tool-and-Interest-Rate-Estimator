from flask import Flask, request, jsonify, render_template
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import io
import base64

# Import statistical and ML libraries
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.vector_ar.vecm import coint_johansen  # Cointegration Test
from statsmodels.tsa.ar_model import AutoReg  # AutoReg for Model Diagnostics
from sklearn.metrics import mean_absolute_error, mean_squared_error  # Error Metrics
from statsmodels.formula.api import ols  # ANOVA Table

# ==============================
# INITIALIZE FLASK APP
# ==============================
app = Flask(__name__)

# ==============================
# BACKTESTING FUNCTION
# ==============================

def backtest_strategy(actual_rates, predicted_rates):
    """ Backtesting: Computes MAE, MSE, RMSE and generates a comparison plot. """
    
    # Compute error metrics
    mae = mean_absolute_error(actual_rates, predicted_rates)
    mse = mean_squared_error(actual_rates, predicted_rates)
    rmse = np.sqrt(mse)
    
    # Create and save the plot as a Base64 string
    plt.figure(figsize=(10, 6))
    plt.plot(actual_rates, label='Actual Rates', marker='o')
    plt.plot(predicted_rates, label='Predicted Rates', marker='x')
    plt.legend()
    plt.title('Backtesting: Actual vs. Predicted Interest Rates')
    
    # Convert plot to Base64 for Flask API
    img = io.BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    plot_url = base64.b64encode(img.getvalue()).decode()

    return {'MAE': mae, 'MSE': mse, 'RMSE': rmse, 'plot_url': plot_url}

# ==============================
# STATISTICAL TESTS & SENSITIVITY ANALYSIS
# ==============================

def adf_test(time_series):
    """ Perform Augmented Dickey-Fuller test for stationarity. """
    result = adfuller(time_series)
    return {'ADF Statistic': result[0], 'p-value': result[1], 'Critical Values': result[4]}

def cointegration_test(time_series_matrix):
    """ Perform Johansen Cointegration Test. """
    result = coint_johansen(time_series_matrix, det_order=0, k_ar_diff=1)
    return {'Trace Statistic': result.lr1.tolist(), 'Critical Values': result.cvt.tolist()}

def sensitivity_analysis(base_value, sensitivity_range, factor):
    """ Perform sensitivity analysis for interest rate changes. """
    analysis = {'Rate Change': [], 'Adjusted Value': []}
    
    for change in sensitivity_range:
        adjusted_value = base_value * (1 + factor * change)
        analysis['Rate Change'].append(change)
        analysis['Adjusted Value'].append(adjusted_value)
    
    return pd.DataFrame(analysis).to_dict(orient="records")

# ==============================
# MODEL DIAGNOSTICS & ANOVA TABLE
# ==============================

def calculate_aic_bic(time_series, lags):
    """ Calculate AIC and BIC for model selection. """
    model = AutoReg(time_series, lags=lags).fit()
    return {'Model': f'AR({lags})', 'AIC': model.aic, 'BIC': model.bic}

def anova_table(models_dict):
    """ Creates an ANOVA-style table for comparing multiple models. """
    results = []
    
    for model_name, model in models_dict.items():
        results.append({
            'Model': model_name,
            'RSS': np.sum(model.resid ** 2),
            'AIC': model.aic,
            'BIC': model.bic
        })
    
    return results

# ==============================
# FLASK ROUTES
# ==============================

@app.route('/')
def home():
    """ Render the Input Form """
    return render_template('index.html')  # Assume index.html contains a form to enter interest rate data

@app.route('/backtest', methods=['POST'])
def backtest():
    """ API Route: Perform backtesting and return error metrics + plot. """
    try:
        actual_rates = np.array([float(x) for x in request.form.get("actual_rates").split(",")])
        predicted_rates = np.array([float(x) for x in request.form.get("predicted_rates").split(",")])
        
        results = backtest_strategy(actual_rates, predicted_rates)
        return jsonify(results)

    except Exception as e:
        return jsonify({"error": str(e)})

@app.route('/adf_test', methods=['POST'])
def run_adf_test():
    """ API Route: Run ADF Test for stationarity. """
    try:
        time_series = np.array([float(x) for x in request.form.get("time_series").split(",")])
        results = adf_test(time_series)
        return jsonify(results)

    except Exception as e:
        return jsonify({"error": str(e)})

@app.route('/cointegration', methods=['POST'])
def run_cointegration():
    """ API Route: Run Johansen Cointegration Test. """
    try:
        time_series_matrix = np.array([[float(x) for x in row.split(",")] for row in request.form.getlist("time_series_matrix")])
        results = cointegration_test(time_series_matrix)
        return jsonify(results)

    except Exception as e:
        return jsonify({"error": str(e)})

@app.route('/sensitivity', methods=['POST'])
def run_sensitivity():
    """ API Route: Perform Sensitivity Analysis """
    try:
        base_value = float(request.form.get("base_value"))
        sensitivity_range = np.linspace(float(request.form.get("min_change")), float(request.form.get("max_change")), 5)
        factor = float(request.form.get("factor"))

        results = sensitivity_analysis(base_value, sensitivity_range, factor)
        return jsonify(results)

    except Exception as e:
        return jsonify({"error": str(e)})

@app.route('/compare_models', methods=['POST'])
def compare_models():
    """ API Route: Compare models using ANOVA Table """
    try:
        time_series = np.array([float(x) for x in request.form.get("time_series").split(",")])
        
        # Fit models with different lags
        model1 = AutoReg(time_series, lags=1).fit()
        model2 = AutoReg(time_series, lags=2).fit()

        # Generate ANOVA Table
        anova_results = anova_table({"AR(1)": model1, "AR(2)": model2})
        return jsonify(anova_results)

    except Exception as e:
        return jsonify({"error": str(e)})

# ==============================
# RUN FLASK APP
# ==============================

if __name__ == '__main__':
    app.run(debug=True)
