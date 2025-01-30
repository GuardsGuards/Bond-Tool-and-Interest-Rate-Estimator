import numpy as np
import pandas as pd

# ==============================
# PART 1: SHORT-RATE MODELS
# ==============================

# Vasicek Model: Mean-reverting short-rate model
def simulate_vasicek(initial_rate, mean_reversion_speed, long_term_mean, volatility, time_horizon, time_step):
    """ Simulates interest rate paths using the Vasicek model. """
    num_steps = int(time_horizon / time_step)
    rates = np.zeros(num_steps)
    rates[0] = initial_rate
    for t in range(1, num_steps):
        dr = (mean_reversion_speed * (long_term_mean - rates[t-1]) * time_step) + \
             (volatility * np.random.normal() * np.sqrt(time_step))
        rates[t] = rates[t-1] + dr
    return rates

# Cox-Ingersoll-Ross (CIR) Model: Ensures non-negative interest rates
def simulate_cir(initial_rate, mean_reversion_speed, long_term_mean, volatility, time_horizon, time_step):
    """ Simulates interest rate paths using the Cox-Ingersoll-Ross (CIR) model. """
    num_steps = int(time_horizon / time_step)
    rates = np.zeros(num_steps)
    rates[0] = initial_rate
    for t in range(1, num_steps):
        dr = (mean_reversion_speed * (long_term_mean - rates[t-1]) * time_step) + \
             (volatility * np.sqrt(rates[t-1]) * np.random.normal() * np.sqrt(time_step))
        rates[t] = max(0, rates[t-1] + dr)  # Ensure non-negativity
    return rates

# Hull-White Model: Extension of Vasicek with time-dependent mean-reversion
def simulate_hull_white(initial_rate, mean_reversion_speed, time_dependent_mean, volatility, time_horizon, time_step):
    """ Simulates interest rate paths using the Hull-White model with a time-dependent mean. """
    num_steps = int(time_horizon / time_step)
    rates = np.zeros(num_steps)
    rates[0] = initial_rate
    for t in range(1, num_steps):
        theta = time_dependent_mean(t * time_step)  # Time-dependent mean function
        dr = (mean_reversion_speed * (theta - rates[t-1]) * time_step) + \
             (volatility * np.random.normal() * np.sqrt(time_step))
        rates[t] = rates[t-1] + dr
    return rates

# Define a time-dependent function for Hull-White model
def time_dependent_theta(time):
    """ Example function for a time-dependent mean level in the Hull-White model. """
    return 0.05 + 0.01 * np.sin(time)

# ==============================
# PART 2: HEATH-JARROW-MORTON (HJM) FRAMEWORK
# ==============================

# HJM Model: Forward rate curve simulation
def simulate_hjm_forward_rates(initial_forward_rates, volatilities, time_horizon, time_step):
    """ Simulates forward rate curves using the Heath-Jarrow-Morton (HJM) model. """
    num_steps = int(time_horizon / time_step)
    num_rates = len(initial_forward_rates)
    rate_paths = np.zeros((num_rates, num_steps))
    rate_paths[:, 0] = initial_forward_rates

    for t in range(1, num_steps):
        dW = np.random.normal(size=num_rates) * np.sqrt(time_step)  # Brownian motion
        drift = -0.5 * volatilities**2 * time_step  # Simplified drift term
        diffusion = volatilities * dW
        rate_paths[:, t] = rate_paths[:, t-1] + drift + diffusion

    return rate_paths

# ==============================
# PART 3: MODEL PARAMETERS & SIMULATION EXECUTION
# ==============================

# Define model parameters
initial_rate = 0.05   # Initial interest rate
mean_reversion_speed = 0.3   # Speed of mean reversion
long_term_mean = 0.05   # Long-term equilibrium rate
volatility = 0.02   # Volatility of interest rate
time_horizon = 10   # Total time in years
time_step = 0.01   # Time step in years

# Simulate interest rate paths
vasicek_rates = simulate_vasicek(initial_rate, mean_reversion_speed, long_term_mean, volatility, time_horizon, time_step)
cir_rates = simulate_cir(initial_rate, mean_reversion_speed, long_term_mean, volatility, time_horizon, time_step)
hull_white_rates = simulate_hull_white(initial_rate, mean_reversion_speed, time_dependent_theta, volatility, time_horizon, time_step)

# Forward rates for HJM
num_forward_rates = 10  # Number of forward rates to simulate
initial_forward_rates = np.linspace(0.02, 0.05, num_forward_rates)  # 10 forward rates from 2% to 5%
volatilities = np.linspace(0.01, 0.03, num_forward_rates)  # Increasing volatilities
hjm_rates = simulate_hjm_forward_rates(initial_forward_rates, volatilities, time_horizon, time_step)

# ==============================
# PART 4: SAVE SIMULATION RESULTS TO FILES
# ==============================

np.savetxt("vasicek_rates.csv", vasicek_rates, delimiter=",", header="Vasicek Model Rates", comments='')
np.savetxt("cir_rates.csv", cir_rates, delimiter=",", header="CIR Model Rates", comments='')
np.savetxt("hull_white_rates.csv", hull_white_rates, delimiter=",", header="Hull-White Model Rates", comments='')
np.savetxt("hjm_forward_rates.csv", hjm_rates, delimiter=",", header="HJM Forward Rates", comments='')

print("Interest rate simulations saved successfully.")

#===============================

from flask import Flask, request, jsonify, render_template

app = Flask(__name__)

# ==============================
# HOME PAGE ROUTE
# ==============================
@app.route('/')
def home():
    return render_template('index.html')  # Assume an HTML file with user inputs

# ==============================
# API ROUTE TO RUN MODELS
# ==============================
@app.route('/run_model', methods=['POST'])
def run_model():
    try:
        model_type = request.form.get("model_type")
        
        # Common input parameters
        initial_rate = float(request.form.get("initial_rate"))
        mean_reversion_speed = float(request.form.get("mean_reversion_speed"))
        long_term_mean = float(request.form.get("long_term_mean"))
        volatility = float(request.form.get("volatility"))
        time_horizon = float(request.form.get("time_horizon"))
        time_step = float(request.form.get("time_step"))

        # Check which model to run
        if model_type == "vasicek":
            rates = simulate_vasicek(initial_rate, mean_reversion_speed, long_term_mean, volatility, time_horizon, time_step)

        elif model_type == "cir":
            rates = simulate_cir(initial_rate, mean_reversion_speed, long_term_mean, volatility, time_horizon, time_step)

        elif model_type == "hull_white":
            rates = simulate_hull_white(initial_rate, mean_reversion_speed, time_dependent_theta, volatility, time_horizon, time_step)

        elif model_type == "hjm":
            num_forward_rates = 10  # Default forward rate count
            initial_forward_rates = np.linspace(0.02, 0.05, num_forward_rates)
            volatilities = np.linspace(0.01, 0.03, num_forward_rates)
            rates = simulate_hjm_forward_rates(initial_forward_rates, volatilities, time_horizon, time_step)
            rates = rates.tolist()  # Convert to list for JSON response

        else:
            return jsonify({"error": "Invalid model type specified."})

        return jsonify({"model": model_type, "simulated_rates": rates.tolist()})

    except Exception as e:
        return jsonify({"error": str(e)})

# ==============================
# RUN FLASK APP
# ==============================
if __name__ == '__main__':
    app.run(debug=True)