import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.mixture import GaussianMixture
from statsmodels.tsa.arima.model import ARIMA

# ==============================
# Part 1: Jump Risk Modeling
# ==============================
def jump_diffusion_model(r0, K, theta, sigma, lambda_j, mu_j, sigma_j, T, dt):
    """ Simulate interest rate path with jumps using a jump-diffusion model. """
    num_steps = int(T / dt)
    rates = np.zeros(num_steps)
    rates[0] = r0
    for t in range(1, num_steps):
        jump = 0
        if np.random.rand() < lambda_j * dt:  # Check if jump occurs
            jump = np.random.normal(mu_j, sigma_j)
        dr = K * (theta - rates[t-1]) * dt + sigma * np.random.normal() * np.sqrt(dt) + jump
        rates[t] = rates[t-1] + dr
    return rates

# ==============================
# Part 2: Bayesian Updating
# ==============================
def bayesian_update(prior_mean, prior_var, new_data_mean, new_data_var):
    """ Bayesian update formula for combining prior and observed data. """
    posterior_mean = (prior_var * new_data_mean + new_data_var * prior_mean) / (prior_var + new_data_var)
    posterior_var = 1 / ((1 / prior_var) + (1 / new_data_var))
    return posterior_mean, posterior_var

# ==============================
# Part 3: Gaussian+ Mixture Model
# ==============================
def fit_gaussian_mixture(data, components=2):
    """ Fit a Gaussian mixture model to the data. """
    model = GaussianMixture(n_components=components, random_state=42)
    model.fit(data.reshape(-1, 1))
    return model

# ==============================
# Part 4: Autoregressive Models (AFEM, PAFEM)
# ==============================
def autoregressive_model(data, order=(1, 0, 0)):
    """ Fit an ARIMA model to the data for forecasting. """
    model = ARIMA(data, order=order)
    model_fit = model.fit()
    return model_fit

# ==============================
# Part 5: Monte Carlo Simulation
# ==============================
def monte_carlo_simulation(start_price, mu, sigma, T, dt, scenarios=1000):
    """ Simulate asset prices using Monte Carlo with Geometric Brownian Motion. """
    num_steps = int(T / dt)
    results = np.zeros((scenarios, num_steps))
    for s in range(scenarios):
        prices = np.zeros(num_steps)
        prices[0] = start_price
        for t in range(1, num_steps):
            prices[t] = prices[t-1] * np.exp((mu - 0.5 * sigma**2) * dt + sigma * np.random.normal() * np.sqrt(dt))
        results[s] = prices
    return results

# ==============================
# Part 6: Markov Chain Analysis
# ==============================
def markov_chain(states, transition_matrix, steps):
    """ Simulate a Markov chain given states and a transition matrix. """
    current_state = np.random.choice(states)
    history = [current_state]
    for _ in range(steps):
        current_state = np.random.choice(states, p=transition_matrix[current_state])
        history.append(current_state)
    return history

# ==============================
# Running the models and saving results
# ==============================

# Example Parameters
r0 = 0.05
K = 0.3
theta = 0.05
sigma = 0.02
lambda_j = 0.1  # Jump intensity (per year)
mu_j = 0.02  # Jump mean size
sigma_j = 0.01  # Jump volatility
T = 10
dt = 0.01
start_price = 100  # Initial asset price for Monte Carlo
mu = 0.05  # Mean return
scenarios = 1000  # Number of Monte Carlo scenarios

# Simulate each model
rates_jump = jump_diffusion_model(r0, K, theta, sigma, lambda_j, mu_j, sigma_j, T, dt)
bayesian_mean, bayesian_var = bayesian_update(0.05, 0.01, 0.06, 0.02)
mc_simulation = monte_carlo_simulation(start_price, mu, sigma, T, dt, scenarios)

# Saving results
np.savetxt("jump_diffusion_rates.csv", rates_jump, delimiter=",")
np.savetxt("monte_carlo_simulations.csv", mc_simulation, delimiter=",")

print("Advanced modeling simulations saved successfully.")

# ==============================
# Flask App Integration (Part 4)
# ==============================
from flask import Flask, request, jsonify, render_template

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')  # Assume index.html allows users to interact with models

@app.route('/predict', methods=['POST'])
def predict():
    """ API to predict using different models based on user input. """
    try:
        model_type = request.form.get("model_type")
        
        if model_type == "jump_diffusion":
            r0_input = float(request.form.get("r0"))
            rates = jump_diffusion_model(r0_input, K, theta, sigma, lambda_j, mu_j, sigma_j, T, dt)
            return jsonify({"interest_rate_path": rates.tolist()})

        elif model_type == "monte_carlo":
            start_price = float(request.form.get("start_price"))
            mc_sim = monte_carlo_simulation(start_price, mu, sigma, T, dt, scenarios=100)
            return jsonify({"simulated_prices": mc_sim.tolist()})

        elif model_type == "bayesian":
            prior_mean = float(request.form.get("prior_mean"))
            prior_var = float(request.form.get("prior_var"))
            new_data_mean = float(request.form.get("new_data_mean"))
            new_data_var = float(request.form.get("new_data_var"))
            post_mean, post_var = bayesian_update(prior_mean, prior_var, new_data_mean, new_data_var)
            return jsonify({"posterior_mean": post_mean, "posterior_variance": post_var})

        else:
            return jsonify({"error": "Invalid model type specified."})
    
    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == '__main__':
    app.run(debug=True)
