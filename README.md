# 📌 Capital Markets & Risk: Interest Rate Modeling & Pricing Project

## A Python & Flask-based application for advanced interest rate modeling, bond pricing, risk analytics, and financial backtesting.

### 📖 Project Overview

This project implements a comprehensive suite of financial models for interest rate forecasting, bond pricing, and risk analysis. It leverages Python-based quantitative finance techniques, including stochastic differential equations, Bayesian inference, Monte Carlo simulations, and autoregressive models, to provide a rigorous, data-driven approach for understanding and managing interest rate risks.


### The application includes:

✅ Interest Rate Models (Vasicek, CIR, Hull-White, HJM)

✅ Advanced Interest Rate Models (Jump-Diffusion, Bayesian, Gaussian+, Autoregressive Functional Exogenous, Pointwise Autoregressive Functional Exogenous, Monte Carlo, Markov Chain)

✅ Bond Pricing (Discounted Cash Flow, Binomial Trees, Option-Embedded Bonds)

✅ Risk Metrics (Duration, Convexity, Sensitivity Analysis, Option-Adjusted Spread, Binomial Tree)

✅ Backtesting & Performance Evaluation (ADF, Cointegration, ANOVA, AIC/BIC)

✅ Flask Web App for User Interaction & Real-Time Scenario Analysis


### 📂 Project Structure

## 🔹 Part 1: Comprehensive Interest Rate & Advanced Modeling

Objective: Implement short-rate models, full yield curve models, and advanced statistical techniques to create robust interest rate forecasts.

### 📌 Interest Rate Models

Short-Rate Models (Simulates short-term interest rate paths)

Vasicek Model – Mean-reverting interest rate model.

Cox-Ingersoll-Ross (CIR) Model – Ensures positive interest rates using a square root diffusion process.

Hull-White Model – An extension of Vasicek allowing time-dependent mean reversion.

Full Yield Curve Model: Heath-Jarrow-Morton (HJM) Framework

## Models the entire term structure of interest rates, directly simulating forward rate curves under a no-arbitrage condition.

### 📌 Advanced Modeling Techniques

Jump Risk Modeling – Models sudden market shocks using jump-diffusion processes.

Bayesian Models – Implements Bayesian inference to dynamically update interest rate forecasts using prior distributions and new data.

Gaussian+ Models – Captures non-linear behaviors and fat tails in interest rate changes.

Autoregressive Models (AFEM, PAFEM) – Integrates macroeconomic indicators to refine forecasts.

Monte Carlo Simulations – Models the impact of volatility & rate fluctuations on financial instruments.

Markov Chain Analysis – Uses probabilistic state transitions to analyze interest rate regime changes.

## 🔹 Part 2: Bond Pricing & Duration Calculations

Objective: Use the interest rate forecasts from Part 1 to price bonds with and without embedded options and calculate key financial metrics.

### 📌 Bond Pricing

Traditional Bonds – Uses Discounted Cash Flow (DCF) method to price bonds based on rate forecasts.

Bonds with Embedded Options – Implements a Binomial Tree Model to price callable/putable bonds, incorporating volatility & jump risks.

### 📌 Financial Metrics Calculation

Duration & Convexity – Measures bond price sensitivity to interest rate changes:

Macaulay Duration – Time-weighted measure of cash flow present values.

Modified Duration – Adjusted for yield changes, used for risk management.

Convexity – Captures curvature effects in interest rate sensitivity.

## 🔹 Part 3: Analysis, Backtesting & Model Evaluation

Objective: Validate model performance, conduct statistical tests, and analyze errors to refine predictive accuracy.

### 📌 Backtesting Framework
Compares historical actual vs. predicted interest rates.
Evaluates model performance using error metrics & statistical diagnostics.

### 📌 Statistical Tests & Sensitivity Analysis

Augmented Dickey-Fuller (ADF) Test – Checks for stationarity in interest rate time series.

Johansen Cointegration Test – Identifies long-term relationships between multiple rate series.

Sensitivity Analysis – Assesses how interest rate fluctuations impact bond valuations.

### 📌 Error Metrics & Model Diagnostics

Mean Absolute Error (MAE), Mean Squared Error (MSE), Root Mean Squared Error (RMSE) – Measures forecast accuracy.

AIC & BIC (Akaike & Bayesian Information Criterion) – Determines optimal model complexity.

ANOVA Table – Compares multiple models based on RSS (Residual Sum of Squares), AIC, and BIC.

## 🔹 Part 4: Flask Web App & User Interaction

Objective: Enable real-time interest rate modeling, bond pricing, and risk assessment via a Flask-powered web application.

### 📌 Features

✅ Interactive Model Selection – Users can choose interest rate models and compare outputs dynamically.

✅ Scenario Testing – Allows users to simulate different rate environments and observe bond price impacts.

✅ Dynamic Visualization – Plots actual vs. predicted interest rates, Monte Carlo simulations, and binomial tree outcomes.

✅ JSON API Responses – Outputs structured financial risk metrics for integration with other tools.
