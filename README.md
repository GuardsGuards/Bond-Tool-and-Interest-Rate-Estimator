# 📌 Capital Markets & Risk

A **Python & Flask**-based application for advanced interest rate modeling, bond pricing, risk analytics, and financial backtesting. The toolkit integrates cutting-edge models and real-world data to help risk analysts, portfolio managers, and researchers make informed financial decisions.

---

## 📖 Project Overview

This project offers a comprehensive framework that blends:

- **Advanced Interest Rate Modeling:**  
  Combines **jump diffusion** (modeling sudden rate changes), **exogenous factors** (e.g., inflation and money velocity), and a **CIR + Hull-White hybrid model** (ensuring non-negative, mean-reverting rates with time-dependent drift).

- **Bond Pricing & Analytics:**  
  Provides tools for computing bond prices, total/annualized returns, duration, convexity, and even pricing using a binomial tree (supporting callable/putable bonds).

- **Credit–FX Jump Model Simulation:**  
  Implements risk–neutral pricing of a zero–coupon bond subject to default risk and an FX jump. It computes both theoretical and Monte Carlo prices for bonds denominated in EUR (converted to USD) and USD, and estimates the FX sensitivity (delta) of the EUR bond price.

- **Visualization & Reporting:**  
  Interactive plots, heatmaps, and summary tables (using Matplotlib and Seaborn) illustrate historical and projected interest rates, bond risk metrics, and pricing analytics.

---

## 🛠️ Key Features

1. **Interest Rate Modeling**
   - **Jump Diffusion:** Models sudden rate jumps with parameters λ (intensity), μ_J (mean jump), and σ_J (volatility).
   - **Exogenous Factors:** Incorporates inflation and money velocity trends.
   - **Hybrid CIR/Hull-White Model:** Combines non-negative rate properties with mean-reversion and calibrated time-dependent drift.

2. **Bond Analytics & Pricing**
   - **Bond Pricing:** Calculates prices from yield, maturity, face value, and coupon rate.
   - **Total Return Analysis:** Determines gross and annualized returns if held to maturity.
   - **Duration & Convexity:** Measures sensitivity and curvature of bond price-yield relationships.
   - **Binomial Tree Pricing:** Supports valuation of bonds with embedded options.
   - **Credit–FX Simulation:** Computes risk–neutral prices for EUR and USD bonds and estimates FX delta via finite differences.

3. **Visualization & Reporting**
   - **Interactive Plots & Heatmaps:** Visualize interest rate projections, bond prices, and risk metrics.
   - **Summary Tables:** Present detailed analytics for bond pricing, returns, duration, and convexity.

---

## 🧩 Project Structure

- **Interest Rate Tool:**  
  Integrates jump diffusion, exogenous factors, and the hybrid CIR/Hull-White model to project future interest rates with confidence intervals.

- **Bond Pricing Tool:**  
  Computes bond prices, analyzes total returns, and evaluates risk measures (duration and convexity) using both analytical and binomial tree methods.

- **Credit–FX Jump Model Simulation:**  
  Simulates risk–neutral pricing of zero–coupon bonds under default risk and FX jump conditions. It provides:
  - **Theoretical Pricing:**  
    - USD Bond: `P_USD = exp(-r*T) * exp(-h*T)`
    - EUR Bond (in USD):  
      `P_EUR = exp(-r*T) * S0 * [exp(-h*T) + exp(J) * (1 - exp(-h*T))]`
  - **Monte Carlo Pricing:** For robust simulation of bond payoffs.
  - **FX Sensitivity (Delta):** Estimated using finite difference approximations.

---

## 🚀 Usage

1. **Data Preparation:**  
   Ensure CSV files such as `Yield Curve.csv`, `M2V.csv`, and `US Inflation.csv` are available for historical data.

2. **Running the Application:**  
   - **Jupyter Notebook:** Execute the cells to calibrate models, compute bond analytics, and generate visualizations.  
   - **Flask App:** Launch the web interface for interactive exploration.

3. **Analyzing Results:**  
   Review output plots and summary tables to gain insights into:
   - Projected interest rates and their drivers.
   - Bond pricing, returns, duration, and convexity.
   - The impact of default risk and FX jumps on bond valuations.

---

## 📦 Dependencies

- **Python 3.x**
- **Libraries:** `numpy`, `pandas`, `matplotlib`, `seaborn`, `statsmodels`, `numpy_financial`, **Flask**

---

## 🔮 Future Work

- **Model Extensions:** Incorporate additional macroeconomic indicators and credit risk factors.
- **Real-Time Integration:** Fetch live yield curve and market data.
- **Enhanced UI:** Expand the web interface with frameworks like Flask or Dash for deeper interactivity.

---

## 📝 Conclusion

This project delivers a robust framework for **interest rate modeling** and **bond pricing**, blending theoretical models with practical data analysis. Whether for risk management, portfolio optimization, or academic research, this toolkit provides the advanced tools necessary for navigating today’s dynamic financial markets.

