# 📌 Capital Markets & Risk: Interest Rate Modeling & Pricing Project

## A Python & Flask-based application for advanced interest rate modeling, bond pricing, risk analytics, and financial backtesting.

---

### 📖 Project Overview

This project is designed to provide a comprehensive toolkit for **interest rate modeling** and **bond pricing**, incorporating advanced financial models and real-world data. The project combines **jump diffusion**, **exogenous factors** (such as inflation and money velocity), and **refinements to traditional models** like the **CIR (Cox-Ingersoll-Ross)** and **Hull-White models** to deliver a more accurate and robust framework for financial analysis.

---

## 🛠️ Key Features

### 1. **Interest Rate Modeling**
   - **Jump Diffusion**: Incorporates sudden jumps in interest rates, modeled using parameters λ (jump intensity), μ_J (mean jump size), and σ_J (jump volatility).
   - **Exogenous Factors**: Integrates external drivers like **inflation** and **money velocity (M2V)** to refine the interest rate projections.
   - **CIR + Hull-White Hybrid Model**: Combines the strengths of the **CIR model** (for non-negative rates) and the **Hull-White model** (for mean-reversion and time-dependent drift) to create a more realistic interest rate simulation.
   - **Time-Dependent Drift**: Captures trends in interest rates using a time-dependent drift term calibrated from historical data.

### 2. **Bond Analytics**
   - **Bond Pricing**: Computes bond prices based on yield, maturity, face value, and coupon rate.
   - **Total Return Analysis**: Calculates the total return and annualized return if the bond is held to maturity.
   - **Duration & Convexity**: Measures the bond's sensitivity to interest rate changes using **Macaulay Duration**, **Modified Duration**, and **Convexity**.
   - **Binomial Tree Pricing**: Prices bonds using a binomial tree model, with support for **callable** and **putable** bonds.

### 3. **Visualization & Reporting**
   - **Interactive Plots**: Visualizes historical and projected interest rates, bond prices, and risk metrics using **Matplotlib** and **Seaborn**.
   - **Heatmaps**: Displays bond prices at different nodes and time steps in a binomial tree model.
   - **Summary Tables**: Provides detailed analytics for bond pricing, returns, duration, and convexity.

---

## 🧩 Project Structure

### 1. **Interest Rate Tool**
   - **Combines Jump Diffusion, Exogenous Factors, and Refined CIR/Hull-White Models**:
     - **Jump Diffusion**: Models sudden changes in interest rates using calibrated parameters (λ, μ_J, σ_J).
     - **Exogenous Factors**: Incorporates inflation and money velocity trends to refine the interest rate projections.
     - **CIR + Hull-White Hybrid**: Ensures non-negative rates and captures mean-reversion and time-dependent drift.
   - **Output**: Projected interest rates with confidence intervals, visualized alongside historical data.

### 2. **Bond Pricing Tool**
   - **Bond Price Calculation**: Computes bond prices using yield, maturity, face value, and coupon rate.
   - **Total Return Analysis**: Calculates gross and annualized returns for bonds held to maturity.
   - **Duration & Convexity**: Measures interest rate sensitivity and curvature of the bond price-yield relationship.
   - **Binomial Tree Pricing**: Prices bonds with optional embedded call or put features using a binomial tree model.

---

## 🚀 Usage

### 1. **Data Preparation**
   - Ensure the following CSV files are available:
     - `Yield Curve.csv`: Historical yield curve data.
     - `M2V.csv`: Money velocity data.
     - `US Inflation.csv`: Inflation data.

### 2. **Running the Notebook**
   - Execute the Jupyter notebook cells to:
     - Calibrate jump parameters and interest rate models.
     - Compute bond prices, returns, duration, and convexity.
     - Generate visualizations and summary tables.

### 3. **Analyzing Results**
   - Review the output plots and tables to gain insights into:
     - Projected interest rates and their sensitivity to exogenous factors.
     - Bond pricing and risk metrics for selected maturities.

---

## 📊 Example Outputs

### 1. **Interest Rate Projections**
   - A plot showing historical 10-year yields and projected yields using the modified Hull-White model.
   - Confidence intervals for projected rates based on jump diffusion and exogenous factors.

### 2. **Bond Analytics**
   - **Bond Price**: Computed price for selected maturities (e.g., 2-year and 10-year bonds).
   - **Total Return**: Gross and annualized returns if held to maturity.
   - **Duration & Convexity**: Measures of interest rate sensitivity.
   - **Binomial Tree Heatmap**: Visual representation of bond prices at different nodes and time steps.

---

## 📦 Dependencies

- **Python 3.x**
- **Libraries**:
  - `numpy`, `pandas`, `matplotlib`, `seaborn`, `statsmodels`, `numpy_financial`

---

## 🔮 Future Work

- **Model Extensions**: Incorporate additional factors like macroeconomic indicators or credit risk.
- **Real-Time Data Integration**: Fetch real-time yield curve data for up-to-date analysis.
- **User Interface**: Develop a web-based interface for interactive analysis using **Flask** or **Dash**.

---

## 📝 Conclusion

This project provides a **robust framework** for **interest rate modeling** and **bond pricing**, combining **theoretical models** with **practical data analysis**. By integrating **jump diffusion**, **exogenous factors**, and **refinements to traditional models**, it offers a more **accurate and realistic approach** to financial analytics. Whether you're a risk analyst, portfolio manager, or financial researcher, this toolkit equips you with the tools to make informed decisions in a dynamic market environment.

--- 

🌟 **Happy Modeling!** 🌟
