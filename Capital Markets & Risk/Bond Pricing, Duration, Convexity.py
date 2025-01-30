from flask import Flask, request, jsonify, render_template
import numpy as np

# ==============================
# PART 1: BOND PRICING FUNCTIONS
# ==============================

def price_bond(face_value, coupon_rate, market_rate, periods):
    """ Traditional Bond Pricing (Discounted Cash Flow Method) """
    cash_flows = np.array([(face_value * coupon_rate) / (1 + market_rate) ** t for t in range(1, periods + 1)])
    final_value = face_value / (1 + market_rate) ** periods
    return np.sum(cash_flows) + final_value

def binomial_tree_bond(face_value, coupon_rate, up_factor, down_factor, risk_free_rate, periods, option_type=None):
    """ Binomial Tree for Bonds with Embedded Options (Call/Put) """
    bond_tree = np.zeros((periods + 1, periods + 1))
    
    # Future bond values at maturity
    for i in range(periods + 1):
        bond_tree[i, periods] = face_value + (face_value * coupon_rate)

    # Work backwards through tree
    for j in range(periods - 1, -1, -1):
        for i in range(j + 1):
            expected_value = 0.5 * (bond_tree[i, j + 1] / (1 + risk_free_rate) + bond_tree[i + 1, j + 1] / (1 + risk_free_rate))
            if option_type == 'call':
                bond_tree[i, j] = min(face_value, expected_value)  # Callable bond
            elif option_type == 'put':
                bond_tree[i, j] = max(face_value, expected_value)  # Putable bond
            else:
                bond_tree[i, j] = expected_value  # Traditional bond pricing

    return bond_tree[0, 0]

# ==============================
# PART 2: RISK METRICS (DURATION & CONVEXITY)
# ==============================

def calculate_duration(face_value, coupon_rate, market_rate, periods):
    """ Computes Macaulay and Modified Duration """
    cash_flows = np.array([(face_value * coupon_rate) / (1 + market_rate) ** t for t in range(1, periods + 1)])
    weighted_times = np.array([t * cf for t, cf in enumerate(cash_flows, 1)])
    bond_price = np.sum(cash_flows) + face_value / (1 + market_rate) ** periods

    macaulay_duration = np.sum(weighted_times) / bond_price
    modified_duration = macaulay_duration / (1 + market_rate)
    return macaulay_duration, modified_duration

def calculate_convexity(face_value, coupon_rate, market_rate, periods):
    """ Computes Convexity of a bond """
    cash_flows = np.array([(face_value * coupon_rate) / (1 + market_rate) ** t for t in range(1, periods + 1)])
    weighted_times = np.array([(t * (t + 1) * cf) for t, cf in enumerate(cash_flows, 1)])
    bond_price = np.sum(cash_flows) + face_value / (1 + market_rate) ** periods

    convexity = np.sum(weighted_times) / (bond_price * (1 + market_rate) ** 2)
    return convexity

# ==============================
# PART 3: FLASK APP
# ==============================

app = Flask(__name__)

@app.route('/')
def home():
    """ Home Route - Render the Input Form """
    return render_template('index.html')  # Assume index.html allows users to enter bond details

@app.route('/calculate', methods=['POST'])
def calculate():
    """ API Route for Bond Pricing & Risk Metrics Calculation """
    try:
        # Get user input from the form
        bond_type = request.form.get("bond_type")  # 'traditional' or 'binomial'
        face_value = float(request.form.get("face_value"))
        coupon_rate = float(request.form.get("coupon_rate"))
        market_rate = float(request.form.get("market_rate"))
        periods = int(request.form.get("periods"))

        if bond_type == "traditional":
            bond_price = price_bond(face_value, coupon_rate, market_rate, periods)

        elif bond_type == "binomial":
            up_factor = float(request.form.get("up_factor", 1.1))
            down_factor = float(request.form.get("down_factor", 0.9))
            risk_free_rate = float(request.form.get("risk_free_rate", market_rate))
            option_type = request.form.get("option_type")  # 'call', 'put', or None

            bond_price = binomial_tree_bond(face_value, coupon_rate, up_factor, down_factor, risk_free_rate, periods, option_type)

        else:
            return jsonify({"error": "Invalid bond type specified."})

        # Calculate risk metrics
        macaulay_duration, modified_duration = calculate_duration(face_value, coupon_rate, market_rate, periods)
        convexity = calculate_convexity(face_value, coupon_rate, market_rate, periods)

        return jsonify({
            "bond_type": bond_type,
            "bond_price": round(bond_price, 2),
            "macaulay_duration": round(macaulay_duration, 2),
            "modified_duration": round(modified_duration, 2),
            "convexity": round(convexity, 2)
        })

    except Exception as e:
        return jsonify({"error": str(e)})

# ==============================
# RUN FLASK APP
# ==============================

if __name__ == '__main__':
    app.run(debug=True)
