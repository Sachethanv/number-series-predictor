import os
import math
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from flask import Flask, render_template, request

app = Flask(__name__)

# Ensure static directory exists
if not os.path.exists('static'):
    os.makedirs('static')

# Function to identify sequence type and return confidence
def identify_sequence_type(input_series):
    if len(input_series) != 5:
        return "Please provide exactly 5 numbers in the input series.", None, None

    # Calculate the differences between consecutive terms
    diff = [round(input_series[i] - input_series[i-1], 6) for i in range(1, len(input_series))]
    ratio = [round(input_series[i] / input_series[i-1], 6) for i in range(1, len(input_series)) if input_series[i-1] != 0]

    # Check for Arithmetic Progression (AP)
    ap_confidence = sum(1 for d in diff if d == diff[0]) / len(diff) * 100
    if all(d == diff[0] for d in diff):
        return "Arithmetic Progression", diff[0], ap_confidence

    # Check for Geometric Progression (GP)
    gp_confidence = sum(1 for r in ratio if r == ratio[0]) / len(ratio) * 100
    if all(r == ratio[0] for r in ratio):
        return "Geometric Progression", ratio[0], gp_confidence

    # Check if the series is a Fibonacci sequence (each number is the sum of the previous two)
    fib_confidence = sum(1 for i in range(2, len(input_series)) if input_series[i] == input_series[i-1] + input_series[i-2]) / (len(input_series) - 2) * 100
    if len(input_series) >= 3 and all(input_series[i] == input_series[i-1] + input_series[i-2] for i in range(2, len(input_series))):
        return "Fibonacci Sequence", None, fib_confidence

    # Check for Harmonic Progression (HP)
    hp_confidence = sum(1 for i in range(1, len(input_series)) if round(1/input_series[i] - 1/input_series[i-1], 6) == diff[0]) / len(input_series) * 100
    if all(round(1/input_series[i] - 1/input_series[i-1], 6) == diff[0] for i in range(1, len(input_series))):
        return "Harmonic Progression", diff[0], hp_confidence

    # Factorial progression check
    factorial_confidence = 100 if input_series == [math.factorial(i) for i in range(1, 6)] else 0
    if input_series == [math.factorial(i) for i in range(1, 6)]:
        return "Factorial Progression", None, factorial_confidence

    return "Series pattern not recognized.", None, 0

# Function to compute the next numbers based on the identified sequence type
def compute_next_numbers(input_series, sequence_type, value):
    next_numbers = []

    if sequence_type == "Arithmetic Progression":
        next_numbers = [input_series[-1] + value * (i + 1) for i in range(3)]
    elif sequence_type == "Geometric Progression":
        next_numbers = [input_series[-1] * (value ** (i + 1)) for i in range(3)]
    elif sequence_type == "Fibonacci Sequence":
        temp_series = input_series[:]
        for _ in range(3):
            next_number = temp_series[-1] + temp_series[-2]
            next_numbers.append(next_number)
            temp_series.append(next_number)
    elif sequence_type == "Harmonic Progression":
        next_numbers = [1 / (1 / input_series[-1] + value * (i + 1)) for i in range(3)]
    elif sequence_type == "Factorial Progression":
        next_numbers = [math.factorial(i) for i in range(6, 9)]
    
    return next_numbers

# Function to generate the graph and save it
def generate_graph(input_series, next_numbers, sequence_type):
    x = list(range(1, len(input_series) + 1))
    x_new = list(range(len(input_series) + 1, len(input_series) + 4))

    plt.figure(figsize=(8, 5))
    plt.plot(x, input_series, 'bo-', label="Input Series")
    plt.scatter(x_new, next_numbers, color='red', label="Predicted Numbers")
    plt.plot([x[-1], x_new[-1]], [input_series[-1], next_numbers[-1]], 'orange', linestyle='dashed', label="Forecast Line")
    plt.xlabel('Position')
    plt.ylabel('Value')
    plt.title(f'{sequence_type} - Predicted Next Numbers')
    plt.legend()
    plt.grid(True)

    # Save the plot as an image in the static folder
    img_path = os.path.join('static', 'graph.png')
    plt.savefig(img_path, format='png')
    plt.close()  # Close the plot to avoid memory issues

    return img_path

@app.route("/how-it-works")
def how_it_works():
    return render_template("how-it-works.html")

@app.route("/about-me")
def about_me():
    return render_template('about-me.html')  

@app.route("/", methods=["GET", "POST"])
def index():
    graph_url = None
    input_series = []
    predicted_numbers = []
    confidence = None
    sequence_type = None
    sequence_confidence = None
    error_message = None

    if request.method == "POST":
        # Get the input series from the form
        input_series = request.form.get("input_series").split()
        input_series = [float(x) for x in input_series]

        # Check if the input series contains exactly 5 numbers
        if len(input_series) != 5:
            error_message = "Please provide exactly 5 numbers in the input series."
        else:
            # Identify sequence type and confidence
            sequence_type, value, sequence_confidence = identify_sequence_type(input_series)

            # If sequence type is recognized, compute the next numbers
            if sequence_type != "Series pattern not recognized.":
                predicted_numbers = compute_next_numbers(input_series, sequence_type, value)
                # Ensure predicted numbers are converted to Python float and rounded
                predicted_numbers = [round(num, 2) for num in predicted_numbers]
            else:
                # Use polynomial regression if no sequence is recognized
                sequence_type = "Polynomial Regression"
                X = np.array(range(1, len(input_series) + 1)).reshape(-1, 1)
                y = np.array(input_series)
                poly = PolynomialFeatures(degree=2)
                X_poly = poly.fit_transform(X)
                model = LinearRegression()
                model.fit(X_poly, y)

                # Predict next 3 numbers
                X_new = np.array(range(len(input_series) + 1, len(input_series) + 4)).reshape(-1, 1)
                X_new_poly = poly.transform(X_new)
                predicted_numbers = model.predict(X_new_poly)

                # Calculate confidence (R² score)
                r2_score = model.score(X_poly, y)
                confidence = r2_score * 100  # Confidence as percentage
                # Ensure predicted numbers are converted to Python float and rounded
                predicted_numbers = [round(num, 2) for num in predicted_numbers]

            # Round the values to 2 decimal places for input series and confidence
            input_series = [round(num, 2) for num in input_series]
            if confidence is not None:
                confidence = round(confidence, 2)
            if sequence_confidence is not None:
                sequence_confidence = round(sequence_confidence, 2)

            # Generate graph with the prediction
            graph_url = generate_graph(input_series, predicted_numbers, sequence_type)

    return render_template("index.html", graph_url=graph_url, input_series=input_series, predicted_numbers=predicted_numbers, confidence=confidence, sequence_confidence=sequence_confidence, error_message=error_message)

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))
