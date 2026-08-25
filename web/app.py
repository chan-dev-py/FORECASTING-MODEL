from flask import Flask, render_template, request, jsonify
import pandas as pd
import os

# ==========================================
# APP CONFIGURATION
# ==========================================

app = Flask(__name__)

# Get the main THESIS directory
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Data directory
DATA_DIR = os.path.join(BASE_DIR, "data")


# ==========================================
# LOAD DATA
# ==========================================

def load_data():

    forecasting_data = pd.read_csv(
        os.path.join(DATA_DIR, "forecasting_data.csv")
    )

    model_comparison = pd.read_csv(
        os.path.join(DATA_DIR, "model_comparison.csv")
    )

    future_forecast = pd.read_csv(
        os.path.join(DATA_DIR, "future_enrollment_forecast.csv")
    )

    future_validation = pd.read_csv(
        os.path.join(DATA_DIR, "future_forecast_validation.csv")
    )

    return (
        forecasting_data,
        model_comparison,
        future_forecast,
        future_validation
    )


# ==========================================
# HOME / DASHBOARD
# ==========================================

@app.route("/")
def index():

    (
        forecasting_data,
        model_comparison,
        future_forecast,
        future_validation
    ) = load_data()

    # Get best model
    best_model = model_comparison.iloc[0]

    # Dashboard statistics
    total_enrollment = forecasting_data[
        "TOTAL_ENROLLMENT"
    ].sum()

    total_programs = forecasting_data[
        "PROGRAM"
    ].nunique()

    total_forecast = future_forecast[
        "PREDICTED_ENROLLMENT"
    ].sum()

    return render_template(
        "index.html",

        total_enrollment=round(total_enrollment),

        total_programs=total_programs,

        total_forecast=round(total_forecast),

        best_model=best_model["MODEL"],

        rmse=round(best_model["RMSE"], 2),

        mae=round(best_model["MAE"], 2),

        mape=round(best_model["MAPE"], 2)
    )


# ==========================================
# FORECAST PAGE
# ==========================================

@app.route("/forecast")
def forecast():

    (
        forecasting_data,
        model_comparison,
        future_forecast,
        future_validation
    ) = load_data()

    programs = sorted(
        forecasting_data["PROGRAM"].unique()
    )

    selected_program = request.args.get(
        "program"
    )

    program_data = None
    program_forecast = None

    if selected_program:

        program_data = forecasting_data[
            forecasting_data["PROGRAM"]
            == selected_program
        ].sort_values(
            ["ACADEMIC_YEAR", "SEMESTER"]
        )

        program_forecast = future_forecast[
            future_forecast["PROGRAM"]
            == selected_program
        ].sort_values(
            ["ACADEMIC_YEAR", "SEMESTER"]
        )

    return render_template(
        "forecast.html",

        programs=programs,

        selected_program=selected_program,

        program_data=program_data,

        program_forecast=program_forecast
    )


# ==========================================
# SIMULATION PAGE
# ==========================================

@app.route("/simulation")
def simulation():

    (
        forecasting_data,
        model_comparison,
        future_forecast,
        future_validation
    ) = load_data()

    programs = sorted(
        future_forecast["PROGRAM"].unique()
    )

    return render_template(
        "simulation.html",
        programs=programs
    )


# ==========================================
# SIMULATION API
# ==========================================

@app.route("/api/simulate", methods=["POST"])
def simulate():

    data = request.get_json()

    selected_program = data.get("program")
    adjustment = float(data.get("adjustment", 0))

    (
        forecasting_data,
        model_comparison,
        future_forecast,
        future_validation
    ) = load_data()

    # Filter selected program
    program_forecast = future_forecast[
        future_forecast["PROGRAM"]
        == selected_program
    ].copy()

    if program_forecast.empty:

        return jsonify({
            "error": "Program not found."
        }), 404

    # Calculate simulation
    program_forecast[
        "SIMULATED_ENROLLMENT"
    ] = (
        program_forecast[
            "PREDICTED_ENROLLMENT"
        ]
        * (1 + adjustment / 100)
    ).round(0)

    # Prevent negative enrollment
    program_forecast[
        "SIMULATED_ENROLLMENT"
    ] = program_forecast[
        "SIMULATED_ENROLLMENT"
    ].clip(lower=0)

    # Convert results to JSON
    results = program_forecast[
        [
            "ACADEMIC_YEAR",
            "SEMESTER",
            "PREDICTED_ENROLLMENT",
            "SIMULATED_ENROLLMENT"
        ]
    ].to_dict(
        orient="records"
    )

    return jsonify({
        "program": selected_program,
        "adjustment": adjustment,
        "results": results
    })


# ==========================================
# MODEL PERFORMANCE PAGE
# ==========================================

@app.route("/model-performance")
def model_performance():

    (
        forecasting_data,
        model_comparison,
        future_forecast,
        future_validation
    ) = load_data()

    return render_template(
        "model_performance.html",

        models=model_comparison.to_dict(
            orient="records"
        )
    )


# ==========================================
# FUTURE FORECAST VALIDATION PAGE
# ==========================================

@app.route("/validation")
def validation():

    (
        forecasting_data,
        model_comparison,
        future_forecast,
        future_validation
    ) = load_data()

    return render_template(
        "validation.html",

        validation_results=future_validation.to_dict(
            orient="records"
        )
    )


# ==========================================
# RUN APPLICATION
# ==========================================

if __name__ == "__main__":

    app.run(
        debug=True
    )