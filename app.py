from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
import joblib
import numpy as np
import requests

# ----------------------------
# Load Model
# ----------------------------
# FIXED: Updated filename to match your notebook output
model = joblib.load("solar_monthly_model.pkl")

app = FastAPI(title="Kerala Smart Solar Intelligence API")

# ----------------------------
# Enable CORS
# ----------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ----------------------------
# Constants
# ----------------------------
PR = 0.8
TEMP_COEFFICIENT = -0.004
COST_PER_KW = 60000
DEGRADATION_RATE = 0.005
YEARS_PROJECTION = 25

MONTH_MAP = {
    1: "JAN", 2: "FEB", 3: "MAR", 4: "APR",
    5: "MAY", 6: "JUN", 7: "JUL", 8: "AUG",
    9: "SEP", 10: "OCT", 11: "NOV", 12: "DEC"
}

DAYS_IN_MONTH = {
    1: 31, 2: 28, 3: 31, 4: 30,
    5: 31, 6: 30, 7: 31, 8: 31,
    9: 30, 10: 31, 11: 30, 12: 31
}

# ----------------------------
# User Input Schema
# ----------------------------
class SolarInput(BaseModel):
    Latitude: float
    Longitude: float
    Month: int
    System_Size: float
    Tariff: float
    Monthly_Consumption: float
    EV_Daily_KM: float
    EV_Efficiency: float
    Export_Tariff: float

    Tilt: Optional[float] = None
    Azimuth: Optional[float] = None
    NOCT: Optional[float] = 45

# ----------------------------
# Fetch NASA Climate
# ----------------------------
def fetch_nasa_climate(lat: float, lon: float):
    nasa_url = (
        "https://power.larc.nasa.gov/api/temporal/climatology/point"
        f"?parameters=T2M,RH2M,WS2M"
        f"&community=RE"
        f"&longitude={lon}"
        f"&latitude={lat}"
        f"&format=JSON"
    )

    try:
        response = requests.get(nasa_url, timeout=10)
        response.raise_for_status()
        return response.json()["properties"]["parameter"]
    except Exception:
        raise HTTPException(status_code=500, detail="Failed to fetch NASA data")


# ----------------------------
# Temperature Derating
# ----------------------------
def calculate_cell_temperature(
    ambient_temp: float,
    irradiance_kwh_m2_day: float,
    noct: float
) -> float:
    # Convert kWh/m²/day to average W/m²
    irradiance_w_m2 = irradiance_kwh_m2_day * 1000 / 24
    cell_temp = ambient_temp + ((noct - 20) / 800) * irradiance_w_m2
    return cell_temp


def temperature_derating_from_cell(cell_temp: float) -> float:
    temp_coefficient = -0.004  # -0.4% per °C
    effect = 1 + temp_coefficient * (cell_temp - 25)
    return max(0.75, min(effect, 1.05))


# ----------------------------
# Simplified Tilt Correction
# ----------------------------
def apply_tilt_correction(ghi: float, latitude: float, tilt: float) -> float:
    latitude_rad = np.radians(latitude)
    tilt_rad = np.radians(tilt)
    tilt_factor = np.cos(latitude_rad - tilt_rad) / np.cos(latitude_rad)
    tilt_factor = max(0.85, min(1.15, tilt_factor))
    return ghi * tilt_factor


# ----------------------------
# Predict Endpoint
# ----------------------------
@app.post("/predict")
def predict_solar(data: SolarInput):

    if data.Month < 1 or data.Month > 12:
        raise HTTPException(status_code=400, detail="Month must be between 1 and 12")

    # ----------------------------
    # Smart Defaults
    # ----------------------------
    def calculate_optimal_tilt(latitude: float) -> float:
        return abs(0.76 * latitude + 3.1)

    tilt = data.Tilt if data.Tilt is not None else calculate_optimal_tilt(data.Latitude)
    azimuth = data.Azimuth if data.Azimuth is not None else 180 

    # ----------------------------
    # Fetch Climate Data
    # ----------------------------
    climate = fetch_nasa_climate(data.Latitude, data.Longitude)

    monthly_results = []
    yearly_energy = 0

    # ----------------------------
    # Monthly Loop
    # ----------------------------
    for month in range(1, 13):
        month_key = MONTH_MAP[month]

        temp = climate["T2M"][month_key]
        humidity = climate["RH2M"][month_key]
        wind_speed = climate["WS2M"][month_key]

        month_sin = np.sin(2 * np.pi * month / 12)
        month_cos = np.cos(2 * np.pi * month / 12)

        # FIXED: Removed Latitude/Longitude from input to match your 5-feature model
        input_data = np.array([[
            temp,
            humidity,
            wind_speed,
            month_sin,
            month_cos
        ]])

        # ML Prediction (GHI in kWh/m²/day)
        predicted_ghi = float(model.predict(input_data)[0])

        # Apply Tilt Correction
        corrected_irradiance = apply_tilt_correction(
            predicted_ghi,
            data.Latitude,
            tilt
        )

        # Temperature Effect
        cell_temp = calculate_cell_temperature(
            ambient_temp=temp,
            irradiance_kwh_m2_day=corrected_irradiance,
            noct=data.NOCT
        )

        temp_effect = temperature_derating_from_cell(cell_temp)

        # Monthly Energy (kWh)
        monthly_energy = (
            data.System_Size *
            corrected_irradiance *
            DAYS_IN_MONTH[month] *
            PR *
            temp_effect
        )

        yearly_energy += monthly_energy

        monthly_results.append({
            "Month": month_key,
            "GHI_kWh_m2_day": round(predicted_ghi, 2),
            "Tilted_Irradiance_kWh_m2_day": round(corrected_irradiance, 2),
            "Temperature_C": round(temp, 2),
            "Monthly_Energy_kWh": round(monthly_energy, 2)
        })

    # ----------------------------
    # Consumption & Financials
    # ----------------------------
    yearly_home_consumption = data.Monthly_Consumption * 12
    ev_daily_energy = (data.EV_Daily_KM / data.EV_Efficiency if data.EV_Efficiency != 0 else 0)
    ev_yearly_energy = ev_daily_energy * 365

    total_yearly_usage = yearly_home_consumption + ev_yearly_energy
    surplus_energy = yearly_energy - total_yearly_usage
    total_cost = data.System_Size * COST_PER_KW

    if surplus_energy >= 0:
        savings_from_usage = total_yearly_usage * data.Tariff
        export_income = surplus_energy * data.Export_Tariff
        grid_purchase_cost = 0
    else:
        savings_from_usage = yearly_energy * data.Tariff
        export_income = 0
        grid_purchase_cost = abs(surplus_energy) * data.Tariff

    annual_net_benefit = savings_from_usage + export_income - grid_purchase_cost
    payback_years = total_cost / annual_net_benefit if annual_net_benefit > 0 else None
    co2_offset = yearly_energy * 0.82 / 1000

    # 25-Year Projection
    total_savings_25 = 0
    for year in range(YEARS_PROJECTION):
        degraded_energy = yearly_energy * ((1 - DEGRADATION_RATE) ** year)
        degraded_surplus = degraded_energy - total_yearly_usage
        if degraded_surplus >= 0:
            yearly_savings = (total_yearly_usage * data.Tariff + degraded_surplus * data.Export_Tariff)
        else:
            yearly_savings = (degraded_energy * data.Tariff - abs(degraded_surplus) * data.Tariff)
        total_savings_25 += yearly_savings

    net_profit_25 = total_savings_25 - total_cost

    return {
        "Panel_Tilt_Used": round(tilt, 2),
        "Panel_Azimuth_Used": round(azimuth, 2),
        "Optimal_Tilt_Suggested": round(calculate_optimal_tilt(data.Latitude), 2),
        "Monthly_Breakdown": monthly_results,
        "Solar_Generation_kWh_per_Year": round(yearly_energy, 2),
        "Home_Consumption_kWh_per_Year": round(yearly_home_consumption, 2),
        "EV_Consumption_kWh_per_Year": round(ev_yearly_energy, 2),
        "Total_Usage_kWh_per_Year": round(total_yearly_usage, 2),
        "Surplus_or_Deficit_kWh": round(surplus_energy, 2),
        "Export_Income_Rs": round(export_income, 2),
        "Grid_Purchase_Cost_Rs": round(grid_purchase_cost, 2),
        "Annual_Net_Benefit_Rs": round(annual_net_benefit, 2),
        "Payback_Years": round(payback_years, 2) if payback_years else None,
        "CO2_Offset_Tons_per_Year": round(co2_offset, 2),
        "25_Year_Total_Savings_Rs": round(total_savings_25, 2),
        "25_Year_Net_Profit_Rs": round(net_profit_25, 2)
    }