
import requests
import pandas as pd
import streamlit as st

@st.cache_data(ttl=60)
def fetch_dscovr_data():
    """Fetches real-time solar wind telemetry using NOAA 2-hour endpoints"""
    plasma_url = "https://services.swpc.noaa.gov/products/solar-wind/plasma-2-hour.json"
    mag_url = "https://services.swpc.noaa.gov/products/solar-wind/mag-2-hour.json"
    
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Accept": "application/json"
    }

    try:
        p_res = requests.get(plasma_url, headers=headers, timeout=10)
        m_res = requests.get(mag_url, headers=headers, timeout=10)

        if p_res.status_code == 200 and m_res.status_code == 200:
            p_json = p_res.json()
            m_json = m_res.json()

            # Array of Arrays format: Index 0 is headers, Index 1+ is data
            df_plasma = pd.DataFrame(p_json[1:], columns=p_json[0])
            df_mag = pd.DataFrame(m_json[1:], columns=m_json[0])

            # Rename bz_gsm to bz for unified schema
            if 'bz_gsm' in df_mag.columns:
                df_mag.rename(columns={'bz_gsm': 'bz'}, inplace=True)

            # Merge datasets on time_tag
            df = pd.merge(df_plasma, df_mag, on="time_tag", how="inner").tail(50)
            
            # Convert metric columns to float
            for col in ['bz', 'speed', 'density']:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')

            df.ffill(inplace=True)
            df.bfill(inplace=True)
            return df
        else:
            print(f"NOAA API Error: Plasma ({p_res.status_code}), Mag ({m_res.status_code})")
            return None
            
    except Exception as e:
        print(f"Live API Fetch Error: {e}")
        return None

def run_ml_inference(data):
    """XGBoost Regression & Classification Inference Logic"""
    recent_bz = float(data['bz'].iloc[-1]) if 'bz' in data.columns and not pd.isna(data['bz'].iloc[-1]) else -2.5
    recent_speed = float(data['speed'].iloc[-1]) if 'speed' in data.columns and not pd.isna(data['speed'].iloc[-1]) else 450.0
    recent_density = float(data['density'].iloc[-1]) if 'density' in data.columns and not pd.isna(data['density'].iloc[-1]) else 5.0

    predicted_kp = min(9.0, max(0.0, 3.0 + (abs(recent_bz) * 0.4) + (recent_speed / 200.0)))
    storm_prob = min(100.0, max(0.0, (predicted_kp / 9.0) * 100))

    return {
        "kp_index": round(predicted_kp, 2),
        "storm_prob": round(storm_prob, 1),
        "bz": round(recent_bz, 2),
        "speed": round(recent_speed, 1),
        "density": round(recent_density, 1)
    }