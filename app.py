import streamlit as st
import requests
import math
import folium
import polyline
import branca.colormap as cm
from streamlit_folium import st_folium
import numpy as np
import pandas as pd
import json
import os
from datetime import datetime, timezone

# ==========================================
# 1. CONSTANTS & DATABASES
# ==========================================
DB_FILE = "segment_database.json"
AIR_DENSITY = 1.225      
GRAVITY = 9.81           
DRIVETRAIN_LOSS = 0.03

st.set_page_config(page_title="KOM Hunter", layout="wide")

@st.cache_data
def load_database():
    if os.path.exists(DB_FILE):
        with open(DB_FILE, "r") as f:
            return json.load(f)
    return {}

# ==========================================
# 2. MATH & PHYSICS ENGINE
# ==========================================
def estimate_cda(weight, height, position_mod, clothing_mod, helmet_mod, drafting_mod):
    bsa = 0.007184 * (weight ** 0.425) * (height ** 0.725)
    return bsa * position_mod * clothing_mod * helmet_mod * drafting_mod

def estimate_max_power(duration_sec, p_max, w_prime_kj, cp):
    if duration_sec <= 0: return p_max
    w_prime_joules = w_prime_kj * 1000.0
    if p_max <= cp: return cp
    k = w_prime_joules / (p_max - cp)
    return float(cp + (w_prime_joules / (duration_sec + k)))

def get_live_weather(lat, lon, hours_ahead=0):
    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": lat,
        "longitude": lon,
        "wind_speed_unit": "ms",
        "timezone": "Europe/Copenhagen"
    }
    
    if hours_ahead == 0:
        params["current"] = "wind_speed_10m,wind_direction_10m"
    else:
        params["hourly"] = "wind_speed_10m,wind_direction_10m"
        
    try:
        response = requests.get(url, params=params, timeout=5)
        if response.status_code == 200:
            data = response.json()
            if hours_ahead == 0:
                return data["current"]["wind_speed_10m"], data["current"]["wind_direction_10m"]
            else:
                now = datetime.now(timezone.utc)
                for i in range(len(data["hourly"]["time"])):
                    if i == hours_ahead: 
                        return data["hourly"]["wind_speed_10m"][i], data["hourly"]["wind_direction_10m"][i]
    except:
        pass
    return 0.0, 0.0

def calculate_dynamic_power_with_wind(streams, target_time, system_weight, CdA, Crr, wind_speed, wind_dir):
    distances = np.array(streams['distance'])
    altitudes = np.array(streams['altitude'])
    latlngs = np.array(streams['latlng'])
    
    if distances[-1] <= 0 or len(distances) < 2:
        return 0, 0
        
    d = np.diff(distances)
    valid = d > 0
    d = d[valid]
    elev = np.diff(altitudes)[valid]
    grade = elev / d
    
    lat1 = np.radians(latlngs[:-1, 0][valid])
    lon1 = np.radians(latlngs[:-1, 1][valid])
    lat2 = np.radians(latlngs[1:, 0][valid])
    lon2 = np.radians(latlngs[1:, 1][valid])
    
    dlon = lon2 - lon1
    x = np.sin(dlon) * np.cos(lat2)
    y = np.cos(lat1) * np.sin(lat2) - (np.sin(lat1) * np.cos(lat2) * np.cos(dlon))
    bearings = (np.degrees(np.arctan2(x, y)) + 360) % 360
    
    hw = wind_speed * np.cos(np.radians(wind_dir - bearings))
    avg_wind_effect = np.mean(hw)
    
    c_aero = 0.5 * AIR_DENSITY * CdA
    c_grav = system_weight * GRAVITY * (grade + Crr)
    eta = 1.0 - DRIVETRAIN_LOSS
    
    low_power, high_power = 0.0, 2500.0
    best_power = 0.0
    
    for _ in range(30): 
        mid_power = (low_power + high_power) / 2.0
        P_eff = mid_power * eta
        
        v_low = np.zeros_like(d)
        v_high = np.full_like(d, 40.0) 
        
        for _ in range(20):
            v_mid = (v_low + v_high) / 2.0
            v_air = v_mid + hw
            F_aero = c_aero * v_air * np.abs(v_air)
            F_resist = c_grav + F_aero
            
            P_req = F_resist * v_mid
            
            mask = P_req > P_eff
            v_high[mask] = v_mid[mask]
            v_low[~mask] = v_mid[~mask]
            
        v = (v_low + v_high) / 2.0
        v = np.maximum(v, 0.1) 
        
        total_time = np.sum(d / v)
        
        if total_time > target_time:
            low_power = mid_power 
        else:
            high_power = mid_power
            best_power = mid_power
            
    return best_power, avg_wind_effect

# ==========================================
# 3. SIDEBAR UI
# ==========================================
with st.sidebar:
    st.header("🎯 Target Filters")
    max_dist = st.slider("Max Distance (m)", min_value=200, max_value=5000, value=5000, step=100)

    st.header("👥 Drafting Strategy")
    drafting = st.selectbox(
        "Tactics",
        options=[
            ("Solo Effort", 1.00),
            ("2-Man Paceline (50/50 Pulls)", 0.85),
            ("Sitting in the Draft (0% Pull)", 0.70)
        ],
        format_func=lambda x: x[0],
        index=0
    )

    st.header("🚴 Rider Profile")
    rider_weight = st.number_input("Rider Weight (kg)", value=None, placeholder="e.g. 85.0", step=1.0)
    rider_height = st.number_input("Height (cm)", value=None, placeholder="e.g. 191", step=1.0)
    bike_weight = st.number_input("Bike & Gear Weight (kg)", value=None, placeholder="e.g. 8.5", step=0.1)

    st.markdown("---")
    position = st.selectbox(
        "Riding Position",
        options=[
            ("Aero Breakaway / Puppy Paws", 0.15),
            ("In the Drops", 0.17),
            ("Aero Hoods (Forearms flat)", 0.18),
            ("Upright Hoods", 0.20),
            ("On the Tops", 0.22)
        ],
        format_func=lambda x: x[0],
        index=2
    )
    
    clothing = st.selectbox(
        "Clothing",
        options=[
            ("Standard Club Fit", 1.00),
            ("Tight Aero Jersey", 0.95),
            ("Full Skinsuit", 0.90)
        ],
        format_func=lambda x: x[0],
        index=1
    )

    helmet = st.selectbox(
        "Helmet",
        options=[
            ("Vented Climbing Helmet", 1.00),
            ("Aero Road Helmet", 0.97),
            ("Full TT Helmet", 0.94)
        ],
        format_func=lambda x: x[0],
        index=1
    )

    surface = st.selectbox(
        "Tires & Surface",
        options=[
            ("Race Tubeless (Smooth Tarmac)", 0.0030),
            ("Standard Clinchers", 0.0045),
            ("Rough Asphalt", 0.0060),
            ("Gravel Setup", 0.0120)
        ],
        format_func=lambda x: x[0]
    )

    st.header("⚡ Power Curve")
    p_max = st.number_input("Peak Sprint (5s) Watts", value=None, placeholder="e.g. 1100", step=25)
    w_prime_kj = st.slider("W' (kJ) [Steady: 15, Punchy: 22, Sprint: 30]", min_value=10.0, max_value=35.0, value=20.0, step=0.5)
    cp = st.number_input("Critical Power / FTP (W)", value=None, placeholder="e.g. 280", step=5)

    st.header("🌦️ Environment")
    hours_ahead = st.slider("Forecast Hours Ahead", 0, 12, 0)

# ==========================================
# 4. MAIN DASHBOARD EXECUTION
# ==========================================
st.title("🏆 Segment Physics Engine")

db = load_database()
if not db:
    st.warning("No segment_database.json found. Please run the Python extractor script first.")
    st.stop()

col_metric1, col_metric2 = st.columns(2)
with col_metric1:
    st.metric("Total Segments in Database", len(db))

if None in [rider_weight, rider_height, bike_weight, p_max, cp]:
    st.info("👈 Please fill out your physical profile, aerodynamics, and power metrics in the sidebar to calculate map physics.")
    st.stop()

system_weight = rider_weight + bike_weight
CdA = estimate_cda(rider_weight, rider_height, position[1], clothing[1], helmet[1], drafting[1])
Crr = surface[1]

with st.sidebar.expander("📈 View Your Power Curve", expanded=False):
    curve_times = np.linspace(5, 1200, 100)
    curve_watts = [estimate_max_power(t, p_max, w_prime_kj, cp) for t in curve_times]
    chart_data = {"Seconds": curve_times, "Max Watts": curve_watts}
    st.line_chart(chart_data, x="Seconds", y="Max Watts")
    st.caption(f"Calculated CdA: {CdA:.3f} | System Mass: {system_weight} kg")

first_seg = list(db.values())[0]
c_lat, c_lon = first_seg["start_latlng"]
wind_speed, wind_dir = get_live_weather(c_lat, c_lon, hours_ahead)

with col_metric2:
    st.metric("Live Forecast Wind", f"{wind_speed:.1f} m/s", f"From {wind_dir}°")

with st.spinner("Integrating meter-by-meter physics..."):
    m = folium.Map(location=[c_lat, c_lon], zoom_start=11, tiles="CartoDB positron")
    
    # Calculate map boundaries for the wind grid
    all_lats = [data["start_latlng"][0] for data in db.values()]
    all_lons = [data["start_latlng"][1] for data in db.values()]
    min_lat, max_lat = min(all_lats) - 0.02, max(all_lats) + 0.02
    min_lon, max_lon = min(all_lons) - 0.02, max(all_lons) + 0.02
    
    # Generate background wind grid
    lat_grid = np.linspace(min_lat, max_lat, 7)
    lon_grid = np.linspace(min_lon, max_lon, 7)
    
    wind_bg_html = f"""
    <div style="font-size: 24px; transform: rotate({(wind_dir + 180) % 360}deg); color: rgba(0, 191, 255, 0.25); text-align: center;">
        ⬆
    </div>
    """
    for glat in lat_grid:
        for glon in lon_grid:
            folium.Marker(
                location=[glat, glon],
                icon=folium.DivIcon(html=wind_bg_html, icon_size=(30, 30), icon_anchor=(15, 15))
            ).add_to(m)

    # Add primary central wind speed indicator
    wind_center_html = f"""
    <div style="text-align: center; width: 80px; transform: translate(-40px, -40px);">
        <div style="font-size: 13px; font-weight: bold; background: white; padding: 2px 4px; border-radius: 4px; border: 1px solid #ccc; box-shadow: 2px 2px 4px rgba(0,0,0,0.3);">
            {wind_speed:.1f} m/s
        </div>
    </div>
    """
    folium.Marker(
        location=[c_lat, c_lon], 
        icon=folium.DivIcon(html=wind_center_html)
    ).add_to(m)
    
    colormap = cm.LinearColormap(
        colors=['green', 'yellow', 'orange', 'red', 'darkred'],
        index=[0.5, 0.8, 0.95, 1.05, 1.2], vmin=0.5, vmax=1.2,
        caption='Difficulty Ratio (Req Watts / Your Max Power)'
    )
    m.add_child(colormap)
    
    table_data = []
    
    for seg_id, data in db.items():
        dist = data["distance"]
        if dist > max_dist: continue # Apply distance filter
        if "streams" not in data or data["kom_sec"] <= 0: continue
        
        kom_sec = data["kom_sec"]
        attempts = data.get("effort_count", 0)
        mins, secs = divmod(kom_sec, 60)
        time_str = f"{mins}:{secs:02d}"
        
        req_watts_wind, hw_avg = calculate_dynamic_power_with_wind(
            data["streams"], kom_sec, system_weight, CdA, Crr, wind_speed, wind_dir
        )
        
        req_watts_nowind, _ = calculate_dynamic_power_with_wind(
            data["streams"], kom_sec, system_weight, CdA, Crr, 0.0, 0.0
        )
        
        wind_str = f"{abs(hw_avg):.1f} m/s {'Headwind' if hw_avg > 0 else 'Tailwind'}"
        
        user_max = estimate_max_power(kom_sec, p_max, w_prime_kj, cp)
        difficulty_ratio = req_watts_wind / user_max if user_max > 0 else 2.0
        possible = req_watts_wind <= user_max
        watt_margin = int(round(user_max - req_watts_wind))
        
        table_data.append({
            "Segment": data["name"],
            "Link": f"https://www.strava.com/segments/{seg_id}",
            "Attempts": attempts,
            "Dist (m)": round(dist),
            "Watts (Wind)": int(round(req_watts_wind)),
            "Margin (W)": watt_margin,
            "Your Limit": int(round(user_max)),
            "Possible?": "🟢" if possible else "🔴"
        })
        
        route = polyline.decode(data["polyline"])
        line_color = colormap(difficulty_ratio)
        
        tooltip_html = f"""
        <div style='font-family: Arial; font-size: 13px; min-width: 150px;'>
            <b>{data['name']}</b><br>
            <hr style='margin: 2px 0;'>
            <b>Attempts:</b> {attempts:,}<br>
            <b>Distance:</b> {round(dist)} m<br>
            <b>Grade:</b> {data['average_grade']}%<br>
            <b>Time to beat:</b> {time_str} ({kom_sec}s)<br>
            <b>Target Speed:</b> {round((dist / kom_sec) * 3.6, 1)} km/h<br>
            <b>Avg Wind:</b> {wind_str}<br>
            <hr style='margin: 2px 0;'>
            <b>Req Watts (With Wind):</b> <span style='color: {"#d32f2f" if not possible else "#388e3c"}; font-weight: bold;'>{int(round(req_watts_wind))}W</span><br>
            <b>Req Watts (No Wind):</b> <span style='font-weight: bold;'>{int(round(req_watts_nowind))}W</span><br>
            <b>Your Limit:</b> {int(round(user_max))}W<br>
            <b>Margin:</b> {watt_margin}W {'(To Spare)' if watt_margin >= 0 else '(Short)'}
        </div>
        """
        
        folium.PolyLine(
            route, color=line_color, weight=3, opacity=0.8, tooltip=folium.Tooltip(tooltip_html)
        ).add_to(m)

    col1, col2 = st.columns([2, 1.2])
    with col1:
        st_folium(m, width=750, height=600, returned_objects=[])
    with col2:
        df = pd.DataFrame(table_data)
        
        sort_mode = st.selectbox(
            "Sort Segments By:", 
            ["Largest Watt Margin (Easiest)", "Most Attempts", "Most Attempts (Achievable Only)"]
        )
        
        if not df.empty:
            if sort_mode == "Largest Watt Margin (Easiest)":
                df = df.sort_values("Margin (W)", ascending=False)
            elif sort_mode == "Most Attempts":
                df = df.sort_values("Attempts", ascending=False)
            elif sort_mode == "Most Attempts (Achievable Only)":
                df = df[df["Possible?"] == "🟢"].sort_values("Attempts", ascending=False)
                
            st.dataframe(
                df, 
                height=530, 
                use_container_width=True,
                column_config={
                    "Link": st.column_config.LinkColumn("Strava", display_text="View")
                }
            )