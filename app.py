import streamlit as st
import pickle
import pandas as pd
import numpy as np
import os
import plotly.graph_objects as go

# ------------------ PAGE CONFIG ------------------
st.set_page_config(
    page_title="AirSight | AQI Forecast",
    page_icon="🍃",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ------------------ CUSTOM CSS & STYLING ------------------
st.markdown("""
    <style>
    /* Main Background and Font */
    .stApp {
        background-color: #0e1117;
        font-family: 'Helvetica Neue', sans-serif;
    }
    
    /* Card Styling */
    .metric-card {
        background-color: #1f2937;
        border-radius: 15px;
        padding: 20px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3);
        border: 1px solid #374151;
        text-align: center;
        transition: transform 0.2s;
    }
    .metric-card:hover {
        transform: translateY(-5px);
    }
    
    /* Headers */
    h1, h2, h3 {
        color: #f3f4f6;
    }
    
    /* Custom Button */
    .stButton>button {
        background: linear-gradient(90deg, #4F46E5 0%, #7C3AED 100%);
        color: white;
        border: none;
        border-radius: 8px;
        font-weight: bold;
        transition: 0.3s;
    }
    .stButton>button:hover {
        opacity: 0.9;
        transform: scale(1.02);
    }
    </style>
    """, unsafe_allow_html=True)

# ------------------ FILE PATH HANDLING ------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "aqi_model.pkl")
ENCODER_PATH = os.path.join(BASE_DIR, "city_encoder.pkl")
DATA_PATH = os.path.join(BASE_DIR, "aqi.csv")

# ------------------ FUNCTIONS ------------------
@st.cache_resource
def load_assets():
    try:
        model = pickle.load(open(MODEL_PATH, "rb"))
        le = pickle.load(open(ENCODER_PATH, "rb"))
        df = pd.read_csv(DATA_PATH)
        # FIX: Use .str.title() on the string accessor (.str)
        df["City_clean"] = df["City"].astype(str).str.strip().str.title()
        
        # We need a lowercase version for robust matching with user input
        df["City_lookup"] = df["City"].astype(str).str.strip().str.lower()
        available_cities = sorted(df["City_clean"].unique())
        return model, le, df, available_cities
    except Exception as e:
        # st.error is usually better than a simple print for Streamlit
        st.error(f"Error loading files: {e}")
        return None, None, None, []

def get_aqi_details(aqi_value):
    """Returns color, category, and health message based on AQI"""
    if aqi_value <= 50:
        return "#00E400", "Good", "Air quality is satisfactory, and air pollution poses little or no risk. Enjoy the fresh air!"
    elif aqi_value <= 100:
        return "#FFFF00", "Satisfactory", "Air quality is acceptable. Sensitive groups may experience minor issues, but the general public is fine."
    elif aqi_value <= 200:
        return "#FF7E00", "Moderate", "Breathing discomfort possible for people with lung disease, asthma, and children. Minimize prolonged exertion outdoors."
    elif aqi_value <= 300:
        return "#FF0000", "Poor", "Breathing discomfort to most people on prolonged exposure. Avoid unnecessary outdoor activity."
    elif aqi_value <= 400:
        return "#8F3F97", "Very Poor", "Respiratory illness on prolonged exposure. Affects healthy people too. Stay indoors and use air purifiers."
    else:
        return "#7E0023", "Severe", "Respiratory effects even on healthy people. Serious impact on those with existing diseases. Strictly avoid outdoor activity."

def create_gauge(value, title, color):
    """Creates a plotly gauge chart"""
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=value,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': title, 'font': {'size': 24, 'color': "white"}},
        number={'font': {'color': "white"}},
        gauge={
            'axis': {'range': [0, 500], 'tickwidth': 1, 'tickcolor': "white"},
            'bar': {'color': color},
            'bgcolor': "rgba(0,0,0,0)",
            'borderwidth': 2,
            'bordercolor': "#333",
            'steps': [
                {'range': [0, 50], 'color': 'rgba(0, 228, 0, 0.3)'},
                {'range': [50, 100], 'color': 'rgba(255, 255, 0, 0.3)'},
                {'range': [100, 200], 'color': 'rgba(255, 126, 0, 0.3)'},
                {'range': [200, 300], 'color': 'rgba(255, 0, 0, 0.3)'},
                {'range': [300, 400], 'color': 'rgba(143, 63, 151, 0.3)'},
                {'range': [400, 500], 'color': 'rgba(126, 0, 35, 0.3)'}
            ],
        }
    ))
    fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", font={'color': "white", 'family': "Arial"})
    return fig

# ------------------ LOAD DATA ------------------
model, le, df, available_cities = load_assets()

# ------------------ SIDEBAR ------------------
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/3262/3262973.png", width=100)
    st.title("AirSight AI")
    st.markdown("---")
    st.markdown("""
    **About this App**
    
    This tool uses Machine Learning (Random Forest/XGBoost) to forecast the Air Quality Index (AQI) for the upcoming day based on historical pollution trends.
    """)
    st.markdown("---")
    st.caption("Built with Streamlit & Python")

# ------------------ MAIN UI ------------------
st.markdown("<h1 style='text-align: center; margin-bottom: 30px;'>🌫️ AI-Powered Air Quality Forecaster</h1>", unsafe_allow_html=True)

col1, col2, col3 = st.columns([1, 2, 1])

with col2:
    # UX CHANGE: Simple Text Input as requested
    city_input = st.text_input(
        "📍 Enter City Name",
        placeholder="e.g. Delhi, Mumbai, Ahmedabad (Case-insensitive)"
    )

    predict_btn = st.button("🚀 Analyze & Predict", use_container_width=True)

# ------------------ PREDICTION LOGIC ------------------
if predict_btn:
    if not city_input.strip():
        st.warning("⚠️ Please enter a city name.")
    else:
        try:
            city_search = city_input.strip().lower()
            
            # Check if city exists using the City_lookup column
            if city_search not in df["City_lookup"].values:
                st.error(f"❌ City **'{city_input.strip()}'** not found in the dataset. Please check the spelling.")
            else:
                with st.spinner(f"Crunching data for {city_input.strip()}..."):
                    
                    # Get the latest record for that city (case-insensitive search)
                    latest = (
                        df[df["City_lookup"] == city_search]
                        .sort_values("Date")
                        .iloc[-1]
                    )

                    # Get the proper City Name (as needed by the Label Encoder)
                    original_city_name = latest['City'] 
                    
                    # Prepare model input
                    input_data = np.array([[
                        latest["AQI_lag_1"],
                        latest["AQI_lag_7"],
                        latest["AQI_roll7_mean"],
                        le.transform([original_city_name])[0]
                    ]])

                    prediction = round(model.predict(input_data)[0])
                    
                    # Get visual details
                    color, category, advice = get_aqi_details(prediction)

                # ------------------ RESULTS DASHBOARD ------------------
                st.markdown("---")
                
                # Layout: Gauge on Left, Details on Right
                res_col1, res_col2 = st.columns([1.5, 1])
                
                with res_col1:
                    st.plotly_chart(create_gauge(prediction, f"Predicted AQI for {latest['City_clean']}", color), use_container_width=True)
                    
                with res_col2:
                    st.markdown("<br>", unsafe_allow_html=True) # Spacer
                    st.markdown(
                        f"""
                        <div class="metric-card" style="border-left: 10px solid {color}; text-align: left;">
                            <h3 style="margin:0; color: #aaa;">Forecast Status</h3>
                            <h1 style="margin:0; font-size: 3rem; color: {color};">{category}</h1>
                            <hr style="border-color: #555;">
                            <p style="font-size: 1.1rem;">**Health Advisory:** {advice}</p>
                        </div>
                        """, 
                        unsafe_allow_html=True
                    )
                    
                    # Historical context stats
                    st.markdown("### 📉 Recent Context")
                    st.info(f"The most recent AQI in the data (Yesterday's): **{int(latest['AQI_lag_1'])}**")
        
        except Exception as e:
            # Catching errors during prediction or data retrieval
            st.error(f"An unexpected error occurred during prediction: {str(e)}")