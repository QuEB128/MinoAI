import streamlit as st
import pandas as pd
import joblib


st.set_page_config(
    page_title="MinoAI | Price Prediction",
    page_icon="🏠",
    layout="wide"
)

# Session State Initialization
if "prediction" not in st.session_state:
    st.session_state.prediction = None

# Load model artifacts
model = joblib.load("./models/random_forest_model.pkl")
feature_columns = joblib.load("./models/feature_columns.pkl")

# Custom CSS
st.markdown(
    """
    <style>
        .main {
            padding: 2rem;
        }
        .title {
            font-size: 3rem;
            font-weight: 700;
            color: #0f172a;
        }
        .subtitle {
            font-size: 1.2rem;
            color: #475569;
            margin-bottom: 2rem;
        }
        .metric-box {
            padding: 2rem;
            border-radius: 12px;
            background-color: #0f172a;
            box-shadow: 0px 4px 12px rgba(0,0,0,0.25);
            text-align: center;
            color: white;
        }

        .metric-box h2 {
            color: #22c55e;
            font-size: 2.5rem;
            margin-bottom: 0.5rem;
        }

        .metric-box p {
            color: #cbd5e1;
            font-size: 1rem;
        }

        .footer {
            text-align: center;
            color: #64748b;
            margin-top: 4rem;
            font-size: 0.9rem;
        }
    </style>
    """,
    unsafe_allow_html=True
)

# Header
st.markdown('<div class="title">MinoAI Price Predictor</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="subtitle">AI-powered Airbnb price estimation using machine learning</div>',
    unsafe_allow_html=True
)

# Sidebar Inputs
st.sidebar.header("📋 Listing Details")

neighbourhood_group = st.sidebar.selectbox(
    "Neighbourhood Group",
    ["Brooklyn", "Manhattan", "Queens", "Bronx", "Staten Island"]
)

room_type = st.sidebar.selectbox(
    "Room Type",
    ["Entire home/apt", "Private room", "Shared room"]
)

minimum_nights = st.sidebar.number_input(
    "Minimum Nights",
    min_value=1,
    max_value=365,
    value=3
)

number_of_reviews = st.sidebar.number_input(
    "Number of Reviews",
    min_value=0,
    value=10
)

reviews_per_month = st.sidebar.number_input(
    "Reviews Per Month",
    min_value=0.0,
    value=1.5
)

availability_365 = st.sidebar.number_input(
    "Availability (Days/Year)",
    min_value=0,
    max_value=365,
    value=180
)

# Main Layout
col1, col2 = st.columns([2, 1])

# ---------- LEFT COLUMN ----------
with col1:
    st.subheader("📊 Listing Summary")

    st.write(
        """
        This application uses a **Random Forest regression model** trained on
        Airbnb listing data to estimate nightly prices.

        Adjust the parameters in the sidebar to see how listing attributes
        affect the predicted price.
        """
    )

    summary_df = pd.DataFrame({
        "Feature": [
            "Neighbourhood Group",
            "Room Type",
            "Minimum Nights",
            "Number of Reviews",
            "Reviews Per Month",
            "Availability (365 days)"
        ],
        "Value": [
            str(neighbourhood_group),
            str(room_type),
            str(minimum_nights),
            str(number_of_reviews),
            str(reviews_per_month),
            str(availability_365)
        ]
    })

    st.dataframe(summary_df, width="stretch")

# ---------- RIGHT COLUMN ----------
with col2:
    st.subheader("💰 Predicted Price")

    # Prepare input dataframe
    input_data = pd.DataFrame({
        "minimum_nights": [minimum_nights],
        "number_of_reviews": [number_of_reviews],
        "reviews_per_month": [reviews_per_month],
        "availability_365": [availability_365],
        "neighbourhood_group": [neighbourhood_group],
        "room_type": [room_type]
    })

    input_encoded = pd.get_dummies(input_data)
    input_encoded = input_encoded.reindex(columns=feature_columns, fill_value=0)

    # Predict button
    if st.button("🔮 Predict Price", width="stretch"):
        st.session_state.prediction = model.predict(input_encoded)[0]

    # Display prediction
    if st.session_state.prediction is None:
        st.markdown(
            """
            <div class="metric-box">
                <h2>—</h2>
                <p>Click "Predict Price" to see estimate</p>
            </div>
            """,
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            f"""
            <div class="metric-box">
                <h2>${st.session_state.prediction:.2f}</h2>
                <p>Estimated nightly price</p>
            </div>
            """,
            unsafe_allow_html=True
        )

# ===============================
# Footer
# ===============================
st.markdown(
    '<div class="footer">© 2026 MinoAI · Built with Streamlit & Machine Learning</div>',
    unsafe_allow_html=True
)
