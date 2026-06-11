
import streamlit as st
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder

# Train model
np.random.seed(42)
n = 300

df = pd.DataFrame({
    'Area_sqft'    : np.random.randint(500, 5000, n),
    'Bedrooms'     : np.random.randint(1, 6, n),
    'Bathrooms'    : np.random.randint(1, 4, n),
    'Age_of_House' : np.random.randint(1, 50, n),
    'Garage'       : np.random.choice(['Yes', 'No'], n),
    'Location'     : np.random.choice(['Urban', 'Suburb', 'Rural'], n),
})

df['Price'] = (
    df['Area_sqft']    * 150 +
    df['Bedrooms']     * 10000 +
    df['Bathrooms']    * 8000 -
    df['Age_of_House'] * 500 +
    (df['Garage'] == 'Yes').astype(int) * 15000 +
    np.random.randint(-20000, 20000, n)
)

df['Garage_enc']   = LabelEncoder().fit_transform(df['Garage'])
df['Location_enc'] = LabelEncoder().fit_transform(df['Location'])

X = df[['Area_sqft','Bedrooms','Bathrooms','Age_of_House','Garage_enc','Location_enc']]
y = df['Price']

model = LinearRegression()
model.fit(X, y)

# Currency rates (vs USD)
currency_options = {
    "🇺🇸 USD - US Dollar"        : ("$",  1.0),
    "🇮🇳 INR - Indian Rupee"     : ("₹",  83.5),
    "🇬🇧 GBP - British Pound"    : ("£",  0.79),
    "🇪🇺 EUR - Euro"             : ("€",  0.92),
    "🇦🇪 AED - UAE Dirham"       : ("AED", 3.67),
    "🇸🇬 SGD - Singapore Dollar" : ("SGD", 1.34),
    "🇦🇺 AUD - Australian Dollar": ("A$", 1.53),
    "🇨🇦 CAD - Canadian Dollar"  : ("C$", 1.36),
}

# App UI
st.title("🏠 House Price Predictor")
st.write("Fill in the details below to predict the house price!")

# Currency selector
st.subheader("💱 Select Currency")
selected_currency = st.selectbox("Choose your currency", list(currency_options.keys()))
symbol, rate = currency_options[selected_currency]

st.divider()

# Inputs
st.subheader("🏡 House Details")
area      = st.number_input("Area (sqft)",         min_value=500,  max_value=5000, value=1500)
bedrooms  = st.number_input("Bedrooms",             min_value=1,    max_value=6,    value=3)
bathrooms = st.number_input("Bathrooms",            min_value=1,    max_value=4,    value=2)
age       = st.number_input("Age of House (yrs)",   min_value=1,    max_value=50,   value=10)
garage    = st.selectbox("Garage",   ["Yes", "No"])
location  = st.selectbox("Location", ["Urban", "Suburb", "Rural"])

garage_enc   = 1 if garage == "Yes" else 0
location_map = {"Urban": 2, "Suburb": 1, "Rural": 0}
location_enc = location_map[location]

# Predict
if st.button("🔮 Predict Price"):
    features      = np.array([[area, bedrooms, bathrooms, age, garage_enc, location_enc]])
    price_usd     = model.predict(features)[0]
    price_converted = price_usd * rate

    st.success(f"💰 Predicted House Price: {symbol} {price_converted:,.0f}")
    st.caption(f"( Base price in USD: $ {price_usd:,.0f} )")
    st.balloons()
