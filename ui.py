import streamlit as st
import requests

st.title('House Price Prediction')
st.write('Enter details of the house to predict price')

with st.form('house_form'):
    col1, col2 = st.columns(2)

    with col1:
        MedInc = st.number_input("MedInc", min_value=0.0, max_value=20.0, value=0.0)
        HouseAge = st.number_input("HouseAge", min_value=0, max_value=60, value=0)
        AveRooms = st.number_input("AveRooms", min_value=0.0, max_value=30.0, value=0.0)
        AveBedrms = st.number_input("AveBedrms", min_value=0.0, max_value=6.0, value=0.0)

    with col2:
        Population = st.number_input("Population", min_value=0, max_value=40000, value=0)
        AveOccup = st.number_input("AveOccup", min_value=0.0, max_value=600.0, value=0.0)
        Latitude = st.number_input("Latitude", min_value=0.0, max_value=50.0, value=0.0)
        Longitude = st.number_input("Longitude", min_value=-130.0, max_value=-115.0, value=-120.0)
    
    submitted = st.form_submit_button("Predict House Price")

if submitted:

    house_data = {
        'MedInc': MedInc,
        'HouseAge': HouseAge,
        'AveRooms': AveRooms,
        'AveBedrms': AveBedrms,
        'Population': Population,
        'AveOccup': AveOccup,
        'Latitude': Latitude,
        'Longitude': Longitude,
    }

    try:
        response = requests.post('http://localhost:8000/predict', json=house_data)

        if response.status_code == 200:
            result = response.json()
            prediction = result['prediction']
            dollars = prediction * 100000

            st.subheader('Prediction Results')
            st.write(f'##### Predicted House Price: ${dollars:.3f}')

    except Exception as e:
        st.error(f'Connection Failed: {e}')