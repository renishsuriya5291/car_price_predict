import pickle
import streamlit as st
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import datetime

# Load the saved Random Forest model and fitted LabelEncoder
with open("random_forest_model.pkl", 'rb') as f:
    rf_model, lb = pickle.load(f)

# Function to predict price based on input dictionary
def predict_price(input_data):
    # preprocess input data (similar to the preprocessing in your training code)
    input_df = preprocess_input(input_data, lb)
    
    # predict the price using the loaded model
    predicted_price = rf_model.predict(input_df)[0]
    return predicted_price

# Preprocessing function
def preprocess_input(input_data, lb):
    # create a DataFrame with the input data
    input_df = pd.DataFrame([input_data])

    # preprocess input data
    input_df['name'] = input_df['name'].str.lower().str.replace(' ', '').str.split(' ').str.get(0)  # Convert to lowercase, remove spaces, and extract the first word
    input_df['name_encoded'] = lb.transform(input_df['name'])  # Use the same LabelEncoder instance
    
    fuel_dic = {'Petrol': 1, 'Diesel': 2, 'CNG': 3, 'LPG': 4}
    owner_dic = {'First Owner': 1, 'Second Owner': 2, 'Third Owner': 3, 'Fourth & Above Owner': 4}
    transmission_dic = {'Manual': 1, 'Automatic': 2}
    seller_type_dic = {'Individual': 1, 'Dealer': 2, 'Trustmark Dealer': 3}

    input_df['fuel'] = input_df['fuel'].map(fuel_dic)
    input_df['owner'] = input_df['owner'].map(owner_dic)
    input_df['transmission'] = input_df['transmission'].map(transmission_dic)
    input_df['seller_type'] = input_df['seller_type'].map(seller_type_dic)

    current_year = datetime.datetime.now().year
    input_df['age'] = current_year - input_df['year']
    input_df.drop(['year', 'name'], axis=1, inplace=True)  # Drop 'year' and 'name'
    
    # Reorder the columns to match the order during training
    input_df = input_df[['km_driven', 'fuel', 'seller_type', 'transmission', 
                         'owner', 'mileage', 'engine', 'max_power', 'seats', 'age', 'name_encoded']]
    
    return input_df

# Streamlit UI code
def main():
    st.title('Car Price Predictor')

    # Input fields
    name = st.text_input('Car Name', '')
    year = st.number_input('Year', min_value=1900, max_value=2024, step=1)
    km_driven = st.number_input('Kilometers Driven', value=0)
    fuel = st.selectbox('Fuel Type', ['Petrol', 'Diesel', 'CNG', 'LPG'])
    seller_type = st.selectbox('Seller Type', ['Individual', 'Dealer', 'Trustmark Dealer'])
    transmission = st.selectbox('Transmission', ['Manual', 'Automatic'])
    owner = st.selectbox('Owner', ['First Owner', 'Second Owner', 'Third Owner', 'Fourth & Above Owner'])
    mileage = st.number_input('Mileage', value=0.0)
    engine = st.number_input('Engine Capacity (cc)', value=0)
    max_power = st.number_input('Maximum Power (bhp)', value=0.0)
    seats = st.number_input('Number of Seats', value=0)

    # Predict button
    if st.button('Predict'):
        input_data = {
            'name': name,
            'year': year,
            'km_driven': km_driven,
            'fuel': fuel,
            'seller_type': seller_type,
            'transmission': transmission,
            'owner': owner,
            'mileage': mileage,
            'engine': engine,
            'max_power': max_power,
            'seats': seats
        }

        predicted_price = predict_price(input_data)

        # Display prediction
        st.subheader('Predicted Price:')
        st.write(f'₹ {predicted_price:.2f}')

if __name__ == '__main__':
    main()
