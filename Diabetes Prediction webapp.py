# -*- coding: utf-8 -*-
"""
Created on Thu Jul 20 03:58:18 2023

@author: HP
"""

import numpy as np;
import pickle
import streamlit as st
  # in python change back slash to forward slash
loaded_model=pickle.load(open('C:/Users/HP/Videos/python/DIABETES_PREDICTION/trained_model.sav','rb'))

# create a function
def diabetes_prediction(input_data):
   


    input_data_as_numpyarray=np.asarray(input_data)

     
    input_data_reshaped=input_data_as_numpyarray.reshape(1,-1)
    prediction=loaded_model.predict(input_data_reshaped)  # using loaded model instead of classifier
    print(prediction)

    if(prediction[0]==1): #prediction[0] because its a list having 0 or 1
        return "Diabetic"
    else:
        return "Non Diabetic"
    
    

def main():
    
    
    # giving a title 
    st.title('Diabetes Prediction Web App')
    
    # getting the input data from user
    Pregnancies=st.text_input("Number of pregnencies")
    Glucose=st.text_input("glucose level")
    BloodPressure=st.text_input("BP level")
    SkinThickness=st.text_input("Skin Thickness")
    Insulin=st.text_input("Insulin level")
    BMI=st.text_input("BMI")
    DiabetesPedigreeFunction=st.text_input("DiabetesPedigreeFunction value ")
    Age=st.text_input("Enter Age")
    
    
    # code for prediction
    diagnosis=''
    
    # creating a button for prediction
    if st.button("Diabetes Test Result"):
        diagnosis=diabetes_prediction([Pregnancies,Glucose,BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age])
        
    st.success(diagnosis)
    
    
if __name__ == '__main__':
    main()

    