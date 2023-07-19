# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy as np
import pickle 

loaded_model=pickle.load(open('C:/Users/HP/Videos/python/DIABETES_PREDICTION/trained_model.sav','rb'))

input_data=(5,116,74,0,0,25.6,0.201,30)


input_data_as_numpyarray=np.asarray(input_data)

 
input_data_reshaped=input_data_as_numpyarray.reshape(1,-1)
prediction=loaded_model.predict(input_data_reshaped)  # using loaded model instead of classifier
print(prediction)

if(prediction[0]==1): #prediction[0] because its a list having 0 or 1
    print("diabetic")
else:
    print("non diabetic")
