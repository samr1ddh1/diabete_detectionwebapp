#import the libraries

import pandas as pd 

from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForrestClassifier
from PIL import Image
import streamlit as st

#create a title and sub-title
st.write("""
#Diabetes Detection 
Detect if someone had diabetes using machine learning and python !
""")

#open and display an image
image = Image.open('S:\P\PYTHON\machine learning\diabete_detectionwebapp\webapp.png')
st.image(image, caption='ML', use_column_width=True)

#get the data
df = pd.read_csv('S:/P/PYTHON/machine learning/diabete_detectionwebapp/diabetes.csv')

#set a subheader
st.subheader('Data Information: ')

#show the data as a table
st.dataframe(df)

#show statistics on the data
st.write(df.describe())

#show the data as a chart
chart = st.bar_chart(df)


#split the data into independent 'X' and dependant 'Y' variables
X = df.iloc[:, 0:8].values
Y = df.iloc[:,-1].values

#split thr dat set into 75% training set and 25% into test set
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.25, random_state=0)

#get the feature input from the user
def get_user_input():
    pregnancies = st.sidebar.slider('pregnancies',0,17,3)
    glucose = st.sidebar.slider('glucose',0,199,117)
    blood_pressure = st.sidebar.slider('blood_pressure',0,122,72)
    skin_thickness = st.sidebar.slider('skin_thickness',0,99,23)
    insulin = st.sidebar.slider('insulin',0.0,846.0,0)
    BMI = st.sidebar.slider('BMI',0.0,67.1,32.0)
    DPF = st.sidebar.slider('DPF',0.078,2.42,0.3725)
    age = st.sidebar.slider('age',21,31,29)

#Store a dictionary into a variable
user_data = {'pregnancies': pregnancies,
            'glucose' : glucose,
            'blood_pressure': blood_pressure,
            'skin_thickness': skin_thickness,
            'insulin': insulin,
            'BMI': BMI,
            'DPF': DPF,
            'age': age
            }

#TRANSFORM THE DATA INTO A DATA FRAME
features = pd.Dataframe(user_data, index = [0])
return features

#Store the user input into a variable
user_input = get_user_input()

#SET A SUBHEADER AND DISPLAY THE USER INOUT
st.subheader('User Input:')
st.write(user_input)

#CREATE AND TRAIN THE MODEL
RandomForrestClassifier = RandomForrestClassifier()
RandomForrestClassifier.fit(X_train, Y_train)

#Show the model metrics
st.subheader('Model Test Accuracy Score:')
st.write(str(accuracy_score(Y_test, RandomForrestClassifier.predict(X_test)) * 100) + '%')

#Store the model predictions in a variable
prediction = RandomForrestClassifier.predict(user_input)

#Set a subheader and display the classification
st.subheader('Classification: ')
st.write(prediction)
