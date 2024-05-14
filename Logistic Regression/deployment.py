import streamlit as st
import pandas as pd
from pickle import load

st.title('Model Deployment: Logistic Regression')
st.sidebar.header('User Input Parameters')

def user_input_features():
    PCLASS = st.sidebar.selectbox('Passenger Class',('1','2','3'))
    CLMSEX = st.sidebar.selectbox('Gender',('Male','Female'))
    CLMAGE = st.sidebar.number_input("Insert the Age", min_value=0, max_value=100, value=30)
    SIBSP = st.sidebar.slider('Number of Siblings/Spouses Aboard', min_value=0, max_value=8, value=1)
    PARCH = st.sidebar.number_input('Number of Parents/Children Aboard', min_value=0, value=0)
    FARE = st.sidebar.number_input('Fare', min_value=0, value=30)
    EMBARKED = st.sidebar.selectbox('Embarked', ['C', 'Q', 'S'])
    data = {
        'Pclass' : PCLASS,
        'Sex' : CLMSEX,
        'Age' : CLMAGE,
        'SibSp' : SIBSP,
        'Parch' : PARCH,
        'Fare' : FARE,
        'Embarked' : EMBARKED
    }
    features = pd.DataFrame(data,index = [0])
    features['Sex'] = features['Sex'].map({'Male': 1, 'Female':0})
    features['Embarked'] = features['Embarked'].map({'C': 0, 'Q':1, 'S':2})
    return features

def main():
    df = user_input_features()
    st.subheader('User Input Parameters')
    st.write(df)

    # loading model
    model = load(open('Logistic_Model.sav', 'rb'))

    prediction = model.predict(df)
    if prediction == 1:
        output = "Survived"
    else:
        output = "Did not Survive"
    prediction_probability = model.predict_proba(df)

    st.subheader('Predicted Result')
    st.write(output)

    st.subheader('Prediction Probability')
    st.write(prediction_probability)

# Run the app
if __name__ == "__main__":
    main()