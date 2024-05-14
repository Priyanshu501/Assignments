''' Deployment of Logistic Regression Model '''
from pickle import load
import streamlit as st
import pandas as pd

st.title('Model Deployment: Logistic Regression')
st.sidebar.header('User Input Parameters')

def user_input_features():
    ''' Taking User Inputs '''
    pclass = st.sidebar.selectbox('Passenger Class',('1','2','3'))
    sex = st.sidebar.selectbox('Gender',('Male','Female'))
    age = st.sidebar.number_input("Insert the Age", min_value=0, max_value=100, value=30)
    sibsp = st.sidebar.slider('Number of Siblings/Spouses Aboard', min_value=0, value=1)
    parch = st.sidebar.number_input('Number of Parents/Children Aboard', min_value=0, value=0)
    fare = st.sidebar.number_input('Fare', min_value=0, value=30)
    embarked = st.sidebar.selectbox('Embarked', ['C', 'Q', 'S'])
    data = {
        'Pclass' : pclass,
        'Sex' : sex,
        'Age' : age,
        'SibSp' : sibsp,
        'Parch' : parch,
        'Fare' : fare,
        'Embarked' : embarked
    }
    features = pd.DataFrame(data,index = [0])
    features['Sex'] = features['Sex'].map({'Male': 1, 'Female':0})
    features['Embarked'] = features['Embarked'].map({'C': 0, 'Q':1, 'S':2})
    return features

def main():
    ''' Streamlit App '''
    df = user_input_features()
    st.subheader('User Input Parameters')
    st.write(df)

    # loading model
    model = load(open('Logistic_Model.sav', 'rb'))

    # Interpreting Prediction
    prediction = model.predict(df)
    if prediction == 1:
        output = "Survived"
    else:
        output = "Did not Survive"

    # Prediction Probability
    prediction_probability = model.predict_proba(df)

    # Displaying Prediction and Prediction Probability
    st.subheader('Predicted Result')
    st.write(output)

    st.subheader('Prediction Probability')
    st.write(prediction_probability)

# Run the app
if __name__ == "__main__":
    main()
