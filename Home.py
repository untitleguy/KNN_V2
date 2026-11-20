from sklearn.neighbors import KNeighborsClassifier
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

st.header("Got heart disease ?")
st.image("./img/heartbreak-heart.gif")
html_7 = """<h3>2 possible way, dead or alive.</h3>"""
st.markdown(html_7, unsafe_allow_html=True)
col1, col2 = st.columns(2)


with col1:
   st.header("Alive")
   st.image("./img/happy.png")

with col2:
   st.header("Dead")
   st.image("./img/grave.jpg")

html_7 = """
<div style="background-color:#EC7063;padding:15px;border-radius:15px 15px 15px 15px;border-style:'solid';border-color:black">
<center><h5>สถิติผู้ที่เป็นโรคหัวใจ</h5></center>
</div>
"""
st.markdown(html_7, unsafe_allow_html=True)
st.markdown("")
dt = pd.read_csv("./data/Heart3.csv")
st.write(dt.head(30))

dt1 = dt['Age'].sum()
dt2 = dt['Sex'].sum()
dt3 = dt['ChestPainType'].sum()
dt4 = dt['RestingBP'].sum()
dt5 = dt['Cholesterol'].sum()
dt6 = dt['FastingBS'].sum()
dt7 = dt['RestingECG'].sum()
dt8 = dt['MaxHR'].sum()
dt9 = dt['ExerciseAngina'].sum()
dt10 = dt['Oldpeak'].sum()
dt11 = dt['ST_Slope'].sum()
dt12 = dt['HeartDisease'].sum()

dx = [dt1, dt2, dt3, dt4, dt5, dt6, dt7, dt8, dt9, dt10, dt11, dt12]
dx2 = pd.DataFrame(dx, index=["Age", "Sex", "ChestPainType", "RestingBP", "Cholesterol", "FastingBS", "RestingECG", "MaxHR", "ExerciseAngina", "Oldpeak", "ST_Slope", "HeartDisease"])

if st.button("แสดงการจินตทัศน์ข้อมูล"):
   #st.write(dt.head(10))
   st.bar_chart(dx2)
   st.button("ไม่แสดงข้อมูล")
else:
    st.write("ไม่แสดงข้อมูล")

html_8 = """
<div style="background-color:#6BD5DA;padding:15px;border-radius:15px 15px 15px 15px;border-style:'solid';border-color:black">
<center><h5>ทำนายข้อมูล</h5></center>
</div>
"""
st.markdown(html_8, unsafe_allow_html=True)
st.markdown("")

st.caption("Categorical fields use the numeric codes in the training dataset.")
with st.form("heart_disease_prediction"):
    age = st.number_input(
        "Age",
        min_value=int(dt["Age"].min()),
        max_value=int(dt["Age"].max()),
        value=int(dt["Age"].median()),
    )
    sex = st.selectbox("Sex", sorted(dt["Sex"].unique()))
    chest_pain_type = st.selectbox(
        "ChestPainType", sorted(dt["ChestPainType"].unique())
    )
    resting_bp = st.number_input(
        "RestingBP",
        min_value=int(dt["RestingBP"].min()),
        max_value=int(dt["RestingBP"].max()),
        value=int(dt["RestingBP"].median()),
    )
    cholesterol = st.number_input(
        "Cholesterol",
        min_value=int(dt["Cholesterol"].min()),
        max_value=int(dt["Cholesterol"].max()),
        value=int(dt["Cholesterol"].median()),
    )
    fasting_bs = st.selectbox("FastingBS", sorted(dt["FastingBS"].unique()))
    resting_ecg = st.selectbox("RestingECG", sorted(dt["RestingECG"].unique()))
    max_hr = st.number_input(
        "MaxHR",
        min_value=int(dt["MaxHR"].min()),
        max_value=int(dt["MaxHR"].max()),
        value=int(dt["MaxHR"].median()),
    )
    exercise_angina = st.selectbox(
        "ExerciseAngina", sorted(dt["ExerciseAngina"].unique())
    )
    oldpeak = st.number_input(
        "Oldpeak",
        min_value=float(dt["Oldpeak"].min()),
        max_value=float(dt["Oldpeak"].max()),
        value=float(dt["Oldpeak"].median()),
        step=0.1,
        format="%.1f",
    )
    st_slope = st.selectbox("ST_Slope", sorted(dt["ST_Slope"].unique()))
    submitted = st.form_submit_button("ทำนายผล")

if submitted:
    features = dt.drop(columns="HeartDisease")
    target = dt["HeartDisease"]
    categorical_features = [
        "Sex",
        "ChestPainType",
        "FastingBS",
        "RestingECG",
        "ExerciseAngina",
        "ST_Slope",
    ]
    numeric_features = [
        column for column in features.columns if column not in categorical_features
    ]
    preprocessor = ColumnTransformer(
        [
            ("numeric", StandardScaler(), numeric_features),
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore"),
                categorical_features,
            ),
        ]
    )
    model = make_pipeline(
        preprocessor, KNeighborsClassifier(n_neighbors=31, weights="distance")
    )
    model.fit(features, target)

    input_data = pd.DataFrame(
        [[
            age,
            sex,
            chest_pain_type,
            resting_bp,
            cholesterol,
            fasting_bs,
            resting_ecg,
            max_hr,
            exercise_angina,
            oldpeak,
            st_slope,
        ]],
        columns=features.columns,
    )
    prediction = model.predict(input_data)[0]

    if prediction == 1:
        st.error("Heart disease predicted.")
        st.image("./img/grave.jpg")
    else:
        st.success("No heart disease predicted.")
        st.image("./img/happy.png")
