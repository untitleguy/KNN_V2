from sklearn.neighbors import KNeighborsClassifier
import streamlit as st
import pandas as pd
import numpy as np
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
dx2 = pd.DataFrame(dx, index=["d1", "d2", "d3", "d4", "d5", "d6", "d7", "d8", "d9", "d10", "d11", "d12"])

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


pt_len = st.slider("กรุณาเลือกข้อมูล petal.length")
pt_wd = st.slider("กรุณาเลือกข้อมูล petal.width")

sp_len = st.number_input("กรุณาเลือกข้อมูล sepal.length")
sp_wd = st.number_input("กรุณาเลือกข้อมูล sepal.width")

if st.button("ทำนายผล"):
   #st.write("ทำนาย")
   #dt = pd.read_csv("./data/iris.csv") 

   X = dt.drop('variety', axis=1)
   y = dt.variety   
   Knn_model = KNeighborsClassifier(n_neighbors=3)
   Knn_model.fit(X, y)  

   x_input = np.array([[pt_len, pt_wd, sp_len, sp_wd]]) #ข้อมูลใหม่สำหรับทำนาย

   st.write(Knn_model.predict(x_input))   
   out=Knn_model.predict(x_input)

   if out[0] == '0':
    st.image("./img/happy.png")
   else:
    st.image("./img/grave.jpg")
else:
    st.write("ไม่ทำนาย")
