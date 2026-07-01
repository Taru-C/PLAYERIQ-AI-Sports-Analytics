import streamlit as st
import pandas as pd
from sklearn.linear_model import LinearRegression
import plotly.express as px


# Page Settings

st.set_page_config(
    page_title="PlayerIQ Analytics",
    page_icon="📊",
    layout="wide"
)



# Custom UI Styling

st.markdown("""
<style>

.main {
    background-color:#0e1117;
}


h1 {
    color:white;
    font-size:42px;
}


h2,h3 {
    color:#d9e6ff;
}


.stMetric {

background-color:#161b22;
padding:20px;
border-radius:12px;

}


[data-testid="stSidebar"] {

background-color:#090d14;

}


div[data-testid="stDataFrame"] {

background-color:#161b22;

border-radius:12px;

}

</style>

""", unsafe_allow_html=True)




# Sidebar


st.sidebar.title("PLAYERIQ")


st.sidebar.caption(
    "AI Sports Analytics Platform"
)


st.sidebar.divider()


st.sidebar.subheader("Dataset")


uploaded_file = st.sidebar.file_uploader(
    "Upload Sports Dataset",
    type=["csv"]
)



st.sidebar.divider()


st.sidebar.subheader("Model Status")


st.sidebar.success(
    "Machine Learning Model Active"
)


st.sidebar.caption(
    "Performance prediction engine ready"
)





# Header


st.title("PLAYERIQ")


st.caption(
    "AI-powered sports performance analytics and prediction platform"
)


st.write(
    "Analyze player statistics, generate predictions, and discover performance insights."
)





# Load Data


if uploaded_file:

    data = pd.read_csv(uploaded_file)

else:

    data = pd.read_csv("data.csv")





# Create Performance Score


data["Performance"] = (

    data["Points"]

    + data["Assists"]

    + data["Rebounds"]

    + data["Recent_Form"]

)





# Dashboard Metrics


col1,col2,col3,col4 = st.columns(4)



with col1:

    st.metric(
        "TOTAL PLAYERS",
        len(data)
    )



with col2:

    st.metric(
        "AVERAGE PERFORMANCE",
        round(data["Performance"].mean(),2)
    )



with col3:


    top_player = data.loc[
        data["Performance"].idxmax()
    ]["Player"]



    st.metric(
        "TOP PERFORMER",
        top_player
    )



with col4:


    st.metric(
        "MODEL STATUS",
        "ACTIVE"
    )






# Player Statistics


st.subheader("Player Statistics Overview")


st.dataframe(
    data,
    use_container_width=True
)






# Machine Learning Model


X = data[
[
"Points",
"Assists",
"Rebounds",
"Minutes",
"Recent_Form"
]
]


y = data["Performance"]



model = LinearRegression()


model.fit(X,y)






# Prediction


st.subheader("AI Performance Prediction")



player = st.selectbox(

    "Select Player",

    data["Player"]

)




player_data = data[
data["Player"] == player
]




prediction = model.predict(

player_data[
[
"Points",
"Assists",
"Rebounds",
"Minutes",
"Recent_Form"
]
]

)





st.metric(

    "Predicted Performance Score",

    f"{prediction[0]:.2f}/100"

)







# AI Insights


st.subheader("AI Performance Insights")



stats = player_data.iloc[0]



if stats["Points"] >= 28:

    st.write(
        "• Strong scoring contribution"
    )


if stats["Assists"] >= 7:

    st.write(
        "• High playmaking ability"
    )


if stats["Rebounds"] >= 7:

    st.write(
        "• Strong rebounding impact"
    )


if stats["Recent_Form"] >= 8:

    st.write(
        "• Excellent recent form"
    )


if stats["Minutes"] >= 34:

    st.write(
        "• High game involvement"
    )








# Player Comparison


st.subheader("Player Comparison")



player1 = st.selectbox(

    "Select First Player",

    data["Player"],

    key="player1"

)



player2 = st.selectbox(

    "Select Second Player",

    data["Player"],

    key="player2"

)




comparison = data[
data["Player"].isin(
[player1,player2]
)
]




fig = px.bar(

    comparison,

    x="Player",

    y="Performance",

    title="Performance Comparison"

)




st.plotly_chart(

    fig,

    use_container_width=True

)






# Visualization


st.subheader("Performance Visualization")



fig2 = px.line(

    data,

    x="Player",

    y="Points",

    markers=True,

    title="Points Trend"

)



st.plotly_chart(

    fig2,

    use_container_width=True

)







# Rankings


st.subheader("AI Player Rankings")



ranking = data.copy()



ranking["Predicted_Score"] = model.predict(

ranking[

[
"Points",
"Assists",
"Rebounds",
"Minutes",
"Recent_Form"
]

]

)





ranking = ranking.sort_values(

    by="Predicted_Score",

    ascending=False

)





st.dataframe(

ranking[

[
"Player",
"Predicted_Score"
]

],

use_container_width=True

)