import pandas as pd

from sklearn.linear_model import LinearRegression


# Load data
data = pd.read_csv("data.csv")


# Create performance score
data["Performance"] = (
    data["Points"]
    + data["Assists"]
    + data["Rebounds"]
    + data["Recent_Form"]
)


# Inputs for AI
X = data[
    [
        "Points",
        "Assists",
        "Rebounds",
        "Minutes",
        "Recent_Form"
    ]
]


# Output we want AI to predict
y = data["Performance"]


# Create AI model
model = LinearRegression()


# Train model
model.fit(X, y)


# Test prediction

prediction_data = pd.DataFrame(
    [
        [30,6,5,34,9]
    ],
    columns=[
        "Points",
        "Assists",
        "Rebounds",
        "Minutes",
        "Recent_Form"
    ]
)


prediction = model.predict(prediction_data)


print("Predicted Performance:")
print(prediction)