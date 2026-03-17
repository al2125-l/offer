import streamlit as st
import pickle
import pandas as pd


st.set_page_config(
    page_title="Offer Recommendation System",
    layout="centered"
)


st.markdown("## 🎯 Personalized Offer Recommendation System")
st.markdown(
    "This system predicts whether a customer should receive a personalized offer "
    "based on purchasing behavior."
)
st.markdown("---")


model = pickle.load(open("model.pkl", "rb"))


avg_order_value = st.number_input("Average Order Value", min_value=0.0)
total_orders = st.number_input("Total Orders", min_value=0)
avg_review_rating = st.slider("Average Review Rating", 1.0, 5.0)

gender = st.selectbox("Gender", ["Male", "Female"])
subscription = st.selectbox("Subscription Status", ["Yes", "No"])


gender_encoded = 1 if gender == "Male" else 0
subscription_encoded = 1 if subscription == "Yes" else 0


if st.button("Predict"):

    input_dict = {
        "avg_order_value": avg_order_value,
        "total_orders": total_orders,
        "avg_review_rating": avg_review_rating,
        "Gender": gender_encoded,
        "Subscription Status": subscription_encoded
    }

    input_df = pd.DataFrame([input_dict])

    
    input_df = input_df[model.feature_names_in_]

    prediction = model.predict(input_df)
    probability = model.predict_proba(input_df)[0][1]

    st.markdown("---")
    st.subheader("Prediction Result")

    if prediction[0] == 1:
        st.success("✅ Offer Recommended")
    else:
        st.error("❌ No Offer Recommended")

    st.metric("Prediction Confidence", f"{probability*100:.2f}%")
