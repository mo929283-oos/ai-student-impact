import streamlit as st
import pandas as pd
import numpy as np
import joblib

# ==========================
# Page Configuration
# ==========================

st.set_page_config(
    page_title="AI Student Impact Predictor",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==========================
# Load Models
# ==========================

@st.cache_resource
def load_models():

    gpa_model = joblib.load("gpa_predictor_model.joblib")

    burnout_model = joblib.load("burnout_classifier_model.joblib")

    encoders = joblib.load("label_encoders.joblib")

    burnout_encoder = joblib.load("burnout_encoder.joblib")

    feature_columns = joblib.load("feature_columns.joblib")

    return (
        gpa_model,
        burnout_model,
        encoders,
        burnout_encoder,
        feature_columns
    )

(
    gpa_model,
    burnout_model,
    encoders,
    burnout_encoder,
    feature_columns
) = load_models()
st.markdown("""
<style>

.main{
    background:#F4F7FA;
}

.block-container{
    padding-top:2rem;
    padding-bottom:2rem;
}

h1{
    color:#0E7490;
    font-weight:700;
}

h2{
    color:#155E75;
}

.stButton>button{

    width:100%;

    background:#0891B2;

    color:white;

    border-radius:12px;

    height:55px;

    font-size:18px;

    font-weight:bold;

    border:none;

}

.stButton>button:hover{

    background:#0E7490;

    color:white;

}

.result-box{

    background:white;

    padding:20px;

    border-radius:15px;

    box-shadow:0px 0px 15px rgba(0,0,0,0.1);

}

.metric-card{

    background:white;

    padding:15px;

    border-radius:12px;

    box-shadow:0px 0px 8px rgba(0,0,0,0.1);

}

</style>
""", unsafe_allow_html=True)
st.title("🎓 AI Student Impact Prediction System")

st.markdown("""

Predict **Student GPA**
and
**Burnout Risk**
using Machine Learning.

""")
st.sidebar.image(
    "https://cdn-icons-png.flaticon.com/512/3135/3135755.png",
    width=120
)

st.sidebar.title("Navigation")

st.sidebar.success(
"""
Machine Learning Models

✅ GPA Prediction

✅ Burnout Classification
"""
)

st.sidebar.info("""
Developer:

Mostafa Ahmed
""")
st.header("📋 Student Information")

col1,col2=st.columns(2)
# ==========================
# Student Information
# ==========================

with col1:

    major = st.selectbox(
        "Major Category",
        encoders["Major_Category"].classes_
    )

    year = st.selectbox(
        "Year of Study",
        encoders["Year_of_Study"].classes_
    )

    pre_gpa = st.number_input(
        "Pre Semester GPA",
        min_value=0.0,
        max_value=4.0,
        value=3.0,
        step=0.01
    )

    weekly_ai = st.slider(
        "Weekly GenAI Hours",
        0,
        60,
        10
    )

    primary_use = st.selectbox(
        "Primary Use Case",
        encoders["Primary_Use_Case"].classes_
    )

    prompt_skill = st.selectbox(
        "Prompt Engineering Skill",
        encoders["Prompt_Engineering_Skill"].classes_
    )

    tool_diversity = st.slider(
        "Tool Diversity",
        1,
        20,
        5
    )


with col2:

    paid = st.selectbox(
        "Paid Subscription",
        ["No","Yes"]
    )

    study_hours = st.slider(
        "Traditional Study Hours",
        0,
        80,
        20
    )

    dependency = st.slider(
        "Perceived AI Dependency",
        1,
        10,
        5
    )

    policy = st.selectbox(
        "Institutional Policy",
        encoders["Institutional_Policy"].classes_
    )

    anxiety = st.slider(
        "Anxiety During Exams",
        1,
        10,
        5
    )

    retention = st.slider(
        "Skill Retention Score",
        1,
        100,
        70
    )

st.divider()
# ==========================
# Encode Inputs
# ==========================

major_encoded = encoders["Major_Category"].transform([major])[0]

year_encoded = encoders["Year_of_Study"].transform([year])[0]

primary_encoded = encoders["Primary_Use_Case"].transform([primary_use])[0]

prompt_encoded = encoders["Prompt_Engineering_Skill"].transform([prompt_skill])[0]

policy_encoded = encoders["Institutional_Policy"].transform([policy])[0]

paid_encoded = 1 if paid == "Yes" else 0
input_df = pd.DataFrame({

    "Major_Category":[major_encoded],

    "Year_of_Study":[year_encoded],

    "Pre_Semester_GPA":[pre_gpa],

    "Weekly_GenAI_Hours":[weekly_ai],

    "Primary_Use_Case":[primary_encoded],

    "Prompt_Engineering_Skill":[prompt_encoded],

    "Tool_Diversity":[tool_diversity],

    "Paid_Subscription":[paid_encoded],

    "Traditional_Study_Hours":[study_hours],

    "Perceived_AI_Dependency":[dependency],

    "Institutional_Policy":[policy_encoded],

    "Anxiety_Level_During_Exams":[anxiety],

    "Skill_Retention_Score":[retention]

})

input_df = input_df[feature_columns]
predict = st.button(
    "Predict Student Performance 🚀",
    use_container_width=True
)
# ==========================
# Prediction
# ==========================

if predict:

    with st.spinner("Running Machine Learning Models..."):

        # GPA Prediction
        predicted_gpa = gpa_model.predict(input_df)[0]

        # Burnout Prediction
        burnout_prediction = burnout_model.predict(input_df)[0]

        burnout_probability = burnout_model.predict_proba(input_df)[0]

        burnout_label = burnout_encoder.inverse_transform(
            [burnout_prediction]
        )[0]

        confidence = np.max(burnout_probability) * 100

    st.success("Prediction Completed Successfully ✅")

    st.divider()

    col1, col2 = st.columns(2)

    # -------------------------
    # GPA RESULT
    # -------------------------

    with col1:

        st.markdown(
            """
            <div class="metric-card">
            <h2>📚 Predicted GPA</h2>
            </div>
            """,
            unsafe_allow_html=True
        )

        st.metric(
            label="Expected GPA",
            value=f"{predicted_gpa:.2f}"
        )

        st.progress(min(predicted_gpa / 4.0, 1.0))

    # -------------------------
    # Burnout RESULT
    # -------------------------

    with col2:

        st.markdown(
            """
            <div class="metric-card">
            <h2>🧠 Burnout Risk</h2>
            </div>
            """,
            unsafe_allow_html=True
        )

        if burnout_label.lower() == "low":

            st.success(f"Risk Level : {burnout_label}")

        elif burnout_label.lower() == "medium":

            st.warning(f"Risk Level : {burnout_label}")

        else:

            st.error(f"Risk Level : {burnout_label}")

        st.metric(
            "Model Confidence",
            f"{confidence:.1f}%"
        )

        st.progress(confidence / 100)

    st.divider()

    # ==========================
    # Personalized Recommendations
    # ==========================

    st.subheader("💡 Recommendations")

    recommendations = []

    if predicted_gpa < 2.5:
        recommendations.append(
            "📚 Increase your traditional study hours."
        )

    if weekly_ai > 30:
        recommendations.append(
            "🤖 Try reducing excessive dependence on Generative AI."
        )

    if dependency >= 8:
        recommendations.append(
            "⚠ High AI dependency detected. Balance AI usage with self-learning."
        )

    if anxiety >= 8:
        recommendations.append(
            "🧘 Consider stress-management techniques before exams."
        )

    if retention < 50:
        recommendations.append(
            "📝 Focus on practicing concepts instead of relying only on AI."
        )

    if burnout_label.lower() == "high":
        recommendations.append(
            "🚨 High burnout risk detected. Consider taking regular breaks."
        )

    if burnout_label.lower() == "medium":
        recommendations.append(
            "😊 Maintain a healthy balance between studying and resting."
        )

    if predicted_gpa >= 3.5 and burnout_label.lower() == "low":
        recommendations.append(
            "🎉 Excellent performance! Keep maintaining your study habits."
        )

    if len(recommendations) == 0:
        recommendations.append(
            "✅ Keep up your current learning strategy."
        )

    for tip in recommendations:
        st.info(tip)

    st.divider()

    # ==========================
    # Input Summary
    # ==========================

    st.subheader("📋 Input Summary")

    st.dataframe(
        input_df,
        use_container_width=True
    )
    st.markdown(
    """
    <hr>
    <center>
        <h5>
        AI Student Impact Prediction System
        </h5>

        Developed using
        ❤️ Streamlit + Scikit-learn
    </center>
    """,
    unsafe_allow_html=True
)

