# ==========================================================
# AI STUDENT IMPACT PREDICTION SYSTEM
# Developed by Mostafa Ahmed
# Streamlit + Scikit-Learn + Plotly
# ==========================================================

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime

# ==========================================================
# PAGE CONFIG
# ==========================================================

st.set_page_config(
    page_title="AI Student Impact Prediction",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==========================================================
# CUSTOM CSS
# ==========================================================

st.markdown("""
<style>

@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600;700&display=swap');

html, body, [class*="css"]{
    font-family: 'Poppins', sans-serif;
}

.main{
    background:#F4F7FC;
}

.block-container{
    padding-top:2rem;
    padding-bottom:2rem;
    padding-left:2rem;
    padding-right:2rem;
}

h1{
    color:#0F4C81;
    font-weight:700;
}

h2, h3{
    color:#0F4C81;
}

.metric-card{
    background:white;
    border-radius:18px;
    padding:20px;
    box-shadow:0px 10px 30px rgba(0,0,0,.08);
}

.card{
    background:white;
    border-radius:20px;
    padding:25px;
    box-shadow:0px 10px 25px rgba(0,0,0,.08);
    margin-bottom:20px;
}

.sidebar .sidebar-content{
    background:#0F4C81;
}

.stButton>button{
    width:100%;
    height:58px;
    font-size:18px;
    font-weight:bold;
    border:none;
    border-radius:15px;
    background:#0F4C81;
    color:white;
    transition:.3s;
}

.stButton>button:hover{
    background:#1976D2;
    color:white;
}

footer, #MainMenu{
    visibility:hidden;
}

</style>
""", unsafe_allow_html=True)

# ==========================================================
# LOAD MODELS
# ==========================================================

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

try:
    (
        gpa_model,
        burnout_model,
        encoders,
        burnout_encoder,
        feature_columns
    ) = load_models()
except Exception as e:
    st.error(f"خطأ في تحميل ملفات الموديلات: {e}")
    st.stop()

# ==========================================================
# SIDEBAR
# ==========================================================

with st.sidebar:
    st.title("🎓 AI Student")
    st.markdown("---")
    st.success("Machine Learning Dashboard")
    st.markdown("""
### Models
✅ GPA Prediction
✅ Burnout Prediction

---
### Algorithms
🌲 Random Forest Regressor
🌲 Random Forest Classifier

---
### Dataset
AI Student Impact Dataset

---
### Version
v1.0
""")
    st.markdown("---")
    st.info(f"Today\n\n{datetime.now().strftime('%d-%m-%Y')}")

# ==========================================================
# HEADER
# ==========================================================

st.title("🎓 AI Student Impact Prediction System")
st.caption("Predict Student Academic Performance and Burnout Risk using Machine Learning")
st.markdown("---")

# ==========================================================
# TOP METRICS
# ==========================================================

m1, m2, m3, m4 = st.columns(4)
with m1:
    st.metric("Regression Model", "Ready ✅")
with m2:
    st.metric("Classification", "Ready ✅")
with m3:
    st.metric("Features", len(feature_columns) if feature_columns is not None else "13")
with m4:
    st.metric("Status", "Online 🟢")

st.markdown("---")

# ==========================================================
# INPUT SECTION
# ==========================================================

st.header("📋 Student Information")
left, right = st.columns(2)

with left:
    st.markdown("### 🎓 Academic Information")
    major = st.selectbox("Major Category", encoders["Major_Category"].classes_)
    year = st.selectbox("Year of Study", encoders["Year_of_Study"].classes_)
    pre_gpa = st.number_input("Pre Semester GPA", min_value=0.00, max_value=4.00, value=3.00, step=0.01, help="Previous Semester GPA")
    weekly_ai = st.slider("Weekly GenAI Hours", 0, 60, 10)
    primary_use = st.selectbox("Primary Use Case", encoders["Primary_Use_Case"].classes_)
    prompt_skill = st.selectbox("Prompt Engineering Skill", encoders["Prompt_Engineering_Skill"].classes_)
    tool_diversity = st.slider("Tool Diversity", 1, 20, 5)

with right:
    st.markdown("### 🤖 AI Usage & Wellbeing")
    paid_subscription = st.radio("Paid Subscription", ["No", "Yes"], horizontal=True)
    study_hours = st.slider("Traditional Study Hours", 0, 80, 20)
    dependency = st.slider("Perceived AI Dependency", 1, 10, 5)
    policy = st.selectbox("Institutional Policy", encoders["Institutional_Policy"].classes_)
    anxiety = st.slider("Anxiety During Exams", 1, 10, 5)
    retention = st.slider("Skill Retention Score", 1, 100, 70)

st.markdown("---")

# ==========================================================
# ENCODING
# ==========================================================

major_encoded = encoders["Major_Category"].transform([major])[0]
year_encoded = encoders["Year_of_Study"].transform([year])[0]
primary_encoded = encoders["Primary_Use_Case"].transform([primary_use])[0]
prompt_encoded = encoders["Prompt_Engineering_Skill"].transform([prompt_skill])[0]
policy_encoded = encoders["Institutional_Policy"].transform([policy])[0]
paid_encoded = 1 if paid_subscription == "Yes" else 0

# ==========================================================
# INPUT DATAFRAME
# ==========================================================

input_data = {
    "Major_Category": [major_encoded],
    "Year_of_Study": [year_encoded],
    "Pre_Semester_GPA": [pre_gpa],
    "Weekly_GenAI_Hours": [weekly_ai],
    "Primary_Use_Case": [primary_encoded],
    "Prompt_Engineering_Skill": [prompt_encoded],
    "Tool_Diversity": [tool_diversity],
    "Paid_Subscription": [paid_encoded],
    "Traditional_Study_Hours": [study_hours],
    "Perceived_AI_Dependency": [dependency],
    "Institutional_Policy": [policy_encoded],
    "Anxiety_Level_During_Exams": [anxiety],
    "Skill_Retention_Score": [retention]
}

input_df = pd.DataFrame(input_data)
# ضمان تطابق ترتيب الأعمدة مع الموديل
input_df = input_df[feature_columns]
input_df = input_df.astype(float)

# ==========================================================
# INPUT PREVIEW
# ==========================================================

with st.expander("📋 Preview Encoded Input Data"):
    st.dataframe(input_df, use_container_width=True)

# ==========================================================
# PREDICT BUTTON
# ==========================================================

predict = st.button("🚀 Predict Student Performance", use_container_width=True)

# ==========================================================
# PREDICTION & RESULTS
# ==========================================================

if predict:
    try:
        with st.spinner("Running AI Models..."):
            predicted_gpa = float(gpa_model.predict(input_df)[0])
            burnout_prediction = burnout_model.predict(input_df)[0]

            # فحص إمكانية استخدام predict_proba
            if hasattr(burnout_model, "predict_proba"):
                burnout_probabilities = burnout_model.predict_proba(input_df)[0]
                confidence = float(np.max(burnout_probabilities) * 100)
            else:
                burnout_probabilities = None
                confidence = 100.0

            burnout_label = str(burnout_encoder.inverse_transform([burnout_prediction])[0])

        st.success("Prediction Completed Successfully ✅")
    except Exception as e:
        st.error(f"حدث خطأ أثناء التنبؤ: {e}")
        st.stop()

    st.markdown("---")

    # Gauges
    left_col, right_col = st.columns(2)

    with left_col:
        gpa_fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=predicted_gpa,
            number={"suffix": "/4.0"},
            title={"text": "Predicted GPA"},
            gauge={
                "axis": {"range": [0, 4]},
                "bar": {"color": "royalblue"},
                "steps": [
                    {"range": [0, 2], "color": "#ffb3b3"},
                    {"range": [2, 3], "color": "#ffe680"},
                    {"range": [3, 4], "color": "#b3ffcc"}
                ]
            }
        ))
        gpa_fig.update_layout(height=380)
        st.plotly_chart(gpa_fig, use_container_width=True)

    with right_col:
        if burnout_probabilities is not None:
            burnout_fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=confidence,
                number={"suffix": "%"},
                title={"text": "Prediction Confidence"},
                gauge={
                    "axis": {"range": [0, 100]},
                    "bar": {"color": "crimson"},
                    "steps": [
                        {"range": [0, 50], "color": "#b3ffcc"},
                        {"range": [50, 75], "color": "#ffe680"},
                        {"range": [75, 100], "color": "#ffb3b3"}
                    ]
                }
            ))
            burnout_fig.update_layout(height=380)
            st.plotly_chart(burnout_fig, use_container_width=True)
        else:
            st.info("Confidence Score is not available for this model type.")

    st.markdown("---")

    # Metrics Summary
    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("Predicted GPA", f"{predicted_gpa:.2f}")
    with c2:
        st.metric("Burnout Risk", burnout_label)
    with c3:
        st.metric("Confidence", f"{confidence:.1f}%" if burnout_probabilities is not None else "N/A")

    # Bar Chart for Probabilities (if available)
    if burnout_probabilities is not None:
        probability_df = pd.DataFrame({
            "Risk": burnout_encoder.classes_,
            "Probability": burnout_probabilities
        })
        fig = px.bar(
            probability_df,
            x="Risk",
            y="Probability",
            color="Risk",
            text_auto=".2f",
            title="Burnout Probability Distribution"
        )
        st.plotly_chart(fig, use_container_width=True)

    # ==========================================================
    # SMART RECOMMENDATIONS
    # ==========================================================
    st.markdown("---")
    st.subheader("💡 AI Recommendations")

    recommendations = []

    # GPA Analysis
    if predicted_gpa >= 3.5:
        recommendations.append(("success", "🎉 Excellent academic performance expected. Keep your current study strategy."))
    elif predicted_gpa >= 3.0:
        recommendations.append(("info", "📚 Good performance predicted. Increasing study hours may further improve your GPA."))
    elif predicted_gpa >= 2.5:
        recommendations.append(("warning", "⚠️ Your predicted GPA is average. Try reducing distractions and reviewing difficult subjects."))
    else:
        recommendations.append(("error", "🚨 Low GPA predicted. Increase traditional studying and reduce overdependence on AI tools."))

    # Burnout Analysis (Flexible string matching)
    burnout_str = burnout_label.lower()
    if "high" in burnout_str:
        recommendations.append(("error", "😴 High burnout risk detected. Schedule regular breaks and improve your sleep routine."))
    elif "medium" in burnout_str or "mod" in burnout_str:
        recommendations.append(("warning", "🧠 Moderate burnout risk. Maintain a healthy balance between studying and resting."))
    else:
        recommendations.append(("success", "😊 Low burnout risk. Your mental workload appears balanced."))

    # AI Dependency
    if dependency >= 8:
        recommendations.append(("warning", "🤖 Your AI dependency is very high. Practice solving problems without AI."))

    # Anxiety
    if anxiety >= 8:
        recommendations.append(("warning", "🧘 High anxiety detected. Consider relaxation techniques before exams."))

    # Study Hours
    if study_hours < 10:
        recommendations.append(("warning", "📖 Your traditional study time is very low."))

    # Skill Retention
    if retention < 50:
        recommendations.append(("warning", "📝 Practice more without AI to improve long-term knowledge retention."))

    # Weekly AI Hours
    if weekly_ai > 35:
        recommendations.append(("warning", "⏳ Excessive AI usage may negatively affect independent learning."))

    # Show Recommendations
    for rec_type, text in recommendations:
        if rec_type == "success":
            st.success(text)
        elif rec_type == "warning":
            st.warning(text)
        elif rec_type == "error":
            st.error(text)
        else:
            st.info(text)

    # ==========================================================
    # PERFORMANCE SUMMARY
    # ==========================================================
    st.markdown("---")
    st.subheader("📊 Performance Summary")

    score = 0
    score += predicted_gpa * 20
    score += retention * 0.3
    score += (10 - dependency) * 3
    score += (10 - anxiety) * 3
    score = max(0, min(score, 100))

    st.progress(score / 100)

    if score >= 85:
        st.success(f"Overall Student Performance Score : {score:.1f}/100")
    elif score >= 70:
        st.info(f"Overall Student Performance Score : {score:.1f}/100")
    elif score >= 50:
        st.warning(f"Overall Student Performance Score : {score:.1f}/100")
    else:
        st.error(f"Overall Student Performance Score : {score:.1f}/100")

    # ==========================================================
    # EXPORT RESULTS
    # ==========================================================
    st.markdown("---")
    result_df = pd.DataFrame({
        "Predicted GPA": [round(predicted_gpa, 2)],
        "Burnout Risk": [burnout_label],
        "Confidence (%)": [round(confidence, 2)],
        "Performance Score": [round(score, 2)]
    })

    csv = result_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        "📥 Download Prediction Results",
        data=csv,
        file_name="prediction_results.csv",
        mime="text/csv"
    )

    # ==========================================================
    # INPUT SUMMARY
    # ==========================================================
    st.markdown("---")
    st.subheader("📋 Input Summary")

    display_df = pd.DataFrame({
        "Feature": [
            "Major", "Year", "Pre GPA", "Weekly AI Hours",
            "Primary Use", "Prompt Skill", "Tool Diversity",
            "Paid Subscription", "Study Hours", "AI Dependency",
            "Policy", "Anxiety", "Retention"
        ],
        "Value": [
            major, year, pre_gpa, weekly_ai,
            primary_use, prompt_skill, tool_diversity,
            paid_subscription, study_hours, dependency,
            policy, anxiety, retention
        ]
    })

    st.dataframe(display_df, use_container_width=True, hide_index=True)

    # ==========================================================
    # ANALYTICS DASHBOARD
    # ==========================================================
    st.markdown("---")
    st.header("📈 Student Analytics Dashboard")

    chart1, chart2 = st.columns(2)

    with chart1:
        radar_categories = ["Study", "Retention", "Low Anxiety", "Low Dependency", "AI Balance"]
        radar_values = [
            min(study_hours / 40, 1) * 100,
            retention,
            (10 - anxiety) * 10,
            (10 - dependency) * 10,
            max(0, 100 - (weekly_ai * 2))
        ]

        radar_values.append(radar_values[0])
        radar_categories.append(radar_categories[0])

        radar = go.Figure()
        radar.add_trace(go.Scatterpolar(
            r=radar_values,
            theta=radar_categories,
            fill='toself',
            name='Student'
        ))
        radar.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
            showlegend=False,
            height=450,
            title="Student Skills Overview"
        )
        st.plotly_chart(radar, use_container_width=True)

    with chart2:
        pie = go.Figure(data=[go.Pie(
            labels=["Study Hours", "AI Usage", "Retention"],
            values=[study_hours, weekly_ai, retention],
            hole=.45
        )])
        pie.update_layout(title="Learning Distribution", height=450)
        st.plotly_chart(pie, use_container_width=True)

    st.markdown("---")

    # Academic Status Card
    if predicted_gpa >= 3.5:
        color, icon, status = "#4CAF50", "🏆", "Excellent"
    elif predicted_gpa >= 3.0:
        color, icon, status = "#2196F3", "📘", "Good"
    elif predicted_gpa >= 2.5:
        color, icon, status = "#FFC107", "📙", "Average"
    else:
        color, icon, status = "#F44336", "📕", "Needs Improvement"

    st.markdown(f"""
    <div style="background:{color}; padding:20px; border-radius:20px; color:white; text-align:center;">
        <h2>{icon} Academic Status</h2>
        <h1>{status}</h1>
    </div>
    """, unsafe_allow_html=True)

    # Burnout Status Card
    if "low" in burnout_str:
        burn_color = "#4CAF50"
    elif "medium" in burnout_str or "mod" in burnout_str:
        burn_color = "#FFC107"
    else:
        burn_color = "#F44336"

    st.markdown(f"""
    <div style="background:{burn_color}; padding:20px; border-radius:20px; margin-top:20px; color:white; text-align:center;">
        <h2>🧠 Burnout Risk</h2>
        <h1>{burnout_label}</h1>
        <p>Confidence : {confidence:.1f}%</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    st.subheader("🤖 AI Usage Analysis")
    usage_df = pd.DataFrame({
        "Metric": ["Weekly AI Hours", "Study Hours", "Dependency", "Retention"],
        "Value": [weekly_ai, study_hours, dependency * 10, retention]
    })
    usage_chart = px.bar(
        usage_df,
        x="Metric",
        y="Value",
        color="Metric",
        text_auto=True,
        title="Learning Behavior"
    )
    st.plotly_chart(usage_chart, use_container_width=True)

    st.markdown("---")
    st.subheader("🎯 Academic Evaluation")

    if predicted_gpa >= 3.5:
        st.balloons()
        st.success("Outstanding academic performance predicted.")
    elif predicted_gpa >= 3.0:
        st.info("Good academic performance.")
    elif predicted_gpa >= 2.5:
        st.warning("Average performance. More practice is recommended.")
    else:
        st.error("High academic risk detected.")

    st.subheader("🧠 Burnout Assessment")
    if "high" in burnout_str:
        st.error("""
High burnout probability detected.

Recommendations:
• Take regular breaks.
• Sleep at least 7 hours.
• Reduce stress.
• Balance AI usage with self-learning.
""")
    elif "medium" in burnout_str or "mod" in burnout_str:
        st.warning("""
Moderate burnout level.

Keep balancing study and rest.
""")
    else:
        st.success("""
Low burnout risk.

Great balance between studying and wellbeing.
""")

    st.markdown("---")

    # Final Dashboard Metrics
    st.subheader("📌 Final Dashboard")
    final_col1, final_col2, final_col3 = st.columns(3)
    final_col1.metric("Overall Score", f"{score:.1f}/100")
    final_col2.metric("Predicted GPA", f"{predicted_gpa:.2f}")
    final_col3.metric("Burnout", burnout_label)

    st.markdown("---")

# ==========================================================
# FOOTER
# ==========================================================

st.markdown("""
<div style="text-align:center; padding:20px;">
    <h2>🎓 AI Student Impact Prediction System</h2>
    <p>Machine Learning Dashboard built using <b>Streamlit</b>, <b>Scikit-Learn</b>, <b>Plotly</b></p>
    <p>Developed by <b>Mostafa Ahmed</b></p>
</div>
""", unsafe_allow_html=True)

st.caption("© 2026 AI Student Impact Prediction System")
