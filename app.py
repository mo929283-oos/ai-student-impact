import streamlit as st
import pandas as pd
import numpy as np
import joblib

# ضبط إعدادات الصفحة لتكون عريضة ومريحة للعين
st.set_page_config(page_title="مستشار الطالب الذكي AI", layout="wide")

# تصميم واجهة التطبيق بالعربي
st.markdown("<h1 style='text-align: center; color: #4CAF50;'>🎓 مستشار الطالب الذكي وأداء الذكاء الاصطناعي</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>أدخل بياناتك الدراسية وسلوك استخدامك للذكاء الاصطناعي لنتوقع أدائك الدراسي ونقدم لك نصائح مخصصة!</p>", unsafe_allow_html=True)
st.write("---")

# تحميل الموديلات والـ Encoders المحفوظة
@st.cache_resource
def load_resources():
    gpa_model = joblib.load('gpa_predictor_model.joblib')
    burnout_model = joblib.load('burnout_classifier_model.joblib')
    encoders = joblib.load('label_encoders.joblib')
    burnout_encoder = joblib.load('burnout_encoder.joblib')
    feature_cols = joblib.load('feature_columns.joblib')
    return gpa_model, burnout_model, encoders, burnout_encoder, feature_cols

try:
    gpa_model, burnout_model, encoders, burnout_encoder, feature_cols = load_resources()
except Exception as e:
    st.error("⚠️ يرجى تشغيل ملف `train_models.py` أولاً لإنشاء الموديلات المحفوظة!")
    st.stop()

# تقسيم الشاشة لجزأين: جزء للمدخلات وجزء للمخرجات والنصائح
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("📝 بيانات الطالب وسلوك المذاكرة")
    
    # المدخلات الأساسية
    major = st.selectbox("التخصص الدراسي", encoders['Major_Category'].classes_)
    year = st.selectbox("السنة الدراسية", encoders['Year_of_Study'].classes_)
    pre_gpa = st.slider("المعدل الدراسي قبل الترم (Pre-Semester GPA)", 0.0, 4.0, 3.0, step=0.1)
    
    st.write("---")
    st.subheader("🤖 سلوك استخدام الذكاء الاصطناعي التوليدي (GenAI)")
    
    weekly_ai_hours = st.slider("ساعات استخدام الـ AI أسبوعياً", 0.0, 40.0, 10.0, step=0.5)
    use_case = st.selectbox("الاستخدام الأساسي للـ AI", encoders['Primary_Use_Case'].classes_)
    skill = st.selectbox("مستواك في كتابة الأوامر (Prompt Engineering)", encoders['Prompt_Engineering_Skill'].classes_)
    tool_diversity = st.number_input("عدد الأدوات المختلفة اللي بتستخدمها (ChatGPT, Claude, etc.)", min_value=1, max_value=10, value=2)
    paid_sub = st.checkbox("هل مشترك في باقة مدفوعة؟ (ChatGPT Plus, Copilot Pro)")
    
    st.write("---")
    st.subheader("📚 المذاكرة التقليدية والصحة النفسية")
    
    trad_study_hours = st.slider("ساعات المذاكرة التقليدية أسبوعياً (بدون AI)", 0.0, 40.0, 15.0, step=0.5)
    ai_dependency = st.slider("مدى اعتمادك واعتماديتك على الـ AI (من 1 لـ 5)", 1, 5, 3)
    policy = st.selectbox("سياسة كليتك تجاه الـ AI", encoders['Institutional_Policy'].classes_)
    exam_anxiety = st.slider("مستوى توترك وقلقك أثناء الامتحانات (من 1 لـ 10)", 1, 10, 5)
    skill_retention = st.slider("نسبة احتفاظك بالمعلومات والمهارات في عقلك (من 0 لـ 100)", 0.0, 100.0, 75.0)

with col2:
    st.subheader("🎯 توقعات الموديل الذكي (Predictions)")
    
    if st.button("🚀 احسب التوقعات وحلل أدائي الدراسي", use_container_width=True):
        # 1. تحضير الداتا للموديل بنفس الترتيب وتحويل القيم النصية لأرقام بالـ encoders
        input_data = {
            'Major_Category': encoders['Major_Category'].transform([major])[0],
            'Year_of_Study': encoders['Year_of_Study'].transform([year])[0],
            'Pre_Semester_GPA': pre_gpa,
            'Weekly_GenAI_Hours': weekly_ai_hours,
            'Primary_Use_Case': encoders['Primary_Use_Case'].transform([use_case])[0],
            'Prompt_Engineering_Skill': encoders['Prompt_Engineering_Skill'].transform([skill])[0],
            'Tool_Diversity': tool_diversity,
            'Paid_Subscription': int(paid_sub),
            'Traditional_Study_Hours': trad_study_hours,
            'Perceived_AI_Dependency': ai_dependency,
            'Institutional_Policy': encoders['Institutional_Policy'].transform([policy])[0],
            'Anxiety_Level_During_Exams': exam_anxiety,
            'Skill_Retention_Score': skill_retention
        }
        
        # تحويل لـ DataFrame بالترتيب الصحيح للأعمدة
        input_df = pd.DataFrame([input_data])[feature_cols]
        
        # 2. التوقعات
        pred_gpa = gpa_model.predict(input_df)[0]
        # حصر الـ GPA بين 0 و 4
        pred_gpa = min(4.0, max(0.0, pred_gpa))
        
        pred_burnout_idx = burnout_model.predict(input_df)[0]
        pred_burnout = burnout_encoder.inverse_transform([pred_burnout_idx])[0]
        
        # 3. عرض النتائج بشكل جمالي
        st.write("### 📈 النتائج المتوقعة لترمك الحالي:")
        
        # عرض الـ GPA المتوقع مقارنة بالقديم
        gpa_diff = pred_gpa - pre_gpa
        st.metric(label="المعدل التراكمي المتوقع في نهاية الترم (Post-Semester GPA)", 
                  value=f"{pred_gpa:.3f}", 
                  delta=f"{gpa_diff:+.3f} (مقارنة بالقديم {pre_gpa:.2f})")
        
        # عرض خطر الاحتراق النفسي بالألوان
        st.write("### 🔥 خطر الإجهاد والاحتراق الدراسي (Burnout Risk):")
        if pred_burnout == 'High':
            st.error(f"🔴 مستوى الخطر: {pred_burnout} (مرتفع جداً!)")
        elif pred_burnout == 'Medium':
            st.warning(f"🟡 مستوى الخطر: {pred_burnout} (متوسط - انتبه لنفسك!)")
        else:
            st.success(f"🟢 مستوى الخطر: {pred_burnout} (آمن ومستقر!)")
            
        st.write("---")
        
        # 4. المستشار الذكي (Rule-Based Advisor Based on ML Predictions)
        st.write("### 💡 نصائح وتوصيات مخصصة لك:")
        
        # نصائح بناءً على التوازن بين المذاكرة والـ AI
        if weekly_ai_hours > trad_study_hours:
            st.write("⚠️ *أنت تقضي ساعات على الـ AI أكثر من المذاكرة التقليدية!* حاول زيادة ساعات القراءة وحل المسائل بنفسك لضمان عدم نسيان المهارات الأساسية.")
        else:
            st.write("✅ *توازن رائع!* ساعات مذاكرتك التقليدية ممتازة وتساعدك على تثبيت الأساسيات بجانب استخدام الذكاء الاصطناعي.")
            
        # نصائح للتوتر والاحتراق النفسي
        if exam_anxiety > 7 or pred_burnout == 'High':
            st.write("🛑 *تنبيه صحي:* مستوى توترك أو احتراقك النفسي مرتفع. ننصحك بتقليل ساعات استخدام الأدوات وتجربة أسلوب البومودورو في المذاكرة، مع أخذ فترات راحة كافية.")
            
        # نصائح لمهارة الـ Prompting
        if skill == 'Beginner':
            st.write("💡 *نصيحة مهارية:* تحسين مستواك في كتابة الأوامر (Prompt Engineering) سيساعدك على توفير الوقت والحصول على إجابات أدق بكثير وبمجهود أقل.")
