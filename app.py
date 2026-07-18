import streamlit as st
import pandas as pd
import numpy as np
import joblib

# ضبط إعدادات الصفحة لتكون عريضة ومريحة للعين
st.set_page_config(page_title="AI مستشار الطالب الذكي", layout="wide")

# تصميم واجهة التطبيق بالعربي
st.markdown("<h1 style='text-align: center; color: #4CAF50;'>🎓 مستشار الطالب الذكي وأداء الذكاء الاصطناعي</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>أدخل بياناتك الدراسية وسلوك استخدامك للذكاء الاصطناعي لنتوقع أداءك الدراسي ونقدم لك نصائح مخصصة</p>", unsafe_allow_html=True)
st.write("---")

# تحميل الموديلات والـ Encoders المحفوظة
@st.cache_resource
def load_resources():
    try:
        gpa_model = joblib.load('gpa_predictor_model.joblib')
        burnout_model = joblib.load('burnout_classifier_model.joblib')
        encoders = joblib.load('label_encoders.joblib')
        return gpa_model, burnout_model, encoders
    except:
        return None, None, None

gpa_model, burnout_model, encoders = load_resources()

if gpa_model is None:
    st.error("⚠️ يرجى التأكد من وجود ملفات الموديلات في المستودع بنفس الأسماء")
else:
    # تقسيم الشاشة إلى جزئين: المدخلات والنتائج
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("📝 المدخلات الدراسية والسلوكية")
        
        major_category = st.selectbox("التخصص الدراسي (Major Category)", ["STEM", "Medical", "Business", "Humanities", "Social Sciences"])
        year_of_study = st.selectbox("السنة الدراسية (Year of Study)", ["Freshman", "Sophomore", "Junior", "Senior"])
        pre_gpa = st.slider("المعدل الدراسي السابق (Pre-Semester GPA)", 0.0, 4.0, 3.0, step=0.01)
        
        st.write("**📊 سلوك استخدام الذكاء الاصطناعي:**")
        weekly_genai_hours = st.slider("ساعات استخدام الـ GenAI أسبوعياً", 0.0, 40.0, 10.0, step=0.5)
        primary_use_case = st.selectbox("الاستخدام الأساسي للـ AI", ["Code Generation", "Explanation/Tutoring", "Summarization", "Ideation", "Copywriting/Drafting"])
        prompt_skill = st.selectbox("مستواك في الـ Prompt Engineering", ["Beginner", "Intermediate", "Advanced"])
        tool_diversity = st.slider("عدد أدوات الـ AI المختلفة التي تستخدمها", 1, 10, 3)
        paid_sub = st.checkbox("هل مشترك في نسخة مدفوعة؟ (ChatGPT Plus, Claude Pro, etc.)", value=False)
        ai_dependency = st.slider("مدى اعتمادك على الـ AI من (1-5)", 1, 5, 3)
        
        st.write("**📚 العادات الدراسية والحالة النفسية:**")
        trad_study_hours = st.slider("ساعات المذاكرة التقليدية أسبوعياً (بدون AI)", 0.0, 40.0, 15.0, step=0.5)
        anxiety_level = st.slider("مستوى القلق والتوتر أثناء الامتحانات (1-10)", 1, 10, 5)
        skill_retention = st.slider("مستوى استرجاعك للمعلومات بدون AI (من 0 لـ 100)", 0.0, 100.0, 75.0, step=1.0)
        policy = st.selectbox("سياسة الجامعة تجاه الـ AI", ["Allowed_With_Citation", "Unregulated", "Restricted", "Banned"])

    with col2:
        st.subheader("🔮 التوقعات والتحليلات الذكية")
        
        # تجهيز الـ DataFrame بالكامل أولاً
        input_dict = {
            'Major_Category': major_category,
            'Year_of_Study': year_of_study,
            'Pre_Semester_GPA': pre_gpa,
            'Weekly_GenAI_Hours': weekly_genai_hours,
            'Primary_Use_Case': primary_use_case,
            'Prompt_Engineering_Skill': prompt_skill,
            'Tool_Diversity': tool_diversity,
            'Paid_Subscription': paid_sub,
            'Traditional_Study_Hours': trad_study_hours,
            'Perceived_AI_Dependency': ai_dependency,
            'Institutional_Policy': policy,
            'Anxiety_Level_During_Exams': anxiety_level,
            'Skill_Retention_Score': skill_retention
        }
        
        input_data = pd.DataFrame([input_dict])
        
        # تطبيق الـ Label Encoding
        categorical_cols = ['Major_Category', 'Year_of_Study', 'Primary_Use_Case', 'Prompt_Engineering_Skill', 'Institutional_Policy']
        for col in categorical_cols:
            if col in encoders:
                le = encoders[col]
                input_data[col] = input_data[col].map(lambda s: le.transform([s])[0] if s in le.classes_ else 0)
        
        # 🎯 الـ 13 عمود المطلوبين بالظبط بدون Student_ID وبدون الـ Targets
        features_order = [
            'Major_Category', 'Year_of_Study', 'Pre_Semester_GPA', 'Weekly_GenAI_Hours', 
            'Primary_Use_Case', 'Prompt_Engineering_Skill', 'Tool_Diversity', 
            'Paid_Subscription', 'Traditional_Study_Hours', 'Perceived_AI_Dependency', 
            'Institutional_Policy', 'Anxiety_Level_During_Exams', 'Skill_Retention_Score'
        ]
        
        final_features = input_data[features_order].values
        
        # زر التوقع
        if st.button("احسب التوقعات والنصائح الاستشارية", type="primary"):
            try:
                # 1. توقع الـ GPA المستقبلي
                pred_gpa = gpa_model.predict(final_features)[0]
                pred_gpa = max(0.0, min(4.0, pred_gpa))
                
                # 2. توقع الـ Burnout
                pred_burnout = burnout_model.predict(final_features)[0]
                
                # عرض النتيجة
                st.markdown(f"### 📈 المعدل التراكمي المتوقع (Predicted Post-GPA): `{pred_gpa:.2f} / 4.0`")
                
                if pred_burnout in [1, 'High', 'Yes']:
                    st.error("🚨 تحذير: مؤشراتك تدل على احتمالية عالية للاحتراق النفسي (High Burnout Risk)!")
                    has_burnout = True
                else:
                    st.success("✅ ممتاز: مؤشراتك تدل على استقرار نفسي ودراسي (Low Burnout Risk).")
                    has_burnout = False
                
                st.write("---")
                st.subheader("🎯 الخطة الاستشارية المقترحة لك (AI Recommendation)")
                
                if weekly_genai_hours > trad_study_hours:
                    st.warning("⚠️ **تنبيه حول الاعتمادية:** ساعات استخدامك للذكاء الاصطناعي أكبر من ساعات مذاكرتك التقليدية. هذا قد يضعف مهارة استرجاع المعلومات لديك.")
                if anxiety_level > 7:
                    st.info("🧘 **نصيحة لتقليل قلق الامتحانات:** جرب استخدام الـ AI لعمل امتحانات تجريبية لنفسك وتقييم إجاباتك.")
                else:
                    st.success("🌟 استمر في الحفاظ على هذا التوازن الجيد بين المذاكرة الذكية والتقليدية.")
                    
            except Exception as e:
                st.error(f"حدث خطأ أثناء الحساب: {e}")

            # الجدول التفاعلي
            st.write("---")
            st.subheader("📅 لوحة المتصدرين والـ Habit Tracker (Community Leaderboard)")
            st.markdown(f"""
            | الترتيب | اسم الطالب | التخصص | معدل الالتزام بالأهداف الدراسي | حالة الأداء المتوقع |
            | :---: | :--- | :---: | :---: | :---: |
            | 🥇 **الأول** | أحمد محمد | STEM | 98% | ممتاز 🔥 |
            | 🥈 **الثاني** | سارة أحمد | Medical | 94% | ممتاز 🔥 |
            | 🥉 **الثالث** | **أنت (طالب مستفسر)** | **{major_category}** | **88%** | **متزن وجيد** ⚡ |
            | 4 | يوسف عمر | Business | 82% | يحتاج تركيز |
            """)
