import streamlit as st
import pandas as pd
import numpy as np
import joblib

# ضبط إعدادات الصفحة
st.set_page_config(page_title="AI مستشار الطالب الذكي", layout="wide")

st.markdown("<h1 style='text-align: center; color: #4CAF50;'>🎓 مستشار الطالب الذكي وأداء الذكاء الاصطناعي</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>أدخل بياناتك الدراسية وسلوك استخدامك للذكاء الاصطناعي لنتوقع أداءك الدراسي ونقدم لك نصائح مخصصة</p>", unsafe_allow_html=True)
st.write("---")

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
    st.error("⚠️ يرجى التأكد من وجود ملفات الموديلات في المستودع")
else:
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("📝 المدخلات الدراسية والسلوكية")
        
        gender = st.selectbox("الجنس", ["Male", "Female"])
        major = st.selectbox("التخصص الدراسي", ["Engineering", "Computer Science", "Business", "Arts", "Sciences"])
        year = st.selectbox("السنة الدراسية", ["Freshman", "Sophomore", "Junior", "Senior"])
        
        stud_hours = st.slider("ساعات المذاكرة التقليدية أسبوعياً", 1, 40, 15)
        ai_hours = st.slider("ساعات استخدام أدوات الذكاء الاصطناعي أسبوعياً", 0, 30, 5)
        attendance = st.slider("نسبة الحضور (%)", 50, 100, 85)
        sleep_hours = st.slider("معدل ساعات النوم اليومية", 4, 10, 7)
        stress_level = st.slider("مستوى التوتر الدراسي (1-10)", 1, 10, 5)

    with col2:
        st.subheader("🔮 التوقعات والتحليلات الذكية")
        
        # عمل الـ DataFrame بنفس ترتيب الـ Features الأصلي تماماً
        input_data = pd.DataFrame([{
            'Gender': gender,
            'Major': major,
            'Year_of_Study': year,
            'Study_Hours_Per_Week': stud_hours,
            'AI_Usage_Hours_Per_Week': ai_hours,
            'Attendance_Percentage': attendance,
            'Sleep_Hours_Per_Day': sleep_hours,
            'Stress_Level': stress_level
        }])
        
        # تحويل النصوص لأرقام
        for col in ['Gender', 'Major', 'Year_of_Study']:
            if col in encoders:
                le = encoders[col]
                input_data[col] = input_data[col].map(lambda s: le.transform([s])[0] if s in le.classes_ else 0)
        
        if st.button("احسب التوقعات والنصائح الاستشارية", type="primary"):
            # 🔥 الحل هنا: نمرر القيم كـ numpy array بدون الأسماء لتجنب تصفير الداتا
            features_array = input_data.values
            
            pred_gpa = gpa_model.predict(features_array)[0]
            pred_gpa = max(0.0, min(4.0, pred_gpa))
            
            pred_burnout = burnout_model.predict(features_array)[0]
            
            st.markdown(f"### 📈 المعدل التراكمي المتوقع (Predicted GPA): `{pred_gpa:.2f} / 4.0`")
            
            if pred_burnout == 1 or pred_burnout == 'Yes':
                st.error("🚨 تحذير: المؤشرات تدل على أنك معرض للاحتراق النفسي (High Burnout Risk)!")
                has_burnout = True
            else:
                st.success("✅ ممتاز: حالتك النفسية والدراسية متزنة حالياً (Low Burnout Risk).")
                has_burnout = False
                
            st.write("---")
            st.subheader("🎯 الخطة الاستشارية المقترحة لك (AI Recommendation)")
            
            if pred_gpa < 3.0:
                st.warning(f"⚠️ **لرفع معدلك التراكمي من {pred_gpa:.2f} إلى +3.5:**")
                st.write(f"• نقترح زيادة ساعات المذاكرة التقليدية من **{stud_hours}** ساعة إلى **{stud_hours + 5}** ساعات أسبوعياً.")
            else:
                st.info("🌟 **للحفاظ على تفوقك الدراسي:**")
                st.write("• استمر على هذا المعدل المتميز من الحضور والمذاكرة.")

            if has_burnout or stress_level > 7 or sleep_hours < 6:
                st.markdown("#### 🧘 روتين مقترح لتقليل التوتر الحاد:")
                st.write(f"• جدولك الجديد يلزمك بزيادة النوم إلى **7 ساعات ونصف** فوراً.")

            st.write("---")
            st.subheader("📅 الـ Habit Tracker اليومي المتكامل (حالة تجريبية)")
            
            col_t1, col_t2, col_t3 = st.columns(3)
            with col_t1:
                st.checkbox("أتممت ساعات المذاكرة المقترحة لليوم 📚", value=True)
            with col_t2:
                st.checkbox("نمت أكثر من 7 ساعات الليلة الماضية 😴")
            with col_t3:
                st.checkbox("قللت الاعتماد على الـ AI واعتمدت على نفسي 🧠", value=True)
                
            st.markdown("""
            | اليوم | تحدي المذاكرة (ساعات) | نقاط الصحة النفسية | ترتيبك في الـ Community |
            | :--- | :---: | :---: | :---: |
            | **السبت** | 3 / 3 ✅ | 85% | الـ 5 على الدفعة 🏆 |
            | **الأحد** | 2 / 3 ⏳ | 60% | الـ 12 على الدفعة |
            """)
