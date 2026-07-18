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
    st.error("⚠️ يرجى التأكد من وجود ملفات الموديلات في المستودع")
else:
    # تقسيم الشاشة إلى جزئين: المدخلات والنتائج
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("📝 المدخلات الدراسية والسلوكية")
        
        # مدخلات نصية (Categorical)
        gender = st.selectbox("الجنس", ["Male", "Female"])
        major = st.selectbox("التخصص الدراسي", ["Engineering", "Computer Science", "Business", "Arts", "Sciences"])
        year = st.selectbox("السنة الدراسية", ["Freshman", "Sophomore", "Junior", "Senior"])
        
        # مدخلات رقمية (Numerical) via Sliders
        stud_hours = st.slider("ساعات المذاكرة التقليدية أسبوعياً", 1, 40, 15)
        ai_hours = st.slider("ساعات استخدام أدوات الذكاء الاصطناعي أسبوعياً", 0, 30, 5)
        attendance = st.slider("نسبة الحضور (%)", 50, 100, 85)
        sleep_hours = st.slider("معدل ساعات النوم اليومية", 4, 10, 7)
        stress_level = st.slider("مستوى التوتر الدراسي (1-10)", 1, 10, 5)

    with col2:
        st.subheader("🔮 التوقعات والتحليلات الذكية")
        
        # تجهيز البيانات للموديل
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
        
        # تطبيق الـ Label Encoding على الأعمدة النصية
        for col in ['Gender', 'Major', 'Year_of_Study']:
            if col in encoders:
                le = encoders[col]
                input_data[col] = input_data[col].map(lambda s: le.transform([s])[0] if s in le.classes_ else 0)
        
        # 🎯 الحل السحري: إجبار الأعمدة على أخذ نفس ترتيب الموديل بالظبط لتفادي أي KeyError
        try:
            model_features = gpa_model.feature_names_in_
            # لو الموديل متخزن جواه أسماء الأعمدة، هنخلي الـ dataframe يترتب زيه بالظبط
            input_data = input_data.reindex(columns=model_features, fill_value=0)
        except AttributeError:
            # لو الموديل مش متخزن جواه الأسماء، هيمشي بالترتيب الافتراضي للموديل
            pass
            
        # زر الحساب والتوقع
        if st.button("احسب التوقعات والنصائح الاستشارية", type="primary"):
            # 1. توقع المعدل التراكمي
            pred_gpa = gpa_model.predict(input_data)[0]
            pred_gpa = max(0.0, min(4.0, pred_gpa))
            
            # 2. توقع الاحتراق النفسي
            pred_burnout = burnout_model.predict(input_data)[0]
            
            # عرض النتائج بشكل جمالي
            st.markdown(f"### 📈 المعدل التراكمي المتوقع (Predicted GPA): `{pred_gpa:.2f} / 4.0`")
            
            if pred_burnout == 1 or pred_burnout == 'Yes':
                st.error("🚨 تحذير: المؤشرات تدل على أنك معرض للاحتراق النفسي (High Burnout Risk) بسبب ضغط الدراسة أو قلة النوم!")
                has_burnout = True
            else:
                st.success("✅ ممتاز: حالتك النفسية والدراسية متزنة حالياً (Low Burnout Risk).")
                has_burnout = False
                
            st.write("---")
            
            # 💡 نظام التوصية والخطط الذكية
            st.subheader("🎯 الخطة الاستشارية المقترحة لك (AI Recommendation)")
            
            if pred_gpa < 3.0:
                st.warning(f"⚠️ **لرفع معدلك التراكمي من {pred_gpa:.2f} إلى +3.5:**")
                st.write(f"• نقترح زيادة ساعات المذاكرة التقليدية من **{stud_hours}** ساعة إلى **{stud_hours + 5}** ساعات أسبوعياً.")
                if attendance < 85:
                    st.write(f"• احذر! نسبة حضورك الحالية ({attendance}%) منخفضة. يجب رفعها فوق **85%** لتجنب خسارة أعمال السنة.")
            else:
                st.info("🌟 **للحفاظ على تفوقك الدراسي:**")
                st.write("• استمر على هذا معدل من الحضور والمذاكرة، وتجنب الاعتماد الكلي على الذكاء الاصطناعي في حل التكليفات بدون فهم.")

            if has_burnout or stress_level > 7 or sleep_hours < 6:
                st.markdown("#### 🧘 روتين مقترح لتقليل التوتر الحاد:")
                st.write(f"• تم رصد معدل نوم منخفض ({sleep_hours} ساعات). جدولك الجديد يلزمك بزيادة النوم إلى **7 ساعات ونصف** فوراً.")
                st.write(f"• قلل ساعات استخدام الـ AI من **{ai_hours}** ساعات إلى **{max(0, ai_hours-3)}** ساعات لتستعيد ثقتك الدراسية وعافيتك النفسية.")

            # 🛠️ جزء الـ Habit Tracker التفاعلي
            st.write("---")
            st.subheader("📅 الـ Habit Tracker اليومي المتكامل (حالة تجريبية)")
            st.info("💡 هكذا سيتابع الطالب خطته يومياً لضمان الوصول للـ GPA المستهدف وتجنب الـ Burnout:")
            
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
