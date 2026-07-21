import streamlit as st
import pandas as pd
import numpy as np
import joblib

# 1. ضبط إعدادات الصفحة
st.set_page_config(
    page_title="مستشار الطالب الذكي AI", 
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 2. حقن CSS احترافي يغير شكل Streamlit بالكامل لـ Dashboard عصرية
st.markdown("""
    <style>
    /* استيراد خط Cairo من Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Cairo:wght@400;600;700;800&display=swap');

    html, body, [class*="css"]  {
        font-family: 'Cairo', sans-serif;
        direction: rtl;
        text-align: right;
    }

    /* خلفية الصفحة كحلي مائل للرمادي الداكن العصرى */
    .stApp {
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
        color: #f8fafc;
    }

    /* هيدر فاخر بـ Gradient */
    .header-box {
        background: linear-gradient(90deg, #3b82f6 0%, #8b5cf6 100%);
        border-radius: 16px;
        padding: 2.5rem;
        text-align: center;
        box-shadow: 0 10px 25px -5px rgba(59, 130, 246, 0.4);
        margin-bottom: 2rem;
    }

    .header-title {
        color: #ffffff;
        font-weight: 800;
        font-size: 2.2rem;
        margin: 0;
    }

    .header-subtitle {
        color: #e2e8f0;
        font-size: 1.1rem;
        margin-top: 0.5rem;
    }

    /* كروت زجاجية أنيقة للتقسيم (Glassmorphism Cards) */
    .custom-card {
        background: rgba(30, 41, 59, 0.7);
        backdrop-filter: blur(12px);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 16px;
        padding: 1.5rem;
        margin-bottom: 1.5rem;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.2);
    }

    .card-title {
        color: #38bdf8;
        font-size: 1.25rem;
        font-weight: 700;
        margin-bottom: 1rem;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }

    /* تخصيص زر التوقع */
    .stButton>button {
        background: linear-gradient(90deg, #10b981 0%, #059669 100%) !important;
        color: white !important;
        font-size: 1.1rem !important;
        font-weight: 700 !important;
        border-radius: 12px !important;
        border: none !important;
        padding: 0.75rem 1.5rem !important;
        box-shadow: 0 4px 14px 0 rgba(16, 185, 129, 0.39) !important;
        transition: all 0.3s ease !important;
    }

    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px 0 rgba(16, 185, 129, 0.55) !important;
    }

    /* كروت النتائج والمقاييس */
    .metric-container {
        background: rgba(15, 23, 42, 0.8);
        border: 1px solid rgba(56, 189, 248, 0.2);
        border-radius: 12px;
        padding: 1.2rem;
        text-align: center;
    }

    .burnout-badge {
        padding: 0.8rem 1.2rem;
        border-radius: 12px;
        font-weight: 700;
        font-size: 1.1rem;
        text-align: center;
        margin-top: 0.5rem;
    }
    
    .burnout-high { background: rgba(239, 68, 68, 0.2); color: #fca5a5; border: 1px solid #ef4444; }
    .burnout-med { background: rgba(245, 158, 11, 0.2); color: #fde047; border: 1px solid #f59e0b; }
    .burnout-low { background: rgba(16, 185, 129, 0.2); color: #6ee7b7; border: 1px solid #10b981; }

    /* تحسين شكل الـ Divider */
    hr {
        border-color: rgba(255, 255, 255, 0.1);
    }
    </style>
""", unsafe_allow_html=True)

# 3. الهيدر الفاخر
st.markdown("""
    <div class="header-box">
        <h1 class="header-title">🎓 مستشار الطالب الذكي (AI Academic Advisor)</h1>
        <p class="header-subtitle">تحليل أداء واستخدام الذكاء الاصطناعي للتنبؤ بالمعدل التراكمي وتجنب الاحتراق النفسي</p>
    </div>
""", unsafe_allow_html=True)

# تحميل الموديلات والـ Encoders
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

# تقسيم الشاشة إلى عمودين منظمين
col1, col2 = st.columns([1.1, 0.9], gap="large")

with col1:
    # كارت 1: البيانات الأساسية
    st.markdown('<div class="custom-card">', unsafe_allow_html=True)
    st.markdown('<div class="card-title">📝 البيانات الأكاديمية الأساسية</div>', unsafe_allow_html=True)
    
    c1, c2 = st.columns(2)
    with c1:
        major = st.selectbox("التخصص الدراسي", encoders['Major_Category'].classes_)
    with c2:
        year = st.selectbox("السنة الدراسية", encoders['Year_of_Study'].classes_)
        
    pre_gpa = st.slider("المعدل الدراسي الحالي/السابق (Pre-GPA)", 0.0, 4.0, 3.0, step=0.01)
    st.markdown('</div>', unsafe_allow_html=True)

    # كارت 2: استخدام الـ AI
    st.markdown('<div class="custom-card">', unsafe_allow_html=True)
    st.markdown('<div class="card-title">🤖 سلوك استخدام الذكاء الاصطناعي (GenAI)</div>', unsafe_allow_html=True)
    
    weekly_ai_hours = st.slider("ساعات استخدام الـ AI أسبوعياً", 0.0, 40.0, 10.0, step=0.5)
    
    c3, c4 = st.columns(2)
    with c3:
        use_case = st.selectbox("الاستخدام الأساسي للـ AI", encoders['Primary_Use_Case'].classes_)
    with c4:
        skill = st.selectbox("مستوى كتابة الأوامر (Prompt Skill)", encoders['Prompt_Engineering_Skill'].classes_)
        
    tool_diversity = st.number_input("عدد أدوات الـ AI المستخدمة", min_value=1, max_value=10, value=2)
    paid_sub = st.checkbox("مشترك في باقة مدفوعة؟ (ChatGPT Plus, Claude Pro, etc.)")
    st.markdown('</div>', unsafe_allow_html=True)

    # كارت 3: المذاكرة التقليدية والصحة النفسية
    st.markdown('<div class="custom-card">', unsafe_allow_html=True)
    st.markdown('<div class="card-title">📚 المذاكرة التقليدية والحالة النفسية</div>', unsafe_allow_html=True)
    
    trad_study_hours = st.slider("ساعات المذاكرة التقليدية أسبوعياً", 0.0, 40.0, 15.0, step=0.5)
    ai_dependency = st.slider("مدى اعتمادك على الـ AI (من 1 لـ 5)", 1, 5, 3)
    policy = st.selectbox("سياسة الكلية تجاه الـ AI", encoders['Institutional_Policy'].classes_)
    
    c5, c6 = st.columns(2)
    with c5:
        exam_anxiety = st.slider("توتر الامتحانات (1-10)", 1, 10, 5)
    with c6:
        skill_retention = st.slider("نسبة الحفظ والاسترجاع (%)", 0.0, 100.0, 75.0)
    st.markdown('</div>', unsafe_allow_html=True)

with col2:
    st.markdown('<div class="custom-card">', unsafe_allow_html=True)
    st.markdown('<div class="card-title">📊 التوقعات والتحليل الاستشاري</div>', unsafe_allow_html=True)
    
    st.write("اضغط على الزر أدناه لمعالجة البيانات عبر موديلي الـ ML وحساب التوقعات:")
    
    if st.button("🚀 احسب التوقعات وحلل أدائي الدراسي", use_container_width=True):
        # 1. تحضير البيانات
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
        
        input_df = pd.DataFrame([input_data])[feature_cols]
        
        # 2. التوقعات
        pred_gpa = gpa_model.predict(input_df)[0]
        pred_gpa = min(4.0, max(0.0, pred_gpa))
        
        pred_burnout_idx = burnout_model.predict(input_df)[0]
        pred_burnout = burnout_encoder.inverse_transform([pred_burnout_idx])[0]
        
        # 3. عرض المخرجات بديزاين Fancier
        gpa_diff = pred_gpa - pre_gpa
        diff_color = "#10b981" if gpa_diff >= 0 else "#ef4444"
        
        st.markdown("---")
        st.markdown("#### 📈 المعدل التراكمي المتوقع (Post-GPA):")
        
        st.markdown(f"""
            <div class="metric-container">
                <h1 style="font-size: 3rem; color: #38bdf8; margin:0;">{pred_gpa:.3f} <span style="font-size: 1.2rem; color: #94a3b8;">/ 4.00</span></h1>
                <p style="color: {diff_color}; font-weight: bold; margin:0;">
                    {"▲" if gpa_diff >= 0 else "▼"} {abs(gpa_diff):.3f} مقارنة بالمعدل السابق ({pre_gpa:.2f})
                </p>
            </div>
        """, unsafe_allow_html=True)
        
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("#### 🔥 مستوى خطر الاحتراق النفسي (Burnout):")
        
        if pred_burnout == 'High':
            st.markdown('<div class="burnout-badge burnout-high">🚨 خطر مرتفع جداً (High Burnout Risk)</div>', unsafe_allow_html=True)
        elif pred_burnout == 'Medium':
            st.markdown('<div class="burnout-badge burnout-med">⚠️ خطر متوسط (Medium Burnout Risk)</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="burnout-badge burnout-low">✅ مستوى آمن ومستقر (Low Burnout Risk)</div>', unsafe_allow_html=True)
            
        st.markdown("---")
        st.markdown("#### 💡 توصيات المستشار الذكي:")
        
        if weekly_ai_hours > trad_study_hours:
            st.warning("⚠️ **تنسيق الوقت:** ساعات استخدام الـ AI تتجاوز المذاكرة التقليدية. ارفع ساعات الحل اليدوي لضمان الفهم العميق.")
        else:
            st.success("✅ **توازن ممتاز:** توزيعك بين المذاكرة الذكية والتقليدية متزن جداً.")
            
        if exam_anxiety > 7 or pred_burnout == 'High':
            st.error("🛑 **تنبيه الصحة النفسية:** توترك مرتفع. جرب أسلوب البومودورو وخذ فترات راحة لتفادي انخفاض الأداء.")
            
        if skill == 'Beginner':
            st.info("💡 **تطوير مهارة:** تعلم الـ Prompt Engineering المتقدم سيوفر عليك 40% من الوقت في البحث.")
            
    st.markdown('</div>', unsafe_allow_html=True)
