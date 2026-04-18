import streamlit as st
import pandas as pd
import numpy as np
import os
from sklearn.ensemble import RandomForestClassifier

# --- 1. ฟังก์ชันสร้างข้อมูลจำลอง (ในกรณีที่ไม่มีไฟล์) ---
def create_mock_data(filename='customer_churn_mock.csv'):
    np.random.seed(42)
    n_samples = 1000
    data = {
        'CustomerID': range(1, n_samples + 1),
        'Age': np.random.randint(18, 70, n_samples),
        'Tenure_Months': np.random.randint(1, 72, n_samples),
        'Monthly_Charge': np.random.uniform(500, 3000, n_samples).round(2),
        'Support_Tickets': np.random.randint(0, 6, n_samples)
    }
    df = pd.DataFrame(data)
    
    churn_prob = (
        (df['Support_Tickets'] * 0.12) + 
        (1 / (df['Tenure_Months'] + 1)) * 0.4 + 
        (df['Monthly_Charge'] / 3000) * 0.1
    )
    df['Churn'] = (np.random.rand(n_samples) < churn_prob).astype(int)
    df.to_csv(filename, index=False)
    return df

# --- 2. การเตรียมข้อมูลและเทรนโมเดล ---
@st.cache_data
def train_model():
    # เช็คว่ามีไฟล์ CSV หรือยัง ถ้ายังไม่มีให้สร้างใหม่ทันที
    if not os.path.exists('customer_churn_mock.csv'):
        df = create_mock_data()
    else:
        df = pd.read_csv('customer_churn_mock.csv')
        
    X = df[['Age', 'Tenure_Months', 'Monthly_Charge', 'Support_Tickets']]
    y = df['Churn']
    
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)
    return model

model = train_model()

# --- 3. การออกแบบส่วนแสดงผล (UI) ---
st.set_page_config(page_title="Customer Churn Prediction", page_icon="📊")

st.title("📊 ระบบคาดการณ์ลูกค้ายกเลิกบริการ")
st.markdown("แอปพลิเคชันสำหรับทีมการตลาดเพื่อ **ประเมินความเสี่ยง** และ **จัดแคมเปญรักษาลูกค้า** เชิงรุก")
st.divider()

# --- 4. แถบเครื่องมือด้านข้าง (Sidebar) สำหรับกรอกข้อมูล ---
st.sidebar.header("📝 กรอกข้อมูลลูกค้าใหม่")

age = st.sidebar.slider("อายุ (ปี)", 18, 70, 35)
tenure = st.sidebar.slider("อายุการใช้งาน (เดือน)", 1, 72, 6)
charge = st.sidebar.number_input("ค่าบริการรายเดือน (บาท)", 500.0, 3000.0, 1500.0)
tickets = st.sidebar.selectbox("จำนวนครั้งที่ติดต่อ Support (ครั้ง)", [0, 1, 2, 3, 4, 5])

# --- 5. การทำนายผล (Prediction) ---
input_data = pd.DataFrame({
    'Age': [age], 
    'Tenure_Months': [tenure], 
    'Monthly_Charge': [charge], 
    'Support_Tickets': [tickets]
})

prediction = model.predict(input_data)[0]
probability = model.predict_proba(input_data)[0][1]

# --- 6. การแสดงผลลัพธ์เชิงธุรกิจ (Business Action) ---
st.subheader("💡 ผลการวิเคราะห์และข้อเสนอแนะทางธุรกิจ")

if prediction == 1:
    st.error(f"⚠️ ความเสี่ยงสูง: ลูกค้ารายนี้มีโอกาสยกเลิกบริการ ({probability:.1%})")
    
    st.markdown("### 🎯 Action Plan แนะนำ:")
    st.info("""
    * **Retention Offer:** ส่ง SMS มอบคูปองส่วนลด 20% สำหรับรอบบิลถัดไป
    * **Proactive Service:** ให้ทีม Customer Success โทรสอบถามปัญหาการใช้งานโดยด่วน
    * **Expected ROI:** ค่าใช้จ่ายในการโทร+ส่วนลด คุ้มค่ากว่าการหาลูกค้าใหม่ 
    """)
else:
    st.success(f"✅ ปลอดภัย: ลูกค้ามีแนวโน้มใช้งานต่อ (ความเสี่ยงเพียง {probability:.1%})")
    
    st.markdown("### 🎯 Action Plan แนะนำ:")
    st.info("""
    * **Relationship Building:** ส่งอีเมลขอบคุณสำหรับการใช้งานอย่างต่อเนื่อง
    * **Cross-sell Opportunity:** เสนอแพ็กเกจเสริม (Add-on) ที่เข้ากับพฤติกรรมการใช้งาน
    """)
