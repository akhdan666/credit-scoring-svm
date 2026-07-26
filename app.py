import sklearn
import joblib
import pandas as pd
import streamlit as st

# Konfigurasi Halaman
st.set_page_config(
    page_title="Credit Scoring DSS - Executive Dashboard",
    page_icon="🏦",
    layout="wide"
)

# Custom Styling untuk Tampilan Lebih Elegan
st.markdown("""
    <style>
    .main {
        background-color: #0e1117;
    }
    .stMetric {
        background-color: #161b22;
        padding: 15px;
        border-radius: 10px;
        border: 1px solid #30363d;
    }
    </style>
""", unsafe_allow_html=True)

# Header Utama
st.markdown("### 🏦 Decision Support System: Evaluasi Risiko Kredit Korporat")
st.markdown("---")

@st.cache_resource
def load_model():
    return joblib.load('svm_credit_model.pkl')

model = load_model()

# Layout Kolom untuk Unggah File
col1, col2 = st.columns([2, 1])
with col1:
    st.subheader("Unggah Data Nasabah Baru")
    uploaded_file = st.file_uploader("Pilih file dataset berformat CSV", type="csv")

with col2:
    st.info("**Panduan Sistem:**\nPastikan kolom CSV sesuai dengan fitur numerik dan kategorikal model German Credit SVM.")

if uploaded_file is not None:
    input_data = pd.read_csv(uploaded_file)
    
    st.markdown("---")
    st.subheader("📋 Pratinjau Data Masukan")
    st.dataframe(input_data.head(), use_container_width=True)
    
    if st.button("🚀 Jalankan Analisis Risiko Kredit", type="primary"):
        with st.spinner('Memproses perhitungan model SVM...'):
            try:
                predictions = model.predict(input_data)
                probabilities = model.predict_proba(input_data)
                
                input_data['Prediction_Class'] = predictions
                input_data['Risk_Status'] = input_data['Prediction_Class'].apply(lambda x: 'Good Risk' if x == 1 else 'Bad Risk')
                input_data['Confidence_Score'] = probabilities.max(axis=1) * 100
                
                # Metrik Ringkasan Hasil
                total_nasabah = len(input_data)
                good_risk_count = (input_data['Risk_Status'] == 'Good Risk').sum()
                bad_risk_count = (input_data['Risk_Status'] == 'Bad Risk').sum()
                
                st.markdown("---")
                st.subheader("📊 Ringkasan Eksekutif Hasil Evaluasi")
                
                m1, m2, m3 = st.columns(3)
                m1.metric("Total Data Dievaluasi", f"{total_nasabah} Nasabah")
                m2.metric("Risiko Rendah (Good)", f"{good_risk_count} Nasabah", delta="Aman")
                m3.metric("Risiko Tinggi (Bad)", f"{bad_risk_count} Nasabah", delta="Perhatian", delta_color="inverse")
                
                st.markdown("### 📑 Hasil Keputusan Terperinci")
                result_df = input_data[['Risk_Status', 'Confidence_Score']]
                
                st.dataframe(
                    result_df.style.map(
                        lambda x: 'background-color: rgba(40, 167, 69, 0.2); color: #28a745; font-weight: bold;' if x == 'Good Risk' 
                        else ('background-color: rgba(220, 53, 69, 0.2); color: #dc3545; font-weight: bold;' if x == 'Bad Risk' else ''),
                        subset=['Risk_Status']
                    ),
                    use_container_width=True
                )
            except Exception as e:
                st.error(f"Terjadi kesalahan komputasi: {e}")
else:
    st.markdown("---")
    st.warning("Silakan unggah file data CSV terlebih dahulu untuk mengaktifkan panel analisis.")
