import sklearn
import joblib
import pandas as pd
import streamlit as st

# 1. Konfigurasi Sistem Utama
st.set_page_config(
    page_title="Credit Risk Analytics",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# 2. Injeksi CSS Modern (Pendekatan SaaS UI)
st.markdown("""
    <style>
    /* Reset padding bawaan Streamlit */
    .block-container {
        padding-top: 2rem;
        max-width: 95%;
    }
    
    /* Tipografi Header Profesional */
    h1 {
        font-weight: 600;
        color: #F8FAFC;
        letter-spacing: -0.05rem;
        font-size: 2.2rem !important;
        margin-bottom: 0rem !important;
    }
    
    /* Penyesuaian Kotak Informasi */
    div.stAlert {
        background-color: rgba(14, 165, 233, 0.05) !important;
        border: 1px solid rgba(14, 165, 233, 0.2) !important;
        color: #38BDF8 !important;
        border-radius: 8px;
    }

    /* Penyesuaian Tombol Eksekusi */
    .stButton>button {
        background-color: #2563EB;
        color: white;
        border: none;
        border-radius: 6px;
        font-weight: 500;
        padding: 0.5rem 2rem;
        transition: all 0.2s ease;
    }
    .stButton>button:hover {
        background-color: #1D4ED8;
        box-shadow: 0 4px 12px rgba(37, 99, 235, 0.2);
    }

    /* Penghapusan Elemen Bawaan Streamlit */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    </style>
""", unsafe_allow_html=True)

# 3. Header Eksekutif
st.markdown("<h1>Data-Driven Credit Risk Assessment</h1>", unsafe_allow_html=True)
st.markdown("<div style='height: 2px; background: linear-gradient(90deg, #2563EB 0%, transparent 100%); margin-top: 10px; margin-bottom: 30px;'></div>", unsafe_allow_html=True)

# Memuat Model Evaluasi
@st.cache_resource
def load_model():
    return joblib.load('svm_credit_model.pkl')

try:
    model = load_model()
except Exception:
    st.error("Kegagalan Sistem: Model SVM (svm_credit_model.pkl) tidak ditemukan di direktori aktif.")
    st.stop()

# 4. Antarmuka Operasional (Grid Asimetris)
col1, col2 = st.columns([7, 3], gap="large")

with col2:
    st.info("📌 **Protokol Data Input**\nPastikan arsitektur matriks CSV identik dengan dataset latih awal untuk mencegah kegagalan komputasi.")

with col1:
    uploaded_file = st.file_uploader("Unggah Berkas Ekstraksi Profil Nasabah (CSV)", type="csv")

if uploaded_file is not None:
    input_data = pd.read_csv(uploaded_file)
    
    st.markdown("<h3 style='color: #E2E8F0; font-size: 1.2rem; margin-top: 20px; font-weight: 500;'>Pratinjau Matriks Data Lintas Sektoral</h3>", unsafe_allow_html=True)
    st.dataframe(input_data.head(5), use_container_width=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    if st.button("Jalankan Mesin Inferensi Risiko", use_container_width=True):
        with st.spinner('Menganalisis probabilitas kelayakan kredit...'):
            try:
                # Eksekusi Prediksi
                predictions = model.predict(input_data)
                probabilities = model.predict_proba(input_data)
                
                # Pemetaan Hasil
                input_data['Status_Risiko'] = ['Low Risk' if x == 1 else 'High Risk' for x in predictions]
                input_data['Confidence (%)'] = probabilities.max(axis=1) * 100
                
                # 5. Dasbor Hasil Terpusat
                st.markdown("<div style='height: 1px; background: #334155; margin: 40px 0px 30px 0px;'></div>", unsafe_allow_html=True)
                st.markdown("<h2 style='color: #F8FAFC; font-size: 1.8rem; margin-bottom: 20px;'>Laporan Eksekutif Portofolio</h2>", unsafe_allow_html=True)
                
                # Kalkulasi Metrik
                total_data = len(input_data)
                low_risk_count = (input_data['Status_Risiko'] == 'Low Risk').sum()
                high_risk_count = (input_data['Status_Risiko'] == 'High Risk').sum()
                
                m1, m2, m3 = st.columns(3)
                m1.metric("Total Profil Dievaluasi", f"{total_data}")
                m2.metric("Kelayakan Tinggi (Low Risk)", f"{low_risk_count}", delta="Aman", delta_color="normal")
                m3.metric("Ancaman Kredit (High Risk)", f"{high_risk_count}", delta="Kritis", delta_color="inverse")
                
                st.markdown("<h3 style='color: #E2E8F0; font-size: 1.2rem; margin-top: 30px; font-weight: 500;'>Distribusi Analisis Keputusan Individu</h3>", unsafe_allow_html=True)
                
                # Pewarnaan Tabel Hasil
                def highlight_risk(val):
                    color = '#10B981' if val == 'Low Risk' else '#EF4444'
                    return f'color: {color}; font-weight: 600;'
                    
                st.dataframe(
                    input_data[['Status_Risiko', 'Confidence (%)']].style.map(highlight_risk, subset=['Status_Risiko']),
                    use_container_width=True
                )
                
            except Exception as e:
                st.error(f"Terjadi deviasi struktural pada data input: {e}")
else:
    # State kosong yang dirancang khusus (Empty State Design)
    st.markdown("""
        <div style='text-align: center; color: #64748B; margin-top: 40px; padding: 50px; border: 1px dashed #334155; border-radius: 8px;'>
            Sistem inferensi dalam mode siaga.<br>Menunggu suplai data profil nasabah untuk dianalisis.
        </div>
    """, unsafe_allow_html=True)
