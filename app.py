import sklearn
import joblib
import pandas as pd
import streamlit as st

# 1. Konfigurasi Halaman (Harus Berada di Baris Pertama)
st.set_page_config(page_title="Executive DSS", page_icon="🏦", layout="wide", initial_sidebar_state="collapsed")

# 2. Injeksi CSS Kustom untuk Tampilan Profesional
st.markdown("""
    <style>
    /* Menyembunyikan elemen default Streamlit yang tidak profesional */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}

    /* Modifikasi skema warna background dan teks */
    .stApp {
        background-color: #0B1120;
        color: #F8FAFC;
    }

    /* Styling Header */
    h1, h2, h3 {
        color: #38BDF8 !important;
        font-weight: 600 !important;
    }

    /* Styling Kotak Info */
    .stAlert {
        background-color: #1E293B;
        color: #E2E8F0;
        border-left: 4px solid #38BDF8;
    }

    /* Styling Tombol Utama */
    .stButton>button {
        background-color: #2563EB;
        color: white;
        border: none;
        border-radius: 6px;
        padding: 0.5rem 2rem;
        font-weight: bold;
        transition: all 0.3s ease;
        width: 100%;
    }
    .stButton>button:hover {
        background-color: #1D4ED8;
        border-color: #38BDF8;
    }

    /* Styling Angka Metrik */
    [data-testid="stMetricValue"] {
        font-size: 2.5rem !important;
        color: #F8FAFC !important;
    }
    </style>
""", unsafe_allow_html=True)

# 3. Header Utama
st.title("🏦 Executive Dashboard: Credit Risk Assessment")
st.markdown("Sistem Pendukung Keputusan Berbasis *Support Vector Machine* (SVM)")
st.markdown("---")

# 4. Pemuatan Model
@st.cache_resource
def load_model():
    return joblib.load('svm_credit_model.pkl')

try:
    model = load_model()
except Exception as e:
    st.error("Sistem gagal memuat model pipeline. Pastikan 'svm_credit_model.pkl' tersedia.")
    st.stop()

# 5. Restrukturisasi UI Menggunakan Tab
tab1, tab2 = st.tabs(["📂 1. Data Input & Pratinjau", "📊 2. Analisis & Dasbor Eksekutif"])

with tab1:
    col1, col2 = st.columns([2, 1])
    with col1:
        uploaded_file = st.file_uploader("Unggah Dataset Nasabah Baru (Format CSV)", type="csv")
    with col2:
        st.info("**Spesifikasi Teknis:**\nFormat dan nama kolom harus identik dengan dataset latih German Credit untuk menghindari kegagalan komputasi.")

    if uploaded_file is not None:
        input_data = pd.read_csv(uploaded_file)
        st.markdown("### 📋 Pratinjau Data Masukan")
        st.dataframe(input_data.head(10), use_container_width=True)

with tab2:
    if uploaded_file is None:
        st.warning("Akses ditolak. Anda harus mengunggah data pada Tab 'Data Input & Pratinjau' terlebih dahulu.")
    else:
        st.markdown("### Eksekusi Model Machine Learning")
        if st.button("🚀 Jalankan Analisis Risiko Kredit"):
            with st.spinner('Memproses perhitungan pipeline SVM...'):
                try:
                    # Proses Prediksi
                    predictions = model.predict(input_data)
                    
                    # Validasi dukungan probabilitas pada model SVM
                    if hasattr(model, "predict_proba"):
                        probabilities = model.predict_proba(input_data)
                        input_data['Confidence_Score'] = probabilities.max(axis=1) * 100
                    else:
                        input_data['Confidence_Score'] = 100.0
                    
                    input_data['Prediction_Class'] = predictions
                    input_data['Risk_Status'] = input_data['Prediction_Class'].apply(lambda x: 'Good Risk' if x == 1 else 'Bad Risk')
                    
                    # Kalkulasi Metrik Dasbor
                    total_nasabah = len(input_data)
                    good_risk_count = (input_data['Risk_Status'] == 'Good Risk').sum()
                    bad_risk_count = (input_data['Risk_Status'] == 'Bad Risk').sum()

                    # Render Metrik Eksekutif
                    st.markdown("---")
                    m1, m2, m3 = st.columns(3)
                    m1.metric("Total Pengajuan", f"{total_nasabah}")
                    m2.metric("Disetujui (Good Risk)", f"{good_risk_count}", delta="Aman", delta_color="normal")
                    m3.metric("Ditolak (Bad Risk)", f"{bad_risk_count}", delta="Risiko Tinggi", delta_color="inverse")
                    
                    # Render Tabel Hasil
                    st.markdown("### 📑 Rincian Status Keputusan")
                    result_cols = ['Risk_Status']
                    if 'Confidence_Score' in input_data.columns:
                        result_cols.append('Confidence_Score')
                    
                    st.dataframe(
                        input_data[result_cols].style.map(
                            lambda x: 'background-color: rgba(16, 185, 129, 0.15); color: #10B981; font-weight: 600;' if x == 'Good Risk' 
                            else ('background-color: rgba(239, 68, 68, 0.15); color: #EF4444; font-weight: 600;' if x == 'Bad Risk' else ''),
                            subset=['Risk_Status']
                        ),
                        use_container_width=True
                    )
                except Exception as e:
                    st.error(f"Kegagalan Sistem Terdeteksi: {e}\nPeriksa kembali apakah Anda mengekspor model lengkap dengan Pipeline (termasuk preprocessor) atau hanya estimator SVM-nya saja.")
