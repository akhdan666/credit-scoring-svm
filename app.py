import sklearn
import joblib
import pandas
import streamlit

st.set_page_config(page_title="Credit Scoring DSS", layout="wide")

st.title("🏦 Decision Support System: Evaluasi Risiko Kredit")
st.markdown("Aplikasi Machine Learning berbasis Support Vector Machine (SVM) untuk memprediksi kelayakan kredit nasabah.")

@st.cache_resource
def load_model():
    return joblib.load('svm_credit_model.pkl')

model = load_model()

st.subheader("Unggah Data Nasabah Baru (Format CSV)")
st.info("Catatan: Format kolom CSV harus sama persis dengan fitur numerik dan kategorikal dataset German Credit.")

uploaded_file = st.file_uploader("Pilih file CSV", type="csv")

if uploaded_file is not None:
    input_data = pd.read_csv(uploaded_file)
    st.write("Preview Data Input:")
    st.dataframe(input_data.head())
    
    if st.button("Jalankan Analisis Risiko"):
        with st.spinner('Memproses data melalui pipeline SVM...'):
            try:
                predictions = model.predict(input_data)
                probabilities = model.predict_proba(input_data)
                
                input_data['Prediction_Class'] = predictions
                input_data['Risk_Status'] = input_data['Prediction_Class'].apply(lambda x: 'Good Risk' if x == 1 else 'Bad Risk')
                input_data['Confidence_Score'] = probabilities.max(axis=1) * 100
                
                st.success("Analisis Selesai!")
                
                st.subheader("Hasil Keputusan Kredit")
                result_df = input_data[['Risk_Status', 'Confidence_Score']]
                st.dataframe(result_df.style.applymap(
                    lambda x: 'background-color: #d4edda; color: green;' if x == 'Good Risk' else 'background-color: #f8d7da; color: red;',
                    subset=['Risk_Status']
                ))
            except Exception as e:
                st.error(f"Terjadi kesalahan saat memproses data. Pastikan format kolom sesuai dengan data latih. Detail error: {e}")
else:
    st.warning("Silakan unggah file CSV untuk memulai prediksi.")
