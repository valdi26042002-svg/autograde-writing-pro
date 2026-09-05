import streamlit as st
import textstat
import pypdf
import docx

# Konfigurasi Halaman
st.set_page_config(
    page_title="AutoGrade Writing Pro",
    page_icon="✍️",
    layout="wide"
)

# Custom CSS untuk tampilan yang lebih profesional dan bersih
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        color: #1E3A8A;
        font-weight: 700;
        margin-bottom: 0px;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #4B5563;
        margin-bottom: 25px;
    }
    .metric-card {
        background-color: #F3F4F6;
        padding: 15px;
        border-radius: 8px;
        border-left: 5px solid #3B82F6;
    }
    </style>
""", unsafe_allow_html=True)

# Judul Aplikasi
st.markdown('<p class="main-header">✍️ AutoGrade Writing Pro</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Aplikasi Penilai dan Analisis Esai Bahasa Inggris Otomatis untuk Tingkat Menengah (SMA / Kelas 11)</p>', unsafe_allow_html=True)

# Sidebar untuk Pengaturan & Navigasi
st.sidebar.header("⚙️ Pengaturan Penilaian")
essay_level = st.sidebar.selectbox(
    "Target Level Pelajar:",
    ["SMA / High School (Grade 11-12)", "Universitas / College", "Pemula / Beginner"]
)

scoring_focus = st.sidebar.selectbox(
    "Fokus Analisis:",
    ["Komprehensif (Struktur & Panjang)", "Kompleksitas Kalimat", "Keterbacaan (Readability)"]
)

st.sidebar.markdown("---")
st.sidebar.info(
    "**Fitur Didukung:**\n"
    "• Unggah Dokumen (.pdf, .docx, .txt)\n"
    "• Input Teks Manual\n"
    "• Analisis Keterbacaan & Statistik\n"
    "• Umpan Balik Otomatis\n"
    "• Ekspor Laporan Analisis"
)

# Pilihan Metode Input Esai
st.subheader("📁 Pilih Metode Masukan Esai")
input_method = st.radio(
    "Bagaimana Anda ingin memasukkan teks esai?", 
    ["Unggah File (PDF / Word / TXT)", "Ketik / Tempel Teks Manual (Paste)"],
    horizontal=True
)

essay_input = ""

if input_method == "Unggah File (PDF / Word / TXT)":
    uploaded_file = st.file_uploader("Pilih file dokumen esai Anda:", type=["txt", "pdf", "docx"])
    
    if uploaded_file is not None:
        file_extension = uploaded_file.name.split(".")[-1].lower()
        try:
            if file_extension == "txt":
                essay_input = uploaded_file.read().decode("utf-8")
            elif file_extension == "pdf":
                reader = pypdf.PdfReader(uploaded_file)
                for page in reader.pages:
                    extracted = page.extract_text()
                    if extracted:
                        essay_input += extracted + "\n"
            elif file_extension == "docx":
                doc = docx.Document(uploaded_file)
                for para in doc.paragraphs:
                    essay_input += para.text + "\n"
            
            if essay_input.strip():
                st.success(f"Berhasil membaca file: **{uploaded_file.name}** ({len(essay_input.split())} kata)")
            else:
                st.warning("File berhasil diunggah, namun teks tidak terdeteksi atau kosong.")
        except Exception as e:
            st.error(f"Terjadi kesalahan saat membaca file: {e}")

else:
    essay_input = st.text_area(
        "Tempel (paste) teks esai bahasa Inggris di sini...", 
        height=250,
        placeholder="Type or paste your English essay here..."
    )

# Tombol Analisis
st.markdown("")
if st.button("🚀 Jalankan Analisis Esai", type="primary", use_container_width=True):
    if not essay_input or essay_input.strip() == "":
        st.warning("Mohon unggah file atau masukkan teks esai terlebih dahulu.")
    else:
        # Perhitungan Statistik Dasar
        word_count = len(essay_input.split())
        char_count = len(essay_input)
        sentence_count = textstat.sentence_count(essay_input)
        
        readability_score = textstat.flesch_reading_ease(essay_input)
        grade_level = textstat.text_standard(essay_input, float_output=False)

        # Menampilkan Metrik Utama
        st.markdown("---")
        st.subheader("📊 Hasil Analisis Teks")
        
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Jumlah Kata", f"{word_count} kata")
        col2.metric("Jumlah Kalimat", f"{sentence_count} kalimat")
        col3.metric("Skor Keterbacaan", f"{readability_score} / 100")
        col4.metric("Estimasi Level", grade_level)

        # Umpan Balik / Feedback Berdasarkan Standar Kelas 11 SMA
        st.markdown("---")
        st.subheader("💡 Umpan Balik & Rekomendasi Perbaikan")
        
        feedback_list = []
        
        # Analisis Panjang Teks
        if word_count < 250:
            feedback_list.append("⚠️ **Panjang Esai:** Esai Anda terlalu pendek (kurang dari 250 kata). Untuk level kelas 11, usahakan menulis minimal 250–400 kata agar argumen dapat dikembangkan dengan matang.")
        elif word_count > 800:
            feedback_list.append("ℹ️ **Panjang Esai:** Esai cukup panjang. Pastikan tidak ada pengulangan ide (*repetitive arguments*) yang tidak perlu.")
        else:
            feedback_list.append("✅ **Panjang Esai:** Panjang esai sudah ideal dan memenuhi standar tugas tingkat menengah.")

        # Analisis Kompleksitas Kalimat
        if sentence_count > 0:
            avg_words_per_sentence = word_count / sentence_count
            if avg_words_per_sentence > 25:
                feedback_list.append("⚠️ **Struktur Kalimat:** Kalimat Anda rata-rata terlalu panjang (lebih dari 25 kata per kalimat). Pertimbangkan untuk memecahnya agar lebih mudah dipahami pembaca.")
            elif avg_words_per_sentence < 10:
                feedback_list.append("⚠️ **Struktur Kalimat:** Kalimat Anda cenderung terlalu pendek dan terkesan terputus-putus. Gunakan *transition words* (misalnya: *furthermore, however, consequently*) untuk merangkai ide.")
            else:
                feedback_list.append("✅ **Struktur Kalimat:** Variasi panjang kalimat sudah seimbang dan mengalir dengan baik.")

        # Analisis Keterbacaan (Flesch Reading Ease)
        if readability_score < 30:
            feedback_list.append("📖 **Keterbacaan:** Teks sangat sulit dibaca (tingkat akademik tinggi/universitas). Pastikan kosa kata dan struktur tidak terlalu rumit untuk pembaca tingkat sekolah.")
        elif readability_score > 70:
            feedback_list.append("📖 **Keterbacaan:** Teks sangat mudah dibaca. Untuk level kelas 11 SMA, Anda bisa meningkatkan penggunaan kosa kata akademik (*advanced vocabulary*) agar esai lebih berbobot.")
        else:
            feedback_list.append("✅ **Keterbacaan:** Tingkat keterbacaan teks sudah sangat sesuai untuk pembaca tingkat sekolah menengah (SMA).")

        # Tampilkan Umpan Balik ke Layar
        for item in feedback_list:
            st.markdown(f"- {item}")

        # Pratinjau Teks & Tombol Unduh Laporan
        st.markdown("---")
        col_preview, col_download = st.columns([2, 1])
        
        with col_preview:
            with st.expander("🔍 Pratinjau Teks Esai yang Dianalisis"):
                st.write(essay_input)
                
        with col_download:
            report_text = f"""=========================================
LAPORAN ANALISIS AUTO-GRADE WRITING PRO
=========================================
Target Level: {essay_level}
Fokus Analisis: {scoring_focus}

[STATISTIK UTAMA]
- Jumlah Kata: {word_count}
- Jumlah Kalimat: {sentence_count}
- Skor Keterbacaan (Flesch): {readability_score} / 100
- Estimasi Level: {grade_level}

[UMPAN BALIK OTOMATIS]
""" + "\n".join([f"• {f.replace('**', '').replace('⚠️ ', '').replace('✅ ', '').replace('📖 ', '').replace('ℹ️ ', '')}" for f in feedback_list])

            st.download_button(
                label="📥 Unduh Laporan (.txt)",
                data=report_text,
                file_name="AutoGrade_Essay_Report.txt",
                mime="text/plain",
                use_container_width=True
            )
