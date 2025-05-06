# SETIA (Sentiment and Topic Insights Analysis)

## Overview
**SETIA** (Sentiment and Topic Insights Analysis) adalah sebuah dashboard interaktif yang dirancang untuk menganalisis ulasan pasien di RSU Queen Latifa. Sistem ini mengintegrasikan analisis sentimen dan topik untuk memberikan wawasan mendalam mengenai kepuasan dan keluhan pasien.

## Features
- **Patient Insight Explorer**: Dashboard untuk eksplorasi sentimen dan topik dalam ulasan pasien.
- **Sentiment Analysis**: Klasifikasi sentimen (Positif, Negatif, Netral) menggunakan model **SVM**.
- **Topic Modeling**: Identifikasi topik utama dari ulasan menggunakan **LDA**.
- **Data Segmentation**: Pemisahan ulasan berdasarkan lokasi (Yogyakarta & Kulon Progo) dan sentimen (Positif & Negatif).
- **Visualizations**:
  - **Pie Chart**: Distribusi sentimen asli dan prediksi.
  - **Word Cloud**: Kata-kata yang sering muncul dalam ulasan positif dan negatif.
  - **Interactive Table**: Menampilkan ulasan dengan sentimen dan topik terkait.
- **Download Processed Data**: Ekspor hasil analisis dalam format CSV.

## Installation
### Prerequisites
- Python 3.8+
- pip
- Virtual Environment (opsional)

### Clone Repository
```sh
git clone https://github.com/username/SETIA.git
cd SETIA
```

### Install Dependencies
```sh
pip install -r requirements.txt
```

### Run the Application
```sh
python app.py
```
Akses dashboard di `http://localhost:5000`.

## Dataset
- Ulasan pasien diperoleh dari Google Maps RSU Queen Latifa (Yogyakarta & Kulon Progo).
- Data telah melalui proses pembersihan dan pemrosesan teks sebelum dianalisis.

## Folder Structure
```
SETIA/
│── app.py                 # Main application file
│── requirements.txt       # Dependencies
│── static/                # Static files (CSS, JS)
│── templates/             # HTML templates
│── data/                  # Raw and processed data
│── models/                # Trained models
│── README.md              # Project documentation
```

## Future Improvements
- Integrasi dengan database untuk pembaruan ulasan secara real-time.
- Model sentimen berbasis deep learning untuk meningkatkan akurasi.
- Dashboard berbasis **Streamlit** untuk tampilan lebih user-friendly.

## Contributors
- **Bagas Dwi Santosa** - [LinkedIn](https://www.linkedin.com/in/bagas-dwi-santosa)

## License
This project is licensed under the MIT License - see the LICENSE file for details.

