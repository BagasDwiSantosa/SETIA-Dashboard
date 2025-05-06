from flask import Flask, render_template, request, jsonify, send_from_directory
import os
import pandas as pd
import numpy as np
import joblib
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from prep import *
from datetime import datetime
from sklearn.decomposition import LatentDirichletAllocation
from gensim.models import CoherenceModel, LdaModel
from gensim.corpora import Dictionary
import math
import plotly.graph_objs as go
from plotly.subplots import make_subplots
import plotly
import json


app = Flask(__name__)

with open("./static/model/tfidf_vectorizer.pkl", "rb") as file:
    vectorizer = joblib.load(file)

svm_model = joblib.load("./static/model/svm_model.pkl")

UPLOAD_FOLDER = 'uploads'
PROCESSED_FOLDER = 'processed'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(PROCESSED_FOLDER, exist_ok=True)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/Data-Reviews-GMaps', methods=['GET'])
def Data_Reviews_GMaps():
    # Load dataset
    data_all = pd.read_csv('./static/data/all_data_QL.csv')
    reviews = data_all[['name', 'review', 'rating', 'rsu']]

    count_yogyakarta = reviews[reviews['rsu'].str.contains('Yogyakarta', case=False)].shape[0]
    count_kulonprogo = reviews[reviews['rsu'].str.contains('KulonProgo', case=False)].shape[0]

    # Baca filter dari query parameters
    location_filter = request.args.get('locationFilter', 'all')
    rating_filter = request.args.get('ratingFilter', 'all')

    # Terapkan filter lokasi
    if location_filter != 'all':
        reviews = reviews[reviews['rsu'].str.contains(location_filter, case=False)]

    # Terapkan filter rating
    if rating_filter != 'all':
        reviews = reviews[reviews['rating'] == float(rating_filter)]

    # Pagination
    per_page = 10
    page = request.args.get('page', 1, type=int)

    start = (page - 1) * per_page
    end = start + per_page

    paginated_reviews = reviews.iloc[start:end]

    # Total pages
    total_reviews = len(reviews)
    total_pages = (total_reviews + per_page - 1) // per_page

    # Rentang halaman untuk pagination
    num_range = 2
    start_range = max(1, page - num_range)
    end_range = min(total_pages, page + num_range)
    page_range = list(range(start_range, end_range + 1))

    return render_template(
        'data.html',
        reviews=paginated_reviews.to_dict(orient='records'),
        page=page,
        total_pages=total_pages,
        page_range=page_range,
        location_filter=location_filter,
        rating_filter=rating_filter,
        count_yogyakarta=count_yogyakarta,
        count_kulonprogo=count_kulonprogo,
    )

@app.route('/Case-Folding', methods=['GET'])
def case_folding():
    data_all = pd.read_excel('./static/data/Sentimen-Done-rev-3.xlsx')
    reviews = data_all[['review_text', 'text_casefoldingText', 'location']]

    # Baca filter dari query parameters
    location_filter = request.args.get('locationFilter', 'all')

    # Terapkan filter lokasi
    if location_filter != 'all':
        reviews = reviews[reviews['location'].str.contains(location_filter, case=False)]

    # Pagination
    per_page = 10
    page = request.args.get('page', 1, type=int)

    start = (page - 1) * per_page
    end = start + per_page

    paginated_reviews = reviews.iloc[start:end]

    # Total pages
    total_reviews = len(reviews)
    total_pages = (total_reviews + per_page - 1) // per_page

    # Rentang halaman untuk pagination
    num_range = 2
    start_range = max(1, page - num_range)
    end_range = min(total_pages, page + num_range)
    page_range = list(range(start_range, end_range + 1))

    return render_template(
        'case-folding.html',
        reviews=paginated_reviews.to_dict(orient='records'),
        page=page,
        total_pages=total_pages,
        page_range=page_range,
        location_filter=location_filter
    )

@app.route('/Data-Cleaning', methods=['GET'])
def data_cleaning():
    data_all = pd.read_excel('./static/data/Sentimen-Done-rev-3.xlsx')
    reviews = data_all[['text_casefoldingText', 'text_clean', 'location']]

    # Baca filter dari query parameters
    location_filter = request.args.get('locationFilter', 'all')

    # Terapkan filter lokasi
    if location_filter != 'all':
        reviews = reviews[reviews['location'].str.contains(location_filter, case=False)]

    # Pagination
    per_page = 10
    page = request.args.get('page', 1, type=int)

    start = (page - 1) * per_page
    end = start + per_page

    paginated_reviews = reviews.iloc[start:end]

    # Total pages
    total_reviews = len(reviews)
    total_pages = (total_reviews + per_page - 1) // per_page

    # Rentang halaman untuk pagination
    num_range = 2
    start_range = max(1, page - num_range)
    end_range = min(total_pages, page + num_range)
    page_range = list(range(start_range, end_range + 1))

    return render_template(
        'cleaning.html',
        reviews=paginated_reviews.to_dict(orient='records'),
        page=page,
        total_pages=total_pages,
        page_range=page_range,
        location_filter=location_filter
    )

@app.route('/Normalization', methods=['GET'])
def normalization():
    data_all = pd.read_excel('./static/data/Sentimen-Done-rev-3.xlsx')
    reviews = data_all[['text_clean', 'text_slang', 'location']]

    # Baca filter dari query parameters
    location_filter = request.args.get('locationFilter', 'all')

    # Terapkan filter lokasi
    if location_filter != 'all':
        reviews = reviews[reviews['location'].str.contains(location_filter, case=False)]

    # Pagination
    per_page = 10
    page = request.args.get('page', 1, type=int)

    start = (page - 1) * per_page
    end = start + per_page

    paginated_reviews = reviews.iloc[start:end]

    # Total pages
    total_reviews = len(reviews)
    total_pages = (total_reviews + per_page - 1) // per_page

    # Rentang halaman untuk pagination
    num_range = 2
    start_range = max(1, page - num_range)
    end_range = min(total_pages, page + num_range)
    page_range = list(range(start_range, end_range + 1))

    return render_template(
        'normalization.html',
        reviews=paginated_reviews.to_dict(orient='records'),
        page=page,
        total_pages=total_pages,
        page_range=page_range,
        location_filter=location_filter
    )

@app.route('/Tokenizing', methods=['GET'])
def tokenizing():
    data_all = pd.read_excel('./static/data/Sentimen-Done-rev-3.xlsx')
    reviews = data_all[['text_slang', 'text_token', 'location']]

    # Baca filter dari query parameters
    location_filter = request.args.get('locationFilter', 'all')

    # Terapkan filter lokasi
    if location_filter != 'all':
        reviews = reviews[reviews['location'].str.contains(location_filter, case=False)]

    # Pagination
    per_page = 10
    page = request.args.get('page', 1, type=int)

    start = (page - 1) * per_page
    end = start + per_page

    paginated_reviews = reviews.iloc[start:end]

    # Total pages
    total_reviews = len(reviews)
    total_pages = (total_reviews + per_page - 1) // per_page

    # Rentang halaman untuk pagination
    num_range = 2
    start_range = max(1, page - num_range)
    end_range = min(total_pages, page + num_range)
    page_range = list(range(start_range, end_range + 1))

    return render_template(
        'tokenizing.html',
        reviews=paginated_reviews.to_dict(orient='records'),
        page=page,
        total_pages=total_pages,
        page_range=page_range,
        location_filter=location_filter
    )

@app.route('/Filtering', methods=['GET'])
def filtering():
    data_all = pd.read_excel('./static/data/Sentimen-Done-rev-3.xlsx')
    reviews = data_all[['text_token', 'text_stopwords', 'location']]

    # Baca filter dari query parameters
    location_filter = request.args.get('locationFilter', 'all')

    # Terapkan filter lokasi
    if location_filter != 'all':
        reviews = reviews[reviews['location'].str.contains(location_filter, case=False)]

    # Pagination
    per_page = 10
    page = request.args.get('page', 1, type=int)

    start = (page - 1) * per_page
    end = start + per_page

    paginated_reviews = reviews.iloc[start:end]

    # Total pages
    total_reviews = len(reviews)
    total_pages = (total_reviews + per_page - 1) // per_page

    # Rentang halaman untuk pagination
    num_range = 2
    start_range = max(1, page - num_range)
    end_range = min(total_pages, page + num_range)
    page_range = list(range(start_range, end_range + 1))

    return render_template(
        'filtering.html',
        reviews=paginated_reviews.to_dict(orient='records'),
        page=page,
        total_pages=total_pages,
        page_range=page_range,
        location_filter=location_filter
    )

@app.route('/Stemming', methods=['GET'])
def stemming():
    data_all = pd.read_excel('./static/data/Sentimen-Done-rev-3.xlsx')
    reviews = data_all[['text_stopwords', 'text_lemmatized', 'location']]

    # Baca filter dari query parameters
    location_filter = request.args.get('locationFilter', 'all')

    # Terapkan filter lokasi
    if location_filter != 'all':
        reviews = reviews[reviews['location'].str.contains(location_filter, case=False)]

    # Pagination
    per_page = 10
    page = request.args.get('page', 1, type=int)

    start = (page - 1) * per_page
    end = start + per_page

    paginated_reviews = reviews.iloc[start:end]

    # Total pages
    total_reviews = len(reviews)
    total_pages = (total_reviews + per_page - 1) // per_page

    # Rentang halaman untuk pagination
    num_range = 2
    start_range = max(1, page - num_range)
    end_range = min(total_pages, page + num_range)
    page_range = list(range(start_range, end_range + 1))

    return render_template(
        'stemming.html',
        reviews=paginated_reviews.to_dict(orient='records'),
        page=page,
        total_pages=total_pages,
        page_range=page_range,
        location_filter=location_filter
    )

@app.route('/Text-Validation', methods=['GET'])
def text_validation():
    data_all = pd.read_excel('./static/data/Sentimen-Done-rev-3.xlsx')
    reviews = data_all[['text_lemmatized', 'text_done', 'location']]

    # Baca filter dari query parameters
    location_filter = request.args.get('locationFilter', 'all')

    # Terapkan filter lokasi
    if location_filter != 'all':
        reviews = reviews[reviews['location'].str.contains(location_filter, case=False)]

    # Pagination
    per_page = 10
    page = request.args.get('page', 1, type=int)

    start = (page - 1) * per_page
    end = start + per_page

    paginated_reviews = reviews.iloc[start:end]

    # Total pages
    total_reviews = len(reviews)
    total_pages = (total_reviews + per_page - 1) // per_page

    # Rentang halaman untuk pagination
    num_range = 2
    start_range = max(1, page - num_range)
    end_range = min(total_pages, page + num_range)
    page_range = list(range(start_range, end_range + 1))

    return render_template(
        'validation.html',
        reviews=paginated_reviews.to_dict(orient='records'),
        page=page,
        total_pages=total_pages,
        page_range=page_range,
        location_filter=location_filter
    )

@app.route('/Sentimen-Analysis')
def sentimen_analysis():
    return render_template('sentimen-analysis.html')

@app.route('/Sentimen-Analysis-RSU-QL-Yogyakarta', methods=['GET'])
def SA_RSU_QL_Yogyakarta():
    data_QL_1 = pd.read_excel('./static/data/Sentimen-Done-rev-3.xlsx')
    sentimen1 = data_QL_1[['text_done', 'location', 'sentimen', 'Predicted_Label']]
    sentimen1 = sentimen1[sentimen1['location'] == 'Yogyakarta']

    sentimen1['Predicted_Label'] = sentimen1['Predicted_Label'].replace({1: 'Positif', -1: 'Negatif'})

    count_sentimen_positif = sentimen1[sentimen1['sentimen'].str.contains('Positif', case=False)].shape[0]
    count_sentimen_negatif = sentimen1[sentimen1['sentimen'].str.contains('Negatif', case=False)].shape[0]
    count_prediction_positif = sentimen1[sentimen1['Predicted_Label'].str.contains('Positif', case=False)].shape[0]
    count_prediction_negatif = sentimen1[sentimen1['Predicted_Label'].str.contains('Negatif', case=False)].shape[0]

    # Pisahkan ulasan berdasarkan sentimen Positif dan Negatif
    positive_reviews = sentimen1[sentimen1['Predicted_Label'] == 'Positif']
    negative_reviews = sentimen1[sentimen1['Predicted_Label'] == 'Negatif']

    # Membuat model CountVectorizer untuk Positif
    vectorizer_pos = CountVectorizer(stop_words='english', max_features=25)
    X_pos = vectorizer_pos.fit_transform(positive_reviews['text_done'])
    words_pos = vectorizer_pos.get_feature_names_out()
    frequencies_pos = X_pos.sum(axis=0).A1
    word_freq_pos = pd.DataFrame(list(zip(words_pos, frequencies_pos)), columns=['Word', 'Frequency'])
    word_freq_pos = word_freq_pos.sort_values(by='Frequency', ascending=False).head(25)

    # Membuat model CountVectorizer untuk Negatif
    vectorizer_neg = CountVectorizer(stop_words='english', max_features=25)
    X_neg = vectorizer_neg.fit_transform(negative_reviews['text_done'])
    words_neg = vectorizer_neg.get_feature_names_out()
    frequencies_neg = X_neg.sum(axis=0).A1
    word_freq_neg = pd.DataFrame(list(zip(words_neg, frequencies_neg)), columns=['Word', 'Frequency'])
    word_freq_neg = word_freq_neg.sort_values(by='Frequency', ascending=False).head(25)

    # Baca filter dari query parameters
    sentimen_filter = request.args.get('sentimenFilter', 'all')
    predict_filter = request.args.get('predictFilter', 'all')

    # Terapkan filter lokasi
    if sentimen_filter != 'all':
        sentimen1 = sentimen1[sentimen1['sentimen'].str.contains(sentimen_filter, case=False)]

    # Terapkan filter predict
    if predict_filter != 'all':
        sentimen1 = sentimen1[sentimen1['Predicted_Label'].str.contains(sentimen_filter, case=False)]

    # Pagination
    per_page = 10
    page = request.args.get('page', 1, type=int)

    start = (page - 1) * per_page
    end = start + per_page

    paginated_reviews = sentimen1.iloc[start:end]

    # Total pages
    total_reviews = len(sentimen1)
    total_pages = (total_reviews + per_page - 1) // per_page

    # Rentang halaman untuk pagination
    num_range = 2
    start_range = max(1, page - num_range)
    end_range = min(total_pages, page + num_range)
    page_range = list(range(start_range, end_range + 1))

    return render_template(
        'sentimen-analysis-QL-Yogyakarta.html', 
        reviews=paginated_reviews.to_dict(orient='records'),
        word_freq_pos=word_freq_pos.to_dict(orient='records'),
        word_freq_neg=word_freq_neg.to_dict(orient='records'),
        page=page,
        total_pages=total_pages,
        page_range=page_range,
        sentimen_filter=sentimen_filter,
        predict_filter=predict_filter,
        count_sentimen_positif=count_sentimen_positif,
        count_sentimen_negatif=count_sentimen_negatif,
        count_prediction_positif=count_prediction_positif,
        count_prediction_negatif=count_prediction_negatif, 
    )

@app.route('/Sentimen-Analysis-RSU-QL-KulonProgo')
def SA_RSU_QL_KulonProgo():
    data_QL_1 = pd.read_excel('./static/data/Sentimen-Done-rev-3.xlsx')
    sentimen1 = data_QL_1[['text_done', 'location', 'sentimen', 'Predicted_Label']]

    sentimen1['Predicted_Label'] = sentimen1['Predicted_Label'].replace({1: 'Positif', -1: 'Negatif'})
    sentimen1 = sentimen1[sentimen1['location'] == 'KulonProgo']

    count_sentimen_positif = sentimen1[sentimen1['sentimen'].str.contains('Positif', case=False)].shape[0]
    count_sentimen_negatif = sentimen1[sentimen1['sentimen'].str.contains('Negatif', case=False)].shape[0]
    count_prediction_positif = sentimen1[sentimen1['Predicted_Label'].str.contains('Positif', case=False)].shape[0]
    count_prediction_negatif = sentimen1[sentimen1['Predicted_Label'].str.contains('Negatif', case=False)].shape[0]

    # Pisahkan ulasan berdasarkan sentimen Positif dan Negatif
    positive_reviews = sentimen1[sentimen1['Predicted_Label'] == 'Positif']
    negative_reviews = sentimen1[sentimen1['Predicted_Label'] == 'Negatif']

    # Membuat model CountVectorizer untuk Positif
    vectorizer_pos = CountVectorizer(stop_words='english', max_features=25)
    X_pos = vectorizer_pos.fit_transform(positive_reviews['text_done'])
    words_pos = vectorizer_pos.get_feature_names_out()
    frequencies_pos = X_pos.sum(axis=0).A1
    word_freq_pos = pd.DataFrame(list(zip(words_pos, frequencies_pos)), columns=['Word', 'Frequency'])
    word_freq_pos = word_freq_pos.sort_values(by='Frequency', ascending=False).head(25)

    # Membuat model CountVectorizer untuk Negatif
    vectorizer_neg = CountVectorizer(stop_words='english', max_features=25)
    X_neg = vectorizer_neg.fit_transform(negative_reviews['text_done'])
    words_neg = vectorizer_neg.get_feature_names_out()
    frequencies_neg = X_neg.sum(axis=0).A1
    word_freq_neg = pd.DataFrame(list(zip(words_neg, frequencies_neg)), columns=['Word', 'Frequency'])
    word_freq_neg = word_freq_neg.sort_values(by='Frequency', ascending=False).head(25)

    # Baca filter dari query parameters
    sentimen_filter = request.args.get('sentimenFilter', 'all')
    predict_filter = request.args.get('predictFilter', 'all')

    # Terapkan filter lokasi
    if sentimen_filter != 'all':
        sentimen1 = sentimen1[sentimen1['sentimen'].str.contains(sentimen_filter, case=False)]

    # Terapkan filter predict
    if predict_filter != 'all':
        sentimen1 = sentimen1[sentimen1['Predicted_Label'].str.contains(sentimen_filter, case=False)]

    # Pagination
    per_page = 10
    page = request.args.get('page', 1, type=int)

    start = (page - 1) * per_page
    end = start + per_page

    paginated_reviews = sentimen1.iloc[start:end]

    # Total pages
    total_reviews = len(sentimen1)
    total_pages = (total_reviews + per_page - 1) // per_page

    # Rentang halaman untuk pagination
    num_range = 2
    start_range = max(1, page - num_range)
    end_range = min(total_pages, page + num_range)
    page_range = list(range(start_range, end_range + 1))

    return render_template(
        'sentimen-analysis-QL-KulonProgo.html', 
        reviews=paginated_reviews.to_dict(orient='records'),
        word_freq_pos=word_freq_pos.to_dict(orient='records'),
        word_freq_neg=word_freq_neg.to_dict(orient='records'),
        page=page,
        total_pages=total_pages,
        page_range=page_range,
        sentimen_filter=sentimen_filter,
        predict_filter=predict_filter,
        count_sentimen_positif=count_sentimen_positif,
        count_sentimen_negatif=count_sentimen_negatif,
        count_prediction_positif=count_prediction_positif,
        count_prediction_negatif=count_prediction_negatif, 
    )

@app.route('/Topic-Analysis')
def topic_analysis():
    return render_template('topic-analysis.html')

def visualize_lda_topics(lda_topics, topic_titles=None, colors=None, top_n=10, word_font_size=10, max_cols=4, height_per_row=350):
    if topic_titles is None:
        topic_titles = [f"Topik {i+1}" for i in range(len(lda_topics))]
    if colors is None:
        default_colors = ['cornflowerblue', 'salmon', 'mediumseagreen', 'mediumpurple', 'goldenrod', 'lightseagreen', 'tomato', 'slateblue']
        colors = [default_colors[i % len(default_colors)] for i in range(len(lda_topics))]

    total_topics = len(lda_topics)
    rows = math.ceil(total_topics / max_cols)
    cols = min(max_cols, total_topics)

    fig = make_subplots(
        rows=rows, cols=cols,
        subplot_titles=topic_titles,
        horizontal_spacing=0.1,
        vertical_spacing=0.07
    )

    for i, topic in enumerate(lda_topics):
        sorted_topic = sorted(topic, key=lambda x: -x[1])[:top_n]
        words, weights = zip(*sorted_topic)
        row = i // max_cols + 1
        col = i % max_cols + 1
        fig.add_trace(
            go.Bar(
                x=weights,
                y=words,
                orientation='h',
                marker=dict(color=colors[i]),
            ),
            row=row, col=col
        )

    for i in range(len(lda_topics)):
        row = i // max_cols + 1
        col = i % max_cols + 1
        fig.update_yaxes(
            tickfont=dict(size=word_font_size, color='white'),
            row=row, col=col
        )

    fig.update_layout(
        height=rows * height_per_row,  # <--- Di sini height dinamis
        title_font=dict(size=12),
        margin=dict(l=10, r=10, t=40, b=10),
        paper_bgcolor='rgba(0,0,0,0)',  # transparan background luar
        plot_bgcolor='rgba(0,0,0,0)',   # transparan background plot
        showlegend=False,
        font=dict(
            family="Fira Code",
            size=word_font_size,
            color="white"
        )
    )
    return fig

@app.route('/Topic-Analysis-RSU-QL-Yogyakarta')
def TA_RSU_QL_Yogyakarta():
    data_all = pd.read_excel('./static/data/dominant_topic_all_yk2.xlsx')
    reviews = data_all[['Dominant_Topic', 'Topic_Contribution', 'Text', 'Sentimen']]

    # Baca filter dari query parameters
    location_filter = request.args.get('locationFilter', 'all')

    # Terapkan filter lokasi
    if location_filter != 'all':
        reviews = reviews[reviews['Sentimen'].str.contains(location_filter, case=False)]

    # Pagination
    per_page = 10
    page = request.args.get('page', 1, type=int)

    start = (page - 1) * per_page
    end = start + per_page

    paginated_reviews = reviews.iloc[start:end]

    # Total pages
    total_reviews = len(reviews)
    total_pages = (total_reviews + per_page - 1) // per_page

    # Rentang halaman untuk pagination
    num_range = 2
    start_range = max(1, page - num_range)
    end_range = min(total_pages, page + num_range)
    page_range = list(range(start_range, end_range + 1))

    # LDA Topics (Positif dan Negatif)
    lda_topics = [
        [("ramah", 0.030), ("rs", 0.026), ("queen", 0.018), ("latifa", 0.017), ("rawat", 0.017),
         ("dokter", 0.016), ("sakit", 0.015), ("rumah", 0.015), ("baik", 0.014), ("nyaman", 0.012)],
        [("ramah", 0.048), ("latifa", 0.036), ("queen", 0.035), ("dokter", 0.022), ("layanan_baik", 0.019),
         ("tugas", 0.018), ("rsu", 0.015), ("jelas", 0.015), ("cepat", 0.015), ("obat", 0.014)],
        [("baik", 0.018), ("periksa", 0.016), ("daftar", 0.016), ("antri", 0.015), ("dokter", 0.014),
         ("bagus", 0.012), ("poli", 0.010), ("rs", 0.009), ("lama", 0.009), ("rawat", 0.008)],
        [("latifa", 0.024), ("queen", 0.022), ("ramah", 0.018), ("dokter", 0.017), ("sakit", 0.017),
         ("rumah", 0.016), ("daftar", 0.013), ("pasien", 0.012), ("baik", 0.012), ("rs", 0.010)]
    ]

    lda_topics2 = [
        [("dokter", 0.019), ("pasien", 0.013), ("ada", 0.011), ("tunggu", 0.010), ("rs", 0.010),
         ("apa", 0.009), ("sakit", 0.008), ("lama", 0.008), ("obat", 0.008), ("bpjs", 0.008)],
        [("kali", 0.010), ("jam", 0.010), ("padahal", 0.009), ("apa", 0.009), ("lama", 0.009),
         ("pasien", 0.009), ("antri", 0.008), ("rs", 0.008), ("kak", 0.008), ("sakit", 0.008)],
        [("dokter", 0.025), ("pasien", 0.022), ("daftar", 0.014), ("bpjs", 0.010), ("harus", 0.010),
         ("kalau", 0.009), ("ramah", 0.009), ("padahal", 0.009), ("kurang", 0.008), ("ada", 0.008)],
        [("antri", 0.023), ("jam", 0.014), ("kalau", 0.013), ("obat", 0.013), ("tunggu", 0.012),
         ("tanggal", 0.012), ("daftar", 0.010), ("datang", 0.010), ("ada", 0.009), ("bagi", 0.009)],
        [("sakit", 0.016), ("dokter", 0.013), ("antri", 0.012), ("rs", 0.012), ("rawat", 0.012),
         ("anak", 0.010), ("rumah", 0.010), ("baik", 0.010), ("daftar", 0.010), ("kalau", 0.010)],
        [("obat", 0.024), ("antri", 0.022), ("farmasi", 0.022), ("lama", 0.019), ("tunggu", 0.017),
         ("jam", 0.015), ("banyak", 0.014), ("banget", 0.011), ("pasien", 0.010), ("bagi", 0.010)],
        [("dokter", 0.027), ("rs", 0.019), ("jam", 0.018), ("anak", 0.016), ("apa", 0.010),
         ("periksa", 0.010), ("kalau", 0.010), ("ada", 0.008), ("banget", 0.008), ("obat", 0.008)],
        [("antri", 0.029), ("jam", 0.028), ("lama", 0.024), ("tunggu", 0.020), ("obat", 0.019),
         ("daftar", 0.015), ("poli", 0.014), ("panggil", 0.012), ("nomor", 0.010), ("farmasi", 0.010)],
    ]

    fig_positive = visualize_lda_topics(lda_topics, topic_titles=["Topik Positif 1", "Topik Positif 2", "Topik Positif 3", "Topik Positif 4"], max_cols=2)
    fig_negative = visualize_lda_topics(lda_topics2, topic_titles=[f"Topik Negatif {i+1}" for i in range(len(lda_topics2))], max_cols=3)

    graphJSON_positive = json.dumps(fig_positive, cls=plotly.utils.PlotlyJSONEncoder)
    graphJSON_negative = json.dumps(fig_negative, cls=plotly.utils.PlotlyJSONEncoder)

    return render_template(
        'topic-analysis-QL-Y.html',
        reviews=paginated_reviews.to_dict(orient='records'),
        page=page,
        total_pages=total_pages,
        page_range=page_range,
        location_filter=location_filter,
        graphJSON_positive=graphJSON_positive,
        graphJSON_negative=graphJSON_negative
    )

@app.route('/Topic-Analysis-RSU-QL-KulonProgo')
def TA_RSU_QL_KulonProgo():
    data_all = pd.read_excel('./static/data/dominant_topic_all_kp2.xlsx')
    reviews = data_all[['Dominant_Topic', 'Topic_Contribution', 'Text', 'Sentimen']]

    # Baca filter dari query parameters
    location_filter = request.args.get('locationFilter', 'all')

    # Terapkan filter lokasi
    if location_filter != 'all':
        reviews = reviews[reviews['Sentimen'].str.contains(location_filter, case=False)]

    # Pagination
    per_page = 10
    page = request.args.get('page', 1, type=int)

    start = (page - 1) * per_page
    end = start + per_page

    paginated_reviews = reviews.iloc[start:end]

    # Total pages
    total_reviews = len(reviews)
    total_pages = (total_reviews + per_page - 1) // per_page

    # Rentang halaman untuk pagination
    num_range = 2
    start_range = max(1, page - num_range)
    end_range = min(total_pages, page + num_range)
    page_range = list(range(start_range, end_range + 1))

    lda_topics3 = [
    [("layanan_ramah", 0.028), ("baik", 0.020), ("dokter", 0.018), ("sakit", 0.014), ("rumah", 0.013),
     ("cepat", 0.012), ("puas", 0.012), ("bersih", 0.010), ("tempat", 0.008), ("nyaman", 0.008)],
    [("ramah", 0.057), ("queen", 0.043), ("latifa", 0.042), ("dokter", 0.031), ("rawat", 0.022),
     ("rsu", 0.017), ("rs", 0.017), ("kulon", 0.017), ("progo", 0.017), ("tugas", 0.016)],
    [("sakit", 0.021), ("rumah", 0.018), ("bagus", 0.016), ("rawat", 0.015), ("ramah", 0.014),
     ("karyawan", 0.011), ("oke", 0.011), ("layanan_cepat", 0.010), ("nyaman", 0.010), ("tempat", 0.009)]
    ]

    lda_topics4 = [
    [("dokter", 0.023), ("tunggu", 0.020), ("kamar", 0.017), ("jam", 0.012), ("lebih", 0.012),
     ("sini", 0.010), ("datang", 0.009), ("lain", 0.009), ("lama", 0.009), ("keluarga", 0.009)],
    [("dokter", 0.023), ("anak", 0.013), ("hari", 0.013), ("untuk", 0.013), ("tanya", 0.013),
     ("beri", 0.013), ("jam", 0.012), ("pasien", 0.011), ("vaksin", 0.011), ("daftar", 0.011)],
    [("jam", 0.035), ("dokter", 0.023), ("jadwal", 0.022), ("ada", 0.013), ("pasien", 0.012),
     ("swab", 0.012), ("datang", 0.011), ("sama", 0.011), ("anak", 0.010), ("rawat", 0.010)],
    [("anak", 0.022), ("kali", 0.018), ("pindah", 0.015), ("banget", 0.011), ("trauma", 0.011),
     ("si", 0.011), ("coba", 0.011), ("kalau", 0.011), ("mau", 0.011), ("pasien", 0.011)],
    [("jadwal", 0.017), ("lama", 0.016), ("jahat", 0.016), ("tunggu", 0.016), ("kelamin", 0.016),
     ("pasien", 0.009), ("baik", 0.009), ("sesuai", 0.009), ("layanan_ibu", 0.009), ("hamil", 0.009)],
    [("obat", 0.033), ("cuma", 0.018), ("baik", 0.018), ("kasih", 0.015), ("usg", 0.015),
     ("lama", 0.014), ("antri", 0.013), ("mau", 0.012), ("sekali", 0.011), ("pasien", 0.009)],
    [("dokter", 0.022), ("jam", 0.022), ("banget", 0.018), ("datang", 0.018), ("bayar", 0.015),
     ("tarif", 0.015), ("mau", 0.011), ("rs", 0.011), ("sama", 0.011), ("kali", 0.011)],
    [("pasien", 0.022), ("tanya", 0.021), ("sama", 0.014), ("dokter", 0.012), ("tugas", 0.012),
     ("kecewa", 0.012), ("bahkan", 0.011), ("mohon", 0.011), ("antri", 0.010), ("datang", 0.008)]
    ]

    fig_positive = visualize_lda_topics(lda_topics3, topic_titles=["Topik Positif 1", "Topik Positif 2", "Topik Positif 3"], max_cols=2)
    fig_negative = visualize_lda_topics(lda_topics4, topic_titles=[f"Topik Negatif {i+1}" for i in range(len(lda_topics4))], max_cols=3)

    graphJSON_positive = json.dumps(fig_positive, cls=plotly.utils.PlotlyJSONEncoder)
    graphJSON_negative = json.dumps(fig_negative, cls=plotly.utils.PlotlyJSONEncoder)

    return render_template(
        'topic-analysis-QL-KP.html',
        reviews=paginated_reviews.to_dict(orient='records'),
        page=page,
        total_pages=total_pages,
        page_range=page_range,
        location_filter=location_filter,
        graphJSON_positive=graphJSON_positive,
        graphJSON_negative=graphJSON_negative
    )

@app.route('/Sentimen-Topic-Tools')
def Sentimen_Topic_Tools():
    return render_template('tools2.html')

def preprocess_text(text):
    text = CaseFoldingText(text)
    text = cleaning_text(text)
    file_path = './static/data/slang_word_list.csv'
    slang_dict = load_slangwords(file_path) 
    text = fix_slangwords(text, slang_dict)  
    text = tokenizingText(text)
    text = remove_stopwords1(text)
    text = remove_stopwords(text)
    text = lemmatize_text(text)
    return text

def predict(texts):
    text_tfidf = vectorizer.transform(texts)
    text_tfidf_dense = text_tfidf.toarray()
    predictions = svm_model.predict(text_tfidf_dense)
    return predictions

# def calculate_coherence_score(texts, num_topics):
#     # Prepare corpus for gensim
#     texts_tokenized = [text.split() for text in texts]
#     dictionary = Dictionary(texts_tokenized)
#     corpus = [dictionary.doc2bow(text) for text in texts_tokenized]
    
#     # Train LDA model
#     lda_model = LdaModel(
#         corpus=corpus,
#         num_topics=num_topics,
#         id2word=dictionary,
#         random_state=42,
#         passes=10
#     )
    
#     # Calculate coherence score
#     coherence_model = CoherenceModel(
#         model=lda_model,
#         texts=texts_tokenized,
#         dictionary=dictionary,
#         coherence='c_v'
#     )
    
#     return coherence_model.get_coherence()

def calculate_coherence_score(texts, n_topics):
    """Calculate coherence score for given number of topics"""
    vectorizer = TfidfVectorizer(max_features=1000)
    dtm = vectorizer.fit_transform(texts)
    
    lda = LatentDirichletAllocation(
        n_components=n_topics,
        random_state=42,
        max_iter=10
    )
    
    lda_output = lda.fit_transform(dtm)
    
    # Calculate coherence score (using topic word distributions)
    coherence = 0
    for topic_idx, topic in enumerate(lda.components_):
        top_term_indices = topic.argsort()[:-10:-1]
        term_frequencies = np.sum(dtm[:, top_term_indices].toarray(), axis=0)
        coherence += np.mean(term_frequencies)
    
    return coherence / n_topics

def find_optimal_topics(texts, max_topics):
    """Find optimal number of topics based on coherence scores"""
    coherence_scores = []
    for n_topics in range(2, max_topics + 1):
        score = calculate_coherence_score(texts, n_topics)
        coherence_scores.append((n_topics, score))
    
    # Return number of topics with highest coherence score
    return max(coherence_scores, key=lambda x: x[1])[0]

def perform_topic_modeling(texts, num_topics=None, max_topics=10):
    if len(texts) == 0:
        return []
    
    # Determine number of topics
    if num_topics is None:
        # Automatic selection
        num_topics = find_optimal_topics(texts, max_topics)
    
    # Perform topic modeling
    topic_vectorizer = TfidfVectorizer()
    tfidf_matrix = topic_vectorizer.fit_transform(texts)
    
    lda = LatentDirichletAllocation(
        n_components=num_topics,
        random_state=42,
        max_iter=10
    )
    lda.fit(tfidf_matrix)
    
    feature_names = topic_vectorizer.get_feature_names_out()
    topics = []
    
    for topic_idx, topic in enumerate(lda.components_):
        top_words = [feature_names[i] for i in topic.argsort()[:-10:-1]]
        topics.append({
            'topic': f'Topic {topic_idx + 1}',
            'words': ', '.join(top_words),
            'coherence_score': calculate_coherence_score(texts, num_topics)
        })
    
    return topics

# Flask routes
@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    topic_selection = request.form.get('topicSelection', 'auto')
    num_topics = int(request.form.get('numTopics', 5)) if topic_selection == 'manual' else None

    if file.filename.endswith('.csv'):
        df = pd.read_csv(file)
    elif file.filename.endswith('.xlsx'):
        df = pd.read_excel(file)
    else:
        return jsonify({'error': 'Unsupported file format'}), 400

    if 'review_text' not in df.columns:
        return jsonify({'error': 'File must contain a review_text column'}), 400

    df['processed_text'] = df['review_text'].apply(preprocess_text)
    predictions = predict(df['processed_text'])
    df['predictions'] = predictions

    positive_df = df[df['predictions'] == 'Positif']
    negative_df = df[df['predictions'] == 'Negatif']

    positive_topics = perform_topic_modeling(
        positive_df['processed_text'].tolist(),
        num_topics=num_topics,
        max_topics=10
    )
    negative_topics = perform_topic_modeling(
        negative_df['processed_text'].tolist(),
        num_topics=num_topics,
        max_topics=10
    )
    
    sentiment_topics = {
        'positive': {
            'count': len(positive_df),
            'topics': positive_topics
        },
        'negative': {
            'count': len(negative_df),
            'topics': negative_topics
        }
    }

    # Save sentiment analysis results
    result_df = df[['review_text', 'processed_text', 'predictions']]
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Save sentiment analysis CSV
    sentiment_filename = f'sentiment_analysis_{timestamp}.csv'
    sentiment_filepath = os.path.join(PROCESSED_FOLDER, sentiment_filename)
    result_df.to_csv(sentiment_filepath, index=False)
    
    # Save topics analysis CSV
    topics_filename = f'topics_analysis_{timestamp}.csv'
    topics_filepath = os.path.join(PROCESSED_FOLDER, topics_filename)
    
    # Create topics DataFrame
    topics_data = []
    for sentiment in ['positive', 'negative']:
        for topic in sentiment_topics[sentiment]['topics']:
            topics_data.append({
                'sentiment': sentiment,
                'topic_number': topic['topic'],
                'words': topic['words'],
                'coherence_score': topic['coherence_score']
            })
    
    topics_df = pd.DataFrame(topics_data)
    topics_df.to_csv(topics_filepath, index=False)

    return jsonify({
        'success': True,
        'data': result_df.to_dict('records'),
        'sentiment_filename': sentiment_filename,
        'topics_filename': topics_filename,
        'sentiment_topics': sentiment_topics
    })

@app.route('/download/<filename>')
def download_file(filename):
    return send_from_directory(PROCESSED_FOLDER, filename, as_attachment=True)

if __name__ == '__main__':
    app.run(debug=True)