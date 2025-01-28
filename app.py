from flask import Flask, render_template, request, jsonify
import os
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer

app = Flask(__name__)

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
    data_all = pd.read_excel('./static/data/Sentimen-Done-rev-2.xlsx')
    reviews = data_all[['text_casefoldingText', 'location']]

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
    data_all = pd.read_excel('./static/data/Sentimen-Done-rev-2.xlsx')
    reviews = data_all[['text_clean', 'location']]

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
    data_all = pd.read_excel('./static/data/Sentimen-Done-rev-2.xlsx')
    reviews = data_all[['text_slang', 'location']]

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
    data_all = pd.read_excel('./static/data/Sentimen-Done-rev-2.xlsx')
    reviews = data_all[['text_token', 'location']]

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
    data_all = pd.read_excel('./static/data/Sentimen-Done-rev-2.xlsx')
    reviews = data_all[['text_stopwords', 'location']]

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
    data_all = pd.read_excel('./static/data/Sentimen-Done-rev-2.xlsx')
    reviews = data_all[['text_lemmatized', 'location']]

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


@app.route('/Representasi-Vektor', methods=['GET'])
def representasi():
    data_all = pd.read_excel('./static/data/Sentimen-Done-rev-3.xlsx')
    reviews = data_all[['tfidf', 'location']]

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
        'representasi.html',
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
    data_QL_1 = pd.read_excel('./static/data/Sentimen-Done-rev-2.xlsx')
    sentimen1 = data_QL_1[['text_lemmatized', 'location', 'sentimen', 'Predicted_Label']]
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
    X_pos = vectorizer_pos.fit_transform(positive_reviews['text_lemmatized'])
    words_pos = vectorizer_pos.get_feature_names_out()
    frequencies_pos = X_pos.sum(axis=0).A1
    word_freq_pos = pd.DataFrame(list(zip(words_pos, frequencies_pos)), columns=['Word', 'Frequency'])
    word_freq_pos = word_freq_pos.sort_values(by='Frequency', ascending=False).head(25)

    # Membuat model CountVectorizer untuk Negatif
    vectorizer_neg = CountVectorizer(stop_words='english', max_features=25)
    X_neg = vectorizer_neg.fit_transform(negative_reviews['text_lemmatized'])
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
    data_QL_1 = pd.read_excel('./static/data/Sentimen-Done-rev-2.xlsx')
    sentimen1 = data_QL_1[['text_lemmatized', 'location', 'sentimen', 'Predicted_Label']]

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
    X_pos = vectorizer_pos.fit_transform(positive_reviews['text_lemmatized'])
    words_pos = vectorizer_pos.get_feature_names_out()
    frequencies_pos = X_pos.sum(axis=0).A1
    word_freq_pos = pd.DataFrame(list(zip(words_pos, frequencies_pos)), columns=['Word', 'Frequency'])
    word_freq_pos = word_freq_pos.sort_values(by='Frequency', ascending=False).head(25)

    # Membuat model CountVectorizer untuk Negatif
    vectorizer_neg = CountVectorizer(stop_words='english', max_features=25)
    X_neg = vectorizer_neg.fit_transform(negative_reviews['text_lemmatized'])
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

@app.route('/Topic-Analysis-RSU-QL-Yogyakarta')
def TA_RSU_QL_Yogyakarta():
    data_all = pd.read_excel('./static/data/dominant_topic_all_yk.xlsx')
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

    return render_template(
        'topic-analysis-QL-Y.html',
        reviews=paginated_reviews.to_dict(orient='records'),
        page=page,
        total_pages=total_pages,
        page_range=page_range,
        location_filter=location_filter
    )

@app.route('/Topic-Analysis-RSU-QL-KulonProgo')
def TA_RSU_QL_KulonProgo():
    data_all = pd.read_excel('./static/data/dominant_topic_all_kp.xlsx')
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

    return render_template(
        'topic-analysis-QL-KP.html',
        reviews=paginated_reviews.to_dict(orient='records'),
        page=page,
        total_pages=total_pages,
        page_range=page_range,
        location_filter=location_filter
    )

@app.route('/Sentimen-Topic-Tools')
def Sentimen_Topic_Tools():
    return render_template('sentimen-topic-tools.html')

if __name__ == '__main__':
    app.run(debug=True)