from flask import Flask, render_template, request, send_file, jsonify
import pandas as pd
import os
from datetime import datetime
from prep import *
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation

app = Flask(__name__)

with open("./static/model/tfidf_vectorizer.pkl", "rb") as file:
    vectorizer = joblib.load(file)

svm_model = joblib.load("./static/model/svm_model.pkl")

UPLOAD_FOLDER = 'uploads'
PROCESSED_FOLDER = 'processed'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(PROCESSED_FOLDER, exist_ok=True)

@app.route('/')
def index():
    return render_template('tools.html')

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

def perform_topic_modeling(texts, n_topics=2):
    if len(texts) == 0:
        return []  
        
    topic_vectorizer = TfidfVectorizer()
    tfidf_matrix = topic_vectorizer.fit_transform(texts)
    
    lda = LatentDirichletAllocation(n_components=n_topics, random_state=42)
    lda.fit(tfidf_matrix)
    
    feature_names = topic_vectorizer.get_feature_names_out()
    topics = []
    for topic_idx, topic in enumerate(lda.components_):
        top_words = [feature_names[i] for i in topic.argsort()[:-10:-1]]
        print(top_words)
        topics.append({
            'topic': f'Topic {topic_idx + 1}',
            'words': ', '.join(top_words)
        })
    return topics

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

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

    positive_topics = perform_topic_modeling(positive_df['processed_text'].tolist())
    negative_topics = perform_topic_modeling(negative_df['processed_text'].tolist())
    
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

    result_df = df[['review_text', 'processed_text', 'predictions']]
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    processed_filename = f'processed_text_{timestamp}.csv'
    processed_filepath = os.path.join(PROCESSED_FOLDER, processed_filename)
    result_df.to_csv(processed_filepath, index=False)

    processed_data = result_df.to_dict('records')

    return jsonify({
        'success': True,
        'data': processed_data,
        'filename': processed_filename,
        'sentiment_topics': sentiment_topics
    })

@app.route('/download/<filename>')
def download_file(filename):
    return send_file(
        os.path.join(PROCESSED_FOLDER, filename),
        as_attachment=True,
        download_name=filename
    )

if __name__ == '__main__':
    app.run(debug=True)

