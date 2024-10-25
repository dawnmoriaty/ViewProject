import os
import nltk
import pandas as pd
import logging
import joblib
from flask import Flask, render_template, request, redirect, url_for, flash
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

nltk.download('punkt')
nltk.download('stopwords')

app = Flask(__name__)
app.secret_key = 'your_secret_key'  # Cần thiết cho flash messages

# Thiết lập logging
logging.basicConfig(filename='app.log', level=logging.INFO)

DATA_FOLDER_EN = 'data/english'
DATA_FOLDER_VI = 'data/vietnamese'

model_en = model_vi = vectorizer_en = vectorizer_vi = None
result = None

# Tự tạo danh sách stopwords cho tiếng Việt
vietnamese_stopwords = {
    'ta', 'mình', 'chúng ta', 'của tôi', 'của mình', 'của chúng ta', 'của chúng mình',
    'họ', 'chúng', 'của họ', 'của chúng', 'nó', 'của nó',
    'nay', 'hôm nay', 'hôm qua', 'ngày mai', 'sáng', 'trưa', 'chiều', 'tối', 'lúc', 'khi', 'trước', 'sau', 'rồi', 'sớm',
    'muộn', 'đây', 'đó', 'này', 'kia', 'đâu', 'nào', 'đến', 'từ', 'qua', 'lên', 'xuống',
    'thôi', 'nào', 'mà', 'chứ', 'nhé', 'à', 'ơi',
    'ôi', 'a', 'hay', 'quá', 'lắm', 'thật', 'nhiều', 'ít', 'mấy', 'một số', 'các'
}


def preprocess_text(text, language):
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    words = word_tokenize(text)

    if language == 'english':
        stop_words = set(stopwords.words('english'))
    elif language == 'vietnamese':
        stop_words = vietnamese_stopwords

    words = [word for word in words if word not in stop_words]
    return ' '.join(words)


def save_model(model, vectorizer, language):
    joblib.dump(model, f'model_{language}.joblib')
    joblib.dump(vectorizer, f'vectorizer_{language}.joblib')
    logging.info(f"Model and vectorizer saved for {language}")


def load_model(language):
    model = joblib.oad(f'model_{language}.joblib')
    vectorizer = joblib.load(f'vectorizer_{language}.joblib')
    logging.info(f"Model and vectorizer loaded for {language}")
    return model, vectorizer


@app.route('/')
def select_file():
    files_en = [f for f in os.listdir(DATA_FOLDER_EN) if f.endswith('.csv')]
    files_vi = [f for f in os.listdir(DATA_FOLDER_VI) if f.endswith('.csv')]
    return render_template('select_file.html', files_en=files_en, files_vi=files_vi)


@app.route('/train', methods=['POST'])
def train_model():
    global model_en, model_vi, vectorizer_en, vectorizer_vi, result
    try:
        selected_file = request.form['file']
        language = request.form['language']
        test_size = float(request.form['test_size'])  # Lấy giá trị test_size từ form

        # Kiểm tra giá trị test_size hợp lệ
        if not (0.1 <= test_size <= 0.9):
            flash("Test size phải nằm trong khoảng từ 0.1 đến 0.9", 'error')
            return redirect(url_for('select_file'))

        logging.info(f"Training model for {language} using file {selected_file} with test_size {test_size}")

        if language == 'english':
            file_path = os.path.join(DATA_FOLDER_EN, selected_file)
        else:
            file_path = os.path.join(DATA_FOLDER_VI, selected_file)

        data = pd.read_csv(file_path)

        if 'text' not in data.columns or 'label' not in data.columns:
            flash(f"File '{selected_file}' không có cột 'text' và 'label'.", 'error')
            return redirect(url_for('select_file'))

        data['text'] = data['text'].apply(lambda x: preprocess_text(x, language))

        vectorizer = TfidfVectorizer()
        X = vectorizer.fit_transform(data['text'])
        y = data['label']

        # Sử dụng test_size từ form
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

        nb_model = MultinomialNB()
        nb_model.fit(X_train, y_train)

        dt_model = DecisionTreeClassifier()
        dt_model.fit(X_train, y_train)

        ensemble_model = VotingClassifier(estimators=[('nb', nb_model), ('dt', dt_model)], voting='hard')
        ensemble_model.fit(X_train, y_train)

        y_pred = ensemble_model.predict(X_test)

        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, pos_label='spam')
        recall = recall_score(y_test, y_pred, pos_label='spam')
        f1 = f1_score(y_test, y_pred, pos_label='spam')

        if language == 'english':
            model_en = ensemble_model
            vectorizer_en = vectorizer
        else:
            model_vi = ensemble_model
            vectorizer_vi = vectorizer

        save_model(ensemble_model, vectorizer, language)

        result = {
            'file': selected_file,
            'language': language,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1
        }

        return redirect(url_for('results'))

    except Exception as e:
        logging.error(f"An error occurred: {str(e)}")
        flash(f"An error occurred: {str(e)}", 'error')
        return redirect(url_for('select_file'))


@app.route('/results')
def results():
    if result is not None:
        return render_template('result.html',
                               file=result['file'],
                               language=result['language'],
                               accuracy=result['accuracy'],
                               precision=result['precision'],
                               recall=result['recall'],
                               f1=result['f1'])
    else:
        flash('No results available yet.', 'info')
        return redirect(url_for('select_file'))


@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        text = request.form['text']
        language = request.form['language']

        if language == 'english' and model_en and vectorizer_en:
            processed_text = preprocess_text(text, 'english')
            vectorized_text = vectorizer_en.transform([processed_text])
            prediction = model_en.predict(vectorized_text)[0]
        elif language == 'vietnamese' and model_vi and vectorizer_vi:
            processed_text = preprocess_text(text, 'vietnamese')
            vectorized_text = vectorizer_vi.transform([processed_text])
            prediction = model_vi.predict(vectorized_text)[0]
        else:
            flash("Model not trained for this language", 'error')
            return redirect(url_for('predict'))

        return render_template('prediction.html', text=text, prediction=prediction)
    return render_template('predict_form.html')


if __name__ == '__main__':
    os.makedirs(DATA_FOLDER_EN, exist_ok=True)
    os.makedirs(DATA_FOLDER_VI, exist_ok=True)

    if os.path.exists('model_english.joblib'):
        model_en, vectorizer_en = load_model('english')
    if os.path.exists('model_vietnamese.joblib'):
        model_vi, vectorizer_vi = load_model('vietnamese')

    app.run(debug=True)
