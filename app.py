import nltk
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from flask import Flask, render_template

# Tải stopwords
nltk.download('punkt')
nltk.download('stopwords')

# Tự tạo danh sách stopwords cho tiếng Việt
vietnamese_stopwords = {
    'ta', 'mình', 'chúng ta', 'của tôi', 'của mình', 'của chúng ta', 'của chúng mình',
    'họ', 'chúng', 'của họ', 'của chúng', 'nó', 'của nó',
    'nay', 'hôm nay', 'hôm qua', 'ngày mai', 'sáng', 'trưa', 'chiều', 'tối', 'lúc', 'khi', 'trước', 'sau', 'rồi', 'sớm',
    'muộn', 'đây', 'đó', 'này', 'kia', 'đâu', 'nào', 'đến', 'từ', 'qua', 'lên', 'xuống',
    'thôi', 'nào', 'mà', 'chứ', 'nhé', 'à', 'ơi',
    'ôi', 'a', 'hay', 'quá', 'lắm', 'thật', 'nhiều', 'ít', 'mấy', 'một số', 'các'
}

# Tiền xử lý dữ liệu
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

# Flask app
app = Flask(__name__)

# Khởi tạo biến toàn cục
model_en = model_vi = vectorizer_en = vectorizer_vi = None
result = None

# Huấn luyện mô hình khi ứng dụng bắt đầu
def load_and_train_models():
    global model_en, vectorizer_en, model_vi, vectorizer_vi
    global result

    try:
        # Đọc dữ liệu
        data_en = pd.read_csv('data/emails_english.csv')
        data_vi = pd.read_csv('data/dataset1.csv')

        # Loại bỏ khoảng trắng trong tên cột
        data_vi.columns = data_vi.columns.str.strip()

        # Kiểm tra xem cột 'label' tồn tại không
        if 'label' not in data_en.columns or 'label' not in data_vi.columns:
            raise ValueError("Dữ liệu phải có cột 'label'.")

        # Xử lý dữ liệu
        data_en['text'] = data_en['text'].apply(lambda x: preprocess_text(x, 'english'))
        data_vi['text'] = data_vi['text'].apply(lambda x: preprocess_text(x, 'vietnamese'))

        # Hàm để xây dựng và đánh giá mô hình
        def build_and_evaluate_model(data, language):
            vectorizer = TfidfVectorizer()
            X = vectorizer.fit_transform(data['text'])
            y = data['label']

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42)

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

            print(f'Results for {language} model:')
            print(f'Accuracy: {accuracy}')
            print(f'Precision: {precision}')
            print(f'Recall: {recall}')
            print(f'F1 Score: {f1}\n')

            return ensemble_model, vectorizer, accuracy, precision, recall, f1

        # Huấn luyện mô hình tiếng Anh
        model_en, vectorizer_en, accuracy_en, precision_en, recall_en, f1_en = build_and_evaluate_model(data_en, 'English')

        # Huấn luyện mô hình tiếng Việt
        model_vi, vectorizer_vi, accuracy_vi, precision_vi, recall_vi, f1_vi = build_and_evaluate_model(data_vi, 'Vietnamese')

        # Lưu kết quả
        result = {
            'accuracy_en': accuracy_en,
            'precision_en': precision_en,
            'recall_en': recall_en,
            'f1_en': f1_en,
            'accuracy_vi': accuracy_vi,
            'precision_vi': precision_vi,
            'recall_vi': recall_vi,
            'f1_vi': f1_vi
        }

        # In kết quả để kiểm tra
        print(f"English Model - Accuracy: {accuracy_en}, Precision: {precision_en}, Recall: {recall_en}, F1: {f1_en}")
        print(f"Vietnamese Model - Accuracy: {accuracy_vi}, Precision: {precision_vi}, Recall: {recall_vi}, F1: {f1_vi}")

    except Exception as e:
        print(f"Error during model training: {e}")

# Huấn luyện mô hình trước khi khởi động Flask
load_and_train_models()

@app.route('/')
def index():
    if result is not None:
        return render_template(
            'index.html',
            accuracy_en=result['accuracy_en'],
            precision_en=result['precision_en'],
            recall_en=result['recall_en'],
            f1_en=result['f1_en'],
            accuracy_vi=result['accuracy_vi'],
            precision_vi=result['precision_vi'],
            recall_vi=result['recall_vi'],
            f1_vi=result['f1_vi']
        )
    else:
        return "Error during model training"

if __name__ == '__main__':
    app.run(debug=True)