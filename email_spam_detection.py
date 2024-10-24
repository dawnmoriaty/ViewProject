import nltk
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('stopwords')
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



# Tự tạo danh sách stopwords cho tiếng Việt
vietnamese_stopwords = {
    'ta', 'mình', 'chúng ta', 'chúng mình', 'của tôi', 'của mình', 'của chúng ta', 'của chúng mình',
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

# Đọc dữ liệu đầu vào
data_en = pd.read_csv('data/english/emails_english.csv')
data_vi = pd.read_csv('data/vietnamese/dataset1.csv')

# # Kiểm tra tên cột trong DataFrame
# print("Tên cột trong emails_vietnamese.csv:", data_vi.columns)

# Loại bỏ khoảng trắng trong tên cột
data_vi.columns = data_vi.columns.str.strip()

# # Kiểm tra lại tên cột
# print("Tên cột sau khi loại bỏ khoảng trắng:", data_vi.columns)

# Xử lý dữ liệu
try:
    # Tham số hoá dữ liệu
    data_en['text'] = data_en['text'].apply(lambda x: preprocess_text(x, 'english'))
    data_vi['text'] = data_vi['text'].apply(lambda x: preprocess_text(x, 'vietnamese'))
except Exception as e:
    print("Error:", e)

# Hàm để xây dựng và đánh giá mô hình
def build_and_evaluate_model(data, language):
    # Kiểm tra xem cột 'label' có tồn tại không
    if 'label' not in data.columns:
        print(f"Cột 'label' không tồn tại trong dữ liệu {language}. Vui lòng kiểm tra lại dữ liệu đầu vào.")
        return None, None

    # Chuyển đổi văn bản thành vector số
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(data['text'])
    y = data['label']

    # Chia dữ liệu
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42)

    # Xây dựng mô hình Naive Bayes
    nb_model = MultinomialNB()
    nb_model.fit(X_train, y_train)

    # Xây dựng mô hình Decision Tree
    dt_model = DecisionTreeClassifier()
    dt_model.fit(X_train, y_train)

    # Kết hợp hai mô hình
    ensemble_model = VotingClassifier(estimators=[('nb', nb_model), ('dt', dt_model)], voting='hard')
    ensemble_model.fit(X_train, y_train)

    # Dự đoán và đánh giá mô hình
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

    return ensemble_model, vectorizer

# Xây dựng và đánh giá mô hình cho tiếng Anh
model_en, vectorizer_en = build_and_evaluate_model(data_en, 'English')

# Xây dựng và đánh giá mô hình cho tiếng Việt
model_vi, vectorizer_vi = build_and_evaluate_model(data_vi, 'Vietnamese')


