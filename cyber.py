import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

nltk.download('punkt')
nltk.download('stopwords')

def preprocess_text(text):
    # Tokenization
    words = word_tokenize(text)

    # Remove stop words
    stop_words = set(stopwords.words("english"))
    filtered_words = [word.lower() for word in words if word.isalnum() and word.lower() not in stop_words]

    # Stemming
    ps = PorterStemmer()
    stemmed_words = [ps.stem(word) for word in filtered_words]

    return " ".join(stemmed_words)

def calculate_similarity(text1, text2):
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform([text1, text2])
    similarity = cosine_similarity(tfidf_matrix[0], tfidf_matrix[1])
    return similarity[0][0]

def detect_threat(input_text, threat_patterns):
    preprocessed_input = preprocess_text(input_text)
    
    for threat_pattern in threat_patterns:
        preprocessed_pattern = preprocess_text(threat_pattern)
        similarity = calculate_similarity(preprocessed_input, preprocessed_pattern)
        
        # You may need to adjust the threshold based on your data and requirements
        if similarity > 0.7:
            print(f"Threat detected: {threat_pattern}")
            return True
    
    print("No threat detected.")
    return False

if __name__ == "__main__":
    # Example usage
    threat_patterns = [
        "malware attack",
        "phishing attempt",
        "data breach",
        # Add more threat patterns as needed
    ]

    input_text = input("Enter a text: ")

    detect_threat(input_text, threat_patterns)
