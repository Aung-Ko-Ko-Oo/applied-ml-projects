import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# NLTK dependencies ကို အလိုအလျောက် Download ဆွဲခိုင်းခြင်း
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('punkt')
    nltk.download('stopwords')

def clean_text(text):
    """စာသားများထဲမှ မလိုလားအပ်သော စာလုံးများကို ဖယ်ထုတ်ပေးသည့် Function"""
    stop_words = set(stopwords.words('english'))
    # Tokenization & Lowercasing
    word_tokens = word_tokenize(text.lower())
    # Stopwords နှင့် သင်္ကေတများ ဖယ်ထုတ်ခြင်း
    filtered_text = [w for w in word_tokens if w.isalpha() and w not in stop_words]
    return " ".join(filtered_text)