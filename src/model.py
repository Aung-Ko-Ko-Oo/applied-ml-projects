# -*- coding: utf-8 -*-
import pandas as pd
from transformers import pipeline

class AIAnalyzer:
    def __init__(self):
        print("Loading Deep Learning Transformer Model (BART)...")
        # သင့်စက်ထဲက Transformers ဗားရှင်းအရ Task Name ကို 'text-generation' ဟု ပြောင်းလဲထားပါသည်
        self.summarizer = pipeline(
            "text-generation", 
            model="facebook/bart-large-cnn"
        )
        print("Model loaded successfully!")

    def train_sentiment_model(self, data_path):
        """ CSV Dataset ကို ဖတ်ပြီး Baseline Model အတွက် ပြင်ဆင်ခြင်း """
        print(f"Loading dataset from: {data_path}")
        try:
            df = pd.read_csv(data_path)
            print(f"Dataset loaded successfully with {len(df)} rows.")
        except Exception as e:
            print(f"Error loading dataset: {e}")

    def summarize_text(self, text):
        """ စာသားရှည်များကို အနှစ်ချုပ်ပေးမည့် Function """
        print("Summarizing text...")
        # text-generation task အတွက် max_new_tokens သုံးပေးခြင်းက ပိုစိတ်ချရပါသည်
        result = self.summarizer(text, max_new_tokens=130, do_sample=False)
        return result[0]['generated_text']