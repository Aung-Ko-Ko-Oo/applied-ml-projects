# -*- coding: utf-8 -*-
import os
from src.model import AIAnalyzer

if __name__ == "__main__":
    # ၁။ Analyzer Object ကို စတင်ဆောက်ခြင်း
    analyzer = AIAnalyzer()

    # ၂။ Baseline Model ကို Dataset ပေး၍ လုပ်ဆောင်ခြင်း
    # (မှတ်ချက် - သင့် project folder ထဲတွင် data/reviews.csv ရှိနေရပါမည်)
    dataset_path = "data/reviews.csv"
    analyzer.train_sentiment_model(dataset_path)

    # ၃။ အနှစ်ချုပ်မည့် နမူနာစာသားရှည်ကြီး
    sample_document = """
    Artificial Intelligence is transforming the educational landscape rapidly. 
    Intelligent Tutoring Systems are being developed to provide personalized feedback 
    to students worldwide. However, many existing systems struggle because they fail 
    to understand the emotional state or the sentiment of the student's inputs. 
    By integrating Natural Language Processing with Deep Learning, we can create a system 
    that not only tracks what a student knows but also understands how they feel about 
    the learning material.
    """

    # ၄။ စာသားကို အနှစ်ချုပ်ခိုင်းပြီး Output ထုတ်ပြခြင်း
    print("\n--- Running Text Summarization ---")
    summary_result = analyzer.summarize_text(sample_document)
    
    print("\n[Original Text]:")
    print(sample_document.strip())
    
    print("\n[Summarized Output]:")
    print(summary_result)