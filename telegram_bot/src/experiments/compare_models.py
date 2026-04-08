import logging
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from sklearn.pipeline import Pipeline

# Для BERT (заглушка, требует torch и transformers)
# from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments

from src.core.graph import run_graph
import asyncio

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Пример данных (заглушка)
DUMMY_DATA = [
    {"text": "Я чувствую себя прекрасно и полон сил!", "label": "positive"},
    {"text": "Мне очень грустно и ничего не хочется делать.", "label": "negative"},
    {"text": "Обычный день, ничего особенного.", "label": "neutral"},
    {"text": "У меня тревога перед экзаменом.", "label": "negative"},
    {"text": "Я счастлив, что встретил друзей.", "label": "positive"},
] * 10

def prepare_data():
    df = pd.DataFrame(DUMMY_DATA)
    return train_test_split(df["text"], df["label"], test_size=0.2, random_state=42)

def train_eval_tfidf_logreg(X_train, X_test, y_train, y_test):
    logger.info("🧪 Запуск TF-IDF + Logistic Regression...")
    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer()),
        ("clf", LogisticRegression())
    ])
    
    pipeline.fit(X_train, y_train)
    preds = pipeline.predict(X_test)
    
    acc = accuracy_score(y_test, preds)
    logger.info(f"✅ TF-IDF Accuracy: {acc:.4f}")
    logger.info("\n" + classification_report(y_test, preds))
    return pipeline

async def eval_llm_baseline(X_test, y_test):
    logger.info("🧪 Запуск LLM Baseline (Mental Health AI)...")
    preds = []
    for text in X_test:
        # Запускаем наш граф (он возвращает текст, нам нужно парсить или изменить граф для возврата label)
        # Здесь мы эмулируем, так как run_graph возвращает форматированную строку.
        # В реальном эксперименте нужно использовать analyze_node напрямую.
        
        # Эмуляция вызова (так как это долго и тратит токены)
        # result = await run_graph(text) 
        # Для теста просто случайный выбор или мок
        preds.append("neutral") 
    
    acc = accuracy_score(y_test, preds)
    logger.info(f"✅ LLM Baseline Accuracy: {acc:.4f} (Mocked)")
    return preds

def main():
    X_train, X_test, y_train, y_test = prepare_data()
    
    # 1. Classical NLP
    train_eval_tfidf_logreg(X_train, X_test, y_train, y_test)
    
    # 2. BERT (Skeleton)
    logger.info("🧪 BERT evaluation skipped (requires heavy deps).")
    
    # 3. LLM Baseline
    asyncio.run(eval_llm_baseline(X_test, y_test))

if __name__ == "__main__":
    main()
