import pandas as pd
import requests
from tqdm import tqdm
import time
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import os

from prompts import ZERO_SHOT_PROMPT, FEW_SHOT_PROMPT, COT_PROMPT

# ========================= НАСТРОЙКИ =========================
SAMPLE_SIZE = 200
RANDOM_STATE = 42

# Поддержка двух возможных расположений датасета
POSSIBLE_PATHS = [
    "data/spam.csv",
    "spam.csv",
    "../spam.csv"
]

# ========================= ЗАГРУЗКА ДАННЫХ =========================
df = None
for path in POSSIBLE_PATHS:
    if os.path.exists(path):
        df = pd.read_csv(path, encoding='latin-1')
        print(f"Датасет успешно загружен из: {path}")
        break

if df is None:
    raise FileNotFoundError("Не найден файл spam.csv. Положите его в папку 'data/' или в корень проекта.")

df = df[['v1', 'v2']].rename(columns={'v1': 'label', 'v2': 'message'})
df = df.sample(SAMPLE_SIZE, random_state=RANDOM_STATE).reset_index(drop=True)

# Преобразуем метки в числовой формат: ham=0, spam=1
df['true_label'] = df['label'].str.lower().map({'ham': 0, 'spam': 1})

print(f"Загружено {len(df)} сообщений для оценки.\n")

# ========================= ФУНКЦИИ =========================
def query_llm(prompt: str, model: str = "qwen2.5:0.5b") -> str:
    try:
        response = requests.post(
            "http://localhost:8000/generate",
            json={
                "prompt": prompt,
                "model": model
            },
            timeout=60
        )
        response.raise_for_status()
        return response.json().get("response", "").strip()
    except Exception as e:
        print(f"Ошибка запроса к LLM: {e}")
        return ""


def extract_verdict(text: str) -> int:
    """Извлекаем 0 (ham) или 1 (spam) из ответа модели"""
    if not text:
        return 0
    
    text_lower = text.lower().strip()
    
    # Проверка последних слов ответа
    last_words = text_lower.split()[-3:]
    if 'spam' in last_words:
        return 1
    if 'ham' in last_words:
        return 0
    
    # Общий поиск
    if "spam" in text_lower and "ham" not in text_lower:
        return 1
    if "ham" in text_lower:
        return 0
    
    return 0  # по умолчанию ham


# ========================= ОСНОВНОЙ ЦИКЛ =========================
results = []
techniques = {
    "zero_shot": ZERO_SHOT_PROMPT,
    "few_shot": FEW_SHOT_PROMPT,
    "cot": COT_PROMPT,
}

print("Начинаю оценку техник промптинга...\n")

for tech_name, prompt_template in techniques.items():
    print(f"→ Обрабатываю технику: {tech_name.upper()}")
    
    predictions = []
    
    for _, row in tqdm(df.iterrows(), total=len(df), desc=tech_name):
        message = row['message']
        prompt = prompt_template.format(message=message)
        
        raw_response = query_llm(prompt)
        pred = extract_verdict(raw_response)
        
        predictions.append(pred)
        
        time.sleep(0.25)  # небольшая задержка
    
    # Сохраняем предсказания
    df[f"pred_{tech_name}"] = predictions
    
    # Вычисляем метрики
    acc = accuracy_score(df['true_label'], predictions)
    prec = precision_score(df['true_label'], predictions, zero_division=0)
    rec = recall_score(df['true_label'], predictions, zero_division=0)
    f1 = f1_score(df['true_label'], predictions, zero_division=0)
    
    results.append({
        "Technique": tech_name.upper(),
        "Accuracy": round(acc, 4),
        "Precision": round(prec, 4),
        "Recall": round(rec, 4),
        "F1-score": round(f1, 4)
    })

# ========================= ИТОГОВАЯ ТАБЛИЦА =========================
print("\n" + "="*60)
print("РЕЗУЛЬТАТЫ СРАВНЕНИЯ ТЕХНИК ПРОМПТИНГА")
print("="*60)

comparison_df = pd.DataFrame(results)
print(comparison_df.to_string(index=False))

# Сохраняем результаты
comparison_df.to_csv("prompting_comparison.csv", index=False)
df.to_csv("detailed_prompting_results.csv", index=False)

print("\nФайлы успешно сохранены:")
print("- prompting_comparison.csv — таблица с метриками")
print("- detailed_prompting_results.csv — детальные результаты")
