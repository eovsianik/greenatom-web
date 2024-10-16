import torch
import numpy as np
from transformers import AutoModelForSequenceClassification
from transformers import AutoTokenizer


def split_long_text(text, max_length=100):
    """
    Разбивает длинные строки на несколько частей для использования в коде.

    Параметры:
    - text (str): исходный текст
    - max_length (int): максимальная длина строки

    Возвращает:
    - str: строка, разбитая на несколько частей, объединённая с помощью символа "+"
    """
    words = text.split()
    lines = []
    current_line = []

    for word in words:
        # Если текущая строка с новым словом будет длиннее max_length, то переносим строку
        if len(" ".join(current_line + [word])) > max_length:
            lines.append(" ".join(current_line))
            current_line = [word]
        else:
            current_line.append(word)

    # Добавляем последнюю строку
    if current_line:
        lines.append(" ".join(current_line))

    # Объединяем все строки в нужный формат
    return " +\n".join([f'"{line}"' for line in lines])


def predict(model, text, tokenizer, device):
    # Новая строка
    text = split_long_text(text)
    # Токенизация текста
    inputs = tokenizer(
        text, return_tensors="pt", truncation=True, padding="max_length", max_length=128
    )

    # Удаляем 'token_type_ids', если они есть (DistilBERT их не использует)
    if "token_type_ids" in inputs:
        del inputs["token_type_ids"]

    # Переносим токенизированные данные на то же устройство, что и модель
    inputs = {key: value.to(device) for key, value in inputs.items()}

    # Делаем предсказание без вычисления градиентов
    with torch.no_grad():
        outputs = model(**inputs)

    # Извлекаем предсказанную оценку
    score = outputs.logits.item()

    return round(score)


def score_review(text: str) -> int:
    model_path = "model"
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    predicted_rating = predict(model, text, tokenizer, device)
    if int(predicted_rating) < 1:
        return 1
    else:
        return int(predicted_rating)
