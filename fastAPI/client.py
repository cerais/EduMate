import requests

# Адрес сервера FastAPI
url = "http://127.0.0.1:8000/process/"

# JSON-данные для отправки в запросе
data = {
    "public_key": "https://disk.yandex.ru/i/IfIXsgaJxoDYzQ"
}

try:
    # Выполнение POST-запроса
    response = requests.post(url, json=data)
    
    # Проверка статуса ответа
    if response.status_code == 200:
        # Обработка успешного ответа
        result = response.json()
        print("Сводка:", result.get("summary"))
    else:
        # Обработка ошибок
        print("Ошибка:", response.status_code, response.json().get("detail"))
except requests.exceptions.RequestException as e:
    print("Ошибка соединения:", e)


