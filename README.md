# Bone-Fracture-Detection

Проект для детекции переломов на фото.  
В проекте есть:
- **YOLO-модель** (YOLO8s)
- **Backend + Frontend**: Streamlit

---

## 1) Чек-лист требований

- [x] Тюнинг модели
- [x] Backend (Streamlit, без очереди)
- [ ] Backend (async с очередью) — для максимального балла (опционально)
- [x] Frontend (Streamlit)
- [ ] Выбор модели пользователем (>=2 моделей)
- [x] Размещение решения на GitHub
- [x] Видео-презентация приложения (после готовности backend+frontend)
- [x] Деплой на Streamlit Cloud/аналог — доп. баллы (опционально)

---

## 2) Датасет

- Bone Fracture Dataset Good Data: https://www.kaggle.com/datasets/mohammedmohsen0404/bone-fracture-dataset-good-data 



---

## 3) Backend (FastAPI)

Backend умеет:
- прогнать изображение и вернуть bbox’ы (JSON)
- вернуть PNG с отрисованными bbox’ами

### 3.1 Запуск backend локально
Из корня репозитория:
```bash
uvicorn backend.app.main:app --host 0.0.0.0 --port 8000 --reload
```

Эндпоинты:
- `GET /models`
- `POST /detect/image`
- `POST /detect/image_render`

---

## 4) Frontend (Streamlit)

UI позволяет:
- загрузить изображение
- увидеть результат и список детекций

### 4.1 Запуск frontend локально
```bash
streamlit run frontend/app.py
```

### 4.2 Адрес backend (через переменную окружения)
По умолчанию используется `http://localhost:8000`.  
Можно переопределить:

**PowerShell**
```powershell
$env:API_URL="http://localhost:8000"
streamlit run frontend/app.py
```

**cmd**
```bat
set API_URL=http://localhost:8000
streamlit run frontend/app.py
```

