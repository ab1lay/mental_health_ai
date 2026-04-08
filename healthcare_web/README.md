# Healthcare Web (FastAPI)

## Setup

1) Install deps:
```
pip install -r healthcare_web/requirements.txt
```

2) Add to .env:
```
WEB_SECRET_KEY=change-me
ADMIN_EMAIL=admin@example.com
ADMIN_PASSWORD=change-me
```

3) Run:
```
uvicorn healthcare_web.app.main:app --reload
```

Open http://127.0.0.1:8000
