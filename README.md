How to run:

1. Create virtualenv:
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```
2. Start server:
   ```bash
    .venv/bin/python -m uvicorn app.main:app --reload
   ```
3. Test via FastApi-swagger:
   - POST /detect - Try out - upload image and excute
