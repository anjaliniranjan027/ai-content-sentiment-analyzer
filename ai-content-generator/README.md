# AI-Powered Content Generator

This project is an intermediate-level AI web app that:
- Generates text based on a prompt using GPT-2 (Generative AI)
- Analyzes the sentiment of the generated text (Machine Learning)

## Requirements

- Python 3.8+
- pip

## Setup and Run

1. Create a virtual environment (recommended):
```
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate   # Windows
```

2. Install dependencies:
```
pip install -r requirements.txt
```

3. Run the Flask app:
```
python app.py
```

4. Open your browser and go to:
```
http://127.0.0.1:5000/
```

## Notes

- Models will download automatically the first time you run.
- You can change the max_length in app.py for longer output.