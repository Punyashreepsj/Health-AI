HealthAI — AI-Powered Personal Health Assistant

Overview:
HealthAI is a smart health analysis and advisory web app that helps users analyze symptoms,
get possible disease predictions, and receive treatment recommendations.
It uses AI models like IBM Granite or Hugging Face for medical reasoning.

Features:
- User details input: Name, Age, Gender, Medical History, Current Medication
- Predicts diseases based on symptoms
- Suggests personalized treatment plans
- Interactive AI-based health chat
- Simple Streamlit frontend and FastAPI backend

Project Structure:
HealthAI/
│
├── app.py             -> Streamlit frontend
├── backend.py         -> FastAPI backend
├── requirements.txt   -> Dependencies
└── README.txt         -> Documentation

Setup Instructions:

1. Clone Repository
   git clone https://github.com/<your-username>/<your-repo-name>.git
   cd <your-repo-name>

2. Create Virtual Environment
   On Windows:
       python -m venv venv
       venv\Scripts\activate
   On Mac/Linux:
       python3 -m venv venv
       source venv/bin/activate

3. Install Requirements
   pip install -r requirements.txt

Running the App:

1. Start Backend Server
   uvicorn backend:app --host 127.0.0.1 --port 8000 --reload

2. Run Streamlit Frontend
   streamlit run app.py

   - Backend runs at: http://127.0.0.1:8000
   - Frontend runs at: http://localhost:8501

Ensure backend is running before launching Streamlit app.

