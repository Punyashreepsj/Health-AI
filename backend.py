import os
import json
import logging
from typing import List, Dict, Optional, Tuple

import requests
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn

LOG = logging.getLogger("healthai")
logging.basicConfig(level=logging.INFO)

# Simple rule DBs
_SYMPTOM_DB = {
     # respiratory / systemic
    "fever": ["Influenza", "COVID-19", "Pneumonia", "Common Cold", "Gastroenteritis"],
    "cough": ["Bronchitis", "COVID-19", "Common Cold", "Pneumonia", "COPD", "Asthma"],
    "productive cough": ["Bronchitis", "Pneumonia", "COPD"],
    "sore throat": ["Common Cold", "Influenza", "COVID-19", "Strep Throat"],
    "nasal congestion": ["Common Cold", "Allergy"],
    "runny nose": ["Common Cold", "Allergy"],
    "anosmia": ["COVID-19"],
    "shortness of breath": ["Asthma", "COPD", "Pneumonia", "Heart Failure", "Pulmonary Embolism"],
    "wheeze": ["Asthma", "Bronchitis", "COPD"],
    "chest pain": ["Angina", "Myocardial Infarction", "GERD", "Pneumonia", "Pericarditis"],
    "pleuritic pain": ["Pneumonia", "Pulmonary Embolism"],
    # neurological / head
    "headache": ["Migraine", "Tension Headache", "Hypertension", "Sinusitis"],
    "unilateral headache": ["Migraine"],
    "photophobia": ["Migraine"],
    "dizziness": ["Arrhythmia", "Dehydration", "Vestibular Neuritis"],
    "sudden weakness": ["Stroke"],
    "slurred speech": ["Stroke"],
    # constitutional / systemic
    "fatigue": ["Anemia", "Hypothyroidism", "Depression", "Chronic Fatigue"],
    "weight loss": ["Hyperthyroidism", "Diabetes", "Malignancy"],
    "night sweats": ["Infection", "Lymphoma", "Tuberculosis"],
    # gastrointestinal / urinary
    "abdominal pain": ["Gastroenteritis", "Appendicitis", "Gallstones", "GERD"],
    "nausea": ["Gastroenteritis", "Migraine", "Myocardial Infarction"],
    "vomiting": ["Gastroenteritis", "Food Poisoning"],
    "diarrhea": ["Gastroenteritis", "Food Poisoning"],
    "heartburn": ["GERD"],
    "dysuria": ["Urinary Tract Infection"],
    "frequency": ["Urinary Tract Infection", "Diabetes"],
    # endocrine / metabolic
    "polyuria": ["Diabetes"],
    "polydipsia": ["Diabetes"],
    # musculoskeletal / skin / rheum
    "joint pain": ["Arthritis", "Rheumatoid Arthritis", "Osteoarthritis"],
    "rash": ["Allergy", "Eczema", "Dermatitis", "Infection"],
    "itching": ["Allergy", "Eczema"],
    # cardiovascular
    "palpitations": ["Arrhythmia", "Anxiety", "Hyperthyroidism"],
    "leg swelling": ["Heart Failure", "Venous Insufficiency"],
    # general catch-alls
    "anemia signs": ["Anemia"],
    "memory loss": ["Dementia", "Alzheimer's", "Depression"],

}

_TREATMENT_DB = {
    
    "Strep Throat": "Confirm with rapid strep test / culture; treat with appropriate antibiotics if positive.",
    "Heart Failure": "Low-salt diet, diuretics, ACE inhibitors/ARBs as prescribed; cardiology follow-up.",
    "Pulmonary Embolism": "Emergency — anticoagulation and hospital evaluation required.",
    "Pericarditis": "NSAIDs and specialist follow-up; seek care for chest pain changes.",
    "Tension Headache": "Stress management, OTC analgesics (ibuprofen/acetaminophen), rest.",
    "Sinusitis": "Saline irrigation, decongestants, antibiotics if bacterial suspected.",
    "Vestibular Neuritis": "Vestibular rehabilitation, antiemetics for severe symptoms.",
    "Stroke": "Emergency — call emergency services immediately.",
    "Gastroenteritis": "Hydration, oral rehydration solutions; seek care for persistent vomiting or bloody stools.",
    "Urinary Tract Infection": "Antibiotics per culture/sensitivity; increase fluids and follow-up.",
    "Diabetes": "Lifestyle measures, blood glucose monitoring, medication/insulin as prescribed.",
    "Arthritis": "Analgesics, physical therapy, rheumatology referral when indicated.",
    "Allergy": "Avoid triggers, antihistamines, intranasal steroids, allergy specialist if severe.",
    "Eczema": "Emollients, topical steroids for flares, avoid irritants.",
    "Arrhythmia": "Cardiology evaluation; rate/rhythm control and anticoagulation depending on type.",
    "Osteoarthritis": "Weight management, exercise, NSAIDs/topical agents, orthopedics/physio referral.",
    "Rheumatoid Arthritis": "Early rheumatology referral; DMARDs and disease control strategies.",

}

# --- Granite / Watson lightweight connector ---
def init_granite_model(api_key_env: str = "GRANITE_API_KEY", url_env: str = "GRANITE_API_URL") -> Optional[Dict]:
    """
    Initialize a lightweight connection descriptor for an IBM Granite/Watson ML text model.
    Returns connection info dict or None if env vars are missing.
    """
    api_key = os.getenv(api_key_env)
    api_url = os.getenv(url_env)
    if not api_key or not api_url:
        LOG.info("Granite env vars missing; continuing without remote model.")
        return None
    return {"api_key": api_key, "api_url": api_url}

def _call_granite_model(conn: Dict, prompt: str, max_tokens: int = 512) -> str:
    """
    Minimal HTTP call to a Granite-style generative endpoint.
    Adjust payload/response parsing to your provider's contract.
    """
    if not conn:
        return ""
    payload = {"prompt": prompt, "max_tokens": max_tokens}
    headers = {"Authorization": f"Bearer {conn['api_key']}", "Content-Type": "application/json"}
    try:
        r = requests.post(conn["api_url"], headers=headers, json=payload, timeout=20)
        r.raise_for_status()
        j = r.json()
        # try standard fields
        if isinstance(j, dict):
            for key in ("text", "generated_text", "output", "result"):
                if key in j and isinstance(j[key], str):
                    return j[key]
            if "choices" in j and isinstance(j["choices"], list) and j["choices"]:
                first = j["choices"][0]
                if isinstance(first, dict):
                    return first.get("text") or first.get("message", {}).get("content", "") or ""
        return json.dumps(j)
    except Exception as e:
        LOG.debug(f"Granite call failed: {e}")
        return ""

# --- Core logic functions ---
def predict_disease(symptoms: List[str], patient_age: Optional[int] = None, conn: Optional[Dict] = None) -> List[Tuple[str, float]]:
    symptoms_norm = [s.strip().lower() for s in symptoms if s and s.strip()]
    scores: Dict[str, float] = {}
    for s in symptoms_norm:
        for key, diagnoses in _SYMPTOM_DB.items():
            if key in s or s in key:
                for d in diagnoses:
                    scores[d] = scores.get(d, 0.0) + 1.0
    if scores:
        maxv = max(scores.values())
        results = [(d, round(v / maxv, 2)) for d, v in scores.items()]
        results.sort(key=lambda x: x[1], reverse=True)
    else:
        results = []

    if conn:
        prompt = (
            f"Patient symptoms: {', '.join(symptoms)}.\n"
            f"Age: {patient_age or 'unknown'}.\n"
            "Provide top 5 possible diagnoses with confidence scores between 0 and 1 as JSON list of objects {diagnosis, confidence}."
        )
        text = _call_granite_model(conn, prompt, max_tokens=300)
        if text:
            try:
                parsed = json.loads(text)
                remote_results = []
                if isinstance(parsed, list):
                    for item in parsed:
                        if isinstance(item, dict):
                            name = item.get("diagnosis") or item.get("name")
                            conf = float(item.get("confidence", 0))
                            if name:
                                remote_results.append((name, round(conf, 2)))
                if remote_results:
                    merged = {d: c for d, c in remote_results}
                    for d, c in results:
                        if d not in merged:
                            merged[d] = round(c * 0.6, 2)
                    results = sorted([(d, round(c, 2)) for d, c in merged.items()], key=lambda x: x[1], reverse=True)
            except Exception:
                LOG.debug("Failed to parse Granite JSON; ignoring remote results.")

    if not results:
        results = [("Undetermined — further evaluation needed", 0.0)]
    return results

def generate_treatment_plan(diagnoses: List[str], patient_info: Optional[Dict] = None, conn: Optional[Dict] = None) -> str:
    patient_info = patient_info or {}
    plan_items = []
    for d in diagnoses:
        snippet = _TREATMENT_DB.get(d, f"General evaluation recommended for {d}. Follow standard clinical guidelines.")
        plan_items.append(f"- {d}: {snippet}")
    plan_text = "Personalized treatment plan:\n" + "\n".join(plan_items)

    if conn:
        prompt = (
            f"Create a concise, patient-friendly treatment plan for: {', '.join(diagnoses)}.\n"
            f"Patient info: {json.dumps(patient_info)}.\n"
            "Limit to 6 bullet points, include red flags that require urgent care, and suggest follow-up timeline."
        )
        extra = _call_granite_model(conn, prompt, max_tokens=400)
        if extra:
            return extra.strip()
    return plan_text

def answer_patient_query(question: str, history: Optional[List[Dict]] = None, conn: Optional[Dict] = None) -> str:
    history = history or []
    q = (question or "").strip()
    if not q:
        return "Please enter a question."
    ql = q.lower()
    if "when to seek" in ql or "emergency" in ql or "call" in ql or "911" in ql or "urgent" in ql:
        return "If you have severe chest pain, severe shortness of breath, sudden weakness or slurred speech, or uncontrolled bleeding — seek emergency care immediately."
    if conn:
        prompt = (
            f"You are a general medical assistant (non-diagnostic). Patient question: \"{q}\".\n"
            "Answer in plain language, cite when to seek emergency care, keep under 300 words."
        )
        resp = _call_granite_model(conn, prompt, max_tokens=350)
        if resp:
            return resp.strip()
    if "fever" in ql:
        return "For fevers, stay hydrated and use antipyretics like acetaminophen. Seek care if fever > 40°C (104°F), persistent >48 hours, or accompanied by severe symptoms."
    if "covid" in ql or "coronavirus" in ql:
        return "If you suspect COVID-19, test per local guidance, isolate, and monitor breathing and oxygenation. Seek care for difficulty breathing."
    return "I can help with general information. For personalized medical advice, consult your healthcare provider. If this is an emergency, seek immediate care."

# --- FastAPI app and request models ---
app = FastAPI(title="Health-AI Backend")

# Allow requests from Streamlit (local)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

conn = init_granite_model()

class ChatRequest(BaseModel):
    message: str
    name: Optional[str] = None
    age: Optional[int] = None
    gender: Optional[str] = None
    medical_history: Optional[str] = None
    current_medication: Optional[str] = None

class PredictRequest(BaseModel):
    symptoms: str  # free text or comma separated
    name: Optional[str] = None
    age: Optional[int] = None
    gender: Optional[str] = None
    medical_history: Optional[str] = None
    current_medication: Optional[str] = None

class TreatmentRequest(BaseModel):
    condition: str
    name: Optional[str] = None
    age: Optional[int] = None
    gender: Optional[str] = None
    medical_history: Optional[str] = None
    current_medication: Optional[str] = None

@app.post("/chat")
def chat_endpoint(req: ChatRequest):
    try:
        reply = answer_patient_query(req.message, history=None, conn=conn)
        return {"reply": reply}
    except Exception as e:
        LOG.exception("Chat error")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict/disease")
# ...existing code...
@app.post("/predict/disease")
def predict_endpoint(req: PredictRequest):
    try:
        # parse symptoms: accept comma-separated or free text (split on commas)
        s = [seg.strip() for seg in req.symptoms.split(",")] if "," in req.symptoms else [req.symptoms.strip()]
        s = [x for x in s if x]
        preds = predict_disease(s, patient_age=req.age, conn=conn)

        # structured list for clients
        prediction_list = [{"diagnosis": d, "confidence": float(conf)} for d, conf in preds]

        # user-readable summary
        lines = []
        for i, (d, conf) in enumerate(preds, start=1):
            pct = int(round(conf * 100))
            lines.append(f"{i}. {d} — {pct}%")
        readable = "Possible diagnoses (ranked):\n" + "\n".join(lines)

        return {"prediction": prediction_list, "readable": readable}
    except Exception as e:
        LOG.exception("Predict error")
        raise HTTPException(status_code=500, detail=str(e))
# ...existing code...
# ...existing code...
        resp = call_backend("predict/disease", payload)
        if resp:
                # prefer backend-provided readable string if available
            if isinstance(resp.get("readable"), str) and resp.get("readable").strip():
                    st.markdown("### Possible diagnoses (from backend):")
                    # preserve newlines
                    st.text(resp["readable"])
            elif "prediction" in resp and isinstance(resp["prediction"], list) and resp["prediction"]:
                    preds = resp["prediction"]
                    df = pd.DataFrame(preds)
                    if "confidence" in df.columns:
                        df["confidence_pct"] = (df["confidence"].astype(float) * 100).round().astype(int).astype(str) + "%"
                        st.write("### Possible diagnoses (ranked):")
                        for _, row in df.iterrows():
                            st.markdown(f"- **{row.get('diagnosis')}** — {row.get('confidence_pct')}")
                    else:
                        st.write(df)
            else:
                    st.info("No predictions returned. Try different symptom phrasing.")
# ...existing code...

@app.post("/treatment/plan")
def treatment_endpoint(req: TreatmentRequest):
    try:
        diagnoses = [req.condition.strip()] if req.condition else []
        patient_info = {"name": req.name, "age": req.age, "gender": req.gender, "medical_history": req.medical_history, "current_medication": req.current_medication}
        plan = generate_treatment_plan(diagnoses, patient_info=patient_info, conn=conn)
        return {"plan": plan}
    except Exception as e:
        LOG.exception("Treatment error")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    # Run with: python backend.py  OR use uvicorn from terminal:
    # uvicorn backend:app --host 127.0.0.1 --port 8000 --reload
    uvicorn.run("backend:app", host="127.0.0.1", port=8000, reload=True)
# ...existing code...
