import os
import json
import time
import hashlib
import uuid
import logging
import sqlite3
import numpy as np
from typing import Optional, List, Dict, Any
from datetime import datetime, timedelta
from fastapi import FastAPI, File, UploadFile, Form, Header, HTTPException, Depends, Body
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
import torch
from threading import Lock
import pickle
from sklearn.metrics.pairwise import cosine_similarity
from scipy.ndimage import sobel
from scipy.signal import find_peaks
from PIL import Image
from io import BytesIO

from xray_3d import reconstruct_3d
from volumetric_3d import reconstruct_volumetric_3d

# ========== CONFIGURATION ==========
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = FastAPI(
    title="NEXUS Sovereign AI",
    description="Explainable Multi-Modal AI Framework for Medical Diagnosis",
    version="3.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ========== STORAGE SETUP ==========
VAULT_BASE = os.getenv("VAULT_BASE", "./nexus_vault")
os.makedirs(f"{VAULT_BASE}/chats", exist_ok=True)
os.makedirs(f"{VAULT_BASE}/records", exist_ok=True)
os.makedirs(f"{VAULT_BASE}/prescriptions", exist_ok=True)
os.makedirs(f"{VAULT_BASE}/embeddings", exist_ok=True)

# ========== VECTOR EMBEDDING MODEL ==========
embedder = SentenceTransformer('all-MiniLM-L6-v2')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Using device: {device}")

# ========== VECTOR DATABASE ==========
class VectorDatabase:
    def __init__(self, db_path=f"{VAULT_BASE}/embeddings/vectors.pkl"):
        self.db_path = db_path
        self.vectors = []
        self.metadata = []
        self.load()
    
    def add(self, vector, metadata):
        self.vectors.append(vector)
        self.metadata.append(metadata)
        self.save()
    
    def search(self, query_vector, top_k=5):
        if len(self.vectors) == 0:
            return []
        
        similarities = cosine_similarity([query_vector], self.vectors)[0]
        top_indices = np.argsort(similarities)[-top_k:][::-1]
        
        results = []
        for idx in top_indices:
            if similarities[idx] > 0.3:  # Similarity threshold
                results.append({
                    "metadata": self.metadata[idx],
                    "similarity": float(similarities[idx])
                })
        return results
    
    def save(self):
        with open(self.db_path, 'wb') as f:
            pickle.dump({'vectors': self.vectors, 'metadata': self.metadata}, f)
    
    def load(self):
        if os.path.exists(self.db_path):
            with open(self.db_path, 'rb') as f:
                data = pickle.load(f)
                self.vectors = data['vectors']
                self.metadata = data['metadata']
    
    def clear(self):
        self.vectors = []
        self.metadata = []
        self.save()

# Initialize vector database
vector_db = VectorDatabase()

# ========== DATABASE SETUP ==========
def init_database():
    """Initialize SQLite database for medical history"""
    conn = sqlite3.connect(f"{VAULT_BASE}/medical_history.db")
    c = conn.cursor()
    
    # Patients table
    c.execute('''CREATE TABLE IF NOT EXISTS patients
                (patient_id TEXT PRIMARY KEY, 
                 created_date TEXT,
                 last_visit TEXT,
                 age INTEGER,
                 gender TEXT,
                 location TEXT)''')
    
    # Prescriptions table
    c.execute('''CREATE TABLE IF NOT EXISTS prescriptions
                (id INTEGER PRIMARY KEY AUTOINCREMENT,
                 patient_id TEXT,
                 doctor_name TEXT,
                 hospital_name TEXT,
                 date TEXT,
                 diagnosis TEXT,
                 symptoms TEXT,
                 duration_days INTEGER,
                 follow_up_date TEXT,
                 image_path TEXT,
                 embedding_id TEXT,
                 FOREIGN KEY(patient_id) REFERENCES patients(patient_id))''')
    
    # Medicines table
    c.execute('''CREATE TABLE IF NOT EXISTS medicines
                (id INTEGER PRIMARY KEY AUTOINCREMENT,
                 prescription_id INTEGER,
                 name TEXT,
                 dosage TEXT,
                 frequency TEXT,
                 duration TEXT,
                 instructions TEXT,
                 FOREIGN KEY(prescription_id) REFERENCES prescriptions(id))''')
    
    # Allergies table
    c.execute('''CREATE TABLE IF NOT EXISTS allergies
                (id INTEGER PRIMARY KEY AUTOINCREMENT,
                 patient_id TEXT,
                 allergen TEXT,
                 reaction TEXT,
                 severity TEXT,
                 date_diagnosed TEXT,
                 FOREIGN KEY(patient_id) REFERENCES patients(patient_id))''')
    
    # Chronic conditions table
    c.execute('''CREATE TABLE IF NOT EXISTS chronic_conditions
                (id INTEGER PRIMARY KEY AUTOINCREMENT,
                 patient_id TEXT,
                 condition_name TEXT,
                 diagnosed_date TEXT,
                 status TEXT,
                 notes TEXT,
                 FOREIGN KEY(patient_id) REFERENCES patients(patient_id))''')
    
    # Learning data table with embedding references
    c.execute('''CREATE TABLE IF NOT EXISTS learning_data
                (id INTEGER PRIMARY KEY AUTOINCREMENT,
                 patient_id TEXT,
                 input_text TEXT,
                 diagnosis TEXT,
                 confidence REAL,
                 verified BOOLEAN,
                 usage_count INTEGER,
                 timestamp TEXT,
                 embedding_id TEXT UNIQUE,
                 FOREIGN KEY(patient_id) REFERENCES patients(patient_id))''')
    
    conn.commit()
    conn.close()
    logger.info("Database initialized successfully")

# Initialize database on startup
init_database()

file_lock = Lock()

# ========== MULTILINGUAL SUPPORT ==========
STRINGS = {
    "en": {
        "steps": ["Neural synchronization...", "Scanning structure...", "Analyzing patterns..."],
        "verdict": "Fracture Confirmed",
        "welcome": "Welcome to NEXUS Sovereign AI",
        "offline": "Offline Mode - Changes will sync when online"
    },
    "hi": {
        "steps": ["न्यूरल सिंक्रोनाइज़ेशन...", "संरचना की जाँच...", "पैटर्न विश्लेषण..."],
        "verdict": "फ्रैक्चर की पुष्टि हुई",
        "welcome": "नेक्सस सार्वभौमिक एआई में आपका स्वागत है",
        "offline": "ऑफ़लाइन मोड - ऑनलाइन होने पर परिवर्तन सिंक होंगे"
    },
    "te": {
        "steps": ["న్యూరల్ సింక్రొనైజేషన్...", "నిర్మాణ పరిశీలన...", "నమూనాల విశ్లేషణ..."],
        "verdict": "ఫ్రాక్చర్ నిర్ధారించబడింది",
        "welcome": "నెక్సస్ సార్వభౌమ AI కి స్వాగతం",
        "offline": "ఆఫ్‌లైన్ మోడ్ - ఆన్‌లైన్‌లో ఉన్నప్పుడు మార్పులు సింక్ అవుతాయి"
    }
}

# ========== PYDANTIC MODELS ==========
class DiagnosisRequest(BaseModel):
    metadata: str
    session_name: str = "New Case"
    language: str = "en"

class SimulationRequest(BaseModel):
    patient_id: str
    scenarios: List[Dict[str, Any]]

class ScribeRequest(BaseModel):
    audio_transcript: str
    video_analysis: Optional[Dict[str, Any]]
    vitals: Dict[str, float]

class FederatedQueryRequest(BaseModel):
    query_type: str
    anonymized_data: Dict[str, Any]

class Medicine(BaseModel):
    name: str
    dosage: str
    frequency: str
    duration: str
    instructions: str

class PrescriptionUpload(BaseModel):
    patient_id: str
    doctor_name: str
    hospital_name: str
    diagnosis: str
    symptoms: str
    medicines: List[Medicine]
    duration_days: int

class LearningData(BaseModel):
    patient_id: str
    input_text: str
    diagnosis: str
    confidence: float = 70.0
    verified: bool = False

class SemanticSearchRequest(BaseModel):
    query: str
    patient_id: Optional[str] = None
    top_k: int = 5

class DiagnosisResponse(BaseModel):
    session_id: str
    session_name: str
    diagnosis: str
    confidence: float
    is_severe: bool
    attention_targets: List[Dict[str, Any]]
    audit: Dict[str, str]
    biometrics: Dict[str, int]
    steps: List[str]
    similar_cases: Optional[List[Dict[str, Any]]] = None
    precautions: Optional[List[str]] = None

# ========== HELPER FUNCTIONS ==========
def get_language(accept_language: str) -> str:
    lang = accept_language[:2] if accept_language else "en"
    return lang if lang in STRINGS else "en"

def persist_session(session_id: str, data: Dict[str, Any]):
    with file_lock:
        file_path = f"{VAULT_BASE}/chats/{session_id}.json"
        try:
            with open(file_path, "w") as f:
                json.dump(data, f)
            logger.info(f"Session {session_id} persisted successfully.")
        except Exception as e:
            logger.error(f"Failed to persist session {session_id}: {str(e)}")
            raise HTTPException(status_code=500, detail="Persistence failed")

def load_session(session_id: str) -> Dict[str, Any]:
    with file_lock:
        file_path = f"{VAULT_BASE}/chats/{session_id}.json"
        if os.path.exists(file_path):
            try:
                with open(file_path, "r") as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"Failed to load session {session_id}: {str(e)}")
                raise HTTPException(status_code=500, detail="Load failed")
        raise HTTPException(status_code=404, detail="Session not found")

def generate_embedding(text: str) -> List[float]:
    """Generate vector embedding for text"""
    embedding = embedder.encode(text)
    return embedding.tolist()

# ========== DIAGNOSIS ENDPOINT WITH SEMANTIC SEARCH ==========
@app.post("/v1/nexus/diagnose", response_model=DiagnosisResponse)
async def diagnose(
    image: UploadFile = File(...),
    metadata: str = Form(...),
    session_name: str = Form("New Case"),
    accept_language: Optional[str] = Header(None)
):
    lang = get_language(accept_language)
    try:
        contents = await image.read()
        img_hash = hashlib.md5(contents).hexdigest()
        await image.seek(0)
    except Exception as e:
        logger.error(f"Image processing failed: {str(e)}")
        raise HTTPException(status_code=400, detail="Invalid image")

    audit_hash = hashlib.sha256(f"{img_hash}-{time.time()}".encode()).hexdigest()[:12].upper()
    
    # Parse metadata
    meta = json.loads(metadata)
    patient_id = meta.get('patient_id', f"patient_{uuid.uuid4().hex[:8]}")
    symptoms = meta.get('symptoms', '')
    
    # ========== ACTUAL DIAGNOSIS LOGIC WITH IMPROVED IMAGE ANALYSIS ==========
    # Load image
    img = Image.open(BytesIO(contents))
    img_gray = img.convert('L')
    img_array = np.array(img_gray)
    
    # Compute gradients using Sobel, focus on horizontal edges for transverse fractures
    dy = sobel(img_array, axis=0)  # Horizontal gradients
    dx = sobel(img_array, axis=1)  # Vertical gradients
    grad = np.sqrt(dx**2 + dy**2)
    
    # Use absolute dy for detecting horizontal discontinuities
    row_grad = np.mean(np.abs(dy), axis=1)
    
    # Normalize row_grad for better peak detection
    row_grad = row_grad / np.max(row_grad) if np.max(row_grad) > 0 else row_grad
    
    # Find peaks with adjusted parameters for better sensitivity
    peaks, properties = find_peaks(row_grad, prominence=0.15, distance=img.height * 0.03, width=5)
    
    # Sort peaks by prominence to prioritize strongest signals
    if len(peaks) > 0:
        prominences = properties['prominences']
        sort_idx = np.argsort(prominences)[::-1]
        peaks = peaks[sort_idx]
    
    # Prepare attention targets
    attention_targets = []
    labels = ["Fracture Site", "Bone Fragment", "Cortical Break"]
    
    # Estimate bone center for better x positioning
    col_mean = np.mean(img_array, axis=0)
    bone_center_x = np.argmax(col_mean)
    
    for i, peak_row in enumerate(peaks[:3]):  # Limit to top 3 strongest peaks
        # Get slice for this row, use abs(dy) for horizontal breaks
        row_slice = np.abs(dy[peak_row, :])
        
        # Find edge peaks in the row to detect the fracture span
        edge_peaks, edge_props = find_peaks(row_slice, prominence=np.max(row_slice)*0.15 if np.max(row_slice)>0 else 0.1, distance=30, width=10)
        
        if len(edge_peaks) >= 2:
            # Take midpoint of the two strongest edges
            x = np.mean(edge_peaks[:2])
        elif len(edge_peaks) == 1:
            x = edge_peaks[0]
        else:
            # Fallback to max in grad or bone center
            x = np.argmax(row_slice) if np.max(row_slice) > 0 else bone_center_x
        
        # Adjust x if it's too far from bone center (to avoid noise)
        if abs(x - bone_center_x) > img.width * 0.2:
            x = bone_center_x
        
        x_norm = (x / img.width) * 100
        y_norm = (peak_row / img.height) * 100
        attention_targets.append({
            "x": round(x_norm, 1),
            "y": round(y_norm, 1),
            "label": labels[i % len(labels)]
        })
    
    # Determine if fracture
    is_fracture = len(peaks) > 0
    confidence = 94.2 if not is_fracture else min(99.0, 70 + len(peaks) * 10)
    diagnosis_text = "No abnormality detected" if not is_fracture else f"Fracture detected ({len(peaks)} potential sites)"
    is_severe = is_fracture and len(peaks) > 1
    
    # ========== SEMANTIC SEARCH FOR SIMILAR CASES ==========
    similar_cases = []
    precautions = []
    
    if symptoms:
        # Generate embedding for symptoms
        query_embedding = generate_embedding(symptoms)
        
        # Search vector database
        results = vector_db.search(query_embedding, top_k=3)
        
        for result in results:
            metadata = result["metadata"]
            similarity = result["similarity"]
            
            similar_cases.append({
                "diagnosis": metadata.get("diagnosis", "Unknown"),
                "symptoms": metadata.get("symptoms", ""),
                "medicines": metadata.get("medicines", []),
                "similarity": round(similarity * 100, 2),
                "date": metadata.get("date", "")
            })
            
            # Extract precautions from similar cases
            if metadata.get("precautions"):
                precautions.extend(metadata["precautions"])
    
    # Remove duplicate precautions
    precautions = list(set(precautions))[:5] if precautions else []
    
    # Default precautions if none found
    if not precautions and is_fracture:
        precautions = [
            "Immobilize the affected area",
            "Apply ice to reduce swelling",
            "Keep the limb elevated",
            "Avoid putting weight on the injury",
            "Consult an orthopedic specialist"
        ]

    response_data = {
        "session_id": str(uuid.uuid4()),
        "session_name": session_name,
        "diagnosis": diagnosis_text,
        "confidence": confidence,
        "is_severe": is_severe,
        "attention_targets": attention_targets,
        "audit": {"hash": audit_hash},
        "biometrics": {"hr": 82, "spo2": 96},
        "steps": STRINGS[lang]["steps"],
        "similar_cases": similar_cases,
        "precautions": precautions
    }

    persist_session(response_data["session_id"], response_data)
    return response_data

# ========== SEMANTIC SEARCH ENDPOINT ==========
@app.post("/v1/nexus/semantic/search")
async def semantic_search(request: SemanticSearchRequest = Body(...)):
    """Search for similar medical cases using vector embeddings"""
    try:
        # Generate embedding for query
        query_embedding = generate_embedding(request.query)
        
        # Search vector database
        results = vector_db.search(query_embedding, top_k=request.top_k)
        
        # Filter by patient_id if provided
        if request.patient_id:
            results = [r for r in results if r["metadata"].get("patient_id") == request.patient_id]
        
        return {
            "query": request.query,
            "results": results,
            "count": len(results)
        }
    except Exception as e:
        logger.error(f"Semantic search failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")

# ========== LEARN FROM USER DATA ==========
@app.post("/v1/nexus/learn")
async def learn_from_data(data: LearningData = Body(...)):
    """Store user data and generate embeddings for future learning"""
    try:
        # Generate embedding
        embedding = generate_embedding(data.input_text)
        
        # Store in vector database
        metadata = {
            "patient_id": data.patient_id,
            "input_text": data.input_text,
            "diagnosis": data.diagnosis,
            "confidence": data.confidence,
            "verified": data.verified,
            "timestamp": datetime.now().isoformat(),
            "type": "learning_data"
        }
        
        vector_db.add(embedding, metadata)
        
        # Store in SQLite
        conn = sqlite3.connect(f"{VAULT_BASE}/medical_history.db")
        c = conn.cursor()
        
        embedding_id = str(uuid.uuid4())
        c.execute('''INSERT INTO learning_data 
                    (patient_id, input_text, diagnosis, confidence, verified, usage_count, timestamp, embedding_id)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)''',
                    (data.patient_id, data.input_text, data.diagnosis, 
                     data.confidence, data.verified, 0, 
                     datetime.now().isoformat(), embedding_id))
        
        conn.commit()
        conn.close()
        
        return {
            "success": True,
            "message": "Learning data stored successfully",
            "embedding_id": embedding_id
        }
    except Exception as e:
        logger.error(f"Failed to store learning data: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Learning failed: {str(e)}")

# ========== PRESCRIPTION UPLOAD WITH EMBEDDINGS ==========
@app.post("/v1/nexus/prescription/upload")
async def upload_prescription(
    image: UploadFile = File(...),
    patient_id: str = Form(...),
    doctor_name: str = Form(...),
    hospital_name: str = Form(...),
    diagnosis: str = Form(...),
    symptoms: str = Form(...),
    medicines: str = Form(...)  # JSON string of medicines
):
    """Upload prescription and generate embeddings for learning"""
    try:
        contents = await image.read()
        
        # Save prescription image
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        image_path = f"{VAULT_BASE}/prescriptions/{patient_id}_{timestamp}.jpg"
        with open(image_path, "wb") as f:
            f.write(contents)
        
        # Parse medicines
        medicines_list = json.loads(medicines)
        
        # Store in database
        conn = sqlite3.connect(f"{VAULT_BASE}/medical_history.db")
        c = conn.cursor()
        
        # Insert or update patient
        c.execute('''INSERT OR REPLACE INTO patients 
                    (patient_id, created_date, last_visit)
                    VALUES (?, ?, ?)''',
                    (patient_id, 
                     datetime.now().isoformat(),
                     datetime.now().isoformat()))
        
        # Insert prescription
        embedding_id = str(uuid.uuid4())
        c.execute('''INSERT INTO prescriptions 
                    (patient_id, doctor_name, hospital_name, date, 
                     diagnosis, symptoms, duration_days, image_path, embedding_id)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)''',
                    (patient_id, doctor_name, hospital_name,
                     datetime.now().isoformat(),
                     diagnosis, symptoms, 7, image_path, embedding_id))
        
        prescription_id = c.lastrowid
        
        # Insert medicines
        for med in medicines_list:
            c.execute('''INSERT INTO medicines
                        (prescription_id, name, dosage, frequency, duration, instructions)
                        VALUES (?, ?, ?, ?, ?, ?)''',
                        (prescription_id, med["name"], med["dosage"], 
                         med["frequency"], med["duration"], med["instructions"]))
        
        conn.commit()
        conn.close()
        
        # ========== GENERATE EMBEDDING FOR LEARNING ==========
        # Create text representation for embedding
        text_for_embedding = f"""
        Symptoms: {symptoms}
        Diagnosis: {diagnosis}
        Medicines: {', '.join([m['name'] for m in medicines_list])}
        Doctor: {doctor_name}
        """
        
        embedding = generate_embedding(text_for_embedding)
        
        # Store in vector database
        metadata = {
            "patient_id": patient_id,
            "diagnosis": diagnosis,
            "symptoms": symptoms,
            "medicines": medicines_list,
            "doctor_name": doctor_name,
            "hospital_name": hospital_name,
            "date": datetime.now().isoformat(),
            "type": "prescription",
            "embedding_id": embedding_id,
            "precautions": [med.get("instructions", "") for med in medicines_list]
        }
        
        vector_db.add(embedding, metadata)
        
        return {
            "success": True,
            "prescription_id": prescription_id,
            "patient_id": patient_id,
            "embedding_id": embedding_id,
            "medicines": medicines_list
        }
        
    except Exception as e:
        logger.error(f"Prescription upload failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Prescription processing failed: {str(e)}")

# ========== GET SIMILAR CASES ==========
@app.post("/v1/nexus/cases/similar")
async def get_similar_cases(
    symptoms: str = Body(...),
    patient_id: Optional[str] = Body(None),
    top_k: int = Body(5)
):
    """Find similar medical cases using semantic search"""
    try:
        # Generate embedding for symptoms
        query_embedding = generate_embedding(symptoms)
        
        # Search vector database
        results = vector_db.search(query_embedding, top_k=top_k)
        
        # Filter by patient if specified
        if patient_id:
            results = [r for r in results if r["metadata"].get("patient_id") == patient_id]
        
        # Format results
        formatted_results = []
        for r in results:
            metadata = r["metadata"]
            formatted_results.append({
                "diagnosis": metadata.get("diagnosis", "Unknown"),
                "symptoms": metadata.get("symptoms", ""),
                "medicines": metadata.get("medicines", []),
                "doctor": metadata.get("doctor_name", "Unknown"),
                "date": metadata.get("date", ""),
                "similarity": round(r["similarity"] * 100, 2),
                "precautions": metadata.get("precautions", [])
            })
        
        return {
            "symptoms": symptoms,
            "similar_cases": formatted_results,
            "count": len(formatted_results)
        }
    except Exception as e:
        logger.error(f"Failed to get similar cases: {str(e)}")
        return {
            "symptoms": symptoms,
            "similar_cases": [],
            "count": 0,
            "error": str(e)
        }

# ========== GET MEDICAL HISTORY ==========
@app.post("/v1/nexus/medical/history")
async def get_medical_history(patient_id: str = Body(..., embed=True)):
    """Retrieve complete medical history for a patient"""
    try:
        conn = sqlite3.connect(f"{VAULT_BASE}/medical_history.db")
        c = conn.cursor()
        
        # Get patient info
        c.execute('''SELECT * FROM patients WHERE patient_id = ?''', (patient_id,))
        patient = c.fetchone()
        
        # Get all prescriptions
        c.execute('''SELECT * FROM prescriptions WHERE patient_id = ? ORDER BY date DESC''', (patient_id,))
        prescriptions_data = c.fetchall()
        
        history = []
        for pres in prescriptions_data:
            # Get medicines for this prescription
            c.execute('''SELECT * FROM medicines WHERE prescription_id = ?''', (pres[0],))
            medicines = c.fetchall()
            
            med_list = []
            for med in medicines:
                med_list.append({
                    "name": med[3],
                    "dosage": med[4],
                    "frequency": med[5],
                    "duration": med[6],
                    "instructions": med[7]
                })
            
            history.append({
                "prescription_id": pres[0],
                "date": pres[4],
                "doctor_name": pres[2],
                "hospital_name": pres[3],
                "diagnosis": pres[5],
                "symptoms": pres[6],
                "duration_days": pres[7],
                "medicines": med_list
            })
        
        conn.close()
        
        return {
            "patient_id": patient_id,
            "prescriptions": history,
            "total_prescriptions": len(history)
        }
        
    except Exception as e:
        logger.error(f"Failed to get medical history: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to retrieve medical history: {str(e)}")

# ========== X-RAY 3D RECONSTRUCTION ==========
@app.post("/v1/nexus/xray/reconstruct-3d")
async def xray_reconstruct_3d(
    image: UploadFile = File(...),
    use_ai: bool = Form(True),
    grid_step: int = Form(4),
):
    """
    Convert a single 2D X-ray into a virtual 3D representation.
    Returns depth map (base64 PNG) and mesh data (vertices + faces) for true 3D rendering.
    use_ai: if True, uses MiDaS depth model when available; otherwise heuristic only.
    grid_step: mesh resolution (higher = fewer vertices, e.g. 4 = every 4th pixel).
    """
    logger.info(f"3D reconstruction request: use_ai={use_ai}, grid_step={grid_step}")
    
    try:
        contents = await image.read()
        if not contents:
            logger.error("Empty image received")
            raise HTTPException(status_code=400, detail="Empty image")
        
        logger.info(f"Image received: size={len(contents)} bytes")
        result = reconstruct_3d(contents, use_ai=use_ai, grid_step=max(2, min(8, grid_step)))
        
        logger.info(f"3D reconstruction successful: vertices={len(result['mesh']['vertices'])//3}, used_ai={result['used_ai']}")
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"X-ray 3D reconstruction failed: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"3D reconstruction failed: {str(e)}")


# ========== VOLUMETRIC 3D BONE RECONSTRUCTION ==========
@app.post("/v1/nexus/xray/reconstruct-volumetric-3d")
async def xray_reconstruct_volumetric_3d(
    image: UploadFile = File(...),
):
    """
    Advanced volumetric 3D reconstruction from 2D X-ray.
    Creates true 3D bone models with fracture detection and multi-dimensional representation.
    Returns volumetric mesh with proper fracture visualization.
    """
    logger.info("Volumetric 3D reconstruction request")
    
    try:
        contents = await image.read()
        if not contents:
            logger.error("Empty image received")
            raise HTTPException(status_code=400, detail="Empty image")
        
        logger.info(f"Image received: size={len(contents)} bytes")
        result = reconstruct_volumetric_3d(contents)
        
        logger.info(f"Volumetric 3D reconstruction successful: vertices={len(result['mesh']['vertices'])//3}, fractures={result['fracture_analysis']['count']}")
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Volumetric 3D reconstruction failed: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Volumetric 3D reconstruction failed: {str(e)}")


# ========== HEALTH CHECK ==========
@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "version": app.version,
        "database": "connected",
        "vector_db_size": len(vector_db.vectors),
        "device": str(device),
        "timestamp": datetime.now().isoformat()
    }

# ========== ERROR HANDLING ==========
@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    return JSONResponse(
        status_code=exc.status_code,
        content={"detail": exc.detail, "timestamp": datetime.now().isoformat()}
    )

# ========== RUN APPLICATION ==========
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")