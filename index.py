from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os
from typing import Optional
import uuid
import face_recognition_service  # This would be your face recognition module

app = FastAPI()

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Database models (simplified for example)
class User(BaseModel):
    id: int
    first_name: str
    last_name: Optional[str] = None
    username: Optional[str] = None

class Claim(BaseModel):
    claim_id: str
    user_id: int
    partner_face_embedding: list
    user_face_embedding: list
    created_at: str

# In-memory "database" for demo purposes
db = {
    "users": {},
    "claims": {},
    "face_embeddings": {}  # face_hash: user_id
}

@app.post("/claim")
async def claim_partner(
    user_id: int = Form(...),
    selfie: UploadFile = File(...),
    partner_photo: UploadFile = File(...)
):
    # Save uploaded files temporarily
    selfie_path = f"temp/{uuid.uuid4()}.jpg"
    partner_path = f"temp/{uuid.uuid4()}.jpg"
    
    with open(selfie_path, "wb") as buffer:
        buffer.write(await selfie.read())
    
    with open(partner_path, "wb") as buffer:
        buffer.write(await partner_photo.read())
    
    # Process face recognition
    try:
        # Get face embeddings
        user_embedding = face_recognition_service.get_face_embedding(selfie_path)
        partner_embedding = face_recognition_service.get_face_embedding(partner_path)
        
        # Check if partner is already claimed
        existing_claim = face_recognition_service.find_similar_face(partner_embedding)
        if existing_claim:
            os.remove(selfie_path)
            os.remove(partner_path)
            return {
                "success": False,
                "message": "This person is already claimed by someone else"
            }
        
        # Create claim
        claim_id = str(uuid.uuid4())
        db["claims"][claim_id] = {
            "claim_id": claim_id,
            "user_id": user_id,
            "partner_face_embedding": partner_embedding,
            "user_face_embedding": user_embedding,
            "created_at": str(datetime.now())
        }
        
        # Store face embedding mapping
        face_hash = face_recognition_service.get_face_hash(partner_embedding)
        db["face_embeddings"][face_hash] = user_id
        
        return {"success": True, "claim_id": claim_id}
    
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
    finally:
        # Clean up temp files
        if os.path.exists(selfie_path):
            os.remove(selfie_path)
        if os.path.exists(partner_path):
            os.remove(partner_path)

@app.post("/check")
async def check_status(photo: UploadFile = File(...)):
    # Save uploaded file temporarily
    photo_path = f"temp/{uuid.uuid4()}.jpg"
    
    with open(photo_path, "wb") as buffer:
        buffer.write(await photo.read())
    
    try:
        # Get face embedding
        embedding = face_recognition_service.get_face_embedding(photo_path)
        
        # Check for matches
        match = face_recognition_service.find_similar_face(embedding)
        
        if match:
            return {
                "claimed": True,
                "claimed_by": f"User #{match['user_id']}"  # In real app, get username
            }
        else:
            return {"claimed": False}
    
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
    finally:
        if os.path.exists(photo_path):
            os.remove(photo_path)

@app.get("/user/{user_id}")
async def get_user(user_id: int):
    if user_id in db["users"]:
        return db["users"][user_id]
    raise HTTPException(status_code=404, detail="User not found")

# Telegram WebApp data validation endpoint
@app.post("/validate")
async def validate_telegram_data(data: dict):
    # In a real app, you would validate the Telegram WebApp initData
    # This is a security-critical step to prevent spoofing
    return {"valid": True}

import face_recognition
import numpy as np
from typing import List, Optional
import hashlib

# Threshold for face matching (adjust as needed)
SIMILARITY_THRESHOLD = 0.6

def get_face_embedding(image_path: str) -> List[float]:
    """Extract face embedding from an image"""
    try:
        image = face_recognition.load_image_file(image_path)
        encodings = face_recognition.face_encodings(image)
        
        if len(encodings) == 0:
            raise ValueError("No faces found in the image")
        
        # For simplicity, use the first face found
        return encodings[0].tolist()
    except Exception as e:
        raise ValueError(f"Error processing image: {str(e)}")

def get_face_hash(embedding: List[float]) -> str:
    """Create a hash of the face embedding for storage"""
    return hashlib.sha256(np.array(embedding).tobytes()).hexdigest()

def find_similar_face(target_embedding: List[float]) -> Optional[dict]:
    """
    Check if a similar face exists in the database
    Returns the claim if found, None otherwise
    """
    target_array = np.array(target_embedding)
    
    # In a real app, you would query your database
    # This is a simplified in-memory version
    for claim_id, claim in db["claims"].items():
        stored_array = np.array(claim["partner_face_embedding"])
        
        # Compare faces
        distance = np.linalg.norm(target_array - stored_array)
        similarity = 1 - distance
        
        if similarity > SIMILARITY_THRESHOLD:
            return claim
    
    return None

def compare_faces(embedding1: List[float], embedding2: List[float]) -> bool:
    """Compare two face embeddings"""
    array1 = np.array(embedding1)
    array2 = np.array(embedding2)
    distance = np.linalg.norm(array1 - array2)
    return (1 - distance) > SIMILARITY_THRESHOLD