# main.py
import os
from datetime import datetime, timedelta
from dotenv import load_dotenv
from fastapi import FastAPI, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sqlalchemy import create_engine, Column, Integer, String
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
from passlib.context import CryptContext
from jose import JWTError, jwt
import google.generativeai as genai

# --- 1. CARGAR CONFIGURACIÓN ---
load_dotenv()

# Cargar variables desde .env
DATABASE_URL = os.getenv("DATABASE_URL")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
SECRET_KEY = os.getenv("SECRET_KEY")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

# Configurar IA de Google
genai.configure(api_key=GOOGLE_API_KEY)
model = genai.GenerativeModel('gemini-2.5-flash')

# --- 2. CONFIGURACIÓN DE BASE DE DATOS (SQLAlchemy) ---
# Usará el "sqlite:///./local.db" de tu .env
engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

# --- 3. MODELOS DE DATOS (Pydantic y SQLAlchemy) ---

# Modelo de la tabla de usuarios en la BD
class User(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True, index=True)
    username = Column(String, unique=True, index=True)
    hashed_password = Column(String)

# Modelos Pydantic (para validación de datos de la API)
class UserCreate(BaseModel): # Para registrarse
    username: str
    password: str

class TextRequest(BaseModel): # Para pedir un resumen
    text: str

class Token(BaseModel): # Lo que devolvemos al hacer login
    access_token: str
    token_type: str

class TokenData(BaseModel): # Para decodificar el token
    username: str | None = None

# --- 4. CONFIGURACIÓN DE SEGURIDAD (Hashing y JWT) ---
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto") # Para contraseñas
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token") # Le dice a FastAPI dónde está el login

# Funciones "helper" de seguridad
def verify_password(plain_password, hashed_password):
    return pwd_context.verify(plain_password, hashed_password)

def get_password_hash(password):
    return pwd_context.hash(password)

def create_access_token(data: dict):
    to_encode = data.copy()
    expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

# --- 5. INICIAR LA APP (FastAPI) ---
app = FastAPI()

# --- 6. CONFIGURACIÓN DE CORS (¡¡MUY IMPORTANTE!!) ---
# Esto le da permiso a tu frontend (que corre en otra dirección)
# para que pueda hablar con este backend.
origins = [
    "http://127.0.0.1:5500", # Para VS Code Live Server
    "http://localhost:5500",
    "http://127.0.0.1:8000", # Para tu config.js local
    "null", # Para cuando abres el .html directamente
    "https://julio-csarli.github.io"
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- 7. DEPENDENCIAS (Helpers para la API) ---

# Dependencia para obtener la sesión de la BD
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# Dependencia "Guardián" - Verifica el token y devuelve al usuario
async def get_current_user(token: str = Depends(oauth2_scheme), db: Session = Depends(get_db)):
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="No se pudieron validar las credenciales",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        if username is None:
            raise credentials_exception
        token_data = TokenData(username=username)
    except JWTError:
        raise credentials_exception
    
    user = db.query(User).filter(User.username == token_data.username).first()
    if user is None:
        raise credentials_exception
    return user

# --- 8. LOS ENDPOINTS (Las "Puertas" de tu API) ---

# Crea la base de datos (el archivo local.db) si no existe
Base.metadata.create_all(bind=engine)

@app.get("/")
def read_root():
    return {"status": "El backend del resumidor está vivo"}

@app.post("/register", status_code=status.HTTP_201_CREATED)
def register_user(user: UserCreate, db: Session = Depends(get_db)):
    # Comprobar si el usuario ya existe
    db_user = db.query(User).filter(User.username == user.username).first()
    if db_user:
        raise HTTPException(status_code=400, detail="El nombre de usuario ya existe")
    
    # Hashear la contraseña y crear el nuevo usuario
    hashed_password = get_password_hash(user.password)
    new_user = User(username=user.username, hashed_password=hashed_password)
    db.add(new_user)
    db.commit()
    db.refresh(new_user)
    return {"message": "Usuario creado exitosamente"}


@app.post("/token", response_model=Token)
def login_for_access_token(user_data: UserCreate, db: Session = Depends(get_db)):
    # (Usamos UserCreate también para el login, ya que el frontend envía JSON)
    user = db.query(User).filter(User.username == user_data.username).first()
    
    # Verificar que el usuario exista Y que la contraseña sea correcta
    if not user or not verify_password(user_data.password, user.hashed_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Nombre de usuario o contraseña incorrectos",
        )
    
    # Crear y devolver el "pase" (token)
    access_token = create_access_token(data={"sub": user.username})
    return {"access_token": access_token, "token_type": "bearer"}


@app.post("/summarize")
async def summarize_text(
    request: TextRequest, 
    current_user: User = Depends(get_current_user) # ¡¡EL GUARDIÁN!!
):
    # Si llegamos aquí, el usuario está autenticado (gracias a Depends)
    try:
        prompt = f"Por favor, resume el siguiente texto en español, en un solo párrafo conciso:\n\n{request.text}"
        response = await model.generate_content_async(prompt)
        return {"summary": response.text}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error al contactar la IA: {str(e)}")