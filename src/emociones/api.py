from contextlib import asynccontextmanager

from fastapi import FastAPI
from pydantic import BaseModel, Field

from emociones.predict import cargar_modelo, predecir


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Carga el modelo al arrancar, antes de aceptar peticiones."""
    cargar_modelo()
    yield


app = FastAPI(
    title="Clasificador de emociones",
    description="Clasifica texto en inglés en seis emociones usando BERT.",
    version="0.1.0",
    lifespan=lifespan,
)


class PeticionPrediccion(BaseModel):
    texto: str = Field(min_length=1, max_length=1000)


class RespuestaPrediccion(BaseModel):
    emocion: str
    confianza: float


@app.get("/health")
def health() -> dict:
    return {"estado": "ok"}


@app.post("/predict", response_model=RespuestaPrediccion)
def predict(peticion: PeticionPrediccion) -> dict:
    return predecir(peticion.texto)