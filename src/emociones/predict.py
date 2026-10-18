from functools import lru_cache
from pathlib import Path

import torch
from transformers import AutoTokenizer

from emociones.model import BertClassifier

MODEL_DIR = Path("models/emociones-bert")
MODEL_NAME = "bert-base-uncased"
ETIQUETAS = ["sadness", "joy", "love", "anger", "fear", "surprise"]


@lru_cache(maxsize=1)
def cargar_modelo():
    """Carga el tokenizer y el modelo entrenado. Solo se ejecuta la primera vez."""
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)

    modelo = BertClassifier(
        num_classes=len(ETIQUETAS),
        model_name=MODEL_NAME,
    )
    pesos = torch.load(MODEL_DIR / "modelo.pt", map_location="cpu")
    modelo.load_state_dict(pesos)
    modelo.eval()

    return tokenizer, modelo


def predecir(texto: str) -> dict:
    """Devuelve la emoción predicha y su probabilidad."""
    tokenizer, modelo = cargar_modelo()

    entradas = tokenizer(
        texto,
        return_tensors="pt",
        truncation=True,
        max_length=128,
    )

    with torch.no_grad():
        logits = modelo(
            input_ids=entradas["input_ids"],
            attention_mask=entradas["attention_mask"],
        )

    probabilidades = torch.softmax(logits, dim=-1)[0]
    indice = int(probabilidades.argmax())

    return {
        "emocion": ETIQUETAS[indice],
        "confianza": round(float(probabilidades[indice]), 4),
    }


if __name__ == "__main__":
    import sys

    print(predecir(sys.argv[1]))