# Clasificador de emociones en texto

Servicio de clasificación de emociones en texto en inglés, construido con DistilBERT y desplegado como API REST y como demo pública.

**[Probar la demo](https://huggingface.co/spaces/Hanser9909/clasificador-emociones)** · **[Modelo en Hugging Face](https://huggingface.co/Hanser9909/emociones-distilbert)**

El modelo clasifica un texto en seis emociones: *sadness, joy, love, anger, fear* y *surprise*. Está ajustado sobre el dataset [`dair-ai/emotion`](https://huggingface.co/datasets/dair-ai/emotion) (16.000 ejemplos de entrenamiento, 2.000 de validación, 2.000 de test).

---

## Resultados

| Modelo | Accuracy (test) | Parámetros | Tamaño |
|---|---|---|---|
| `bert-base-uncased` | 0.9265 | 109M | 439 MB |
| `distilbert-base-uncased` | **0.9250** | **66M** | **266 MB** |

DistilBERT pierde 0,15 puntos de accuracy con un 40% menos de parámetros. Por esa razón es el modelo que sirve la demo.

### Métricas por clase (DistilBERT)

| Clase | Precision | Recall | F1 | Soporte |
|---|---|---|---|---|
| sadness | 0.959 | 0.976 | 0.968 | 581 |
| joy | 0.967 | 0.928 | 0.947 | 695 |
| love | 0.793 | 0.893 | 0.840 | 159 |
| anger | 0.941 | 0.920 | 0.930 | 275 |
| fear | 0.890 | 0.871 | 0.880 | 224 |
| surprise | 0.680 | 0.773 | 0.723 | 66 |

**Sobre el rendimiento en `surprise`:** su F1 de 0,723 está muy por debajo del resto, y la causa es el desbalanceo del dataset. La clase tiene 66 ejemplos en test frente a los 695 de `joy`, una proporción de 1 a 10. La matriz de confusión muestra que la mayoría de sus errores se confunden con `fear`, lo cual es razonable: ambas emociones comparten vocabulario de sobresalto. El accuracy global de 0,926 oculta este problema, que solo se ve en el macro-F1 (0,881).

Mejorar esta clase requeriría reponderar la función de pérdida o aumentar los ejemplos, no un modelo más grande.

---

## Decisiones de ingeniería

### Reducción de la imagen Docker: de 4,62 GB a 438 MB

La primera versión del contenedor pesaba 4,62 GB. El análisis con `docker history` localizó el origen en dos puntos.

**PyTorch con CUDA (−3,8 GB).** La instalación por defecto incluye las librerías de NVIDIA completas (cuDNN, cuBLAS y los kernels compilados para cada arquitectura de GPU). El servicio hace inferencia en CPU, así que nada de eso se utiliza. Se resolvió apuntando `torch` al índice de solo-CPU de PyTorch:

```toml
[tool.uv.sources]
torch = [{ index = "pytorch-cpu" }]

[[tool.uv.index]]
name = "pytorch-cpu"
url = "https://download.pytorch.org/whl/cpu"
explicit = true
```

**El modelo dentro de la imagen (−405 MB).** Los pesos se copiaban al contenedor en tiempo de construcción. Se movieron a Hugging Face Hub y ahora se descargan al arrancar, con caché local. El repositorio de código queda ligero y el artefacto tiene su propio registro versionado.

| Versión | Tamaño |
|---|---|
| Inicial (CUDA + modelo embebido) | 4,62 GB |
| Solo-CPU | 843 MB |
| Solo-CPU + modelo externo | 438 MB |

### Elección del modelo según las restricciones de despliegue

El plan gratuito de Render ofrece 512 MB de RAM. BERT base ocupa unos 440 MB solo en pesos, a los que hay que sumar el runtime de PyTorch y el intérprete de Python.

El cambio a DistilBERT redujo el modelo a 266 MB con una pérdida de accuracy de 0,15 puntos. Aun así, el servicio siguió excediendo el límite: PyTorch cargado consume entre 200 y 300 MB antes de instanciar ningún modelo, de modo que **512 MB no son suficientes para servir un transformer en PyTorch**, independientemente del tamaño del modelo.

La demo se desplegó finalmente en Hugging Face Spaces, que ofrece 16 GB. La API REST de este repositorio se mantiene desplegable en cualquier entorno con 1 GB o más.

### Carga del modelo en memoria

El modelo se carga una sola vez, no en cada petición:

```python
@lru_cache(maxsize=1)
def cargar_modelo():
    ...
```

Además, FastAPI lo carga durante el arranque mediante `lifespan`, antes de aceptar tráfico. Sin esto, la primera petición de cada despliegue pagaría varios segundos de latencia.

---

## Arquitectura

```
src/emociones/
├── data.py      Carga y tokenización del dataset
├── model.py     Definición del clasificador (AutoModel + capa lineal)
├── train.py     Entrenamiento, evaluación y guardado
├── predict.py   Carga del modelo e inferencia
└── api.py       API REST con FastAPI
```

El clasificador toma la representación del token `[CLS]` de la última capa oculta, le aplica dropout y la proyecta a seis clases con una capa lineal.

### Endpoints

| Método | Ruta | Descripción |
|---|---|---|
| `GET` | `/health` | Comprobación de estado |
| `POST` | `/predict` | Clasifica un texto |
| `GET` | `/docs` | Documentación interactiva (OpenAPI) |

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"texto": "i feel so tired after this shift"}'
```

```json
{
  "emocion": "sadness",
  "confianza": 0.9716
}
```

La validación de entrada se delega en Pydantic: un texto vacío o de más de 1.000 caracteres devuelve un 422 sin comprobaciones manuales.

---

## Cómo ejecutarlo

### Con Docker

```bash
docker build -t emociones-api .
docker run -p 8000:8000 emociones-api
```

El contenedor descarga el modelo de Hugging Face al arrancar. Documentación en `http://localhost:8000/docs`.

### En local

```bash
uv sync
uv run uvicorn emociones.api:app --reload
```

### Reentrenar

El proyecto está configurado para inferencia en CPU. Para entrenar con GPU hay que apuntar `torch` al índice de CUDA en `pyproject.toml`:

```toml
url = "https://download.pytorch.org/whl/cu128"
```

```bash
uv lock && uv sync
uv run python -m emociones.train
```

Los hiperparámetros están en `config.yaml`. El entrenamiento tarda unos 10 minutos en una RTX 3060 Laptop.

---

## Stack

Python 3.12 · PyTorch · Transformers · FastAPI · Pydantic · Docker · uv · Hugging Face Hub · Gradio

---

## Limitaciones conocidas

- **Solo funciona en inglés.** El modelo base y el dataset lo son; con texto en español las predicciones no son fiables.
- **`surprise` tiene un rendimiento bajo** por desbalanceo del dataset (ver más arriba).
- **El modelo se descarga en cada arranque de contenedor.** La caché vive dentro del contenedor y no sobrevive a su ciclo de vida. Descargarlo durante el `build` eliminaría esa latencia a costa de una imagen más pesada.
- **La demo puede tardar en la primera petición.** Está desplegada en hardware compartido bajo demanda.
- **Las dependencias de entrenamiento e inferencia no están separadas.** Alternar entre CUDA y CPU requiere editar `pyproject.toml` a mano; lo correcto serían grupos de dependencias independientes.
