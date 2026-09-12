import yaml
import numpy as np
import torch
import torch.nn as nn

from sklearn.metrics import classification_report, confusion_matrix

from emociones.data import cargar_datos, tokenizar_datos
from emociones.model import BertClassifier


# --------------------------------------------------
# CONFIGURACIÓN
# --------------------------------------------------

with open("config.yaml", "r", encoding="utf-8") as f:
    config = yaml.safe_load(f)

SEED = 42
PATIENCE = 2


# --------------------------------------------------
# SEMILLAS
# --------------------------------------------------

np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)


# --------------------------------------------------
# DISPOSITIVO
# --------------------------------------------------

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("PyTorch:", torch.__version__)
print("Dispositivo:", device)


# --------------------------------------------------
# CARGAR DATOS
# --------------------------------------------------

print("\nCargando dataset...")

(
    df_train,
    df_val,
    df_test,
    class_names,
    num_classes,
) = cargar_datos(config)

print("Número de clases:", num_classes)
print("Clases:", class_names)

print(
    "Tamaños -> train:",
    df_train.shape,
    "| val:",
    df_val.shape,
    "| test:",
    df_test.shape
)


# --------------------------------------------------
# TOKENIZACIÓN
# --------------------------------------------------

print("\nTokenizando datos...")

train_dl, val_dl, test_dl, tokenizer = tokenizar_datos(
    df_train,
    df_val,
    df_test,
    config
)


# --------------------------------------------------
# MODELO
# --------------------------------------------------

print("\nCreando modelo...")

model = BertClassifier(
    num_classes,
    model_name=config["model"]["name"]
).to(device)

n_params = sum(
    p.numel()
    for p in model.parameters()
)

print(f"Número total de parámetros: {n_params:,}")


# --------------------------------------------------
# LOSS Y OPTIMIZADOR
# --------------------------------------------------

criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=config["training"]["learning_rate"]
)


# --------------------------------------------------
# FUNCIÓN DE EVALUACIÓN
# --------------------------------------------------

def evaluar(model, dataloader, criterion):

    model.eval()

    total_loss = 0.0
    correct = 0
    n = 0

    with torch.no_grad():

        for input_ids, attention_mask, labels in dataloader:

            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            labels = labels.to(device)

            logits = model(
                input_ids,
                attention_mask
            )

            loss = criterion(
                logits,
                labels
            )

            total_loss += (
                loss.item() * labels.size(0)
            )

            correct += (
                logits.argmax(dim=1) == labels
            ).sum().item()

            n += labels.size(0)

    return total_loss / n, correct / n


# --------------------------------------------------
# ENTRENAMIENTO
# --------------------------------------------------

history = {
    "train_loss": [],
    "val_loss": [],
    "train_acc": [],
    "val_acc": []
}

best_val_loss = float("inf")
best_state = None
epochs_sin_mejora = 0


for epoch in range(1, config["training"]["epochs"] + 1):

    print(f"\nEpoch {epoch}/{config['training']['epochs']}")

    model.train()

    total_loss = 0.0
    correct = 0
    n = 0

    for input_ids, attention_mask, labels in train_dl:

        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        logits = model(
            input_ids,
            attention_mask
        )

        loss = criterion(
            logits,
            labels
        )

        loss.backward()

        optimizer.step()

        total_loss += (
            loss.item() * labels.size(0)
        )

        correct += (
            logits.argmax(dim=1) == labels
        ).sum().item()

        n += labels.size(0)

    train_loss = total_loss / n
    train_acc = correct / n

    val_loss, val_acc = evaluar(
        model,
        val_dl,
        criterion
    )

    history["train_loss"].append(train_loss)
    history["val_loss"].append(val_loss)
    history["train_acc"].append(train_acc)
    history["val_acc"].append(val_acc)

    print(
        f"train_loss={train_loss:.4f} "
        f"acc={train_acc:.4f} | "
        f"val_loss={val_loss:.4f} "
        f"acc={val_acc:.4f}"
    )

    # Early stopping

    if val_loss < best_val_loss:

        best_val_loss = val_loss

        best_state = {
            k: v.cpu().clone()
            for k, v in model.state_dict().items()
        }

        epochs_sin_mejora = 0

    else:

        epochs_sin_mejora += 1

        if epochs_sin_mejora >= PATIENCE:

            print(
                f"Early stopping en la época {epoch}"
            )

            break


# --------------------------------------------------
# RECUPERAR MEJORES PESOS
# --------------------------------------------------

if best_state is not None:

    model.load_state_dict(best_state)

    print(
        "\nRestaurados los pesos "
        "con mejor val_loss."
    )


# --------------------------------------------------
# TEST
# --------------------------------------------------

test_loss, test_acc = evaluar(
    model,
    test_dl,
    criterion
)

print(f"\nTest loss: {test_loss:.4f}")
print(f"Test accuracy: {test_acc:.4f}")


# --------------------------------------------------
# PREDICCIONES
# --------------------------------------------------

model.eval()

preds = []

with torch.no_grad():

    for input_ids, attention_mask, labels in test_dl:

        logits = model(
            input_ids.to(device),
            attention_mask.to(device)
        )

        preds.append(
            logits.argmax(dim=1)
            .cpu()
            .numpy()
        )

y_pred = np.concatenate(preds)

print("\nInforme de clasificación:")

print(
    classification_report(
        df_test["label"].values,
        y_pred,
        target_names=class_names,
        digits=3
    )
)


# --------------------------------------------------
# MATRIZ DE CONFUSIÓN
# --------------------------------------------------

cm = confusion_matrix(
    df_test["label"].values,
    y_pred
)

print("\nMatriz de confusión:")
print(cm)