# EmoEvent-ES: Clasificación de emociones en tweets (estudio de 3 modelos)

Estudio reproducible sobre **clasificación de emociones** en español usando el dataset **EmoEvent**.  
Se evalúan **tres enfoques**:

1. **M1 — TF-IDF + LinearSVC** (baseline clásico)  
2. **M2 — BETO (BERT español) fine-tuning**  
3. **M3 — Zero-shot NLI** (mDeBERTa-v3 XNLI, sin entrenamiento)

Se reportan métricas **por emoción** y **por evento**, además de **matrices de confusión** y EDA básico.

---

## Dataset

- Repo oficial: `fmplaza/EmoEvent`  
- Idioma: **español**  
- Etiquetas (8): `anger, sadness, joy, disgust, fear, surprise, offensive, other`  
- Splits: `train/dev/test` bajo `splits/es/*.tsv`

> Este repo **no** re-distribuye el dataset. El notebook lo clona automáticamente desde GitHub.


