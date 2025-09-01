# EmoEvent-ES: Clasificación de emociones en tweets (estudio de 3 modelos)

Estudio reproducible sobre **clasificación de emociones** en español usando el dataset **EmoEvent**.  
Se evalúan **tres enfoques**:

1. **M1 — TF-IDF + LinearSVC** (baseline clásico)  
2. **M2 — BETO (BERT español) fine-tuning**  
3. **M3 — Zero-shot NLI** 

Se reportan métricas **por emoción** y **por evento**, además de **matrices de confusión** y EDA básico.

---

## 1) Problema y objetivo

- **Tarea:** multi-clase (una etiqueta por tweet).  
- **Etiquetas (8):** `anger, sadness, joy, disgust, fear, surprise, offensive, other`.  
- **Objetivo:** comparar 3 modelos y **reportar métricas por clase y por evento**, además de análisis cualitativos.

- 
## 2) Requisitos y entorno

- **Python:** 3.10+ (Colab funciona bien)
- **GPU:** opcional, recomendable para M2 y M3  
- **Paquetes:**  
  `transformers datasets evaluate accelerate scikit-learn pandas numpy matplotlib seaborn tqdm emoji unidecode`

## 3) Reproducibilidad
- Semilla fija `SEED=42` (numpy / sklearn / torch / transformers).
- Se guardan CSV y figuras en `outputs/m*/`.
- Versiones clave (Colab): `transformers>=4.55`.
- Subclase de Trainer compatible y:
  - `eval_strategy="epoch"`,  
  - `save_strategy="epoch"`,  
  - `load_best_model_at_end=True`.

---

## 4) EDA (resumen)
- **Clases:** `other` y `joy` concentran la mayoría; `fear`/`disgust`/`surprise` son muy escasas.  
- **Eventos top:** NotreDame, GameOfThrones, SpainElection, Venezuela, ChampionsLeague…  
- **Longitud:** mediana ≈ 22 tokens; emojis/hashtags/URLs frecuentes.  
- Se generan histogramas de longitudes, distribución por clase y mapa evento×clase.

---

## 5) Preprocesamiento
**Dos variantes según el modelo:**

- **Baseline (SVM):**
  - Normalización NFC, minúsculas, colapso de espacios.  
  - Quitar URLs, menciones, RT, #, USER.  
  - Eliminar emojis y stopwords.  
  - Reducir repeticiones de caracteres.

- **BETO / Zero-shot:**
  - Limpieza ligera (conservar emojis y stopwords).  
  - Menciones → `@user`; quitar URLs/RT/# manteniendo el contenido emocional.

---

## 6) Modelos

### M1 — TF-IDF + LinearSVC (baseline)
- Features: word n-grams (1–2) + char n-grams (3–5), `min_df=2`, `sublinear_tf`.  
- Clasificador: `LinearSVC(C=0.5, class_weight='balanced', random_state=42)`.  
- Entrenamiento: selección de `C` por dev; re-entreno en train+dev y evaluación en test.

### M2 — BETO (BERT español, fine-tuning)
- Checkpoint: `dccuchile/bert-base-spanish-wwm-uncased`.  
- Tokenización: `max_length=160`, `DataCollatorWithPadding`.  
- Pérdida: CrossEntropy con pesos por clase.  
- Hiperparámetros:  
  - lr=2e-5, epochs=4  
  - batch train=16, eval=32  
  - weight_decay=0.01, early_stopping  
- Criterio: mejor `f1_macro` en dev.

### M3 — Zero-shot NLI
- Checkpoint: `MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7`.  
- Pipeline: `zero-shot-classification` con hipótesis:  
  *“Este texto expresa {}.”*  
- Verbalizadores: enojo, tristeza, alegría, asco, miedo, sorpresa, lenguaje ofensivo.  
- `other`: no es candidata; se asigna si `score_top < τ`.  
- Barrido de `τ`: optimizado en dev (0.30–0.60) y aplicado en test.

---

## 7) Métricas y artefactos generados
- Por emoción: `per_label.csv` + `m*_f1_per_label.png`.  
- Matriz de confusión (normalizada).  
- Por evento: `per_event.csv` + barras Top/Bottom.  
- Resumen global: `summary.csv` con `f1_macro_test`, `f1_weighted_test` y parámetros.

---

## 8) Resultados (split de test)

| Modelo                    | F1-macro | F1-weighted | Accuracy |
|----------------------------|----------|-------------|----------|
| M1 — TF-IDF + LinearSVC    | 0.193    | 0.355       | 0.367    |
| M2 — BETO (fine-tuning)    | 0.217    | 0.347       | 0.336    |
| M3 — Zero-shot NLI         | 0.156    | 0.280       | 0.268    |

**Lecturas clave:**
- `offensive` y `other` suelen rendir mejor; `joy` mejora con BETO.  
- Clases minoritarias (fear, disgust, surprise) siguen siendo difíciles → desbalance severo.  
- BETO (M2) logra el mejor macro-F1; M3 sirve como baseline “modelo listo” y es sensible a `τ` y a los verbalizadores.

---

## 9) Cómo ejecutar (Colab o local)
1. Abrir `notebooks/01_emoevent_es.ipynb`.  
2. Parte 1 — Instalación y Setup (descarga de EmoEvent y carpetas).  
3. Parte 2 — Limpieza (crea `text_svm` y `text_beto`).  
4. EDA (gráficas).  
5. Modelos: M1 → M2 (GPU recomendada) → M3 (opcional, usa GPU si hay).  
6. Revisar `outputs/m*/` para CSVs y PNGs.  

En Colab, activar GPU: *Runtime → Change runtime type → GPU (T4/V100)* y re-ejecutar desde el inicio.

---

## 10) Consejos de replicación / tuning
- **M1:** `C ∈ {0.25, 0.5, 1, 2}`, char n-grams 3–6, `min_df ∈ {1,2,3}`; probar conservar emojis/stopwords.  
- **M2:** `max_length=192–256`, epochs=3–5, label smoothing 0.05–0.1, sampler balanceado/oversampling, Focal Loss.  
- **M3:** afinar `τ` (0.30–0.40 paso 0.01); verbalizadores alternativos (ej. “insulto / ataque personal / lenguaje ofensivo”, “asombro” para *surprise*).

---

## 11) Limitaciones y trabajo futuro
- Desbalance pronunciado (fear/disgust/surprise): ex

