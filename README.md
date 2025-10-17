# Proyecto 7 · Clasificación Multiclase · Grupo 5

## Descripción del Proyecto

Este proyecto tiene como objetivo desarrollar un modelo de machine learning capaz de resolver un problema del mundo real utilizando algoritmos de clasificación multiclase. A lo largo del proyecto, se aplican técnicas de análisis y visualización de datos, preprocesamiento, construcción de modelos supervisados y evaluación de resultados.

La clasificación multiclase es un enfoque de aprendizaje supervisado en el que cada instancia se asigna a una única clase entre tres o más posibles. A diferencia de la clasificación binaria, este tipo de modelos deben distinguir entre múltiples categorías excluyentes.

Como recurso opcional, se sugiere el uso del dataset "Forest Cover Type Dataset". Sin embargo, se alienta a lxs participantes a elegir datasets auténticos que representen sus propios intereses o problemas relevantes.


## Estructura del proyecto
```
.
├── backend/
│   ├── __init__.py
│   └── notebooks/
│       ├── EDA.ipynb
│       ├── Random_Forest_Model.ipynb
│       ├── forest_covertype_xgboost.ipynb
│       └── __init__.py
├── frontend/
│   └── __init__.py
├── resources/
│   ├── __init__.py
│   └── models/
│       ├── random_forest_model.pkl
│       ├── xgboost_optimized.pkl
│       └── xgboost_scaler.pkl
├── .github/ISSUE_TEMPLATE/
│   ├── bug_request.yml
│   ├── chore_request.yml
│   ├── daily_meeting.yml
│   ├── documentation_request.yml
│   ├── feature_request.yml
│   ├── task_request.yml
│   └── test_request.yml
├── .env.example
├── .gitignore
├── requirements.txt
└── README.md
```

## Estado actual
- **EDA**: Notebook `backend/notebooks/EDA.ipynb` con análisis exploratorio inicial.
- **Modelado**:
  - `Random_Forest_Model.ipynb`: prototipo con Random Forest.
  - `forest_covertype_xgboost.ipynb`: experimento con XGBoost.
- **Artefactos de modelo** en `resources/models/`:
  - `xgboost_optimized.pkl` (modelo grande), `xgboost_scaler.pkl` (preprocesamiento).
  - `random_forest_model.pkl` (placeholder/artefacto inicial).
- **Plantillas de issues** en `.github/ISSUE_TEMPLATE/` para gestión de tareas.

## Requisitos
- Python 3.x
- Dependencias en `requirements.txt` (GitPython, python-dotenv, etc.).

Instalación rápida (entorno virtual recomendado):
```bash
python -m venv .venv
.venv\Scripts\activate  # Windows
pip install -r requirements.txt
```

## Uso (notebooks)
```bash
pip install jupyterlab  # si no lo tienes
jupyter lab
```
Abrir y ejecutar los notebooks en `backend/notebooks/`.

## Próximos pasos sugeridos
- Exponer el mejor modelo como servicio (API en `backend/`).
- Definir flujo de entrenamiento/inferencia reproducible (scripts o pipeline).
- Integrar `frontend/` con la API para inferencia.
- Añadir validación, tests y CI.


## Convenciones
- Ramas: trabajo en feature branches y merge hacia `development` mediante PR/MR.