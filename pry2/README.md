# Proyecto 2: Agente Rainbow DQN para Space Invaders

Proyecto individual del curso **CC3092 Deep Learning y Sistemas Inteligentes**.

Se entrenó un agente de aprendizaje por refuerzo para jugar
`ALE/SpaceInvaders-v5` utilizando una implementación de Rainbow DQN.

## Modelo final

El modelo seleccionado corresponde al checkpoint de **11,750,000 pasos** del
experimento `Rainbow_01_full_12h`.

La selección se realizó comparando los mejores checkpoints sobre 100 episodios
nuevos, organizados en 20 simulaciones de competencia de cinco episodios.

| Modelo | Promedio de recompensa | Máximo promedio por grupo de 5 | Victorias |
|---|---:|---:|---:|
| Rainbow 11.75M | 1398.3 | 1936.0 | 14/20 |
| Rainbow 10.75M | 1340.0 | 1889.0 | 6/20 |

Ensayo final de cinco episodios:

- Promedio: **1500.0**
- Máximo: **2190.0**
- Mínimo: **1145.0**
- Recompensa acumulada: **7500.0**

## Algoritmo

La implementación Rainbow combina:

- Double DQN
- Dueling Network
- Noisy Networks
- Prioritized Experience Replay
- Retornos n-step
- Distribución categórica C51
- Red objetivo

## Preprocesamiento

- Entorno: `ALE/SpaceInvaders-v5`
- Acciones pegajosas: `repeat_action_probability=0.25`
- Espacio mínimo de 6 acciones
- Frames RGB convertidos a escala de grises
- Resolución de entrada: `84 x 84`
- Frame skip: 4
- Apilamiento de 4 frames
- Observación final: `(4, 84, 84)`
- Reward clipping solamente durante entrenamiento
- Recompensa real durante evaluación
- Política greedy durante evaluación
- Las vidas se conservan dentro del mismo episodio durante evaluación

## Descargar los pesos

El modelo final se publica como archivo de un GitHub Release porque supera el
límite de 100 MB para archivos normales de Git.

Con GitHub CLI:

```bash
mkdir -p models/competencia

gh release download pry2-rainbow-v1 \
  --repo ecarcamo/CC3092-DEEP-LEARNING \
  --pattern "SpaceInvaders_Rainbow_FINAL.pt" \
  --dir models/competencia
```

El archivo debe quedar en:

```text
models/competencia/SpaceInvaders_Rainbow_FINAL.pt
```

SHA-256 esperado:

```text
27162a02f57dcff282e2264c745d775ff1a8b8f97771cb6c42582308c7657eaf
```

## Instalación

Se recomienda Python 3.11 y una GPU compatible con CUDA.

```bash
python3.11 -m venv .venv-qrdqn
source .venv-qrdqn/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

La instalación de PyTorch puede ajustarse a la versión de CUDA disponible
siguiendo las instrucciones oficiales de PyTorch.

## Evaluación para la competencia

Ejecutar cinco episodios:

```bash
python scripts/competencia_final.py \
  --episodes 5 \
  --seed 2026
```

El programa muestra:

- Recompensa de cada episodio
- Promedio
- Máximo
- Mínimo
- Recompensa acumulada

La competencia utiliza como puntaje el mejor de los cinco episodios.

## Generación del video

```bash
python scripts/generar_video_final.py
```

El resultado queda en:

```text
videos/SpaceInvaders_Rainbow_FINAL.mp4
```

El video incluido reproduce un episodio completo de **2190 puntos**.

## Entrenamiento

Entrenamiento Rainbow desde cero:

```bash
python scripts/train_rainbow_local.py \
  --experiment Rainbow_01_full_12h \
  --max-steps 30000000 \
  --hours 12
```

El script guarda checkpoints, evaluaciones y el mejor modelo encontrado.

## Experimentos realizados

Durante el proyecto se evaluaron:

1. Baseline aleatorio.
2. Baseline de disparo constante.
3. DQN de 500,000 pasos.
4. DQN de 2,000,000 pasos.
5. QR-DQN de 8,000,000 pasos.
6. Rainbow DQN con semilla 42.
7. Rainbow DQN con semilla 123.

Rainbow DQN obtuvo el mejor desempeño y fue seleccionado como agente final.

## Archivos principales

```text
scripts/
├── competencia_final.py
├── generar_video_final.py
├── select_competition_model.py
├── train_rainbow_local.py
├── evaluate_champions_local.py
├── evaluate_finalists_local.py
├── evaluate_qrdqn_local.py
└── resume_qrdqn_local.py

videos/
├── SpaceInvaders_Rainbow_FINAL.mp4
└── SpaceInvaders_Rainbow_FINAL.json
```

## Hardware utilizado

- NVIDIA Tesla T4 en Google Colab
- NVIDIA GeForce RTX 4050 Laptop GPU para entrenamiento local
- PyTorch con CUDA

## Reproducibilidad

Para reproducir correctamente la evaluación deben conservarse sin cambios:

- El entorno `ALE/SpaceInvaders-v5`.
- Los wrappers y el preprocesamiento descritos anteriormente.
- La arquitectura almacenada junto con el checkpoint.
- La política greedy sin exploración durante evaluación.
- La recompensa real, sin clipping, durante evaluación.

El archivo `scripts/competencia_final.py` carga la configuración guardada con
el modelo y ejecuta el protocolo de cinco episodios utilizado en la
competencia.
