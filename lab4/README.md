# Laboratorio 4 — Fundamentos de Aprendizaje por Refuerzo y Gymnasium

Investigación de los fundamentos del aprendizaje por refuerzo (RL) y de la librería
**Gymnasium**, más un módulo exploratorio que ejecuta un **agente aleatorio** en dos
entornos (CartPole-v1 y FrozenLake-v1) y una **política simple no aprendida** para
CartPole-v1, comparando su desempeño.

## Estructura

```
lab4/
├── notebooks/
│   └── lab4_rl_gymnasium.ipynb   # entregable: investigación + módulo exploratorio + gráficas
├── results/
│   └── figures/                  # figuras generadas por el notebook (PNG)
├── README.md
├── requirements.txt
└── .gitignore
```

## Reproducir

```bash
# venv compartido del curso, en la raíz del repo
../venv/bin/pip install -r requirements.txt
../venv/bin/jupyter nbconvert --to notebook --execute --inplace notebooks/lab4_rl_gymnasium.ipynb
```

O bien abrir el notebook en Jupyter y ejecutar todas las celdas de arriba a abajo.

## Contenido del notebook

1. **Fundamentos del aprendizaje por refuerzo** — RL vs. aprendizaje supervisado/no
   supervisado, componentes (agente, entorno, estado, acción, recompensa, política),
   MDP, funciones de valor V(s) y Q(s,a), ecuación de Bellman, exploración vs.
   explotación, taxonomías (episódico/continuo, on/off-policy, model-based/model-free)
   y Q-Learning.
2. **La librería Gymnasium** — qué es y su relación con OpenAI Gym, la API de un `Env`
   (`reset`, `step`, `render`, `close`), los espacios (`Discrete`, `Box`,
   `MultiDiscrete`), un catálogo de entornos y los *wrappers*.
3. **Módulo de prueba** — agente aleatorio en CartPole-v1 y FrozenLake-v1, gráficas de
   recompensa por episodio y una política simple para CartPole-v1 comparada contra el
   agente aleatorio.

<!-- La tabla de resultados se completa al final del laboratorio. -->
