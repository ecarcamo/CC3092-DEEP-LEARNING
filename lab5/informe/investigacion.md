# Laboratorio 5 — Investigación

**CC3092 · Deep Learning y Sistemas Inteligentes**
Agentes en el Arcade Learning Environment (ALE): Space Invaders

> Documento fuente de la investigación. Cubre los puntos 1 y 2 del enunciado. De aquí
> se condensa el informe en PDF (máximo 3 páginas) y las celdas de texto del notebook.
> Todos los valores numéricos de este documento fueron **verificados ejecutando el
> entorno** en `lab5/.venv` (gymnasium 1.3.0, ale-py 0.12.1), no copiados de la documentación.

---

## 1. El Arcade Learning Environment (ALE)

### 1.1 ¿Qué es ALE y qué problema resuelve?

El **Arcade Learning Environment** es una plataforma de evaluación que convierte los juegos
originales de la consola **Atari 2600** en entornos de aprendizaje por refuerzo con una
interfaz uniforme. Fue presentado por Bellemare, Naddaf, Veness y Bowling en 2013 (*The
Arcade Learning Environment: An Evaluation Platform for General Agents*, JAIR) y revisado
por Machado et al. en 2018 (*Revisiting the Arcade Learning Environment*, JAIR).

El problema que resuelve es **metodológico**, no técnico. Antes de ALE, cada investigador
evaluaba su algoritmo en un dominio construido por él mismo (un laberinto, un péndulo, un
mundo de rejilla). Eso tiene dos defectos graves:

1. **Sesgo del diseñador.** Quien diseña el algoritmo diseña también la prueba, así que es
   fácil —incluso sin querer— ajustar el entorno a las fortalezas del método.
2. **Falta de comparabilidad.** Dos artículos con dominios distintos no se pueden comparar,
   y no hay forma de saber si un método generaliza o solo funciona en su propio problema.

ALE ofrece en cambio un conjunto de **más de 100 tareas diseñadas por terceros** —programadores
de videojuegos de los años 80, sin ninguna intención de servir como *benchmark* de IA— que son
diversas (disparos, plataformas, deportes, laberintos), no triviales y comparten exactamente la
misma interfaz: el agente recibe píxeles y devuelve una de 18 acciones posibles del joystick.
Esto permite plantear la pregunta de la **inteligencia general de dominio**: ¿puede un mismo
agente, con los mismos hiperparámetros, aprender a jugar decenas de juegos distintos sin
conocimiento específico de cada uno?

ALE es el banco de pruebas donde se validaron los hitos del RL profundo: **DQN** (Mnih et al.,
2013 y Nature 2015), Double DQN, Dueling DQN, A3C, Rainbow, IMPALA, R2D2 y Agent57.

### 1.2 Relación con el emulador Stella y con los ROMs originales

**Stella** es un emulador libre y maduro del Atari 2600, escrito para que las personas jueguen
los cartuchos originales en una computadora. ALE **no reimplementa los juegos**: se construye
*encima* de Stella y ejecuta los mismos ROMs comerciales, con la misma lógica, la misma
física y los mismos errores que el cartucho original de 1980.

Lo que ALE añade sobre Stella es la capa que lo convierte en un entorno de RL:

| Capa | Responsabilidad |
|---|---|
| **Stella** | Emula el hardware: CPU 6507, chip gráfico TIA, 128 bytes de RAM. Produce la imagen de 210×160 y avanza el estado cuadro a cuadro. |
| **ALE** | Expone una API de agente: `reset`, `act(action)`, lectura de pantalla y de RAM, señal de terminación y **extracción de la puntuación** desde la RAM del juego. |
| **ale-py / Gymnasium** | Envuelve ALE en la API estándar `Env` de Gymnasium (`reset` / `step` / `render` / `close`) y registra los identificadores de entorno. |

Tres detalles importantes de esa capa intermedia:

- **Elimina la interfaz humana.** No hay ventana ni sonido obligatorios: el entorno corre
  *headless* y mucho más rápido que en tiempo real, que es lo que hace viable entrenar con
  millones de cuadros.
- **Define dónde termina un episodio.** El cartucho original nunca "termina": vuelve a la
  pantalla de inicio. ALE incorpora, juego por juego, la condición de fin (típicamente perder
  todas las vidas).
- **Define la recompensa.** El ROM no emite recompensas; solo dibuja un marcador en pantalla.
  ALE lee la puntuación directamente de una dirección conocida de la RAM del juego y entrega
  como recompensa la **diferencia de puntuación entre dos pasos consecutivos**.

Desde **ale-py 0.9 los ROMs vienen incluidos** en el paquete, con licencia resuelta. Ya no hace
falta `AutoROM` ni descargar cartuchos por separado, como sí ocurría en versiones anteriores
(este es el paso donde más se atasca la gente que sigue tutoriales viejos).

### 1.3 Variantes del mismo juego: sufijos y parámetros

Un mismo ROM da lugar a muchos entornos distintos. En la instalación de este laboratorio, los
identificadores registrados para Space Invaders son exactamente estos:

| Identificador | `frameskip` | `repeat_action_probability` | `full_action_space` |
|---|---|---|---|
| `SpaceInvaders-v0` | `(2, 5)` aleatorio | 0.25 | False |
| `SpaceInvaders-v4` | `(2, 5)` aleatorio | 0.0 | False |
| `SpaceInvadersNoFrameskip-v0` | 1 | 0.25 | False |
| `SpaceInvadersNoFrameskip-v4` | 1 | 0.0 | False |
| **`ALE/SpaceInvaders-v5`** | **4** | **0.25** | **False** |

Los `v0`/`v4` son los identificadores heredados de OpenAI Gym y se conservan solo por
compatibilidad. El namespace **`ALE/…-v5`** es el actual y sus valores por defecto implementan
las recomendaciones de evaluación de Machado et al. (2018), que es lo que hoy se considera el
protocolo correcto para publicar resultados.

Los parámetros que definen cada variante:

**`frameskip`** — cuántos cuadros del emulador se ejecutan por cada acción del agente.
Un entero `k` repite la acción durante `k` cuadros; una tupla `(2, 5)` muestrea `k`
uniformemente en cada paso (así introducía Gym estocasticidad en `v0`/`v4`, antes de que
existieran las *sticky actions*). `ALE/SpaceInvaders-v5` usa `frameskip=4`.

**`repeat_action_probability`** (*sticky actions*) — con probabilidad *p*, el entorno **ignora
la acción nueva y repite la anterior**. En `v5`, *p* = 0.25. Su razón de ser: el Atari 2600 es
**determinista**, así que sin ninguna fuente de aleatoriedad un agente puede "resolver" un juego
memorizando una **secuencia fija de acciones en lazo abierto**, sin mirar la pantalla. Machado
et al. demostraron que algoritmos triviales de búsqueda (*the Brute*) superaban así a agentes de
RL sofisticados —un resultado que no mide inteligencia, sino memorización—. Las *sticky actions*
rompen esa estrategia sin alterar la dinámica del juego y obligan al agente a ser **reactivo**.

**`full_action_space`** — el joystick del 2600 admite 18 combinaciones legales (8 direcciones ×
{disparo, sin disparo}, más `NOOP` y `FIRE`). Con `full_action_space=False` (valor por defecto
en `v5`) el entorno expone solo el **conjunto mínimo** relevante para ese juego; en Space
Invaders son 6, porque el cañón solo se mueve horizontalmente y `UP`/`DOWN` no hacen nada. Con
`True` se exponen las 18 en todos los juegos: es lo que se necesita para un agente **multijuego**
con una sola cabeza de salida compartida. La contrapartida es que aprender con 18 acciones,
12 de las cuales son inertes, es más lento.

**Sufijo `-ram` / `obs_type`** — cambia *qué* observa el agente, no la dinámica. En ale-py 0.12
los identificadores `-ram` fueron retirados y se usa el argumento `obs_type`:

```python
gym.make("ALE/SpaceInvaders-v5", obs_type="rgb")        # Box(0, 255, (210, 160, 3), uint8)
gym.make("ALE/SpaceInvaders-v5", obs_type="grayscale")  # Box(0, 255, (210, 160), uint8)
gym.make("ALE/SpaceInvaders-v5", obs_type="ram")        # Box(0, 255, (128,), uint8)
```

Un último parámetro relevante: **`max_num_frames_per_episode = 108_000`** en `v5`. A 60 cuadros
por segundo son **30 minutos** de juego; con `frameskip=4` equivalen a 27 000 pasos del agente.
Es lo que produce `truncated=True` y evita episodios infinitos en agentes que aprenden a
sobrevivir sin avanzar.

### 1.4 ¿Por qué importa el *frame skipping*?

El Atari 2600 corre a 60 cuadros por segundo. Sin *frame skipping*, un agente tendría que
decidir 60 veces por segundo, y esto es un problema por tres razones:

**Velocidad de simulación.** El costo dominante no es emular el juego (Stella es muy rápido),
sino la **inferencia de la red neuronal** en cada decisión. Con `frameskip=4` el agente hace
una cuarta parte de las pasadas hacia adelante para la misma cantidad de cuadros emulados:
la simulación se acelera de forma prácticamente lineal en `k`. En un entrenamiento de 50
millones de cuadros esto es la diferencia entre días y semanas.

**Longitud del episodio y asignación de crédito.** Con `frameskip=4`, un episodio de 1 800
cuadros se convierte en 450 pasos. Menos pasos entre la acción y su consecuencia significa
**menos descuento acumulado** y menos pasos por los que propagar la recompensa hacia atrás: el
problema de asignación de crédito se vuelve considerablemente más fácil.

**Escala temporal adecuada.** Las decisiones útiles en Space Invaders no cambian cada 16
milisegundos; un humano reacciona en torno a 200 ms. `k = 4` da 15 decisiones por segundo, una
escala razonable para el juego, y además repetir la acción varios cuadros produce movimientos
**más sostenidos y coherentes** que alternar entre acciones contradictorias 60 veces por segundo.

También tiene costos, y por eso `k` es un hiperparámetro real:

- Con `k` demasiado grande el agente **pierde control fino** y puede no reaccionar a eventos
  rápidos (un proyectil enemigo que cruza la pantalla en pocos cuadros).
- Con `k` demasiado pequeño el entrenamiento se vuelve caro sin ganar capacidad de control.
- Cambiar `k` **cambia el horizonte efectivo** que representa un mismo factor de descuento γ,
  así que los resultados entre configuraciones distintas no son directamente comparables.

Un detalle técnico ligado al *frame skipping*: en el Atari 2600 varios sprites se dibujan en
**cuadros alternos** (limitación del chip TIA), lo que produce parpadeo —Space Invaders es un
caso de manual—. Si uno se queda solo con el último cuadro del salto, algunos objetos pueden
sencillamente no aparecer. Por eso el preprocesamiento estándar toma el **máximo píxel a píxel
de los dos últimos cuadros** del salto en lugar del último.

**Advertencia práctica:** el salto de cuadros puede aplicarse dos veces sin darse cuenta. Si se
va a usar `AtariPreprocessing` (que implementa su propio `frame_skip=4` con *max-pooling*), el
entorno base debe crearse con `frameskip=1`, o el salto efectivo será de 16 cuadros.

**Medición propia del parpadeo.** Ejecutando 1 200 cuadros del juego con la acción `NOOP` —es
decir, sin disparar nunca, de modo que todo proyectil visible es enemigo— y contando en cuántas
observaciones aparece el color de los proyectiles en la franja entre la formación y el cañón:

| Configuración | Observaciones con proyectil visible |
|---|---|
| `frameskip=1` | **346 de 1 200** |
| `frameskip=4` (defecto de la v5) | **0 de 300** |

Con el salto por defecto los proyectiles **no aparecen nunca**: se dibujan justo en los cuadros
que el salto descarta. No es que se vean poco, es que son invisibles. Esto tiene una consecuencia
que va más allá de lo estético: un agente reactivo que trabaje sobre la observación por defecto
**no puede esquivar**, porque las bombas enemigas no existen para él. Es la justificación
empírica, y no meramente teórica, del *max-pooling* de los dos últimos cuadros.

### 1.5 Space Invaders: mecánica, objetivo y recompensa

**Origen.** Space Invaders es un arcade de Taito (1978) que Atari adaptó al 2600 en 1980. Fue
la primera "aplicación estrella" de la consola y multiplicó sus ventas.

**Mecánica y objetivo.** El jugador controla un cañón láser en la parte inferior de la pantalla
que solo puede **moverse horizontalmente y disparar hacia arriba**. Enfrente avanza una
formación rectangular de invasores que se desplaza lateralmente y **desciende un escalón cada
vez que toca un borde**, mientras deja caer proyectiles. El objetivo es destruir toda la
formación antes de que llegue abajo. Tres elementos definen la dificultad:

- La formación **acelera conforme quedan menos invasores**: el final de cada oleada es la parte
  más difícil.
- Hay **búnkeres destructibles** que protegen al cañón, pero se erosionan tanto con los disparos
  enemigos como con los propios.
- Cada cierto tiempo cruza por arriba una **nave nodriza** que otorga una bonificación mayor
  que cualquier invasor normal.

En la versión de ALE el jugador dispone de **3 vidas** (verificado: `info["lives"] == 3` al
reiniciar). Se pierde una vida al recibir un impacto, y el episodio **termina cuando se agotan
las tres** (verificado: `terminated=True` con `info["lives"] == 0`).

**Traducción de la puntuación a recompensa.** El juego original solo incrementa un marcador en
pantalla. ALE lee esa puntuación de la RAM y define:

$$r_t = \text{puntuación}(t) - \text{puntuación}(t-1)$$

Consecuencias para el diseño del agente:

- La recompensa es **no negativa**: se suma al destruir un invasor o la nave nodriza, y **nunca
  se resta al morir**. Perder una vida no produce castigo directo; su costo es implícito, por
  la recompensa futura que se deja de obtener. Por eso muchas implementaciones añaden el wrapper
  `terminal_on_life_loss` durante el entrenamiento, que trata cada vida como un episodio.
- Es **escalonada y relativamente dispersa**: la mayoría de pasos dan recompensa 0, y solo los
  disparos acertados producen un salto. Los invasores de las filas superiores valen más puntos
  que los de las inferiores, así que el valor de la recompensa depende de *qué* se destruye.
- Su **escala es arbitraria y distinta en cada juego** (decenas de puntos aquí, miles en otros
  títulos). Como no se puede usar la misma tasa de aprendizaje con magnitudes tan dispares, el
  protocolo estándar de DQN **recorta la recompensa a su signo**, `{-1, 0, +1}`, para entrenar.
  Nótese que con esto se pierde la información de que un invasor vale más que otro; es un
  compromiso consciente a favor de la estabilidad.
- El **retorno del episodio** (suma de recompensas sin descontar) coincide exactamente con la
  puntuación final de la partida, que es la métrica que se reporta en la literatura.

*Referencia medida en este laboratorio:* sobre 10 episodios, un agente **aleatorio** obtiene
un retorno medio de **123.5 puntos** (mínimo 30, máximo 235) sobreviviendo 461 pasos de media.
Ese es el punto de partida contra el que se compara cualquier agente posterior.

---

## 2. Espacios de observación y acción en entornos Atari

### 2.1 Observación por defecto vs. CartPole-v1

| | `ALE/SpaceInvaders-v5` | `CartPole-v1` |
|---|---|---|
| Espacio | `Box(0, 255, (210, 160, 3), uint8)` | `Box(low, high, (4,), float32)` |
| Naturaleza | Imagen RGB de la pantalla | Vector de estado físico |
| Nº de valores | **100 800** | **4** |
| Significado | Ninguno explícito: intensidades de color | `[posición, velocidad, ángulo, velocidad angular]` |
| ¿Markoviano? | **No** con un solo cuadro | **Sí** |
| Acciones | `Discrete(6)` | `Discrete(2)` |
| Límite de episodio | 108 000 cuadros (≈27 000 pasos) | 500 pasos |

La diferencia de **25 200×** en dimensionalidad no es lo más importante. Lo verdaderamente
distinto es que en CartPole el estado **ya viene interpretado**: alguien decidió que las cuatro
variables relevantes son esas y las entrega medidas. En Atari el agente recibe lo mismo que un
humano —píxeles crudos— y tiene que **descubrir por sí mismo** que ciertos grupos de píxeles son
un cañón, otros son proyectiles y otros son el marcador irrelevante.

Implicaciones concretas:

**1. Hace falta aprender la representación.** Con 4 números basta una política lineal o una red
diminuta; de hecho en el laboratorio anterior una regla reactiva escrita a mano resolvía
CartPole. Con 100 800 píxeles hay que aprender un extractor de características, y esa es
precisamente la razón por la que aquí aparecen las **redes convolucionales**: aportan invarianza
a la traslación y comparten pesos, en lugar de tratar cada píxel como una variable independiente.

**2. Un solo cuadro no es un estado.** Una imagen fija no dice **hacia dónde** se mueve la
formación ni si un proyectil sube o baja. La observación es **parcialmente observable** y viola
la propiedad de Markov, mientras que el vector de CartPole incluye las velocidades y por lo tanto
sí la cumple. Esta es la justificación del apilamiento de cuadros (sección 2.4).

**3. Hay muchísima información irrelevante.** El color, el marcador, el contador de vidas y el
fondo ocupan píxeles que no aportan a la decisión. De ahí la escala de grises y el recorte.

**4. El costo en memoria y cómputo cambia de escala.** Un cuadro RGB ocupa ~100 KB; un búfer de
repetición de un millón de transiciones sin preprocesar necesitaría ~100 GB. Tras el
preprocesamiento, 84×84 en `uint8` son 7 KB por cuadro, y guardar cuadros individuales en lugar
de pilas completas lo vuelve manejable. Mantener `uint8` en vez de `float32` (`scale_obs=False`)
ahorra un factor de 4 y es una decisión deliberada, no un descuido.

**5. La eficiencia de muestras se desploma.** CartPole se resuelve en miles de pasos y en CPU;
los resultados publicados en Atari se miden en **decenas de millones de cuadros** y requieren GPU.

### 2.2 La variante de observación en RAM (`obs_type="ram"`)

Con `obs_type="ram"` la observación pasa a ser `Box(0, 255, (128,), uint8)`: los **128 bytes
completos de memoria** del Atari 2600 (la consola entera tenía esa RAM, en el chip RIOT). Ahí
viven, codificadas por el programador del juego, las posiciones de los sprites, la puntuación,
las vidas restantes y los contadores internos. Es, literalmente, **el estado del juego**.

**Cuándo conviene:**

- **Recursos limitados.** 128 valores admiten un MLP pequeño que entrena en CPU. Para un
  laboratorio, una prueba rápida o un experimento con presupuesto acotado, es la opción sensata.
- **Cuando el objeto de estudio no es la percepción.** Si se quiere evaluar exploración,
  planificación o un algoritmo de control, la visión por computadora solo añade ruido y costo;
  con RAM se aísla la parte que interesa.
- **Estado completo y sin ambigüedad.** No hay parpadeo, ni oclusiones, ni sprites de un píxel
  que desaparecen al redimensionar. El estado es casi markoviano sin necesidad de apilar cuadros.
- **Depuración e interpretabilidad.** Permite saber exactamente qué información tiene el agente.
- **Cota superior de referencia.** Sirve para separar "el agente no aprende la tarea" de "el
  agente no logra extraer la información de los píxeles".

**Cuándo no conviene:**

- La codificación es **específica de cada juego y opaca**: los bytes son campos de bits, valores
  BCD y contadores sin ninguna estructura común. Lo aprendido **no transfiere** a otro juego,
  que es justo lo que ALE existe para medir.
- Es **información privilegiada**: un humano no ve la RAM. Comparar contra puntajes humanos deja
  de ser legítimo.
- **No es comparable con la literatura**, que reporta resultados sobre píxeles.
- Empíricamente **no siempre resulta más fácil**: los bytes cambian de forma discontinua y no
  lineal, y los agentes sobre RAM a menudo rinden por debajo de los agentes sobre píxeles.

### 2.3 Espacio de acción de Space Invaders

`Discrete(6)`, verificado con `env.unwrapped.get_action_meanings()`:

| Índice | Acción | Efecto sobre el cañón |
|---|---|---|
| 0 | `NOOP` | Ninguno: no se mueve ni dispara. Útil para esperar a que se recargue el disparo o mantener una posición segura. |
| 1 | `FIRE` | Dispara sin moverse. |
| 2 | `RIGHT` | Se desplaza a la derecha sin disparar. |
| 3 | `LEFT` | Se desplaza a la izquierda sin disparar. |
| 4 | `RIGHTFIRE` | Se desplaza a la derecha **y** dispara en el mismo paso. |
| 5 | `LEFTFIRE` | Se desplaza a la izquierda **y** dispara en el mismo paso. |

Las 6 acciones son el **conjunto mínimo** del juego: `UP` y `DOWN` existen en el joystick pero no
hacen nada aquí. Con `full_action_space=True` el espacio pasa a `Discrete(18)` e incluye todas
las combinaciones (`UPFIRE`, `DOWNLEFTFIRE`, etc.), inertes en este juego.

Dos observaciones de diseño:

- Las acciones combinadas (`RIGHTFIRE`, `LEFTFIRE`) importan mucho: reflejan que en el 2600 el
  botón y la palanca son **independientes**. Un agente restringido a moverse *o* disparar jugaría
  claramente peor.
- El espacio es **discreto y pequeño**, lo que hace que Space Invaders sea un candidato natural
  para métodos basados en valores (DQN y su familia), que requieren evaluar $Q(s,a)$ para cada
  acción.

### 2.4 `AtariPreprocessing` y `FrameStackObservation`

**`gymnasium.wrappers.AtariPreprocessing`** empaqueta el preprocesamiento estándar del artículo
de DQN (Nature, 2015) con las correcciones de Machado et al. (2018). Su firma real en Gymnasium
1.3.0 es:

```python
AtariPreprocessing(env, noop_max=30, frame_skip=4, screen_size=84,
                   terminal_on_life_loss=False, grayscale_obs=True,
                   grayscale_newaxis=False, scale_obs=False)
```

| Transformación | Qué hace y por qué |
|---|---|
| `noop_max=30` | Ejecuta entre 1 y 30 `NOOP` al reiniciar. Aleatoriza el estado inicial para que el agente no memorice una única apertura. |
| `frame_skip=4` | Repite la acción 4 cuadros y devuelve el **máximo píxel a píxel de los 2 últimos**, lo que corrige el parpadeo de sprites del TIA. Exige `frameskip=1` en el entorno base. |
| `screen_size=84` | Redimensiona a 84×84 (usa OpenCV). Reduce el cómputo y da una entrada cuadrada conveniente para la CNN. |
| `grayscale_obs=True` | Pasa a escala de grises: elimina el canal de color, que en Atari casi nunca es informativo, y divide el tamaño entre 3. |
| `scale_obs=False` | Mantiene `uint8` en `[0, 255]` en lugar de `float32` en `[0, 1]`: **4× menos memoria** en el búfer de repetición. El escalado se hace en la GPU, ya dentro de la red. |
| `terminal_on_life_loss` | Opcional. Trata la pérdida de una vida como fin de episodio, lo que da una señal negativa que la recompensa de ALE no proporciona. Solo se usa al entrenar, nunca al evaluar. |

> **Precisión sobre el recorte de recompensa:** el enunciado lo menciona entre las
> transformaciones típicas, y en efecto forma parte del pipeline clásico de DQN, pero en
> Gymnasium **no está dentro de `AtariPreprocessing`**: vive en un wrapper aparte
> (`gymnasium.wrappers.ClipReward`, o `TransformReward` con `np.sign`). Conviene separarlos
> porque el recorte altera el objetivo que se optimiza —deja de coincidir con la puntuación—,
> mientras que las demás transformaciones solo cambian la representación de la observación.

**`gymnasium.wrappers.FrameStackObservation(env, stack_size=4)`** (llamado `FrameStack` en las
versiones anteriores a Gymnasium 1.0) apila las últimas `N` observaciones en un nuevo eje, de
modo que el agente recibe una **ventana temporal** en vez de una foto.

Su razón de ser es la señalada en 2.1: **un cuadro no basta para determinar el estado**. Con una
sola imagen no se puede saber si un proyectil sube o baja, ni hacia dónde avanza la formación.
Con 4 cuadros consecutivos, la velocidad y la dirección quedan implícitas en la diferencia entre
ellos, y la observación recupera —de forma aproximada— la propiedad de Markov. Por eso ambos
wrappers **se usan siempre juntos**: `AtariPreprocessing` hace cada cuadro barato y limpio, y
`FrameStackObservation` le devuelve la dimensión temporal que el preprocesamiento no aporta.

Efecto medido del pipeline completo sobre `ALE/SpaceInvaders-v5`:

```python
env = gym.make("ALE/SpaceInvaders-v5", frameskip=1)   # Box(0,255,(210,160,3))  → 100 800 valores
env = AtariPreprocessing(env, frame_skip=4, screen_size=84, grayscale_obs=True)
                                                      # Box(0,255,(84,84))      →   7 056 valores
env = FrameStackObservation(env, stack_size=4)        # Box(0,255,(4,84,84))    →  28 224 valores
```

La observación final es **3.6 veces más pequeña** que la imagen original y, aun así, contiene
**más información útil**, porque incorpora cuatro instantes de tiempo en lugar de uno solo. Ese
es exactamente el resultado que se busca: menos datos, mejor estado.

---

## Referencias

1. Bellemare, M. G., Naddaf, Y., Veness, J., Bowling, M. (2013). *The Arcade Learning
   Environment: An Evaluation Platform for General Agents*. Journal of Artificial Intelligence
   Research, 47, 253–279.
2. Machado, M. C., Bellemare, M. G., Talvitie, E., Veness, J., Hausknecht, M., Bowling, M.
   (2018). *Revisiting the Arcade Learning Environment: Evaluation Protocols and Open Problems
   for General Agents*. Journal of Artificial Intelligence Research, 61, 523–562.
3. Mnih, V. et al. (2015). *Human-level control through deep reinforcement learning*. Nature,
   518, 529–533.
4. Documentación de Gymnasium — Atari environments y wrappers: <https://gymnasium.farama.org/>
5. Documentación de ale-py (Farama Foundation): <https://ale.farama.org/>
6. Stella, emulador de Atari 2600: <https://stella-emu.github.io/>
