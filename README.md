# Diseño inverso de metamateriales spinodoidales

**Mini-proyecto BME513, Universidad de Valparaíso**
**Autor:** Vicente José Allendes Galleguillos
*Escuela de Ingeniería Civil Biomédica*

---

## Resumen

Comparación controlada de modelos subrogados para el problema
$\mathbf{G} \leftrightarrow \boldsymbol{\Theta}$ en metamateriales
spinodoidales, con foco en **caracterizar la no-unicidad angular** del mapeo
inverso — hasta ahora tratada en la literatura como un obstáculo, no como un
objeto de estudio.

Sobre 10 011 muestras (partición fija 80/20 con semilla), tres subrogados
directos superan $R^2 > 0{,}985$. El GP obtiene la mejor calibración
(ECE $= 2{,}8 \times 10^{-3}$) y es el único que **infla su varianza al
extrapolar** ($\times 3{,}47$ en $\beta$); la $f$-NN-prob, en cambio, la
*reduce* fuera del dominio ($\times 0{,}75$). Un *temperature scaling* de
solución cerrada ($s^\ast = 0{,}425$) recalibra la cobertura de la red del
$84{,}2\,\%$ al $65{,}8\,\%$ para un intervalo nominal del $68\,\%$.

En el mapeo inverso, la importancia de variables muestra que $\beta$ y
$\rho$ concentran $>99\,\%$ de la influencia sobre $G$; los ángulos son
irrelevantes. Consistente con eso, entre el $16$ y $22\,\%$ del espacio
paramétrico reproduce cada geometría, con dispersión angular de $26^\circ$
— **exactamente la de una uniforme en $[0^\circ, 90^\circ]$**
($90/\sqrt{12} = 25{,}98^\circ$). Cuatro estimaciones independientes (Monte
Carlo, ensamble, MDN $K{=}3$, baseline MDN $K{=}1$) convergen al mismo
valor. La comparación $K{=}1$ vs $K{=}3$ demuestra que la degeneración es
**continua, no discreta**: un núcleo no trivial de dimensión tres inducido
por la representación escalar de $G$, que no puede romperse con más datos
sino sólo enriqueciendo la representación.

---

## Estructura del repositorio

```
.
├── mini_proyecto_final.ipynb   # Notebook principal, autocontenido
├── db_spinodoid.csv            # Dataset: 10,011 muestras
├── requirements.txt            # Dependencias mínimas
├── README.md
├── .gitignore
│
├── modelos/                    # Modelos entrenados y serializados
│   ├── scalers.pkl             # StandardScaler de X e Y (imprescindible)
│   ├── config.json             # Arquitecturas para reconstruir sin adivinar
│   ├── f_nn.pt / f_nn_prob.pt  # Redes directas (determinista y probabilística)
│   ├── i_nn.pt                 # Red inversa (baseline)
│   ├── mdn_k1.pt / mdn_k3.pt   # Mixture Density Networks (K=1 baseline, K=3)
│   ├── ensemble.pt             # Ensamble de 10 i-NN
│   └── gp_kernels.pkl          # Kernels del GP ya optimizados (no la Cholesky)
│
├── figuras/                    # Todas las figuras del informe (PNG)
├── jsones/                     # Métricas y resultados numéricos por experimento
├── mds/                        # Contexto, guía técnica y preview del informe
└── salida_oveja/               # Descriptores extraídos del microCT (aplicación)
```

## Datos

- **10 011 muestras** generadas por campo aleatorio gaussiano anisótropo
  ([Kumar et al., 2020](#refs)) mallado con `pygalmesh`.
- **Parámetros de diseño** $\Theta = (\beta, \rho, \theta_1, \theta_2, \theta_3)$
  muestreados uniformemente en:
  - $\beta \in [5\pi,\, 15\pi]$
  - $\rho \in [0{,}3,\, 0{,}7]$
  - $\theta_i \in [0^\circ,\, 89{,}9^\circ]$
- **Descriptores geométricos** $G$: `Area_Inner`, `Area_Outer`, `Total_Area`,
  `Volume`, `SA_Vol_Ratio`, `Porosity`, `Pore_Size`.
- Partición fija 80/20 con semilla (8 008 train / 2 003 test). Los scalers se
  ajustan sólo con train para evitar fuga de información.

## Métodos comparados

**Mapeo directo** $\Theta \to G$:

| Modelo | Arquitectura / kernel | $n_\text{train}$ | Rol |
|---|---|---|---|
| f-NN         | MLP [512,256,256,128], MSE                 | 8 008 | Baseline determinista |
| f-NN-prob    | Mismo cuerpo, dos cabezas $(\mu, s)$, NLL  | 8 008 | Verosimilitud heteroscedástica |
| GP           | $C \cdot \text{RBF} + \text{White}$, ARD   | 2 240 | Referencia bayesiana |

**Mapeo inverso** $G \to \Theta$:

| Modelo | Objetivo |
|---|---|
| i-NN                 | Pérdida de reconstrucción con la f-NN congelada |
| Ensamble de 10 i-NN  | Descomposición sesgo-varianza y detección de multimodalidad |
| Monte Carlo Dropout  | Aproximación bayesiana barata sobre la i-NN |
| MDN ($K{=}3$)        | Densidad condicional $p(\Theta\mid G)$ explícita |
| MDN ($K{=}1$)        | Baseline unimodal — aísla el efecto de la mezcla |

## Reproducibilidad

### 1. Entorno

```bash
python -m venv .venv
# Windows PowerShell:
.venv\Scripts\Activate.ps1
# Bash / WSL:
source .venv/bin/activate

pip install -r requirements.txt
```

Probado en Python 3.10. CPU es suficiente; GPU acelera las redes pero no el
GP. El entrenamiento completo toma ~25 minutos en CPU, dominado por el GP
(~20 min).

### 2. Ejecutar el notebook

```bash
jupyter notebook mini_proyecto_final.ipynb
```

El notebook está organizado en secciones lineales y puede ejecutarse de
principio a fin (*Kernel → Restart & Run All*).

### 3. Reutilizar modelos entrenados (sin re-entrenar)

La carpeta `modelos/` contiene todos los pesos. La sección **"Carga de
modelos"** del notebook reconstruye cada objeto desde disco:

- **Redes** — `torch.load()` sobre los `state_dict`, reinstanciando las
  clases con la arquitectura declarada en `config.json`.
- **GP** — sólo se guardan los kernels optimizados, no los objetos completos
  (cada uno arrastra su matriz de Cholesky de $2240 \times 2240$, ~40 MB por
  propiedad). Al cargar, se hace
  `GaussianProcessRegressor(kernel=..., optimizer=None).fit(...)`, que
  ejecuta sólo la factorización (segundos) y evita la optimización de
  hiperparámetros (~20 min).
- **Scalers** — imprescindibles: sin ellos los pesos no sirven, porque no se
  puede transformar una entrada nueva ni interpretar la salida.

## Resultados principales

| Métrica | f-NN | f-NN-prob | GP |
|---|---:|---:|---:|
| $R^2$ (test)                | 0,9853 | 0,9871 | **0,9887** |
| NLL (test)                  | —      | −2,1065 | — |
| ECE                         | —      | $5{,}7 \times 10^{-3}$ | **$2{,}8 \times 10^{-3}$** |
| PICP nominal 68 %           | —      | 0,842 (0,658 recalibrada) | **0,739** |
| PICP nominal 95 %           | —      | 0,989 (0,922 recalibrada) | **0,943** |
| Factor de $\sigma$ al extrapolar en $\beta$ | — | 0,75 | **3,47** |

- **Speedup** del subrogado frente a la simulación física (133,3 s/muestra):
  $\sim 4{,}4 \times 10^7$ para la f-NN.
- **Curva de aprendizaje**: con el 5 % de los datos (400 muestras) la f-NN
  alcanza el 99 % de su $R^2$ final.
- **Diagnóstico inverso** replicado sobre cinco muestras y tres umbrales
  $\epsilon \in \{0{,}3,\, 0{,}5,\, 0{,}7\}$: la dispersión angular
  permanece invariante en $25{,}9^\circ$.

## Contribución

La contribución central es demostrar experimentalmente que **la
degeneración angular del diseño inverso spinodoidal no corresponde a un
conjunto discreto de modos, sino a una degeneración continua inducida por
la representación escalar de $G$**. La no identificabilidad es propiedad de
la representación, no del modelo, y sólo puede romperse enriqueciendo $G$
con descriptores direccionales.

Consecuencia metodológica: en problemas inversos degenerados, las métricas
puntuales, la descomposición sesgo-varianza clásica y la dispersión de un
ensamble pierden su interpretación habitual. La evaluación correcta exige
modelar la distribución condicional completa y validar físicamente sus
muestras a través del mapeo directo.

## Referencias clave <a id="refs"></a>

1. Kumar, S., Tan, S., Zheng, L., Kochmann, D. M. *Inverse-designed
   spinodoid metamaterials.* npj Computational Materials 6, art. 73 (2020).
2. Bishop, C. M. *Mixture density networks.* Technical Report NCRG/94/004,
   Aston University (1994).
3. El informe completo del proyecto está en `mds/preview_informe_final.pdf`
   con la lista completa de referencias.
