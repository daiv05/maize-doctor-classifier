# Interpretabilidad y Explicabilidad (XAI)

Una red neuronal convolucional es, por defecto, una caja negra. Le entregamos la fotografía de una hoja, hace millones de multiplicaciones internas y nos devuelve una etiqueta con un porcentaje de confianza. Pero no nos dice qué vio para llegar a esa conclusión.

En agricultura esta opacidad es peligrosa. Si la aplicación le dice a un productor que su cultivo tiene roya cuando en realidad sufre deficiencia de nitrógeno, el agricultor podría gastar dinero escaso en fungicidas innecesarios mientras las plantas siguen muriendo por falta de nutrientes. Peor aún: el modelo podría estar acertando por los motivos equivocados, fijándose en el color de la mesa de laboratorio, en la sombra de los dedos o en el suelo del fondo en lugar de mirar los síntomas biológicos de la hoja.

Para auditar y entender las decisiones de los modelos exploramos tres técnicas complementarias: Grad-CAM, LIME y SHAP.

---

## Cómo mira el modelo: tres formas de explicar una predicción

No todas las herramientas de explicabilidad funcionan igual ni responden a las mismas preguntas. Algunas aprovechan las matemáticas internas de las convoluciones, mientras que otras tratan a la red como una caja cerrada a la que interrogan con pruebas modificadas.

### Grad-CAM: el mapa de calor directo
Grad-CAM aprovecha la estructura interna de la red. Va directamente a la última capa convolucional —que es la que todavía conserva la distribución espacial de la imagen antes de condensarse en la decisión final— y calcula qué regiones activaron con más fuerza la clase predicha.

El resultado es un mapa de calor visual muy intuitivo. Las zonas rojas y cálidas indican dónde se concentró la atención del modelo, mientras que las azules fueron ignoradas. Es una técnica extremadamente rápida que no cuesta casi nada computacionalmente y nos permite comprobar al instante si la red está mirando la pústula de roya o si se distrajo con una brizna de pasto al fondo. Su única limitación es que la resolución es algo gruesa, mostrando manchas de atención más que bordes microscópicos.

### LIME: apagando piezas del rompecabezas
LIME no necesita saber cómo está construida la red por dentro. Toma la imagen original, la divide en grupos de píxeles homogéneos llamados superpíxeles y empieza a jugar con ellos: apaga algunos, enciende otros y le pregunta una y otra vez al modelo qué opina de cada variante.

Con cientos de esas consultas, LIME ajusta un modelo lineal simple que nos dice cuánto sumó o restó cada zona. Lo interesante de LIME es que distingue importancias con signo: muestra en un color las regiones de la hoja que empujan a favor del diagnóstico y en otro las que contradicen esa decisión. Sin embargo, tiene un precio: hacer mil inferencias por cada foto es computacionalmente lento y, al basarse en perturbaciones aleatorias, dos ejecuciones pueden dar mapas ligeramente distintos.

### SHAP: rigor matemático y visión global
SHAP toma prestada una idea de la teoría de juegos cooperativos (los valores de Shapley). Imagina que cada región de la imagen es un jugador en un equipo y calcula cuál es la contribución justa y exacta de cada una al resultado final, promediando lo que aporta en todas las combinaciones posibles.

La gran ventaja de SHAP frente a LIME es su consistencia matemática: no depende de aproximaciones caprichosas y permite comparar valores numéricos de importancia entre distintas imágenes. Además de explicar una foto puntual, los valores de SHAP se pueden promediar sobre cientos de ejemplos para construir un perfil global por clase, confirmando de manera sólida si la red suele buscar los patrones donde la agronomía manda.

---

## La combinación adoptada en el proyecto

Cada técnica tiene sus fortalezas y sus costes:

| Herramienta | Cómo trabaja | Resolución | Costo de cálculo | Uso en DoctorMaiz |
|---|---|---|---|---|
| **Grad-CAM** | Gradientes en la última capa | Mapa de calor continuo | Muy bajo (una pasada) | Inspección visual rápida y auditoría de campo |
| **LIME** | Perturbación de superpíxeles | Regiones discretas (+ / −) | Alto (cientos de pasadas) | Análisis local de casos dudosos |
| **SHAP** | Valores de Shapley | Atribuciones consistentes | Medio / Alto | Auditoría global y comparación rigurosa |

En nuestro flujo combinamos las tres: usamos **Grad-CAM y LIME** para revisar imágenes individuales y casos de error, y dejamos **SHAP** para el pipeline principal, donde necesitamos verificar que la atención promedio sobre el tejido foliar sea genuina y no un simple reflejo del encuadre fotográfico.

---

## Referencias

- Selvaraju, R. R., et al. (2017). Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization. *ICCV*.
- Ribeiro, M. T., Singh, S., & Guestrin, C. (2016). "Why Should I Trust You?": Explaining the Predictions of Any Classifier. *KDD*.
- Lundberg, S. M., & Lee, S.-I. (2017). A Unified Approach to Interpreting Model Predictions. *NeurIPS*.
