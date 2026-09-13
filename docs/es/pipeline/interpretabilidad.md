# Interpretabilidad en el Pipeline Principal

Entender el comportamiento de un modelo no puede limitarse a mirar una tabla de aciertos. Para comprobar que las redes aprenden patrones agronómicos genuinos y no atajos visuales, el pipeline principal integra un flujo post-hoc que combina **Grad-CAM**, **LIME** y **SHAP**.

La ventaja de ejecutar la explicabilidad de forma desacoplada del entrenamiento es que no ralentiza la convergencia del modelo y nos permite auditar con el mismo rigor tanto predicciones individuales como tendencias globales de la red.

---

## El panel comparativo: LIME, SHAP y Grad-CAM

Para comparar las explicaciones sin sesgos, implementamos una segmentación en superpíxeles compartida (algoritmo SLIC). Al calcular los superpíxeles una sola vez y pasárselos tanto a LIME como a KernelSHAP, nos aseguramos de que ambas técnicas evalúen exactamente las mismas piezas de la hoja.

![Panel LIME, SHAP y Grad-CAM](/xai/panel_compare_common_rust.png)

Sobre esa base común evaluamos tres preguntas concretas:
1. **¿Coinciden en qué región mirar?** Medido mediante la intersección sobre unión ($IoU$) de los segmentos con mayor peso positivo.
2. **¿Coinciden en el orden de importancia?** Cuantificado con el coeficiente de correlación de Spearman sobre los segmentos.
3. **¿Coinciden en la dirección del empuje?** Verificando si ambos métodos concuerdan en qué partes de la hoja suman a favor del diagnóstico y cuáles restan.

Cuando LIME y SHAP muestran un acuerdo alto en estas tres métricas, la confianza en que el modelo se está apoyando en síntomas patológicos reales se multiplica.

---

## Perfil global y atención foliar

Además de mirar hojas individuales, nos interesaba saber hacia dónde se va la atención del modelo en promedio. ¿Pasa más tiempo analizando el tejido verde de la hoja o se distrae con la tierra y el fondo?

![Perfil global por clase](/xai/class_profile.png)

Acumulamos valores de Shapley sobre cientos de predicciones del conjunto de prueba para construir un perfil por clase, midiendo la proporción de la atribución positiva que cae dentro de la lámina foliar frente a lo que cae en el fondo.

![Auditoría de la máscara foliar](/xai/mask_audit.png)

Para auditar esta relación de forma objetiva, contrastamos la atribución con una máscara que delimita la hoja. Los resultados confirman que en patologías con síntomas definidos —como la roya común o las manchas necróticas— la red concentra de forma dominante sus gradientes sobre el tejido enfermo, superando con creces la cobertura que daría una atención repartida al azar.
