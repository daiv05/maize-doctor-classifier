# Detección de imágenes fuera de dominio (OOD)

Cualquier clasificador estándar de aprendizaje profundo opera bajo un supuesto peligroso: asume que todo lo que se le ponga enfrente pertenece obligatoriamente a una de sus clases de entrenamiento. Nuestro modelo conoce nueve diagnósticos de hojas de maíz. Si alguien le toma una foto a una hoja de frijol, al cielo, a una mesa de madera o a sus propios zapatos, la red no tiene un botón de "no sé": la función softmax está forzada matemáticamente a repartir un 100% de probabilidad entre esas nueve opciones. 

El resultado es predecible: el modelo puede asegurar con total tranquilidad que una taza de café tiene "roya común" con 85% de certeza. En una aplicación para agricultores, eso destruiría la confianza en el sistema desde el primer día.

Por eso incorporamos un detector de anomalías fuera de dominio (*Out-of-Distribution* o OOD), diseñado para evaluar si la imagen que entra se parece de verdad a una hoja de maíz antes de emitir un veredicto.

---

## Por qué no basta con mirar la confianza del modelo

La primera tentación al diseñar un filtro de seguridad es poner un umbral simple sobre la probabilidad más alta: si ninguna clase supera el 60% o si la diferencia entre el primer y segundo lugar es muy pequeña, se rechaza la predicción. 

Esto ayuda con imágenes borrosas o confusas de maíz, pero falla estrepitosamente con objetos que no tienen nada que ver con el cultivo. La confianza softmax mide qué tan seguro está el modelo *dado que tiene que elegir entre sus nueve etiquetas*, no si la foto es remotamente parecida a lo que vio durante el entrenamiento. Una imagen completamente ajena con colores o texturas coincidentes puede activar con fuerza una neurona y producir una confianza altísima por pura coincidencia estadística.

Para saber si una imagen pertenece al dominio del maíz, tenemos que mirar más profundo en la red: el vector de características (*features*) que se genera justo antes de la capa final de clasificación.

---

## Midiendo distancias en el espacio de características

En las capas intermedias de `EfficientNet-Lite0`, cada imagen se resume en un vector de 1,280 números que captura su estructura visual. En ese espacio matemático multidimensional, las miles de fotos de hojas de maíz que usamos para entrenar forman cúmulos o nubes agrupadas por patología.

Para saber si una foto nueva encaja con esas nubes, utilizamos la **distancia de Mahalanobis**. A diferencia de una distancia euclidiana común (que mide en línea recta ignorando cómo se dispersan los datos), Mahalanobis tiene en cuenta la forma y la correlación de la nube: mide a cuántas "desviaciones estándar" cae la foto del centroide de la clase más cercana.

Para afinar la puntería, el pipeline aplica dos ajustes clave:

1. **Reducción de ruido con PCA:** De las 1,280 dimensiones del vector, la inmensa mayoría contienen ruido numérico o redundancias. Proyectando el vector sobre las componentes principales que retienen el 99% de la varianza útil, nos quedamos con unas 186 dimensiones limpias y robustas.
2. **Distancia Relativa (RMD):** Muchas dimensiones de una red solo codifican información genérica de cualquier fotografía (como brillo general o contraste). La distancia relativa resta un modelo de fondo global: si una característica no ayuda a diferenciar una hoja de maíz de cualquier otra imagen natural, se cancela y no influye en la decisión.

---

## Cómo responde la aplicación en la práctica

Durante la calibración previa, calculamos el umbral de aceptación evaluando miles de hojas legítimas del conjunto de validación y fijando el límite en el percentil 95. Cualquier imagen cuya distancia supere ese umbral se clasifica como fuera de dominio.

En las pruebas de validación con el teléfono y en pruebas sintéticas, el comportamiento demostró ser muy efectivo:

- **Imágenes absurdas y ruidos:** Pantallas en negro, gris sólido, blanco o ruido aleatorio RGB disparan distancias altísimas (valores de RMD superiores a 300 o 600, frente a un umbral de corte de ~31) y son rechazadas de inmediato.
- **Fotografías reales no agrícolas:** Fotos de personas, gráficos o paisajes quedan marcadas como "OOD", evitando diagnósticos disparatados.
- **Hojas legítimas de maíz:** Las fotos reales de campo se mantienen cómodamente en valores bajos de distancia (incluso negativos en RMD), permitiendo que el diagnóstico fluya con normalidad.

El resultado para el productor es directo y transparente: si la cámara enfoca accidentalmente el suelo o una planta distinta, la pantalla de resultado no inventa una plaga inexistente, sino que le indica con prudencia: **"Imagen no reconocida. Por favor, asegúrese de encuadrar una hoja de maíz"**.

---

## Referencias

- Lee, K., Lee, K., Lee, H., & Shin, J. (2018). A Simple Unified Framework for Detecting Out-of-Distribution Samples and Adversarial Attacks. *NeurIPS*.
- Ren, J., Fort, S., Liu, J., et al. (2021). A Simple Fix to Mahalanobis Distance for Improving Near-OOD Detection. *arXiv:2106.09022*.
