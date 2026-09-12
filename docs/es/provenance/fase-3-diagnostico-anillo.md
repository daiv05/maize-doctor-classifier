# Fase 3 — Qué mide realmente el anillo bajo LOSO

## Resultado

Bajo validación por fuente, la evaluación sobre el anillo exterior **no mide dependencia del
marco**. Mide si el colapso del modelo coincide con la clase mayoritaria del conjunto retenido.

## Evidencia

Para cada pliegue de `aligned_source_baseline`, sobre las predicciones almacenadas:

| fuente | n | clases | mayoritaria | acierto anillo | moda de la predicción |
| --- | ---: | ---: | ---: | ---: | ---: |
| corn-leaf-diseases-classification-roboflow | 476 | 1 | 1,0000 | 0,4370 | 0,437 |
| corn-leaf-roboflow | 2 919 | 6 | 0,2765 | 0,2169 | 0,523 |
| cropdg-unified-multidomain | 2 560 | 3 | 0,4531 | 0,1730 | 0,637 |
| maize-2-roboflow | 845 | 3 | 0,3645 | 0,0024 | 0,559 |
| maize-beans-tomatoes-africa | 12 112 | 4 | 0,3487 | 0,1450 | 0,457 |
| maize-deficiency-scanner-roboflow | 158 | 3 | 0,4810 | 0,1899 | 0,595 |
| maize-diseases | 6 970 | 3 | 0,6785 | 0,7049 | 0,698 |
| maize-in-field-dataset | 851 | 3 | 0,7121 | 0,3114 | 0,395 |
| maize-leaf-roboflow | 331 | 1 | 1,0000 | 0,0755 | 0,912 |
| maize-nutrient-deficiency | 337 | 4 | 0,3323 | 0,2166 | 0,825 |
| multicrop-disease-maiz | 5 709 | 3 | 0,5656 | 0,4936 | 0,330 |

El acierto del anillo supera a la tasa de la clase mayoritaria en **1 de 11 pliegues**
(`maize-diseases`, por 2,6 puntos). En los diez restantes, predecir siempre la clase más
frecuente del conjunto retenido iguala o supera al modelo entrenado.

La columna de moda explica por qué. Los modelos de anillo concentran entre el 33 % y el 91 % de
sus predicciones en una sola clase. El acierto que obtienen depende de si esa clase coincide con
la mayoritaria del retenido, no de información extraída del marco:

- `maize-diseases`: moda 0,698, mayoritaria 0,678, acierto 0,705 — las tres cifras son la misma.
- `maize-leaf-roboflow`: moda 0,912 sobre un retenido de una sola clase, acierto 0,0755 — el
  colapso cayó en la clase equivocada.

El contraste es más marcado en `minimal_colour`, cuyos modelos de anillo colapsan más:

| fuente | moda base | moda minimal_colour | acierto base | acierto minimal_colour | mayoritaria |
| --- | ---: | ---: | ---: | ---: | ---: |
| cropdg-unified-multidomain | 0,637 | **0,988** | 0,1730 | 0,4539 | 0,4531 |
| maize-beans-tomatoes-africa | 0,457 | 0,879 | 0,1450 | 0,2966 | 0,3487 |
| maize-in-field-dataset | 0,395 | 0,875 | 0,3114 | 0,0270 | 0,7121 |
| maize-diseases | 0,698 | 0,808 | 0,7049 | 0,8303 | 0,6785 |

En `cropdg` el modelo emite una sola clase el 98,8 % de las veces y obtiene 0,4539 de acierto,
que es la tasa de la mayoritaria (0,4531). En `maize-in-field` colapsa igual de fuerte y obtiene
0,0270. El mismo grado de degeneración produce acierto alto o nulo según el retenido.

## Alcance de la corrección

Esto **no** afecta a la medición sobre partición aleatoria. Allí las mismas fuentes están en
train y test, el anillo identifica la fuente, la fuente arrastra su prior de clase y el 78,3 %
de acierto es real y transferible dentro de esa partición. Ese sigue siendo el número que
demuestra el atajo.

Lo que queda invalidado es el cociente `acierto_anillo / acierto_original` **bajo LOSO**, usado
como «dependencia del marco» en la consolidación y en la comparación de brazos. En particular,
los +12,9 puntos atribuidos a `minimal_colour` sobre `maize-diseases` y los +14,8 puntos
agrupados no son evidencia de que ese brazo se apoye más en el marco.

El cociente sobre macro-F1 resiste mejor, porque un predictor colapsado obtiene macro-F1 bajo
por construcción: 0,1769 en `baseline` frente a 0,2851 en `minimal_colour` sobre los mismos
ocho pliegues. La diferencia sigue apuntando en la misma dirección, pero está contaminada por el
mismo colapso y no puede sostener sola una decisión de pipeline.

## Consecuencia para la métrica

Una medida de dependencia del marco utilizable bajo LOSO tiene que descontar el prior del
retenido. Las tres formas directas:

1. **Exceso sobre la mayoritaria**: `acierto_anillo − tasa_mayoritaria`, negativo en 10 de 11
   pliegues para `baseline`.
2. **Macro-F1 del anillo contra el azar** de nueve clases, insensible al prior pero sensible al
   colapso.
3. **Medir el atajo sólo donde es medible**: sobre partición aleatoria, que es donde el atajo
   opera y donde la cifra del 78,3 % ya está establecida.
