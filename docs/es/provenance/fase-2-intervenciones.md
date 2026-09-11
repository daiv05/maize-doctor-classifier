# Fase 2 — Cuánto se recupera con datos

Cribado de las dos intervenciones que la literatura respalda para corregir sesgo de
procedencia, medidas bajo el mismo protocolo de validación por fuente. Fecha de ejecución:
2026-09-10. GPU A10 en Modal, cuatro corridas en paralelo.

Reproducible con:

```bash
make modal-loso DETACH=1 BALANCE=1
make modal-loso DETACH=1 BALANCE=1 ARM=border_ring
make modal-loso DETACH=1 BACKMIX=0.5
make modal-loso DETACH=1 BACKMIX=0.5 ARM=border_ring
make modal-pull-provenance
```

Evidencia bruta en [`evidencia/`](/es/provenance/evidencia/).

## Las dos intervenciones

**Balanceo de grupos** (`--balance-groups`). El cupo de entrenamiento de cada clase se
reparte entre sus fuentes por llenado progresivo, en lugar de tomarse del total. Sin él, una
clase como `healthy` —cuyo 80,7 % procede de tres fuentes que no contienen ninguna otra
clase— entrena casi sólo con esas fuentes. El tamaño total del conjunto no cambia, así que la
comparación es a igualdad de datos: sobre un pliegue de ejemplo, `healthy` pasa de 547/48
entre su fuente mayor y su menor a 232 en cada una.

**BackMix** (`--backmix 0.5`). Sustituye el fondo por el de otra imagen del conjunto en lugar
de eliminarlo. Sólo actúa sobre imágenes que ya vienen recortadas sobre negro, que son las
únicas cuya máscara es recuperable sin segmentar: en la práctica, las de `common_rust` de
laboratorio. Es la intervención dirigida a la clase donde el marco recuperaba el 104 %.

Ambas se midieron con los dos brazos —imagen completa y sólo el marco— para poder evaluar las
dos condiciones de la compuerta.

## Resultado agregado

| Brazo | macro-F1 | Exactitud | El marco recupera |
|---|---:|---:|---:|
| Base (fuera de fuente) | 0,5573 | 0,6884 | 69,5 % |
| Balanceo de grupos | 0,5594 | 0,7009 | 71,6 % |
| BackMix 0,5 | 0,5609 | 0,6981 | 70,6 % |

Las dos ganancias —**+0,0022** y **+0,0036**— están por debajo de la desviación entre semillas
de la línea base, que es **0,0072**. Ninguna reduce la recuperación del marco; las dos la
suben ligeramente.

## Compuerta 2 por clase

| Clase | Banda | F1 base | F1 balanceo | F1 BackMix | Mejor Δ | Marco base | Mejor marco | Veredicto |
|---|---|---:|---:|---:|---:|---:|---:|---|
| `healthy` | Por procedencia | 0,8092 | 0,8335 | 0,8246 | +0,024 | 56,0 % | 65,8 % | Techo alcanzado |
| `lethal_necrosis` | Por procedencia | 0,7912 | 0,8180 | 0,7893 | +0,027 | 99,7 % | 93,0 % | Techo alcanzado |
| `common_rust` | Por procedencia | 0,7804 | 0,7427 | 0,7504 | −0,030 | 98,5 % | 97,0 % | Techo alcanzado |
| `northern_corn_leaf_blight` | Por procedencia | 0,7171 | 0,7026 | 0,7248 | +0,008 | 62,4 % | 63,2 % | Techo alcanzado |
| `fall_armyworm` | Frágil | 0,6194 | 0,6411 | 0,6395 | +0,022 | 68,1 % | 58,7 % | Techo alcanzado |
| `nitrogen_deficiency` | Frágil | 0,4190 | 0,4606 | 0,4486 | +0,042 | 62,8 % | 46,1 % | Techo alcanzado |
| `gray_leaf_spot` | No soportada | 0,3972 | 0,3910 | 0,3882 | −0,006 | 47,0 % | 50,2 % | Techo alcanzado |
| `phosphorus_deficiency` | No soportada | 0,2861 | 0,2514 | 0,2718 | −0,014 | 16,3 % | 39,1 % | Techo alcanzado |
| `potassium_deficiency` | No soportada | 0,1958 | 0,1940 | 0,2109 | +0,015 | 54,2 % | 45,9 % | Techo alcanzado |

El criterio era ganar **≥ 0,10 de F1** en las frágiles, o bajar el marco **por debajo del
60 %** en las sostenidas por procedencia. **Ninguna clase cumple ninguna de las dos.** La
mayor ganancia de F1 en todo el experimento es +0,042 en `nitrogen_deficiency`, menos de la
mitad del umbral.

## Lectura

**El balanceo no ayuda porque el problema no es de proporción, es de cobertura.** Repartir el
cupo de `healthy` entre sus cinco fuentes no crea variedad de captura donde no la hay: las
tres fuentes que aportan el 80,7 % siguen siendo fuentes de una sola clase, y la cuarta y
quinta son pequeñas. El modelo sigue pudiendo reconocer la sesión.

**BackMix no ayuda porque su alcance es minúsculo.** Sólo puede actuar sobre las imágenes ya
recortadas sobre negro, que son unas 1 124 de 33 268. Descorrelacionar el fondo del 3 % del
corpus no cambia el comportamiento agregado. Y en `common_rust`, su objetivo directo, la
recuperación del marco baja sólo de 98,5 % a 97,0 %: el atajo de esa clase no es únicamente
el fondo negro, también son resolución, compresión y encuadre, que BackMix no toca.

**Hay un efecto real pero por debajo del umbral en las frágiles.** El balanceo baja la
recuperación del marco de `nitrogen_deficiency` 16,7 puntos y la de `fall_armyworm` 9,4. Es
la única señal consistente del experimento, y va en la dirección esperada. No alcanza para
cambiar ninguna banda, pero indica que el balanceo hace *algo* donde hay varias fuentes
comparables.

## Consecuencia

Con la Compuerta 2 cerrada en las nueve clases, **todas pasan a limitación documentada con
las bandas de la [Fase 1](/es/provenance/fase-1-particion-honesta)**. El plan no contempla
una tercera intentona con estos datos, y no se hará.

Lo que queda es la [Fase 3](/es/provenance/): predicción selectiva. El objetivo deja de ser
subir el número y pasa a ser que el sistema sepa cuándo no responder.

## Fase 2b — Endurecimiento por augmentation

Las dos intervenciones anteriores actúan sobre la **composición** de los datos. Ninguna toca
la firma de captura, que es lo que el brazo del marco mide. Esta tercera sí.

### Un bug previo que había que arreglar

`LeakDataset` sembraba su generador con `default_rng(seed * 1_000_003 + index)`, determinista
por imagen. **Cada imagen recibía el mismo volteo en todas las épocas**: una asignación
aleatoria fija, no una augmentation. Con recorte aleatorio el defecto sería fatal, porque
cada imagen vería siempre el mismo recorte. Se sustituyó por un generador propio de cada
proceso trabajador, cuyo estado avanza entre épocas.

Los experimentos anteriores llevaban el defecto, pero como todos sus brazos lo compartían,
sus comparaciones internas siguen siendo válidas.

### Diseño B

Los experimentos anteriores entrenan un modelo sobre anillos para medir **cuánta información
contiene el marco**. Aquí la pregunta es otra: **cuánto sigue dependiendo del marco un modelo
entrenado con la imagen completa**. El modelo se entrena una vez y se evalúa dos veces sobre
el mismo conjunto retenido, con la imagen entera y con sólo el anillo.

Las cifras de esta sección **no son comparables** con las de las secciones anteriores.

### La augmentation endurecida

Recorte aleatorio del 30–100 % del área, remuestreo por reducción y ampliación,
recompresión JPEG con calidad entre 30 y 95, y alteración de brillo, contraste, saturación y
tono. El recorte ataca la presencia del marco; el remuestreo y la recompresión, la huella del
codificador y de la resolución nativa; el color, la respuesta cromática de cada cámara.

Conviene notar que el pipeline del proyecto usa `T.Resize` de la imagen completa y
`ColorJitter(saturation=0.0, hue=0.0)`, así que **el marco está presente en el 100 % de las
muestras de entrenamiento y la respuesta cromática se preserva intacta**.

### Resultado

| | Base | Endurecida | Δ |
|---|---:|---:|---:|
| macro-F1 con imagen completa | 0,5863 | 0,5470 | −0,039 |
| macro-F1 con sólo el marco | 0,2629 | 0,1630 | −0,100 |
| Exactitud con imagen completa | 0,7225 | 0,6702 | −0,052 |
| **Dependencia del marco** | **44,8 %** | **29,8 %** | **−15,0 pp** |

**La augmentation endurecida sí rompe parte de la dependencia del atajo**, y lo hace con
diferencia: quince puntos. El coste son cuatro centésimas de macro-F1.

### Por clase

| Clase | F1 base | F1 endurecida | Δ | Dep. base | Dep. endurecida | Δ dep. |
|---|---:|---:|---:|---:|---:|---:|
| `nitrogen_deficiency` | 0,4739 | **0,5219** | +0,048 | 27,5 % | 18,5 % | −9,0 |
| `phosphorus_deficiency` | 0,2889 | **0,3303** | +0,041 | 21,5 % | **0,0 %** | −21,5 |
| `potassium_deficiency` | 0,2324 | **0,2677** | +0,035 | 28,4 % | **0,0 %** | −28,4 |
| `fall_armyworm` | 0,6594 | 0,6561 | −0,003 | 62,5 % | 39,4 % | −23,1 |
| `northern_corn_leaf_blight` | 0,7548 | 0,7165 | −0,038 | 18,8 % | 9,5 % | −9,3 |
| `healthy` | 0,8265 | 0,7665 | −0,060 | 64,6 % | 64,8 % | +0,2 |
| `lethal_necrosis` | 0,8231 | 0,7487 | −0,075 | 73,2 % | 41,5 % | −31,7 |
| `common_rust` | 0,7939 | 0,6529 | −0,141 | 43,7 % | 28,6 % | −15,1 |
| `gray_leaf_spot` | 0,4241 | 0,2625 | −0,162 | 16,5 % | 19,0 % | +2,5 |

### Lectura

**Las tres deficiencias mejoran, contra lo previsto.** Antes de ejecutar se anotó el riesgo de
que la alteración de tono dañara a `nitrogen`, `phosphorus` y `potassium`, porque el color
clorótico es su señal diagnóstica. Ocurre lo contrario: las tres suben, y la dependencia del
marco de fósforo y potasio cae a **cero**. La predicción era errónea y queda registrada como
tal.

**El comportamiento se separa en tres grupos según de qué depende cada clase.**

- Clases cuya señal es **color y estructura de gran escala** (las tres deficiencias): la
  augmentation actúa como regularización y mejora.
- Clases que **dependían del marco** (`lethal_necrosis` 73,2 %, `common_rust` 43,7 %): pierden
  el atajo y su F1 baja. Es el comportamiento esperado y deseable: ese rendimiento estaba
  inflado.
- Clases cuya señal es **textura fina**: `gray_leaf_spot` cae 0,162 y su dependencia del marco
  **no** baja (16,5 % → 19,0 %). No perdió un atajo, perdió señal. La recompresión y el
  remuestreo destruyen exactamente el detalle de lesión que esa clase necesita.

**`healthy` es el único caso donde la augmentation no reduce la dependencia** (64,6 % →
64,8 %) y además pierde F1. Su atajo sobrevive al recorte, al remuestreo y al color.

### Consecuencia

Frente a la conclusión de la Fase 2, esta tercera intervención **sí mueve el indicador que
importa**, pero no de forma uniforme: compra independencia del marco y recupera las clases
escasas, a costa de las de textura fina.

Eso sugiere una ablación por componentes: separar recorte, compresión y color para ver si el
recorte por sí solo compra la independencia del marco sin el coste en `gray_leaf_spot`.

## Fase 2c — Ablación por componentes

Cada componente se aisló conservando la probabilidad que tiene dentro de la combinación, de
modo que la ablación descompone exactamente esa transformación. Mismo diseño B, mismos once
pliegues, misma semilla.

| Brazo | macro-F1 completa | macro-F1 marco | Dependencia | Δ F1 |
|---|---:|---:|---:|---:|
| Base | 0,5863 | 0,2629 | 44,8 % | — |
| `crop` | 0,5643 | 0,1395 | **24,7 %** | −0,022 |
| `codec` | 0,6171 | 0,2463 | 39,9 % | +0,031 |
| `colour` | **0,6349** | 0,2499 | 39,4 % | **+0,049** |
| `hardened` (los tres) | 0,5470 | 0,1630 | 29,8 % | −0,039 |

### Los componentes compran cosas distintas

**El recorte compra independencia del marco.** Baja la dependencia de 44,8 % a 24,7 %, más
que la combinación de los tres, y pagando menos F1 que ella.

**El color compra rendimiento.** Sube el macro-F1 a 0,6349, el mejor número del proyecto bajo
partición honesta, y mejora ocho de las nueve clases.

**Los tres por separado superan a la combinación en F1.** `hardened` (−0,039) es peor que
cualquiera de sus componentes aislados. Apilarlos no suma.

### Dos predicciones propias refutadas

**El color no daña las deficiencias; es lo que más las ayuda.** Antes de ejecutar se anotó
dos veces el riesgo de que la alteración de tono perjudicara a `nitrogen`, `phosphorus` y
`potassium`, porque la clorosis es su señal. El brazo de color sube las tres, y
`potassium_deficiency` encabeza la tabla con +0,106. El
`ColorJitter(saturation=0.0, hue=0.0)` del pipeline del proyecto, elegido con ese mismo
razonamiento, está costando rendimiento.

**La compresión no es lo que hunde `gray_leaf_spot`.** Se atribuyó su caída de 0,162 en
`hardened` al remuestreo y la recompresión. Aislado, `codec` deja esa clase **plana**
(+0,0003) y sube el conjunto +0,031.

### Los componentes no son aditivos

| Clase | Δ `crop` | Δ `codec` | Δ `colour` | Suma | Δ `hardened` real |
|---|---:|---:|---:|---:|---:|
| `gray_leaf_spot` | −0,071 | +0,000 | +0,042 | −0,029 | **−0,162** |
| `common_rust` | −0,219 | −0,007 | −0,002 | −0,228 | **−0,141** |

En `gray_leaf_spot` la combinación destruye cinco veces más de lo que predice la suma de sus
partes. En `common_rust` ocurre lo contrario: la combinación daña menos que el recorte solo.
**La interacción entre componentes domina sobre sus efectos individuales**, en ambas
direcciones, así que ninguna conclusión sobre la combinación se deduce de las partes.

### Consecuencia

La respuesta a si algo ayuda es **sí**, y con dos palancas separadas: el color para el
rendimiento y el recorte para la independencia del atajo. Lo que no está probado es si pueden
combinarse sin que la interacción se las coma, que es lo que la tabla anterior advierte.

## Qué queda sin verificar

| Cuestión | Estado |
|---|---|
| Semillas | Cribado con una sola semilla por brazo. El plan preveía consolidar a tres sólo lo que mostrara efecto; nada superó el ruido, así que no se consolidó. Efectos reales de +0,02 a +0,04 no son resolubles con una semilla, pero el umbral de la compuerta era +0,10 |
| Probabilidad de BackMix | Se probó únicamente 0,5. No se barrió el parámetro |
| Grado de balanceo | El reparto es equitativo por celda. No se probó un balanceo parcial ni un repesado en la pérdida en lugar de submuestreo |
| BackMix fuera de las pre-enmascaradas | Requeriría máscaras que sólo el segmentador puede dar, y su calidad está medida y descartada para este uso |
| Interacción entre ambas | No se probó balanceo y BackMix a la vez |
