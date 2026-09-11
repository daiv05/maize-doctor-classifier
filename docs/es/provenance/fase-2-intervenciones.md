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

## Qué queda sin verificar

| Cuestión | Estado |
|---|---|
| Semillas | Cribado con una sola semilla por brazo. El plan preveía consolidar a tres sólo lo que mostrara efecto; nada superó el ruido, así que no se consolidó. Efectos reales de +0,02 a +0,04 no son resolubles con una semilla, pero el umbral de la compuerta era +0,10 |
| Probabilidad de BackMix | Se probó únicamente 0,5. No se barrió el parámetro |
| Grado de balanceo | El reparto es equitativo por celda. No se probó un balanceo parcial ni un repesado en la pérdida en lugar de submuestreo |
| BackMix fuera de las pre-enmascaradas | Requeriría máscaras que sólo el segmentador puede dar, y su calidad está medida y descartada para este uso |
| Interacción entre ambas | No se probó balanceo y BackMix a la vez |
