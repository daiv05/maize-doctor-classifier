# Evaluación y diferencias por entorno

Actualización: 11–12 septiembre 2026. Esta página distingue métricas guardadas,
recálculos y conclusiones que todavía requieren experimentación. Ver
[corridas recibidas](../../reviews/2026-09-11-corridas-recibidas) y
[registro de reparación](../../reviews/2026-09-11-reparacion-integral).

## Procedencia de resultados

B0 y ShuffleNet pertenecen al pipeline principal, con test de 5,015 filas.
Sus macro-F1 recalculados son 0.948333 y 0.932980; sus accuracies, 0.979661 y 0.973081.
El ensemble declara 0.950676, pero faltan sus predicciones/probabilidades para
recalcularlo. Estos resultados no son comparables directamente con 0.903520 de Etapa 2:
sus holdouts tienen distintas imágenes. El hash de pesos no acredita calidad ni
independencia de datos.

La revisión del 12/09 confirmó una misma fotografía GLS a distinta resolución en train
y val de main, más cuatro conflictos healthy/fall_armyworm en el corpus fuente.
Los nuevos splits excluyen provisionalmente ocho entradas y agrupan el par confirmado;
quedan 372 pares candidatos sin resolver. Ver [dataset y piloto](../../reviews/2026-09-12-piloto-y-dataset).

## Entornos y clases comparables

Laboratorio aporta 532 imágenes de tres clases; campo, 4,483 de nueve clases.
Para B0, el macro-F1 en las clases soportadas de laboratorio es 0.886365, no 0.295455:
el cálculo viejo promediaba seis clases sin muestras como ceros.

La comparación común usa roya, mancha gris y tizón norteño en ambos entornos:
laboratorio 0.886365 frente a campo 0.907094, brecha 0.020729 y razón min/max 0.977148.
La composición de clases y los soportes siguen limitando la interpretación.
Las clases ausentes tienen recall/FNR indefinidos (`null`), no FNR=1.
Un único entorno no puede demostrar ausencia de disparidad.

El evaluador registra soportes e intervalos Wilson de accuracy/recall cuando son
estimables. No aplica una regla del 80% para certificar equidad con macro-F1 ni
mezcla esa razón con una razón de accuracy.

## Oclusiones: sensibilidad, no localización causal

El rectángulo del 20% al 80% de cada eje ocupa aproximadamente el **36% del área**;
su complemento, el 64%. Un cero en tensor normalizado representa el color medio
de normalización, no negro. Esas regiones no son «lesión» y «fondo» sin máscaras
verificadas, y Grad-CAM no demuestra que la red ignore por completo el contexto.

La reparación compara probabilidad de una clase fija, registra cambios de clase,
y añade un control aleatorio de igual área. Se retiraron umbrales de riesgo
arbitrarios y frases de confirmación causal.

La página histórica publicaba 84% y 70% de accuracy tras ocluir; el JSON recibido
contiene 80.46% y 71.76%. No se conservan como cifras actuales ni se intenta recuperar
probabilidades de clase fija desde promedios de máximos. Hace falta repetir el
análisis con predicciones por imagen y el contrato correcto.

## Segmentación y OOD

Segmentar es una hipótesis a evaluar, no una mitigación garantizada ni requisito
universal demostrado por estas oclusiones. El piloto compara originales y derivados
en desarrollo, conserva fallbacks y requiere revisar transferencia de etiqueta,
especialmente en N/P/K y campo. No se selecciona contra test.

El piloto ya se ejecutó con 288 imágenes y tres semillas: solo 31 máscaras aceptadas,
257 fallbacks originales y ninguna mejora media global de los heads nuevos. No habilitar
segmentación obligatoria a partir de esta evidencia; faltan revisión humana y calibración.

OOD necesita calibración de desarrollo y evaluación con ejemplos fuera de dominio.
Un percentil ID no garantiza detectar todas las hojas ajenas, manos o fondos.
Modelo, etiquetas, preproceso y estadísticas deben corresponder a la misma corrida.

## Entrega

La exportación distingue creación, paridad y evaluación; sincronizar exige caída de
macro-F1 ≤0.01 y paquete coherente con recuperación. El informe anterior no demuestra
una API/APK disponible ni latencia, memoria o inferencia offline medidas en Android.

La [comprobación real TFLite](../../reviews/2026-09-12-exportacion-real) reprodujo PyTorch
en FP32 para ambos checkpoints sobre 5 015 imágenes históricas. ShuffleNet INT8 pasó;
B0 INT8 falló paridad de probabilidades/features. Los desgloses corregidos usan clases
con soporte, y no se interpreta esta reevaluación de conversión como un holdout nuevo.
