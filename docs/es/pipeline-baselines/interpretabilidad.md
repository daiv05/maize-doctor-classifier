# Interpretabilidad

Se combina LIME (regiones de superpíxeles que sostienen el diagnóstico) y Grad-CAM (mapa de activación de la
clase predicha). **SHAP no aplica a los baselines:** los subcomandos `compare` y `global` de `scripts/pipeline/explain.py`
son exclusivos del pipeline principal (`outputs/main`, ver [interpretabilidad del pipeline principal](../pipeline/interpretabilidad.md)).
Los baselines existen para comparar arquitecturas rápido y barato, y LIME + Grad-CAM ya cubre ese propósito sin el costo
extra de correr KernelSHAP sobre cada arquitectura candidata.

## Subcomandos

Tres subcomandos de `scripts/pipeline/explain.py` cubren distintos niveles de detalle:

| Subcomando | Target | Salida |
|---|---|---|
| `visual` | `make explain-visual-baselines` | Reporte visual LIME + Grad-CAM por imagen |
| `fidelity` | `make explain-fidelity-baselines` | Fidelidad agregada y dispersión por clase |
| `errors` | `make explain-errors-baselines` | LIME dirigido a errores (label ≠ pred) |

## Qué observar

Revisar los mapas de atribución con ojo crítico ayuda a distinguir cuándo el modelo realmente aprende rasgos de la enfermedad y cuándo se apoya en atajos poco confiables. En las clases fáciles (roya, necrosis letal) los mapas se concentran en el tejido de la hoja. En cambio, en las deficiencias N/P/K y en imágenes de campo con fondo cargado, la atribución puede dispersarse hacia el fondo, una señal de contexto a vigilar y coherente con la confusión entre deficiencias observada en la evaluación.

### Caso sano: atribución sobre el tejido correcto

En una imagen de laboratorio de `common_rust` bien clasificada, LIME resalta superpíxeles sobre la lámina de la hoja y Grad-CAM concentra su activación en la zona con pústulas.

![LIME + Grad-CAM sobre roya común correctamente clasificada](/baselines/samples/lime_common_rust_ok.png)

### Caso de error: potasio diagnosticado como nitrógeno

Este panel es uno de los más informativos de todo el análisis, el modelo clasifica una imagen de `potassium_deficiency` como `nitrogen_deficiency` **con 99.7 % de confianza**. Grad-CAM se enciende sobre la punta necrótica y amarillenta de la hoja, un síntoma real de clorosis, pero ese mismo rasgo es ambiguo entre las tres deficiencias. El modelo mira el lugar correcto y aun así llega a la etiqueta equivocada.

![LIME + Grad-CAM: potasio clasificado como nitrógeno con alta confianza](/baselines/samples/lime_potasio_error_nitrogeno.png)

## Confianza del modelo

Para `efficientnet_b0`:

| | Confianza media |
|---|---:|
| Predicciones **correctas** | 0.987 |
| Predicciones **incorrectas** | 0.914 |
| Errores solo de `potassium_deficiency` | 0.897 |

El modelo **se equivoca con alta confianza**, cuando falla, no lo hace "dudando". La diferencia entre acertar y fallar es de apenas ~7 puntos de confianza, y los errores de potasio siguen por encima de 0.89.

Este hallazgo inicial es muy importante para el caso de uso que busca darle, un mensaje del tipo "confianza baja, consulte a un técnico" no capturaría estos casos donde el modelo esté equivocado.

En caso de que no se logre llegar a mejores resultados, una alternativa más prometedora sería agrupar N/P/K en una respuesta de "deficiencia nutricional" cuando la predicción caiga dentro de ese clúster, pero eso es algo que se decidirá luego de continuar con el pipeline principal.
