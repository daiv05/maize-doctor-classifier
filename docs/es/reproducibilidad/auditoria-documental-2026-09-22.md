# Auditoría documental — 22 de septiembre de 2026

## Alcance y método

Se contrastaron `README.md`, `LOCAL.md`, `CLAUDE.md`, `docs/`, informes, historial Git, scripts, tests y artefactos locales/remotos. No se entrenó, no se ejecutó Optuna, no se regeneraron splits y no se modificó el dataset ni código funcional.

## Inconsistencias encontradas y resolución

| Hallazgo | Clasificación | Resolución documental |
|---|---|---|
| El README presentaba el Lite0 desplegado de agosto como único resultado actual | histórico mezclado con vigente | separar desplegado histórico de baseline de desarrollo `20260921_204608` |
| `0.6026 ± 0.1240` aparecía como generalización/LOSO | etiqueta experimental ambigua | identificarlo como CV source-grouped histórica de cinco pliegues |
| Accuracy asociada a esa CV figuraba como 76.4 % | contradicción con el CSV | corregir a 0.796900 ± 0.1195 o no resumirla sin dispersión |
| El inventario de 33 438 se mezclaba con 33 429 elegibles | versiones/contextos distintos | conservar 33 438 como corpus ampliado histórico y usar 33 437 descubiertas / 33 429 elegibles para la materialización vigente |
| `healthy=8 744` y `fall_armyworm=4 858` seguían como conteos actuales | cinco exclusiones no reflejadas | actualizar el resumen vigente a 8 740 y 4 853; preservar la tabla histórica donde se explicita su contexto |
| “8 fuentes” se trataba como conteo actual | inventario documental ≠ `source_id` resuelto | declarar 11 fuentes en el manifest vigente; conservar el catálogo histórico de datasets por separado |
| `LOCAL.md` decía que el loop principal estaba pendiente | obsoleto | reemplazar por comandos actuales de entrenamiento/evaluación |
| Los estudios Optuna antiguos podían parecer la fase nueva | protocolos mezclados | etiquetar 15/25 trials como históricos y mantener los 60 trials como plan pendiente |
| `run_afinada_b0_summary.json` contiene internamente `model=efficientnet_lite0` | evidencia contradictoria | conservar, marcar y prohibir citarlo como B0 hasta resolver el origen |
| El run `180112` dice `split_identifier=seed_42` aunque el hash identifica la materialización agrupada histórica | ruta lógica ambigua | documentar el hash y el protocolo observado, sin reescribir el artefacto original |
| Se afirmaba o sugería ausencia de fuga perceptual | no demostrado | declarar PHash desactivado y solapamiento perceptual desconocido |
| Métricas por ambiente se comparaban sin siempre declarar clases con soporte | interpretación incompleta | añadir advertencia de soporte desigual |
| “producción” se usaba para el checkpoint nuevo | conclusión excesiva | usar “baseline principal de desarrollo”; reservar “desplegado histórico” para el artefacto móvil existente |
| Figuras sin run/script/artefacto explícito | procedencia incompleta | crear [registro de figuras](./figuras.md) y marcar lo desconocido como `PENDIENTE` |
| Referencias con metadatos parciales y sin validación externa | bibliografía incompleta | crear [auditoría bibliográfica](./referencias.md), sin fabricar DOI |

## `docs/etapa_2`

La carpeta contiene únicamente `main.aux`, `main.log`, `main.out` y `main.toc`. Son salidas auxiliares de LaTeX, no una fuente documental autónoma; no existe allí el `.tex` ni un PDF con el que validar sus cifras. Se clasifican **NO VERIFICABLES COMO DOCUMENTACIÓN FUENTE** y se preservan por no eliminar historia sin autorización. Las fuentes históricas recuperables están en `reports/firts-phase/` y `reports/second-phase/`, donde sí existen `.tex` y PDF.

## Clasificación de cifras

- **VIGENTE:** manifest/splits actuales, baseline `20260921_204608`, contratos e invariantes actuales.
- **HISTÓRICA:** baselines antiguos, Lite0 desplegado, ensamble, CV 2×2, multi-seed anterior, LOSO e informes de fases.
- **REEMPLAZADA:** uso de source grouping como split principal de selección; queda vigente solo como benchmark.
- **NO VERIFICABLE:** cifras aisladas sin artefacto recuperado y contenido numérico de `docs/etapa_2` sin fuente.
- **PENDIENTE:** HPO 60 trials, entrenamiento formal, multi-seed/CV/LOSO finales, PHash, exportación y dispositivo del checkpoint actual.

## Comandos revisados

`make help` resolvió correctamente y confirma los targets documentados. Los comandos costosos se revisaron por definición/parser, sin ejecutarlos. El comando de HPO queda planificado con `N_TRIALS=60`; no debe confundirse con el default de Makefile.

## Resultado de la auditoría

La documentación canónica queda dividida en metodología, experimentos, resultados, reproducibilidad y fuentes para tesis. Los artefactos generados permanecen separados de la narrativa manual, y el [registro de evidencias](../tesis/EVIDENCE_REGISTRY.md) es el punto de control antes de usar una cifra en tesis.
