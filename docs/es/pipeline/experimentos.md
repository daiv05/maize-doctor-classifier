# Experimentos

Con la arquitectura ya acotada a unos pocos candidatos, los experimentos del pipeline principal dejarán de comparar modelos y pasarán a comparar decisiones de entrenamiento. 

El protocolo de reparación empieza con un piloto pareado original/segmentado en desarrollo, conservando identidades y particiones, arquitectura y presupuesto, con varias semillas. Main mantiene pérdida ponderada sin sampler. Cualquier variación de fondo, regularización o cambio de balanceo requiere un protocolo separado; no se seleccionan variantes repitiendo test. Ver el [registro de reparación](../../reviews/2026-09-11-reparacion-integral).
