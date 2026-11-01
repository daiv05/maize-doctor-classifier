# Entrenamiento

Cada arquitectura se construye desde un registro de modelos único, con backbone pre-entrenado en ImageNet y solo la capa de clasificación reemplazada por una lineal de 9 salidas.

## Configuración canónica

Todos los baselines comparten exactamente los mismos hiperparámetros de entrenamiento. Fijarlos así permite que las diferencias de resultado entre arquitecturas se deban al modelo en sí, no a condiciones de entrenamiento distintas.

| Hiperparámetro | Valor |
|---|---|
| Clases | 9 (perfil `baseline`, cap 1 500/clase) |
| Épocas | 30 |
| Optimizador | AdamW (lr 1e-4, weight decay 1e-4) |
| Pérdida | CrossEntropyLoss sin ponderar por clase |
| Balanceo | WeightedRandomSampler + augmentation minority en caliente |
| Pesos iniciales | ImageNet (`pretrained=True`) |
| `image_size` | por modelo (los 3 baselines usan 224x224) |
| Batch size | 32 (auto-escalado según resolución del modelo) |

Cada corrida se versiona en `outputs/baselines/<modelo>/<run_id>/` con `best.pth`, `summary.json`, `predictions.csv`, `train_history.csv` y los reportes de test. El `summary.json`, como su nombre indica, resume la corrida y permite comparar resultados generales entre modelos y runs.

## Modelos baseline

Se adoptan tres arquitecturas ligeras que cubren el eje precisión / eficiencia y que son convertibles a TensorFlow Lite para despliegue móvil: `efficientnet_b0`, `shufflenet_v2_x1_0` y `efficientnet_lite0`.

## Dinámica de convergencia

Las tres corridas de 30 épocas sobre GPU en Modal (~60 s/época, ~30 min por modelo) muestran un comportamiento sano: el macro-F1 de validación sube rápido, se estabiliza y no colapsa. La figura traza el macro-F1 de validación por época; el punto marca la mejor época de cada modelo (la que produce `best.pth`).

![Convergencia - macro-F1 de validación por época](/baselines/baseline_convergencia.png)

Dos observaciones tienen valor práctico:

- **El backbone pre-entrenado en ImageNet paga desde el inicio.** `efficientnet_b0` y `efficientnet_lite0` cruzan la meta de 0.85 en 3–7 épocas. `shufflenet_v2_x1_0` arranca mucho más abajo (0.57 en la época 1) y necesita 6 épocas para alcanzarla: converge más lento pero llega al mismo rango final. Esto sugiere que, para shufflenet, un presupuesto de épocas mayor o un warmup de learning rate podrían dar margen adicional.
- **No hay sobreajuste destructivo.** El macro-F1 de validación se mantiene en meseta alta (~0.88–0.91) hasta la época 30 sin degradarse, aunque el train macro-F1 ya supera 0.99. El gap train/val es el esperado con este tamaño de dataset y no exige early-stopping agresivo; aun así, se conserva `best.pth` (mejor época) además de `last.pth`.

## Ejecución

El proyecto y su pipeline permiten que el entrenamiento se puede lanzar en local o delegarlo a una GPU en <a href="https://modal.com" rel="noopener noreferrer" target="_blank">Modal</a>; ambos caminos usan el mismo `train_baselines.py` por debajo.

```bash
# Local
make train-baselines

# GPU en Modal (regenera splits si cambió el perfil)
make modal-train-baselines REGEN_SPLITS=1
```
