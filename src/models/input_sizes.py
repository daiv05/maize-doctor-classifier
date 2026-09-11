"""Resolución de entrada nativa por modelo.

Algunas arquitecturas fueron diseñadas y preentrenadas a una resolución distinta
de los 224x224 por defecto del pipeline. Entrenarlas a 224 desperdicia capacidad
(EfficientNet-B4) o penaliza los pesos preentrenados (FastViT-T8). Este mapa las
declara explícitamente; los modelos ausentes usan el `fallback` (target_size del YAML)
"""

from __future__ import annotations

# Solo excepciones: (alto, ancho), convención (h, w) de torchvision.
MODEL_INPUT_SIZES: dict[str, tuple[int, int]] = {
    "efficientnet_b4": (380, 380),  # diseñado para 380x380
    "fastvit_t8": (256, 256),  # preentrenado a 256x256
}


def resolve_input_size(
    name: str, fallback: tuple[int, int] = (224, 224)
) -> tuple[int, int]:
    """Devuelve la resolución nativa del modelo, o `fallback` si no está declarada."""
    return MODEL_INPUT_SIZES.get(name, fallback)
