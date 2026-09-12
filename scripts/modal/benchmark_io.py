"""Localiza el cuello de botella de la carga de datos en Modal.

Separa tres costes que el DataLoader mezcla: leer los bytes del Volume, decodificar el
JPEG y leer de un memmap ya decodificado. Sin esa separación no se puede saber si añadir
núcleos sirve o si el límite es la red del Volume.

Uso:
    modal run scripts/modal/benchmark_io.py
"""

import time

import modal

from scripts.modal._common import dataset_vol, image, outputs_vol

app = modal.App("corn-benchmark-io", image=image)

MUESTRA = 400


@app.function(
    cpu=32.0,
    volumes={"/data": dataset_vol, "/outputs": outputs_vol},
    timeout=1800,
)
def benchmark(workers: int = 32) -> dict:
    """Mide lectura cruda, decodificación y lectura de memmap sobre el mismo muestreo.

    @param {int} workers Hilos usados en las fases paralelas.
    @returns {dict} Throughput medido en cada fase.
    """
    import io
    from concurrent.futures import ThreadPoolExecutor
    from pathlib import Path

    import numpy as np
    import pandas as pd
    from PIL import Image

    dataset_vol.reload()
    outputs_vol.reload()

    import os

    print(f"[*] cores visibles: {os.cpu_count()}", flush=True)

    df = pd.read_csv("/outputs/splits/seed_42/train.csv")
    rutas = [Path("/data") / r for r in df.image_path.sample(MUESTRA, random_state=0)]

    def leer(p):
        return p.read_bytes()

    def decodificar(p):
        with Image.open(p) as im:
            return im.convert("RGB").resize((224, 224), Image.Resampling.BILINEAR).tobytes()

    def leer_y_decodificar(datos):
        with Image.open(io.BytesIO(datos)) as im:
            return im.convert("RGB").resize((224, 224), Image.Resampling.BILINEAR).tobytes()

    resultados = {"cores": os.cpu_count(), "workers": workers, "n": MUESTRA}

    inicio = time.perf_counter()
    with ThreadPoolExecutor(max_workers=workers) as pool:
        crudos = list(pool.map(leer, rutas))
    t = time.perf_counter() - inicio
    resultados["lectura_volume_img_s"] = round(MUESTRA / t, 1)
    resultados["mb_totales"] = round(sum(len(c) for c in crudos) / 1e6, 1)
    resultados["lectura_volume_mb_s"] = round(resultados["mb_totales"] / t, 1)

    inicio = time.perf_counter()
    with ThreadPoolExecutor(max_workers=workers) as pool:
        list(pool.map(leer_y_decodificar, crudos))
    t = time.perf_counter() - inicio
    resultados["decodificacion_en_ram_img_s"] = round(MUESTRA / t, 1)

    inicio = time.perf_counter()
    with ThreadPoolExecutor(max_workers=workers) as pool:
        list(pool.map(decodificar, rutas))
    t = time.perf_counter() - inicio
    resultados["volume_mas_decodificacion_img_s"] = round(MUESTRA / t, 1)

    arr = np.lib.format.open_memmap(
        "/tmp/bench.npy", mode="w+", dtype=np.uint8, shape=(MUESTRA, 256, 256, 3)
    )
    arr[:] = 7
    arr.flush()
    lectura = np.load("/tmp/bench.npy", mmap_mode="r")
    inicio = time.perf_counter()
    for indice in range(MUESTRA):
        np.array(lectura[indice])
    t = time.perf_counter() - inicio
    resultados["memmap_local_img_s"] = round(MUESTRA / t, 1)

    import shutil

    total, usado, libre = shutil.disk_usage("/tmp")
    resultados["disco_local_libre_gb"] = round(libre / 1e9, 1)
    tamano_medio = resultados["mb_totales"] / MUESTRA
    resultados["corpus_estimado_gb"] = round(tamano_medio * 33433 / 1000, 1)

    destino = Path("/tmp/copia")
    destino.mkdir(exist_ok=True)

    def copiar(par):
        indice, origen = par
        (destino / f"{indice}.jpg").write_bytes(origen.read_bytes())

    inicio = time.perf_counter()
    with ThreadPoolExecutor(max_workers=workers) as pool:
        list(pool.map(copiar, enumerate(rutas)))
    t = time.perf_counter() - inicio
    resultados["copia_a_local_mb_s"] = round(resultados["mb_totales"] / t, 1)
    resultados["copia_corpus_min_estimado"] = round(
        resultados["corpus_estimado_gb"] * 1000 / resultados["copia_a_local_mb_s"] / 60, 1
    )

    locales = sorted(destino.glob("*.jpg"))
    inicio = time.perf_counter()
    with ThreadPoolExecutor(max_workers=workers) as pool:
        list(pool.map(decodificar, locales))
    t = time.perf_counter() - inicio
    resultados["decodificacion_disco_local_img_s"] = round(MUESTRA / t, 1)

    for clave, valor in resultados.items():
        print(f"  {clave}: {valor}", flush=True)
    return resultados


@app.local_entrypoint()
def main(workers: int = 32) -> None:
    """Ejecuta el benchmark y deja el resultado en la salida estándar."""
    benchmark.remote(workers=workers)
