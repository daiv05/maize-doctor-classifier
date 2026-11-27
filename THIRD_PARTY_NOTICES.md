# Material de terceros

La licencia del repositorio no cubre ni restringe el material listado aquí.

## Fuentes de datos

| Fuente | Autor o publicador | Licencia | Imágenes |
|---|---|---|---:|
| [Maize, Beans and Tomatoes Image Dataset for Africa](https://www.kaggle.com/datasets/osutokaggle/maize-beans-and-tomatoes-image-dataset-for-africa) (compila TOM2024, Makerere, NUST y YEESI Lab) | osutokaggle y los autores de cada subconjunto | Apache 2.0; subconjuntos CC BY 4.0 y CC0 1.0 | 12 136 |
| [Maize Diseases](https://www.kaggle.com/datasets/kaustavbiswal/maize-diseases/data) | Kaustav Biswal (Kaggle) | CC BY-NC-SA 4.0 | 6 989 |
| [Maize Crop Disease (Leaf)](https://doi.org/10.17632/6w6gsvghfw.1) | Abebe Adege (2026) | CC BY 4.0 | 5 816 |
| [Corn Leaf](https://universe.roboflow.com/labonis-workspace/corn-leaf-hd9iy) | Labonis Workspace (Roboflow) | CC BY 4.0 | 2 919 |
| [CropDG Unified Multi-Domain](https://doi.org/10.34740/KAGGLE/DSV/12009964) (PlantVillage, CCMT y PlantDoc) | Saniya Verma (2024) | CC BY-NC-SA 4.0 | 2 562 |
| [Maize in Field Dataset](https://doi.org/10.34740/KAGGLE/DSV/3603983) | Hamish Craze y Dave Berger (2022) | CC BY-NC-SA 4.0 | 852 |
| [Maize 2](https://universe.roboflow.com/david-deras/maize-2-qag1o-76kth) | Workspace david-deras (Roboflow) | CC BY 4.0 | 846 |
| [Corn Leaf Diseases Classification](https://universe.roboflow.com/david-deras/corn-leaf-diseases-classifcation-nwmtk) | Workspace david-deras (Roboflow) | CC BY 4.0 | 480 |
| [Maize Nutrient Deficiency](https://doi.org/10.17632/34gb2gr7p2.1) | Vishnu Khade (2025) | CC BY 4.0 | 340 |
| [Maize Leaf](https://universe.roboflow.com/david-deras/maize-leaf-7zwoi-he5vd) | Workspace david-deras (Roboflow) | CC BY 4.0 | 331 |
| [Maize Deficiency Scanner](https://universe.roboflow.com/david-deras/maize-deficiency-scanner-z2whz) | Workspace david-deras (Roboflow) | CC BY 4.0 | 158 |

Textos de las licencias: [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/),
[CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/),
[CC0 1.0](https://creativecommons.org/publicdomain/zero/1.0/) y
[Apache 2.0](https://www.apache.org/licenses/LICENSE-2.0).

**Cambios realizados:** se seleccionaron nueve clases, se armonizaron las etiquetas, se
convirtieron las anotaciones por región en una etiqueta por imagen, se eliminaron duplicados e
imágenes con etiquetas contradictorias, y se renombraron y reorganizaron los archivos. Ninguno de
los autores originales respalda este proyecto.

## Imágenes incluidas en este repositorio

Las muestras de `public/` y las figuras con fotografías del corpus (análisis exploratorio,
aumentaciones, LIME, Grad-CAM y SHAP), incluidas las de `reports/`, son copias o adaptaciones de
imágenes de las fuentes anteriores. Cada una conserva la licencia de su fuente; las provenientes de
fuentes CC BY-NC-SA 4.0 no pueden usarse con fines comerciales.

## Librerías principales

| Librería | Licencia |
|---|---|
| torch, torchvision, pandas, scikit-learn, scikit-image, seaborn, python-dotenv | BSD |
| timm, opencv-python-headless, onnx, litert-torch, ai-edge-litert, huggingface_hub, modal, imagededup | Apache 2.0 |
| PyYAML, shap, gdown, onnxruntime, onnxscript, optuna, optuna-dashboard, VitePress | MIT |
| lime | BSD 2-Clause |
| Pillow | MIT-CMU |
| matplotlib | PSF |
| tqdm | MPL 2.0 y MIT |

Cada dependencia, incluidas las transitivas, se rige por la licencia que distribuye su paquete.
