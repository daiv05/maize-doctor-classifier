# Licencia de los modelos entrenados

Aplica a los pesos entrenados (`*.pth`), a sus exportaciones (`*.onnx`, `*.tflite`) y a los
archivos derivados que los acompañan (`labels.json`, `ood_stats.json`).

Titular: Universidad de El Salvador.

Licencia: [Creative Commons Atribución-NoComercial-CompartirIgual 4.0 Internacional
(CC BY-NC-SA 4.0)](https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode.es).

## Motivo

El 31 % de las imágenes del corpus (10 403 de 33 429) proviene de fuentes publicadas bajo
CC BY-NC-SA 4.0. No está resuelto jurídicamente si unos pesos son material adaptado de las
imágenes de entrenamiento, y se adopta la lectura conservadora.

## Condiciones

- **Atribución:** citar a la Universidad de El Salvador, a los autores del proyecto (Josias Abner
  Rivas Fuentes, David Alejandro Deras Cerros y Elmer Edenilson Rosales Molina) y a las fuentes
  de datos listadas en `THIRD_PARTY_NOTICES.md`.
- **No comercial:** no se permite venderlos ni usarlos con fines principalmente comerciales.
- **Compartir igual:** las adaptaciones que se compartan usan esta misma licencia.

Quien reciba el modelo, por ejemplo dentro de la aplicación, conserva los derechos que otorga esta
licencia. La licencia propietaria del código no los restringe.

Los pesos preentrenados de partida provienen de [timm](https://github.com/huggingface/pytorch-image-models)
(Apache 2.0) y se entrenaron con ImageNet, que tiene términos de acceso propios.
