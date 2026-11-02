---
title: Preguntas Frecuentes y Solución de Problemas
description: Guía de resolución de dudas técnicas, problemas de cámara, conectividad y buenas prácticas en campo.
---

# Preguntas Frecuentes y Solución de Problemas

Esta sección reúne las consultas y dificultades más comunes reportadas por productores, agrónomos y técnicos durante el uso de DoctorMaiz en parcelas reales.

---

<PhoneGallery gap="2rem">
  <PhoneMockup 
    src="/app/pixel4_history.png" caption="Historial y Filtros" subtitle="Revisión de escaneos guardados" />
  <PhoneMockup 
    src="/app/pixel4_forgot_password.png" caption="Recuperación de Cuenta" subtitle="Autogestión de credenciales" />
</PhoneGallery>

---

## Preguntas Frecuentes (FAQ)

### ¿DoctorMaiz requiere conexión a internet para dar un diagnóstico?
**No.** La red neuronal de visión artificial (`efficientnet_lite0`) corre de forma autónoma dentro del procesador del teléfono mediante TensorFlow Lite. Puedes estar en la montaña más aislada sin señal de celular y obtendrás el diagnóstico. En el teléfono de pruebas la inferencia toma 23 ms y el preprocesado de la foto 98 ms (medianas). Solo se requiere conexión para consultar el clima en vivo, cargar los mosaicos del mapa, sincronizar con la nube o aportar fotos al dataset.

### ¿Por qué la aplicación me muestra el mensaje "No Reconocida"?
El mensaje *"No Reconocida"* no es un error de la app: es el resultado de tres controles que se aplican a cada foto. La imagen se rechaza si la clase más probable tiene menos de 60 % de probabilidad, si las dos clases más probables quedan a menos de 15 puntos porcentuales entre sí, o si el detector OOD (*Out-Of-Distribution*) mide que la foto está lejos de las imágenes de entrenamiento. En las 95 fotos de campo externas al corpus que se probaron en el teléfono, el detector OOD marcó el 52.6 %, frente al 8.3 % de las 48 imágenes del corpus. Una imagen no reconocida no se guarda en el historial.

**Solución:** Acércate a 20–30 cm, haz que la hoja llene el marco central, evita suelo y sombras fuertes en el encuadre y vuelve a presionar el disparador.

### ¿Cómo distingo rápidamente en campo entre Roya y Mancha Gris?
- **Roya Común:** Son pústulas redondeadas de 1 a 2 mm que rompen la cutícula foliar. Al frotar la lesión con el dedo, este se manchará con un polvo fino color canela o marrón.
- **Mancha Gris:** Son lesiones rectangulares de 1 a 6 cm, delimitadas por las nervaduras. No producen polvo al frotarlas.

### ¿Mis fotografías y coordenadas son públicas o privadas?
- **Escaneos y fotos:** el historial, las fotos y las coordenadas de tus escaneos se guardan en el teléfono y no se suben al servidor, tengas o no una cuenta.
- **Datos que sí salen del teléfono sin cuenta:** tus coordenadas GPS se envían a Open-Meteo para obtener el clima, y el mapa de cobertura descarga mosaicos de ArcGIS y OpenStreetMap, que reciben las coordenadas de los mosaicos que se visualizan. Si la app está configurada con un servidor propio, también consulta si hay una versión nueva.
- **Cuenta registrada:** la cuenta solo sirve para sincronizar tus correcciones y tus aportes al dataset; los escaneos no se respaldan.
- **Al Contribuir al Dataset Nacional:** Solo se envían las fotos que tú decidas aportar voluntariamente, junto con la etiqueta que elegiste, tu nota opcional y la fecha del aporte. El envío no incluye tu nombre, teléfono, correo ni coordenadas GPS. El aporte se guarda primero en el teléfono y se sube cuando hay una sesión iniciada y conexión a internet.

### ¿Por qué algunos escaneos no aparecen sobre el mapa de cobertura?
Si un escaneo aparece en la lista de historial pero no se ve sobre el mapa, el escaneo no tiene coordenadas guardadas: el teléfono no obtuvo ubicación al momento del disparo.  
**Solución:** Antes de iniciar el recorrido en la milpa, asegúrate de tener encendida la opción de **Ubicación** en la barra de ajustes rápidos de Android y dale 30 segundos al teléfono a cielo abierto para fijar satélites.

### ¿DoctorMaiz puede sustituir el criterio de un agrónomo?
**No.** DoctorMaiz es una herramienta de asistencia diagnóstica rápida. Las recomendaciones de la app mencionan grupos de productos (por ejemplo triazoles, estrobilurinas o *Bacillus*), pero la dosis comercial exacta y la decisión final de compra deben ajustarse según la edad fenológica del cultivo y las recomendaciones del técnico de tu cooperativa o centro de extensión agrícola (como CENTA o MAG).

---

## Diagnóstico Rápido de Errores Comunes

| Síntoma en la App | Causa Probable | Solución Inmediata |
|---|---|---|
| **Pantalla de cámara en negro** | Permiso de cámara revocado en Android | Ve a *Ajustes del Teléfono > Aplicaciones > DoctorMaiz > Permisos* y activa "Cámara". |
| **El mapa aparece en gris** | Falta de conexión a internet: las capas Satelital y Estándar descargan sus mosaicos en línea | Conéctate a internet para ver el mapa. |
| **El escaneo tarda más de 2 segundos** | Teléfono con modo de ahorro extremo de batería activo | Desactiva el modo de ahorro de batería de Android para permitir que la CPU ejecute el modelo TFLite a su frecuencia nominal. |
