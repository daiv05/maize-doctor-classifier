---
title: Preguntas Frecuentes y Solución de Problemas
description: Guía de resolución de dudas técnicas, problemas de cámara, conectividad y buenas prácticas en campo.
---

# Preguntas Frecuentes y Solución de Problemas

Esta sección reúne las consultas y dificultades más comunes reportadas por productores, agrónomos y técnicos durante el uso de DoctorMaiz en parcelas reales.

---

<PhoneGallery gap="2rem">
  <PhoneMockup 
    src="/app/pixel4_history.png" 
    caption="Historial y Filtros" 
    subtitle="Revisión de escaneos guardados" 
  />
  <PhoneMockup 
    src="/app/pixel4_forgot_password.png" 
    caption="Recuperación de Cuenta" 
    subtitle="Autogestión de credenciales" 
  />
</PhoneGallery>

---

## Preguntas Frecuentes (FAQ)

### ¿DoctorMaiz requiere conexión a internet para dar un diagnóstico?
**No.** La red neuronal de visión artificial (`efficientnet_lite0`) corre de forma autónoma dentro del procesador del teléfono mediante TensorFlow Lite. Puedes estar en la montaña más aislada sin señal de celular y obtendrás el diagnóstico en aproximadamente **60 milisegundos**. Solo se requiere conexión para consultar el clima en vivo, sincronizar con la nube o aportar fotos al dataset.

### ¿Por qué la aplicación me muestra el mensaje "No Reconocida"?
El mensaje *"No Reconocida"* no es un error de la app: es una **salvaguarda de seguridad fitosanitaria** activada por el detector OOD (*Out-Of-Distribution*). Significa que la foto no cumple con los patrones visuales de una hoja de maíz analizable. Ocurre comúnmente si:
- La foto está desenfocada o movida.
- La hoja está demasiado lejos y la foto contiene más suelo que hoja.
- Hay un destello de luz solar directa muy fuerte sobre la superficie foliar.
- Se enfocó una maleza o cultivo diferente al maíz.

**Solución:** Acércate a 20–30 cm, haz que la hoja llene el marco verde central, haz sombra con tu cuerpo si el sol es muy intenso y vuelve a presionar el disparador.

### ¿Cómo distingo rápidamente en campo entre Roya y Mancha de Asfalto?
Aplica la **"Prueba del Frotado con la Yema del Dedo"**:
- **Roya Común:** Las pústulas rompen la cutícula foliar. Al frotar la lesión con el dedo, este se manchará con un polvo fino color canela o marrón (esporas libres).
- **Mancha de Asfalto:** Las lesiones son estromas negros lisos y brillantes que crecen dentro del tejido. Al frotar o raspar suavemente con la uña, **no manchan el dedo y no se desprenden**.

### ¿Mis fotografías y coordenadas son públicas o privadas?
- **En Modo Invitado:** Todos los registros, fotos y coordenadas permanecen estrictamente dentro de la memoria interna de tu teléfono celular. Nada se envía a servidores externos.
- **En Modo Registrado:** Tus escaneos se respaldan en tu cuenta privada.
- **Al Contribuir al Dataset Nacional:** Solo las fotos que tú decidas postular voluntariamente se comparten con la investigación científica. En ese proceso, **se borran tus datos personales** y las coordenadas GPS se aproximan a nivel de municipio para resguardar la privacidad de tu parcela.

### ¿Por qué algunos escaneos no aparecen sobre el mapa de cobertura?
Si un escaneo aparece en la lista de historial pero no se ve sobre el mapa, se debe a que el teléfono no tenía señal de satélites GPS en el momento exacto del disparo.  
**Solución:** Antes de iniciar el recorrido en la milpa, asegúrate de tener encendida la opción de **Ubicación** en la barra de ajustes rápidos de Android y dale 30 segundos al teléfono a cielo abierto para fijar satélites.

### ¿DoctorMaiz puede sustituir el criterio de un agrónomo?
**No.** DoctorMaiz es una herramienta de asistencia diagnóstica rápida. Las recomendaciones del acordeón de severidad sugieren ingredientes activos homologados (como *Azoxistrobina*, *Tebuconazol* o *Bacillus subtilis*), pero la dosis comercial exacta y la decisión final de compra deben ajustarse según la edad fenológica del cultivo y las recomendaciones del técnico de tu cooperativa o centro de extensión agrícola (como CENTA o MAG).

---

## Diagnóstico Rápido de Errores Comunes

| Síntoma en la App | Causa Probable | Solución Inmediata |
|---|---|---|
| **Pantalla de cámara en negro** | Permiso de cámara revocado en Android | Ve a *Ajustes del Teléfono > Aplicaciones > DoctorMaiz > Permisos* y activa "Cámara". |
| **El mapa satelital aparece en gris** | Falta de conexión a internet para descargar mosaicos | Cambia la capa a "Mapa Normal" o conéctate brevemente a Wi-Fi antes de salir al campo para precargar el cuadrante del lote. |
| **El escaneo tarda más de 2 segundos** | Teléfono con modo de ahorro extremo de batería activo | Desactiva el modo de ahorro de batería de Android para permitir que la CPU ejecute el modelo TFLite a su frecuencia nominal. |
| **Alerta "Memoria insuficiente"** | Historial de fotos locales saturó el almacenamiento | Dirígete a *Perfil*, realiza una copia de seguridad o sincroniza tus datos y limpia las fotos antiguas del almacenamiento temporal. |
