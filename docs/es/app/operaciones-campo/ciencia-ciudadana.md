---
title: Aporte a la Ciencia y Dataset Nacional
description: Módulo de contribución ciudadana para enriquecer el modelo con datos agrícolas de Centroamérica y El Salvador.
---

# Aporte a la Ciencia: Dataset Nacional de Maíz

El entrenamiento de algoritmos de inteligencia artificial depende de la diversidad geográfica de sus fotografías. Sin embargo, en la investigación inicial de DoctorMaiz se evidenció una realidad contundente: **de los 14 datasets públicos internacionales evaluados, ninguno incluía imágenes capturadas en Centroamérica**.

DoctorMaiz convierte a cada agricultor en un colaborador activo de la ciencia mediante su módulo de **Contribución Ciudadana**.

---

<PhoneGallery gap="2rem">
  <PhoneMockup 
    src="/app/pixel4_contribute.png" 
    caption="Módulo de Contribución" 
    subtitle="Selección y envío de fotografías" 
  />
  <PhoneMockup 
    src="/app/contribuir.png" 
    caption="Validación Comunitaria" 
    subtitle="Aporte a la investigación regional" 
  />
</PhoneGallery>

---

## Por qué los Modelos Globales Fallan en Nuestra Región

Los modelos entrenados con imágenes de otros continentes sufren una pérdida de precisión cuando se aplican en parcelas locales debido a factores ecológicos particulares:

1. **Variedades y Genética Criolla:** Las variedades locales (como híbridos *CENTA H-59*, *Pasaquina* o maíces criollos de montaña) presentan una textura foliar, vellosidad y coloración distinta a las variedades de Norteamérica.
2. **Suelos Volcánicos y Deficiencias Complejas:** Los andosoles y vertisoles salvadoreños presentan dinámicas de fijación de fósforo muy marcadas, provocando manchas rojizas que modelos extranjeros confunden fácilmente con virosis.
3. **Clases poco representadas en datasets internacionales:** Las deficiencias nutricionales de fósforo y potasio tienen muy pocas imágenes públicas de campo, sobre todo de variedades locales.

---

## Cómo Funciona el Envío de Muestras

El agricultor puede postular fotografías de sus escaneos para que formen parte del **Dataset Abierto Nacional**:

```mermaid
graph TD
    A[Foto en Parcela] --> B[DoctorMaiz App]
    B --> C[Opción: 'Aportar al Dataset']
    C --> D[Envío al servidor con cuenta registrada]
    D --> E[Revisión por Especialistas Agrónomos]
    E --> F[Inclusión en una futura versión del modelo, si se decide]
```

### Qué se envía
- **Lo que se envía:** la fotografía, la etiqueta que elegiste, tu nota opcional y la fecha del escaneo.
- **Lo que no se envía:** tu nombre, teléfono, correo y coordenadas GPS del escaneo. La app no los incluye en el envío.
- **Pendiente de confirmar:** la app todavía no limpia los metadatos de la fotografía antes de enviarla.
