import { defineConfig } from "vitepress";

const esDatasetSidebar = [
  {
    text: "Guía",
    items: [
      { text: "Empezando", link: "/es/guia/" },
      { text: "Trabajos previos", link: "/es/previous-work/" },
    ],
  },
  {
    text: "Datos",
    collapsed: false,
    items: [
      {
        text: "Fuentes de datos",
        link: "/es/datasets/",
        collapsed: true,
        items: [
          { text: "Maize in Field Dataset", link: "/es/datasets/maize-in-field-dataset" },
          { text: "Maize Diseases", link: "/es/datasets/maize-diseases" },
          { text: "Corn Leaf Diseases", link: "/es/datasets/corn-leaf-diseases" },
          { text: "CropDG Unified Multi-Domain", link: "/es/datasets/cropdg-unified-multidomain" },
          { text: "Maize, Beans and Tomatoes (Africa)", link: "/es/datasets/maize-beans-tomatoes-africa" },
          { text: "Maize Crop Disease (Leaf)", link: "/es/datasets/multicrop-disease-maiz-disease-pests-and-disease" },
          { text: "Maize Nutrient Deficiency", link: "/es/datasets/maize-nutrient-deficiency" },
          { text: "Corn Leaf - Roboflow", link: "/es/datasets/corn-leaf-roboflow" },
          { text: "Maize 2 - Roboflow", link: "/es/datasets/maize-2-roboflow" },
          { text: "Maize Leaf - Roboflow", link: "/es/datasets/maize-leaf-roboflow" },
          { text: "Maize Deficiency Scanner - Roboflow", link: "/es/datasets/maize-deficiency-scanner-roboflow" },
          { text: "Corn Leaf Diseases Classification - Roboflow", link: "/es/datasets/corn-leaf-diseases-classification-roboflow" },
        ],
      },
      { text: "Clases foliares", link: "/es/dataset-classes/" },
      { text: "Limpieza y ordenado", link: "/es/cleanup-and-ordered/" },
      { text: "Análisis exploratorio (EDA)", link: "/es/exploratory-data-analysis/" },
      { text: "Preprocesado", link: "/es/preprocessed/" },
      { text: "Pipeline de datos e identidad", link: "/es/metodologia/pipeline-datos" },
    ],
  },
  {
    text: "Modelado",
    collapsed: false,
    items: [
      { text: "Fundamentos teóricos", link: "/es/deep-learning/" },
      { text: "Arquitecturas y baselines", link: "/es/baselines/" },
      { text: "Interpretabilidad (XAI)", link: "/es/deep-learning/interpretability" },
      { text: "Detección OOD (Mahalanobis)", link: "/es/deep-learning/ood-detection" },
    ],
  },
  {
    text: "Pipelines",
    collapsed: false,
    items: [
      {
        text: "Pipeline baselines",
        collapsed: true,
        items: [
          { text: "Preprocesado", link: "/es/pipeline-baselines/preprocessed" },
          { text: "Entrenamiento", link: "/es/pipeline-baselines/entrenamiento" },
          { text: "Evaluación", link: "/es/pipeline-baselines/evaluacion" },
          { text: "Interpretabilidad", link: "/es/pipeline-baselines/interpretabilidad" },
          { text: "Experimentos", link: "/es/pipeline-baselines/experimentos" },
        ],
      },
      {
        text: "Pipeline principal",
        collapsed: false,
        items: [
          { text: "Preprocesado", link: "/es/pipeline/preprocessed" },
          { text: "Optimización (Optuna)", link: "/es/pipeline/optimizacion" },
          { text: "Entrenamiento", link: "/es/pipeline/entrenamiento" },
          { text: "Contratos de runs", link: "/es/pipeline/contratos-runs" },
          { text: "Ensamble", link: "/es/pipeline/ensamble" },
          { text: "Evaluación y equidad", link: "/es/pipeline/evaluacion" },
          { text: "Interpretabilidad", link: "/es/pipeline/interpretabilidad" },
          { text: "Experimentos", link: "/es/pipeline/experimentos" },
        ],
      },
      { text: "Protocolos experimentales", link: "/es/metodologia/protocolos-experimentales" },
      { text: "Optimización de hiperparámetros", link: "/es/experimentos/hpo" },
      { text: "Comparación multi-semilla", link: "/es/experimentos/multiseed" },
    ],
  },
  {
    text: "Resultados",
    collapsed: false,
    items: [
      { text: "Modelo oficial y resumen", link: "/es/resultados/" },
      { text: "Ficha del modelo oficial", link: "/es/resultados/run-20260921-efficientnet-lite0" },
      { text: "Generalización entre fuentes", link: "/es/resultados/generalizacion" },
    ],
  },
  {
    text: "Reproducibilidad",
    collapsed: false,
    items: [
      { text: "Registro de evidencias", link: "/es/reproducibilidad/evidencias" },
      { text: "Procedencia de figuras", link: "/es/reproducibilidad/figuras" },
    ],
  },
  {
    text: "Despliegue y app",
    collapsed: false,
    items: [
      { text: "GPU en Modal", link: "/es/deployment/modal" },
      { text: "Aplicación React Native", link: "/es/deployment/react-native" },
      { text: "Prototipo en dispositivo", link: "/es/deployment/prototipo" },
      { text: "Manual de la app", link: "/es/app/" },
    ],
  },
];

const esAppSidebar = [
  {
    text: "DoctorMaiz App",
    items: [
      { text: "Visión General", link: "/es/app/" },
    ],
  },
  {
    text: "Guía de Inicio",
    collapsed: false,
    items: [
      { text: "Instalación y Requisitos", link: "/es/app/guia-inicio/instalacion" },
      { text: "Primeros Pasos y Cuentas", link: "/es/app/guia-inicio/primeros-pasos" },
    ],
  },
  {
    text: "Operaciones en Campo",
    collapsed: false,
    items: [
      { text: "Escaneo Guiado en Surco", link: "/es/app/operaciones-campo/escaneo-guiado" },
      { text: "Diagnóstico y Severidad", link: "/es/app/operaciones-campo/diagnostico-severidad" },
      { text: "Monitoreo Agroclimático", link: "/es/app/operaciones-campo/monitoreo-agroclimatico" },
      { text: "Mapa y Cobertura Satelital", link: "/es/app/operaciones-campo/mapa-cobertura" },
      { text: "Aporte al Dataset Nacional", link: "/es/app/operaciones-campo/ciencia-ciudadana" },
    ],
  },
  {
    text: "Sanidad Vegetal",
    collapsed: false,
    items: [
      { text: "Catálogo de Enfermedades", link: "/es/app/catalogo-enfermedades/" },
    ],
  },
  {
    text: "Arquitectura Técnica",
    collapsed: false,
    items: [
      { text: "Pipeline de Inferencia Edge", link: "/es/app/arquitectura-tecnica/pipeline-inferencia" },
      { text: "Base de Datos y Persistencia", link: "/es/app/arquitectura-tecnica/base-datos-offline" },
    ],
  },
  {
    text: "Soporte y FAQ",
    items: [
      { text: "Preguntas Frecuentes", link: "/es/app/faq" },
    ],
  },
];

export default defineConfig({
  vite: {
    publicDir: "../public",
  },

  head: [["meta", { name: "robots", content: "noindex, nofollow" }]],

  markdown: {
    math: true,
    config: (md) => {
      const defaultFence = md.renderer.rules.fence!;
      md.renderer.rules.fence = (tokens, idx, options, env, self) => {
        const token = tokens[idx];
        const info = token.info.trim();
        if (info === "mermaid") {
          const code = encodeURIComponent(token.content);
          return `<MermaidChart code="${code}" />`;
        }
        return defaultFence(tokens, idx, options, env, self);
      };
    },
  },

  lastUpdated: true,

  locales: {
    es: {
      label: "Español",
      lang: "es-SV",
      link: "/es/",
      title: "DoctorMaiz",
      head: [
        [
          "meta",
          {
            name: "description",
            content:
              "Detección de Enfermedades Foliares en Cultivos de Maíz mediante Deep Learning en Dispositivos Móviles",
          },
        ],
        [
          "meta",
          {
            name: "keywords",
            content:
              "maíz, enfermedades foliares, detección, deep learning, dispositivos móviles, datasets, agricultura de precisión",
          },
        ],
        ["link", { rel: "icon", type: "image/svg+xml", href: "/logo.svg" }],
        ["link", { rel: "icon", type: "image/png", href: "/logo.png" }],
      ],
      description:
        "Detección de Enfermedades Foliares en Cultivos de Maíz mediante Deep Learning en Dispositivos Móviles",
      themeConfig: {
        nav: [
          { text: "Empezando", link: "/es/guia/" },
          {
            text: "Datos",
            items: [
              { text: "Fuentes de datos", link: "/es/datasets/" },
              { text: "Limpieza", link: "/es/cleanup-and-ordered/" },
              { text: "Análisis exploratorio", link: "/es/exploratory-data-analysis/" },
              { text: "Preprocesado", link: "/es/preprocessed/" },
            ],
          },
          {
            text: "Modelado",
            items: [
              { text: "Fundamentos teóricos", link: "/es/deep-learning/" },
              { text: "Arquitecturas y baselines", link: "/es/baselines/" },
              { text: "Interpretabilidad (XAI)", link: "/es/deep-learning/interpretability" },
              { text: "Detección OOD", link: "/es/deep-learning/ood-detection" },
              { text: "Entrenamiento", link: "/es/pipeline/entrenamiento" },
              { text: "Contratos de runs", link: "/es/pipeline/contratos-runs" },
            ],
          },
          {
            text: "Resultados",
            items: [
              { text: "Modelo oficial y resumen", link: "/es/resultados/" },
              { text: "Generalización entre fuentes", link: "/es/resultados/generalizacion" },
              { text: "Registro de evidencias", link: "/es/reproducibilidad/evidencias" },
            ],
          },
          {
            text: "Despliegue",
            items: [
              { text: "GPU en Modal", link: "/es/deployment/modal" },
              { text: "App React Native", link: "/es/deployment/react-native" },
              { text: "Prototipo en dispositivo", link: "/es/deployment/prototipo" },
            ],
          },
          { text: "App Móvil", link: "/es/app/" },
        ],
        sidebar: {
          "/es/app/": esAppSidebar,
          "/es/": esDatasetSidebar,
        },
        search: {
          provider: "local",
          options: {
            translations: {
              button: {
                buttonText: "Buscar",
                buttonAriaLabel: "Buscar",
              },
              modal: {
                noResultsText: "No se encontraron resultados",
                resetButtonTitle: "Limpiar búsqueda",
                footer: {
                  selectText: "Seleccionar",
                  navigateText: "Navegar",
                  closeText: "Cerrar",
                },
              },
            },
          },
        },
        outlineTitle: "En esta página",
        docFooter: { prev: "Anterior", next: "Siguiente" },
        darkModeSwitchLabel: "Apariencia",
        sidebarMenuLabel: "Menú",
        returnToTopLabel: "Volver arriba",
        langMenuLabel: "Idioma",
        logo: "/logo.svg",
        lastUpdated: {
          text: "Última actualización",
          formatOptions: {
            year: "numeric",
            month: "long",
            day: "numeric",
          },
        },
      },
    },
  },

  themeConfig: {
    search: {
      provider: "local",
    },
  },

});
