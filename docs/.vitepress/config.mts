import { defineConfig } from "vitepress";

const esDatasetSidebar = [
  {
    text: "Guía e inicio",
    items: [
      { text: "Empezando", link: "/es/guia/" },
    ],
  },
  {
    text: "Antecedentes",
    items: [
      { text: "Trabajos previos y literatura", link: "/es/previous-work/" },
    ],
  },
  {
    text: "Procesamiento de Datos",
    collapsed: false,
    items: [
      {
        text: "Limpieza de datos",
        link: "/es/cleanup-and-ordered/",
      },
      {
        text: "Análisis exploratorio de datos (EDA)",
        link: "/es/exploratory-data-analysis/",
      },
      {
        text: "Preprocesado y normalización",
        link: "/es/preprocessed/",
      },
    ],
  },
  {
    text: "Estudio de Atajos",
    collapsed: false,
    items: [
      { text: "Marco del estudio y objetivos", link: "/es/provenance/" },
      { text: "Detección de atajos por cámara", link: "/es/provenance/fase-0-auditoria" },
      { text: "Partición honesta por fuente (LOSO)", link: "/es/provenance/fase-1-particion-honesta" },
      { text: "Intervenciones y mitigación", link: "/es/provenance/fase-2-intervenciones" },
      { text: "Consolidación de hallazgos", link: "/es/provenance/consolidacion" },
      { text: "Evidencia y registros brutos", link: "/es/provenance/evidencia/" },
    ],
  },
  {
    text: "Modelado Deep Learning",
    collapsed: false,
    items: [
      { text: "Fundamentos teóricos", link: "/es/deep-learning/" },
      { text: "Interpretabilidad (XAI)", link: "/es/deep-learning/interpretability" },
      { text: "Detección OOD (Mahalanobis)", link: "/es/deep-learning/ood-detection" },
      { text: "Arquitecturas baseline", link: "/es/baselines/" },
    ],
  },
  {
    text: "Pipelines de Ejecución",
    collapsed: false,
    items: [
      {
        text: "Pipeline Baselines",
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
        text: "Pipeline Principal",
        collapsed: true,
        items: [
          { text: "Preprocesado", link: "/es/pipeline/preprocessed" },
          { text: "Optimización (Optuna)", link: "/es/pipeline/optimizacion" },
          { text: "Entrenamiento", link: "/es/pipeline/entrenamiento" },
          { text: "Contratos de runs", link: "/es/pipeline/contratos-runs" },
          { text: "Ensamble multimodelo", link: "/es/pipeline/ensamble" },
          { text: "Evaluación", link: "/es/pipeline/evaluacion" },
          { text: "Interpretabilidad", link: "/es/pipeline/interpretabilidad" },
          { text: "Experimentos", link: "/es/pipeline/experimentos" },
        ],
      },
    ],
  },
  {
    text: "Metodología Experimental",
    collapsed: false,
    items: [
      { text: "Pipeline de datos", link: "/es/metodologia/pipeline-datos" },
      { text: "Protocolos experimentales", link: "/es/metodologia/protocolos-experimentales" },
      { text: "Registro de experimentos", link: "/es/experimentos/" },
      { text: "Estado actual de ejecución", link: "/es/experimentos/current-status" },
      { text: "Optimización de hiperparámetros (HPO)", link: "/es/experimentos/hpo" },
      { text: "Entrenamiento formal", link: "/es/experimentos/formal-training" },
      { text: "Estabilidad multi-semilla", link: "/es/experimentos/multiseed" },
      { text: "Validación cruzada", link: "/es/experimentos/cross-validation" },
    ],
  },
  {
    text: "Resultados Experimentales",
    collapsed: false,
    items: [
      { text: "Resumen ejecutivo", link: "/es/resultados/" },
      { text: "Optimización e hiperparámetros", link: "/es/resultados/optimizacion" },
      { text: "Modelos avanzados y ensamble", link: "/es/resultados/ensamble" },
      { text: "Evaluación rigurosa y métricas finales", link: "/es/resultados/evaluacion" },
      { text: "Ficha técnica: Baseline Lite0 (seed 42)", link: "/es/resultados/run-20260921-efficientnet-lite0" },
      { text: "Ficha técnica: Benchmark por fuentes (LOSO)", link: "/es/resultados/run-20260921-source-grouped" },
      { text: "Análisis de equidad y sesgos", link: "/es/resultados/equidad" },
    ],
  },
  {
    text: "Auditoría Metodológica",
    collapsed: false,
    items: [
      { text: "Marco de auditoría", link: "/es/auditoria/" },
      { text: "Estabilidad entre semillas", link: "/es/auditoria/estabilidad-semillas" },
      { text: "Aislamiento de fuentes (Protocolo LOSO)", link: "/es/auditoria/analisis-fuentes-loso" },
      { text: "Auditoría forense de píxeles", link: "/es/auditoria/auditoria-pixeles" },
      { text: "Ensamble fuera de dominio", link: "/es/auditoria/ensamble-loso" },
    ],
  },
  {
    text: "Reproducibilidad e Integridad",
    collapsed: false,
    items: [
      { text: "Trazabilidad de reportes históricos", link: "/es/reproducibilidad/auditoria-documental-2026-09-22" },
      { text: "Continuidad del HPO y workspace", link: "/es/reproducibilidad/hpo-continuidad" },
      { text: "Procedencia y origen de figuras", link: "/es/reproducibilidad/figuras" },
      { text: "Auditoría bibliográfica y DOIs", link: "/es/reproducibilidad/referencias" },
    ],
  },
  {
    text: "Fuentes para Tesis",
    collapsed: false,
    items: [
      { text: "Resultados experimentales consolidados", link: "/es/tesis/EXPERIMENTAL_RESULTS_SUMMARY" },
      { text: "Comparativa HPO vs Baseline", link: "/es/tesis/HPO_BASELINE_COMPARISON" },
      { text: "Evolución cronológica del proyecto", link: "/es/tesis/PROJECT_EVOLUTION" },
      { text: "Resumen histórico consolidado", link: "/es/tesis/PROJECT_HISTORY_SUMMARY" },
      { text: "Registro de decisiones de diseño (ADR)", link: "/es/tesis/DECISION_LOG" },
      { text: "Hitos y entregables", link: "/es/tesis/MILESTONES" },
      { text: "Registro de evidencias experimentales", link: "/es/tesis/EVIDENCE_REGISTRY" },
      { text: "Backlog de investigación pendiente", link: "/es/tesis/RESEARCH_BACKLOG" },
      { text: "Plantillas de tablas para tesis", link: "/es/tesis/TABLE_TEMPLATES" },
      { text: "Empaquetado de modelos (README)", link: "/es/tesis/MODEL_PACKAGE_README" },
    ],
  },
  {
    text: "Datasets y Fuentes",
    collapsed: false,
    items: [
      { text: "Catálogo y recopilación general", link: "/es/datasets/" },
      { text: "Taxonomía y clases foliares", link: "/es/dataset-classes/" },
      {
        text: "Maize in Field Dataset",
        link: "/es/datasets/maize-in-field-dataset",
      },
      { text: "Maize Diseases", link: "/es/datasets/maize-diseases" },
      { text: "Corn Leaf Diseases", link: "/es/datasets/corn-leaf-diseases" },
      {
        text: "CropDG Unified Multi-Domain",
        link: "/es/datasets/cropdg-unified-multidomain",
      },
      {
        text: "Maize, Beans and Tomatoes (Africa)",
        link: "/es/datasets/maize-beans-tomatoes-africa",
      },
      {
        text: "Multicrop disease maiz pests and disease",
        link: "/es/datasets/multicrop-disease-maiz-disease-pests-and-disease",
      },
      {
        text: "Maize Nutrient Deficiency",
        link: "/es/datasets/maize-nutrient-deficiency",
      },
      {
        text: "Corn Leaf - Roboflow",
        link: "/es/datasets/corn-leaf-roboflow",
      },
      {
        text: "Maize 2 - Roboflow",
        link: "/es/datasets/maize-2-roboflow",
      },
      {
        text: "Maize Leaf - Roboflow",
        link: "/es/datasets/maize-leaf-roboflow",
      },
      {
        text: "Maize Deficiency Scanner - Roboflow",
        link: "/es/datasets/maize-deficiency-scanner-roboflow",
      },
      {
        text: "Corn Leaf Diseases Classification - Roboflow",
        link: "/es/datasets/corn-leaf-diseases-classification-roboflow",
      },
    ],
  },
  {
    text: "Deployment y Aplicación",
    collapsed: false,
    items: [
      { text: "Servicio de inferencia GPU en Modal", link: "/es/deployment/modal" },
      { text: "Aplicación móvil React Native", link: "/es/deployment/react-native" },
      { text: "Prototipo de inferencia en dispositivo", link: "/es/deployment/prototipo" },
      { text: "Manual de usuario DoctorMaiz App", link: "/es/app/" },
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
  ignoreDeadLinks: true,
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
          {
            text: "Empezando",
            link: "/es/guia/",
          },
          {
            text: "Datos",
            items: [
              {
                text: "Exploración",
                items: [
                  { text: "Datasets", link: "/es/datasets/" },
                  { text: "Limpieza", link: "/es/cleanup-and-ordered/" },
                  { text: "Análisis Exploratorio", link: "/es/exploratory-data-analysis/" },
                ],
              },
              {
                text: "Procesamiento",
                items: [
                  { text: "Preprocesado", link: "/es/preprocessed/" },
                ],
              },
            ],
          },
          {
            text: "Modelado",
            items: [
              {
                text: "Deep Learning",
                items: [
                  { text: "Teoría", link: "/es/deep-learning/" },
                  { text: "Interpretabilidad (XAI)", link: "/es/deep-learning/interpretability" },
                  { text: "Detección OOD (Mahalanobis)", link: "/es/deep-learning/ood-detection" },
                  { text: "Baselines", link: "/es/baselines/" },
                ],
              },
              {
                text: "Pipeline · Baselines",
                items: [
                  { text: "Preprocesado", link: "/es/pipeline-baselines/preprocessed" },
                  { text: "Entrenamiento", link: "/es/pipeline-baselines/entrenamiento" },
                  { text: "Evaluación", link: "/es/pipeline-baselines/evaluacion" },
                  { text: "Interpretabilidad", link: "/es/pipeline-baselines/interpretabilidad" },
                  { text: "Experimentos", link: "/es/pipeline-baselines/experimentos" },
                ],
              },
              {
                text: "Pipeline · Principal",
                items: [
                  { text: "Preprocesado", link: "/es/pipeline/preprocessed" },
                  { text: "Optimización", link: "/es/pipeline/optimizacion" },
                  { text: "Entrenamiento", link: "/es/pipeline/entrenamiento" },
                  { text: "Contratos de runs", link: "/es/pipeline/contratos-runs" },
                  { text: "Ensamble", link: "/es/pipeline/ensamble" },
                  { text: "Evaluación", link: "/es/pipeline/evaluacion" },
                  { text: "Interpretabilidad", link: "/es/pipeline/interpretabilidad" },
                  { text: "Experimentos", link: "/es/pipeline/experimentos" },
                ],
              },
            ],
          },
          {
            text: "Resultados",
            items: [
              { text: "Resumen ejecutivo", link: "/es/resultados/" },
              { text: "Optimización e hiperparámetros", link: "/es/resultados/optimizacion" },
              { text: "Modelos avanzados y ensamble", link: "/es/resultados/ensamble" },
              { text: "Evaluación rigurosa y métricas finales", link: "/es/resultados/evaluacion" },
              { text: "Ficha técnica: Baseline Lite0 (seed 42)", link: "/es/resultados/run-20260921-efficientnet-lite0" },
              { text: "Ficha técnica: Benchmark por fuentes (LOSO)", link: "/es/resultados/run-20260921-source-grouped" },
              { text: "Análisis de equidad y sesgos", link: "/es/resultados/equidad" },
            ],
          },
          {
            text: "Investigación",
            items: [
              { text: "Auditoría metodológica (5 puntos)", link: "/es/auditoria/" },
              { text: "Pipeline y protocolos", link: "/es/metodologia/pipeline-datos" },
              { text: "Estado experimental", link: "/es/experimentos/current-status" },
              { text: "Evolución histórica", link: "/es/tesis/PROJECT_EVOLUTION" },
              { text: "Registro de evidencias", link: "/es/tesis/EVIDENCE_REGISTRY" },
              { text: "Trazabilidad de reportes históricos", link: "/es/reproducibilidad/auditoria-documental-2026-09-22" },
            ],
          },
          {
            text: "Deployment",
            items: [
              { text: "GPU en Modal", link: "/es/deployment/modal" },
              { text: "App React Native", link: "/es/deployment/react-native" },
              { text: "Prototipo en dispositivo", link: "/es/deployment/prototipo" },
              { text: "Documentación App Móvil", link: "/es/app/" },
            ],
          },
          {
            text: "App Móvil",
            link: "/es/app/",
          },
        ],
        sidebar: {
          "/es/app/": esAppSidebar,
          "/es/guia/": esDatasetSidebar,
          "/es/": esDatasetSidebar,
          "/es/datasets/": esDatasetSidebar,
          "/es/dataset-classes/": esDatasetSidebar,
          "/es/cleanup-and-ordered/": esDatasetSidebar,
          "/es/exploratory-data-analysis/": esDatasetSidebar,
          "/es/preprocessed/": esDatasetSidebar,
          "/es/previous-work/": esDatasetSidebar,
          "/es/baselines/": esDatasetSidebar,
          "/es/pipeline-baselines/": esDatasetSidebar,
          "/es/deep-learning/": esDatasetSidebar,
          "/es/pipeline/": esDatasetSidebar,
          "/es/provenance/": esDatasetSidebar,
          "/es/resultados/": esDatasetSidebar,
          "/es/metodologia/": esDatasetSidebar,
          "/es/experimentos/": esDatasetSidebar,
          "/es/reproducibilidad/": esDatasetSidebar,
          "/es/tesis/": esDatasetSidebar,
          "/es/deployment/": esDatasetSidebar,
          "/es/auditoria/": esDatasetSidebar,
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
    // en: {
    //   label: 'English',
    //   lang: 'en',
    //   link: '/en/',
    //   title: 'Corn Leaf Disease Detection',
    //   description: 'Detection of Corn Leaf Diseases using Deep Learning on Mobile Devices',
    //   themeConfig: {
    //     nav: [
    //       { text: 'Home', link: '/en/' },
    //       { text: 'Datasets', link: '/en/datasets/' },
    //     ],
    //     sidebar: {
    //       '/en/': enDatasetSidebar,
    //       '/en/datasets/': enDatasetSidebar,
    //     },
    //   },
    // },
  },

  themeConfig: {
    search: {
      provider: "local",
    },
  },

});
