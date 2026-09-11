import { defineConfig } from "vitepress";

const esDatasetSidebar = [
  {
    text: "Antecedentes",
    items: [
      { text: "Trabajos previos", link: "/es/previous-work/" },
    ],
  },
  {
    text: "Datasets",
    items: [
      { text: "Recopilación", link: "/es/datasets/" },
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
        text: "Maize, Beans and Tomatoes image dataset for Africa",
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
    text: "Limpieza",
    items: [
      {
        text: "Limpieza y ordenado",
        link: "/es/cleanup-and-ordered/",
      },
    ],
  },
  {
    text: "Análisis Exploratorio",
    items: [
      {
        text: "Exploración",
        link: "/es/exploratory-data-analysis/",
      },
    ],
  },
  {
    text: "Preprocesado",
    items: [
      { text: "Flujo realizado", link: "/es/preprocessed/" },
    ],
  },
  {
    text: "Procedencia y fuga",
    items: [
      { text: "Plan de experimentación", link: "/es/provenance/" },
      { text: "Consolidación y siguientes pasos", link: "/es/provenance/consolidacion" },
      { text: "Fase 0 - Auditoría", link: "/es/provenance/fase-0-auditoria" },
      { text: "Fase 1 - Partición honesta", link: "/es/provenance/fase-1-particion-honesta" },
      { text: "Fase 2 - Intervenciones", link: "/es/provenance/fase-2-intervenciones" },
      { text: "Evidencia bruta", link: "/es/provenance/evidencia/" },
    ],
  },
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
    text: "Pipelines",
    items: [
      {
        text: "Baselines",
        items: [
          { text: "Preprocesado", link: "/es/pipeline-baselines/preprocessed" },
          { text: "Entrenamiento", link: "/es/pipeline-baselines/entrenamiento" },
          { text: "Evaluación", link: "/es/pipeline-baselines/evaluacion" },
          { text: "Interpretabilidad", link: "/es/pipeline-baselines/interpretabilidad" },
          { text: "Experimentos", link: "/es/pipeline-baselines/experimentos" },
        ],
      },
      {
        text: "Principal",
        items: [
          { text: "Preprocesado", link: "/es/pipeline/preprocessed" },
          { text: "Entrenamiento", link: "/es/pipeline/entrenamiento" },
          { text: "Evaluación", link: "/es/pipeline/evaluacion" },
          { text: "Interpretabilidad", link: "/es/pipeline/interpretabilidad" },
          { text: "Experimentos", link: "/es/pipeline/experimentos" },
        ],
      },
    ],
  },
  {
    text: "Deployment",
    items: [
      { text: "GPU en Modal", link: "/es/deployment/modal" },
      { text: "App React Native", link: "/es/deployment/react-native" },
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
                  { text: "Entrenamiento", link: "/es/pipeline/entrenamiento" },
                  { text: "Evaluación", link: "/es/pipeline/evaluacion" },
                  { text: "Interpretabilidad", link: "/es/pipeline/interpretabilidad" },
                  { text: "Experimentos", link: "/es/pipeline/experimentos" },
                ],
              },
            ],
          },
          {
            text: "Deployment",
            items: [
              { text: "GPU en Modal", link: "/es/deployment/modal" },
              { text: "App React Native", link: "/es/deployment/react-native" },
            ],
          },
        ],
        sidebar: {
          "/es/": esDatasetSidebar,
          "/es/datasets/": esDatasetSidebar,
          "/es/cleanup-and-ordered/": esDatasetSidebar,
          "/es/exploratory-data-analysis/": esDatasetSidebar,
          "/es/preprocessed/": esDatasetSidebar,
          "/es/previous-work/": esDatasetSidebar,
          "/es/baselines/": esDatasetSidebar,
          "/es/pipeline-baselines/": esDatasetSidebar,
          "/es/deep-learning/": esDatasetSidebar,
          "/es/pipeline/": esDatasetSidebar,
          "/es/deployment/": esDatasetSidebar,
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
