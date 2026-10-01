<script setup lang="ts">
import { ref, onMounted, watch, nextTick } from 'vue'
import { useData } from 'vitepress'

const props = defineProps<{
  code: string
}>()

const { isDark } = useData()
const svgContent = ref('')
const errorMsg = ref('')
const isRendering = ref(true)

let renderCounter = 0

async function renderDiagram() {
  if (typeof window === 'undefined') return

  isRendering.value = true
  errorMsg.value = ''

  try {
    if (document.fonts) {
      await document.fonts.ready
    }

    const mermaid = (await import('mermaid')).default
    
    mermaid.initialize({
      startOnLoad: false,
      theme: isDark.value ? 'dark' : 'neutral',
      flowchart: {
        htmlLabels: true,
        useMaxWidth: true,
        padding: 24,
        nodeSpacing: 50,
        rankSpacing: 50,
        curve: 'basis',
        wrappingWidth: 350,
        minNodeWidth: 170
      },
      themeVariables: {
        fontFamily: 'Inter, system-ui, -apple-system, sans-serif',
        fontSize: '13.5px',
        darkMode: isDark.value,
        primaryColor: isDark.value ? '#1e293b' : '#f1f5f9',
        primaryTextColor: isDark.value ? '#f1f5f9' : '#0f172a',
        primaryBorderColor: isDark.value ? '#38bdf8' : '#0284c7',
        lineColor: isDark.value ? '#94a3b8' : '#64748b',
        secondaryColor: isDark.value ? '#0f172a' : '#f8fafc',
        tertiaryColor: isDark.value ? '#1e293b' : '#ffffff'
      },
      securityLevel: 'loose'
    })

    const rawCode = decodeURIComponent(props.code).trim()
    const uniqueId = `mermaid-svg-${Date.now()}-${++renderCounter}`

    const { svg } = await mermaid.render(uniqueId, rawCode)
    svgContent.value = svg
  } catch (err: any) {
    console.error('Mermaid render error:', err)
    errorMsg.value = err?.message || 'Error renderizando diagrama Mermaid'
  } finally {
    isRendering.value = false
  }
}

onMounted(() => {
  nextTick(() => {
    renderDiagram()
  })
})

watch(isDark, () => {
  renderDiagram()
})
</script>

<template>
  <div class="mermaid-container">
    <div v-if="errorMsg" class="mermaid-error">
      <div class="error-header">⚠️ Error en diagrama Mermaid</div>
      <pre>{{ errorMsg }}</pre>
    </div>
    <div
      v-else-if="svgContent"
      class="mermaid-svg-wrapper"
      v-html="svgContent"
    ></div>
    <div v-else class="mermaid-placeholder">
      <div class="spinner"></div>
      <span>Cargando diagrama...</span>
    </div>
  </div>
</template>

<style scoped>
.mermaid-container {
  margin: 1.75rem 0;
  padding: 1.5rem;
  background: var(--vp-c-bg-soft);
  border: 1px solid var(--vp-c-divider);
  border-radius: 12px;
  overflow-x: auto;
  box-shadow: 0 4px 16px rgba(0, 0, 0, 0.04);
  transition: border-color 0.25s, background-color 0.25s;
}

.mermaid-container:hover {
  border-color: var(--vp-c-brand-1);
}

.mermaid-svg-wrapper {
  display: flex;
  justify-content: center;
  align-items: center;
  width: 100%;
}

.mermaid-svg-wrapper :deep(svg) {
  max-width: 100%;
  height: auto;
  filter: drop-shadow(0 2px 4px rgba(0, 0, 0, 0.05));
}

/* Fix text clipping in Mermaid nodes */
.mermaid-svg-wrapper :deep(foreignObject) {
  overflow: visible !important;
}

.mermaid-svg-wrapper :deep(.node foreignObject) {
  overflow: visible !important;
}

.mermaid-svg-wrapper :deep(.node foreignObject div) {
  line-height: 1.4 !important;
  font-size: 13.5px !important;
  text-align: center !important;
  display: flex !important;
  flex-direction: column !important;
  justify-content: center !important;
  align-items: center !important;
}

.mermaid-svg-wrapper :deep(.nodeLabel),
.mermaid-svg-wrapper :deep(.label) {
  line-height: 1.4 !important;
  font-size: 13.5px !important;
  text-align: center !important;
}

.mermaid-placeholder {
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  gap: 0.75rem;
  min-height: 120px;
  color: var(--vp-c-text-2);
  font-size: 0.875rem;
}

.spinner {
  width: 24px;
  height: 24px;
  border: 2px solid var(--vp-c-divider);
  border-top-color: var(--vp-c-brand-1);
  border-radius: 50%;
  animation: spin 0.8s linear infinite;
}

@keyframes spin {
  to {
    transform: rotate(360deg);
  }
}

.mermaid-error {
  background: rgba(239, 68, 68, 0.1);
  border: 1px solid rgba(239, 68, 68, 0.3);
  color: #ef4444;
  padding: 1rem;
  border-radius: 8px;
  font-size: 0.85rem;
}

.error-header {
  font-weight: 600;
  margin-bottom: 0.5rem;
}

.mermaid-error pre {
  margin: 0;
  white-space: pre-wrap;
  word-break: break-all;
}
</style>
