<script setup lang="ts">
import { ref, onMounted, onUnmounted, watch, nextTick } from 'vue'
import { useData } from 'vitepress'

const props = defineProps<{
  code: string
}>()

const { isDark } = useData()
const svgContent = ref('')
const errorMsg = ref('')
const isRendering = ref(true)

const isFullscreen = ref(false)
const zoomLevel = ref(1)

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
        nodeSpacing: 45,
        rankSpacing: 45,
        curve: 'basis',
        wrappingWidth: 320,
        minNodeWidth: 180
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

function openFullscreen() {
  isFullscreen.value = true
  zoomLevel.value = 1
  if (typeof document !== 'undefined') {
    document.body.style.overflow = 'hidden'
    window.addEventListener('keydown', handleKeydown)
  }
}

function closeFullscreen() {
  isFullscreen.value = false
  zoomLevel.value = 1
  if (typeof document !== 'undefined') {
    document.body.style.overflow = ''
    window.removeEventListener('keydown', handleKeydown)
  }
}

function toggleFullscreen() {
  if (isFullscreen.value) {
    closeFullscreen()
  } else {
    openFullscreen()
  }
}

function zoomIn() {
  if (zoomLevel.value < 2.5) {
    zoomLevel.value = +(zoomLevel.value + 0.2).toFixed(2)
  }
}

function zoomOut() {
  if (zoomLevel.value > 0.5) {
    zoomLevel.value = +(zoomLevel.value - 0.2).toFixed(2)
  }
}

function resetZoom() {
  zoomLevel.value = 1
}

function handleKeydown(e: KeyboardEvent) {
  if (e.key === 'Escape') {
    closeFullscreen()
  } else if (e.key === '+' || e.key === '=') {
    e.preventDefault()
    zoomIn()
  } else if (e.key === '-' || e.key === '_') {
    e.preventDefault()
    zoomOut()
  } else if (e.key === '0') {
    e.preventDefault()
    resetZoom()
  }
}

function handleWheel(e: WheelEvent) {
  if (e.ctrlKey || e.metaKey) {
    e.preventDefault()
    if (e.deltaY < 0) {
      zoomIn()
    } else {
      zoomOut()
    }
  }
}

onMounted(() => {
  nextTick(() => {
    renderDiagram()
  })
})

onUnmounted(() => {
  if (typeof document !== 'undefined') {
    document.body.style.overflow = ''
    window.removeEventListener('keydown', handleKeydown)
  }
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

    <template v-else-if="svgContent">
      <div class="mermaid-actions">
        <button
          type="button"
          class="action-btn"
          @click="openFullscreen"
          title="Ver en pantalla completa (Doble clic en el diagrama)"
          aria-label="Ver en pantalla completa"
        >
          <svg viewBox="0 0 24 24" width="16" height="16" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
            <path d="M8 3H5a2 2 0 0 0-2 2v3m18 0V5a2 2 0 0 0-2-2h-3m0 18h3a2 2 0 0 0 2-2v-3M3 16v3a2 2 0 0 0 2 2h3" />
          </svg>
        </button>
      </div>

      <div
        class="mermaid-svg-wrapper"
        v-html="svgContent"
        @dblclick="openFullscreen"
        title="Doble clic para ver en pantalla completa"
      ></div>
    </template>

    <div v-else class="mermaid-placeholder">
      <div class="spinner"></div>
      <span>Cargando diagrama...</span>
    </div>

    <!-- Modal de Pantalla Completa -->
    <Teleport to="body">
      <Transition name="fade">
        <div
          v-if="isFullscreen"
          class="mermaid-fullscreen-modal"
          @click.self="closeFullscreen"
        >
          <!-- Barra superior de controles -->
          <div class="modal-header">
            <div class="modal-title">
              <svg viewBox="0 0 24 24" width="18" height="18" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                <rect x="3" y="3" width="18" height="18" rx="2" ry="2"/>
                <line x1="3" y1="9" x2="21" y2="9"/>
                <line x1="9" y1="21" x2="9" y2="9"/>
              </svg>
              <span>Vista Extendida de Diagrama</span>
            </div>

            <div class="modal-controls">
              <div class="zoom-group">
                <button
                  type="button"
                  class="ctrl-btn"
                  @click="zoomOut"
                  :disabled="zoomLevel <= 0.5"
                  title="Reducir (-) o Ctrl + Rueda"
                >
                  <svg viewBox="0 0 24 24" width="15" height="15" fill="none" stroke="currentColor" stroke-width="2">
                    <line x1="5" y1="12" x2="19" y2="12"/>
                  </svg>
                </button>

                <button
                  type="button"
                  class="zoom-badge"
                  @click="resetZoom"
                  title="Clic para reiniciar zoom al 100%"
                >
                  {{ Math.round(zoomLevel * 100) }}%
                </button>

                <button
                  type="button"
                  class="ctrl-btn"
                  @click="zoomIn"
                  :disabled="zoomLevel >= 2.5"
                  title="Aumentar (+) o Ctrl + Rueda"
                >
                  <svg viewBox="0 0 24 24" width="15" height="15" fill="none" stroke="currentColor" stroke-width="2">
                    <line x1="12" y1="5" x2="12" y2="19"/>
                    <line x1="5" y1="12" x2="19" y2="12"/>
                  </svg>
                </button>

                <button
                  type="button"
                  class="ctrl-btn text-ctrl"
                  @click="resetZoom"
                  title="Restablecer tamaño original (0)"
                >
                  Restablecer
                </button>
              </div>

              <div class="separator"></div>

              <button
                type="button"
                class="ctrl-btn close-btn"
                @click="closeFullscreen"
                title="Cerrar vista (Esc)"
              >
                <svg viewBox="0 0 24 24" width="16" height="16" fill="none" stroke="currentColor" stroke-width="2">
                  <line x1="18" y1="6" x2="6" y2="18"/>
                  <line x1="6" y1="6" x2="18" y2="18"/>
                </svg>
                <span class="shortcut-tag">Esc</span>
              </button>
            </div>
          </div>

          <!-- Lienzo con scroll, fondo cuadriculado y tarjeta contenedora -->
          <div class="modal-viewport" @wheel="handleWheel">
            <div
              class="modal-canvas-wrapper"
              :style="{ transform: `scale(${zoomLevel})` }"
            >
              <div class="modal-card">
                <div class="modal-svg-content" v-html="svgContent"></div>
              </div>
            </div>
          </div>
        </div>
      </Transition>
    </Teleport>
  </div>
</template>

<style scoped>
.mermaid-container {
  position: relative;
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

.mermaid-actions {
  position: absolute;
  top: 0.75rem;
  right: 0.75rem;
  z-index: 5;
  display: flex;
  gap: 0.5rem;
  opacity: 0.75;
  transition: opacity 0.2s ease;
}

.mermaid-container:hover .mermaid-actions {
  opacity: 1;
}

.action-btn {
  display: inline-flex;
  align-items: center;
  justify-content: center;
  width: 32px;
  height: 32px;
  padding: 0;
  color: var(--vp-c-text-2);
  background: var(--vp-c-bg);
  border: 1px solid var(--vp-c-divider);
  border-radius: 6px;
  cursor: pointer;
  transition: all 0.2s ease;
  box-shadow: 0 1px 3px rgba(0, 0, 0, 0.08);
}

.action-btn:hover {
  color: var(--vp-c-brand-1);
  border-color: var(--vp-c-brand-1);
  background: var(--vp-c-bg-elv);
  transform: scale(1.06);
}

.mermaid-svg-wrapper {
  display: flex;
  justify-content: center;
  align-items: center;
  width: 100%;
  cursor: zoom-in;
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

/* ==============================================================================
   Modal de Pantalla Completa - Estilo Tablero de Diseño
   ============================================================================== */
.mermaid-fullscreen-modal {
  position: fixed;
  inset: 0;
  z-index: 99999;
  display: flex;
  flex-direction: column;
  background: var(--vp-c-bg);
}

.modal-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 0.85rem 1.5rem;
  background: var(--vp-c-bg-soft);
  border-bottom: 1px solid var(--vp-c-divider);
  z-index: 10;
  box-shadow: 0 2px 10px rgba(0, 0, 0, 0.08);
}

.modal-title {
  display: flex;
  align-items: center;
  gap: 0.6rem;
  font-size: 0.95rem;
  font-weight: 600;
  color: var(--vp-c-text-1);
}

.modal-controls {
  display: flex;
  align-items: center;
  gap: 0.85rem;
}

.zoom-group {
  display: flex;
  align-items: center;
  background: var(--vp-c-bg);
  border: 1px solid var(--vp-c-divider);
  border-radius: 8px;
  padding: 2px;
  gap: 2px;
}

.ctrl-btn {
  display: inline-flex;
  align-items: center;
  justify-content: center;
  padding: 0.35rem 0.55rem;
  background: transparent;
  border: none;
  border-radius: 6px;
  color: var(--vp-c-text-2);
  cursor: pointer;
  transition: all 0.15s ease;
}

.ctrl-btn:hover:not(:disabled) {
  background: var(--vp-c-bg-soft);
  color: var(--vp-c-brand-1);
}

.ctrl-btn:disabled {
  opacity: 0.35;
  cursor: not-allowed;
}

.text-ctrl {
  font-size: 0.75rem;
  font-weight: 500;
  padding: 0.35rem 0.65rem;
}

.zoom-badge {
  min-width: 52px;
  padding: 0.35rem 0.4rem;
  background: transparent;
  border: none;
  font-size: 0.8rem;
  font-weight: 600;
  font-variant-numeric: tabular-nums;
  color: var(--vp-c-text-1);
  cursor: pointer;
  text-align: center;
}

.zoom-badge:hover {
  color: var(--vp-c-brand-1);
}

.separator {
  width: 1px;
  height: 24px;
  background: var(--vp-c-divider);
}

.close-btn {
  display: inline-flex;
  align-items: center;
  gap: 0.4rem;
  padding: 0.35rem 0.75rem;
  background: var(--vp-c-bg);
  border: 1px solid var(--vp-c-divider);
  border-radius: 8px;
  font-weight: 500;
  color: var(--vp-c-text-1);
}

.close-btn:hover {
  background: rgba(239, 68, 68, 0.1);
  border-color: #ef4444;
  color: #ef4444;
}

.shortcut-tag {
  font-size: 0.7rem;
  padding: 0.1rem 0.35rem;
  border-radius: 4px;
  background: var(--vp-c-bg-soft);
  border: 1px solid var(--vp-c-divider);
}

.modal-viewport {
  flex: 1;
  overflow: auto;
  display: flex;
  padding: 2.5rem;
  background-color: var(--vp-c-bg);
  background-image: radial-gradient(var(--vp-c-divider) 1.25px, transparent 1.25px);
  background-size: 24px 24px;
  cursor: grab;
}

.modal-viewport:active {
  cursor: grabbing;
}

.modal-canvas-wrapper {
  margin: auto;
  transform-origin: center center;
  transition: transform 0.2s cubic-bezier(0.16, 1, 0.3, 1);
  display: flex;
  align-items: center;
  justify-content: center;
}

.modal-card {
  background: var(--vp-c-bg-soft);
  border: 1px solid var(--vp-c-divider);
  border-radius: 16px;
  padding: 2.5rem 3rem;
  box-shadow: 0 16px 40px rgba(0, 0, 0, 0.2);
  display: flex;
  justify-content: center;
  align-items: center;
  min-width: min(880px, 86vw);
}

.modal-svg-content {
  display: flex;
  justify-content: center;
  align-items: center;
  width: 100%;
}

.modal-svg-content :deep(svg) {
  width: 100% !important;
  min-width: min(820px, 82vw) !important;
  max-width: min(1200px, 88vw) !important;
  height: auto !important;
  filter: drop-shadow(0 4px 12px rgba(0, 0, 0, 0.08));
}

.modal-svg-content :deep(foreignObject) {
  overflow: visible !important;
}

.modal-svg-content :deep(.node foreignObject) {
  overflow: visible !important;
}

.modal-svg-content :deep(.node foreignObject div) {
  line-height: 1.45 !important;
  font-size: 14px !important;
  text-align: center !important;
  display: flex !important;
  flex-direction: column !important;
  justify-content: center !important;
  align-items: center !important;
}

.modal-svg-content :deep(.nodeLabel),
.modal-svg-content :deep(.label) {
  line-height: 1.45 !important;
  font-size: 14px !important;
  text-align: center !important;
}

/* Animaciones */
.fade-enter-active,
.fade-leave-active {
  transition: opacity 0.2s ease, transform 0.2s ease;
}

.fade-enter-from,
.fade-leave-to {
  opacity: 0;
}
</style>
