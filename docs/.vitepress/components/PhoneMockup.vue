<template>
  <div class="phone-mockup-wrapper" :class="{ 'align-center': center }">
    <div class="phone-device" :style="{ maxWidth: maxWidth || '300px' }">
      <!-- Phone Bezel -->
      <div class="phone-bezel">
        <!-- Top Speaker / Dynamic Island bar -->
        <div class="phone-island">
          <div class="island-camera"></div>
          <div class="island-speaker"></div>
        </div>
        <!-- Screen Content -->
        <div class="phone-screen">
          <img :src="src" :alt="alt || caption || 'Captura de pantalla'" loading="lazy" />
        </div>
      </div>
    </div>
    <!-- Caption -->
    <div v-if="caption" class="phone-caption">
      <span class="caption-title">{{ caption }}</span>
      <span v-if="subtitle" class="caption-sub">{{ subtitle }}</span>
    </div>
  </div>
</template>

<script setup lang="ts">
defineProps<{
  src: string;
  caption?: string;
  subtitle?: string;
  maxWidth?: string;
  center?: boolean;
  alt?: string;
}>();
</script>

<style scoped>
.phone-mockup-wrapper {
  display: flex;
  flex-direction: column;
  align-items: center;
  margin: 1.5rem auto;
}

.phone-mockup-wrapper.align-center {
  margin-left: auto;
  margin-right: auto;
}

.phone-device {
  width: 100%;
  position: relative;
  border-radius: 36px;
  padding: 10px;
  background: linear-gradient(145deg, #2b3137, #161b22);
  box-shadow: 0 20px 40px -15px rgba(0, 0, 0, 0.45), 0 0 0 1px rgba(255, 255, 255, 0.08);
}

:root:not(.dark) .phone-device {
  background: linear-gradient(145deg, #e1e7ed, #cfd7df);
  box-shadow: 0 18px 36px -12px rgba(31, 38, 135, 0.22), 0 0 0 1px rgba(0, 0, 0, 0.08);
}

.phone-bezel {
  position: relative;
  background: #000;
  border-radius: 28px;
  overflow: hidden;
  box-shadow: inset 0 0 4px rgba(0, 0, 0, 0.8);
}

.phone-island {
  position: absolute;
  top: 8px;
  left: 50%;
  transform: translateX(-50%);
  width: 64px;
  height: 12px;
  background: #0d1117;
  border-radius: 12px;
  z-index: 10;
  display: flex;
  align-items: center;
  justify-content: center;
  gap: 6px;
}

.island-camera {
  width: 5px;
  height: 5px;
  background: #21262d;
  border-radius: 50%;
}

.island-speaker {
  width: 24px;
  height: 3px;
  background: #30363d;
  border-radius: 2px;
}

.phone-screen {
  width: 100%;
  aspect-ratio: 9 / 19.5;
  background: #000;
  overflow: hidden;
  display: flex;
  align-items: center;
  justify-content: center;
}

.phone-screen img {
  width: 100%;
  height: 100%;
  object-fit: cover;
  display: block;
  user-select: none;
}

.phone-caption {
  margin-top: 0.75rem;
  text-align: center;
  display: flex;
  flex-direction: column;
  gap: 2px;
}

.caption-title {
  font-weight: 600;
  font-size: 0.88rem;
  color: var(--vp-c-text-1);
}

.caption-sub {
  font-size: 0.78rem;
  color: var(--vp-c-text-2);
}
</style>
