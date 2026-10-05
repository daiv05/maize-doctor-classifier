<template>
  <div class="phone-gallery" :style="{ gap: gap || '1.5rem' }">
    <slot></slot>
  </div>
</template>

<script setup lang="ts">
defineProps<{
  gap?: string;
}>();
</script>

<style scoped>
.phone-gallery {
  display: flex;
  flex-wrap: wrap;
  justify-content: center;
  align-items: flex-start;
  margin: 2rem 0;
  width: 100%;
}

.phone-gallery :deep(.phone-mockup-wrapper) {
  margin: 0;
  flex: 1 1 260px;
  max-width: 290px;
}
</style>
