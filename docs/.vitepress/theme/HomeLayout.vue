<template>
  <NatureBackground />
  <Layout>
    <template #home-hero-image>
      <HeroLogo />
    </template>
  </Layout>
</template>

<script setup lang="ts">
import DefaultTheme from "vitepress/theme";
import HeroLogo from "../components/HeroLogo.vue";
import NatureBackground from "../components/NatureBackground.vue";

const { Layout } = DefaultTheme;
</script>
