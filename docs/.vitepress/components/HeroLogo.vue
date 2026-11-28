<template>
  <div class="hero-logo-wrapper">
    <div class="glass-container">
      <img src="/logo.svg" alt="DoctorMaíz Logo" class="hero-logo-img" />
    </div>
  </div>
</template>

<style scoped>
.hero-logo-wrapper {
  display: flex;
  justify-content: center;
  align-items: center;
  margin-bottom: 2rem;
}

.glass-container {
  position: relative;
  padding: 2rem;
  border-radius: 2rem;
  background: linear-gradient(
    135deg,
    rgba(105, 143, 38, 0.18) 0%,
    rgba(240, 193, 38, 0.10) 40%,
    rgba(37, 83, 30, 0.18) 100%
  );
  backdrop-filter: blur(16px) saturate(1.6);
  -webkit-backdrop-filter: blur(16px) saturate(1.6);
  border: 1.5px solid rgba(240, 193, 38, 0.28);
  box-shadow:
    0 8px 32px rgba(37, 83, 30, 0.22),
    0 2px 8px rgba(105, 143, 38, 0.18),
    inset 0 1px 0 rgba(240, 193, 38, 0.22),
    inset 0 -1px 0 rgba(37, 83, 30, 0.12);
  transition: box-shadow 0.3s ease, transform 0.3s ease;
}

.glass-container::before {
  content: "";
  position: absolute;
  inset: 0;
  border-radius: 2rem;
  background: radial-gradient(
    ellipse at 30% 20%,
    rgba(240, 193, 38, 0.14) 0%,
    transparent 60%
  );
  pointer-events: none;
}

.glass-container:hover {
  box-shadow:
    0 12px 48px rgba(37, 83, 30, 0.32),
    0 4px 16px rgba(105, 143, 38, 0.24),
    inset 0 1px 0 rgba(240, 193, 38, 0.32),
    inset 0 -1px 0 rgba(37, 83, 30, 0.16);
  transform: translateY(-3px);
}

.hero-logo-img {
  display: block;
  width: 180px;
  height: 180px;
  object-fit: contain;
  filter: drop-shadow(0 4px 12px rgba(37, 83, 30, 0.4));
}

@media (max-width: 768px) {
  .hero-logo-img {
    width: 130px;
    height: 130px;
  }
  .glass-container {
    padding: 1.5rem;
  }
}
</style>
