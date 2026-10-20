<template>
  <div class="nature-background-root" aria-hidden="true">
    <!-- Subtle Agro-Architectural Dot Lattice with Smooth Radial Mask -->
    <div class="nature-dot-matrix"></div>

    <!-- Ambient Canopy Sunray & Chlorophyll Auroras -->
    <div class="nature-aurora-top"></div>
    <div class="nature-aurora-lateral"></div>

    <!-- Interactive Organic Pollen & Bio-Neural Canvas -->
    <canvas ref="canvasRef" class="nature-canvas"></canvas>
  </div>
</template>

<script setup lang="ts">
import { ref, onMounted, onUnmounted } from "vue";

const canvasRef = ref<HTMLCanvasElement | null>(null);
let animId: number | null = null;

onMounted(() => {
  const canvas = canvasRef.value;
  if (!canvas) return;
  const ctx = canvas.getContext("2d");
  if (!ctx) return;

  let width = (canvas.width = window.innerWidth);
  let height = (canvas.height = window.innerHeight);

  const resize = () => {
    if (!canvas) return;
    width = canvas.width = window.innerWidth;
    height = canvas.height = window.innerHeight;
  };

  window.addEventListener("resize", resize, { passive: true });

  // Natural particle system: pollen, chlorophyll, dew & amber seeds
  const isMobile = window.innerWidth < 768;
  const particleCount = isMobile ? 32 : 68;
  const maxDistance = isMobile ? 95 : 135;

  interface BioParticle {
    x: number;
    y: number;
    vx: number;
    vy: number;
    radius: number;
    color: string;
    alpha: number;
    baseAlpha: number;
    floatPhase: number;
    floatSpeed: number;
  }

  // Palette: Gold corn pollen, Chlorophyll green, Morning dew, Warm amber
  const colors = [
    "240, 193, 38",  // Golden Corn Pollen (#f0c126)
    "164, 216, 82",  // Fresh Chlorophyll / Sprout (#a4d852)
    "105, 143, 38",  // Deep Leaf Green (#698f26)
    "255, 255, 255", // Morning Dew / Starlight
    "245, 158, 11"   // Warm Harvest Amber
  ];

  const particles: BioParticle[] = [];

  for (let i = 0; i < particleCount; i++) {
    const color = colors[Math.floor(Math.random() * colors.length)];
    const baseAlpha = Math.random() * 0.40 + 0.12;
    particles.push({
      x: Math.random() * width,
      y: Math.random() * height,
      vx: (Math.random() - 0.5) * 0.32,
      vy: (Math.random() - 0.5) * 0.28 - 0.08, // Subtle upward floating breeze
      radius: Math.random() * 1.5 + 0.8,
      color,
      alpha: baseAlpha,
      baseAlpha,
      floatPhase: Math.random() * Math.PI * 2,
      floatSpeed: 0.015 + Math.random() * 0.02
    });
  }

  // Interactive mouse interaction (gentle air displacement)
  const mouse = { x: -1000, y: -1000, radius: 160 };

  const onMouseMove = (e: MouseEvent) => {
    mouse.x = e.clientX;
    mouse.y = e.clientY;
  };

  const onMouseLeave = () => {
    mouse.x = -1000;
    mouse.y = -1000;
  };

  window.addEventListener("mousemove", onMouseMove, { passive: true });
  window.addEventListener("mouseleave", onMouseLeave, { passive: true });

  let isPageVisible = true;
  const onVisibilityChange = () => {
    isPageVisible = !document.hidden;
  };
  document.addEventListener("visibilitychange", onVisibilityChange);

  let frameCount = 0;

  const render = () => {
    if (!ctx) return;

    if (!isPageVisible) {
      animId = requestAnimationFrame(render);
      return;
    }

    frameCount++;
    ctx.clearRect(0, 0, width, height);

    // Render faint bio-neural filament connections
    for (let i = 0; i < particles.length; i++) {
      for (let j = i + 1; j < particles.length; j++) {
        const dx = particles[i].x - particles[j].x;
        const dy = particles[i].y - particles[j].y;
        const dist = Math.sqrt(dx * dx + dy * dy);

        if (dist < maxDistance) {
          const strength = (1 - dist / maxDistance) * 0.22;
          ctx.beginPath();
          ctx.moveTo(particles[i].x, particles[i].y);
          ctx.lineTo(particles[j].x, particles[j].y);
          ctx.strokeStyle = `rgba(164, 216, 82, ${strength * Math.min(particles[i].alpha, particles[j].alpha)})`;
          ctx.lineWidth = 0.85;
          ctx.stroke();
        }
      }
    }

    // Render floating bio-particles
    for (let i = 0; i < particles.length; i++) {
      const p = particles[i];

      // Gentle floating sway
      p.floatPhase += p.floatSpeed;
      p.x += p.vx + Math.sin(p.floatPhase) * 0.22;
      p.y += p.vy + Math.cos(p.floatPhase) * 0.15;

      // Mouse gentle breeze repulsion
      const mdx = p.x - mouse.x;
      const mdy = p.y - mouse.y;
      const mdist = Math.sqrt(mdx * mdx + mdy * mdy);
      if (mdist < mouse.radius && mdist > 0) {
        const force = (1 - mdist / mouse.radius) * 1.8;
        p.x += (mdx / mdist) * force;
        p.y += (mdy / mdist) * force;
        p.alpha = Math.min(0.9, p.baseAlpha * 1.7);
      } else {
        p.alpha += (p.baseAlpha - p.alpha) * 0.05;
      }

      // Wrap-around screen bounds
      if (p.x < -20) p.x = width + 20;
      if (p.x > width + 20) p.x = -20;
      if (p.y < -20) p.y = height + 20;
      if (p.y > height + 20) p.y = -20;

      // Draw particle with gentle glow
      ctx.beginPath();
      ctx.arc(p.x, p.y, p.radius, 0, Math.PI * 2);
      ctx.fillStyle = `rgba(${p.color}, ${p.alpha})`;
      ctx.fill();

      // Soft ambient aura for larger particles
      if (p.radius > 1.4) {
        ctx.beginPath();
        ctx.arc(p.x, p.y, p.radius * 2.8, 0, Math.PI * 2);
        ctx.fillStyle = `rgba(${p.color}, ${p.alpha * 0.20})`;
        ctx.fill();
      }
    }

    animId = requestAnimationFrame(render);
  };

  render();

  onUnmounted(() => {
    if (animId) cancelAnimationFrame(animId);
    window.removeEventListener("resize", resize);
    window.removeEventListener("mousemove", onMouseMove);
    window.removeEventListener("mouseleave", onMouseLeave);
    document.removeEventListener("visibilitychange", onVisibilityChange);
  });
});
</script>

<style scoped>
.nature-background-root {
  position: fixed;
  inset: 0;
  width: 100%;
  height: 100%;
  pointer-events: none;
  z-index: 0;
  overflow: hidden;
  user-select: none;
  -webkit-user-select: none;
}

/* Sutil cuadrícula bio-celular con máscara radial suave */
.nature-dot-matrix {
  position: absolute;
  inset: 0;
  opacity: 0.18;
  background-image: radial-gradient(rgba(164, 216, 82, 0.35) 1px, transparent 1px);
  background-size: 34px 34px;
  mask-image: radial-gradient(ellipse 80% 65% at 50% 25%, #000 35%, transparent 100%);
  -webkit-mask-image: radial-gradient(ellipse 80% 65% at 50% 25%, #000 35%, transparent 100%);
}

.dark .nature-dot-matrix {
  opacity: 0.14;
  background-image: radial-gradient(rgba(164, 216, 82, 0.40) 1px, transparent 1px);
}

/* Halo superior difuso: luz de sol filtrada entre maizales */
.nature-aurora-top {
  position: absolute;
  top: -120px;
  left: 50%;
  transform: translateX(-50%);
  width: 950px;
  height: 520px;
  border-radius: 50%;
  background: radial-gradient(
    ellipse at center,
    rgba(164, 216, 82, 0.09) 0%,
    rgba(240, 193, 38, 0.05) 35%,
    rgba(105, 143, 38, 0.02) 60%,
    transparent 75%
  );
  filter: blur(55px);
  pointer-events: none;
}

.dark .nature-aurora-top {
  background: radial-gradient(
    ellipse at center,
    rgba(164, 216, 82, 0.11) 0%,
    rgba(240, 193, 38, 0.06) 40%,
    transparent 75%
  );
}

/* Halo lateral sutil: profundidad de follaje y clorofila */
.nature-aurora-lateral {
  position: absolute;
  bottom: 10%;
  right: -100px;
  width: 600px;
  height: 600px;
  border-radius: 50%;
  background: radial-gradient(
    circle,
    rgba(105, 143, 38, 0.05) 0%,
    rgba(164, 216, 82, 0.03) 45%,
    transparent 70%
  );
  filter: blur(60px);
  pointer-events: none;
}

.nature-canvas {
  display: block;
  width: 100%;
  height: 100%;
  opacity: 0.65;
}

.dark .nature-canvas {
  opacity: 0.75;
}
</style>
