<template>
  <div class="nature-background-root" aria-hidden="true">
    <div class="nature-dot-matrix"></div>
    <div class="nature-aurora-top"></div>
    <div class="nature-aurora-lateral"></div>
    <canvas ref="canvasRef" class="nature-canvas"></canvas>
  </div>
</template>

<script setup lang="ts">
import { ref, onMounted, onUnmounted } from "vue";

interface BioParticle {
  x: number;
  y: number;
  vx: number;
  vy: number;
  radius: number;
  color: string;
  alpha: number;
  baseAlpha: number;
  pulsePhase: number;
  pulseSpeed: number;
  floatPhase: number;
}

interface PollenDust {
  x: number;
  y: number;
  vx: number;
  vy: number;
  radius: number;
  alpha: number;
  life: number;
  maxLife: number;
}

const PARTICLE_COLORS = [
  "240, 193, 38",
  "164, 216, 82",
  "105, 143, 38",
  "245, 158, 11",
  "255, 255, 255",
];

const canvasRef = ref<HTMLCanvasElement | null>(null);
let animId: number | null = null;
let cleanup: (() => void) | null = null;

/**
 * Starts the particle animation on the canvas unless the user prefers reduced motion.
 *
 * @param canvas Canvas element that receives the drawing.
 * @returns Function that stops the animation and removes every listener.
 */
function startAnimation(canvas: HTMLCanvasElement): (() => void) | null {
  const reducedMotion = window.matchMedia("(prefers-reduced-motion: reduce)");
  if (reducedMotion.matches) return null;

  const ctx = canvas.getContext("2d");
  if (!ctx) return null;

  let width = (canvas.width = window.innerWidth);
  let height = (canvas.height = window.innerHeight);

  const resize = () => {
    width = canvas.width = window.innerWidth;
    height = canvas.height = window.innerHeight;
  };
  window.addEventListener("resize", resize, { passive: true });

  const isMobile = window.innerWidth < 768;
  const particleCount = isMobile ? 36 : 72;
  const maxConnectionDistance = isMobile ? 90 : 130;

  const particles: BioParticle[] = [];
  for (let i = 0; i < particleCount; i++) {
    const baseAlpha = Math.random() * 0.42 + 0.15;
    particles.push({
      x: Math.random() * width,
      y: Math.random() * height,
      vx: (Math.random() - 0.5) * 0.25,
      vy: (Math.random() - 0.5) * 0.22 - 0.06,
      radius: Math.random() * 1.5 + 0.8,
      color: PARTICLE_COLORS[Math.floor(Math.random() * PARTICLE_COLORS.length)],
      alpha: baseAlpha,
      baseAlpha,
      pulsePhase: Math.random() * Math.PI * 2,
      pulseSpeed: 0.02 + Math.random() * 0.03,
      floatPhase: Math.random() * Math.PI * 2,
    });
  }

  const pollenDustTrail: PollenDust[] = [];
  const maxDust = 28;
  const mouse = { x: -1000, y: -1000, prevX: -1000, prevY: -1000, vx: 0, vy: 0, radius: 175 };

  const onMouseMove = (e: MouseEvent) => {
    if (mouse.prevX !== -1000) {
      mouse.vx = (e.clientX - mouse.prevX) * 0.25;
      mouse.vy = (e.clientY - mouse.prevY) * 0.25;
    }
    mouse.prevX = mouse.x;
    mouse.prevY = mouse.y;
    mouse.x = e.clientX;
    mouse.y = e.clientY;

    if (Math.random() < 0.35 && pollenDustTrail.length < maxDust) {
      pollenDustTrail.push({
        x: mouse.x + (Math.random() - 0.5) * 16,
        y: mouse.y + (Math.random() - 0.5) * 16,
        vx: (Math.random() - 0.5) * 0.45 + mouse.vx * 0.1,
        vy: (Math.random() - 0.5) * 0.45 + mouse.vy * 0.1 - 0.1,
        radius: Math.random() * 1.3 + 0.6,
        alpha: 0.6,
        life: 0,
        maxLife: 35 + Math.random() * 25,
      });
    }
  };

  const onMouseLeave = () => {
    mouse.x = -1000;
    mouse.y = -1000;
    mouse.vx = 0;
    mouse.vy = 0;
  };

  let isPageVisible = true;
  const onVisibilityChange = () => {
    isPageVisible = !document.hidden;
  };

  let time = 0;
  let stopped = false;

  const render = () => {
    if (stopped) return;
    if (!isPageVisible) {
      animId = requestAnimationFrame(render);
      return;
    }

    time++;
    ctx.clearRect(0, 0, width, height);
    mouse.vx *= 0.92;
    mouse.vy *= 0.92;

    for (let i = 0; i < particles.length; i++) {
      for (let j = i + 1; j < particles.length; j++) {
        const dx = particles[i].x - particles[j].x;
        const dy = particles[i].y - particles[j].y;
        const dist = Math.sqrt(dx * dx + dy * dy);
        if (dist < maxConnectionDistance) {
          const strength = (1 - dist / maxConnectionDistance) * 0.22;
          ctx.beginPath();
          ctx.moveTo(particles[i].x, particles[i].y);
          ctx.lineTo(particles[j].x, particles[j].y);
          ctx.strokeStyle = `rgba(164, 216, 82, ${strength * Math.min(particles[i].alpha, particles[j].alpha)})`;
          ctx.lineWidth = 0.85;
          ctx.stroke();
        }
      }
    }

    for (let i = 0; i < particles.length; i++) {
      const p = particles[i];
      p.pulsePhase += p.pulseSpeed;
      const pulseFactor = Math.sin(p.pulsePhase);
      p.alpha = p.baseAlpha * (0.65 + pulseFactor * 0.35);

      const windAngle =
        Math.sin(p.x * 0.0018 + time * 0.007) * Math.cos(p.y * 0.0018 + time * 0.005) * 1.4 - 0.2;
      const windSpeed = 0.28 + Math.sin(time * 0.009 + i) * 0.12;
      const wx = Math.cos(windAngle) * windSpeed;
      const wy = Math.sin(windAngle) * windSpeed - 0.08;

      p.floatPhase += 0.02;
      p.x += p.vx + wx + Math.sin(p.floatPhase) * 0.18;
      p.y += p.vy + wy + Math.cos(p.floatPhase) * 0.14;

      const mdx = p.x - mouse.x;
      const mdy = p.y - mouse.y;
      const mdist = Math.sqrt(mdx * mdx + mdy * mdy);
      if (mdist < mouse.radius && mdist > 0) {
        const force = (1 - mdist / mouse.radius) * 1.9;
        p.x += (mdx / mdist) * force + mouse.vx * 0.3;
        p.y += (mdy / mdist) * force + mouse.vy * 0.3;
        p.alpha = Math.min(0.95, p.alpha * 1.8);
      }

      if (p.x < -25) p.x = width + 25;
      if (p.x > width + 25) p.x = -25;
      if (p.y < -25) p.y = height + 25;
      if (p.y > height + 25) p.y = -25;

      ctx.beginPath();
      ctx.arc(p.x, p.y, p.radius, 0, Math.PI * 2);
      ctx.fillStyle = `rgba(${p.color}, ${p.alpha})`;
      ctx.fill();

      const haloRadius = p.radius * (2.2 + pulseFactor * 0.8);
      ctx.beginPath();
      ctx.arc(p.x, p.y, haloRadius, 0, Math.PI * 2);
      ctx.fillStyle = `rgba(${p.color}, ${p.alpha * 0.22})`;
      ctx.fill();
    }

    for (let k = pollenDustTrail.length - 1; k >= 0; k--) {
      const d = pollenDustTrail[k];
      d.life++;
      d.x += d.vx;
      d.y += d.vy;
      const progress = d.life / d.maxLife;
      ctx.beginPath();
      ctx.arc(d.x, d.y, d.radius * (1 - progress * 0.4), 0, Math.PI * 2);
      ctx.fillStyle = `rgba(240, 193, 38, ${(1 - progress) * d.alpha})`;
      ctx.fill();
      if (d.life >= d.maxLife) pollenDustTrail.splice(k, 1);
    }

    animId = requestAnimationFrame(render);
  };

  const stop = () => {
    stopped = true;
    if (animId) cancelAnimationFrame(animId);
    ctx.clearRect(0, 0, width, height);
    window.removeEventListener("resize", resize);
    window.removeEventListener("mousemove", onMouseMove);
    window.removeEventListener("mouseleave", onMouseLeave);
    document.removeEventListener("visibilitychange", onVisibilityChange);
    reducedMotion.removeEventListener("change", onMotionPreferenceChange);
  };

  const onMotionPreferenceChange = (e: MediaQueryListEvent) => {
    if (e.matches) stop();
  };

  window.addEventListener("mousemove", onMouseMove, { passive: true });
  window.addEventListener("mouseleave", onMouseLeave, { passive: true });
  document.addEventListener("visibilitychange", onVisibilityChange);
  reducedMotion.addEventListener("change", onMotionPreferenceChange);

  render();
  return stop;
}

onMounted(() => {
  if (canvasRef.value) cleanup = startAnimation(canvasRef.value);
});

onUnmounted(() => {
  if (cleanup) cleanup();
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
  opacity: 0.70;
}

.dark .nature-canvas {
  opacity: 0.85;
}
</style>
