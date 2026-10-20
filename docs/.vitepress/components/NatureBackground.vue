<template>
  <div class="nature-background-root" aria-hidden="true">
    <!-- Subtle Agro-Architectural Dot Lattice with Smooth Radial Mask -->
    <div class="nature-dot-matrix"></div>

    <!-- Ambient Canopy Sunray & Chlorophyll Auroras -->
    <div class="nature-aurora-top"></div>
    <div class="nature-aurora-lateral"></div>

    <!-- Interactive Organic Pollen, Fireflies & Computer Vision Canvas -->
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

  const isMobile = window.innerWidth < 768;
  const particleCount = isMobile ? 36 : 72;
  const maxConnectionDistance = isMobile ? 90 : 130;

  // ── 1. Partículas Biológicas & Luciérnagas ──
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

  const colors = [
    "240, 193, 38",  // Polen Dorado de Maíz (#f0c126)
    "164, 216, 82",  // Clorofila Fresca (#a4d852)
    "105, 143, 38",  // Verde Follaje (#698f26)
    "245, 158, 11",  // Ámbar de Cosecha (#f59e0b)
    "255, 255, 255"  // Rocío / Luz Pura
  ];

  const particles: BioParticle[] = [];

  for (let i = 0; i < particleCount; i++) {
    const color = colors[Math.floor(Math.random() * colors.length)];
    const baseAlpha = Math.random() * 0.42 + 0.15;
    particles.push({
      x: Math.random() * width,
      y: Math.random() * height,
      vx: (Math.random() - 0.5) * 0.25,
      vy: (Math.random() - 0.5) * 0.22 - 0.06,
      radius: Math.random() * 1.5 + 0.8,
      color,
      alpha: baseAlpha,
      baseAlpha,
      pulsePhase: Math.random() * Math.PI * 2,
      pulseSpeed: 0.02 + Math.random() * 0.03,
      floatPhase: Math.random() * Math.PI * 2
    });
  }

  // ── 2. Estela Efímera de Polen al Mover el Mouse ──
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

  const pollenDustTrail: PollenDust[] = [];
  const maxDust = 28;

  // ── 3. Micro-Retículas de Inspección Edge AI (Computer Vision Focus) ──
  interface VisionTarget {
    particleIndex: number;
    size: number;
    alpha: number;
    label: string;
    stage: "fade-in" | "hold" | "fade-out";
    timer: number;
  }

  const visionLabels = [
    "CNN: 98.4%",
    "LEAF: SANA",
    "EDGE AI: 12ms",
    "OOD: 0.01",
    "TFLITE: OK",
    "SCAN: FITO"
  ];

  let currentTarget: VisionTarget | null = null;
  let targetSpawnTimer = 0;

  // ── 4. Interacción del Mouse (Campo de Viento) ──
  const mouse = {
    x: -1000,
    y: -1000,
    prevX: -1000,
    prevY: -1000,
    vx: 0,
    vy: 0,
    radius: 175
  };

  const onMouseMove = (e: MouseEvent) => {
    if (mouse.prevX !== -1000) {
      mouse.vx = (e.clientX - mouse.prevX) * 0.25;
      mouse.vy = (e.clientY - mouse.prevY) * 0.25;
    }
    mouse.prevX = mouse.x;
    mouse.prevY = mouse.y;
    mouse.x = e.clientX;
    mouse.y = e.clientY;

    // Emite motas de polen en la estela de brisa
    if (Math.random() < 0.35 && pollenDustTrail.length < maxDust) {
      pollenDustTrail.push({
        x: mouse.x + (Math.random() - 0.5) * 16,
        y: mouse.y + (Math.random() - 0.5) * 16,
        vx: (Math.random() - 0.5) * 0.45 + mouse.vx * 0.1,
        vy: (Math.random() - 0.5) * 0.45 + mouse.vy * 0.1 - 0.1,
        radius: Math.random() * 1.3 + 0.6,
        alpha: 0.6,
        life: 0,
        maxLife: 35 + Math.random() * 25
      });
    }
  };

  const onMouseLeave = () => {
    mouse.x = -1000;
    mouse.y = -1000;
    mouse.vx = 0;
    mouse.vy = 0;
  };

  window.addEventListener("mousemove", onMouseMove, { passive: true });
  window.addEventListener("mouseleave", onMouseLeave, { passive: true });

  let isPageVisible = true;
  const onVisibilityChange = () => {
    isPageVisible = !document.hidden;
  };
  document.addEventListener("visibilitychange", onVisibilityChange);

  let time = 0;

  // ── Función de Renderizado Principal ──
  const render = () => {
    if (!ctx) return;

    if (!isPageVisible) {
      animId = requestAnimationFrame(render);
      return;
    }

    time++;
    ctx.clearRect(0, 0, width, height);

    // Amortiguación de velocidad del mouse
    mouse.vx *= 0.92;
    mouse.vy *= 0.92;

    // ── A. Conexiones Filamentosas (Micelio & Red Bio-Neuronal) ──
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

    // ── B. Campo de Viento & Movimiento de Luciérnagas ──
    for (let i = 0; i < particles.length; i++) {
      const p = particles[i];

      // Pulso bioluminiscente independiente (respiración)
      p.pulsePhase += p.pulseSpeed;
      const pulseFactor = Math.sin(p.pulsePhase);
      p.alpha = p.baseAlpha * (0.65 + pulseFactor * 0.35);

      // Campo de flujo de viento ondulatorio
      const windAngle =
        Math.sin(p.x * 0.0018 + time * 0.007) * Math.cos(p.y * 0.0018 + time * 0.005) * 1.4 - 0.2;
      const windSpeed = 0.28 + Math.sin(time * 0.009 + i) * 0.12;
      const wx = Math.cos(windAngle) * windSpeed;
      const wy = Math.sin(windAngle) * windSpeed - 0.08;

      p.floatPhase += 0.02;
      p.x += p.vx + wx + Math.sin(p.floatPhase) * 0.18;
      p.y += p.vy + wy + Math.cos(p.floatPhase) * 0.14;

      // Perturbación del cursor (Brisa / Empuje de aire)
      const mdx = p.x - mouse.x;
      const mdy = p.y - mouse.y;
      const mdist = Math.sqrt(mdx * mdx + mdy * mdy);

      if (mdist < mouse.radius && mdist > 0) {
        const force = (1 - mdist / mouse.radius) * 1.9;
        p.x += (mdx / mdist) * force + mouse.vx * 0.3;
        p.y += (mdy / mdist) * force + mouse.vy * 0.3;
        p.alpha = Math.min(0.95, p.alpha * 1.8);
      }

      // Envoltura continua de bordes
      if (p.x < -25) p.x = width + 25;
      if (p.x > width + 25) p.x = -25;
      if (p.y < -25) p.y = height + 25;
      if (p.y > height + 25) p.y = -25;

      // Dibujo de luciérnaga / espora
      ctx.beginPath();
      ctx.arc(p.x, p.y, p.radius, 0, Math.PI * 2);
      ctx.fillStyle = `rgba(${p.color}, ${p.alpha})`;
      ctx.fill();

      // Halo bioluminiscente suave (respiración expansiva)
      const haloRadius = p.radius * (2.2 + pulseFactor * 0.8);
      ctx.beginPath();
      ctx.arc(p.x, p.y, haloRadius, 0, Math.PI * 2);
      ctx.fillStyle = `rgba(${p.color}, ${p.alpha * 0.22})`;
      ctx.fill();
    }

    // ── C. Render de la Estela Efímera de Polen ──
    for (let k = pollenDustTrail.length - 1; k >= 0; k--) {
      const d = pollenDustTrail[k];
      d.life++;
      d.x += d.vx;
      d.y += d.vy;
      const progress = d.life / d.maxLife;
      const currentAlpha = (1 - progress) * d.alpha;

      ctx.beginPath();
      ctx.arc(d.x, d.y, d.radius * (1 - progress * 0.4), 0, Math.PI * 2);
      ctx.fillStyle = `rgba(240, 193, 38, ${currentAlpha})`;
      ctx.fill();

      if (d.life >= d.maxLife) {
        pollenDustTrail.splice(k, 1);
      }
    }

    // ── D. Micro-Retícula de Detección Edge AI (Computer Vision Target) ──
    targetSpawnTimer++;
    if (!currentTarget && targetSpawnTimer > 360 && particles.length > 0) {
      targetSpawnTimer = 0;
      const chosenIndex = Math.floor(Math.random() * particles.length);
      const label = visionLabels[Math.floor(Math.random() * visionLabels.length)];
      currentTarget = {
        particleIndex: chosenIndex,
        size: 26,
        alpha: 0,
        label,
        stage: "fade-in",
        timer: 0
      };
    }

    if (currentTarget) {
      currentTarget.timer++;
      const targetParticle = particles[currentTarget.particleIndex];

      if (currentTarget.stage === "fade-in") {
        currentTarget.alpha += 0.035;
        if (currentTarget.alpha >= 0.55) {
          currentTarget.alpha = 0.55;
          currentTarget.stage = "hold";
        }
      } else if (currentTarget.stage === "hold") {
        if (currentTarget.timer > 160) {
          currentTarget.stage = "fade-out";
        }
      } else if (currentTarget.stage === "fade-out") {
        currentTarget.alpha -= 0.025;
        if (currentTarget.alpha <= 0) {
          currentTarget = null;
        }
      }

      if (currentTarget && targetParticle) {
        const tx = targetParticle.x;
        const ty = targetParticle.y;
        const s = currentTarget.size / 2;
        const arm = 6;
        const a = currentTarget.alpha;

        ctx.strokeStyle = `rgba(164, 216, 82, ${a})`;
        ctx.lineWidth = 1.1;

        // Esquinas de retícula de bounding box [ · ]
        // Superior izquierda
        ctx.beginPath();
        ctx.moveTo(tx - s, ty - s + arm);
        ctx.lineTo(tx - s, ty - s);
        ctx.lineTo(tx - s + arm, ty - s);
        ctx.stroke();

        // Superior derecha
        ctx.beginPath();
        ctx.moveTo(tx + s - arm, ty - s);
        ctx.lineTo(tx + s, ty - s);
        ctx.lineTo(tx + s, ty - s + arm);
        ctx.stroke();

        // Inferior izquierda
        ctx.beginPath();
        ctx.moveTo(tx - s, ty + s - arm);
        ctx.lineTo(tx - s, ty + s);
        ctx.lineTo(tx - s + arm, ty + s);
        ctx.stroke();

        // Inferior derecha
        ctx.beginPath();
        ctx.moveTo(tx + s - arm, ty + s);
        ctx.lineTo(tx + s, ty + s);
        ctx.lineTo(tx + s, ty + s - arm);
        ctx.stroke();

        // Etiqueta técnica en micro-fuente
        ctx.font = "9px ui-monospace, SFMono-Regular, Menlo, monospace";
        ctx.fillStyle = `rgba(164, 216, 82, ${a * 0.9})`;
        ctx.fillText(currentTarget.label, tx - s, ty - s - 4);
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
  opacity: 0.70;
}

.dark .nature-canvas {
  opacity: 0.85;
}
</style>
