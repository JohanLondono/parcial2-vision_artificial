# Sistema de Detección de Objetos en Tráfico Vehicular
## Documentación Técnica - Enfoque en Extracción de Características

**Universidad del Quindío - Visión Artificial**  
**Fecha:** Octubre 2025  
**Versión:** 5.0 - Edición Optimizada y Enfocada

---

## Tabla de Contenidos

1. [Introducción](#1-introducción)
2. [Arquitectura del Sistema](#2-arquitectura-del-sistema)
3. [Fundamentos Matemáticos Básicos](#3-fundamentos-matemáticos-básicos)
4. [Preprocesamiento (Resumen)](#4-preprocesamiento-resumen)
5. [Extracción de Características - Descriptores Clásicos](#5-extracción-de-características---descriptores-clásicos)
6. [Extracción de Características - Algoritmos Avanzados](#6-extracción-de-características---algoritmos-avanzados)
7. [Análisis de Texturas](#7-análisis-de-texturas)
8. [Detección de Objetos Específicos](#8-detección-de-objetos-específicos)
9. [Evaluación y Comparación](#9-evaluación-y-comparación)
10. [Referencias](#10-referencias)

---

## 1. Introducción

### 1.1 Objetivos del Sistema

Sistema especializado en **extracción de características visuales** para análisis de tráfico vehicular, implementando:

- ✅ **Descriptores Clásicos:** HOG, SIFT, SURF, ORB, KAZE, AKAZE
- ✅ **Descriptores Avanzados:** FREAK (bio-inspirado)
- ✅ **Análisis de Texturas:** GLCM, LBP
- ✅ **Análisis de Movimiento:** Optical Flow (Lucas-Kanade, Farneback)
- ✅ **Segmentación Avanzada:** GrabCut con modelos GMM
- ✅ **Detección de Formas:** Transformada de Hough, LoG

### 1.2 Enfoque Principal

Este documento se centra en:
1. **Fundamentos matemáticos** de cada algoritmo de extracción
2. **Comparación rigurosa** entre métodos
3. **Aplicaciones específicas** en tráfico vehicular
4. **Ventajas y limitaciones** de cada técnica

---

## 2. Arquitectura del Sistema

### 2.1 Módulos de Extracción

```
Sistema de Extracción de Características
│
├── modules/
│   ├── hog_kaze.py                # HOG + KAZE
│   ├── surf_orb.py                # SURF + ORB
│   ├── advanced_algorithms.py     # AKAZE, FREAK, GrabCut, LoG
│   ├── texture_analysis.py        # GLCM, LBP
│   └── comparison_metrics.py      # Evaluación comparativa
│
└── detectores/
    ├── detector_llantas.py        # Hough + Textura
    ├── detector_senales.py        # Color + Forma
    └── detector_semaforos.py      # Estructura + Estado
```

### 2.2 Pipeline de Extracción

```
Imagen → Preprocesamiento (mínimo) → Extracción Características → Análisis → Resultados
                                            ↓
                        ┌────────────────────┴────────────────────┐
                        │                                         │
                   Descriptores                              Análisis
                   Locales/Globales                         Temporal
```

---

## 3. Fundamentos Matemáticos Básicos

### 3.1 Convolución y Gradientes

**Convolución (base de todo filtro):**
$$
(I * K)(x, y) = \sum_i \sum_j I(x-i, y-j) \cdot K(i, j)
$$

**Gradientes (detección de características):**
- **Magnitud:** $|G| = \sqrt{G_x^2 + G_y^2}$
- **Orientación:** $\theta = \arctan(G_y/G_x)$

### 3.2 Espacio de Escala

**Gaussiano (lineal):**
$$
L(x, y, \sigma) = G(x, y, \sigma) * I(x, y)
$$

**No Lineal (KAZE/AKAZE):**
$$
\frac{\partial L}{\partial t} = \text{div}(c(x,y,t) \cdot \nabla L)
$$

Donde $c$ es función de conductividad que preserva bordes.

---

## 4. Preprocesamiento (Resumen)

### 4.1 Operaciones Esenciales

| Operación | Propósito | Complejidad |
|-----------|-----------|-------------|
| **Filtro Gaussiano** | Reducción de ruido | $O(n^2)$ |
| **Normalización** | Invarianza a iluminación | $O(n)$ |
| **Redimensionamiento** | Uniformidad de escala | $O(n)$ |
| **Conversión Grises** | Reducción dimensionalidad | $O(n)$ |

### 4.2 Filtros Aplicados

- **Suavizado:** Gaussiano con $\sigma = 1.0-2.0$
- **Realce de bordes:** Sobel, Canny
- **Morfología:** Apertura/cierre para ruido

**Nota:** El preprocesamiento es mínimo para preservar información de características.

---

## 5. Extracción de Características - Descriptores Clásicos

### 5.1 HOG (Histogram of Oriented Gradients)

#### Fundamento Matemático

HOG captura la distribución de gradientes orientados en regiones locales.

**Pipeline completo:**

1. **Normalización Gamma:**
   $$I'(x,y) = I(x,y)^\gamma, \quad \gamma \approx 0.5$$

2. **Cálculo de Gradientes:**
   $$G_x = I * [-1, 0, +1], \quad G_y = I * [-1, 0, +1]^T$$
   $$|G| = \sqrt{G_x^2 + G_y^2}, \quad \theta = \arctan(G_y/G_x)$$

3. **Histogramas por Celda:**
   - Celda: 8×8 píxeles
   - 9 bins de orientación (0°-180°)
   - Voto ponderado por magnitud

4. **Normalización por Bloques:**
   - Bloque: 2×2 celdas = 16×16 píxeles
   - Normalización L2-Hys:
   $$v' = \frac{v}{||v||_2 + \epsilon}, \quad v'' = \min(v', 0.2), \quad v_{final} = \frac{v''}{||v''||_2 + \epsilon}$$

5. **Descriptor Final:**
   - Dimensión típica: 3780D para imagen 128×64
   - Concatenación de todos los bloques

#### Características Clave

- **Tipo:** Descriptor denso (grid regular)
- **Invarianzas:** Escala parcial, iluminación moderada
- **Dimensionalidad:** Alta (miles de valores)
- **Velocidad:** Moderada (más lento que binarios)

#### Aplicaciones en Tráfico

- 🚗 **Detección de vehículos:** Template matching
- 🚶 **Detección de peatones:** Clasificador SVM
- 🪧 **Clasificación de señales:** Forma + color

---

### 5.2 SIFT (Scale-Invariant Feature Transform)

#### Fundamento Matemático

**Espacio de Escala Gaussiano:**
$$
L(x, y, \sigma) = G(x, y, \sigma) * I(x, y)
$$

**Difference of Gaussians (DoG):**
$$
D(x, y, \sigma) = L(x, y, k\sigma) - L(x, y, \sigma) \approx \sigma \nabla^2 G
$$

**Detección de extremos:** Buscar máximos/mínimos en espacio 3D (x, y, σ)

**Construcción del Descriptor:**

1. **Región 16×16 alrededor del keypoint**
2. **4×4 subregiones de 4×4 píxeles**
3. **Histograma de 8 orientaciones por subregión**
4. **Descriptor final:** 4×4×8 = **128 dimensiones**

**Orientación dominante:**
$$
\theta = \text{pico del histograma de orientaciones en ventana circular}
$$

#### Características Clave

- **Invarianzas:** ⭐⭐⭐⭐⭐ (escala, rotación, iluminación, viewpoint parcial)
- **Precisión:** Muy alta
- **Velocidad:** Lenta
- **Memoria:** Media (128D flotante)
- **Patente:** Expirada (2020)

#### Ventajas vs Desventajas

**✅ Ventajas:**
- Extremadamente robusto
- Gold standard para matching
- Excelente localización

**❌ Desventajas:**
- Computacionalmente costoso
- Descriptor flotante (no para matching binario)
- Muchos keypoints redundantes

---

### 5.3 SURF (Speeded-Up Robust Features)

#### Fundamento Matemático

**Imagen Integral:** Acelera cómputo de sumas en regiones rectangulares
$$
I_{\text{sum}}(x,y) = \sum_{i=0}^x \sum_{j=0}^y I(i,j)
$$

**Aproximación de Laplaciano con Box Filters:**
$$
\det(H_{\text{approx}}) = D_{xx} D_{yy} - (0.9 D_{xy})^2
$$

Donde $D_{xx}$, $D_{yy}$, $D_{xy}$ son respuestas de filtros box.

**Descriptor:**

1. **Región 20s×20s** (s = escala)
2. **4×4 subregiones**
3. **Por subregión:** $\sum dx, \sum dy, \sum |dx|, \sum |dy|$
4. **Descriptor final:** 4×4×4 = **64 dimensiones** (o 128 extendido)

#### Características Clave

- **Velocidad:** 3-4× más rápido que SIFT
- **Precisión:** Comparable a SIFT
- **Descriptor:** Más compacto (64D vs 128D)
- **Invarianzas:** Escala, rotación

#### Comparación SURF vs SIFT

| Aspecto | SURF | SIFT |
|---------|------|------|
| **Velocidad** | ⚡⚡⚡⚡ | ⚡⚡ |
| **Precisión** | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| **Descriptor** | 64D/128D | 128D |
| **Implementación** | Imagen integral | Pirámide DoG |
| **Patente** | Sí (restrictiva) | Expirada |

---

### 5.4 ORB (Oriented FAST and Rotated BRIEF)

#### Fundamento Matemático

**Componente 1: FAST (Features from Accelerated Segment Test)**

Detector de esquinas ultra-rápido:
1. Considerar círculo de 16 píxeles alrededor de punto $p$
2. Si $n$ píxeles consecutivos son todos más brillantes o más oscuros que $I_p + t$
3. Entonces $p$ es esquina

**Componente 2: Orientación con Momentos de Intensidad**
$$
\theta = \arctan2(m_{01}, m_{10})
$$

Donde $m_{pq} = \sum x^p y^q I(x,y)$ en vecindad del keypoint.

**Componente 3: BRIEF Rotado (rBRIEF)**

1. **Selección de pares:** 256 pares $(x_i, y_i)$ predefinidos
2. **Test binario:**
   $$b_i = \begin{cases} 1 & \text{si } I(x_i) < I(y_i) \\ 0 & \text{en otro caso} \end{cases}$$
3. **Rotación según orientación:** Rotar patrón con matriz de rotación $R_\theta$
4. **Descriptor final:** 256 bits = 32 bytes

**Matching con Distancia Hamming:**
$$
d_H(a, b) = \text{popcount}(a \oplus b)
$$

#### Características Clave

- **Velocidad:** ⚡⚡⚡⚡⚡ (más rápido de todos)
- **Descriptor:** Binario (32 bytes)
- **Matching:** Hardware-accelerated (XOR + popcount)
- **Licencia:** Libre (no patentado)
- **Invarianza:** Rotación ✓, Escala ✗

#### Aplicaciones Ideales

- 📱 **Dispositivos móviles:** Recursos limitados
- 🎥 **Tiempo real:** Video tracking, SLAM
- 🤖 **Robótica:** Navegación, localización
- 🔧 **Prototipado rápido:** Desarrollo ágil

---

### 5.5 KAZE y AKAZE

#### Fundamento Matemático

**KAZE:** Difusión No Lineal en Espacio de Escala

**Ecuación de Difusión:**
$$
\frac{\partial L}{\partial t} = \text{div}(c(x,y,t) \cdot \nabla L)
$$

**Función de Conductividad (Perona-Malik G2):**
$$
c(x,y,t) = \frac{1}{1 + \left(\frac{|\nabla L_\sigma|}{K}\right)^2}
$$

**Efecto:**
- $|\nabla L| \ll K$: $c \approx 1$ → difusión completa (regiones uniformes)
- $|\nabla L| \gg K$: $c \approx 0$ → no difusión (bordes preservados)

**Ventaja sobre Gaussiano:**
- Gaussiano: Borra bordes junto con ruido
- KAZE: Preserva bordes mientras suaviza texturas

#### AKAZE: Aceleración con FED

**Fast Explicit Diffusion (FED):**
- Esquema numérico más eficiente
- 2-3× más rápido que KAZE
- Convergencia similar

**Descriptor M-LDB (Modified Local Difference Binary):**

1. **Grid de muestreo** adaptativo por escala
2. **Comparaciones binarias** entre regiones
3. **Descriptor:** 486 bits (61 bytes)

#### Comparación Completa

| Característica | KAZE | AKAZE | SIFT | SURF | ORB |
|----------------|------|-------|------|------|-----|
| **Espacio escala** | No lineal | No lineal (FED) | Gaussiano | Gaussiano | Pirámide |
| **Descriptor** | M-SURF (64D) | M-LDB (486b) | 128D | 64D | 256b |
| **Tipo** | Flotante | Binario | Flotante | Flotante | Binario |
| **Velocidad** | ⚡⚡ | ⚡⚡⚡ | ⚡ | ⚡⚡⚡ | ⚡⚡⚡⚡ |
| **Precisión** | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐ |
| **Memoria** | Alta | Media | Media | Media | Baja |
| **Patente** | No | No | No | Sí | No |

#### Aplicaciones en Tráfico

- 🎯 **Matching robusto:** Imágenes con bordes fuertes
- 🌧️ **Condiciones adversas:** Lluvia, niebla (bordes preservados)
- 📸 **Alta resolución:** Detalles finos en señales
- 🔄 **Multi-escala:** Vehículos a diferentes distancias

---

## 6. Extracción de Características - Algoritmos Avanzados

### 6.1 FREAK (Fast Retina Keypoint)

#### Inspiración Biológica

**Retina Humana:**
- **Fóvea central:** Alta densidad de fotoreceptores (visión detallada)
- **Periferia:** Baja densidad (visión periférica)

**Patrón Log-Polar de FREAK:** Imita esta distribución

#### Fundamento Matemático

**Patrón de Muestreo:**

```
43 puntos en 7-8 capas concéntricas:
- Capa 0 (centro): 1 punto (σ₀)
- Capa 1: 6 puntos (σ₁ = 2σ₀)
- Capa 2: 6 puntos (σ₂ = 4σ₀)
- Capa 3: 6 puntos (σ₃ = 8σ₀)
- ...
- Capa 7: 6 puntos (σ₇ = 128σ₀)

Radio crece exponencialmente: r_i = 2^i · r_0
```

**Construcción del Descriptor:**

**Paso 1: Suavizado Gaussiano por Punto**
$$
S_i = \int\int I(x,y) \cdot G_{\sigma_i}(x - x_i, y - y_i) \, dx \, dy
$$

**Paso 2: Selección de Pares (Aprendizaje)**
$$
\text{De 43 puntos} \rightarrow \frac{43 \times 42}{2} = 903 \text{ pares posibles}
$$
$$
\text{Seleccionar 512 mejores usando correlación de Pearson}
$$

**Paso 3: Descriptor Binario**
$$
\text{Bit}_k = \begin{cases} 1 & \text{si } S_i > S_j \text{ para par } k=(i,j) \\ 0 & \text{en otro caso} \end{cases}
$$

**Paso 4: Orientación Cascada**
$$
\theta = \arctan2\left(\sum_{k} w_k S_k \sin(\phi_k), \sum_{k} w_k S_k \cos(\phi_k)\right)
$$

Donde $w_k$ son pesos aprendidos y $\phi_k$ ángulos de los puntos.

#### Características Clave

- **Descriptor:** 512 bits (64 bytes)
- **Patrón:** Bio-inspirado, aprendido
- **Matching:** Distancia Hamming (rápido)
- **Invarianzas:** Rotación ✓, Escala ✓
- **Velocidad:** ⚡⚡⚡

#### Comparación con Otros Binarios

| Aspecto | FREAK | BRIEF | ORB | BRISK | AKAZE |
|---------|-------|-------|-----|-------|-------|
| **Patrón** | Log-polar (43 pts) | Aleatorio | Rotado | Multi-escala | Grid adaptivo |
| **Bits** | 512 | 128-512 | 256 | 512 | 486 |
| **Orientación** | ✓ Cascada | ✗ | ✓ Momentos | ✓ Gradiente | ✓ Dominante |
| **Aprendizaje** | ✓ Pares | ✗ | ✗ | ✗ | ✗ |
| **Velocidad** | ⚡⚡⚡ | ⚡⚡⚡⚡ | ⚡⚡⚡⚡ | ⚡⚡ | ⚡⚡⚡ |
| **Precisión** | ⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ |

#### Aplicaciones Específicas

- 📸 **Reconocimiento de señales:** Matching con base de datos
- 🎬 **Tracking temporal:** Seguimiento frame-a-frame
- 🤖 **SLAM:** Localización y mapeo simultáneo
- 📱 **Apps móviles:** Balance precisión-eficiencia

---

### 6.2 GrabCut - Segmentación Avanzada

#### Fundamento Matemático

**Formulación como Optimización de Energía:**
$$
E(\alpha, k, \theta, z) = U(\alpha, k, \theta, z) + V(\alpha, z)
$$

**Variables:**
- $\alpha \in \{0,1\}^n$: Etiquetas binarias (0=fondo, 1=objeto)
- $k \in \{1,...,K\}^n$: Componente GMM por píxel
- $\theta$: Parámetros GMM (medias $\mu$, covarianzas $\Sigma$, pesos $\pi$)
- $z$: Observaciones (colores RGB)

#### Modelos de Mezcla Gaussiana (GMM)

**GMM para Objeto:**
$$
p(z|\alpha=1, \theta) = \sum_{k=1}^K \pi_k \cdot \mathcal{N}(z; \mu_k, \Sigma_k)
$$

**GMM para Fondo:**
$$
p(z|\alpha=0, \theta) = \sum_{k=K+1}^{2K} \pi_k \cdot \mathcal{N}(z; \mu_k, \Sigma_k)
$$

**Gaussiana Multivariada (RGB, 3D):**
$$
\mathcal{N}(z; \mu, \Sigma) = \frac{1}{(2\pi)^{3/2}|\Sigma|^{1/2}} \exp\left(-\frac{1}{2}(z-\mu)^T\Sigma^{-1}(z-\mu)\right)
$$

Típicamente: $K=5$ componentes para objeto, $K=5$ para fondo.

#### Término de Datos (Unario)

$$
U(\alpha, k, \theta, z) = \sum_{n=1}^N D(\alpha_n, k_n, \theta, z_n)
$$

$$
D(\alpha_n, k_n, \theta, z_n) = -\log p(z_n | \alpha_n, k_n, \theta) = -\log \pi_{k_n} - \log \mathcal{N}(z_n; \mu_{k_n}, \Sigma_{k_n})
$$

**Interpretación:** Penaliza asignaciones de píxeles a modelos improbables.

#### Término de Suavidad (Pairwise)

$$
V(\alpha, z) = \gamma \sum_{(n,m) \in \mathcal{C}} [\alpha_n \neq \alpha_m] \cdot \frac{1}{\text{dist}(n,m)} \cdot \exp(-\beta||z_n - z_m||^2)
$$

**Parámetro de contraste:**
$$
\beta = \frac{1}{2 \mathbb{E}[||z_n - z_m||^2]}
$$

**Interpretación:**
- Si $||z_n - z_m||$ grande (colores diferentes): Permite borde ($\exp(-\beta \cdot \text{grande}) \approx 0$)
- Si $||z_n - z_m||$ pequeño (colores similares): Penaliza borde ($\exp(-\beta \cdot \text{pequeño}) \approx 1$)

#### Algoritmo Iterativo

```
Inicialización:
1. Usuario dibuja rectángulo R
2. α_n = {fondo definitivo si n fuera de R
         {probable objeto si n dentro de R
3. Inicializar GMMs con k-means en píxeles conocidos

Iteración (hasta convergencia):
  
  ┌─ Paso 1: Asignar Componentes GMM ─────────────────┐
  │ Para cada píxel n:                                 │
  │   k_n = argmax_k p(z_n | k, θ_α_n)                │
  └────────────────────────────────────────────────────┘
  
  ┌─ Paso 2: Aprender Parámetros GMM ─────────────────┐
  │ Actualizar {π_k, μ_k, Σ_k} usando píxeles         │
  │ asignados a cada componente (EM algorithm)         │
  └────────────────────────────────────────────────────┘
  
  ┌─ Paso 3: Estimar Segmentación ────────────────────┐
  │ Resolver min-cut en grafo G=(V,E):                │
  │   - V = píxeles + source + sink                   │
  │   - E con capacidades según U y V                 │
  │ Algoritmo: Max-flow/Min-cut (Boykov-Kolmogorov)   │
  └────────────────────────────────────────────────────┘
  
  Si cambio en α < umbral: CONVERGE
```

#### Graph Cut - Formulación

**Grafo G = (V, E):**
- **Vértices:** $V = \{\text{píxeles}\} \cup \{\text{source}, \text{sink}\}$
- **Aristas terminales (t-links):**
  - $\text{source} \rightarrow n$: capacidad = $-\log p(z_n|\alpha=1)$
  - $n \rightarrow \text{sink}$: capacidad = $-\log p(z_n|\alpha=0)$
- **Aristas de vecindad (n-links):**
  - $n \leftrightarrow m$: capacidad = $V(\alpha_n, \alpha_m, z_n, z_m)$

**Teorema:** El corte mínimo del grafo equivale a minimizar $E(\alpha, k, \theta, z)$.

#### Ventajas y Limitaciones

**✅ Ventajas:**
- Segmentación precisa con mínima interacción
- Modela distribución de color compleja (no solo umbrales)
- Considera contexto espacial (smoothness)
- Iterativo y refinable

**❌ Limitaciones:**
- Requiere imagen en **color** (no funciona en escala de grises)
- Sensible a inicialización (rectángulo inicial)
- Computacionalmente costoso ($O(n^3)$ en peor caso)
- Asume separación clara de colores objeto/fondo

#### Aplicaciones en Tráfico

- 🚗 **Segmentación de vehículos:** Extraer vehículo del fondo
- 🪧 **Extracción de señales:** Aislar señal para análisis detallado
- 🎨 **Dataset preparation:** Generar máscaras ground-truth
- 📊 **Análisis de forma:** Post-segmentación para clasificación

---

### 6.3 Optical Flow - Análisis de Movimiento

#### Fundamento Matemático

**Ecuación de Restricción de Brillo:**

Asunción: Intensidad de un punto se conserva entre frames.

$$
I(x, y, t) = I(x + dx, y + dy, t + dt)
$$

**Expansión de Taylor (primer orden):**
$$
I(x+dx, y+dy, t+dt) \approx I(x,y,t) + \frac{\partial I}{\partial x}dx + \frac{\partial I}{\partial y}dy + \frac{\partial I}{\partial t}dt
$$

**Igualando y dividiendo por $dt$:**
$$
I_x \cdot u + I_y \cdot v + I_t = 0
$$

Donde $u = \frac{dx}{dt}$, $v = \frac{dy}{dt}$ son las velocidades del flujo.

**Problema:** 1 ecuación, 2 incógnitas → **Aperture Problem** (mal planteado)

#### 6.3.1 Lucas-Kanade (Flujo Disperso)

**Restricción Adicional:** Flujo constante en vecindad local.

**Sistema sobredeterminado (ventana de $n$ píxeles):**
$$
\begin{bmatrix}
I_x(p_1) & I_y(p_1) \\
I_x(p_2) & I_y(p_2) \\
\vdots & \vdots \\
I_x(p_n) & I_y(p_n)
\end{bmatrix}
\begin{bmatrix} u \\ v \end{bmatrix}
=
-\begin{bmatrix}
I_t(p_1) \\
I_t(p_2) \\
\vdots \\
I_t(p_n)
\end{bmatrix}
$$

**Solución por Mínimos Cuadrados:**
$$
\begin{bmatrix} u \\ v \end{bmatrix} = 
\left(A^T A\right)^{-1} A^T b
$$

$$
= \begin{bmatrix} 
\sum I_x^2 & \sum I_xI_y \\ 
\sum I_xI_y & \sum I_y^2 
\end{bmatrix}^{-1}
\begin{bmatrix}
-\sum I_xI_t \\
-\sum I_yI_t
\end{bmatrix}
$$

**Condición de Invertibilidad:**
$$
\det(A^T A) = \left(\sum I_x^2\right)\left(\sum I_y^2\right) - \left(\sum I_xI_y\right)^2 > \text{umbral}
$$

- $\det$ grande → Esquina bien definida → **Flujo confiable** ✅
- $\det$ pequeño → Región uniforme o borde → **Flujo no confiable** ❌

**Pirámide Multiescala:**

Para desplazamientos grandes (violación de linealización):

```
Nivel 3 (baja res)  →  Estimar flujo grueso
        ↓
Nivel 2             →  Refinar flujo (warp + estimar)
        ↓
Nivel 1             →  Refinar flujo (warp + estimar)
        ↓
Nivel 0 (alta res)  →  Flujo final preciso
```

#### 6.3.2 Farneback (Flujo Denso)

**Aproximación Polinomial Cuadrática:**
$$
I(x) \approx x^T A x + b^T x + c
$$

Para desplazamiento $d$:
$$
I(x-d) \approx (x-d)^T A (x-d) + b^T (x-d) + c
$$

Expandiendo:
$$
= x^T A x - 2d^T A x + d^T A d + b^T x - b^T d + c
$$

**Igualando con $I_2(x)$ (siguiente frame):**

Asumiendo $A$ constante entre frames:
$$
b_2 = b_1 - 2Ad
$$
$$
d = \frac{1}{2}A^{-1}(b_1 - b_2)
$$

**Estimación:**
1. Estimar $A$, $b_1$, $c_1$ en frame 1
2. Estimar $A$, $b_2$, $c_2$ en frame 2
3. Calcular desplazamiento $d$ de los coeficientes

**Parámetros Clave:**

| Parámetro | Valor Típico | Efecto |
|-----------|--------------|--------|
| `pyr_scale` | 0.5 | Reducción entre niveles pirámide |
| `levels` | 3-5 | Número de niveles (más = desplaz. grandes) |
| `winsize` | 15-25 | Ventana promediado (más = smooth) |
| `iterations` | 3 | Iteraciones por nivel |
| `poly_n` | 5-7 | Vecindad para polinomio (impar) |
| `poly_sigma` | 1.1-1.5 | Gaussiana para suavizado coeficientes |

#### Métricas Extraídas del Flujo

**Por píxel:**
- **Magnitud:** $m(x,y) = \sqrt{u^2 + v^2}$
- **Dirección:** $\theta(x,y) = \arctan2(v, u) \in [-\pi, \pi]$

**Globales:**
- **Velocidad promedio:** $\bar{m} = \frac{1}{N}\sum m(x,y)$
- **Dirección dominante:** Pico de histograma circular
- **Coherencia:** Similitud de direcciones locales
  $$\text{coherencia} = 1 - \frac{|\theta(x,y) - \theta_{\text{smoothed}}(x,y)|}{\pi}$$

#### Visualización HSV

**Codificación Color:**
- **Hue (Tono):** Dirección del flujo ($\theta$ → [0, 180])
- **Saturation:** 255 (máximo)
- **Value (Brillo):** Magnitud normalizada

```
Para cada píxel (x, y):
  HSV[0] = θ(x,y) * 180/π / 2    # Dirección → Tono
  HSV[1] = 255                    # Saturación máxima
  HSV[2] = normalize(m(x,y))      # Magnitud → Brillo
```

#### Comparación Lucas-Kanade vs Farneback

| Aspecto | Lucas-Kanade | Farneback |
|---------|--------------|-----------|
| **Tipo** | Disperso (sparse) | Denso (dense) |
| **Salida** | Vectores en puntos clave | Vector por píxel |
| **Velocidad** | ⚡⚡⚡ Rápido | ⚡⚡ Moderado |
| **Memoria** | Baja (solo keypoints) | Alta (H×W×2 array) |
| **Precisión** | Alta en puntos | Moderada global |
| **Complejidad** | $O(k \cdot n)$ | $O(H \cdot W \cdot n)$ |
| **Aplicación** | Tracking objetos | Flujo global, análisis campo |

#### Aplicaciones en Tráfico Vehicular

**1. Análisis de Flujo:**
- Velocidad promedio del tráfico
- Identificación de cuellos de botella
- Patrones de movimiento dominantes

**2. Detección de Eventos:**
- Frenadas bruscas ($\Delta m$ grande)
- Cambios de carril (dirección perpendicular)
- Congestión (velocidad baja, coherencia alta)

**3. Conteo de Vehículos:**
- Tracking de trayectorias
- Detección de cruces de línea virtual
- Estimación de densidad

**4. Seguridad:**
- Movimientos erráticos (coherencia baja)
- Detección de accidentes (movimiento nulo súbito)
- Validación de señales (velocidad compatible)

---

### 6.4 Laplaciano de Gaussiana (LoG)

#### Fundamento Matemático

**Laplaciano:**
$$
\nabla^2 I = \frac{\partial^2 I}{\partial x^2} + \frac{\partial^2 I}{\partial y^2}
$$

**Laplaciano de Gaussiana:**
$$
\nabla^2 G(x,y,\sigma) = -\frac{1}{\pi\sigma^4}\left[1 - \frac{x^2+y^2}{2\sigma^2}\right]\exp\left(-\frac{x^2+y^2}{2\sigma^2}\right)
$$

**Propiedad clave:**
$$
\nabla^2 (G * I) = (\nabla^2 G) * I
$$

Podemos pre-calcular $\nabla^2 G$ como kernel.

#### Detección de Blobs

**Principio:** LoG tiene respuesta máxima en centro de blob cuando $\sigma$ coincide con tamaño del blob.

**Algoritmo:**
1. Calcular $\text{LoG}(I, \sigma)$ para múltiples escalas $\sigma$
2. Detectar extremos en espacio 3D $(x, y, \sigma)$
3. Radio del blob: $r \approx \sqrt{2} \sigma$

#### Aplicaciones

- 🔴 **Detección de círculos:** Semáforos, señales circulares
- ⚪ **Detección de llantas:** Regiones circulares oscuras
- 🔍 **Preprocesamiento:** Antes de Hough circles

---

## 7. Análisis de Texturas

### 7.1 GLCM (Gray-Level Co-occurrence Matrix)

#### Fundamento Matemático

**Definición:**
$$
P(i, j | d, \theta) = \#\{(x_1,y_1), (x_2,y_2) : I(x_1,y_1)=i, I(x_2,y_2)=j, (x_2-x_1, y_2-y_1)=(d\cos\theta, d\sin\theta)\}
$$

**Normalización:**
$$
P_{\text{norm}}(i,j) = \frac{P(i,j)}{\sum_{i,j} P(i,j)}
$$

#### Métricas de Haralick (Selección)

**1. Contraste (Variabilidad Local):**
$$
\text{Contraste} = \sum_{i=0}^{N-1} \sum_{j=0}^{N-1} (i-j)^2 \cdot P(i,j)
$$
- Valores altos → Bordes fuertes, transiciones abruptas
- Valores bajos → Textura suave, uniforme

**2. Energía/ASM (Uniformidad):**
$$
\text{Energía} = \sum_{i=0}^{N-1} \sum_{j=0}^{N-1} P(i,j)^2
$$
- Valores altos → Textura repetitiva, ordenada
- Valores bajos → Textura caótica, aleatoria

**3. Homogeneidad (Suavidad):**
$$
\text{Homogeneidad} = \sum_{i=0}^{N-1} \sum_{j=0}^{N-1} \frac{P(i,j)}{1 + |i-j|}
$$
- Valores altos → Variaciones graduales
- Valores bajos → Cambios abruptos

**4. Correlación (Dependencia Lineal):**
$$
\text{Correlación} = \sum_{i=0}^{N-1} \sum_{j=0}^{N-1} \frac{(i-\mu_i)(j-\mu_j) \cdot P(i,j)}{\sigma_i \cdot \sigma_j}
$$

#### Aplicaciones en Tráfico

| Objeto | Contraste | Energía | Homogeneidad | Interpretación |
|--------|-----------|---------|--------------|----------------|
| **Llanta** | Alto | Bajo | Bajo | Patrón radial, textura compleja |
| **Carretera** | Bajo | Alto | Alto | Uniforme, lisa |
| **Señal** | Medio | Medio | Medio | Diseño estructurado |
| **Vegetación** | Medio | Bajo | Medio | Textura irregular |

---

### 7.2 LBP (Local Binary Patterns)

#### Fundamento Matemático

**Definición para LBP básico:**
$$
\text{LBP}(x_c, y_c) = \sum_{p=0}^{P-1} s(I_p - I_c) \cdot 2^p
$$

Donde:
$$
s(x) = \begin{cases} 1 & \text{si } x \geq 0 \\ 0 & \text{en otro caso} \end{cases}
$$

**LBP Uniforme:** Solo patrones con máximo 2 transiciones 0→1 o 1→0

**LBP Invariante a Rotación:** Tomar valor mínimo entre todas las rotaciones

#### Ventajas

- ⚡ Extremadamente rápido (solo comparaciones)
- 🔄 Invariante a iluminación monotónica
- 📦 Descriptor compacto (histograma)
- 🎯 Captura microestructuras locales

---

## 8. Detección de Objetos Específicos

### 8.1 Detector de Llantas

#### Estrategia Multi-Método

**Método 1: Hough Circles + Validación GLCM**

1. **Transformada de Hough para Círculos:**
   $$
   (x - a)^2 + (y - b)^2 = r^2
   $$
   Espacio de parámetros 3D: $(a, b, r)$

2. **Filtros geométricos:**
   - Circularidad: $C = \frac{4\pi A}{P^2} \in [0.7, 1.0]$
   - Ratio aspecto: $\frac{W}{H} \in [0.8, 1.2]$
   - Rango radio: $r \in [20, 300]$ px

3. **Validación con textura:**
   - Extraer ROI circular
   - Calcular GLCM
   - Validar: Contraste alto, energía baja (patrón radial)

**Método 2: Contornos + Forma**

1. Umbralización adaptativa
2. Operaciones morfológicas (apertura)
3. Encontrar contornos
4. Filtrar por circularidad y área

**Método 3: Template Matching**

1. Templates pre-definidos de llantas
2. Multi-escala (0.5× a 2×)
3. Multi-rotación (0° a 360° cada 15°)
4. Correlación cruzada normalizada

#### Fusión de Resultados

```
Confianza_final = 0.4·Score_Hough + 0.3·Score_Contorno + 0.3·Score_Template
```

Umbral de aceptación: Confianza > 0.6

---

### 8.2 Detector de Señales de Tráfico

#### Pipeline

**1. Segmentación por Color (Espacio HSV)**

| Color | Rango H | Rango S | Rango V |
|-------|---------|---------|---------|
| Rojo | [0,10] ∪ [160,180] | [100,255] | [100,255] |
| Azul | [100,130] | [50,255] | [50,255] |
| Amarillo | [20,40] | [100,255] | [100,255] |

**2. Análisis de Forma Geométrica**

**Aproximación poligonal (Douglas-Peucker):**
$$
\epsilon = k \cdot \text{perímetro}
$$

**Clasificación:**
- **Círculo:** 3-6 vértices + circularidad > 0.7
- **Triángulo:** 3 vértices exactos
- **Rectángulo:** 4 vértices + ángulos ~90°
- **Octágono:** 7-9 vértices (STOP)

**3. Validación Multi-Criterio**
- ✅ Área: [500, 50000] px²
- ✅ Forma reconocida
- ✅ Color dominante correcto
- ✅ Ratio relleno > 0.7

---

### 8.3 Detector de Semáforos

#### Características Estructurales

**1. Detección de Caja:**
- Forma rectangular vertical
- Ratio alto/ancho: [2.5, 3.5]
- Color fondo: Negro/gris oscuro

**2. Detección de Luces:**

**Por color (HSV):**
- 🔴 Rojo: H∈[0,10]∪[170,180], S>150, V>150
- 🟡 Amarillo: H∈[20,30], S>150, V>150
- 🟢 Verde: H∈[40,80], S>100, V>100

**Por geometría:**
- Círculos con Hough
- Alineación vertical
- Espaciado uniforme

**3. Validación de Estado:**
- Solo 1 luz encendida (típico)
- Posiciones relativas coherentes
- Tamaño uniforme de luces

---

## 9. Evaluación y Comparación

### 9.1 Tabla Comparativa Global de Descriptores

| Algoritmo | Keypoints | Descriptor | Velocidad | Precisión | Invarianzas | Uso Ideal |
|-----------|-----------|------------|-----------|-----------|-------------|-----------|
| **HOG** | N/A (denso) | 3780D flotante | ⚡⚡ | ⭐⭐⭐⭐ | Escala parcial | Detección objetos, clasificación |
| **SIFT** | 500-3000 | 128D flotante | ⚡ | ⭐⭐⭐⭐⭐ | Escala, rotación, ilum. | Matching preciso, panoramas |
| **SURF** | 300-2000 | 64D flotante | ⚡⚡⚡ | ⭐⭐⭐⭐ | Escala, rotación | Balance velocidad-precisión |
| **ORB** | 500-1500 | 256b binario | ⚡⚡⚡⚡ | ⭐⭐⭐ | Rotación | Tiempo real, embebidos |
| **KAZE** | 400-2500 | 64D flotante | ⚡⚡ | ⭐⭐⭐⭐ | Escala, rotación | Bordes finos, alta calidad |
| **AKAZE** | 400-2000 | 486b binario | ⚡⚡⚡ | ⭐⭐⭐⭐ | Escala, rotación | Balance completo |
| **FREAK** | 300-1500 | 512b binario | ⚡⚡⚡ | ⭐⭐⭐⭐ | Rotación, escala | Bio-inspirado, eficiente |

### 9.2 Criterios de Selección

**Para Tráfico Vehicular:**

| Escenario | Algoritmo Recomendado | Justificación |
|-----------|----------------------|---------------|
| **Tiempo real (30+ FPS)** | ORB o AKAZE | Velocidad + binario |
| **Alta precisión** | SIFT o KAZE | Mejor localización |
| **Dispositivo móvil** | ORB o FREAK | Memoria baja |
| **Condiciones adversas** | KAZE o AKAZE | Preserva bordes |
| **Matching robusto** | SIFT o FREAK | Invarianzas múltiples |
| **Clasificación** | HOG + SVM | Descriptor denso |

### 9.3 Métricas de Evaluación

**Detección:**
- **Precisión:** $P = \frac{TP}{TP + FP}$
- **Recall:** $R = \frac{TP}{TP + FN}$
- **F1-Score:** $F_1 = \frac{2PR}{P+R}$
- **IoU:** $\frac{\text{Intersección}}{\text{Unión}}$

**Descriptores:**
- **Número de keypoints:** Cobertura
- **Tiempo extracción:** Eficiencia
- **Ratio matching:** $\frac{\text{Buenos matches}}{\text{Total keypoints}}$
- **Repeatability:** Consistencia bajo transformaciones

---

## 10. Referencias

### 10.1 Papers Fundamentales

1. **HOG:** Dalal & Triggs (2005). *Histograms of oriented gradients for human detection.* CVPR.

2. **SIFT:** Lowe (2004). *Distinctive image features from scale-invariant keypoints.* IJCV.

3. **SURF:** Bay et al. (2008). *Speeded-up robust features.* CVIU.

4. **ORB:** Rublee et al. (2011). *ORB: An efficient alternative to SIFT or SURF.* ICCV.

5. **KAZE:** Alcantarilla et al. (2012). *KAZE features.* ECCV.

6. **AKAZE:** Alcantarilla et al. (2013). *Fast explicit diffusion for accelerated features.* BMVC.

7. **FREAK:** Alahi et al. (2012). *FREAK: Fast retina keypoint.* CVPR.

8. **GrabCut:** Rother et al. (2004). *"GrabCut" - Interactive foreground extraction.* ACM TOG.

9. **Lucas-Kanade:** Lucas & Kanade (1981). *An iterative image registration technique.* IJCAI.

10. **Farneback:** Farnebäck (2003). *Two-frame motion estimation based on polynomial expansion.* SCIA.

11. **GLCM:** Haralick et al. (1973). *Textural features for image classification.* IEEE Trans. SMC.

### 10.2 Libros Recomendados

- **Szeliski, R.** (2022). *Computer Vision: Algorithms and Applications* (2nd ed.). [Cobertura completa moderna]

- **Gonzalez & Woods** (2018). *Digital Image Processing* (4th ed.). [Fundamentos sólidos]

- **Prince, S.J.D.** (2012). *Computer Vision: Models, Learning, and Inference*. [Enfoque probabilístico]

### 10.3 Recursos Online

- **OpenCV Docs:** https://docs.opencv.org/
- **PyImageSearch:** https://pyimagesearch.com/
- **Papers with Code:** https://paperswithcode.com/area/computer-vision

---

## Conclusión

### Resumen de Fortalezas por Categoría

**Descriptores Flotantes (Alta Precisión):**
- ✅ SIFT: Gold standard, máxima robustez
- ✅ SURF: Balance velocidad-precisión
- ✅ KAZE: Preservación de bordes, alta calidad

**Descriptores Binarios (Alta Velocidad):**
- ✅ ORB: Más rápido, ideal tiempo real
- ✅ AKAZE: Balance completo
- ✅ FREAK: Bio-inspirado, eficiente

**Descriptores Densos (Detección):**
- ✅ HOG: Clasificación de objetos, SVM

**Análisis Temporal:**
- ✅ Optical Flow: Movimiento, velocidad, tracking

**Segmentación:**
- ✅ GrabCut: Objeto/fondo con modelos GMM

**Texturas:**
- ✅ GLCM: Caracterización cuantitativa
- ✅ LBP: Rápido, invariante iluminación

### Recomendación Final

Para un sistema **robusto** de análisis de tráfico vehicular:

1. **Detección inicial:** HOG + SVM o YOLO
2. **Tracking:** ORB o AKAZE (velocidad)
3. **Matching preciso:** SIFT o KAZE (precisión)
4. **Análisis temporal:** Optical Flow (Farneback)
5. **Segmentación:** GrabCut (cuando necesario)
6. **Validación:** GLCM para texturas

**Combinación óptima** = Múltiples métodos + Fusión de resultados

---

**Desarrollado para Universidad del Quindío**  
**Visión Artificial - Octubre 2025**  
**Versión 5.0 - Documentación Optimizada y Enfocada en Características**
