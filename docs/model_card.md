# Model Card · CNN pequeña para Fashion-MNIST

## Detalles del modelo
- **Desarrollador:** Melvin H. Rivas Genao (INF-8239, UASD), 2026-10-11.
- **Tipo:** red convolucional de 2 capas convolucionales + 2 densas, 56,714 parámetros, PyTorch 2.11.0+cpu.
- **Archivo:** `reports/e04/cnn_best.pt` (225 KB). Época elegida por validación: 5 de 6.

## Uso previsto
- Fines educativos: clasificar imágenes 28×28 en escala de grises de prendas en 10 categorías.
- **Fuera de alcance:** fotografías reales de productos, imágenes a color o de otra resolución, decisiones comerciales sin validación adicional.

## Datos
- Fashion-MNIST (Xiao et al., 2017), licencia MIT. 60,000 imágenes de entrenamiento (54,000 entrenamiento + 6,000 validación estratificada) y 10,000 de prueba oficial, 10 clases balanceadas.
- Auditoría: 0 duplicados exactos en entrenamiento, 0 en prueba y 0 imágenes de prueba repetidas en entrenamiento.

## Métricas (prueba oficial, 10,000 imágenes)
- Accuracy: 0.8719 · F1-macro: 0.8726.
- Línea base (regresión logística sobre píxeles): accuracy 0.8218.
- Clase más difícil: Shirt (F1 0.636). Confusión más frecuente: Shirt → T-shirt/top (143 casos).

## Costo
- Entrenamiento: 133.4 s en CPU de Colab (2 núcleos). Inferencia de 10,000 imágenes: 2213 ms.

## Limitaciones y riesgos
- Imágenes centradas, sin fondo y de baja resolución: el desempeño no se traslada a fotos reales.
- Confunde prendas superiores de silueta similar (camisa, camiseta, suéter, abrigo).
- Los tiempos dependen del hardware asignado por Colab y varían entre sesiones.
- Una sola semilla de entrenamiento; no se estimó la variación entre ejecuciones.

## Consideraciones éticas
- Datos sin personas ni información sensible. El sesgo principal es de dominio (catálogo de Zalando).
