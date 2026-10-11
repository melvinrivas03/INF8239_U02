# Declaración de uso de inteligencia artificial

**Herramienta:** Claude (Anthropic), usado como tutor y asistente de programación durante el Ejercicio 03.

## Para qué se utilizó
- Proponer y comparar corpus candidatos según los criterios del mandato (licencia, tamaño, duplicados).
- Generar el código base del notebook, de los módulos `src/inf8239_u02/` y de las pruebas.
- Proponer la estrategia contra la fuga (deduplicación y partición por grupos de casi duplicados).
- Redactar borradores de la interpretación, el análisis de errores y la conclusión.

## Prompts relevantes (resumidos)
- "Revisa los dos PDF (introducción de la Unidad 02 y mandato del Ejercicio 03) y dime qué necesitas para empezar."
- Elección del corpus: "SMS Spam Collection", repositorio nuevo `INF8239_U02`.
- "Construye el notebook con auditoría, duplicados, baseline, dos pipelines comparables y el análisis de 20 errores."

## Qué verifiqué personalmente
- La licencia (CC BY 4.0), los autores y la composición del corpus en la ficha oficial de UCI.
- Que el notebook se ejecuta de principio a fin en Google Colab desde las instrucciones del README y que las pruebas pasan.
- Que las cifras del informe coinciden con `reports/cv_resultados.csv`, `reports/resultados_test.csv` y `reports/errores_20.csv`.
- La lectura de los 20 mensajes mal clasificados y su categoría.

## Correcciones realizadas
- Se cambió el modelo seleccionado de la regresión logística a Naive Bayes, tras comprobar que la validación cruzada no favorecía a la regresión logística y que Naive Bayes no produce falsos positivos, el error definido como más costoso.
- Se sustituyó la partición aleatoria por una partición por grupos al cuantificar que el 29.5 % del spam de prueba tenía casi duplicados en entrenamiento.
- Se ajustó la lectura del archivo con `quoting=QUOTE_NONE` porque las comillas de algunos mensajes rompían el parseo.

La responsabilidad sobre los datos, el código, las referencias y las conclusiones es del estudiante.
