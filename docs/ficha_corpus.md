# Ficha del corpus

## Pregunta
¿Es posible clasificar automáticamente un SMS como **spam** o **ham** (legítimo) a partir de su texto, de forma que un filtro bloquee la mayoría del spam sin bloquear mensajes legítimos?

- **Unidad de análisis:** un mensaje SMS (una fila).
- **Target:** `label` ∈ {ham, spam}. Clase positiva: `spam`.
- **Decisión apoyada:** enviar el mensaje a la bandeja normal o a la carpeta de spam.
- **Error más costoso:** falso positivo (un SMS legítimo bloqueado, p. ej. un código o una cita). Un falso negativo (spam que llega) es molesto pero recuperable.
- **Métricas:** F1 de la clase spam (principal), precisión de spam (controla falsos positivos), recall de spam, F1-macro y PR-AUC.

## Procedencia y licencia
- **Fuente:** UCI Machine Learning Repository, dataset #228. https://archive.ics.uci.edu/dataset/228/sms+spam+collection
- **Descarga directa:** https://archive.ics.uci.edu/static/public/228/sms+spam+collection.zip (archivo `SMSSpamCollection`, separado por tabulación, sin encabezado).
- **Autores:** Tiago A. Almeida y José María Gómez Hidalgo. **Donación:** 21-06-2012. **DOI:** 10.24432/C5CC84.
- **Licencia:** Creative Commons Attribution 4.0 (CC BY 4.0): uso académico permitido con atribución.
- **Cita:** Almeida, T. A., Gómez Hidalgo, J. M., & Yamakami, A. (2011). Contributions to the study of SMS spam filtering: new collection and results. *ACM DocEng 2011*.

## Trazabilidad (composición según los autores)
| Fuente | Mensajes | Clase |
|---|---|---|
| Grumbletext (foro del Reino Unido donde usuarios reportan spam) | 425 | spam |
| NUS SMS Corpus (voluntarios de la Universidad Nacional de Singapur) | 3,375 | ham |
| Tesis doctoral de Caroline Tag | 450 | ham |
| SMS Spam Corpus v.0.1 Big | 1,002 ham + 322 spam | ambas |
| **Total** | **5,574** (4,827 ham · 747 spam) | |

## Representatividad y límites
- Inglés de Reino Unido y Singapur, mensajes de 2003–2011; no representa el spam actual (enlaces acortados, phishing bancario, WhatsApp) ni SMS en español.
- Ham y spam provienen mayoritariamente de **fuentes distintas**: el modelo puede aprender rasgos de la fuente (estilo de Singapur vs. Reino Unido) además del contenido.
- Las etiquetas provienen de las fuentes originales; no hay acuerdo entre anotadores documentado.
