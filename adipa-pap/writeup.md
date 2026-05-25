# Writeup Técnico — ADIPA PAP Classifier

**Candidato:** Juan Sebastián Páez Cortés  
**Rol:** Machine Learning Engineer  
**Fecha:** Mayo 2026

---

## 1. ¿LLM few-shot o modelo entrenado, y por qué?

**Usé LLM few-shot (Claude Haiku) como clasificador principal, con TF-IDF + LogReg como baseline de referencia y fallback.**

La evidencia empírica lo justifica directamente: el baseline entrenado sobre 8 casos alcanza **99% de accuracy en train pero solo 62% en held-out** (gap de 37 puntos). Ese colapso no es un problema de hiperparámetros — es el síntoma clásico de *memorización de estilo de autor*. Los 10 guiones los escribió el mismo equipo clínico con vocabulario altamente consistente por caso: "encerrona", "4-4-4", "citófono" son señales léxicas del caso Camila, no de la fase. Un modelo TF-IDF las aprende como features de fase porque co-ocurren, pero no generalizan a un caso nuevo.

Un LLM pre-entrenado en español clínico resuelve esto porque **clasifica por comprensión semántica**, no por co-ocurrencia léxica. No necesita haber visto "mina" o "derrumbe" (caso Luis) para entender que "Puede quedarse en silencio" es contención emocional.

**¿Con qué señal migraría de LLM a modelo entrenado?**

Con tres condiciones simultáneas: (a) ≥1.000 turnos anotados por humanos (no weak labels), (b) latencia promedio del LLM > 700 ms en producción real, y (c) costo mensual del LLM superior a mantener un endpoint Hugging Face en producción. Con esos datos, fine-tuning de BETO o DistilBERT-multilingual en multi-task (fase + actos verbales) superaría al few-shot en calidad y latencia. Hoy no tenemos nada de esto.

---

## 2. Detección de riesgo clínico

**NO implementada en el clasificador — discutida aquí.**

La detección de ideación suicida, autolesión y violencia es un problema **categorialmente distinto** a la clasificación de fase. Los costos de error son asimétricos: un falso negativo (no detectar riesgo real) puede implicar daño grave o muerte; un falso positivo genera fricción pero no daño irreversible.

Por eso, la métrica prioritaria es **recall, no precision**. Prefiero escalar 10 casos de no-riesgo a revisión humana antes que perder un caso de riesgo real. Un threshold de 0.3 en vez de 0.5 es razonable si el volumen de alertas es manejable por el equipo clínico.

La arquitectura correcta no es un clasificador binario aislado. Es un **modelo de abstención con calibración probabilística**: si la probabilidad de riesgo supera el umbral, el sistema no clasifica — deriva a revisión humana inmediata con el turno completo y el contexto. El clasificador no resuelve el caso de riesgo: lo escala. El rol de la IA es detección de señal, no toma de decisión clínica.

Para producción real usaría un ensemble: (a) keywords de alta precisión clínica ("quiero morir", "no quiero seguir", "voy a hacerlo") como trigger rápido + (b) un clasificador semántico calibrado (Platt scaling o isotonic regression sobre las probabilidades del modelo) + (c) log obligatorio de todos los turnos marcados como riesgo, con revisión posterior por clínicos para active learning del módulo.

---

## 3. LLM externo vs modelo local para el endpoint en vivo

**Para la sesión de voz en tiempo real, la arquitectura correcta depende del volumen.**

En la fase actual (producto recién arranca, <100 sesiones/día): **LLM externo por request es la opción correcta**. El costo de Claude Haiku es ~$0.002/turno (~$0.05 por sesión de 25 turnos). Con 100 sesiones/día son $5/día — completamente viable. La latencia promedio de Haiku es 800–1.200 ms/request, que con la arquitectura cliente-lado (la clasificación ocurre *después* de que el alumno habla, no bloqueando la respuesta de la IA) es perfectamente aceptable.

Si el volumen escala a >10.000 sesiones/mes, el trade-off cambia: considerar BETO fine-tuneado en Hugging Face Inference Endpoints (~$0.0001/request, <100 ms) o SageMaker serverless. La señal de migración es cuando el costo mensual del LLM supere el costo de mantenimiento de un endpoint dedicado (~$200-400/mes en AWS).

**Qué le falta a este servicio para producción real:**
- Autenticación (API key o JWT) en `/classify`
- Rate limiting por alumno
- Observabilidad: métricas de latencia P95, tasa de fallback a baseline, error rate (Prometheus + Grafana)
- Persistencia de clasificaciones: cada turno clasificado debe loggearse a base de datos para auditoría clínica y para alimentar el flywheel de datos
- Drift detection: monitorear distribución de fases predichas — si cambia significativamente, los guiones cambiaron o el modelo degradó
- Módulo de riesgo con escalamiento a revisor humano (no implementado aquí intencionalmente)
- Pruebas de carga: validar latencia bajo 50 sesiones simultáneas
- Secretos en AWS Secrets Manager / GCP Secret Manager (no en variables de entorno del contenedor)

---

*Este servicio es el primer ladrillo funcional, no el edificio completo. El criterio de diseño fue: que corra, que tenga contratos claros y que las decisiones de arquitectura sean defensibles cuando haya más datos.*
