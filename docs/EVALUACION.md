# Evaluación del prototipo

## Lectura de los resultados

El modo **aprendizaje** obtuvo 81.59% de exactitud (421 aciertos de 516 casos) en test. Es una señal útil para el catálogo actual, pero no demuestra traducción abierta: las particiones comparten expresiones base y cambian las plantillas de consulta en español. La exactitud de entrenamiento (95.22%) es mayor que la de validación (71.12%), lo que indica una brecha de generalización.

El modo **conversaciones** llegó a 100% en sus 42 ejemplos de entrenamiento, distribuidos en 29 clases. No hay validación ni test. Ese 100% describe ajuste —probablemente memorización— y no debe presentarse como rendimiento ante conversaciones nuevas.

Los umbrales de aceptación (`0.60` de puntuación y `0.15` de margen) son heurísticos. La salida softmax no está calibrada y no representa la probabilidad de que una expresión sea lingüísticamente correcta.

## Mejoras prioritarias

1. **Gobernanza comunitaria.** Definir desde el inicio qué comunidades, variante y ortografía se representan; incorporar hablantes remunerados como autores, revisores y responsables de decisiones.
2. **Más conversación real.** Reunir ejemplos consentidos y anonimizados por intención, con diversidad de edad, localidad, grafías y formas naturales de preguntar. El primer objetivo debería ser al menos 20–50 ejemplos revisados por clase antes de comparar arquitecturas.
3. **Evaluación independiente.** Crear un test escrito por personas distintas de quienes construyen el entrenamiento, sin paráfrasis casi idénticas. Reportar macro-F1, precisión y cobertura por clase, no solo exactitud global.
4. **Clase fuera de dominio.** Añadir entradas que el bot debe rechazar, faltas ortográficas y variantes cercanas. Medir tasa de falsas aceptaciones y calibrar umbral/margen con validación.
5. **Revisar los 95 errores.** Agruparlos por confusión, longitud, plantilla y fuente; fusionar intenciones indistinguibles o aumentar ejemplos donde dos clases se solapan.
6. **Separar variantes.** No mezclar ortografías o variantes como si fueran equivalentes. Mostrar procedencia y variante junto a cada respuesta, y permitir que la persona usuaria elija.
7. **Línea base de recuperación.** Comparar la red con búsqueda difusa o embeddings sobre un glosario. En catálogos pequeños, una solución recuperativa puede ser más explicable y sencilla de corregir.
8. **Pruebas con usuarios.** Medir utilidad educativa, comprensión, respeto cultural y facilidad para reportar errores. Una métrica automática no sustituye la evaluación de hablantes.

## Próximos experimentos recomendados

- Validación cruzada estratificada por intención, manteniendo grupos de expresiones base juntos.
- Matriz de confusión y macro-F1 por fuente y longitud de entrada.
- Calibración con temperature scaling o, como mínimo, curvas de cobertura frente a precisión.
- Registro opcional y consentido de consultas rechazadas, con eliminación de datos personales.
- Respuestas con fuente, variante, ortografía y estado de revisión visibles en la interfaz.
