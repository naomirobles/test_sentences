# Selección del conjunto de oraciones para la anotación manual de pictogramas

**Proyecto:** TT 2027-A036 — Traducción Automática de Texto a Pictogramas mediante Modelos Transformer
**Herramienta de anotación:** Pictograms Annotator (`anotador-pictogramas.onrender.com`)
**Cuaderno utilizado:** `seleccion_oraciones_test.ipynb` (Google Colab)
**Fecha:** octubre de 2026

---

## 1. Objetivo

Se construyó un conjunto de oraciones en español para que personas anotaran a mano su traducción a secuencias de pictogramas ARASAAC. Las anotaciones sirven para dos fines:

1. **Evaluar el modelo** con referencias hechas por personas, y no solo con las referencias generadas automáticamente.
2. **Validar el método de construcción del corpus.** El corpus de entrenamiento se obtuvo traduciendo con NLLB-200 las oraciones de Propicto-CommonVoice, que están en francés, y conservando sus secuencias de pictogramas originales. Al comparar la anotación humana con esa referencia automática se puede medir cuánto se aleja el corpus de lo que haría una persona.

Este conjunto se usa **solo para evaluación**; ninguna de sus oraciones se usa para entrenar.

---

## 2. Configuración final

| Parámetro | Valor |
|---|---|
| Oraciones únicas | **600** |
| Propicto (validación, español) | **300** (50%) |
| Common Voice en español | **300** (50%) |
| Reparto por longitud en cada fuente | 100 cortas · 100 medianas · 100 largas |
| Semilla aleatoria | `2027` |

En el cuaderno corresponde a `TOTAL = 600` y `FRAC_CV = 0.5`.

---

## 3. Fuentes de datos

### 3.1 Propicto (split de validación)

- **Dataset:** [`licaaron/propicto-qwen3-sft`](https://huggingface.co/datasets/licaaron/propicto-qwen3-sft), en Hugging Face.
- **Contenido:** oraciones de Propicto-CommonVoice traducidas del francés a 5 idiomas (alemán, inglés, español, ruso y turco), con el prompt en formato de chat de Qwen3. Cada fila tiene una dirección: `text2picto` (texto → pictogramas) o `picto2text` (pictogramas → texto).
- **Particiones:** *train* (~1.05 millones de filas) y *validation* (~5,270 filas).
- **Columnas usadas:** `text`, `lang`, `direction`, `row_id`.

**Por qué esta fuente:** cada oración ya tiene una secuencia de pictogramas de referencia, obtenida con el método de construcción del corpus. Esto permite comparar directamente la anotación humana con la referencia automática.

### 3.2 Common Voice en español

- **Origen:** repositorio público de Mozilla Common Voice en GitHub ([`common-voice/common-voice`](https://github.com/common-voice/common-voice)), carpeta `server/data/es/`.
- **Archivos usados:** `sentence-collector.txt` y `wiki.es.txt`.
- **Archivo excluido:** `singleword-benchmark.txt`, porque contiene palabras sueltas y no oraciones.
- **Licencia:** dominio público (CC0).

**Por qué esta fuente:** son oraciones escritas originalmente en español, no traducidas. Sirven para comprobar si el modelo funciona con español natural, sin los rasgos propios de una traducción automática. Además son del mismo estilo que Propicto-CommonVoice (frases cortas pensadas para leerse en voz alta), así que ambas fuentes son comparables.

**Limitación:** estas oraciones no tienen secuencia de pictogramas de referencia; su única referencia será la anotación humana.

---

## 4. Proceso

### 4.1 Extracción de la oración en español (Propicto)

1. Se cargó el split de **validación** y se conservaron las filas con `lang = es`, en **ambas direcciones**.
2. El campo `text` contiene el prompt completo con la plantilla de chat:

   ```
   <|im_start|>user <instrucción>: <contenido><|im_end|>
   <|im_start|>assistant <think> </think> <respuesta><|im_end|>
   ```

   La instrucción termina en el primer `:`; lo que sigue es el contenido.
3. Según la dirección, la oración en español está en un lugar distinto:

   | Dirección | Oración en español | Secuencia de pictogramas |
   |---|---|---|
   | `text2picto` | contenido del mensaje del usuario | respuesta del asistente |
   | `picto2text` | respuesta del asistente | contenido del mensaje del usuario |

4. Se guardó también la secuencia de referencia (`pictos_ref`) para compararla después con la anotación humana.

### 4.2 Eliminación de duplicados

Cada oración original parece aparecer dos veces en el dataset, una por dirección y no necesariamente en el mismo idioma. Esto se deduce de que `row_id` llega hasta ~527 mil y el dataset tiene el doble de filas. Si un mismo `row_id` aparecía en español en ambas direcciones, se conservó una sola vez, dando preferencia a `text2picto`.

### 4.3 Control de fuga de datos

**Decisión:** se descartó toda oración de validación cuyo `row_id` también apareciera en *train*.

**Motivo:** si una oración aparece en la prueba como texto → pictogramas, pero su versión pictogramas → texto (aunque sea en otro idioma) está en entrenamiento, el modelo ya vio esa secuencia de pictogramas. Evaluar con ella daría un resultado artificialmente bueno.

Para no descargar el texto completo de *train*, se leyó únicamente su columna `row_id` desde los archivos Parquet.

### 4.4 Filtros

Ambas fuentes pasaron por los mismos filtros.

**Decisión clave:** los filtros solo revisan el **texto en español**, nunca la calidad de la secuencia de pictogramas de referencia. Si se eliminaran las oraciones con referencias malas, se escondería justo el problema que la anotación busca medir.

| Filtro | Criterio | Motivo |
|---|---|---|
| Longitud | 4 a 15 palabras | Las muy cortas aportan poco; las muy largas cansan y bajan la calidad de la anotación. |
| Cifras | Se excluyen oraciones con dígitos | Los números rara vez tienen un pictograma claro. |
| Siglas | Se excluyen palabras de 2 o más mayúsculas seguidas (ONU, BBC…) | No tienen pictograma. |
| Símbolos | Se excluyen comillas, paréntesis, corchetes y `/ @ # & ;` | Suelen indicar citas, títulos o texto técnico. |
| Palabras poco comunes | Máximo 1 palabra con frecuencia Zipf < 3.0 en español (biblioteca `wordfreq`) | Detecta nombres propios, tecnicismos y errores de traducción, que no se pueden representar con pictogramas. |
| Mayúsculas internas | Máximo 1 palabra con mayúscula inicial después de la primera | Detecta nombres propios en Common Voice, donde se conserva el uso de mayúsculas. |

La frecuencia Zipf mide qué tan común es una palabra en una escala aproximada de 1 a 7. Por ejemplo, *casa* tiene 5.77 y *minicab* tiene 0.

### 4.5 Muestreo

1. Cada oración se clasificó por longitud:

   | Grupo | Palabras |
   |---|---|
   | Corta | 4–7 |
   | Mediana | 8–11 |
   | Larga | 12–15 |

2. De cada fuente se tomaron 300 oraciones, repartidas en partes iguales entre los tres grupos (100 por grupo). Si un grupo no tuviera suficientes oraciones, el faltante se pasaría a los demás grupos.
3. Se juntaron las dos fuentes y se mezclaron al azar para que el orden no revelara de dónde viene cada oración.
4. A cada oración se le asignó un identificador: `es-0001` … `es-0600`.
5. Todo el muestreo usa la semilla `2027`, así que repetir el proceso con los mismos datos produce la misma muestra.

---

## 5. Resultados

### 5.1 Oraciones disponibles

| Etapa | Common Voice | Propicto |
|---|---|---|
| Oraciones únicas en español | 1,075,523 | _completar_ |
| Descartadas por aparecer en *train* | — | _completar_ |
| Pasan los filtros | 631,991 | _completar_ |
| Seleccionadas | 300 | 300 |

> Los valores de Propicto los imprime la sección 4 del cuaderno. Copiarlos aquí.

### 5.2 Archivos generados

| Archivo | Contenido |
|---|---|
| `oraciones_test.csv` | Las 600 oraciones seleccionadas. |
| `candidatas_propicto_es.csv` | Todas las oraciones de Propicto que pasaron los filtros. |
| `candidatas_cv_es.csv` | Todas las oraciones de Common Voice que pasaron los filtros. |

Columnas de `oraciones_test.csv`:

| Columna | Descripción |
|---|---|
| `id` | Identificador de la oración (`es-0001`…). |
| `fuente` | `propicto` o `common_voice`. |
| `oracion` | Texto en español que se anota. |
| `longitud` | `corta`, `mediana` o `larga`. |
| `row_id` | Identificador original en Propicto (vacío en Common Voice). |
| `direction` | Dirección de la fila original en Propicto (vacío en Common Voice). |
| `pictos_ref` | Secuencia de pictogramas de referencia de Propicto (vacío en Common Voice). |
| `archivo` | Archivo de origen en Common Voice (vacío en Propicto). |

Los archivos de candidatas permiten reemplazar una oración de la muestra si en la revisión manual resulta inadecuada.

---

## 6. Limitaciones conocidas

- **Referencias con estructura del francés.** Las secuencias de Propicto siguen el orden y la gramática del francés (`passé`, `il`, `de`…), y varias son ruidosas. Se espera que difieran de las anotaciones humanas en español; medir esa diferencia es parte del objetivo.
- **Calidad de la traducción automática.** Algunas oraciones de Propicto traducidas con NLLB tienen errores. Se recomienda que el anotador tenga una opción para marcar una oración como **mal traducida**, en vez de forzar su anotación. Esto da además una medida de la calidad de la traducción.
- **El filtro de palabras raras no es perfecto.** Al permitir una palabra poco común, deja pasar casos como *"también utiliza el marco emberjs"*. Por eso la muestra final se revisó a mano antes de cargarla al anotador.
- **Dominio enciclopédico.** Buena parte de Common Voice (`wiki.es.txt`) viene de Wikipedia, así que hay oraciones más formales que el lenguaje cotidiano al que suelen orientarse los pictogramas.
- **Estructura del `row_id`.** La interpretación de que cada `row_id` corresponde a una oración original que aparece una vez por dirección se dedujo de los tamaños del dataset, no de una documentación oficial.

---

## 7. Reproducibilidad

Para regenerar exactamente el mismo conjunto:

1. Abrir `seleccion_oraciones_test.ipynb` en Google Colab.
2. En la celda de configuración: `TOTAL = 600`, `FRAC_CV = 0.5`, `SEED = 2027`.
3. Ejecutar todas las celdas.

Como las fuentes pueden cambiar con el tiempo, conviene registrar sus versiones exactas:

| Elemento | Versión |
|---|---|
| Commit del repositorio de Common Voice | _completar_ (`git -C /content/seleccion/common-voice rev-parse HEAD`) |
| Revisión del dataset `licaaron/propicto-qwen3-sft` | _completar_ (hash del commit en Hugging Face) |
| Versión de `wordfreq` | _completar_ (`pip show wordfreq`) |
