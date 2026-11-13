# El asistente guiado

Es la forma más simple de usar Perceptron: contás el problema con tus palabras y el asistente te
lleva paso a paso hasta un modelo validado. No hace falta saber de machine learning.

## Paso a paso

1. **Contá el problema.** En **Inicio**, escribí qué querés lograr, por ejemplo «Tengo una
   planilla de ventas y quiero estimar cuánto vamos a vender». Tocá **Empezar**.
   - **¿Ya tenés los datos?** Adjuntalos con el clip: una o varias planillas (CSV, Excel,
     Parquet), o un zip con fotos, grabaciones o textos separados en carpetas.
   - **Lo que te propone:** el asistente los lee y te dice qué puede aprender con ellos, por
     ejemplo «Predecir “se dio de baja”» o «Pronosticar “unidades” para cada producto».
   - **Varias planillas:** si tienen las mismas columnas, las junta. Si se relacionan por una
     columna, como clientes y sus compras, las une.
   - **Cómo elegir:** tocá el objetivo que te sirve, o contalo con tus palabras.
2. **Respondé las preguntas.** El asistente pregunta de a una cosa por vez, con ejemplos: qué
   datos tenés, qué querés obtener, qué error es peor, qué resultado te sirve. Podés contestar
   escribiendo o con el control que aparece debajo. En **Lo que entendí** podés corregir
   cualquier dato. Cuando está todo, revisalo y tocá **Sí, es esto**.
3. **Prepará los datos.** El asistente te muestra cómo tienen que quedar, con una tabla de ejemplo
   (o cómo armar las carpetas del ZIP), una plantilla para descargar y consejos. Subí el archivo.
4. **Revisión.** Perceptron revisa los datos y te dice qué encontró. Si algo impide seguir (por
   ejemplo, muy pocos ejemplos), te dice qué corregir.
5. **Entrenamiento.** Tocá **Preparar y entrenar**. Perceptron diseña el modelo, lo entrena y lo
   evalúa con ejemplos que nunca vio. Ves en qué etapa está y podés salir: te avisa al terminar.
6. **Resultado.** Te explica cuánto acierta, o cuánto se equivoca en promedio, y si cumple lo que
   pediste. Si encontró una fórmula exacta, podés copiarla para Excel o Python.
7. **Validá con casos nuevos.** Cargá casos que no estén en los datos que subiste, con su
   resultado real. El modelo predice sin verlo y después se compara. En tablas podés subir un CSV
   con varios casos a la vez. Con al menos 10 casos (y todas las categorías) queda listo para
   usar.
8. **Usalo.** Completá solo los datos que el modelo necesita, o subí la foto o el audio, y obtené
   la predicción.

## Qué casos resuelve

- **Planillas:** elegir entre respuestas posibles, estimar un número, descubrir una fórmula,
  pronosticar los próximos valores de algo que medís en el tiempo (ventas, consumo, desgaste) o
  detectar momentos raros en mediciones (aunque no tengas casos marcados).
- **Fotos:** decir de qué tipo es cada una, ubicar dónde están los objetos, marcar la zona exacta
  (por ejemplo, el área dañada) o leer un texto (una etiqueta, un número de serie).
- **Textos y grabaciones:** elegir entre respuestas posibles.

Si tu caso no entra tal cual, el asistente te propone una forma equivalente de plantearlo. Por
ejemplo, agrupar en rangos un número que querés estimar a partir de fotos.

## Opciones avanzadas

Todas las herramientas técnicas (datos, diseño, entrenamiento, experimentos, modelos, monitoreo)
siguen en **Opciones avanzadas** dentro del proyecto. Ver [Tu primer modelo](primer-modelo.md)
para el recorrido manual.
