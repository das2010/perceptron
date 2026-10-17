# El asistente guiado

Es la forma más simple de usar Perceptron: contás el problema con tus palabras y el asistente te
lleva paso a paso hasta un modelo validado. No hace falta saber de machine learning.

## Paso a paso

1. **Contá el problema.** En **Inicio**, escribí qué querés lograr, por ejemplo «Tengo una
   planilla de ventas y quiero estimar cuánto vamos a vender». Tocá **Empezar**.
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

- Tablas: elegir una categoría, estimar un número o descubrir una fórmula.
- Imágenes, textos y audios: elegir una categoría.

Para lo demás (pronósticos de series, casos raros, ubicar objetos en una foto, datos sin la
respuesta correcta) el asistente te explica por qué y te lleva al **wizard avanzado** o a
**Etiquetado**.

## Opciones avanzadas

Todas las herramientas técnicas (datos, diseño, entrenamiento, experimentos, modelos, monitoreo)
siguen en **Opciones avanzadas** dentro del proyecto. Ver [Tu primer modelo](primer-modelo.md)
para el recorrido manual.
