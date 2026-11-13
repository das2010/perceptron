from docx import Document
from docx.shared import Inches, Pt, RGBColor
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from pathlib import Path

OUT=Path(r'C:\Users\aschneider\OneDrive - preteco.com\Escritorio\Archivos\I&D\Desarrollo\Diagnostico_y_plan_de_reconversion_IA_2026-10-23.docx')
d=Document()
s=d.sections[0]
s.page_height=Inches(11.7); s.page_width=Inches(8.3)
s.top_margin=s.bottom_margin=Inches(.7)
s.left_margin=s.right_margin=Inches(.8)
for name in ['Normal','Title','Subtitle','Heading 1','Heading 2','Heading 3']:
 st=d.styles[name]; st.font.name='Calibri'; st.font.color.rgb=RGBColor(0,0,0)
d.styles['Normal'].font.size=Pt(11)
d.styles['Normal'].paragraph_format.space_after=Pt(7)
d.styles['Normal'].paragraph_format.line_spacing=1.08
d.styles['Title'].font.size=Pt(25)
d.styles['Heading 1'].font.size=Pt(18)
d.styles['Heading 2'].font.size=Pt(13)
d.styles['Heading 1'].paragraph_format.space_after=Pt(12)
for st in d.styles:
 for border in list(st.element.iter(qn('w:pBdr'))):
  border.getparent().remove(border)
d.core_properties.title='Reconversión del equipo de desarrollo ante la IA'
d.core_properties.subject='Evidencia de mercado y plan de acción'
d.core_properties.author=''
foot=s.footer.paragraphs[0]; foot.alignment=2
run=foot.add_run('Página '); run.font.size=Pt(9)
fld=OxmlElement('w:fldSimple'); fld.set(qn('w:instr'),'PAGE'); foot._p.append(fld)
def p(t): d.add_paragraph(t)
def h(t): d.add_heading(t,2)
def page(t): d.add_page_break(); d.add_heading(t,1)
def table(headers,rows,widths):
 t=d.add_table(rows=1, cols=len(headers)); t.autofit=False
 for c,w in zip(t.columns,widths): c.width=Inches(w)
 for c,txt in zip(t.rows[0].cells,headers): c.text=txt
 for row in rows:
  for c,txt in zip(t.add_row().cells,row): c.text=txt
 for i,row in enumerate(t.rows):
  pr=row._tr.get_or_add_trPr(); pr.append(OxmlElement('w:cantSplit'))
  if i==0: pr.append(OxmlElement('w:tblHeader'))
  for c,w in zip(row.cells,widths):
   c.width=Inches(w); c.vertical_alignment=1
   tcpr=c._tc.get_or_add_tcPr(); shade=OxmlElement('w:shd'); shade.set(qn('w:fill'),'E7EDF3' if i==0 else ('F7F7F7' if i%2==0 else 'FFFFFF')); tcpr.append(shade)
   borders=OxmlElement('w:tcBorders')
   for side in ['top','left','bottom','right','insideH','insideV']:
    el=OxmlElement('w:'+side); el.set(qn('w:val'),'single'); el.set(qn('w:sz'),'4'); el.set(qn('w:color'),'D9D9D9'); borders.append(el)
   tcpr.append(borders)
   mar=OxmlElement('w:tcMar')
   for side in ['top','left','bottom','right']:
    el=OxmlElement('w:'+side); el.set(qn('w:w'),'100'); el.set(qn('w:type'),'dxa'); mar.append(el)
   tcpr.append(mar)
   for pa in c.paragraphs:
    pa.paragraph_format.space_after=Pt(3); pa.paragraph_format.line_spacing=1.0
    for r in pa.runs: r.font.size=Pt(10); r.bold=(i==0)
 d.add_paragraph().paragraph_format.space_after=Pt(0)
def link(label,url):
 pa=d.add_paragraph(); hy=OxmlElement('w:hyperlink')
 hy.set(qn('r:id'),pa.part.relate_to(url,'http://schemas.openxmlformats.org/officeDocument/2006/relationships/hyperlink',is_external=True))
 r=OxmlElement('w:r'); props=OxmlElement('w:rPr'); color=OxmlElement('w:color'); color.set(qn('w:val'),'174A70'); props.append(color); r.append(props)
 tx=OxmlElement('w:t'); tx.text=label; r.append(tx); hy.append(r); pa._p.append(hy)

d.add_heading('Reconversión del equipo de desarrollo ante la IA',0)
p('Diagnóstico de mercado y estrategia de actualización profesional')
p('Documento para Dirección y responsables de Desarrollo e I&D\nFecha de corte de la investigación 23 de octubre de 2026')
h('Conclusión ejecutiva')
p('El equipo presenta una brecha de capacidades para trabajar con IA y agentes, según la situación planteada. Revertirla requiere una decisión de capacidad y de organización del trabajo, además de capacitación. La recomendación es iniciar un programa de 90 días con tiempo protegido, práctica sobre proyectos reales y evaluación de resultados antes de extenderlo al conjunto de la operación.')
p('La evidencia respalda la urgencia de actualizar habilidades, pero no demuestra que los profesionales fullstack, diseñadores, analistas funcionales, líderes técnicos y scrum masters hayan perdido todo su valor de mercado. La experiencia en negocio, arquitectura, calidad y coordinación sigue siendo una base para incorporar IA. La brecha debe evaluarse por persona y por tarea, sin equiparar falta de exposición con incapacidad.')
h('Qué cambió en el mercado')
p('En la encuesta de Stack Overflow publicada el 6 de octubre de 2026, el 66% de quienes respondieron utiliza asistentes o agentes de programación y el 26,2% utiliza flujos automatizados con agentes. El uso de asistentes ya es extendido; la automatización agéntica todavía no es una competencia universal. [1]')
p('PwC informa en junio de 2026 que las habilidades requeridas en los empleos más expuestos a IA cambian más del doble de rápido que en los menos expuestos. Esto vuelve costoso mantener al equipo permanentemente ocupado sin espacio para actualizarse. [2]')
h('Decisiones propuestas')
p('Asignar un patrocinador de Dirección y un responsable operativo. Reservar inicialmente entre 10% y 20% de la jornada de los participantes durante 12 semanas, replanificando compromisos. Seleccionar dos pilotos de riesgo acotado, habilitar herramientas aprobadas y evaluar al día 90 si mejoran tiempo o costo sin deteriorar calidad.')
p('Los porcentajes de dedicación, plazos y metas de este documento son propuestas de gestión, no benchmarks de mercado ni compromisos de ahorro. Los riesgos se presentan como escenarios a validar con datos internos.')

page('1 Evidencia reciente para interpretar la brecha')
h('Adopción y demanda de habilidades')
p('Stack Overflow 2026 reúne más de 30.000 participantes de 169 países. Además del uso de asistentes y agentes citado, sólo el 24% señala que su organización publica herramientas aprobadas o políticas de IA. La adopción individual puede adelantarse a la capacidad de gobernarla. La encuesta describe a sus participantes y no constituye un censo representativo de empresas argentinas. [1]')
p('El AI Jobs Barometer 2026 de PwC analiza más de mil millones de avisos de empleo de seis continentes. Encuentra cambios de habilidades acelerados y mayor valoración de juicio y liderazgo. Reporta un crecimiento de productividad 40% superior en las compañías más expuestas a IA frente a las menos expuestas; no significa que cada desarrollador produzca 40% más ni prueba por sí solo causalidad. [2]')
h('La productividad requiere comprobarse')
p('DORA 2025 concluye que la IA amplifica fortalezas y debilidades organizacionales. La implicación para este equipo es que comprar licencias sin mejorar especificaciones, pruebas, documentación y flujo de entrega puede aumentar trabajo generado sin aumentar valor entregado. Es una orientación sobre el sistema de trabajo, no un porcentaje de ahorro aplicable a la empresa. [3]')
p('METR comunicó en febrero de 2026 que su estudio inicial de 2025 observó tareas 19% más lentas con IA en desarrolladores experimentados de proyectos abiertos. En el estudio posterior detectó sesgos de selección y dificultades de medición; considera probable una mejora respecto de 2025, pero no puede estimar con solidez su magnitud. No corresponde usar el resultado inicial como prueba de que la IA actual siempre ralentiza, ni el seguimiento como garantía de aceleración. [4]')
h('Transformación del empleo y reconversión')
p('El Future of Jobs Report 2025 del World Economic Forum proyecta que 39% de las habilidades cambiará o quedará desactualizado entre 2025 y 2030; 59 de cada 100 trabajadores requerirían formación y 85% de los empleadores consultados prioriza actualizar capacidades. Son expectativas empresariales, no resultados ya observados. [5]')
p('El mismo informe ubica a desarrolladores de software entre los roles de crecimiento y a diseñadores gráficos cerca de los diez de mayor caída esperada. Esto sugiere revisar tareas y propuesta de valor, no asumir la desaparición uniforme de las profesiones. El horizonte es 2030 y no predice despidos dentro de esta empresa. [5]')
h('Alcance de la comparación')
p('Estos informes son señales globales. Para estimar competitividad local deben complementarse con precios ganados y perdidos, requerimientos de clientes, avisos comparables de Argentina y mercados de exportación, tiempos de entrega y márgenes propios. No permiten cuantificar hoy la pérdida comercial atribuible a la brecha.')

page('2 Situación actual y consecuencias de no actuar')
h('Diagnóstico organizacional')
p('El punto de partida informado es un equipo absorbido por proyectos de tecnología y metodología tradicionales, sin oportunidades suficientes de preparación. La hipótesis principal es una brecha acumulada por falta de capacidad disponible para aprender. Si toda hora se compromete a entregas, el equipo seguirá sin poder adquirir habilidades, aun cuando exista voluntad individual.')
p('Programación agéntica significa definir objetivos y restricciones para un sistema que puede planificar pasos, consultar el repositorio, modificar archivos y ejecutar herramientas o pruebas bajo supervisión. Exige dividir tareas, aportar contexto verificable, limitar permisos y revisar resultados. No equivale únicamente a conversar con un chatbot o aceptar autocompletado.')
table(['Horizonte','Consecuencia probable','Señal interna a vigilar'],[
('Corto plazo\n0 a 3 meses','Más esfuerzo relativo en tareas repetibles y prototipos frente a equipos que ya dominan IA.','Demoras en propuestas y entregas; costo por cambio aceptado.'),
('Corto plazo\n0 a 3 meses','Uso informal de herramientas o resistencia por falta de guía y tiempo.','Herramientas no declaradas; retrabajo; formación cancelada.'),
('Mediano plazo\n3 a 12 meses','Presión sobre precios y margen si clientes esperan ciclos más breves.','Descuentos, negocios perdidos y objeciones documentadas.'),
('Mediano plazo\n3 a 12 meses','Dependencia de pocos expertos, dificultad de reasignación y riesgo de rotación.','Concentración de conocimiento; vacantes internas sin cubrir.'),
('Mediano plazo\n3 a 12 meses','Menor capacidad para ofrecer modernización y automatización a clientes.','Oportunidades descartadas por falta de competencias.')
],[1.1,3.45,2.15])
p('Son riesgos razonados para este contexto, no hechos comprobados ni probabilidades calculadas. La intensidad depende de contratos, complejidad del legado, requisitos de calidad y ritmo de adopción de clientes y competidores.')
h('Qué se conserva y qué debe evolucionar')
p('El conocimiento de sistemas existentes, clientes y dominio reduce errores al dirigir y verificar agentes. Conviene preservarlo. Lo que debe evolucionar es el modo de producir y validar entregables: pasar de ejecutar manualmente cada paso a decidir qué delegar, preparar contexto, supervisar y demostrar que el resultado cumple la necesidad.')
p('Una adopción apresurada también tiene costos: más defectos, revisión saturada, dependencia de proveedores y pérdida de comprensión. La respuesta adecuada es una transición controlada, no exigir velocidad inmediata ni reemplazar personas basándose sólo en su rol.')

page('3 Estrategias para revertir la situación')
h('Liberar capacidad y cambiar incentivos')
p('Dirección debe convertir la formación en trabajo planificado. Reservar bloques semanales, reducir alcance o renegociar fechas cuando sea necesario y registrar las cancelaciones de esos bloques. No pedir que la actualización dependa del tiempo personal. Medir aprendizaje aplicado y resultados del equipo, no cantidad de prompts ni líneas generadas.')
h('Combinar formación interna con acompañamiento experto')
table(['Alternativa','Ventaja','Límite y uso recomendado'],[
('Formación interna','Preserva dominio y desarrolla autonomía.','Requiere tiempo y mentores. Base de la estrategia.'),
('Mentoría externa acotada','Acelera la instalación de prácticas y revisión.','Exigir transferencia y trabajo sobre casos propios.'),
('Incorporación selectiva','Cubre capacidades avanzadas inexistentes.','Usar tras diagnosticar brechas; no sustituye aprendizaje del resto.'),
('Tercerizar pilotos','Permite probar oportunidades con rapidez.','Riesgo de dependencia. Exigir documentación y participación interna.')
],[1.45,2.25,3.0])
p('Se recomienda formación interna acompañada por un referente experimentado, propio o externo. Evitar un programa centrado sólo en cursos: cada módulo debe terminar en un entregable revisado de un proyecto real o un entorno de práctica representativo.')
h('Preparar el entorno para trabajar con agentes')
p('Antes del piloto, disponer de repositorios con instrucciones de instalación, convenciones, decisiones de arquitectura y pruebas ejecutables. Establecer herramientas autorizadas, cuentas corporativas, límites de gasto y tratamiento de información de clientes. Usar ramas aisladas, permisos mínimos y aprobación humana para integración y despliegue.')
p('La definición de terminado debe incluir revisión humana responsable, pruebas funcionales pertinentes, chequeos de seguridad y evidencia de aceptación. Un agente que modifica código no debe ser la única autoridad que certifica que ese código es correcto. Registrar herramienta, versión o modelo cuando sea posible y contexto relevante para interpretar resultados.')
h('Elegir trabajo apto para aprender')
p('Comenzar con mantenimiento acotado, pruebas, documentación, prototipos y cambios reversibles. Postergar autonomía sobre producción, migraciones críticas, credenciales y decisiones irreversibles. Comparar al menos dos opciones de herramienta sobre el mismo conjunto de tareas antes de estandarizar, considerando integración, controles, costo y calidad.')
h('Sostener la transición')
p('Crear una clínica semanal de casos, un repositorio de ejemplos y revisión entre pares. Rotar referentes para evitar dependencia de una sola persona. Mantener ejercicios de explicación y resolución sin asistencia para comprobar comprensión y conservar criterio técnico.')

page('4 Competencias y práctica por perfil')
p('Objetivo común: poder encargar una tarea acotada a IA, aportar contexto suficiente, detectar errores y justificar la aceptación del resultado. No todos los perfiles necesitan programar agentes; cada uno debe dominar la colaboración con IA en su trabajo.')
table(['Perfil','Competencias prioritarias','Evidencia práctica de dominio'],[
('Fullstack','Descomposición de tareas; contexto de repositorio; agentes en IDE o terminal; revisión de cambios; pruebas y depuración.','Completar un cambio con pruebas, explicar el diff y corregir un fallo introducido deliberadamente.'),
('Diseño gráfico','Ideación y variantes con IA; dirección visual; consistencia de marca; accesibilidad; trazabilidad de activos y permisos de uso.','Entregar variantes de una pieza con criterios de selección, control de marca y revisión humana de calidad.'),
('Análisis funcional','Especificaciones verificables; reglas de negocio; ejemplos y criterios de aceptación; validación de resúmenes y trazabilidad.','Convertir un requerimiento ambiguo en casos de aceptación y detectar requisitos inventados por IA.'),
('Liderazgo técnico','Arquitectura; permisos de agentes; evaluación de herramientas; calidad del contexto; revisión y medición del costo total.','Diseñar un flujo seguro de tarea a integración, con pruebas, responsables y mecanismo de reversión.'),
('Scrum masters','Gestión del flujo humano y agentes; protección de aprendizaje; límites de trabajo en curso; métricas y facilitación del cambio.','Replanificar un sprint con formación, medir esperas y retrabajo y conducir una retrospectiva del piloto.')
],[1.15,2.75,2.8])
h('Secuencia formativa sugerida')
p('Semanas 1 y 2: fundamentos compartidos, límites de los modelos, información permitida y evaluación inicial. Semanas 3 a 6: prácticas por rol con tutoría y entregables pequeños. Semanas 7 a 10: flujo integrado desde necesidad y diseño hasta pruebas y revisión. Semanas 11 y 12: evaluación práctica y transferencia a un segundo caso.')
h('Rúbrica de evaluación')
p('Nivel 0: requiere acompañamiento en fundamentos. Nivel 1: utiliza asistencia en tareas acotadas y verifica resultados. Nivel 2: ejecuta un flujo supervisado reproducible y explica decisiones. Nivel 3: diseña, evalúa y enseña flujos a otros. El nivel se asigna con evidencia, no por antigüedad, asistencia a cursos o autoevaluación.')
p('Meta inicial propuesta: todos los participantes del piloto alcanzan nivel 1; al menos una persona por disciplina representada alcanza nivel 2. Quienes no lo logren reciben más práctica y acompañamiento. Una evaluación aislada no debe convertirse en una decisión automática de desvinculación.')

page('5 Plan de ejecución y criterios de decisión')
table(['Etapa','Acciones y entregables','Responsable'],[
('Días 1 a 15','Inventario de tareas y habilidades; línea base; dos pilotos; herramientas y tiempo reservado.','Dirección y líder técnico'),
('Días 16 a 30','Formación común; entorno de práctica; ejercicios por rol; primera revisión de capacidades.','Referente IA y mentores'),
('Días 31 a 60','Piloto de mantenimiento y piloto de necesidad a prototipo validado; registro de tiempo, calidad y costo.','Equipo multidisciplinario'),
('Días 61 a 90','Repetición en tareas comparables; evaluación práctica; informe de resultados y decisión de expansión.','Líder técnico y responsable de entrega'),
('Meses 4 a 6','Extensión gradual a otros proyectos; actualización de estimaciones y oferta comercial con evidencia propia.','Dirección y responsables de área')
],[1.1,4.1,1.5])
h('Cómo medir sin confundir actividad con valor')
p('Tomar una línea base de dos semanas y comparar tareas similares en complejidad, repositorio y experiencia. Cuando sea viable, alternar tareas con y sin asistencia; no repetir exactamente la misma tarea porque el aprendizaje sesga el resultado. Registrar cambios de alcance y herramientas. Con muestras pequeñas, informar resultados descriptivos sin presentar conclusiones causales generales.')
p('Medir tiempo desde inicio hasta aceptación, horas humanas totales incluyendo revisión y corrección, defectos posteriores, retrabajo, costo de herramientas por entrega aceptada y capacidad práctica alcanzada. Separar tiempo transcurrido de esfuerzo humano cuando los agentes ejecuten en paralelo. No comparar puntos de historia entre equipos.')
h('Puertas para ampliar o corregir')
p('Umbral propuesto al día 90: al menos 15% de reducción de tiempo o costo por entrega comparable, calidad al menos equivalente y ausencia de incidentes críticos atribuibles al flujo. Buscar el resultado en dos ciclos consecutivos y no en una demostración única. Son metas internas a ajustar tras la línea base, no resultados garantizados por los informes.')
p('Si aumenta el retrabajo o aparecen fallas de control, reducir la autonomía, mejorar contexto y pruebas y repetir el piloto. Si no hay beneficio neto, conservar sólo los casos útiles y revisar la selección de tareas antes de comprar más licencias. Si hay beneficio sostenido, ampliar por cohortes y mantener capacitación recurrente.')
h('Presupuesto y capacidad')
p('Presupuestar horas de formación y mentoría, licencias o consumo, preparación de repositorios y evaluación. Como ejemplo de capacidad, 10 personas × 40 horas semanales × 15% × 12 semanas requieren reservar 720 horas; no son 720 horas adicionales. El costo se calcula con la tarifa interna y la replanificación acordada.')
p('Beneficio neto = valor de capacidad efectivamente reutilizada y retrabajo evitado menos costos incrementales. Las horas liberadas no son ahorro de caja automáticamente. Revisar mensualmente avance, compromisos comerciales y retención de talento con Dirección.')

page('6 Informes y fuentes consultadas')
p('Fuentes primarias consultadas al 23 de octubre de 2026. Las referencias numeradas vinculan datos y conclusiones externas con su origen. Las propuestas operativas y los escenarios son análisis para la situación planteada.')
h('1 Stack Overflow Developer Survey 2026')
p('Publicado el 6 de octubre de 2026. Más de 30.000 participantes de 169 países. Útil para contrastar adopción y gobernanza; respuestas voluntarias y autorreportadas. No interpretar sus porcentajes como prevalencia exacta en todo el mercado.')
link('Resultados oficiales y datos de la encuesta','https://survey.stackoverflow.co/2026/ai/data')
link('Comunicado con fecha y principales resultados','https://stackoverflow.co/company/press/archive/stack-overflow-annual-developer-survey-ai-trust-conditional')
h('2 PwC Global AI Jobs Barometer 2026')
p('Publicado el 15 de junio de 2026. Análisis de avisos de empleo y datos empresariales; útil para demanda de habilidades. Las asociaciones entre exposición y productividad no equivalen a un experimento causal ni a una mejora individual esperable.')
link('Informe y metodología del AI Jobs Barometer 2026','https://www.pwc.com/gx/en/1/services/ai/ai-jobs-barometer.html')
link('Informe completo en PDF','https://www.pwc.com/gx/en/issues/artificial-intelligence/job-barometer/2026/2026-global-ai-jobs-barometer-full-report.pdf')
h('3 DORA State of AI assisted Software Development 2025')
p('Edición 2025 de Google Cloud y socios de investigación. Útil para entender capacidades organizacionales; no trasladar asociaciones del estudio a promesas de retorno del equipo. La página oficial ofrece versión abreviada en español para América Latina.')
link('Informe completo y versiones en español','https://dora.dev/research/2025/dora-report/')
h('4 METR actualización del estudio de productividad')
p('Publicado el 24 de febrero de 2026. Seguimiento experimental en desarrolladores de código abierto. Explicita problemas de selección y medición; lectura necesaria para evitar usar resultados de 2025 como si describieran herramientas actuales.')
link('We are Changing our Developer Productivity Experiment Design','https://metr.org/blog/2026-02-24-uplift-update/')
h('5 World Economic Forum Future of Jobs Report 2025')
p('Publicado en enero de 2025; proyecciones empresariales para 2025 a 2030. Sirve para dimensionar la reconversión y diferencias entre ocupaciones. No es una medición de pérdidas actuales de empleo ni una previsión específica para Argentina.')
link('Informe y resumen de resultados','https://www.weforum.org/publications/the-future-of-jobs-report-2025/')
link('Detalle sobre evolución de ocupaciones','https://www.weforum.org/publications/the-future-of-jobs-report-2025/2-jobs-outlook')

d.save(OUT)
print(OUT)
