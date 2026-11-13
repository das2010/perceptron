from docx import Document
from docx.shared import Inches,Pt,RGBColor
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from pathlib import Path
d=Document(); s=d.sections[0]
s.page_width=Inches(8.3); s.page_height=Inches(11.7)
s.top_margin=s.bottom_margin=Inches(.75); s.left_margin=s.right_margin=Inches(.85)
for name in ['Normal','Title','Heading 1','Heading 2']:
 st=d.styles[name]; st.font.name='Calibri'; st.font.color.rgb=RGBColor(0,0,0)
for st in d.styles:
 for b in list(st.element.iter(qn('w:pBdr'))): b.getparent().remove(b)
d.styles['Normal'].font.size=Pt(11)
d.styles['Normal'].paragraph_format.space_after=Pt(8)
d.styles['Normal'].paragraph_format.line_spacing=1.08
d.styles['Title'].font.size=Pt(23)
d.styles['Heading 1'].font.size=Pt(15)
d.core_properties.title='Fragmentos de fuentes sobre IA y reconversión del equipo'
d.core_properties.author=''
def p(t): d.add_paragraph(t)
def link(url):
 pa=d.add_paragraph(); hy=OxmlElement('w:hyperlink'); hy.set(qn('r:id'),pa.part.relate_to(url,'http://schemas.openxmlformats.org/officeDocument/2006/relationships/hyperlink',is_external=True))
 r=OxmlElement('w:r'); pr=OxmlElement('w:rPr'); co=OxmlElement('w:color');co.set(qn('w:val'),'174A70');pr.append(co);r.append(pr);t=OxmlElement('w:t');t.text='Consultar el texto original';r.append(t);hy.append(r);pa._p.append(hy)
def entry(title,date,location,quote,translation,meaning,limit,url):
 d.add_heading(title,1);p(date+' · '+location)
 pa=d.add_paragraph();pa.add_run('Cita textual en inglés\n').bold=True;pa.add_run('“'+quote+'”').italic=True
 p('Traducción propia: '+translation)
 p('Relevancia para el equipo: '+meaning)
 p('Alcance: '+limit);link(url)
d.add_heading('Fragmentos de fuentes sobre IA y reconversión del equipo',0)
p('Selección documental complementaria al diagnóstico y plan de reconversión\nConsulta realizada el 23 de octubre de 2026')
p('Las citas entre comillas reproducen fragmentos breves de las cinco fuentes principales del informe. Las traducciones y los comentarios se identifican por separado. Los enlaces permiten revisar el pasaje en su contexto original.')
entry('1 Stack Overflow Developer Survey 2026','6 de octubre de 2026','Principales resultados sobre IA',
'Coding assistants or coding agents are used by 66% of respondents',
'El 66% de los encuestados utiliza asistentes o agentes de programación.',
'La asistencia con IA ya forma parte del trabajo de una proporción importante de los participantes. La falta de exposición del equipo merece un plan explícito de actualización.',
'La categoría combina asistentes y agentes; no demuestra que el 66% domine flujos agénticos autónomos. La encuesta es voluntaria y global.',
'https://stackoverflow.co/company/press/archive/stack-overflow-annual-developer-survey-ai-trust-conditional')
entry('2 PwC Global AI Jobs Barometer 2026','15 de junio de 2026','Introducción',
'Skills needed for the most AI-exposed jobs are changing more than twice as fast as for the least AI-exposed jobs.',
'Las habilidades necesarias para los empleos más expuestos a IA están cambiando más del doble de rápido que las de los menos expuestos.',
'Si la dedicación a proyectos impide aprender, la brecha puede ampliarse. La actualización debe tener tiempo y presupuesto recurrentes.',
'Compara grupos de ocupaciones según su exposición a IA. No mide directamente las habilidades del equipo ni establece cuánto tiempo necesita para reconvertirse.',
'https://www.pwc.com/gx/en/1/services/ai/ai-jobs-barometer.html')
d.add_page_break()
entry('3 DORA State of AI assisted Software Development 2025','Edición 2025','Presentación del informe',
'AI’s primary role is as an amplifier, magnifying an organization’s existing strengths and weaknesses.',
'La función principal de la IA es amplificar las fortalezas y debilidades que ya existen en una organización.',
'Agregar herramientas a un flujo con requisitos ambiguos, pruebas insuficientes o revisión saturada puede aumentar esos problemas. La reconversión debe incluir prácticas de ingeniería y coordinación.',
'Es una conclusión organizacional del informe, no una garantía de retorno ni una tasa de mejora aplicable a cualquier proyecto.',
'https://dora.dev/research/2025/dora-report/')
entry('4 METR seguimiento del experimento de productividad','24 de febrero de 2026','Discusión inicial de los resultados',
'our data is only very weak evidence for the size of this increase.',
'Nuestros datos constituyen evidencia muy débil sobre la magnitud de este aumento.',
'Conviene medir el efecto en tareas propias antes de exigir un aumento fijo de productividad o comprometer ahorros comerciales.',
'El aumento se refiere a la aceleración con IA que los autores consideran probable frente a comienzos de 2025. El seguimiento presenta sesgos de selección y problemas de medición; la frase no significa que hayan demostrado ausencia de beneficios.',
'https://metr.org/blog/2026-02-24-uplift-update/')
entry('5 World Economic Forum Future of Jobs Report 2025','7 de enero de 2025','Resultados sobre brechas de habilidades',
'85% of employers surveyed plan to prioritize upskilling their workforce',
'El 85% de los empleadores encuestados planea priorizar la actualización de habilidades de su personal.',
'La formación del equipo existente es una respuesta empresarial ampliamente contemplada. En este caso exige replanificar carga y ofrecer práctica acompañada durante la jornada.',
'Describe planes empresariales para el horizonte 2025 a 2030, no programas ya ejecutados ni evidencia de que cualquier capacitación produzca resultados.',
'https://www.weforum.org/publications/the-future-of-jobs-report-2025/')
footer=s.footer.paragraphs[0];footer.alignment=2;footer.add_run('Página ').font.size=Pt(9)
fld=OxmlElement('w:fldSimple');fld.set(qn('w:instr'),'PAGE');footer._p.append(fld)
out=Path(r'C:\Users\aschneider\OneDrive - preteco.com\Escritorio\Archivos\I&D\Desarrollo\Fragmentos_de_fuentes_IA_2026-10-23.docx');d.save(out);print(out)
