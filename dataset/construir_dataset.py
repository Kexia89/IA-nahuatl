from pathlib import Path
import csv
import hashlib
import json
import re
import unicodedata
from collections import Counter

BASE = Path(__file__).resolve().parent
HASH_CORPUS = 'ed8e85f560e78c43385f91ac99d08dee071183749bc1f1aa1b88f1b3a2530878'
if hashlib.sha256((BASE/'fuentes/axolotl.csv').read_bytes()).hexdigest()!=HASH_CORPUS:
    raise ValueError('La versión del corpus cambió. Revisar la selección editorial antes de reconstruir.')

def guardar(nombre, obj):
    destino = BASE / nombre
    destino.parent.mkdir(parents=True, exist_ok=True)
    destino.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')

def normalizar(s):
    return ' '.join(unicodedata.normalize('NFC', s).casefold().split())

def clave(s):
    return ''.join(c for c in normalizar(s) if c.isalnum() or c.isspace()).strip()

def limpiar(s):
    return ' '.join(unicodedata.normalize('NFC', s).split()).strip('“”" .,:;')

FUENTES = {
    'inali': {
        'titulo': 'Prontuario de frases de cortesía: náhuatl de la Huasteca',
        'institucion': 'Instituto Nacional de Lenguas Indígenas',
        'voz_acreditada': 'Maximina Félix Allende',
        'url': 'https://site.inali.gob.mx/Micrositios/Prontuarios/nahuatlV.html',
        'variante': 'Huasteca; localidad y código ISO no especificados en este prontuario',
        'ortografia': 'Grafías del prontuario (cualli, tlazohcamati, etc.)',
        'licencia': 'No se identificó una licencia de datos. Solo expresiones breves de uso común; no se redistribuye el folleto.'
    },
    'uv': {
        'titulo': 'Alfabeto Náhuatl, tema 3: Conversaciones',
        'institucion': 'Universidad Veracruzana',
        'url': 'https://www.uv.mx/celulaode/alfabeto-nahuatl/tema3.html',
        'variante': 'Huasteca; localidad y código ISO no especificados en la lección',
        'ortografia': 'Grafías de la sección Huasteca (k, w, h); se conservan diferencias internas de la fuente',
        'licencia': 'No se identificó una licencia de datos. Se utilizan expresiones breves de uso común, no se reproduce la lección completa.',
        'nota': 'Se excluyó la segunda versión de ¿A dónde vas?: el HTML la rotula Huasteca, pero sus formas coinciden con el bloque de Zongolica. No se corrigió por conjetura.'
    },
    'axolotl': {
        'titulo': 'Axolotl, distribuido en Py-Elotl',
        'institucion': 'GIL-UNAM / Comunidad Elotl',
        'url': 'https://github.com/ElotlMX/py-elotl/blob/master/elotl/corpora/axolotl.csv',
        'corpus': 'https://axolotl-corpus.mx/',
        'cita': 'Gutierrez-Vasques, X., Sierra, G., y Pompa, I. H. (2016). Axolotl: a Web Accessible Parallel Corpus for Spanish-Nahuatl. LREC 2016.',
        'articulo': 'https://aclanthology.org/L16-1666/',
        'variante': 'Western Huasteca Nahuatl [nhw], según la columna de metadatos del corpus',
        'documento': 'Lo que relatan de antes (cuentos tének y nahuas de la huasteca)',
        'ortografia': 'Original de cada registro; no se regularizan préstamos ni alternancias gráficas',
        'licencia': 'El repositorio Py-Elotl publica MPL-2.0. Se conserva su licencia. No se afirma que esa licencia sustituya derechos de cada obra de origen.',
        'sha256_csv_consultado': hashlib.sha256((BASE/'fuentes/axolotl.csv').read_bytes()).hexdigest(),
        'archivo_original_incluido_en_zip': False,
    }
}

MANUALES = {
 'inali': [
  ('bienvenidos','Mah cualli xihualacan'),
  ('buenos días','Mah cualli tonalli'),
  ('buenas tardes','Mah cualli teotlac'),
  ('buenas noches','Mah cualli yohualli'),
  ('que disfruten de esta reunión','Mah xiyopalquican ini misentiliztli'),
  ('pase usted','Mah xipano'),
  ('con permiso; disculpe usted','Mah xinechtlahpopolhui'),
  ('muchas gracias','Tlazohcamati miyac'),
  ('buen provecho','Mah xitlacuacan cualli'),
  ('muchas gracias por su visita','Tlazohcamati ipampa otitechitaco'),
  ('vuelva pronto','Icihuaca ximocuepa'),
  ('que te vaya bien; buen camino','Mah cualli ohtli'),
  ('adiós; nos vemos','Timoitase'),
 ],
 'uv': [
  ('hola','Piyali'), ('buenos días','Tonalti'),
  ('buenas tardes','Tiotlakilti'), ('buenas noches','Tlapoyawalti'),
  ('nos veremos luego','Timoitasehyok'), ('nos encontraremos luego','Timopantisehyok'),
  ('¿cómo te llamas?','Tlen motoka?'),
  ('yo me llamo Juana','Nah notoka Juana'),
  ('está bien','Kualtitok'), ('sí','Kena'), ('nos vemos','Timoittaseh'),
  ('¿a dónde vas?','Kanke tiyah?'), ('voy al tianguis','Niyah tiankis'),
  ('voy a mi milpa','Niyah nomilla'), ('que te vaya bien','Kualli xiyah'),
  ('¿cómo estás?','Kenihkatzah tiitztok?'),
  ('estoy bien','Kualli niitztok'), ('también estoy bien','Nohkia kualli niitztok'),
  ('¿aquí estudias?','Nikan timomachtia?'),
  ('sí, aquí estudio','Kena, nikan nimomachtia'),
 ]
}

SELECCION = [
 9,224,226,248,276,310,333,547,638,813,855,935,1114,1151,1197,1253,
 1366,1392,1418,1645,1787,1844,2072,2155,2214,2258,2304,2584,2789,
 2862,2878,2961,2971,3077,3167,3318,3334,3371,3641,3811,3885,3891,
 3964,4274,4437,4734,4841,4953,5203,5209,5235,5333,5380,5614,5650,
 5657,5678,5809,5866,6200,6240,6439,6452,6463,6642,6674,7010,
 7397,7435,7486,7508,7611,7617,7694,7797,7945,8046,8193,8403,8574,
 8602,8778,8838,8947,8960,9199,9254,9267,9437,9491,9548,9774,9802,
 9953,9954,10130,10262,10287,10438,10475,10888,10984,11103,11247,
 11413,11455,11586,11727,11982,12007,12008,12417,12460,12664,12779,
 12957,13006,13008,13043,13061,13068,13079,13325,13579,13587,13610,
 13701,13711,13757,13764,13802,13828,13911,13982,14124,14601,14892,
 14995,15089,15144,15220,15250,15353,15565,15605,15759,
]

frases=[]
for fuente, pares in MANUALES.items():
    for numero,(es,nah) in enumerate(pares,1):
        frases.append(dict(id=f'{fuente}_{numero:03}', es=es, nah=nah, fuente=fuente,
                           estado='expresion_documentada', original_es=es, original_nah=nah,
                           cambios='Extracción de expresiones breves; mayúscula inicial; sin cambio de grafemas. Glosas españolas de apoyo.'))

registros=list(csv.reader((BASE/'fuentes/axolotl.csv').open(encoding='utf-8',newline='')))
for num in SELECCION:
    fila=registros[num-1]
    assert fila[4]=='nhw', (num,fila)
    es,nah=limpiar(fila[0]),limpiar(fila[1])
    frases.append(dict(id=f'ax_{num:05}',es=es,nah=nah,fuente='axolotl',registro_csv=num,
                       estado='seleccion_editorial_de_corpus_no_validada_por_hablante',
                       original_es=fila[0],original_nah=fila[1],variante_iso=fila[4],
                       documento=fila[3],cambios='NFC, espacios y retiro de puntuación periférica de cita; no traducción automática.'))

unicas={}
for f in frases:
    k=(clave(f['es']),clave(f['nah']))
    if k in unicas:
        unicas[k].setdefault('registros_equivalentes',[]).append(f['id'])
    else:
        unicas[k]=f
frases=list(unicas.values())
por_id={f['id']:f for f in frases}

grupos_es={}
grupos_nah={}
for f in frases:
    grupos_es.setdefault(clave(f['es']),[]).append(f)
    grupos_nah.setdefault(clave(f['nah']),[]).append(f)

aprendizaje=[]
procedencia=[]
for direccion,grupos in [('es_nah',grupos_es),('nah_es',grupos_nah)]:
 for i,grupo in enumerate(grupos.values(),1):
    primero=grupo[0]
    tag=f'{direccion}_{i:04}'
    if direccion=='es_nah':
        termino=primero['es']
        patterns=[
          f'¿Cómo se dice «{termino}» en náhuatl de la Huasteca?',
          f'Traduce al náhuatl de la Huasteca: {termino}',
          f'Quiero aprender a decir «{termino}» en náhuatl.',
          f'¿Cuál es la expresión en náhuatl para «{termino}»?',
          f'Enséñame a escribir en náhuatl: {termino}',
          f'Necesito la traducción al náhuatl de «{termino}».',
          f'En náhuatl de la Huasteca, ¿cómo expreso «{termino}»?',
          f'Dime el equivalente de «{termino}» en náhuatl.',
        ]
        equivalentes=list(dict.fromkeys(f['nah'] for f in grupo))
        respuestas=[' / '.join(equivalentes)]
    else:
        termino=primero['nah']
        patterns=[
          f'¿Qué significa «{termino}» en español?',
          f'Traduce al español esta expresión en náhuatl: {termino}',
          f'Quiero entender la frase «{termino}».',
          f'¿Cuál es el significado en español de «{termino}»?',
          f'Explícame en español «{termino}».',
          f'En la lección de náhuatl aparece «{termino}», ¿qué quiere decir?',
        ]
        equivalentes=list(dict.fromkeys(f['es'] for f in grupo))
        respuestas=[' / '.join(equivalentes)]
    aprendizaje.append(dict(tag=tag,patterns=patterns,responses=respuestas))
    procedencia.append(dict(archivo='intents_aprender_nahuatl.json',tag=tag,
                           frases=[f['id'] for f in grupo],tipo='consulta_didactica',
                           patterns='Plantillas sintéticas en español; no son consultas recogidas de usuarios.',
                           responses='Equivalencias del registro; glosa contextual, no definición exhaustiva.',
                           validacion_hablante=False))

conversaciones=[]
conversaciones_es=[]
def conv(tag,fuente,pat,res,es_pat,es_res,nota='Emparejamiento editorial de expresiones documentadas; no diálogo espontáneo grabado.'):
    assert len(pat)==len(es_pat) and len(res)==len(es_res)
    tag=fuente+'_'+tag
    conversaciones.append(dict(tag=tag,patterns=pat,responses=res))
    conversaciones_es.append(dict(tag=tag,patterns=es_pat,responses=es_res))
    procedencia.append(dict(archivo='intents_conversaciones_nahuatl.json',tag=tag,fuente=fuente,
                           tipo='dialogo_de_practica',estado='ensamblado_de_expresiones_documentadas',
                           nota=nota,validacion_hablante=False))

conv('saludo','uv',['Piyali','Piyali Juana','Piyali Pedro'],['Piyali'],['Hola','Hola Juana','Hola Pedro'],['Hola'])
conv('saludo_manana','uv',['Tonalti'],['Tonalti'],['Buenos días'],['Buenos días'])
conv('saludo_tarde','uv',['Tiotlakilti'],['Tiotlakilti'],['Buenas tardes'],['Buenas tardes'])
conv('saludo_noche','uv',['Tlapoyawalti'],['Tlapoyawalti'],['Buenas noches'],['Buenas noches'])
conv('preguntar_nombre','uv',['Tlen motoka?','Piyali, tlen motoka?'],['Nah notoka Juana'],['¿Cómo te llamas?','Hola, ¿cómo te llamas?'],['Yo me llamo Juana'], 'Personaje ficticio fijo: Juana. No es la identidad del usuario ni un dato de una persona real.')
conv('estado_bien','uv',['Kenihkatzah tiitztok?','Piyali Juana, kenihkatzah tiitztok?'],['Kualli niitztok'],['¿Cómo estás?','Hola Juana, ¿cómo estás?'],['Estoy bien'])
conv('reciprocidad_bien','uv',['Kualli niitztok','Nohkia kualli niitztok'],['Kualtitok'],['Estoy bien','También estoy bien'],['Está bien'])
conv('preguntar_destino','uv',['Kanke tiyah?'],['Niyah tiankis'],['¿A dónde vas?'],['Voy al tianguis'],'Escenario ficticio: una persona se dirige al tianguis; el bot no afirma realizar viajes reales.')
conv('desear_buen_camino','uv',['Niyah tiankis','Niyah nomilla'],['Kualli xiyah'],['Voy al tianguis','Voy a mi milpa'],['Que te vaya bien'])
conv('preguntar_estudio','uv',['Nikan timomachtia?'],['Kena, nikan nimomachtia'],['¿Aquí estudias?'],['Sí, aquí estudio'],'Escenario ficticio de dos estudiantes. Pregunta y respuesta documentadas en la misma conversación.')
conv('despedida','uv',['Timoitasehyok','Timopantisehyok','Timoittaseh','Kualli xiyah'],['Timoittaseh'],['Nos veremos luego','Nos encontraremos luego','Nos vemos','Que te vaya bien'],['Nos vemos'])
conv('aceptacion','uv',['Kena'],['Kualtitok'],['Sí'],['Está bien'],'Solo acuse de recibo conversacional; no implica ejecutar una acción ni confirmar consentimiento operativo.')

conv('saludo_manana','inali',['Mah cualli tonalli'],['Mah cualli tonalli'],['Buenos días'],['Buenos días'])
conv('saludo_tarde','inali',['Mah cualli teotlac'],['Mah cualli teotlac'],['Buenas tardes'],['Buenas tardes'])
conv('saludo_noche','inali',['Mah cualli yohualli'],['Mah cualli yohualli'],['Buenas noches'],['Buenas noches'])
conv('recibir_bienvenida','inali',['Mah cualli xihualacan'],['Tlazohcamati miyac'],['Bienvenidos'],['Muchas gracias'])
conv('invitar_pasar','inali',['Mah xipano'],['Tlazohcamati miyac'],['Pase usted'],['Muchas gracias'])
conv('buen_provecho','inali',['Mah xitlacuacan cualli'],['Tlazohcamati miyac'],['Buen provecho'],['Muchas gracias'])
conv('pedir_permiso','inali',['Mah xinechtlahpopolhui'],['Mah xipano'],['Con permiso; disculpe usted'],['Pase usted'],'Contexto limitado a pedir paso o entrada; no usar como respuesta general a una disculpa por un daño.')
conv('despedida','inali',['Timoitase','Icihuaca ximocuepa','Mah cualli ohtli'],['Timoitase'],['Adiós; nos vemos','Vuelva pronto','Que te vaya bien'],['Nos vemos'])
conv('agradecer_visita','inali',['Tlazohcamati ipampa otitechitaco'],['Timoitase'],['Muchas gracias por su visita'],['Nos vemos'])

def frase_ax(num):
    r=registros[num-1]
    assert r[4]=='nhw'
    return limpiar(r[1]),limpiar(r[0])

def conv_ax(tag,nums_pat,nums_res,nota):
    p=[frase_ax(n) for n in nums_pat]
    r=[frase_ax(n) for n in nums_res]
    conv(tag,'axolotl',[x[0] for x in p],[x[0] for x in r],
         [x[1] for x in p],[x[1] for x in r],nota)
    procedencia[-1]['registros_patterns']=nums_pat
    procedencia[-1]['registros_responses']=nums_res
    procedencia[-1]['variante_iso']='nhw'

conv_ax('comprobar_veracidad',[9254,15687],[4734], 'Pregunta de comprobación y respuesta afirmativa ensambladas. Afirmación de ejemplo, no verificación de hechos externos.')
conv_ax('recibir_regreso',[11727],[5866], 'Respuesta breve de agradecimiento. Se conserva el préstamo español «Bueno».')
conv_ax('recibir_visita',[813],[5866], 'Intercambio de cortesía ensamblado.')
conv_ax('recibir_elogio',[15089],[5866], 'Intercambio de cortesía ensamblado; se conserva «Buen Joven» como aparece en el cuento.')
conv_ax('avisar_salida',[6463,15144],[2304], 'Práctica entre personas cercanas: «¡Vete!» es imperativo directo, no una fórmula universal de cortesía.')
conv_ax('anunciar_agua',[13008],[4734], 'Acuse de recibo: «Voy a traer agua» / «Sí». No implica una acción física del sistema.')
conv_ax('aceptar_trabajo',[12008],[4734], 'Práctica sobre planes de trabajo con respuesta afirmativa.')
conv_ax('aceptar_estancia',[13006],[4734], 'Acuerdo con una propuesta de estancia; ejemplo ficticio.')
conv_ax('pedir_explicacion_accion',[2033,8354],[1378], 'La respuesta incluye «le dijo» en el corpus; este par se exporta solo en candidatos, no en el conjunto principal.')

descartado=conversaciones.pop()
conversaciones_es.pop()
procedencia.pop()

guardar('intents_conversaciones_nahuatl.json',{'intents':conversaciones})
guardar('traducciones_conversaciones_es.json',{'intents':conversaciones_es})
guardar('intents_aprender_nahuatl.json',{'intents':aprendizaje})
guardar('frases_documentadas.json',{'frases':frases})
guardar('procedencia.json',{'fecha_consulta':'2026-09-15','fuentes':FUENTES,'registros':procedencia})
guardar('descartes_ejemplos.json',{
 'nota':'Ejemplos de problemas detectados; lista ilustrativa, no auditoría exhaustiva del corpus.',
 'ejemplos':[
   {'registro_csv':612,'es':registros[611][0],'nah':registros[611][1],'motivo':'Equivalencia léxica sospechosa: aguaceros / pejki; no utilizar.'},
   {'registro_csv':796,'es':registros[795][0],'nah':registros[795][1],'motivo':'No corresponden como equivalentes aislados: muchacha / kipantito.'},
   {'registro_csv':10203,'es':registros[10202][0],'nah':registros[10202][1],'motivo':'Negación en náhuatl ausente de la glosa española.'},
   {'registro_csv':10717,'es':registros[10716][0],'nah':registros[10716][1],'motivo':'Desalineación evidente: comer / Magdalena.'},
   {'tag':descartado['tag'],'motivo':'La respuesta conservaba una acotación del narrador; no es un turno conversacional limpio.'}
 ]})

for fuente in FUENTES:
    tags={m['tag'] for m in procedencia if m['archivo']=='intents_conversaciones_nahuatl.json' and m.get('fuente')==fuente}
    guardar(f'perfiles/intents_conversaciones_{fuente}.json',{'intents':[x for x in conversaciones if x['tag'] in tags]})

def auditar(nombre,intents):
    tags=set(); patrones={}; duplicados=[]
    for x in intents:
        assert set(x)=={'tag','patterns','responses'}
        assert x['tag'] not in tags
        tags.add(x['tag'])
        assert x['patterns'] and x['responses']
        for campo in ('patterns','responses'):
            assert all(isinstance(s,str) and s.strip() and '\ufffd' not in s for s in x[campo])
        assert len(x['patterns'])==len(set(map(normalizar,x['patterns'])))
        for p in x['patterns']:
            k=clave(p)
            if k in patrones and patrones[k]!=x['tag']:
                duplicados.append((k,patrones[k],x['tag']))
            patrones[k]=x['tag']
    assert not duplicados,(nombre,duplicados)
    return dict(intents=len(intents),patterns=sum(len(x['patterns']) for x in intents),
                respuestas=sum(len(x['responses']) for x in intents),
                patrones_duplicados_entre_etiquetas=0,esquema_valido=True)

auditoria={
 'conversaciones':auditar('conversaciones',conversaciones),
 'aprendizaje':auditar('aprendizaje',aprendizaje),
 'frases_bilingues_distintas':len(frases),
 'frases_por_fuente':dict(Counter(f['fuente'] for f in frases)),
 'registros_axolotl_consultados':len(registros),
 'registros_huasteca_en_corpus':sum(x[4] in {'nhe','nhw','nch'} for x in registros),
 'validacion_por_hablante_nativo':False,
 'traduccion_automatica_usada':False,
 'advertencia':'Validación estructural y selección editorial no garantizan exactitud lingüística. Los patterns españoles son sintéticos; no equivalen a conversaciones independientes.'
}
guardar('auditoria.json',auditoria)
print(json.dumps(auditoria,ensure_ascii=False,indent=2))
