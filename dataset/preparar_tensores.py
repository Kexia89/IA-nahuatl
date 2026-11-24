from pathlib import Path
import argparse
import hashlib
import json
import unicodedata
import numpy as np

BASE=Path(__file__).resolve().parent

def normalizar(texto):
    return ' '.join(unicodedata.normalize('NFC',texto).casefold().split())

def guardar_json(path,obj):
    path.write_text(json.dumps(obj,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')

def preparar(path,out):
    contenido=path.read_bytes()
    intents=json.loads(contenido)['intents']
    datos={'train':[],'validation':[],'test':[]}
    etiquetas=[]
    educativos=all(x['tag'].startswith(('es_nah_','nah_es_')) for x in intents)
    for intent in intents:
        if not intent['patterns']:
            continue
        etiqueta=len(etiquetas)
        etiquetas.append(intent['tag'])
        for j,p in enumerate(intent['patterns']):
            split='train'
            if educativos:
                if intent['tag'].startswith('es_nah_'):
                    split='train' if j<4 else ('validation' if j<6 else 'test')
                else:
                    split='train' if j<4 else ('validation' if j==4 else 'test')
            datos[split].append({'text':p,'tag':intent['tag'],'label':etiqueta,'template_index':j})
    assert datos['train']
    textos_train=[normalizar(r['text']) for r in datos['train']]
    chars=sorted(set(''.join(textos_train)))
    tokens=['<PAD>','<UNK>']+chars
    char_to_id={s:i for i,s in enumerate(tokens)}
    longitud=((max(map(len,textos_train))+63)//64)*64
    out.mkdir(parents=True,exist_ok=True)
    estadisticas={}
    matrices={}
    for split,rows in datos.items():
        X=np.zeros((len(rows),longitud),dtype=np.int32)
        y=np.array([r['label'] for r in rows],dtype=np.int32)
        truncados=0
        for i,r in enumerate(rows):
            text=normalizar(r['text'])
            ids=[char_to_id.get(c,1) for c in text]
            truncados+=len(ids)>longitud
            X[i,:min(len(ids),longitud)]=ids[:longitud]
        mask=X!=0
        matrices[f'X_{split}']=X
        matrices[f'y_{split}']=y
        matrices[f'mask_{split}']=mask
        estadisticas[split]={'filas':len(rows),'forma_X':list(X.shape),
                            'desconocidos':int(np.sum(X==1)),'textos_truncados':truncados}
        with (out/f'{split}.jsonl').open('w',encoding='utf-8') as f:
            for r in rows:
                f.write(json.dumps(r,ensure_ascii=False)+'\n')
    sets=[set(normalizar(r['text']) for r in datos[s]) for s in datos]
    assert all(not a&b for i,a in enumerate(sets) for b in sets[i+1:])
    np.savez_compressed(out/'datos.npz',**matrices)
    guardar_json(out/'tokenizador.json',{
        'tipo':'caracteres','normalizacion':'Unicode NFC, casefold y espacios; conservar acentos, h y apóstrofos',
        'pad_id':0,'unk_id':1,'max_length':longitud,'tokens':tokens,
        'ajustado_solo_con_train':True,'truncation':'right','padding':'right',
    })
    guardar_json(out/'etiquetas.json',{'tags':etiquetas})
    guardar_json(out/'respuestas.json',{x['tag']:x['responses'] for x in intents})
    guardar_json(out/'informe.json',{
        'entrada':path.name,'sha256_entrada':hashlib.sha256(contenido).hexdigest(),
        'clases':len(etiquetas),'vocabulario':len(tokens),'particiones':estadisticas,
        'estrategia_particion':'Familias de plantillas españolas: evaluación cerrada del clasificador.' if educativos else 'Solo train: no hay suficientes ejemplos independientes por clase para una evaluación válida.',
        'limite_evaluacion':'No mide conversación libre, traducción general ni validación lingüística.',
    })
    with np.load(out/'datos.npz',allow_pickle=False) as saved:
        for split,rows in datos.items():
            X,y,mask=(saved[f'{k}_{split}'] for k in ('X','y','mask'))
            assert X.dtype==np.int32 and y.dtype==np.int32 and mask.dtype==np.bool_
            assert X.shape==(len(rows),longitud) and y.shape==(len(rows),)
            assert np.array_equal(mask,X!=0)
            assert not y.size or (y.min()>=0 and y.max()<len(etiquetas))
            assert not X.size or (X.min()>=0 and X.max()<len(tokens))
            for i,r in enumerate(rows):
                expected=normalizar(r['text'])[:longitud]
                reconstructed=''.join(tokens[n] for n in X[i] if n)
                assert reconstructed==''.join(c if c in char_to_id else '<UNK>' for c in expected)
    print(path.name,json.dumps(estadisticas,ensure_ascii=False))

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input',type=Path,help='Otro archivo de intents con la misma estructura.')
    parser.add_argument('--output',type=Path,help='Directorio de salida para --input.')
    args=parser.parse_args()
    if args.input:
        preparar(args.input,args.output or BASE/'tensores/personalizado')
    else:
        preparar(BASE/'intents_conversaciones_nahuatl.json',BASE/'tensores/conversaciones')
        preparar(BASE/'intents_aprender_nahuatl.json',BASE/'tensores/aprendizaje')
