from pathlib import Path
import json
import threading
import urllib.request
import urllib.error

from interfaz import LocalServer

BASE = Path(__file__).resolve().parent


def main():
    server = LocalServer(0, BASE / 'models')
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base = f'http://127.0.0.1:{server.server_port}'

    def request(path, data=None):
        req = urllib.request.Request(base + path,
            data=json.dumps(data, ensure_ascii=False).encode('utf-8') if data is not None else None,
            headers={'Content-Type': 'application/json'} if data is not None else {})
        try:
            with urllib.request.urlopen(req, timeout=90) as response:
                body=response.read()
                return response.status, json.loads(body) if response.headers.get_content_type()=='application/json' else body
        except urllib.error.HTTPError as error:
            return error.code, json.loads(error.read())

    results=[]
    try:
        assert request('/health')[0]==200
        assert request('/')[0]==200
        code,status=request('/api/status')
        assert code==200 and all(m['available'] for m in status['models'])
        assert not any(m['loaded'] for m in status['models']), 'El estado no debe cargar TensorFlow.'
        for mode,message,expected in [
            ('aprendizaje','¿Qué significa «Piyali» en español?','hola'),
            ('aprendizaje','¿Cómo se dice «buenos días» en náhuatl de la Huasteca?','Mah cualli tonalli / Tonalti'),
            ('conversaciones','Piyali','Piyali'),
            ('conversaciones','Kenihkatzah tiitztok?','Kualli niitztok'),
        ]:
            code,answer=request('/api/chat',{'mode':mode,'message':message})
            assert code==200 and answer['accepted'] and answer['answer']==expected, (message,code,answer)
            results.append({'modo':mode,'entrada':message,**answer})
        for data in ({'mode':'incorrecto','message':'hola'},
                     {'mode':'conversaciones','message':'a'*65},
                     {'mode':'aprendizaje','message':''},
                     {'mode':'aprendizaje','message':'???'}):
            assert request('/api/chat',data)[0]==400
        code,unknown=request('/api/chat',{'mode':'aprendizaje','message':'漢字漢字'})
        assert code==200 and unknown['accepted'] is False and unknown['tag'] is None
        assert request('/../motor.py')[0]==404
        report={'pruebas_http':'correctas','casos_con_modelos_reales':results,
                'rechazos_verificados':['modo inválido','entrada demasiado larga','vacío','solo signos','caracteres desconocidos'],
                'nota':'Pruebas funcionales de ejemplos del catálogo; no evaluación de generalización.'}
        (BASE/'verificacion_interfaz.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
        print(json.dumps(report,ensure_ascii=False,indent=2))
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


if __name__=='__main__':
    main()
