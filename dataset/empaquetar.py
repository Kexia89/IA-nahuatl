from pathlib import Path
import hashlib
import json
import zipfile

BASE=Path(__file__).resolve().parent
EXCLUIR={'fuentes/axolotl.csv'}
archivos=sorted(p for p in BASE.rglob('*') if p.is_file()
                and p.relative_to(BASE).as_posix() not in EXCLUIR
                and '__pycache__' not in p.parts
                and p.suffix.lower() in {'.json','.jsonl','.npz','.md','.py','.txt'}
                and p.name!='SHA256SUMS.txt')

for path in archivos:
    if path.suffix=='.json':
        json.loads(path.read_text(encoding='utf-8'))
    elif path.suffix=='.jsonl':
        for line in path.read_text(encoding='utf-8').splitlines():
            json.loads(line)

manifest=BASE/'SHA256SUMS.txt'
manifest.write_text(''.join(hashlib.sha256(p.read_bytes()).hexdigest()+'  '+p.relative_to(BASE).as_posix()+'\n' for p in archivos),encoding='utf-8')
destino=BASE.parent/'nahuatl_huasteca_dataset.zip'
with zipfile.ZipFile(destino,'w',compression=zipfile.ZIP_DEFLATED) as z:
    for path in archivos+[manifest]:
        z.write(path,BASE.name+'/'+path.relative_to(BASE).as_posix())
with zipfile.ZipFile(destino) as z:
    assert z.testzip() is None
    assert not any(n.endswith('/axolotl.csv') for n in z.namelist())
    for path in archivos+[manifest]:
        assert z.read(BASE.name+'/'+path.relative_to(BASE).as_posix())==path.read_bytes()
print(json.dumps({'zip':str(destino),'archivos':len(archivos)+1,'bytes':destino.stat().st_size},ensure_ascii=False))
