from pathlib import Path
import zipfile


BASE = Path(__file__).resolve().parent
OUTPUT = BASE.parent / "tlahtolli-github.zip"
EXCLUDED_PARTS = {".venv", "venv", "__pycache__", ".git", ".idea", ".vscode"}


def included(path: Path) -> bool:
    relative = path.relative_to(BASE)
    return (
        path.is_file()
        and not any(part in EXCLUDED_PARTS for part in relative.parts)
        and not any("_anterior_" in part for part in relative.parts)
    )


files = [path for path in BASE.rglob("*") if included(path)]
with zipfile.ZipFile(OUTPUT, "w", compression=zipfile.ZIP_DEFLATED) as archive:
    for path in files:
        archive.write(path, (Path("tlahtolli") / path.relative_to(BASE)).as_posix())

with zipfile.ZipFile(OUTPUT) as archive:
    assert archive.testzip() is None
    names = set(archive.namelist())
    assert "tlahtolli/app.py" in names
    for mode in ("aprendizaje", "conversaciones"):
        assert f"tlahtolli/models/{mode}/modelo.h5" in names

print(f"{OUTPUT}\n{OUTPUT.stat().st_size:,} bytes; ZIP verificado.")
