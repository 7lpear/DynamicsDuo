"""Download pinned official SO-101 geometry and local browser dependencies."""
import json
from pathlib import Path
from urllib.request import urlopen

ROOT = Path(__file__).parent / "viewer"


def download(url, destination):
    destination.parent.mkdir(parents=True, exist_ok=True)
    with urlopen(url, timeout=45) as response:
        content = response.read()
    if content.startswith(b"version https://git-lfs.github.com/spec/v1"):
        url = url.replace("raw.githubusercontent.com/", "media.githubusercontent.com/media/")
        with urlopen(url, timeout=45) as response:
            content = response.read()
        if content.startswith(b"version https://git-lfs.github.com/spec/v1"):
            raise RuntimeError(f"Unresolved LFS asset: {url}")
    destination.write_bytes(content)
    print(destination.relative_to(ROOT))


def main():
    with urlopen("https://api.github.com/repos/huggingface/leLab/git/trees/main?recursive=1", timeout=45) as response:
        tree = json.load(response)
    revision = tree["sha"]
    prefix = "frontend/public/so-101-urdf/"
    for entry in tree["tree"]:
        path = entry["path"]
        if path.startswith(prefix) and entry["type"] == "blob":
            download(f"https://raw.githubusercontent.com/huggingface/leLab/{revision}/{path}", ROOT / "model" / path[len(prefix):])
    download(f"https://raw.githubusercontent.com/huggingface/leLab/{revision}/LICENSE", ROOT / "LELAB-LICENSE.txt")
    version = "0.180.0"
    for path in ["build/three.module.js", "build/three.core.js", "examples/jsm/controls/OrbitControls.js", "examples/jsm/loaders/STLLoader.js"]:
        download(f"https://cdn.jsdelivr.net/npm/three@{version}/{path}", ROOT / "vendor" / Path(path).name)
    download(f"https://cdn.jsdelivr.net/npm/three@{version}/LICENSE", ROOT / "vendor" / "LICENSE.txt")
    (ROOT / "assets-version.json").write_text(json.dumps({"lelab_commit": revision, "three": version}, indent=2))


if __name__ == "__main__":
    main()
