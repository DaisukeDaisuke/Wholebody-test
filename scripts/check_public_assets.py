"""GET pinned browser dependencies and resolve static ESM references in Actions.

This is a deployment asset check, not browser execution or tracking accuracy.
It requires only Python's standard library; no npm install or browser is used.
"""
from __future__ import annotations

import argparse
from collections import deque
import hashlib
from html.parser import HTMLParser
import json
import os
from pathlib import Path
import re
import subprocess
import time
import urllib.parse
import urllib.request


CDN = "https://cdn.jsdelivr.net/npm/"
ROOTS = {
    "three": CDN + "three@0.170.0/",
    "vrm": CDN + "@pixiv/three-vrm@3.3.0/",
    "vision": CDN + "@mediapipe/tasks-vision@0.10.21/",
    "ort": CDN + "onnxruntime-web@1.27.0/",
}
DEFAULT_IMPORTS = {
    "three": ROOTS["three"] + "build/three.module.js",
    "three/addons/": ROOTS["three"] + "examples/jsm/",
    "@pixiv/three-vrm": ROOTS["vrm"] + "lib/three-vrm.module.min.js",
}
REQUIRED = [
    DEFAULT_IMPORTS["three"],
    ROOTS["three"] + "examples/jsm/loaders/GLTFLoader.js",
    ROOTS["three"] + "examples/jsm/controls/OrbitControls.js",
    DEFAULT_IMPORTS["@pixiv/three-vrm"],
    ROOTS["vision"] + "vision_bundle.mjs",
    ROOTS["vision"] + "wasm/vision_wasm_internal.js",
    ROOTS["vision"] + "wasm/vision_wasm_internal.wasm",
    ROOTS["vision"] + "wasm/vision_wasm_nosimd_internal.js",
    ROOTS["vision"] + "wasm/vision_wasm_nosimd_internal.wasm",
    ROOTS["ort"] + "dist/ort.webgpu.min.mjs",
    ROOTS["ort"] + "dist/ort-wasm-simd-threaded.jsep.mjs",
    ROOTS["ort"] + "dist/ort-wasm-simd-threaded.jsep.wasm",
]
TOKEN = re.compile(
    r"(?P<comment>//[^\n]*|/\*[\s\S]*?\*/)"
    r"|(?P<string>\"(?:\\.|[^\"\\])*\"|'(?:\\.|[^'\\])*')"
    r"|(?P<template>`(?:\\.|[^`\\])*`)"
    r"|(?P<word>[A-Za-z_$][A-Za-z0-9_$]*)|(?P<punct>[^\s])")


def imports(text: str, include_dynamic: bool = False):
    tokens = [(m.lastgroup, m.group()) for m in TOKEN.finditer(text) if m.lastgroup != "comment"]
    for index, (kind, value) in enumerate(tokens):
        if kind != "word" or value not in ("from", "import"):
            continue
        next_index = index + 1
        if next_index >= len(tokens):
            continue
        if value == "import" and tokens[next_index][1] == "(":
            if not include_dynamic:
                continue
            next_index += 1
        if next_index < len(tokens) and tokens[next_index][0] == "string":
            yield tokens[next_index][1][1:-1]


class HtmlAssets(HTMLParser):
    def __init__(self):
        super().__init__()
        self.import_map, self.local_assets, self.inline_modules = {}, [], []
        self.script_type, self.script_text = None, []

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if tag == "script":
            self.script_type, self.script_text = attrs.get("type", ""), []
            if attrs.get("src"):
                self.local_assets.append(attrs["src"])
        if tag == "link" and attrs.get("rel") == "stylesheet" and attrs.get("href"):
            self.local_assets.append(attrs["href"])

    def handle_data(self, data):
        if self.script_type is not None:
            self.script_text.append(data)

    def handle_endtag(self, tag):
        if tag != "script" or self.script_type is None:
            return
        content = "".join(self.script_text)
        if self.script_type == "importmap":
            self.import_map.update(json.loads(content).get("imports", {}))
        elif self.script_type == "module":
            self.inline_modules.append(content)
        self.script_type, self.script_text = None, []


def allowed_url(url: str) -> bool:
    return any(url.startswith(root) for root in ROOTS.values())


def get_public(url: str):
    if not allowed_url(url):
        raise ValueError(f"Unexpected unpinned dependency URL: {url}")
    for attempt in range(3):
        try:
            request = urllib.request.Request(url, headers={"User-Agent": "wholebody-static-assets/1.0"})
            with urllib.request.urlopen(request, timeout=90) as response:
                if response.status != 200 or not allowed_url(response.url):
                    raise RuntimeError(f"Unexpected CDN response for {url}")
                mime = response.headers.get("Content-Type", "").split(";", 1)[0].lower()
                cors = response.headers.get("Access-Control-Allow-Origin")
                if cors != "*":
                    raise RuntimeError(f"CDN asset lacks anonymous cross-origin access: {url}")
                data = response.read(100_000_001)
            if not data or len(data) > 100_000_000:
                raise RuntimeError(f"Unexpected asset size: {url}")
            is_wasm = urllib.parse.urlparse(url).path.endswith(".wasm")
            if is_wasm:
                if data[:4] != b"\0asm" or mime != "application/wasm":
                    raise RuntimeError(f"Invalid WebAssembly bytes/MIME: {url}")
                text = None
            else:
                if mime not in ("application/javascript", "text/javascript", "application/x-javascript"):
                    raise RuntimeError(f"Non-JavaScript CDN MIME {mime}: {url}")
                text = data.decode("utf-8")
            return text, {"url": url, "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest(),
                          "mime": mime, "cors": cors, "utf8": not is_wasm}
        except Exception:
            if attempt == 2:
                raise
            time.sleep(2 ** attempt)
    raise AssertionError("Unreachable")


def resolve(specifier: str, parent: str | Path, import_map: dict, static: Path):
    if specifier.startswith("https://"):
        return specifier
    if specifier.startswith(("./", "../", "/")):
        if isinstance(parent, str):
            return urllib.parse.urljoin(parent, specifier)
        base = static if specifier.startswith("/") else parent.parent
        target = (base / specifier.lstrip("/")).resolve()
        if not target.is_relative_to(static):
            raise ValueError(f"Module escapes static directory: {specifier}")
        return target
    for key in sorted(import_map, key=len, reverse=True):
        if specifier == key or key.endswith("/") and specifier.startswith(key):
            suffix = specifier[len(key):]
            mapped = import_map[key] + suffix
            return resolve(mapped, static / "index.html", {}, static)
    raise ValueError(f"Bare module has no import-map entry: {specifier}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--static-directory", type=Path, default=Path("dist"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/public-assets.json"))
    args = parser.parse_args()
    if os.environ.get("GITHUB_ACTIONS") != "true":
        raise SystemExit("Deployment dependency checks run in GitHub Actions.")
    static = args.static_directory.resolve()
    document = HtmlAssets()
    index = static / "index.html"
    if index.exists():
        document.feed(index.read_text(encoding="utf-8"))
        import_map = document.import_map
        for key, expected in DEFAULT_IMPORTS.items():
            if import_map.get(key) != expected:
                raise RuntimeError(f"Missing or different pinned import-map entry: {key}")
    else:
        import_map = DEFAULT_IMPORTS
    local_files, edges = [], []
    queue = deque(REQUIRED)
    if static.exists():
        result = subprocess.run(["rg", "--files", str(static), "-g", "*.js", "-g", "*.mjs"],
                                check=False, capture_output=True, text=True, encoding="utf-8")
        if result.returncode not in (0, 1):
            raise RuntimeError("Cannot enumerate delivered JavaScript modules")
        local_files = [Path(line) for line in result.stdout.splitlines() if line]
        for path in local_files:
            source = path.read_text(encoding="utf-8")
            for name, version in (("VISION_VERSION", "0.10.21"), ("ORT_VERSION", "1.27.0")):
                declared = re.search(r"\bconst\s+" + name + r"\s*=\s*['\"]([^'\"]+)['\"]", source)
                if declared and declared.group(1) != version:
                    raise RuntimeError(f"Runtime {name} does not match checked CDN version")
            for specifier in imports(source, include_dynamic=True):
                dependency = resolve(specifier, path, import_map, static)
                edges.append({"from": str(path.relative_to(static)), "import": specifier})
                if isinstance(dependency, Path):
                    if not dependency.is_file():
                        raise RuntimeError(f"Missing local module: {dependency.relative_to(static)}")
                    dependency.read_text(encoding="utf-8")
                else:
                    queue.append(dependency)
        for source in document.inline_modules:
            for specifier in imports(source, include_dynamic=True):
                dependency = resolve(specifier, index, import_map, static)
                if isinstance(dependency, str):
                    queue.append(dependency)
                elif not dependency.is_file():
                    raise RuntimeError(f"Missing entry module: {dependency.relative_to(static)}")
        for asset in document.local_assets:
            if asset.startswith("https://"):
                queue.append(asset)
            else:
                target = static / asset.removeprefix("./").lstrip("/")
                if not target.is_file():
                    raise RuntimeError(f"Missing HTML asset: {asset}")
                target.read_text(encoding="utf-8")
    verified, seen = [], set()
    while queue:
        url = queue.popleft()
        if url in seen:
            continue
        seen.add(url)
        source, report = get_public(url)
        verified.append(report)
        if source is not None:
            for specifier in imports(source):
                dependency = resolve(specifier, url, import_map, static)
                if not isinstance(dependency, str):
                    raise RuntimeError("Remote ESM unexpectedly resolves to a workspace path")
                queue.append(dependency)
        print(json.dumps({"verifiedDependency": url, "bytes": report["bytes"]}), flush=True)
    result = {"schema": "wholebody-public-assets-v1", "publicAssets": verified,
              "localModuleCount": len(local_files), "staticImportReferences": edges,
              "htmlEntryChecked": index.exists(), "browserExecuted": False,
              "gpuInferenceMeasured": False, "trackingAccuracyCertified": False,
              "scope": "HTTP GET, UTF-8 JavaScript, pinned static ESM/import-map resolution, CORS and WASM bytes/MIME"}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
