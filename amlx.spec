# -*- mode: python ; coding: utf-8 -*-
from PyInstaller.utils.hooks import collect_all, collect_data_files

mlx_datas,    mlx_binaries,    mlx_hidden    = collect_all("mlx")
mlx_lm_datas, mlx_lm_binaries, mlx_lm_hidden = collect_all("mlx_lm")
np_datas,     np_binaries,     np_hidden     = collect_all("numpy")

a = Analysis(
    ["src/amlx/cli.py"],
    pathex=["src"],
    binaries=mlx_binaries + mlx_lm_binaries + np_binaries,
    datas=[
        ("src/amlx/ui", "amlx/ui"),
    ] + mlx_datas + mlx_lm_datas + np_datas,
    hiddenimports=[
        "amlx",
        "amlx.api", "amlx.api.app", "amlx.api.routes",
        "amlx.cache", "amlx.inference", "amlx.model_manager", "amlx.adapters",
        "amlx.adapters.mlx_adapter",
        "uvicorn.logging",
        "uvicorn.loops", "uvicorn.loops.asyncio", "uvicorn.loops.uvloop",
        "uvicorn.http", "uvicorn.http.h11_impl", "uvicorn.http.httptools_impl",
        "uvicorn.protocols",
        "uvicorn.protocols.websockets", "uvicorn.protocols.websockets.websockets_impl",
        "uvicorn.lifespan", "uvicorn.lifespan.off", "uvicorn.lifespan.on",
        "fastapi", "fastapi.staticfiles", "fastapi.responses",
        "starlette.staticfiles", "starlette.responses",
        "setuptools", "pkg_resources",
    ] + mlx_hidden + mlx_lm_hidden + np_hidden,
    hookspath=[],
    runtime_hooks=[],
    excludes=["tkinter", "matplotlib", "scipy", "pandas"],
    noarchive=False,
)

pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    [],
    name="amlx",
    debug=False,
    strip=True,
    upx=False,
    console=True,
    target_arch="arm64",
    # Extract once to persistent cache — survives reboots, reused every run
    runtime_tmpdir="~/Library/Caches/amlx",
)
