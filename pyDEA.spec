# -*- mode: python ; coding: utf-8 -*-
from PyInstaller.utils.hooks import collect_data_files

# pyDEA: logging_config.ini and logo files; pulp: bundled CBC solver binaries.
datas = collect_data_files('pyDEA') + collect_data_files('pulp')

a = Analysis(
    ['run_pyDEA.py'],
    pathex=[],
    binaries=[],
    datas=datas,
    hiddenimports=['logging.handlers'],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=['pytest', 'xlrd'],
    noarchive=False,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='pyDEA',
    debug=False,
    strip=False,
    upx=False,
    console=False,
    icon='pyDEA/pyDEAlogo.ico',
)
coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    name='pyDEA',
)
