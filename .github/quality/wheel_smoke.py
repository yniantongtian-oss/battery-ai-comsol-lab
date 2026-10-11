"""Install the built wheel in a clean environment and test its public imports."""
from __future__ import annotations
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import venv

root = Path(__file__).resolve().parents[2]
plan = json.loads((root / '.github/quality/wheel-smoke.json').read_text(encoding='utf-8'))
wheels = list((root / 'wheel-dist').glob('*.whl'))
assert len(wheels) == 1, f'Expected exactly one current wheel: {wheels}'
with tempfile.TemporaryDirectory(prefix='installed-wheel-') as folder:
    external = Path(folder).resolve()
    environment = external / 'venv'
    venv.create(environment, with_pip=True)
    scripts = environment / ('Scripts' if os.name == 'nt' else 'bin')
    python = scripts / ('python.exe' if os.name == 'nt' else 'python')
    wheel = str(wheels[0].resolve())
    if plan['extras']: wheel += '[' + plan['extras'] + ']'
    clean_env = dict(os.environ)
    clean_env.pop('PYTHONPATH', None)
    clean_env.pop('PYTHONHOME', None)
    clean_env['PYTHONUTF8'] = '1'
    def run(arguments, timeout=120):
        subprocess.run(arguments, cwd=external, env=clean_env, check=True, timeout=timeout)
    run([str(python), '-m', 'pip', 'install', wheel], timeout=900)
    run([str(python), '-m', 'pip', 'check'])
    code = (
        'import importlib,json,pathlib,sys; '
        'modules=json.loads(sys.argv[1]); prefix=pathlib.Path(sys.prefix).resolve(); '
        'origins={name:str(pathlib.Path(importlib.import_module(name).__file__).resolve()) for name in modules}; '
        'assert all(pathlib.Path(p).is_relative_to(prefix) for p in origins.values()), origins; '
        'print(json.dumps(origins,indent=2))'
    )
    run([str(python), '-I', '-c', code, json.dumps(plan['modules'])])
    cli = scripts / (plan['cli'] + ('.exe' if os.name == 'nt' else ''))
    assert cli.exists(), cli
    for arguments in plan['commands']:
        run([str(cli), *[value.replace('{root}', str(root)) for value in arguments]])
print('Installed wheel public imports and CLI checks passed outside the checkout')
