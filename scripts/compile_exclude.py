"""Compile all .py files under the repo root excluding .venv and gpt folders.
Prints failures and exits with code 1 if any compile errors found.
"""
import os, py_compile, sys
root = os.path.dirname(os.path.abspath(__file__))
root = os.path.dirname(root)  # project root
errors = []
for dirpath, dirnames, filenames in os.walk(root):
    # normalize
    rel = os.path.relpath(dirpath, root)
    if rel.startswith('.venv') or rel.split(os.sep)[0] == '.venv' or rel.startswith('gpt') or rel.split(os.sep)[0] == 'gpt':
        continue
    # skip hidden .git etc
    if rel.startswith('.git'):
        continue
    for f in filenames:
        if not f.endswith('.py'):
            continue
        path = os.path.join(dirpath, f)
        try:
            py_compile.compile(path, doraise=True)
        except Exception as e:
            errors.append((path, str(e)))

if not errors:
    print('COMPILE_OK')
    sys.exit(0)
print('COMPILE_FAIL')
for p, e in errors:
    print('---')
    print(p)
    print(e)
sys.exit(1)
