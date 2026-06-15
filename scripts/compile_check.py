import compileall, os, sys
root='.'
bad=[]
for dirpath, dirs, files in os.walk(root):
    if '.venv' in dirpath.split(os.sep) or os.path.normpath('gpt') in os.path.normpath(dirpath):
        continue
    for f in files:
        if f.endswith('.py'):
            path=os.path.join(dirpath,f)
            ok=compileall.compile_file(path, force=False, quiet=1)
            if not ok:
                bad.append(path)
if bad:
    print('COMPILE_ERRORS')
    for p in bad:
        print(p)
    sys.exit(2)
print('COMPILE_OK')
