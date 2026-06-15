"""Package entrypoint: forward to web_cam.main()

Allows running the repository as a package with `python -m BTL` (when invoked
from the parent directory) which will delegate to the existing web_cam CLI.
"""
import sys

try:
    # Import local web_cam module and call its main()
    import web_cam

    if hasattr(web_cam, 'main'):
        web_cam.main()
    else:
        print('web_cam.main() not found')
        sys.exit(2)
except Exception as e:
    print('Failed to launch web_cam as package entrypoint:', e)
    raise
