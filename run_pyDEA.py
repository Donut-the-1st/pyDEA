''' Launcher used to build the standalone executable with PyInstaller. '''

import os
import sys


def _use_writable_working_dir():
    # The log file and the tmp folder are created relative to the current
    # directory, which may be read-only when the app is installed.
    base = os.environ.get('LOCALAPPDATA') or os.path.expanduser('~')
    work_dir = os.path.join(base, 'pyDEA')
    os.makedirs(work_dir, exist_ok=True)
    os.chdir(work_dir)


if __name__ == '__main__':
    if getattr(sys, 'frozen', False):
        _use_writable_working_dir()
    from pyDEA.main_gui import main
    main()
