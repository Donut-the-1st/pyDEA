import sys
import os
from pathlib import Path

# Automatically locate and inject Tcl/Tk paths from the base Python installation
base_prefix = Path(sys.base_prefix)

# Windows Paths
tcl_dir_win = base_prefix / "tcl"
# macOS/Linux Paths
tcl_dir_unix = base_prefix / "lib"

if tcl_dir_win.exists():
    # Find the specific tcl8.x or tk8.x directories
    tcl_lib = list(tcl_dir_win.glob("tcl8*"))
    if tcl_lib:
        os.environ["TCL_LIBRARY"] = str(tcl_dir_win / tcl_lib[0].name)
        os.environ["TK_LIBRARY"] = str(tcl_dir_win / tcl_lib[0].name.replace("tcl", "tk"))
elif tcl_dir_unix.exists():
    tcl_lib = list(tcl_dir_unix.glob("tcl8*"))
    if tcl_lib:
        os.environ["TCL_LIBRARY"] = str(tcl_lib[0])
        # Tk might be inside /lib or /lib/tk8.x
        tk_lib = list(tcl_dir_unix.glob("tk8*"))
        if tk_lib:
            os.environ["TK_LIBRARY"] = str(tk_lib[0])
