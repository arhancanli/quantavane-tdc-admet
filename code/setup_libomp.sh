#!/bin/sh
# Make LightGBM importable on macOS without Homebrew.
#
# The LightGBM wheel's lib_lightgbm.dylib asks the loader for @rpath/libomp.dylib
# and records only two rpaths, /opt/homebrew/opt/libomp/lib and
# /opt/local/lib/libomp. With neither Homebrew nor MacPorts present, the import
# fails. Setting DYLD_LIBRARY_PATH is not a usable workaround here because macOS
# strips DYLD_* variables when a system binary (such as /usr/bin/time) execs.
#
# So: copy the libomp that the scikit-learn wheel already bundles into LightGBM's
# own lib directory, repoint the dependency at @loader_path, and ad-hoc re-sign
# the patched dylib. Everything stays inside this project's .venv; no system
# directory is touched and no sudo is needed.
#
# On a machine with Homebrew, `brew install libomp` replaces this script.
set -eu

HERE=$(cd "$(dirname "$0")" && pwd)
BASE=$(dirname "$HERE")
PY="$BASE/.venv/bin/python"
export DEVELOPER_DIR="${DEVELOPER_DIR:-/Library/Developer/CommandLineTools}"

SRC=$("$PY" -c "import os,sklearn;print(os.path.join(os.path.dirname(sklearn.__file__),'.dylibs','libomp.dylib'))")
LGBDIR=$("$PY" -c "import os,lightgbm.libpath as l;print(os.path.dirname(l.__file__))" 2>/dev/null || true)
if [ -z "$LGBDIR" ]; then
  LGBDIR=$("$PY" -c "import os,site;print(os.path.join(site.getsitepackages()[0],'lightgbm'))")
fi
LIB="$LGBDIR/lib/lib_lightgbm.dylib"

[ -f "$SRC" ] || { echo "no bundled libomp at $SRC" >&2; exit 1; }
[ -f "$LIB" ] || { echo "no lib_lightgbm.dylib at $LIB" >&2; exit 1; }

if otool -L "$LIB" | grep -q "@loader_path/libomp.dylib"; then
  echo "already patched: $LIB"
else
  cp -f "$SRC" "$LGBDIR/lib/libomp.dylib"
  install_name_tool -change @rpath/libomp.dylib @loader_path/libomp.dylib "$LIB"
  codesign --force --sign - "$LIB"
  echo "patched $LIB -> @loader_path/libomp.dylib"
fi

"$PY" -c "import lightgbm; print('lightgbm', lightgbm.__version__, 'imports cleanly')"
