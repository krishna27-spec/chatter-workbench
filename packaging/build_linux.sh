#!/usr/bin/env bash
# Build the Linux installer: build/chatter-workbench_<version>_amd64.deb
#   1. PyInstaller packs Python, the app, every library and the top-3 models into build/dist/chatter-workbench/
#   2. the packed app runs its self-test (formats, fonts, models predict, NI driver); the build stops if it fails
#   3. dpkg-deb wraps it: /opt/chatter-workbench, a `chatter-workbench` command, a menu entry and an icon
# Install on any Ubuntu/Debian PC:  sudo apt install ./build/chatter-workbench_<version>_amd64.deb
# The .deb contains the trained models (from confidential lab data): copy it only to machines you trust.
set -euo pipefail
cd "$(dirname "$0")/.."
PY=.venv-wb/bin/python
VERSION=$(env -u PYTHONPATH $PY -c "import workbench; print(workbench.__version__)")

env -u PYTHONPATH $PY -m PyInstaller --noconfirm --clean --log-level WARN \
    --distpath build/dist --workpath build/work packaging/chatter_workbench.spec

echo "== self-test of the packed app"
env -u PYTHONPATH QT_QPA_PLATFORM=offscreen build/dist/chatter-workbench/chatter-workbench --self-test

PKG=build/deb/chatter-workbench_${VERSION}_amd64
rm -rf "$PKG"
mkdir -p "$PKG/DEBIAN" "$PKG/opt" "$PKG/usr/bin" "$PKG/usr/share/applications" \
         "$PKG/usr/share/icons/hicolor/256x256/apps"
cp -a build/dist/chatter-workbench "$PKG/opt/chatter-workbench"
ln -s /opt/chatter-workbench/chatter-workbench "$PKG/usr/bin/chatter-workbench"
cp packaging/icon.png "$PKG/usr/share/icons/hicolor/256x256/apps/chatter-workbench.png"
cat > "$PKG/usr/share/applications/chatter-workbench.desktop" <<DESK
[Desktop Entry]
Type=Application
Name=Chatter Workbench
Comment=Analyse micromilling recordings and run the live chatter rig
Exec=chatter-workbench %f
Icon=chatter-workbench
Terminal=false
Categories=Science;Engineering;
DESK
cat > "$PKG/DEBIAN/control" <<CTRL
Package: chatter-workbench
Version: ${VERSION}
Architecture: amd64
Maintainer: Srikrishna (local build)
Installed-Size: $(du -sk "$PKG/opt" | cut -f1)
Depends: libxcb-cursor0, libxkbcommon-x11-0, libegl1, libgl1, libfontconfig1, libdbus-1-3
Section: science
Priority: optional
Description: Chatter Workbench: micromilling chatter analysis and live control
 Opens .lvm/.csv recordings and scope.py / workbench sessions (time plot, spectrum,
 spectrogram), and runs the live rig on NI cDAQ hardware with the top-3 chatter models
 and the Liao & Young spindle-speed controller. Live needs the NI-DAQmx driver (ni.com).
CTRL
dpkg-deb --build --root-owner-group "$PKG" "build/chatter-workbench_${VERSION}_amd64.deb"
ls -lh "build/chatter-workbench_${VERSION}_amd64.deb"
