@echo off
rem Build Chatter Workbench for Windows. Run from the unzipped source folder (double-click, or from a terminal).
rem Needs: Python 3.12 from python.org ("Add python.exe to PATH" ticked). Optional: Inno Setup 6 for one Setup.exe.
rem Result: build\dist\ChatterWorkbench\ChatterWorkbench.exe  (and build\ChatterWorkbench-Setup-<version>.exe)
setlocal
cd /d "%~dp0\.."

where py >nul 2>nul
if errorlevel 1 (
  echo Python launcher "py" not found. Install Python 3.12 from https://www.python.org/downloads/windows/
  echo and tick "Add python.exe to PATH" in the installer, then run this again.
  pause & exit /b 1
)
if not exist .venv-win\Scripts\python.exe (
  echo == creating .venv-win with Python 3.12
  py -3.12 -m venv .venv-win || (echo Python 3.12 is needed: py -3.12 failed & pause & exit /b 1)
)
echo == installing the exact library versions (packaging\requirements.txt)
.venv-win\Scripts\python -m pip install --upgrade pip || (pause & exit /b 1)
.venv-win\Scripts\python -m pip install -r packaging\requirements.txt || (
  echo pip failed. If it was pycatch22 or another package needing a C compiler, install
  echo "Microsoft C++ Build Tools" ^(Desktop development with C++^) and run this again.
  pause & exit /b 1
)

echo == drawing the icon
.venv-win\Scripts\python packaging\make_icon.py || (pause & exit /b 1)

echo == packing the app (several minutes)
.venv-win\Scripts\python -m PyInstaller --noconfirm --clean --log-level WARN --distpath build\dist --workpath build\work packaging\chatter_workbench.spec || (pause & exit /b 1)

echo == self-test of the packed app (a window shows the report; close it to continue)
build\dist\ChatterWorkbench\ChatterWorkbench.exe --self-test
if errorlevel 1 (echo Self-test FAILED: see the report window and Documents\Chatter Workbench\self-test.txt & pause & exit /b 1)

set ISCC="%ProgramFiles(x86)%\Inno Setup 6\ISCC.exe"
if exist %ISCC% (
  echo == making the installer
  for /f %%v in ('.venv-win\Scripts\python -c "import workbench; print(workbench.__version__)"') do set VER=%%v
  %ISCC% /DMyVersion=%VER% packaging\windows_installer.iss || (pause & exit /b 1)
  echo Done: build\ChatterWorkbench-Setup-%VER%.exe
) else (
  echo Inno Setup 6 not installed: no Setup.exe made. The app itself is ready to run:
  echo     build\dist\ChatterWorkbench\ChatterWorkbench.exe   ^(copy the whole ChatterWorkbench folder^)
  echo For a one-file installer, install Inno Setup 6 from https://jrsoftware.org/isdl.php and run this again.
)
pause
