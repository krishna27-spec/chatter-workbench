; Inno Setup script: one ChatterWorkbench-Setup-<version>.exe from the PyInstaller folder.
; Called by build_windows.bat (ISCC /DMyVersion=x.y.z). Installs per machine into Program Files,
; with a Start-menu entry and an optional desktop icon. The installer contains the trained models.
#ifndef MyVersion
  #define MyVersion "0.0.0"
#endif

[Setup]
AppName=Chatter Workbench
AppVersion={#MyVersion}
AppPublisher=Chatter Workbench
DefaultDirName={autopf}\Chatter Workbench
DefaultGroupName=Chatter Workbench
OutputDir=..\build
OutputBaseFilename=ChatterWorkbench-Setup-{#MyVersion}
SetupIconFile=icon.ico
UninstallDisplayIcon={app}\ChatterWorkbench.exe
Compression=lzma2
SolidCompression=yes
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible

[Tasks]
Name: "desktopicon"; Description: "Create a desktop icon"; Flags: unchecked

[Files]
Source: "..\build\dist\ChatterWorkbench\*"; DestDir: "{app}"; Flags: recursesubdirs createallsubdirs ignoreversion

[Icons]
Name: "{group}\Chatter Workbench"; Filename: "{app}\ChatterWorkbench.exe"
Name: "{group}\Chatter Workbench self-test"; Filename: "{app}\ChatterWorkbench.exe"; Parameters: "--self-test"
Name: "{autodesktop}\Chatter Workbench"; Filename: "{app}\ChatterWorkbench.exe"; Tasks: desktopicon

[Run]
Filename: "{app}\ChatterWorkbench.exe"; Description: "Start Chatter Workbench"; Flags: nowait postinstall skipifsilent
