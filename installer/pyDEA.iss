[Setup]
AppName=pyDEA
AppVersion=1.7
AppPublisher=pyDEA authors
DefaultDirName={autopf}\pyDEA
DefaultGroupName=pyDEA
UninstallDisplayIcon={app}\pyDEA.exe
OutputDir=..\dist
OutputBaseFilename=pyDEA-setup
Compression=lzma2
SolidCompression=yes
ArchitecturesInstallIn64BitMode=x64compatible
LicenseFile=..\License.txt
PrivilegesRequired=lowest
PrivilegesRequiredOverridesAllowed=dialog

[Tasks]
Name: "desktopicon"; Description: "Create a &desktop shortcut"; Flags: unchecked

[Files]
Source: "..\dist\pyDEA\*"; DestDir: "{app}"; Flags: recursesubdirs createallsubdirs ignoreversion

[Icons]
Name: "{group}\pyDEA"; Filename: "{app}\pyDEA.exe"
Name: "{autodesktop}\pyDEA"; Filename: "{app}\pyDEA.exe"; Tasks: desktopicon

[Run]
Filename: "{app}\pyDEA.exe"; Description: "Launch pyDEA"; Flags: nowait postinstall skipifsilent
