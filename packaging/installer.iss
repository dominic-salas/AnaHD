; Inno Setup 6 script: turns the standalone build (dist\AnaHD, from
; scripts/build_windows.py) into one installer:
;   iscc /DAppVersion=0.1.0 packaging\installer.iss
; (add /O<folder> to put the setup somewhere other than dist\).
; Installs per user by default (no admin rights needed), or for all users.

#ifndef AppVersion
  #define AppVersion "0.0.0"
#endif

[Setup]
AppId={{3C9E4A71-8B2D-4F6A-9E15-7D0B2A6C4E83}
AppName=AnaHD
AppVersion={#AppVersion}
AppPublisher=Dominic Salas
DefaultDirName={autopf}\AnaHD
DefaultGroupName=AnaHD
DisableProgramGroupPage=yes
LicenseFile=..\LICENSE
OutputDir=..\dist
OutputBaseFilename=AnaHD-{#AppVersion}-setup
SetupIconFile=..\build\windows\anahd.ico
UninstallDisplayIcon={app}\AnaHD.exe
; PyTorch's CUDA libraries are most of the size. Strongest settings: a GitHub release
; file must stay under 2 GiB (the folder is ~5 GB).
Compression=lzma2/ultra64
SolidCompression=yes
LZMAUseSeparateProcess=yes
LZMANumBlockThreads=8
DiskSpanning=no
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
MinVersion=10.0
PrivilegesRequired=lowest
PrivilegesRequiredOverridesAllowed=dialog
WizardStyle=modern

[Tasks]
Name: "desktopicon"; Description: "Create a desktop shortcut (drop recordings on it)"; GroupDescription: "Shortcuts:"

[Files]
Source: "..\dist\AnaHD\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs

[Icons]
Name: "{group}\AnaHD"; Filename: "{app}\AnaHD.exe"
Name: "{group}\Uninstall AnaHD"; Filename: "{uninstallexe}"
Name: "{autodesktop}\AnaHD"; Filename: "{app}\AnaHD.exe"; Tasks: desktopicon

[Run]
Filename: "{app}\AnaHD.exe"; Description: "Start AnaHD"; Flags: nowait postinstall skipifsilent

[Code]
// AnaHD keeps its model cache in the user's local app-data folder. Offer to remove it;
// never touch anything else (recordings and cleaned clips stay where they are).
procedure CurUninstallStepChanged(CurUninstallStep: TUninstallStep);
var
  Data: String;
begin
  if CurUninstallStep = usPostUninstall then
  begin
    Data := ExpandConstant('{localappdata}\AnaHD');
    if DirExists(Data) and (not UninstallSilent) then
      if MsgBox('Also remove AnaHD''s model cache (' + Data + ')? ' +
                'Your videos are not affected.', mbConfirmation, MB_YESNO) = IDYES then
        DelTree(Data, True, True, True);
  end;
end;
