@echo off
setlocal
title Murmur
cd /d "%~dp0"

rem ---- Optional settings: remove "rem" in front of a line to use it ----------
rem "Precise" mode uses the turbo model: about half the memory, much faster, nearly as accurate
rem (not for "Transcribe to English"). For the full model use: set MURMUR_PRECISE_MODEL=large-v3
set MURMUR_PRECISE_MODEL=large-v3-turbo
rem Force the processor even if an NVIDIA card is present:
rem set MURMUR_DEVICE=cpu
rem Speaker recognition (also needs: .venv\Scripts\pip install pyannote.audio):
rem set HF_TOKEN=hf_your_token_here
rem -----------------------------------------------------------------------------

rem Models you downloaded yourself into the folder "models" are used instead of downloading them:
rem   models\sprint, models\steady, models\precise  (each one a complete model folder)
if exist "%~dp0models\sprint" set "MURMUR_SPRINT_MODEL=%~dp0models\sprint"
if exist "%~dp0models\steady" set "MURMUR_STEADY_MODEL=%~dp0models\steady"
if exist "%~dp0models\precise" set "MURMUR_PRECISE_MODEL=%~dp0models\precise"

if not defined MURMUR_PORT set MURMUR_PORT=8000
rem The server opens the browser itself once it is really ready.
set MURMUR_OPEN_BROWSER=1

if exist ".venv\Scripts\python.exe" goto run

echo [murmur] First start: installing the Python packages. This takes a few minutes.
set PY=python
where py >nul 2>nul && set PY=py -3
%PY% -m venv .venv
if errorlevel 1 goto nopython
".venv\Scripts\python.exe" -m pip install --upgrade pip
".venv\Scripts\python.exe" -m pip install -r requirements.txt
if errorlevel 1 goto failed

:run
echo [murmur] Starting on http://localhost:%MURMUR_PORT% - close this window to stop Murmur.
".venv\Scripts\python.exe" -m uvicorn server:app --host 127.0.0.1 --port %MURMUR_PORT% --no-access-log
if errorlevel 1 goto crashed
goto end

:nopython
echo.
echo [murmur] Python was not found. Install Python 3.10 or newer from
echo          https://www.python.org/downloads/ and tick "Add python.exe to PATH",
echo          then start this file again.
goto stop

:failed
echo.
echo [murmur] Installing the packages failed. Check the internet connection
echo          and start this file again.
rmdir /s /q .venv
goto stop

:crashed
echo.
echo [murmur] The server stopped with an error, see the lines above.
echo          "address already in use" means Murmur is already running:
echo          keep using the browser tab at http://localhost:%MURMUR_PORT%

:stop
pause

:end
endlocal
