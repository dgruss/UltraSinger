@echo off
set "TL=D:\games\usdxwork\tools\UltraSinger\.venv\Lib\site-packages\torch\lib"
set PATH=%TL%;%PATH%
start cmd /k ".venv\Scripts\activate && cd src"