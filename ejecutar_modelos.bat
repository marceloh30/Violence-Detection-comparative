@echo off
REM Script para ejecutar los 4 modelos de clasificación de vídeo secuencialmente.

echo ###################################################
echo # Iniciando ejecución del modelo I3D #
echo ###################################################
C:/Users/Chelo/AppData/Local/Microsoft/WindowsApps/python3.12.exe main_I3D.py
echo.
echo --- Modelo I3D completado ---
echo.

echo ###################################################
echo # Iniciando ejecución del modelo TSM #
echo ###################################################
C:/Users/Chelo/AppData/Local/Microsoft/WindowsApps/python3.12.exe main_TSM.py
echo.
echo --- Modelo TSM completado ---
echo.

echo ###################################################
echo # Iniciando ejecución del modelo SlowFast #
echo ###################################################
C:/Users/Chelo/AppData/Local/Microsoft/WindowsApps/python3.12.exe main_SlowFast.py
echo.
echo --- Modelo SlowFast completado ---
echo.

echo ###################################################
echo # Iniciando ejecución del modelo ViViT #
echo ###################################################
C:/Users/Chelo/AppData/Local/Microsoft/WindowsApps/python3.12.exe main_ViViT.py
echo.
echo --- Modelo ViViT completado ---
echo.

echo ###################################################
echo # Todas las ejecuciones de modelos han finalizado. #
echo ###################################################

pause