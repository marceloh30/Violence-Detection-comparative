@echo off
REM Script para ejecutar los 4 modelos de clasificación de vídeo secuencialmente.

echo =====================================================================
echo == Selección del Dataset para Entrenamiento ==
echo =====================================================================
echo.

:askDataset
SET /P TRAIN_DATASET_INPUT="Por favor, ingrese el dataset para entrenar ('rwf2000' o 'rlvs'): "

REM Validar la entrada
IF /I "%TRAIN_DATASET_INPUT%"=="rwf2000" GOTO :datasetOk
IF /I "%TRAIN_DATASET_INPUT%"=="rlvs" GOTO :datasetOk

echo.
echo ERROR: Dataset '%TRAIN_DATASET_INPUT%' no es valido.
echo Por favor, ingrese 'rwf2000' o 'rlvs'.
echo.
GOTO :askDataset

:datasetOk
echo.
echo =====================================================================
echo == Iniciando ejecuciones para el dataset de entrenamiento: %TRAIN_DATASET_INPUT% ==
echo =====================================================================
echo.

echo ###################################################
echo # Iniciando ejecución del modelo I3D #
echo ###################################################
C:/Users/Chelo/AppData/Local/Microsoft/WindowsApps/python3.12.exe main_I3D.py %TRAIN_DATASET_INPUT%
echo.
echo --- Modelo I3D completado ---
echo.

echo ###################################################
echo # Iniciando ejecución del modelo TSM #
echo ###################################################
C:/Users/Chelo/AppData/Local/Microsoft/WindowsApps/python3.12.exe main_TSM.py %TRAIN_DATASET_INPUT%
echo.
echo --- Modelo TSM completado ---
echo.

echo ###################################################
echo # Iniciando ejecución del modelo SlowFast #
echo ###################################################
C:/Users/Chelo/AppData/Local/Microsoft/WindowsApps/python3.12.exe main_SlowFast.py %TRAIN_DATASET_INPUT%
echo.
echo --- Modelo SlowFast completado ---
echo.

echo ###################################################
echo # Iniciando ejecución del modelo ViViT #
echo ###################################################
C:/Users/Chelo/AppData/Local/Microsoft/WindowsApps/python3.12.exe main_ViViT.py %TRAIN_DATASET_INPUT%
echo.
echo --- Modelo ViViT completado ---
echo.

echo ###################################################
echo # Todas las ejecuciones de modelos han finalizado. #
echo ###################################################

pause