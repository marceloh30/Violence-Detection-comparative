@echo off
REM Script para ejecutar los 4 modelos de clasificacion de video secuencialmente.

echo =====================================================================
echo == Seleccion del Dataset para Entrenamiento ==
echo =====================================================================
echo.

:askDataset
SET /P TRAIN_DATASET_INPUT="Por favor, ingrese el dataset para entrenar al modelo ('rwf2000' o 'rlvs'): "

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
echo Dataset de entrenamiento seleccionado correctamente: %TRAIN_DATASET_INPUT%
echo.
echo.
echo Desea ejecutar un modelo en particular? En caso contrario se van a ejecutar un modelo tras otro
:askModels
SET /P MODEL_SELECTION="Escriba un modelo ('i3d', 'slowfast', 'tsm' o 'vivit') :"

REM Validar la entrada
IF /I "%MODEL_SELECTION%"=="i3d" GOTO :i3d
IF /I "%MODEL_SELECTION%"=="slowfast" GOTO :slowfast
IF /I "%MODEL_SELECTION%"=="tsm" GOTO :tsm
IF /I "%MODEL_SELECTION%"=="vivit" GOTO :vivit

echo.
echo No se ha elegido un modelo. Se procede a ejecutar los modelos secuencialmente.
echo.


echo.
echo =====================================================================
echo == Iniciando ejecuciones para el dataset de entrenamiento: %TRAIN_DATASET_INPUT% ==
echo =====================================================================
echo.

:i3d
echo ###################################################
echo # Iniciando ejecucion del modelo I3D #
echo ###################################################
C:/Users/Chelo/AppData/Local/Microsoft/WindowsApps/python3.12.exe -m violence_detection.main i3d %TRAIN_DATASET_INPUT%
echo.
echo --- Modelo I3D completado ---
echo.
IF /I "%MODEL_SELECTION%"=="i3d" GOTO :end

:tsm
echo ###################################################
echo # Iniciando ejecucion del modelo TSM #
echo ###################################################
C:/Users/Chelo/AppData/Local/Microsoft/WindowsApps/python3.12.exe -m violence_detection.main tsm %TRAIN_DATASET_INPUT%
echo.
echo --- Modelo TSM completado ---
echo.
IF /I "%MODEL_SELECTION%"=="tsm" GOTO :end

:slowfast
echo ###################################################
echo # Iniciando ejecucion del modelo SlowFast #
echo ###################################################
C:/Users/Chelo/AppData/Local/Microsoft/WindowsApps/python3.12.exe -m violence_detection.main slowfast %TRAIN_DATASET_INPUT%
echo.
echo --- Modelo SlowFast completado ---
echo.
IF /I "%MODEL_SELECTION%"=="slowfast" GOTO :end

:vivit
echo ###################################################
echo # Iniciando ejecucion del modelo ViViT #
echo ###################################################
C:/Users/Chelo/AppData/Local/Microsoft/WindowsApps/python3.12.exe -m violence_detection.main vivit %TRAIN_DATASET_INPUT%
echo.
echo --- Modelo ViViT completado ---
echo.
IF /I "%MODEL_SELECTION%"=="vivit" GOTO :end

echo ###################################################
echo # Todas las ejecuciones de modelos han finalizado. #
echo ###################################################

:end
pause