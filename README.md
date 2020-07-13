# MODELO SEIIHR

Modelo simple del tipo SEIR para transmisión de SARS-CoV-2. Basado en https://github.com/mrc-ide/squire.

## Configuración

Para correr los archivos de este repositorio es necesario una versión de Python 3 con una versión reciente de las librerías `datetime`, `numpy`, `pandas`, `plotly` y `scipy` instalada.

## Contenidos

1. El archivo `Metodologia.pdf` documenta la metodología del modelo.
2. El archivo `SEIIHR.py` contiene el código necesario para correr los escenarios.
3. El archivo `SEIIHR_Notebook.ipynb` es un jupyter notebook que contiene el código de los escenarios simulados con el módulo `SEIIHR.py`.
4. El archivo `Datos_Movimiento.xlsx` contiene el índice de movilidad utilizado para ajustar los modelos.
5. Los archivos `Escenario_[i].xlsx` son los resultados diarios de los escenarios simulados (ver archivo 3.).
