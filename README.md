# RNA - Descripción
## Archivos
El archivo *Observaciones.xlsx* contiene los datos de interés de todos los experimentos realizados.

El archivo *data_format.py* es usado para preprocesar y formatear los datos para que la red neuronal pueda utilizarlos.

El archivo *train_network.py* entrena a una red neuronal de la clase especificada y gurada información pertinente de esta.

El archivo *ann_log.txt* presenta el porcentaje de aciertos en el conjunto de validación con el paso de las épocas al entrenar la red neuronal.

El archivo *load_network.py* carga la red neuronal entrenada y la información asociada a esta, también muestra gráficamente los resultados así como el porcentaje de aciertos en el conjunto de prueba (solamente usar esta última funcionalidad con la red neuronal final).

## Evidencias
La carpeta *Evidencias* contiene las capturas de pantalla y gráficas generadas durtante los experimentos. Está dividida en tres subcarpetas correspondientes a los experimentos hechos con la red neuronal básica, la red neuronal con backpropagation con inercia y a la red nuronal con cross-entropy, respectivamente.

## data
La carpeta *data* contiene los datos utilizados y generados en los experimentos.
- preprocessed: contiene los conjuntos de entrenamiento, validación y prueba
- mnist.pkl.gz: datos comprimidos
- network_data.pkl: archivo generado al entrenar la red neuronal, contiene los valores de los pesos y biases
- training_info.csv: contiene datos relacionados con el entrenamiento de la red (costo y porcentaje de aciertos en el conjunto de validación)

## src
La carpeta *src* contiene la implementación de la red neuronal y las modificaiones de esta, así como funciones auxiliares para graficar, preprocesar los datos y funciones para carga y guardado de datos.