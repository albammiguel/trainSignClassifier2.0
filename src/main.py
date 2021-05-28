import argparse

import cv2
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as lda

from FileManager import FileManager
from ImageTrainClass import ImageTrainClass


def processData(trainImagesList):
    for trainImage in trainImagesList:
        image = ImageTrainClass.__getattribute__(trainImage, 'img')
        image_grey = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        equalize_image = cv2.equalizeHist(image_grey)
        resized_image = cv2.resize(equalize_image, (30, 30), interpolation=cv2.INTER_AREA)
        cell_size = (4, 4)  # h x w in pixels
        block_size = (2, 2)  # h x w in cells
        nbins = 9  # number of orientation bins

        hog = cv2.HOGDescriptor(_winSize=(resized_image.shape[1] // cell_size[1] * cell_size[1],
                                          resized_image.shape[0] // cell_size[0] * cell_size[0]),
                                _blockSize=(block_size[1] * cell_size[1],
                                            block_size[0] * cell_size[0]),
                                _blockStride=(cell_size[1], cell_size[0]),
                                _cellSize=(cell_size[1], cell_size[0]),
                                _nbins=nbins)
        descriptor = hog.compute(resized_image)
        ImageTrainClass.__setattr__(trainImage, 'vector_caract', descriptor)



if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description='Entrena sober train y ejecuta el clasificador sobre imgs de test')
    parser.add_argument(
        '--train_path', type=str, default="./train", help='Path al directorio de imgs de train')
    parser.add_argument(
        '--test_path', type=str, default="./test", help='Path al directorio de imgs de test')
    parser.add_argument(
        '--classifier', type=str, default="BAYES", help='String con el nombre del clasificador')

    args = parser.parse_args()

    # Cargar los datos de entrenamiento
    fileManager = FileManager()
    numberTrainDirectories = fileManager.countNumberOfFiles(args.train_path)
    trainImagesList = fileManager.generateImagesList(args.train_path, numberTrainDirectories)

    #Tratamiento de los datos
    processData(trainImagesList)

    # Crear el clasificador 
    if args.classifier == "BAYES":
        #detector = ...
        None
    else:
        raise ValueError('Tipo de clasificador incorrecto')

    # Entrenar el clasificador si es necesario ...
    # detector ...

    # Cargar y procesar imgs de test 
    # args.train_path ...

    # Guardar los resultados en ficheros de texto (en el directorio donde se 
    # ejecuta el main.py) tal y como se pide en el enunciado.





