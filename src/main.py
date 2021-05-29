import argparse

import cv2
import numpy
from sklearn.discriminant_analysis import  LinearDiscriminantAnalysis

from FileManager import FileManager
from ImageTrainClass import ImageTrainClass


def processImage(image):
    image_grey = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    equalize_image = cv2.equalizeHist(image_grey)
    resized_image = cv2.resize(equalize_image, (30, 30), interpolation=cv2.INTER_AREA)

    return resized_image

#Parte del siguiente código está extraído de la siguiente página:
#https://cristianrohr.github.io/datascience/python/machine%20learning/im%C3%A1genes/deteccion-peatones/
def processData(trainImagesList):

    trainingData = numpy.array([])
    classes = numpy.array([])

    for trainImage in trainImagesList:
        processedImage = processImage(ImageTrainClass.__getattribute__(trainImage, 'img'))
        cell_size = (5, 5)  # h x w in pixels
        block_size = (3, 3)  # h x w in cells
        nbins = 9  # number of orientation bins

        hog = cv2.HOGDescriptor(_winSize=(processedImage.shape[1] // cell_size[1] * cell_size[1],
                                          processedImage.shape[0] // cell_size[0] * cell_size[0]),
                                _blockSize=(block_size[1] * cell_size[1],
                                            block_size[0] * cell_size[0]),
                                _blockStride=(cell_size[1], cell_size[0]),
                                _cellSize=(cell_size[1], cell_size[0]),
                                _nbins=nbins)

        v = hog.compute(processedImage)
        v2 = v.ravel()
        ImageTrainClass.__setattr__(trainImage, 'vector_caract', v2)
        trainingData = numpy.hstack((trainingData, v2))
        classes = numpy.hstack((classes, numpy.array(ImageTrainClass.__getattribute__(trainImage, 'signClass'))))

    trainingData = trainingData.reshape((len(trainImagesList), len(v2)))

    return trainingData, classes


def test(imagen, clasificador):

    img = cv2.imread(imagen, cv2.IMREAD_COLOR)
    processedImage = processImage(img)
    cell_size = (5, 5)  # h x w in pixels
    block_size = (3, 3)  # h x w in cells
    nbins = 9  # number of orientation bins

    hog = cv2.HOGDescriptor(_winSize=(processedImage.shape[1] // cell_size[1] * cell_size[1],
                                      processedImage.shape[0] // cell_size[0] * cell_size[0]),
                            _blockSize=(block_size[1] * cell_size[1],
                                        block_size[0] * cell_size[0]),
                            _blockStride=(cell_size[1], cell_size[0]),
                            _cellSize=(cell_size[1], cell_size[0]),
                            _nbins=nbins)

    v = hog.compute(processedImage)
    h2 = v.reshape((1, -1))
    return (clasificador.predict(h2))



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
    trainingData, classesList = processData(trainImagesList)

    lda = LinearDiscriminantAnalysis()

    lda.fit(trainingData, classesList)

    EXAMPLE= "C:\\Users\\albam\\Desktop\\URJC\\SEGUNDO_CUATRIMESTRE\\VISION_ARTIFICIAL\\PRACTICA_2\\test_reconocimiento\\38-00012.ppm"
    res = test(EXAMPLE, lda)
    print(EXAMPLE, " fue clasificado como: ", res)

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





