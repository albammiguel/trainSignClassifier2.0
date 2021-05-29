import argparse
import math

import cv2
import cv2.ml
import numpy
from matplotlib import pyplot as plt
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import classification_report, confusion_matrix, plot_confusion_matrix

from FileManager import FileManager
from ImageClass import ImageClass
from ImageTestClass import ImageTestClass


def processImage(image):
    image_grey = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    equalize_image = cv2.equalizeHist(image_grey)
    resized_image = cv2.resize(equalize_image, (30, 30), interpolation=cv2.INTER_AREA)

    return resized_image

#https://stackoverflow.com/questions/44972099/opencv-hog-features-explanation
def createHogDescriptor(image):
    cell_size = (5, 5)  # h x w in pixels
    block_size = (3, 3)  # h x w in cells
    nbins = 9  # number of orientation bins

    hog = cv2.HOGDescriptor(_winSize=(image.shape[1] // cell_size[1] * cell_size[1],
                                      image.shape[0] // cell_size[0] * cell_size[0]),
                            _blockSize=(block_size[1] * cell_size[1],
                                        block_size[0] * cell_size[0]),
                            _blockStride=(cell_size[1], cell_size[0]),
                            _cellSize=(cell_size[1], cell_size[0]),
                            _nbins=nbins)

    return hog

#Parte del siguiente código está extraído de la siguiente página:
#https://cristianrohr.github.io/datascience/python/machine%20learning/im%C3%A1genes/deteccion-peatones/
def processData(imagesList):

    data = numpy.array([])
    classes = numpy.array([])

    for image in imagesList:
        processedImage = processImage(ImageClass.__getattribute__(image, 'img'))
        hog = createHogDescriptor(processedImage)
        v = hog.compute(processedImage)
        v2 = v.ravel()
        ImageClass.__setattr__(image, 'vector_caract', v2)
        data = numpy.hstack((data, v2))
        classes = numpy.hstack((classes, numpy.array(ImageClass.__getattribute__(image, 'signClass'))))

    data = data.reshape((len(imagesList), len(v2)))

    return data, classes


def savePredictedLabels(testImagesList, predictedClasses):
    for i in range(len(testImagesList)):
        ImageTestClass.__setattr__(testImagesList[i], 'predictedSignClass', predictedClasses[i])


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


    # Crear el clasificador
    if args.classifier == "BAYES":
        detector = LinearDiscriminantAnalysis()
    else:
        raise ValueError('Tipo de clasificador incorrecto')

    # Entrenar el clasificador si es necesario ...
    detector.fit(trainingData, classesList)


    # Cargar y procesar imgs de test
    testImagesList = fileManager.generateImagesTestList(args.test_path)

    # Guardar los resultados en ficheros de texto (en el directorio donde se 
    # ejecuta el main.py) tal y como se pide en el enunciado.
    testData, realClasses = processData(testImagesList)

    predictedClasses = detector.predict(testData)

    savePredictedLabels(testImagesList, predictedClasses)
    fileManager.generateResultFile("resultado.txt", testImagesList)


    print(classification_report(realClasses, predictedClasses))


    tasa_acierto = numpy.sum(1*(realClasses == predictedClasses))/realClasses.shape[0]
    print("La tasa de acierto es: ", format(tasa_acierto*100, ".2f"), "%")



    fig, ax = plt.subplots(figsize=(40, 40))
    plot_confusion_matrix(detector, testData, realClasses, cmap=plt.cm.Blues, ax=ax)
    plt.title('Matriz de confusion clasificador Bayes sklearn')
    plt.show()






