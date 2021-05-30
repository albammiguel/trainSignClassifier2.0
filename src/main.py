import argparse

import cv2
import cv2.ml
import numpy
import numpy as np
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import classification_report, plot_confusion_matrix, confusion_matrix
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import NearestCentroid, KNeighborsClassifier

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


def printConfussionMatrix(detector, testData, realClasses):
    fig, ax = plt.subplots(figsize=(40, 40))
    plot_confusion_matrix(detector, testData, realClasses, cmap=plt.cm.Blues, ax=ax)
    if args.classifier[1] == "BAYES":
        plt.title('Matriz de confusion clasificador Bayes sklearn')
    elif args.classifier[1] == "EUCLIDEAN":
        plt.title('Matriz de confusión clasificador Euclídeo')
    elif args.classifier[1] == "KNN":
        plt.title('Matriz de confusión clasificador KNN')
    plt.show()


def calculateCrossValidation(detector, totalData, totalClasses):

    cv_data = cross_val_score(detector, totalData, totalClasses, cv=5, n_jobs=2)
    print("Varianza: %0.4f (+/- %0.4f)" % (cv_data.mean(), cv_data.std()*2))






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

    args.classifier = args.classifier.split("-")

    # Crear el clasificador
    if args.classifier[1] == "BAYES":
        #detector = cv2.ml.NormalBayesClassifier_create()
        detector = LinearDiscriminantAnalysis()
    elif args.classifier[1] == "EUCLIDEAN":
        detector = NearestCentroid(metric='euclidean')
    elif args.classifier[1] == "KNN":
        detector = KNeighborsClassifier(n_neighbors=3)
    else:
        raise ValueError('Tipo de clasificador incorrecto')

    # Entrenar el clasificador si es necesario ...
    if args.classifier[0] == "LDA":
        if args.classifier[1] != "BAYES":
            lda = LinearDiscriminantAnalysis()
            lda.fit(trainingData, classesList)
            trainingData = lda.transform(trainingData)
        """
        lda = LinearDiscriminantAnalysis()
        lda.fit(trainingData, classesList)
        trainingData = lda.transform(trainingData)"""

    elif args.classifier[0] == "PCA":
        pca = PCA()
        pca.fit(trainingData, classesList)
        trainingData = pca.transform(trainingData)

    detector.fit(trainingData, classesList)
    #detector.train(numpy.float32(trainingData), cv2.ml.ROW_SAMPLE, numpy.int32(classesList))


    # Cargar y procesar imgs de test
    testImagesList = fileManager.generateImagesTestList(args.test_path)

    # Guardar los resultados en ficheros de texto (en el directorio donde se 
    # ejecuta el main.py) tal y como se pide en el enunciado.
    testData, realClasses = processData(testImagesList)


    if args.classifier[0] == "LDA":
        if args.classifier[1] != "BAYES":
            testData = lda.transform(testData)

        #testData = lda.transform(testData)

    elif args.classifier[0] == "PCA":
        testData = pca.transform(testData)


    predictedClasses = detector.predict(testData)
    #_, predictedClasses = detector.predict(numpy.float32(testData))

    savePredictedLabels(testImagesList, predictedClasses)
    fileManager.generateResultFile("resultado.txt", testImagesList)

    #realClasses = numpy.int32(realClasses)
    print(classification_report(realClasses, predictedClasses))

    #Tasa de acierto
    """tasa_acierto = numpy.sum(1 * (realClasses.reshape((realClasses.shape[0], 1)) == predictedClasses)) / \
                   realClasses.shape[0]"""
    tasa_acierto = numpy.sum(1*(realClasses == predictedClasses))/realClasses.shape[0]
    print("La tasa de acierto es: ", format(tasa_acierto*100, ".2f"), "%")


    #Matríz de confusion
    #cfm = confusion_matrix(realClasses, predictedClasses)
    printConfussionMatrix(detector, testData, realClasses)


    #validacionCruzada
    # https://cristianrohr.github.io/datascience/python/machine%20learning/im%C3%A1genes/deteccion-peatones/
    totalData = np.concatenate((trainingData, testData), axis=0)
    totalClasses = np.concatenate((classesList, realClasses))

    calculateCrossValidation(detector, totalData, totalClasses)

















