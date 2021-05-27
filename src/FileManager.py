import os
import cv2

class FileManager:

    #método que cuenta el número de archivos/carpetas que hay en un directorio.
    def countNumberOfFiles(self, path):
        listPath = os.listdir(path)
        print("Numero de archivos: " + str(len(listPath)))
        return len(listPath)

    def generateDirectoriesList(self, path, numberTrainDirectories):

        trainDirectoriesArray = [0] * numberTrainDirectories
        for i in range(numberTrainDirectories):
            trainDirectoriesArray[i] = path + "\\" + ('%02d' % i)
            print("Guardado nombre directorio: " + trainDirectoriesArray[i])

        return trainDirectoriesArray

