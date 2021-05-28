import os
import cv2

from ImageTrainClass import ImageTrainClass


class FileManager:

    #método que cuenta el número de archivos/carpetas que hay en un directorio.
    def countNumberOfFiles(self, path):
        listPath = os.listdir(path)
        print("Numero de archivos: " + str(len(listPath)))
        return len(listPath)

    def generateImagesList(self, path, numberTrainDirectories):

        trainImagesList = []
        for i in range(numberTrainDirectories):
            signClass = ('%02d' % i)
            pathCurrentDirectory = path + "\\" + signClass
            imagesDirectory = os.listdir(pathCurrentDirectory)
            print("----Directorio  " + signClass + "-------")
            for image in imagesDirectory:
                print("Readed image " + pathCurrentDirectory+"\\"+image)
                img = cv2.imread(pathCurrentDirectory+"\\"+image)
                imageTrain = ImageTrainClass(img, signClass)
                trainImagesList.append(imageTrain)


        return trainImagesList




