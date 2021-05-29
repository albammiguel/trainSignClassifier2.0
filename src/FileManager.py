import os
import cv2

from ImageTrainClass import ImageTrainClass
from ImageTestClass import ImageTestClass


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

    def generateImagesTestList(self, path):

        testImagesList = []
        listPath = os.listdir(path)
        len_path = len(listPath)
        for file in listPath[1:len_path]:
            print("Readed test image " + path + "\\" + file)
            img = cv2.imread(path+ "\\" + file)
            splitName = file.split("-")
            signClass = splitName[0]
            imageTest = ImageTestClass(img, signClass, file)
            testImagesList.append(imageTest)

        return testImagesList

    def generateResultFile(self, path, testImagesList):
            f = open(path, "w")

            for testImage in testImagesList:
                img_name = ImageTestClass.__getattribute__(testImage,'imgName')
                img_name = img_name.replace("-", "_")
                predictedClass = ImageTestClass.__getattribute__(testImage, 'predictedSignClass')
                text = img_name +"; "+ str(predictedClass) + "\n"
                f.write(text)

            f.close()





