from ImageClass import ImageClass


class ImageTestClass(ImageClass):

    imgName = ''
    predictedSignClass = ''

    def __init__(self, img, signClass, imgName, predictedSignClass):
        ImageClass.__init__(self, img, signClass)
        self.imgName = imgName
        self.predictedSignClass = predictedSignClass