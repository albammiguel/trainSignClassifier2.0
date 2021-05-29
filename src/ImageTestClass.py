from ImageClass import ImageClass


class ImageTestClass(ImageClass):

    imgName = ''
    predictedSignClass = ''

    def __init__(self, img, signClass, imgName):
        ImageClass.__init__(self, img, signClass)
        self.imgName = imgName