from ImageClass import ImageClass


class ImageTrainClass(ImageClass):

    vector_caract = None

    def __init__(self, img, signClass, vector_caract):
        ImageClass.__init__(self, img, signClass)
        self.vector_caract = vector_caract
