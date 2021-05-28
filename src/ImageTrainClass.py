from ImageClass import ImageClass


class ImageTrainClass(ImageClass):

    vector_caract = None

    def __init__(self, img, signClass):
        ImageClass.__init__(self, img, signClass)
