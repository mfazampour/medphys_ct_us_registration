import imfusion


class MyAlgorithm(imfusion.Algorithm):
    def __init__(self, imageset):
        super().__init__()
        self.data_list = ['C:/Users/maria/OneDrive/Desktop/txt_files/spine9_vert5_11_0.txt',
                          'C:/Users/maria/OneDrive/Desktop/txt_files/spine9_vert1_11_0.txt']

        self.add_param('save_path', "")

        self.input_pc = imageset[0]

        self.idx = 0

    @classmethod
    def convert_input(cls, data):
        if len(data) == 1:
            return data
        raise IncompatibleError('Requires no input data')


    def find_connections(self, line, pc):
        return (0, 0)

    def compute(self):

        am = imfusion.imfusion.app.annotationModel()
        annotations = am.annotations()
        a = annotations[0]

        print(self.input_pc)

        # todo: extract point cloud from annotations - if more than one, use all in the find_connections
        pc = None

        for annotation in annotations:
            if annotation.type == imfusion.app.AnnotationType.Line:
                self.find_connections(annotation, pc)

        #imfusion.app.open(self.data_list[self.idx])
        imfusion.app.open("C:/Users/maria/OneDrive/Desktop/txt_files/tmp.iws")
        self.idx += 1


imfusion.registerAlgorithm('My Algorithm', MyAlgorithm)