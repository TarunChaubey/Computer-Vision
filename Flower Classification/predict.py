import numpy as np
from keras.models import load_model
from keras.preprocessing import image


class multiclass:
    def __init__(self, filename):
        self.filename = filename

    def predictionmulticlass(self):
        # load model
        model = load_model('Multiclass_CNN.h5')

        # summarize model
        # model.summary()
        imagename = self.filename
        test_image = image.load_img(imagename, target_size=(64, 64))
        test_image = image.img_to_array(test_image)
        test_image = np.expand_dims(test_image, axis=0)
        result = model.predict(test_image)
        

        if np.argmax(result) == 0:
            prediction = 'daisy'
            print(prediction)
            return [{"image": prediction}]
        elif np.argmax(result) == 1:
            prediction = 'dandelion'
            return [{"image": prediction}]
        elif np.argmax(result) == 2:
            prediction = 'rose'
            return [{"image": prediction}]
        elif np.argmax(result) == 3:
            prediction = 'sunflower'
            return [{"image": prediction}]
        else:
            prediction = 'tulip'
            return [{"image": prediction}]
