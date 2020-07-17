from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.applications.resnet_v2 import preprocess_input
from keras.applications.resnet_v2 import decode_predictions
from keras.applications.resnet_v2 import ResNet101V2
from keras.applications.vgg16 import preprocess_input
from keras.applications.vgg16 import decode_predictions
from keras.applications.vgg16 import VGG16
from keras.applications.vgg19 import preprocess_input
from keras.applications.vgg19 import decode_predictions
from keras.applications.vgg19 import VGG19
from keras.applications.resnet50 import preprocess_input
from keras.applications.resnet50 import decode_predictions
from keras.applications.resnet50 import ResNet50
from keras.applications.mobilenet import preprocess_input
from keras.applications.mobilenet import decode_predictions
from keras.applications.mobilenet import MobileNet
# load the model
class Landmark:
    def DefineLandmark(self):
        model = MobileNet()
        # load an image from file
        image = load_img('birthday-cake-600x600.jpg', target_size=(224, 224))
        # convert the image pixels to a numpy array
        image = img_to_array(image)
        # reshape data for the model
        image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
        # prepare the image for the VGG model
        image = preprocess_input(image)
        # predict the probability across all output classes
        yhat = model.predict(image)
        # convert the probabilities to class labels
        label = decode_predictions(yhat)
        # retrieve the most likely result, e.g. highest probability
        label2 = label[0]
        print(type(label2))
        for i in label2:
            print('%s (%.2f%%)' % (i[1], i[2] * 100))
        return label2

ob = Landmark().DefineLandmark()

"""for eachObject in label:
    print(eachObject["name"] , " : " , eachObject["percentage_probability"] )"""

"""label = label[0][0]
print(label)
# print the classification
print('%s (%.2f%%)' % (label[1], label[2]*100))"""
