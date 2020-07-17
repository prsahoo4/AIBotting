from imageai.Prediction.Custom import CustomImagePrediction
from imageai.Detection import ObjectDetection
import os



class Object:
    def ObjectDetect(self):
        execution_path = "C:\Tensorflow\models\Research\object_detection\Engine\customPrediction"
        print(execution_path)
        detector = ObjectDetection()
        detector.setModelTypeAsRetinaNet()
        detector.setModelPath(os.path.join(execution_path, "resnet50_coco_best_v2.0.1.h5"))
        detector.loadModel()
        detections1 = detector.detectCustomObjectsFromImage(input_image=os.path.join(execution_path, "trail1.jpg"),
                                                            output_image_path=os.path.join(execution_path,
                                                                                           "example3.jpg"))

        detector2 = ObjectDetection()
        detector2.setModelTypeAsYOLOv3()
        detector2.setModelPath(os.path.join(execution_path, "yolo.h5"))
        detector2.loadModel()
        detections2 = detector2.detectCustomObjectsFromImage(input_image=os.path.join(execution_path, "trail1.jpg"),
                                                             output_image_path=os.path.join(execution_path,
                                                                                            "example4.jpg"))

        detector3 = ObjectDetection()
        detector3.setModelTypeAsTinyYOLOv3()
        detector3.setModelPath(os.path.join(execution_path, "yolo-tiny.h5"))
        detector3.loadModel()
        detections3 = detector3.detectCustomObjectsFromImage(input_image=os.path.join(execution_path, "trail1.jpg"),
                                                             output_image_path=os.path.join(execution_path,
                                                                                            "example5.jpg"))

        prediction = CustomImagePrediction()
        prediction.setModelTypeAsResNet()
        prediction.setModelPath(os.path.join(execution_path, "model_ex-027_acc-0.843750.h5"))
        prediction.setJsonPath(os.path.join(execution_path, "model_class.json"))
        prediction.loadModel(num_objects=2)
        predictions, probabilities = prediction.predictImage(os.path.join(execution_path, "trail1.jpg"),
                                                             result_count=5)

        detections = detections1 + detections2 + detections3
        List = []
        for i in detections:
            List.append(i["name"])

        for eachPrediction,eachProbability in zip(predictions, probabilities):
            if eachProbability > 50:
                List.append(eachPrediction)



        """for eachObject in detections:
            print(eachObject["name"], " : ", eachObject["percentage_probability"])

        for eachPrediction, eachProbability in zip(predictions, probabilities):
            print(eachPrediction, " : ", eachProbability)"""
        return List

"""ob= Object()
print(ob.ObjectDetect())"""
"""for eachObject in detections:
    print(eachObject["name"] , " : " , eachObject["percentage_probability"] )


for eachPrediction, eachProbability in zip(predictions, probabilities):
    print(eachPrediction , " : " , eachProbability)"""