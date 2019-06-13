from darkflow.net.build import TFNet
import cv2
import numpy as np

options = {"model": "cfg/yolov2.cfg", "load": "bin/yolov2.weights", "threshold": 0.1}

tfnet = TFNet(options)

def main():
    cap = cv2.VideoCapture(0)

    while(True):

        ret, frame = cap.read()
        #print(frame)
        #cv2.imshow('Stream IP Camera OpenCV', frame)
        result = tfnet.return_predict(frame)
        #print(result)
        newresult = boxing(frame, result)
        #print(newresult)
        cv2.imshow('Stream IP Camera OpenCV', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


def boxing(original_img, predictions):
    newImage = np.copy(original_img)

    for result in predictions:
        top_x = result['topleft']['x']
        top_y = result['topleft']['y']

        btm_x = result['bottomright']['x']
        btm_y = result['bottomright']['y']

        confidence = result['confidence']
        label = result['label'] + " " + str(round(confidence, 3))

        if confidence > 0.3:
            newImage = cv2.rectangle(newImage, (top_x, top_y), (btm_x, btm_y), (255, 0, 0), 3)
            newImage = cv2.putText(newImage, label, (top_x, top_y - 5), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.8,
                                   (0, 230, 0), 1, cv2.LINE_AA)


    return newImage

if __name__ == '__main__':
    main()