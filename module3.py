#..........................................Testing(prediction).........................................#

import numpy as np
import argparse
import pickle
import cv2
import os
import time
from keras.models import load_model
from collections import deque


def main(video):
        ap = argparse.ArgumentParser()
        ap.add_argument("--model", default = 'model/activity.model',
                help="path to trained serialized model")
        ap.add_argument("--label-bin", default ='model/lb.pickle',
                help="path to  label binarizer")
        
        ap.add_argument('--input', default = video)
        
        ap.add_argument("--output", default = 'output/result.png',                
                help="path to our output video")
        ap.add_argument("-s", "--size", type=int, default=128,
                help="size of queue for averaging")
        args = vars(ap.parse_args())

        print("Loading model and label binarizer...")
        model = load_model(args["model"])
        lb = pickle.loads(open(args["label_bin"], "rb").read())

        # predictions queue
        mean = np.array([123.68, 116.779, 103.939][::1], dtype="float32")
        Q = deque(maxlen=args["size"])

        vs = cv2.VideoCapture(args["input"])
        writer = None
        (W, H) = (None, None)

        while True:
                (grabbed, frame) = vs.read()
                if not grabbed:
                        break

                # if the frame dimensions are empty, grab them
                if W is None or H is None:
                        (H, W) = frame.shape[:2]

                output = frame.copy()
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = cv2.resize(frame, (224, 224)).astype("float32")
                frame -= mean

                preds = model.predict(np.expand_dims(frame, axis=0))[0]
                Q.append(preds)

                results = np.array(Q).mean(axis=0)
                i = np.argmax(results)
                label = lb.classes_[i]

                # draw the activity on the output frame
                text = "Detected: {}".format(label)
                print('prediction:',text)    
                file = open("output.txt",'w')
                file.write(text)
                file.close()
                cv2.putText(output, text, (35, 50), cv2.FONT_HERSHEY_SIMPLEX,
                        1.25, (0, 255, 0), 5)
   
                if writer is None:
                        fourcc = cv2.VideoWriter_fourcc(*"MJPG")
                        writer = cv2.VideoWriter(args["output"], fourcc, 30,
                                (W, H), True)

                # write the output frame to disk
                writer.write(output)

                # show the output image
                cv2.imshow("Output", output)
                key = cv2.waitKey(1) & 0xFF

                if key == ord("q"):
                        break

        print("Cleaning up...")
        writer.release()
        vs.release()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    TestData="Input"
    while True:
        for(direcpath,direcnames,files) in os.walk(TestData):
            for file in files:
                if 'png' in file or 'jpg' in file:
                    print(file)
                    time.sleep(1)
                    filename= TestData +'/'+ file
                    clf = main(filename)
                    print('Filename:',filename)    
                    os.remove(filename)

