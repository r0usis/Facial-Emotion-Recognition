import os
import sys
import cv2
import glob
import random
import math
import numpy as np
import csv
import dlib
import itertools
import pylab as pl

from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix

emotions = ["disgust", "happiness", "neutral"]  # Emotion list, before len(emotions) = 8, then len(emotions) = 7
fishface = cv2.face.FisherFaceRecognizer_create()
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")  # Or set this to whatever you named the downloaded file
clf = SVC(kernel='linear', probability=True, tol=1e-3, verbose = True) #Set the classifier as a support vector machines with polynomial kernel
data = {}  # Make dictionary for all values


def get_files(emotion):  # Define function to get file list, randomly shuffle it and split 80/20
    files = glob.glob("dataset/%s/*" % emotion)
    # fotos = glob.glob("frames/video%s/*.png" %numerovideo)
    random.shuffle(files)
    training = files[:int(len(files)*0.8)]
    prediction = fotos
    return training, prediction, files


def get_landmarks(image):
    detections = detector(image, 1)
    for k, d in enumerate(detections):  # For all detected face instances individually
        shape = predictor(image, d)  # Draw Facial Landmarks with the predictor class
        xlist = []
        ylist = []
        for i in range(1, 68):  # Store X and Y coordinates in two lists
            xlist.append(float(shape.part(i).x))
            ylist.append(float(shape.part(i).y))
        xmean = np.mean(xlist)
        ymean = np.mean(ylist)
        xcentral = [(x - xmean) for x in xlist]
        ycentral = [(y - ymean) for y in ylist]
        landmarks_vectorised = []
        for x, y, w, z in zip(xcentral, ycentral, xlist, ylist):
            landmarks_vectorised.append(w)
            landmarks_vectorised.append(z)
            meannp = np.asarray((ymean, xmean))
            coornp = np.asarray((z, w))
            dist = np.linalg.norm(coornp - meannp)
            landmarks_vectorised.append(dist)
            landmarks_vectorised.append((math.atan2(y, x) * 360) / (2 * math.pi))
        data['landmarks_vectorised'] = landmarks_vectorised
        for i in range(1, 68):  # There are 68 landmark points on each face
            cv2.circle(image, (shape.part(i).x, shape.part(i).y), 1, (0, 0, 255), thickness=2)  # For each point, draw a red circle with thickness2 on the original frame
            # cv2.imshow("image", image) #Display the frame
            # if cv2.waitKey(1) & 0xFF == ord('q'): #Exit program when the user presses 'q'
            # break
    if len(detections) < 1:
        data['landmarks_vestorised'] = "error"


def make_sets():
    training_data = []
    training_labels = []
    prediction_data = []
    prediction_labels = []
    prediction_names = []
    names_data = []
    i=0

    for emotion in emotions:
        print(" working on %s" % emotion)
        training, prediction, names = get_files(emotion)
        names_data.append(names)
        prediction_names.append(prediction)
        # Append data to training and prediction list, and generate labels 0-7
        for item in training:
            image = cv2.imread(item)  # open image
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # convert to grayscale
            clahe_image = clahe.apply(gray)
            get_landmarks(clahe_image)
            if data['landmarks_vectorised'] == "error":
                print("no face detected on this one")
            else:
                training_data.append(data['landmarks_vectorised'])  # append image array to training data list
                training_labels.append(i)

        for item in prediction:
            image = cv2.imread(item)
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            clahe_image = clahe.apply(gray)
            get_landmarks(clahe_image)
            if data['landmarks_vectorised'] == "error":
                print("no face detected on this one")
            else:
                prediction_data.append(data['landmarks_vectorised'])
                prediction_labels.append(i)
        i+=1
    return training_data, training_labels, prediction_data, prediction_labels, names_data, prediction_names


accur_lin = []


def put_labels(names_data, Probabilidade, Predicoes2):
    Temporal = 0
    for item in range(0, len(names_data) - 1):
        for item2 in range(0, len(names_data[item]) - 1):
            # print(item)
            # clf.classes_  # Vector of classes
            font = cv2.FONT_HERSHEY_SIMPLEX
            img = cv2.imread(names_data[item][item2], 0)
            # print(names_data[item2])
            IndexPredict = item2 + Temporal
            P1 = "disgust: " + str(Probabilidade[item2][0])
            P2 = "happiness: " + str(Probabilidade[item2][1])
            P3 = "neutral: " + str(Probabilidade[item2][2])
            P8 = "predict: " + str(Predicoes2[IndexPredict])

            cv2.putText(img, names_data[item][item2], (0, 25), font, 0.8, (200, 0, 0), 1)
            cv2.putText(img, P1, (0, 50), font, 0.8, (200, 0, 0), 1)
            cv2.putText(img, P2, (0, 75), font, 0.8, (200, 0, 0), 1)
            cv2.putText(img, P3, (0, 100), font, 0.8, (200, 0, 0), 1)
            #cv2.putText(img, P4, (0, 125), font, 0.8, (200, 0, 0), 1)
            #cv2.putText(img, P5, (0, 150), font, 0.8, (200, 0, 0), 1)
            #cv2.putText(img, P6, (0, 175), font, 0.8, (200, 0, 0), 1)
            #cv2.putText(img, P7, (0, 200), font, 0.8, (200, 0, 0), 1)
            cv2.putText(img, P8, (0, 225), font, 0.8, (200, 0, 0), 1)
            #cv2.imshow('result', img)
            newnames = "testeFotos/%s" % names_data[item][item2]
            print(str(newnames))
            cv2.imwrite(newnames, img)
        Temporal = Temporal + len(names_data[item])
        # cv2.waitKey(0)


nomes = ["gripezinha", "resfriadinho"]

for nome in nomes:
    fotos = glob.glob("frames1/%s/*.png" %nome)
    out = csv.writer(open(str("%s.csv" %nome), "a"), delimiter=';', quoting=csv.QUOTE_ALL)
    for i in range(0, 10):
        print("Making sets %s" % i)  # Make sets by random sampling 80/20%
        training_data, training_labels, prediction_data, prediction_labels, names_data, prediction_names = make_sets()
        # Turn the training set into a numpy array for the classifier
        npar_train = np.array(training_data)
        npar_trainlabs = np.array(training_labels)
        print("training SVM linear %s" % i)
        # train SVM
        clf.fit(npar_train, training_labels)
        print("getting accuracies %s" % i)
        npar_pred = np.array(prediction_data)
        npar_predlabs = np.array(prediction_labels)
        # Use score() function to get accuracy
        pred_lin = clf.score(npar_pred, npar_predlabs)
        print("linear: ", pred_lin)
        accur_lin.append(pred_lin)  # Store accuracy in a list

    Predicoes = clf.predict_proba(npar_pred)
    # Class in which were classify
    Predicoes2 = clf.predict(npar_pred)
    for item2 in range(0, len(prediction_names[0]) - 1):
        P1 = "disgust: " + str(Predicoes[item2][0])
        P2 = "happiness: " + str(Predicoes[item2][1])
        P3 = "neutral: " + str(Predicoes[item2][2])
        P8 = "Predicao: " + str(Predicoes2[item2])
        # Store the probabilities and the class of each image
        out.writerow([prediction_names[0][item2], P1, P2, P3, P8])
    cm = confusion_matrix(prediction_labels, Predicoes2)
    print("Mean value lin svm: %s" % np.mean(accur_lin))
    # put_labels(prediction_names, Predicoes, Predicoes2)

