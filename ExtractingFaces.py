import cv2
import glob
import os

faceDet = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
faceDet_two = cv2.CascadeClassifier("haarcascade_frontalface_alt2.xml")
faceDet_three = cv2.CascadeClassifier("haarcascade_frontalface_alt.xml")
faceDet_four = cv2.CascadeClassifier("haarcascade_frontalface_alt_tree.xml")

def detect_faces(emotion, files):
    filenumber = 0
    for f in files:
        frame = cv2.imread(f)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        #Detect face using 4 different classifiers
        face = faceDet.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=10, minSize=(5, 5), flags=cv2.CASCADE_SCALE_IMAGE)
        face_two = faceDet_two.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=10, minSize=(5, 5), flags=cv2.CASCADE_SCALE_IMAGE)
        face_three = faceDet_three.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=10, minSize=(5, 5), flags=cv2.CASCADE_SCALE_IMAGE)
        face_four = faceDet_four.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=10, minSize=(5, 5), flags=cv2.CASCADE_SCALE_IMAGE)
        #Go over detected faces, stop at first detected face, return empty if no face.
        if len(face) == 1:
            facefeatures = face
        elif len(face_two) == 1:
            facefeatures = face_two
        elif len(face_three) == 1:
            facefeatures = face_three
        elif len(face_four) == 1:
            facefeatures = face_four
        else:
            facefeatures = ""


        #Cut and save face
        for (x, y, w, h) in facefeatures: #get coordinates and size of rectangle containing face
            print ("face found in file: %s" %f)

            filename = f.replace(".jpg", ".png")
            print(filename)
            gray = gray[y-30:y+h+50, x-30:x+w+50] #Cut the frame to size
            try:
                out = cv2.resize(gray, (350, 350)) #Resize face so all images have same size
                cv2.imwrite(filename, out) #Write image
                print ("filename %s" %filename)
            except:
               pass #If error, pass file
        filenumber += 1 #Increment image number

print("1- Recortar frames \n2-Recortar dataset")
prop = input("Escolha uma opção: ")
qntvideos = input("Digite a quantidade de videos: ")

for _, _, num in os.walk('./videos'):
    print()

if prop == '1':
    for i in range(0, int(qntvideos)):
        num[i] = num[i].replace(".mp4", "")
        files = glob.glob("frames1/%s/*" %str(num[i])) #Get list of all images with emotion
        detect_faces(str(num[i]), files)
elif prop == '2':
    for emotion in ["neutral", "anger", "disgust", "fear", "happiness", "sadness", "surprise"]:
        files = glob.glob("dataset/%s/*" % emotion)  # Get list of all images with emotion
        detect_faces(emotion, files) #Call functiona
