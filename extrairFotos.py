import cv2
import os

def fun(cam):
    try:

        # creating a folder named data
        if not os.path.exists('video'):
            os.makedirs('video')

        # if not created then raise error
    except OSError:
        print('Error: Creating directory of data')

    # frame
    currentframe = 0

    while (True):

        # reading from frame
        ret, frame = cam.read()

        if ret:
            # if video is still left continue creating images
            name = './data/frame' + str(currentframe) + '.jpg'
            print('Creating...' + name)

            # writing the extracted images
            cv2.imwrite(name, frame)

            # increasing counter so that it will
            # show how many frames are created
            currentframe += 1
        else:
            break

    # Release all space and windows once done
    cam.release()
    cv2.destroyAllWindows()


n = 0
cam = list(range(20))
cam[0] = cv2.VideoCapture('33ok.mp4')
cam[1] = cv2.VideoCapture('34ok.mp4')
cam[2] = cv2.VideoCapture('36ok.mp4')
cam[3] = cv2.VideoCapture('39ok.mp4')
cam[4] = cv2.VideoCapture('41ok.mp4')
cam[5] = cv2.VideoCapture('48ok.mp4')
cam[6] = cv2.VideoCapture('57ok.mp4')
cam[7] = cv2.VideoCapture('58ok.mp4')
cam[8] = cv2.VideoCapture('60ok.mp4')
cam[9] = cv2.VideoCapture('61ok.mp4')
cam[10] = cv2.VideoCapture('622ok.mp4')
cam[11] = cv2.VideoCapture('63ok.mp4')
cam[12] = cv2.VideoCapture('64ok.mp4')
cam[13] = cv2.VideoCapture('65ok.mp4')
cam[14] = cv2.VideoCapture('66ok.mp4')
cam[15] = cv2.VideoCapture('67ok.mp4')
cam[16] = cv2.VideoCapture('68ok.mp4')
cam[17] = cv2.VideoCapture('69ok.mp4')
cam[18] = cv2.VideoCapture('70ok.mp4')
cam[19] = cv2.VideoCapture('71ok.mp4')

while(n<len(cam)):
    print(n)

    fun(cam[n],n)
    n+=1