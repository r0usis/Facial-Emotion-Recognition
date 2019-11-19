import cv2
import os



def recorte (n,num):

    cam[n] = cv2.VideoCapture('./videos/' + str(num))

    a = num.replace(".mp4", "")
    path = './frames1/' + a

    try:
        if not os.path.exists('./frames1'):
            os.makedirs('./frames1')

    except OSError:
        print('Error: Creating directory of data')

    try:
        if not os.path.exists(path):
            os.makedirs(path)

    except OSError:
        print('Error: Creating directory of data')

    currentframe = 0

    while True:


        ret, frame = cam[n].read()

        if ret:

            name = path + '/' + 'frame' + str(currentframe) + '.jpg'
            cv2.imwrite(name, frame)
            print('Salvando...' + name)

            currentframe += 1
        else:
            break

    cam[n].release()
    cv2.destroyAllWindows()


# calcula numero de itens na pasta videos
# num é uma tupla com os nomes dos arquivos


for _, _, num in os.walk('./videos'):
    print()


cam = list(range(len(num)))
cont = 0
i = 0
n = 0


while n < len(cam):
    print(n)
    recorte(n, str(num[n]))
    n += 1
