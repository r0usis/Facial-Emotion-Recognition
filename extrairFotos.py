import cv2
import os

def recorte (n):
    try:
        if not os.path.exists('./frames'):
            os.makedirs('./frames')

    except OSError:
        print('Error: Creating directory of data')

    try:
        if not os.path.exists('./frames/video'+str(n)):
            os.makedirs('./frames/video'+str(n))

    except OSError:
        print('Error: Creating directory of data')

    currentframe = 0

    while True:

        ret, frame = cam[n].read()

        if ret:

            name = './video'+str(n)+'/frame' + str(currentframe) + '.jpg'
            print('Salvando...' + name)

            cv2.imwrite(name, frame)

            currentframe += 1
        else:
            break

    cam[n].release()
    cv2.destroyAllWindows()



n = 0
cam = list(range(20))
i = 0;
# onde começa a contagem dos videos
cont = 33;

# exceções, videos que nao existem
# 35, 37, 38, 40, 42, 43, 44, 45, 46, 47, 49, 50, 51, 52, 53, 54, 55, 56, 57, 59
# adicionar if para experimentos que nao deram certo

for i in range(len(cam)):

    if cont == 35:
        cont = 36
    elif cont == 37 or cont == 38:
        cont = 39
    elif cont == 40:
        cont = 41
    elif cont == 42 or cont == 43 or cont == 44 or cont == 45 or cont == 46 or cont == 47:
        cont = 48
    elif cont == 49 or cont == 50 or cont == 51 or cont == 52 or cont == 53 or cont == 54 or cont == 55 or cont == 56 or cont == 57:
        cont = 58
    elif cont == 59:
        cont = 60

    cam[i] = cv2.VideoCapture('./videos/' + str(cont) + 'ok.mp4')
    cont = cont + 1



while n < len(cam):

    print(n)
    recorte(n)
    n += 1
