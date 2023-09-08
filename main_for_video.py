import os

import argparse

import cv2
import mediapipe as mp 

# Обернём всё в функцию
def process_img(img, face_detection):
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    out = face_detection.process(img_rgb)
    H, W, _ = img.shape
    # print(out.detections) можем проверить работу на фото cat.jpg - получим None

    # сначала проверяем, есть ли вообще какое-то лицо на изображении/видео
    if out.detections is not None:
        for detection in out.detections:
            location_data = detection.location_data #данные о местоположении точек
            bbox = location_data.relative_bounding_box #данные об ограничивающей рамке

            x1, y1, w, h = bbox.xmin, bbox.ymin, bbox.width, bbox.height
            # полученные значения в относительных единицах, мы преобразуем в реальные размеры 
            x1 = int(x1*W)
            y1 = int(y1*H)
            w = int(w*W)
            h = int(h*H)

            #нарисуем зеленую ограничивающую рамку
            # img = cv2.rectangle(img, (x1, y1), (x1+w, y1+h), (0, 255, 0), 10)

            # заблюрим лицо на изображении
            img[y1:y1+h, x1:x1+w, :] = cv2.blur(img[y1:y1+h, x1:x1+w, :], (30,30))
        
    return img

# создадим объект с аргументами, чтобы вызывать нужную программу
args = argparse.ArgumentParser()

args.add_argument('--mode', default = 'video') #Здесь выбираем режим
args.add_argument('--filePath', default = './data/testVideo.mp4')

args = args.parse_args()

output_data = './ouput'
# если такого каталога не существует, мы его создаём
if not os.path.exists(output_data):
    os.makedirs(output_data)

#Создаем объект, который будет использован для детекции
mp_face_detection = mp.solutions.face_detection

#первый параметр- Минимальный показатель достоверности, позволяющий считать обнаружение лица успешным
#второй- выбор модели, 0 - распознает лица в пределах 2-х метров от камеры
with mp_face_detection.FaceDetection(min_detection_confidence = 0.5, model_selection=0) as face_detection:

    if args.mode in ['image']:
        img = cv2.imread(args.filePath)
    
        img = process_img(img, face_detection) #вызываем функцию

        cv2.imwrite(os.path.join(output_data, 'output_img.png'), img)

    elif args.mode in ['video']:
        #откроем видео из папки папку
        cap = cv2.VideoCapture(args.filePath)
        ret, frame = cap.read()

        output_video = cv2.VideoWriter(os.path.join(output_data, 'output.mp4'),
                                      cv2.VideoWriter_fourcc(*'MP4V'),
                                      25, #здесь указана частота кадров
                                      (frame.shape[1], frame.shape[0])) #размер кадра
        
#while ret - цикл, который выполняется до тех пор, пока условие 'ret' является истинным
# Когда цикл 'while' начинается, сначала проверяется условие 'ret'. 
# Если оно является истинным, тогда код внутри цикла выполняется. 
# После выполнения кода цикла, условие 'ret' снова проверяется.
        while ret:
            frame = process_img(frame, face_detection)
            output_video.write(frame)
            ret, frame = cap.read()

        cap.release()
        output_video.release()

    elif args.mode in ['webcam']:
            #сохраним видео в отдельную папку
            cap = cv2.VideoCapture(0)
            ret, frame = cap.read()
            while ret:
                frame = process_img(frame, face_detection) #читаем кадры
                cv2.imshow('frame', frame)
                cv2.waitKey(25)
                ret, frame = cap.read()

            cap.release()