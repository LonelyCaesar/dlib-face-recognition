# dlib臉部影像辨識
# 一、說明
本專題使用dlib套件及進行人臉辨識，dlib是一套包含了機器學習、電腦視覺、圖像處理等的函式庫，使用C++開發而成，目前廣泛使用於工業界及學術界，也應不用再機器人、嵌入式系統、手機、甚至於大型的運算架構中，而且最重要的是，它不但開源且完全免費，也可以跨平台使用。
# 二、相關文章
此專案利用 Pre-train 好的 Dlib model，進行人臉辨識　(Face Detection)　，並且實現僅用一張照片作為 database　就可以作出達到一定效果的人臉識別 (Face Recognition)。 除此之外，更加入了活體偵測 (Liveness Detection) 技術，以避免利用靜態圖片通過系統識別的問題。
![image](https://github.com/LonelyCaesar/dlib-face-recognition/assets/101235367/51655e7d-3173-4d69-8699-1a84ae91b2ba)

Dlib 是一套基於 C++ 的機器學習工具包，藉由 Dlib 可以使用這些機器學習工具在任何的專案上，目前無論在機器人、嵌入設備、移動設備甚至是大型高效運算環境中都被廣泛使用。
# 三、實作
使用dlib套件之前請先下載安裝visual Studio2019以後版本。安裝時要勾選使用C++桌面開發、使用Windows平台，安裝好visual Studio2019後皆可執行安裝pip install dlib套件。
```
pip install dlib
```
![image](https://github.com/LonelyCaesar/dlib-face-recognition/assets/101235367/1748dba9-ab85-416b-8b4b-49b05ed3e600)

Dlib官方網站提供了許多已經訓練好的模型，我們先將一些常用的模型下載，許多範例需要這些模型檔才能執行。需要的模型依照片為主：(模型檔網頁：http://dlib.net/files/)
![image](https://github.com/LonelyCaesar/dlib-face-recognition/assets/101235367/563a354c-a1ca-409d-a54e-45104642a1ae)

### 1.	5點特徵人臉影像辨識：
dlib套件偵測人臉的矩形區塊，也可以偵測鼻子、左右點及嘴巴等。使用<shape_predictor_5_face_landmarks.dat>特徵模型進行人臉矩形區域中偵測5點特徵。然後建議請先安裝pip install opencv-python確保執行時出現缺少套件的問題所在。
### 程式碼：
```python
import dlib
import cv2
predictor = "shape_predictor_5_face_landmarks.dat"  #模型(5點)
sp = dlib.shape_predictor(predictor)  #讀入模型
detector = dlib.get_frontal_face_detector()  #偵測臉部

img = dlib.load_rgb_image("media\\LINE_230625.jpg")  #讀取圖片
win = dlib.image_window()  #建立顯示視窗
win.clear_overlay()  #清除圖形
win.set_image(img)  #顯示圖片

dets = detector(img, 1)  #臉部偵測,1為彩色
print("人臉數：{}".format(len(dets)))
#繪製人臉矩形及5點特徵
for k, det in enumerate(dets):
    print("偵測人臉 {}: 左：{}  上：{}  右：{}  下：{}".format(k, det.left(), det.top(), det.right(), det.bottom()))  #人臉坐標
    win.add_overlay(det)  #顯示矩形
    shape = sp(img, det)  #取得5點特徵 
    win.add_overlay(shape)  #顯示5點特徵
    dlib.hit_enter_to_continue()  #保持影像
```
### 執行結果：
![image](https://github.com/LonelyCaesar/dlib-face-recognition/assets/101235367/410a90e3-9906-4bf8-9d09-d7a8d22740e3)
![image](https://github.com/LonelyCaesar/dlib-face-recognition/assets/101235367/c37fcafa-a70b-4041-8a72-13ac8cf71c8c)

### 2.	CNN訓練模型人臉偵測：
Dlib官網提供的模型中有一個利用CNN訓練模型檔，此模型偵測人臉可得到表情較佳效果。使用< mmod_human_face_detector.dat>特徵模型進行人臉矩形表情。
### 程式碼：
```python
import dlib
import cv2

cnn_detector = dlib.cnn_face_detection_model_v1("mmod_human_face_detector.dat")  #CNN模型
img = cv2.imread("media\\LINE_230625.jpg")
dets = cnn_detector(img, 1)  #偵測人臉
print("人臉數：{}".format(len(dets)))
for i, det in enumerate(dets):
    #det.rect是人臉矩形坐標,det.confidence為信心指數
    face = det.rect
    left = face.left()
    top = face.top()
    right = face.right()
    bottom = face.bottom()
    print("偵測人臉 {}: 左：{}  上：{}  右：{}  下：{}  信心指數：{}".format(i, left, top, right, bottom, det.confidence))
    cv2.rectangle(img, (left, top), (right, bottom), (0, 255, 0), 1)  #畫人臉矩形

cv2.namedWindow("win", cv2.WINDOW_AUTOSIZE)
cv2.imshow("win", img)
k = cv2.waitKey(0)
cv2.destroyAllWindows()
```
### 執行結果：
![image](https://github.com/LonelyCaesar/dlib-face-recognition/assets/101235367/0662ddff-2eca-4a08-ac78-373abb99e356)
![image](https://github.com/LonelyCaesar/dlib-face-recognition/assets/101235367/69af6a6a-d0c2-4a9f-8c56-43c72f9a3871)

### 3.	68點特徵人臉偵測
Dilb除了可以進行基本及5點特徵人臉偵測外，還提供68點特徵人臉偵測，包括了雙眼、鼻子、嘴巴及頭部的輪廓詳細資訊，我們可以畫出這些部位的輪廓圖形。
使用<shape_predictor_68_face_landmarks.dat>特徵模型進行人臉特徵辨識。
### 程式碼：
```python
import numpy as np
import cv2  #影象處理庫OpenCV
import dlib

predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')   #構建68點特徵
detector = dlib.get_frontal_face_detector()  #偵測臉部正面

img = cv2.imread("media\\LINE_ALBUM_1120609.jpg")  #讀取影象
dets = detector(img, 1)  #偵測人臉
for det in dets:
    #人臉關鍵點識別
    landmarks = []
    for p in predictor(img, det).parts():  
        landmarks.append(np.matrix([p.x, p.y])) 
    # 取得68點座標
    for idx, point in enumerate(landmarks):
        pos = (point[0, 0], point[0, 1])  #[0,0]為x坐標,[0,1]為y坐標
        cv2.circle(img, pos, 5, color=(0, 255, 0))  #畫出68個小圓點
        font = cv2.FONT_HERSHEY_SIMPLEX  # 利用cv2.putText輸出1-68
        #引數依次是：圖片，新增的文字，座標，字型，字型大小，顏色，字型粗細
        cv2.putText(img, str(idx+1), pos, font, 0.4, (0, 0, 255), 1,cv2.LINE_AA)

cv2.namedWindow("img", 2)     
cv2.imshow("img", img)       #顯示影象
cv2.waitKey(0)
cv2.destroyWindow("img")
```
### 執行結果：
![image](https://github.com/LonelyCaesar/dlib-face-recognition/assets/101235367/52d6fe02-dce1-4004-aa10-c9b1f3dc7119)

### 4.	攝影機圖像中劃出點輪廓
Imutils模組是一個圖形處理的模組，它有為dlib提供臉部68點特徵各部位範圍的功能，使用imutils模組就可輕鬆取得各部位的特徵點範圍。
### 請先安裝
```
pip install imutils
```
### 程式碼：
```python
from imutils.video import VideoStream
from imutils import face_utils
import imutils
import time
import dlib
import cv2

detector=dlib.get_frontal_face_detector()  #偵測臉部正面
predictor=dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")  #構建68點特徵

#左右眼特徵點索引
(left_Start,left_End)=face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(right_Start,right_End)=face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]
#嘴特徵點索引
(leftMouth,rightMouth)=face_utils.FACIAL_LANDMARKS_IDXS['mouth']
#下巴特徵點索引
(leftJaw,rightJaw)=face_utils.FACIAL_LANDMARKS_IDXS['jaw']
#鼻子特徵點索引
(leftNose,rightNose)=face_utils.FACIAL_LANDMARKS_IDXS['nose']
#左右眉毛特徵點索引
(left_leftEyebrow,left_rightEyebrow)=face_utils.FACIAL_LANDMARKS_IDXS['left_eyebrow']
(right_leftEyebrow,right_rightEyebrow)=face_utils.FACIAL_LANDMARKS_IDXS['right_eyebrow']

vsThread=VideoStream(src=0).start()  #開啟攝影機
time.sleep(2.0)

while True:
    frame = vsThread.read()  #讀取影格
    frame = imutils.resize(frame, width=720)
    faces = detector(frame, 0)  #偵測人臉
    for face in faces:
        shape = predictor(frame, face)  #取得人臉
        shape = face_utils.shape_to_np(shape)  #轉為numpy
        #左右眼特徵點
        leftEye = shape[left_Start:left_End]
        rightEye = shape[right_Start:right_End]
        #轉為外殼
        leftEyeHull = cv2.convexHull(leftEye)
        rightEyeHull = cv2.convexHull(rightEye)
        #畫出輪廓
        cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
        cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)
        #嘴
        mouth=shape[leftMouth:rightMouth]
        mouthHull=cv2.convexHull(mouth)
        cv2.drawContours(frame, [mouthHull], -1, (0, 255, 0), 1)
        #鼻子
        nose=shape[leftNose:rightNose]
        noseHull=cv2.convexHull(nose)
        cv2.drawContours(frame, [noseHull], -1, (0, 255, 0), 1)
        #下巴
        jaw=shape[leftJaw:rightJaw]
        jawHull=cv2.convexHull(jaw)
        cv2.drawContours(frame, [jawHull], -1, (0, 255, 0), 1)
        #眉毛
        leftEyebrow=shape[left_leftEyebrow:left_rightEyebrow]
        rightEyebrow=shape[right_leftEyebrow:right_rightEyebrow]
        leftEyebrowHull=cv2.convexHull(leftEyebrow)
        rightEyebrowHull=cv2.convexHull(rightEyebrow)
        cv2.drawContours(frame, [leftEyebrowHull], -1, (0, 255, 0), 1)
        cv2.drawContours(frame, [rightEyebrowHull], -1, (0, 255, 0), 1)

    cv2.imshow("Frame", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
```
### 執行結果：
![image](https://github.com/LonelyCaesar/dlib-face-recognition/assets/101235367/56445a8d-f60c-4912-8dac-db115fdf2dfb)

### 5.	攝影機臉部偵測
與上述的使用模型套件相同，此範例需要使用openCV套件框選人臉的數值比數之特點範圍。
### 程式碼：
```python
import dlib
import cv2
import imutils
# 開啟影片檔案
cap = cv2.VideoCapture(0)
# Dlib 的人臉偵測器
detector = dlib.get_frontal_face_detector()
# 以迴圈從影片檔案讀取影格，並顯示出來
while(cap.isOpened()):
  ret, frame = cap.read()
  # 偵測人臉
  face_rects, scores, idx = detector.run(frame, 0)
  # 取出所有偵測的結果
  for i, d in enumerate(face_rects):
    x1 = d.left()
    y1 = d.top()
    x2 = d.right()
    y2 = d.bottom()
    text = "%2.2f(%d)" % (scores[i], idx[i])
    # 以方框標示偵測的人臉
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 4, cv2.LINE_AA)
    # 標示分數
    cv2.putText(frame, text, (x1, y1), cv2.FONT_HERSHEY_DUPLEX,
            0.7, (255, 255, 255), 1, cv2.LINE_AA)
  # 顯示結果
  cv2.imshow("Face Detection", frame)
  if cv2.waitKey(1) & 0xFF == ord('q'):
    break
cap.release()
cv2.destroyAllWindows()
```
### 執行結果：
![image](https://github.com/LonelyCaesar/dlib-face-recognition/assets/101235367/5dfdf44d-8201-4c5f-944b-f0192311ccb2)

### 6.	辨識不同的臉：
人臉偵測只能得知圖片中是否有人臉存在，並測得人臉的位置；人臉辨識則會進一步指出人臉是屬於何人，或者比對兩個人臉圖形是否為同一人。使用dlib_face_recognition_resnet_model_v1.dat模型做預測。
### 程式碼：
```python
import dlib, numpy

predictor = "shape_predictor_68_face_landmarks.dat"  #人臉68特徵點模型
recogmodel = "dlib_face_recognition_resnet_model_v1.dat"  #人臉辨識模型

detector = dlib.get_frontal_face_detector()  #偵測臉部正面
sp = dlib.shape_predictor(predictor)  #讀入人臉特徵點模型
facerec = dlib.face_recognition_model_v1(recogmodel)  #讀入人臉辨識模型

#取得人臉特徵點向量
def getFeature(imgfile):
    img = dlib.load_rgb_image(imgfile)
    dets = detector(img, 1)
    for det in dets:
        shape = sp(img, det)  #特徵點偵測
        feature = facerec.compute_face_descriptor(img, shape)  #取得128維特徵向量
        return numpy.array(feature)  #轉換numpy array格式

#判斷是否同一人 
def samePerson(pic1, pic2):
    feature1 = getFeature(pic1)
    feature2 = getFeature(pic2)
    dist = numpy.linalg.norm(feature1-feature2)  # 計算歐式距離,越小越像
    print("歐式距離={}".format(dist))
    if dist < 0.3: 
        print("{} 和 {} 為同一個人！".format(pic1, pic2))
    else:
        print("{} 和 {} 不是同一個人！".format(pic1, pic2))
    print()
    
samePerson("media\\LINE_ALBUM_111.jpg", "media\\LINE_ALBUM_11106.jpg")  #不同人
samePerson("media\\LINE_ALBUM_1125.jpg", "media\\LINE_ALBUM_1120610.jpg")  #同一人
```
### 執行結果：
![image](https://github.com/LonelyCaesar/dlib-face-recognition/assets/101235367/82cbfb13-2b7d-4082-9c66-c5e9829464f7)
# 四、結論
使用了以上的四個模型做人臉辨識後對於論文所寫出來或者是所提出了的議題與FaceNet其概念幾乎完全一樣，但是用了更快的方法來達成而已，或許，這個專案未來可以試著用 FaceNet 來替代 Dlib 的方式進行辨識。對於人工智慧與大數據來說是個專題實作上的作品，對求職的人來說是不可或缺的作品，請於嘗試者做出來。
# 五、參考
巨匠電腦python深度學習開發。
巨匠電腦python人工智慧整合開發。

