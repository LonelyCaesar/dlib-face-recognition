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

