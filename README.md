# dlib臉部影像辨識
# 一、說明
本專題使用dlib套件及進行人臉辨識，dlib是一套包含了機器學習、電腦視覺、圖像處理等的函式庫，使用C++開發而成，目前廣泛使用於工業界及學術界，也應不用再機器人、嵌入式系統、手機、甚至於大型的運算架構中，而且最重要的是，它不但開源且完全免費，也可以跨平台使用。
# 二、相關文章
此專案利用 Pre-train 好的 Dlib model，進行人臉辨識　(Face Detection)　，並且實現僅用一張照片作為 database　就可以作出達到一定效果的人臉識別 (Face Recognition)。 除此之外，更加入了活體偵測 (Liveness Detection) 技術，以避免利用靜態圖片通過系統識別的問題。
![image](https://github.com/LonelyCaesar/dlib-face-recognition/assets/101235367/51655e7d-3173-4d69-8699-1a84ae91b2ba)

Dlib 是一套基於 C++ 的機器學習工具包，藉由 Dlib 可以使用這些機器學習工具在任何的專案上，目前無論在機器人、嵌入設備、移動設備甚至是大型高效運算環境中都被廣泛使用。
