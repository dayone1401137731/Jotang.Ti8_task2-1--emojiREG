#人脸表情识别项目:sweat_drops:
##Pt1. 配置环境 :dash:
遇到的麻烦不少，首先就是环境的文件地址，因为纯纯萌新，文档的管理有点不够clean，导致在pip install和test的环节上状况多多。
同时配置时最傻的是我误以为已经安装了CUDA和CUDNN，却忙这安装keras库，导致报错频频:worried:
![](%E5%B1%8F%E5%B9%95%E6%88%AA%E5%9B%BE%202022-10-05%20100947.png)

不过最后也还算是顺利解决问题了

![](%E5%B1%8F%E5%B9%95%E6%88%AA%E5%9B%BE%202022-10-09%20235206.png)

##Pt2.复现train.py的小成果:dash:

> ~~别问我为啥子不用电脑截屏（因为手机有美颜）~~:triumph:

![](QQ%E5%9B%BE%E7%89%8720221009235632.jpg)
![](QQ%E5%9B%BE%E7%89%8720221009235644.jpg)
![](QQ%E5%9B%BE%E7%89%8720221009235650.jpg)

##Pt3.对train.py的一点分析(?):dash:


1. ==加载数据包 设置训练器和验证器==
```
train_dir = 'D:/网络下载/emoji-creator-project-code/emoji-creator-project-code/Emoji Recognition/train'
val_dir = 'D:/网络下载/emoji-creator-project-code/emoji-creator-project-code/Emoji Recognition/test'
train_datagen = ImageDataGenerator(rescale=1./255)
val_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=(48,48),
        batch_size=64,
        color_mode="grayscale",
        class_mode='categorical')

validation_generator = val_datagen.flow_from_directory(
        val_dir,
        target_size=(48,48),
        batch_size=64,
        color_mode="grayscale",
        class_mode='categorical')

```

2. ==创建model(Squential形式)==
```
emotion_model = Sequential()

emotion_model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(48,48,1)))
emotion_model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
emotion_model.add(MaxPooling2D(pool_size=(2, 2)))
emotion_model.add(Dropout(0.25))

emotion_model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
emotion_model.add(MaxPooling2D(pool_size=(2, 2)))
emotion_model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
emotion_model.add(MaxPooling2D(pool_size=(2, 2)))
emotion_model.add(Dropout(0.25))

emotion_model.add(Flatten())
emotion_model.add(Dense(1024, activation='relu'))
emotion_model.add(Dropout(0.5))
emotion_model.add(Dense(7, activation='softmax'))
```

3.==神经网络的训练==



```
emotion_model.compile(loss='categorical_crossentropy',optimizer=Adam(lr=0.001, decay=1e-6),metrics=['accuracy'])
emotion_model_info = emotion_model.fit_generator(
        train_generator,
        steps_per_epoch=28709 // 64,
        epochs=1,
        validation_data=validation_generator,
        validation_steps=10)

#问题可能出现在这一步，导致gui.py运行时内存过大？
#因此可能需要调整batch_size来减少运行内存。
```


设置较小的batch size不仅可以解决一些内存上不足的问题，还可以即时更新权重。       //至于为什么会更新权重我还没搞懂
但是使用过少量数据训练时可能因为数据量较少而造成训练中的梯度值较大的波动。         //ps：好像跟优化器的梯度清零有关(?)

4.==OpenCV库的调用和应用== （*亟待掌握*）
```
cap = cv2.VideoCapture(0)
while True:
    # Find haar cascade to draw bounding box around face
    ret, frame = cap.read()
    if not ret:
        break

    pathf = 'C:/ProgramData/Anaconda3/Lib/site-packages/cv2/data/haarcascade_frontalface_default.xml'
    bounding_box = face_cascade = cv2.CascadeClassifier(pathf)

    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    num_faces = bounding_box.detectMultiScale(gray_frame,scaleFactor=1.3, minNeighbors=5)

    for (x, y, w, h) in num_faces:
        cv2.rectangle(frame, (x, y-50), (x+w, y+h+10), (255, 0, 0), 2)
        roi_gray_frame = gray_frame[y:y + h, x:x + w]
        cropped_img = np.expand_dims(np.expand_dims(cv2.resize(roi_gray_frame, (48, 48)), -1), 0)
        emotion_prediction = emotion_model.predict(cropped_img)
        maxindex = int(np.argmax(emotion_prediction))
        cv2.putText(frame, emotion_dict[maxindex], (x+20, y-60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

    cv2.imshow('Video', cv2.resize(frame,(1200,860),interpolation = cv2.INTER_CUBIC))
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
```

##Pt4 运行gui.py遇到的一些问题:dash:

> 某运行成功的一次:point_down:

![](QQ%E5%9B%BE%E7%89%8720221010002808.jpg)

运行结果不准确的原因是因为我测试的时候为了减少神经网络训练的时间，我把训练的epoch从50减到了2次:smile:

但是这个gui.py的运行是在条条红字中运行的，不仅遇到调用Opencv库的时候出错:point_down:
```
error: (-215:Assertion failed) !_src.empty() in function ‘cv::cvtColor‘
```
还遇到了pycharm在运行这个.py时内存不足的报错:point_down:
```
Process finished with exit code 3
```
前一个错误我在重看代码时发现的问题可能是在以下代码中
```
def show_vid2():

    frame2 = cv2.imread(emoji_dist[show_text[0]])
    pic2 = cv2.cvtColor(frame2,cv2.COLOR_BGR2RGB)
    img2 =Image.fromarray(frame2)
    imgtk2=ImageTk.PhotoImage(image=img2)
    lmain2.imgtk2 = imgtk2
    lmain3.configure(text=emotion_dict[show_text[0]],font=('arial',45,'bold'))
    
    lmain2.configure(image=imgtk2)
    lmain2.after(10, show_vid2)

    #line 3 的变量pic2没有被赋以实际意义，导致cv2在调用图片的时候这种错误发生是因为图片没有被正确读取。所以问题可能出现在前一行的cv2.imread中，我查阅得到的解决方法有两种：
    #   （1）检查所给的路径是否有该图片存在；
    #   （2）检查路径后面的变量是否为有效数字，0代表灰度图像。

    #但我因为还没完全明白这个gui.py的程序逻辑，因此这个pic2的含义还未完全理解，~~因为个人情况，做题的时间确实有点仓促~~
    #所以这个gui.py的实现只能以半成品的样子呈现了
```
而至于第二个错误我在推测之所会占用内存过大，可能是训练时产生的过多的文件目录占用了过多的内存(pycharm的运行确实会产生大量的缓存)
解决方法可能就是通过改变batch_size的大小来降低pycharm运行时的内存占用。


*更新* **关于第二个错误我的方向好像错了**
> 为什么会产生内存不足的原因,好像是因为opencv调用摄像头不断地拍摄照片，而我没有及时停止并开始识别而导致内存不断减小。

~~可能的罪魁祸首~~:point_down:
```
    elif flag1:
        global last_frame1
        last_frame1 = frame1.copy()
        pic = cv2.cvtColor(last_frame1, cv2.COLOR_BGR2RGB)     
        img = Image.fromarray(pic)
        imgtk = ImageTk.PhotoImage(image=img)
        lmain.imgtk = imgtk
        lmain.configure(image=imgtk)
        lmain.after(10, show_vid)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        exit()
```
> 在执行gui.py后，摄像头不断读入照片，但我却不能通过按q来终止读入。

*欸  啥时候得恶补一下OpenCV*