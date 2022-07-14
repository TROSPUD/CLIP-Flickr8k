import torch
import os
import clip
import pandas as pd
import numpy as np
from pathlib import Path
from torchvision.datasets import Flickr8k
from torch.utils.data import Dataset
from PIL import Image

import sys
from PyQt5 import QtWidgets,QtGui,QtCore,Qt
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.Qt import *
imgName=''
output_text=''
#准备数据集
#读取图片数据集
class MyData(Dataset):
    def __init__(self,path):
        self.path=path
        self.train_class=os.listdir(self.path)
        self.img_path=[]
        self.img_item_path=path
        cnt=0
        for i in range(400):
            self.img_path.append(self.train_class[i])
    def __getitem__(self, idx):
        img_name=self.img_path[idx]
        self.img_item_path=os.path.join(self.path,img_name)
        img=Image.open(self.img_item_path)
        return img
    def __len__(self):
        return len(self.img_path)

image_path="ai/archive/Images"
jpgs=MyData(image_path)
print(jpgs.__len__())
print(type(jpgs))

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)
#下载flickr8k数据集
dir_Flickr_text="ai/archive/captions"#要用/不能用\
#jpgs=os.listdir(image_path)
#print("total images in dataset={}".format(len(jpgs)))

#file=open(dir_Flickr_text,'r')
#text = file.read()
#file.close()
path = (Path(__file__).parent) / 'archive/captions.csv'
data=pd.read_csv(str(path),nrows=2000,encoding='UTF-8')
print(type(data))

#gui
class GUI(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        self.initGUI()
    def initGUI(self):
        self.resize(1000, 1000)
        qr=self.frameGeometry()
        cp=QtWidgets.QDesktopWidget().availableGeometry().center()
        qr.moveCenter(cp)
        self.setWindowTitle("Image intelligent description and inversion generator")
        self.setStyleSheet('''QWidget{background-color:#19335c;}''')
        self.move(qr.topLeft()) 
        self.label=QLabel(self)
        self.label.setText("INPUT-image")
        self.label.setFixedSize(300,200)
        self.label.move(100, 160)
        self.label.setStyleSheet("QLabel{background:white;}"
        "QLabel{color:rgb(300,300,300,120);font-size:20px;font-weight:bold;font-family:Helvetica;}"
        )
        self.label1=QLabel(self)
        self.label1.setFixedSize(1000, 300)
        self.label1.move(100,310)
        self.label1.setStyleSheet("color:white")
        self.label1.setText("TEXT OF IMAGE CONVERSION")
        self.setStyleSheet("QLabel{font-size:20px;}")

        self.label2=QLabel(self)
        self.label2.setText("OUTPUT-image")
        self.label2.setFixedSize(300,200)
        self.label2.move(100, 560)
        self.label2.setStyleSheet("QLabel{background:white;}"
        "QLabel{color:rgb(300,300,300,120);font-size:20px;font-weight:bold;font-family:Helvetica;}"
        )
        self.bot=QtWidgets.QPushButton('upload',self)
        self.bot.resize(100, 30)
        self.bot.move(100, 100)
        self.bot.setStyleSheet("background-color: #e9dbcc;"
        "border-color: #e9dbcc;"
        "font: 75 12pt \"Arial Narrow\";"
        "color: #19335c;")
        self.bot1=QtWidgets.QPushButton('tranform',self)
        self.bot1.resize(100, 30)
        self.bot1.move(100, 400)
        self.bot1.setStyleSheet("background-color: #e9dbcc;"
        "border-color: #e9dbcc;"
        "font: 75 12pt \"Arial Narrow\";"
        "color: #19335c;")
        self.bot2=QtWidgets.QPushButton('tranform',self)
        self.bot2.resize(100, 30)
        self.bot2.move(100, 500)
        self.bot2.setStyleSheet("background-color: #e9dbcc;"
        "border-color: #e9dbcc;"
        "font: 75 12pt \"Arial Narrow\";"
        "color: #19335c;")
        self.bot.clicked.connect(self.openimag)
        self.bot1.clicked.connect(self.text)
        self.bot2.clicked.connect(self.pic)
        
        

    def openimag(self):
        global imgName
        global output_text
        imgName,imgType=QFileDialog.getOpenFileName(self, "打开图片", "", "*.jpg;;*.png;;All Files(*)")
        jpg=QtGui.QPixmap(imgName).scaled(self.label.width(),self.label.height())
        self.label.setPixmap(jpg)
        image = preprocess(Image.open(imgName)).unsqueeze(0).to(device)
        #text = clip.tokenize(["a diagram", "a dog", "a cat"]).to(device)
        text = torch.cat([clip.tokenize(f"a photo of a {c}") for c in data['caption']]).to(device)

        with torch.no_grad():
            image_features = model.encode_image(image)
            text_features = model.encode_text(text)

            #logits_per_image, logits_per_text = model(image, text)
            #probs = logits_per_image.softmax(dim=-1).cpu().numpy()
            #print(type(probs))
        image_features /= image_features.norm(dim=-1, keepdim=True)
        text_features /= text_features.norm(dim=-1, keepdim=True)
        similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)
        values, indices = similarity[0].topk(1)

        print("\nTop predictions:\n")
        for value, index in zip(values, indices):
            data1=data['caption'].to_list()#series和list可以相互转换
            output_text=data1[index]
            print(data1[index],end=' :')
            print(f"{100 * value.item():.2f}%")
    def text(self):
        global output_text
        self.label1.setText(output_text)
    
    def pic(self):
        global output_text
        global jpgs
        #文字转图片
        out_text=clip.tokenize(output_text).to(device)
        jpgs_input=[]
        for j in range(100):
            jpg_input=preprocess(jpgs.__getitem__(j)).unsqueeze(0).to(device)
            jpgs_input.append(jpg_input)
        print(type(jpgs_input))
        all_prob=[]
        for items in jpgs_input:
            with torch.no_grad():
                image_features=model.encode_image(items)
                text_features=model.encode_text(out_text)
                logits_per_image, logits_per_text = model(items, out_text)
                # probs = logits_per_text.softmax(dim=-1).cpu().numpy()
                probs = logits_per_text.numpy()
            all_prob.append(probs[0][0])
        t=type(all_prob)
        all_prob=np.array(all_prob)
        print(type(all_prob))
        ma=np.max(all_prob)
        print(ma)

        ind=np.where(all_prob==np.max(all_prob))
        print(ind)
        j=ind[0][0]
        print(j)
        pic=jpgs.__getitem__(j)
        pic.show()
        p=jpgs.img_item_path
        jpgg=QtGui.QPixmap(p).scaled(self.label2.width(),self.label2.height())
        self.label2.setPixmap(jpgg)
        





if __name__ == '__main__':
    app=QtWidgets.QApplication(sys.argv)
    gui=GUI() 
    gui.show()
    sys.exit(app.exec_())
    print('done')


