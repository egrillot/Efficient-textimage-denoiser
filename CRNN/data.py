import os
import comtypes.client
import pdf2image
import numpy as np
import sklearn.utils
import pickle as pkl
import matplotlib.pyplot as plt
from docx.shared import Pt
from docx import Document
from docx.enum.text import WD_COLOR_INDEX
from docx.shared import RGBColor

class Data:

    def __init__(self,max_size,alphabet):
        self.max_size=max_size
        self.size=0
        self.X=[]
        self.Y=[]
        self.alphabet=np.array([c for c in alphabet])
        self.alphabet_size=self.alphabet.shape[0]
        self.dict={'black':RGBColor(0,0,0),'blue':RGBColor(0,0,255),'white':RGBColor(255,255,255),'orange':RGBColor(255,153,51),'grey':RGBColor(64,64,64)}
        self.dict_key=np.array(['black','blue','white','orange','grey'])
        self.black_background=np.array([WD_COLOR_INDEX.BLUE,WD_COLOR_INDEX.YELLOW,WD_COLOR_INDEX.BRIGHT_GREEN,WD_COLOR_INDEX.RED,WD_COLOR_INDEX.TURQUOISE,WD_COLOR_INDEX.WHITE])
        self.blue_background=np.array([WD_COLOR_INDEX.YELLOW,WD_COLOR_INDEX.BRIGHT_GREEN,WD_COLOR_INDEX.RED,WD_COLOR_INDEX.WHITE])
        self.white_background=np.array([WD_COLOR_INDEX.BLUE,WD_COLOR_INDEX.RED,WD_COLOR_INDEX.TURQUOISE])
        self.x_train=None
        self.y_train=None
        self.x_val=None
        self.y_val=None
        self.x_test=None
        self.y_test=None
        self.encoding=None
        self.training_txt=None
        self.val_txt=None
        self.train_input_length=None
        self.train_label_length=None
        self.val_input_length=None
        self.val_label_length=None
        self.max_word_length=None
    
    def set_encoding(self):
        i=0
        encoding={}
        while i!=self.alphabet_size:
            encoding[self.alphabet[i]]=i
            i+=1
        self.encoding=encoding
    
    def get_size(self):
        return self.size
    
    def add_data(self,x,y):
        self.size+=len(x)
        self.X+=x
        self.Y+=y

    def convert_doc_to_pdf(doc_path):
        path=doc_path[:-4]
        word = comtypes.client.CreateObject('Word.Application')
        doc = word.Documents.Open(doc_path)
        doc.SaveAs(path+'pdf', FileFormat=17)
        doc.Close()
        word.Quit()
        os.remove(doc_path)
        return path+'pdf'
    
    def generate_full_page(self,dir_path_for_doc_writing): #3.3 sec
        image,txt=[],[]
        document = Document()
        font=document.styles['Normal'].font
        font.name='Calibri'
        color_txt_key=np.random.choice(self.dict_key,size=1)[0]
        txt_color=self.dict[color_txt_key]
        if color_txt_key=='black':
            color_background=np.random.choice(self.black_background,size=1)[0]
        if color_txt_key=='blue':
            color_background=np.random.choice(self.blue_background,size=1)[0]
        if color_txt_key=='white':
            color_background=np.random.choice(self.white_background,size=1)[0]
        if color_txt_key=='orange':
            color_background=WD_COLOR_INDEX.WHITE
        if color_txt_key=='grey':
            color_background=WD_COLOR_INDEX.WHITE
        i=0
        length=np.random.randint(3,8)
        while i!=24:
            text=''
            a=np.random.choice(self.alphabet,size=length,replace=True)
            for ch in a:
                text+=ch 
            txt.append(text)
            font.size = Pt(12)
            font.highlight_color=color_background
            font.color.rgb = txt_color
            document.add_paragraph(text)
            i+=1
        document.save(dir_path_for_doc_writing+'doc.docx')
        pdf_path=Data.convert_doc_to_pdf(dir_path_for_doc_writing+'doc.docx')
        im_txt=pdf2image.convert_from_path(pdf_path)[0]
        for j in range(24):
            image.append(np.array(im_txt.crop((232,197+j*74,450,197+(j+1)*74)).convert('L').resize((128,32))).tolist())
        os.remove(pdf_path)
        return image,txt
    
    def build(self,dir_path):
        while self.size<=self.max_size:
            im,txt=self.generate_full_page(dir_path)
            self.add_data(im,txt)
            if self.size%1008==0:
                print('Current size of the database : '+str(self.size))
            if self.size%4032==0:
                self.save_fill_data(dir_path)
        self.save_fill_data(dir_path)      
    
    def shuffle(self):
        self.X,self.Y=sklearn.utils.shuffle(self.X,self.Y)
    
    def normalize(self):
        self.X=self.X/255. 
    
    def set_train_val_test(self):
        a,b=int(0.1*0.9*self.size),int(0.1*self.size)
        self.x_train,self.x_val,self.x_test=self.X[:self.size-a-b],self.X[self.size-a-b:self.size-b],self.X[self.size-b:]
        self.y_train,self.y_val,self.y_test=self.Y[:self.size-a-b],self.Y[self.size-a-b:self.size-b],self.Y[self.size-b:]
    
    def arr(self):
        self.X=np.array(self.X)
        self.Y=np.array(self.Y)
    
    def prepareCTCenv(self):
        training_txt = []
        val_txt = []
        max_word_length=max([len(t) for t in self.Y])
        train_input_length=[]
        val_input_length=[]
        for y in self.y_train: 
            word_encoded=np.ones(max_word_length)*self.encoding[' ']
            train_input_length.append(len(y))
            i=0
            for character in y:
                word_encoded[i]=self.encoding[character]
                i+=1
            training_txt.append(word_encoded)
        for y in self.y_val: 
            word_encoded=np.ones(max_word_length)*self.encoding[' ']
            val_input_length.append(len(y))
            i=0
            for character in y:
                word_encoded[i]=self.encoding[character]
                i+=1
            val_txt.append(word_encoded)
        self.training_txt,self.train_input_length,self.train_label_length,self.val_txt,self.val_input_length,self.val_label_length,self.max_word_length=np.array(training_txt),np.array(train_input_length),np.ones((self.x_train.shape[0]))*31,np.array(val_txt),np.array(val_input_length),np.ones((self.x_val.shape[0]))*31,max_word_length

    def add_noise(self,saltp,noise):
        bin1=np.random.binomial(1,saltp/2,self.X.shape)
        bin2=np.random.binomial(1,saltp/2,self.X.shape)
        self.X=np.uint8(np.clip(bin2+(1-bin2)*(1-bin1)*self.X+np.random.normal(0,noise,self.X.shape),0,255))
    
    def watch_impact_sample(self,saltp,noise):
        i=np.random.randint(0,len(self.X))
        sample=np.array(self.X[i])
        bin1=np.random.binomial(1,saltp/2,sample.shape)
        bin2=np.random.binomial(1,saltp/2,sample.shape)
        sample=np.uint8(np.clip(bin2+(1-bin2)*(1-bin1)*sample+np.random.normal(0,noise,sample.shape),0,255))
        plt.imshow(sample,cmap='Greys')
        plt.show()

    def preprocess(self,saltp,noise,dir_path):
        self.set_encoding()
        self.arr()
        self.add_noise(saltp,noise)
        self.normalize()
        self.shuffle()
        self.set_train_val_test()
        self.prepareCTCenv()
        self.save(dir_path)

    def get_data(self):
        return [self.x_train,self.training_txt,self.train_input_length,self.train_label_length],[self.x_val,self.val_txt,self.val_input_length,self.val_label_length],self.max_word_length,self.x_test,self.y_test
    
    def save_fill_data(self,dir_path):
        with open(dir_path+'dataset_filling.pkl','wb') as f:
            pkl.dump((self.X,self.Y),f)
            f.close()
    
    def load_predata(self,dir_path):
        with open(dir_path+'dataset_filling.pkl','rb') as f:
            self.X,self.Y=pkl.load(f)
            f.close()
        self.size=len(self.X)
    
    def save(self,dir_path):
        with open(dir_path+'dataset_generated.pkl','wb') as f:
            pkl.dump((self.get_data()),f)
            f.close()
    
    def load(self,dir_path):
        with open(dir_path+'dataset_generated.pkl','rb') as f:
            [self.training_txt,self.train_input_length,self.train_label_length],[self.val_txt,self.val_input_length,self.val_label_length],self.max_word_length,self.x_test,self.y_test=pkl.load(f)
            f.close()
    
    def get_data_for_training(self):
        return [self.x_train,self.training_txt,self.train_input_length,self.train_label_length],[self.x_val,self.val_txt,self.val_input_length,self.val_label_length],self.max_word_length
    
    
