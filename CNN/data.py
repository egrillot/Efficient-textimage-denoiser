import numpy as np
import sklearn.utils
import matplotlib.pyplot as plt

class Data:

  def __init__(self,y_train,y_test,x_train=None,x_test=None):
    self.x_train=None
    self.x_test=None
    self.y_train=y_train
    self.y_test=y_test
  
  def set_size(self):
    self.train_size=self.y_train.shape[0]
    self.test_size=self.y_test.shape[0]
  
  def get_size(self):
    print('Size of training set : '+str(self.train_size))
    print('Size of testing set : '+str(self.test_size))
    return self.train_size,self.test_size
  
  def set_x(self,salt_p,noise):
    bin1_train=np.random.binomial(1,salt_p/2,self.y_train.shape)
    bin2_train=np.random.binomial(1,salt_p/2,self.y_train.shape)
    bin1_test=np.random.binomial(1,salt_p/2,self.y_test.shape)
    bin2_test=np.random.binomial(1,salt_p/2,self.y_test.shape)
    self.x_train=np.uint8(np.clip(bin2_train+(1-bin2_train)*(1-bin1_train)*self.y_train+np.random.normal(0,noise,self.y_train.shape),0,255))
    self.x_test=np.uint8(np.clip(bin2_test+(1-bin2_test)*(1-bin1_test)*self.y_test+np.random.normal(0,noise,self.y_test.shape),0,255))
  
  def shuffle(self):
    self.x_train,self.y_train=sklearn.utils.shuffle(self.x_train,self.y_train)
  
  def normalize(self):
    self.x_train=self.x_train/255.
    self.x_test=self.x_test/255.
    self.y_train=self.y_train/255.
    self.y_test=self.y_test/255.
  
  def get_train_set(self):
    return self.x_train,self.y_train
  
  def get_test_set(self):
    return self.x_test,self.y_test
  
  def build_train(self,salt_p,noise):
    self.set_size()
    self.set_x(salt_p,noise)
  
  def reshape(self):
    self.x_train=self.x_train.reshape((self.x_train.shape[0],self.x_train.shape[1],self.x_train.shape[2],1))
    self.x_test=self.x_test.reshape((self.x_test.shape[0],self.x_test.shape[1],self.x_test.shape[2],1))
  
  def preprocess(self):
    self.shuffle()
    self.normalize()
    self.reshape()
  
  def get_element_test(self,i,display=False):
    x,y=self.x_test[i].reshape((self.x_test.shape[1],self.x_test.shape[2]))*255,self.y_test[i]*255
    if display:
      plt.imshow(x,cmap='Greys')
      plt.show()
      plt.imshow(y,cmap='Greys')
      plt.show()
    return x,y
  
  def get_shape(self):
    return (self.x_train.shape[1],self.x_train.shape[2],1)
