import keras
import matplotlib.pyplot as plt

class Model:
  
  def __init__(self,input_shape):
    input=keras.layers.Input(shape=input_shape)
    x=keras.layers.Conv2D(128,(3,3),activation='sigmoid',padding='same')(input)
    x=keras.layers.Conv2D(128,(3,3),activation='relu',padding='same')(x)
    x=keras.layers.BatchNormalization()(x)
    x=keras.layers.Conv2D(64,(3,3),activation='relu',padding='same')(x)
    x=keras.layers.Conv2D(64,(3,3),activation='relu',padding='same')(x)
    x=keras.layers.BatchNormalization()(x)
    x=keras.layers.Conv2D(32,(3,3),activation='relu',padding='same')(x)
    x=keras.layers.Conv2D(32,(3,3),activation='relu',padding='same')(x)
    x=keras.layers.BatchNormalization()(x)
    x=keras.layers.Conv2D(1,(3,3),activation='sigmoid',padding='same')(x)
    self.model=keras.Model(input,x)
  
  def summary(self):
    self.model.summary()

  def train(self,X,Y,optimizer='adam',batch_size=32,loss='mse',val_split=0.15,epochs=50,display=False,save=False,save_path=''):
    self.model.compile(optimizer=optimizer,loss=loss)
    if not save:
      history=self.model.fit(x=X,y=Y,batch_size=batch_size,epochs=epochs,validation_split=val_split) 
    else:
      if len(save_path)==0:
        print('Please enter the path to save the model')
      else:
        checkpoint = keras.callbacks.ModelCheckpoint(filepath=save_path, monitor='val_loss', save_best_only=True, mode='auto')
        callbacks_list = [checkpoint]
        history=self.model.fit(x=X,y=Y,batch_size=batch_size,epochs=epochs,validation_split=val_split,callbacks=callbacks_list) 
    if display:
      plt.plot(history.history['loss'])
      plt.plot(history.history['val_loss'])
      plt.title('model loss')
      plt.ylabel('loss')
      plt.xlabel('epoch')
      plt.legend(['train', 'validation'], loc='upper left')
      plt.show()
  
  def load_model(self,path):
    self.model.load_weights(path)
    
  def pred(self,X):
    return self.model.predict(X).reshape((X.shape[0],X.shape[1],X.shape[2]))*255