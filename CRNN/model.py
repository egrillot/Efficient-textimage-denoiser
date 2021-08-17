import keras
import numpy as np
import matplotlib.pyplot as plt

class Model:
  
    def __init__(self,input_shape,alphabet):
        self.alphabet=np.array([c for c in alphabet])  
        self.alphabet_size=self.alphabet.shape[0]
        self.input=keras.layers.Input(shape=input_shape)
        x=keras.layers.Conv2D(64,(3,3),activation='relu',padding='same')(self.input)
        x=keras.layers.MaxPool2D(pool_size=(2,2))(x)
        x=keras.layers.Conv2D(128,(3,3),activation='relu',padding='same')(x)
        x=keras.layers.MaxPool2D(pool_size=(2,2))(x)
        x=keras.layers.Conv2D(256,(3,3),activation='relu',padding='same')(x)
        x=keras.layers.Conv2D(256,(3,3),activation='relu',padding='same')(x)
        x=keras.layers.MaxPool2D(pool_size=(2,1))(x)
        x=keras.layers.Conv2D(512,(3,3),activation='relu',padding='same')(x)
        x=keras.layers.BatchNormalization()(x)
        x=keras.layers.Conv2D(512,(3,3),activation='relu',padding='same')(x)
        x=keras.layers.BatchNormalization()(x)
        x=keras.layers.MaxPool2D(pool_size=(2,1))(x)
        x=keras.layers.Conv2D(512,(2,2),activation='relu')(x)
        x=keras.layers.Lambda(lambda x: keras.backend.squeeze(x,1))(x)
        x=keras.layers.Bidirectional(keras.layers.LSTM(128,return_sequences=True,dropout=0.2))(x)
        x=keras.layers.Bidirectional(keras.layers.LSTM(128,return_sequences=True,dropout=0.2))(x)
        self.output=keras.layers.Dense(self.alphabet_size+1,activation='softmax')(x)
        self.model_for_summary=keras.Model(self.input,self.output)
  
    def summary(self):
        self.model_for_summary.summary()

    def ctc_lambda_func(args):
        y_pred, labels, input_length, label_length = args
        return keras.backend.ctc_batch_cost(labels, y_pred, input_length, label_length)
  
    def set_CTCloss(self,max_word_length):
        labels = keras.layers.Input(name='the_labels', shape=[max_word_length], dtype='float32')
        input_length = keras.layers.Input(name='input_length', shape=[1], dtype='int64')
        label_length = keras.layers.Input(name='label_length', shape=[1], dtype='int64')
        loss_out = keras.layers.Lambda(Model.ctc_lambda_func, output_shape=(1,), name='ctc')([self.output, labels, input_length, label_length])
        self.model = keras.Model(inputs=[self.input, labels, input_length, label_length], outputs=loss_out) 

    def train(self,train,validation,save_path,batch_size=32,epochs=10,display=False):
        training_im, training_txt, train_label_length, train_input_length =train
        val_im, val_txt, val_label_length, val_input_length =validation
        self.model.compile(loss={'ctc': lambda y_true, y_pred: y_pred}, optimizer = 'adam')
        checkpoint = keras.callbacks.ModelCheckpoint(filepath=save_path, monitor='val_loss', verbose=1, save_best_only=True, mode='auto')
        callbacks_list = [checkpoint]
        history=self.model.fit(x=[training_im.reshape((training_im.shape[0],training_im.shape[1],training_im.shape[2],1)), training_txt, train_input_length, train_label_length], y=np.zeros(training_im.shape[0]), batch_size=batch_size,epochs = epochs, validation_data = ([val_im.reshape((val_im.shape[0],val_im.shape[1],val_im.shape[2],1)), val_txt, val_input_length, val_label_length], [np.zeros(val_im.shape[0])]), verbose = 1,callbacks=callbacks_list)
        if display:
            plt.plot(history.history['loss'])
            plt.plot(history.history['val_loss'])
            plt.title('model loss')
            plt.ylabel('CTC loss')
            plt.xlabel('epoch')
            plt.legend(['train', 'validation'], loc='upper left')
            plt.show()
  
    def load_model(self,path):
        self.model_for_summary.load_weights(path)
    
    def pred(self,X):
        prediction=self.model_for_summary.predict(X.reshape((X.shape[0],X.shape[1],X.shape[2],1)))
        out = keras.backend.get_value(keras.backend.ctc_decode(prediction, input_length=np.ones(prediction.shape[0])*prediction.shape[1],greedy=True)[0][0])
        res=np.zeros((X.shape[0]),dtype=object)
        i=0
        for x in out:
            txt=''
            for char_encoded in x:
                if int(char_encoded)!=-1:
                    txt+=self.alphabet[int(char_encoded)]
            res[i]=txt
            i+=1
        return res