#Loading dependencies
import numpy as np
import pandas as pd
from keras.models import Model
from keras.preprocessing.image import ImageDataGenerator
from keras.models import load_model
from keras.callbacks import EarlyStopping
from sklearn.metrics import accuracy_score,f1_score,recall_score,precision_score,confusion_matrix
from plotly.offline import plot
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import datetime
from IPython.display import clear_output
import os
from keras.applications import InceptionResNetV2
from keras.layers import Input, Dense, Flatten, BatchNormalization, Conv2D, Dropout, MaxPooling2D, Activation
import keras
from keras.models import Sequential

class lettuce:


  def __init__(self):

    #class variables
    self.model=None   #public
    self.__IMAGE_SIZE = [224, 224]  #private
#Instantiate an empty model
  def get_model(self):
    model = Sequential()
    # 1st Convolutional Layer
    model.add(Conv2D(filters=96, input_shape=(224,224,3), kernel_size=(11,11), strides=(4,4), padding='valid'))

    model.add(Activation('relu'))
    # Max Pooling
    model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='valid'))

    # 2nd Convolutional Layer
    model.add(Conv2D(filters=256, kernel_size=(11,11), strides=(1,1), padding='valid'))
    model.add(Activation('relu'))
    # Max Pooling
    model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='valid'))

    # 3rd Convolutional Layer
    model.add(Conv2D(filters=384, kernel_size=(3,3), strides=(1,1), padding='valid'))
    model.add(Activation('relu'))

    # 4th Convolutional Layer
    model.add(Conv2D(filters=384, kernel_size=(3,3), strides=(1,1), padding='valid'))
    model.add(Activation('relu'))

    # 5th Convolutional Layer
    model.add(Conv2D(filters=256, kernel_size=(3,3), strides=(1,1), padding='valid'))
    model.add(Activation('relu'))
    # Max Pooling
    model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='valid'))

    # Passing it to a Fully Connected layer
    model.add(Flatten())
    # 1st Fully Connected Layer
    model.add(Dense(4096, input_shape=(224*224*3,)))
    model.add(Activation('relu'))
    # Add Dropout to prevent overfitting
    model.add(Dropout(0.4))

    # 2nd Fully Connected Layer
    model.add(Dense(4096))
    model.add(Activation('relu'))
    # Add Dropout
    model.add(Dropout(0.4))

    # 3rd Fully Connected Layer
    model.add(Dense(1000))
    model.add(Activation('relu'))
    # Add Dropout
    model.add(Dropout(0.4))

    # Output Layer
    model.add(Dense(16))
    model.add(Activation('softmax'))
    # Compile the model
    model.compile(loss=keras.losses.categorical_crossentropy, optimizer='adam', metrics=["accuracy"])
    return model
  
  def retrain(self,training_dir,validation_dir,epochs=30,notebook=False):
    input_shape=(224,224,3)
    n_out=16

    model=self.get_model()

    datagen = ImageDataGenerator(zoom_range=0.2,shear_range=0.2,horizontal_flip=True)
    train = datagen.flow_from_directory(training_dir, class_mode='categorical', batch_size=64,target_size =[224,224],shuffle=False)
    validate = datagen.flow_from_directory(validation_dir, class_mode='categorical', batch_size=16,target_size =[224,224],shuffle=False)
    history=model.fit_generator(train,
                      steps_per_epoch = 498,   
                      epochs = epochs,
                      validation_data = validate,
                      validation_steps = 5,verbose=1)    
    self.model=model

    history=pd.DataFrame(history.history)

    fig = make_subplots(specs=[[{"secondary_y": True}]])

    # Add traces
    fig.add_trace(
        go.Scatter(x=[*range(len(history))], y=history.loss, name="TRAIN_LOSS"),
        secondary_y=False,
    )
    fig.add_trace(
        go.Scatter(x=[*range(len(history))], y=history.val_loss, name="VALID_LOSS"),
        secondary_y=False,
    )
    fig.add_trace(
        go.Scatter(x=[*range(len(history))], y=history.accuracy, name="TRAIN_ACC"),
        secondary_y=True,
    )
    fig.add_trace(
        go.Scatter(x=[*range(len(history))], y=history.val_accuracy, name="VALID_ACC"),
        secondary_y=True,
    )


    # Add figure title
    fig.update_layout(
        title_text="<b>LOSS AND ACCURACY HISTORY</b>",
        title_x=0.5,
        plot_bgcolor='white'
    )

    # Set x-axis title
    fig.update_xaxes(title_text="EPOCHS")


    # Set y-axes titles
    fig.update_yaxes(title_text="LOSS", secondary_y=False)
    fig.update_yaxes(title_text="ACCURACY", secondary_y=True)
    plot(fig,filename='results/Alexnet_Loss_and_Accuracy_History.html',image_width=1080, image_height=720)


    try:
      fig.write_image("results/Alexnet_Loss_and_Accuracy_History.png",width=800,image_height=600)
    except:
      pass
    
    
    mod_name='models/Alexnet_'+datetime.datetime.now().strftime("%d-%m-%Y_%H-%M-%S")+'.h5'
    model.save(mod_name)
    print('\n\nTraining Complete.\nModel Saved as : ',mod_name,'\n') 
    if notebook:
      clear_output()
      return fig   
    else:
      print('Loss and Accuracy graph saved to Results directory')

  def load_trained_model(self,path=0):
    if path:
      self.model=load_model(path)
      print('loaded model : ',path)
    else:
      mod=sorted([i for i in os.listdir('models') if 'Resnet' in i])[-1]
      self.model=load_model(mod)
      print('\n\nNo Path Given. Loaded latest model : ',mod)
  
  def setter(self,x):
    return round(x*100,1)
  
  
  def test(self,input_dir,wide=1200,high=600,notebook=False):
    inp_datagen = ImageDataGenerator(rescale = 1./255)
    inp_generator = inp_datagen.flow_from_directory(input_dir, target_size = [224,224], batch_size = 64, class_mode = 'categorical',shuffle=False)
    Y_pred = self.model.predict_generator(inp_generator)
    y_pred=np.argmax(Y_pred, axis=1)
    y_true=inp_generator.classes
    mapper={v: k for k, v in inp_generator.class_indices.items()}
    df=pd.DataFrame({'True':y_true,'Prediction':y_pred})
    df['True']=df['True'].map(mapper)
    df['Prediction']=df['Prediction'].map(mapper)
    labels=list(df["True"].unique())
    results={'matrix':['Accuracy','F1 Score','Precision','Recall','False Postives']}
    for i in labels:
        tdf=df[df['True']==i]
        tr=tdf['True'].values;prd=tdf['Prediction'].values
        dft=df[df['Prediction']==i]
        
        results[i]=np.array([accuracy_score(tr,prd),
                    f1_score(tr,prd,average='weighted'),
                    precision_score(tr,prd,average='weighted'),
                  recall_score(tr,prd,average='weighted'),
                  (1*(dft['True']!=dft['Prediction'])).mean()])
    res=pd.DataFrame(results).set_index(['matrix'])
    res=res.apply(self.setter)
    textpos='outside'
    fig = go.Figure(data=[
        go.Bar(name='ACCURACY', x=labels, y=res.T.Accuracy,text=res.T.Accuracy,textposition=textpos,textangle=-30),
        go.Bar(name='F1 SCORE', x=labels, y=res.T['F1 Score'],text=res.T['F1 Score'],textposition=textpos,textangle=-30),
        go.Bar(name='PRECISION', x=labels, y=res.T.Precision,text=res.T.Precision,textposition=textpos,textangle=-30),
        go.Bar(name='RECALL', x=labels, y=res.T.Recall,text=res.T.Recall,textposition=textpos,textangle=-30),
        go.Bar(name='FALSE POS', x=labels, y=res.T['False Postives'],text=res.T['False Postives'],textposition=textpos,textangle=-30),
        
    ])
    # Change the bar mode
    #fig.update_traces(texttemplate=res.T, textposition='outside')
    fig.update_layout(barmode='group',plot_bgcolor='white',width=wide, height=high)
    plot(fig,filename='results/Alexnet_Results.html', image_width=wide, image_height=high)
    res.to_csv('results/Alexnet_Results.csv')
    try:
      fig.write_image("results/Alexnet_Results.png")
    except:
      pass
    if notebook:
      clear_output()
      return fig
    else:
      print('Graph saved to results directory')

if __name__=='__main__':
  obj=lettuce()
  train_dir='data/train'
  validation_dir='data/val'
  test_dir='data/test'
  obj.retrain(train_dir,validation_dir,5,notebook=False)
  
  obj.test('data/test',wide=1200,high=600,notebook=False)
