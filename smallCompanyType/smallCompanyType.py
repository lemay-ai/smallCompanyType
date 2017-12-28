import gensim
from gensim.models import Word2Vec
from sklearn import preprocessing
from keras.models import load_model
import numpy as np
import re 
from os import path

class SmallCompanyType:
    
    le=None
    embedding_model=None
    dnn_model=None
    
    def __init__(self):
       
        models_dir = path.join(path.dirname(__file__), 'models')
        self.le = preprocessing.LabelEncoder()
        self.le.classes_ = np.load(models_dir+'/labelEncoderClasses.npy')
        self.embedding_model = gensim.models.Word2Vec.load(models_dir+'/companyVectors')
        self.dnn_model = load_model(models_dir+'/0_1024_companySector_model.h5')
    
    def doit(self):
        print("Dirty Mind!")
        return
    
    def getCompanyVector(self, name):
        overallVec=np.zeros(200)
        wordCount=0
        for word in name.split():
            if word in self.embedding_model:
                overallVec=np.add(self.embedding_model[word],overallVec)
                wordCount+=1
        if wordCount>0:
            overallVec=overallVec/wordCount
        return overallVec

    def getCompanyType(self, name):
        sample=self.getCompanyVector(name)
        sampleMod = sample[np.newaxis,:]
        predicted = (self.dnn_model.predict(sampleMod)[0])
        return self.le.inverse_transform(np.argmax(predicted))
    
    #title_except is from https://stackoverflow.com/questions/3728655/titlecasing-a-string-with-exceptions
    def title_except(self, s, exceptions=['a', 'an', 'of', 'the', 'is']):
        word_list = re.split(' ', s)       # re.split behaves as expected
        final = [word_list[0].capitalize()]
        for word in word_list[1:]:
            final.append(word if word in exceptions else word.capitalize())
        return " ".join(final)
