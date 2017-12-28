import gensim
from gensim.models import Word2Vec
from sklearn import preprocessing
from keras.models import load_model
import numpy as np
import re 

#title_except is from https://stackoverflow.com/questions/3728655/titlecasing-a-string-with-exceptions
articles = ['a', 'an', 'of', 'the', 'is']
def title_except(s, exceptions):
    word_list = re.split(' ', s)       # re.split behaves as expected
    final = [word_list[0].capitalize()]
    for word in word_list[1:]:
        final.append(word if word in exceptions else word.capitalize())
    return " ".join(final)

embedding_model = gensim.models.Word2Vec.load('../models/companyVectors')
dnn_model = load_model('../models/0_1024_companySector_model.h5')
le = preprocessing.LabelEncoder()
le.classes_ = np.load('../models/labelEncoderClasses.npy')

def getCompanyVector(name):
    overallVec=np.zeros(200)
    wordCount=0
    for word in name.split():
        if word in embedding_model:
            overallVec=np.add(embedding_model[word],overallVec)
            wordCount+=1
    if wordCount>0:
        overallVec=overallVec/wordCount
    return overallVec

def getCompanyType(name):
    sample=getCompanyVector(name)
    sampleMod = sample[np.newaxis,:]
    predicted = (dnn_model.predict(sampleMod)[0])
    return le.inverse_transform(np.argmax(predicted))
