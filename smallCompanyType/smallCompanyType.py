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
    typeDict= { 'Acupuncturist'			:'B2C',
                'Adult Entertainment Store'	:'B2C',
                'Animal Clinic/Hospital'	:'B2C',
                'Animal Services'		:'B2C',
                'Artist'			:'B2BC',
                'Artist Live/Work Studio'	:'B2BC',
                'Assembly Hall'			:'PUB',
                'Auctioneer'			:'B2BC',
                'Auto Dealer'			:'B2BC',
                'Auto Detailing'		:'B2C',
                'Auto Painter & Body Shop'	:'B2C',
                'Auto Parking Lot/Parkade'	:'B2C',
                'Auto Repairs'			:'B2C',
                'Auto Washer'			:'B2C',
                'Auto Wholesaler'		:'B2B',
                'Beauty Services'		:'B2C',
                'Bed and Breakfast'		:'B2C',
                'Boat Charter Services'		:'B2BC',
                'Booking Agency'		:'B2BC',
                'Boot & Shoe Repairs'		:'B2C',
                'Business Services'		:'B2B',
                'Carpet/Upholstery Cleaner'	:'B2BC',
                'Caterer'			:'B2BC',
                'Club'				:'B2BC',
                'Community Association'		:'PUB',
                'Computer Services'		:'B2BC',
                'Contractor'			:'B2BC',
                'Contractor - Special Trades'	:'B2BC',
                'Cosmetologist'			:'B2C',
                'Dance Hall'			:'B2C',
                'Dating Services'		:'B2C',
                'ESL Instruction'		:'B2C',
                'Educational'			:'PUB',
                'Electrical Contractor'		:'B2BC',
                'Electrical-Security Alarm Installation' :'B2BC',
                'Employment Agency'		:'B2BC',
                'Entertainment Services'	:'B2BC',
                'Equipment Operator'		:'B2BC',
                'Exhibitions/Shows/Concerts'	:'B2BC',
                'Financial Institution'		:'B2BC',
                'Financial Services'		:'B2BC',
                'Fitness Centre'		:'B2C',
                'Food Processing'		:'B2B',
                'Gas Contractor'		:'B2BC',
                'Gasoline Station'		:'B2C',
                'Hair Stylist/Hairdresser'	:'B2C',
                'Health Services'		:'B2C',
                'Health and Beauty'		:'B2C',
                'Home Business'			:'B2BC',
                'Homecraft'			:'B2B',
                'Hotel'				:'B2BC',
                'Instruction'			:'B2BC',
                'Janitorial Services'		:'B2B',
                'Jeweller'			:'B2C',
                'Laboratory'			:'B2B',
                'Landscape Gardener'		:'B2BC',
                'Late Night Dance Event'	:'B2C',
                'Laundry'			:'B2C',
                'Liquor Equipment'		:'B2B',
                'Liquor Establishment'		:'B2C',
                'Liquor License Application'	:'B2C',
                'Liquor Retail Store'		:'B2C',
                'Locksmith'			:'B2BC',
                'Manufacturer'			:'B2B',
                'Manufacturer - Food'		:'B2B',
                'Marina Operator'		:'B2BC',
                'Marine Services'		:'B2BC',
                'Massage Therapist'		:'B2C',
                'Money Services'		:'B2BC',
                'Moving/Transfer Service'	:'B2BC',
                'Non-profit Housing'		:'PUB',
                'Office'			:'B2BC',
                'Painter'			:'B2BC',
                'Pawnbroker'			:'B2C',
                'Personal Care Home'		:'B2C',
                'Personal Services'		:'B2C',
                'Pest Control/Exterminator'	:'B2BC',
                'Pet Store'			:'B2C',
                'Photo Services'		:'B2C',
                'Photographer'			:'B2BC',
                'Physical Therapist'		:'B2C',
                'Plumber'			:'B2BC',
                'Plumber & Gas Contractor'	:'B2BC',
                'Plumber & Sprinkler Contractor':'B2BC',
                'Plumber Sprinkler & Gas Contractor':'B2BC',
                'Postal Rental Agency'		:'B2BC',
                'Power/ Pressure Washing'	:'B2BC',
                'Printing Services'		:'B2BC',
                'Product Assembly'		:'B2B',
                'Production Company'		:'B2BC',
                'Property Management'		:'B2BC',
                'Psychic/Fortune Teller'	:'B2C',
                'Real Estate Dealer'		:'B2BC',
                'Recycling Depot'		:'PUB',
                'Referral Services'		:'B2BC',
                'Rentals'			:'B2BC',
                'Repair/ Service/Maintenance'	:'B2BC',
                'Residential/Commercial'	:'B2BC',
                'Restaurant'			:'B2C',
                'Retail Dealer'			:'B2C',
                'Retail Dealer - Food'		:'B2C',
                'Retail Dealer - Grocery'	:'B2C',
                'Roofer'			:'B2BC',
                'Rooming House'			:'PUB',
                'Scavenging'			:'B2BC',
                'School (Business & Trade)'	:'PUB',
                'School (Private)'		:'PUB',
                'Seamstress/Tailor'		:'B2C',
                'Secondary Suite - Permanent'	:'B2B',
                'Secondhand Dealer'		:'B2C',
                'Security Services'		:'B2BC',
                'Social Escort Services'	:'B2C',
                'Soliciting For Charity'	:'PUB',
                'Sprinkler Contractor'		:'B2BC',
                'Studio'			:'B2BC',
                'Talent Agency'			:'B2BC',
                'Tanning Salon'			:'B2C',
                'Tattoo Parlour'		:'B2C',
                'Telecommunications'		:'B2BC',
                'Theatre'			:'B2C',
                'Therapeutic Touch Technique'	:'B2C',
                'Travel Agent'			:'B2BC',
                'Venue'				:'B2BC',
                'Warehouse Operator'		:'B2B',
                'Wholesale  Dealer'		:'B2B',
                'Wholesale Dealer - Food'	:'B2B',
                'Window Cleaner'		:'B2BC'}
    
    # Load the model components
    def __init__(self):
        models_dir = path.join(path.dirname(__file__), 'models')
        self.le = preprocessing.LabelEncoder()
        self.le.classes_ = np.load(models_dir+'/labelEncoderClasses.npy')
        self.embedding_model = gensim.models.Word2Vec.load(models_dir+'/companyVectors')
        self.dnn_model = load_model(models_dir+'/0_1024_companySector_model.h5')
    
    # Get a vector embedding for a company name string
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

    # Predict and return a categorical label for an input company name string
    def getCompanySubtype(self, name):
        sample=self.getCompanyVector(name)
        sampleMod = sample[np.newaxis,:]
        predicted = (self.dnn_model.predict(sampleMod)[0])
        return self.le.inverse_transform(np.argmax(predicted))
    
    # Predict and return a categorical label for an input company name string
    def getCompanyType(self, name):)
        return self.typeDict[self.getCompanySubtype(self, name)]
    