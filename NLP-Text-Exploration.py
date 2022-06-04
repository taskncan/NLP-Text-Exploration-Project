"""
@author: cantaskin33
"""
import re
import nltk
import gensim 
import nltk.data 
import math
import numpy as np
from nltk import ngrams
from wordcloud import WordCloud
from nltk.corpus import stopwords
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer,TfidfTransformer
from operator import itemgetter
from scipy.stats import zipf
from nltk.lm import MLE
from nltk.lm.preprocessing import padded_everygram_pipeline
from nltk import word_tokenize
from nltk.lm.models import KneserNeyInterpolated
from nltk.lm.api import LanguageModel
from gensim.models import Word2Vec 
nltk.download('punkt')
nltk.download('stopwords')
Stopwords = set(stopwords.words('turkish')) 

import pickle
with open('T_sample5000.pkl', 'rb') as f:
    Docs = pickle.load(f)

         
def create_WordCloud(Docs,size,wordcloud_outputfile,mode,stopwords):

   lst = []

   for item in Docs: # Creating lst contain item in Docs

       item = re.sub(r'[^a-zçğıöşüA-ZÇĞİÖŞÜ]', ' ', item)

       for element in item.split(' '): 

           lst.append(element)   
           
   if stopwords == True: #BagOfWords with Turkish Stopwords 

       bagOfWords = lst
       uniqueWords = set(bagOfWords)
       numOfWords = dict.fromkeys(uniqueWords, 0)

       for word in bagOfWords:

           numOfWords[word] += 1   

   else:                #BagOfWords without Turkish Stopwords

       bagOfWords = lst
       uniqueWords = set(bagOfWords)
       numOfWords = dict.fromkeys(uniqueWords, 0)

       for word in bagOfWords:

           if word not in Stopwords:

               numOfWords[word] += 1  
           
   if mode == "TF": #Term weighting option: TF

     tfDict = {}
     bagOfWordsCount = len(bagOfWords)

     for word, count in numOfWords.items():

         tfDict[word] = count / float(bagOfWordsCount)

     wordcloud = WordCloud(background_color="white",height = size, width = size,stopwords= Stopwords).generate_from_frequencies(tfDict) 
     wordcloud.to_file(wordcloud_outputfile)
      
   else:          #Term weighting option: TFIDF

     tfDict = {}
     idfDict = {}
     tfidf = {}
     bagOfWordsCount = len(bagOfWords)

     for word, count in numOfWords.items():

         tfDict[word] = count / float(bagOfWordsCount)

     length = len(numOfWords)
     idfDict = dict.fromkeys(numOfWords.keys(), 0)

     for word, val in idfDict.items():

        idfDict[word] = math.log(length / (float(val) + 1))

     for word, val in tfDict.items():

        tfidf[word] = val*idfDict[word]

     wordcloud = WordCloud(background_color="white",height = size, width = size, stopwords = Stopwords).generate_from_frequencies(tfidf)
     wordcloud.to_file(wordcloud_outputfile)



def create_ZiphsPlot(Docs,zips_outputfile):

    frequency = {}
    lst = []

    for item in Docs:

       for element in item.split(' '): 

           lst.append(element)    

    for word in lst:

        if word not in Stopwords:

            count = frequency.get(word,0)
            frequency[word] = count + 1 

    collection = sorted(frequency.items(), key=itemgetter(1), reverse = True)
    total = sum([value for key, value in collection[:30]])

    fig = plt.figure(figsize=(10,10)) 
    plt.ylabel("Frequency")
    plt.xlabel("Words")
    plt.xticks(rotation=90)

    for word , freq in collection[:30]:
        plt.bar(word, freq)  
    plt.plot(range(len(collection[:30])), [zipf.pmf(p, 1.4) * total for p in range(1, len(collection[:30]) + 1)],color = 'red') #Ziphs Law Plot
    fd = nltk.FreqDist(frequency)
    fd.plot(30) #Calculated Values Plot 
    fig.savefig(zips_outputfile)

    
def create_HeapsPlot(Docs,heaps_outputfile):

    lst = []

    for item in Docs:

       for element in item.split(' '): 

           lst.append(element)

    uniqueWords = set(lst)
    uniqueDict = dict.fromkeys(uniqueWords, 0)
    
    for word in lst:

        if word not in Stopwords:

            uniqueDict[word] += 1  

    fig = plt.figure(figsize=(10,10))
    plt.xticks(rotation=90)
    plt.xlabel("Total Words")
    plt.ylabel("Unique Words")
    plt.plot([100*(v**0.4) for v in range(1,len(lst))],color="red") # Heaps Law / Numbers can be change 
    fig.savefig(heaps_outputfile)
    

def create_LanguageModel(Docs,model_type,ngram):

    tokenized_text = []

    for item in Docs:

       item = re.sub(r'[^a-zçğıöşüA-ZÇĞİÖŞÜ\.\,]', ' ', item)
       tokenize = word_tokenize(item.lower())
       tokenized_text.append(tokenize)

    if model_type == 'MLE':  
        
       train_data, vocab = padded_everygram_pipeline(ngram, tokenized_text)
       model = MLE(ngram) # Lets train a N-grams maximum likelihood estimation model.
       model.fit(train_data, vocab)
       return model

    if model_type == 'KneserNeyInterpolated':

       train_data, vocab = padded_everygram_pipeline(ngram, tokenized_text)
       model = KneserNeyInterpolated(ngram) # Lets train a N-grams Kneser Ney Interpolated model.
       model.fit(train_data, vocab)
       return model

def generate_sentence(LM,text):

    sentences = []
    lst = []
    n = 3
    perplexity = []
    counter = 0
    check = True
    iterations = 5

    sentences.append(text)

    for _ in range(iterations):

       while check: 

         word = LM.generate(num_words= 1,text_seed=[sentences]) #Generate new word from previous generated word 
         sentences.append(word) #Append the new word to sentences list
         counter += 1  

         if word == "</s>":

            sentence = " ".join(sentences) #Join words to create sentece
            
            if sentence not in lst:

                sentences.clear()
                sentences.append(text)
                counter = 0
                lst.append(sentence.replace("</s>","."))      
                ngram = ngrams(sentence.split(),n,left_pad_symbol='.') #Create Ngram model 
                perplexity.append(LM.perplexity(ngram)) #Find perplexity of ngram model 
                break

    for i in range(0,len(perplexity)):

        if perplexity[i] == min(perplexity):

             return lst[i],perplexity[i] #Return minimum perplexity sentence (Return 'inf' for perplexity WHY?)


def create_WordVectors(Docs,dimension,modelType,windowSize):

    tokenized_text = []

    for item in Docs:

       item = re.sub(r'[^a-zçğıöşüA-ZÇĞİÖŞÜ]', ' ', item)

       if item != '':

          tokenized_text.append(word_tokenize(item.lower()))    

    if modelType == 'cbow':

        modelCbow = gensim.models.Word2Vec(tokenized_text,min_count=1,size = dimension, window = windowSize)
        return modelCbow

    if modelType == 'skipgram':

        modelSkipgram = gensim.models.Word2Vec(tokenized_text,min_count=1,size = dimension, window = windowSize,sg = 1)
        return modelSkipgram
    

def use_WordRelationship(WE,example_tuple_list,example_tuple_test):

     lst = []
     tuple_list = list(example_tuple_list)

     for item in tuple_list:

         if item[0] not in WE.wv.vocab or item[1] not in WE.wv.vocab:

            example_tuple_list.remove(item)

         if len(tuple_list) == 0 or example_tuple_test[0] not in WE.wv.vocab:

            print("Sorry, this operation cannot be performed!")
            break

         vec = WE[item[0]] - WE[item[1]]
         lst.append(vec)
         
     lst = sum(lst)/len(lst)
     lst = lst + WE.wv.get_vector(example_tuple_test[0]) 
     print(WE.wv.similar_by_vector(lst,topn=5))
     
