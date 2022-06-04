# NLP-Text-Exploration-Project

Here are the functions implemented:

- Create_WordCloud:
This function will take the following parameters as input:
o List of documents
o The dimension size (same dimension will be used for both height and width of the output figure)
o The full path of the output file
o The term weighting option (either TF or TFIDF weighting, default is TF)
o The stopwords option (either to remove the stopwords or not, default is to keep them)
The function will create the word cloud and save it in the provided output file in png format.

- Create_ZipfsPlot:
This function will take the following parameters as input:
o List of documents
o The full path of the output file
The function will create the Zipf’s plot and save it in the provided output file in png format.

- Create_HeapsPlot:
This function will take the following parameters as input:
o List of documents
o The full path of the output file
The function will create the Heap’s plot and save it in the provided output file in png format.

- Create_LanguageModel:
This function will take the following parameters as input:
o Type of the model (MLE or KneserNeyInterpolated)
o List of documents
o Maximum ngram size
The function will return the trained language model.

- Generate_Sentence:
This function will take the following parameters as input:
o Trained language model
o Text for starting the sentence
