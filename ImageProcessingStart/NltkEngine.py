import os
import nltk
import nltk.corpus
from VGGModel import Landmark
from Engine.Predict import PredictFood
from Engine.customPrediction.RunCustomTrain import Object
from nltk.stem import wordnet,WordNetLemmatizer
from nltk.corpus import stopwords
stopwords.words("english")
class SearchEngine:
    def StartEngine(self):
        print("hello \n Enter Your Questions below \n")
        string = input()
        """quotes_token = nltk.word_tokenize(string)"""
        # function to test if something is a noun
        noun = self.FilterNoun(string)
        print(noun)
        self.FilterOutput(noun)


    def FilterNoun(self,str):
        tokenized = nltk.word_tokenize(str)
        nouns = [word for (word, pos) in nltk.pos_tag(tokenized) if pos[:2] == 'NN']
        return nouns

    def FilterOutput(self,nouns):
        land = Landmark().DefineLandmark()
        food = PredictFood().FoodDetection()
        object = Object().ObjectDetect()
        landmark = []
        for each in nouns:
            noun = each.lower()
            for i in land:
                landmark.append(i[1])

            if noun.__contains__("land"):
                print(landmark)
            elif noun.__contains__("food"):
                print(food)
            elif noun.__contains__("item"):
                print(object)
            else:
                if str(landmark).__contains__(noun) or str(food).__contains__(noun) or str(object).__contains__(noun):
                    print("Yes ", noun, " is present")
                else:
                    print(noun, " is not present in Database")



ob = SearchEngine()
ob.StartEngine()