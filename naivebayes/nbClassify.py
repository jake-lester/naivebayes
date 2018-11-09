'''Loads test data and determines category of each test.  Assumes
train/test data with one text-document per line.  First item of each
line is category; remaining items are space-delimited words.  

Author: Jacob Lester

Date: 6.Nov.2017

project:naivebayesPractice

'''
from __future__ import print_function
import sys
import math

class NaiveBayes():
    '''Naive Bayes classifier for text data.
    Assumes input text is one text sample per line.  
    First word is classification, a string.
    Remainder of line is space-delimited text.
    '''

    def __init__(self,train):
        '''Create classifier using train, the name of an input
        training file.
        '''
        self.makeData()
        self.learn(train) # loads train data, fills prob. table



    def makeData(self):
        """ Creates the motherload dictionary:"""
        self.Data = {}                  #empty dicitionary to store data
        self.Data["train"] = {}         #contains only data from train.txt
        self.Data["train"]["nw"] = 0    #total count of word (w) instances
        self.Data["train"]["nv"] = 0    #total count of category (v) instances
        self.Data["train"]["v"] = {}    #contains each distinct-category (v_j) as a key
        self.Data["train"]["w"] = {}    #contains each distinct-word (w_k) as a key
        self.Data["test"] = {}          #contains only data from test.txt
        self.Data["test"] = {}     #contains each distinct category (v_j) as a key


    def learn(self,traindat):
        '''Load data for training; adding to 
        dictionary of classes and counting words.'''
        with open(traindat,'r') as fd:
            for line in fd.readlines():
                id,*words = line.split()
                self.Data["train"]["nv"] = self.Data["train"]["nv"] + 1 #counts v instances
                
                if id not in self.Data["train"]["v"]:
                    self.Data["train"]["v"][id] = {}            #updates Data with new v_j as a key
                    self.Data["train"]["v"][id]["prior_j"] = {} #contains prior probability of v_j
                    self.Data["train"]["v"][id]["nw_j"] = 0     #initializes total count of words in v_j
                    self.Data["train"]["v"][id]["docs_j"] = 0   #initializes total count of v_j instances
                    self.Data["train"]["v"][id]["w_j"] = {}     #contains each w_k in v_j as a key
                self.Data["train"]["v"][id]["docs_j"] = self.Data["train"]["v"][id]["docs_j"] + 1 #counts v_j instances
                
                for word in range(len(words)):
                    self.Data["train"]["nw"] = self.Data["train"]["nw"] + 1 #counts w instances
                    self.Data["train"]["v"][id]["nw_j"] = self.Data["train"]["v"][id]["nw_j"] + 1 #counts w instances in v_j

                    if words[word] not in self.Data["train"]["w"]:
                        self.Data["train"]["w"][words[word]] = {}           #updates Data with new w_k as a key
                        self.Data["train"]["w"][words[word]]["n_k"] = 0     #initializes count of w_k instances
                        self.Data["train"]["w"][words[word]]["docs"] = 0    #initializes count of w_k present in each v instance

                    if words[word] not in self.Data["train"]["v"][id]["w_j"]:
                        self.Data["train"]["w"][words[word]]["docs"] = self.Data["train"]["w"][words[word]]["docs"] + 1
                        self.Data["train"]["v"][id]["w_j"][words[word]] = 0 #initializes count of w_k instances in v_j
                        
            
                    self.Data["train"]["v"][id]["w_j"][words[word]] = self.Data["train"]["v"][id]["w_j"][words[word]] + 1 #counts w_k instances in v_j
                    self.Data["train"]["w"][words[word]]["n_k"] = self.Data["train"]["w"][words[word]]["n_k"] + 1 #counts w_k instances


            for w_k in self.Data["train"]["w"]:
                for v_j in self.Data["train"]["v"]:
                    if w_k not in self.Data["train"]["v"][v_j]["w_j"]:
                        self.Data["train"]["v"][v_j]["w_j"][w_k] = 0    #updates Data with w_k not in v_j
                    

    def runTest(self,test):
        with open(test,'r') as sd:
            for line in sd.readlines():
                id,*words = line.split()
                if id not in self.Data["test"]:
                    self.Data["test"][id] = {}             #updates new v_j as a key
                    self.Data["test"][id]["docs_j"] = 0
                    self.Data["test"][id]["raw"] = {}
                    self.Data["test"][id]["raw"]["nCorrect"] = 0
                    self.Data["test"][id]["mest"] = {}
                    self.Data["test"][id]["mest"]["nCorrect"] = 0
                    self.Data["test"][id]["tfidf"] = {}
                    self.Data["test"][id]["tfidf"]["nCorrect"] = 0
                self.Data["test"][id]["docs_j"] = self.Data["test"][id]["docs_j"] + 1 #keeps a running count of v_j instances in test data

            self.raw(test)
            self.mest(test)
            self.tfidf(test)

    def raw(self, test):
        with open(test,'r') as sd:
            for line in sd.readlines():
                id,*words = line.split()
                Token = {}
                for v_j in self.Data["train"]["v"]:
                    j_prob = self.Data["train"]["v"][v_j]["docs_j"] / self.Data["train"]["nv"] #calculates P(v_j)
                    for word in range(len(words)):
                        w_k = words[word]
                        if w_k not in self.Data["train"]["w"]:
                            pass
                        else:
                            j_prob = j_prob * (self.Data["train"]["v"][v_j]["w_j"][w_k] / self.Data["train"]["v"][v_j]["nw_j"])
                    Token[j_prob] = v_j
                v_h = Token[max(Token)]

                if v_h == id:
                    self.Data["test"][id]["raw"]["nCorrect"] = self.Data["test"][id]["raw"]["nCorrect"] + 1

    def mest(self, test):
        with open(test,'r') as sd:
            for line in sd.readlines():
                id,*words = line.split()
                Token = {}
                
                for v_j in self.Data["train"]["v"]:
                    j_prob = self.Data["train"]["v"][v_j]["docs_j"] / self.Data["train"]["nv"] #prior P(v_j)
                    for word in range(len(words)):
                        if words[word] not in self.Data["train"]["w"]:
                            pass
                        else:
                            j_prob = j_prob * (self.Data["train"]["v"][v_j]["w_j"][words[word]] + 1)/ (self.Data["train"]["v"][v_j]["nw_j"] + self.Data["train"]["nw"])
                            #Calculates P(v_j|a_1,a_2,...a_n)
                    Token[j_prob] = v_j
                v_h = Token[max(Token)]
                if v_h == id:
                    self.Data["test"][id]["mest"]["nCorrect"] = self.Data["test"][id]["mest"]["nCorrect"] + 1

    def tfidf(self, test):
        with open(test,'r') as sd:
            for line in sd.readlines():
                id,*words = line.split()
                Token = {}
                for v_j in self.Data["train"]["v"]:
                    j_prob = 1
                    for word in range(len(words)):
                        if words[word] not in self.Data["train"]["w"]:
                            pass
                        else:
                            if words[word] in self.Data["train"]["v"][v_j]["w_j"]:
                                tf = ((self.Data["train"]["v"][v_j]["w_j"][words[word]] + 1) / (self.Data["train"]["v"][v_j]["nw_j"]))
                            else:
                                tf = (1)/((self.Data["train"]["v"][v_j]["nw_j"]))
                            idf = math.log(len(self.Data["train"]["v"]))/(self.Data["train"]["w"][words[word]]["docs"])
                            #Tf-Idf weight = ((w_k instances count in v_j) / (w instances count in v_j)) * log((total count w instances)/(count of w_j instances))
                            j_prob = j_prob * tf * idf
                    Token[j_prob] = v_j
                v_h = Token[max(Token)]
                if v_h == id:
                    self.Data["test"][id]["tfidf"]["nCorrect"] = self.Data["test"][id]["tfidf"]["nCorrect"] + 1


    def printClasses(self):
        for version in ["raw","mest","tfidf"]:
            print("VERSION: ",version)
            print("Category ---------------------- NCorrect --- N ------- Accuracy")
            for v_j in self.Data["test"]:
                accuracy = round(self.Data["test"][v_j][version]["nCorrect"] / self.Data["test"][v_j]["docs_j"], 3)
                spacing1 = ""
                spacing2 = ""
                for i in range(30 - len(v_j)):
                    spacing1 = spacing1 + "-"
                for i in range (11 - len(str(self.Data["test"][v_j][version]["nCorrect"]))):    
                    spacing2 = spacing2 + "-"

                print(v_j,spacing1,self.Data["test"][v_j][version]["nCorrect"],spacing2,self.Data["test"][v_j]["docs_j"]," ----- ",accuracy)


def argmax(lst):
    return lst.index(max(lst))
    
def main():
    if len(sys.argv) != 4:
        print("Usage: %s trainfile testfile" % sys.argv[0])
        sys.exit(-1)

    nbclassifier = NaiveBayes(sys.argv[1])
    nbclassifier.runTest(sys.argv[2])
    nbclassifier.printClasses()


if __name__ == "__main__":
    main()


#cd "C:\Users\Jacob\Desktop\ai hw 4\hw4_nb\"
#type fakey_1.txt
#type nbClassify.py fakey_1.txt fakey_1.txt
#python nbClassify.py fakey_1.txt fakey_1.txt
#python nbClassify.py 20ng-train-stemmed.txt 20ng-test-stemmed.txt ['raw','mest','tfidf']


    
