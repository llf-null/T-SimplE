# Copyright (c) 2018-present, Royal Bank of Canada.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
import random
import math
import copy
import time
import numpy as np
from random import shuffle
from scripts import shredFacts

class Dataset:
    """Implements the specified dataloader"""
    def __init__(self, 
                 ds_name,model_name):
        """
        Params:
                ds_name : name of the dataset 
        """
        self.name = ds_name
        self.ds_path = "datasets/" + ds_name.lower() + "/"
        self.ent2id = {}
        self.rel2id = {}
        self.time2id={}
        self.year2id={}
        self.month2id={}
        self.day2id={}

        self.data = {"train": self.readFile(self.ds_path + "train.txt",model_name),
                     "valid": self.readFile(self.ds_path + "valid.txt",model_name),
                     "test":  self.readFile(self.ds_path + "test.txt",model_name)}
        
        self.start_batch = 0
        self.all_facts_as_tuples = None
        self.all_facts_as_tuples = set([tuple(d) for d in self.data["train"] + self.data["valid"] + self.data["test"]])
        
        for spl in ["train", "valid", "test"]:
            self.data[spl] = np.array(self.data[spl])

        
    def readFile(self, 
                 filename,model_name):

        with open(filename, "r",encoding='UTF-8') as f:
            data = f.readlines()
        
        facts = []
        for line in data:
            elements = line.strip().split("\t")
            elements[3]=elements[3].replace('#','0')
            head_id =  self.getEntID(elements[0])
            rel_id  =  self.getRelID(elements[1])
            tail_id =  self.getEntID(elements[2])
            time_id= self.getTimeID(elements[3])
            year_id=self.getYearID(elements[3])
            month_id=self.getMonthID(elements[3])
            day_id=self.getDayID(elements[3])
            timestamp = elements[3]
            facts.append([head_id, rel_id, tail_id,time_id])
            
        return facts
                
    
    def numEnt(self):
    
        return len(self.ent2id)

    def numRel(self):
    
        return len(self.rel2id)
    def numTime(self):
        
        return len(self.time2id)
    def numYear(self):
        return len(self.year2id)
    def numMonth(self):
        return len(self.month2id)
    def numDay(self):
        return len(self.day2id)
        

    
    def getEntID(self,
                 ent_name):

        if ent_name in self.ent2id:
            return self.ent2id[ent_name] 
        self.ent2id[ent_name] = len(self.ent2id)
        return self.ent2id[ent_name]
    
    def getRelID(self, rel_name):
        if rel_name in self.rel2id:
            return self.rel2id[rel_name] 
        self.rel2id[rel_name] = len(self.rel2id)
        return self.rel2id[rel_name]

    # def getTimeID(self,time_date):
    #     if time_date in self.time2id:
    #         return self.time2id[time_date]
    #     self.time2id[time_date]=len(self.time2id)
    #     return self.time2id[time_date]
    
#___________________________________________________________
#The granularity is month rather than date
    def getTimeID(self,time_date):
        if time_date[:7] in self.time2id:
            return self.time2id[time_date[:7]]
        self.time2id[time_date[:7]]=len(self.time2id)
        return self.time2id[time_date[:7]]
#____________________________________________________________


#___________________________________________________________
#The granularity is year rather than date
    # def getTimeID(self,time_date):
    #     if time_date[:4] in self.time2id:
    #         return self.time2id[time_date[:4]]
    #     self.time2id[time_date[:4]]=len(self.time2id)
    #     return self.time2id[time_date[:4]]
#____________________________________________________________    
    
    def getYearID(self,time_date):
        if time_date[:4] in self.year2id:
            return self.year2id[time_date[:4]]
        self.year2id[time_date[:4]]=len(self.year2id)
        return self.year2id[time_date[:4]]

    def getMonthID(self,time_date):
        if time_date[5:7] in self.month2id:
            return self.month2id[time_date[5:7]]
        self.month2id[time_date[5:7]]=len(self.month2id)
        return self.month2id[time_date[5:7]]
    
    def getDayID(self,time_date):
        if time_date[8:11] in self.day2id:
            return self.day2id[time_date[8:11]]
        self.day2id[time_date[8:11]]=len(self.day2id)
        return self.day2id[time_date[8:11]]
#____________________________________________________________________    
    def nextPosBatch(self, batch_size):
        if self.start_batch + batch_size > len(self.data["train"]):
            ret_facts = self.data["train"][self.start_batch : ]
            self.start_batch = 0
        else:
            ret_facts = self.data["train"][self.start_batch : self.start_batch + batch_size]
            self.start_batch += batch_size
        return ret_facts
    

    def addNegFacts(self, bp_facts, neg_ratio):
        ex_per_pos = 2 * neg_ratio + 2
        facts = np.repeat(np.copy(bp_facts), ex_per_pos, axis=0)
        for i in range(bp_facts.shape[0]):
            s1 = i * ex_per_pos + 1
            e1 = s1 + neg_ratio
            s2 = e1 + 1
            e2 = s2 + neg_ratio
            
            facts[s1:e1,0] = (facts[s1:e1,0] + np.random.randint(low=1, high=self.numEnt(), size=neg_ratio)) % self.numEnt()
            facts[s2:e2,2] = (facts[s2:e2,2] + np.random.randint(low=1, high=self.numEnt(), size=neg_ratio)) % self.numEnt()
            
        return facts
    
    def addNegFacts2(self, bp_facts, neg_ratio):
        pos_neg_group_size = 1 + neg_ratio
        facts1 = np.repeat(np.copy(bp_facts), pos_neg_group_size, axis=0)
        facts2 = np.copy(facts1)
        rand_nums1 = np.random.randint(low=1, high=self.numEnt(), size=facts1.shape[0])
        rand_nums2 = np.random.randint(low=1, high=self.numEnt(), size=facts2.shape[0])
        for i in range(facts1.shape[0] // pos_neg_group_size):
            rand_nums1[i * pos_neg_group_size] = 0
            rand_nums2[i * pos_neg_group_size] = 0

        facts1[:,0] = (facts1[:,0] + rand_nums1) % self.numEnt()
        facts2[:,2] = (facts2[:,2] + rand_nums2) % self.numEnt()
        return np.concatenate((facts1, facts2), axis=0)
    
    def nextBatch(self, batch_size,model_name, neg_ratio=1):
        bp_facts = self.nextPosBatch(batch_size)
        batch = shredFacts(self.addNegFacts2(bp_facts, neg_ratio))
        return batch
    
    
    def wasLastBatch(self):
        return (self.start_batch == 0)
            
