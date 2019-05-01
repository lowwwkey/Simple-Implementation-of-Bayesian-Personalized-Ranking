#!/usr/bin/env python
#-*-coding:utf-8-*-

'''
@File       : imp_bpr.py
@Discription: 对BPR算法进行改进及简单实现
@Author     : Guangkai Li
@Date:      : 2018/08/15
'''

import json
import numpy as np
import operator
import sys
from collections import defaultdict
from math import exp, log
from random import choice
from sklearn import preprocessing
from sklearn.metrics import roc_auc_score

class BPR(object):
    """对BPR算法进行改进，应用于需求匹配.

    Args:
        path: 数据集路径.
        k: 矩阵维度.
        max_iter: 最大训练迭代次数.
        learning_rate: 学习率.
    """
    
    def __init__(self, training_path, test_path):
        self.training_path = training_path
        self.test_path = test_path
        self.k = 50
        self.max_iter = 100
        self.learning_rate = 0.05
        self.max_learning_rate = 0.1
        self.reg_u = 0.01
        self.reg_i = 0.01
        self.user = {}
        self.item = {}
        self.id2user = {}
        self.id2item = {}
        self.training_set_u = defaultdict(dict)
        
    def load_training_data(self):
        with open(self.training_path, 'r') as f:
            return [i.strip().split(' ') for i in f.readlines()]
        
    def data_pre_processing(self):
        print 'Preparing...'
        self.training_data = self.load_training_data()
        for entry in self.training_data:
            user_name, item_name, rating = entry
            if user_name not in self.user:
                self.user[user_name] = len(self.user)
                self.id2user[self.user[user_name]] = user_name
            if item_name not in self.item:
                self.item[item_name] = len(self.item)
                self.id2item[self.item[item_name]] = item_name
            if float(rating) > 30:
                rating = 5.0
            elif float(rating)>15 and float(rating)<=30:
                rating = 4.0
            elif float(rating)>7 and float(rating)<=15:
                rating = 3.0
            elif float(rating)>3 and float(rating)<=7:
                rating = 2.0
            else:
                rating = 1.0
            self.training_set_u[user_name][item_name] = float(rating)
        with open('./training_results/user2id.json','w') as f_u:
            json.dump(self.user,f_u,ensure_ascii=False)
            f_u.write('\n')
        with open('./training_results/item2id.json','w') as f_i:
            json.dump(self.item,f_i,ensure_ascii=False)
            f_i.write('\n')
            
    def print_data_statistic(self, flag=True):
        width = 60
        width_header = width//3
        print '-'*width 
        if flag:
            print 'Training set size'.center(width)
            set_size = self.training_size
        else:
            print 'Test set size'.center(width)
            set_size = self.test_size
        print '-'*width
        print 'user count'.ljust(width_header) + 'item count'.ljust(width_header) + 'record count'.ljust(width_header)
        print '-'*width
        print '{0:<20}{1:<20}{2:<20}'.format(set_size()[0], set_size()[1], set_size()[2])
        print '-'*width
        
    
    def init_model(self):
        print 'initialize model...'
        self.U = np.random.rand(self.training_size()[0], self.k)/3 # latent user matrix
        self.I = np.random.rand(self.training_size()[1], self.k)/3  # latent item matrix
        self.loss, self.last_loss = 0, 0
        
    def build_model_level(self):
        self.positive_set_5 = defaultdict(dict)
        self.positive_set_4 = defaultdict(dict)
        self.positive_set_3 = defaultdict(dict)
        self.positive_set_2 = defaultdict(dict)
        self.positive_set_1 = defaultdict(dict)
        self.positive_set = defaultdict(dict)
        
        for user in self.user:
            for item in self.training_set_u[user]:
                self.positive_set[user][item] = self.training_set_u[user][item]
                if self.training_set_u[user][item] == 5.0:
                    self.positive_set_5[user][item] = 5.0
                elif self.training_set_u[user][item] == 4.0:
                    self.positive_set_4[user][item] = 4.0
                elif self.training_set_u[user][item] == 3.0:
                    self.positive_set_3[user][item] = 3.0
                elif self.training_set_u[user][item] == 2.0:
                    self.positive_set_2[user][item] = 2.0
                else:
                    self.positive_set_1[user][item] = 1.0
        
        print 'training...'
        iteration = 0
        item_list = list(self.item.keys())
        while iteration < self.max_iter:
            if iteration%10==0:
                self.evaluate(iteration=iteration)
            self.loss = 0
            for user in self.positive_set:
                u = self.user[user]
                positive_item_list = self.positive_set[user]
                for item in self.positive_set[user]:
                    i = self.item[item]
                    for n in range(3):
                        if self.positive_set[user][item] == 5.0:
                            if user in self.positive_set_4:
                                item_j = choice(list(self.positive_set_4[user]))
                            elif user in self.positive_set_3:
                                item_j = choice(list(self.positive_set_3[user]))
                            elif user in self.positive_set_2:
                                item_j = choice(list(self.positive_set_2[user]))
                            elif user in self.positive_set_1:
                                item_j = choice(list(self.positive_set_1[user]))
                            else:
                                for m in range(3):
                                    item_j = choice(item_list)
                                    while item_j in positive_item_list:
                                        item_j = choice(item_list)
                                    j = self.item[item_j]
                                    self.optimization(u, i, j)
                        elif self.positive_set[user][item] == 4.0:
                            if user in self.positive_set_3:
                                item_j = choice(list(self.positive_set_3[user]))
                            elif user in self.positive_set_2:
                                item_j = choice(list(self.positive_set_2[user]))
                            elif user in self.positive_set_1:
                                item_j = choice(list(self.positive_set_1[user]))
                            else:
                                for m in range(3):
                                    item_j = choice(item_list)
                                    while item_j in positive_item_list:
                                        item_j = choice(item_list)
                                    j = self.item[item_j]
                                    self.optimization(u, i, j)
                        elif self.positive_set[user][item] == 3.0:
                            if user in self.positive_set_2:
                                item_j = choice(list(self.positive_set_2[user]))
                            elif user in self.positive_set_1:
                                item_j = choice(list(self.positive_set_1[user]))
                            else:
                                for m in range(3):
                                    item_j = choice(item_list)
                                    while item_j in positive_item_list:
                                        item_j = choice(item_list)
                                    j = self.item[item_j]
                                    self.optimization(u, i, j)
                        elif self.positive_set[user][item] == 2.0:
                            if user in self.positive_set_1.keys():
                                item_j = choice(list(self.positive_set_1[user]))
                            else:
                                for m in range(3):
                                    item_j = choice(item_list)
                                    while item_j in positive_item_list:
                                        item_j = choice(item_list)
                                    j = self.item[item_j]
                                    self.optimization(u, i, j)
                        else:
                            for m in range(3):
                                item_j = choice(item_list)
                                while item_j in positive_item_list:
                                    item_j = choice(item_list)
                                j = self.item[item_j]
                                self.optimization(u, i, j)
                        j = self.item[item_j]
                        self.optimization(u, i, j)
            self.loss += self.reg_u * (self.U * self.U).sum() + self.reg_i * (self.I * self.I).sum()
            self.U = preprocessing.normalize(self.U, norm='l2')
            self.I = preprocessing.normalize(self.I, norm='l2')
            iteration += 1
            if self.isConverged(iteration):
                break
        print 'training completed!'
        self.evaluate(iteration=iteration)
        np.save('./training_results/latentUserMatrix.npy', self.U)
        np.save('./training_results/latentItemMatrix.npy', self.I)
                
    def training_size(self):
        return [len(self.user), len(self.item), len(self.training_data)]
    
    def sigmoid(self, val):
        if val < 0:
            return 1-1/(1 + exp(val))
        return 1/(1+exp(-val))
    
    def optimization(self,u,i,j):
        alpha = self.U[u].dot(self.I[i]) - self.U[u].dot(self.I[j])
        if alpha < 0.0:
            s = self.sigmoid(alpha) 
            self.U[u] += self.learning_rate * (1 - s) * (self.I[i] - self.I[j]) - self.learning_rate * self.reg_u * self.U[u]
            self.I[i] += self.learning_rate * (1 - s) * self.U[u] - self.learning_rate * self.reg_i * self.I[i]
            self.I[j] -= self.learning_rate * (1 - s) * self.U[u] - self.learning_rate * self.reg_i * self.I[j]
            
            s_ = self.sigmoid(self.U[u].dot(self.I[i]) - self.U[u].dot(self.I[j])) 
            if s_ != 0:
                self.loss += -log(s_)
                       
    def update_learning_rate(self,iteration):
        if iteration > 1:
            if abs(self.last_loss) > abs(self.loss):
                self.learning_rate *= 1.05
            else:
                self.learning_rate *= 0.5

            if self.max_learning_rate > 0 and self.learning_rate > self.max_learning_rate:
                self.learning_rate = self.max_learning_rate
                
    def isConverged(self,iteration):
        from math import isnan
        if isnan(self.loss):
            print 'Loss = NaN or Infinity: current settings does not fit the recommender! Change the settings and try again!'
            exit(-1)
        delta_loss = (self.last_loss-self.loss)
        print 'iteration %-3d: loss = %-15.4f delta_loss = %-15.5f learning_Rate = %-15.5f' %(iteration, self.loss, delta_loss, self.learning_rate)
        # check if converged
        cond = abs(delta_loss) < 1e-3
        converged = cond
        if not converged:
            self.update_learning_rate(iteration)
        self.last_loss = self.loss
        return converged
    
    #----------------------------evaluate--------------------------------
    
    def dcg_at_k(self, r, k):
        r = np.asfarray(r)[:k]
        if r.size:
            return np.sum(np.subtract(np.power(2, r), 1) / np.log2(np.arange(2, r.size + 2)))
        return 0.


    def ndcg_at_k(self, r, k):
        idcg = self.dcg_at_k(sorted(r, reverse=True), k)
        if not idcg:
            return 0.
        return float(self.dcg_at_k(r, k)) / idcg     

    def precision_at_k(self, r, k):
        assert k >= 1
        r = np.asarray(r)[:k] != 0
        if r.size != k:
            raise ValueError('Relevance score length < k')
        return np.mean(r)

    def average_precision(self, r):
        r = np.asarray(r) != 0
        out = [self.precision_at_k(r, k + 1) for k in range(r.size) if r[k]]
        if not out:
            return 0.
        return np.mean(out)

    def mean_average_precision(self, rs):
        return np.mean([self.average_precision(r) for r in rs])

    def roc_auc(self, r, pre_score):
        r = [1 if i>0 else 0 for i in r]
        if len(set(r)) == 0 or len(set(r)) == 1:
            return 1
        return roc_auc_score(r, pre_score)
    
    def load_test_data(self):
        with open(self.test_path, 'r') as f:
            return [i.strip().split(' ') for i in f.readlines()]
        
    def test_data_pre_processing(self):
        self.test_set_u = defaultdict(dict)
        self.test_data = self.load_test_data()
        self.test_item_list = []
        for entry in self.test_data:
            user_name, item_name, rating = entry
            if item_name not in self.test_item_list:
                self.test_item_list.append(item_name)
            self.test_set_u[user_name][item_name] = float(self.grading(rating))
        self.test_item_set = set(self.test_item_list)
            
    def test_size(self):
        return len(self.test_set_u), len(self.test_item_list), len(self.test_data)

    def grading(self, rating):
        if float(rating) > 30:
            return 5.0
        elif float(rating)>15 and float(rating)<=30:
            return 4.0
        elif float(rating)>7 and float(rating)<=15:
            return 3.0
        elif float(rating)>3 and float(rating)<=7:
            return 2.0
        else:
            return 1.0

    def get_pre_item(self, u, flag=True):
        score_matrix = self.U.dot(self.I.T)
        score = score_matrix[self.user[str(u)]]
        item_sort = (-score).argsort()
        if flag:
            predict_list = [self.id2item[i] for i in item_sort]
            pre_score = [score[i] for i in item_sort]
        else:
            predict_list = [self.id2item[i] for i in item_sort if self.id2item[i] in self.test_item_set]
            pre_score = [score[i] for i in item_sort if self.id2item[i] in self.test_item_set]
        return predict_list, pre_score

    def calculate_ndcg(self, u, flag=True):
        if flag:
            pre_list, pre_score = self.get_pre_item(u)
            r = [self.training_set_u[str(u)][str(j)] if j in self.training_set_u[str(u)] else 0 for j in pre_list]
            l = len(self.training_set_u[str(u)])
        else:
            pre_list, pre_score = self.get_pre_item(u, False)
            r = [self.test_set_u[str(u)][str(j)] if j in self.test_set_u[str(u)] else 0 for j in pre_list]
            l = len(self.test_set_u[str(u)])
        return self.ndcg_at_k(r, len(r)), float(l), r, pre_score

    def evaluate(self, flag=True, iteration=None):
        ndcg = []
        n_item = []
        m_a_p = []
        auc = []
        if flag:
            set_u = self.training_set_u
        else:
            set_u = self.test_set_u
        for u in list(set_u.keys()):
            if str(u) in self.user.keys():
                ndcg_at_k, len_item, r, pre_score = self.calculate_ndcg(u, flag)
                ndcg.append(ndcg_at_k)
                n_item.append(len_item)
                m_a_p.append(self.average_precision(r))
                auc.append(self.roc_auc(r, pre_score))
        ndcg = np.array(ndcg)
        m_a_p = np.array(m_a_p)
        auc = np.array(auc)
        weight = np.array(n_item)/sum(n_item)
        weighted_ndcg = np.dot(ndcg,weight.T)
        weighted_map = np.dot(m_a_p,weight.T)
        weighted_auc = np.dot(auc,weight.T)
        if flag:
            print '{0:{fill}{align}60}'.format('iteration %-3d training set evaluation results'%iteration, fill='-', align='^')
        else:
            print '{0:{fill}{align}60}'.format('test set evaluation results', fill='-', align='^')
        print 'NDCG = %-15.5f MAP = %-15.5f AUC = %-15.5f' % (weighted_ndcg, weighted_map, weighted_auc)
        print '-'*60
    
    def print_test_evaluate(self):
        print 'evaluate test set...'
        self.test_data_pre_processing()
        self.evaluate(False)   
    
    #----------------------------------------------------------------
        
    def execute(self):
        self.data_pre_processing()
        self.print_data_statistic()
        self.init_model()
        self.build_model_level()
        self.test_data_pre_processing()
        self.print_data_statistic(False)
        self.print_test_evaluate()
    

def main():
    _, training_path, test_path = sys.argv
    a = BPR(training_path, test_path)
    a.execute()

if __name__ == '__main__':
    main()
