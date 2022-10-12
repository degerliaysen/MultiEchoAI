#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 10 11:41:30 2019

@author: degerli
"""

from sklearn.metrics import confusion_matrix
from models import * 
import numpy as np
import os
import argparse
     
ap = argparse.ArgumentParser()
ap.add_argument('-gpu', '--gpu', default='0')
ap.add_argument('-view', '--view', default='multi')
args = vars(ap.parse_args())

os.environ["CUDA_VISIBLE_DEVICES"] = args['gpu']
if not os.path.exists(os.path.join(os.getcwd(),'output', 'matrices')): os.makedirs(os.path.join(os.getcwd(),'output', 'matrices'))

MODEL = ['SVM', 'DT', 'KNN', 'RF', 'CNN']
REFIT= ['AUC', 'Accuracy', 'Recall', 'F1-Score', 'Precision']

x_train = np.load(os.path.join(os.path.join(os.getcwd(),'DataSplits'), 'x_train_' + args['view'] + '.npy'))
x_test = np.load(os.path.join(os.path.join(os.getcwd(),'DataSplits'), 'x_test_' + args['view'] + '.npy'))
y_train = np.load(os.path.join(os.path.join(os.getcwd(),'DataSplits'), 'y_train_' + args['view'] + '.npy'))
y_test = np.load(os.path.join(os.path.join(os.getcwd(),'DataSplits'), 'y_test_' + args['view'] + '.npy'))

for f in range(0,5):    
    for i in range(len(MODEL)):           
        for j in range(len(REFIT)):            
            #Shuffle train data
            np.random.seed(seed=3)
            idx = np.random.permutation(len(x_train[f]))
            x_train[f], y_train[f] = x_train[f][idx], y_train[f][idx]
        
            if MODEL[i] == 'SVM':    
                best_parameters, best_model = SVM_train(x_train[f], y_train[f], REFIT[j])
                score = best_model.predict(x_test[f])
                CM = confusion_matrix(y_test[f], score)
                metrics = performance_metrics(CM)
                
                #Save the results
                text_file = open(os.path.join(os.path.join(os.getcwd(), 'output'), MODEL[i] + '_' + args['view'] + '.txt'), "a")
                text_file.write ("\n\n\n----------FOLD " + str(f) + "-------------\n")
                text_file.write("\n\n\nConfusion Matrix :" + str(CM) + "\n")
                text_file.write ("\nScoring:" + REFIT[j])
                text_file.write ("\nSensitivity:" + str(metrics[0]))
                text_file.write ("\nSpecificity:" + str(metrics[1]))
                text_file.write("\nPrecision:" + str(metrics[2]))
                text_file.write ("\nF1-Score:" + str(metrics[3]))
                text_file.write ("\nF2-Score:" + str(metrics[4]))   
                text_file.write("\nAccuracy:" + str(metrics[5]))
                text_file.write('\nBest paramters:' + str(best_parameters))
                text_file.close()
                    
                np.save(os.path.join(os.path.join(os.getcwd(), 'output', 'matrices'), args['view'] +'_score_' + MODEL[i] + '_' + REFIT[j] + '_fold' + str(f) + '.npy'), score)
                np.save(os.path.join(os.path.join(os.getcwd(), 'output', 'matrices'), args['view'] + '_y_test_' + MODEL[i] + '_' + REFIT[j] + '_fold' + str(f) + '.npy'), y_test[f])
                                
            elif MODEL[i] == 'KNN':
                best_parameters, best_model = KNN_train(x_train[f], y_train[f], REFIT[j])
                score = best_model.predict(x_test[f])
                
                CM = confusion_matrix(y_test[f], score)
                metrics = performance_metrics(CM)
                
                #Save the results
                text_file = open(os.path.join(os.path.join(os.getcwd(), 'output'), MODEL[i] + '_' + args['view'] + '.txt'), "a")
                text_file.write ("\n\n\n----------FOLD " + str(f) + "-------------\n")
                text_file.write("\n\n\nConfusion Matrix :" + str(CM) + "\n")
                text_file.write ("\nScoring:" + REFIT[j])
                text_file.write ("\nSensitivity:" + str(metrics[0]))
                text_file.write ("\nSpecificity:" + str(metrics[1]))
                text_file.write("\nPrecision:" + str(metrics[2]))
                text_file.write ("\nF1-Score:" + str(metrics[3]))
                text_file.write ("\nF2-Score:" + str(metrics[4]))   
                text_file.write("\nAccuracy:" + str(metrics[5]))
                text_file.write('\nBest paramters:' + str(best_parameters))
                text_file.close()
                
                np.save(os.path.join(os.path.join(os.getcwd(), 'output', 'matrices'), args['view'] +'_score_' + MODEL[i] + '_' + REFIT[j] + '_fold' + str(f) + '.npy'), score)
                np.save(os.path.join(os.path.join(os.getcwd(), 'output', 'matrices'), args['view'] + '_y_test_' + MODEL[i] + '_' + REFIT[j] + '_fold' + str(f) + '.npy'), y_test[f])
                                       
            elif MODEL[i] == 'DT':
                best_parameters, best_model = DT_train(x_train[f], y_train[f], REFIT[j])
                score = best_model.predict(x_test[f])
                
                CM = confusion_matrix(y_test[f], score)
                metrics = performance_metrics(CM)
                
                #Save the results
                text_file = open(os.path.join(os.path.join(os.getcwd(), 'output'), MODEL[i] + '_' + args['view'] + '.txt'), "a")
                text_file.write ("\n\n\n----------FOLD " + str(f) + "-------------\n")
                text_file.write("\n\n\nConfusion Matrix :" + str(CM) + "\n")
                text_file.write ("\nScoring:" + REFIT[j])
                text_file.write ("\nSensitivity:" + str(metrics[0]))
                text_file.write ("\nSpecificity:" + str(metrics[1]))
                text_file.write("\nPrecision:" + str(metrics[2]))
                text_file.write ("\nF1-Score:" + str(metrics[3]))
                text_file.write ("\nF2-Score:" + str(metrics[4]))   
                text_file.write("\nAccuracy:" + str(metrics[5]))
                text_file.write('\nBest paramters:' + str(best_parameters))
                text_file.close()
                
                np.save(os.path.join(os.path.join(os.getcwd(), 'output', 'matrices'), args['view'] +'_score_' + MODEL[i] + '_' + REFIT[j] + '_fold' + str(f) + '.npy'), score)
                np.save(os.path.join(os.path.join(os.getcwd(), 'output', 'matrices'), args['view'] + '_y_test_' + MODEL[i] + '_' + REFIT[j] + '_fold' + str(f) + '.npy'), y_test[f])
                                
            elif MODEL[i] == 'RF':
                best_parameters, best_model = RF_train(x_train[f], y_train[f], REFIT[j])
                score = best_model.predict(x_test[f])
                
                CM = confusion_matrix(y_test[f], score)
                metrics = performance_metrics(CM)
                
                #Save the results
                text_file = open(os.path.join(os.path.join(os.getcwd(), 'output'), MODEL[i] + '_' + args['view'] + '.txt'), "a")
                text_file.write ("\n\n\n----------FOLD " + str(f) + "-------------\n")
                text_file.write("\n\n\nConfusion Matrix :" + str(CM) + "\n")
                text_file.write ("\nScoring:" + REFIT[j])
                text_file.write ("\nSensitivity:" + str(metrics[0]))
                text_file.write ("\nSpecificity:" + str(metrics[1]))
                text_file.write("\nPrecision:" + str(metrics[2]))
                text_file.write ("\nF1-Score:" + str(metrics[3]))
                text_file.write ("\nF2-Score:" + str(metrics[4]))   
                text_file.write("\nAccuracy:" + str(metrics[5]))
                text_file.write('\nBest paramters:' + str(best_parameters))
                text_file.close()
                
                np.save(os.path.join(os.path.join(os.getcwd(), 'output', 'matrices'), args['view'] +'_score_' + MODEL[i] + '_' + REFIT[j] + '_fold' + str(f) + '.npy'), score)
                np.save(os.path.join(os.path.join(os.getcwd(), 'output', 'matrices'), args['view'] + '_y_test_' + MODEL[i] + '_' + REFIT[j] + '_fold' + str(f) + '.npy'), y_test[f])
                                
            elif MODEL[i] == 'CNN':      
                x_train = np.expand_dims(x_train, axis = -1)
                x_test = np.expand_dims(x_test, axis = -1)
                best_model, best_parameters = CNN_train(x_train[f], y_train[f], REFIT[j])    
                score = best_model.predict(x_test[f])
                
                CM = confusion_matrix(y_test[f], score)
                metrics = performance_metrics(CM)
                
                #Save the results
                text_file = open(os.path.join(os.path.join(os.getcwd(), 'output'), MODEL[i] + '_' + args['view'] + '.txt'), "a")
                text_file.write ("\n\n\n----------FOLD " + str(f) + "-------------\n")
                text_file.write("\n\n\nConfusion Matrix :" + str(CM) + "\n")
                text_file.write ("\nScoring:" + REFIT[j])
                text_file.write ("\nSensitivity:" + str(metrics[0]))
                text_file.write ("\nSpecificity:" + str(metrics[1]))
                text_file.write("\nPrecision:" + str(metrics[2]))
                text_file.write ("\nF1-Score:" + str(metrics[3]))
                text_file.write ("\nF2-Score:" + str(metrics[4]))   
                text_file.write("\nAccuracy:" + str(metrics[5]))
                text_file.write('\nBest paramters:' + str(best_parameters))
                text_file.close()
                
                np.save(os.path.join(os.path.join(os.getcwd(), 'output', 'matrices'), args['view'] +'_score_' + MODEL[i] + '_' + REFIT[j] + '_fold' + str(f) + '.npy'), score)
                np.save(os.path.join(os.path.join(os.getcwd(), 'output', 'matrices'), args['view'] + '_y_test_' + MODEL[i] + '_' + REFIT[j] + '_fold' + str(f) + '.npy'), y_test[f])
                                