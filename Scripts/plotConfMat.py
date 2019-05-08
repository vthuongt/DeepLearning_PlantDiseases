# -*- coding: utf-8 -*-
"""
Created on Wed Jan  9 17:00:39 2019

@author: TVA8FE
"""
import numpy as np
import matplotlib.pyplot as plt
import pickle
from itertools import product

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues, printScores=False):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        #TODO fix error by 0 division if not correctly classified
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    if printScores:
        for i, j in product(range(cm.shape[0]), range(cm.shape[1])):
            plt.text(j, i, format(cm[i, j], fmt),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black",
                     fontsize=5)

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()


###############################################


with open(r"alexnet_retrain_deep.pickle", "rb") as infile:
    deep = pickle.load(infile)

with open(r"alexnet_train.pickle", "rb") as infile:
    train = pickle.load(infile)

with open(r"alexnet_retrain_shallow.pickle", "rb") as infile:
    shallow = pickle.load(infile)


dset_classes = []
name = 'alexnet'
plt.figure()
plot_confusion_matrix(deep['confusion_matrix'], classes=dset_classes, normalize=False,
                      title='confusion matrix of %s retrained deep' % name, printScores=True)
plt.draw()


plt.figure()
plot_confusion_matrix(train['confusion_matrix'], classes=dset_classes, normalize=False,
                      title='confusion matrix of %s train' % name, printScores=True)
plt.draw()


plt.figure()
plot_confusion_matrix(shallow['confusion_matrix'], classes=dset_classes, normalize=False,
                      title='confusion matrix of %s retrained shallow' % name, printScores=True)
plt.draw()


plt.show()






