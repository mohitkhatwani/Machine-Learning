'''
Name: Mohit Khatwani
Student ID: AJ75499
HomeWork 3: SVM
'''

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
import matplotlib.pyplot as plot
from random import random,shuffle,sample
from operator import itemgetter


def extract_data_shuffled(train_images,train_labels,test_images,test_labels,count,num1,num2):
    image_train,label_train,image_test,label_test = [],[],[],[]
    #1 denoted by num1 and -1 denoted by num2
    #training data
    num1_count,num2_count = 0,0
    for i in range(len(train_labels)):
        #extract only num1's
        if num1_count < count and train_labels[i][num1] == 1:
            image_train.append(train_images[i])
            label_train.append([1])
            num1_count += 1
        #extract only num2's
        if num2_count < count and train_labels[i][num2] == 1:
            image_train.append(train_images[i])
            label_train.append([-1])
            num2_count += 1

    #shuffle training data
    train = list(zip(image_train,label_train))
    shuffle(train)
    image_train, label_train = zip(*train)

    #Testing data
    num1_count,num2_count = 0,0
    for i in range(len(test_labels)):
        if num1_count < count and test_labels[i][num1] == 1:
            image_test.append(test_images[i])
            label_test.append([1])
            num1_count += 1
        #extract only num2's
        if num2_count < count and test_labels[i][num2] == 1:
            image_test.append(test_images[i])
            label_test.append([-1])
            num2_count += 1
    #shuffle testing data
    test = list(zip(image_test,label_test))
    shuffle(test)
    image_test, label_test = zip(*test)

    return image_train,label_train,image_test,label_test



def extract_data(train_images,train_labels,test_images,test_labels,num1,num2):
    image_train,label_train,image_test,label_test = [],[],[],[]
    #1 denoted by num1 and -1 denoted by num2
    #training data
    num_count = 0
    for i in range(len(train_labels)):
        #extract only num1's
        if num_count < 500 and train_labels[i][num1] == 1:
            image_train.append(train_images[i])
            label_train.append([1])
            num_count += 1
    num_count = 0
    for i in range(len(train_labels)):
        #extract only num2's
        if num_count < 500 and train_labels[i][num2] == 1:
            image_train.append(train_images[i])
            label_train.append([-1])
            num_count += 1
    num_count = 0
    #Testing data
    for i in range(len(test_labels)):
        if num_count < 500 and test_labels[i][num1] == 1:
            image_test.append(test_images[i])
            label_test.append([1])
            num_count += 1
    num_count = 0
    for i in range(len(test_labels)):
        #extract only num2's
        if num_count < 500 and test_labels[i][num2] == 1:
            image_test.append(test_images[i])
            label_test.append([-1])
            num_count += 1
    return image_train,label_train,image_test,label_test


def extract_multiclass_data(train_images,train_labels,test_images,test_labels,num1):
    image_train,label_train,image_test,label_test = [],[],[],[]
    num1_count,num2_count = 0,0
    for i in range(len(train_labels)):
        #extract only num1's
        if num1_count < 1000 and train_labels[i][num1] == 1:
            image_train.append(train_images[i])
            label_train.append([1])
            num1_count += 1
        #extract only num2's
        if num2_count < 1000 and train_labels[i][num1] != 1:
            image_train.append(train_images[i])
            label_train.append([-1])
            num2_count += 1
    #shuffle training data
    train = list(zip(image_train,label_train))
    shuffle(train)
    image_train, label_train = zip(*train)

    #Testing data
    num1_count,num2_count = 0,0
    for i in range(len(test_labels)):
        if num1_count < 500 and test_labels[i][num1] == 1:
            image_test.append(test_images[i])
            label_test.append([1])
            num1_count += 1
        if num2_count < 500 and test_labels[i][num1] != 1:
            image_test.append(test_images[i])
            label_test.append([-1])
            num2_count += 1
    #shuffle testing data
    test = list(zip(image_test,label_test))
    shuffle(test)
    image_test, label_test = zip(*test)

    return image_train,label_train,image_test,label_test

def prediction(inputs,weights):
    activation = 0
    for input,weight in zip(inputs,weights):
        activation += input*weight
    if activation > 0:
        return 1
    else:
        return -1

def accuracy(testdata,actual_label,weights):
    correct = 0
    for i in range(len(testdata)):
        pred = prediction(testdata[i],weights)
        if pred == actual_label[i][0]: correct += 1
    return correct/float(len(testdata))*100

def accuracy_iteration(train_data,train_label,test_data,test_label,iteration,num1,num2):

    x = [i*len(train_data) for i in range(iteration)]
    y = []
    for i in range(0,iteration):
        y.append(svm(train_data,train_label,test_data,test_label,i)[0])
    plot.ylim(0,100)
    plot.plot(x,y)
    print(y)
    plot_name = "svm_accuracy_iteration%s_%s%s.png" %(iteration,num1,num2)
    plot.savefig(plot_name)


def svm(train_images,train_labels,test_images,test_labels,epochs, num1 = None, num2 = None):
    session = tf.Session()
    #weight = tf.placeholder(dtype=tf.float32,shape=[28*28])
    weight = np.zeros(28*28)
    for epoch in range(1,epochs):
        for i in range(1,len(train_images)):
            learningRate = 1/i
            C = 0.00001
            if train_labels[i][0]*np.dot(train_images[i], weight) < 1:
                weight += learningRate*((train_images[i]*train_labels[i]) + (-2 * C * weight))
            else:
                weight += learningRate * (-2 * C * weight)
    if num1 != None and num2 != None:
        correct_num1, correct_num2, total_num1, total_num2 = get_count(test_images, test_labels, weight, num1, num2)
        return correct_num1, correct_num2, total_num1, total_num2
    return accuracy(test_images,test_labels,weight), weight

def get_count(test_images, test_labels, weight, num1, num2):
    #count for correct predictions in num1
    total_num1, total_num2, correct_num1, correct_num2 = 0 ,0 , 0 , 0
    for i in range(len(test_labels)):
        if test_labels[i] == [1]:
            total_num1 += 1
            if prediction(test_images[i], weight) == 1:
                correct_num1 += 1
        else:
            total_num2 += 1
            if prediction(test_images[i], weight) == -1:
                correct_num2 += 1
    return correct_num1, correct_num2, total_num1, total_num2

def sorted_data_visualization(train_data,train_label,test_data,test_label,iteration,num1,num2):
    x = [i*len(train_data) for i in range(iteration)]
    y = []
    for i in range(0,iteration):
        y.append(svm(train_data,train_label,test_data,test_label,i)[0])
    plot.ylim(0,100)
    plot.plot(x,y)
    plot_name = "svm_sorted_accuracy_iteration%s%s.png" %(num1,num2)
    plot.savefig(plot_name)

def multiclass_data_visualization(train_data,train_label,test_data,test_label,iteration,num1):
    x = [i*len(train_data) for i in range(iteration)]
    y = []
    for i in range(0,iteration):
        y.append(svm(train_data,train_label,test_data,test_label,i)[0])
    plot.ylim(0,100)
    plot.plot(x,y)
    plot_name = "svm_multiclass_accuracy_iteration%s.png" %(num1)
    plot.savefig(plot_name)

def getConfusionMatrix(mnist):
    confusion = [[0 for j in range(10)] for i in range(10)]
    for i in range(0,10):
        for j in range(0,10):
            if i != j:
                train_images,train_labels,test_images,test_labels = extract_data_shuffled(mnist.train.images,mnist.train.labels,mnist.test.images,mnist.test.labels,500,i,j)
                train_images = np.asarray(train_images)
                test_images = np.asarray(test_images)
                correct_num1, correct_num2, total_num1, total_num2 = svm(train_images, train_labels, test_images, test_labels, 10, i, j)
                confusion[i][i] += correct_num1
                confusion[i][j] += total_num1 - correct_num1
    for con in confusion:
        temp = sum(con)
        for i in range(len(con)):
            con[i] = round(con[i] / temp,3)
    return confusion

def getIncorrectImages(test_images,test_labels, weight):
    incorrect_images = []
    for i in range(len(test_images)):
        if test_labels[i] == [1]:
            if prediction(test_images[i], weight) == -1:
                incorrect_images.append(test_images[i])
    return incorrect_images


def getMaxIncorrectImage(incorrect_images, weight):
    scores = []
    for image in incorrect_images:
        score = 0
        for i in range(len(image)):
            score += abs(weight[i] - image[i])
        scores.append(score)
    ind = scores.index(max(scores))
    return incorrect_images[ind]

def visualize_weights(weights,num1,num2):
    weight_pos, weight_neg = [], []

    for i in range(len(weights)):
        if weights[i] >= 0:
            weight_pos.append(weights[i])
        else:
            weight_pos.append(0)

    for i in range(len(weights)):
        if weights[i] <= 0:
            weight_neg.append(abs(weights[i]))
        else:
            weight_neg.append(0)
    pos_weight,neg_weight = [],[]

    for i in range(0,len(weight_pos),28):
        pos_weight.append(weight_pos[i:i+28])
    for i in range(0,len(weight_neg),28):
        neg_weight.append(weight_neg[i:i+28])

    plot.imshow(pos_weight,'gray_r')
    plt_name = "svm_weight_%s.png"%(num1)
    plot.savefig(plt_name)
    plot.imshow(neg_weight,'gray_r')
    plt_name = "svm_weight_%s.png"%(num2)
    plot.savefig(plt_name)

def getPrecision(confusion):
    relevant,retrieved = 0,0
    for i in range(len(confusion)):
        for j in range(len(confusion[i])):
            retrieved += confusion[i][j]
            if i == j:
                relevant += confusion[i][j]
    return round(relevant/retrieved,5)

def main():

    print("Importing MNIST Dataset")
    mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
    train_images,train_labels,test_images,test_labels = extract_data_shuffled(mnist.train.images,mnist.train.labels,mnist.test.images,mnist.test.labels,500,1,6)
    train_images = np.asarray(train_images)
    test_images = np.asarray(test_images)


    print("(a). Accuracy plot with number of iterations for classifying digits 1 and 6\n(b). Accuracy plot for classifying digits 1 and 6 with sorted training data\n(c). 1-vs-all multi-class SVM\n(d). Confusion Matrix for all 10 digits in MNIST dataset\n(e). Top mistakes along with ground truth labels and predicted labels\n(f). Weight Vector visualization for 1 and 6")

    choice = input("***Enter your choice***")
    if choice == 'a':
        accuracy_iteration(train_images,train_labels,test_images,test_labels,10,1,6)
        print("Accuracy-iteration plot for digits 1 and 6 ploted!")
    elif choice == 'b':
        train_images_sorted,train_labels_sorted,test_images_sorted,test_labels_sorted = extract_data(mnist.train.images,mnist.train.labels,mnist.test.images,mnist.test.labels,1,6)
        train_images = np.asarray(train_images)
        test_images = np.asarray(test_images)
        sorted_data_visualization(train_images_sorted,train_labels_sorted,test_images_sorted,test_labels_sorted,10,1,6)
        print("Accuracy-iteration plot for digits 1 and 6 with sorted training data!")
    elif choice == 'c':
        train_images_sorted,train_labels_sorted,test_images_sorted,test_labels_sorted = extract_multiclass_data(mnist.train.images,mnist.train.labels,mnist.test.images,mnist.test.labels,1)
        train_images = np.asarray(train_images)
        test_images = np.asarray(test_images)
        multiclass_data_visualization(train_images_sorted,train_labels_sorted,test_images_sorted,test_labels_sorted,10,1)
        print("Accuracy-iteration plot for 1 vs. all mulit-class SVM!")
    elif choice == 'd':
        confusion = getConfusionMatrix(mnist)
        for con in confusion:
            print(con)
        print("Average Precision is:",getPrecision(confusion))
    elif choice == 'e':
        confusion = getConfusionMatrix(mnist)
        print(confusion)
        maximum_incorrect = []
        for i in range(0,10):
            conf = confusion[i][0:i] + confusion[i][i+1:]
            maxi = max(conf)
            maximum_incorrect.append(conf.index(maxi))
        print(maximum_incorrect)
        for i in range(len(maximum_incorrect)):
            train_images,train_labels,test_images,test_labels = extract_data_shuffled(mnist.train.images,mnist.train.labels,mnist.test.images,mnist.test.labels,500,i,maximum_incorrect[i])
            weight = svm(train_images,train_labels,test_images,test_labels,10)[1]
            incorrect_images = getIncorrectImages(test_images,test_labels, weight)
            Incorrect_image = getMaxIncorrectImage(incorrect_images, weight)
            temp = []
            for j in range(0,len(Incorrect_image),28):
                temp.append(Incorrect_image[j:j+28])
            pl = plot.subplot(2,5,i+1)
            plot.axis('off')
            s = str(i) + ',' + str(maximum_incorrect[i])
            pl.set_title(s)
            plot.imshow(temp,'gray_r')
        plot.savefig('Worst_mistakes.png')
    elif choice == 'f':
        weight = svm(train_images,train_labels,test_images,test_labels,10)[1]
        visualize_weights(weight,1,6)

if __name__ == "__main__":
    main()
