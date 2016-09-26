import tensorflow as tf
import numpy as np
import cv2
import math
import random
import os
import sys
import matplotlib.pyplot as plt
from SvhnNet import *

debug = False

whiteWash = True
dataRatio = [0.4,0.4,0.2] # train, test, validate

allLabelsTrain = []
allImagesTrain = []
allLabelsTest = []
allImagesTest = []
allLabelsValidate = []
allImagesValidate = []
validateFiles = []

imageSize = 32
batchSize = 128
trainingIters = 1000000 # in terms of sample size
onehotLabels = None
displayStep = 1 # how often to print details
step = 0

#imagePath = "/Users/jiyan/Desktop/class/"
logPath = "svhnlogs"
modelPath = "svhnModel/"
imagePath = "/home/deeplearningdev/class/"

def setupSummaries():
  with tf.variable_scope('monitor') as scope:
    loss = tf.Variable(0.0)
    tf.scalar_summary("Loss", loss)
    trainAcc = tf.Variable(0.0)
    tf.scalar_summary("Train Accuracy", trainAcc)
    testAcc = tf.Variable(0.0)
    tf.scalar_summary("Test Accuracy", testAcc)
    summaryVars = [loss, trainAcc, testAcc]
    summaryPlaceholders = [tf.placeholder("float") for i in range(len(summaryVars))]
    updateOps = [summaryVars[i].assign(summaryPlaceholders[i]) for i in range(len(summaryVars))]
    return summaryPlaceholders, updateOps

def load():
  global onehotLabels
  global allImagesTrain
  global allLabelsTrain
  global allImagesTest
  global allLabelsTest
  global allImagesValidate
  global allLabelsValidate
  allImages = []
  allLabels = []
  with open("digit.out") as f:
    content = f.readlines()
    for line in content:
      parts = line.split(",")
      fileName = imagePath + parts[0]
      print fileName
      img = cv2.resize(cv2.imread(fileName, cv2.IMREAD_UNCHANGED), (imageSize, imageSize))

      # white wash image
      if whiteWash:
        imgMean = np.mean(img)
        #std = np.sqrt(np.sum(np.square(img - imgMean)) / (32 * 32))
        img = img.astype(np.float32)
        img -= imgMean
        #img /= std

      allImages.append(img)
      allLabels.append(parts[1])
      if debug and len(allLabels) > 1000:
        break

  onehotLabels = np.zeros((len(allLabels), 10))
  onehotLabels[np.arange(len(allLabels)), allLabels] = 1

  trainIdx = int(len(allLabels) * dataRatio[0])
  testIdx = int(trainIdx + len(allLabels) * dataRatio[1])
  allImagesTrain = allImages[:trainIdx]
  allLabelsTrain = onehotLabels[:trainIdx]
  allImagesTest = allImages[trainIdx:testIdx]
  allLabelsTest = onehotLabels[trainIdx:testIdx]
  allImagesValidate = allImages[testIdx:]
  allLabelsValidate = onehotLabels[testIdx:]

def next(size, imgs, labels):
  indices = random.sample(range(len(imgs)), size)
  batchImages = np.array(imgs)[indices]
  batchLabels = np.array(labels)[indices]
  return batchImages,batchLabels

def train():
  global allImagesValidate
  global allLabelsValidate

  x = tf.placeholder(tf.float32, [None, imageSize, imageSize, 3])
  y = tf.placeholder(tf.float32, [None, 10])

  load()
  cnn = SvhnNet(x, y)
  monitorPh, monitorOps = setupSummaries()
  init = tf.initialize_all_variables()
  saver = tf.train.Saver()
  summaryOps = tf.merge_all_summaries()
  with tf.Session() as sess:
      sess.run(init)
      writer = tf.train.SummaryWriter(logPath, sess.graph)
      if not os.path.exists(modelPath):
        os.makedirs(modelPath)
      checkpoint = tf.train.get_checkpoint_state(modelPath)
      if checkpoint and checkpoint.model_checkpoint_path:
          saver.restore(sess, checkpoint.model_checkpoint_path)
          print "successfully loaded checkpoint"

      step = 1
      while step * batchSize < trainingIters:
          trainImages, trainLabels = next(batchSize, allImagesTrain, allLabelsTrain)
          sess.run(cnn.optimizer, feed_dict={x: trainImages, y: trainLabels})
          if step % displayStep == 0:
              # Calculate training loss and accuracy
              loss, trainAcc = sess.run([cnn.cost,cnn.accuracy], feed_dict={x: trainImages,
                                                                            y: trainLabels})
              # calculate test accuracy
              testImages, testLabels = next(batchSize, allImagesTest, allLabelsTest)
              testAcc = sess.run(cnn.accuracy, feed_dict={x: testImages, y: testLabels})
              sess.run([monitorOps[0], monitorOps[1], monitorOps[2]], feed_dict={monitorPh[0]:float(loss),
                                                                                 monitorPh[1]:trainAcc,
                                                                                 monitorPh[2]:testAcc})

              print("Iter " + str(step*batchSize) + ", Minibatch Loss= " + \
                    "{:.6f}".format(loss) + ", Training Accuracy= " + \
                    "{:.5f}".format(trainAcc) + ", Test Accuracy= " + "{:.5f}".format(testAcc))
              
              savePath = saver.save(sess, modelPath + "svhn.ckpt")
              print("Model saved in file: %s" % savePath)
              summaryStr = sess.run(summaryOps, feed_dict={x: trainImages,
                                                           y: trainLabels})
              writer.add_summary(summaryStr, step)
              writer.add_summary(summaryStr, step)
          step += 1

      print("Optimization Finished!")
      validateImages, validateLabels = next(len(allImagesValidate), allImagesValidate, allLabelsValidate)
      # Calculate validate loss and accuracy
      validateAcc, correctPred = sess.run([cnn.accuracy, cnn.correctPred], feed_dict={x: validateImages,
                                                                                      y: validateLabels})
      print("Validation Accuracy= " + "{:.5f}".format(validateAcc))
      correctIndices = [i for i, x in enumerate(correctPred) if x]
      incorrectIndices = [i for i, x in enumerate(correctPred) if not x]
      allImagesValidate = np.array(allImagesValidate)
      correct = allImagesValidate[correctIndices]
      incorrect = allImagesValidate[incorrectIndices]

      if not os.path.exists("correct"):
        os.makedirs("correct")
      if not os.path.exists("incorrect"):
        os.makedirs("incorrect")

      for i, x in enumerate(correct):
        cv2.imwrite("correct/img" + str(i) + ".jpg", x)

      for i, x in enumerate(incorrect):
        cv2.imwrite("incorrect/img" + str(i) + ".jpg", x)


if __name__ == '__main__':
  if len(sys.argv) == 1:
    train()
  else:
    # output pred for passed in image
    fileName = str(sys.argv[1])
    img = cv2.resize(cv2.imread(fileName, cv2.IMREAD_UNCHANGED), (imageSize, imageSize))

    # white wash image
    if whiteWash:
      imgMean = np.mean(img)
      #std = np.sqrt(np.sum(np.square(img - imgMean)) / (32 * 32))
      img = img.astype(np.float32)
      img -= imgMean
      #img /= std

    x = tf.placeholder(tf.float32, [None, imageSize, imageSize, 3])
    y = tf.placeholder(tf.float32, [None, 10])
    cnn = SvhnNet(x, y)
    saver = tf.train.Saver()
    with tf.Session() as sess:
        checkpoint = tf.train.get_checkpoint_state(modelPath)
        if checkpoint and checkpoint.model_checkpoint_path:
            saver.restore(sess, checkpoint.model_checkpoint_path)
            print "successfully loaded model"
            dummyLabel = [0] * 10
            logits = sess.run(cnn.logits, feed_dict={x: [img],
                                                       y: [dummyLabel]})
            print logits[0]
            objects = ('0', '1', '2', '3', '4', '5', '6', '7', '8', '9')
            yPos = np.arange(len(objects))
             
            plt.bar(yPos, logits[0], align='center', alpha=0.5)
            plt.xticks(yPos, objects)
            plt.ylabel('Unnormalized Probability (logits)')
            plt.title('Digit')
             
            plt.show()



