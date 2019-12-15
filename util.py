# Library imports
import cv2
import itertools
import numpy as np
from imutils import paths
from PIL import Image, ImageOps
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, classification_report,confusion_matrix

# Global Variables
desired_size = 64
base_path = "malaria/cell_images"
image_paths = list(paths.list_images(base_path))


def resize(w, h):

    for im_pth in image_paths:
        src = cv2.imread(im_pth, cv2.IMREAD_UNCHANGED)

        width = w
        height = h

        # dsize
        dsize = (width, height)

        # resize image
        output = cv2.resize(src, dsize)

        cv2.imwrite(im_pth,output) 

    print("All done!~ ^-^")

def plot_ROC(Y_test, y_pred, num_classes):
    #compute the ROC-AUC values
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(num_classes):
        fpr[i], tpr[i], _ = roc_curve(Y_test[:, i], y_pred[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], _ = roc_curve(Y_test.ravel(), y_pred.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

    #Plot ROC curve for the positive class
    plt.figure(figsize=(20,10), dpi=300)
    lw = 1 #true class label
    plt.plot(fpr[1], tpr[1], color='red',
             lw=lw, label='ROC curve (area = %0.4f)' % roc_auc[1])
    plt.plot([0, 1], [0, 1], color='black', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristics')
    plt.legend(loc="lower right")
    plt.show()


def calc_confusion(Y_test, y_pred):
    #plot the confusion matrix
    classes = ['class 0(abnormal)', 'class 1(normal)']
    print(classification_report(Y_test,y_pred,target_names=classes))
    print(confusion_matrix(Y_test,y_pred))
    cm = (confusion_matrix(Y_test,y_pred))
    np.set_printoptions(precision=4)
    plt.figure(figsize=(20,10), dpi=300)
    return cm, classes

def plot_confusion(Y_test, y_pred):
    
    # Calculate matrix
    cm, classes = calc_confusion(Y_test, y_pred)

    # Plot non-normalized confusion matrix
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()

def plot_loss_acc(num_epoch, hist):
    # visualizing losses and accuracy
    train_loss=hist.history['loss']
    val_loss=hist.history['val_loss']
    train_acc=hist.history['acc']
    val_acc=hist.history['val_acc']
    xc=range(num_epoch)

    plt.figure(1,figsize=(20,10), dpi=300)
    plt.plot(xc,train_loss)
    plt.plot(xc,val_loss)
    plt.xlabel('num of Epochs')
    plt.ylabel('loss')
    plt.title('train_loss vs val_loss')
    plt.grid(True)
    plt.legend(['train','val'])
    plt.style.use(['classic'])

    plt.figure(2,figsize=(20,10), dpi=300)
    plt.plot(xc,train_acc)
    plt.plot(xc,val_acc)
    plt.xlabel('num of Epochs')
    plt.ylabel('accuracy')
    plt.title('train_acc vs val_acc')
    plt.grid(True)
    plt.legend(['train','val'])
    plt.style.use(['classic'])

if __name__ == "__main__":
    resize(150, 150)