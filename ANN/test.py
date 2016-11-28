import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix


def prepare_data(targets, predicts):
    target_match = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    predict_match = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

    for target_index, target in enumerate(targets):
        index_t = target.index(1)
        target_match[index_t] += 1
    for predict_index, predict in enumerate(predicts):
        m = max(predict)
        index_p = predict.index(m)
        predict_match[index_p] += 1

    print(target_match)
    print(predict_match)
    target_match = [1,23,34,50,1,0,23,54,34,2, 2]
    predict_match = [1,22,33,40,1,0,23,54,33,1, 3]
    return target_match, predict_match


def d_confusion_matrix(targets, predicted):
    y_tar, y_pred = prepare_data(targets, predicted)
    cm = confusion_matrix(y_tar, y_pred)
    plt.imshow(cm, interpolation='nearest')
    plt.title("Confusion matrix for IMDB ratings prediction")
    plt.colorbar()
    plt.xticks([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10], rotation=45)
    plt.yticks([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    plt.tight_layout()
    plt.ylabel("True")
    plt.xlabel("Predicted")
    plt.show()


target = [[0, 0, 0, 0, 0, 0, 0, 0, 0, 1], [0, 0, 0, 0, 0, 0, 0, 0, 0, 1]]
predict = [[0.005713738750424112, 0.00609160083540838, 0.006259006100390485, 0.006804233473251715,
             0.00701450122797203, 0.01068455950766015, 0.0502865402778061, 0.22202610071897963,
             0.2605204693883913, 0.3240050755523023], [0.005713738750424112, 0.00609160083540838, 0.006259006100390485, 0.006804233473251715,
             0.00701450122797203, 0.01068455950766015, 0.0502865402778061, 0.22202610071897963,
             0.2605204693883913, 0.3240050755523023]]
d_confusion_matrix(target, predict)
