from sklearn.metrics import accuracy_score

def cal_acc(labels, preds):
    return accuracy_score(labels, preds)