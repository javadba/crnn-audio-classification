import torch
import numpy as np


def accuracyOrig(output, target, percent=0.1):
    with torch.no_grad():

        assert output.shape[0] == len(target)
        preds = torch.argmax(output,dim=1)
        tp = 0
        tp = torch.sum(preds == target).item()

    return tp / len(target)

accFile = open('/tmp/acc.txt','w')
def accuracyBal(output, target, percent=0.1):
    with torch.no_grad():

        assert output.shape[0] == len(target)
        preds = torch.argmax(output,dim=1)
        npTarg = np.array(target.cpu())
        npPreds = np.array(preds.cpu())
        targs1 = np.where(npTarg==1) ; len1 = len(targs1[0])
        targs0 = np.where(npTarg==0) ; len0 = len(targs0[0])
        tp0 = 0 if not len0 else np.sum(npPreds[targs1] == npTarg[targs1])/len0
        tp1 = 0 if not len1 else np.sum(npPreds[targs0] == npTarg[targs0])/len1
        tpAvg = (tp1+tp0)/2.
        # if (tp1>0 and tp0 > 0):
        msg = f'Accur0={tp0} Accur1={tp1} tpAvg={tpAvg}'
        accFile.write(f'{msg}\n')
        accFile.flush()
    return tpAvg

def accuracyNew(output, target, percent=0.1):
    with torch.no_grad():

        assert output.shape[0] == len(target)
        preds = torch.argmax(output,dim=1)
        nt = target.numpy()
        np = preds.numpy()
        from collections import defaultdict
        acc = defaultdict(defaultdict)
        for [t,p] in list(zip(nt,np)):
          x = (t == p) + 0
          newx = (0 if not x in acc[t] else acc[t][x]) + 1
          acc[t][p] = newx

        # accurs = { t:xv/sum(v.values()) for t,v in acc.items() for xk,xv in v if xv }
        accurs = {}
        print(f'acc={acc}')
        for k,v in acc.items():
          for xk, xv in ([xk, xv] for xk, xv in v.items()):
            accurs[k] = xv/sum(v.values())
        
        mean_accur = sum(accurs) / len(accurs)
        print(f'MeanAccuracy={mean_accur} Accuracies = {accurs}')
        return mean_accur

accuracy = accuracyBal

def avg_precision(output, target, num_classes, mode='macro'):
    
    with torch.no_grad():

        assert output.shape[0] == len(target)
    
        if mode == 'micro':
            return _precision_micro_agg(output, target)
        elif mode == 'macro':
            return _precision_macro_agg(output, target, num_classes)
        else:
            raise ValueError('Pass a valid type of aggregation to the precision metric.')



def avg_recall(output, target, num_classes, mode='macro'):
    
    with torch.no_grad():

        assert output.shape[0] == len(target)

        if mode == 'micro':
            return _recall_micro_agg(output, target)
        elif mode == 'macro':
            return _recall_macro_agg(output, target, num_classes)
        else:
            raise ValueError('Pass a valid type of aggregation to the precision metric.')




def _precision_macro_agg(output, target, num_classes):
    preds = torch.argmax(output,dim=1)

    ret = torch.zeros(num_classes)

    for ind in target.unique():

        tp = torch.sum( (preds == target) * (preds == ind) ).item()
        fp = torch.sum( (preds != target) * (preds == ind) ).item()

        denom = (tp + fp)
        ret[ind] = tp / denom if denom > 0 else 0

    return ret.mean()



def _precision_micro_agg(output, target):
    preds = torch.argmax(output,dim=1)

    tp_cumsum = 0
    fp_cumsum = 0

    for ind in target.unique():

        tp = torch.sum( (preds == target) * (preds == ind) ).item()
        fp = torch.sum( (preds != target) * (preds == ind) ).item()

        tp_cumsum += tp
        fp_cumsum += fp

    return tp_cumsum / (tp_cumsum + fp_cumsum)




def _recall_macro_agg(output, target, num_classes):
    preds = torch.argmax(output,dim=1)

    ret = torch.zeros(num_classes)

    for ind in target.unique():

        tp = torch.sum( (preds == target) * (target == ind) ).item()
        fn = torch.sum( (preds != target) * (target == ind) ).item()
        
        denom = (tp + fn)
        ret[ind] = tp / denom if denom > 0 else 0

    return ret.mean()



def _recall_micro_agg(output, target):
    preds = torch.argmax(output,dim=1)

    tp_cumsum = 0
    fn_cumsum = 0

    for ind in target.unique():

        tp = torch.sum( (preds == target) * (target == ind) ).item()
        fn = torch.sum( (preds != target) * (target == ind) ).item()

        tp_cumsum += tp
        fn_cumsum += fn


    return tp_cumsum / (tp_cumsum + fn_cumsum)


def classification_metrics(num_classes):
    avg_p = lambda x,y: avg_precision(x,y,num_classes,mode='macro')
    avg_p.__name__ = 'avg_precision'

    avg_r = lambda x,y: avg_recall(x,y,num_classes,mode='macro')
    avg_r.__name__ = 'avg_recall'
    return [accuracy, avg_p, avg_r]
