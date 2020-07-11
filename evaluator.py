import sys
import os
from sklearn.metrics import average_precision_score

if __name__=="__main__":
    #   load the ground-truth file list (val)
    gt_fn=open(sys.argv[1]).readlines()
    pred_fn=open(sys.argv[2]).readlines()
    event_name = sys.argv[2].split("/")[1].split("_")[0]
    print event_name

    print("Evaluating the average precision (AP)")

    y_gt=[]
    y_score=[]
    assert(len(y_gt)==len(y_score))

    for lines in gt_fn:
        lines = lines.split(" ")
        line = lines[0]
        label = lines[1].replace('\n', '')
        if label != event_name:
            label = "0"
        else:
            label = "1"
        y_gt.append(float(label.strip()))

    for lines in pred_fn:
        y_score.append(float(lines.strip()))

    assert(len(y_gt) == len(y_score))
    print "Average precision: ",average_precision_score(y_gt,y_score)
