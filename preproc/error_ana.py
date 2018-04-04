import os
from collections import defaultdict

def count_overlap(file):
    with open(file, "r") as indsf:
        indsmap = defaultdict(dict)
        for l in indsf:
            if len(l.split("\t")) == 3:
                task, model, inds = l.strip("\n").split("\t")
                indsmap[task][model] = inds.split(" ")
            else:
                task, model, iter, inds = l.strip("\n").split("\t")
                indsmap[task + "_" + iter][model] = inds.split(" ")
        for task, entries in indsmap.items():
            main_correct = 0.0
            relabel_correct = 0.0
            both_correct = 0.0
            both_incorrect = 0.0
            len_gold = len(indsmap[task]["Gold"])
            all = float(len_gold)
            for i in range(0, len_gold):
                if (indsmap[task]["Gold"][i] == indsmap[task]["Relabel model"][i]) and (indsmap[task]["Relabel model"][i] == indsmap[task]["Main model"][i]):
                    both_correct += 1
                elif (indsmap[task]["Relabel model"][i] == indsmap[task]["Main model"][i]) and  (indsmap[task]["Main model"][i] != indsmap[task]["Gold"][i]):
                    both_incorrect += 1
                elif indsmap[task]["Gold"][i] == indsmap[task]["Relabel model"][i]:
                    relabel_correct += 1
                else:
                    main_correct += 1
            rate_both_correct = (both_correct/all)
            rate_both_incorect = (both_incorrect / all)
            rate_relab_correct = (relabel_correct / all)
            rate_main_correct = (main_correct / all)
            prop_main = rate_main_correct / (rate_both_correct + rate_relab_correct + rate_main_correct)
            prop_relab = rate_relab_correct / (rate_both_correct + rate_relab_correct + rate_main_correct)
            print(task, "Rate both correct", str(rate_both_correct))
            print(task, "Rate both incorrect", str(rate_both_incorect))
            print(task, "Rate only relabel correct", str(rate_relab_correct))
            print(task, "Rate only main correct", str(rate_main_correct))
            print(task, "Prop main", str(prop_main * 100))
            print(task, "Prop relab", str(prop_relab * 100))


if __name__ == "__main__":
    #reformat_log_tabs()
    dirpath = "../"
    files = os.listdir(dirpath)
    for f in files:
        if f.endswith("_inds.txt"):
            if not "learningcurve" in f:
                continue
            if not "label-transfer" in f:
                continue
            if not "multi" in f:
                continue
            print("Reading file", f)
            count_overlap(os.path.join(dirpath, f))
            print("")