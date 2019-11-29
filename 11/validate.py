import os
import numpy as np
import io

def load_index_lists(data_dir,file_name):
    with open(os.path.join(data_dir,file_name),"r") as f:
        lines = f.readlines()
        lines = [l.strip('\n') for l in lines]
        imgPath = ([l.strip().split(' ')[1] for l in lines])
        imgPath.pop(0)
        imgName = ([p.strip().split('/')[3] for p in imgPath])
        imgName = [n.strip(".ppm") for n in imgName]
        enrollId = ([n.strip().split('-')[0] for n in imgName])
    return enrollId

def load_score_lists(data_dir,file_name):
    with open(os.path.join(data_dir,file_name),"r") as f:
        lines = f.readlines()
        lines = [l.strip('\n') for l in lines]
        score = ([l.strip().split(' ')[2] for l in lines])
        score.pop(0)
    return score
        
def main():
    try:
        mode=int(input('Give FR confidence threshold: '))
    except ValueError:
        print ("Not a int number")
    FRconfidence = float(mode*0.01)
    knownToUnknown = 0
    unknownToKnown = 0
    Known = 0
    Unknown = 0
    FR_data_dir = "11/validation"
    enrollId = load_index_lists(FR_data_dir,"enroll.log")
    verifId = load_index_lists(FR_data_dir,"verif.log")
    matchScore = load_score_lists(FR_data_dir, "match.log")
    for i, score in enumerate(matchScore):
        if float(score) >= FRconfidence:
            if enrollId[i] == verifId[i]:
                Known += 1
            else:
                unknownToKnown += 1
        else:
            if enrollId[i] != verifId[i]:
                Unknown += 1
            else:
                knownToUnknown += 1
    print("\n[SUCCESS] NIST frvt validation for confidence: ",FRconfidence,
    "\nAll count: ", len(matchScore),
    "\nKnown: ", Known, 
    "\nUnknown: ", Unknown,
    "\nknownToUnknown: ", knownToUnknown, 
    "\nunknownToKnown", unknownToKnown,
    "\nFAR: ", unknownToKnown/len(matchScore)*100,
    "%\nFRR: ", knownToUnknown/len(matchScore)*100,"%\n")
     
if __name__ == "__main__":
    main()              