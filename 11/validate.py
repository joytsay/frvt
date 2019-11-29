import os
import numpy as np
import io
import shutil

def load_index_lists(data_dir,file_name):
    with open(os.path.join(data_dir,file_name),"r") as f:
        lines = f.readlines()
        lines = [l.strip('\n') for l in lines]
        imgPath = ([l.strip().split(' ')[1] for l in lines])
        imgPath.pop(0)
        imgName = ([p.strip().split('/')[3] for p in imgPath])
        noExtImgName = [n.strip(".ppm") for n in imgName]
        enrollId = ([n.strip().split('-')[0] for n in noExtImgName])
    return enrollId, imgName

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
    FR_img_dir = "common/images"
    FAR_dir = "11/FAR"
    FRR_dir = "11/FRR"
    enrollId, enrollImg= load_index_lists(FR_data_dir,"enroll.log")
    verifId, verifImg = load_index_lists(FR_data_dir,"verif.log")
    matchScore = load_score_lists(FR_data_dir, "match.log")
    for i, score in enumerate(matchScore):
        if float(score) >= FRconfidence:
            if enrollId[i] == verifId[i]:
                Known += 1
            else:
                unknownToKnown += 1
                enrollDestFileName = str(unknownToKnown) + "_enrollLine(" + str(i+2) + ")_score("  + str(score) + ")_" + enrollImg[i]
                shutil.copy(os.path.join(FR_img_dir,enrollImg[i]), 
                    os.path.join(FAR_dir,enrollDestFileName))
                verifDestFileName = str(unknownToKnown) + "_verifLine(" + str(i+2) + ")_score("  + str(score) + ")_" + verifImg[i]
                shutil.copy(os.path.join(FR_img_dir,verifImg[i]),
                    os.path.join(FAR_dir,verifDestFileName))
        else:
            if enrollId[i] != verifId[i]:
                Unknown += 1
            else:
                knownToUnknown += 1
                enrollDestFileName = str(knownToUnknown) + "_enrollLine(" + str(i+2) + ")_score("  + str(score) + ")_" + enrollImg[i]
                shutil.copy(os.path.join(FR_img_dir,enrollImg[i]), 
                    os.path.join(FRR_dir,enrollDestFileName))
                verifDestFileName = str(knownToUnknown) + "_verifLine(" + str(i+2) + ")_score("  + str(score) + ")_" + verifImg[i]
                shutil.copy(os.path.join(FR_img_dir,verifImg[i]),
                    os.path.join(FRR_dir,verifDestFileName))
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