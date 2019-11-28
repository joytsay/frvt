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
    # fileNameSets = ["enroll","verif","match"]
    FR_data_dir = "11/validation"
    enrollId = load_index_lists(FR_data_dir,"enroll.log")
    verifId = load_index_lists(FR_data_dir,"verif.log")
    matchScore = load_score_lists(FR_data_dir, "match.log")
    if
    print("[SUCCESS]")
     
if __name__ == "__main__":
    main()              