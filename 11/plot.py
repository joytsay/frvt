import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def loadDataframe(fileDirectory):
    # Load from a file
    enrolldir = fileDirectory + "/enroll.log"
    verifdir = fileDirectory + "/verif.log"
    matchdir = fileDirectory + "/match.log"
    dfEnroll = pd.read_csv(enrolldir, delimiter= '\s+')
    dfVerif = pd.read_csv(verifdir, delimiter= '\s+')
    dfMatch = pd.read_csv(matchdir, delimiter= '\s+')    
    # Get UUID
    imgEnrollName = ([p.strip().split('/')[3] for p in dfEnroll['image']])
    noExtEnrollName = [n.strip(".ppm") for n in imgEnrollName]
    enrollUUID = ([n.strip().split('-')[0] for n in noExtEnrollName]) 
    dfEnroll['UUID'] = enrollUUID
    imgVerifName = ([p.strip().split('/')[3] for p in dfVerif['image']])
    noExtVerifName = [n.strip(".ppm") for n in imgVerifName]
    verifUUID = ([n.strip().split('-')[0] for n in noExtVerifName]) 
    dfVerif['UUID'] = verifUUID
    # print(dfEnroll.head())
    # print(dfVerif.head())
    # print(dfMatch.head())
    return dfEnroll,dfVerif,dfMatch

def plotGIBoxScatter(dfGIbox):
    #label for same id(Genuine) and different id(Imposter)
    # ax = sns.boxplot(x="GIlabel", y="score", data=dfGIbox, showfliers = False)
    ax = sns.swarmplot(x="GIlabel", y="score", data=dfGIbox, hue="returnCode")
    minVal = dfGIbox['score'].min()
    maxVal = dfGIbox['score'].max()
    tickLen = float(maxVal - minVal)/20.0
    plt.yticks(np.arange(minVal, maxVal, tickLen))
    plt.grid()
    plt.savefig('GIboxPlot.png')
    plt.show()
    return plt

#load image list
dfEnroll, dfVerif, dfMatch = loadDataframe('validation')

#match pairs similarity score
dfGIbox = pd.DataFrame(columns=['GIlabel','score','returnCode'])
for i in range(len(dfMatch)):
    dfGIbox.at[i,'score'] = dfMatch.at[i,'simScore']
    if dfEnroll.at[i, 'returnCode'] != 0 or dfVerif.at[i, 'returnCode'] != 0:
        dfGIbox.at[i, 'returnCode'] = 'FaceDetectionError'
    else:
        dfGIbox.at[i, 'returnCode'] = 'MatchSuccess'
    if dfEnroll.at[i, 'UUID'] == dfVerif.at[i, 'UUID']:
        dfGIbox.at[i, 'GIlabel'] = 'Genuine'
    else:
        dfGIbox.at[i, 'GIlabel'] = 'Imposter'
# print(dfGIbox)
plotGIBoxScatter(dfGIbox)