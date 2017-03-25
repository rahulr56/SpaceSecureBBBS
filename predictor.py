import pandas as pd
from sklearn.neighbors import KNeighborsClassifier as knc
from pandas import Series
from sklearn.model_selection import cross_val_score as cvs
from sklearn import svm, datasets
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsRestClassifier
matchAttributes=["Hybrid","MatchType","MatchStatus","QueueDescription","TimeInQueue","MatchSupportLevel","MatchReportSources","PendingMatchDate","MatchClosureReasons","MatchLength", "CouplesMatch","MatchCountChild","SegmentMatchCountChild","MatchCountVolunteer","SegmentMatchCountVolunteer","ChildGender", "ChildEthnicity","ChildAge","IncarceratedParent","ChildGrade","ChildLivingSituation","ChildIncomeLevel","MilitaryParent","ParentDeployed", "ChildFamilyAssistance","ChildFreeReducedlunch","ChildAutomaticProgramName","ChildReportSources","ChildActiveQueue","VolGender","VolEthnicity","VolEducationLevel","VolMaritalStatus","VolAutomaticProgramName","VolReportSources","VolActiveQueue","Beg","Open","Close","End"]
yorAttributes=['MatchType', 'MatchStatus', 'Q1', 'Q2Neg', 'Q3Neg', 'Q4Neg', 'Q5', 'Q6', 'SocAccept', 'Q1b', 'Q2b', 'Q3b', 'Q4b', 'Q5b', 'Q6b', 'SocAcceptB', 'SocAcceptPrcnt', 'Q7Neg', 'Q8', 'Q9', 'Q10Neg', 'Q11Neg', 'Q12', 'SchComp', 'Q7b', 'Q8b', 'Q9b', 'Q10b', 'Q11b', 'Q12b', 'SchCompB', 'SchCompPrcnt', 'Q13', 'Q14', 'Q15', 'EdExpect', 'Q13b', 'Q14b', 'Q15b', 'EdExpectb', 'EdExpectPrcnt', 'Q16', 'Q17', 'Q18', 'Q19', 'Grades', 'Q16b', 'Q17b', 'Q18b', 'Q19b', 'Gradesb', 'GradesPrcnt', 'Q20Neg', 'Q21Neg', 'Q22Neg', 'Q23Neg', 'Q24Neg', 'Q25Neg', 'Q26Neg', 'RiskAtt', 'Q20b', 'Q21b', 'Q22b', 'Q23b', 'Q24b', 'Q25b', 'Q26b', 'RiskAttb', 'RiskAttPrcnt', 'Q27', 'Q28', 'Q29', 'PTrust', 'Q27b', 'Q28b', 'Q29b', 'PTrustb', 'PTrustPrcnt', 'Q30Neg', 'Q31Neg', 'Truancy', 'Q30b', 'Q31b', 'Truancyb', 'TruancyPrcnt', 'Q32', 'SpAdult', 'Q32b', 'SpAdultb', 'SpAdultPrcnt', 'Q33Neg', 'JJustice', 'Q33b', 'JJusticeB', 'JJusticePrcnt', 'MatchSupportLevel', 'MatchLength', 'ChildGender', 'ChildEthnicity', 'CouplesMatch', 'MatchCountChild', 'SegmentMatchCountChild', 'MatchCountVolunteer', 'SegmentMatchCountVolunteer', 'ChildGrade', 'ChildFamilyAssistance', 'ChildFreeReducedLunch', 'VolGender', 'VolEthnicity', 'VolEducationLevel', 'VolMaritalStatus', 'ChildPartKey' , 'VolPartKey']

YOS_DATA_TRAIN_PATH="/Users/rrachapa/Desktop/DATA/shared/bbbs/matches/all/youth_outcome_reports_new.bsv"
YOS_DATA_TEST_ACTIVE_PATH="/Users/rrachapa/Desktop/DATA/shared/bbbs/matches/active/youth_outcome_reports_new.bsv"
YOS_DATA_TEST_UNS_PATH="/Users/rrachapa/Desktop/DATA/shared/bbbs/matches/active/match_details_new.bsv"
MATH_DETAILS_FILE_PATH="/Users/rrachapa/Desktop/DATA/shared/bbbs/matches/all/match_details_new.bsv"


yosData = pd.read_csv(YOS_DATA_TRAIN_PATH, delimiter="|",index_col=1)
yosTestData = pd.read_csv(YOS_DATA_TEST_ACTIVE_PATH, delimiter="|",index_col=1)

matchDetails=pd.read_csv(MATH_DETAILS_FILE_PATH,delimiter="|",index_col=1)

yorTrainData = yosData.loc[:,yorAttributes]
yorTestData= yosTestData.loc[:,yorAttributes]
matchTrainData = matchDetails.loc[:,matchAttributes]
matchTypeMap={"C":0}
matchStatusMap={}
#print yorTrainData.dtypes

yorTrainData['MatchType']=yorTrainData['MatchType'].map({'C':1})#apply(lambda x: 0 if x =='C' else 1)
yorTrainData['MatchStatus']=yorTrainData['MatchStatus'].map({'Active':1,'Completed':2})
yorTrainData['ChildFamilyAssistance']=yorTrainData['ChildFamilyAssistance'].map({'Y':1,'N':-1})
yorTrainData['CouplesMatch']=yorTrainData['CouplesMatch'].map({'Y':1,'N':-1})
yorTrainData['VolGender']=yorTrainData['VolGender'].map({'M':2,'F':1})
yorTrainData['ChildGender']=yorTrainData['ChildGender'].map({'M':2,'F':1})
yorTrainData['ChildGrade']=yorTrainData['ChildGrade'].map({'K':0})
yorTrainData['ChildEthnicity']=yorTrainData['ChildEthnicity'].map({"White":1,"Black":5,"Hispanic":2,"Separated":4,"Multi-race (Black & White)":3})
yorTrainData['VolEthnicity']=yorTrainData['VolEthnicity'].map({"White":1,"Black":0,"Hispanic":2,"Separated":0,"Multi-race (Black & White)":3})
yorTrainData['VolMaritalStatus']=yorTrainData['VolMaritalStatus'].map({"Single":1,"Married":2,"Divorced":-2,"Separated":0,"Living w/ Significant Other":-1})
yorTrainData['ChildFreeReducedLunch']=yorTrainData['ChildFreeReducedLunch'].map({"Yes":1,"No":0})
yorTrainData['VolEducationLevel']=yorTrainData['VolEducationLevel'].map({"Masters Degree":3,"Bachelors Degree":2,"Juris Doctorate (JD)":4,"Some College":1,"Some High School":0})
yorTrainData['MatchSupportLevel']=yorTrainData['MatchSupportLevel'].map({"Green":3,"Red":1,"Yellow":2})


yorTestData['MatchType']=yorTestData['MatchType'].map({'C':1})#apply(lambda x: 0 if x =='C' else 1)
yorTestData['MatchStatus']=yorTestData['MatchStatus'].map({'Active':1,'Completed':2})
yorTestData['ChildFamilyAssistance']=yorTestData['ChildFamilyAssistance'].map({'Y':1,'N':-1})
yorTestData['CouplesMatch']=yorTestData['CouplesMatch'].map({'Y':1,'N':-1})
yorTestData['VolGender']=yorTestData['VolGender'].map({'M':2,'F':1})
yorTestData['ChildGender']=yorTestData['ChildGender'].map({'M':2,'F':1})
yorTestData['ChildGrade']=yorTestData['ChildGrade'].map({'K':0})
yorTestData['ChildEthnicity']=yorTestData['ChildEthnicity'].map({"White":1,"Black":5,"Hispanic":2,"Separated":4,"Multi-race (Black & White)":3})
yorTestData['VolEthnicity']=yorTestData['VolEthnicity'].map({"White":1,"Black":0,"Hispanic":2,"Separated":0,"Multi-race (Black & White)":3})
yorTestData['VolMaritalStatus']=yorTestData['VolMaritalStatus'].map({"Single":1,"Married":2,"Divorced":-2,"Separated":0,"Living w/ Significant Other":-1})
yorTestData['ChildFreeReducedLunch']=yorTestData['ChildFreeReducedLunch'].map({"Yes":1,"No":0})
yorTestData['VolEducationLevel']=yorTestData['VolEducationLevel'].map({"Masters Degree":3,"Bachelors Degree":2,"Juris Doctorate (JD)":4,"Some College":1,"Some High School":0})
yorTestData['MatchSupportLevel']=yorTestData['MatchSupportLevel'].map({"Green":3,"Red":1,"Yellow":2})

for x in yorAttributes:
    yorTrainData.fillna(inplace=True,value=0)#method='backfill')
    yorTestData.fillna(inplace=True,value=0)

listAttr=yorAttributes[:-2]
listAttr.append(yorAttributes[-1])
knc_map_child = knc(n_neighbors=8, weights='distance')
knc_map_child.fit((yorTrainData[listAttr])[100:], (yorTrainData['ChildPartKey'])[100:])
y_score=knc_map_child.fit((yorTrainData[listAttr])[100:], (yorTrainData['ChildPartKey'])[100:])
print "CROSS VALIDATING : "+str(cvs(knc_map_child, X=(yorTrainData[listAttr])[:100], y=(yorTrainData['ChildPartKey'])[:100], verbose=1,cv=2))*100
print "PREDICTION      -      ACTUAL"
predictedTestResults=list(knc_map_child.predict(((yorTrainData[listAttr])[:100])))
for x in predictedTestResults:
    print str(x)+"	- 	"+str(yorTrainData['ChildPartKey'])[100:])
#fpr = dict()
#tpr = dict()
#roc_auc = dict()
#for i in range(100):
#    fpr[i], tpr[i], _ = roc_curve((yorTrainData[listAttr])[:100], y_score[:, i])
#    roc_auc[i] = auc(fpr[i], tpr[i])
print "SCORE : " + str(knc_map_child.score(yorTestData[listAttr], yorTestData['ChildPartKey'])*100)
# Compute micro-average ROC curve and ROC area
#fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(), y_score.ravel())
#roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
