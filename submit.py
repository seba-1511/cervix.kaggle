import pandas as pd
import numpy as np

def createSubmission( model, data, submission_name = "C:/Users/Neel Tiruviluamala/Desktop/SubmissionTest615.csv"):
    
    names = np.array([str(i) + '.jpg' for i in range(512)]+['1'+str(i).zfill(4) + '.jpg' for i in range(3506)]).reshape(4018,1)
    #This potentially needs to be modified using the argument data
    
    t1p = (np.ones(4018)*.17).reshape(4018,1)
    #This needs to be filled in using arguments model and data
    
    t2p = (np.ones(4018)*.53).reshape(4018,1)
    #This needs to be filled in using arguments model and data
    
    t3p = 1.0 - t1p - t2p
    
    submission = pd.DataFrame(data = np.concatenate((names, t1p, t2p, t3p),axis=1), columns = ['image_name', 'Type_1', 'Type_2', 'Type_3'])
    
    submission.to_csv(submission_name, index = False)
    
createSubmission(model = 'blank', data = 'blank')