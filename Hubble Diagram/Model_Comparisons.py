import numpy as np
from ModelMergerUpdated import *
from Gen_Emcee import *

# This works for data using covariance matrices for log likelood only

def get_bestparams(model,label):
    #params = np.loadtxt('/Users/RyanCamo/Documents/GitHub/Honours/Hubble Diagram/Chains_OUTPUT/%s_PARAMS_covloglike.txt' %(model))
    #print(params)
    #exit()
    cols = []
    for i, l in enumerate(label):
        cols.append(i+2)

    samples = np.loadtxt('/Users/RyanCamo/Desktop/Cobaya/chains/%s_DES5YR_UNBIN.1.txt' %(model), usecols=(cols), comments='#')

    params = get_param(samples,label,model,0)


    return params 

def get_table(models,zz,mu,mu_error):
    ## Printing the Table for each model
    print(r'\begin{table}')
    print(r' \centering')    
    print(r' \caption{Model comparisons!}')    
    print(r' \label{tab:model_comp}')    
    print(r' \begin{tabular}{ccccc}')    
    print(r'     \hline')        
    print(r'     Model & $\chi^2$/dof & GoF (\%) & $\Delta$AIC & $\Delta$BIC \\')         
    print(r'     \hline')    
    for i, model in enumerate(models):
        # Calculating # AIC & BIC
        label, begin, legend = get_info(model.__name__)
        params = get_bestparams(model.__name__,label)
        params = np.array(params)
        #if model.__name__ == 'FLCDM':
        #    params = [params]
        p1 = L_tot(params,zz,mu,mu_error,model,begin)
        #print(p1)
        chi2 = -p1/0.5  # NOT SURE IF THIS IS ACTUALLY chi2. or chi2/0.5??
        n = len(mu)
        free_params = len(label)
        dof = len(mu) - free_params
        #print(chi2/dof)
        #refAIC = -2*np.max(p2) + 2#*free_params # may need to set LCDM as reference but may be able to define one based on the max/min in a list
        #refBIC = -2*np.max(p2) + np.log(n)#free_params*np.log(n)
        if model.__name__ == 'FLCDM':
            refAIC = -2*np.max(p1) + 2*free_params # may need to set LCDM as reference but may be able to define one based on the max/min in a list
            refBIC = -2*np.max(p1) + free_params*np.log(n)
        AIC = -2*p1 + 2*free_params - refAIC
        BIC = -2*p1 + free_params*np.log(n) - refBIC
        print(r'        %s  & $%s/%s$ & GoF &  %s    &   %s    \\' % (model.__name__,round(chi2,1) , dof, round(AIC,1), round(BIC,1)))             
    print(r'     \hline')        
    print(r' \end{tabular}')    
    print(r'\end{table}')

if __name__ == "__main__":
    # Models to Create Data from
    models = [FLCDM, LCDM, FwCDM, wCDM, IDE1, IDE2, IDE4, FGChap, GChap, Chap, FCa, Fwa, Fwz]



    # Data to test against
    DataToUse = 'DES5YR_UNBIN'
    DES5YR_UNBIN = np.genfromtxt("/Users/RyanCamo/Desktop/Cobaya/data/%s_data.txt" % (DataToUse), names=True)
    zz = DES5YR_UNBIN['zCMB']
    mu = DES5YR_UNBIN['MU']
    error = DES5YR_UNBIN['MUERR']
    cov_arr = np.genfromtxt("/Users/RyanCamo/Desktop/Cobaya/data/%s_cov.txt" % (DataToUse), comments='#',dtype=None)
    cov1 = cov_arr.reshape(1867,1867) 
    mu_diag = np.diag(error)**2
    mu_error = mu_diag+cov1



    # Data to test against - worked
    #data = np.genfromtxt("Hubble Diagram/DATA_simdes5yr_binned.txt",names=True,dtype=None, encoding=None, delimiter=',')
    #zz = data['zCMB']
    #mu = data['MU']
    #error = data['MUERR']
    #cov_arr = np.genfromtxt("Hubble Diagram/COVsyst_simdes5yr_binned.txt",comments='#',dtype=None, skip_header=1)
    #cov = cov_arr.reshape(20,20)
    #cov2 = np.diagonal(cov) 
    #mu_diag = np.diag(error)**2
    #mu_error = mu_diag+cov

    get_table(models,zz,mu,mu_error)