import numpy as np
from ModelMergerUpdated import *
from Gen_Emcee import *

# This works for data using covariance matrices for log likelood only

def get_bestparams(model):
    params = np.loadtxt('/Users/RyanCamo/Documents/GitHub/Honours/Hubble Diagram/Chains_OUTPUT/%s_PARAMS_covloglike.txt' %(model))
    return params 

def get_table(models,zz,mu,mu_error):
    ## Printing the Table for each model
    print(r'\begin{table}')
    print(r' \centering')    
    print(r' \caption{Model comparisons!}')    
    print(r' \label{tab:model_comp}')    
    print(r' \begin{tabular}{ccc}')    
    print(r'     \hline')        
    print(r'     Model & $\Delta$AIC & $\Delta$BIC \\')         
    print(r'     \hline')    
    for i, model in enumerate(models):
        # Calculating # AIC & BIC
        label, begin, legend = get_info(model.__name__)
        params = get_bestparams(model.__name__)
        if model.__name__ == 'FLCDM':
            params = [params]
        p1 = L_tot(params,zz,mu,mu_error,model,begin)
        n = len(mu)
        free_params = len(label)
        if model.__name__ == 'FLCDM':
            refAIC = -2*np.max(p1) + 2*free_params # may need to set LCDM as reference but may be able to define one based on the max/min in a list
            refBIC = -2*np.max(p1) + free_params*np.log(n)
        AIC = -2*p1 + 2*free_params - refAIC
        BIC = -2*p1 + free_params*np.log(n) - refBIC
        print(r'        %s  &   %s    &   %s    \\' % (model.__name__, round(AIC), round(BIC)))             
    print(r'     \hline')        
    print(r' \end{tabular}')    
    print(r'\end{table}')

if __name__ == "__main__":
    # Models to Create Data from
    models = [FLCDM,LCDM,FGChap, FwCDM, wCDM]

    # Data to test against
    data = np.genfromtxt("Hubble Diagram/DATA_simdes5yr_binned.txt",names=True,dtype=None, encoding=None, delimiter=',')
    zz = data['zCMB']
    mu = data['MU']
    error = data['MUERR']
    cov_arr = np.genfromtxt("Hubble Diagram/COVsyst_simdes5yr_binned.txt",comments='#',dtype=None, skip_header=1)
    cov = cov_arr.reshape(20,20)
    cov2 = np.diagonal(cov) 
    mu_diag = np.diag(error)**2
    mu_error = mu_diag+cov

    get_table(models,zz,mu,mu_error)