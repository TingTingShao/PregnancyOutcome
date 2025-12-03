import os
import pandas as pd
import numpy as np
import pickle
from sksurv.util import Surv

class Utils:

    def __init__(self):
        pass

    def load_data(self, filename):
        if filename.endswith(".xls"):
            return pd.read_excel(filename,engine="xlrd", header=2)
        elif filename.endswith(".xlsx"):
            return pd.read_excel(filename)
        elif filename.endswith(".csv"):
            return pd.read_csv(filename)
        elif filename.endswith(".pkl"):
            with open(filename, 'rb') as file:
                return pickle.load(file)
        else:
            raise ValueError("Unsupported file format")
    
    def save_data(self, data, filename):
        if filename.endswith(".pkl"):
            with open(filename, 'wb') as file:
                pickle.dump(data, file)
        elif filename.endswith(".csv"):
            data.to_csv(filename, index=False)
        elif filename.endswith('.txt'):
            with open(filename, 'w') as file:
                file.write("\n".join(data))
        else:
            raise ValueError("Unsupported file format")
    
    def fit_scale(self, X, num_features):
        std=X[num_features].std(ddof=0)
        # avoid std = 0
        std_replaced=std.replace(0, 1) 
        return {"cols": num_features, "sd": std_replaced.to_dict()}

    def transform_scale(self, X, params):
        """
        Apply \tilde{x} = x / sd using SDs computed on the TRAIN set.
        Non-numeric columns are left untouched.
        """
        cols = params["cols"]
        sd = pd.Series(params["sd"])
        Xs = X.copy()
        Xs[cols] = Xs[cols].astype(float).div(sd, axis=1)
        return Xs

    def construct_y_array(self, y):
        y_surv=y.to_numpy()
        event= y_surv[:, 0].astype(bool)
        time= y_surv[:, 1].astype(float)
        y_surv=Surv.from_arrays(event, time)
        return y_surv

    def build_stacked_test(self, X_test, cov_cols, event_times):
        """
        Construct a stacked test set for the SDA-MCLERNON method.
        Parameters
        ----------
        X_test : pd.DataFrame
            Test set with covariates.
        cov_cols : list of str
            List of covariate column names.
        event_times : list of int
            List of unique event times in the training set.
        Returns
        -------
        X_test_stacked : np.ndarray
            Stacked test set with covariates.
        """

        X = X_test[cov_cols].to_numpy(copy=False)       # (n, p)
        n, p = X.shape
        K = len(event_times)

        # repeat each row K times
        base_rep = np.repeat(X, K, axis=0)             # (n*K, p)

        # one-hot riskset: I_K tiled n times
        risk_block = np.tile(np.eye(K, dtype=base_rep.dtype), (n, 1))  # (n*K, K)

        X_stk = np.hstack([risk_block, base_rep])      # (n*K, K+p)  — pure numpy
        cols = [f'risk_{t}' for t in event_times] + cov_cols
        X_stk = pd.DataFrame(X_stk, columns=cols)      # (n*K, K+p)  — pandas
        return X_stk

    def evolution_IVF_treatment(self, data):
        """
        input: data with pregnancy_outcome_binary and total_cycle/transfer_taken"""

        cycle_totals=[]
        max_num_cycles=data['time'].max()
        for i in range(max_num_cycles):
            cycle_totals.append(sum(data['time']>i))
        
        results=data.groupby(['time', 'event']).size().unstack().reindex(index=range(1, max_num_cycles+1), columns=[0,1])

        dropout_num=results[0]
        success_num=results[1]
        df=pd.DataFrame({
            'cycle': range(1, max_num_cycles+1),
            'num_patients': cycle_totals,
            'success_num': success_num,
            'dropout_num': dropout_num         
        })
        return df.reset_index(drop=True)