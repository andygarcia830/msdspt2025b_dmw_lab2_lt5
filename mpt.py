import itertools
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# from tqdm import tqdm

class MPT:
    def generate_weights(self, len, n_samples):
        """
        Internal convenience method to generate the default weights
        for analysis.

        Parameters
        ----------
        self : PortfolioAnalysis
            The current instance of this class
        len : int
            The number of Tickets in the given Portfolio

        Returns
        -------
        increments : array-like
            The generated default weights
        """
        weights=[]

        # Generate n random values
        for i in range(n_samples):
            random_values = np.random.rand(len)
            # Normalize the values to sum to 1
            weights.append(random_values / random_values.sum()) 
        return np.array(weights)
    
    def normalize_weights(self):
         for idx, row in self.weight_matrix.iterrows():
              row = row + abs(row.min()) 
              row = row / row.sum()
              self.weight_matrix.iloc[idx] = row

    def calculate_return_variance(self, columns):
            """
            Internal convenience method to calculate the return variance.

            Parameters
            ----------
            self : PortfolioAnalysis
                The current instance of this class
            columns : list
                The column names in the data frame that contain the
                weights

            Returns
            -------
            """

            variance = pd.Series()
            for idx, row in self.mpt_results.iterrows():
                var_val = 0
                var_val = sum([row[i] ** 2 * np.var(self.returns[i]) for i in columns])
                cov = np.cov(self.returns.T)
                var_val += sum([np.dot(row[i], row[j]) * cov[i][j] for j in columns for i in columns if i != j])

                variance[idx] = var_val
            return variance

    def calculate_expected_return(self, columns):
            """
            Internal convenience method to calculate the return variance.

            Parameters
            ----------
            self : PortfolioAnalysis
                The current instance of this class
            columns : list
                The column names in the data frame that contain the
                weights

            Returns
            -------
            """
            ret = pd.Series()
            
            means = np.array([self.returns[x].mean() for x in columns])
            for idx, row in self.weight_matrix.iterrows():

                ret_val = np.dot(row, means)
                ret[idx] = ret_val
            
            return ret
    
    def __init__(self, R, weights=None):
        print(R.shape)
        self.returns=R.copy()
        # print(self.returns)

        
        # minimum = min(R.shape[0], )
        if type(weights) == type(None):
             self.weight_matrix = pd.DataFrame(self.generate_weights(R.shape[1],300))
        else:
             self.weight_matrix=pd.DataFrame(weights)
             self.normalize_weights()
        self.mpt_results = pd.DataFrame(self.weight_matrix)
        columns = self.mpt_results.columns.values
        self.returns.columns=columns
        # print('Calculating Expected Returns')
        # print(f'RETURNS {len(self.returns[0])}')
        # print(f'RESULTS {len(self.mpt_results[0])}')
        self.mpt_results['Expected Returns'] = self.calculate_expected_return(columns)
        # print('Calculating Return Variance')
        self.mpt_results['Portfolio Variance'] =\
            self.calculate_return_variance(columns)
        self.mpt_results.fillna(0, inplace=True)
