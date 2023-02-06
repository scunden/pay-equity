#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
import logging
import statsmodels.api as sm
from scipy.stats import ttest_ind

plt.style.use('ggplot')
warnings.simplefilter(action='ignore', category=FutureWarning)
pd.options.mode.chained_assignment = None  # default='warn'

class Regressor():
#     def __str__(self):
#         pass
    
#     def __repr__(self):
#         return "<Job Group Object> {} | Headcount: {}".format(self.name, self.df.shape[0])
    
    def __init__(
        self, 
        df,
        eeid,
        numerical,
        categorical,
        y,
        references, 
        hug_vars,
        predictive_vars, 
        diagnostic_vars=None, 
        iter_order=None,
        column_map=None, 
        column_map_inv=None, 
        div_vars=None, 
        div_min=None, 
        div_ref=None,
        name="Company"):
        
        self.name=name
        self.logger = self._get_logger(name=self.name+" Regressor")

        self.df=df
        self.eeid = eeid
        self.numerical=numerical
        self.categorical=categorical
        self.y=y
        self.references=references
        self.hug_vars = hug_vars
        self.predictive_vars=predictive_vars
        self.diagnostic_vars=diagnostic_vars
        self.iter_order=iter_order
        self.column_map=column_map
        self.column_map_inv=column_map_inv
        self.eeid_original = self.column_map_inv[self.eeid]
        self.div_vars=div_vars
        self.div_min=div_min
        self.div_ref=div_ref
            
    def _get_logger(self, name='__name__'):
        logger = logging.getLogger(name)
        logger.setLevel(logging.INFO)
        formatter = logging.Formatter('%(name)s| %(levelname)s: %(message)s'.format(name))
        
        if not logger.handlers:
            # Prevent logging from propagating to the root logger
            logger.propagate = 0
            console = logging.StreamHandler()
            console.setLevel(logging.DEBUG)
            console.setFormatter(formatter)
            logger.addHandler(console)
            
        return logger
                
    def _format_X(self, df, variables):   
        X = df[variables].copy()
        X = pd.get_dummies(
            X, 
            columns=[x for x in [y for y in self.categorical if y in variables]], 
            drop_first=True, 
            prefix_sep='|'
        )
        X = sm.add_constant(X)
        return X
    
    def _generate_coefficients(self, results, label="diagnostic"):
        self.logger.debug("Generating {} coefficients".format(label))
        coef = results.summary2().tables[1]
        coef['Exp. Coef.'] = np.exp(coef['Coef.'])-1
        coef['Job Group'] = self.name
        coef = coef.reset_index()
        coef['Variable'] = coef['index'].str.split('|').str[0]
        coef['Reference'] = coef['Variable'].map(self.references)  
        coef['Variable'] = coef['Variable'].map(self.column_map_inv) 
        coef['Feature'] = coef['index'].str.split('|').str[1].str.strip('_')
        
        coef['Significant'] = np.where(coef['P>|t|']<0.05,"Yes","No")
        coef = coef[['Job Group','Variable','Feature','Reference','Exp. Coef.','Significant','P>|t|']]
        coef[(coef['Variable'].isna())&(coef['Feature'].isna())]['Variable'] = 'Constant'
        coef['Variable'] = np.where(
            (coef['Variable'].isna())&(coef['Feature'].isna()),
            'Constant',
            coef['Variable']
        )
        
        return coef

    def _format_coefficients(self, coef, vars, vif=None):
        coef = coef.merge(self._headcount_by_feature(vars), how='left', on=['Variable','Feature'])
        if vif is not None:
            coef = coef.merge(vif, how='left', on=['Variable','Feature'])
        return coef    
    
    def _dof(self, X, label, results, y=None, predictions=None):

        y = self.y if y is None else y
        predictions = self.predictions.copy() if predictions is None else predictions
        
        self.logger.debug("Generating {} DoF".format(label))
        dof_df = pd.DataFrame()
        n, k = X.shape
        dof = n-k
        dof_ratio = dof/k
        r2=results.rsquared
        adj_r2=results.rsquared_adj
        mape = self._MAPE(np.exp(y), predictions['Prediction'])

        values = [n, k,dof, dof_ratio, r2, adj_r2, mape]
        columns = ["Observations","Features","DoF","Ratio","R2","Adj. R2","MAPE"]
        
        dof_df.loc[self.name, columns] = values
        
        return dof_df

    def _regression(self, y, X, label):
        
        self.logger.debug("Running {} regression".format(label))

        model = sm.OLS(y, X)
        results  = model.fit()

        return results

        
    def run_predictive_regression(self, label='predictive'):
        
        self.predictive_X = self._format_X(self.df, self.predictive_vars)
        self.predictive_results = self._regression(y=self.y, X=self.predictive_X,label=label)

        predictions = self._generate_predictions(self.predictive_X, self.predictive_results)
        self.predictions = self._generate_outliers(predictions)
        self.predictions = self.predictions.join(self.df[self.eeid])

        self.predictive_corr = self._generate_correlations(self.predictive_X , label=label)
        self.predictive_dof = self._dof(self.predictive_X , label=label, results=self.predictive_results)

        vif = self._generate_vif(self.predictive_X , label=label)
        coef = self._generate_coefficients(self.predictive_results, label=label)
        coef = self._format_coefficients(coef, self.predictive_vars, vif)

        self.predictive_coef = coef.copy()

    def run_diagnostic_regression(self, label='diagnostic'):
        
        self.diagnostic_X = self._format_X(self.df, self.diagnostic_vars)
        self.diagnostic_results = self._regression(y=self.y, X=self.diagnostic_X,label=label)
        predictions = self._generate_predictions(self.diagnostic_X, self.diagnostic_results)
        self.diagnostic_corr = self._generate_correlations(self.diagnostic_X , label=label)
        self.diagnostic_dof = self._dof(self.diagnostic_X , label=label, results=self.diagnostic_results, predictions=predictions)

        vif = self._generate_vif(self.diagnostic_X , label=label)
        coef = self._generate_coefficients(self.diagnostic_results, label=label)
        coef = self._format_coefficients(coef, self.diagnostic_vars, vif)

        self.diagnostic_coef = coef.copy()

    def run_regression(self):

        self.run_predictive_regression()
        self.run_diagnostic_regression()

        self.hug_coef = self.diagnostic_coef[(self.diagnostic_coef.Feature.isin(self.div_min.values()))
        ]
        
    def _generate_correlations(self, X,label="diagnostic"):
        self.logger.debug("Generating {} correlation table".format(label))
        corr = X.corr().stack().reset_index()
        corr['Job Group'] = self.name
        corr['Variable 1'] = corr['level_0'].str.split('|').str[0]
        corr['Feature 1'] = corr['level_0'].str.split('|').str[1].str.strip('_')
        corr['Variable 2'] = corr['level_1'].str.split('|').str[0]
        corr['Feature 2'] = corr['level_1'].str.split('|').str[1].str.strip('_')
        
        corr['Variable 1'] = corr['Variable 1'].map(self.column_map_inv)
        corr['Variable 2'] = corr['Variable 2'].map(self.column_map_inv)
        corr.rename({0:'Correlation'}, axis=1, inplace=True)
        corr['Key 1'] = corr['Variable 1'].astype(str)+corr['Feature 1'].astype(str)
        corr['Key 2'] = corr['Variable 2'].astype(str)+corr['Feature 2'].astype(str)
        corr = corr[corr['Key 1']!= corr['Key 2']]
        
        return corr[['Job Group','Variable 1','Feature 1','Variable 2','Feature 2','Correlation']]
        
    def _generate_vif(self, X, label="diagnostic"):
        self.logger.debug("Generating {} VIF scores".format(label))
        # VIF dataframe
        vif = pd.DataFrame()
        vif["feature"] = X.columns

        # calculating VIF for each feature
        vif["VIF"] = [self.variance_inflation_factor(X, i) for i in range(len(X.columns))]
        vif['Variable'] = vif['feature'].str.split('|').str[0]
        vif['Variable'] = vif['Variable'].map(self.column_map_inv)
        vif['Feature'] = vif['feature'].str.split('|').str[1].str.strip('_')
        
        return vif[['Variable','Feature','VIF']]
    
    def _MAPE(self, y, y_pred):
        return np.mean(np.abs((y - y_pred)/y))
    
    def variance_inflation_factor(self, X, exog_idx):
        """
        exog : ndarray, (nobs, k_vars)
            design matrix with all explanatory variables, as for example used in
            regression
        exog_idx : int
            index of the exogenous variable in the columns of exog
        """
        exog = X.values
        self.logger.debug("VIF Score for {}".format(X.columns[exog_idx]))
        k_vars = exog.shape[1]
        x_i = exog[:, exog_idx]
        mask = np.arange(k_vars) != exog_idx
        x_noti = exog[:, mask]
        r_squared_i = sm.OLS(x_i, x_noti).fit().rsquared
        np.seterr(divide='ignore')
        vif = 1. / (1. - r_squared_i)
        return vif
    
    def _headcount_by_feature(self, variables):
        hc_df = pd.DataFrame()
        for variable in [x for x in [y for y in self.categorical if y in variables]]:
            count = self.df[variable].value_counts().reset_index().rename({'index':'Feature',
                                                                           variable:'Headcount'}, axis=1)
            count['Variable'] = self.column_map_inv[variable]
            hc_df = pd.concat([hc_df, count])
            
        return hc_df
  
    def iterative_regression(self, export=False):
        label='iterative'
        if self.iter_order is None:
            self.logger.error("No order has been set - cannot start iterative regression. Use set_order()")
        else:
            self.iterative_coef = pd.DataFrame()
            for diversity in self.hug_vars:
                variables = [diversity]+self.iter_order
                
                for i in range(len(variables)):
                    modelling_variables = variables[:i+1]
                    self.logger.debug('{} Iteration {}: Adding {}'.format(
                        self.column_map_inv[diversity],
                        i+1, 
                        self.column_map_inv[modelling_variables[-1]]
                    ))

                    X = self._format_X(self.df, modelling_variables)
                    results = self._regression(y=self.y, X=X, label=label)

                    coef = self._generate_coefficients(results, label=label)
                    coef = coef.merge(self._headcount_by_feature(modelling_variables), how='left', on=['Variable','Feature'])
                    coef['Variable Added'] = self.column_map_inv[modelling_variables[i]]
                    coef['Iteration'] = i
                    self.iterative_coef = pd.concat([self.iterative_coef, coef]) 
            
            self.hug_iter_coef = self.iterative_coef[self.iterative_coef['Variable'].isin(\
                [self.column_map_inv[x] for x in self.hug_vars])]
            
            return self.hug_iter_coef

    def _generate_predictions(self, X, results, alpha=0.05, y=None):
        
        self.logger.debug("Generating predictions")

        y = self.y if y is None else y
        pred = results.get_prediction(X)
        pred = np.exp(pred.summary_frame(alpha=alpha))
        pred = pred[['mean','obs_ci_lower','obs_ci_upper']]
        pred.rename({ 'mean':'Prediction',
                                 'obs_ci_lower':"Lower",
                                 'obs_ci_upper':"Upper"}, 
                                axis=1, inplace=True)

        pred['Job Group'] = self.name                        
        pred['Actual'] = np.exp(y)

        return pred

    def _generate_outliers(self, pred, _from='Actual', prefix=""):

        lower_o = "{}Lower Outlier".format(prefix)
        upper_o = "{}Upper Outlier".format(prefix)
        total_o = "{}Outlier".format(prefix)

        pred[lower_o] = np.where(pred[_from]<pred['Lower'],1,0)
        pred[upper_o] = np.where(pred[_from]>pred['Upper'],1,0)
        pred[total_o] = pred[lower_o]+pred[upper_o]
        
        return pred
    
    def _calculate_adjusted_pay(self, pred, eligibility, _from='Actual', eeid='EMPLOYEE_ID', prefix=""):
        
        pred = pred.copy().join(self.df[eeid])
        
        pred = self._generate_outliers(pred, _from=_from, prefix=prefix)    
        lower_o = "{}Lower Outlier".format(prefix)  
        pred['Eligibility'] = np.where(pred[eeid].isin(eligibility),1,0) 
        
        pred['Pay Adjustment'] = np.where(pred[lower_o]==1,(pred['Lower']-pred[_from])*pred['Eligibility'], 0)
        pred['Adjusted Pay'] = pred[_from]+pred['Pay Adjustment'] 

        return pred['Adjusted Pay']
    
    def _remediate_to_alpha(self, scenario, variable, eligibility, alpha=0.95, _from="Actual"):
        assert(_from in self.predictions.columns)
        label = "remediation"
        prefix="Adj "

        results = self._regression(self.y, self.predictive_X, label=label)
        pred = self._generate_predictions(self.predictive_X, results, alpha=1-alpha, y=None)
        pred = pred.join(self.predictions[_from]) if _from not in pred.columns else pred

        pred = self._generate_outliers(pred, _from=_from, prefix=prefix)
        pred['Adjusted Pay'] = self._calculate_adjusted_pay(pred, eligibility, _from=_from, prefix=prefix)
        
        adj_y = np.log(pred['Adjusted Pay'])
        adj_results = self._regression(y=adj_y, X=self.diagnostic_X, label=label)   

        adj_coef = self._generate_coefficients(adj_results, label=label)
        adj_coef = self._format_coefficients(adj_coef, self.diagnostic_vars, vif=None)
        
        df_variable = self.div_vars[variable]
        minority = self.div_min[variable]
        gap, sig = self._get_gap_and_sig(adj_coef, variable)        

        return pred, gap, sig, alpha
    
    def _remediate_outliers(self, variable, scenario="S1"):

        assert(variable in self.div_vars.keys())
        eligibility = self.df['EMPLOYEE_ID'].unique()
        pred, gap, sig, alpha = self._remediate_to_alpha(scenario, variable, eligibility)

        return pred, gap, sig, alpha
    
    def _format_adjustments(self, pred, scenario):
        adj_ind = "{} Adjusted?".format(scenario)
        adj = "{} Pay Adjustments".format(scenario)
        adj_pay = "{} Adjusted Pay".format(scenario)
        
        self.predictions[adj_pay] = self.predictions["Actual"] if pred is None else pred["Adjusted Pay"]
        self.predictions[adj] = 0 if pred is None else pred["Adjusted Pay"]-pred["Actual"]
        self.predictions[adj_ind] = 0 if pred is None else np.where(self.predictions[adj]>0,1,0)
    
    def _remediation_loop(self, scenario, stat, crit, eligibility, variable, _from="Actual",init_gap=None, init_sig=None):

        pred=None
        alpha = 0.95
        gap = init_gap
        sig = init_sig

        stat = sig if "S2" in scenario else gap

        while (round(stat,3) < crit) & (alpha >=0):

            alpha = alpha - 0.01        
            pred, gap, sig, alpha = self._remediate_to_alpha(scenario, variable, eligibility, alpha, _from)
            stat = sig if "S2" in scenario else gap

        self._format_adjustments(pred, scenario)

        return pred, gap, sig, alpha
    
    def _get_gap_and_sig(self, coef, variable):
        df_variable = self.div_vars[variable]
        minority = self.div_min[variable]

        df_variable = self.column_map_inv[df_variable]
        variable_filter=(coef.Variable.isin([df_variable]))
        feature_filter=(coef.Feature.isin([minority]))
        cols = ['Exp. Coef.','P>|t|']
        
        return coef[variable_filter&feature_filter][cols].values[0]    
    

    def _remediate(self, scenario, variable, eligibility, gap=None, sig=None, _from='Actual'):

        assert(variable in self.div_vars.keys())
        assert(("S1" in scenario)|("S2" in scenario)|("S3" in scenario))
        
        df_variable = self.div_vars[variable]
        minority = self.div_min[variable]
        eeid = 'EMPLOYEE_ID'

        variable_eligibility = self.df[self.df[df_variable]==minority][eeid].unique()
        eligibility = variable_eligibility if eligibility is None else eligibility

        if gap is None or sig is None:
            y= np.log(self.predictions[_from])
            results = self._regression(y=y, X=self.diagnostic_X, label="initial")   
            coef = self._generate_coefficients(results, label="initial")
            coef = self._format_coefficients(coef, self.diagnostic_vars, vif=None)
            gap, sig = self._get_gap_and_sig(coef, variable)
        
        init_stat = sig if "S2" in scenario else gap
        crit = 0.05 if "S2" in scenario else 0
        self.logger.info("Variable: {} | Init Gap: {:.2%} | Init Sig: {:.3f}".format(variable, gap, sig))

        if "S1" in scenario:
            pred, gap, sig, alpha = self._remediate_outliers(variable)

        elif init_stat < crit:
            pred, gap, sig, alpha =  self._remediation_loop(scenario, init_stat, crit, 
            eligibility, variable, _from, init_gap=gap, init_sig=sig)
        
        else:
            pred = self.predictions[["Actual",_from]].copy().rename({_from:'Adjusted Pay'}, axis=1)
            alpha=0.95
        
        self.logger.info("Variable: {} | Scenario: {} | Alpha: {:.3f} | Gap: {:.2%} | Sig: {:.3f}".format(variable, scenario, alpha, gap, sig))
        self._format_adjustments(pred, scenario)

        return gap, sig, alpha

    def individual_remediation(self):
        self.remediation_summary = self.diagnostic_coef[self.diagnostic_coef.Variable.isin([self.column_map_inv[x] for x in self.div_vars.values()])]
        self.remediation_summary["Scenario"] = "Initial Gap"

        for variable in self.div_vars.keys():
            gap, sig, eligibility = None, None, None
            _from="Actual"
            for scenario in ["S1","S2","S3"]:
                scenario = scenario+"_"+"".join([x[0] for x in variable.split("_")])

                gap, sig, alpha = self._remediate(
                    scenario=scenario, 
                    variable=variable, 
                    eligibility=eligibility, 
                    gap=gap, 
                    sig=sig, 
                    _from=_from
                     )
                _from = scenario+" Adjusted Pay"

                remediation_summary = self._evaluate_remediation(_from=_from, variable=variable, scenario=scenario)
                self.remediation_summary = pd.concat([self.remediation_summary, remediation_summary])

        budget_summary = self._budget_summary(self.remediation_summary)
        self.remediation_summary = self.remediation_summary.merge(budget_summary, how='left', on=["Job Group", "Scenario"])
    
    def _evaluate_remediation(self, _from, scenario, variable, label='evaluation'):
        y= np.log(self.predictions[_from])
        pr_results = self._regression(y=y, X=self.diagnostic_X, label=label)   

        pr_coef = self._generate_coefficients(pr_results, label=label)
        pr_coef = self._format_coefficients(pr_coef, self.diagnostic_vars, vif=None)
        pr_coef["Scenario"] = scenario

        return pr_coef[pr_coef['Variable']==self.column_map_inv[self.div_vars[variable]]]

    def _budget_summary(self, pr_coef):
        scenarios = [x for x in pr_coef["Scenario"].unique() if x not in ["Initial Gap"]]
        scenario_summaries = pd.DataFrame()

        for scenario in scenarios:
            summary = pd.DataFrame()
            adj = self.predictions[scenario+" Adjusted?"].sum()
            pay_adj = self.predictions[scenario+" Pay Adjustments"].sum()
            max_pay_adj = self.predictions[scenario+" Pay Adjustments"].max()
            
            summary["Total Adjusted"] = [adj]
            summary["Total Pay Adjustments"] = [pay_adj]
            summary["Avg. Pay Adjustments"] = summary["Total Pay Adjustments"]/summary["Total Adjusted"]
            summary["Max Pay Adjustments"] = [max_pay_adj]
            summary["Job Group"] = [self.name]
            summary["Scenario"] =[scenario]

            scenario_summaries = pd.concat([scenario_summaries, summary])


        return scenario_summaries

    def segment_gap(self, segment):
        df_temp = self.df.copy()
        df_temp = df_temp.merge(self.predictions, on=self.eeid, how='right')
        df_temp["Residual"] = np.log(df_temp["Actual"])-np.log(df_temp["Prediction"])

        results = pd.DataFrame()
        for feature in df_temp[segment].unique():
            df_temp_2 = df_temp[df_temp[segment]==feature]
            for diversity in self.div_vars:

                div = self.div_vars[diversity]
                minority = self.div_min[diversity]
                ref = self.div_ref[diversity]

                df_min = df_temp[(df_temp[segment]==feature)&(df_temp_2[div]==minority)]
                df_ref = df_temp[(df_temp[segment]==feature)&(df_temp_2[div]==ref)]

                minority_res = df_min["Residual"].mean()
                ref_res = df_ref["Residual"].mean()

                gap = np.exp(minority_res)/np.exp(ref_res)-1
                t,p = ttest_ind(df_min["Residual"], df_ref["Residual"])

                minority_hc = df_temp_2[df_temp_2[div]==minority].shape[0]
                ref_hc = df_temp_2[df_temp_2[div]==ref].shape[0]

                results_temp = pd.DataFrame({
                    'Job Group':self.name,
                    'Variable':[segment],
                    'Feature':[feature],
                    'Diversity':[self.column_map_inv[div]],
                    'Minority':[minority], 
                    'Minority HC':[minority_hc] ,  
                    'Reference':[ref],
                    'Reference HC':[ref_hc],  
                    'Gap':[gap],
                    'P Value':[p]
                    } )
                results = pd.concat([results,results_temp]).sort_values(["Diversity"])
        
        return results


            
            

        
