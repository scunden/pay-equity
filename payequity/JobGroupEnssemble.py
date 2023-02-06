#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import matplotlib.pyplot as plt
import warnings
from .JobGroup import JobGroup
from .Audit import Audit


plt.style.use('ggplot')
warnings.simplefilter(action='ignore', category=FutureWarning)
pd.options.mode.chained_assignment = None  # default='warn'

class JobGroupEnssemble(JobGroup):
#     def __str__(self):
#         pass
    
#     def __repr__(self):
#         return "<Job Group Enssemble> {} | Headcount: {}".format(self.name, self.df.shape[0])
    
    def __init__(
        self, 
        df, 
        eeid, 
        pay_component, 
        predictive_vars, 
        diagnostic_vars=None, 
        iter_order=None,
        column_map=None, 
        column_map_inv=None, 
        div_vars=None, 
        div_min=None, 
        div_ref=None,
        name="Company",
        job_group_column='Business Category',
        headcount_cutoff=100
        ):
        
        super().__init__(
            df, 
            eeid, 
            pay_component, 
            predictive_vars, 
            diagnostic_vars, 
            iter_order,
            column_map, 
            column_map_inv, 
            div_vars, 
            div_min, 
            div_ref,
            name=name
        ) 
        self.job_group_column = self.column_map[job_group_column]
        self.headcount_cutoff = headcount_cutoff
    
    def generate_job_groups(self):

        
        self.job_groups = []
        
        for jg in self.df[self.job_group_column].unique():
            df_temp = self.df[self.df[self.job_group_column]==jg]
            headcount = df_temp.shape[0]
            
            if headcount>self.headcount_cutoff:
                self.job_groups.append(JobGroup(
                    df=df_temp, 
                    eeid=self.eeid, 
                    pay_component=self.pay_component, 
                    predictive_vars=self.predictive_vars, 
                    diagnostic_vars=self.diagnostic_vars, 
                    iter_order=self.iter_order,
                    column_map=self.column_map, 
                    column_map_inv=self.column_map_inv, 
                    div_vars=self.div_vars, 
                    div_min=self.div_min, 
                    div_ref=self.div_ref,
                    name=jg
                ))
                self.logger.info("{} Job Group Created and Validated".format(jg))
            else:
                self.logger.error("Failed to build {} job group. Not enough headcount ({})".format(jg,headcount))
    
    def set_overall_references(self, specified):
        
        for jg in self.job_groups:
            self.logger.info("Setting reference for {}".format(jg.name))
            jg.set_references(specified)
    
    def run_regressions(self):
        for job_group in self.job_groups:
            job_group._run_regression()
        self.regressors = [x.regressor for x in self.job_groups]
        self.logger.info("Use generate_audit() to compile and access regression results")
    
    def run_iter_regressions(self):
        for job_group in self.job_groups:
            job_group._iterative_regression()
        self.logger.info("Use generate_audit() to compile and access iter regression results") 
        
    def run_individual_remediation(self):
        for job_group in self.job_groups:
            job_group._run_individual_remediation()
        self.logger.info("Use generate_audit(remediatied=True) to compile and access remediation summary") 

    def run_segment_gap(self, segment):
        results = pd.DataFrame()
        for job_group in self.job_groups:
            if segment in job_group.df.columns:
                res = job_group.segment_gap(segment)
                results = pd.concat([results, res], axis=0)
        return results
                
    def generate_audit(self, remediated=False):
        
        self.audit = Audit(
            self.regressors,
            div_vars=self.div_vars, 
            div_min=self.div_min, 
            column_map_inv=self.column_map_inv,
            name=self.name,
            remediated=remediated
        )

    
    


        