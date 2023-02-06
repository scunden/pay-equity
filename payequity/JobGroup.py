#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import warnings
import logging
from .Regressor import Regressor

plt.style.use('ggplot')
warnings.simplefilter(action='ignore', category=FutureWarning)
pd.options.mode.chained_assignment = None  # default='warn'

class JobGroup():
#     def __str__(self):
#         pass
    
    def __str__(self):
        return "<Job Group Object> {} | Headcount: {}".format(self.name, self.df.shape[0])
    
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
        name="C&W"):

        ################### Create a function that validates all the inputs #####
        
        self.name=name
        self.logger = self._get_logger(name=self.name)
        
        # Remap columns and keep track of mapping
        self.original_cols = df.columns
        self.df = self._format_columns(df.copy())
        self.column_map = self._initialize_column_map(column_map, self.original_cols)
        self.column_map_inv = self._initialize_column_map(column_map_inv, self.original_cols, inv=True)
        self._initial_null_check()
        
        # key variables
        
        self.eeid = eeid if eeid in self.df.columns else self.column_map[eeid]
#         self.eeid = self.column_map[eeid]
        self.pay_component = pay_component if pay_component in self.df.columns else self.column_map[pay_component]
        self.key_variables = [
            self.eeid, 
            self.pay_component, 
            ]

        self.numerical, self.categorical = self._categorize_columns()
        
        # Diversity dictionaries
        self._initialize_diversities(div_vars, div_min, div_ref)
                
        # Initialize predictive, diagnostic and iterative variables
        self._initialize_variables(predictive_vars, diagnostic_vars, iter_order)
        
        self.predictions = None
        
        self.log_pay_component = 'LOG_'+self.pay_component
        self.df[self.log_pay_component] = np.log(self.df[self.pay_component])
        self.y = self.df[self.log_pay_component]
        
        self.regressor = None
        self.categorized=False
        
    
    def _initialize_column_map(self, column_map, original, inv=False):
        if column_map is None:
            column_map = dict(zip(original, self.df.columns))
            
            if not inv:
                return column_map
            else:
                return {v: k for k, v in column_map.items()}
        else:
            return column_map
    
    def _initialize_diversities(self, div_vars, div_min, div_ref):

        self._check_diversity()

        self.div_vars = {div: self.column_map[div_vars[div]] for div in div_vars}
        self.div_min = div_min
        self.div_ref = div_ref
        
        self.hug_vars = list(self.div_vars.values())

    def _check_diversity(self, div_vars, div_min, div_ref):

        if set(div_vars.values()).issubset(self.df.columns):
            for div in div_vars:
                minority = div_min[div]
                ref = div_ref[div]
                
                if not set([minority, ref]).issubset(self.df[div_vars[div]].tolist()):
                    self.logger.error("Minority and/or reference for '{}' not present in the df".format(div))
                    self.logger.error("Check the div_min and div_ref dictionaries")
                    raise ValueError()
        
        
        elif set(div_vars.values()).issubset(self.original_cols):
            for div in div_vars:
                minority = div_min[div]
                ref = div_ref[div]
                
                if not set([minority, ref]).issubset(self.df[self.column_map[div_vars[div]]].tolist()):
                    self.logger.error("Minority and/or reference for '{}' not present in the df".format(div))
                    self.logger.error("Check the div_min and div_ref dictionaries")
                    raise ValueError()

        else:
            self.logger.error("Some diversities in div_var are not present in df. Update the div_var dictionary")
            raise ValueError()
        
    
    def _initialize_variables(self, predictive_vars, diagnostic_vars, iter_order):
        self.predictive_vars = self.set_variables(predictive_vars, label='predictive')
        assert self.predictive_vars is not None

        self.diagnostic_vars = self._generate_diagnostic_variables(diagnostic_vars)
        self.iter_order = self.set_iter_order(iter_order)
    
    def remove_diversity_from_scope(self, diversities):
        excluded = []
        for div in diversities:
            if div in self.div_vars:
                excluded.append(self.div_vars.pop(div))
                self.logger.info("{} removed from scope of analysis".format(div))
            else:
                self.logger.error("{} not in current list of diversities. Use div_var to see list".format(div))
                raise ValueError()
        
        self.hug_vars = list(self.div_vars.values())
        if self.diagnostic_vars is not None:
            self.diagnostic_vars = [x for x in self.diagnostic_vars if x not in excluded]
            
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
    
    def _generate_diagnostic_variables(self, diagnostic_vars):
        
        self.logger.info("Diagnostic variables generated")
        if diagnostic_vars is None:
            return self.hug_vars+ self.predictive_vars
        else:
            diagnostic_vars = [self.column_map[x] if x in self.column_map else x for x in diagnostic_vars ]
            return self.set_variables(
                list(set(self.predictive_vars+diagnostic_vars+self.hug_vars)), 
                label='diagnostic'
            )
        
    def _generate_references(self, variables):

        cat_vars = [x for x in variables if x in self.categorical]
        self.references = {var: self.df[var].value_counts().index[0] for var in cat_vars}
        for div in self.div_vars:
            self.references.update({self.div_vars[div]:self.div_ref[div]})
        self.logger.debug("References generated based mode. Use set_references() for specific references")
    
    def _map_specified(self, specified):
        return {self.column_map[column]: specified[column] for column in specified}
    
    def set_references(self, specified):
        try:
            mapped_specified = self._map_specified(specified)
            
            for column in mapped_specified:
                if mapped_specified[column] in self.df[column].unique():
                    self.logger.debug("References updated.")
                    self.references.update(mapped_specified)
                else:
                    self.logger.debug("References not updated. Specified values not in dataframe.")                    
            
        except:
            self.logger.debug("References not updated.")
            self._validate_specified_references(specified)
            
    def _validate_feature(self, feature, variable):   
        if variable in self.column_map:
            new_variable = self.column_map[variable]
            
            if feature not in self.df[new_variable].unique():
                self.logger.error("Feature '{}' not in variable '{}'".format(feature, variable))
                
    def _validate_specified_references(self, specified):
        for variable in specified.keys():
            self._validate_variable(variable)
            self._validate_feature(specified[variable], variable)
    
    def _categorize_columns(self):

        numerical = self.df.select_dtypes(include=np.number).columns.tolist()
        categorical = [x for x in self.df.columns if x not in numerical+[self.eeid]]
        
        return numerical, categorical
    
    def _create_job_group_column(self): 
        if self.job_group_column not in self.df.columns:
            self.logger.error("Cannot create column: Job group variable not found in columns")
            raise ValueError()
        else:
            self.df[self.job_group] = self.df[self.job_group_column] 
    
    def _generate_variables(self, variables, label):

        if variables is None:
            self.logger.error("{} variables is not initialized. Use set_variables() to do so.".format(
                label.title()
            ))
        else:
            variables = variables if set(variables).issubset(self.df) else\
            [self.column_map[x] for x in variables]
            variables = [x for x in variables if x not in self.key_variables]
            
            if self.df[variables].isna().sum().sum()>0:
                self.logger.warn("{} initialized but NaNs detected.".format(label.title()))
                self.logger.warn("Use null_check() and fill_categorical_nas().")
            self._generate_references(variables)
            self.logger.info("{} variables validated and references generated.".format(label.title()))
            
            return variables
            
    def set_variables(self, variables, label='predictive'):
        try:
            return self._generate_variables(variables, label)
        except:
            self.logger.error("{} variables not initialized".format(label.title()))
            for variable in variables:
                self._validate_variable(variable)
                
    def _validate_variable(self, variable):
        if variable not in self.column_map:
            if variable in self.column_map_inv:
                self.logger.error("Variable '{}' not recognized. Did you mean '{}'?".format(
                    variable, 
                    self.column_map_inv[variable]
                ))
            elif variable not in self.column_map_inv:
                self.logger.error("Variable '{}' not in df columns.".format(variable))
            else:
                self.logger.error("Unrecognized error")
                                  
    def _validate_iter_order(self, iter_order):

        if iter_order is None:
            self.logger.error("iter_order is not initialized. Order set to predictive_vars.")
            self.logger.error("Use set_iter_order() to specifiy an order.")
            
            return self.predictive_vars
            
        elif self.predictive_vars is None:
            self.logger.error("predictive_vars is not initialized. Use set_variables() to do so.")
            raise ValueError()
                    
        else:
            iter_order = iter_order if set(iter_order).issubset(self.df) else \
            [self.column_map[x] for x in iter_order]
            
            exclude = self.key_variables+self.hug_vars
            iter_order = [x for x in iter_order if x not in exclude]
            
            if set(iter_order).issubset(self.predictive_vars):
            
                if self.df[iter_order].isna().sum().sum()>0:
                    self.logger.warn("iter_order initialized but NaNs detected.")
                    self.logger.warn("Use null_check() and fill_categorical_nas().")

                self.logger.info("iter_order validated and generated.")

                return iter_order
            else:
                self.logger.error("iter_order needs to be a subset of predictive_vars")
                self.logger.error("iter_order is not initialized. Order set to predictive_vars.")
                self.logger.error("Use set_iter_order() to specifiy an order.")
                
                return self.predictive_vars
    
    def set_iter_order(self, iter_order):
        try:
            return self._validate_iter_order(iter_order)
        except:
            self.logger.error("Iterative order not initialized")
            for variable in iter_order:
                self._validate_variable(variable)
        
    def _format_columns(self, df): 
        df.columns = df.columns.str.upper().str.replace('[^0-9a-zA-Z]+', '_')
        return df
    
    def _initial_null_check(self):

        na_df = self.df.isna().sum()
        na_vals = na_df.sum()
        na_df_vals = na_df[na_df>0].shape[0]
        
        if na_vals!=0:
            self.logger.warning("{} NaN values present in the data, across {} columns".format(
                na_vals, 
                na_df_vals
            ))
            self.logger.warning("Run null_check() to investigate, and fill_categorical_nas() to fill nulls")
            
        all_na_cols = [self.column_map_inv[x] for x in self.df[self.df.columns[self.df.isna().all()]].columns]
        if len(all_na_cols) > 0:
            self.df = self.df.dropna(axis=1, how='all')
            self.logger.warning("{} column(s) only contain NaN - dropped from df".format(', '.join(all_na_cols)))
            
    def null_check(self, df=None, figsize=(10,20), by=None):
        
        if df is None: df = self.df.copy()
            
        if by is None:
            return self._plot_nans(df, figsize=figsize)
        else:
            by = self.column_map[by]
            tracking_dict = {}
            for feature in df[by].unique():
                if feature != feature:
                    self.logger.warning("Cannot process NaN features")
                else:
                    tracking_dict.update({feature:self._plot_nans(
                        df[df[by]==feature], 
                        figsize=figsize, 
                        feature=feature)})
                    
            return tracking_dict
        
    def _plot_nans(self, df, figsize=(10,20), feature=None):
        
        perc_missing = df.isna().sum()/df.shape[0]
        perc_missing = perc_missing[perc_missing>0]
        perc_missing.index = perc_missing.rename(self.column_map_inv).index

        fig, ax = plt.subplots()
        perc_missing.sort_values().plot(kind="barh", figsize=figsize, ax=ax)

        ax.xaxis.tick_top()
        ax.set_xlabel("% of values as Null")
        
        if feature is None:
            ax.set_title("Missing Values per Variable\n") 
        else:
            ax.set_title("Missing Values per Variable ({})\n".format(feature)) 
            
        ax.xaxis.set_label_position('top')
        ax.minorticks_on()
        ax.grid('on', which='minor', axis='x' ,linestyle='--', linewidth='0.5', color='black', alpha=0.2)
        ax.grid('on', which='major', axis='x' ,linestyle='--', linewidth='0.5', color='black')
        ax.grid('on', which='major', axis='y' ,linestyle='--', linewidth='0.5', color='black')
        ax.xaxis.set_major_formatter(mtick.PercentFormatter(1.0))

        plt.show()
        
        return perc_missing

    def fill_categorical_nas(self, specified={}):

        try:
            specified = self._map_specified(specified)
            unspecified = {column: "Not Available" for column in self.categorical if column not in specified}

            specified.update(unspecified)
            self.df[self.categorical] = self.df[self.categorical].fillna(specified)
            self.logger.info("All categorical NAs have been filled")
        except:
            for variable in specified.keys():
                self._validate_variable(variable)

    def diversity_distributions(self, by=None, min_visible=0.05, figsize=(12, 8)):
        
        for div in self.div_vars:
            self._plot_distributions(
                var=self.div_vars[div],
                by=self.column_map[by], 
                min_visible=min_visible, 
                figsize=figsize
            )
            
    def _plot_distributions(self, var, by=None, min_visible=0.05, figsize=(12, 8)):
        
        df_plot = self.df.groupby([by,var]).size().reset_index().pivot(index=by, columns=var, values=0).fillna(0)
        df_plot = df_plot.div(df_plot.sum(axis=1), axis=0)
        ax = df_plot.plot(stacked=True, kind='bar', figsize=figsize, rot='horizontal')

        # .patches is everything inside of the chart
        for rect in ax.patches:
            # Find where everything is located
            height = rect.get_height()
            width = rect.get_width()
            x = rect.get_x()
            y = rect.get_y()

            # The height of the bar is the data value and can be used as the label
            label_text = "{:.0%}".format(height)

            # ax.text(x, y, text)
            label_x = x + width / 2
            label_y = y + height / 2

            # plot only when height is greater than specified value
            if height > min_visible:
                ax.text(label_x, label_y, label_text, ha='center', va='center', fontsize=8)

        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', biter_orderaxespad=0.)    
        ax.set_ylabel("Distribution")
        ax.set_xlabel("{}".format(self.column_map_inv[by]))
        ax.set_title("{} Distribution by {}".format(
            self.column_map_inv[var], 
            self.column_map_inv[by]
        ), fontsize=14)
        plt.show()
        
    def _create_diversity_coef(self, coef):
        hug_coef = pd.DataFrame()
        try:
            for div in self.div_vars:
                variable = self.column_map_inv[self.div_vars[div]]
                ref = self.div_min[div]
                hug_coef = pd.concat([hug_coef, coef[((coef['Variable']==variable)&(coef['Feature']==ref))]])
        except:
            self.logger.error("Cannot create diversity coefficient dataframe")
            raise ValueError()
            
        return hug_coef
    
    def _reorder_categories(self, specified={}):

        specified = self._map_specified(specified)
        all_vars = self.predictive_vars+list(self.div_vars.values())
        
        for column in [x for x in self.categorical if x in all_vars]:
            self.df[column] = self.df[column].cat.reorder_categories(
                self.df[column].value_counts().index.tolist(),ordered=True
            )
            self.logger.debug("Highest Count: Variable {} with reference {}".format(
                column, 
                self.df[column].cat.categories[0]
            ))
            
            if column in self.references.keys():
                if self.references[column] in self.df[column].unique():
                    ls = self.df[column].cat.categories.tolist()
                    ls.remove(self.references[column])
                    ls.insert(0,self.references[column])
                    self.df[column] = self.df[column].cat.reorder_categories(ls,ordered=True)
                    self.logger.debug("Generic: Variable {} with reference {}".format(
                        column, 
                        self.df[column].cat.categories[0]
                    ))
                else:
                    self.logger.warning("'{}' does not contain reference '{}'.".format(
                        column, 
                        self.references[column]
                    ))
                    self.logger.warning("Refence set to the maximum value '{}'".format(
                        self.df[column].cat.categories[0]
                    ))

            if column in specified.keys():
                if specified[column] in self.df[column].unique():
                    ls = self.df[column].cat.categories.tolist()
                    ls.remove(specified[column])
                    ls.insert(0,specified[column])
                    self.df[column] = self.df[column].cat.reorder_categories(ls,ordered=True)
                    self.logger.debug("Specified: variable '{}' with reference '{}'".format(
                        column, 
                        self.df[column].cat.categories[0]
                    ))
                else:
                    self.logger.error("Variable '{}' does not contain '{}' (in specified dictionary)".format(
                        column, 
                        specified[column]
                    ))
                    raise ValueError()
                    
            self.logger.debug("Final: variable '{}' set with reference {}".format(
                self.column_map_inv[column], 
                self.df[column].cat.categories[0]
            ))
                
    def _generate_category_reference(self):

        for column in self.categorical:
            self.references.update({column: self.df[column].cat.categories[0]}) 
    
    def _categorize_data(self, specified={}):

        try:
            self.df[self.categorical] = self.df[self.categorical].astype('category')
            self.df[self.numerical] = self.df[self.numerical].apply(pd.to_numeric)
            self._reorder_categories(specified=specified)
            self._generate_category_reference()
            self.categorized=True
        except:
            for variable in specified.keys():
                self._validate_variable(variable)
                
    def exclude(self, variables):
        try:
            variables = [self.column_map[x] for x in variables]
            self.predictive_vars = [x for x in self.predictive_vars if x not in variables]
            self.logger.info("Variables removed from predictive_vars. Update iter_order if applicable")
        except:
            for variable in variables:
                self._validate_variable(variable)
        
    def include(self, variables):
        try:
            variables = [self.column_map[x] for x in variables]
            self.predictive_vars = list(set(self.predictive_vars+variables))
            self.logger.info("Variables added to predictive_vars. Update iter_order if applicable")
        except:
            for variable in variables:
                self._validate_variable(variable)
                
    
    def _create_regressor(self):
        if self.regressor is None:
            self.regressor = Regressor(
            df=self.df,
            numerical=self.numerical,
            categorical=self.categorical,
            y=self.y,
            references=self.references,
            hug_vars=self.hug_vars,
            predictive_vars=self.predictive_vars, 
            diagnostic_vars=self.diagnostic_vars, 
            iter_order=self.iter_order,
            column_map=self.column_map, 
            column_map_inv=self.column_map_inv, 
            div_vars=self.div_vars, 
            div_min=self.div_min, 
            div_ref=self.div_ref,
            name=self.name
        )
            
    def run_regression(self):
        if not self.categorized:
            self._categorize_data()
        self._create_regressor()
        self.regressor.run_regression()

    def run_individual_remediation(self):
        self.regressor.individual_remediation()
        
    def iterative_regression(self, export=False):
        if not self.categorized:
            self._categorize_data()
        self._create_regressor()
        self.regressor.iterative_regression()

    def segment_gap(self,segment):
        return self.regressor.segment_gap(segment)
    
    
            


        
