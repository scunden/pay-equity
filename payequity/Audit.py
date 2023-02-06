import pandas as pd
import logging
from datetime import date
import string
import pandas.io.formats.excel
pandas.io.formats.excel.ExcelFormatter.header_style = None

class Audit():
#     def __str__(self):
#         pass
    
#     def __repr__(self):
#         return "<Job Group Object> {} | Headcount: {}".format(self.name, self.df.shape[0])
    
    def __init__(
        self, 
        regressors,
        div_vars=None, 
        div_min=None,
        column_map_inv=None,
        remediated=False,
        name="Company"):
        
        self.name=name
        self.logger = self._get_logger(name=self.name+" Audit")
        self.div_vars=div_vars
        self.div_min=div_min
        self.column_map_inv=column_map_inv
        self.regressors = regressors
        self.remediated = remediated
        self.logger.info("Audit initialized")
        self._compile_regressions()

        self.df = pd.concat([reg.df for reg in self.regressors])
        self.df = self.df[[x for x in self.df.columns if x in self.column_map_inv]]
        self.df.columns = [self.column_map_inv[x] for x in self.df.columns]

        self.eeid = self.regressors[0].eeid
        self.eeid_original = self.column_map_inv[self.eeid]


        if self.remediated:
            self._compile_remediation_summary()
        self._summary()
    
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
                    
    def _create_diversity_coef(self, coef):
        hug_coef = pd.DataFrame()
        try:
            for div in self.div_vars:
                variable = self.column_map_inv[self.div_vars[div]]
                hug_coef = pd.concat([hug_coef, coef[((coef['Variable']==variable))]])
        except:
            self.logger.error("Cannot create diversity coefficient dataframe")
            
        return hug_coef
    
    def _compile_regressions(self):
        diagnostic_coef = pd.DataFrame()
        predictive_coef = pd.DataFrame()
        
        diagnostic_corr = pd.DataFrame()
        predictive_corr = pd.DataFrame()
        
        predictive_dof = pd.DataFrame()
        
        predictions = pd.DataFrame()
        
        for regressor in self.regressors:
            diagnostic_coef = pd.concat([diagnostic_coef, regressor.diagnostic_coef])
            predictive_coef = pd.concat([predictive_coef, regressor.predictive_coef])
            
            diagnostic_corr = pd.concat([diagnostic_corr, regressor.diagnostic_corr])
            predictive_corr = pd.concat([predictive_corr, regressor.predictive_corr])
            
            predictive_dof = pd.concat([predictive_dof, regressor.predictive_dof])
            
            predictions = pd.concat([predictions, regressor.predictions])
            
        self.diagnostic_coef = diagnostic_coef.reset_index(drop=True)
        self.predictive_coef = predictive_coef.reset_index(drop=True)
        self.predictive_corr = predictive_corr.reset_index(drop=True)
        self.predictive_dof = predictive_dof.reset_index().rename({'index':'Job Group'}, axis=1)
        self.hug_coef = self._create_diversity_coef(diagnostic_coef)
        self.predictions = predictions.reset_index(drop=True)
        self.outlier_summary = self._summarize_outliers(self.predictions)
        self.logger.info("Audit updated with compiled regressors outputs")
            
    def _compile_iter_regressions(self):
        self.iterative_coef = pd.DataFrame()
        
        for regressor in self.regressors:
            try:
                coef = regressor.iterative_coef
                coef = coef.loc[coef['Variable'].isin([self.column_map_inv[x] for x in self.div_vars.values()])]
                self.iterative_coef = pd.concat([self.iterative_coef, coef])
            except:
                self.logger.warn("No iterative regression detected for {}".format(regressor.name))
        
    def _export_iter_regressions(self, version=""):
        self._compile_iter_regressions()
        writer = pd.ExcelWriter('Iterative Coef {}.xlsx'.format(version), engine='xlsxwriter')
        workbook = writer.book

        header_fmt = workbook.add_format({"bg_color": "#ed1b2c","bold": True,'font_color': '#FFFFFF','border': 1})
        default_fmt_dict = {"bg_color":"#FFFFFF", 'border': 1}
        default_fmt = workbook.add_format(default_fmt_dict)

        perc_fmt = workbook.add_format(dict(list({'num_format':'0.0%'}.items())+ list(default_fmt_dict.items())))
        dec_fmt = workbook.add_format(dict(list({'num_format':'#,##0.000'}.items())+ list(default_fmt_dict.items())))

        self.iterative_coef.to_excel(writer, index=False, sheet_name="Iter") 
        fmt = {  "P>|t|":dec_fmt, "Exp. Coef.":perc_fmt,"Default":default_fmt,"Header":header_fmt,}
        self._format_worksheet(writer.sheets["Iter"],self.iterative_coef, fmt)
            
        writer.save()

    
    def _summarize_outliers(self, predictions):
        summary = predictions.groupby(['Job Group']).agg({
            'Prediction':'count',
            'Lower Outlier':'sum',
            'Upper Outlier':'sum',
            'Outlier':'sum'}).reset_index().rename({'Prediction':'Headcount'}, axis=1)
        outlier_cols = [x for x in summary.columns if 'Outlier' in x]
        summary[[x+' Rate' for x in outlier_cols]] = summary[outlier_cols].div(summary.Headcount, axis=0)
        
        return summary

    def _compile_remediation_summary(self):
        self.remediation_summary = pd.DataFrame()
        for regressor in self.regressors:
            self.remediation_summary = pd.concat([self.remediation_summary, regressor.remediation_summary])

        self.remediation_summary = self.remediation_summary.sort_values(['Job Group','Variable','Scenario'])

        self.logger.info("Audit updated with compiled remediation summary")
        
    def _summary(self):
        summary = self.hug_coef.copy()
        summary = summary[summary['Feature'].isin(self.div_min.values())]
        summary['Feature'] = summary['Feature'] + ' vs. '+ summary['Reference']
        summary =pd.pivot_table(summary, 
                                         values = ['Exp. Coef.','P>|t|'], 
                                         index=['Job Group'], 
                                         columns = 'Feature').reset_index()
        
        summary.columns = [x[0]+'~'+x[1] if x[0]!="Job Group" else x[0] for x in summary.columns]
        summary.set_index(['Job Group'], inplace=True)

        sig_cols = [x for x in summary.columns if 'P>|t|' in x]

        summary.columns = [x.split('~')[1]+' Gap' if 'Coef' in x else x.split('~')[1]+' Sig.' for x in\
                           summary.columns]
        summary = summary.reindex(sorted(summary.columns), axis=1)
        
        summary.reset_index(inplace=True)
        summary = summary.merge(self.outlier_summary, how='left', on='Job Group')
        summary = self.predictive_dof[['Job Group','Observations','Ratio','MAPE','R2']].merge(summary,on='Job Group')
        exclude = ['Headcount','Outlier','Lower Outlier','Upper Outlier']

        summary["Model"]=self.name
        self.summary = summary[[x for x in summary.columns if x not in exclude]].rename({'Ratio':'DoF Ratio'},
                                                                                        axis=1
                                                                                       )

        self.logger.info("Audit updated with compiled summary")

    def export_all(self, version="", diagnostic=True, predictions=True, iter=False):

        version = date.today().strftime("%d.%m.%Y")+version
        writer = pd.ExcelWriter('{} Pay Equity Audit {}.xlsx'.format(self.name, version), engine='xlsxwriter')
        workbook = writer.book

        header_fmt = workbook.add_format({"bg_color": "#ed1b2c","bold": True,'font_color': '#FFFFFF','border': 1})
        default_fmt_dict = {"bg_color":"#FFFFFF", 'border': 1}
        default_fmt = workbook.add_format(default_fmt_dict)
        
        money_fmt = workbook.add_format(dict(list({'num_format':'$#,##0'}.items())+ list(default_fmt_dict.items())))
        perc_fmt = workbook.add_format(dict(list({'num_format':'0.0%'}.items())+ list(default_fmt_dict.items())))
        dec_fmt = workbook.add_format(dict(list({'num_format':'#,##0.000'}.items())+ list(default_fmt_dict.items())))
        int_fmt = workbook.add_format(dict(list({'num_format':'#,##0'}.items())+ list(default_fmt_dict.items())))

        df_dict = {
            "Gap Summary":self.summary,
            "Remediation Summary":self.remediation_summary if  self.remediated else pd.DataFrame(),
            "Diagnostic Coef.":self.diagnostic_coef if diagnostic else pd.DataFrame(),
            "Predictive Coef.":self.predictive_coef,
            "Model Performance":self.predictive_dof,
            "Correlations":self.predictive_corr,
            "Predictions":self.df.merge(self.predictions, left_on=self.eeid_original, right_on=self.eeid, how='left') if predictions else pd.DataFrame(),
        }

        summary_fmt = {
            "Default":default_fmt,
            "Header":header_fmt,
            "DoF Ratio":dec_fmt,
            "Observations":int_fmt
            }
        summary_fmt.update({x: perc_fmt for x in self.summary.columns if "Gap" in x or "Rate" in x})
        summary_fmt.update({x: dec_fmt for x in self.summary.columns if ("Sig" in x)&("?" not in x)})

        remediation_fmt = {
            "Default":default_fmt,
            "Header":header_fmt,
            "Total Adjusted":int_fmt,
            "Total Pay Adjustments":money_fmt,
            "Avg. Pay Adjustments":money_fmt,
            "Max Pay Adjustments":money_fmt,
            "Exp. Coef.":perc_fmt,
            "P>|t|":dec_fmt,
            "VIF":dec_fmt,
            "Headcount":int_fmt
            }

        diagnostic_fmt = {
            "Default":default_fmt,
            "Header":header_fmt,
            "Exp. Coef.":perc_fmt,
            "P>|t|":dec_fmt,
            "VIF":dec_fmt,
            "Headcount":int_fmt
            }
        
        predictive_fmt = {
            "Default":default_fmt,
            "Header":header_fmt,
            "Exp. Coef.":perc_fmt,
            "P>|t|":dec_fmt,
            "VIF":dec_fmt,
            "Headcount":int_fmt
            }

        performance_fmt = {
            "Default":default_fmt,
            "Header":header_fmt,

            "Observations":int_fmt,
            "Features":int_fmt,
            "DoF":int_fmt,
            "Ratio":dec_fmt,
            "R2":perc_fmt,
            "Adj. R2":perc_fmt,
            "MAPE":perc_fmt,
            }

        corr_fmt = {
            "Default":default_fmt,
            "Header":header_fmt,

            "Correlation":perc_fmt
            }

        pred_fmt = {
            "Default":default_fmt,
            "Header":header_fmt
            }
        pred_fmt.update({x: money_fmt for x in self.predictions.columns if "Pay" in x})
        pred_fmt.update({x: money_fmt for x in self.predictions.columns if x in ["Prediction","Upper","Lower","Actual"]})
        pred_fmt.update({x: int_fmt for x in self.predictions.columns if "?" in x})

        sheet_format = {
            "Gap Summary":summary_fmt,
            "Remediation Summary":remediation_fmt,
            "Diagnostic Coef.":diagnostic_fmt,
            "Predictive Coef.":predictive_fmt,
            "Model Performance":performance_fmt,
            "Correlations":corr_fmt,
            "Predictions":pred_fmt,
        }
        
        for sheet in df_dict:
            # print(sheet)
            df_dict[sheet].to_excel(writer, index=False, sheet_name=sheet) 
            worksheet = writer.sheets[sheet]
            self._format_worksheet(worksheet, df_dict[sheet], sheet_format[sheet])
            
        
        writer.close()
        if iter:
            self._export_iter_regressions(version=version)

    def _format_worksheet(self, worksheet, df, format_dict):

        last_idx = 0
        for idx, col in enumerate(df):  # loop through all columns
            series = df[col]
            
            max_len = max((
                series.astype(str).map(len).max(),  # len of largest item
                len(str(series.name))  # len of column name/header
                )) + 1  # adding a little extra space
            
            if col in format_dict:
                worksheet.set_column(idx, idx, max_len, format_dict[col])
            else:
                worksheet.set_column(idx, idx, max_len, format_dict["Default"])
                
            last_idx = idx+1
                
        worksheet.set_default_row(hide_unused_rows=True)

        q, r = divmod(last_idx, 26)
        if last_idx<=25:
            idx = string.ascii_uppercase[last_idx]
        else:
            idx = string.ascii_uppercase[q-1]+string.ascii_uppercase[r]

        worksheet.set_column('{}:XFD'.format(idx), None, None, {'hidden': True})
        worksheet.set_row(0, None, format_dict["Header"])


        


        

