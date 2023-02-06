# Pay Equity Library

This purpose of this libary is to allow the organization carry out pay equity audits in a practical and sustainable manner. The library contains multiple functionalities, including regression modelling, t-test of residuals and remediation. Outlined below is guide for use of the library.

## Installation and Import

To install the library, please ensure that you have access to https://github.com/scunden/pay-equity. Once you do, you can install the library via command line, using:

```
pip install git+https://github.com/scunden/pay-equity.git
```

To import the library and check the version, use the commands below:
```
import payequity
payequity.__version__
```

## Creating Job Groups and Enssemble

### Initialization

A job group enssemble is simply a collection of job groups. The enssemble allows the user to perform actions on all of the job groups it contains - i.e., running a regression on an essemble will run a regression on all the job groups it contains. This allows for more efficient and comprehensive coding.

Below is an example of how to initialize an essemble. The fields that are specified in the instantiation are required, while those left as None or unspecicified are only optional.

```
jge = JobGroupEnssemble(
    df=df, 
    eeid='Employee ID',
    pay_component='AZBP', 
    predictive_vars=predictive_vars_bp, 
    diagnostic_vars=None, 
    iter_order=None,
    div_vars=None, 
    div_min=None, 
    div_ref=None,
    name="C&W Base Pay",
    job_group_column='Job Group'
)
```
* `df`: Dataframe  -  Employee level data. Note that once instantiated, the nomanclature of each column will be changed - all strings will be capitalized and special chars ans white spaced replaced by underscores
* `eeid`: String - Variable used to identify Employee ID
* `pay_component`: String - Variable used to identify pay component. The pay component in the data should not be log transformed.
* `predictive_vars`: List of Strings - Variables to be used in pay prediction modelling - should all be legitimate factors
* `diagnostic_vars`: List of Strings - Variables to be used in diagnostic modelling - any variable to be tested, but not included in `predictive_vars`. Do not include diversity variables.
* `iter_order`: List of Strings - This list should mirror in content `predictive_vars` but should have an intended order for the iterative anlaysis.
* `div_vars`: Dictionary - Contains all of the relevant diversities and their corresponding variables to be evaluated in this audit. More details below
* `div_min`: Dictionary - Contains all of the relevant minorities and their corresponding variables to be evaluated in this audit. More details below
* `div_ref`: Dictionary - Contains all of the relevant references and their corresponding variables to be evaluated in this audit. More details below
* `name`: String - Name of the enssemble
* `job_group_column`: String - Variable used to identify the job group column in `df`

### Specifying Diversities

If left unspecified, the three diversity dictionaries mentioned above will take the following default values:

```
div_vars = {
    'GENDER':'Gender Grouped',
    'ETHNICITY':'Ethnicity',
    'ETHNICITY_BINARY':'Ethnicity Binary',
    'AGE_GROUP':'Age Group'}

div_min = {
    'GENDER':'Female & NDT',
    'ETHNICITY':'Black or African American',
    'ETHNICITY_BINARY':'HUG',
    'AGE_GROUP':'40 & Above'}

div_ref = {
    'GENDER':'Male',
    'ETHNICITY':'White',
    'ETHNICITY_BINARY':'White',
    'AGE_GROUP':'Below 40'}
```

The default values have been built to match C&W's nomanclature and should preferably not be changed. However, if there are any changes in the data that render the default values incorrect, new values can be specified in the instantiation of the job groups. Simply use the same format for the dictionaries.

### Removing a Diversity from Scope

Note that the analysis will be carried out for all the diversities listed in `div_vars`. Use `jge.remove_diversity_from_scope([discarded_diversity])` to remove the desired diversity from the entire scope across all job groups. If the diversity only needs to be removed from select job groups, you can apply the same method on that given job group instead.

### Data Wrangling

It is highly recommended that the dataset be cleaned ahead of using the library. However, the library does allow for some wrangling.

`jge.null_check()` will allow the user to view a bar chart indicating the % of values in each categorical columns of `df` that are NaN. In order to remediate to these, the user can call `jge.fill_categorical_nas()` to fill out all categorical NaNs with "N/A" string values. Alternatively, the user can specify a dictionary as the sole argument to the previous function, specifying the null value replacement for a given column, for example: `jge.fill_categorical_nas(specified={'Worker is Manager?':'No'})`

The user can also view the headcount distribution across the diversities listed in `div_vars` using `jge.diversity_distributions(by="Business Unit)`.

<font color='red'>Need to update library to ensure all inputs can be capitalized - then specify in MD</font>

### Initializing Job Groups

One the enssemble has been created, a `JobGroup` object can be created for each job group under `job_group_column`. This can be achieve by simply calling:
```
jge.generate_job_groups()
```

Please be sure to read all INFO and WARNING logs to ensure that all the job groups are properly created.


### Variable Inclusion and Exclusion

All job groups are initialized with `predicitive_vars` as the default variables to use during modelling. However, these can be individually changed by calling the method `include` and `exclude` on a select job group. 

### Category References

When a regression is ran within a `JobGroup` object, the object will automatically set the regression reference of the categorical variable in `predictive_vars` as its mode. The `set_references()` method can be used to override that behavior for a specific job group, prior to running the regression. In order to change the reference of a variable for the whole enssemble, use the `set_overall_references()` method. Both take in a `specified` dictionary as sole parameter:

```
jge.set_overall_references(specified={"High Performer":"No"})
```