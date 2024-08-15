Author:
Katherine Markham

Publication_Date:
20240815

Title:
PythonMachineLearningScriptsForConflictInZimbabwe2024

Online_Linkage: 
Link to published manuscript forthcoming.

Abstract:
Runs multi-layer perceptron (MLP) and deep learning neural network algorithms to model where conflict occurs using presence and absence point locations. Uses 10-fold cross validation. Script can be used with any dataset containing an equal number of presences and absences (scored as 1 and 0 for conflict and non conflict). Conflict data should be in a csv file. One column (titled RAIDED in this script) indicates if conflict occurred or not. The remaining columns are model predictors, such as distance from roads. All model predictors are type float64. The output value, the column that indicates if conflict occurred or not, is type int64. Model predictors or inputs are calculated elsewhere. Refer to published manuscript for more information on methods.

Dependencies:
numpy 
pandas
matplotlib 
sklearn
Keras 

Supplemental_Information:
Other scripts related to this HEC project include machine learning algorithms in R and a Google Earth Engine script for creating model imputs.

Progress:
Complete.

Point_of_Contact:
Katherine Markham

Native_Environment:
Run using Jupyter Notebook 6.1.4 through Anaconda.Navigator on Windows 10 Enterprise, OS build 19045.4651.

Completeness_Report:
Users must provide own conflict data.
