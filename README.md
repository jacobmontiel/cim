# Cascade Imputation

This project contains the implementation of the "Cascade Imputation" described in our paper:

“Scalable Model-Based Cascaded Imputation of Missing Data”.
Montiel, Jacob, Jesse Read, Albert Bifet, and Talel Abdessalem.
In 2018 Pacific-Asia Conference on Knowledge Discovery and Data Mining (PAKDD), pp. 64-76. Springer, Cham, 2018.

### Dependencies:
* scikit-learn
* pandas

___

### Usage:
Required arguments:
- data: DataFrame to be processed D = (X,y)
- classifier: Classifier/regressor to be used for imputation. Options:

  "LR" - Logistic/Linear Regression
  
  "DT" - Decision Trees
  
  "RF" - Random Forest
  
  "ET" - Extreme Randomized Trees
  
- num_attr: List of numerical attributes
- cat_attr: List of categorical attributes

### Optional arguments
- verbose: Enable verbose debug messages
- fp_log: Pointer to log file
- target_col: Name of the target column, "target" by default

### Returns:
- DataFrame with imputed values D' = (X', y)
