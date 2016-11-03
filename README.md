# Ensemble Learning

- Following is the structure of files and folder
	- Anisha_Gartia_HW2.zip
		- README
		- HW2.pdf - description, results, plots, and discussions
		- AdaBoost.ipynb - Jupyter notebook for Adaboost
		- RF.ipyb - Jupyter notebook for RandomForest
		- my_code
			-Adaboost.py - python code for AdaBoost
			-RF.py - python code for Random Forest

Note: The datasets have not been included as they are large in size.

-Language of code: PYTHON 3.0

-Contains independent codes for RandomForest (RF.py) and SAMME AdaBoost (AdaBoost.py)

-To execute my random forest code, you may simply run the code as it is, with the following filenames for the datasets
1. wine - wine.data
2. MNIST - train.csv , test.csv
3. Office dataset - office_train.csv , office_test.csv

-Sklearn library will be required. All the necessary imports are already present in the python code.

-To select the dataset to work on, the first three lines of code after header may be set by user in the following format.
1. for wine data set:  input_dataset = 1;
2. for MNIST data set:  input_dataset = 2;
3. for Office data set:  input_dataset = 3;

-The number of iterations (with B trees per iterations) may be set in the next line as follows:
1. iteration
2. step
This will give us iterations of i = range(0,iteration,step)

-For Random Forest, we may set the starting number of trees, and number of additional trees to be generated per iteration as follows:
1. B
2. stepb 
This will give us (B + i*stepb) where i = range(0,iteration,step) which we defined earlier.

-Execution of the python code will generate the confusion matrix, give accuracy score, generate the necessary graphs. 

-For feedback to user, the loop execution number will be displayed at the start of each loop cycle.

-For AdaBoost, a section for performing PCA has been provided. The user may uncomment this block and hence obtain reaults of SAMME AdaBoost performed on the PCA Data. NO other changes need to be made.

The ipython notebook can be viewed at   
http://nbviewer.jupyter.org/github/anishagartia/ensemble-learning/blob/master/RF.ipynb
