# Dual-Component Deep Domain Adaptation
- Hi everyone, these are the source code samples for reproducing the experimental results in our paper "Dual-Component Deep Domain Adaptation: A New Approach for Cross Project Software Vulnerability Detection" (PAKDD-2020) https://link.springer.com/chapter/10.1007/978-3-030-47426-3_54.

## Data sets
- We use the real-world data sets, collected by Lin et al. (https://github.com/DanielLin1986/TransferRepresentationLearning), which contain the source codes of vulnerable and non-vulnerable functions obtained from five real-world software projects, namely FFmpeg, LibTIFF, LibPNG, VLC and Pidgin. These data sets cover both multimedia and image application categories. The summary statistics of these projects are shown in Table I in our paper. In our experiment, some of the data sets from the multimedia category were used as the source domain whilst other data sets from the image category were used as the target domain (see Table II, in our paper).
- We split the data of the source domain into two random partitions. The first partition contains 80% for training and the second partition contains 20% for validation. We also split the data of the target domain into two random partitions containing 80% for training without using any label information and 20% for testing the model. We additionally apply gradient clipping regularization to prevent over-fitting when training the model.

## Folder structure
-	The folder having “data_sets” in its name contains data sets used in the training and testing processes for mentioned methods in our paper. For example, we use the data set from the software projects FFmpeg (used as the source domain) and LibPNG (used as the target domain).
-	The folder having “model” in its name contains trained models for mentioned methods in our paper. For example, we save the trained models for our proposed method (dual generator-discriminator deep domain adaptation network, Dual-GD-DDAN) and the most relevant method (DDAN) proposed in [16] (i.e., Deep
Domain Adaptation For Vulnerable Code Function Identification (IJCNN-2019) https://ieeexplore.ieee.org/document/8851923) on a pair of data sets from the software projects FFmpeg and LibPNG.

## Training, validating and testing processes
-	For each model, we use the file having “train” in its name to train the model. After training the model, we can find out the best model (i.e., based on the results of AUC, Recall, Precision and F1-measure on the training set) which will be used in the testing process.
-	For each model, we use the file having “predict” in its name to test the trained model on testing set. For example, we use the trained model for Dual-GD-DDAN and the trained model for DDAN stored in the folder having “model” in its name gained after training process to obtain the result on testing set.
- In our sample source codes, to let you run and use the source codes easily, we design to train the model using the training set and compute the results on the testing set after each iteration, and summary the highest results (saved in a high_values variable) on the testing set corresponding to the used sets of hyper-parameters. To Dual-GD-DDAN method, you can use dual_dan.py file or preserved_codes/dual_dan_HG.py file. To DDAN method, you can use dan.py file.

## Implementation
- We implemented all mentioned methods in Python using Tensorflow (version 1.14), an opensource software library for Machine Intelligence developed by the Google Brain Team, and Python 3.6. We ran our experiments on an Intel Xeon Processor E5-1660 which has 8 cores at 3.0 GHz and 128 GB of RAM.

## Additional reading about data sets
- We use the real-world data sets, collected by Lin et al. (https://github.com/DanielLin1986/TransferRepresentationLearning), which contain the source codes of vulnerable and non-vulnerable functions obtained from five real-world software projects, namely FFmpeg, LibTIFF, LibPNG, VLC and Pidgin. These datasets cover both multimedia and image application categories. 
- We preprocess data sets before inputting into the deep neural networks (i.e., baselines and our proposed method). Firstly, we standardize the source codes by removing comments, blank lines and non-ASCII characters. Secondly, we map user-defined variables to symbolic names (e.g., “var1”, “var2”) and user-defined functions to symbolic names (e.g., “func1”, “func2”). We also replace integers, real and hexadecimal numbers with a generic "num" token and strings with a generic "str" token. We use https://joern.readthedocs.io/en/latest/ to analyze the source codes to get user-defined variables and functions.
- We observe that to source codes, vulnerabilities are almost relevant to variables (i.e., local variables or parameters's variables), so that to reduce the length of the source codes as well as remove unimportant code statements, we proposed to keep code statements having variables. We name this type data set as "data_sets_gadget". We have two types of data sets, the first one contains all code statements in functions to obtain "data_sets_full" and the second one contains only statements having variables in functions to obtain "data_sets_gadget".

## Citation
If you use our source code samples and our processed data in your work, please kindly cite our paper.

> @article{van-nguyen-dual-dan-2020,<br/>
  title={Dual-Component Deep Domain Adaptation: A New Approach for Cross Project Software Vulnerability Detection},<br/>
  author={Nguyen, Van and Le, Trung and De Vel, Olivier and Montague, Paul and Grundy, John and Phung, Dinh},<br/>
  booktitle={Pacific-Asia Conference on Knowledge Discovery and Data Mining},<br/>
  year={2020}}

