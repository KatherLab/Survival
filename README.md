# EE-Surv

This repository contains the scripts for the End-to-End Survival prediction from histopathological images.
The main script to run the survival is Survival_Main.py and the following table shows the description for 
the main required variables for the script:

Input Variable name | Description
--- | --- 
-datadir_train | Path to the normalized tiles | 
-slide_dir | Path to the slide table, containing the name of patients and their corresponding slides
-clini_dir | Path to the clini table, containing the name of patients and their corresponding clinical data
-outputPath | Path to the folder to save the results
-checkPointName | Name which will be used to save the trained model
-evenCol | The name of event column in the clini table (OS\Death\...)
-timeCol | The name of time column in the clini table (OS_time\follow_up\...)
-timeInDays | if the measured follow-up time is in days, this variable should be defined as True. 
-patientCol | The name of patient column in the clini table (PATIENT\ID\...)
-trainFull | If the aim is to use all the patients to train a single model, this variable should be defined as True.

If the variable -trainFull in the training  script is defined as True, then the Survival_Deploy script can be used to evaluate the
performance of the model on a new data set.



