
import json
import pandas as pd
import os
import glob
import numpy as np


summary_metric_files = glob.glob('*_synthetic*/samples_test_*.json')

metrics = pd.DataFrame()
for the_file in summary_metric_files:
    dir_name = os.path.dirname(the_file)
    num_synthetic = int(dir_name.split('_')[0])
    
    test_on = "original_test"
    
    if "customSplit" in dir_name:
        if "test_single_obj" in the_file:
            test_on = "single_obj"
        elif "test_low_score" in the_file:
            test_on = "low_score"
        else:
            test_on = "new_test_splits"
    
    file_name = os.path.basename(the_file)
    file_info = os.path.splitext(file_name)[0].split('_')
    mouse_id = file_info[2]
    model_step = int(file_info[3].replace("at", ""))
    aggregate_test = "Yes" if "aggTest" in file_info else "No"
    
    # record = [mouse_id, model_step, num_synthetic, test_on, aggregate_test]
    data = pd.read_json(the_file)
    data = data[[c for c in data.columns if "avg" not in c]]
    
    data["mouse_id"] = mouse_id
    data["step"] = model_step
    data["num_synthetic"] = num_synthetic
    data["test_on"] = test_on
    data["aggregate_test"] = aggregate_test
    data["the_file"] = the_file
    
    metrics = pd.concat([metrics, data], axis=0, ignore_index=True)
    
# remove all (avg) in column names
# metrics.columns = [c.replace(" (avg)", "") for c in metrics.columns]
metrics.to_csv("performances.csv", index=False)


image_metric_files = glob.glob('*_synthetic*/samples_test_*.txt')
metrics = pd.DataFrame()

for the_file in image_metric_files:
    dir_name = os.path.dirname(the_file)
    num_synthetic = int(dir_name.split('_')[0])
    
    test_on = "original_test"
    
    if "customSplit" in dir_name:
        if "test_single_obj" in the_file:
            test_on = "single_obj"
        elif "test_low_score" in the_file:
            test_on = "low_score"
        else:
            test_on = "new_test_splits"
    
    file_name = os.path.basename(the_file)
    file_info = os.path.splitext(file_name)[0].split('_')
    mouse_id = file_info[2]
    model_step = int(file_info[3].replace("at", ""))
    aggregate_test = "Yes" if "aggTest" in file_info else "No"
    
    if "pw" in file_info:
        metrics_type = "-".join(file_info[-2:])
    else:
        metrics_type = file_info[-1]
    
    data = pd.read_csv(the_file, sep="\t", header=None, names=["rep1", "rep2", "rep3", "rep4"])
    data["mouse_id"] = mouse_id
    data["step"] = model_step
    data["num_synthetic"] = num_synthetic
    data["test_on"] = test_on
    data["aggregate_test"] = aggregate_test
    data["metrics_type"] = metrics_type
    data["the_file"] = the_file
    
    metrics = pd.concat([metrics, data], axis=0, ignore_index=True)

metrics.to_csv("performance_image_level.csv", index=False)
