数据准备：训练数据需要整理成 data——external_id——group_id 三级文件夹，group_id 内有 ECG PPG ACC txt
	   删除group_id 小于30个的external_id，保证后续构造正负对样本充足

测试结果复现：
Step1: 将 "a01_ECG.py" 置于 data 同级文件夹下，运行代码，生成 ECG 特征

Step2: 将 "a02_PPG.py" 置于 data 同级文件夹下，运行代码，生成 PPG 特征

Step3: 将 "b01_feature_statistics.py" 置于 data 同级文件夹下，运行代码，生成 ecg_vectors 与 ppg_vectors

Step4: 运行 "b02_pair_features.py"，生成 ecg_scaler ecg_xgb ppg_scaler ppg_xgb

Step5: 将 "b03_executable_scripts.py" 和 ecg_scaler ecg_xgb ppg_scaler ppg_xgb 放置在 registration_data 和 test_data 同级文件夹下，运行代码，生成测试结果.xlsx