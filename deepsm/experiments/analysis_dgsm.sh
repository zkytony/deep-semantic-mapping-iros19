# All combinations of parameters

# Default setting:
# batch_size = 10
# learning_rate = 0.001
# balance_data = True

# Balanced data versus not balanced
./train_test_dgsm_full_model.py DGSM_SAME_BUILDING -d Stockholm \
				--config "{'test_case':'456-7', 'category_type':'SIMPLE', 'training_params': '--batch-size 10 --learning-rate 0.001 --balance-data'}"\
				--name balanced_default_Stockholm_456-7

./train_test_dgsm_full_model.py DGSM_SAME_BUILDING -d Stockholm \
				--config "{'test_case':'456-7', 'category_type':'SIMPLE', 'training_params': '--batch-size 10 --learning-rate 0.001'}"\
				--name unbalanced_default_Stockholm_456-7

# batch size 10
# Learning rate - Stockholm
./train_test_dgsm_full_model.py DGSM_SAME_BUILDING -d Stockholm \
				--config "{'test_case':'456-7', 'category_type':'SIMPLE', 'training_params': '--batch-size 10 --learning-rate 0.01 --balance-data'}"\
				--name balanced_lr01_b10_Stockholm_456-7
./train_test_dgsm_full_model.py DGSM_SAME_BUILDING -d Stockholm \
				--config "{'test_case':'456-7', 'category_type':'SIMPLE', 'training_params': '--batch-size 10 --learning-rate 0.001 --balance-data'}"\
				--name balanced_lr001_b10_Stockholm_456-7
./train_test_dgsm_full_model.py DGSM_SAME_BUILDING -d Stockholm \
				--config "{'test_case':'456-7', 'category_type':'SIMPLE', 'training_params': '--batch-size 10 --learning-rate 0.0001 --balance-data'}"\
				--name balanced_lr0001_b10_Stockholm_456-7

# Learning rate - Freiburg
./train_test_dgsm_full_model.py DGSM_SAME_BUILDING -d Freiburg \
				--config "{'test_case':'12-3', 'category_type':'SIMPLE', 'training_params': '--batch-size 10 --learning-rate 0.01 --balance-data'}"\
				--name balanced_lr01_b10_Freiburg_12-3
./train_test_dgsm_full_model.py DGSM_SAME_BUILDING -d Freiburg \
				--config "{'test_case':'12-3', 'category_type':'SIMPLE', 'training_params': '--batch-size 10 --learning-rate 0.001 --balance-data'}"\
				--name balanced_lr001_b10_Freiburg_12-3
./train_test_dgsm_full_model.py DGSM_SAME_BUILDING -d Freiburg \
				--config "{'test_case':'12-3', 'category_type':'SIMPLE', 'training_params': '--batch-size 10 --learning-rate 0.0001 --balance-data'}"\
				--name balanced_lr0001_b10_Freiburg_12-3

# batch size 50
# Learning rate - Stockholm
./train_test_dgsm_full_model.py DGSM_SAME_BUILDING -d Stockholm \
				--config "{'test_case':'456-7', 'category_type':'SIMPLE', 'training_params': '--batch-size 50 --learning-rate 0.01 --balance-data'}"\
				--name balanced_lr01_b50_Stockholm_456-7
./train_test_dgsm_full_model.py DGSM_SAME_BUILDING -d Stockholm \
				--config "{'test_case':'456-7', 'category_type':'SIMPLE', 'training_params': '--batch-size 50 --learning-rate 0.001 --balance-data'}"\
				--name balanced_lr001_b50_Stockholm_456-7
./train_test_dgsm_full_model.py DGSM_SAME_BUILDING -d Stockholm \
				--config "{'test_case':'456-7', 'category_type':'SIMPLE', 'training_params': '--batch-size 50 --learning-rate 0.0001 --balance-data'}"\
				--name balanced_lr0001_b50_Stockholm_456-7

# Learning rate - Freiburg
./train_test_dgsm_full_model.py DGSM_SAME_BUILDING -d Freiburg \
				--config "{'test_case':'12-3', 'category_type':'SIMPLE', 'training_params': '--batch-size 50 --learning-rate 0.01 --balance-data'}"\
				--name balanced_lr01_b50_Freiburg_12-3
./train_test_dgsm_full_model.py DGSM_SAME_BUILDING -d Freiburg \
				--config "{'test_case':'12-3', 'category_type':'SIMPLE', 'training_params': '--batch-size 50 --learning-rate 0.001 --balance-data'}"\
				--name balanced_lr001_b50_Freiburg_12-3
./train_test_dgsm_full_model.py DGSM_SAME_BUILDING -d Freiburg \
				--config "{'test_case':'12-3', 'category_type':'SIMPLE', 'training_params': '--batch-size 50 --learning-rate 0.0001 --balance-data'}"\
				--name balanced_lr0001_b50_Freiburg_12-3

# batch size 100
# Learning rate - Stockholm
./train_test_dgsm_full_model.py DGSM_SAME_BUILDING -d Stockholm \
				--config "{'test_case':'456-7', 'category_type':'SIMPLE', 'training_params': '--batch-size 100 --learning-rate 0.01 --balance-data'}"\
				--name balanced_lr01_b100_Stockholm_456-7
./train_test_dgsm_full_model.py DGSM_SAME_BUILDING -d Stockholm \
				--config "{'test_case':'456-7', 'category_type':'SIMPLE', 'training_params': '--batch-size 100 --learning-rate 0.001 --balance-data'}"\
				--name balanced_lr001_b100_Stockholm_456-7
./train_test_dgsm_full_model.py DGSM_SAME_BUILDING -d Stockholm \
				--config "{'test_case':'456-7', 'category_type':'SIMPLE', 'training_params': '--batch-size 100 --learning-rate 0.0001 --balance-data'}"\
				--name balanced_lr0001_b100_Stockholm_456-7

# Learning rate - Freiburg
./train_test_dgsm_full_model.py DGSM_SAME_BUILDING -d Freiburg \
				--config "{'test_case':'12-3', 'category_type':'SIMPLE', 'training_params': '--batch-size 100 --learning-rate 0.01 --balance-data'}"\
				--name balanced_lr01_b100_Freiburg_12-3
./train_test_dgsm_full_model.py DGSM_SAME_BUILDING -d Freiburg \
				--config "{'test_case':'12-3', 'category_type':'SIMPLE', 'training_params': '--batch-size 100 --learning-rate 0.001 --balance-data'}"\
				--name balanced_lr001_b100_Freiburg_12-3
./train_test_dgsm_full_model.py DGSM_SAME_BUILDING -d Freiburg \
				--config "{'test_case':'12-3', 'category_type':'SIMPLE', 'training_params': '--batch-size 100 --learning-rate 0.0001 --balance-data'}"\
				--name balanced_lr0001_b100_Freiburg_12-3



