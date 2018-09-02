# All combinations of parameters

# Default setting:
# batch_size = 10
# learning_rate = 0.001
# balance_data = True

# Stopping condition
./train_test_dgsm_full_model.py DGSM_SAME_BUILDING -d Stockholm \
				--config "{'test_case':'456-7', 'category_type':'SIMPLE', 'training_params': '--batch-size 200 --learning-rate 0.001 --update-threshold 0.00001 --balance-data'}"\
				--save-loss

./train_test_dgsm_full_model.py DGSM_SAME_BUILDING -d Stockholm \
				--config "{'test_case':'457-6', 'category_type':'SIMPLE', 'training_params': '--batch-size 200 --learning-rate 0.001 --update-threshold 0.00001 --balance-data'}"\
				--save-loss

./train_test_dgsm_full_model.py DGSM_SAME_BUILDING -d Stockholm \
				--config "{'test_case':'456-7', 'category_type':'SIMPLE', 'training_params': '--batch-size 200 --learning-rate 0.01 --update-threshold 0.00001 --balance-data'}"\
				--save-loss

./train_test_dgsm_full_model.py DGSM_SAME_BUILDING -d Stockholm \
				--config "{'test_case':'457-6', 'category_type':'SIMPLE', 'training_params': '--batch-size 200 --learning-rate 0.01 --update-threshold 0.00001 --balance-data'}"\
				--save-loss

./train_test_dgsm_full_model.py DGSM_SAME_BUILDING -d Stockholm \
				--config "{'test_case':'456-7', 'category_type':'SIMPLE', 'training_params': '--batch-size 10 --learning-rate 0.001 --update-threshold 0.00001 --balance-data'}"\
				--save-loss

./train_test_dgsm_full_model.py DGSM_SAME_BUILDING -d Stockholm \
				--config "{'test_case':'457-6', 'category_type':'SIMPLE', 'training_params': '--batch-size 10 --learning-rate 0.001 --update-threshold 0.00001 --balance-data'}"\
				--save-loss

./train_test_dgsm_full_model.py DGSM_SAME_BUILDING -d Stockholm \
				--config "{'test_case':'456-7', 'category_type':'SIMPLE', 'training_params': '--batch-size 10 --learning-rate 0.01 --update-threshold 0.00001 --balance-data'}"\
				--save-loss

./train_test_dgsm_full_model.py DGSM_SAME_BUILDING -d Stockholm \
				--config "{'test_case':'457-6', 'category_type':'SIMPLE', 'training_params': '--batch-size 10 --learning-rate 0.01 --update-threshold 0.00001 --balance-data'}"\
				--save-loss


# More learning rate
./train_test_dgsm_full_model.py DGSM_SAME_BUILDING -d Stockholm \
				--config "{'test_case':'457-6', 'category_type':'SIMPLE', 'training_params': '--batch-size 200 --learning-rate 0.01 --balance-data --value-inference MPE'}"\
				--save-loss

./train_test_dgsm_full_model.py DGSM_SAME_BUILDING -d Stockholm \
				--config "{'test_case':'456-7', 'category_type':'SIMPLE', 'training_params': '--batch-size 200 --learning-rate 0.01 --balance-data --value-inference MPE'}"\
				--save-loss


./train_test_dgsm_full_model.py DGSM_SAME_BUILDING -d Stockholm \
				--config "{'test_case':'456-7', 'category_type':'SIMPLE', 'training_params': '--batch-size 200 --learning-rate 0.00001 --balance-data '}"\
				--save-loss


./train_test_dgsm_full_model.py DGSM_SAME_BUILDING -d Stockholm \
				--config "{'test_case':'457-6', 'category_type':'SIMPLE', 'training_params': '--batch-size 200 --learning-rate 0.00001 --balance-data '}"\
				--save-loss


./train_test_dgsm_full_model.py DGSM_SAME_BUILDING -d Stockholm \
				--config "{'test_case':'456-7', 'category_type':'SIMPLE', 'training_params': '--batch-size 200 --learning-rate 0.001 --balance-data --value-inference MPE'}"\
				--save-loss

./train_test_dgsm_full_model.py DGSM_SAME_BUILDING -d Stockholm \
				--config "{'test_case':'457-6', 'category_type':'SIMPLE', 'training_params': '--batch-size 200 --learning-rate 0.001 --balance-data '}"\
				--save-loss



# MPE
./train_test_dgsm_full_model.py DGSM_SAME_BUILDING -d Stockholm \
				--config "{'test_case':'457-6', 'category_type':'SIMPLE', 'training_params': '--batch-size 10 --learning-rate 0.001 --value-inference mpe --balance-data'}"\
				--save-loss

./train_test_dgsm_full_model.py DGSM_SAME_BUILDING -d Stockholm \
				--config "{'test_case':'456-7', 'category_type':'SIMPLE', 'training_params': '--batch-size 10 --learning-rate 0.001 --value-inference mpe --balance-data'}"\
				--save-loss

./train_test_dgsm_full_model.py DGSM_SAME_BUILDING -d Stockholm \
				--config "{'test_case':'457-6', 'category_type':'SIMPLE', 'training_params': '--batch-size 200 --learning-rate 0.001 --value-inference mpe --balance-data'}"\
				--save-loss

./train_test_dgsm_full_model.py DGSM_SAME_BUILDING -d Stockholm \
				--config "{'test_case':'456-7', 'category_type':'SIMPLE', 'training_params': '--batch-size 200 --learning-rate 0.001 --value-inference mpe --balance-data'}"\
				--save-loss

# Batch size

./train_test_dgsm_full_model.py DGSM_SAME_BUILDING -d Stockholm \
				--config "{'test_case':'456-7', 'category_type':'SIMPLE', 'training_params': '--batch-size 10 --learning-rate 0.001 --balance-data'}"\
				--save-loss

./train_test_dgsm_full_model.py DGSM_SAME_BUILDING -d Stockholm \
				--config "{'test_case':'456-7', 'category_type':'SIMPLE', 'training_params': '--batch-size 10 --learning-rate 0.00001 --balance-data'}"\
				--save-loss

./train_test_dgsm_full_model.py DGSM_SAME_BUILDING -d Stockholm \
				--config "{'test_case':'457-6', 'category_type':'SIMPLE', 'training_params': '--batch-size 10 --learning-rate 0.001 --balance-data'}"\
				--save-loss

./train_test_dgsm_full_model.py DGSM_SAME_BUILDING -d Stockholm \
				--config "{'test_case':'457-6', 'category_type':'SIMPLE', 'training_params': '--batch-size 10 --learning-rate 0.00001 --balance-data'}"\
				--save-loss

