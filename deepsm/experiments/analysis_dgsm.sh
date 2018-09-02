# All combinations of parameters

# Default setting:
# batch_size = 10
# learning_rate = 0.001
# balance_data = True

# Stopping condition
./train_test_dgsm_full_model.py DGSM_SAME_BUILDING -d Stockholm \
				--config "{'test_case':'456-7', 'category_type':'SIMPLE', 'training_params': '--batch-size 200 --learning-rate 0.001 --update-threshold 0.00001 --balance-data'}"\
				--name balanced_lr001_b200_uc00001_Stockholm_456-7 \
				--save-loss

./train_test_dgsm_full_model.py DGSM_SAME_BUILDING -d Stockholm \
				--config "{'test_case':'457-6', 'category_type':'SIMPLE', 'training_params': '--batch-size 200 --learning-rate 0.001 --update-threshold 0.00001 --balance-data'}"\
				--name balanced_lr001_b200_uc00001_Stockholm_457-6 \
				--save-loss

./train_test_dgsm_full_model.py DGSM_SAME_BUILDING -d Stockholm \
				--config "{'test_case':'456-7', 'category_type':'SIMPLE', 'training_params': '--batch-size 200 --learning-rate 0.01 --update-threshold 0.00001 --balance-data'}"\
				--name balanced_lr01_b200_uc0001_Stockholm_456-7 \
				--save-loss

./train_test_dgsm_full_model.py DGSM_SAME_BUILDING -d Stockholm \
				--config "{'test_case':'457-6', 'category_type':'SIMPLE', 'training_params': '--batch-size 200 --learning-rate 0.01 --update-threshold 0.00001 --balance-data'}"\
				--name balanced_lr01_b200_uc0001_Stockholm_457-6 \
				--save-loss

./train_test_dgsm_full_model.py DGSM_SAME_BUILDING -d Stockholm \
				--config "{'test_case':'456-7', 'category_type':'SIMPLE', 'training_params': '--batch-size 10 --learning-rate 0.001 --update-threshold 0.00001 --balance-data'}"\
				--name balanced_lr001_b10_uc00001_Stockholm_456-7 \
				--save-loss

./train_test_dgsm_full_model.py DGSM_SAME_BUILDING -d Stockholm \
				--config "{'test_case':'457-6', 'category_type':'SIMPLE', 'training_params': '--batch-size 10 --learning-rate 0.001 --update-threshold 0.00001 --balance-data'}"\
				--name balanced_lr001_b10_uc00001_Stockholm_457-6 \
				--save-loss

./train_test_dgsm_full_model.py DGSM_SAME_BUILDING -d Stockholm \
				--config "{'test_case':'456-7', 'category_type':'SIMPLE', 'training_params': '--batch-size 10 --learning-rate 0.01 --update-threshold 0.00001 --balance-data'}"\
				--name balanced_lr01_b10_uc00001_Stockholm_456-7 \
				--save-loss

./train_test_dgsm_full_model.py DGSM_SAME_BUILDING -d Stockholm \
				--config "{'test_case':'457-6', 'category_type':'SIMPLE', 'training_params': '--batch-size 10 --learning-rate 0.01 --update-threshold 0.00001 --balance-data'}"\
				--name balanced_lr01_b10_uc00001_Stockholm_457-6 \
				--save-loss



# # More learning rate
# ./train_test_dgsm_full_model.py DGSM_SAME_BUILDING -d Stockholm \
# 				--config "{'test_case':'457-6', 'category_type':'SIMPLE', 'training_params': '--batch-size 200 --learning-rate 0.01 --balance-data --value-inference MPE'}"\
# 				--name balanced_lr01_b200_mpe_Stockholm_457-6 \
# 				--save-loss

# ./train_test_dgsm_full_model.py DGSM_SAME_BUILDING -d Stockholm \
# 				--config "{'test_case':'456-7', 'category_type':'SIMPLE', 'training_params': '--batch-size 200 --learning-rate 0.01 --balance-data --value-inference MPE'}"\
# 				--name balanced_lr01_b200_mpe_Stockholm_456-7 \
# 				--save-loss


# ./train_test_dgsm_full_model.py DGSM_SAME_BUILDING -d Stockholm \
# 				--config "{'test_case':'456-7', 'category_type':'SIMPLE', 'training_params': '--batch-size 200 --learning-rate 0.00001 --balance-data '}"\
# 				--name balanced_lr00001_b200_Stockholm_456-7 \
# 				--save-loss


# ./train_test_dgsm_full_model.py DGSM_SAME_BUILDING -d Stockholm \
# 				--config "{'test_case':'457-6', 'category_type':'SIMPLE', 'training_params': '--batch-size 200 --learning-rate 0.00001 --balance-data '}"\
# 				--name balanced_lr00001_b200_Stockholm_457-6 \
# 				--save-loss


# ./train_test_dgsm_full_model.py DGSM_SAME_BUILDING -d Stockholm \
# 				--config "{'test_case':'456-7', 'category_type':'SIMPLE', 'training_params': '--batch-size 200 --learning-rate 0.001 --balance-data --value-inference MPE'}"\
# 				--name balanced_lr001_b200_mpe_Stockholm_456-7 \
# 				--save-loss

# ./train_test_dgsm_full_model.py DGSM_SAME_BUILDING -d Stockholm \
# 				--config "{'test_case':'457-6', 'category_type':'SIMPLE', 'training_params': '--batch-size 200 --learning-rate 0.001 --balance-data '}"\
# 				--name balanced_lr001_b200_Stockholm_457-6 \
# 				--save-loss



# # MPE
# ./train_test_dgsm_full_model.py DGSM_SAME_BUILDING -d Stockholm \
# 				--config "{'test_case':'457-6', 'category_type':'SIMPLE', 'training_params': '--batch-size 10 --learning-rate 0.001 --value-inference mpe --balance-data'}"\
# 				--name balanced_lr001_b10_mpe_Stockholm_457-6 \
# 				--save-loss

# ./train_test_dgsm_full_model.py DGSM_SAME_BUILDING -d Stockholm \
# 				--config "{'test_case':'456-7', 'category_type':'SIMPLE', 'training_params': '--batch-size 10 --learning-rate 0.001 --value-inference mpe --balance-data'}"\
# 				--name balanced_lr001_b10_mpe_Stockholm_456-7 \
# 				--save-loss

# ./train_test_dgsm_full_model.py DGSM_SAME_BUILDING -d Stockholm \
# 				--config "{'test_case':'457-6', 'category_type':'SIMPLE', 'training_params': '--batch-size 200 --learning-rate 0.001 --value-inference mpe --balance-data'}"\
# 				--name balanced_lr001_b10_mpe_Stockholm_457-6 \
# 				--save-loss

# ./train_test_dgsm_full_model.py DGSM_SAME_BUILDING -d Stockholm \
# 				--config "{'test_case':'456-7', 'category_type':'SIMPLE', 'training_params': '--batch-size 200 --learning-rate 0.001 --value-inference mpe --balance-data'}"\
# 				--name balanced_lr001_b10_mpe_Stockholm_456-7 \
# 				--save-loss

# # Batch size

# ./train_test_dgsm_full_model.py DGSM_SAME_BUILDING -d Stockholm \
# 				--config "{'test_case':'456-7', 'category_type':'SIMPLE', 'training_params': '--batch-size 10 --learning-rate 0.001 --balance-data'}"\
# 				--name balanced_lr001_b10_Stockholm_456-7 \
# 				--save-loss

# ./train_test_dgsm_full_model.py DGSM_SAME_BUILDING -d Stockholm \
# 				--config "{'test_case':'456-7', 'category_type':'SIMPLE', 'training_params': '--batch-size 10 --learning-rate 0.00001 --balance-data'}"\
# 				--name balanced_lr00001_b10_Stockholm_456-7 \
# 				--save-loss

# ./train_test_dgsm_full_model.py DGSM_SAME_BUILDING -d Stockholm \
# 				--config "{'test_case':'457-6', 'category_type':'SIMPLE', 'training_params': '--batch-size 10 --learning-rate 0.001 --balance-data'}"\
# 				--name balanced_lr001_b10_Stockholm_457-6 \
# 				--save-loss

# ./train_test_dgsm_full_model.py DGSM_SAME_BUILDING -d Stockholm \
# 				--config "{'test_case':'457-6', 'category_type':'SIMPLE', 'training_params': '--batch-size 10 --learning-rate 0.00001 --balance-data'}"\
# 				--name balanced_lr00001_b10_Stockholm_457-6 \
# 				--save-loss

