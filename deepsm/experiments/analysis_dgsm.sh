# All combinations of parameters

# Default setting:
# batch_size = 10
# learning_rate = 0.001
# balance_data = True


# batch size
./train_test_dgsm_full_model.py DGSM_SAME_BUILDING -d Stockholm \
				--config "{'test_case':'456-7', 'category_type':'SIMPLE', 'training_params': '--batch-size 200 --learning-rate 0.001 --balance-data'}"\
				--name balanced_lr0001_b200_Stockholm_456-7 \
				--save-loss

./train_test_dgsm_full_model.py DGSM_SAME_BUILDING -d Stockholm \
				--config "{'test_case':'457-6', 'category_type':'SIMPLE', 'training_params': '--batch-size 200 --learning-rate 0.001 --balance-data'}"\
				--name balanced_lr0001_b200_Stockholm_457-6 \
				--save-loss

# learning rate 0.001 vs 0.00001
./train_test_dgsm_full_model.py DGSM_SAME_BUILDING -d Stockholm \
				--config "{'test_case':'456-7', 'category_type':'SIMPLE', 'training_params': '--batch-size 10 --learning-rate 0.001 --balance-data'}"\
				--name balanced_lr001_b10_Stockholm_456-7 \
				--save-loss

n./train_test_dgsm_full_model.py DGSM_SAME_BUILDING -d Stockholm \
				--config "{'test_case':'456-7', 'category_type':'SIMPLE', 'training_params': '--batch-size 10 --learning-rate 0.00001 --balance-data'}"\
				--name balanced_lr00001_b10_Stockholm_456-7 \
				--save-loss


./train_test_dgsm_full_model.py DGSM_SAME_BUILDING -d Stockholm \
				--config "{'test_case':'457-6', 'category_type':'SIMPLE', 'training_params': '--batch-size 10 --learning-rate 0.001 --balance-data'}"\
				--name balanced_lr001_b10_Stockholm_457-6 \
				--save-loss

./train_test_dgsm_full_model.py DGSM_SAME_BUILDING -d Stockholm \
				--config "{'test_case':'457-6', 'category_type':'SIMPLE', 'training_params': '--batch-size 10 --learning-rate 0.00001 --balance-data'}"\
				--name balanced_lr00001_b10_Stockholm_457-6 \
				--save-loss

# mpe inference
./train_test_dgsm_full_model.py DGSM_SAME_BUILDING -d Stockholm \
				--config "{'test_case':'456-7', 'category_type':'SIMPLE', 'training_params': '--batch-size 10 --learning-rate 0.001 --value-inference mpe --balance-data'}"\
				--name balanced_lr001_b10_mpe_Stockholm_456-7 \
				--save-loss

./train_test_dgsm_full_model.py DGSM_SAME_BUILDING -d Stockholm \
				--config "{'test_case':'457-6', 'category_type':'SIMPLE', 'training_params': '--batch-size 10 --learning-rate 0.001 --value-inference mpe --balance-data'}"\
				--name balanced_lr001_b10_mpe_Stockholm_457-6 \
				--save-loss

# stopping conditions
./train_test_dgsm_full_model.py DGSM_SAME_BUILDING -d Stockholm \
				--config "{'test_case':'456-7', 'category_type':'SIMPLE', 'training_params': '--batch-size 10 --learning-rate 0.001 --update-threshold 1 --balance-data'}"\
				--name balanced_lr001_b10_uc10_Stockholm_456-7 \
				--save-loss

./train_test_dgsm_full_model.py DGSM_SAME_BUILDING -d Stockholm \
				--config "{'test_case':'457-6', 'category_type':'SIMPLE', 'training_params': '--batch-size 10 --learning-rate 0.001 --update-threshold 1 --balance-data'}"\
				--name balanced_lr001_b10_uc10_Stockholm_457-6 \
				--save-loss

./train_test_dgsm_full_model.py DGSM_SAME_BUILDING -d Stockholm \
				--config "{'test_case':'456-7', 'category_type':'SIMPLE', 'training_params': '--batch-size 10 --learning-rate 0.001 --update-threshold 0.1 --balance-data'}"\
				--name balanced_lr001_b10_uc1_Stockholm_456-7 \
				--save-loss

./train_test_dgsm_full_model.py DGSM_SAME_BUILDING -d Stockholm \
				--config "{'test_case':'457-6', 'category_type':'SIMPLE', 'training_params': '--batch-size 10 --learning-rate 0.001 --update-threshold 0.1 --balance-data'}"\
				--name balanced_lr001_b10_uc1_Stockholm_457-6 \
				--save-loss

./train_test_dgsm_full_model.py DGSM_SAME_BUILDING -d Stockholm \
				--config "{'test_case':'456-7', 'category_type':'SIMPLE', 'training_params': '--batch-size 10 --learning-rate 0.001 --update-threshold 0.01 --balance-data'}"\
				--name balanced_lr001_b10_uc01_Stockholm_456-7 \
				--save-loss

./train_test_dgsm_full_model.py DGSM_SAME_BUILDING -d Stockholm \
				--config "{'test_case':'457-6', 'category_type':'SIMPLE', 'training_params': '--batch-size 10 --learning-rate 0.001 --update-threshold 0.01 --balance-data'}"\
				--name balanced_lr001_b10_uc01_Stockholm_457-6 \
				--save-loss

# balance data
./train_test_dgsm_full_model.py DGSM_SAME_BUILDING -d Stockholm \
				--config "{'test_case':'456-7', 'category_type':'SIMPLE', 'training_params': '--batch-size 10 --learning-rate 0.001'}"\
				--name unbalanced_lr001_b10_Stockholm_456-7 \
				--save-loss

n./train_test_dgsm_full_model.py DGSM_SAME_BUILDING -d Stockholm \
				--config "{'test_case':'457-6', 'category_type':'SIMPLE', 'training_params': '--batch-size 10 --learning-rate 0.00001'}"\
				--name unbalanced_lr01_b10_Stockholm_457-6 \
				--save-loss
