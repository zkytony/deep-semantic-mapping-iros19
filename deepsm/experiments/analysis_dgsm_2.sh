# unbalanced
./train_test_dgsm_full_model.py DGSM_SAME_BUILDING -d Stockholm \
				--config "{'test_case':'457-6', 'category_type':'SIMPLE', 'training_params': '--batch-size 10 --learning-rate 0.00001'}"\
				--name unbalanced_lr01_b10_Stockholm_457-6 \
				--save-loss


# stopping conditions
./train_test_dgsm_full_model.py DGSM_SAME_BUILDING -d Stockholm \
				--config "{'test_case':'456-7', 'category_type':'SIMPLE', 'training_params': '--batch-size 200 --learning-rate 0.001 --update-threshold 1 --balance-data'}"\
				--name balanced_lr001_b200_uc10_Stockholm_456-7 \
				--save-loss

./train_test_dgsm_full_model.py DGSM_SAME_BUILDING -d Stockholm \
				--config "{'test_case':'457-6', 'category_type':'SIMPLE', 'training_params': '--batch-size 200 --learning-rate 0.001 --update-threshold 1 --balance-data'}"\
				--name balanced_lr001_b200_uc10_Stockholm_457-6 \
				--save-loss

./train_test_dgsm_full_model.py DGSM_SAME_BUILDING -d Stockholm \
				--config "{'test_case':'456-7', 'category_type':'SIMPLE', 'training_params': '--batch-size 200 --learning-rate 0.001 --update-threshold 0.1 --balance-data'}"\
				--name balanced_lr001_b200_uc1_Stockholm_456-7 \
				--save-loss

./train_test_dgsm_full_model.py DGSM_SAME_BUILDING -d Stockholm \
				--config "{'test_case':'457-6', 'category_type':'SIMPLE', 'training_params': '--batch-size 200 --learning-rate 0.001 --update-threshold 0.1 --balance-data'}"\
				--name balanced_lr001_b200_uc1_Stockholm_457-6 \
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
				--config "{'test_case':'456-7', 'category_type':'SIMPLE', 'training_params': '--batch-size 200 --learning-rate 0.001 --update-threshold 0.01 --balance-data'}"\
				--name balanced_lr001_b200_uc001_Stockholm_456-7 \
				--save-loss

./train_test_dgsm_full_model.py DGSM_SAME_BUILDING -d Stockholm \
				--config "{'test_case':'457-6', 'category_type':'SIMPLE', 'training_params': '--batch-size 200 --learning-rate 0.001 --update-threshold 0.01 --balance-data'}"\
				--name balanced_lr001_b200_uc001_Stockholm_457-6 \
				--save-loss


./train_test_dgsm_full_model.py DGSM_SAME_BUILDING -d Stockholm \
				--config "{'test_case':'456-7', 'category_type':'SIMPLE', 'training_params': '--batch-size 200 --learning-rate 0.001 --update-threshold 0.001 --balance-data'}"\
				--name balanced_lr001_b001_uc10_Stockholm_456-7 \
				--save-loss

./train_test_dgsm_full_model.py DGSM_SAME_BUILDING -d Stockholm \
				--config "{'test_case':'457-6', 'category_type':'SIMPLE', 'training_params': '--batch-size 200 --learning-rate 0.001 --update-threshold 0.001 --balance-data'}"\
				--name balanced_lr001_b001_uc10_Stockholm_457-6 \
				--save-loss


./train_test_dgsm_full_model.py DGSM_SAME_BUILDING -d Stockholm \
				--config "{'test_case':'456-7', 'category_type':'SIMPLE', 'training_params': '--batch-size 200 --learning-rate 0.001 --update-threshold 0.0001 --balance-data'}"\
				--name balanced_lr001_b200_uc0001_Stockholm_456-7 \
				--save-loss

./train_test_dgsm_full_model.py DGSM_SAME_BUILDING -d Stockholm \
				--config "{'test_case':'457-6', 'category_type':'SIMPLE', 'training_params': '--batch-size 200 --learning-rate 0.001 --update-threshold 0.0001 --balance-data'}"\
				--name balanced_lr001_b200_uc0001_Stockholm_457-6 \
				--save-loss



# balance data
./train_test_dgsm_full_model.py DGSM_SAME_BUILDING -d Stockholm \
				--config "{'test_case':'456-7', 'category_type':'SIMPLE', 'training_params': '--batch-size 10 --learning-rate 0.001'}"\
				--name unbalanced_lr001_b10_Stockholm_456-7 \
				--save-loss

./train_test_dgsm_full_model.py DGSM_SAME_BUILDING -d Stockholm \
				--config "{'test_case':'457-6', 'category_type':'SIMPLE', 'training_params': '--batch-size 200 --learning-rate 0.001'}"\
				--name unbalanced_lr001_b10_Stockholm_456-7 \
				--save-loss


./train_test_dgsm_full_model.py DGSM_SAME_BUILDING -d Stockholm \
				--config "{'test_case':'457-6', 'category_type':'SIMPLE', 'training_params': '--batch-size 200 --learning-rate 0.001 --value-inference mpe'}"\
				--name unbalanced_lr001_b10_Stockholm_456-7 \
				--save-loss

./train_test_dgsm_full_model.py DGSM_SAME_BUILDING -d Stockholm \
				--config "{'test_case':'457-6', 'category_type':'SIMPLE', 'training_params': '--batch-size 10 --learning-rate 0.001 --value-inference mpe'}"\
				--name unbalanced_lr001_b10_Stockholm_456-7 \
				--save-loss

