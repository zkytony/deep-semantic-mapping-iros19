# TRY 2

# stopping condition
./train_test_dgsm_full_model.py DGSM_SAME_BUILDING -d Stockholm \
				--config "{'test_case':'456-7', 'category_type':'SIMPLE', 'training_params': '--batch-size 200 --learning-rate 0.001 --update-threshold 0.001 --balance-data'}"\
				--save-loss

./train_test_dgsm_full_model.py DGSM_SAME_BUILDING -d Stockholm \
				--config "{'test_case':'457-6', 'category_type':'SIMPLE', 'training_params': '--batch-size 200 --learning-rate 0.001 --update-threshold 0.001 --balance-data'}"\
				--save-loss


./train_test_dgsm_full_model.py DGSM_SAME_BUILDING -d Stockholm \
				--config "{'test_case':'456-7', 'category_type':'SIMPLE', 'training_params': '--batch-size 200 --learning-rate 0.01 --update-threshold 0.001 --balance-data'}"\
				--save-loss

./train_test_dgsm_full_model.py DGSM_SAME_BUILDING -d Stockholm \
				--config "{'test_case':'457-6', 'category_type':'SIMPLE', 'training_params': '--batch-size 200 --learning-rate 0.01 --update-threshold 0.001 --balance-data'}"\
				--save-loss


./train_test_dgsm_full_model.py DGSM_SAME_BUILDING -d Stockholm \
				--config "{'test_case':'456-7', 'category_type':'SIMPLE', 'training_params': '--batch-size 10 --learning-rate 0.001 --update-threshold 0.001 --balance-data'}"\
				--save-loss

./train_test_dgsm_full_model.py DGSM_SAME_BUILDING -d Stockholm \
				--config "{'test_case':'457-6', 'category_type':'SIMPLE', 'training_params': '--batch-size 10 --learning-rate 0.001 --update-threshold 0.001 --balance-data'}"\
				--save-loss

./train_test_dgsm_full_model.py DGSM_SAME_BUILDING -d Stockholm \
				--config "{'test_case':'456-7', 'category_type':'SIMPLE', 'training_params': '--batch-size 10 --learning-rate 0.01 --update-threshold 0.001 --balance-data'}"\
				--save-loss

./train_test_dgsm_full_model.py DGSM_SAME_BUILDING -d Stockholm \
				--config "{'test_case':'457-6', 'category_type':'SIMPLE', 'training_params': '--batch-size 10 --learning-rate 0.01 --update-threshold 0.001 --balance-data'}"\
				--save-loss

# unbalanced
./train_test_dgsm_full_model.py DGSM_SAME_BUILDING -d Stockholm \
				--config "{'test_case':'457-6', 'category_type':'SIMPLE', 'training_params': '--batch-size 10 --learning-rate 0.00001'}"\
				--save-loss


# stopping conditions
./train_test_dgsm_full_model.py DGSM_SAME_BUILDING -d Stockholm \
				--config "{'test_case':'456-7', 'category_type':'SIMPLE', 'training_params': '--batch-size 200 --learning-rate 0.001 --update-threshold 1 --balance-data'}"\
				--save-loss

./train_test_dgsm_full_model.py DGSM_SAME_BUILDING -d Stockholm \
				--config "{'test_case':'457-6', 'category_type':'SIMPLE', 'training_params': '--batch-size 200 --learning-rate 0.001 --update-threshold 1 --balance-data'}"\
				--save-loss

./train_test_dgsm_full_model.py DGSM_SAME_BUILDING -d Stockholm \
				--config "{'test_case':'456-7', 'category_type':'SIMPLE', 'training_params': '--batch-size 200 --learning-rate 0.001 --update-threshold 0.1 --balance-data'}"\
				--save-loss

./train_test_dgsm_full_model.py DGSM_SAME_BUILDING -d Stockholm \
				--config "{'test_case':'457-6', 'category_type':'SIMPLE', 'training_params': '--batch-size 200 --learning-rate 0.001 --update-threshold 0.1 --balance-data'}"\
				--save-loss


./train_test_dgsm_full_model.py DGSM_SAME_BUILDING -d Stockholm \
				--config "{'test_case':'456-7', 'category_type':'SIMPLE', 'training_params': '--batch-size 10 --learning-rate 0.001 --update-threshold 0.1 --balance-data'}"\
				--save-loss

./train_test_dgsm_full_model.py DGSM_SAME_BUILDING -d Stockholm \
				--config "{'test_case':'457-6', 'category_type':'SIMPLE', 'training_params': '--batch-size 10 --learning-rate 0.001 --update-threshold 0.1 --balance-data'}"\
				--save-loss


./train_test_dgsm_full_model.py DGSM_SAME_BUILDING -d Stockholm \
				--config "{'test_case':'456-7', 'category_type':'SIMPLE', 'training_params': '--batch-size 200 --learning-rate 0.001 --update-threshold 0.01 --balance-data'}"\
				--save-loss

./train_test_dgsm_full_model.py DGSM_SAME_BUILDING -d Stockholm \
				--config "{'test_case':'457-6', 'category_type':'SIMPLE', 'training_params': '--batch-size 200 --learning-rate 0.001 --update-threshold 0.01 --balance-data'}"\
				--save-loss


./train_test_dgsm_full_model.py DGSM_SAME_BUILDING -d Stockholm \
				--config "{'test_case':'456-7', 'category_type':'SIMPLE', 'training_params': '--batch-size 200 --learning-rate 0.001 --update-threshold 0.001 --balance-data'}"\
				--save-loss

./train_test_dgsm_full_model.py DGSM_SAME_BUILDING -d Stockholm \
				--config "{'test_case':'457-6', 'category_type':'SIMPLE', 'training_params': '--batch-size 200 --learning-rate 0.001 --update-threshold 0.001 --balance-data'}"\
				--save-loss


./train_test_dgsm_full_model.py DGSM_SAME_BUILDING -d Stockholm \
				--config "{'test_case':'456-7', 'category_type':'SIMPLE', 'training_params': '--batch-size 200 --learning-rate 0.001 --update-threshold 0.0001 --balance-data'}"\
				--save-loss

./train_test_dgsm_full_model.py DGSM_SAME_BUILDING -d Stockholm \
				--config "{'test_case':'457-6', 'category_type':'SIMPLE', 'training_params': '--batch-size 200 --learning-rate 0.001 --update-threshold 0.0001 --balance-data'}"\
				--save-loss



# balance data
./train_test_dgsm_full_model.py DGSM_SAME_BUILDING -d Stockholm \
				--config "{'test_case':'456-7', 'category_type':'SIMPLE', 'training_params': '--batch-size 10 --learning-rate 0.001'}"\
				--save-loss

./train_test_dgsm_full_model.py DGSM_SAME_BUILDING -d Stockholm \
				--config "{'test_case':'457-6', 'category_type':'SIMPLE', 'training_params': '--batch-size 200 --learning-rate 0.001'}"\
				--save-loss


./train_test_dgsm_full_model.py DGSM_SAME_BUILDING -d Stockholm \
				--config "{'test_case':'457-6', 'category_type':'SIMPLE', 'training_params': '--batch-size 200 --learning-rate 0.001 --value-inference mpe'}"\
				--save-loss

./train_test_dgsm_full_model.py DGSM_SAME_BUILDING -d Stockholm \
				--config "{'test_case':'457-6', 'category_type':'SIMPLE', 'training_params': '--batch-size 10 --learning-rate 0.001 --value-inference mpe'}"\
				--save-loss

