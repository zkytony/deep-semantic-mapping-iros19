# View template
# - trial 0 Stockholm
./train_test_graphspn_colddb_multiple_sequences.py DGSM_SAME_BUILDING -d Stockholm -e ColdExperiment -t fullEdgeRelP5T0Stockholm -l 0 --skip-placeholders --template VIEW
# - trial 1 Freiburg
./train_test_graphspn_colddb_multiple_sequences.py DGSM_SAME_BUILDING -d Freiburg -e ColdExperiment -t fullEdgeRelP5T0Freiburg -l 0 --skip-placeholders --template VIEW
# - trial 2 Saarbrucken
./train_test_graphspn_colddb_multiple_sequences.py DGSM_SAME_BUILDING -d Saarbrucken -e ColdExperiment -t fullEdgeRelP5T0Saarbrucken -l 0 --skip-placeholders --template VIEW

# Three-node template
# - trial 0 Stockholm
./train_test_graphspn_colddb_multiple_sequences.py DGSM_SAME_BUILDING -d Stockholm -e ColdExperiment -t fullThreeP5T0Stockholm -l 0 --skip-placeholders --template THREE
# - trial 1 Freiburg
./train_test_graphspn_colddb_multiple_sequences.py DGSM_SAME_BUILDING -d Freiburg -e ColdExperiment -t fullThreeP5T0Freiburg -l 0 --skip-placeholders --template THREE
# - trial 2 Saarbrucken
./train_test_graphspn_colddb_multiple_sequences.py DGSM_SAME_BUILDING -d Saarbrucken -e ColdExperiment -t fullThreeP5T0Saarbrucken -l 0 --skip-placeholders --template THREE
