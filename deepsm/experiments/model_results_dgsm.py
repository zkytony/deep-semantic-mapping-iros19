#!/usr/bin/env python3
#
# Generate result for a DGSM that combines several submodels;
# Specifically, we would like to produce arrays of normalized
# likelihood values.
#
# In this experiment framework, the virtual scans of each graph
# is grouped into the same room. Therefore, when saving the results
# of likelihoods, we save one file per room (i.e. per graph). These
# will later act as input to GraphSPN.
#
# We report the DGSM's classification result per node, and accuracy
# 1) per graph per class,   2) per graph all classes in total,
# 3) all graphs per class,  4) all graphs all classes in total.
#
# Refer to dgsm/result.py:Result:get_classification_results_for_graphs()

import deepsm.dgsm.model_results as dgsm_results

if __name__ == "__main__":
    dgsm_results.main(trained_classes=['1PO', 'CR'])
