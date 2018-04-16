# Running experiments

The experiment we try to perform is to unify DGSM and GraphSPN. DGSM provides
likelihood values of local virtual scan classification, where virtual scans
have corresponding topological maps. The GraphSPN built upon each topological
map will take as input these likelihoods and try to disambiguate them.

The second experiment we might perform is given the disambiguated likelihood values
produced by GraphSPN, can DGSM use MPE to figure out the geometry of the
virtual scan? (Although this demonstrates the utility of inferring low-level evidence
based on high-level semantics, there are reasons this is not a valuable experiment,
for example, we do not train DGSM and GraphSPN together, so we have nothing to expect
for what the geometry should be like. This experiment might just be a future work
when it's actually possible to train both together.)

## DGSM experiment framework

#### Generate data

     TODO: complete this README