import salty
import salty.adaptive_learn as adl

data = ['cpt', 'density']
T = [298.1, 298.16]
P = [101, 102]
devmodel = salty.aggregate_data(data, T=T, P=P)
targets = adl.expand_convex_hull(devmodel, expansion_target=[1, 1.005],
                                 target_number=10)
print(targets)

