import salty
from sklearn.model_selection import cross_val_score
from sklearn.neural_network import MLPRegressor
from sklearn.multioutput import MultiOutputRegressor
model = MLPRegressor(activation='relu', alpha=0.92078, batch_size='auto',
       beta_1=0.9, beta_2=0.999, early_stopping=False, epsilon=1e-08,
       hidden_layer_sizes=75, learning_rate='constant',
       learning_rate_init=0.001, max_iter=1e8, momentum=0.9,
       nesterovs_momentum=True, power_t=0.5, random_state=None,
       shuffle=True, solver='lbfgs', tol=1e-08, validation_fraction=0.1,
       verbose=False, warm_start=False)
multi_model = MultiOutputRegressor(model)
imputeData = salty.aggregate_data(['cpt', 'density'], merge='union',
                                  impute=True)
interData = salty.aggregate_data(['cpt', 'density'])
unionData = salty.aggregate_data(['cpt', 'density'], merge='union')
property_models = [imputeData, interData, unionData]
model_scores = []
for i, name in enumerate(property_models):
    X_train, Y_train, X_test, Y_test = salty.devmodel_to_array(name)
    model_scores.append(cross_val_score(multi_model, X_train,
                                        Y_train, cv=5))
