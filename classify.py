import pandas
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import hamming_loss
import random
import plot
from sklearn import metrics
import GradientBoostingTrees.functions as gbt_func
import SupportVectorMachine.functions as svc_func
import RandomForest.functions as rf_func

DATA_PATH = 'DataFile/roadcasualties.csv'
RANDOM_SEED = 27
PREDICTING_ATTRIBUTE = 'Crash_Police_Region'


def get_determine_train_test_indexes(data_length):
    return {i for i in range(0, data_length) if i % 5 != 0}, \
           {i for i in range(0, data_length) if i % 5 == 0}


def get_random_train_test_indexes(data_length):
    rnd = random.Random(RANDOM_SEED)

    all_indices = {i for i in range(0, data_length)}
    train_percent = .20

    train = rnd.sample(all_indices, int(data_length * train_percent))
    test = all_indices.difference(train)

    return train, test


class TestingMode:
    SingleTest = 0
    ByComplexityGrowing = 1


class DataPreparation:
    NotPrepare = 1
    MakeUnified = 2
    AddCovarianceColumns = 6
    Binarise = 10

CLASSIFIERS = {'gbt': GradientBoostingClassifier, 'svc':SVC, 'rf': RandomForestClassifier }
CLASSIFIER_PARAMETERS = {
    'gbt': {'warm_start': True, 'max_depth': 5, 'verbose': 1, 'n_estimators': 50},
    'svc':{'kernel':'rbf','tol': 0.00001, 'max_iter':100000},
    'rf': {'n_estimators': 148,'n_jobs': 2,'random_state': RANDOM_SEED,'max_features': 5}
}
ADDITIONAL_CLASSIFIER_FUNCTIONS = {
    'gbt': [gbt_func.gbt_print_importances]
}
COMPLEXITY_GROWING_ALGORITHMS = {
    'gbt': gbt_func.gbt_estimators_growing,
    'svc': svc_func.gnb_estimators_growing,
    'rf': rf_func.rf_estimators_growing
}
TRAIN_TEST_INDEXES = {
    'determine': get_determine_train_test_indexes,
    'random': get_random_train_test_indexes
}

TESTING_MODE = TestingMode.ByComplexityGrowing
TRAIN_TEST_SELECTING_MODE = 'random'
TESTED_CLASSIFIERS = 'svc'
DATA_PREPARATION = DataPreparation.AddCovarianceColumns


def unique_attribute(arr):
    return np.unique(arr, return_inverse=True)[1]

if __name__ == '__main__':

    print('Loading data...')
    data = pandas.read_csv(DATA_PATH)
    print("Dataset loaded")

    if DATA_PREPARATION & DataPreparation.MakeUnified:
        DATA_PREPARATION -= DataPreparation.MakeUnified

        print('Transforming data set to int array...')
        frame = {'Crash_Year': data['Crash_Year'],
                 'Casualty_Severity': unique_attribute(data['Casualty_Severity']),
                 'Casualty_AgeGroup': unique_attribute(data['Casualty_AgeGroup']),
                 'Casualty_Gender': unique_attribute(data['Casualty_Gender']),
                 'Casualty_RoadUserType': unique_attribute(data['Casualty_RoadUserType']),
                 'Casualty_Count': data['Casualty_Count'],
                 'Crash_Police_Region': unique_attribute(data['Crash_Police_Region'])}

        data = pandas.DataFrame(data=frame)

        if DATA_PREPARATION & DataPreparation.AddCovarianceColumns:
            rnd = random.Random(RANDOM_SEED)

            cols = list(data.columns)
            for i in range(0, len(cols)):
                if cols[i] != PREDICTING_ATTRIBUTE:
                    for j in range(i+1, len(cols)):
                        if cols[j] != PREDICTING_ATTRIBUTE:
                            rand_add = 0
                            if rnd.randint(1, 100) <= 15:
                                rand_add = rnd.randint(-5, 5) 
                            data[cols[i] + ' ' + cols[j]] = data[cols[i]] + data[cols[j]] + rand_add

        if DATA_PREPARATION & DataPreparation.Binarise:
            print('Binarising data set...')

            count_values_on_column = []
            for col in data.columns:
                if col != PREDICTING_ATTRIBUTE \
                        and col != 'Casualty_Count'\
                        and col != 'Crash_Year':
                    count_values_on_column.append((col, len(set(data[col]))))

            bin_data = pandas.DataFrame(data={PREDICTING_ATTRIBUTE: data[PREDICTING_ATTRIBUTE]})
            bin_data['Casualty_Count'] = data['Casualty_Count']
            bin_data['Crash_Year'] = data['Crash_Year']

            for item in count_values_on_column:
                col_name = item[0]
                for j in range(0, item[1]):
                    bin_data[col_name + str(j)] = pandas.Series(data[col_name] == j)

            data = bin_data
        print('Data transforming done')

    print('Making x and y sets...')
    x_cols = set(data.columns).difference([PREDICTING_ATTRIBUTE])
    x = data[list(x_cols)]
    y = data[PREDICTING_ATTRIBUTE]
    print('x, y are made')

    print('Making train set...')
    train_indices, test_indices = TRAIN_TEST_INDEXES[TRAIN_TEST_SELECTING_MODE](len(data))
    x_train, y_train = x.ix[train_indices], y.ix[train_indices]
    print('Train set made successfully')

    print('Making test set...')
    x_test, y_test = x.ix[test_indices], y.ix[test_indices]
    print('Test set made successfully')

    if (DATA_PREPARATION & DataPreparation.NotPrepare) != DataPreparation.NotPrepare:
        x_train = np.array(x_train, dtype=np.float32, order='C')
        y_train = np.array(y_train, dtype=np.float32, order='C')
        x_test = np.array(x_test, dtype=np.float32, order='C')
        y_test = np.array(y_test, dtype=np.float32, order='C')

    if TESTED_CLASSIFIERS == 'all':
        chosen_clfs = list(CLASSIFIERS.keys())
    else:
        chosen_clfs = TESTED_CLASSIFIERS
        if TESTED_CLASSIFIERS.__class__ != list:
            chosen_clfs = [TESTED_CLASSIFIERS]

    for clf_name in chosen_clfs:
            parameters = CLASSIFIER_PARAMETERS.get(clf_name)
            additional_functions = ADDITIONAL_CLASSIFIER_FUNCTIONS.get(clf_name)

            if additional_functions is None:
                additional_functions = []

            if parameters is not None and len(parameters) != 0:
                classifier = CLASSIFIERS[clf_name](**parameters)
            else:
                classifier = CLASSIFIERS[clf_name]()

            print('Classifier:', clf_name, 'parameters:', classifier.get_params())

            if TESTING_MODE == TestingMode.SingleTest:
                classifier.fit(x_train, y_train)
                train_predict = classifier.predict(x_train)
                test_predict = classifier.predict(x_test)
                
                plot.plot_classification_report(metrics.confusion_matrix(y_train, train_predict))
                print('Train err:', hamming_loss(y_train, train_predict))
                
                plot.plot_classification_report(metrics.confusion_matrix(y_test, test_predict))
                print('Test err:', hamming_loss(y_test, test_predict))

                for function in additional_functions:
                    function(classifier)

            elif TESTING_MODE == TestingMode.ByComplexityGrowing:
                grawing_func = COMPLEXITY_GROWING_ALGORITHMS[clf_name]
                grawing_func(classifier, x_test, y_test, x_train, y_train)

