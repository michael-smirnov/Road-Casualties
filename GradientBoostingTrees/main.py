from DataSet import DataSet as ds
from sklearn.ensemble import GradientBoostingClassifier
import matplotlib.pyplot as plt

if __name__ == '__main__':
    data = ds.load_data()
    data = ds.transform_data_to_ints(data)
    x = ds.get_x_all(data)
    y = ds.get_y_all(data)

    train_indices = ds.get_train_indices(data)
    x_train, y_train = ds.get_fortran_array(x.ix[train_indices]), ds.get_fortran_array(y.ix[train_indices])
    x_test, y_test = ds.get_fortran_array(x.drop(train_indices)), ds.get_fortran_array(y.drop(train_indices))

    clf = GradientBoostingClassifier(n_estimators=400, max_depth=5)

    choice = 0
    if choice == 0:
        clf.fit(x_train, y_train)

        print('Test-train check')
        print('Train err:', ds.error(y_train, clf.predict(x_train)))
        print('Test err:', ds.error(y_test, clf.predict(x_test)))

    elif choice == 1:
        y_train_err = []
        y_test_err = []

        clf.n_estimators = 300
        max_depth_list = [i for i in range(1, 10)]
        for max_depth in max_depth_list:
            print('solving for', max_depth, end=' ')
            clf.max_depth = max_depth
            clf.fit(x_train, y_train)

            y_train_err.append(ds.error(y_train, clf.predict(x_train)))
            y_test_err.append(ds.error(y_test, clf.predict(x_test)))
            print('done.')

        plt.plot(max_depth_list, y_train_err, 'b-')
        plt.plot(max_depth_list, y_test_err, 'r-')
        plt.show()

    elif choice == 2:
        y_train_err = []
        y_test_err = []

        n_estimators_list = [i for i in range(100, 600, 100)]
        for n_estimators in n_estimators_list:
            print('solving for', n_estimators, end=' ')
            clf.n_estimators = n_estimators
            clf.fit(x_train, y_train)

            y_train_err.append(ds.error(y_train, clf.predict(x_train)))
            y_test_err.append(ds.error(y_test, clf.predict(x_test)))
            print('done.')

        plt.plot(n_estimators_list, y_train_err, 'b-')
        plt.plot(n_estimators_list, y_test_err, 'r-')
        plt.show()

    elif choice == 3:
        print('K-Fold check')
        y_train_err = []
        y_test_err = []
        for train_indices, test_indices in ds.get_k_fold_iterator(data):
            x_train, y_train = ds.get_fortran_array(x.ix[train_indices]), ds.get_fortran_array(y.ix[train_indices])
            x_test, y_test = ds.get_fortran_array(x.ix[test_indices]), ds.get_fortran_array(y.ix[test_indices])
            clf.fit(x_train, y_train)
            y_train_err.append(ds.error(y_train, clf.predict(x_train)))
            y_test_err.append(ds.error(y_test, clf.predict(x_test)))
        print('Train err:', sum(y_train_err)/len(y_train_err))
        print('Test err:', sum(y_test_err)/len(y_test_err))
    elif choice == 4:
        print('LOO check')
        y_train_err = []
        y_test_err = []
        for train_indices, test_indices in ds.get_loo_iterator(data):
            x_train, y_train = ds.get_fortran_array(x.ix[train_indices]), ds.get_fortran_array(y.ix[train_indices])
            x_test, y_test = ds.get_fortran_array(x.ix[test_indices]), ds.get_fortran_array(y.ix[test_indices])
            clf.fit(x_train, y_train)
            y_train_err.append(ds.error(y_train, clf.predict(x_train)))
            y_test_err.append(ds.error(y_test, clf.predict(x_test)))
        print('Train err:', sum(y_train_err)/len(y_train_err))
        print('Test err:', sum(y_test_err)/len(y_test_err))
