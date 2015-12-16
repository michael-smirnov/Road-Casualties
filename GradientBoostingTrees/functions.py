import matplotlib.pyplot as plt
from sklearn.metrics import hamming_loss


def gbt_print_importances(clf):
    print('Importances:', clf.feature_importances_)


def gbt_estimators_growing(clf, x_test, y_test, x_train, y_train):
    estimators = [1, 2, 3, 4, 5, 10, 20, 30, 40, 50] + \
                 [i for i in range(100, 1000, 100)] #+ \
                 #[i for i in range(1000, 6000, 1000)]

    err_train = []
    err_test = []
    for n_estimators in estimators:
        print('For n_estimators:', n_estimators)

        clf.n_estimators = n_estimators
        clf.fit(x_train, y_train)
        err_train.append(hamming_loss(y_train, clf.predict(x_train)))
        err_test.append(hamming_loss(y_test, clf.predict(x_test)))

    plt.plot(estimators, err_train, 'b-')
    plt.plot(estimators, err_test, 'r-')
    plt.show()
    print('Growing algorithm ended')