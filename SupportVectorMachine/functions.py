import matplotlib.pyplot as plt
from sklearn.metrics import hamming_loss

def gnb_estimators_growing(clf, x_test, y_test, x_train, y_train):
    penalty = [i/100 for i in range(1, 10, 1)] + \
                 [i/10 for i in range(1, 10, 1)] + \
                 [i for i in range(1, 10, 1)]+\
                 [i for i in range(10, 100, 10)]
    gamma = [i/100 for i in range(1, 10, 1)] + \
                 [i/10 for i in range(1, 10, 1)] + \
                 [i for i in range(1, 10, 1)]+\
                 [i for i in range(10, 100, 10)]
                 
    err_train = []
    err_test = []
    for n_penalty in penalty:
        print('For n_penalty:', n_penalty)

        clf.C = n_penalty
        clf.fit(x_train, y_train)
        err_train.append(hamming_loss(y_train, clf.predict(x_train)))
        err_test.append(hamming_loss(y_test, clf.predict(x_test)))

    plt.plot(penalty, err_train, 'b-')
    plt.plot(penalty, err_test, 'r-')
    plt.show()
    
    err_train = []
    err_test = []
    clf.C = 1.0
    for n_gamma in gamma:
        print('For n_gamma:', n_gamma)

        clf.gamma = n_gamma
        clf.fit(x_train, y_train)
        err_train.append(hamming_loss(y_train, clf.predict(x_train)))
        err_test.append(hamming_loss(y_test, clf.predict(x_test)))

    plt.plot(gamma, err_train, 'b-')
    plt.plot(gamma, err_test, 'r-')
    plt.show()
    
    print('Growing algorithm ended')