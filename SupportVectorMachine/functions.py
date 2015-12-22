import matplotlib.pyplot as plt
from sklearn.metrics import hamming_loss

def gnb_estimators_growing(clf, x_test, y_test, x_train, y_train):
    penalty = [2**i for i in range(-5, 15, 3)]
    gamma = [2**i for i in range(-15, 3, 2)]
    
    err_train = []
    err_test = []
    
   
    clf.C = 1.0
    for n_gamma in gamma:
        print('For n_gamma:', n_gamma)

        clf.gamma = n_gamma
        for n_penalty in penalty:
            print('For n_penalty:', n_penalty)

            clf.C = n_penalty
            clf.fit(x_train, y_train)
            err_train.append(hamming_loss(y_train, clf.predict(x_train)))
            err_test.append(hamming_loss(y_test, clf.predict(x_test)))
        
        plt.plot(penalty, err_train, color="blue",label="train")
        plt.plot(penalty, err_test, color="red",label="test")
        plt.xlabel("Penalty")
        plt.ylabel("Error")
        plt.legend(loc="upper right",fancybox=True);      
        plt.show()
    
        err_train = []
        err_test = []
    
    print('Growing algorithm ended')