import pandas as pd
import random
import numpy as np
from sklearn import preprocessing
from sklearn.cross_validation import KFold
from sklearn.cross_validation import LeaveOneOut
from sklearn.metrics import hamming_loss

DATA_PATH = '../DataFile/roadcasualties.csv'
RANDOM_SEED = 27
PREDICTING_ATTRIBUTE = 'Crash_Police_Region'


def load_data():
    return pd.read_csv(DATA_PATH)


# Служебный метод. Получает числовое уникальное значение строкового атрибута
def unique_attribute(arr):
    return np.unique(arr, return_inverse=True)[1]


# Трансформирует строковые данные в числовые. Строковые данные методами sklearn не воспринимаются
def transform_data_to_ints(data):
    frame = {'Crash_Year': data['Crash_Year'],
             'Casualty_Severity': unique_attribute(data['Casualty_Severity']),
             'Casualty_AgeGroup': unique_attribute(data['Casualty_AgeGroup']),
             'Casualty_Gender': unique_attribute(data['Casualty_Gender']),
             'Casualty_RoadUserType': unique_attribute(data['Casualty_RoadUserType']),
             'Casualty_Count': data['Casualty_Count'],
             'Crash_Police_Region': unique_attribute(data['Crash_Police_Region'])}
    return pd.DataFrame(data=frame)


def get_x_all(data):
    x_cols = set(data.columns).difference([PREDICTING_ATTRIBUTE])
    return data[list(x_cols)]


def get_y_all(data):
    return data[PREDICTING_ATTRIBUTE]


# Нормировка числовых данных
def scale_x(x):
    scaler = preprocessing.StandardScaler()
    return scaler.fit_transform(x)


def get_k_fold_iterator(data):
    return KFold(len(data), n_folds=10, random_state=RANDOM_SEED, shuffle=True)


def get_loo_iterator(data):
    return LeaveOneOut(len(data))


def get_train_indices(data):
    return [i for i in range(0, len(data)) if i % 5 != 0]


# Для производительности лучше скармливать методам sklearn фортрановский массив чем DataFrame из pandas
def get_fortran_array(data):
    return np.asfortranarray(data, dtype=np.float32)


def error(y_exact, y_predict):
    return hamming_loss(y_exact, y_predict)


# Данные классы собственного написания, делают все тоже самое. Оставлены на всякий случай :)
class DataLoader:
    def __init__(self, file_path):
        self.data_csv = []
        self.feature_names = []
        self.read(file_path)

    def read(self, file_path):
        file = open(file_path)
        self.data_csv = []
        self.feature_names = file.readline().replace('\n', '').split(',')
        for row in file:
            self.data_csv.append((row.replace('\n', '')).split(','))


class DataConverter:
    def __init__(self, feature_names, data_csv, data_types):
        self.data_set = np.zeros((len(data_csv), len(feature_names)), dtype=np.int)
        self.feature_names = feature_names
        self.data_types = data_types
        self.attributes = []

        keys = [[] for i in range(0, len(feature_names))]
        values = [[] for i in range(0, len(feature_names))]

        row_number = 0
        for row in data_csv:
            attribute_number = 0
            for element in row:
                if keys[attribute_number].count(element) == 0:
                    keys[attribute_number].append(element)

                    if data_types[attribute_number] == int:
                        values[attribute_number].append(int(element))
                    else:
                        values[attribute_number].append(len(keys[attribute_number]) - 1)

                self.data_set[row_number, attribute_number] = \
                    values[attribute_number][keys[attribute_number].index(element)]
                attribute_number += 1
            row_number += 1

        attributes_list = []
        for attribute_number in range(0, len(feature_names)):
            attribute_dict = {keys[attribute_number][i]: values[attribute_number][i] for i in
                              range(0, len(keys[attribute_number]))}
            attributes_list.append(attribute_dict)

        self.attributes = {feature_names[i]: attributes_list[i] for i in range(0, len(feature_names))}

    def get_attribute_value(self, feature_name, attribute_name):
        feature_index = self.feature_names.index(feature_name)
        if self.data_types[feature_index] == int:
            return attribute_name
        else:
            return self.attributes.get(feature_name).get(attribute_name)

    def get_x_value(self, x_name, target_feature):
        x_value = []
        i = 0
        for feature in self.feature_names:
            if feature != target_feature:
                x_value.append(self.get_attribute_value(feature, x_name[i]))
                i += 1
        return x_value

    def get_y_name(self, y_value, target_feature):
        y = list(self.attributes[target_feature].items())
        for item in y:
            if item[1] == y_value:
                return item[0]
        return None


class DataSetMaker:
    def __init__(self, feature_names, data, target_feature, training_percent):
        self.data = data
        self.target_feature = feature_names.index('Crash_Police_Region')
        self.feature_names = feature_names

        split_data = np.split(self.data, [self.target_feature, self.target_feature + 1], axis=1)
        self.x_all = np.hstack((split_data[0], split_data[2]))
        self.y_all = split_data[1][:, 0]

        random.seed(RANDOM_SEED)
        self.training_percent = training_percent

        self.current_set_indexes = []
        self.x_train, self.y_train = [], []
        self.x_test, self.y_test = [], []
        self.x_cross, self.y_cross = [], []

        self.generate_train_test()

    def generate_train_test(self):
        data_length = len(self.data)
        self.current_set_indexes = random.sample([i for i in range(0, data_length)],
                                                 int(data_length * self.training_percent))

        self.x_train = np.zeros((len(self.current_set_indexes), len(self.feature_names) - 1), dtype=np.int)
        self.y_train = np.zeros(len(self.current_set_indexes), dtype=np.int)

        self.x_test = np.zeros((data_length - len(self.current_set_indexes), len(self.feature_names) - 1), dtype=np.int)
        self.y_test = np.zeros(data_length - len(self.current_set_indexes), dtype=np.int)

        last_train = 0
        last_test = 0
        for row in range(0, data_length):
            if self.current_set_indexes.count(row) == 0:
                self.x_test[last_test], self.y_test[last_test] = self.x_all[row], self.y_all[row]
                last_test += 1
            else:
                self.x_train[last_train], self.y_train[last_train] = self.x_all[row], self.y_all[row]
                last_train += 1

    def generate_cross_validation_sets(self, separation_count):
        separation_length = int(len(self.data) / separation_count)
        selectable_indexes = {i for i in range(0, len(self.data))}

        x = np.zeros((separation_length, len(self.feature_names) - 1), dtype=np.int)
        y = np.zeros(separation_length, dtype=np.int)

        for i in range(0, separation_count - 1):
            current_separation = set(random.sample(selectable_indexes, separation_length))

            last_row = 0
            for row in current_separation:
                x[last_row] = self.x_all[row]
                y[last_row] = self.y_all[row]
                last_row += 1
            self.x_cross.append(np.array(x))
            self.y_cross.append(np.array(y))
            selectable_indexes -= current_separation

        x = np.zeros((len(selectable_indexes), len(self.feature_names) - 1), dtype=np.int)
        y = np.zeros(len(selectable_indexes), dtype=np.int)

        last_row = 0
        for row in selectable_indexes:
            x[last_row] = self.x_all[row]
            y[last_row] = self.y_all[row]
            last_row += 1
        self.x_cross.append(np.array(x))
        self.y_cross.append(np.array(y))

