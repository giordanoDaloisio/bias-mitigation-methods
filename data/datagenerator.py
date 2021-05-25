import pandas as pd
import random


class Dataset:
    def __init__(self, size, num_men, num_women):
        self.size = size
        self.num_men = num_men
        self.num_women = num_women
        self.data = self.__make__()

    def __make__(self):
        array = []
        for i in range(self.num_women):
            array.append('woman')
        for j in range(self.num_men):
            array.append('male')
        random.shuffle(array)
        return pd.DataFrame(array, columns='sex')

    def get_data(self):
        return self.data

    def export(self):
        self.data.to_csv('data.csv')

    def add_field(self, field_name, field_values):
        self.data[field_name] = ''
        male_set = self.data[data['sex'] == 'male']
        self.data[field_name][self.data[data['sex'] == 'male']]
        


if __name__ == '__main__':
    data = Dataset(size=268461, num_men=109256, num_women=159205)
    print(data.get_data())
