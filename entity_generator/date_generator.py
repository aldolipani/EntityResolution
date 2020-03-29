import numpy as np

from entity_generator.entity_generator import EntityGenerator


class DateGenerator(EntityGenerator):

    def __init__(self, sep='/'):
        super(DateGenerator, self).__init__("Date Generator")
        self.DATA_SEPS = ['/', '.', ' ']
        self.MONTHS = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October',
                       'November', 'December']
        self.year = 0
        self.month = 0
        self.day = 0
        self.sep = sep

    def generate(self):
        self.year = np.random.randint(low=1970, high=2051)
        self.month = np.random.randint(low=1, high=13)
        self.day = np.random.randint(low=1, high=31)

    def variant(self):
        n = np.random.randint(low=0, high=5)
        if n == 0:  # other separator
            sep = np.random.choice(self.DATA_SEPS)[0]
            return str(self.day) + sep + str(self.month) + sep + str(self.year)
        elif n == 1:  # add zeros if needed
            return str(self.day).zfill(2) + self.sep + str(self.month).zfill(2) + self.sep + str(self.year)
        elif n == 2:  # use two digits for year
            return str(self.day) + self.sep + str(self.month) + self.sep + str(self.year)[2:4]
        elif n == 3:  # substitute month number with string
            return str(self.day) + ' ' + self.MONTHS[self.month - 1] + ' ' + str(self.year)
        elif n == 4:
            return str(self.day) + self.sep + str(self.month) + self.sep + str(self.year)

if __name__ == '__init__':
    open('../data/date_dataset', 'w')