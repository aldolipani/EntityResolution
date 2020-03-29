import numpy as np

from corruptor_models.corrupter_model import CurroptorModel


class OCRCorrupter(CurroptorModel):

    def __init__(self):
        super(OCRCorrupter, self).__init__("OCR Curroptor")
        self.common_alterations = {
            '|': ['/', '\\'],
            '\\': ['|'],
            '/': ['|'],
            '1': ['I'],
            'I': ['1'],
            ',': ['.'],
            'n': ['m'],
            'm': ['n']
        }

    def corrupt(self, entity: str):
        shuffled_common_alterations = list(self.common_alterations.items())
        np.random.shuffle(shuffled_common_alterations)
        for target, alterations in shuffled_common_alterations:
            if target in entity:
                items = entity.split(target)
                alteration = np.random.choice(alterations)
                i = np.random.randint(0, len(items) - 1)
                entity = items[0]
                for n, item in enumerate(items[1:]):
                    if n == i:
                        entity += alteration
                    else:
                        entity += target
                    entity += item
                return entity
        return entity
