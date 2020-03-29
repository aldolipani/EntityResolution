import numpy as np
from tqdm import tqdm

from corruptor_models import OCRCorrupter
from entity_generator import DateGenerator


def corrupt(entity, corruptor_models, p=0.1):
    n = np.random.binomial(len(corruptor_models), p)
    for i in range(n):
        entity = corruptor_models[i].corrupt(entity)
    return entity


def main():
    data_generators = [DateGenerator()]
    corrupter_models = [OCRCorrupter()]

    with open('./data/dataset.tsv', 'w') as f:
        for _ in tqdm(range(10000000)):
            for data_generator in data_generators:
                data_generator.generate()
                entity1 = data_generator.variant()
                entity1 = corrupt(entity1, corrupter_models)
                entity2 = data_generator.variant()
                entity2 = corrupt(entity2, corrupter_models)
                f.write(entity1 + '\t' + entity2 + '\n')


if __name__ == '__main__':
    main()
