from string_similarity import NormalizedLevenshteinSimilarity, UnigramTransformerSimilarity, LanguageModelSimilarity


def main():
    string_similarities = [NormalizedLevenshteinSimilarity(),
                           LanguageModelSimilarity(),
                           UnigramTransformerSimilarity()]

    with open('data/examples.tsv', 'r') as f:
        for line in f.readlines():
            items = line.split('\t')
            entity1 = items[0].strip()
            entity2 = items[1].strip()
            print('E1 =', entity1, ', E2 =', entity2)
            for string_similarity in string_similarities:
                print('{:30s}'.format(string_similarity.name), '{:.4f}'.format(string_similarity.compute(entity1, entity2)))
            print()

if __name__ == "__main__":
    main()
