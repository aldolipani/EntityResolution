import nltk

from string_similarity import EntitySimilarity


class LanguageModelSimilarity(EntitySimilarity):

    def __init__(self):
        super(LanguageModelSimilarity, self).__init__("Language Model")

    def compute(self, entity1: str, entity2: str):
        bag_of_words1 = self.preprocess(entity1)
        bag_of_words2 = self.preprocess(entity2)
        len_entity1 = sum(bag_of_words1.values())
        len_entity2 = sum(bag_of_words2.values())
        res1 = 0.0
        res2 = 0.0
        for term in set(bag_of_words1.keys()).intersection(bag_of_words2.keys()):
            res1 += bag_of_words1[term]/len_entity1
            res2 += bag_of_words2[term]/len_entity2
        return (res1 + res2)/2

    def preprocess(self, str_entity):
        res = {}
        for token in nltk.word_tokenize(str_entity):
            token = token.lower()
            if token not in res:
                res[token] = 0
            res[token] += 1
        return res
