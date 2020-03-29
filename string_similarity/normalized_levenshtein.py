import textdistance

from string_similarity.entity_similarity import EntitySimilarity


class NormalizedLevenshteinSimilarity(EntitySimilarity):

    def __init__(self):
        super(NormalizedLevenshteinSimilarity, self).__init__("Normalized Levenshtein")
        self.similarity = textdistance.Levenshtein()

    def compute(self, entity1: str, entity2: str):
        return self.similarity.normalized_similarity(entity1, entity2)
