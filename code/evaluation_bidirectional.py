from collections import Counter

class Eval:
    def __init__(self, gold, pred):
        # assert len(gold)==len(pred)
        self.gold = gold
        self.pred = pred

    def accuracy(self):
        numer = sum(1 for p,g in zip(self.pred, self.gold) if p in g)
        return numer / len(self.gold)
