
class PolynomialDecay:
    def __init__(self, maxEpochs=100, power=0.9):
        self.maxEpochs = maxEpochs
        self.power = power

    def __call__(self, epoch):
        decay = (1 - (epoch / float(self.maxEpochs))) ** self.power

        return float(decay)
