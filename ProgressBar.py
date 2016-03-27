import sys

class ProgressBar:
    def __init__(self,total,barlength=50,symbol = "#"):
        self.total = total
        self.barlength = barlength
        self.symbol = symbol
        self.counter = 0

    def update(self):
        self.counter += 1
        percent = float(self.counter) / self.total
        hashes = self.symbol * int(round(percent * self.barlength))
        spaces = ' ' * (self.barlength - len(hashes))
        sys.stdout.write("\rPercent: [{0}] {1}%".format(hashes + spaces, int(round(percent * 100))))
        sys.stdout.flush()