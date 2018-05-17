from gan.Interface import *
from gan.Generator import Generator
from gan.Discriminator import Discriminator

class Adversarial:
    def __init__(self):
        self.structure = Sequential()
        self.structure.add(Generator().structure)
        self.structure.add(Discriminator().structure)