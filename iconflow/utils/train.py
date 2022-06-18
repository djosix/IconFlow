import random

def infinite_loop(iterable):
    while True:
        yield from iterable

def random_sampler(size):
    while True:
        yield random.randrange(size)
