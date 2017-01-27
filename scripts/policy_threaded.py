# because multiprocessing Pool cannot run in interactive mode, we put it in a file

from multiprocessing import Pool
from starter import Agent
import numpy as np

def make_and_run(init_w):
    d = 8
    w = np.full((3,d,d), init_w)
    a = Agent(side_size=d, weights=w)
    for e in range(50):
        a.episode()
    return a.escape_times

def main():
    p = Pool(6)
    # init_w -- 12 examples of each
    # agents = [0] * 12
    agents = [0] * 12
    results = []

    for res in p.map(make_and_run, agents):
        results.append(res)

    print(results)

if __name__ == "__main__":
    main()

