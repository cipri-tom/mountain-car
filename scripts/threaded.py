# because multiprocessing Pool cannot run in interactive mode, we put it in a file

from multiprocessing import Pool
from starter import Agent

def make_and_run(size):
    a = Agent(side_size=size)
    for e in range(60):
        a.episode()
        # if 5 times in a row we escape in less than 150 steps
        if e > 5 and sum(a.escape_times[-5:]) / 5 < 150:
            break
    return [size, e]

def main():
    p = Pool(6)
    episodes_per_size = []

    # 5 agents of each size from 6 -> 11
    for res in p.map(make_and_run, sorted(list(range(7,12)) * 5)):
        episodes_per_size.append(res)

    print(episodes_per_size)

if __name__ == "__main__":
    main()

