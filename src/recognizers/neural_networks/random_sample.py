import argparse

import numpy
import scipy.stats

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--log', action='store_true', default=False)
    parser.add_argument('--int', action='store_true', default=False)
    parser.add_argument('min', type=float)
    parser.add_argument('max', type=float)
    args = parser.parse_args()

    if args.log:
        result = scipy.stats.loguniform(args.min, args.max).rvs()
        if args.int:
            result = round(result)
    elif args.int:
        result = numpy.random.randint(round(args.min), round(args.max)+1)
    else:
        result = numpy.random.uniform(args.min, args.max)
    print(result)

if __name__ == '__main__':
    main()
