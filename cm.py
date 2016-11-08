from metrics_mc import *
import numpy as np
import sys


def main():
    z = np.loadtxt(sys.argv[1])
    print "GC^2: %f" % gcsq(z)
    print "Gorodkin: %f" % gorodkin(z)
    print "Kappa: %f" % kappa(z)
    print "IC: %f" % IC(z)

if __name__ == '__main__':
    main()
