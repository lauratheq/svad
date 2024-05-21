import matplotlib as plt
from itertools import cycle

SAMPLE_RATE		= 22050
BUFFER_SIZE		= 1024
ERROR_MARGIN    = 50 # in percentile
LOG_LEVEL       = 'DEBUG'

COLOR_PAL = plt.rcParams["axes.prop_cycle"].by_key()["color"]
COLOR_CYCLE = cycle(plt.rcParams["axes.prop_cycle"].by_key()["color"])
