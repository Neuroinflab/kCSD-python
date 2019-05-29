# Based on
# Color Universal Design (CUD)
# - How to make figures and presentations that are friendly to Colorblind people
#
#
# Masataka Okabe
# Jikei Medial School (Japan)
#
# Kei Ito
# University of Tokyo, Institute for Molecular and Cellular Biosciences (Japan)
# (both are strong protanopes)
# 11.20.2002 (modified on 2.15.2008, 9.24.2008)
# http://jfly.iam.u-tokyo.ac.jp/color/#pallet

import collections
from matplotlib import colors


_Color = collections.namedtuple('_Color', ['red', 'green', 'blue'])

def _html(r, g, b):
    return "#{:02X}{:02X}{:02X}".format(r, g, b)


_BLACK     = _Color(  0,   0,   0)
_ORANGE    = _Color(230, 159,   0)
_SKY_BLUE  = _Color( 86, 180, 233)
_GREEN     = _Color(  0, 158, 115)
_YELLOW    = _Color(240, 228,  66)
_BLUE      = _Color(  0, 114, 178)
_VERMILION = _Color(213,  94,   0)
_PURPLE    = _Color(204, 121, 167)

BLACK     = _html(*_BLACK)
ORANGE    = _html(*_ORANGE)
SKY_BLUE  = _html(*_SKY_BLUE)
GREEN     = _html(*_GREEN)
YELLOW    = _html(*_YELLOW)
BLUE      = _html(*_BLUE)
VERMILION = _html(*_VERMILION)
PURPLE    = _html(*_PURPLE)

def _BipolarColormap(name, negative, positive):
    return colors.LinearSegmentedColormap(
                      name,
                      {k: [(0.0,) + (getattr(negative, k) / 255.,) * 2,
                           (0.5, 1.0, 1.0),
                           (1.0,) + (getattr(positive, k) / 255.,) * 2,]
                       for k in ['red', 'green', 'blue']})

bwr = _BipolarColormap('cbf.bwr', _BLUE, _VERMILION)
PRGn = _BipolarColormap('cbf.PRGn', _PURPLE, _GREEN)