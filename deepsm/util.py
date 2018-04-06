import os
import math
import random
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.lines as lines

##########Convenient##########
def sure_add(dictionary, key, value):
    """Adds value to the set dictionary[key]. If key does
    not exist in dictionary, create it. If value is None
    or key is None, do nothing"""
    if value is None or key is None:
        return
    if key in dictionary:
        dictionary[key].add(value)
    else:
        dictionary[key] = set({value})


##########Topo-map related##########
def compute_view_number(node, neighbor, divisions=8):
    """
    Assume node and neighbor have the 'pose' attribute. Return an integer
    within [0, divisions-1] indicating the view number of node that is
    connected to neighbor.
    """
    x, y = node.anchor_pose[0], node.anchor_pose[1]
    nx, ny = neighbor.anchor_pose[0], neighbor.anchor_pose[1]
    angle_rad = math.atan2(ny-y, nx-x)
    if angle_rad < 0:
        angle_rad = math.pi*2 - abs(angle_rad)
    view_number = int(math.floor(angle_rad / (2*math.pi / divisions)))  # floor division
    return view_number


def abs_view_distance(v1, v2, num_divisions=8):
    return min(abs(v1-v2), num_divisions-abs(v1-v2))


################# Colors ##################
def linear_color_gradient(rgb_start, rgb_end, n):
    colors = [rgb_start]
    for t in range(1, n):
        colors.append(tuple(
            rgb_start[i] + float(t)/(n-1)*(rgb_end[i] - rgb_start[i])
            for i in range(3)
        ))
    return colors


def rgb_to_hex(rgb):
    r,g,b = rgb
    return '#%02x%02x%02x' % (int(r), int(g), int(b))

def hex_to_rgb(hx):
    """hx is a string, begins with #. ASSUME len(hx)=7."""
    if len(hx) != 7:
        raise ValueError("Hex must be #------")
    hx = hx[1:]  # omit the '#'
    r = int('0x'+hx[:2], 16)
    g = int('0x'+hx[2:4], 16)
    b = int('0x'+hx[4:6], 16)
    return (r,g,b)

def inverse_color_rgb(rgb):
    r,g,b = rgb
    return (255-r, 255-g, 255-b)

def inverse_color_hex(hx):
    """hx is a string, begins with #. ASSUME len(hx)=7."""
    return inverse_color_rgb(hex_to_rgb(hx))

def random_unique_color(colors, ctype=1):
    """
    ctype=1: completely random
    ctype=2: red random
    ctype=3: blue random
    ctype=4: green random
    ctype=5: yellow random
    """
    if ctype == 1:
        color = "#%06x" % random.randint(0x444444, 0x999999)
        while color in colors:
            color = "#%06x" % random.randint(0x444444, 0x999999)
    elif ctype == 2:
        color = "#%02x0000" % random.randint(0xAA, 0xFF)
        while color in colors:
            color = "#%02x0000" % random.randint(0xAA, 0xFF)
    elif ctype == 4:  # green
        color = "#00%02x00" % random.randint(0xAA, 0xFF)
        while color in colors:
            color = "#00%02x00" % random.randint(0xAA, 0xFF)
    elif ctype == 3:  # blue
        color = "#0000%02x" % random.randint(0xAA, 0xFF)
        while color in colors:
            color = "#0000%02x" % random.randint(0xAA, 0xFF)
    elif ctype == 5:  # yellow
        h = random.randint(0xAA, 0xFF)
        color = "#%02x%02x00" % (h, h)
        while color in colors:
            h = random.randint(0xAA, 0xFF)
            color = "#%02x%02x00" % (h, h)
    else:
        raise ValueError("Unrecognized color type %s" % (str(ctype)))
    return color


################# Plotting ##################
def plot_dot(ax, rx, ry, map_spec, img, color='blue', dotsize=2, fill=True, zorder=0, linewidth=0, edgecolor=None):
    px, py = transform_coordinates(rx, ry, map_spec, img)
    very_center = plt.Circle((px, py), dotsize, facecolor=color, fill=fill, zorder=zorder, linewidth=linewidth, edgecolor=edgecolor)
    ax.add_artist(very_center)
    return px, py


def plot_line(ax, g1, g2, map_spec, img, linewidth=1, color='black', zorder=0):
    # g1, g2 are two points with gmapping coordinates
    p1x, p1y = transform_coordinates(g1[0], g1[1], map_spec, img)
    p2x, p2y = transform_coordinates(g2[0], g2[1], map_spec, img)
    ax = plt.gca()
    line = lines.Line2D([p1x, p2x], [p1y, p2y], linewidth=linewidth, color=color, zorder=zorder)
    ax.add_line(line)



def zoom_plot(p, img, ax, zoom_level=0.35):
    # Zoom by setting limits. Center around p
    px, py = p
    h, w = img.shape
    sidelen = min(w*zoom_level*0.2, h*zoom_level*0.2)
    ax.set_xlim(px - sidelen/2, px + sidelen/2)
    ax.set_ylim(py - sidelen/2, py + sidelen/2)


def zoom_rect(p, img, ax, h_zoom_level=0.35, v_zoom_level=0.35):
    # Zoom by setting limits
    px, py = p
    h, w = img.shape
    xsidelen = w*h_zoom_level*0.2
    ysidelen = h*v_zoom_level*0.2
    ax.set_xlim(px - xsidelen/2, px + xsidelen/2)
    ax.set_ylim(py - ysidelen/2, py + ysidelen/2)


def print_banner(text, ch='=', length=78):
    """Source: http://code.activestate.com/recipes/306863-printing-a-bannertitle-line/"""
    spaced_text = ' %s ' % text
    banner = spaced_text.center(length, ch)
    print(banner)


def print_in_box(msgs, ho="=", vr="||"):
    max_len = 0
    for msg in msgs:
        max_len = max(len(msg), max_len)
    print(ho*(max_len+2*(len(vr)+1)))
    for msg in msgs:
        print(vr + " " + msg + " " + vr)
    print(ho*(max_len+2*(len(vr)+1)))



class CategoryManager:

    # Invariant: categories have an associated number. The number
    #            goes from 0 to NUM_CATEGORIES - 1

    # SKIP_UNKNOWN is True if we want to SKIP unknown classes, which means all remaining
    # classes should be `known` and there is no 'UN' label.
    SKIP_UNKNOWN = True
    # TYPE of label mapping scheme. Offers SIMPLE and FULL.
    TYPE = 'FULL'
    
    # Category mappings
    CAT_MAP_ALL = {
        'SIMPLE': {
            'FW': {
                # DW = 0; CR = 1; 1PO = 2; 2PO = 3; UN = 4
                'OC': -1,  # occluded
                'DW': 0,
                'CR': 1,
                'PT' : 1,
                '1PO': 2,
                '2PO': 3,
                'MPO': 3,
                'PRO': 3,
                'UN': 4
            },
            'BW': {
                -1: 'OC',
                0: 'DW',
                1: 'CR',
                2: '1PO',
                3: '2PO',
                4: 'UN'
            },
            'CL': {
                -1: '#000000',
                0: '#00FFFF', # cyan
                1: '#FF0000',
                2: '#0000FF',
                3: '#00FF00',
                4: '#FFFFFF',
            }
        },
        'FULL': {
            'FW': {
                'OC': -1,
                'DW': 0,   #               (DW)
                'CR': 1,   # corridor      (CR)
                'AT': 1,   # anteroom      (CR)
                'ST': 1,   # stairs        (CR)
                'EV': 1,   # elevator      (CR)
                '1PO': 2,  #               (1PO)
                '2PO': 3,  #               (2PO)
                'PRO': 3,  #               (2PO)
                'MPO': 4,  #               (LO)
                'LO': 4,   # large office  (LO)
                'LAB': 5,  # --lab---LAB-------
                'RL': 5,   # robot lab     (LAB)
                'LR': 5,   # living room   (LAB)
                'TR': 5,   # terminal room (LAB)
                'BA': 6,   # bathroom      (BA)
                'KT': 7,   # kitchen       (KT)
                'MR': 8,   # meeting room  (MR)
                'LMR': 8,  # lg meeting rm (MR)
                'CF': 8,   # conference rm (MR)
                'UT': 9,   # --utility-UT-----
                'WS': 9,   # workshop      (UT)
                'PT': 9,   # printer room  (UT)
                'PA': 9,   # -same as PT-  (UT)
                'UN': 10
            },
            'BW': {
                -1: 'OC',
                0: 'DW',
                1: 'CR',
                2: '1PO',
                3: '2PO',
                4: 'LO',
                5: 'LAB',
                6: 'BA',
                7: 'KT',
                8: 'MR',
                9: 'UT',  # utility
                10: 'UN'
            },
            'CL': {
                -1: '#000000', # OC
                0: '#f6f23d',
                1: '#f08055',
                2: '#40d9ff',
                3: '#48ff80',
                4: '#59be3c',
                5: '#f2b6c6', 
                6: '#7bcf99',
                7: '#d99188',
                8: '#f2e1ea',
                9: '#cf22ef',
                10: '#FFFFFF' # UN
            }
        }
    }

    CAT_MAP = CAT_MAP_ALL[TYPE]['FW']
    CAT_REV_MAP = CAT_MAP_ALL[TYPE]['BW']
    CAT_COLOR = CAT_MAP_ALL[TYPE]['CL']
    PLACEHOLDER_COLOR = '#b8b8b8'

    if SKIP_UNKNOWN:
        NUM_CATEGORIES = len(CAT_COLOR) - 2  # exclude '-1' and catg_num('UN')
    else:
        NUM_CATEGORIES = len(CAT_COLOR) - 1  # exclude '-1'.
    
    @staticmethod
    def category_map(category, rev=False, checking=False):
        """
        Return a number for a given category

        `rev` is True if want to return a category string for a given number
        `checking` is true if it is allowed to return 'UN' or its number, ignoring what is set
                   in SKIP_UNKNOWN. Namely, if SKIP_UNKNOWN is True, this function won't return
                   'UN' or its number, unless `checking` is True. This is useful for checking
                   if some node should be skipped when loading from the database.
        """
        # If we 'skip unknown', and the passed in `category` is 'unknown' to our
        # category mapping, then the category map is missing something. Throw
        # an exception.
        if not checking and CategoryManager.SKIP_UNKNOWN:
            if not rev:  # str -> num
                if category not in CategoryManager.CAT_MAP or CategoryManager.CAT_MAP[category] == CategoryManager.CAT_MAP['UN']:
                    raise ValueError("Unknown category: %s" % category)
            else:  # num -> str
                if category not in CategoryManager.CAT_REV_MAP or CategoryManager.CAT_REV_MAP[category] == 'UN':
                    raise ValueError("Unknown category number: %d" % category)
        if not rev:
            if category in CategoryManager.CAT_MAP:
                return CategoryManager.CAT_MAP[category]
            else:
                if not checking:
                    assert CategoryManager.SKIP_UNKNOWN == False
                return CategoryManager.CAT_MAP['UN']
        else:
            if category in CategoryManager.CAT_REV_MAP:
                return CategoryManager.CAT_REV_MAP[category]
            else:
                if not checking:
                    assert CategoryManager.SKIP_UNKNOWN == False
                return 'UN'

    @staticmethod
    def canonical_catgegory(catg):
        """
        Returns the canonical category for category `catg`.

        Args:
          `catg` needs to be a string.

        Returns the string abbreviation of the canonical category.
        """
        return CategoryManager.category_map(
            CategoryManager.category_map(
                CategoryManager.category_map(catg),
                rev=True))

    @staticmethod
    def category_color(category):
        if CategoryManager.category_map(category) in CategoryManager.CAT_COLOR:
            return CategoryManager.CAT_COLOR[CategoryManager.category_map(category)]
        else:  # unknown
            return CategoryManager.CAT_COLOR[CategoryManager.category_map('UN')]
