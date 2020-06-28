import math
from flask import Flask, request
import numpy as np
import matplotlib.pyplot
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon

def Area(corners):
    n = len(corners) # of corners
    area = 0.0
    for i in range(n):
        j = (i + 1) % n
        area += corners[i][0] * corners[j][1]
        area -= corners[j][0] * corners[i][1]
    area = abs(area) / 2.0
    return area

def distance(p1, p2):
    return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

def Perimeter(corners):
    perimeter = 0
    points = corners + [corners[0]]
    print(points)
    for i in range(len(corners)):
        perimeter += distance(points[i], points[i+1])
    return perimeter

def rgb2gray(rgb):
    r, g, b = rgb[:,:,0], rgb[:,:,1], rgb[:,:,2]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
    return gray

def fractal_dimension(Z, threshold=0.9):
    # Only for 2d image
    assert(len(Z.shape) == 2)

    # From https://github.com/rougier/numpy-100 (#87)
    def boxcount(Z, k):
        S = np.add.reduceat(
            np.add.reduceat(Z, np.arange(0, Z.shape[0], k), axis=0),
                               np.arange(0, Z.shape[1], k), axis=1)

        # We count non-empty (0) and non-full boxes (k*k)
        return len(np.where((S > 0) & (S < k*k))[0])

    # Transform Z into a binary array
    Z = (Z < threshold)

    # Minimal dimension of image
    p = min(Z.shape)

    # Greatest power of 2 less than or equal to p
    n = 2**np.floor(np.log(p)/np.log(2))

    # Extract the exponent
    n = int(np.log(n)/np.log(2))

    # Build successive box sizes (from 2**n down to 2**1)
    sizes = 2**np.arange(n, 1, -1)

    # Actual box counting with decreasing size
    counts = []
    for size in sizes:
        counts.append(boxcount(Z, size))

    # Fit the successive log(sizes) with log (counts)
    coeffs = np.polyfit(np.log(sizes), np.log(counts), 1)
    return -coeffs[0]

def convert_points_image(points):
    pts = np.array(points)
    p = Polygon(pts, closed=False)
    ax = plt.gca()
    ax.add_patch(p)
    ax.set_xlim(0, 500)
    ax.set_ylim(0, 500)

    # make axis invisible
    ax.spines['bottom'].set_linewidth(0)
    ax.spines['left'].set_linewidth(0)
    ax.spines['right'].set_linewidth(0)
    ax.spines['top'].set_linewidth(0)

    # remove the tick
    ax.set_xticks([])
    ax.set_yticks([])

    plt.savefig('test.png')


app = Flask(__name__)

@app.route('/', methods = ["POST"])
def demo():
    body =request.json
    convert_points_image(body)
    image = rgb2gray(matplotlib.pyplot.imread("test.png"))
    FR = fractal_dimension(image)
    return {"fractal dimension": FR}

app.run()