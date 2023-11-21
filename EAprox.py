import numpy as np
from math import fabs
import matplotlib.pyplot as plt
from numpy import linalg as LA

def GregsCircles(matrix, eigvals):
    if not isSquare(matrix):
        print('Your input matrix is not square!')
        return []
    
    circles = []
    for x in range(0, len(matrix)):
        radius = 0
        piv = matrix[x][x]
        for y in range(0, len(matrix)):
            if x != y:
                radius += fabs(matrix[x][y])
        circles.append([piv, radius])
    
    return circles

def plotCircles(circles, eigvals):
    index, radi = zip(*circles)
    Xupper = max(index) + np.std(index)
    Xlower = min(index) - np.std(index)
    Ylimit = max(radi) + np.std(index)
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6), gridspec_kw={'height_ratios': [1, 6]})
    
    eigenvalue_text = '\n'.join([fr'$\lambda_{i} = {eigval}$' for i, eigval in enumerate(eigvals, start=1)])
    ax1.text(0.5, 0.5, f'Eigenvalues:\n{eigenvalue_text}', va='center', ha='center', fontsize=12)
    ax1.axis('off')
    
    ax2.set_xlim((Xlower, Xupper))
    ax2.set_ylim((-Ylimit, Ylimit))
    ax2.set_xlabel('Real')
    ax2.set_ylabel('Imaginary')

    for i, (eigval, label) in enumerate(zip(eigvals, [fr'$\lambda_{i}$' for i in range(1, len(eigvals) + 1)])):
        circ = plt.Circle((index[i], 0), radius=radi[i], edgecolor='black', facecolor='none', linewidth=1)
        ax2.add_artist(circ)
        ax2.plot(eigval.real, eigval.imag, 'ro')
        ax2.annotate(label, (eigval.real, eigval.imag), textcoords="offset points", xytext=(0,10), ha='center')

    ax2.plot([Xlower, Xupper], [0, 0], 'k--')
    ax2.plot([0, 0], [-Ylimit, Ylimit], 'k--')
    
    plt.tight_layout()
    fig.savefig('plotcircles.png')
    plt.show()

def isSquare(m):
    cols = len(m)
    for row in m:
        if len(row) != cols:
            return False
    return True

def main():
    matrix = np.array([[2, 0.1, -0.5], [0.2, 3, 0.5], [-0.4, 0.1, 5]]) # enter your matrix
    eigvals = LA.eigvals(matrix)
    temp = GregsCircles(matrix, eigvals)
    plotCircles(temp, eigvals)

if __name__ == '__main__':
    main()
