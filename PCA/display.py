import math
import numpy as np
import matplotlib.pyplot as plt

def displayData(X, example_width=None):
    if example_width == None:
        example_width = round(math.sqrt(X.shape[1]))

    m, n = X.shape
    example_height = (n//example_width)

    display_rows = math.floor(math.sqrt(m))
    display_cols = math.ceil(m//display_rows)
    pad = 1

    display_mtx = -np.ones((pad + display_rows * (example_height + pad), pad + display_cols * (example_width + pad)))

    curr_ex = 0
    for i in range(display_rows):
        for j in range(display_cols):
            if curr_ex >= m:
                break
            maxval = np.max(np.abs(X[i,:]))
            row_add = pad + (i * (pad + example_height))
            col_add = pad + (j * (pad + example_width))
            display_mtx[(row_add + 0):(row_add + example_height),
                        (col_add + 0):(col_add + example_width)] = X[curr_ex,:].reshape(example_height, example_width, order = 'F') / maxval
            curr_ex += 1
            if curr_ex >= m:
                break
            '''
            .mat image data can be ordered in fortran style. -> that's why I put some order option in there
            '''

    #display_mtx += 0.5
    #display_mtx /= 2.0

    plt.imshow(display_mtx, cmap='gray')