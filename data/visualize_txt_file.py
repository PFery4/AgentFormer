import os
import numpy as np
import matplotlib.pyplot as plt

if __name__ == '__main__':
    root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "datasets", "eth_ucy", "eth"))
    delimiter = " "
    for file in os.scandir(root_dir):
        file_path = os.path.abspath(file)
        assert os.path.exists(file_path)
        print(f"extracting trajectory data from:\n{file_path}")

        data = np.genfromtxt(file_path, delimiter=delimiter, dtype=str)
        print(data)
        x_ind = 13
        z_ind = 15

        fig, ax = plt.subplots()

        xs = data[:, x_ind].astype(np.float32)
        print(xs)
        zs = data[:, z_ind].astype(np.float32)
        print(zs)

        ax.scatter(xs, zs, c="r", alpha=0.5)
        ax.set_aspect('equal', 'box')
        plt.show()
