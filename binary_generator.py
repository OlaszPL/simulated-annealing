from binary_functions import *
from tqdm import tqdm, trange
import pickle

def load_images(filename):
    """
    Wczytuje listę images z pliku w formacie pickle.
    
    Args:
        filename (str): Ścieżka do pliku, z którego dane zostaną wczytane.
    
    Returns:
        list: Lista 3 list, które zawierają macierze np.ndarray.
    """
    with open(filename, 'rb') as f:
        images = pickle.load(f)
    print(f"Images zostały wczytane z pliku: {filename}")
    return images

# -------------

max_iter = 500000
break_cnt = 1000
eps = 1e-6
init_temps = [100, 1000]
a = 0.9999
temp = exponentional_temperature
energies = [
    (energy_ising_4neighbors, update_energy_ising_4neighbors), 
    (energy_ising_8neighbors, update_energy_ising_8neighbors),
    # (energy_ising_16neighbors, update_energy_ising_16neighbors),
    # (energy_custom, update_energy_custom),
    # (energy_diagonal, update_energy_diagonal),
    # (energy_diagonal_proximity, update_energy_diagonal_proximity),
    # (energy_rotated_square, update_energy_rotated_square)
            ]


N = [128, 256, 1024]
deltas = [0.1, 0.3, 0.4]
# images = [
#     [generate_binary_image(i, delta) for i in N] for delta in deltas
# ]

filename = "images_data.pkl"
images = load_images(filename)

for energy, update in tqdm(energies):
    for i in trange(len(deltas)):
        for img in tqdm(images[i]):
            n = len(img)
            for T0 in tqdm(init_temps):
                full_name = f"{energy.__name__}_d={deltas[i]}_N={n}_{temp.__name__}_T0={T0}_a={a}"
                os.makedirs(os.path.dirname(f"./BIN_IMG/{energy.__name__}/"), exist_ok = True)
                path = f"./BIN_IMG/{energy.__name__}/{full_name}"

                best_energy, best_state, energies, temperatures = simulated_annealing_mod(
                    img, energy, update, apply_changes_pic, temp, neighbour_flip_pairs_pic, T0,
                    max_iter, eps, break_cnt, a, path, f"Tworzenie obrazu binarnego dla:\n{full_name}")

                plot_saoptimset(energies, temperatures, full_name, f"{temp.__name__}\nT0: {T0}, a: {a}", path)