import numpy as np
import os
from matplotlib import gridspec
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import imageio.v2 as imageio

def P(prev, next, T):
    return np.exp((prev - next) / T) if T > 0 else 0

def exponentional_temperature(init_temp, a, i):
    return init_temp * (a ** i)

def plot_saoptimset(energies, temperatures, name: str, temp_parameters: str, path: str):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))  # Zwiększono szerokość z 8 do 10
    
    # Wykres wartości funkcji celu
    ax1.plot(energies)
    ax1.set_title(f'Wartość energii od iteracji dla:\n{name}')
    ax1.set_xlabel('Liczba iteracji')
    ax1.set_ylabel('Energia')
    
    # Wykres temperatury
    ax2.plot(temperatures)
    ax2.set_title(f'Temperatura\n {temp_parameters}')
    ax2.set_xlabel('Liczba iteracji')
    ax2.set_ylabel('Temperatura')
    
    plt.tight_layout()
    plt.savefig(f'{path}_saoptimset.png')
    plt.close()


def generate_binary_image(n, delta):
    img = np.random.random((n, n))
    return np.where(img < delta, 0, 255).astype(np.uint8) # tam gdzie wylosowana wartość jest mniejsza od delty to wstawiamy czarny,
    # w przeciwnym przypadku biały


def energy_ising_4neighbors(img: np.ndarray):
    """Energia z pełnym sąsiedztwem 4-kierunkowym (von Neumanna) (liczona raz)"""
    energy = 0
    rows, cols = img.shape
    moves = ((-1,0), (1,0), (0,-1), (0,1))
    for i in range(rows):
        for j in range(cols):
            for di, dj in moves:
                ni, nj = i + di, j + dj
                if -1 < ni < rows and -1 < nj < cols:
                    energy -= 1 if img[i,j] == img[ni,nj] else -1
    return energy / 2  # Dzielenie przez 2 dla uniknięcia podwójnego liczenia

def update_energy_ising_4neighbors(curr_energy, changes : list, curr_state : np.ndarray):
    """Wylicza nową energię dla zmian (zmiany to lista współrzędnych punktów zamienianych)
    1. Dodajemy/odejmujemy energię dla starej pozycji (w zależności od tego jaka ona jest)
    2. Odejmujemy/dodajemy energię dla symulowanych nowych pozycji.
    3. Na koniec zwracamy nową wartość energii.
    """
    to_subtract = 0
    to_add = 0
    moves = ((-1,0), (1,0), (0,-1), (0,1))
    rows, cols = curr_state.shape

    for ((i1, j1), (i2, j2)) in changes:
        for di, dj in moves:
            # zarządzenie starymi wartościami
            ni1, nj1 = i1 + di, j1 + dj
            ni2, nj2 = i2 + di, j2 + dj
            if -1 < ni1 < rows and -1 < nj1 < cols:
                # jeżeli kolory były takie same to cofamy ich wpływ (oryginalną zmianę dodajemy do to_subtract, którą na koniec odejmiemy)
                to_subtract -= 1 if curr_state[i1,j1] == curr_state[ni1,nj1] else -1
                # analizując nowy symulowany stan zamieniam współrzędne przy porównywnaniu
                to_add -= 1 if curr_state[i2,j2] == curr_state[ni1,nj1] else -1
            if -1 < ni2 < rows and -1 < nj2 < cols:
                to_subtract -= 1 if curr_state[i2,j2] == curr_state[ni2,nj2] else -1
                to_add -= 1 if curr_state[i1,j1] == curr_state[ni2,nj2] else -1

    return curr_energy - (to_subtract / 2) + (to_add / 2)


def energy_ising_8neighbors(img: np.ndarray):
    """Energia z pełnym sąsiedztwem 8-kierunkowym (Moore'a) (liczona raz)"""
    energy = 0
    rows, cols = img.shape
    for i in range(rows):
        for j in range(cols):
            for di in (-1,0,1):
                for dj in (-1,0,1):
                    if di == 0 and dj == 0:
                        continue
                    ni, nj = i + di, j + dj
                    if -1 < ni < rows and -1 < nj < cols:
                        energy -= 1 if img[i,j] == img[ni,nj] else -1
    return energy / 2


def update_energy_ising_8neighbors(curr_energy, changes : list, curr_state : np.ndarray):
    to_subtract = 0
    to_add = 0
    rows, cols = curr_state.shape

    for ((i1, j1), (i2, j2)) in changes:
        for di in (-1,0,1):
            for dj in (-1,0,1):
                if di == 0 and dj == 0:
                    continue
                # zarządzenie starymi wartościami
                ni1, nj1 = i1 + di, j1 + dj
                ni2, nj2 = i2 + di, j2 + dj
                if -1 < ni1 < rows and -1 < nj1 < cols:
                    # jeżeli kolory były takie same to cofamy ich wpływ (oryginalną zmianę dodajemy do to_subtract, którą na koniec odejmiemy)
                    to_subtract -= 1 if curr_state[i1,j1] == curr_state[ni1,nj1] else -1
                    # analizując nowy symulowany stan zamieniam współrzędne przy porównywnaniu
                    to_add -= 1 if curr_state[i2,j2] == curr_state[ni1,nj1] else -1
                if -1 < ni2 < rows and -1 < nj2 < cols:
                    to_subtract -= 1 if curr_state[i2,j2] == curr_state[ni2,nj2] else -1
                    to_add -= 1 if curr_state[i1,j1] == curr_state[ni2,nj2] else -1

    return curr_energy - (to_subtract / 2) + (to_add / 2)


def energy_ising_16neighbors(img: np.ndarray):
    """Energia dla 16-sąsiedztwa (Moore + 2 kroki w kierunkach głównych i diagonalnych)"""
    energy = 0
    rows, cols = img.shape
    
    # Lista wszystkich kierunków sąsiedztwa
    # Sąsiedztwo Moore'a (8 kierunków)
    directions = [
        (-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1),
        (-2, 0), (2, 0), (0, -2), (0, 2), (-2, -2), (-2, 2), (2, -2), (2, 2)
    ]
    
    for i in range(rows):
        for j in range(cols):
            for di, dj in directions:
                ni, nj = i + di, j + dj
                if -1 < ni < rows and -1 < nj < cols:
                    # Przyciąganie dla tych samych kolorów, odpychanie dla różnych
                    energy -= 1 if img[i,j] == img[ni,nj] else -1
    
    return energy / 2  # Dzielimy przez 2 by uniknąć podwójnego liczenia par


def update_energy_ising_16neighbors(curr_energy, changes : list, curr_state : np.ndarray):
    to_subtract = 0
    to_add = 0
    rows, cols = curr_state.shape

    directions = [
        (-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1),
        (-2, 0), (2, 0), (0, -2), (0, 2), (-2, -2), (-2, 2), (2, -2), (2, 2)
    ]

    for ((i1, j1), (i2, j2)) in changes:
        for di, dj in directions:
            # zarządzenie starymi wartościami
            ni1, nj1 = i1 + di, j1 + dj
            ni2, nj2 = i2 + di, j2 + dj
            if -1 < ni1 < rows and -1 < nj1 < cols:
                # jeżeli kolory były takie same to cofamy ich wpływ (oryginalną zmianę dodajemy do to_subtract, którą na koniec odejmiemy)
                to_subtract -= 1 if curr_state[i1,j1] == curr_state[ni1,nj1] else -1
                # analizując nowy symulowany stan zamieniam współrzędne przy porównywnaniu
                to_add -= 1 if curr_state[i2,j2] == curr_state[ni1,nj1] else -1
            if -1 < ni2 < rows and -1 < nj2 < cols:
                to_subtract -= 1 if curr_state[i2,j2] == curr_state[ni2,nj2] else -1
                to_add -= 1 if curr_state[i1,j1] == curr_state[ni2,nj2] else -1

    return curr_energy - (to_subtract / 2) + (to_add / 2)


def energy_custom(img: np.ndarray):
    """Energia z przyciąganiem/odpychaniem i pełnym sąsiedztwem 5x5"""
    energy = 0
    rows, cols = img.shape
    
    for i in range(rows):
        for j in range(cols):
            # Pełne sąsiedztwo 5x5 (wszystkie kierunki)
            for di in range(-2, 3):
                for dj in range(-2, 3):
                    if di == 0 and dj == 0:
                        continue
                    ni, nj = i + di, j + dj
                    if -1 < ni < rows and -1 < nj < cols:
                        d1, d2 = abs(di), abs(dj)
                        distance = d1 if d1 > d2 else d2  # odległość Czebyszewa
                        if distance == 1:  # bezpośredni sąsiedzi (przyciąganie)
                            energy -= 1 if img[i,j] == img[ni,nj] else -1
                        else:  # dalsi sąsiedzi (odpychanie)
                            energy += 1 if img[i,j] == img[ni,nj] else -1
                            
    return energy / 2  # dzielenie przez 2 dla uniknięcia podwójnego liczenia


def update_energy_custom(curr_energy, changes : list, curr_state : np.ndarray):
    to_subtract = 0
    to_add = 0
    rows, cols = curr_state.shape

    for ((i1, j1), (i2, j2)) in changes:
        for di in range(-2, 3):
            for dj in range(-2, 3):
                if di == 0 and dj == 0:
                    continue
                ni1, nj1 = i1 + di, j1 + dj
                ni2, nj2 = i2 + di, j2 + dj
                if -1 < ni1 < rows and -1 < nj1 < cols:
                    d1, d2 = abs(di), abs(dj)
                    distance = d1 if d1 > d2 else d2  # odległość Czebyszewa
                    if distance == 1:  # bezpośredni sąsiedzi (przyciąganie)
                        to_subtract -= 1 if curr_state[i1,j1] == curr_state[ni1,nj1] else -1
                        to_add -= 1 if curr_state[i2,j2] == curr_state[ni1,nj1] else -1
                    else:  # dalsi sąsiedzi (odpychanie)
                        to_subtract += 1 if curr_state[i1,j1] == curr_state[ni1,nj1] else -1
                        to_add += 1 if curr_state[i2,j2] == curr_state[ni1,nj1] else -1
                if -1 < ni2 < rows and -1 < nj2 < cols:
                    d1, d2 = abs(di), abs(dj)
                    distance = d1 if d1 > d2 else d2  # odległość Czebyszewa
                    if distance == 1:  # bezpośredni sąsiedzi (przyciąganie)
                        to_subtract -= 1 if curr_state[i2,j2] == curr_state[ni2,nj2] else -1
                        to_add -= 1 if curr_state[i1,j1] == curr_state[ni2,nj2] else -1
                    else:  # dalsi sąsiedzi (odpychanie)
                        to_subtract += 1 if curr_state[i2,j2] == curr_state[ni2,nj2] else -1
                        to_add += 1 if curr_state[i1,j1] == curr_state[ni2,nj2] else -1

    return curr_energy - (to_subtract / 2) + (to_add / 2)


def energy_diagonal(img: np.ndarray):
    """Energia zależna od zgodności kolorów na przekątnych"""
    energy = 0
    rows, cols = img.shape
    directions = [(-1, -1), (-1, 1), (1, -1), (1, 1)]
    
    # Przechodzimy przez wszystkie piksele
    for i in range(rows):
        for j in range(cols):
            # Sprawdzamy 4 kierunki diagonalne
            for di, dj in directions:
                ni, nj = i + di, j + dj
                # Sprawdź czy sąsiad jest w granicach obrazu
                if -1 < ni < rows and -1 < nj < cols:
                    # Oblicz energię dla pary pikseli
                    if img[i, j] == img[ni, nj]:
                        energy -= 1  # Nagroda za zgodność
                    else:
                        energy += 1  # Kara za niezgodność
                        
    return energy / 2


def update_energy_diagonal(curr_energy, changes : list, curr_state : np.ndarray):
    to_subtract = 0
    to_add = 0
    rows, cols = curr_state.shape
    directions = [(-1, -1), (-1, 1), (1, -1), (1, 1)]

    for ((i1, j1), (i2, j2)) in changes:
        for di, dj in directions:
            ni1, nj1 = i1 + di, j1 + dj
            ni2, nj2 = i2 + di, j2 + dj
            # Sprawdź czy sąsiad jest w granicach obrazu
            if -1 < ni1 < rows and -1 < nj1 < cols:
                # Oblicz energię dla pary pikseli
                to_subtract -= 1 if curr_state[i1,j1] == curr_state[ni1,nj1] else -1
                to_add -= 1 if curr_state[i2,j2] == curr_state[ni1,nj1] else -1
            if -1 < ni2 < rows and -1 < nj2 < cols:
                to_subtract -= 1 if curr_state[i2,j2] == curr_state[ni2,nj2] else -1
                to_add -= 1 if curr_state[i1,j1] == curr_state[ni2,nj2] else -1   

    return curr_energy - (to_subtract / 2) + (to_add / 2)


def energy_diagonal_proximity(img: np.ndarray):
    """Energia maleje, im bliżej przekątnej znajduje się czarny piksel (0)."""
    energy = 0.0
    size = img.shape[0]
    for i in range(size):
        for j in range(size):
            if img[i, j] == 0:  # Tylko czarne piksele
                d_main = abs(i - j)
                d_anti = abs(i + j - (size - 1))
                d = min(d_main, d_anti)
                energy -= 1 / (d + 1)
    return energy


def update_energy_diagonal_proximity(curr_energy: float, changes: list, curr_state: np.ndarray):
    """Aktualizuje energię po zamianie pikseli, uwzględniając odległość od przekątnych."""
    delta = 0.0
    size = curr_state.shape[0]
    
    for ((i1, j1), (i2, j2)) in changes:
        # Pobierz kolory i zamień na int (0 lub 1)
        C1 = int(curr_state[i1, j1] == 0)
        C2 = int(curr_state[i2, j2] == 0)
        
        # Oblicz odległości
        d1 = min(abs(i1 - j1), abs(i1 + j1 - (size - 1)))
        d2 = min(abs(i2 - j2), abs(i2 + j2 - (size - 1)))
        
        # Oblicz zmianę energii
        delta += (C2 - C1) * (1/(d2 + 1) - 1/(d1 + 1))
    
    return curr_energy + delta


# ====================== funkcja specjalna

_MAIN_OFFSETS_TUPLES = ((-3, 0), (3, 0), (0, -3), (0, 3))
_DIAG_OFFSETS_TUPLES = ((-2, -2), (-2, 2), (2, -2), (2, 2))
_ALL_OFFSETS_TUPLES = _MAIN_OFFSETS_TUPLES + _DIAG_OFFSETS_TUPLES

def _get_color(r: int, c: int, state: np.ndarray, changes_dict: dict) -> int:
    return changes_dict.get((r, c), state[r, c])

def energy_rotated_square(img: np.ndarray) -> float:
    """
    Energia zmniejsza się dla wzorców: piksel centralny zgodny z 4 głównymi sąsiadami (odległość 3)
    oraz 4 sąsiadami diagonalnymi (odległość 2).
    """
    energy = 0.0
    rows, cols = img.shape

    if rows < 7 or cols < 7:
        return 0.0
    
    for i in range(3, rows - 3):
        for j in range(3, cols - 3):
            current_color = img[i, j]
            main_count = 0
            diag_count = 0

            main_count = sum(img[i + di, j + dj] == current_color for di, dj in [(-3, 0), (3, 0), (0, -3), (0, 3)])

            diag_count = sum(img[i + di, j + dj] == current_color for di, dj in [(-2, -2), (-2, 2), (2, -2), (2, 2)])

            if main_count == 4:
                 energy += -20.0 if diag_count == 4 else -5.0
            elif diag_count == 4:
                 energy += -5.0
            elif main_count > 0 or diag_count > 0:
                 energy += -1.0

    return energy


def update_energy_rotated_square(curr_energy, changes : list, curr_state: np.ndarray):
    rows, cols = curr_state.shape
    affected_centers = set()

    pixel_coords_changed = set()
    temp_changes_dict = {}
    for (r1, c1), (r2, c2) in changes:
        pixel_coords_changed.add((r1, c1))
        pixel_coords_changed.add((r2, c2))
        if (r1, c1) not in temp_changes_dict:
             temp_changes_dict[(r1, c1)] = curr_state[r2, c2]
        if (r2, c2) not in temp_changes_dict:
             temp_changes_dict[(r2, c2)] = curr_state[r1, c1]


    min_rc, max_r, max_c = 3, rows - 3, cols - 3
    for r, c in pixel_coords_changed:
        if min_rc <= r < max_r and min_rc <= c < max_c:
            affected_centers.add((r, c))

        for dr, dc in _ALL_OFFSETS_TUPLES:
            nr, nc = r - dr, c - dc
            if min_rc <= nr < max_r and min_rc <= nc < max_c:
                 affected_centers.add((nr, nc))

    delta_energy = 0.0

    for r, c in affected_centers:
        old_main_count = 0
        old_diag_count = 0
        center_color_old = curr_state[r, c]

        old_main_count = sum(curr_state[r + dr, c + dc] == center_color_old for dr, dc in [(-3, 0), (3, 0), (0, -3), (0, 3)])

        old_diag_count = sum(curr_state[r + dr, c + dc] == center_color_old for dr, dc in [(-2, -2), (-2, 2), (2, -2), (2, 2)])

        old_contrib = 0.0
        if old_main_count == 4:
            old_contrib = -20.0 if old_diag_count == 4 else -5.0
        elif old_diag_count == 4:
            old_contrib = -5.0
        elif old_main_count > 0 or old_diag_count > 0:
            old_contrib = -1.0

        new_main_count = 0
        new_diag_count = 0
        center_color_new = _get_color(r, c, curr_state, temp_changes_dict)

        new_main_count = sum(
            _get_color(r + dr, c + dc, curr_state, temp_changes_dict) == center_color_new
            for dr, dc in [(-3, 0), (3, 0), (0, -3), (0, 3)]
        )

        new_diag_count = sum(
            _get_color(r + dr, c + dc, curr_state, temp_changes_dict) == center_color_new
            for dr, dc in [(-2, -2), (-2, 2), (2, -2), (2, 2)]
        )

        new_contrib = 0.0
        if new_main_count == 4:
            new_contrib = -20.0 if new_diag_count == 4 else -5.0
        elif new_diag_count == 4:
            new_contrib = -5.0
        elif new_main_count > 0 or new_diag_count > 0:
            new_contrib = -1.0

        delta_energy += (new_contrib - old_contrib)

    return curr_energy + delta_energy


# funkcja losująca naiwnie punkty do wymiany

def neighbour_flip_pairs_pic(curr_state: np.ndarray, num=5):
    n = len(curr_state)
    used = set()
    changes = []

    for _ in range(num):
        p1 = (np.random.randint(0, n), np.random.randint(0, n))
        while p1 in used:
            p1 = (np.random.randint(0, n), np.random.randint(0, n))
        used.add(p1)
        p2 = (np.random.randint(0, n), np.random.randint(0, n))
        while p2 in used:
            p2 = (np.random.randint(0, n), np.random.randint(0, n))
        
        changes.append((p1, p2))

    return changes


def apply_changes_pic(curr_state : np.ndarray, changes : list):
    for ((i1, j1), (i2, j2)) in changes:
        tmp = curr_state[i1, j1]
        curr_state[i1, j1] = curr_state[i2, j2]
        curr_state[i2, j2] = tmp


def create_animation(figsize=(12, 7), title="Tworzenie obrazu binarnego"):
    fig = plt.figure(figsize=figsize)
    gs = gridspec.GridSpec(1, 3, width_ratios=[1, 4, 1])

    # Panel informacyjny po lewej
    info_ax = plt.subplot(gs[0])
    info_ax.axis('off')

    # Obraz w środku
    img_ax = plt.subplot(gs[1])

    # Panel postępu po prawej
    progress_ax = plt.subplot(gs[2])
    progress_ax.axis('off')

    img_ax.set_title(title, fontsize=14)
    img_ax.set_xticks([])
    img_ax.set_yticks([])

    return fig, img_ax, info_ax, progress_ax

def update_animation(frame_idx, img_obj, info_text, progress_text, state, energy, initial_energy, total_frames, max_frames):
    improvement = ((initial_energy - energy) / abs(initial_energy) * 100) if initial_energy != 0 else 0
    img_obj.set_data(state)
    info_text.set_text(f"Energia: {energy:.2f}\nPoprawa: {improvement:.1f}%\n\nKlatka: {frame_idx+1}/{max_frames}")
    progress_text.set_text(f"{frame_idx+1}/{total_frames} (klatka {frame_idx+1}/{max_frames})")
    return img_obj, info_text, progress_text

def finalize_animation(fig, anim, save_path, interval):
    anim.save(f'{save_path}_animation.gif', writer='pillow', fps=1000/interval, dpi=100)
    plt.savefig(f'{save_path}_result.png')
    plt.close(fig)


def simulated_annealing_mod(S, E, update_E, apply_changes, temperature, neighbour, init_temp, max_iter, eps, break_cnt, a, save_path, title, P=P):
    curr_state, best_state = S.copy(), S.copy()
    curr_energy = E(S)
    best_energy = curr_energy
    diff_cnt = 0
    prev_energy = float('inf')

    # Historia wartości energii i temperatury
    energies = [curr_energy]
    temperatures = [init_temp]

    # Tworzenie animacji
    fig, img_ax, info_ax, progress_ax = create_animation(title=title)
    img_obj = img_ax.imshow(curr_state, cmap='gray', interpolation='nearest')
    info_text = info_ax.text(0.5, 0.5, f"Energia: {curr_energy:.2f}\nPoprawa: 0.0%", fontsize=12, va='center', ha='center', transform=info_ax.transAxes, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))
    progress_text = progress_ax.text(0.5, 0.5, f"1/{max_iter} (klatka 1)", ha='center', va='center', fontsize=10, transform=progress_ax.transAxes, bbox=dict(boxstyle='round', facecolor='white', alpha=0.5))

    # Otwórz GIF writer
    with imageio.get_writer(f"{save_path}_animation.gif", mode='I', fps=8, loop = 0) as gif_writer:
        j = 1
        for i in range(max_iter):
            changes = neighbour(curr_state)
            energy = update_E(curr_energy, changes, curr_state)

            T = temperature(init_temp, a, i)
            temperatures.append(T)

            if energy < curr_energy:
                curr_energy = energy
                apply_changes(curr_state, changes)
                if curr_energy < best_energy:
                    best_energy = curr_energy
                    best_state = curr_state.copy()
            elif np.random.uniform() < P(curr_energy, energy, T):
                curr_energy = energy
                apply_changes(curr_state, changes)

            energies.append(curr_energy)

            if abs(prev_energy - curr_energy) < eps:
                diff_cnt += 1
                if diff_cnt == break_cnt:
                    break
            else:
                diff_cnt = 0

            prev_energy = curr_energy

            # Dodaj klatkę co `interval` iteracji
            if i % (max_iter // 100) == 0 or i == max_iter - 1:
                # Aktualizuj wskaźniki energii i klatek
                improvement = ((energies[0] - curr_energy) / abs(energies[0]) * 100) if energies[0] != 0 else 0
                info_text.set_text(f"Energia: {curr_energy:.2f}\nPoprawa: {improvement:.1f}%")
                progress_text.set_text(f"Iteracja: {i + 1}/{max_iter}\nKlatka: {j}")

                img_obj.set_data(curr_state)
                fig.canvas.draw()
                frame = np.frombuffer(fig.canvas.tostring_argb(), dtype='uint8').reshape(fig.canvas.get_width_height()[::-1] + (4,))
                frame = frame[:, :, [1, 2, 3]]  # Konwersja z ARGB na RGB
                gif_writer.append_data(frame)
                j += 1

    plt.savefig(f'{save_path}_result.png')
    plt.close(fig)
    return best_energy, best_state, energies, temperatures