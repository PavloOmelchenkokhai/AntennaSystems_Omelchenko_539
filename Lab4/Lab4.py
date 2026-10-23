import matplotlib.pyplot as plt
import numpy as np

# ВХІДНІ ДАНІ (ВАРІАНТ № 13)
lambd = 4.45  # Довжина хвилі, см
l = 23.7  # Довжина стрижня, см
h = 5.0  # Відстань між стрижнями, см

# Коефіцієнт уповільнення для ДСА (виправлено синтаксис)
xi = 1.0 + lambd / (2.0 * l)  # xi ≈ 1.0939

# Діапазон кутів від 0 до 90 градусів
theta_deg = np.linspace(0, 90, 1801)
theta_rad = np.radians(theta_deg)

# РОЗРАХУНОК МНОЖНИКІВ
arg_num = (np.pi * l / lambd) * (xi - np.cos(theta_rad))
arg_den = xi - np.cos(theta_rad)
norm_const = (xi - 1.0) / np.sin((np.pi * l / lambd) * (xi - 1.0))

with np.errstate(divide='ignore', invalid='ignore'):
  F_b = norm_const * (np.sin(arg_num) / arg_den)
  F_b = np.nan_to_num(F_b, nan=1.0)
  F_b_abs = np.abs(F_b)

F_env = np.abs(np.cos(theta_rad))

# ДС однострижневої
FH_1 = F_b_abs
FE_1 = F_b_abs * F_env

# Множник решітки та ДС двохстрижневої
F_C = np.abs(np.cos((np.pi * h / lambd) * np.sin(theta_rad)))
FH_2 = F_b_abs * F_C
FE_2 = F_b_abs * F_C * F_env

# АНАЛІТИЧНІ ТОЧКИ (ДО 5-Ї ПЕЛЮСТКИ ВКЛЮЧНО)
# Нулі випромінювання (theta_0m)
nulls_deg = []
for m in [1, 2, 3, 4, 5]:
  val = xi - m * lambd / l
  if -1 <= val <= 1:
    deg = np.degrees(np.arccos(val))
    if deg <= 90:
      nulls_deg.append(deg)

# Максимуми бічних пелюсток (theta_mm) (виправлено синтаксис)
maxes_deg = []
for m in [1, 2, 3, 4, 5]:
  val = xi - (2 * m + 1) * lambd / (2.0 * l)
  if -1 <= val <= 1:
    deg = np.degrees(np.arccos(val))
    if deg <= 90:
      maxes_deg.append(deg)


def find_theta_level(theta, profile, level=0.707):
  idx = np.argmin(np.abs(profile - level))
  return theta[idx]


theta_0707_1 = find_theta_level(theta_deg, FH_1, 0.707)
theta_0707_2 = find_theta_level(theta_deg, FH_2, 0.7)

# РИСУНОК 1: ДС Однострижневої ДСА
plt.figure(figsize=(12, 6.5))
plt.plot(
    theta_deg,
    F_env,
    'k-',
    linewidth=1.2,
    label=r'$|\cos(\Theta)|$ (Огинаюча)',
)
plt.plot(
    theta_deg,
    FH_1,
    'k-',
    linewidth=1.8,
    label=r'$|F_H(\Theta)| = |F_\sigma(\Theta)|$ (H-площина)',
)
plt.plot(
    theta_deg,
    FE_1,
    'k--',
    linewidth=1.5,
    label=r'$|F_E(\Theta)|$ (E-площина)',
)

# Рівень 0.707
plt.plot(
    [0, theta_0707_1, theta_0707_1],
    [0.707, 0.707, 0],
    'r--',
    linewidth=1.2,
    label=fr'Рівень 0.707 ($\Theta_{{0.707}} \approx {theta_0707_1:.2f}^\circ$)',
)
plt.scatter(
    [theta_0707_1],
    [0.707],
    color='red',
    zorder=5,
    label=r'Точка рівня $\Theta_{0.707}$',
)

# Позначення нулів та додавання першої точки у легенду
for i, th in enumerate(nulls_deg):
  lbl = r'Напрямки нулів ($\Theta_{0m}$)' if i == 0 else ''
  plt.scatter([th], [0], color='purple', zorder=5, label=lbl)
  plt.annotate(
      fr'$\Theta_{{0{i+1}}}$',
      (th, 0),
      textcoords='offset points',
      xytext=(0, -18),
      ha='center',
      fontsize=8,
      color='purple',
  )

# Позначення максимумів бокових пелюсток та додавання точок у легенду
for i, th in enumerate(maxes_deg):
  val_h = np.interp(th, theta_deg, FH_1)
  val_e = np.interp(th, theta_deg, FE_1)

  lbl_h = r'Максимуми $\Theta_{mm}$ (H-площина)' if i == 0 else ''
  lbl_e = r'Максимуми $\Theta_{mm}$ (E-площина)' if i == 0 else ''

  plt.scatter([th], [val_h], color='green', zorder=5, label=lbl_h)
  plt.annotate(
      fr'$\Theta_{{m{i+1}}}$ (H)',
      (th, val_h),
      textcoords='offset points',
      xytext=(0, 10),
      ha='center',
      fontsize=7,
      color='green',
  )

  plt.scatter([th], [val_e], color='blue', zorder=5, label=lbl_e)
  plt.annotate(
      fr'$\Theta_{{m{i+1}}}$ (E)',
      (th, val_e),
      textcoords='offset points',
      xytext=(0, -15),
      ha='center',
      fontsize=7,
      color='blue',
  )

plt.axhline(0, color='black', linewidth=0.8)
plt.axvline(0, color='black', linewidth=0.8)
plt.title(
    r'Розрахункова ДС однострижневої діелектричної антени (Варіант №13)',
    fontsize=12,
)
plt.xlabel(r'Кут $\Theta^\circ$, град')
plt.ylabel(r'Нормована амплітуда')
plt.xlim(0, 90)
plt.ylim(-0.05, 1.05)
plt.xticks(np.arange(0, 91, 15))
plt.yticks([0, 0.5, 0.707, 1.0])
plt.grid(True, linestyle=':', alpha=0.6)
plt.legend(loc='upper right', fontsize=8)
plt.tight_layout()
plt.savefig('DS_single_rod_5petals.png', dpi=300)
plt.show()

# РИСУНОК 2: ДС Двохстрижневої ДСА
plt.figure(figsize=(12, 6.5))
plt.plot(
    theta_deg,
    F_env,
    'r-',
    linewidth=1.2,
    label=r'$|\cos(\Theta)|$ (Огинаюча)',
)
plt.plot(
    theta_deg,
    F_C,
    'k-',
    linewidth=1.5,
    label=r'$|F_C(\Theta)|$ (Множник решітки, $h=5$ см)',
)
plt.plot(
    theta_deg,
    FH_2,
    'k-',
    linewidth=1.8,
    label=r'$|F_H(\Theta)|$ (Двохстрижнева, H-площина)',
)
plt.plot(
    theta_deg,
    FE_2,
    'k--',
    linewidth=1.5,
    label=r'$|F_E(\Theta)|$ (Двохстрижнева, E-площина)',
)

# Рівень 0.7
plt.plot(
    [0, theta_0707_2, theta_0707_2],
    [0.7, 0.7, 0],
    'r--',
    linewidth=1.2,
    label=fr'Рівень 0.7 ($\Theta_{{0.7}} \approx {theta_0707_2:.2f}^\circ$)',
)
plt.scatter(
    [theta_0707_2],
    [0.7],
    color='red',
    zorder=5,
    label=r'Точка рівня $\Theta_{0.7}$',
)

# Позначення всіх нулів
for i, th in enumerate(nulls_deg):
  lbl = r'Напрямки нулів ($\Theta_{0m}$)' if i == 0 else ''
  plt.scatter([th], [0], color='purple', zorder=5, label=lbl)
  plt.annotate(
      fr'$\Theta_{{0{i+1}}}$',
      (th, 0),
      textcoords='offset points',
      xytext=(0, -18),
      ha='center',
      fontsize=8,
      color='purple',
  )

# Позначення всіх максимумів
for i, th in enumerate(maxes_deg):
  val_h2 = np.interp(th, theta_deg, FH_2)
  val_e2 = np.interp(th, theta_deg, FE_2)

  lbl_h = r'Максимуми $\Theta_{mm}$ (H-площина)' if i == 0 else ''
  lbl_e = r'Максимуми $\Theta_{mm}$ (E-площина)' if i == 0 else ''

  plt.scatter([th], [val_h2], color='green', zorder=5, label=lbl_h)
  plt.annotate(
      fr'$\Theta_{{m{i+1}}}$ (H)',
      (th, val_h2),
      textcoords='offset points',
      xytext=(0, 10),
      ha='center',
      fontsize=7,
      color='green',
  )

  plt.scatter([th], [val_e2], color='blue', zorder=5, label=lbl_e)
  plt.annotate(
      fr'$\Theta_{{m{i+1}}}$ (E)',
      (th, val_e2),
      textcoords='offset points',
      xytext=(0, -15),
      ha='center',
      fontsize=7,
      color='blue',
  )

plt.axhline(0, color='black', linewidth=0.8)
plt.axvline(0, color='black', linewidth=0.8)
plt.title(
    r'Розрахункова ДС двохстрижневої діелектричної антени (Варіант №13)',
    fontsize=12,
)
plt.xlabel(r'Кут $\Theta^\circ$, град')
plt.ylabel(r'Нормована амплітуда')
plt.xlim(0, 90)
plt.ylim(-0.05, 1.05)
plt.xticks(np.arange(0, 91, 15))
plt.yticks([0, 0.5, 0.7, 1.0])
plt.grid(True, linestyle=':', alpha=0.6)
plt.legend(loc='upper right', fontsize=8)
plt.tight_layout()
plt.savefig('DS_double_rod_5petals.png', dpi=300)
plt.show()

# РИСУНОК 3: Полярна ДС Однострижневої антени
fig, ax = plt.subplots(figsize=(7, 7), subplot_kw={'projection': 'polar'})
ax.set_theta_zero_location('N')
ax.set_theta_direction(-1)

theta_p = np.linspace(-np.pi / 2, np.pi / 2, 3601)
FH_1_p = np.interp(np.abs(theta_p), theta_rad, FH_1)
FE_1_p = np.interp(np.abs(theta_p), theta_rad, FE_1)

ax.plot(theta_p, FH_1_p, label='H-площина', color='blue', linewidth=1.8)
ax.plot(
    theta_p,
    FE_1_p,
    label='E-площина',
    color='red',
    linestyle='--',
    linewidth=1.5,
)

ax.set_title(
    'Полярна ДС однострижневої ДСА (Варіант №13)', pad=20, fontsize=12
)
ax.legend(loc='lower right', bbox_to_anchor=(1.25, 0.0), fontsize=9)
plt.tight_layout()
plt.savefig('DS_polar_single_Var13.png', dpi=300)
plt.show()

# РИСУНОК 4: Полярна ДС Двохстрижневої антени
fig, ax = plt.subplots(figsize=(7, 7), subplot_kw={'projection': 'polar'})
ax.set_theta_zero_location('N')
ax.set_theta_direction(-1)

FH_2_p = np.interp(np.abs(theta_p), theta_rad, FH_2)
FE_2_p = np.interp(np.abs(theta_p), theta_rad, FE_2)

ax.plot(theta_p, FH_2_p, label='H-площина', color='blue', linewidth=1.8)
ax.plot(
    theta_p,
    FE_2_p,
    label='E-площина',
    color='red',
    linestyle='--',
    linewidth=1.5,
)

ax.set_title(
    'Полярна ДС двохстрижневої ДСА (Варіант №13)', pad=20, fontsize=12
)
ax.legend(loc='lower right', bbox_to_anchor=(1.25, 0.0), fontsize=9)
plt.tight_layout()
plt.savefig('DS_polar_double_Var13.png', dpi=300)
plt.show()