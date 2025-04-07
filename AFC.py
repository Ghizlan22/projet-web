import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

# 1. Affichage du tableau initial
data = {
    'Yeux/Cheveux': ['Vert', 'Bleu', 'Marron', 'Noir'],
    'Blond': [50, 43, 8, 2],
    'Marron': [10, 8, 66, 44],
    'Noir': [3, 0, 33, 50]
}

df = pd.DataFrame(data).set_index('Yeux/Cheveux')
print("1. Tableau de contingence initial:")
print(df)
print("\n")

# 2. Tableau de fréquence (f_ij = n_ij / n)
total = df.values.sum()
table_freq = df / total

# Ajout de la colonne fi. (somme des lignes)
table_freq['f_i.'] = table_freq.sum(axis=1)

# Ajout de la ligne f_j (somme des colonnes)
f_j_row = table_freq.sum(axis=0).to_frame().T
f_j_row.index = ['f_j']

# Ajout de la ligne à la table
table_freq = pd.concat([table_freq, f_j_row])

print("2. Tableau de fréquence (avec f_i. et f_j):")
print(table_freq.round(5))





# 3. Tableau de profils ligne (f_j|i = n_ij / n_i)
profils_ligne = df.div(df.sum(axis=1), axis=0)

# Ajout de la colonne f_i. (identique à celle de la table de fréquence, sauf dernière ligne)
profils_ligne['f_i.'] = table_freq.loc[df.index, 'f_i.']

print("3. Tableau de profils ligne (avec f_i.):")
print(profils_ligne.round(4))
print("\n")



# 4. Tableau de profils colonne (f_i|j = n_ij / n_j)
profils_colonne = df.div(df.sum(axis=0), axis=1)

# Ajout de la ligne f_j (extraite depuis la table de fréquence, sauf dernière colonne)
f_j_row_only = table_freq.loc['f_j', profils_colonne.columns]
f_j_row_only.name = 'f_j'

# Ajout de la ligne à la table des profils colonne
profils_colonne = pd.concat([profils_colonne, pd.DataFrame([f_j_row_only])])

print("4. Tableau de profils colonne (avec f_j uniquement):")
print(profils_colonne.round(4))
print("\n")

# 5. Calcul de la matrice X comme dans l'image
# Extraction des f_i. et f_j (sans la dernière ligne et colonne)
f_i = table_freq.loc[df.index, 'f_i.'].values
f_j = table_freq.loc['f_j', df.columns].values

# Calcul de la matrice (f_ij - f_i*f_j)
f_ij = table_freq.loc[df.index, df.columns].values
mat_diff = f_ij - np.outer(f_i, f_j)

print("5. Matrice (f_ij - f_i*f_j):")
print(pd.DataFrame(mat_diff, index=df.index, columns=df.columns).round(5))
print("\n")

# Calcul des matrices diagonales D_i^-1 et D_j^-1
D_i_inv = np.diag(1/np.sqrt(f_i))
D_j_inv = np.diag(1/np.sqrt(f_j))

print("Matrice D_i^-1:")
print(pd.DataFrame(D_i_inv, index=df.index, columns=df.index).round(3))
print("\n")

print("Matrice D_j^-1:")
print(pd.DataFrame(D_j_inv, index=df.columns, columns=df.columns).round(3))
print("\n")

# Calcul de la matrice X
X = D_i_inv @ mat_diff @ D_j_inv

print("6. Matrice X:")
print(pd.DataFrame(X, index=df.index, columns=df.columns).round(4))
print("\n")

# 6. Calcul de la matrice S = X^T X
S = X.T @ X

print("7. Matrice S = X^T X:")
print(pd.DataFrame(S, index=df.columns, columns=df.columns).round(5))
print("\n")

# 7. Calcul des valeurs propres et vecteurs propres
eigenvalues, eigenvectors = np.linalg.eig(S)

# Tri des valeurs propres dans l'ordre décroissant
idx = eigenvalues.argsort()[::-1]
eigenvalues = eigenvalues[idx]
eigenvectors = eigenvectors[:, idx]

print("Valeurs propres:")
print(eigenvalues.round(6))
print("\n")

print("Vecteurs propres (colonnes):")
print(pd.DataFrame(eigenvectors, index=df.columns, 
                  columns=[f"Vecteur {i+1}" for i in range(len(df.columns))]).round(5))
print("\n")

# Pour l'analyse AFC, on garde les 2 premières dimensions (comme dans l'image)
print("Pour la présentation graphique, on garde les 2 premières dimensions:")
print(f"Valeur propre 1 (A1): {eigenvalues[0]:.5f}")
print(f"Valeur propre 2 (A2): {eigenvalues[1]:.5f}")

# 8. Calcul des composantes principales comme dans l'image

# Extraction des vecteurs propres U1 et U2 (première et deuxième colonne)
U1 = eigenvectors[:, 0]
U2 = eigenvectors[:, 1]

print("8. Vecteurs propres U1 et U2:")
print(pd.DataFrame({'U1': U1, 'U2': U2}, index=df.columns).round(3))
print("\n")

# 9. Calcul des composantes principales pour les profils lignes (Ck = X * Uk)
C1_ligne = X @ U1
C2_ligne = X @ U2

print("9. Composantes principales des profils lignes:")
profils_ligne_afc = pd.DataFrame({'C1': C1_ligne, 'C2': C2_ligne}, index=df.index)
print(profils_ligne_afc.round(4))
print("\n")

# 10. Calcul des composantes principales pour les profils colonnes (Ck' = sqrt(lambda) * Uk)
C1_col = np.sqrt(eigenvalues[0]) * U1
C2_col = np.sqrt(eigenvalues[1]) * U2

print("10. Composantes principales des profils colonnes:")
profils_col_afc = pd.DataFrame({'C1': C1_col, 'C2': C2_col}, index=df.columns)
print(profils_col_afc.round(4))
print("\n")

import matplotlib.pyplot as plt

# Création du graphique
plt.figure(figsize=(10, 8))

# Affichage des profils lignes (yeux)
plt.scatter(profils_ligne_afc['C1'], profils_ligne_afc['C2'], color='blue', label='Yeux')
for i, txt in enumerate(profils_ligne_afc.index):
    plt.annotate(txt, (profils_ligne_afc['C1'][i], profils_ligne_afc['C2'][i]), color='blue')

# Affichage des profils colonnes (cheveux)
plt.scatter(profils_col_afc['C1'], profils_col_afc['C2'], color='red', label='Cheveux')
for i, txt in enumerate(profils_col_afc.index):
    plt.annotate(txt, (profils_col_afc['C1'][i], profils_col_afc['C2'][i]), color='red')

# Lignes d'axes
plt.axhline(0, color='grey', linestyle='--')
plt.axvline(0, color='grey', linestyle='--')

# Titres et légende
plt.title('Plan factoriel AFC - Axes 1 et 2')
plt.xlabel(f'Axe 1 (Inertie: {eigenvalues[0]:.2%})')
plt.ylabel(f'Axe 2 (Inertie: {eigenvalues[1]:.2%})')
plt.legend()
plt.grid(True)
plt.show()

