import SimpleITK as sitk
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from SSIM_PIL import compare_ssim
from scipy.ndimage import sobel


# Charger les images .hdr et .mhd avec SimpleITK
image1 = sitk.ReadImage(r"C:\Users\alici\Desktop\Polytech\cours\ETS S1\IA\projet\image_58\OAS1_0454_MR1_mpr_nn_anon_111_t88_masked_gfc.hdr")
image2 = sitk.ReadImage(r"C:\Users\alici\Desktop\Polytech\cours\ETS S1\IA\projet\reconstructed_volume_454_k3.mhd")
#BIEN PENSER A CHANGER LE CHEMIN D'ACCES DES FICHIERS .HDR ET .MHD (celui de la reconstruction)


# Convertir les images SimpleITK en tableaux NumPy
image1_array = sitk.GetArrayFromImage(image1)
image2_array = sitk.GetArrayFromImage(image2)

# Calculer la différence absolue entre les deux images
difference_array = np.abs(image1_array - image2_array)

# Application d'un seuil pour amplifier les différences significatives
threshold = np.percentile(difference_array, 90)  # seuil basé sur le 90e percentile des différences
difference_highlighted = np.where(difference_array > threshold, difference_array, 0)

# Créer une image des ressemblances
threshold_similarity = np.percentile(difference_array, 90)  # seuil basé sur le 10e percentile pour les ressemblances
similarity_highlighted = np.where(difference_array < threshold_similarity, difference_array, 0)


# Convertir la différence et la similarité en images SimpleITK
difference_image = sitk.GetImageFromArray(difference_highlighted)
similarity_image = sitk.GetImageFromArray(similarity_highlighted)
difference_image.CopyInformation(image1)
similarity_image.CopyInformation(image1)

# Sauvegarder l'image de différence complète
output_difference_path = r"C:\Users\alici\Desktop\Polytech\cours\ETS S1\IA\projet\difference_highlighted_image.mhd"
#BIEN PENSER A CHANGER LE CHEMIN D'ACCES DES FICHIERS .MHD

sitk.WriteImage(difference_image, output_difference_path)
print(f"Image de différence avec seuil sauvegardée à : {output_difference_path}")

# Extraire la tranche médiane
median_slice_index = difference_array.shape[0] // 2
median_slice_image1 = image1_array[median_slice_index, :, :]
median_slice_image2 = image2_array[median_slice_index, :, :]
median_slice_diff = difference_highlighted[median_slice_index, :, :]
median_slice_similarity = similarity_highlighted[median_slice_index, :, :]


# Détection des contours des différences
# Appliquer un filtre Sobel pour détecter les contours (utilisation de scipy.ndimage)
edges_x = sobel(median_slice_diff, axis=0)
edges_y = sobel(median_slice_diff, axis=1)
edges = np.hypot(edges_x, edges_y)  # Combine les gradients pour détecter les contours

# Convertir les bords en une image binaire (contours visibles)
contours = np.where(edges > 0.2, 1, 0).astype(np.uint8)  # Seuil pour contrôler l'intensité du contour

# Superposer ces contours sur l'image reconstruite (en violet)
highlighted_image = np.copy(median_slice_image2)
highlighted_image[contours == 1] = 255  # Applique le violet/rose (255) là où les contours sont détectés

# Calculer le MSE (Mean Squared Error)
mse = np.mean((image1_array - image2_array) ** 2)

# Calculer le SSIM
ssim_values = []
for i in range(image1_array.shape[0]):
    slice_image1 = image1_array[i, :, :]
    slice_image2 = image2_array[i, :, :]
    
    slice_image1_pil = Image.fromarray(slice_image1).convert('L')
    slice_image2_pil = Image.fromarray(slice_image2).convert('L')
    
    ssim_value = compare_ssim(slice_image1_pil, slice_image2_pil)
    ssim_values.append(ssim_value)

# Moyenne des SSIM sur toutes les tranches
mean_ssim = np.mean(ssim_values)
print(f"SSIM moyen sur toutes les tranches : {mean_ssim}")

# Affichage avec un subplot pour voir les images côte à côte
plt.figure(figsize=(15, 5))

# Image 1
plt.subplot(1, 4, 1)
plt.imshow(median_slice_image1, cmap='gray')
plt.title('Coupe médiane image 454')

# Image 2
plt.subplot(1, 4, 2)
plt.imshow(median_slice_image2, cmap='gray')
plt.title('Coupe médiane image 454 reconstruite')

# Différences
plt.subplot(1, 4, 3)
plt.imshow(median_slice_diff, cmap='hot')  # Utilisation d'une palette chaude pour mettre en évidence les différences
plt.title('Différences entre images')

# Ressemblances (mis en valeur avec un seuil faible)
plt.subplot(1, 4, 4)
plt.imshow(highlighted_image, cmap='magma')  # Utilisation de 'magma' ou d'un autre cmap pour un effet visuel fort
plt.title('Ressemblances sur l\'image reconstruite')

# Ajouter le MSE sous l'image
plt.figtext(0.5, 0.05, f'MSE: {mse:.4f}     SSIM: {mean_ssim:.4f}', ha='center', fontsize=12)

# Ajuster l'espace pour ajouter plus de marge sous le graphique
plt.subplots_adjust(bottom=0.2)

plt.show()