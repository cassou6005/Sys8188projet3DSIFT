# Sys818_projet_3DSIFT
Ce projet développe une méthode de reconstruction 3D utilisant les points clés SIFT 3D et l’algorithme KNN pour analyser des IRM cérébrales. Testée sur le jeu de données OASIS, la méthode est évaluée via les métriques SSIM et MSE. Elle présente des limites et des perspectives d’amélioration avec des modèles d’apprentissage profond.

## Installation des Bibliothèques Nécessaires
Avant d'exécuter le projet, assurez-vous d'installer toutes les bibliothèques requises. Vous pouvez utiliser pip install pour les installer. Voici une liste des principales bibliothèques utilisées :

- SimpleITK
- NumPy
- Matplotlib
- Pillow (PIL)
- SSIM_PIL
- SciPy

Vous pouvez les installer en utilisant la commande suivante :
pip install SimpleITK numpy matplotlib pillow SSIM_PIL scipy


## Chemins d'Accès aux Fichiers
Vérifiez et modifiez les chemins d'accès des fichiers avant d'exécuter les scripts. Les chemins doivent pointer vers les fichiers d'images correspondants (.hdr, .img, .key, .mhd) et les fichiers de sortie.
Modifiez ces chemins en fonction de votre environnement de travail.

### Récupération des Images
Les images d'IRM à utiliser dans ce projet peuvent être récupérées dans le dossier suivant :

Données : (keypoints, IRM 3D cérébrales, déjà recalés pour référence)
http://www.matthewtoews.com/projects/oasis/
http://www.matthewtoews.com/projects/oasis/oasis_brains.zip
• Note : Il y a 416 images, utiliser images 0001 à 0399 pour entrainement,
images 0400 à 0416 pour évaluation.

Veuillez vous assurer d'avoir les fichiers necessaires.

## Problèmes Actuels
### Maintenance de la Salle Labo A-3340
Le service TI effectue actuellement une maintenance dans la salle de Labo A-3340. Cette maintenance nous empêche de récupérer les codes des GANs et U-Net qui étaient en cours d'entraînement sur les machines de la salle. Nous codions dans cette salle à cause de l'absence de carte GPU sur nos ordinateurs personnels pour exécuter ces modèles.

La réouverture de la salle est prévue pour le *10 janvier 2025*. Nous mettrons à jour ce répertoire avec les codes correspondants si nécessaire après cette date.
