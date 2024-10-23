# Utiliser une image de base Python 3.9
FROM python:3.10-slim

# Définir le répertoire de travail
WORKDIR /app

# Installer les dépendances système nécessaires
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Installer Jupyter et nbconvert pour gérer les notebooks
RUN pip install --no-cache-dir jupyter nbconvert

# Copier le fichier requirements.txt dans le conteneur
COPY requirements.txt /app/requirements.txt

# Installer les dépendances Python
RUN pip install --no-cache-dir -r requirements.txt

# Copier le reste des fichiers de l'application (incluant le notebook)
COPY . /app

# Convertir et exécuter le notebook
# RUN jupyter nbconvert --to notebook --execute --inplace models/train_model.ipynb
RUN jupyter nbconvert --to notebook --execute --inplace model.ipynb

# Commande par défaut (facultatif si tu veux exécuter autre chose après)
#CMD ["python", "models/train_model.py"]
CMD ["python", "model.py"]