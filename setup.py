import shutil
import os
import zipfile
import subprocess
import urllib.request
from setuptools import setup, find_packages
from setuptools.command.install import install


with open("README.md", "r") as fh:
	long_description = fh.read()


# URL do arquivo de dados
DATA_URL = "https://github.com/MICLab-Unicamp/LobePrior/releases/download/LobePrior/data.zip"
DATA_DIR = "data"  # Pasta onde os dados serão extraídos


def download_and_extract_data():
	"""
	Faz o download e a extração do arquivo data.zip se a pasta 'data' não existir.
	"""
	if not os.path.exists(DATA_DIR):
		print(f"[LobePrior] Folder '{DATA_DIR}' not found. Starting download...")
		zip_path = "data.zip"

		# Download do arquivo ZIP
		print(f"[LobePrior] Downloading from {DATA_URL}...")
		urllib.request.urlretrieve(DATA_URL, zip_path)
		print("[LobePrior] Download completed.")

		# Extração do ZIP
		print(f"[LobePrior] Extraindo {zip_path} para {DATA_DIR}...")
		with zipfile.ZipFile(zip_path, "r") as zip_ref:
			zip_ref.extractall(".")
		print("[LobePrior] Extraction completed.")

		# Remove o arquivo zipado
		os.remove(zip_path)
		print("[LobePrior] Zip file removed.")
	else:
		print(f"[LobePrior] Folder '{DATA_DIR}' already exists. Skipping download.")

	data_dir = "data"
	root_dir = "."  # root folder (it can be an absolute path if needed)

	# List the folders inside ‘data’
	subfolders = ["weights", "raw_images"]

	for folder in subfolders:
		src = os.path.join(data_dir, folder)
		dst = os.path.join(root_dir, folder)
		if os.path.exists(src):
			if not os.path.exists(dst):
				print(f"Moving {src} to {dst}")
				shutil.move(src, root_dir)
			else:
				print(f"Destination folder {dst} already exists. Skipping..")
		else:
			print(f"Folder {src} not found.")

	if os.path.exists(data_dir) and os.path.isdir(data_dir):
		shutil.rmtree(data_dir)
		print(f"[LobePrior] Directory {data_dir} removed.")

class CustomInstallCommand(install):
	"""
	Comando de instalação personalizado para baixar e extrair os dados.
	"""
	def run(self):
		install.run(self)  # Performs the default installation
		download_and_extract_data()  # Downloads and extracts the data



found = find_packages()
print(f"Found these packages to add: {found}")

setup(
	name="lobeprior",
	version="1.0",
	description="LobePrior: Segmenting Lung Lobes on Computed Tomography Images in the Presence of Severe Abnormalities",
	author="MICLab Unicamp",
	url="https://github.com/MICLab-Unicamp/LobePrior",
	packages=found,
	python_requires=">=3.9",  # Added to ensure sm_120 compatibility
	install_requires=[
		"numpy==1.24.4",
		"SimpleITK==2.4.0",
		"torchio==0.20.1",
		"torchvision",
		"nibabel",
		"pydicom",
		"scipy",
		"dipy==1.7.0",
		"connected-components-3d",
		"pytorch_lightning",
		"torch",
		"monai",
		"matplotlib",
		"scikit-image",
		"tensorboard",
		"scikit-learn",
		"psutil"
	],
	cmdclass={
		'install': CustomInstallCommand,  # Substitui o comando install padrão
	},
	entry_points={
		"console_scripts": [
			"lobeprior=predict:main",
			"lungprior=predict_lung_register:main"
		],
	},
	include_package_data=True,
	package_data={'lobeprior': ["LightningLobes_no_template.ckpt", "LightningLobes.ckpt", "LightningLung.ckpt"]}
)
