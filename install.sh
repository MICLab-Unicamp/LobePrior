# An optional install script that does all installation steps
echo "Creating lobeprior environment..."
eval "$(conda shell.bash hook)"
conda create --name lobeprior python=3.9
conda activate lobeprior
echo "Installing PyTorch with GPU support..."

echo "Installing requirements..."
pip install -r requirements.txt
echo "Installing LobePrior..."
pip install .
echo "Done. Please always activate the lobeprior environment with 'conda activate lobeprior' before running LobePrior method."

