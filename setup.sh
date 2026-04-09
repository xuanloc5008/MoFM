# Install dependencies
pip install -r requirements.txt

# Install gdown
pip install gdown

# Create checkpoint folder
mkdir -p checkpoints

# Download MedSAM checkpoint
gdown 1UAmWL88roYR7wKlnApw5Bcuzf2iQgk6_ -O checkpoints/medsam_vit_b.pth

echo "✅ Setup completed!"