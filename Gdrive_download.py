import gdown

# Define folders
folder1 = 'https://drive.google.com/drive/folders/1KpZ8nFsmA7OUinq5lpp2dCwFpFnoH3E8',
folder2 = 'https://drive.google.com/drive/folders/1NPfihQm_yoL0RJ2paCGvdp5mN98UXriw',
folder3 = 'https://drive.google.com/drive/folders/1EkFfkx5oiPsWLZe4h4mFyOR3ptzRugM7',
folder4 = 'https://drive.google.com/drive/folders/1dFKA0w_MUxSurK0jc3_SklFMTYAd8m_0'
output_folder = '/gpfs/glad1/Exch/USA_2022/Data/Country_geojson'

# Try to download the first folder
gdown.download_folder(folder1, output=output_folder)