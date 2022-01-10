# UltraColdCNN

A collection of jupyter notebooks designed to generate a machine learning model which can fit the defocus parameter of a series of images and compare to a chi-squared fitting method.

## Building env from YAML file

1. Download PurdueUltraCold.yml from main branch
2. Open with a text editor and replace both instances of <env-name> with desired python environment name (make sure to save as .yml file)
3. Replace the one instance of </path/to/your/anaconda/distribution> with the path to your anaconda distribution (this was created using Anaconda 4.10)
4. In a command line run the following: conda env create --file <env-name>.yml 

## Importing images

1. Place image folders in raw_image folder (a list of image folders used here is in raw_image titled raw_im_folders_used.txt)

## Running the modules

### Once images are placed in correct folder run modules in the following order
  
1. GetParamRanges.ipynb
2. RandomNoise_V6.ipynb
3. UltraColdCNN_V9.ipynb + RealDataPrep.ipynb + Fit_Single.ipynb
4. GraphGeneratorArtificialV2.ipynb + GraphGeneratorRealV2.ipynb
