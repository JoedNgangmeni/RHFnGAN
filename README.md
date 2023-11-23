# RHFnGAN

# Getting Started

1. Clone this repo
2. Visit [Anaconda's website](https://docs.anaconda.com/free/anaconda/install/index.html) and download the version that suits your operating system.

**Each model runs in its own conda environment.** When tryinf to run a specific model, be sure to activate that models conda environment by following these steps
1. In your systems command line, navigate to the base directory for the model you want to run
2. **Make sure your terminal is in the base directory for that model.** Create the conda environment for that specific model by running the following code  

   ```conda env create -f environment.yml``` 

4. Once the environment is created, activate it by running the following code

   ```conda activate environment_name```

**REPLACE "environment_name" with the type of model you want to run.**
   
   - For Traditional Random Forest - TRF
   - For Random Hinge Forest - RHF
   - For XGBoost - XG
   - For LightGBM - LG
   - For Deep Neural Network - DNN
   - For Recurrent Neural Network - RNN

