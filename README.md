SML_EMG:

    This project is focused on comparing different types of supervised machine learning models on unfiltered EMG recordings

Installation and Usage:

Follow these steps to set up and run the code:

1. Clone the repository
    
    git clone https://github.com/ariasarch/SML_EMG.git
    cd SML_EMG

2. Create a virtual environment

    conda create --name new_environment_name --file conda_requirements.txt 

3. Install the package using the provided setup.py script:

    python setup.py install

4. Update the base filepaths in config.py as needed for your system.

5. (Optional) Modify the test script

    Modify the number of types of models to run in the test script (if necessary).

6. (Optional) Modify the eval script

    Modify the best model to run in the test script (if necessary).
    Adjust the p-value calculation settings in the test script (if necessary).

7. Run the test script

    python test.py
        
