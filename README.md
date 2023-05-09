SML_EMG:

    This project is focused on comparing different types of supervised machine learning models on unfiltered EMG recordings

Installation and Usage:

    Follow these steps to set up and run the code:

        1. Clone the repository
        
            git clone https://github.com/ariasarch/SML_EMG.git
            cd SML_EMG
        
        2. (Optional) Create a virtual environment
        
            python -m venv venv_name
            source venv_name/bin/activate  # For Linux and macOS
            venv_name\Scripts\activate     # For Windows
        
        3. Install the package using the provided setup.py script:
        
            python setup.py install
        
        4. Install additional required dependencies using the requirements.txt file:
        
            pip install -r requirements.txt
            
        5.  Update the filepaths in config.py as needed for your system.
        
        6.  (Optional) Modify the test script
        
            Modify the number of types of models to run in the test script (if necessary).
            Adjust the p-value calculation settings in the test script (if necessary).
            
        7. Run the test script
      
            python test.py
        
        
