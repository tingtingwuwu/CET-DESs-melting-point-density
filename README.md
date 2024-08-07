# My Project
This project aims to predict the properties of Deep Eutectic Solvents (DESs) using CET model.

## Installation
1. Clone the repository:
    ```bash
    git clone https://github.com/your_username/my_project.git
    ```
2. Navigate to the project directory:
    ```bash
    cd my_project
    ```
3. Install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```

## Usage
To run the main script, use the following command:
```bash
python main.py --model chemberta
```

You can also choose to use the K-BERT model:
```bash
python main.py --model kbert
```

### Example
Here is an example of how to use the project:
1. Ensure you have the necessary data file (e.g., `compound_data.xlsx`) in the appropriate path.
2. Run the main script with the desired model:
    ```bash
    python main.py --model chemberta
    ```
   or
    ```bash
    python main.py --model kbert
    ```
This will preprocess the data, extract features using the chosen model, and evaluate the regression models.
