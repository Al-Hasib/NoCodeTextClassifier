# mPowerOnlineTest

### Demo test for the Machine Learning Engineer Position

The Exploratory Data Analysis report has given in the ```experiments/EDA.ipynb``` file. The ```experiments/ml.ipynb``` file shows the experiments training, evaluation with different ML models, GridSearch for finding the best model and hyperparameter tuning. The Question asked to focus on simplicity and effectiveness. That's why I started from Machine Learning models and thought to explore the Deep Learning and transformer models. But I get enough good performance with simple machine learning models.

### Instruction for running the python files

Create Virtual Environment
```bash
python -m venv env
```

Activate the Virtual Environment On Windows
```bash
env\Scripts\activate
```

Install the requirements
```bash
pip install -r requirements.txt
```

**Perform Training**

After running, you will get the model saved in the current directory.
```python
python train.py
```

**Perform Evaluation**

For getting the Output in the csv format for the text dataset, run the evalution file. It will generate the submission file.

```python
python evaluation.py
```

For performing the inference on a string to predict the emails is spam or not, run the inference file. Then provide the emails. Then it will provide the class of the given email.

```python
python inference.py
```

As Machine Learning model, I get the highest result with Linear Support Vector Machine model. With GridSearchCV, The mean accuracy of the model is 99.04%. 

## **Thanks**
