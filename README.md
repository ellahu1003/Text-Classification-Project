# Text-classification-project

## Overview
This project demonstrates text classification using the 20 Newsgroups dataset. The dataset contains approximately 20,000 newsgroup documents, partitioned across 20 different newsgroups. The aim is to classify these documents into their respective categories using the Multinomial Naive Bayes model. The project follows a structured approach that includes data loading and exploration, data cleaning and preprocessing, feature extraction using TF-IDF vectorization, and model training and evaluation.

## Dataset
The 20 Newsgroups dataset is fetched using sklearn.datasets.fetch_20newsgroups and includes the following:
1) Text Data: The content of the newsgroup posts.
2) Target: Category labels for the posts.
3) Target Names: Names of the categories.

## TechnologieS Used
1) Python
2) Pandas
3) Numpy
3) NLTK
4) Scikit-Learn
5) Seaborn
6) Matplotlib
7) Jupyter Notebook

## Setup and installations
1) Clone the repository:
   ```bash
    git clone https://github.com/ellahu1003/text-classification-project.git
    cd text-classification-project
    ```
2) Install the required packages:
   ```bash
    pip install -r Requirements.txt
    ```
3) Run the Jupyter Notebook:
   ```bash
    jupyter notebook notebook/Text_classification_project.ipynb
    ```
## Project Structure
```markdown
text-classification/
│
├── data/
│   └── 20_newsgroups_dataset.csv
│
├── notebook/
│   └── text_classification_project.ipynb
│
├── results/
│   ├── distribution_of_categories.png
│   └── classification_report.txt
│
└── requirements.txt
```

## Requirements
The 'Requirements.txt' file lists all the Python packages required to run the project. Install these dependencies to avoid any compatibility issues.

## Results
1) Accuracy score: [0.88]
2) The visual distribution of categories are available in distribution_of_categories_.png
3) Detailed information about the model's performance in each class is available in classification_report.txt

## Conclusion:
The model performs well overall with a high accuracy of 88%. However, there are some classes where performance could be improved, particularly class 15, which has very low recall. This suggests that the model struggles to identify this class effectively. 
