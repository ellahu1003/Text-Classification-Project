# Text-classification-project with Naive Bayes

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

## Methodology
1) Data Collection:
   The project utilises the 'fetch_20newsgroups' dataset from the sklearn.datasets library. This dataset includes text data from 20 different newsgroups, which are categorised into various topics.

2) Data Preprocessing:
   1. Loading the Data: The dataset is fetched and loaded using the fetch_20newsgroups function from sklearn.datasets.
   2. Creating DataFrame: The data and target variables are extracted and combined into a Pandas DataFrame. The target variable is also mapped to category names for better interpretability.
   3. Saving Dataset: The DataFrame is saved as a CSV file for easy access and future use.

3) Exploratory Data Analysis (EDA):
   1. Checking Data Types: The data types of the DataFrame columns are checked to ensure correctness.
   2. Inspecting Missing Values: The dataset is inspected for any missing values, no missing values present in the dataset.
   3. Descriptive Statistics: Basic descriptive statistics are computed to understand the datase.
   4. Data Distribution Visualisation: The distribution of categories in the dataset is visualised.

4) Text Preprocessing and Cleaning:
   1. Removing Stopwords: Stopwords, which carry less meaningful information, are removed using NLTK's list of English stopwords.
   2. Text Cleaning: A function is defined to clean the text data by removing special characters, numbers, and converting text to lowercase.

5) Feature Extraction:

   TF-IDF Vectorization: The cleaned text data is transformed into numerical features using the TF-IDF vectoriser (TfidfVectoriser from sklearn.feature_extraction.text).
   
6) Model Building and Training:
   1. Train-Test Split: The dataset is split into training and testing sets using an 80-20 split.
   2. Naive Bayes Classifier: A Multinomial Naive Bayes classifier (MultinomialNB from sklearn.naive_bayes) is trained on the training set.

7) Model Evaluation: 
   1. Accuracy Score: The accuracy of the model is evaluated on the test set.
   2. Classification Report: A classification report is generated to provide detailed performance metrics for each category.



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

## Requirements
The 'Requirements.txt' file lists all the Python packages required to run the project. Install these dependencies to avoid any compatibility issues.

## Results
1) Accuracy score: [0.88]
2) The visual distribution of categories are available in distribution_of_categories_.png
3) Detailed information about the model's performance in each class is available in classification_report.txt

## Conclusion:
The model performs well overall with a high accuracy of 88%. However, there are some classes where performance could be improved, particularly class 15, which has very low recall. This suggests that the model struggles to identify this class effectively. 
