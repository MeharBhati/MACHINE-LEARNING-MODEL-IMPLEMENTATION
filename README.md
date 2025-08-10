# MACHINE-LEARNING-MODEL-IMPLEMENTATION

Company : CODETECH IT SOLUTIONS

Name : Bhati Mehar Mansoor Akhtar

Intern ID : CT04DH2154

Domain : Python

Duration : 4 weeks

Mentor : Neela Santosh

***************************************************************


**Task 4 – Machine Learning Model Implementation**

In Task 4 of my CODTECH Python internship, I developed a **predictive machine learning model** using **Scikit-learn** to classify or predict outcomes from a dataset. The example problem I chose was **spam email detection**, where the goal was to distinguish between spam and non-spam (ham) emails based on textual data. This task combined data preprocessing, model training, evaluation, and presentation in a **Jupyter Notebook** environment.

**1. Dataset Selection and Loading**
I began by selecting a suitable dataset containing labeled examples of spam and ham emails. The dataset was loaded into the project using **Pandas**, which allowed for easy inspection and manipulation of the data. Each record consisted of the email text and its corresponding label (spam or ham).

**2. Data Preprocessing**
Since raw text cannot be fed directly into machine learning models, I performed several preprocessing steps:

* **Text Cleaning** – Removing special characters, punctuation, and numbers to standardize the data.
* **Tokenization** – Splitting sentences into individual words.
* **Stopword Removal** – Eliminating common words like “the” and “and” that carry little meaning.
* **Lemmatization/Stemming** – Reducing words to their root form for consistency.

For converting text into numerical form, I used **TF-IDF Vectorization (Term Frequency–Inverse Document Frequency)**, which measures the importance of each word in relation to the entire dataset. This representation was crucial for the model to understand and weigh relevant keywords.

**3. Model Selection and Training**
I experimented with different classification algorithms available in Scikit-learn, including:

* **Naive Bayes (MultinomialNB)** – commonly used for text classification.
* **Logistic Regression** – effective for binary classification tasks.
* **Support Vector Machines (SVM)** – for achieving high accuracy with complex boundaries.

The dataset was split into **training** and **testing** sets (typically 80/20 split) using Scikit-learn’s `train_test_split()` function. Models were trained on the training set and evaluated on the test set.

**4. Model Evaluation**
I used metrics like **accuracy score**, **precision**, **recall**, and the **F1-score** to assess model performance. A **confusion matrix** was also plotted using **Matplotlib** to visualize how well the model distinguished between spam and ham. The Naive Bayes classifier performed particularly well, balancing accuracy and computational efficiency.

**5. Jupyter Notebook Presentation**
The entire workflow—from data loading to final predictions—was documented and executed within **Jupyter Notebook**. This environment allowed for combining explanatory text (Markdown cells), code cells, and output visualizations in one place, making the project both reproducible and easy to understand.

**Tools and Technologies Used:**

* **Python** – core programming language.
* **Pandas** – for data handling and manipulation.
* **Scikit-learn** – for machine learning algorithms, model evaluation, and data splitting.
* **NLTK** – for text preprocessing.
* **Matplotlib** – for plotting the confusion matrix and visualizations.
* **Jupyter Notebook** – for implementation and documentation.

******Output*******

http://localhost:8888/notebooks/spamdetectionModel.ipynb

<img width="1192" height="865" alt="Image" src="https://github.com/user-attachments/assets/2d0f26d6-6107-481d-bb1c-f33ce0f843e5" />

