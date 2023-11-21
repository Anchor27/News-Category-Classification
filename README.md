# NEWS HEADLINE CATEGORY CLASSIFICATION MODEL

<img src="https://github.com/Anchor27/Projects-Data-Repo/blob/d328ec824642867a952f42beebf03aedaddd2498/NEWS-IMG.png">

## Overview
In an era characterized by information abundance, the efficient categorization of news headlines becomes paramount. This project endeavors to contribute to this objective by developing a robust machine learning model capable of accurately classifying news headlines into distinct categories. Leveraging advanced natural language processing techniques, the model aims to bring sophistication to news categorization.

## Table of Contents
1. Objectives
2. Motivation
3. News Categories
4. Dataset Exploration
5. Feature Engineering
6. Model Evaluation
7. Usage Guide
8. File Structure
9. Future Enhancements
10. Significance
11. Scope and Limitatiions
12. Author Information

## 1. Objectives
The primary objective of this project is to develop a robust, adaptable, and high-performing model capable of categorizing news headlines with a nuanced understanding of their content. Beyond mere classification, the model aspires to contribute to the broader conversation on information organization, offering insights into the complex interplay between language patterns, semantics, and the ever-evolving landscape of news media.

## 2. Motivation
The impetus behind this project lies in the recognition of the challenges posed by the sheer volume and diversity of news content available today. Efficiently categorizing news headlines is not only a technological challenge but also a societal necessity. The ability to swiftly and accurately identify the nature of news articles aids in streamlining information dissemination, enhancing user experience, and facilitating more informed decision-making.

## 3. News Categories
The various news-headline categories on which my model was trained are:
1. Declarative
2. Interrogative
3. Imperative
4. Exclamatory
5. Fragment
6. Miscellaneous (nq_drop, qws)

## 4. Dataset Exploration
### Data Source and Preprocessing
The dataset comprises a diverse collection of news headlines, each meticulously labeled with its corresponding category. Preprocessing steps involve not only the standardization of textual data but also the strategic handling of missing values to ensure a comprehensive dataset for model training.

### Exploratory Data Analysis (EDA)
An in-depth exploratory data analysis is conducted to unravel insights into the distribution of categories, potential imbalances, and the inherent linguistic characteristics of the headlines. This preliminary analysis guides subsequent feature engineering decisions.

## 5. Feature Engineering
### Text Representation Strategies
#### TF-IDF Vectorization
The textual content of the headlines undergoes transformation using Term Frequency-Inverse Document Frequency (TF-IDF) vectorization. This dual-vectorization strategy encompasses both character and word n-grams, enabling the model to discern intricate semantic nuances and syntactic structures within the text.

### Linguistic Features
Beyond textual representations, linguistic features such as text length and word count are meticulously extracted. These features add an additional layer of context, allowing the model to recognize patterns beyond the lexical domain.

### Model Architecture
The model architecture is anchored by a Random Forest Classifier with 100 estimators. This ensemble learning approach is chosen for its adaptability to high-dimensional feature spaces and its potential to deliver robust results.Evaluation
To assess the model's efficacy, a meticulous evaluation process is executed on a distinct test set. The evaluation metrics, including accuracy, and a comprehensive classification report, are presented for a nuanced understanding of the model's performance.

#### Code snippet for model training
<pre>
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(combined_features, y_train)
</pre>



## 6. Model Evaluation
To ascertain the model's efficacy, a comprehensive evaluation is executed on a distinct test set. The evaluation metrics, including accuracy, precision, recall, and a detailed classification report, are presented to provide a nuanced understanding of the model's performance.

Code Snippet for Model Evaluation-
<pre>
y_pred = model.predict(X_test_combined_features)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:")
print(classification_report(y_test, y_pred))
</pre>

## 7. Usage Guide
Implementing and reproducing this project is streamlined through the following steps:

### Clone the Repository:

<pre>
git clone https://github.com/Anchor27/News-Category-Classification.git
cd News-Category-Classification
</pre>
  
### Install Dependencies:

<pre>
pip install scikit-learn
pip install pandas
pip install numpy
</pre>

  
## 8. File Structure
- `data/`       :         Contains datasets used for training and testing the model
- `src/`         :        Source code and Google Collab Notebook
- `README.md`     :       This README file

## 9. Future Enhancements
Anticipated future improvements for this project include:

- Hyperparameter tuning for optimizing model performance.
- Exploring deep learning architectures to facilitate more intricate feature extraction.
- Investigating alternative vectorization techniques, such as embeddings or transformers, to refine text representation.

## 10. Significance
The significance of this project extends beyond its immediate application in news categorization. It delves into the intricate dynamics of language representation, feature engineering, and the symbiosis between machine learning and linguistic analysis. By addressing the challenges inherent in news categorization, this project seeks to contribute to the broader discourse on the responsible use of technology in information management.

## 11. Scope and Limitations
While this project is designed to make significant strides in news headline categorization, it is crucial to acknowledge its scope and limitations. The model's efficacy may vary based on the nature of the dataset and the dynamic nature of news content. Continuous iterations and adaptations will be necessary to ensure the model's relevance and accuracy in the face of evolving linguistic patterns and news reporting styles.

In essence, this project is not just a technical endeavor but a conscientious exploration of the symbiotic relationship between technology and information, with a vision of enhancing the way we interact with and understand the ever-expanding world of news.

## 12. Author Information
This project was meticulously crafted by Aryan Joshi, an aspiring data scientist and machine learning enthusiast. 
