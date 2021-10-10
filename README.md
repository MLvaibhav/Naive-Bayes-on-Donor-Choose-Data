# Naive-Bayes-on-Donor-Choose-Data

# About Donor Choose

DonorsChoose.org receives hundreds of thousands of project proposals each year for classroom projects in need of funding. Right now, a large number of volunteers is needed to manually screen each submission before it's approved to be posted on the DonorsChoose.org website.

Next year, DonorsChoose.org expects to receive close to 500,000 project proposals.

# Objective

The goal is to predict whether or not a DonorsChoose.org project proposal submitted by a teacher will be approved, using the text of project descriptions as well as additional metadata about the project, teacher, and school.

DonorsChoose.org can then use this information to identify projects most likely to need further review before approval.

# About donor chose dataset

| Feature       | Description   |
| ------------- | ------------- |
| project_id    | A unique identifier for the proposed project. Example: p036502  |
| project_title | Title of project  |
| project_grade_category  | Grade level of students for which the project is targeted  |
| project_subject_categories  | subject categories for the project  |
| school_state  | State where school is located   |
| project_subject_subcategories  | subject subcategories for the project  |
| project_resource_summary  | An explanation of the resources needed for the project.  |
| project_essay_1  | First application essay*  |
| project_essay_2 | Second application essay*  |
| project_essay_3  | Third application essay*  |
| project_essay_4 | Fourth application essay*  |
| project_submitted_datetime  | Datetime when project application was submitted.  |
| teacher_id | A unique identifier for the teacher of the proposed project.  |
| teacher_prefix | Teacher's title.  |
| teacher_number_of_previously_posted_projects | Number of project applications previously submitted by the same teacher.  |

Notes on essay Data 

Prior to May 17, 2016, the prompts for the essays were as follows:

__project_essay_1:__ "Introduce us to your classroom".

__project_essay_2:__ "Tell us more about your students".

__project_essay_3:__ "Describe how your students will use the materials you're requesting".

__project_essay_3:__ "Close by sharing why your project will make a difference".

Starting on May 17, 2016, the number of essays was reduced from 4 to 2, and the prompts for the first 2 essays were changed to the following:

__project_essay_1:__ "Describe your students: What makes your students special? Specific details about their background, your neighborhood, and your school are all helpful."

__project_essay_2:__ "About your project: How will these materials make a difference in your students' learning and improve their school lives?"

For all projects with project_submitted_datetime of 2016-05-17 and later, the values of project_essay_3 and project_essay_4 will be NaN.

Lets get started with data 

Importing all the important libararies 

```python

%matplotlib inline
import warnings
warnings.filterwarnings("ignore")

import sqlite3
import pandas as pd
import numpy as np
import nltk
import string
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import confusion_matrix
from sklearn import metrics
from sklearn.metrics import roc_curve, auc
from nltk.stem.porter import PorterStemmer

import re
# Tutorial about Python regular expressions: https://pymotw.com/2/re/
import string
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.stem.wordnet import WordNetLemmatizer

from gensim.models import Word2Vec
from gensim.models import KeyedVectors
import pickle

from tqdm import tqdm
import os

from chart_studio import plotly
import plotly.offline as offline
import plotly.graph_objs as go
offline.init_notebook_mode()
from collections import Counter
```

Reading the provided data 

```python
project_data = pd.read_csv('train_data.csv')
```

```python
print("Number of data points in train data", project_data.shape)
print('-'*50)
```

1. Data analysis

Defining function stack_plot to create a stacked bar chart 

```python

def stack_plot(data, xtick, col2='project_is_approved', col3='total'):
    ind = np.arange(data.shape[0])
    
    plt.figure(figsize=(20,5))
    p1 = plt.bar(ind, data[col3].values)
    p2 = plt.bar(ind, data[col2].values)

    plt.ylabel('Projects')
    plt.title('% of projects aproved state wise')
    plt.xticks(ind, list(data[xtick].values))
    plt.legend((p1[0], p2[0]), ('total', 'accepted'))
    plt.show()
    ```
    
    Again defining new function univariate_barplots thattakes below mentioned inputs and stack_plot is also getting called inside it to create stacked bar chart in barplots 
    
    stacked_plot takes column 1 as xtick that means x axis will have col1 values
    
   ```python
   
   def univariate_barplots(data, col1, col2='project_is_approved', top=False):
    
    #Count number of zeros in dataframe python: https://stackoverflow.com/a/51540521/4084039
    temp = pd.DataFrame(project_data.groupby(col1)[col2].agg(lambda x: x.eq(1).sum())).reset_index()
    #print(temp)

     #Pandas dataframe grouby count: https://stackoverflow.com/a/19385591/4084039
    temp['total'] = pd.DataFrame(project_data.groupby(col1)[col2].agg(total='count')).reset_index()['total']
    temp['Avg'] = pd.DataFrame(project_data.groupby(col1)[col2].agg(Avg='mean')).reset_index()['Avg']
    
    temp.sort_values(by=['total'],inplace=True, ascending=False)
    
    #if top:
        #temp = temp[0:top]
    
    stack_plot(temp, xtick=col1, col2=col2, col3='total')
    print(temp.head(5))
    print("="*50)
    print(temp.tail(5))
    ```
    
    Calling the created function
    
    ```python
    univariate_barplots(project_data, 'school_state', 'project_is_approved', False)
    ```
    
    ![Screen Shot 2021-10-10 at 10 49 32 AM](https://user-images.githubusercontent.com/90976062/136683336-d033061c-892e-4c7f-91c7-d91fbd59faf9.png)
    
       school_state  project_is_approved  total       Avg
4            CA                13205  15388  0.858136
43           TX                 6014   7396  0.813142
34           NY                 6291   7318  0.859661
9            FL                 5144   6185  0.831690
27           NC                 4353   5091  0.855038
==================================================
   school_state  project_is_approved  total       Avg
39           RI                  243    285  0.852632
26           MT                  200    245  0.816327
28           ND                  127    143  0.888112
50           WY                   82     98  0.836735
46           VT                   64     80  0.800000
```

![Screen Shot 2021-10-10 at 10 49 32 AM](https://user-images.githubusercontent.com/90976062/136683667-e803617b-b186-411d-9836-f4e583bc9b6e.png)

Every state is having more than 80% success rate in approval.

Univariate Analysis: teacher_prefix

```python
univariate_barplots(project_data, 'teacher_prefix', 'project_is_approved' , top=False)
```

![Screen Shot 2021-10-10 at 11 04 10 AM](https://user-images.githubusercontent.com/90976062/136683767-0c671f54-4c33-4e28-88fc-6c082557d6a8.png)

```python
  teacher_prefix  project_is_approved  total       Avg
2           Mrs.                48997  57269  0.855559
3            Ms.                32860  38955  0.843537
1            Mr.                 8960  10648  0.841473
4        Teacher                 1877   2360  0.795339
0            Dr.                    9     13  0.692308
==================================================
  teacher_prefix  project_is_approved  total       Avg
2           Mrs.                48997  57269  0.855559
3            Ms.                32860  38955  0.843537
1            Mr.                 8960  10648  0.841473
4        Teacher                 1877   2360  0.795339
0            Dr.                    9     13  0.692308
```

Univariate Analysis: project_grade_category

```python
univariate_barplots(project_data, 'project_grade_category', 'project_is_approved', top=False)
```

![Screen Shot 2021-10-10 at 11 09 16 AM](https://user-images.githubusercontent.com/90976062/136683874-9bd67d7b-df06-4d5f-ad6b-8fa9bfb459c6.png)

```python
  project_grade_category  project_is_approved  total       Avg
3          Grades PreK-2                37536  44225  0.848751
0             Grades 3-5                31729  37137  0.854377
1             Grades 6-8                14258  16923  0.842522
2            Grades 9-12                 9183  10963  0.837636
==================================================
  project_grade_category  project_is_approved  total       Avg
3          Grades PreK-2                37536  44225  0.848751
0             Grades 3-5                31729  37137  0.854377
1             Grades 6-8                14258  16923  0.842522
2            Grades 9-12                 9183  10963  0.837636
```

Univariate Analysis: project_subject_categories

Before doing the analysis on the above feature we have to do some data cleaning

```python
catogories = list(project_data['project_subject_categories'].values)
# remove special characters from list of strings python: https://stackoverflow.com/a/47301924/4084039

# https://www.geeksforgeeks.org/removing-stop-words-nltk-python/
# https://stackoverflow.com/questions/23669024/how-to-strip-a-specific-word-from-a-string
# https://stackoverflow.com/questions/8270092/remove-all-whitespace-in-a-string-in-python
cat_list = []
for i in catogories:
    temp = ""
    # consider we have text like this "Math & Science, Warmth, Care & Hunger"
    for j in i.split(','): # it will split it in three parts ["Math & Science", "Warmth", "Care & Hunger"]
        for k in j.split(): # this will split each of the catogory based on space "Math & Science"=> "Math","&", "Science"
            j=j.replace('The','') # if we have the words "The" we are going to replace it with ''(i.e removing 'The')
        j = j.replace(' ','') # we are placeing all the ' '(space) with ''(empty) ex:"Math & Science"=>"Math&Science"
        temp+=j.strip()+" " #" abc ".strip() will return "abc", remove the trailing spaces
        temp = temp.replace('&','_') # we are replacing the & value into 
    cat_list.append(temp.strip())
  ```
  
  ```python
  project_data['clean_categories'] = cat_list
project_data.drop(['project_subject_categories'], axis=1, inplace=True)

univariate_barplots(project_data, 'clean_categories', 'project_is_approved', top=20)
```

![Screen Shot 2021-10-10 at 11 20 53 AM](https://user-images.githubusercontent.com/90976062/136684157-5fdee1de-f416-437f-81c2-66fcdec1bd9d.png)

Univariate Analysis: project_subject_subcategories

```python
sub_catogories = list(project_data['project_subject_subcategories'].values)
# remove special characters from list of strings python: https://stackoverflow.com/a/47301924/4084039

# https://www.geeksforgeeks.org/removing-stop-words-nltk-python/
# https://stackoverflow.com/questions/23669024/how-to-strip-a-specific-word-from-a-string
# https://stackoverflow.com/questions/8270092/remove-all-whitespace-in-a-string-in-python

sub_cat_list = []
for i in sub_catogories:
    temp = ""
    # consider we have text like this "Math & Science, Warmth, Care & Hunger"
    for j in i.split(','): # it will split it in three parts ["Math & Science", "Warmth", "Care & Hunger"]
        if 'The' in j.split(): # this will split each of the catogory based on space "Math & Science"=> "Math","&", "Science"
            j=j.replace('The','') # if we have the words "The" we are going to replace it with ''(i.e removing 'The')
        j = j.replace(' ','') # we are placeing all the ' '(space) with ''(empty) ex:"Math & Science"=>"Math&Science"
        temp +=j.strip()+" "#" abc ".strip() will return "abc", remove the trailing spaces
        temp = temp.replace('&','_')
    sub_cat_list.append(temp.strip())
```

```python
project_data['clean_subcategories'] = sub_cat_list
project_data.drop(['project_subject_subcategories'], axis=1, inplace=True)
```

```python
univariate_barplots(project_data, 'clean_subcategories', 'project_is_approved', top=50)
```

![Screen Shot 2021-10-10 at 11 20 53 AM](https://user-images.githubusercontent.com/90976062/136684632-830624b4-740c-4bc4-b0ea-8bcd7b7e8ea9.png)

```python
                clean_subcategories  project_is_approved  total       Avg
317                        Literacy                 8371   9486  0.882458
319            Literacy Mathematics                 7260   8325  0.872072
331  Literature_Writing Mathematics                 5140   5923  0.867803
318     Literacy Literature_Writing                 4823   5571  0.865733
342                     Mathematics                 4385   5379  0.815207
==================================================
                    clean_subcategories  project_is_approved  total       Avg
196       EnvironmentalScience Literacy                  389    444  0.876126
127                                 ESL                  349    421  0.828979
79                   College_CareerPrep                  343    421  0.814727
17   AppliedSciences Literature_Writing                  361    420  0.859524
3    AppliedSciences College_CareerPrep                  330    405  0.814815
```

Univariate Analysis: Cost per project

 we get the cost of the project using resource.csv file
```python 
resource_data.head(2)
```

```python 
 https://stackoverflow.com/questions/22407798/how-to-reset-a-dataframes-indexes-for-all-groups-in-one-step
price_data = resource_data.groupby('id').agg({'price':'sum', 'quantity':'sum'}).reset_index()
``` 

 join two dataframes in python: 
 ```python 
project_data = pd.merge(project_data, price_data, on='id', how='left')
 ```

```python 
 approved_price = project_data[project_data['project_is_approved']==1]['price'].values

rejected_price = project_data[project_data['project_is_approved']==0]['price'].values
```
```python
plt.boxplot([approved_price, rejected_price])
plt.title('Box Plots of Cost per approved and not approved Projects')
plt.xticks([1,2],('Approved Projects','Rejected Projects'))
plt.ylabel('Words in project title')
plt.grid()
plt.show()
```

![Screen Shot 2021-10-10 at 1 03 49 PM](https://user-images.githubusercontent.com/90976062/136686949-31da2117-5261-41cf-a847-7f9374d34752.png)

```python
plt.figure(figsize=(10,3))
sns.distplot(approved_price, hist=False, label="Approved Projects")
sns.distplot(rejected_price, hist=False, label="Not Approved Projects")
plt.title('Cost per approved and not approved Projects')
plt.xlabel('Cost of a project')
plt.legend()
plt.show()
```
![Screen Shot 2021-10-10 at 1 07 27 PM](https://user-images.githubusercontent.com/90976062/136687014-31881195-b743-4080-bb1e-dd2ac49221f3.png)

```python
xj = project_data['teacher_number_of_previously_posted_projects'].head(100)
print(xj)
project_data['xj']=xj
univariate_barplots(project_data, 'xj', 'project_is_approved', False)
```

![Screen Shot 2021-10-10 at 1 11 25 PM](https://user-images.githubusercontent.com/90976062/136687091-884bc3f6-22ed-44c1-9964-ac94ad667f7d.png).

2. Preprocessing of data 

SET 1 :Preprocessing Categorical Features: project_grade_category

```python
project_data['project_grade_category'].value_counts()

Grades PreK-2    32352
Grades 3-5       27244
Grades 6-8       12383
Grades 9-12       8021
Name: project_grade_category, dtype: int64

project_data['project_grade_category'] = project_data['project_grade_category'].str.replace(' ','_')
project_data['project_grade_category'] = project_data['project_grade_category'].str.replace('-','_')
project_data['project_grade_category'] = project_data['project_grade_category'].str.lower()
project_data['project_grade_category'].value_counts()

grades_prek_2    32352
grades_3_5       27244
grades_6_8       12383
grades_9_12       8021
Name: project_grade_category, dtype: int64
```

```python
project_data['project_subject_categories'] = project_data['project_subject_categories'].str.replace(' The ','')
project_data['project_subject_categories'] = project_data['project_subject_categories'].str.replace(' ','')
project_data['project_subject_categories'] = project_data['project_subject_categories'].str.replace('&','_')
project_data['project_subject_categories'] = project_data['project_subject_categories'].str.replace(',','_')
project_data['project_subject_categories'] = project_data['project_subject_categories'].str.lower()
```

 check if we have any nan values are there
 ```python
print(project_data['teacher_prefix'].isnull().values.any())
print("number of nan values",project_data['teacher_prefix'].isnull().values.sum())
True
number of nan values 3
```

```python
project_data['teacher_prefix']=project_data['teacher_prefix'].fillna('Mrs.')
```

```python
project_data['teacher_prefix'] = project_data['teacher_prefix'].str.replace('.','')
project_data['teacher_prefix'] = project_data['teacher_prefix'].str.lower()
project_data['teacher_prefix'].value_counts()
mrs        41776
ms         28741
mr          7770
teacher     1705
dr             8
Name: teacher_prefix, dtype: int64
```

```python
project_data['project_subject_subcategories'] = project_data['project_subject_subcategories'].str.replace(' The ','')
project_data['project_subject_subcategories'] = project_data['project_subject_subcategories'].str.replace(' ','')
project_data['project_subject_subcategories'] = project_data['project_subject_subcategories'].str.replace('&','_')
project_data['project_subject_subcategories'] = project_data['project_subject_subcategories'].str.replace(',','_')
project_data['project_subject_subcategories'] = project_data['project_subject_subcategories'].str.lower()
project_data['project_subject_subcategories'].value_counts()

literacy                              6995
literacy_mathematics                  6089
literature_writing_mathematics        4304
literacy_literature_writing           4108
mathematics                           3940
                                      ... 
economics_nutritioneducation             1
gym_fitness_socialsciences               1
communityservice_music                   1
appliedsciences_warmth_care_hunger       1
economics_health_lifescience             1
Name: project_subject_subcategories, Length: 395, dtype: int64
```

```python
project_data['school_state'].value_counts()
project_data['school_state'] = project_data['school_state'].str.lower()
```
```python
def decontracted(phrase):
    # specific
    phrase = re.sub(r"won't", "will not", phrase)
    phrase = re.sub(r"can\'t", "can not", phrase)

     general
    phrase = re.sub(r"n\'t", " not", phrase)
    phrase = re.sub(r"\'re", " are", phrase)
    phrase = re.sub(r"\'s", " is", phrase)
    phrase = re.sub(r"\'d", " would", phrase)
    phrase = re.sub(r"\'ll", " will", phrase)
    phrase = re.sub(r"\'t", " not", phrase)
    phrase = re.sub(r"\'ve", " have", phrase)
    phrase = re.sub(r"\'m", " am", phrase)
    return phrase
    ```
    
 we are removing the words from the stop words list: 'no', 'nor', 'not'
 
 ```python
stopwords= ['i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're", "you've",\
            "you'll", "you'd", 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', \
            'she', "she's", 'her', 'hers', 'herself', 'it', "it's", 'its', 'itself', 'they', 'them', 'their',\
            'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', "that'll", 'these', 'those', \
            'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', \
            'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of', \
            'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through', 'during', 'before', 'after',\
            'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further',\
            'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more',\
            'most', 'other', 'some', 'such', 'only', 'own', 'same', 'so', 'than', 'too', 'very', \
            's', 't', 'can', 'will', 'just', 'don', "don't", 'should', "should've", 'now', 'd', 'll', 'm', 'o', 're', \
            've', 'y', 'ain', 'aren', "aren't", 'couldn', "couldn't", 'didn', "didn't", 'doesn', "doesn't", 'hadn',\
            "hadn't", 'hasn', "hasn't", 'haven', "haven't", 'isn', "isn't", 'ma', 'mightn', "mightn't", 'mustn',\
            "mustn't", 'needn', "needn't", 'shan', "shan't", 'shouldn', "shouldn't", 'wasn', "wasn't", 'weren', "weren't", \
            'won', "won't", 'wouldn', "wouldn't"]
```

```python
print("printing some random reviews")
print(9, project_data['project_title'].values[9])
print(38, project_data['project_title'].values[38])
print(177, project_data['project_title'].values[177])

printing some random reviews
9 Just For the Love of Reading--\r\nPure Pleasure
38 Kinders Inspired to be on Target in Fitness Part One
177 My Education, My Seating Choice! Flexible Seating in the Classroom.
```
#Combining all the above stundents 
```python
from tqdm import tqdm
def preprocess_text(text_data):
    preprocessed_text = []
    # tqdm is for printing the status bar
    for sentance in tqdm(text_data):
        sent = decontracted(sentance)
        sent = sent.replace('\\r', ' ')
        sent = sent.replace('\\n', ' ')
        sent = sent.replace('\\"', ' ')
        sent = re.sub('[^A-Za-z0-9]+', ' ', sent)
        # https://gist.github.com/sebleier/554280
        sent = ' '.join(e for e in sent.split() if e.lower() not in stopwords)
        preprocessed_text.append(sent.lower().strip())
    return preprocessed_text
    
    preprocessed_titles = preprocess_text(project_data['project_title'].values)
    ```
```python    
print("printing some random reviews")
print(9, preprocessed_titles[9])
print(38,preprocessed_titles[38])
print(177,preprocessed_titles[177])

printing some random reviews
9 love reading pure pleasure
38 kinders inspired target fitness part one
177 education seating choice flexible seating classroom
``` 

Preprocessing Categorical Features: essay

 merge  column text dataframe: 
 ```python
project_data["essay"] = project_data["project_essay_1"].map(str) +\
                        project_data["project_essay_2"].map(str) + \
                        project_data["project_essay_3"].map(str) + \
                        project_data["project_essay_4"].map(str)
```
check if we have any nan values are there
```python
print(project_data['essay'].isnull().values.any())
print("number of nan values",project_data['essay'].isnull().values.sum())

False
number of nan values 0
```
```python
preprocessed_essays = preprocess_text(project_data['essay'].values)
project_data['essay']=preprocessed_essays
```
Preprocessing Numerical Values: price
```python
price_data = resource_data.groupby('id').agg({'price':'sum', 'quantity':'sum'}).reset_index()
```

Now Vectorizing Text data using Bag of words technique
```python
data  = pd.read_csv('preprocessed_data.csv', nrows=80000)
```
```python
y = data['project_is_approved'].values
X = data.drop(['project_is_approved'], axis=1)
```

Now splitting the data 

```python
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, stratify=y)
X_train, X_cv, y_train, y_cv = train_test_split(X_train, y_train, test_size=0.33, stratify=y_train)
```
```python
print(X_train.shape, y_train.shape)
print(X_cv.shape, y_cv.shape)
print(X_test.shape, y_test.shape)

print("="*100)


vectorizer = CountVectorizer(min_df=10,ngram_range=(1,2))
vectorizer.fit(X_train['essay'].values) # fit has to happen only on train data

# we use the fitted CountVectorizer to convert the text to vector
X_train_essay_bow = vectorizer.transform(X_train['essay'].values)
X_cv_essay_bow = vectorizer.transform(X_cv['essay'].values)
X_test_essay_bow = vectorizer.transform(X_test['essay'].values)

print("After vectorizations")
print(X_train_essay_bow.shape, y_train.shape)
print(X_cv_essay_bow.shape, y_cv.shape)
print(X_test_essay_bow.shape, y_test.shape)
print("="*100)

(35912, 8) (35912,)
(17688, 8) (17688,)
(26400, 8) (26400,)
====================================================================================================
After vectorizations
(35912, 77692) (35912,)
(17688, 77692) (17688,)
(26400, 77692) (26400,)
====================================================================================================
```
```python
vectorizer = CountVectorizer()
vectorizer.fit(X_train['school_state'].values) # fit has to happen only on train data

# we use the fitted CountVectorizer to convert the text to vector
X_train_state_ohe = vectorizer.transform(X_train['school_state'].values)
X_cv_state_ohe = vectorizer.transform(X_cv['school_state'].values)
X_test_state_ohe = vectorizer.transform(X_test['school_state'].values)

print("After vectorizations")
print(X_train_state_ohe.shape, y_train.shape)
print(X_cv_state_ohe.shape, y_cv.shape)
print(X_test_state_ohe.shape, y_test.shape)
print(vectorizer.get_feature_names())
print("="*100)

After vectorizations
(35912, 51) (35912,)
(17688, 51) (17688,)
(26400, 51) (26400,)
['ak', 'al', 'ar', 'az', 'ca', 'co', 'ct', 'dc', 'de', 'fl', 'ga', 'hi', 'ia', 'id', 'il', 'in', 'ks', 'ky', 'la', 'ma', 'md', 'me', 'mi', 'mn', 'mo', 'ms', 'mt', 'nc', 'nd', 'ne', 'nh', 'nj', 'nm', 'nv', 'ny', 'oh', 'ok', 'or', 'pa', 'ri', 'sc', 'sd', 'tn', 'tx', 'ut', 'va', 'vt', 'wa', 'wi', 'wv', 'wy']
====================================================================================================
```
```python
vectorizer = CountVectorizer()
vectorizer.fit(X_train['teacher_prefix'].values) # fit has to happen only on train data

# we use the fitted CountVectorizer to convert the text to vector
X_train_teacher_ohe = vectorizer.transform(X_train['teacher_prefix'].values)
X_cv_teacher_ohe = vectorizer.transform(X_cv['teacher_prefix'].values)
X_test_teacher_ohe = vectorizer.transform(X_test['teacher_prefix'].values)

print("After vectorizations")
print(X_train_teacher_ohe.shape, y_train.shape)
print(X_cv_teacher_ohe.shape, y_cv.shape)
print(X_test_teacher_ohe.shape, y_test.shape)
print(vectorizer.get_feature_names())
print("="*100)

After vectorizations
(35912, 5) (35912,)
(17688, 5) (17688,)
(26400, 5) (26400,)
['dr', 'mr', 'mrs', 'ms', 'teacher']
====================================================================================================
```
```python
vectorizer.fit(X_train['project_grade_category'].values) # fit has to happen only on train data

# we use the fitted CountVectorizer to convert the text to vector
X_train_grade_ohe = vectorizer.transform(X_train['project_grade_category'].values)
X_cv_grade_ohe = vectorizer.transform(X_cv['project_grade_category'].values)
X_test_grade_ohe = vectorizer.transform(X_test['project_grade_category'].values)

print("After vectorizations")
print(X_train_grade_ohe.shape, y_train.shape)
print(X_cv_grade_ohe.shape, y_cv.shape)
print(X_test_grade_ohe.shape, y_test.shape)
print(vectorizer.get_feature_names())
print("="*100)

After vectorizations
(35912, 4) (35912,)
(17688, 4) (17688,)
(26400, 4) (26400,)
['grades_3_5', 'grades_6_8', 'grades_9_12', 'grades_prek_2']
===================================================================================================
```
```python
from sklearn.preprocessing import Normalizer
normalizer = Normalizer()
normalizer.fit(X_train['price'].values.reshape(-1,1))

X_train_price_norm = normalizer.transform(X_train['price'].values.reshape(-1,1))
X_cv_price_norm = normalizer.transform(X_cv['price'].values.reshape(-1,1))
X_test_price_norm = normalizer.transform(X_test['price'].values.reshape(-1,1))

print("After vectorizations")
print(X_train_price_norm.shape, y_train.shape)
print(X_cv_price_norm.shape, y_cv.shape)
print(X_test_price_norm.shape, y_test.shape)
print("="*100)

After vectorizations
(35912, 1) (35912,)
(17688, 1) (17688,)
(26400, 1) (26400,)
====================================================================================================
```
```python
from sklearn.preprocessing import Normalizer
normalizer1 = Normalizer()
normalizer1.fit(X_train['teacher_number_of_previously_posted_projects'].values.reshape(-1,1))

X_train_post_project_norm1 = normalizer1.transform(X_train['teacher_number_of_previously_posted_projects'].values.reshape(-1,1))
X_cv_post_project_norm1 = normalizer1.transform(X_cv['teacher_number_of_previously_posted_projects'].values.reshape(-1,1))
X_test_post_project_norm1 = normalizer1.transform(X_test['teacher_number_of_previously_posted_projects'].values.reshape(-1,1))

print("After vectorizations")
print(X_train_post_project_norm1.shape, y_train.shape)
print(X_cv_post_project_norm1.shape, y_cv.shape)
print(X_test_post_project_norm1.shape, y_test.shape)
print("="*100)

After vectorizations
(35912, 1) (35912,)
(17688, 1) (17688,)
(26400, 1) (26400,)
====================================================================================================
```
```python
vectorizer1 = CountVectorizer()
vectorizer1.fit(X_train['clean_subcategories'].values) # fit has to happen only on train data

# we use the fitted CountVectorizer to convert the text to vector
X_train_subcat_ohe1 = vectorizer1.transform(X_train['clean_subcategories'].values)
X_cv_subcat_ohe1 = vectorizer1.transform(X_cv['clean_subcategories'].values)
X_test_subcat_ohe1 = vectorizer1.transform(X_test['clean_subcategories'].values)

print("After vectorizations")
print(X_train_subcat_ohe1.shape, y_train.shape)
print(X_cv_subcat_ohe1.shape, y_cv.shape)
print(X_test_subcat_ohe1.shape, y_test.shape)
print(vectorizer1.get_feature_names())
print("="*100

After vectorizations
(35912, 30) (35912,)
(17688, 30) (17688,)
(26400, 30) (26400,)
['appliedsciences', 'care_hunger', 'charactereducation', 'civics_government', 'college_careerprep', 'communityservice', 'earlydevelopment', 'economics', 'environmentalscience', 'esl', 'extracurricular', 'financialliteracy', 'foreignlanguages', 'gym_fitness', 'health_lifescience', 'health_wellness', 'history_geography', 'literacy', 'literature_writing', 'mathematics', 'music', 'nutritioneducation', 'other', 'parentinvolvement', 'performingarts', 'socialsciences', 'specialneeds', 'teamsports', 'visualarts', 'warmth']
===================================================================================================
```
```python
vectorizer = CountVectorizer()
vectorizer.fit(X_train['clean_subcategories'].values) # fit has to happen only on train data

# we use the fitted CountVectorizer to convert the text to vector
X_train_cat_ohe2 = vectorizer1.transform(X_train['clean_categories'].values)
X_cv_cat_ohe2 = vectorizer1.transform(X_cv['clean_categories'].values)
X_test_cat_ohe2 = vectorizer1.transform(X_test['clean_categories'].values)

print("After vectorizations")
print(X_train_cat_ohe2.shape, y_train.shape)
print(X_cv_cat_ohe2.shape, y_cv.shape)
print(X_test_cat_ohe2.shape, y_test.shape)
print(vectorizer1.get_feature_names())
print("="*100)

After vectorizations
(35912, 30) (35912,)
(17688, 30) (17688,)
(26400, 30) (26400,)
['appliedsciences', 'care_hunger', 'charactereducation', 'civics_government', 'college_careerprep', 'communityservice', 'earlydevelopment', 'economics', 'environmentalscience', 'esl', 'extracurricular', 'financialliteracy', 'foreignlanguages', 'gym_fitness', 'health_lifescience', 'health_wellness', 'history_geography', 'literacy', 'literature_writing', 'mathematics', 'music', 'nutritioneducation', 'other', 'parentinvolvement', 'performingarts', 'socialsciences', 'specialneeds', 'teamsports', 'visualarts', 'warmth']
===================================================================================================
```
```python
from scipy.sparse import hstack
X_tr = hstack((X_train_essay_bow, X_train_state_ohe , X_train_teacher_ohe, X_train_grade_ohe, X_train_price_norm,X_train_post_project_norm1,X_train_subcat_ohe1,X_train_cat_ohe2)).tocsr()
X_cr = hstack((X_cv_essay_bow, X_cv_state_ohe, X_cv_teacher_ohe, X_cv_grade_ohe, X_cv_price_norm,X_cv_post_project_norm1,X_cv_subcat_ohe1,X_cv_cat_ohe2)).tocsr()
X_te = hstack((X_test_essay_bow, X_test_state_ohe, X_test_teacher_ohe, X_test_grade_ohe, X_test_price_norm,X_test_post_project_norm1,X_test_subcat_ohe1,X_test_cat_ohe2)).tocsr()

print("Final Data matrix")
print(X_tr.shape, y_train.shape)
print(X_cr.shape, y_cv.shape)
print(X_te.shape, y_test.shape)
print("="*100)

Final Data matrix
(35912, 77814) (35912,)
(17688, 77814) (17688,)
(26400, 77814) (26400,)
====================================================================================================
```

HYPERPARAMETER TUNING AND APPLYING MODEL 
```python
def batch_predict(clf, data):
    # roc_auc_score(y_true, y_score) the 2nd parameter should be probability estimates of the positive class
    # not the predicted outputs thts why we are using predict_proba not predict
    #here we are dividing data ib batch size of 1000

    y_data_pred = []
    # here we check weather data is divisible by 1000 or not if not , to avoid use decimal we subtract it eg data = 49041 which is not divisible so what we do is we subtract 41     from the data to make it perfectly divisible
    
    tr_loop = data.shape[0] - data.shape[0]%1000 
    # consider you X_tr shape is 49041, then your tr_loop will be 49041 - 49041%1000 = 49000
    # in this for loop we will iterate unti the last 1000 multiplier
    for i in range(0, tr_loop, 1000):
        y_data_pred.extend(clf.predict_proba(data[i:i+1000])[:,1]) # use of [:,1] is because predict_proba gives output for class 1 ( positive) and class0 (negative ) but we                                                                        only  need first one for roc_auc score.
    # we will be predicting for the last data points
    if data.shape[0]%1000 !=0:
        y_data_pred.extend(clf.predict_proba(data[tr_loop:])[:,1])
    
    return y_data_pred
    ```
    
 ```python   
import matplotlib.pyplot as plt
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import roc_auc_score
import math


train_auc = []
cv_auc = []
K = [0.00001,0.0005, 0.0001,0.005,0.001,0.05,0.01,0.1,0.5,1,5,10,50,100]
K1 = [math.log(x) for x in K]


#K = [-5,-3.3,-4,-2.3,-3,-1.3,]
for i in tqdm(K):
    neigh = MultinomialNB(alpha=i, fit_prior=True, class_prior=[0.5,0.5])
    neigh.fit(X_tr, y_train)

    y_train_pred = batch_predict(neigh, X_tr)    
    y_cv_pred = batch_predict(neigh, X_cr)

    # roc_auc_score(y_true, y_score) the 2nd parameter should be probability estimates of the positive class
    # not the predicted outputs        
    train_auc.append(roc_auc_score(y_train,y_train_pred))
    cv_auc.append(roc_auc_score(y_cv, y_cv_pred))

plt.plot(K1, train_auc, label='Train AUC')
plt.plot(K1, cv_auc, label='CV AUC')

plt.scatter(K1, train_auc, label='Train AUC points')
plt.scatter(K1, cv_auc, label='CV AUC points')

plt.legend()
plt.xlabel("K1: hyperparameter")
plt.ylabel("AUC")
plt.title("ERROR PLOTS")
plt.grid()
plt.show()
```
![Screen Shot 2021-10-10 at 8 15 17 PM](https://user-images.githubusercontent.com/90976062/136700706-72ac776c-9a28-4213-83a1-7a5bbdf1de85.png)

BEST HYPERPARAMETER K = 0.1
Applyint k = 0.1 to model and firstly just computing predicted output and then computing AUC plot

```python
from sklearn.naive_bayes import MultinomialNB

# 2. instantiate a Multinomial Naive Bayes model
nb = MultinomialNB(alpha=0.1, fit_prior=True, class_prior=[0.5,0.5])
```
```python
nb.fit(X_tr, y_train)

y_pred_class = nb.predict(X_te)
```



```python
from sklearn.metrics import roc_curve, auc


neigh = MultinomialNB(alpha=K, fit_prior=True, class_prior=[0.5,0.5])
neigh.fit(X_tr, y_train)
# roc_auc_score(y_true, y_score) the 2nd parameter should be probability estimates of the positive class
# not the predicted outputs

y_train_pred = batch_predict(neigh, X_tr)    
y_test_pred = batch_predict(neigh, X_te)

train_fpr, train_tpr, tr_thresholds = roc_curve(y_train, y_train_pred)
test_fpr, test_tpr, te_thresholds = roc_curve(y_test, y_test_pred)

plt.plot(train_fpr, train_tpr, label="train AUC ="+str(auc(train_fpr, train_tpr)))
plt.plot(test_fpr, test_tpr, label="test AUC ="+str(auc(test_fpr, test_tpr)))
plt.legend()
plt.xlabel("FPR")
plt.ylabel("TPR")
plt.title("AUC PLOTS")
plt.grid()
plt.show()
```

![Screen Shot 2021-10-10 at 8 20 26 PM](https://user-images.githubusercontent.com/90976062/136700959-6ce91cd6-0db5-4ac3-a51b-4c24008f1482.png)














    
    
    




