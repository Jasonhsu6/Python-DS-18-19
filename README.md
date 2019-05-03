# Python-DS-18-19
## Data Visualization and Machine Learning

##### Core techniques: Python 3, Pandas, Matplotlib, Scikit-learn, NLTK, Gensim

### 1. Washington metro visualization

Question: How the silver line opening affects ridership of metro stations in DC area. 

Speculation: silver line transport between western and eastern DC and it has many transfer stations with other lines. The line would relieve the stress of some transfer stations that are not part of it, like Gallery Place – Chinatown station. At the same time, other transfer stations like Metro Center and L’Enfant Plaza will receive more passengers.

Data can be accessed: 

<https://planitmetro.com/2016/03/24/data-download-metrorail-ridership-by-station-by-month-2010-2015/>

 <https://planitmetro.com/wp-content/uploads/2016/03/Avg-Weekday-Rail-Ridership-by-Month-by-Station-2010-to-20161.xlsx> 

<https://planitmetro.com/wp-content/uploads/2016/03/Avg-Weekday-Rail-Ridership-by-Month-by-Station-by-Period-2010-to-2016.xlsx> 

### 2. Detroit blight classification

A data challenge from the Michigan Data Science Team ([MDST](http://midas.umich.edu/mdst/)).

The Michigan Data Science Team ([MDST](http://midas.umich.edu/mdst/)) and the Michigan Student Symposium for Interdisciplinary Statistical Sciences ([MSSISS](https://sites.lsa.umich.edu/mssiss/)) have partnered with the City of Detroit to help solve one of the most pressing problems facing Detroit - blight. [Blight violations](http://www.detroitmi.gov/How-Do-I/Report/Blight-Complaint-FAQs) are issued by the city to individuals who allow their properties to remain in a deteriorated condition. Every year, the city of Detroit issues millions of dollars in fines to residents and every year, many of these fines remain unpaid. Enforcing unpaid blight fines is a costly and tedious process, so the city wants to know: how can we increase blight ticket compliance?

The first step in answering this question is understanding when and why a resident might fail to comply with a blight ticket. This is where predictive modeling comes in. The task is to predict whether a given blight ticket will be paid on time.

All data has been provided to us through the [Detroit Open Data Portal](https://data.detroitmi.gov/).

- [Building Permits](https://data.detroitmi.gov/Property-Parcels/Building-Permits/xw2a-a7tf)
- [Trades Permits](https://data.detroitmi.gov/Property-Parcels/Trades-Permits/635b-dsgv)
- [Improve Detroit: Submitted Issues](https://data.detroitmi.gov/Government/Improve-Detroit-Submitted-Issues/fwz3-w3yn)
- [DPD: Citizen Complaints](https://data.detroitmi.gov/Public-Safety/DPD-Citizen-Complaints-2016/kahe-efs3)
- [Parcel Map](https://data.detroitmi.gov/Property-Parcels/Parcel-Map/fxkw-udwf)

------

Two data files for use in training and validating your models: train.csv and test.csv. Each row in these two files corresponds to a single blight ticket, and includes information about when, why, and to whom each ticket was issued. The target variable is compliance, which is True if the ticket was paid early, on time, or within one month of the hearing data, False if the ticket was paid after the hearing date or not at all, and Null if the violator was found not responsible. Compliance, as well as a handful of other variables that will not be available at test-time, are only included in train.csv.

Note: All tickets where the violators were found not responsible are not considered during evaluation. They are included in the training set as an additional source of data for visualization, and to enable unsupervised and semi-supervised approaches. However, they are not included in the test set.



### 3. MNIST 

The MNIST database of handwritten digits, available from this page, has a training set of 60,000 examples, and a test set of 10,000 examples. It is a subset of a larger set available from NIST. The digits have been size-normalized and centered in a fixed-size image.

![img](data:image/png;base64,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%0A)

Data can be accessed through: <http://yann.lecun.com/exdb/mnist/> 

### 4. Adult income

**Abstract**: Predict whether income exceeds $50K/yr based on census data. Also known as "Census Income" dataset.

**Data Set Information:**

Extraction was done by Barry Becker from the 1994 Census database. A set of reasonably clean records was extracted using the following conditions: ((AAGE>16) && (AGI>100) && (AFNLWGT>1)&& (HRSWK>0)) 

### Prediction task is to determine whether a person makes over 50K a year. 

data can be accessed through: <https://archive.ics.uci.edu/ml/datasets/adult>

### 5. NLP topic modelling

## The 20 Newsgroups data set

The 20 Newsgroups data set is a collection of approximately 20,000 newsgroup documents, partitioned (nearly) evenly across 20 different newsgroups. To the best of my knowledge, it was originally collected by Ken Lang, probably for his *Newsweeder: Learning to filter netnews* paper, though he does not explicitly mention this collection. The 20 newsgroups collection has become a popular data set for experiments in text applications of machine learning techniques, such as text classification and text clustering.

#### Organization

The data is organized into 20 different newsgroups, each corresponding to a different topic. Some of the newsgroups are very closely related to each other (e.g. **comp.sys.ibm.pc.hardware / comp.sys.mac.hardware**), while others are highly unrelated (e.g **misc.forsale / soc.religion.christian**). Here is a list of the 20 newsgroups, partitioned (more or less) according to subject matter:

comp.graphics
comp.os.ms-windows.misc
comp.sys.ibm.pc.hardware
comp.sys.mac.hardware
comp.windows.xrec.autos
rec.motorcycles
rec.sport.baseball
rec.sport.hockeysci.crypt
sci.electronics
sci.med
sci.spacemisc.forsaletalk.politics.misc
talk.politics.guns
talk.politics.mideasttalk.religion.misc
alt.atheism
soc.religion.christian

Data can be accessed through: <http://qwone.com/~jason/20Newsgroups/>