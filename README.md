# Categorical Naive Bayes

Categorical Naive Bayes prediction method implemented in SparkML. See (presentation)[http://www.slideshare.net/BorysBiletskyy/distributed-categorical-bayes-method) .

Apart from Bernoulli and Multinomial variations of Naive Bayes classifier available in SparkML and inspired by NLP-specific use-cases ( [see e-book referenced in sources](https://nlp.stanford.edu/IR-book/html/htmledition/naive-bayes-text-classification-1.html) ), Categorical Naive Bayes can be applied for classification of objects with conditionally independent categorical attributes.

An example of such a use-case can be predicting fraud transactions based on transaction attributes or predicting an illness based on patients' symptoms and complains.

The proposed Categorical Naive Bayes classifier was tested on (acute inflammations dataset)[https://archive.ics.uci.edu/ml/datasets/Acute+Inflammations)].


## How to run

```
sbt run
```

## How to run tests

```
sbt test
```

