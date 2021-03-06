# Python Package for DS-Student-Resources Companion Notebook

This package was built to help students receive a better learning experience and achieve greater level of success in the Data Science Program.
___



[![PyPI version](https://badge.fury.io/py/DS-Students.svg)](https://badge.fury.io/py/DS-Students)[![PyPi license](https://badgen.net/pypi/license/DS-Students/)](https://pypi.com/DS-Students/)[![Downloads](https://pepy.tech/badge/ds-students)](https://pepy.tech/project/ds-students)[![Downloads](https://pepy.tech/badge/ds-students/month)](https://pepy.tech/project/ds-students)[![Downloads](https://pepy.tech/badge/ds-students/week)](https://pepy.tech/project/ds-students)[![Maintenance](https://img.shields.io/badge/Maintained%3F-yes-blue.svg)](https://github.com/woz-u/DS-Student-Resources/graphs/commit-activity)[![GitHub stars](https://badgen.net/github/stars/woz-u/DS-Student-Resources)](https://GitHub.com/woz-u/DS-Student-Resources/)[![GitHub forks](https://badgen.net/github/forks/woz-u/DS-Student-Resources/)](https://GitHub.com/woz-u/DS-Student-Resources/)[![GitHub forks](https://badgen.net/github/watchers/woz-u/DS-Student-Resources/)](https://GitHub.com/woz-u/DS-Student-Resources/)

**Note:** To install the JupyterLab extension into JupyterLab, you also need to have [nodejs](https://nodejs.org/en/download/) installed.


-----

**From the terminal:** 
    
    pip install DS-Students

**From Jupyter:** 
    
    !pip install DS_Students

To Import
-----

    From DS_Students import *

## Use Case 1
#### Using the MultipleChoice function
### Input
```

Q1 = MultipleChoice('Which of these fruits start with the letter A?',['Apples','Bananas','Strawberries'],'Apples')
```
```
display(Q1)
```
### Output
----

![DS-Students](https://github.com/woz-u/DS-Student-Resources/blob/main/DS101-Basic-Statistics/Media/Multiplechoice.png?raw=true)




## Use Case 2
#### Using the MC function
### Input
```

Q1 = MC('Which of these fruits start with the letter A?',['Apples','Bananas','Strawberries'],'Apples', cor='Fun Fact: Over 2,500 varieties of apples are grown in the United States!!', incor='Sorry.. try again!')
```
```
display(Q1)
```
### Output
----

![DS-Students](https://github.com/woz-u/DS-Student-Resources/blob/main/DS101-Basic-Statistics/Media/MC-Correct.png?raw=true)


![DS-Students](https://github.com/woz-u/DS-Student-Resources/blob/main/DS101-Basic-Statistics/Media/MC-Incorrect.png?raw=true)