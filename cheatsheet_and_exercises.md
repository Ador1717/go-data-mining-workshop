# Go cheatsheet and exercises

## Golang vs Python syntax
| Concept   | Golang    |   Python  |
|-----------|-----------|-----------|
| Variable declaration  | var x int = 10 or x := 10 | x = 10|
| Print| fmt.Println("Hello")| print("Hello")
| Function| func add(a int, b int) int { return a + b } | def add(a, b): return a + b|
| If statement| if x > 0 { ... } else { ... }| if x > 0:\n ...\nelse:\n ...
|For loop| for i := 0; i < 5; i++ { ... }| for i in range(5):|
| While loop| for x < 10 {...}| while x < 10:|
| Array / List| [3]int{1, 2, 3}| [1, 2, 3]|
| Slice / List| []int{1, 2, 3}| [1, 2, 3]|
|Map /Dict| map[string]int{"a":1}| {"a":1}|
|Import| import "fmt"| import math|
|Error handling| if err != nil {...}| try:\n ...\n except Exception as e:|
|Comments| // single line| # single line|
|  | /* multi-line*/ | '''multi-line'''|

## To run an exercise:
- open a terminal in VSCode
- navigate to the project root 
- run: ``` go run main.go```
- enter number of selected exercise

__Note the results of the exercises for an advantage in the kahoot session__

## Exercise 1: Regression
### 1.1  Choose a different feature to predict the house prices on and find the best predictor.​
Which feature is the best predictor for house prices?
- Open the regression dataset -> datasets/housing_prices.csv
- choose a feature and remember the index of it
- Open exercises/regression/housing_regression.go
- change the featureIndex to the feature you want to use
- rerun main.go to see the results
- try other features


### 1.2 Predict the price of a house based on its size.
How much is a house of 15 square meters?
- uncomment the code in the regression file that let's you predict the price
- don't forget to adjust the size
- rerun main.go to see the results

## Exercise 2: Classification
### 2.1 Find the best k value to classify people into morning persons and night owls
Which k value results in the highest accuracy and what is the accuracy?
Does increasing k always improve accuracy?
- Open exercises/classification/sleep_classification.go
- modify the k value in the Run() function
- try at least three different values and observe the accuracy changes
- don't forget to rerun main after making changes

### 2.2 Use the classifier to predict based on new values
Are you a night owl or a morning person? Do you agree with the result?
- uncomment the code that lets you make a prediction
- rerun main.go
- in the terminal, enter separated by a space:
    - wheter you drink energy drinks (0=false, 1=true)
    - hours of sleep you get
    - your avg screen time before bed
    - the hour you usually wake up at
- use the best k value you found in the previous task

## Exercise 3: Clustering
### 3.1 Cluster lifestyles based on entertainment spending and gaming hours
Which number of clusters makes most sense here?
- Open exercises/clustering/zones_clustering.go
- adjust the value of k in the Run() function 
- rerun main.go
- check the visual output and adjust if necessary

### 3.2 Create clusters based on weekly exercise hours and hours of sleep per night
What number of clusters fits best here?
- adjust the code to change the selected features
- rerun main.go
- check the visual output and adjust k if necessary
