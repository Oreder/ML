# ML

# Part 01. Regression

Install the dependencies and devDependencies and let start.

```sh
pip install numpy

pip install scipy

pip install scikit-learn

pip install matplotlib

pip install pandas

pip install quandl
```

To begin, what is regression in terms of us using it with machine learning? The goal is to take continuous data, find the equation that best fits the data, and be able forecast out a specific value. With simple linear regression, you are just simply doing this by creating a best fit line.

A popular use with regression is to predict stock prices. This is done because we are considering the fluidity of price over time, and attempting to forecast the next fluid price in the future using a continuous dataset.

Regression is a form of supervised machine learning, which is where the scientist teaches the machine by showing it features and then showing it what the correct answer is, over and over, to teach the machine. Once the machine is taught, the scientist will usually "test" the machine on some unseen data, where the scientist still knows what the correct answer is, but the machine doesn't. The machine's answers are compared to the known answers, and the machine's accuracy can be measured. If the accuracy is high enough, the scientist may consider actually employing the algorithm in the real world.

Since regression is so popularly used with stock prices, we can start there with an example. To begin, we need data. Sometimes the data is easy to acquire, and sometimes you have to go out and scrape it together, like what we did in an older tutorial series using machine learning with stock fundamentals for investing. In our case, we're able to at least start with simple stock price and volume information from Quandl.

* 1.1 - Intro and data