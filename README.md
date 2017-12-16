# Abstract
Language is “...a systematic means of communicating ideas or feelings by the use of conventionalized signs, sounds, gestures, or marks having understood meanings”. In natural language processing, we try to come up with algorithms that can parse information, facts and sentiments quickly, efficiently and accurately from written text or speech. The fundamental problem in solving any of these tasks involve understanding the language itself, specifically its semantics and grammar. Language modeling addresses this key issue - it is a statistical measure of whether a sequence of words semantically makes sense in a particular language. In this report we will take a look at how we can model the English language using a deep learning architecture called Recurrence Neural Network (RNN) with Long-Short Term Memory (LSTM) units. Two modeling approaches are explored - one at a word level and another at a character level. The report compares and contrasts the two approaches through exhaustive experiments and identify the tradeoff and limitations of both the approaches.

# Setup
* Use **setup.sh** to install required packages.
  ```
  ./setup.sh
  ```
* Change hyper parameters in **config.py**
* For training word based model
  ```
  python lstm_word_based.py
  ```
* For character based model
  ```
  ./runCoderun.sh
  ```
* Models are saved in **character_models** OR **word_models** folder

# Read more about the project

* [LSTM presentation](https://docs.google.com/presentation/d/1LwtnqwPpYlWleJ2xdqnkRjoCJ0Y_u8JudFOc1Q4OEMA/edit?usp=sharing)
* [Project presentation](https://docs.google.com/presentation/d/1tXgI2NodX3_m8GfjVIFuq8O2XW7uCTvCGCyOikiALxM/edit?usp=sharing)
* [Paper](https://drive.google.com/file/d/1-dbmYEo_USoIhr_xu0V1nMAEHrK1wN-_/view?usp=sharing)
