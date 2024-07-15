# arXivNN

Simple neural network classifier of arXiv's new submissions
(e.g.: https://arxiv.org/list/astro-ph/new).

Usage:

1. Generate a file called "classifier_NN.csv" with two columns named 'class' and
   'abstract' with as many rows as desired. The first column stores the labels
   from 1 to 4, the second one stores the text
2. Define the `categ` and `subcategs` arguments as desired
3. Run the code

Each new submission in `categ` that matches any of the sub-categories in `subcategs`
will be assigned a label according to the trained NN. The user will be presented
with the classification for each new submission and asked to either:

1. store a new entry with the assigned label (by pressing `Return`)
2. store it with a different label (a number from 1 to 4, in case the NN
    miss-classified it)
3. continue without storing the entry (pressing `c`), or
4. quitting the code (pressing `q`)


