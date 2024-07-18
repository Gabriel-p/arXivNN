import os
import re
import warnings
import bs4
import requests_cache
from dateutil import parser
import datetime
import numpy as np
import pandas as pd
import textwrap
from keras.models import Sequential
from keras.layers import Dense, Input
from keras.utils import to_categorical
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer


def main(
    classif_file: str = "labels_text.csv",
    categ: str = "astro-ph",
    subcategs: tuple[str, ...] = ("astro-ph.GA", "astro-ph.IM"),
    max_features: int = 1000,
    verbose: bool = False,
) -> None:
    """
    Main function to process and classify new arXiv submissions using a Neural Network.

    This function performs the following steps:
    1. Loads pre-classified data from a CSV file.
    2. Retrieves the new submissions from arXiv for the specified category and
       subcategories.
    3. Trains a Neural Network classifier.
    4. Predicts labels for the new arXiv submissions.
    5. Updates the classification dataset if required, via user input.

    Parameters
    ----------
    classif_file : str
        Name of the CSV file with the pre-classified data. Should contain two columns,
        one named 'label' with integers, and one named 'text' with the classified
        text.
    categ : str
        The main arXiv category to fetch articles from (default is "astro-ph").
    subcategs : tuple[str, ...]
        Subcategories within the main category to consider (default is
        ("astro-ph.GA", "astro-ph.IM")).
    max_features : int
        The maximum number of features to use in the text vectorization
        (default is 1000).
    verbose : bool
        Verbosity level for the training process (default is False).

    Returns
    -------
    None

    Notes
    -----
    - The function expects a CSV file named {classif_file}. This file should contain
      two columns: 'label' and 'text'.
    - The function modifies the {classif_file} in-place with new predictions.
    """
    # Load previously classified data
    df_class, num_classes = load_classif_file(classif_file)

    # Get latest submissions to arXiv
    articles = get_arxiv_new(categ, subcategs)

    # Train NN
    vectorizer, model = train_NN(df_class, num_classes, max_features, verbose)

    # Predict labels and update 'df_class' if required
    predicted_labels, i_sort = predict_label(articles, vectorizer, model, verbose)

    # Update dataset requesting user input
    get_user_input(classif_file, df_class, articles, predicted_labels, i_sort)


def load_classif_file(classif_file: str) -> tuple[pd.DataFrame, int]:
    """
    Load a file containing classified data and determine the number of classes.

    This function reads a CSV file containing classified data. The file should have
    two columns named 'label' and 'text' with as many rows as desired. The 'label'
    column stores the labels, the 'text' columns stores the text.
    The number of unique classes is extracted from the 'label' column.

    Parameters
    ----------
    classif_file : str
        The path to the CSV file containing the classified data.

    Returns
    -------
    df_class : pd.DataFrame
        A pandas DataFrame containing the loaded data.
    num_classes : int
        The number of unique classes in the 'label' column.

    Notes
    -----
    The input file should have two columns:
    1. 'label': Contains the classification labels.
    2. 'text': Contains the text data associated with each label.

    The number of classes is determined by the maximum value in the 'label' column.

    Examples
    --------
    >>> df, num_classes = load_classif_file("classified_data.csv")
    >>> print(df.head())
       label                                 text
    0      1  This is an example of class 1 text.
    1      2  This is an example of class 2 text.
    2      1  Another example of class 1 text.
    >>> print(num_classes)
    2
    """
    df_class = pd.read_csv(classif_file)

    # Extract number of labels from input file
    num_classes = int(max(set(df_class['label'])))

    return df_class, num_classes


def get_arxiv_new(categ: str, subcategs: tuple[str, ...]) -> list:
    """
    Download new submissions from arXiv for the given category and sub-categories.

    This function fetches new arXiv submissions for a specified main category,
    then filters them based on the provided sub-categories. It extracts the title,
    abstract, and URL for each matching article.

    Parameters
    ----------
    categ : str
        The main arXiv category to search in (e.g., 'cs', 'physics').
    subcategs : tuple[str, ...]
        A list of sub-categories to filter the results. Only articles belonging
        to at least one of these sub-categories will be included.

    Returns
    -------
    list
        A list of articles, where each article is represented as a list containing
        three elements: [title, abstract, url].

    Notes
    -----
    - The function converts all category and sub-category inputs to lowercase.
    - It uses a helper function `get_soup` to fetch and parse the arXiv page.
    - Another helper function `extract_text_in_parentheses` is used to parse
      sub-categories.

    Examples
    --------
    >>> articles = get_arxiv_new('astro-ph', ['astro-ph.GA', 'astro-ph.EP'])
    >>> print(f"Found {len(articles)} articles")
    Found 85 articles
    >>> print(articles[0][0])  # Print the title of the first article
    "Influences of modified Chaplygin dark fluid around a black hole"
    """
    categ = categ.lower()
    subcategs = [subcat.lower() for subcat in subcategs]

    # Download new articles from arXiv.
    print(f"Downloading new arXiv {categ} submissions for sub-categories {subcategs}")
    soup = get_soup(categ)

    # Store urls for later
    dt_tags = soup.find_all('dt')
    all_urls = [tag.find_all('a')[1].get('id') for tag in dt_tags]
    all_urls = ["https://arxiv.org/abs/" + url for url in all_urls]

    # Extract titles and abstracts, only for the matching sub-categories
    dd_tags = soup.find_all('dd')
    articles = []
    for i, dd_element in enumerate(dd_tags):
        subjects = dd_element.find(class_='list-subjects').text
        subcategs_new = extract_text_in_parentheses(subjects)
        # Check if submission fits any sub-category
        if any(element in subcategs_new for element in subcategs):
            title = dd_element.find(class_='list-title mathjax').text
            title = title.split('\n')[1].strip()
            abstract = dd_element.find_all(class_='mathjax')[-1].text.strip()
            articles.append([title, abstract, all_urls[i]])

    return articles


def get_soup(categ: str) -> bs4.BeautifulSoup:
    """
    Fetches and caches the HTML content of the latest articles from a specified
    category on arXiv.

    This function retrieves the HTML content of the latest articles from a specified
    category on arXiv, caches the content to avoid repeated network requests, and
    ensures the cache is up-to-date by checking the date of the cached content against
    the current date. If the cache is outdated, it deletes the old cache and fetches
    the latest content.

    Notes
    -----
    - It uses a helper function `inner_soup` defined internally, to get the BS4 soup
    - The helper function `extract_date_from_soup` is used to extract the date

    Parameters
    ----------
    categ : str
        The category code for the arXiv category (e.g., 'astro-cs.AI' for Artificial
        Intelligence).

    Returns
    -------
    bs4.BeautifulSoup
        A BeautifulSoup object containing the parsed HTML content of the latest
        articles from the specified category.
    """
    cache_name = 'arxiv_cache'
    url = f"http://arxiv.org/list/{categ}/new"

    def inner_soup():
        session = requests_cache.CachedSession(cache_name)
        response = session.get(url)
        return bs4.BeautifulSoup(response.content, features="xml")

    # Get cached version if it exists, or cache a new one
    soup = inner_soup()
    # Extract date
    today_parsed = extract_date_from_soup(soup)

    # Check if cached file is old. If so, delete it and get a new one
    if today_parsed != datetime.date.today():
        os.remove(cache_name + ".sqlite")
        soup = inner_soup()

    return soup


def extract_date_from_soup(soup: bs4.BeautifulSoup) -> datetime.date:
    """
    Extract the date from a BeautifulSoup object containing arXiv HTML.

    This function searches for an <h3> tag in the provided BeautifulSoup object,
    extracts a date string from its text content, and parses it into a date object.

    Parameters
    ----------
    soup : bs4.BeautifulSoup
        A BeautifulSoup object containing the parsed HTML from an arXiv page.

    Returns
    -------
    datetime.date
        The extracted date as a date object.

    Raises
    ------
    ValueError
        If the <h3> tag is not found in the soup object.
        If a date string cannot be extracted from the <h3> tag's text.

    Examples
    --------
    >>> from bs4 import BeautifulSoup
    >>> html = '<h3>New submissions for Wednesday, 17 July 2024</h3>'
    >>> soup = BeautifulSoup(html, 'html.parser')
    >>> extract_date_from_soup(soup)
    datetime.date(2024, 7, 17)

    >>> html_no_date = '<h3>New submissions</h3>'
    >>> soup_no_date = BeautifulSoup(html_no_date, 'html.parser')
    >>> extract_date_from_soup(soup_no_date)
    Traceback (most recent call last):
        ...
    ValueError: Could not parse arXiv HTML, date not found

    >>> html_no_h3 = '<p>Some text</p>'
    >>> soup_no_h3 = BeautifulSoup(html_no_h3, 'html.parser')
    >>> extract_date_from_soup(soup_no_h3)
    Traceback (most recent call last):
        ...
    ValueError: Could not parse arXiv HTML, h3 tag not found

    Notes
    -----
    This function assumes the date format in the <h3> tag is 'DD Month YYYY'.
    It uses a regular expression to extract this pattern and then parses it
    using dateutil.parser.
    """
    # Find the <h3> tag
    h3_tag = soup.find('h3')
    if h3_tag:
        h3_text = h3_tag.text
        # Extract the date
        date_match = re.search(r'\b\d{1,2}\s+[A-Za-z]+\s+\d{4}\b', h3_text)
        if date_match:
            date_str = date_match.group()
            # Parse the date string
            today_parsed = parser.parse(date_str).date()
        else:
            raise ValueError("Could not parse arXiv HTML, date not found")
    else:
        raise ValueError("Could not parse arXiv HTML, h3 tag not found")

    return today_parsed


def extract_text_in_parentheses(text: str) -> list:
    """
    Extracts all text found within parentheses in the given string.

    This function uses a regular expression to find all occurrences of text
    enclosed in parentheses, extracts them, converts them to lowercase,
    and returns them as a list.

    Parameters
    ----------
    text : str
        The input string to search for parenthesized text.

    Returns
    -------
    list
        A list of lowercase strings found within parentheses in the input text.

    Examples
    --------
    >>> extract_text_in_parentheses("Hello (World)! (Python) is (awesome)")
    ['world', 'python', 'awesome']
    """
    pattern = r'\((.*?)\)'
    matches = re.findall(pattern, text)
    matches = [_.lower() for _ in matches]
    return matches


def preprocess_text(texts: list) -> list:
    """
    Preprocess a list of text strings for natural language processing tasks.

    This function performs several preprocessing steps on each text in the input list:
    1. Converts the text to lowercase.
    2. Removes special characters and numbers.
    3. Tokenizes the text into individual words.
    4. Removes stop words (common words that typically don't carry significant meaning).
    5. Applies stemming to reduce words to their root form.

    Parameters
    ----------
    texts : list
        A list of text strings to be preprocessed.

    Returns
    -------
    list
        A list of preprocessed text strings, where each string contains
        space-separated preprocessed tokens.

    Notes
    -----
    Ensure that the NLTK library is installed and the necessary NLTK data
    (punkt tokenizer and stopwords) are downloaded before using this function.

    The function assumes English language texts. For other languages, modify
    the stop words and consider using an appropriate stemmer or lemmatizer.

    This function depends on:
    - re: For regular expression operations.
    - nltk: For tokenization and stop words.
    - nltk.stem.porter.PorterStemmer: For stemming.

    Example
    -------
    >>> texts = ["Hello, world!", "Natural Language Processing is fun!"]
    >>> preprocessed = preprocess_text(texts)
    >>> print(preprocessed)
    ['hello world', 'natur languag process fun']
    """
    preprocessed_texts = []
    for text in texts:
        # Convert to lowercase
        text = text.lower()
        
        # Remove special characters and numbers
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        
        # Tokenize
        tokens = nltk.word_tokenize(text)
        
        # Remove stop words
        stop_words = set(stopwords.words('english'))
        tokens = [token for token in tokens if token not in stop_words]
        
        # Stemming
        stemmer = PorterStemmer()
        tokens = [stemmer.stem(token) for token in tokens]
        
        preprocessed_texts.append(' '.join(tokens))

    return preprocessed_texts


def train_NN(
    df: pd.DataFrame,
    num_classes: int,
    max_features: int,
    verbose: bool
) -> tuple[TfidfVectorizer, Sequential]:
    """
    Train a neural network classifier on text data.

    This function preprocesses text data, converts it to TF-IDF representation,
    and trains a neural network classifier. It performs the following steps:
    1. Preprocesses the input texts.
    2. Converts texts to TF-IDF representation.
    3. Prepares labels for multi-class classification.
    4. Splits data into training and testing sets.
    5. Builds and trains a neural network model.
    6. Evaluates the model on the test set.

    Parameters
    ----------
    df : pd.DataFrame
        A DataFrame containing 'text' and 'label' columns.
    num_classes : int
        The number of classes for classification.
    max_features : int
        The maximum number of features to use in the TF-IDF vectorizer.
    verbose : bool
        Verbosity level for model predictions.

    Returns
    -------
    vectorizer : TfidfVectorizer
        The fitted TF-IDF vectorizer.
    model : Sequential
        The trained neural network model.

    Notes
    -----
    - Assumes the existence of a `preprocess_text` function for text preprocessing.
    - The neural network architecture is fixed: 3 hidden layers (64, 32, 16 units)
      with ReLU activation, and an output layer with softmax activation.
    - Uses 'adam' optimizer and 'categorical_crossentropy' loss.
    - Trains for 100 epochs with a batch size of 32 and 20% validation split.
    - Class labels are assumed to start from 1, not 0 (subtraction is performed before
      conversion to categorical).

    Dependencies:
    - numpy
    - pandas
    - sklearn (TfidfVectorizer, train_test_split)
    - tensorflow.keras (to_categorical, Sequential, Dense)

    Side Effects:
    - Prints the test accuracy after model evaluation.

    Example
    -------
    >>> df = pd.DataFrame({'abstract': ['text1', 'text2'], 'class': [1, 2]})
    >>> vectorizer, model = train_NN(num_classes=2, max_features=1000, df=df, verb=0)
    """
    print(f"Training NN: num_classes={num_classes}, max_features={max_features}")

    texts = list(df['text'].values)
    labels = df['label'].values
    # Preprocess texts
    preprocessed_texts = preprocess_text(texts)

    # Convert to TF-IDF representation
    vectorizer = TfidfVectorizer(max_features=max_features)
    tfidf_matrix = vectorizer.fit_transform(preprocessed_texts)
    X = tfidf_matrix.toarray()
    # Check that the shape matches the vectorized text
    if X.shape[1] < max_features:
        warnings.warn(
            f"Lowering max_features to match the vectorized text: {X.shape[1]}"
        )
        max_features = X.shape[1]

    # Convert labels to categorical
    y = to_categorical(np.array(labels) - 1, num_classes=num_classes)

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)

    # Build the neural network model
    model = Sequential([
        Input(shape=(max_features,)),
        Dense(64, activation='relu'),
        Dense(32, activation='relu'),
        Dense(16, activation='relu'),
        Dense(num_classes, activation='softmax')
    ])

    # Compile the model
    model.compile(
        optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy']
    )

    # Train the model
    model.fit(
        X_train, y_train, epochs=100, batch_size=32, validation_split=0.2,
        verbose=verbose
    )

    # Evaluate the model
    loss, accuracy = model.evaluate(X_test, y_test, verbose=verbose)
    print(f"Test accuracy: {accuracy:.4f}")

    return vectorizer, model


def predict_label(
    articles: list,
    vectorizer: TfidfVectorizer,
    model: Sequential,
    verbose: bool
) -> tuple[list, np.ndarray]:
    """
    Predict labels for new articles and sort them based on predictions.

    This function processes a list of articles, predicts their labels using a
    pre-trained model, and sorts the articles based on their predicted labels.

    Parameters
    ----------
    articles : list
        A list of articles, where each article is a tuple containing
        (title, abstract, url).
    vectorizer : TfidfVectorizer
        The text vectorizer used to transform the preprocessed text.
    model : Sequential
        The pre-trained classification model used for predictions.
    verbose : bool
        Verbosity level for model predictions.

    Returns
    -------
    predicted_labels : list
        Predicted labels for each article.
    i_sort : np.ndarray
        Indices that would sort the articles based on their predicted labels.

    Notes
    -----
    The function includes a nested function 'classif' which:
    1. Preprocesses the input text.
    2. Transforms the preprocessed text using the vectorizer.
    3. Predicts the label using the provided model.

    This function requires the 'preprocess_text' function to be defined.

    The function prints a summary of the number of articles for each predicted label.

    Examples
    --------
    >>> articles = [("Title 1", "Abstract 1", "url1"), ("Title 2", "Abstract 2", "url2")]
    >>> vectorizer = TfidfVectorizer()  # Assume it's fitted
    >>> model = keras.Sequential()  # Assume it's trained
    >>> labels, sort_indices = predict_label(articles, vectorizer, model, verbose=False)
    >>> print(labels)
    [1, 2]
    >>> print(sort_indices)
    [0 1]
    """
    def classif(text):
        preprocessed_text = preprocess_text([text])
        text_vector = vectorizer.transform(preprocessed_text).toarray()
        prediction = model.predict(text_vector, verbose=verbose)
        return np.argmax(prediction) + 1

    predicted_labels = []
    for article in articles:
        text = article[0] + ' ' + article[1]
        predicted_labels.append(classif(text))
    i_sort = np.argsort(predicted_labels)

    print("\n")
    all_labels = list(set(predicted_labels))
    for N_label in all_labels:
        NX = (np.array(predicted_labels) == N_label).sum()
        print(f"Articles labeled {N_label}: {NX}")

    return predicted_labels, i_sort


def get_user_input(
    classif_file: str,
    df: pd.DataFrame,
    articles: list,
    predicted_labels: list,
    i_sort: np.ndarray
) -> None:
    """
    Display results and update a classification file based on user input.

    This function displays the predicted results and allows the user to confirm or
    modify the predictions. The updated data is then saved to a CSV file.

    Parameters
    ----------
    classif_file : str
        Name of the CSV file with the pre-classified data. Should contain two columns,
        one named 'label' with integers, and one named 'text' with the classified
        text.
    df : pd.DataFrame
        The existing DataFrame containing classified articles.
    articles : list
        A list of articles, where each article is a tuple containing
        (title, abstract, url).
    predicted_labels : list
        A list of predicted labels (integers) for each article.
    i_sort : np.ndarray
        An array of indices that sort the articles based on their predicted labels.

    Returns
    -------
    None

    Notes
    -----
    This function has several side effects:
    - Prints prediction results and article information to the console.
    - Updates the input DataFrame 'df' with new classifications.

    The main function:
    1. Displays each article's information and predicted label.
    2. Prompts the user for input:
       - 'q': Quit the function.
       - 'c': Continue to the next article without saving.
       - Any other input: Save the article with the given or predicted label.
    3. Updates the DataFrame and CSV file with new classifications.

    Assumes the existence of a CSV file named {classif_file} for saving results.
    The classification labels are assumed to start from 1, not 0.

    Examples
    --------
    >>> df = pd.DataFrame(columns=['label', 'text'])
    >>> articles = [("Title 1", "Abstract 1", "url1"), ("Title 2", "Abstract 2", "url2")]
    >>> predicted_labels = [1, 2]
    >>> i_sort = np.array([0, 1])
    >>> get_user_input(df, articles, predicted_labels, i_sort)
    # This will start an interactive session in the console
    """
    print(
        "\nAccepted user inputs:\n"
        + "q      : quit,\n"
        + "c      : continue without storing,\n"
        + "Return : store predicted label,\n"
        + "{int}  : store this integer as label"
    )
    for i in i_sort:
        print("\n")
        title, abstract, art_url = articles[i]
        print(f"* Predicted label: {predicted_labels[i]}; {art_url}")
        print("# " + title + "\n")
        print(textwrap.fill(abstract, width=80))
        while True:
            user_input = input("> (q, c, Return, {int}): ")
            if user_input == 'q' or user_input == 'c' or user_input == '' \
                    or is_integer(user_input):
                break
            else:
                print("Unrecognized input. Try again")
        if user_input.lower() == 'q':
            print("\nQuitting")
            return
        elif user_input.lower() == 'c':
            print("\nContinuing to next article without saving")
            continue
        else:
            if user_input != '':
                train = int(user_input)
            else:
                train = int(predicted_labels[i])
            print(f"Add '{train}' to classification file")
            # Update file
            row = {'label': train, 'text': title + ' ' + abstract}
            df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
            df.to_csv(classif_file, index=False)


def is_integer(user_input: str) -> bool:
    """
    Check if the given string can be converted to an integer.

    This function attempts to convert the input string to an integer.
    If successful, it returns True; otherwise, it returns False.

    Parameters
    ----------
    user_input : str
        The string to be checked for integer convertibility.

    Returns
    -------
    bool
        True if the input can be converted to an integer, False otherwise.

    Examples
    --------
    >>> is_integer("123")
    True
    >>> is_integer("-456")
    True
    >>> is_integer("3.14")
    False
    >>> is_integer("abc")
    False
    >>> is_integer("")
    False

    Notes
    -----
    This function uses a try-except block to attempt the conversion,
    which is generally faster than using regular expressions for simple cases.
    """
    try:
        int(user_input)
        return True
    except ValueError:
        return False


if __name__ == "__main__":
    # Download corpora if not already downloaded
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
    main()
