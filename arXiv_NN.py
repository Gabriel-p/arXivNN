import re
from bs4 import BeautifulSoup as BS
import requests
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

# nltk.download('punkt')
# nltk.download('stopwords')


def main(
    categ: str = "astro-ph",
    subcategs: tuple = ("astro-ph.GA", "astro-ph.IM"),
    max_features: int = 1000,
    verb: int = 0,
) -> None:
    """
    Main function to process and classify arXiv submissions using a Neural Network.

    This function performs the following steps:
    1. Retrieves the latest submissions from arXiv for the specified category and
       subcategories.
    2. Loads pre-classified data from a CSV file.
    3. Trains a Neural Network classifier.
    4. Predicts labels for the new arXiv submissions and updates the classification
       dataset if required.

    Parameters:
    -----------
    categ : str, optional
        The main arXiv category to fetch articles from (default is "astro-ph").
    subcategs : tuple of str, optional
        Subcategories within the main category to consider (default is
        ("astro-ph.GA", "astro-ph.IM")).
    max_features : int, optional
        The maximum number of features to use in the text vectorization
        (default is 1000).
    verb : int, optional
        Verbosity level for the training process (default is 0, which means no verbose
        output).

    Returns:
    --------
    None

    Notes:
    ------
    - The function expects a CSV file named 'labels_text.csv' in the current directory.
      This file should contain two columns: 'label' and 'text'.
    - The function modifies the 'df_class' DataFrame in-place with new predictions.
    """

    # Get latest submissions to arXiv
    articles = get_arxiv_new(categ, subcategs)

    # Load file with classified data. Should have two columns named 'label' and
    # 'text' with as many rows as desired. The first column stores the labels,
    # the second one stores the text.
    df_class = pd.read_csv("labels_text.csv")
    # Extract number of labels from input file
    num_classes = int(max(set(df_class['label'])))

    # Train NN
    vectorizer, model = train_NN(num_classes, max_features, df_class, verb)

    # Predict labels and update 'df_class' if required
    predict_label(articles, df_class, vectorizer, model, verb)


def get_arxiv_new(categ, subcategs):
    """
    Download recent submissions from arXiv for the given category. Keep only those
    that belong to any of the sub-categories selected.
    """
    categ = categ.lower()
    subcategs = [_.lower() for _ in subcategs]

    # Download new articles from arXiv.
    print(f"Downloading new arXiv {categ} submissions for sub-categories {subcategs}")
    url = "http://arxiv.org/list/" + categ + "/new"
    html = requests.get(url)
    soup = BS(html.content, features="xml")

    # import pickle
    # # with open('temp.pkl', 'wb') as f:
    # #     pickle.dump((soup), f)
    # # breakpoint()
    # with open('temp.pkl', 'rb') as f:
    #     soup = pickle.load(f)

    # Store urls for later
    dt_tags = soup.find_all('dt')
    all_urls = [_.find_all('a')[1].get('id') for _ in dt_tags]
    all_urls = ["https://arxiv.org/abs/" + _ for _ in all_urls]

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


def extract_text_in_parentheses(text):
    pattern = r'\((.*?)\)'
    matches = re.findall(pattern, text)
    matches = [_.lower() for _ in matches]
    return matches


def preprocess_text(texts):
    """
    Preprocess a list of text strings for natural language processing tasks.

    This function performs several preprocessing steps on each text in the input list:
    1. Converts the text to lowercase.
    2. Removes special characters and numbers.
    3. Tokenizes the text into individual words.
    4. Removes stop words (common words that typically don't carry significant meaning).
    5. Applies stemming to reduce words to their root form.

    Parameters:
    -----------
    texts : list of str
        A list of text strings to be preprocessed.

    Returns:
    --------
    preprocessed_texts : list of str
        A list of preprocessed text strings, where each string contains
        space-separated preprocessed tokens.

    Dependencies:
    - re: For regular expression operations.
    - nltk: For tokenization and stop words.
    - nltk.stem.porter.PorterStemmer: For stemming.

    Note:
    - Ensure that the NLTK library is installed and the necessary NLTK data
      (punkt tokenizer and stopwords) are downloaded before using this function.
    - The function assumes English language texts. For other languages, modify
      the stop words and consider using an appropriate stemmer or lemmatizer.

    Example:
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


def train_NN(num_classes, max_features, df, verb):
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

    Parameters:
    -----------
    num_classes : int
        The number of classes for classification.
    max_features : int
        The maximum number of features to use in the TF-IDF vectorizer.
    df : pandas.DataFrame)
        A DataFrame containing 'text' and 'label' columns.
    verb : int
        Verbosity level for model predictions.

    Returns:
    --------
    A tuple containing two elements:
        - vectorizer (TfidfVectorizer): The fitted TF-IDF vectorizer.
        - model (keras.Sequential): The trained neural network model.

    Dependencies:
    - numpy
    - pandas
    - sklearn (TfidfVectorizer, train_test_split)
    - tensorflow.keras (to_categorical, Sequential, Dense)

    Notes:
    - Assumes the existence of a `preprocess_text` function for text preprocessing.
    - The neural network architecture is fixed: 3 hidden layers (64, 32, 16 units)
      with ReLU activation, and an output layer with softmax activation.
    - Uses 'adam' optimizer and 'categorical_crossentropy' loss.
    - Trains for 100 epochs with a batch size of 32 and 20% validation split.
    - Class labels are assumed to start from 1, not 0 (subtraction is performed before
      conversion to categorical).

    Side Effects:
    - Prints the test accuracy after model evaluation.

    Example:
    >>> df = pd.DataFrame({'abstract': ['text1', 'text2'], 'class': [1, 2]})
    >>> vectorizer, model = train_NN(num_classes=2, max_features=1000, df=df, verb=0)
    """
    texts = df['text'].values
    labels = df['label'].values
    # Preprocess texts
    preprocessed_texts = preprocess_text(texts)

    # Convert to TF-IDF representation
    vectorizer = TfidfVectorizer(max_features=max_features)
    tfidf_matrix = vectorizer.fit_transform(preprocessed_texts)
    X = tfidf_matrix.toarray()

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
        X_train, y_train, epochs=100, batch_size=32, validation_split=0.2, verbose=verb
    )

    # Evaluate the model
    loss, accuracy = model.evaluate(X_test, y_test, verbose=verb)
    print(f"Test accuracy: {accuracy:.4f}")

    return vectorizer, model


def predict_label(articles, df, vectorizer, model, verb):
    """
    Predict labels for new articles, display results, and update a classification
    file based on user input.

    This function processes a list of articles, predicts their labels using a
    pre-trained model, displays the results, and allows the user to confirm or
    modify the predictions. The updated data is then saved to a CSV file.

    Parameters:
    -----------
    df : pandas.DataFrame
        The existing DataFrame containing classified articles.
    articles : list of tuple
        A list of articles, where each article is a tuple containing
        (title, abstract, url).
    vectorizer : TfidfVectorizer
        The text vectorizer used to transform the preprocessed text.
    model :keras.Sequential
        The pre-trained classification model used for predictions.
    verb : int
        Verbosity level for model predictions.

    Returns:
    -------
    None

    Side Effects:
    - Prints prediction results and article information to the console.
    - Updates the "labels_text.csv" file with new classifications.
    - Modifies the input DataFrame 'df' with new entries.

    The function includes a nested function 'classif' which:
    1. Preprocesses the input text.
    2. Transforms the preprocessed text using the vectorizer.
    3. Predicts the label using the provided model.

    The main function:
    1. Predicts labels for all input articles.
    2. Sorts articles based on predicted labels.
    3. Displays each article's information and predicted label.
    4. Prompts the user for input:
       - 'q': Quit the function.
       - 'c': Continue to the next article without saving.
       - Any other input: Save the article with the given or predicted label.
    5. Updates the DataFrame and CSV file with new classifications.

    Note:
    - Requires 'preprocess_text' function to be defined.
    - Assumes the existence of a "labels_text.csv" file for saving results.
    - The classification labels are assumed to start from 1, not 0.

    Dependencies:
    - numpy
    - pandas
    - textwrap

    Example:
    >>> df = pd.DataFrame(columns=['class', 'abstract'])
    >>> articles = [("Title 1", "Abstract 1", "url1"), ("Title 2", "Abstract 2", "url2")]
    >>> predict_label(df, articles, my_vectorizer, my_model, verb)
    """
    def classif(text):
        preprocessed_text = preprocess_text([text])
        text_vector = vectorizer.transform(preprocessed_text).toarray()
        prediction = model.predict(text_vector, verbose=verb)
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
                    or type(user_input) is int:
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
            df.to_csv("labels_text.csv", index=False)


if __name__ == "__main__":
    main()
