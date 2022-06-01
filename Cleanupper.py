import re
import pandas as pd
import numpy as np
from collections import Counter
import time
import os
from sklearn.decomposition import PCA
from Predictor import Logit

class Processor:

    def __init__(self, data, stopwords,alpha=0.005,err=.01,min_count=500,tol=10000):
        '''
        Here we intialize all the required attributes of our class.
        Take notice of the structure of feature_voc since this will be
        relevant latter on.
        :param data_path: the path to the data source.
        :param stopwords: a list of stopwords.
        '''
        self.sw = stopwords
        self.voc = None
        self.data = data
        self.model = None
        self.dim_redux = None
        self.alpha = alpha
        self.err = err
        self.min_count=min_count
        self.tol=tol
        self.feature_voc = {'f0': {'pos': None, 'neg': None},
                            'f1': {'pos': None, 'neg': None},
                            'f2': {'pos': None, 'neg': None},
                            'f3': {'pos': None, 'neg': None}}
        print(f"{self.err} | {self.tol}")

    def clean_text_df(self, text_col):
        '''
        This function applies the proc_text function to text_col within self.data
        and returns the values in a new column with name: words_df
        :param text_col: the column to be cleaned
        '''
        self.data = (self
                     .data
                     .assign(clean_text=lambda df: self.proc_text_df(df, text_col, self.sw))
                     .rename({'clean_text': f'words_df'}, axis=1))

    def clean_text(self, text_col):
        '''
        This function applies the proc_text function to text_col within self.data
        and returns the values in a new column with name: words
        :param text_col: the column to be cleaned

        TODO

        Apply your implementation of proc_text from utils to text_col (in this case the column of interest is review)
        and store the result in a new column named words in self.data

        [10 points]
        '''
        needed_col = self.data[text_col]
        needed_fun = lambda row : self.proc_text(row,self.sw)
        self.data['words'] = needed_col.apply(needed_fun)

    def gen_voc(self):
        '''
        This function extracts the vocabulary from the *words* column (the output of clean_text).
        :param min_count: The minimum number of entries the word most have within the corpus.

        TODO

        1.- Generate the vocabulary from *words* column in self.data. as a **SERIES** with ordered
        index. The vocabluary most contain words that have at least *min_count* entries within the corpus.

        2.- Make sure to clean self.data.words (removing the words that are not inside of the vocabulary

        [25 points]
        '''
        query = "count >= " + str(self.min_count)
        #'count+>@min_count'
        self.voc = self.data.words.value_counts().reset_index(name='count').query(query)["index"]

        # Filter words only those in voc.
        self.data.words = self.data.words.isin(self.voc)

    def gen_voc_features(self, feature_col):
        '''

        This function generates a series of positive and negative entries per feature_col

        :param feature_col: the feature col from which the pos an neg vocabularies are generated
        :return: positive voc of features, negative voc of features.
        '''
        w_counts = (self.data
                    .groupby('class')
                    [feature_col]
                    .apply(list)
                    .map(lambda x: [y for z in x for y in z])
                    .reset_index())
        pos_voc = pd.Series(w_counts.reset_index()[feature_col][0]).value_counts(normalize=True)
        neg_voc = pd.Series(w_counts.reset_index()[feature_col][1]).value_counts(normalize=True)
        return pos_voc, neg_voc

    def build_data_features(self):
        '''
        This function adds three feature columns to self.data:

             - bigrams: a column of bigrams from words
             - neg_words: a column of words with 'neg_' prefixes
             - neg_bigrams: a column of bigrams from neg_words

        TODO

        - Apply get_bigrams to words and assign it to a bigrams column in self.data
        - Apply prefix_neg to words and assign it to a neg_words column in self.data
        - Apply get_bigrams to neg_words and assign it to a neg_bigrams column in self.data

        [10 points]
        '''

        self.data['bigrams'] = self.get_bigrams(self.data.words)
        self.data['neg_words'] = self.prefix_neg(self.data.words)
        self.data['neg_bigrams'] = self.get_bigrams(self.data.neg_words)

    def get_feature_voc(self):
        '''
        This function populates feature_voc with features for: words, bigrams, neg_words and neg_bigrams.

        TODO

        Call gen_voc_features for each feature and populate feature_voc.

        [10 points]
        '''
        # Generate data features
        self.build_data_features()
        # Iterate over each feature
        feature_cols = ['words', 'bigrams', 'neg_words', 'neg_bigrams']
        for i, feature in enumerate(feature_cols):
            print(f'Processing feature: {feature}')
            pos, neg = self.gen_voc_features(feature)
            self.feature_voc[f'f{i}']['pos']= pos
            self.feature_voc[f'f{i}']['neg']= neg

    def text_to_features(self, text):
        '''
        This function takes as input a sample text and generates all the necessary features.
        :param text: the text to be processed
        :return: a dataframe with all the textual features
        '''
        features = {'f0_pos': 0, 'f0_neg': 0,
                    'f1_pos': 0, 'f1_neg': 0,
                    'f2_pos': 0, 'f2_neg': 0,
                    'f3_pos': 0, 'f3_neg': 0}
        words = self.proc_text(text, self.sw)
        neg_words = self.prefix_neg(words)
        data_features = {'f0': words, 'f1': self.get_bigrams(words),
                         'f2': neg_words, 'f3': self.get_bigrams(neg_words)}
        for i in range(4):
            if i in [0,2]:
                features[f'f{i}_pos'] = np.mean([self.feature_voc[f'f{i}']['pos'][w]
                                                 if w in self.feature_voc[f'f{i}']['pos'] else 0
                                                 for w in data_features[f'f{i}']])
                features[f'f{i}_neg'] = np.mean([self.feature_voc[f'f{i}']['neg'][w]
                                                 if w in self.feature_voc[f'f{i}']['neg'] else 0
                                                 for w in data_features[f'f{i}']])
            else:
                features[f'f{i}_pos'] = np.mean([self.feature_voc[f'f{i}']['pos'][tuple(w)]
                                                 if tuple(w) in self.feature_voc[f'f{i}']['pos'] else 0
                                                 for w in data_features[f'f{i}']])
                features[f'f{i}_neg'] = np.mean([self.feature_voc[f'f{i}']['neg'][tuple(w)]
                                                 if tuple(w) in self.feature_voc[f'f{i}']['neg'] else 0
                                                 for w in data_features[f'f{i}']])
        return pd.DataFrame(features, index=[1])

    def create_feature_matrix(self, df, text_col):
        '''
        This function constructs a dataframe of features from a given text_col
        :param df: a data frame to process
        :param text_col: the text_col from which features are extracted
        :return: a feature dataframe
        '''
        feature_df = df[text_col].map(lambda x: self.text_to_features(x))
        return pd.concat(feature_df.values, axis=0)*100

    def predict(self, df, text_col):
        '''
        This function takes as input a new data frame together with a text column to predict sentiment.
        :param df: new dataset
        :param text_col: text column over which predict sentiment
        :return: an array of probabilities of sentiment.
        '''
        X = self.create_feature_matrix(df, text_col)
        X_redux = self.dim_redux.transform(X)
        new_X = np.hstack((np.ones(X_redux.shape[0]).reshape(-1, 1), X_redux)).T
        # Generate predictions and shape them as grid.
        return self.model.forward(new_X)

    def train_model(self, text_col, sent_col,plot=True):
        '''
        This function trains our model with the given text and sentiment column.
        :param text_col: the column containing the text to be classified
        :param sent_col: the column containing the true sentiment labels
        '''
        print('Creating feature matrix...')
        X = self.create_feature_matrix(self.data, text_col)
        self.dim_redux = PCA(n_components=2).fit(X)
        X_redux = self.dim_redux.transform(X)
        print(X_redux[:10])
        print(self.data[sent_col].values[:10])
        # Instantiate model
        
        self.model = Logit(X_redux, self.data[sent_col].values,alpha=self.alpha,err=self.err,tol=self.tol)
        # Optmize parameters
        self.model.optimize()
        # Generate classification region plot.
        if plot:
            self.model.plot_region(X_redux)

    def get_bigrams(self,words):
        '''
        :param words: a list of words
        :return: a list of bigrams
        This function returns a list of bigrams (sets of two words) from a given list of words. For example:
        ['this', 'is', 'a', '.', 'very', 'simple', 'example', '.']
        ->
        [['this', 'is'], ['is', 'a'], ['a', '.'], ['.', 'very'], ['very', 'simple'], ['simple', 'example'], ['example', '.']]
        TODO
        Implement get_bigrams
        [10 points]
        '''
        bigrams =  [words[x:x+2] for x in range(0, len(words),1)]  # YOUR CODE GOES HERE
        return bigrams

    def prefix_neg(self,words):
        '''
        :param words: a list of words to process.
        :return: a list of words with the 'neg_' suffix.
        This function receives a list of words and appends a 'neg_' suffix to every word following a negation (no, not, nor)
        and up to the next punctuation mark (., !, ?). For example:
        ['not', 'complex', 'example', '.', 'could', 'not', 'simpler', '!']
        ->
        ['not', 'neg_complex', 'neg_example', '.', 'could', 'not', 'neg_simpler', '!']
        TODO
        Implement prefix_neg
        HINT: you might find the statment 'continue' useful in your implementation, althought it is not neceessary.
        [15 points]
        '''
        after_neg = False
        proc_words = []
        for word in words:
            if word == 'no' or word == 'not' or word == 'nor':
                after_neg = True
                proc_words.append(word)
            else:
                if word =='.'or word == '!' or word == '?':
                    proc_words.append(word)
                    after_neg = False
                else:
                    if after_neg:
                        proc_words.append('neg_'+word)
                    else:
                        proc_words.append(word)# YOUR CODE GOES HERE
        return proc_words

    def filter_voc(self,words, voc):
        mask = np.ma.masked_array(words, ~np.in1d(words, voc))
        return mask[~mask.mask].data

    def proc_text(self,text, sw):
        '''
            This function takes as input a non processed string and returns a list
            with the clean words.
            :param text: The text to be processed
            :param sw: A list of stop words
            :return: a list containing the words of the clean text.
            TODO
            Implement the following transformations:
            1.- Set the text to lowercase.
            2.- Make explicit negations:
                  - don't -> do not
                  - shouldn't -> should not
                  - can't -> ca not (it's ok to leave 'ca' like that making it otherwise will need extra fine tunning)
            3.- Clean html and non characters (except for '.', '?' and '!')
            4.- Add spacing between punctuation and letters, for example:
                    - .hello -> . hello
                    - goodbye. -> goodbye
            5.- Truncaters punctuation and characters with multiple repetitions into three repetitions.
                Punctuation marks are consider 'multiple' with at least **two** consecutive instances: ??, !!, ..
                Characters are considre 'multiple' with at least **three**  consecutive instances: goood, baaad.
                for example:
                    - very gooooooood! -> very goood
                    - awesome !!!!!! -> awesome !!!
                    - nooooooo !!!!!! -> nooo !!!
                    - how are you ?? -> how are you ???
                NOTE: Think about the logic behind this transformation.
            6.- Remove whitespace from start and end of string
            7.- Return a list with the clean words after removing stopwords (DO NOT REMOVE NEGATIONS!) and
                **character** (letters) strings of length 2 or less. for example:
                   - ll, ss
            8.- Removes any single letter: 'z', 'w', 'b', ...
        This is the most important function and the only one that will include a test case:
        input = 'this...... .is a!! .somewhat??????,, # % & messy 234234 astring... noooo.asd HELLLLOOOOOO!!! what is thiiiiiiissssss?? <an> <HTML/> <tag>tagg</tag>final. test! z z w w y y t t'
        expected_output = ['...', '.', '!!!', '.', 'somewhat', '???', 'messy', 'astring', '...', 'nooo', '.', 'asd', 'helllooo', '!!!', 'thiiisss', '???', 'tagg', 'final', '.', 'test', '!']
        [40 points]
        '''

        words = str.lower(text)
        words=re.sub(r"n\'t", " not", words)
        words=re.sub(re.compile('<.*?>'),'',words)
        words=re.sub(r'[^a-z-.?! ]','', words)
        words=re.sub( r'([!,?.])([a-z])', r'\1 \2', words)
        words=re.sub( r'([a-z])([,.!])', r'\1 \2', words)
        letter_ocurrances = Counter(words).items()
        for key, value in letter_ocurrances:
            if value>3 and key!=' ' and key*value in words:
                words = words.replace(key*value,key*3)
            if value == 2 and key != ' ' and key*value in words:
                words = words.replace(key*value,key)
        words=re.sub(r"^\s+|\s+$", "",words)
        lista =[word for word in words.split() if word not in sw]
        # YOUR CODE GOES HERE

        return lista

