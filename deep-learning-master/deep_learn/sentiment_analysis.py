{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 59,
   "metadata": {
    "collapsed": false
   },
   "outputs": [],
   "source": [
    "import nltk\n",
    "from nltk.tokenize import word_tokenize\n",
    "from nltk.stem import WordNetLemmatizer \n",
    "import numpy as np\n",
    "import random\n",
    "import pickle\n",
    "from collections import Counter\n",
    "from io import open"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 60,
   "metadata": {
    "collapsed": true
   },
   "outputs": [],
   "source": [
    "lemmatizer = WordNetLemmatizer()\n",
    "hm_lines = 10000000\n",
    "                "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 61,
   "metadata": {
    "collapsed": false
   },
   "outputs": [],
   "source": [
    "def create_lexicon(pos, neg): #Output=> {'format': 500, 'language': 450}\n",
    "    lexicon = []\n",
    "    for fi in [pos, neg]:\n",
    "        with open(fi, 'r') as f:\n",
    "            contents = f.readlines()\n",
    "            for l in contents[:hm_lines]:\n",
    "                all_words = word_tokenize(l.lower())\n",
    "                lexicon += list(all_words)\n",
    "    \n",
    "    lexicon = [lemmatizer.lemmatize(i) for i in lexicon] #starts stemming process\n",
    "    w_counts = Counter(lexicon)#creates a list: {'the': 3444, 'a':889999}\n",
    "    \n",
    "    l2 = []\n",
    "    for w in w_counts:\n",
    "        if 1000 > w_counts[w] > 50:\n",
    "            l2.append(w)\n",
    "    print len(l2)        \n",
    "    return l2"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 62,
   "metadata": {
    "collapsed": false
   },
   "outputs": [],
   "source": [
    "def sample_handling(sample, lexicon, classification): #Output=> {{[0,0,1,0,...],[0,1]},{[1,0,1,0,...],[1,0]}}\n",
    "    featureset = []\n",
    "    with open(sample, 'r') as f:\n",
    "        contents = f.readlines()\n",
    "        for l in contents[:hm_lines]:\n",
    "            current_words = word_tokenize(l.lower())\n",
    "            current_words = [lemmatizer.lemmatize(i) for i in current_words]\n",
    "            \n",
    "            features = np.zeros(len(lexicon))\n",
    "            \n",
    "            for word in current_words:\n",
    "                if word.lower() in lexicon:\n",
    "                    index_value = lexicon.index(word.lower())\n",
    "                    features[index_value] += 1\n",
    "                             \n",
    "            features = list(features)\n",
    "            featureset.append([features, classification])\n",
    "    #print(featureset[:5])\n",
    "        \n",
    "    return featureset"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 63,
   "metadata": {
    "collapsed": true
   },
   "outputs": [],
   "source": [
    "def create_feature_sets_and_labels(pos, neg, test_size=0.1):\n",
    "    lexicon = create_lexicon(pos, neg)\n",
    "    featureset = []\n",
    "    featureset = sample_handling('pos.txt', lexicon, [1,0])\n",
    "    featureset = sample_handling('neg.txt', lexicon, [0,1])\n",
    "    random.shuffle(featureset)\n",
    "    \n",
    "    features = np.array(featureset)\n",
    "    \n",
    "    testing_size = int(test_size * len(features))\n",
    "    \n",
    "    train_x = list(features[:,0][:-testing_size])\n",
    "    train_y = list(features[:,1][:-testing_size])\n",
    "    \n",
    "    test_x = list(features[:,0][-testing_size:])\n",
    "    test_y = list(features[:,1][-testing_size:])\n",
    "    \n",
    "    return train_x, train_y, test_x, test_y\n",
    "    "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 64,
   "metadata": {
    "collapsed": false
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "423\n"
     ]
    }
   ],
   "source": [
    "if __name__ == '__main__':\n",
    "    train_x, train_y, test_x, test_y = create_feature_sets_and_labels('pos.txt', 'neg.txt')\n",
    "    with open('sentiment_set.pickle', 'wb') as f:\n",
    "        pickle.dump([train_x, train_y, test_x, test_y], f)\n",
    "        "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "collapsed": true
   },
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "hide_input": false,
  "kernelspec": {
   "display_name": "Python 2",
   "language": "python",
   "name": "python2"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 2
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython2",
   "version": "2.7.6"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 2
}
