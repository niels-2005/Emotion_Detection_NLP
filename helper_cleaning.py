import random
import re
import string

import demoji
import nltk
from nltk.corpus import stopwords

nltk.download("stopwords")


def remove_stopwords(text: str) -> str:
    """
    Removes stopwords from the input text.

    Args:
        text (str): The input text from which stopwords need to be removed.

    Returns:
        str: The text with all stopwords removed.
    """
    stop_words = set(stopwords.words("english"))
    words = text.split()
    filtered_text = " ".join(word for word in words if word.lower() not in stop_words)
    return filtered_text


def remove_emojis(text: str) -> str:
    """
    Removes emojis from the input text using the `demoji` library.

    Args:
        text (str): The input text from the emojis needs to be removed.

    Returns:
        str: The input text with all emojis removed.
    """
    return demoji.replace(text, "")


def strip_all_entities(text: str) -> str:
    """
    Cleans the input text by removing new line characters, converting to lowercase,
    and stripping out URLs, mentions, non-ASCII characters, and certain punctuation marks.

    Args:
        text (str): The input text to be cleaned.

    Returns:
        str: The cleaned text, with specified characters and patterns removed.
    """
    text = text.replace("\r", "").replace("\n", " ").replace("\n", " ").lower()
    text = re.sub(r"(?:\@|https?\://)\S+", "", text)
    text = re.sub(r"[^\x00-\x7f]", r"", text)
    banned_list = string.punctuation + "Ã" + "±" + "ã" + "¼" + "â" + "»" + "§"
    table = str.maketrans("", "", banned_list)
    text = text.translate(table)
    return text


def clean_hashtags(tweet: str) -> str:
    """
    Processes hashtags in a tweet by removing hashtags at the end of the tweet and stripping the hashtag symbol from those in the middle.

    Args:
        tweet (str): The tweet text to be processed.

    Returns:
    str: The modified tweet text, where middle-positioned hashtags have been converted into plain words by removing the '#' symbol,
         and any hashtags at the tweet's end have been deleted.
    """
    new_tweet = " ".join(
        word.strip()
        for word in re.split("#(?!(?:hashtag)\b)[\w-]+(?=(?:\s+#[\w-]+)*\s*$)", tweet)
    )
    new_tweet2 = " ".join(word.strip() for word in re.split("#|_", new_tweet))
    return new_tweet2


def filter_chars(a: str) -> str:
    """
    Filters out words containing special characters such as '&' and '$' from the input string.

    Args:
        a (str): The input string from which words containing '&' and '$' should be removed.

    Returns:
        str: The cleaned string with words containing the specified special characters removed.
    """
    sent = []
    for word in a.split(" "):
        if ("$" in word) | ("&" in word):
            sent.append("")
        else:
            sent.append(word)
    return " ".join(sent)


def remove_mult_spaces(text: str) -> str:
    """
    Reduces multiple consecutive spaces in the input text to a single space.

    Args:
        text (str): The input text with potentially multiple consecutive spaces.

    Returns:
        str: The input text with all multiple spaces reduced to single spaces.
    """
    return re.sub("\s\s+", " ", text)
