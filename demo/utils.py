from channels import Group
from nltk.tokenize import word_tokenize

import json

def log_to_terminal(socketid, message):
    Group(socketid).send({"text": json.dumps(message)})


def tokenize(sentence):
    return [i for i in re.split(r"([-.\"',:? !\$#@~()*&\^%;\[\]/\\\+<>\n=])", sentence) if i!='' and i!=' ' and i!='\n'];


def prepro_question(s, method='nltk'):
    if method == 'nltk':
        txt = word_tokenize(str(s).lower())
    else:
        txt = tokenize(s)
    return txt
