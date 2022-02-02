import dis
import numpy as np
import matplotlib.pyplot as plt
from transformers import AutoTokenizer, AutoModel
import streamlit as st
import re


def attend(corpus, query, model, tokenizer, blacklist=False):
    token_blacklist = [119, 136, 106]
    query = '\n\n---\n\n' + query
    full_ids = tokenizer(corpus + '\n\n' + query,
                         return_tensors='pt')['input_ids']
    query_ids = tokenizer(query,
                          return_tensors='pt')['input_ids']
    corpus_ids = tokenizer(corpus + '\n\n',
                           return_tensors='pt')['input_ids']

    attention = [e.detach().numpy() for e in model(full_ids)[-1]][-1]
    attention = np.array([e[1:-1]
                         for e in np.mean(attention, axis=(0, 1))[1:-1]])

    if blacklist:
        prune_idx = [e_idx - 1 for e_idx, e in enumerate(
            corpus_ids[0]) if e in token_blacklist]
        valid = [r for r in range(attention.shape[0]) if r not in prune_idx]
        attention = attention[valid][:, valid]
        corpus_ids = [[e for e in corpus_ids[0] if e not in token_blacklist]]

    attention = [e[:len(corpus_ids[0]) - 2]
                 for e in attention[-(len(query_ids[0]) - 2):]]

    attention = np.mean(attention, axis=0)
    corpus_tokens = tokenizer.convert_ids_to_tokens(
        corpus_ids[0], skip_special_tokens=True)
    # plot_attention(attention, corpus_tokens)
    return corpus_tokens, attention


def plot_attention(attention, corpus_tokens):
    plt.matshow(attention)

    x_pos = np.arange(len(corpus_tokens))
    plt.xticks(x_pos, corpus_tokens)

    y_pos = np.arange(len(attention))
    plt.yticks(y_pos, ['query'] * len(attention))

    plt.show()


def softmax(x, temperature):
    e_x = np.exp(x / temperature)
    return e_x / e_x.sum()


def render_html(corpus_tokens, attention, focus=0.99):
    # focus = focus * 0.5 + 0.5
    mu = np.median(attention)
    sigma = np.std(attention)

    raw = ''

    distribution = [0, 0, 0]
    for e_idx, e in enumerate(corpus_tokens):
        if attention[e_idx] > mu + focus * sigma * 8:
            distribution[2] += 1
            raw += ' <span class="glow-large">' + e + '</span>'
        elif attention[e_idx] > mu + focus * sigma * 2:
            distribution[1] += 1
            raw += ' <span class="glow-medium">' + e + '</span>'
        elif attention[e_idx] > mu + focus * sigma * 1:
            distribution[0] += 1
            raw += ' <span class="glow-small">' + e + '</span>'
        else:
            raw += ' ' + e

    print(distribution)
    raw = re.sub(r'\s##', '', raw)
    raw = re.sub(r'\s(\.|,|!|\?|;|\))', r'\1', raw)
    raw = re.sub(r'\(\s', r'(', raw)
    raw = re.sub(r'\s(-|\'|’)\s', r'\1', raw)
    raw = re.sub(r'\s<span class="glow-(small|medium|large)">##',
                 r'<span class="glow-\1">', raw)
    raw = raw.strip()
    return raw


@ st.cache(allow_output_mutation=True)
def load(model='distilbert-base-cased'):
    tokenizer = AutoTokenizer.from_pretrained(model)
    model = AutoModel.from_pretrained(model, output_attentions=True)
    return tokenizer, model
