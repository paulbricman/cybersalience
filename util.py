from transformers import AutoTokenizer, AutoModel
import numpy as np
import matplotlib.pyplot as plt


model = 'distilbert-base-cased'
tokenizer = AutoTokenizer.from_pretrained(model)
model = AutoModel.from_pretrained(model, output_attentions=True)


def attend(corpus, query, model, tokenizer, blacklist=False):
    token_blacklist = [119, 136, 106]  # [1012, 1029, 999]
    full_ids = tokenizer(corpus + '\n\n' + query,
                         return_tensors='pt')['input_ids']
    query_ids = tokenizer(query,
                          return_tensors='pt')['input_ids']
    corpus_ids = tokenizer(corpus + '\n\n',
                           return_tensors='pt')['input_ids']

    attention = [e.detach().numpy() for e in model(full_ids)[-1]][-2]
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
    corpus_tokens = tokenizer.convert_ids_to_tokens(corpus_ids[0])[1:-1]
    plot_attention(attention, corpus_tokens)
    return corpus_tokens, attention


def plot_attention(attention, corpus_tokens):
    plt.matshow([attention])

    x_pos = np.arange(len(corpus_tokens))
    plt.xticks(x_pos, corpus_tokens)

    y_pos = np.arange(1)
    plt.yticks(y_pos, ['query'])

    plt.show()


attend('Anna lived in Constantinople. Omeir lived close to it.',
       'Where did the two live?', model, tokenizer)
