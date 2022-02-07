import streamlit as st
from util import attend, render_html, load


st.set_page_config(
    page_title='👁️ cybersalience')

tokenizer, model = load('bert-base-cased')

st.info(
    'ℹ️ This tool is part of [a suite of experimental tools for thought](https://paulbricman.com/thoughtware) which incorporate AI primitives in knowledge work.')

hide_streamlit_style = '''
            <style>
            # MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            </style>
            '''
st.markdown(hide_streamlit_style, unsafe_allow_html=True)

st.title('👁️ cybersalience')
st.markdown('A synergy of artificial and organic attention.')
st.markdown('---')

cols = st.columns([1, 1])
query = st.sidebar.text_input(
    'driving query', help='Specify the overarching query which will drive the salience map.', value='cybersalience')
duration = st.sidebar.slider('pulse duration (seconds)', 0., 5., step=0.1, value=2.,
                             help='Specify how long the pulse should take')
focus = st.sidebar.slider('focus strength', 0., 1., step=0.01, value=1.,
                          help='Specify how sharp the focus of the salience map should be. Low focus means the salience is distributed more broadly across tokens. High focus means only a handful of tokens will be attended to. `softmax_temperature = 1 - focus`')
color = st.sidebar.color_picker(
    'halo color', help='Specify the color of the halo around tokens being attended to.', value='#2160EA')

font_family = st.sidebar.selectbox(
    'font family', ['Space Grotesk', 'Monospace', 'Times New Roman', 'Arial', 'Helvetica', 'Courier', 'Calibri', 'Georgia'])
font_size = st.sidebar.slider('font size', 10, 20, step=1, value=14,
                              help='Specify how big the text should be.')

style = f'''
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400&display=swap" rel="stylesheet"> 
<style>
container {{ 
    font-size: {font_size}pt;
    font-family: {font_family}; 
    text-align: justify; }}

.glow-large {{
  animation: glow-large {duration}s ease-out infinite alternate;
}}

@-webkit-keyframes glow-large {{
  from {{
    text-shadow: 0 0 10px #fff;
  }}
  
  to {{
    text-shadow: 0 0 10px {color}, 0 0 20px {color}, 0 0 30px {color}, 0 0 40px {color}, 0 0 50px {color};
  }}
}}

.glow-medium {{
  animation: glow-medium {duration}s ease-out infinite alternate;
}}

@-webkit-keyframes glow-medium {{
  from {{
    text-shadow: 0 0 10px #fff;
  }}
  
  to {{
    text-shadow: 0 0 10px {color}, 0 0 20px {color};
  }}
}}

.glow-small {{
  animation: glow-small {duration}s ease-out infinite alternate;
}}

@-webkit-keyframes glow-small {{
  from {{
    text-shadow: 0 0 10px #fff;
  }}
  
  to {{
    text-shadow: 0 0 10px {color};
  }}
}}
</style>'''

if 'content' not in st.session_state.keys():
    st.session_state['content'] = open('write-up.txt').read()

if st.session_state['content'] == None:
    content = st.text_area('content', height=300)
    if st.button('save'):
        st.session_state['content'] = content
        st.experimental_rerun()
else:
    if ('query' not in st.session_state.keys() or query != st.session_state['query']) or \
            ('focus' not in st.session_state.keys() or focus != st.session_state['focus']):
        raw_pars = st.session_state['content'].split('\n')
        pars = []

        with st.spinner('attending to text...'):
            for raw_par in raw_pars:
                if raw_par.strip() != '':
                    corpus_tokens, attention = attend(
                        raw_par, query, model, tokenizer)
                    pars += [render_html(corpus_tokens, attention, focus)]

    if st.sidebar.button('reset content'):
        st.session_state['content'] = None
        st.experimental_rerun()

    content = style + '<container>' + ''.join(pars) + '</container>'
    st.components.v1.html(content, scrolling=True, height=5000)

st.sidebar.markdown('''---
#### resources
- [lexiscore](https://paulbricman.com/thoughtware/lexiscore) and [decontextualizer](https://paulbricman.com/thoughtware/decontextualizer)
- [humane representation of thought](https://vimeo.com/115154289)
- NLP models [attending to, and forming memories of text](https://distill.pub/2016/augmented-rnns/)
- [representational resources](https://paulbricman.com/reflections/representational-resources)
- [attention in machine translation](https://www.youtube.com/watch?v=SysgYptB198)
- ideas related to my [bachelor's project on interpretability](https://paulbricman.com/reflections/distributions-of-meaning)
- [intro resource on attention in cognitive science](https://mitpress.mit.edu/books/handbook-attention)
- awesome [token-level BERT visualization tool](https://github.com/jessevig/bertviz)
- motivation behind [choosing second-to-last layer](https://github.com/jessevig/bertviz)
- a cognitive architecture which doubles down on [firing synchrony across biologically-plausible neurons](https://www.nengo.ai/)
''')
