import streamlit as st
from util import attend, render_html, load


st.set_page_config(
    page_title='👁️ semantic salience',
    layout='wide')

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

st.title('👁️ semantic salience')
st.markdown('A fusion of artificial and organic attention.')
st.markdown('---')

cols = st.columns([1, 2.5])
query = cols[0].text_input(
    'driving query', help='Specify the overarching query which will drive the salience map.')
duration = cols[0].slider('pulse duration (seconds)', 0., 5., step=0.1, value=1.,
                          help='Specify how long the pulse should take')
focus = cols[0].slider('focus strength', 0., 1., step=0.01, value=0.95,
                       help='Specify how sharp the focus of the salience map should be. Low focus means the salience is distributed more broadly across tokens. High focus means only a handful of tokens will be attended to. `softmax_temperature = 1 - focus`')
color = cols[0].color_picker(
    'halo color', help='Specify the color of the halo around tokens being attended to.', value='#EA2121')

font_family = cols[0].selectbox(
    'font family', sorted(['Monospace', 'Times New Roman', 'Arial', 'Helvetica', 'Courier', 'Calibri', 'Georgia', 'Space Grotesk']))
font_size = cols[0].slider('font size', 10, 20, step=1, value=14,
                           help='Specify how big the text should be.')

style = f'''
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

if 'content' not in st.session_state.keys() or st.session_state['content'] == None:
    content = cols[1].text_area('content', height=300)
    if cols[1].button('save'):
        st.session_state['content'] = content
        st.experimental_rerun()
else:
    if ('query' not in st.session_state.keys() or query != st.session_state['query']) or \
            ('focus' not in st.session_state.keys() or focus != st.session_state['focus']):
        corpus_tokens, attention = attend(
            st.session_state['content'], query, model, tokenizer)
        content = render_html(corpus_tokens, attention, focus)

    if cols[0].button('reset content'):
        st.session_state['content'] = None
        st.experimental_rerun()

    content = style + '<container><p>' + content + '</p></container>'
    with cols[1]:
        st.components.v1.html(content, scrolling=True, height=5000)
