import streamlit as st


st.set_page_config(
    page_title='👁️ semantic salience')

st.info(
    'ℹ️ This tool is part of [a suite of experimental tools for thought](https://paulbricman.com/thoughtware) which incorporate AI primitives in knowledge work.')

hide_streamlit_style = '''
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            </style>
            '''
st.markdown(hide_streamlit_style, unsafe_allow_html=True)

st.title('👁️ semantic salience')
st.markdown('A fusion of artificial and organic attention.')
st.markdown('---')

query = st.text_input(
    'driving query', help='Specify the overarching query which will drive the salience map.')
with st.expander('salience map settings', expanded=False):
    pulsing = st.checkbox('pulsing effect', value=True,
                          help='Specify whether the salience map should be pulsing.')
    duration = None
    if pulsing:
        duration = st.slider('pulse duration (seconds)', 0., 5., step=0.1, value=1.,
                             help='Specify how long the pulse should take')
    focus = st.slider('focus strength', 0., 1., step=0.1, value=0.8,
                      help='Specify how sharp the focus of the salience map should be. Low focus means the salience is distributed more broadly across tokens. High focus means only a handful of tokens will be attended to. `softmax_temperature = 1 - focus`')
    color = st.color_picker(
        'halo color', help='Specify the color of the halo around tokens being attended to.')

with st.expander('text settings', expanded=False):
    font_family = st.selectbox(
        'font family', sorted(['Monospace', 'Times New Roman', 'Arial', 'Helvetica', 'Courier', 'Calibri', 'Georgia', 'Space Grotesk']))
    font_size = st.slider('font size', 10, 20, step=1, value=14,
                          help='Specify how big the text should be.')

style = f'''
<style>
container {{ 
    font-size: {font_size}pt;
    font-family: {font_family}; 
    text-align: justify; }}

.glow {{
  color: black;
  animation: glow {duration}s ease-in-out infinite alternate;
}}

@-webkit-keyframes glow {{
  from {{
    text-shadow: 0 0 10px #fff;
  }}
  
  to {{
    text-shadow: 0 0 10px {color}, 0 0 20px {color}, 0 0 30px {color}, 0 0 40px {color}, 0 0 50px {color};
  }}
}}
</style>'''

if 'content' not in st.session_state.keys() or st.session_state['content'] == None:
    content = st.text_area('content', height=300)
    if st.button('save'):
        st.session_state['content'] = content
        st.experimental_rerun()
else:
    if st.button('reset'):
        st.session_state['content'] = None
        st.experimental_rerun()
    content = style + '<container>' + ''.join(['<p>' + e +
                                               '</p>' for e in st.session_state['content'].split('\n')]) + '</container>'
    st.components.v1.html(content, scrolling=True, height=5000)
