import streamlit as st
import os
import imageio
import tensorflow as tf
import difflib

from utils import load_data, num_to_char
from modelutil import load_model

st.set_page_config(layout="wide")

st.markdown("""
<style>
.block-container {
    padding-top: 2rem;
    padding-left: 4rem;
    padding-right: 4rem;
}
.title {
    font-size: 36px;
    font-weight: 700;
    color: white;
}
.subtitle {
    color: #9aa0a6;
    margin-bottom: 20px;
}
.section {
    background-color: #161b22;
    padding: 15px;
    border-radius: 10px;
    margin-bottom: 15px;
}
.output-box {
    background-color: #1f6f4a;
    padding: 10px;
    border-radius: 6px;
    color: white;
    text-align: center;
}
.center {
    display: flex;
    justify-content: center;
}
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="title">Visual Speech Recognition</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Understand speech from lip movements</div>', unsafe_allow_html=True)

data_path = os.path.join('data', 's1')
options = os.listdir(data_path)

selected_video = st.selectbox("Select Video", options)

file_path = os.path.join(data_path, selected_video)

col1, col2 = st.columns([1,1])

with col1:
    st.markdown('<div class="section">', unsafe_allow_html=True)
    st.markdown("### Video")

    output_path = "temp.mp4"
    os.system(f'ffmpeg -loglevel quiet -i "{file_path}" -vcodec libx264 -acodec aac "{output_path}" -y')

    video_bytes = open(output_path, 'rb').read()

    st.markdown('<div class="center">', unsafe_allow_html=True)
    st.video(video_bytes)
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)

with col2:
    st.markdown('<div class="section">', unsafe_allow_html=True)
    st.markdown("### Model View")

    video, _ = load_data(tf.convert_to_tensor(file_path))
    video_np = video.numpy()
    video_np = (video_np * 255).astype('uint8')
    video_np = video_np.squeeze(-1)

    imageio.mimsave('anim.gif', video_np, fps=10)

    st.markdown('<div class="center">', unsafe_allow_html=True)
    st.image('anim.gif', width=350)
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="section">', unsafe_allow_html=True)
    st.markdown("### Prediction")

    model = load_model()
    yhat = model.predict(tf.expand_dims(video, axis=0))

    decoder = tf.keras.backend.ctc_decode(
        yhat,
        input_length=[75],
        greedy=False,
        beam_width=100
    )[0][0].numpy()

    decoded = decoder[decoder != -1]
    prediction = tf.strings.reduce_join(num_to_char(decoded)).numpy().decode('utf-8')

    try:
        from textblob import TextBlob
        corrected = str(TextBlob(prediction).correct())
    except:
        corrected = prediction

    valid_words = ["bin","blue","by","again","set","lay","place","red","green","at","in","with"]

    def closest_word(word):
        match = difflib.get_close_matches(word, valid_words, n=1)
        return match[0] if match else word

    final = " ".join([closest_word(w) for w in corrected.split()])

    st.write("Raw:", prediction)
    st.markdown(f'<div class="output-box">{final}</div>', unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)