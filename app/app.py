from flask import Flask, render_template, request, url_for
import os
from generate_video import create_vision_video

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        visions_input = request.form['visions']
        visions = [vision.strip() for vision in visions_input.split(',')]
        video_path = create_vision_video(visions)
        video_url = url_for('static', filename=os.path.basename(video_path))
        return render_template('result.html', video_url=video_url)
    return render_template('index.html')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
