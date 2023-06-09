from re import DEBUG, sub
from flask import Flask, render_template, request, redirect, send_file, url_for
from werkzeug.utils import secure_filename, send_from_directory
import os
import subprocess
import boto3

app = Flask(__name__)

session = boto3.session.Session()
e_url ="https://csucc.sgp1.digitaloceanspaces.com"
client = session.client('s3',
                        region_name='sgp1',
                        endpoint_url=e_url,
                        aws_access_key_id="DO00VPT27JE4BC4JV9Z6",
                        aws_secret_access_key="Ym1/UTBzW+05lKIuL6LuYaVZ1H8D1h/Of7W8nOZF1jA")

uploads_dir = os.path.join(app.instance_path, 'uploads')

os.makedirs(uploads_dir, exist_ok=True)

@app.route("/")
def hello_world():
    return render_template('index.html')


@app.route("/detect", methods=['POST'])
def detect():
    print("ok 1")
    video = request.files['video']
    print("ok")
    video.save(os.path.join(uploads_dir, secure_filename(video.filename)))
    print(video)
    #subprocess.run("ls", shell=True)
    subprocess.run(['py', 'detect.py', '--source', os.path.join(uploads_dir, secure_filename(video.filename))], shell=True)

    # return os.path.join(uploads_dir, secure_filename(video.filename))
    obj = secure_filename(video.filename)
    return obj
    
@app.route("/detect2", methods=['POST'])
def detect2():
    print("checking if valid")
    if 'video' not in request.files:
        return 'No file provided', 400
    file = request.files['video']
    if file.filename == '':
        return 'Invalid file', 400
    print("uploading")
    client.put_object(Body=file.read(), ACL='public-read', Bucket='yolo', Key='detect.mp4')
    url = e_url + "/yolo/detect.mp4"
    print(url)
    print("detecting objects")
    #video = request.files['video']
    #video.save(os.path.join(uploads_dir, secure_filename(video.filename)))
    #print(video)
    #subprocess.run("ls", shell=True)
    #os.path.join(uploads_dir, secure_filename(video.filename))
    subprocess.run(['py', 'detect.py', '--source', url], shell=True)

    # return os.path.join(uploads_dir, secure_filename(video.filename))
    #obj = secure_filename(video.filename)
    return url  #obj
    
@app.route("/opencam", methods=['GET'])
def opencam():
    print("here")
    subprocess.run(['py', 'detect.py', '--source', '0'], shell=True)
    return "done"
    

@app.route('/return-files', methods=['GET'])
def return_file():
    obj = request.args.get('obj')
    loc = os.path.join("static", obj)
    print(loc)
    try:
        return send_file(os.path.join("static", obj), attachment_filename=obj)
        # return send_from_directory(loc, obj)
    except Exception as e:
        return str(e)

@app.route('/display/<filename>')
def display_video(filename):
 	print('display_video filename: ' + filename)
 	return redirect(url_for('static/{}'.format(filename), code=200))

#if __name__ == "__main__":
#	app.run(host="0.0.0.0", port="8080", debug=True)