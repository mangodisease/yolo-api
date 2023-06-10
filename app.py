from re import DEBUG, sub
from flask import Flask, render_template, request, redirect, send_file, url_for
from werkzeug.utils import secure_filename, send_from_directory
import os
import subprocess
import boto3
import sys

import argparse
#import os
#import sys
from pathlib import Path

import torch
import torch.backends.cudnn as cudnn

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from models.common import DetectMultiBackend
from utils.dataloaders import IMG_FORMATS, VID_FORMATS, LoadImages, LoadStreams
from utils.general import (LOGGER, check_file, check_img_size, check_imshow, check_requirements, colorstr, cv2,
                           increment_path, non_max_suppression, print_args, scale_coords, strip_optimizer, xyxy2xywh)
from utils.plots import Annotator, colors, save_one_box
from utils.torch_utils import select_device, time_sync

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

@app.route("/detect", methods=["POST"])
def det():
    if 'video' not in request.files:
        return 'No file provided', 400
    file = request.files['video']
    if file.filename == '':
        return 'Invalid file', 400
    file.save(os.path.join(uploads_dir, secure_filename(file.filename)))
    url = os.path.join(uploads_dir, secure_filename(file.filename))
    #print("uploading")
    #client.put_object(Body=file.read(), ACL='public-read', Bucket='yolo', Key='detect.mp4')
    #url = e_url + "/yolo/detect.mp4"
    #print(url)
    print("detecting objects")
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default=ROOT / 'best.pt', help='model path(s)')
    parser.add_argument('--source', type=str, default=url, help='file/dir/URL/glob, 0 for webcam')
    parser.add_argument('--data', type=str, default=ROOT / 'data/coco128.yaml', help='(optional) dataset.yaml path')
    parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=[640], help='inference size h,w')
    parser.add_argument('--conf_thres', type=float, default=0.25, help='confidence threshold')
    parser.add_argument('--iou_thres', type=float, default=0.45, help='NMS IoU threshold')
    parser.add_argument('--max_det', type=int, default=1000, help='maximum detections per image')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view_img', action='store_true', help='show results')
    parser.add_argument('--save_txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save_conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--save_crop', action='store_true', help='save cropped prediction boxes')
    parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --classes 0, or --classes 0 2 3')
    parser.add_argument('--agnostic_nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--visualize', action='store_true', help='visualize features')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--project', default='static', help='save results to project/name')
    parser.add_argument('--name', default='video', help='save results to project/name')
    parser.add_argument('--exist_ok', action='store_false', help='existing project/name ok, do not increment')
    parser.add_argument('--line_thickness', default=3, type=int, help='bounding box thickness (pixels)')
    parser.add_argument('--hide_labels', default=False, action='store_true', help='hide labels')
    parser.add_argument('--hide_conf', default=False, action='store_true', help='hide confidences')
    parser.add_argument('--half', action='store_true', help='use FP16 half-precision inference')
    parser.add_argument('--dnn', action='store_true', help='use OpenCV DNN for ONNX inference')
    #global opt
    opt = parser.parse_args()
    opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1  # expand
    print_args(vars(opt))
    check_requirements(exclude=('tensorboard', 'thop'))#'pycocotools',
    with torch.no_grad():
        detect_yolo(opt=opt)

    obj = secure_filename(file.filename)
    print('Detection Completed and Saved in Run Folder')
    return obj
    #print('Detection Completed and Saved in Run Folder')
    #loc = os.path.join("runs/video", file.filename)
    #print(loc)
    #try:
    #    return url_for('static', filename='video/{}'.format(file.filename))
    #except Exception as e:
    #    return str(e)
        
#yolo detection here
def detect_yolo(opt):
    print("I was here")
    source, weights, view_img, save_txt, imgsz, nosave, project, name, exist_ok, device, dnn, data, half, visualize, augment, conf_thres, iou_thres, classes, agnostic_nms, max_det, save_crop, line_thickness, hide_labels, hide_conf, update  = opt.source, opt.weights, opt.view_img, opt.save_txt, opt.imgsz, opt.nosave, opt.project, opt.name, opt.exist_ok, opt.device, opt.dnn, opt.data, opt.half, opt.visualize, opt.augment, opt.conf_thres, opt.iou_thres, opt.classes, opt.agnostic_nms, opt.max_det, opt.save_crop, opt.line_thickness, opt.hide_labels, opt.hide_conf, opt.update
    source = str(source)
    save_img = not nosave and not source.endswith('.txt')  # save inference images
    is_file = Path(source).suffix[1:] in (IMG_FORMATS + VID_FORMATS)
    is_url = source.lower().startswith(('rtsp://', 'rtmp://', 'http://', 'https://'))
    webcam = source.isnumeric() or source.endswith('.txt') or (is_url and not is_file)
    if is_url and is_file:
        source = check_file(source)  # download

    # Directories
    save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)  # increment run
    (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

    txt='runs'
    # Load model
    device = select_device(device)
    model = DetectMultiBackend(weights, device=device, dnn=dnn, data=data, fp16=half)
    stride, names, pt = model.stride, model.names, model.pt
    imgsz = check_img_size(imgsz, s=stride)  # check image size

    # Dataloader
    if webcam:
        view_img = check_imshow()
        cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadStreams(source, img_size=imgsz, stride=stride, auto=pt)
        bs = len(dataset)  # batch_size
    else:
        dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=pt)
        bs = 1  # batch_size
    vid_path, vid_writer = [None] * bs, [None] * bs
    
    # Run inference
    model.warmup(imgsz=(1 if pt else bs, 3, *imgsz))  # warmup
    seen, windows, dt = 0, [], [0.0, 0.0, 0.0]
    for path, im, im0s, vid_cap, s in dataset:
        t1 = time_sync()
        im = torch.from_numpy(im).to(device)
        im = im.half() if model.fp16 else im.float()  # uint8 to fp16/32
        im /= 255  # 0 - 255 to 0.0 - 1.0
        if len(im.shape) == 3:
            im = im[None]  # expand for batch dim
        t2 = time_sync()
        dt[0] += t2 - t1

        # Inference
        visualize = increment_path(save_dir / Path(path).stem, mkdir=True) if visualize else False
        pred = model(im, augment=augment, visualize=visualize)
        t3 = time_sync()
        dt[1] += t3 - t2

        # NMS
        pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)
        dt[2] += time_sync() - t3

        # Second-stage classifier (optional)
        # pred = utils.general.apply_classifier(pred, classifier_model, im, im0s)

        # Process predictions
        for i, det in enumerate(pred):  # per image
            seen += 1
            if webcam:  # batch_size >= 1
                p, im0, frame = path[i], im0s[i].copy(), dataset.count
                s += f'{i}: '
            else:
                p, im0, frame = path, im0s.copy(), getattr(dataset, 'frame', 0)

            p = Path(p)  # to Path
            save_path = str(save_dir / p.name)  # im.jpg
            txt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  # im.txt
            s += '%gx%g ' % im.shape[2:]  # print string
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            imc = im0.copy() if save_crop else im0  # for save_crop
            annotator = Annotator(im0, line_width=line_thickness, example=str(names))
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(im.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                # Write results
                for *xyxy, conf, cls in reversed(det):
                    if save_txt:  # Write to file
                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                        line = (cls, *xywh, conf) if save_conf else (cls, *xywh)  # label format
                        with open(f'{txt_path}.txt', 'a') as f:
                            f.write(('%g ' * len(line)).rstrip() % line + '\n')


                    if save_img or save_crop or view_img:  # Add bbox to image
                        c = int(cls)  # integer class
                        label = None if hide_labels else (names[c] if hide_conf else f'{names[c]} {conf:.2f}')
                        annotator.box_label(xyxy, label, color=colors(c, True))
                    if save_crop:
                        save_one_box(xyxy, imc, file=save_dir / 'crops' / names[c] / f'{p.stem}.jpg', BGR=True)

            # Stream results
            im0 = annotator.result()
            if view_img:
                if p not in windows:
                    windows.append(p)
                    cv2.namedWindow(str(p), cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)  # allow window resize (Linux)
                    cv2.resizeWindow(str(p), im0.shape[1], im0.shape[0])
                cv2.imshow(str(p), im0)
                cv2.waitKey(1)  # 1 millisecond

            # Save results (image with detections)
            if save_img:
                if dataset.mode == 'image':
                    cv2.imwrite(save_path, im0)
                else:  # 'video' or 'stream'
                    if vid_path[i] != save_path:  # new video
                        vid_path[i] = save_path
                        if isinstance(vid_writer[i], cv2.VideoWriter):
                            vid_writer[i].release()  # release previous video writer
                        if vid_cap:  # video
                            fps = vid_cap.get(cv2.CAP_PROP_FPS)
                            w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                            h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        else:  # stream
                            fps, w, h = 30, im0.shape[1], im0.shape[0]
                        save_path = str(Path(save_path).with_suffix('.mp4'))  # force *.mp4 suffix on results videos
                        vid_writer[i] = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                    vid_writer[i].write(im0)

        # Print time (inference-only)
        LOGGER.info(f'{s}Done. ({t3 - t2:.3f}s)')
    
    # Print results
    t = tuple(x / seen * 1E3 for x in dt)  # speeds per image
    LOGGER.info(f'Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS per image at shape {(1, 3, *imgsz)}' % t)
    if save_txt or save_img:
        s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''
        LOGGER.info(f"Results saved to {colorstr('bold', save_dir)}{s}")
    if update:
        strip_optimizer(weights)  # update model (to fix SourceChangeWarning)
    #print('uploading to s3')
    #my_path = os.path.abspath(os.path.dirname(__file__))
    #path = os.path.join(my_path, save_path)
    #client.put_object(Body=path, ACL='public-read', Bucket='yolo', Key='detected.mp4')

@app.route("/detectX", methods=['POST'])
def detect():
    video = request.files['video']
    
    video.save(os.path.join(uploads_dir, secure_filename(video.filename)))
    print("video")
    #subprocess.run("ls", shell=True)
    subprocess.run([sys.executable, 'detect.py', '--source', os.path.join(uploads_dir, secure_filename(video.filename))], shell=True)

    # return os.path.join(uploads_dir, secure_filename(video.filename))
    obj = secure_filename(video.filename)
    return os.path.join(uploads_dir, secure_filename(video.filename))#obj
    
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
    subprocess.run([sys.executable, 'detect.py', '--source', url], shell=True)

    # return os.path.join(uploads_dir, secure_filename(video.filename))
    #obj = secure_filename(video.filename)
    return url  #obj
    
@app.route("/opencam", methods=['GET'])
def opencam():
    print("here")
    subprocess.run([sys.executable, 'detect.py', '--source', '0'], shell=True)
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
 	return redirect(url_for('static/video/{}'.format(filename), code=200))

if __name__ == "__main__":
	app.run(host="0.0.0.0", port="8080", debug=True)