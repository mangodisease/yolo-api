from csv import DictWriter
import pandas as pd
from utils.torch_utils import select_device, time_sync
from utils.plots import Annotator, colors, save_one_box
from utils.general import (LOGGER, check_file, check_img_size, check_imshow, check_requirements, colorstr, cv2,
                           increment_path, non_max_suppression, print_args, scale_coords, strip_optimizer, xyxy2xywh,
                           set_logging, apply_classifier)
from utils.dataloaders import IMG_FORMATS, VID_FORMATS, LoadImages, LoadStreams
from models.common import DetectMultiBackend
from re import DEBUG, sub
from flask import Flask, flash, render_template, request, redirect, send_file, url_for, send_from_directory, session
from werkzeug.utils import secure_filename
import os
import subprocess
import boto3
import sys

import argparse
# import os
# import sys
from pathlib import Path

import torch
import torch.backends.cudnn as cudnn
import time

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative


app = Flask(__name__)
app.secret_key = b'_5#y2L"F4Q8z\n\xec]/'


uploads_dir = os.path.join(app.instance_path, 'uploads')

os.makedirs(uploads_dir, exist_ok=True)


@app.route("/")
def home():
    print(session.get('login'))
    if session.get('login') is None:
        return redirect(url_for('login'))
    else:
        return render_template('index.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    error = ""
    if session.get('login') is True:
        return redirect(url_for('home'))
    if request.method == 'POST':
        if request.form['username'] != 'admin' or request.form['password'] != 'password':
            error = 'Invalid Credentials. Please try again.'
        else:
            session['login'] = True
            return redirect(url_for('home'))
    return render_template('login.html', error=error)

@app.route('/logout' , methods=['POST'])
def logout():
    session.clear()
    return redirect(url_for('home'))

@app.route("/detect", methods=["POST"])
def det():
    if 'video' not in request.files:
        return 'No file provided', 400
    file = request.files['video']
    if file.filename == '':
        return 'Invalid file', 400
    file.save(os.path.join(uploads_dir, secure_filename(file.filename)))
    url = os.path.join(uploads_dir, secure_filename(file.filename))
    print("detecting objects")
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default=ROOT / 'best.pt', help='model path(s)')
    parser.add_argument('--source', type=str, default=url, help='file/dir/URL/glob, 0 for webcam')
    parser.add_argument('--data', type=str, default=ROOT / 'data/coco128.yaml', help='(optional) dataset.yaml path')
    parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=[640], help='inference size h,w')
    parser.add_argument('--conf_thres', type=float, default=0.75, help='confidence threshold')
    parser.add_argument('--iou_thres', type=float, default=0.95, help='NMS IoU threshold')
    parser.add_argument('--max_det', type=int, default=10, help='maximum detections per image')
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
    parser.add_argument('--project', default=ROOT / 'static', help='save results to project/name')
    parser.add_argument('--name', default='video', help='save results to project/name')
    parser.add_argument('--exist_ok', action='store_false', help='existing project/name ok, do not increment')
    parser.add_argument('--line_thickness', default=3, type=int, help='bounding box thickness (pixels)')
    parser.add_argument('--hide_labels', default=False, action='store_true', help='hide labels')
    parser.add_argument('--hide_conf', default=False, action='store_true', help='hide confidences')
    parser.add_argument('--half', action='store_true', help='use FP16 half-precision inference')
    parser.add_argument('--dnn', action='store_true', help='use OpenCV DNN for ONNX inference')
    # global opt
    opt = parser.parse_args()
    opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1  # expand
    print_args(vars(opt))
    check_requirements(exclude=('tensorboard', 'thop'))  # 'pycocotools',
    with torch.no_grad():
        detect_yolo(opt=opt)

    obj = secure_filename(file.filename)
    print('Detection Completed and Saved in Run Folder')
    return obj
    # print('Detection Completed and Saved in Run Folder')
    # loc = os.path.join("runs/video", file.filename)
    # print(loc)
    # try:
    #    return url_for('static', filename='video/{}'.format(file.filename))
    # except Exception as e:
    #    return str(e)

# yolo detection here


def detect_yolo(opt):
    # for CSV Download Detections
    detections = []
    _log = ''
    # YOLOV5 Param Values
    source, weights, view_img, save_txt, imgsz, nosave, project, name, exist_ok, device, dnn, data, half, visualize, augment, conf_thres, iou_thres, classes, agnostic_nms, max_det, save_crop, line_thickness, hide_labels, hide_conf, update, save_conf = opt.source, opt.weights, opt.view_img, opt.save_txt, opt.imgsz, opt.nosave, opt.project, opt.name, opt.exist_ok, opt.device, opt.dnn, opt.data, opt.half, opt.visualize, opt.augment, opt.conf_thres, opt.iou_thres, opt.classes, opt.agnostic_nms, opt.max_det, opt.save_crop, opt.line_thickness, opt.hide_labels, opt.hide_conf, opt.update, opt.save_conf
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

    # Initialize
    set_logging()

    txt = 'runs'
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

    # Get names and colors
    names = model.module.names if hasattr(model, 'module') else model.names
    # colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]
    print(names)

    # Run inference
    with open('static/detected.csv', 'a') as csvfile:
        # 'bicycle', 'car', 'motorcycle', 'tricycle', 'van'
        header = ('Frame', 'bicycle', 'car', 'motorcycle', 'tricycle', 'van')
        csv_writer = DictWriter(csvfile, fieldnames=header, lineterminator='\n', delimiter=',')
        csv_writer.writeheader()

    t0 = time.time()

    model.warmup(imgsz=(1 if pt else bs, 3, *imgsz))  # warmup
    seen, windows, dt = 0, [], [0.0, 0.0, 0.0]
    _name = ''
    _n_det = 0
    
    for path, im, im0s, vid_cap, s in dataset:
        t1 = time_sync()
        im = torch.from_numpy(im).to(device)
        im = im.half() if model.fp16 else im.float()  # uint8 to fp16/32
        im /= 255  # 0 - 255 to 0.0 - 1.0
        if len(im.shape) == 3:
            im = im[None]  # expand for batch dim
        t2 = time_sync()
        dt[0] += t2 - t1
        _log = s[s.find('(') + 1:s.find(')')]
        _log = _log.split("/")
        # Inference
        visualize = increment_path(save_dir / Path(path).stem, mkdir=True) if visualize else False
        pred = model(im, augment=augment, visualize=visualize)
        t3 = time_sync()
        dt[1] += t3 - t2

        # NMS
        pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)
        dt[2] += time_sync() - t3
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
                    _name = names[int(c)]
                    _n_det = int(n)

                # Write results
                tl1 = cnf1 = tl2 = cnf2 = tl3 = cnf3 = tl4 = cnf4 = tl5 = cnf5 = ''
                for *xyxy, conf, cls in reversed(det):
                    if save_txt:  # save_txt:  # Write to file
                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                        line = (cls, *xywh, conf) if save_conf else (cls, *xywh)  # label format
                        # with open(f'{txt_path}.txt', 'a') as f:
                        #    f.write(('%g ' * len(line)).rstrip() % line + '\n')

                    if save_img or save_crop or view_img:  # Add bbox to image
                        c = int(cls)  # integer class
                        label = None if hide_labels else (names[c] if hide_conf else f'{names[c]} {conf:.2f}')
                        annotator.box_label(xyxy, label, color=colors(c, True))
                    if save_crop:
                        save_one_box(xyxy, imc, file=save_dir / 'crops' / names[c] / f'{p.stem}.jpg', BGR=True)

                    # save csv report
                    if True:
                        icls = int(cls)
                        if icls == 0 or icls == 1:
                            tl1 = int(cls)
                            cnf1 = '%.2f' % (conf)
                        if icls == 2 or icls == 3:
                            tl2 = int(cls)
                            cnf2 = '%.2f' % (conf)
                        if icls == 4 or icls == 5 or icls == 6:
                            tl3 = int(cls)
                            cnf3 = '%.2f' % (conf)
                        if icls == 7 or icls == 8:
                            tl4 = int(cls)
                            cnf4 = '%.2f' % (conf)
                        if icls == 9 or icls == 10:
                            tl5 = int(cls)
                            cnf5 = '%.2f' % (conf)

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
                        vid_writer[i] = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'H264'), fps, (w, h))
                    vid_writer[i].write(im0)

        # Print time (inference-only)
        # 'bicycle', 'car', 'motorcycle', 'tricycle', 'van'
        if _name == 'bicycle':
            with open('static/detected.csv', 'a') as csvfile:
                csv_writer = DictWriter(csvfile, fieldnames=header, lineterminator='\n', delimiter=',')
                csv_writer.writerow({'Frame': frame, 'bicycle': int(n), 'car': 0,
                                    'motorcycle': 0, 'tricycle': 0, 'van': 0})
        if _name == 'car':
            with open('static/detected.csv', 'a') as csvfile:
                csv_writer = DictWriter(csvfile, fieldnames=header, lineterminator='\n', delimiter=',')
                csv_writer.writerow({'Frame': frame, 'bicycle': 0, 'car': int(n),
                                                'motorcycle': 0, 'tricycle': 0, 'van': 0})
        if _name == 'motorcycle':
            with open('static/detected.csv', 'a') as csvfile:
                csv_writer = DictWriter(csvfile, fieldnames=header, lineterminator='\n', delimiter=',')
                csv_writer.writerow({'Frame': frame, 'bicycle': 0, 'car': 0, 'motorcycle': int(n), 'tricycle': 0, 'van': 0 })
        if _name == 'tricycle':
            with open('static/detected.csv', 'a') as csvfile:
                csv_writer = DictWriter(csvfile, fieldnames=header, lineterminator='\n', delimiter=',')
                csv_writer.writerow({'Frame': frame, 'bicycle': 0, 'car': 0, 'motorcycle': 0, 'tricycle': int(n), 'van': 0})
        if _name == 'van':
            with open('static/detected.csv', 'a') as csvfile:
                csv_writer = DictWriter(csvfile, fieldnames=header, lineterminator='\n', delimiter=',')
                csv_writer.writerow({'Frame': frame, 'bicycle': 0, 'car': 0, 'motorcycle': 0, 'tricycle': 0, 'van': int(n)})

        fr = int(_log[0])
        to = int(_log[1])
        perc = round((fr/to) * 100)
        LOGGER.info(f'{s}Done. ({t3 - t2:.3f}s)')
        LOGGER.info('{}%'.format(perc))

    # Print results
    t = tuple(x / seen * 1E3 for x in dt)  # speeds per image
    LOGGER.info(f'Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS per image at shape {(1, 3, *imgsz)}' % t)
    if save_txt or save_img:
        s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''
        LOGGER.info(f"Results saved to {colorstr('bold', save_dir)}{s}")
    if update:
        strip_optimizer(weights)  # update model (to fix SourceChangeWarning)


@app.route('/return-files', methods=['GET'])
def return_file():
    obj = request.args.get('obj')
    loc = os.path.join("static", obj)
    print(loc)
    try:
        return send_file(os.path.join("static/video/", obj), attachment_filename=obj)
    except Exception as e:
        return str(e)


@app.route('/uploads/<filename>', methods=['GET', 'POST'])
def download(filename):
    return send_from_directory(os.path.join("static/video/", filename), filename, as_attachment=True)


@app.route('/display/<filename>')
def display_video(filename):
    print('display_video filename: ' + filename)
    return redirect(url_for('static/video/{}'.format(filename), code=200))


if __name__ == "__main__":
    app.run(host="0.0.0.0", port="8080", debug=True)
