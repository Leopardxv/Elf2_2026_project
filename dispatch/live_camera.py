#!/usr/bin/env python3
"""ELF2 Live Camera — ESP32 WiFi + YOLOv5s NPU → Web Browser."""
import cv2, time, numpy as np, threading, requests
from rknnlite.api import RKNNLite
from http.server import HTTPServer, BaseHTTPRequestHandler

CAM = "http://10.127.96.16:81/stream"
MODEL = "/home/elf/Projects/models/yolov5s-640-640.rknn"
PORT = 8080

COCO = ["person","bicycle","car","motorcycle","airplane","bus","train","truck",
    "boat","traffic light","fire hydrant","stop sign","parking meter","bench",
    "bird","cat","dog","horse","sheep","cow","elephant","bear","zebra",
    "giraffe","backpack","umbrella","handbag","tie","suitcase","frisbee",
    "skis","snowboard","sports ball","kite","baseball bat","baseball glove",
    "skateboard","surfboard","tennis racket","bottle","wine glass","cup",
    "fork","knife","spoon","bowl","banana","apple","sandwich","orange",
    "broccoli","carrot","hot dog","pizza","donut","cake","chair","couch",
    "potted plant","bed","dining table","toilet","tv","laptop","mouse",
    "remote","keyboard","cell phone","microwave","oven","toaster","sink",
    "refrigerator","book","clock","vase","scissors","teddy bear",
    "hair drier","toothbrush"]
COL = [(0,255,0),(255,0,0),(0,0,255),(255,255,0)]

latest_jpeg = b""
lock = threading.Lock()
running = True

def sig(x): return 1/(1+np.exp(-x))

def detect(rknn, frame):
    h0,w0 = frame.shape[:2]
    img = cv2.resize(frame, (640,640))
    inp = np.transpose(img, (2,0,1)).astype(np.float32)[None,:]
    outs = rknn.inference(inputs=[inp])
    boxes, scores, cls = [], [], []
    for i,o in enumerate(outs):
        o = o[0]; na,h,w = o.shape[0],o.shape[2],o.shape[3]
        nc = o.shape[1]-5; stride = [8,16,32][i]
        o = o.reshape(na,5+nc,h*w).transpose(0,2,1).reshape(-1,5+nc)
        gy,gx = np.mgrid[0:h,0:w]
        gy = np.tile(gy.reshape(-1),3); gx = np.tile(gx.reshape(-1),3)
        o[:,0] = sig(o[:,0])*2-0.5+gx; o[:,1] = sig(o[:,1])*2-0.5+gy
        o[:,2] = (sig(o[:,2])*2)**2; o[:,3] = (sig(o[:,3])*2)**2
        o[:,0] *= stride; o[:,1] *= stride
        o[:,2] *= stride*4; o[:,3] *= stride*4
        obj = sig(o[:,4]); cs = sig(o[:,5:])
        sc = obj*cs.max(axis=1); cl = cs.argmax(axis=1)
        mask = sc > 0.35
        if not mask.any(): continue
        b = np.zeros((mask.sum(),4))
        b[:,0] = o[mask,0]-o[mask,2]/2; b[:,1] = o[mask,1]-o[mask,3]/2
        b[:,2] = o[mask,0]+o[mask,2]/2; b[:,3] = o[mask,1]+o[mask,3]/2
        boxes.append(b); scores.append(sc[mask]); cls.append(cl[mask])
    if not boxes: return []
    boxes = np.concatenate(boxes); scores = np.concatenate(scores); cls = np.concatenate(cls)
    x1,y1,x2,y2 = boxes[:,0],boxes[:,1],boxes[:,2],boxes[:,3]
    areas = (x2-x1)*(y2-y1); order = scores.argsort()[::-1]; keep = []
    while order.size > 0:
        i = order[0]; keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]]); yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]]); yy2 = np.minimum(y2[i], y2[order[1:]])
        w = np.maximum(0,xx2-xx1); h = np.maximum(0,yy2-yy1)
        iou = w*h/(areas[i]+areas[order[1:]]-w*h)
        order = order[np.where(iou<=0.45)[0]+1]
    res = []
    for i in keep:
        x1 = int(boxes[i,0]*w0/640); y1 = int(boxes[i,1]*h0/640)
        x2 = int(boxes[i,2]*w0/640); y2 = int(boxes[i,3]*h0/640)
        res.append((COCO[int(cls[i])], float(scores[i]), (x1,y1,x2,y2)))
    return res

def cap_loop():
    global latest_jpeg, running
    rknn = RKNNLite(); rknn.load_rknn(MODEL); rknn.init_runtime(core_mask=RKNNLite.NPU_CORE_0)
    fps_times = []
    while running:
        try:
            r = requests.get(CAM, stream=True, timeout=5)
            buf = b""
            soi_marker = bytes([0xff, 0xd8])
            eoi_marker = bytes([0xff, 0xd9])
            for chunk in r.iter_content(chunk_size=4096):
                if not running: break
                buf += chunk
                while True:
                    soi = buf.find(soi_marker)
                    if soi < 0: break
                    eoi = buf.find(eoi_marker, soi)
                    if eoi < 0: break
                    jpeg = buf[soi:eoi+2]; buf = buf[eoi+2:]
                    frame = cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR)
                    if frame is None: continue
                    res = detect(rknn, frame)
                    fps_times.append(time.time())
                    if len(fps_times)>30: fps_times.pop(0)
                    fps = len(fps_times)/(fps_times[-1]-fps_times[0]) if len(fps_times)>1 else 0
                    for lb,cf,(xa,ya,xb,yb) in res:
                        c = COL[hash(lb)%len(COL)]
                        cv2.rectangle(frame, (xa,ya), (xb,yb), c, 2)
                        cv2.putText(frame, f"{lb} {cf:.0%}", (xa,ya-5),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, c, 2)
                    cv2.putText(frame, f"FPS:{fps:.0f} Objs:{len(res)}", (5,18),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)
                    _, out = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 75])
                    with lock: latest_jpeg = out.tobytes()
        except Exception as e:
            time.sleep(2)
    rknn.release()

class H(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path.startswith('/stream'):
            self.send_response(200)
            self.send_header('Content-type', 'image/jpeg')
            self.end_headers()
            with lock:
                if latest_jpeg: self.wfile.write(latest_jpeg)
        else:
            self.send_response(200)
            self.send_header('Content-type', 'text/html; charset=utf-8')
            self.end_headers()
            html = """<!DOCTYPE html><html><head>
<title>ELF2 Live Camera</title><meta charset=utf-8>
<style>body{margin:0;background:#000;text-align:center}
h2{color:#0f0;font-family:sans-serif;margin:8px 0}
img{max-width:100%;border:2px solid #0f0}</style></head>
<body><h2>ELF2 Live Camera + YOLOv5s NPU</h2>
<img id=c src=/stream>
<script>setInterval(function(){document.getElementById('c').src='/stream?'+Date.now()},150)</script>
</body></html>"""
            self.wfile.write(html.encode())

def main():
    global running
    print(f"Server on http://localhost:{PORT}")
    threading.Thread(target=cap_loop, daemon=True).start()
    time.sleep(4)
    HTTPServer(('0.0.0.0', PORT), H).serve_forever()

if __name__ == "__main__":
    main()
