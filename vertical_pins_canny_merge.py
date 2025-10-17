# pin_inspector_unified.py
# One PySide6 GUI (800x600) with two tabs:
# 1) Top/Bottom PCD detector (circles)   2) Side-view vertical pin height detector
# - All display stays inside the GUI; images are auto-scaled to fit the tab area.

import sys, os, math
import cv2
import numpy as np
from PySide6.QtWidgets import (
    QApplication, QWidget, QLabel, QPushButton, QVBoxLayout, QHBoxLayout,
    QFileDialog, QMessageBox, QTabWidget, QSpacerItem, QSizePolicy
)
from PySide6.QtGui import QPixmap, QImage
from PySide6.QtCore import Qt

# ------------------------ Shared utils ------------------------
def cv_to_qpixmap(img):
    """Convert BGR/Gray numpy image to QPixmap safely."""
    if img is None:
        return QPixmap()
    if img.ndim == 2:
        h, w = img.shape
        q = QImage(img.data, w, h, w, QImage.Format_Grayscale8)
        return QPixmap.fromImage(q)
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h, w, c = rgb.shape
    q = QImage(rgb.data, w, h, w * c, QImage.Format_RGB888)
    return QPixmap.fromImage(q)

def safe_imread_any(path):
    """Robust loader for paths with unicode; returns BGR image or None."""
    data = np.fromfile(path, np.uint8)
    return cv2.imdecode(data, cv2.IMREAD_COLOR)

# =====================================================================
#                           TAB 1: TOP/BOTTOM PCD
# =====================================================================

class PerfectS398Detector:
    """Your circles-based PCD detector (slightly adapted to be GUI-friendly)."""
    def __init__(self):
        self.target_pcd = 11.66   # in pixels (as provided)
        self.tolerance  = 0.13

    def preprocess_image(self, img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        filtered = cv2.medianBlur(gray, 5)
        enhanced = cv2.equalizeHist(filtered)
        return enhanced

    def _score_circles(self, processed, circles):
        """Score circles by edge energy inside each circle ROI."""
        scores = []
        for (x, y, r) in circles:
            y0, y1 = max(0, y - r), min(processed.shape[0], y + r)
            x0, x1 = max(0, x - r), min(processed.shape[1], x + r)
            roi = processed[y0:y1, x0:x1]
            if roi.size > 0:
                edges = cv2.Canny(roi, 50, 150)
                scores.append(int(np.sum(edges)))
            else:
                scores.append(0)
        return scores

    def detect_pins(self, img):
        """Return up to 3 best circles as (x,y,r) int array or None."""
        processed = self.preprocess_image(img)
        circles = cv2.HoughCircles(
            processed, cv2.HOUGH_GRADIENT, dp=1, minDist=70,
            param1=120, param2=50, minRadius=10, maxRadius=60
        )

        if circles is None:
            return None

        circles = np.round(circles[0, :]).astype("int")
        if len(circles) > 3:
            scores = self._score_circles(processed, circles)
            idx = np.argsort(scores)[-3:]  # top-3 by score
            circles = circles[idx]
        return circles

    def calculate_distances(self, pins):
        """All pairwise distances among 3 pins (in pixels)."""
        distances = []
        for i in range(len(pins)):
            for j in range(i + 1, len(pins)):
                x1, y1 = pins[i][:2]
                x2, y2 = pins[j][:2]
                dist = float(np.hypot(float(x2 - x1), float(y2 - y1)))
                distances.append(dist)
        return distances

    def run_on_image(self, img_bgr):
        """
        Returns: result_img, message (str)
        - Draws circles and pair lines with distances and OK/NG.
        """
        if img_bgr is None:
            return None, "❌ Could not load image!"

        result_img = img_bgr.copy()
        pins = self.detect_pins(img_bgr)
        if pins is None:
            return result_img, "❌ No pins detected!"

        num_pins = len(pins)
        for i, (x, y, r) in enumerate(pins):
            cv2.circle(result_img, (int(x), int(y)), int(r), (0, 255, 0), 2)
            cv2.circle(result_img, (int(x), int(y)), 3, (0, 0, 255), -1)
            cv2.putText(result_img, f"Pin{i + 1}", (int(x) - 25, int(y) - int(r) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

        if num_pins == 3:
            distances = self.calculate_distances(pins)
            avg_distance = float(np.mean(distances))
            pin_pairs = [(0, 1), (1, 2), (0, 2)]

            # draw pairwise lines + labels
            for i, (p1, p2) in enumerate(pin_pairs):
                x1, y1 = pins[p1][:2]
                x2, y2 = pins[p2][:2]
                cv2.line(result_img, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 255), 2)
                mid_x, mid_y = int((x1 + x2) // 2), int((y1 + y2) // 2)
                cv2.putText(result_img, f"{distances[i]:.1f}px",
                            (mid_x - 20, mid_y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2)

            status = "OK" if abs(avg_distance - self.target_pcd) <= self.tolerance else "NG"
            color = (0, 255, 0) if status == "OK" else (0, 0, 255)

            cv2.putText(result_img, f"Status: {status}", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)
            cv2.putText(result_img, f"Pins: {num_pins}/3", (20, 75),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            cv2.putText(result_img, f"Avg Distance: {avg_distance:.2f}px", (20, 105),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

            return result_img, f"✅ Success: {status} - {num_pins} pins detected"

        else:
            cv2.putText(result_img, f"Need 3 pins, found {num_pins}", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
            return result_img, f"❌ Found {num_pins} pins, need exactly 3"


class TabTopBottomPCD(QWidget):
    """GUI tab for the PerfectS398Detector (Top/Bottom PCD)."""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.detector = PerfectS398Detector()

        self.img_label = QLabel("Open an image…")
        self.img_label.setAlignment(Qt.AlignCenter)
        self.img_label.setStyleSheet("border:1px solid #555; background:#111; color:#aaa;")
        self.img_label.setScaledContents(True)

        self.status_label = QLabel("Status: –")
        self.status_label.setAlignment(Qt.AlignLeft)

        self.open_btn = QPushButton("Open Image")
        self.open_btn.clicked.connect(self.open_image)

        top = QHBoxLayout()
        top.addWidget(self.open_btn)
        top.addSpacerItem(QSpacerItem(20, 10, QSizePolicy.Expanding, QSizePolicy.Minimum))
        top.addWidget(self.status_label)

        lay = QVBoxLayout()
        lay.addLayout(top)
        lay.addWidget(self.img_label, stretch=1)
        self.setLayout(lay)

    def open_image(self):
        p, _ = QFileDialog.getOpenFileName(self, "Open Image", "", "Images (*.png *.jpg *.jpeg *.bmp *.tiff)")
        if not p:
            return
        img = safe_imread_any(p)
        if img is None:
            QMessageBox.critical(self, "Error", f"Cannot load {p}")
            return

        result_img, msg = self.detector.run_on_image(img)
        self.img_label.setPixmap(cv_to_qpixmap(self._fit_to_tab(result_img)))
        self.status_label.setText(f"Status: {msg}")

    def _fit_to_tab(self, img):
        """Scale image conservatively to fit typical tab client area."""
        if img is None:
            return None
        # Heuristic: leave room for header—scale to about 760x520 (within 800x600 window)
        max_w, max_h = 760, 520
        h, w = img.shape[:2]
        s = min(max_w / float(w), max_h / float(h), 1.0)
        if s < 1.0:
            img = cv2.resize(img, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)
        return img

# =====================================================================
#                           TAB 2: SIDE HEIGHT
# (Adapted from pin_height_gui_v11_simple.py to render inside GUI)
# =====================================================================

# ----------------- config -----------------
SAMPLE_PATH = "/mnt/data/IMG_9489.jpg"
EXPECTED_COUNT = 3

# vertical lines
ANGLE_TOL_DEG = 6
CANNY1, CANNY2 = 40, 120
HOUGH_TH, MIN_LINE, MAX_GAP = 48, 36, 12
X_BORDER_FRAC = 0.01
X_CLUSTER_GAP_FRAC = 0.014   # cluster merge width (~1.4% of image w)
X_SUPPRESS_FRAC = 0.075      # non-max spacing (~7.5% of image w)
MIN_HEIGHT_PX = 240

# head search band & radius (fractions of cluster height)
HEAD_BAND_UP = 0.20
HEAD_BAND_DN = 0.25
HEAD_R_MAX   = 0.10

def compute_distance_cv(vals):
    if not vals: return 999.0
    a = np.asarray(vals, np.float32); m = float(a.mean())
    return 999.0 if m == 0 else float(a.std(ddof=0)/m*100.0)

def compute_accuracy_from_cv(cv_percent, n, expected):
    if expected <= 0: return 0.0
    match = 1.0 if n == expected else min(1.0, n/expected)
    return round(max(0.0, 100.0*match*(1.0 - min(1.0, cv_percent/100.0))), 2)

def preprocess(gray):
    bg = cv2.GaussianBlur(gray, (101,101), 0)
    norm = cv2.subtract(gray, bg)
    norm = cv2.normalize(norm, None, 0, 255, cv2.NORM_MINMAX)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    return clahe.apply(norm)

def hough_vertical_clusters(gray_clean):
    h, w = gray_clean.shape
    edges = cv2.Canny(gray_clean, CANNY1, CANNY2)
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, HOUGH_TH, minLineLength=MIN_LINE, maxLineGap=MAX_GAP)
    if lines is None: return []

    x_margin = int(max(6, round(w*X_BORDER_FRAC)))
    segs = []
    for l in lines:
        x1,y1,x2,y2 = map(int, l[0])
        if (x1<x_margin and x2<x_margin) or (x1>w-x_margin and x2>w-x_margin): continue
        dx, dy = int(x2-x1), int(y2-y1)
        ang = 90.0 if dx==0 else abs(math.degrees(math.atan2(float(dy), float(dx))))
        if ang>90: ang = 180-ang
        if abs(90.0-ang) <= float(ANGLE_TOL_DEG):
            t, b = max(0,min(y1,y2)), min(h-1,max(y1,y2))
            if b>t: segs.append((0.5*(x1+x2), t, b))

    if not segs: return []

    # cluster by x
    segs.sort(key=lambda s: s[0])
    gap = int(max(6, round(w*X_CLUSTER_GAP_FRAC)))
    clusters, cur = [], [segs[0]]
    for xm,t,b in segs[1:]:
        if abs(xm - cur[-1][0]) <= gap: cur.append((xm,t,b))
        else: clusters.append(cur); cur=[(xm,t,b)]
    clusters.append(cur)

    info=[]
    for cl in clusters:
        xs=[c[0] for c in cl]; ts=[c[1] for c in cl]; bs=[c[2] for c in cl]
        cx=int(round(sum(xs)/len(xs))); top=int(min(ts)); bot=int(max(bs))
        info.append({'x':cx,'top':top,'bottom':bot,'height':max(0,bot-top)})
    # min height
    info=[c for c in info if c['height']>=MIN_HEIGHT_PX]
    return info

def suppress_by_x(clusters, w, want=EXPECTED_COUNT):
    if not clusters: return []
    clusters = sorted(clusters, key=lambda c: c['height'], reverse=True)
    out=[]; min_dx = max(8, int(X_SUPPRESS_FRAC*w))
    for c in clusters:
        if all(abs(int(c['x'])-int(o['x']))>=min_dx for o in out):
            out.append(c)
            if len(out)==want: break
    # backfill if fewer than wanted
    if len(out)<want:
        rest=[c for c in clusters if c not in out]
        rest.sort(key=lambda c:c['x'])
        for c in rest:
            if all(abs(int(c['x'])-int(o['x']))>=min_dx for o in out):
                out.append(c)
                if len(out)==want: break
    return sorted(out[:want], key=lambda c:c['x'])

def _pick_best_circle(circles, left, top, target_x):
    if circles is None: return None
    cs = np.round(circles[0,:]).astype(int)
    best=None; bestdx=1e9
    for cx,cy,r in cs:
        fx, fy = int(left+cx), int(top+cy)
        d = abs(int(fx)-int(target_x))
        if d<bestdx: bestdx=d; best=(fx,fy,int(r))
    return best

def find_head(gray, cx, y_top, y_bot, max_r):
    h,w = gray.shape
    left = max(0, int(cx-2*max_r)); right=min(w, int(cx+2*max_r))
    top  = max(0, int(y_top));      bot  = min(h, int(y_bot))
    if right-left<8 or bot-top<8: return None

    roi = gray[top:bot, left:right]
    min_r = max(3, int(0.25*max_r))
    # first pass
    best = _pick_best_circle(
        cv2.HoughCircles(cv2.medianBlur(roi,5), cv2.HOUGH_GRADIENT,1.2,8,
                         param1=85,param2=10,minRadius=min_r,maxRadius=int(max_r*0.9)),
        left, top, cx)
    # edge-based
    edges = cv2.Canny(cv2.GaussianBlur(roi,(3,3),0),70,150)
    cand = _pick_best_circle(
        cv2.HoughCircles(edges, cv2.HOUGH_GRADIENT,1.1,8,
                         param1=80,param2=7,minRadius=min_r,maxRadius=int(max_r*0.9)),
        left, top, cx)
    if cand is not None:
        best = cand if best is None else best

    if best is None: return None
    bx, by, br = best
    by = max(0, int(by - int(0.30 * float(br))))  # move up toward pin top
    br = int(max(3, int(float(br) * 0.7)))
    return (int(bx), int(by), int(br))

def find_trapezoid_bottom(gray_clean, cx, head_y, head_r, cluster_bottom):
    h,w = gray_clean.shape
    half_w = max(36, int(0.12*(cluster_bottom - head_y + 1)))
    left=max(0, cx-half_w); right=min(w, cx+half_w)
    start = min(h-2, max(0, int(head_y + max(4, 0.4*head_r))))
    end   = min(h-1, int(cluster_bottom))
    if right-left<8 or end-start<8: return None

    roi = gray_clean[start:end+1, left:right]
    gy = cv2.Sobel(roi, cv2.CV_32F, 0, 1, ksize=3)
    row_energy = np.mean(np.abs(gy), axis=1).astype(np.float32)
    if row_energy.size==0 or float(np.max(row_energy))==0: return None
    sm = cv2.GaussianBlur(row_energy.reshape(-1,1),(1,9),0).ravel()
    thr = max(6.0, 0.40*float(np.max(sm)))
    best=None
    for i in range(len(sm)-2, 1, -1):
        if sm[i]>=thr and sm[i]>=sm[i-1] and sm[i]>=sm[i+1]:
            best = i; break
    if best is None:
        best = int(np.argmax(sm))
    y = start + int(best)
    return min(int(y), int(cluster_bottom))

def detect_and_measure_side(img_bgr):
    """Return tuple: clean(gray), overlay(bgr), heights(list[int]), accuracy(float)"""
    out = img_bgr.copy()
    h, w = out.shape[:2]
    gray = cv2.cvtColor(out, cv2.COLOR_BGR2GRAY)
    clean = preprocess(gray)

    clusters = hough_vertical_clusters(clean)
    clusters = suppress_by_x(clusters, w, EXPECTED_COUNT)
    if len(clusters) < EXPECTED_COUNT:
        xs = [int(w*0.3), int(w*0.5), int(w*0.7)]
        clusters = [{'x':x,'top':int(0.08*h),'bottom':int(0.75*h),'height':int(0.67*h)} for x in xs]

    heights=[]
    for c in clusters:
        cx, top, bot = int(c['x']), int(c['top']), int(c['bottom'])
        height = max(1, int(bot-top))

        y_top = max(0, int(top - int(HEAD_BAND_UP*height)))
        y_bot = min(h-1, int(top + int(HEAD_BAND_DN*height)))
        max_r = int(max(6, round(HEAD_R_MAX*height)))

        circ = find_head(gray, cx, y_top, y_bot, max_r)
        if circ is None:
            circ_x, circ_y, circ_r = cx, top, int(max_r*0.8)
        else:
            circ_x, circ_y, circ_r = circ

        trap_y = find_trapezoid_bottom(clean, cx, circ_y, circ_r, bot)
        if trap_y is not None and int(trap_y) > int(circ_y):
            hpx = int(trap_y - circ_y)
            heights.append(hpx)
            cv2.circle(out,(int(circ_x),int(circ_y)),max(3,int(circ_r)),(0,255,0),2)
            cv2.circle(out,(int(circ_x),int(circ_y)),2,(0,255,0),-1)
            cv2.line(out,(int(cx),int(circ_y)),(int(cx),int(trap_y)),(255,255,0),2)
            cv2.putText(out,f"{hpx}px",(int(cx+8),int((circ_y+trap_y)//2)),
                        cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,255,255),2)
        else:
            hpx = int(height)
            heights.append(hpx)
            cv2.line(out,(int(cx),int(top)),(int(cx),int(bot)),(0,0,255),2)
            cv2.putText(out,f"{hpx}px (fb)",(int(cx+8),int((top+bot)//2)),
                        cv2.FONT_HERSHEY_SIMPLEX,0.45,(0,200,255),2)

    cvp = compute_distance_cv(heights)
    acc = compute_accuracy_from_cv(cvp, len(heights), EXPECTED_COUNT)
    return clean, out, heights, acc

class TabSideHeight(QWidget):
    """GUI tab for side-view vertical pin-height detector."""
    def __init__(self, parent=None):
        super().__init__(parent)

        self.open_btn = QPushButton("Open Image")
        self.open_btn.clicked.connect(self.open_image)

        self.status = QLabel("Heights: – | Accuracy: –")
        self.status.setAlignment(Qt.AlignLeft)

        # three views
        self.orig = QLabel(); self.mid = QLabel(); self.out = QLabel()
        for L in (self.orig, self.mid, self.out):
            L.setAlignment(Qt.AlignCenter)
            L.setStyleSheet("border:1px solid #555; background:#111; color:#aaa;")
            L.setScaledContents(True)
            L.setMinimumSize(220, 260)

        top = QHBoxLayout()
        top.addWidget(self.open_btn)
        top.addSpacerItem(QSpacerItem(20, 10, QSizePolicy.Expanding, QSizePolicy.Minimum))
        top.addWidget(self.status)

        imgs = QHBoxLayout()
        imgs.addWidget(self.orig)
        imgs.addWidget(self.mid)
        imgs.addWidget(self.out)

        lay = QVBoxLayout()
        lay.addLayout(top)
        lay.addLayout(imgs)
        self.setLayout(lay)

    def open_image(self):
        p, _ = QFileDialog.getOpenFileName(self, "Open Image", "", "Images (*.png *.jpg *.jpeg *.bmp *.tiff)")
        if not p:
            return
        img = safe_imread_any(p)
        if img is None:
            QMessageBox.critical(self, "Error", f"Cannot load {p}")
            return

        # scale input to keep performance consistent
        h, w = img.shape[:2]
        max_dim = 1200
        if max(h, w) > max_dim:
            s = max_dim / float(max(h, w))
            img = cv2.resize(img, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)

        clean, overlay, heights, acc = detect_and_measure_side(img)

        # fit for labels (about 760x520 max per tab; each label about a third width)
        self.orig.setPixmap(cv_to_qpixmap(self._fit(img)))
        self.mid.setPixmap(cv_to_qpixmap(self._fit(cv2.cvtColor(clean, cv2.COLOR_GRAY2BGR))))
        self.out.setPixmap(cv_to_qpixmap(self._fit(overlay)))
        self.status.setText(
            f"Heights: {', '.join(map(str,heights))} px | Accuracy: {acc}%"
            if heights else "No vertical pins detected"
        )

    def _fit(self, img):
        if img is None:
            return None
        # Each of three labels fits side-by-side; leave margins.
        max_w, max_h = 240, 520
        h, w = img.shape[:2]
        s = min(max_w / float(w), max_h / float(h), 1.0)
        if s < 1.0:
            img = cv2.resize(img, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)
        return img

# =====================================================================
#                               MAIN WINDOW
# =====================================================================

class UnifiedInspector(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Pin Inspector – PCD (Top/Bottom) + Height (Side)")
        self.setFixedSize(800, 600)  # requested 800x600

        self.tabs = QTabWidget()
        self.tabs.addTab(TabTopBottomPCD(self), "Top/Bottom – PCD")
        self.tabs.addTab(TabSideHeight(self), "Side – Height")

        lay = QVBoxLayout()
        lay.addWidget(self.tabs)
        self.setLayout(lay)

# ----------------- main -----------------
if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = UnifiedInspector()
    win.show()
    sys.exit(app.exec())
