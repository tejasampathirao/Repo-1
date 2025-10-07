# s398_pindetector_gui.py  (updated: improved side-view vertical height & distances)
import sys
import math
import cv2
import numpy as np
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QLabel, QPushButton, QVBoxLayout, QHBoxLayout,
    QFileDialog, QComboBox, QCheckBox, QSpinBox, QDoubleSpinBox, QMessageBox, QGroupBox
)
from PySide6.QtGui import QPixmap, QImage
from PySide6.QtCore import Qt

# ---------------- Utility functions ----------------

def cv2_to_qpixmap(bgr):
    """Convert BGR OpenCV image to QPixmap (RGB888)"""
    if bgr is None:
        return QPixmap()
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    h, w, ch = rgb.shape
    bytes_per_line = ch * w
    qimg = QImage(rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)
    return QPixmap.fromImage(qimg)

def select_best_group_1d(sorted_coords, expected):
    """Select a contiguous group of expected coords with minimal span."""
    coords = sorted_coords
    n = len(coords)
    if n <= expected:
        return coords
    best_span = float('inf')
    best_group = coords[:expected]
    for i in range(0, n - expected + 1):
        window = coords[i:i + expected]
        span = window[-1] - window[0]
        if span < best_span:
            best_span = span
            best_group = window
    return best_group

# ---------------- Detection helpers ----------------

def detect_circles(frame, blur_type='median', blur_k=5,
                   dp=1.2, minDist=40, param1=120, param2=40, minRadius=8, maxRadius=120, draw=True):
    """Detect circles using HoughCircles. Returns (image_with_drawing, centers_list)"""
    out = frame.copy()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    if blur_type == 'median':
        gray = cv2.medianBlur(gray, blur_k)
    elif blur_type == 'gaussian':
        k = blur_k if blur_k % 2 == 1 else blur_k + 1
        gray = cv2.GaussianBlur(gray, (k, k), 1.5)

    circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT,
                               dp=dp, minDist=minDist,
                               param1=param1, param2=param2,
                               minRadius=minRadius, maxRadius=maxRadius)
    centers = []
    if circles is not None:
        circles = np.uint16(np.around(circles[0]))
        for (x, y, r) in circles:
            centers.append((int(x), int(y), int(r)))
            if draw:
                cv2.circle(out, (int(x), int(y)), int(r), (0, 255, 0), 2)
                cv2.circle(out, (int(x), int(y)), 2, (0, 0, 255), 3)
    return out, centers

def detect_vertical_lines(frame, blur_type='gaussian', blur_k=5,
                          canny_th1=50, canny_th2=150,
                          hough_threshold=50, minLineLength=60, maxLineGap=15,
                          angle_tol_deg=12, draw=True):
    """
    Detect roughly vertical line segments using HoughLinesP and return clustered vertical pins.
    Returns:
        out: image with drawn short vertical segments (top->bottom of each cluster)
        clusters: list of dicts [{ 'x': int(center_x), 'top': int, 'bottom': int, 'height': int }, ...] sorted by x
        raw_lines: original Hough lines (for debug)
    """
    out = frame.copy()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    if blur_type == 'median':
        gray = cv2.medianBlur(gray, blur_k)
    elif blur_type == 'gaussian':
        k = blur_k if blur_k % 2 == 1 else blur_k + 1
        gray = cv2.GaussianBlur(gray, (k, k), 1.5)

    edges = cv2.Canny(gray, canny_th1, canny_th2)
    raw_lines = cv2.HoughLinesP(edges, rho=1, theta=np.pi / 180, threshold=hough_threshold,
                                minLineLength=minLineLength, maxLineGap=maxLineGap)
    if raw_lines is None:
        return out, [], []

    # collect candidate vertical segments: (xm, top_y, bottom_y)
    vert_segs = []
    h_img, w_img = frame.shape[:2]
    for l in raw_lines:
        x1, y1, x2, y2 = l[0]
        dx = x2 - x1
        dy = y2 - y1
        if dx == 0:
            angle = 90.0
        else:
            angle = abs(math.degrees(math.atan2(dy, dx)))
            if angle > 90:
                angle = 180 - angle
        # vertical if angle close to 90
        if abs(90 - angle) <= angle_tol_deg:
            top = min(y1, y2)
            bottom = max(y1, y2)
            xm = (x1 + x2) / 2.0
            # normalize within image bounds
            top = max(0, int(top))
            bottom = min(h_img - 1, int(bottom))
            vert_segs.append((xm, top, bottom))

    if not vert_segs:
        return out, [], raw_lines

    # sort by xm
    vert_segs.sort(key=lambda v: v[0])
    # cluster by proximity in x
    min_gap = max(6, int(w_img * 0.01))
    clusters = []
    curr = [vert_segs[0]]
    for xm, top, bottom in vert_segs[1:]:
        if abs(xm - curr[-1][0]) <= min_gap:
            curr.append((xm, top, bottom))
        else:
            clusters.append(curr)
            curr = [(xm, top, bottom)]
    clusters.append(curr)

    # build cluster summary: center x, top=min(top), bottom=max(bottom), height
    cluster_infos = []
    for cl in clusters:
        xs = [c[0] for c in cl]
        tops = [c[1] for c in cl]
        bottoms = [c[2] for c in cl]
        center_x = int(round(sum(xs) / len(xs)))
        top_y = int(min(tops))
        bottom_y = int(max(bottoms))
        height = max(0, bottom_y - top_y)
        cluster_infos.append({'x': center_x, 'top': top_y, 'bottom': bottom_y, 'height': height})

    # sort cluster_infos by x ascending
    cluster_infos.sort(key=lambda c: c['x'])

    # draw short vertical segments for each cluster (exact pin height)
    if draw:
        for c in cluster_infos:
            cx = c['x']; t = c['top']; b = c['bottom']
            # main vertical segment limited to pin extents
            cv2.line(out, (cx, t), (cx, b), (255, 0, 0), 2)
            # small caps for visual clarity
            cv2.line(out, (cx - 6, t), (cx + 6, t), (255, 0, 0), 2)
            cv2.line(out, (cx - 6, b), (cx + 6, b), (255, 0, 0), 2)
            # annotate height near middle
            mid_y = (t + b) // 2
            txt = f"{c['height']}px"
            # put text to the right of the line (ensure inside image)
            tx = min(w_img - 80, cx + 8)
            ty = mid_y
            cv2.putText(out, txt, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

    return out, cluster_infos, raw_lines

# --- Note: minLineLength/maxLineGap are used as parameter names inside function call ---
# Define them (placeholders) so the function body references exist in global scope if needed.
# They will be shadowed by keyword arguments when the function is called.
minLineLength = 60
maxLineGap = 15

# ---------------- Accuracy helpers ----------------

def compute_distance_cv(distances):
    """Return coefficient of variation percentage if distances list exists, else large number"""
    if not distances:
        return 999.0
    arr = np.array(distances, dtype=float)
    mean = arr.mean()
    if mean == 0:
        return 999.0
    return float(np.std(arr) / mean * 100.0)

def compute_accuracy_from_cv(cv_percent, detected_count, expected_count):
    """Compute simple accuracy % based on CV and how many found vs expected"""
    if expected_count <= 0:
        return 0.0
    match_factor = min(1.0, detected_count / expected_count)
    # If CV is 0 -> perfect; higher CV reduces accuracy
    acc = max(0.0, 100.0 * match_factor * (1.0 - min(1.0, cv_percent / 100.0)))
    return acc

# ---------------- Main GUI App ----------------

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("S398 Pin Detector - PySide6")
        self.resize(1200, 720)

        # Left: original, Right: processed
        self.lbl_orig = QLabel("Original")
        self.lbl_proc = QLabel("Processed")
        for lbl in (self.lbl_orig, self.lbl_proc):
            lbl.setFixedSize(560, 420)
            lbl.setAlignment(Qt.AlignCenter)
            lbl.setStyleSheet("background: #222; color: white;")

        # Controls
        self.btn_upload = QPushButton("Upload Image")
        self.btn_webcam = QPushButton("Capture Webcam Frame")
        self.btn_process = QPushButton("Process")
        self.btn_upload.clicked.connect(self.upload_image)
        self.btn_webcam.clicked.connect(self.capture_webcam)
        self.btn_process.clicked.connect(self.process_current)

        # Mode selection
        self.combo_mode = QComboBox()
        self.combo_mode.addItems(["Top View (Circles)", "Side View (Vertical Lines)", "Bottom View (Circles-Alt)", "Auto"])
        self.combo_mode.setCurrentIndex(0)

        # Blur selection
        self.combo_blur = QComboBox()
        self.combo_blur.addItems(["median", "gaussian", "none"])
        self.spin_blur_k = QSpinBox(); self.spin_blur_k.setRange(1, 31); self.spin_blur_k.setValue(5)

        # Grayscale toggle
        self.chk_gray = QCheckBox("Use Grayscale"); self.chk_gray.setChecked(True)

        # Expected count & tolerance
        self.spin_expected = QSpinBox(); self.spin_expected.setRange(1, 10); self.spin_expected.setValue(3)
        self.spin_tol = QDoubleSpinBox(); self.spin_tol.setRange(0.1, 200.0); self.spin_tol.setValue(25.0)
        self.spin_tol.setSuffix(" % (CV)")

        # Hough circle params
        circle_group = QGroupBox("Circle (Top/Bottom) params")
        self.spin_c_param1 = QSpinBox(); self.spin_c_param1.setRange(10, 300); self.spin_c_param1.setValue(120)
        self.spin_c_param2 = QSpinBox(); self.spin_c_param2.setRange(5, 200); self.spin_c_param2.setValue(40)
        self.spin_c_minR = QSpinBox(); self.spin_c_minR.setRange(1, 500); self.spin_c_minR.setValue(8)
        self.spin_c_maxR = QSpinBox(); self.spin_c_maxR.setRange(1, 1000); self.spin_c_maxR.setValue(120)
        self.spin_c_minDist = QSpinBox(); self.spin_c_minDist.setRange(1, 1000); self.spin_c_minDist.setValue(40)

        # Line params
        line_group = QGroupBox("Line (Side) params")
        self.spin_canny1 = QSpinBox(); self.spin_canny1.setRange(1, 500); self.spin_canny1.setValue(50)
        self.spin_canny2 = QSpinBox(); self.spin_canny2.setRange(1, 500); self.spin_canny2.setValue(150)
        self.spin_hough_thresh = QSpinBox(); self.spin_hough_thresh.setRange(1, 500); self.spin_hough_thresh.setValue(50)
        self.spin_minLineLen = QSpinBox(); self.spin_minLineLen.setRange(1, 1000); self.spin_minLineLen.setValue(60)
        self.spin_maxLineGap = QSpinBox(); self.spin_maxLineGap.setRange(1, 1000); self.spin_maxLineGap.setValue(15)
        self.spin_angle_tol = QSpinBox(); self.spin_angle_tol.setRange(1, 89); self.spin_angle_tol.setValue(12)

        # Pass/Fail and accuracy labels
        self.lbl_status = QLabel("Status: -")
        self.lbl_accuracy = QLabel("Accuracy: - %")
        self.lbl_status.setStyleSheet("font-weight: bold; font-size: 16px;")
        self.lbl_accuracy.setStyleSheet("font-weight: bold; font-size: 14px;")

        # Layout assembly
        left_v = QVBoxLayout()
        left_v.addWidget(self.lbl_orig)

        right_v = QVBoxLayout()
        right_v.addWidget(self.lbl_proc)

        top_h = QHBoxLayout()
        top_h.addLayout(left_v); top_h.addLayout(right_v)

        controls_v = QVBoxLayout()
        row1 = QHBoxLayout()
        row1.addWidget(self.btn_upload); row1.addWidget(self.btn_webcam); row1.addWidget(self.btn_process)
        controls_v.addLayout(row1)

        row2 = QHBoxLayout()
        row2.addWidget(QLabel("Mode:")); row2.addWidget(self.combo_mode)
        row2.addWidget(QLabel("Blur:")); row2.addWidget(self.combo_blur); row2.addWidget(QLabel("k:")); row2.addWidget(self.spin_blur_k)
        row2.addWidget(self.chk_gray)
        controls_v.addLayout(row2)

        row3 = QHBoxLayout()
        row3.addWidget(QLabel("Expected count:")); row3.addWidget(self.spin_expected)
        row3.addWidget(QLabel("Dist tolerance (CV):")); row3.addWidget(self.spin_tol)
        controls_v.addLayout(row3)

        # circle params layout
        c_row = QHBoxLayout()
        c_row.addWidget(QLabel("param1:")); c_row.addWidget(self.spin_c_param1)
        c_row.addWidget(QLabel("param2:")); c_row.addWidget(self.spin_c_param2)
        c_row.addWidget(QLabel("minR:")); c_row.addWidget(self.spin_c_minR)
        c_row.addWidget(QLabel("maxR:")); c_row.addWidget(self.spin_c_maxR)
        c_row.addWidget(QLabel("minDist:")); c_row.addWidget(self.spin_c_minDist)
        circle_group.setLayout(c_row)
        controls_v.addWidget(circle_group)

        # line params layout
        l_row = QHBoxLayout()
        l_row.addWidget(QLabel("Canny1:")); l_row.addWidget(self.spin_canny1)
        l_row.addWidget(QLabel("Canny2:")); l_row.addWidget(self.spin_canny2)
        l_row.addWidget(QLabel("Hough thresh:")); l_row.addWidget(self.spin_hough_thresh)
        l_row.addWidget(QLabel("minLen:")); l_row.addWidget(self.spin_minLineLen)
        l_row.addWidget(QLabel("maxGap:")); l_row.addWidget(self.spin_maxLineGap)
        l_row.addWidget(QLabel("angleTol:")); l_row.addWidget(self.spin_angle_tol)
        line_group.setLayout(l_row)
        controls_v.addWidget(line_group)

        status_row = QHBoxLayout()
        status_row.addWidget(self.lbl_status); status_row.addStretch(); status_row.addWidget(self.lbl_accuracy)
        controls_v.addLayout(status_row)

        main_v = QVBoxLayout()
        main_v.addLayout(top_h)
        main_v.addLayout(controls_v)

        central = QWidget()
        central.setLayout(main_v)
        self.setCentralWidget(central)

        # internal state
        self.current_image = None
        self.processed_image = None

    # ---------------- UI actions ----------------
    def upload_image(self):
        path, _ = QFileDialog.getOpenFileName(self, "Open image", "", "Images (*.png *.jpg *.jpeg *.bmp *.tiff)")
        if not path:
            return
        img = cv2.imread(path)
        if img is None:
            QMessageBox.warning(self, "Read error", "Could not read that image file.")
            return
        self.current_image = img
        self.show_original(img)
        self.lbl_status.setText("Status: Image loaded. Press Process.")

    def capture_webcam(self):
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            QMessageBox.critical(self, "Webcam", "Could not open webcam.")
            return
        ret, frame = cap.read()
        cap.release()
        if not ret or frame is None:
            QMessageBox.critical(self, "Webcam", "Failed to capture frame.")
            return
        self.current_image = frame
        self.show_original(frame)
        self.lbl_status.setText("Status: Webcam frame captured. Press Process.")

    def show_original(self, img):
        pix = cv2_to_qpixmap(img)
        self.lbl_orig.setPixmap(pix.scaled(self.lbl_orig.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))

    def show_processed(self, img):
        pix = cv2_to_qpixmap(img)
        self.lbl_proc.setPixmap(pix.scaled(self.lbl_proc.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))

    # ---------------- Main process ----------------
    def process_current(self):
        if self.current_image is None:
            QMessageBox.information(self, "No image", "Please upload an image or capture from webcam first.")
            return

        # read UI parameters
        mode = self.combo_mode.currentText()
        blur = self.combo_blur.currentText()
        blur_k = int(self.spin_blur_k.value())
        use_gray = bool(self.chk_gray.isChecked())
        expected = int(self.spin_expected.value())
        tol_percent = float(self.spin_tol.value())

        # circle params
        c_param1 = int(self.spin_c_param1.value())
        c_param2 = int(self.spin_c_param2.value())
        c_minR = int(self.spin_c_minR.value())
        c_maxR = int(self.spin_c_maxR.value())
        c_minDist = int(self.spin_c_minDist.value())

        # line params
        canny1 = int(self.spin_canny1.value())
        canny2 = int(self.spin_canny2.value())
        hough_thresh = int(self.spin_hough_thresh.value())
        minLineLen = int(self.spin_minLineLen.value())
        maxLineGap = int(self.spin_maxLineGap.value())
        angle_tol = int(self.spin_angle_tol.value())

        frame = self.current_image.copy()
        processed = frame.copy()
        detected = False
        accuracy = 0.0
        info_text = ""

        # Mode handling
        if "Top" in mode or "Bottom" in mode:
            processed, centers = detect_circles(frame,
                                                blur_type=blur if blur != "none" else 'median',
                                                blur_k=blur_k,
                                                dp=1.2, minDist=c_minDist,
                                                param1=c_param1, param2=c_param2,
                                                minRadius=c_minR, maxRadius=c_maxR, draw=True)
            centers_xy = [(x, y) for (x, y, r) in centers]
            if centers_xy:
                xs = [c[0] for c in centers_xy]
                xs_sorted = sorted(xs)
                group = select_best_group_1d(xs_sorted, expected)
                distances = [abs(group[i+1] - group[i]) for i in range(len(group)-1)] if len(group) >= 2 else []
                cv_percent = compute_distance_cv(distances) if distances else 999.0
                accuracy = compute_accuracy_from_cv(cv_percent, len(group), expected)
                detected = (len(group) >= expected and (cv_percent <= tol_percent))
                info_text = f"Found centers: {len(centers_xy)}, Selected: {len(group)}, Distances(px): {distances}, CV%: {cv_percent:.1f}"
            else:
                info_text = "No circles found."
                accuracy = 0.0
                detected = False

        elif "Side" in mode:
            # Detect vertical lines and clusters (clusters contain exact top/bottom heights)
            processed, clusters, vlines = detect_vertical_lines(frame,
                                                                 blur_type=blur if blur != "none" else 'gaussian',
                                                                 blur_k=blur_k,
                                                                 canny_th1=canny1, canny_th2=canny2,
                                                                 hough_threshold=hough_thresh,
                                                                 minLineLength=minLineLen, maxLineGap=maxLineGap,
                                                                 angle_tol_deg=angle_tol, draw=True)
            if clusters:
                # clusters is a list of dicts: {'x', 'top', 'bottom', 'height'}
                xs = [c['x'] for c in clusters]
                group_xs = select_best_group_1d(xs, expected)
                # find selected clusters preserving original clusters order
                selected_clusters = []
                # clusters are sorted by x already
                for xval in group_xs:
                    # find cluster with matching x (exact match should hold since we stored ints)
                    match = next((c for c in clusters if c['x'] == int(round(xval))), None)
                    if match is None:
                        # fallback: choose nearest
                        match = min(clusters, key=lambda cc: abs(cc['x'] - xval))
                    selected_clusters.append(match)

                # compute horizontal distances between selected cluster centers
                distances = [abs(selected_clusters[i+1]['x'] - selected_clusters[i]['x']) for i in range(len(selected_clusters)-1)] if len(selected_clusters) >= 2 else []
                # heights of each selected pin
                heights = [c['height'] for c in selected_clusters]
                cv_percent = compute_distance_cv(distances) if distances else 999.0
                accuracy = compute_accuracy_from_cv(cv_percent, len(selected_clusters), expected)
                detected = (len(selected_clusters) >= expected and (cv_percent <= tol_percent))

                # highlight selected clusters on processed image with thicker green line and annotate distances/heights
                h_img, w_img = processed.shape[:2]
                for idx, c in enumerate(selected_clusters):
                    cx, top, bottom, height = c['x'], c['top'], c['bottom'], c['height']
                    cv2.line(processed, (cx, top), (cx, bottom), (0, 255, 0), 3)  # selected green
                    cv2.circle(processed, (cx, top), 4, (0,255,0), -1)
                    cv2.circle(processed, (cx, bottom), 4, (0,255,0), -1)
                    # annotate height near center
                    midy = (top + bottom) // 2
                    tx = min(w_img - 120, cx + 8)
                    cv2.putText(processed, f"H:{height}px", (tx, midy - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)

                # annotate distances between consecutive selected clusters
                for i in range(len(selected_clusters)-1):
                    c1 = selected_clusters[i]; c2 = selected_clusters[i+1]
                    midx = (c1['x'] + c2['x']) // 2
                    # place text above the top of the two
                    top_y = min(c1['top'], c2['top'])
                    text_y = max(10, top_y - 8)
                    cv2.putText(processed, f"D:{abs(c2['x']-c1['x']):.0f}px", (midx - 30, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255), 2)

                info_text = f"Found vertical clusters: {len(clusters)}, Selected: {len(selected_clusters)}, Distances(px): {distances}, Heights(px): {heights}, CV%: {cv_percent:.1f}"
            else:
                info_text = "No vertical lines found."
                accuracy = 0.0
                detected = False

        else:  # Auto: try circles then side
            proc1, centers = detect_circles(frame,
                                           blur_type=blur if blur != "none" else 'median',
                                           blur_k=blur_k,
                                           dp=1.2, minDist=c_minDist,
                                           param1=c_param1, param2=c_param2,
                                           minRadius=c_minR, maxRadius=c_maxR, draw=True)
            centers_xy = [(x, y) for (x, y, r) in centers]
            if centers_xy:
                xs = [c[0] for c in centers_xy]
                group = select_best_group_1d(sorted(xs), expected)
                distances = [abs(group[i+1] - group[i]) for i in range(len(group)-1)] if len(group) >= 2 else []
                cv_percent = compute_distance_cv(distances) if distances else 999.0
                accuracy = compute_accuracy_from_cv(cv_percent, len(group), expected)
                detected = (len(group) >= expected and (cv_percent <= tol_percent))
                processed = proc1
                info_text = f"(Auto->circles) Found: {len(centers_xy)}, Selected: {len(group)}, Distances: {distances}, CV%: {cv_percent:.1f}"
            else:
                proc2, clusters, _ = detect_vertical_lines(frame,
                                                           blur_type=blur if blur != "none" else 'gaussian',
                                                           blur_k=blur_k,
                                                           canny_th1=canny1, canny_th2=canny2,
                                                           hough_threshold=hough_thresh,
                                                           minLineLength=minLineLen, maxLineGap=maxLineGap,
                                                           angle_tol_deg=angle_tol, draw=True)
                if clusters:
                    xs = [c['x'] for c in clusters]
                    group_xs = select_best_group_1d(xs, expected)
                    selected_clusters = []
                    for xval in group_xs:
                        match = next((c for c in clusters if c['x'] == int(round(xval))), None)
                        if match is None:
                            match = min(clusters, key=lambda cc: abs(cc['x'] - xval))
                        selected_clusters.append(match)
                    distances = [abs(selected_clusters[i+1]['x'] - selected_clusters[i]['x']) for i in range(len(selected_clusters)-1)] if len(selected_clusters) >= 2 else []
                    heights = [c['height'] for c in selected_clusters]
                    cv_percent = compute_distance_cv(distances) if distances else 999.0
                    accuracy = compute_accuracy_from_cv(cv_percent, len(selected_clusters), expected)
                    detected = (len(selected_clusters) >= expected and (cv_percent <= tol_percent))
                    processed = proc2
                    info_text = f"(Auto->side) Found positions: {len(clusters)}, Selected: {len(selected_clusters)}, Distances: {distances}, Heights: {heights}, CV%: {cv_percent:.1f}"
                else:
                    processed = frame.copy()
                    info_text = "Auto: nothing found."
                    accuracy = 0.0
                    detected = False

        # update GUI
        self.processed_image = processed
        self.show_processed(processed)
        self.lbl_accuracy.setText(f"Accuracy: {accuracy:.1f} %")
        if detected:
            self.lbl_status.setText("Status: PASS")
            self.lbl_status.setStyleSheet("color: green; font-weight: bold; font-size: 16px;")
        else:
            self.lbl_status.setText("Status: FAIL")
            self.lbl_status.setStyleSheet("color: red; font-weight: bold; font-size: 16px;")
        # log info on status bar
        try:
            self.statusBar().showMessage(info_text)
        except Exception:
            pass

# ---------------- Run app ----------------

def main():
    app = QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
