import cv2
import numpy as np
import math
import os
import sys

# -----------------------
# USER CONFIG
# -----------------------
# Replace with your IP camera URL (or use 0 for local webcam)
CAMERA_URL = "rtsp://admin:idt12345@192.168.0.160:554/cam/realmonitor?channel=1&subtype=0"

# Max display size (so window fits screen). The script maps clicks back to original image coords.
MAX_DISPLAY_W = 1280
MAX_DISPLAY_H = 720

SAVE_DIR = "homography_measure_outputs"
os.makedirs(SAVE_DIR, exist_ok=True)

# -----------------------
# UTILS
# -----------------------
def image_to_world(pt, H):
    """Convert image pixel (x,y) to world coordinates (X_mm, Y_mm) using homography H."""
    x, y = float(pt[0]), float(pt[1])
    vec = np.array([x, y, 1.0], dtype=np.float64)
    w = H.dot(vec)            # 3-vector
    if abs(w[2]) < 1e-8:
        return None
    X = w[0] / w[2]
    Y = w[1] / w[2]
    return (float(X), float(Y))

def world_distance_mm(pw1, pw2):
    dx = pw1[0] - pw2[0]
    dy = pw1[1] - pw2[1]
    return math.hypot(dx, dy)

# -----------------------
# GLOBAL STATE (used by mouse callback)
# -----------------------
ref_pts_img = []      # list of 4 (x,y) image coordinates for reference rectangle (original image coords)
measure_pts_img = []  # list of two (x,y) image coords to measure
homography = None
display_scale = 1.0   # display scaling factor
orig_frame_shape = None
save_count = 0

mode_msg = ("Step 1: Click 4 corners of a known rectangular reference in THIS order:\n"
            "  top-left, top-right, bottom-right, bottom-left\n"
            "Then type its real width and height in mm when prompted.\n"
            "After homography is computed, click any two points to measure distance in mm.\n"
            "Keys: r=reset, s=save snapshot, ESC=quit")

# -----------------------
# Mouse callback
# -----------------------
def on_mouse(event, x, y, flags, param):
    global ref_pts_img, measure_pts_img, display_scale, orig_frame_shape, homography
    # Map clicked display coords back to original image coords
    if orig_frame_shape is None:
        return
    img_h, img_w = orig_frame_shape[:2]
    inv_scale = 1.0 / display_scale if display_scale != 0 else 1.0
    img_x = int(round(x * inv_scale))
    img_y = int(round(y * inv_scale))
    if event == cv2.EVENT_LBUTTONDOWN:
        if homography is None:
            # collecting reference corners
            if len(ref_pts_img) < 4:
                ref_pts_img.append((img_x, img_y))
                print(f"Ref corner {len(ref_pts_img)}: (img) {img_x, img_y}")
            else:
                print("Reference already has 4 points. Press 'r' to reset or 'c' to recompute.")
        else:
            # homography ready -> collecting measurement points
            measure_pts_img.append((img_x, img_y))
            if len(measure_pts_img) > 2:
                measure_pts_img = measure_pts_img[-2:]
            print(f"Selected measure point #{len(measure_pts_img)}: (img) {img_x, img_y}")

# -----------------------
# MAIN
# -----------------------
def main():
    global ref_pts_img, measure_pts_img, homography, display_scale, orig_frame_shape, save_count

    print(mode_msg)
    cap = cv2.VideoCapture(CAMERA_URL)
    if not cap.isOpened():
        print("ERROR: cannot open camera URL:", CAMERA_URL)
        sys.exit(1)

    # read first valid frame to get resolution
    orig = None
    for _ in range(50):
        ret, frame = cap.read()
        if ret and frame is not None:
            orig = frame
            break
    if orig is None:
        print("ERROR: failed to read frames from camera.")
        cap.release()
        sys.exit(1)

    orig_h, orig_w = orig.shape[:2]
    orig_frame_shape = orig.shape

    # determine display scale so image fits screen
    sx = min(1.0, MAX_DISPLAY_W / orig_w)
    sy = min(1.0, MAX_DISPLAY_H / orig_h)
    display_scale = min(sx, sy)
    print(f"Camera resolution: {orig_w}x{orig_h}, display scale = {display_scale:.3f}")

    cv2.namedWindow("HomographyMeasure", cv2.WINDOW_NORMAL)
    cv2.setMouseCallback("HomographyMeasure", on_mouse)

    snapshot_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            # try again
            continue

        img = frame.copy()

        # draw reference points (if any) - scaled for display
        disp = cv2.resize(img, (int(orig_w * display_scale), int(orig_h * display_scale)),
                          interpolation=cv2.INTER_LINEAR)

        # draw ref corners
        for i, p in enumerate(ref_pts_img):
            px = int(round(p[0] * display_scale))
            py = int(round(p[1] * display_scale))
            cv2.circle(disp, (px, py), 6, (0, 255, 255), -1)
            cv2.putText(disp, f"R{i+1}", (px + 6, py - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255), 2)
        # if 4 ref corners present, draw polygon
        if len(ref_pts_img) == 4:
            pts = np.array([ (int(round(x*display_scale)), int(round(y*display_scale))) for (x,y) in ref_pts_img ], np.int32)
            cv2.polylines(disp, [pts], isClosed=True, color=(0,255,255), thickness=2)

        # draw measurement points and compute distance if homography exists
        if homography is not None:
            # draw measure points
            for i, p in enumerate(measure_pts_img):
                px = int(round(p[0] * display_scale))
                py = int(round(p[1] * display_scale))
                cv2.circle(disp, (px, py), 6, (0, 255, 0), -1)
                cv2.putText(disp, f"P{i+1}", (px + 6, py - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

            if len(measure_pts_img) >= 2:
                p1_img = measure_pts_img[-2]
                p2_img = measure_pts_img[-1]
                # compute world coords
                w1 = image_to_world(p1_img, homography)
                w2 = image_to_world(p2_img, homography)
                if w1 is not None and w2 is not None:
                    dist_mm = world_distance_mm(w1, w2)
                    # draw line
                    p1_disp = (int(round(p1_img[0]*display_scale)), int(round(p1_img[1]*display_scale)))
                    p2_disp = (int(round(p2_img[0]*display_scale)), int(round(p2_img[1]*display_scale)))
                    cv2.line(disp, p1_disp, p2_disp, (255, 0, 0), 2)
                    mid = ((p1_disp[0]+p2_disp[0])//2, (p1_disp[1]+p2_disp[1])//2)
                    cv2.putText(disp, f"{dist_mm:.2f} mm", (mid[0]+8, mid[1]-8),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,0,0), 2)
                else:
                    cv2.putText(disp, "Homography mapping error", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)

        # Instructions overlay
        y0 = 20
        for idx, line in enumerate([
            "Instructions: click 4 reference corners (TL,TR,BR,BL) then enter real dims in mm",
            "After homography computed, click two points to measure distance (mm).",
            "Keys: r=reset, s=save annotated image, q or ESC=quit"
        ]):
            cv2.putText(disp, line, (10, y0 + idx*22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1)

        cv2.imshow("HomographyMeasure", disp)
        key = cv2.waitKey(1) & 0xFF

        if key in (27, ord('q')):  # ESC or q
            break
        elif key == ord('r'):
            # reset everything
            ref_pts_img = []
            measure_pts_img = []
            homography = None
            print("Reset: reference points and homography cleared.")
        elif key == ord('s'):
            # save annotated full-res image with drawn overlays (draw on copy of original)
            save_img = img.copy()
            # draw reference
            for i, p in enumerate(ref_pts_img):
                cv2.circle(save_img, (int(p[0]), int(p[1])), 6, (0,255,255), -1)
                cv2.putText(save_img, f"R{i+1}", (int(p[0])+6, int(p[1])-6), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255), 2)
            if homography is not None:
                for i, p in enumerate(measure_pts_img):
                    cv2.circle(save_img, (int(p[0]), int(p[1])), 6, (0,255,0), -1)
                    cv2.putText(save_img, f"P{i+1}", (int(p[0])+6, int(p[1])-6), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
                if len(measure_pts_img) >= 2:
                    w1 = image_to_world(measure_pts_img[-2], homography)
                    w2 = image_to_world(measure_pts_img[-1], homography)
                    if w1 is not None and w2 is not None:
                        dist_mm = world_distance_mm(w1, w2)
                        cv2.line(save_img, (int(measure_pts_img[-2][0]), int(measure_pts_img[-2][1])),
                                 (int(measure_pts_img[-1][0]), int(measure_pts_img[-1][1])), (255,0,0), 2)
                        cv2.putText(save_img, f"{dist_mm:.2f} mm", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,0,0), 2)
            snapshot_idx += 1
            fname = os.path.join(SAVE_DIR, f"annotated_{snapshot_idx}.png")
            cv2.imwrite(fname, save_img)
            print("Saved annotated image:", fname)

        # if we have exactly 4 reference points and homography not computed yet -> compute now
        if len(ref_pts_img) == 4 and homography is None:
            # ask user for real dims
            print("\nFour reference corners clicked. Please enter the real rectangle size (mm).")
            try:
                w_mm = float(input("Enter reference width in mm (e.g. 85.6 for credit card width): ").strip())
                h_mm = float(input("Enter reference height in mm (e.g. 53.98 for credit card height): ").strip())
            except Exception as e:
                print("Invalid input. Resetting reference points. Press 'r' and try again.")
                ref_pts_img = []
                continue

            # Build source and destination arrays
            src = np.array(ref_pts_img, dtype=np.float64)  # shape (4,2) image pixels
            # destination world coordinates in mm in the same order as clicked:
            dst = np.array([
                [0.0, 0.0],
                [w_mm, 0.0],
                [w_mm, h_mm],
                [0.0, h_mm]
            ], dtype=np.float64)

            # compute homography mapping image -> world(mm)
            H, status = cv2.findHomography(src, dst, method=0)  # simple DLT, no RANSAC (points should be correct)
            if H is None:
                print("Homography computation failed. Reset and try again.")
                ref_pts_img = []
                continue
            homography = H
            print("Homography computed successfully. You can now click two points to measure distance in mm.")
            # clear measure points
            measure_pts_img = []

    # end loop
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
