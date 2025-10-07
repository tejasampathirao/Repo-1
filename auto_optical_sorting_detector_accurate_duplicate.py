import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog, messagebox
import os

class PerfectS398Detector:
    def __init__(self):
        self.target_pcd = 11.66
        self.tolerance = 0.13
        
    def preprocess_image(self, img):
        """Perfect preprocessing with median filter"""
        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Apply median filter (better than Gaussian for noise removal)
        filtered = cv2.medianBlur(gray, 5)
        
        # Enhance contrast
        enhanced = cv2.equalizeHist(filtered)
        
        return enhanced
    
    def detect_pins(self, img):
        """Detect exactly 3 pins"""
        processed = self.preprocess_image(img)
        
        # Detect circles with optimized parameters
        circles = cv2.HoughCircles(
            processed,
            cv2.HOUGH_GRADIENT,
            dp=1,
            minDist=60,
            param1=120,
            param2=40,
            minRadius=15,
            maxRadius=80
        )
        
        if circles is not None:
            circles = np.round(circles[0, :]).astype("int")
            
            # If more than 3, keep the 3 best ones
            if len(circles) > 3:
                # Calculate circle quality (edge strength)
                scores = []
                for (x, y, r) in circles:
                    roi = processed[max(0,y-r):min(processed.shape[0],y+r), 
                                  max(0,x-r):min(processed.shape[1],x+r)]
                    if roi.size > 0:
                        edges = cv2.Canny(roi, 50, 150)
                        score = np.sum(edges)
                        scores.append(score)
                    else:
                        scores.append(0)
                
                # Keep top 3 circles
                top3_indices = np.argsort(scores)[-3:]
                circles = circles[top3_indices]
            
            return circles
        
        return None
    
    def calculate_distances(self, pins):
        """Calculate all distances between pins"""
        distances = []
        for i in range(len(pins)):
            for j in range(i+1, len(pins)):
                x1, y1 = pins[i][:2]
                x2, y2 = pins[j][:2]
                dist = np.sqrt((x2-x1)**2 + (y2-y1)**2)
                distances.append(dist)
        return distances
    
    def process_image(self, image_path):
        """Main processing function"""
        print(f"\n🔍 Processing: {os.path.basename(image_path)}")
        
        # Load image
        img = cv2.imread(image_path)
        if img is None:
            return None, "❌ Could not load image!"
        
        # Make a copy for drawing
        result_img = img.copy()
        
        # Detect pins
        pins = self.detect_pins(img)
        
        if pins is None:
            return result_img, "❌ No pins detected!"
        
        num_pins = len(pins)
        print(f"✅ Detected {num_pins} pins")
        
        # Draw detected pins
        for i, (x, y, r) in enumerate(pins):
            # Draw circle
            cv2.circle(result_img, (x, y), r, (0, 255, 0), 3)
            # Draw center
            cv2.circle(result_img, (x, y), 3, (0, 0, 255), -1)
            # Label
            cv2.putText(result_img, f"Pin{i+1}", (x-25, y-r-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        
        # Calculate distances if exactly 3 pins
        if num_pins == 3:
            distances = self.calculate_distances(pins)
            avg_distance = np.mean(distances)
            
            # Draw distance lines
            pin_pairs = [(0,1), (1,2), (0,2)]
            for i, (p1, p2) in enumerate(pin_pairs):
                x1, y1 = pins[p1][:2]
                x2, y2 = pins[p2][:2]
                cv2.line(result_img, (x1, y1), (x2, y2), (255, 0, 255), 2)
                
                # Distance text
                mid_x, mid_y = (x1+x2)//2, (y1+y2)//2
                cv2.putText(result_img, f"{distances[i]:.1f}px", 
                           (mid_x-20, mid_y), cv2.FONT_HERSHEY_SIMPLEX, 
                           0.5, (255, 0, 255), 2)
            
            # Status
            status = "OK" if abs(avg_distance - self.target_pcd) <= self.tolerance else "NG"
            color = (0, 255, 0) if status == "OK" else (0, 0, 255)
            
            # Add status text
            cv2.putText(result_img, f"Status: {status}", (20, 40), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)
            cv2.putText(result_img, f"Pins: {num_pins}/3", (20, 80), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            cv2.putText(result_img, f"Avg Distance: {avg_distance:.2f}px", (20, 120), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
            
            # Print results
            print(f"📏 Distances: {[f'{d:.1f}' for d in distances]} pixels")
            print(f"📊 Average: {avg_distance:.2f} pixels")
            print(f"🎯 Status: {status}")
            
            return result_img, f"✅ Success: {status} - {num_pins} pins detected"
            
        else:
            # Not exactly 3 pins
            cv2.putText(result_img, f"Need 3 pins, found {num_pins}", (20, 40), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
            return result_img, f"❌ Found {num_pins} pins, need exactly 3"


def upload_and_process():
    """Image upload interface"""
    # Create file dialog
    root = tk.Tk()
    root.withdraw()  # Hide main window
    
    # Open file dialog
    file_path = filedialog.askopenfilename(
        title="Select S398 Terminal Image",
        filetypes=[
            ("Image files", "*.jpg *.jpeg *.png *.bmp *.tiff"),
            ("All files", "*.*")
        ]
    )
    
    if not file_path:
        print("❌ No file selected!")
        root.destroy()
        return
    
    # Process the image
    detector = PerfectS398Detector()
    result_img, message = detector.process_image(file_path)
    
    if result_img is not None:
        # --- RESIZE DISPLAY IMAGE (only for showing) ---
        # Adjust these to the maximum display size you want:
        max_w, max_h = 900, 700  # change to 800,600 if you prefer smaller
        h, w = result_img.shape[:2]
        scale = min(max_w / w, max_h / h, 1.0)  # never upscale, only shrink
        if scale < 1.0:
            new_w = int(w * scale)
            new_h = int(h * scale)
            disp_img = cv2.resize(result_img, (new_w, new_h), interpolation=cv2.INTER_AREA)
        else:
            disp_img = result_img

        # Show result
        window_name = f'S398 Detection - {os.path.basename(file_path)}'
        cv2.imshow(window_name, disp_img)
        print(f"\n{message}")
        print("👀 Press any key to close...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        # result_img is None
        messagebox.showerror("Error", message)
    
    root.destroy()

def main():
    """Main function"""
    print("=== PERFECT S398 PIN DETECTOR ===")
    print("📁 Click OK to select your image file...")
    
    while True:
        try:
            upload_and_process()
            
            # Ask if want to process another image
            root = tk.Tk()
            root.withdraw()
            
            result = messagebox.askyesno("Continue?", 
                                       "Process another image?")
            root.destroy()
            
            if not result:
                break
                
        except Exception as e:
            print(f"❌ Error: {e}")
            break
    
    print("👋 Done!")

if __name__ == "__main__":
    main()
