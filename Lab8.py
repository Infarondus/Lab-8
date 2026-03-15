import cv2
import numpy as np

class ImageResizer:
    def __init__(self, image_path, scale_factor):
        self.image_path = image_path
        self.scale_factor = scale_factor
        self.image = cv2.imread(image_path)

    def resize_my_image(self, output_path='result.png'):
        if self.image is None:
            raise FileNotFoundError(f"Не удалось загрузить изображение: {self.image_path}")
        h, w = self.image.shape[:2]
        new_w = int(w * self.scale_factor)
        new_h = int(h * self.scale_factor)
        resized = cv2.resize(self.image, (new_w, new_h))
        cv2.imwrite(output_path, resized)
        print(f"Изображение сохранено как {output_path}")


class ReferenceMarker:
    def __init__(self):
        self.template = None
        self.template_size = None
        self.is_calibrated = False

    def calibrate_from_frame(self, gray_frame, center_x, center_y, radius):
        r = int(radius * 1.1)
        x1 = max(0, center_x - r)
        y1 = max(0, center_y - r)
        x2 = min(gray_frame.shape[1], center_x + r)
        y2 = min(gray_frame.shape[0], center_y + r)
        crop = gray_frame[y1:y2, x1:x2]
        if crop.size == 0:
            return False
        self.template = cv2.resize(crop, (120, 120))
        self.template_size = (120, 120)
        self.is_calibrated = True
        print("Метка захвачена как эталон")
        return True

    def reset(self):
        self.template = None
        self.template_size = None
        self.is_calibrated = False
        print("Эталон сброшен")

    def match_score(self, candidate_gray):
        if not self.is_calibrated or self.template is None:
            return 0.0
        try:
            candidate_resized = cv2.resize(candidate_gray, self.template_size)
            result = cv2.matchTemplate(candidate_resized, self.template, cv2.TM_CCOEFF_NORMED)
            return float(result[0, 0])
        except Exception:
            return 0.0


class MarkerDetector:
    def __init__(self, min_radius=25, max_radius=180):
        self.min_radius = min_radius
        self.max_radius = max_radius
        self.reference = ReferenceMarker()
        self.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

    def _preprocess(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = self.clahe.apply(gray)
        blurred = cv2.GaussianBlur(gray, (7, 7), 1.5)
        return gray, blurred

    def detect(self, frame):
        gray, blurred = self._preprocess(frame)

        circles = cv2.HoughCircles(
            blurred,
            cv2.HOUGH_GRADIENT,
            dp=1.2,
            minDist=60,
            param1=60,
            param2=22,
            minRadius=self.min_radius,
            maxRadius=self.max_radius,
        )

        if circles is None:
            return None, 0.0

        circles = np.round(circles[0, :]).astype("int")
        best_match = None
        best_score = -1.0

        for (x, y, r) in circles:
            crop = gray[max(0, y - r):min(gray.shape[0], y + r),
                        max(0, x - r):min(gray.shape[1], x + r)]
            if crop.shape[0] < 30 or crop.shape[1] < 30:
                continue
            score = self.reference.match_score(crop)
            if score > best_score:
                best_score = score
                best_match = (x, y, r)

        threshold = 0.60
        if best_match and best_score > threshold:
            return best_match, best_score
        return None, 0.0

    def detect_biggest_circle(self, frame):
        _, blurred = self._preprocess(frame)
        circles = cv2.HoughCircles(
            blurred, cv2.HOUGH_GRADIENT, 1.2, 60,
            param1=60, param2=22,
            minRadius=self.min_radius, maxRadius=self.max_radius,
        )
        if circles is None:
            return None
        circles = np.round(circles[0, :]).astype("int")
        return max(circles, key=lambda c: c[2])


class TrackerApp:
    def __init__(self, camera_index=0, overlay_path="overlay.png"):
        self.cap = cv2.VideoCapture(camera_index)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

        self.detector = MarkerDetector()
        self.overlay_renderer = OverlayRenderer(overlay_path)
        self.calibrating = True
        self.left_count = 0
        self.right_count = 0
        self.last_side = None

    def run(self):
        if not self.cap.isOpened():
            print("Не удалось открыть камеру")
            return

        print("Q — захватить метку как эталон | R — сбросить эталон | ESC — выход")

        while True:
            ret, frame = self.cap.read()
            if not ret:
                break

            display_frame = frame.copy()
            frame_width = frame.shape[1]

            if self.calibrating:
                cv2.putText(
                    display_frame,
                    "The ancient evil has awaken. Press Q to capture your mark",
                    (40, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2,
                )
            else:
                marker, score = self.detector.detect(frame)
                if marker:
                    x, y, r = marker

                    
                    side = "left" if x < frame_width // 2 else "right"
                    if side != self.last_side:
                        if side == "left":
                            self.left_count += 1
                        else:
                            self.right_count += 1
                        self.last_side = side

                    cv2.circle(display_frame, (x, y), r, (0, 255, 0), 3)
                    cv2.circle(display_frame, (x, y), 8, (0, 0, 255), -1)
                    cv2.putText(
                        display_frame,
                        f"Match: {score:.3f}  ({x},{y})    L: {self.left_count}  R: {self.right_count}",
                        (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 120), 2,
                    )
                    diameter = 2 * r
                    display_frame = self.overlay_renderer.apply(display_frame, x, y, diameter)

            cv2.imshow("Marker Tracker", display_frame)
            key = cv2.waitKey(30) & 0xFF

            if key == 27:   # ESC
                break
            elif key in (ord('q'), ord('Q')) and self.calibrating:
                circle = self.detector.detect_biggest_circle(frame)
                if circle is not None:
                    x, y, r = circle
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    if self.detector.reference.calibrate_from_frame(gray, x, y, r):
                        self.calibrating = False
                        print("Калибровка завершена")
                else:
                    print("Круг не найден — попробуйте снова (Q)")
            elif key in (ord('r'), ord('R')):
                self.detector.reference.reset()
                self.calibrating = True
                self.left_count = 0
                self.right_count = 0

        self.cap.release()
        cv2.destroyAllWindows()


class OverlayRenderer:
    def __init__(self, overlay_path):
        self.overlay = cv2.imread(overlay_path, cv2.IMREAD_UNCHANGED)
        if self.overlay is None:
            print(f"Не удалось загрузить {overlay_path}")

    def apply(self, frame, center_x, center_y, diameter):
        if self.overlay is None:
            return frame
        overlay_size = int(diameter * 0.9)
        if overlay_size < 30:
            return frame
        resized = cv2.resize(self.overlay, (overlay_size, overlay_size))
        ox = center_x - overlay_size // 2
        oy = center_y - overlay_size // 2
        if ox < 0 or oy < 0 or ox + overlay_size > frame.shape[1] or oy + overlay_size > frame.shape[0]:
            return frame
        if resized.shape[2] == 4:
            alpha = resized[:, :, 3:] / 255.0
            roi = frame[oy:oy + overlay_size, ox:ox + overlay_size]
            frame[oy:oy + overlay_size, ox:ox + overlay_size] = (
                alpha * resized[:, :, :3] + (1 - alpha) * roi
            ).astype(np.uint8)
        else:
            frame[oy:oy + overlay_size, ox:ox + overlay_size] = resized
        return frame

resizer = ImageResizer("variant-6.png", scale_factor=2.0)
resizer.resize_my_image(output_path="variant-6_x2.png")
print("Изображение variant-6 увеличено и сохранено как variant-6_x2.png")

if __name__ == "__main__":
    app = TrackerApp(camera_index=0, overlay_path="fly64.png")
    app.run()
