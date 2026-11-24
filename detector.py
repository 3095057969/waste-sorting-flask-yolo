"""加载 YOLO 权重，在内存中绘制中文检测框并释放每次推理的数据。"""
import base64
import io
import os
from pathlib import Path
from threading import Lock
from time import perf_counter

from PIL import Image, ImageDraw, ImageFont

CATEGORIES = (
    {"id": 0, "name": "可回收垃圾", "color": "#2866C7"},
    {"id": 1, "name": "其他垃圾", "color": "#64716B"},
    {"id": 2, "name": "厨余垃圾", "color": "#20844F"},
    {"id": 3, "name": "有害垃圾", "color": "#CE4945"},
)


class ModelNotLoadedError(RuntimeError):
    """在没有显式加载模型时进行推理。"""



def label_font(size):
    candidates = (
        os.environ.get("WASTE_FONT_PATH", ""),
        "C:/Windows/Fonts/msyh.ttc",
        "C:/Windows/Fonts/simhei.ttf",
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        "/System/Library/Fonts/PingFang.ttc",
    )
    for candidate in candidates:
        if candidate and Path(candidate).is_file():
            return ImageFont.truetype(candidate, size)
    return ImageFont.load_default(size=size)


class WasteDetector:
    def __init__(self, model_path):
        self.model_path = Path(model_path)
        self.model_name = None
        self._model = None
        self._temporary_model_path = None
        self._lock = Lock()

    @property
    def is_loaded(self):
        return self._model is not None

    def load_model(self, model_path=None, display_name=None, temporary=False):
        """显式加载一个模型，成功后才替换当前正在使用的模型。"""
        path = Path(model_path) if model_path is not None else self.model_path
        if not path.is_file():
            raise FileNotFoundError(path)
        from ultralytics import YOLO

        with self._lock:
            model = YOLO(str(path))
            previous_temporary_path = self._temporary_model_path
            self._model = model
            self.model_path = path
            self.model_name = display_name or path.name
            self._temporary_model_path = path if temporary else None
            if previous_temporary_path and previous_temporary_path != path:
                self._remove_temporary_file(previous_temporary_path)
        return self.model_name

    def close(self):
        """释放模型引用并清理由网页上传的临时模型文件。"""
        with self._lock:
            self._model = None
            self.model_name = None
            temporary_path = self._temporary_model_path
            self._temporary_model_path = None
            if temporary_path:
                self._remove_temporary_file(temporary_path)

    @staticmethod
    def _remove_temporary_file(path):
        try:
            Path(path).unlink(missing_ok=True)
        except OSError:
            pass

    def detect(self, image, confidence):
        # YOLO 的 predictor 包含当前输入及结果，共享模型的整个推理周期需要串行化。
        with self._lock:
            if self._model is None:
                raise ModelNotLoadedError("模型尚未加载，请先加载模型。")
            started = perf_counter()
            try:
                result = self._model.predict(
                    source=image, conf=confidence, iou=0.45, imgsz=640,
                    save=False, save_txt=False, save_crop=False, verbose=False,
                )[0]
                elapsed = (perf_counter() - started) * 1000
                detections = []
                for box, score, category_id in zip(
                    result.boxes.xyxy.tolist(), result.boxes.conf.tolist(), result.boxes.cls.tolist()
                ):
                    category = CATEGORIES[int(category_id)]
                    detections.append({
                        "id": len(detections) + 1,
                        "class_id": category["id"], "class_name": category["name"],
                        "color": category["color"], "confidence": round(float(score), 4),
                        "box": [round(float(value), 1) for value in box],
                    })
            finally:
                # Ultralytics 会保留 predictor.results / batch。释放 predictor 以清除上次图片。
                self._model.predictor = None

        annotated = image.copy()
        font = label_font(max(14, min(image.size) // 35))
        draw = ImageDraw.Draw(annotated)
        line_width = max(2, min(image.size) // 220)
        counts = [{**category, "count": 0} for category in CATEGORIES]
        for item in detections:
            counts[item["class_id"]]["count"] += 1
            left, top, right, bottom = item["box"]
            draw.rectangle((left, top, right, bottom), outline=item["color"], width=line_width)
            label = f'{item["id"]} {item["class_name"]} {item["confidence"]:.0%}'
            bounds = draw.textbbox((0, 0), label, font=font)
            label_width = bounds[2] - bounds[0] + 12
            label_height = bounds[3] - bounds[1] + 10
            x = max(0, min(left, image.width - label_width))
            y = max(0, top - label_height)
            draw.rectangle((x, y, x + label_width, y + label_height), fill=item["color"])
            draw.text((x + 6, y + 5 - bounds[1]), label, fill="white", font=font)
        try:
            with io.BytesIO() as buffer:
                annotated.save(buffer, format="JPEG", quality=92)
                encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
        finally:
            annotated.close()
        return {
            "image": f"data:image/jpeg;base64,{encoded}",
            "detections": detections, "counts": counts, "total": len(detections),
            "inference_ms": round(elapsed, 1), "width": image.width, "height": image.height,
        }
