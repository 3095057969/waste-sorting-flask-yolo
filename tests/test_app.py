"""回归验证上传校验、内存处理和真实模型接口的边界行为。"""
import base64
import io
import threading
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
from PIL import Image

from app import create_app, MemoryRequest
from detector import WasteDetector


class DemoDetector:
    def __init__(self, loaded=True):
        self.calls = 0
        self.is_loaded = loaded
        self.model_name = None
        self.loaded_models = []

    def load_model(self, model_path, display_name=None, temporary=False):
        path = Path(model_path)
        self.loaded_models.append((path.suffix.lower(), path.read_bytes(), display_name, temporary))
        self.is_loaded = True
        self.model_name = display_name or path.name

    def detect(self, image, confidence):
        self.calls += 1
        return {"image": "data:image/jpeg;base64,AA==", "detections": [],
                "counts": [], "total": 0, "inference_ms": 1,
                "width": image.width, "height": image.height}


def image_file():
    stream = io.BytesIO()
    Image.new("RGB", (80, 60), "white").save(stream, "PNG")
    stream.seek(0)
    return stream


class UploadTests(unittest.TestCase):
    def setUp(self):
        self.detector = DemoDetector()
        self.app = create_app({"TESTING": True, "MAX_CONTENT_LENGTH": 1024 * 1024}, self.detector)
        self.client = self.app.test_client()

    def test_home_and_assets(self):
        self.assertEqual(self.client.get("/").status_code, 200)
        with self.client.get("/static/app.js") as response:
            self.assertEqual(response.status_code, 200)

    def test_missing_file(self):
        response = self.client.post("/api/detect")
        self.assertEqual(response.status_code, 400)
        self.assertIn("error", response.json)

    def test_disguised_image_is_rejected(self):
        response = self.client.post("/api/detect", data={"image": (io.BytesIO(b"text"), "fake.jpg")})
        self.assertEqual(response.status_code, 400)
        self.assertEqual(self.detector.calls, 0)

    def test_sequential_uploads_are_independent_and_uncached(self):
        for _ in range(2):
            response = self.client.post("/api/detect", data={"image": (image_file(), "sample.png")})
            self.assertEqual(response.status_code, 200)
            self.assertEqual(response.json["total"], 0)
            self.assertIn("no-store", response.headers["Cache-Control"])
            self.assertNotIn("Set-Cookie", response.headers)
        self.assertEqual(self.detector.calls, 2)

    def test_invalid_confidence(self):
        for value in ("nan", "inf", "-0.1", "1.1", "abc"):
            with self.subTest(value=value):
                response = self.client.post("/api/detect", data={
                    "image": (image_file(), "sample.png"), "confidence": value})
                self.assertEqual(response.status_code, 400)
        self.assertEqual(self.detector.calls, 0)

    def test_oversized_request_has_json_error(self):
        self.app.config["MAX_CONTENT_LENGTH"] = 128
        response = self.client.post("/api/detect", data={
            "image": (io.BytesIO(b"a" * 256), "large.jpg")})
        self.assertEqual(response.status_code, 413)
        self.assertIn("error", response.json)

    def test_stream_stays_in_memory_even_for_large_upload(self):
        request = MemoryRequest.from_values()
        stream = request._get_file_stream(2 * 1024 * 1024, "image/png", "large.png")
        self.assertIsInstance(stream, io.BytesIO)
        stream.close()
        request.close()

    def test_busy_upload_rejected_before_extra_inference(self):
        entered = threading.Event()
        release = threading.Event()
        responses = []
        def busy_detect(image, confidence):
            entered.set()
            release.wait(timeout=5)
            return {"total": 0}
        def send_first():
            with self.app.test_client() as client:
                responses.append(client.post("/api/detect", data={"image": (image_file(), "a.png")}).status_code)
        with patch.object(self.detector, "detect", side_effect=busy_detect):
            thread = threading.Thread(target=send_first)
            thread.start()
            try:
                self.assertTrue(entered.wait(timeout=3))
                response = self.client.post("/api/detect", data={"image": (image_file(), "b.png")})
                self.assertEqual(response.status_code, 429)
            finally:
                release.set()
                thread.join(timeout=5)
        self.assertEqual(responses, [200])

    def test_detection_requires_an_explicitly_loaded_model(self):
        detector = DemoDetector(loaded=False)
        app = create_app({"TESTING": True}, detector)
        response = app.test_client().post("/api/detect", data={
            "image": (image_file(), "sample.png")})
        self.assertEqual(response.status_code, 409)
        self.assertEqual(detector.calls, 0)

    def test_model_upload_waits_for_manual_load_by_default(self):
        detector = DemoDetector(loaded=False)
        app = create_app({"TESTING": True}, detector)
        client = app.test_client()
        response = client.post("/api/model/upload", data={
            "model": (io.BytesIO(b"pt-model-bytes"), "waste.pt")})
        self.assertEqual(response.status_code, 200)
        self.assertFalse(response.json["loaded"])
        self.assertEqual(detector.loaded_models, [])

        response = client.post("/api/model/load")
        self.assertEqual(response.status_code, 200)
        self.assertTrue(response.json["loaded"])
        self.assertEqual(detector.loaded_models[0][:3], (".pt", b"pt-model-bytes", "waste.pt"))
        self.assertTrue(detector.loaded_models[0][3])

    def test_onnx_upload_can_be_loaded_automatically(self):
        detector = DemoDetector(loaded=False)
        app = create_app({"TESTING": True}, detector)
        response = app.test_client().post("/api/model/upload", data={
            "model": (io.BytesIO(b"onnx-model-bytes"), "waste.onnx"),
            "auto_load": "1",
        })
        self.assertEqual(response.status_code, 200)
        self.assertTrue(response.json["loaded"])
        self.assertEqual(detector.loaded_models[0][:3], (".onnx", b"onnx-model-bytes", "waste.onnx"))

    def test_model_upload_rejects_unsupported_extensions(self):
        detector = DemoDetector(loaded=False)
        app = create_app({"TESTING": True}, detector)
        response = app.test_client().post("/api/model/upload", data={
            "model": (io.BytesIO(b"not-a-model"), "waste.pt.txt")})
        self.assertEqual(response.status_code, 400)
        self.assertEqual(detector.loaded_models, [])

    def test_load_without_upload_uses_the_configured_default_model(self):
        detector = DemoDetector(loaded=False)
        app = create_app({"TESTING": True, "MODEL_PATH": str(Path(__file__))}, detector)
        response = app.test_client().post("/api/model/load")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(detector.loaded_models[0][0], ".py")
        self.assertEqual(detector.loaded_models[0][2], Path(__file__).name)
        self.assertFalse(detector.loaded_models[0][3])

    def test_model_selection_controls_match_uploaded_formats_and_manual_default(self):
        response = self.client.get("/")
        self.assertIn(b'id="model-pt-button"', response.data)
        self.assertIn(b'id="model-onnx-button"', response.data)
        self.assertIn(b'id="load-model-button"', response.data)
        self.assertIn(b'id="pt-model-input" type="file" accept=".pt', response.data)
        self.assertIn(b'id="onnx-model-input" type="file" accept=".onnx', response.data)
        self.assertIn(b'id="auto-load-model" type="checkbox"', response.data)
        self.assertIn("默认模型始终手动加载".encode(), response.data)

    def test_default_model_path_is_local_to_the_project(self):
        import os
        project_root = Path(__file__).resolve().parents[1]
        with patch.dict(os.environ):
            os.environ.pop("WASTE_MODEL_PATH", None)
            app = create_app({"TESTING": True}, DemoDetector(loaded=False))
        self.assertEqual(Path(app.config["MODEL_PATH"]), project_root / "best.pt")

    def test_default_model_is_not_loaded_until_requested(self):
        detector = DemoDetector(loaded=False)
        app = create_app({"TESTING": True}, detector)
        response = app.test_client().get("/api/model")
        self.assertEqual(response.status_code, 200)
        self.assertFalse(response.json["loaded"])
        self.assertFalse(response.json["pending"])
        self.assertEqual(detector.loaded_models, [])

    def test_model_failure_returns_message(self):
        with patch.object(self.detector, "detect", side_effect=RuntimeError("internal trace")):
            response = self.client.post("/api/detect", data={"image": (image_file(), "sample.png")})
        self.assertEqual(response.status_code, 500)
        self.assertIn("error", response.json)
        self.assertNotIn("internal trace", response.json["error"])


class DetectorTests(unittest.TestCase):
    def test_pt_loading_enables_restricted_checkpoint_loading(self):
        import ultralytics.nn.tasks as tasks

        with tempfile.TemporaryDirectory() as directory:
            model_path = Path(directory) / "model.pt"
            model_path.write_bytes(b"test checkpoint")
            model = SimpleNamespace()

            def create_model(_path):
                self.assertTrue(tasks.SAFE_LOAD)
                return model

            detector = WasteDetector(model_path)
            with patch.object(tasks, "SAFE_LOAD", False), patch("ultralytics.YOLO", side_effect=create_model):
                detector.load_model()
                self.assertTrue(tasks.SAFE_LOAD)

    def test_pt_loading_refuses_when_restricted_loader_is_unavailable(self):
        import ultralytics.nn.tasks as tasks

        with tempfile.TemporaryDirectory() as directory:
            model_path = Path(directory) / "model.pt"
            model_path.write_bytes(b"test checkpoint")
            detector = WasteDetector(model_path)
            with patch.object(tasks._SafeLoad, "SUPPORTED", False), patch("ultralytics.YOLO") as create_model:
                with self.assertRaises(RuntimeError):
                    detector.load_model()
                create_model.assert_not_called()

    def test_pt_loading_registers_dill_checkpoint_helper(self):
        import dill._dill as dill_impl

        with tempfile.TemporaryDirectory() as directory:
            model_path = Path(directory) / "model.pt"
            model_path.write_bytes(b"test checkpoint")
            detector = WasteDetector(model_path)
            with patch("torch.serialization.add_safe_globals") as register:
                with patch("ultralytics.YOLO", return_value=SimpleNamespace()):
                    detector.load_model()
            registered = register.call_args.args[0]
            self.assertIn(dill_impl._load_type, registered)

    def test_annotation_uses_chinese_classes_and_releases_prediction(self):
        boxes = SimpleNamespace(xyxy=np.array([[10, 8, 60, 50]]),
                                conf=np.array([0.88]), cls=np.array([2]))
        result = SimpleNamespace(boxes=boxes)
        class Model:
            predictor = None
            def predict(self, **kwargs):
                self.options = kwargs
                self.predictor = SimpleNamespace(results=[result], batch=kwargs["source"])
                return [result]
        model = Model()
        detector = WasteDetector(Path(__file__))
        with patch("ultralytics.YOLO", return_value=model):
            detector.load_model()
            output = detector.detect(Image.new("RGB", (80, 60)), 0.25)
        self.assertEqual(output["detections"][0]["class_name"], "厨余垃圾")
        self.assertEqual(output["total"], 1)
        self.assertEqual(output["counts"][2]["count"], 1)
        self.assertFalse(model.options["save"])
        self.assertIsNone(model.predictor)
        data = base64.b64decode(output["image"].split(",", 1)[1])
        with Image.open(io.BytesIO(data)) as annotated:
            self.assertEqual(annotated.size, (80, 60))

    def test_predictor_released_after_failure(self):
        class Model:
            predictor = object()
            def predict(self, **kwargs):
                raise RuntimeError("failure")
        model = Model()
        detector = WasteDetector(Path(__file__))
        with patch("ultralytics.YOLO", return_value=model):
            detector.load_model()
            with self.assertRaises(RuntimeError):
                detector.detect(Image.new("RGB", (80, 60)), 0.25)
        self.assertIsNone(model.predictor)


if __name__ == "__main__":
    unittest.main()
