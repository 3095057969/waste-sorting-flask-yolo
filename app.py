"""Flask 图片检测入口。上传图片和返回图像均在内存中处理。"""
import atexit
import io
import math
import os
import tempfile
import warnings
from pathlib import Path
from threading import BoundedSemaphore, RLock

from flask import Flask, Request, jsonify, render_template, request
from PIL import Image, ImageOps, UnidentifiedImageError
from PIL.Image import open as open_image
from werkzeug.exceptions import RequestEntityTooLarge

from detector import CATEGORIES, ModelNotLoadedError, WasteDetector

DEFAULT_MODEL = Path(__file__).resolve().parent / "best.pt"
MAX_PIXELS = 25_000_000
SUPPORTED_FORMATS = {"JPEG", "PNG", "WEBP", "BMP"}
SUPPORTED_MODEL_SUFFIXES = {".pt", ".onnx"}


class MemoryRequest(Request):
    """覆盖 Werkzeug 的磁盘回退行为，使 multipart 文件始终使用 BytesIO。"""

    def _get_file_stream(self, total_content_length, content_type, filename=None, content_length=None):
        return io.BytesIO()


def create_app(config=None, detector=None):
    app = Flask(__name__)
    app.request_class = MemoryRequest
    app.config.from_mapping(
        MAX_CONTENT_LENGTH=12 * 1024 * 1024,
        MODEL_MAX_CONTENT_LENGTH=256 * 1024 * 1024,
        MAX_FORM_PARTS=8,
        MODEL_PATH=os.environ.get("WASTE_MODEL_PATH", str(DEFAULT_MODEL)),
    )
    app.config.update(config or {})
    app.json.ensure_ascii = False
    engine = detector if detector is not None else WasteDetector(Path(app.config["MODEL_PATH"]))

    admission = BoundedSemaphore(1)
    model_state_lock = RLock()
    pending_model_path = None
    pending_model_name = None
    active_model_source = "unknown" if getattr(engine, "is_loaded", False) else "none"

    @app.before_request
    def set_route_upload_limit():
        if request.path == "/api/model/upload":
            request.max_content_length = app.config["MODEL_MAX_CONTENT_LENGTH"]

    @app.after_request
    def prevent_image_cache(response):
        response.headers["Cache-Control"] = "no-store, max-age=0"
        response.headers["X-Content-Type-Options"] = "nosniff"
        return response

    def remove_file(path):
        if path:
            try:
                Path(path).unlink(missing_ok=True)
            except OSError:
                app.logger.warning("无法删除临时模型文件 %s", path)

    def model_payload():
        with model_state_lock:
            loaded = bool(getattr(engine, "is_loaded", False))
            return {
                "loaded": loaded,
                "model_name": getattr(engine, "model_name", None) if loaded else None,
                "source": active_model_source if loaded else "none",
                "pending": pending_model_path is not None,
                "pending_name": pending_model_name,
            }

    def cleanup_models():
        nonlocal pending_model_path, pending_model_name
        with model_state_lock:
            pending_path = pending_model_path
            pending_model_path = None
            pending_model_name = None
        remove_file(pending_path)
        close = getattr(engine, "close", None)
        if close:
            close()

    atexit.register(cleanup_models)

    @app.get("/")
    def index():
        return render_template(
            "index.html",
            categories=CATEGORIES,
            max_upload_mb=app.config["MAX_CONTENT_LENGTH"] // (1024 * 1024),
            max_model_mb=app.config["MODEL_MAX_CONTENT_LENGTH"] // (1024 * 1024),
        )

    @app.get("/api/model")
    def model_status():
        return jsonify(model_payload())

    @app.post("/api/model/upload")
    def upload_model():
        nonlocal pending_model_path, pending_model_name, active_model_source
        upload = request.files.get("model")
        if upload is None or not upload.filename:
            return jsonify(error="请选择一个 .pt 或 .onnx 模型文件。"), 400

        original_name = Path(upload.filename).name
        suffix = Path(original_name).suffix.lower()
        if suffix not in SUPPORTED_MODEL_SUFFIXES:
            upload.close()
            return jsonify(error="模型格式仅支持 .pt 或 .onnx。"), 400

        temporary_path = None
        try:
            with tempfile.NamedTemporaryFile(prefix="waste-model-", suffix=suffix, delete=False) as target:
                upload.save(target)
                temporary_path = Path(target.name)
            if temporary_path.stat().st_size == 0:
                remove_file(temporary_path)
                return jsonify(error="模型文件为空，请重新选择模型。"), 400

            auto_load = request.form.get("auto_load", "").strip().lower() in {"1", "true", "yes", "on"}
            with model_state_lock:
                previous_pending_path = pending_model_path
                pending_model_path = temporary_path
                pending_model_name = original_name
                remove_file(previous_pending_path)

                if auto_load:
                    try:
                        engine.load_model(temporary_path, display_name=original_name, temporary=True)
                    except FileNotFoundError:
                        app.logger.exception("上传模型文件缺失")
                        state = model_payload()
                        state["error"] = "模型文件无法读取，请重新上传。"
                        return jsonify(state), 422
                    except Exception:
                        app.logger.exception("上传模型加载失败")
                        state = model_payload()
                        state["error"] = "模型加载失败，请确认模型与当前 Ultralytics 环境兼容。"
                        return jsonify(state), 422
                    pending_model_path = None
                    pending_model_name = None
                    active_model_source = "uploaded"

            return jsonify(model_payload())
        except OSError:
            app.logger.exception("保存上传模型失败")
            remove_file(temporary_path)
            return jsonify(error="模型文件暂时无法保存，请重试。"), 500
        finally:
            upload.close()

    @app.post("/api/model/load")
    def load_model():
        nonlocal pending_model_path, pending_model_name, active_model_source
        with model_state_lock:
            path = pending_model_path or Path(app.config["MODEL_PATH"])
            display_name = pending_model_name or path.name
            temporary = pending_model_path is not None
            try:
                engine.load_model(path, display_name=display_name, temporary=temporary)
            except FileNotFoundError:
                app.logger.exception("模型文件缺失")
                return jsonify(error="未找到模型文件，请检查默认模型路径或重新上传模型。"), 503
            except Exception:
                app.logger.exception("模型加载失败")
                return jsonify(error="模型加载失败，请确认模型与当前 Ultralytics 环境兼容。"), 422

            if temporary:
                pending_model_path = None
                pending_model_name = None
                active_model_source = "uploaded"
            else:
                active_model_source = "default"

        return jsonify(model_payload())

    @app.post("/api/detect")
    def detect():
        # 在解码前限制并发，避免连续更换图片时堆积高分辨率图像。
        if not admission.acquire(blocking=False):
            response = jsonify(error="模型正在处理图片，请稍候重试。")
            response.headers["Retry-After"] = "1"
            return response, 429
        try:
            return process_upload()
        finally:
            admission.release()

    def process_upload():
        upload = request.files.get("image")
        if upload is None or not upload.filename:
            return jsonify(error="请选择一张图片后再上传。"), 400
        try:
            confidence = float(request.form.get("confidence", "0.25"))
        except (TypeError, ValueError):
            upload.close()
            return jsonify(error="置信度阈值需要介于 0 与 1 之间。"), 400
        if not math.isfinite(confidence) or not 0 <= confidence <= 1:
            upload.close()
            return jsonify(error="置信度阈值需要介于 0 与 1 之间。"), 400

        image = None
        try:
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("error", Image.DecompressionBombWarning)
                    with open_image(upload.stream, formats=tuple(SUPPORTED_FORMATS)) as source:
                        if source.format not in SUPPORTED_FORMATS:
                            return jsonify(error="请上传 JPG、PNG、WEBP 或 BMP 格式的图片。"), 400
                        if source.width * source.height > MAX_PIXELS:
                            return jsonify(error="图片分辨率过大，请使用 2500 万像素以内的图片。"), 400
                        source.load()
                        image = ImageOps.exif_transpose(source).convert("RGB")
            except (Image.DecompressionBombError, Image.DecompressionBombWarning):
                return jsonify(error="图片分辨率过大，请缩小后重新上传。"), 400
            except (UnidentifiedImageError, OSError, ValueError):
                return jsonify(error="图片无法读取，请选择有效且完整的图片。"), 400

            if not getattr(engine, "is_loaded", False):
                return jsonify(error="模型尚未加载，请先点击“加载模型”后再识别图片。"), 409

            try:
                return jsonify(engine.detect(image, confidence))
            except ModelNotLoadedError:
                return jsonify(error="模型尚未加载，请先点击“加载模型”后再识别图片。"), 409
            except FileNotFoundError:
                app.logger.exception("模型文件缺失")
                return jsonify(error="未找到模型文件，请检查模型路径后重试。"), 503
            except Exception:
                app.logger.exception("图片推理失败")
                return jsonify(error="识别失败，请检查模型依赖和终端日志后重试。"), 500
        finally:
            if image is not None:
                image.close()
            upload.close()

    @app.errorhandler(RequestEntityTooLarge)
    def too_large(_error):
        limit = app.config["MAX_CONTENT_LENGTH"] // (1024 * 1024)
        if request.path == "/api/model/upload":
            limit = app.config["MODEL_MAX_CONTENT_LENGTH"] // (1024 * 1024)
        return jsonify(error=f"上传文件超过 {limit} MB，请压缩后重试。"), 413

    return app


app = create_app()


def run_server():
    app.run(
        host=os.environ.get("HOST", "0.0.0.0"),
        port=int(os.environ.get("PORT", "5000")),
        debug=False,
    )


if __name__ == "__main__":
    run_server()
