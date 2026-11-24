# 垃圾四类识别展示网页

本仓库不包含训练权重文件。网页可加载本机默认 `.pt` 模型，也可上传 `.pt` 或 `.onnx` 模型。上传模型暂存于系统临时目录，不会作为项目文件保存。

使用 Python Flask 调用四分类 YOLO 模型，支持 `.pt` 与 `.onnx` 权重。上传或拖入图片后，网页自动显示中文检测框、类别、置信度、目标坐标及四类数量统计。

## 启动

Windows 下双击 `启动网页.bat`，随后访问 <http://127.0.0.1:5000>。

也可以在当前目录的 PowerShell 终端中运行以下命令。

```powershell
.\.venv\Scripts\python.exe app.py
```

首次在其他电脑配置环境时，使用 Python 3.10 或更新版本，并安装依赖。项目要求 PyTorch 2.6 或更新版本、Ultralytics 8.4.208 或更新版本，确保 `.pt` 检查点使用受限的 `weights_only` 加载。

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe app.py
```

当前电脑已创建 `.venv`，复用了系统环境已有的 PyTorch、Ultralytics、ONNX Runtime 和 Pillow，并在虚拟环境中安装 Flask 与 dill。网页启动时不会自动加载模型，首次手动加载模型会花费一些时间。

## 模型与类别

网页提供三个模型操作按钮。选择 PT 模型并上传、选择 ONNX 模型并上传，以及加载模型。未上传自定义权重时，加载模型按钮使用默认模型。网页启动后模型不会自动加载，必须先点击加载模型，才可以识别图片。

勾选“上传模型后自动加载”后，上传的自定义权重会立即加载。未勾选时，模型上传后处于待加载状态，需要再点击加载模型。此选项只影响上传的权重，默认模型始终由按钮手动加载。

上传的模型暂存于系统临时目录，供 Ultralytics 从文件读取。加载新模型或服务退出时会清理暂存文件。上传大小限制为 256 MB。ONNX 推理使用 `onnxruntime` CPU 运行时。

默认权重文件名为 `best.pt`，放在项目根目录即可。模型权重已加入 `.gitignore`，不会被推送到仓库。也可以通过环境变量 `WASTE_MODEL_PATH` 指定 `.pt` 或 `.onnx` 权重。

类别编号沿用数据目录的 `data.yaml`、`classes.txt` 及权重附带的 X-AnyLabeling 配置。

| 编号 | 类别 |
| --- | --- |
| 0 | 可回收垃圾 |
| 1 | 其他垃圾 |
| 2 | 厨余垃圾 |
| 3 | 有害垃圾 |

如果移动权重文件，可以通过环境变量设置新路径。

```powershell
$env:WASTE_MODEL_PATH = 'D:\新目录\best.pt'
.\.venv\Scripts\python.exe app.py
```

中文标签默认使用 Windows 微软雅黑或黑体。其他系统可以将 `WASTE_FONT_PATH` 指向本机支持中文的 `.ttf` 或 `.ttc` 字体文件。端口可通过 `PORT` 环境变量调整，默认仅监听本机 `127.0.0.1:5000`。

## 图片与结果生命周期

- 选择新图时立即清空上一张标注图、明细和统计，并释放上一张原图的浏览器对象 URL。
- 模型加载后，选择新图片会立即识别，更改置信度阈值会重新识别当前图片。模型未加载时仍可预览图片，加载模型后会自动识别当前图片。
- 后端使用内存文件流接收上传，用 Pillow 在内存中绘制标注，以 Base64 图像返回页面。
- 推理时显式关闭保存图片、标签与裁剪结果。每次推理结束释放 YOLO predictor 对图片与结果的引用。
- 页面仅保留当前图片与结果，支持手动清空。页面关闭或刷新时释放前端状态，HTTP 响应使用 `Cache-Control: no-store`。
- 后端在图片解码前限制并发，模型繁忙时页面自动等待重试。已开始的推理会执行结束，前端会忽略已被替换图片的响应。

支持 JPG、PNG、WEBP 和 BMP，单次上传限制为 12 MB，分辨率限制为 2500 万像素。输入方向会根据 EXIF 信息纠正。模型实际使用 640 的推理尺寸，检测框绘制在纠正方向后的原始分辨率图片上。

## 文件

| 文件 | 用途 |
| --- | --- |
| `app.py` | Flask 页面、上传校验与接口 |
| `detector.py` | 模型调用、中文标注与结果编码 |
| `templates/index.html` | 页面结构 |
| `static/style.css` | 桌面与移动端样式 |
| `static/app.js` | 上传、结果替换、统计和错误反馈 |
| `static/retry-delay.js` | 按服务端 `Retry-After` 响应头设置繁忙重试间隔 |
| `requirements.txt` | Python 运行依赖 |
| `启动网页.bat` | Windows 启动入口 |
| `tests/test_app.py` | 图片与模型上传、内存流和推理释放的回归测试 |
| `tests/retry-delay.test.cjs` | 浏览器重试间隔解析测试 |

## 验证

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
node --test tests/retry-delay.test.cjs
```
