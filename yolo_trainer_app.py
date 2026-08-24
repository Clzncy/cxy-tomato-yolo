"""番茄 YOLO 训练平台 - 桌面版 (PySide6)

依赖: pip install pyside6 -i https://mirrors.aliyun.com/pypi/simple
运行: python yolo_trainer_app.py
"""

import os
import random
import re
import shutil
import subprocess
import sys
from datetime import datetime

from PySide6.QtCore import QProcess, QRectF, QSettings, Qt
from PySide6.QtGui import QColor, QFont, QIcon, QPainter, QPen, QPixmap
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDialog,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMessageBox,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QTabWidget,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

APP_DIR = os.path.dirname(os.path.abspath(__file__))
ANSI_RE = re.compile(r"\x1b\[[0-9;?]*[a-zA-Z]|\x1b\][^\x07]*\x07")
EPOCH_RE = re.compile(r"Epoch\s*(\d+)/(\d+)")

MODEL_SCAN_DIRS = [
    r"E:\yolo训练平台\YoloTrain212\YoloTrain\models",
    r"E:\PlantVillage-Dataset\tomato-ready\tomato_dataset\runs\detect\tomato_yolo11m_1024\weights",
    r"E:\tomato\tomato_dataset",
    os.path.join(APP_DIR, "models"),
]

AMP_FALLBACKS = [
    r"E:\yolo训练平台\YoloTrain212\YoloTrain\models\yolo26n.pt",
    r"E:\迅雷下载\yolo26n.pt",
]

STYLE = """
QMainWindow, QWidget { background: #f4f6fb; color: #26303f; font-size: 13px; }
QGroupBox { background: white; border: 1px solid #dde2ec; border-radius: 10px;
            margin-top: 14px; padding: 10px; font-weight: 600; color: #2b3a55; }
QGroupBox::title { subcontrol-origin: margin; left: 12px; padding: 0 5px; }
QPushButton { background: #4a6cf7; color: white; border: none; border-radius: 7px;
              padding: 7px 16px; font-weight: 600; }
QPushButton:hover { background: #3b5ce0; }
QPushButton:pressed { background: #2f4bc4; }
QPushButton:disabled { background: #b9c1d8; }
QPushButton#ghost { background: #eef1f8; color: #2b3a55; border: 1px solid #d0d6e4; }
QPushButton#ghost:hover { background: #e2e7f2; }
QPushButton#danger { background: #e74c3c; }
QPushButton#danger:hover { background: #c9412f; }
QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox, QListWidget, QTableWidget {
    background: white; border: 1px solid #d0d6e4; border-radius: 6px; padding: 4px 6px; }
QLineEdit:focus, QSpinBox:focus, QDoubleSpinBox:focus, QComboBox:focus { border: 1px solid #4a6cf7; }
QPlainTextEdit { background: #202433; color: #d8e2ff; border: 1px solid #202433;
                 border-radius: 8px; padding: 6px; }
QTabWidget::pane { border: 1px solid #dde2ec; border-radius: 10px; background: white; top: -1px; }
QTabBar::tab { background: #e7ebf4; color: #4a5568; padding: 9px 22px; margin-right: 2px;
               border-top-left-radius: 8px; border-top-right-radius: 8px; font-weight: 600; }
QTabBar::tab:selected { background: #4a6cf7; color: white; }
QTabBar::tab:hover:!selected { background: #d7ddef; }
QProgressBar { border: none; border-radius: 6px; background: #e7ebf4; text-align: center; height: 16px; }
QProgressBar::chunk { background: #4a6cf7; border-radius: 6px; }
QCheckBox { spacing: 6px; }
QTableWidget { gridline-color: #e6eaf2; }
QHeaderView::section { background: #eef1f8; border: none; padding: 5px; font-weight: 600; }
QLabel#title { font-size: 18px; font-weight: 700; color: #2b3a55; }
QLabel#metric { font-size: 20px; font-weight: 700; color: #4a6cf7; }
QLabel#caption { color: #7a8699; font-size: 12px; }
"""


def strip_ansi(text: str) -> str:
    return ANSI_RE.sub("", text).replace("\r", "\n")


def quiet_process(proc: QProcess) -> None:
    """Windows 下让子进程不弹出黑色控制台窗口。"""
    if hasattr(proc, "setCreateProcessArgumentsModifier"):
        proc.setCreateProcessArgumentsModifier(
            lambda args: setattr(args, "flags", getattr(args, "flags", 0) | subprocess.CREATE_NO_WINDOW)
        )


def scan_models() -> list[str]:
    found: dict[str, str] = {}
    for d in MODEL_SCAN_DIRS:
        if not os.path.isdir(d):
            continue
        for root, dirs, files in os.walk(d):
            depth = root[len(d):].count(os.sep)
            if depth > 3:
                dirs[:] = []
                continue
            for name in files:
                if name.lower().endswith((".pt", ".onnx")):
                    found[name] = os.path.join(root, name)
    return list(found.values())


def parse_yaml_simple(path: str) -> dict:
    info = {"path": "", "train": "train/images", "val": "val/images", "test": "test/images", "nc": 0, "names": []}
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            for line in f:
                line = line.strip()
                if line.startswith("#") or ":" not in line:
                    continue
                key, _, value = line.partition(":")
                key = key.strip()
                value = value.strip().strip("'\"")
                if key in ("path", "train", "val", "test"):
                    info[key] = value
                elif key == "nc":
                    try:
                        info["nc"] = int(value)
                    except ValueError:
                        pass
                elif key == "names":
                    names = re.findall(r"[\u4e00-\u9fff\w.-]+", value)
                    if names:
                        info["names"] = names
    except OSError:
        pass
    return info


def count_dir(path: str) -> int:
    if not os.path.isdir(path):
        return 0
    return sum(1 for f in os.listdir(path) if os.path.isfile(os.path.join(path, f)))


def dataset_summary(yaml_path: str) -> dict:
    info = parse_yaml_simple(yaml_path)
    base = info["path"] or os.path.dirname(yaml_path)
    result = {"names": info["names"], "nc": info["nc"], "splits": {}}
    for split in ("train", "val", "test"):
        rel = info.get(split, split)
        imgs = os.path.join(base, rel)
        if not os.path.isdir(imgs):
            imgs = os.path.join(base, rel, "images")
        n_img = count_dir(imgs)
        n_lbl = count_dir(os.path.join(os.path.dirname(imgs), "labels"))
        result["splits"][split] = (n_img, n_lbl)
    return result


def latest_metrics(run_dir: str) -> dict[str, str]:
    """读取 results.csv 最后一行，按表头名返回 {列名: 值}。"""
    csv = os.path.join(run_dir, "results.csv")
    if not os.path.isfile(csv):
        return {}
    try:
        with open(csv, "r", encoding="utf-8", errors="replace") as f:
            lines = [l for l in f.read().splitlines() if l.strip()]
        if len(lines) < 2:
            return {}
        header = [x.strip() for x in lines[0].split(",")]
        row = [x.strip() for x in lines[-1].split(",")]
    except OSError:
        return {}
    out: dict[str, str] = {}
    for i, key in enumerate(header):
        if i < len(row):
            out[key] = row[i]
    return out


def run_folders(data_yaml: str) -> list[str]:
    base = os.path.join(os.path.dirname(data_yaml), "runs", "detect") if data_yaml else ""
    if not os.path.isdir(base):
        return []
    folders = [f for f in os.listdir(base) if os.path.isdir(os.path.join(base, f))]
    folders.sort(key=lambda n: os.path.getmtime(os.path.join(base, n)), reverse=True)
    return [os.path.join(base, n) for n in folders]


class LogPane(QPlainTextEdit):
    def __init__(self) -> None:
        super().__init__()
        self.setReadOnly(True)
        self.setFont(QFont("Consolas", 9))
        self.setMaximumBlockCount(8000)

    def append(self, text: str) -> None:
        self.appendPlainText(text.rstrip("\n"))


def browse_model(combo: QComboBox, parent) -> None:
    """打开文件对话框，手动选择一个 .pt/.onnx 模型。"""
    path, _ = QFileDialog.getOpenFileName(
        parent, "选择模型文件", "", "模型文件 (*.pt *.onnx);;所有文件 (*.*)"
    )
    if path:
        combo.setCurrentText(path)


def make_row(edit: QLineEdit, slot) -> QHBoxLayout:
    h = QHBoxLayout()
    h.addWidget(edit, 1)
    btn = QPushButton("浏览")
    btn.setObjectName("ghost")
    btn.clicked.connect(slot)
    h.addWidget(btn)
    return h


def set_pixmap(label: QLabel, path: str) -> None:
    if not os.path.isfile(path):
        label.setText("(图片不存在)")
        label.setPixmap(QPixmap())
        return
    pix = QPixmap(path)
    label.setPixmap(pix.scaledToWidth(760, Qt.SmoothTransformation))


class TrainTab(QWidget):
    def __init__(self, main_window: "MainWindow") -> None:
        super().__init__()
        self.main = main_window
        self.settings = main_window.settings
        self.proc: QProcess | None = None

        root = QVBoxLayout(self)
        cfg = QGroupBox("训练配置")
        form = QFormLayout(cfg)

        self.model_combo = QComboBox()
        self.model_combo.setEditable(True)
        self.model_combo.setInsertPolicy(QComboBox.NoInsert)
        self.refresh_models()
        btn_refresh = QPushButton("刷新")
        btn_refresh.setObjectName("ghost")
        btn_refresh.clicked.connect(self.refresh_models)
        mr = QHBoxLayout()
        mr.addWidget(self.model_combo, 1)
        mr.addWidget(btn_refresh)

        self.data_edit = QLineEdit(
            self.settings.value("data", r"E:\tomato\tomato_multi_wholeleaf\dataset.yaml")
        )
        self.data_edit.textChanged.connect(self.main.overview.refresh)
        self.data_info = QLabel("选择数据集后自动显示信息")
        self.data_info.setWordWrap(True)

        self.epochs = QSpinBox()
        self.epochs.setRange(1, 1000)
        self.epochs.setValue(100)
        self.imgsz = QSpinBox()
        self.imgsz.setRange(320, 1920)
        self.imgsz.setSingleStep(32)
        self.imgsz.setValue(640)
        self.batch = QSpinBox()
        self.batch.setRange(-1, 256)
        self.batch.setValue(16)
        self.batch.setSpecialValueText("auto")
        self.device = QLineEdit("0")
        self.name = QLineEdit()
        self.name.setPlaceholderText("留空自动编号")
        self.extra = QLineEdit()
        self.extra.setPlaceholderText("其它参数，如 cutmix=0.2 lr0=0.005 seed=42")

        form.addRow("模型文件", mr)
        form.addRow("数据集配置 (data.yaml)", make_row(self.data_edit, self._browse_data))
        form.addRow("", self.data_info)
        form.addRow("训练轮数", self.epochs)
        form.addRow("图像尺寸 imgsz", self.imgsz)
        form.addRow("批量 batch (-1=自动)", self.batch)
        form.addRow("GPU device", self.device)
        form.addRow("实验名", self.name)
        form.addRow("其它参数", self.extra)
        root.addWidget(cfg)

        tools = QGroupBox("数据集与权重工具")
        th = QHBoxLayout(tools)
        for text, slot in (
            ("数据集生成", self._open_dataset_gen),
            ("数据集浏览", self._open_dataset_browse),
            ("权重配置", self._open_weights),
        ):
            b = QPushButton(text)
            b.setObjectName("ghost")
            b.clicked.connect(slot)
            th.addWidget(b)
        th.addStretch(1)
        root.addWidget(tools)

        aug = QGroupBox("数据增强（自由选择，勾选即启用）")
        ag = QGridLayout(aug)
        self.aug = {}

        def add_aug(row, col, name, key, default, maxv, step, decimals=1, spin_step=0.1):
            cb = QCheckBox(name)
            sp = QDoubleSpinBox()
            sp.setRange(0.0, maxv)
            sp.setSingleStep(step)
            sp.setDecimals(decimals)
            sp.setValue(default)
            cb.toggled.connect(sp.setEnabled)
            cb.setChecked(default > 0)
            box = QHBoxLayout()
            box.addWidget(cb)
            box.addWidget(sp)
            ag.addLayout(box, row, col)
            self.aug[key] = (cb, sp)

        add_aug(0, 0, "马赛克 mosaic", "mosaic", 1.0, 1.0, 0.1)
        add_aug(0, 1, "水平翻转 fliplr", "fliplr", 0.5, 1.0, 0.1)
        add_aug(0, 2, "垂直翻转 flipud", "flipud", 0.0, 1.0, 0.1)
        add_aug(1, 0, "随机缩放 scale", "scale", 0.5, 1.0, 0.1)
        add_aug(1, 1, "随机平移 translate", "translate", 0.1, 1.0, 0.1)
        add_aug(1, 2, "随机旋转 degrees", "degrees", 0.0, 180.0, 5.0, decimals=1, spin_step=5)
        add_aug(2, 0, "混合增强 mixup", "mixup", 0.0, 1.0, 0.1)
        add_aug(2, 1, "随机擦除 erasing", "erasing", 0.4, 1.0, 0.1)

        self.hsv_cb = QCheckBox("色彩增强 hsv (H/S/V)")
        self.hsv_h = QDoubleSpinBox()
        self.hsv_h.setRange(0.0, 0.1)
        self.hsv_h.setSingleStep(0.005)
        self.hsv_h.setDecimals(3)
        self.hsv_h.setValue(0.015)
        self.hsv_s = QDoubleSpinBox()
        self.hsv_s.setRange(0.0, 1.0)
        self.hsv_s.setSingleStep(0.1)
        self.hsv_s.setValue(0.7)
        self.hsv_v = QDoubleSpinBox()
        self.hsv_v.setRange(0.0, 1.0)
        self.hsv_v.setSingleStep(0.1)
        self.hsv_v.setValue(0.4)
        self.hsv_cb.toggled.connect(self._toggle_hsv)
        hbox = QHBoxLayout()
        hbox.addWidget(self.hsv_cb)
        hbox.addWidget(QLabel("H"))
        hbox.addWidget(self.hsv_h)
        hbox.addWidget(QLabel("S"))
        hbox.addWidget(self.hsv_s)
        hbox.addWidget(QLabel("V"))
        hbox.addWidget(self.hsv_v)
        hbox.addStretch(1)
        ag.addLayout(hbox, 2, 1, 1, 2)
        root.addWidget(aug)

        presets = QGroupBox("快捷预设")
        ph = QHBoxLayout(presets)
        for label, epochs, imgsz, batch in (
            ("快速测试 (20轮)", 20, 640, 8),
            ("标准训练 (100轮)", 100, 640, 16),
            ("高精度 (150轮,1024)", 150, 1024, 8),
        ):
            btn = QPushButton(label)
            btn.setObjectName("ghost")
            btn.clicked.connect(lambda _=False, e=epochs, i=imgsz, b=batch: self.apply_preset(e, i, b))
            ph.addWidget(btn)
        ph.addStretch(1)
        root.addWidget(presets)

        buttons = QHBoxLayout()
        self.btn_start = QPushButton("开始训练")
        self.btn_start.clicked.connect(self.start)
        self.btn_stop = QPushButton("停止训练")
        self.btn_stop.setObjectName("danger")
        self.btn_stop.clicked.connect(self.stop)
        self.btn_stop.setEnabled(False)
        self.btn_export = QPushButton("导出 ONNX")
        self.btn_export.setObjectName("ghost")
        self.btn_export.clicked.connect(self.export_onnx)
        self.btn_open = QPushButton("打开结果文件夹")
        self.btn_open.setObjectName("ghost")
        self.btn_open.clicked.connect(self.open_results)
        buttons.addWidget(self.btn_start)
        buttons.addWidget(self.btn_stop)
        buttons.addWidget(self.btn_export)
        buttons.addWidget(self.btn_open)
        buttons.addStretch(1)
        root.addLayout(buttons)

        self.progress = QProgressBar()
        self.progress.setVisible(False)
        root.addWidget(self.progress)
        self.log = LogPane()
        root.addWidget(self.log, 1)
        self.update_dataset_info()

    def _toggle_hsv(self, checked: bool) -> None:
        for sp in (self.hsv_h, self.hsv_s, self.hsv_v):
            sp.setEnabled(checked)

    def refresh_models(self) -> None:
        current = self.model_combo.currentText()
        self.model_combo.clear()
        for path in scan_models():
            self.model_combo.addItem(path)
        if current:
            self.model_combo.setCurrentText(current)
        elif self.model_combo.count():
            self.model_combo.setCurrentIndex(0)

    def apply_preset(self, epochs: int, imgsz: int, batch: int) -> None:
        self.epochs.setValue(epochs)
        self.imgsz.setValue(imgsz)
        self.batch.setValue(batch)
        self.log.append(f"[预设] epochs={epochs} imgsz={imgsz} batch={batch}")

    def _open_dataset_gen(self) -> None:
        dlg = DatasetGenDialog(self, on_done=self._dataset_generated)
        dlg.exec()

    def _dataset_generated(self, yaml_path: str) -> None:
        self.data_edit.setText(yaml_path)
        self.log.append(f"[数据集生成] 已切换到 {yaml_path}")
        self.main.overview.refresh()

    def _open_dataset_browse(self) -> None:
        data = self.data_edit.text().strip().strip('"')
        if not os.path.isfile(data):
            QMessageBox.information(self, "提示", "请先在训练页选择 data.yaml")
            return
        DatasetBrowseDialog(self, data).exec()

    def _open_weights(self) -> None:
        WeightsDialog(self, on_select=self._set_model).exec()

    def _set_model(self, path: str) -> None:
        self.model_combo.setCurrentText(path)
        self.log.append(f"[权重配置] 当前模型 -> {path}")

    def _browse_data(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "选择 dataset.yaml", "", "YAML (*.yaml);;所有文件 (*.*)")
        if path:
            self.data_edit.setText(path)

    def update_dataset_info(self) -> None:
        path = self.data_edit.text().strip().strip('"')
        if os.path.isfile(path):
            s = dataset_summary(path)
            names = ", ".join(s["names"]) if s["names"] else "?"
            parts = [f"类别: {s['nc'] or '?'} ({names})"]
            for k, (ni, nl) in s["splits"].items():
                if ni:
                    parts.append(f"{k}: {ni}图/{nl}标")
            self.data_info.setText(" | ".join(parts))
        else:
            self.data_info.setText("找不到该 data.yaml")

    def build_aug_args(self) -> str:
        tokens = []
        for key, (cb, sp) in self.aug.items():
            if cb.isChecked():
                tokens.append(f"{key}={sp.value()}")
        if self.hsv_cb.isChecked():
            tokens.append(f"hsv_h={self.hsv_h.value()}")
            tokens.append(f"hsv_s={self.hsv_s.value()}")
            tokens.append(f"hsv_v={self.hsv_v.value()}")
        return " ".join(tokens)

    def _ensure_amp_model(self, workdir: str) -> None:
        target = os.path.join(workdir, "yolo26n.pt")
        if os.path.isfile(target):
            return
        for src in AMP_FALLBACKS:
            if os.path.isfile(src):
                try:
                    shutil.copy2(src, target)
                    self.log.append(f"[信息] 已复制 yolo26n.pt (AMP 自检用) -> {target}")
                except OSError as exc:
                    self.log.append(f"[警告] 复制 yolo26n.pt 失败: {exc}")
                return
        self.log.append("[警告] 未找到本地 yolo26n.pt，AMP 自检可能尝试联网。")

    def start(self) -> None:
        if self.proc is not None and self.proc.state() != QProcess.NotRunning:
            QMessageBox.warning(self, "提示", "训练正在进行中。")
            return
        data = self.data_edit.text().strip().strip('"')
        model = self.model_combo.currentText().strip().strip('"')
        if not os.path.isfile(data):
            QMessageBox.critical(self, "错误", f"找不到数据集配置:\n{data}")
            return
        if not os.path.isfile(model):
            QMessageBox.critical(self, "错误", f"找不到模型文件:\n{model}")
            return
        workdir = os.path.dirname(data)
        self._ensure_amp_model(workdir)

        aug = self.build_aug_args()
        extra = " ".join(x for x in (aug, self.extra.text().strip()) if x)
        cmd = [
            os.path.join(APP_DIR, "train_worker.py"),
            "--data", data,
            "--model", model,
            "--epochs", str(self.epochs.value()),
            "--imgsz", str(self.imgsz.value()),
            "--batch", str(self.batch.value()),
            "--device", self.device.text().strip() or "0",
            "--name", self.name.text().strip(),
            "--extra", extra,
        ]
        self.log.append("\n" + "=" * 60)
        self.log.append(f"开始训练: {os.path.basename(model)}  epochs={self.epochs.value()} "
                        f"imgsz={self.imgsz.value()} batch={self.batch.value()}")
        self.log.append(f"数据集: {data}")
        if extra:
            self.log.append(f"增强/参数: {extra}")
        self.log.append("=" * 60)

        self.proc = QProcess(self)
        quiet_process(self.proc)
        self.proc.setProcessChannelMode(QProcess.MergedChannels)
        self.proc.setWorkingDirectory(workdir)
        self.proc.readyReadStandardOutput.connect(self._read_output)
        self.proc.finished.connect(self._on_finished)
        self.proc.start(sys.executable, cmd)

        self.btn_start.setEnabled(False)
        self.btn_stop.setEnabled(True)
        self.progress.setRange(0, self.epochs.value())
        self.progress.setValue(0)
        self.progress.setVisible(True)
        self.main.statusBar().showMessage("训练中 ...")

    def stop(self) -> None:
        if self.proc is not None and self.proc.state() != QProcess.NotRunning:
            self.proc.kill()
            self.log.append("\n[已发送停止指令]")
            self.main.statusBar().showMessage("正在停止 ...")

    def export_onnx(self, model: str = "") -> None:
        model = model or self.model_combo.currentText().strip().strip('"')
        if not os.path.isfile(model):
            QMessageBox.critical(self, "错误", f"找不到模型文件:\n{model}")
            return
        self.log.append(f"\n[导出 ONNX] {model} imgsz={self.imgsz.value()}")
        self.proc = QProcess(self)
        quiet_process(self.proc)
        self.proc.setProcessChannelMode(QProcess.MergedChannels)
        self.proc.setWorkingDirectory(os.path.dirname(model))
        self.proc.readyReadStandardOutput.connect(self._read_output)
        self.proc.finished.connect(self._on_export_finished)
        self.proc.start(sys.executable, [
            "-m", "ultralytics", "export",
            "model=" + model, "format=onnx", f"imgsz={self.imgsz.value()}",
        ])

    def open_results(self) -> None:
        data = self.data_edit.text().strip().strip('"')
        base = os.path.join(os.path.dirname(data), "runs", "detect") if data else ""
        if os.path.isdir(base):
            os.startfile(base)
        else:
            QMessageBox.information(self, "提示", f"还没有结果文件夹:\n{base}")

    def _read_output(self) -> None:
        assert self.proc is not None
        raw = bytes(self.proc.readAllStandardOutput()).decode("utf-8", errors="replace")
        for line in strip_ansi(raw).split("\n"):
            if not line.strip():
                continue
            self.log.append(line)
            m = EPOCH_RE.search(line)
            if m and self.progress.isVisible():
                self.progress.setMaximum(int(m.group(2)))
                self.progress.setValue(int(m.group(1)))

    def _on_finished(self, code: int, _status) -> None:
        self.proc = None
        self.btn_start.setEnabled(True)
        self.btn_stop.setEnabled(False)
        self.progress.setVisible(False)
        if code == 0:
            self.main.statusBar().showMessage("训练完成 ✓")
            self.log.append("\n[训练完成]")
            self.main.overview.refresh()
            self.main.report.refresh_runs()
        else:
            self.main.statusBar().showMessage(f"训练已停止 (退出码 {code})")
            self.log.append(f"\n[进程结束，退出码 {code}]")

    def _on_export_finished(self, code: int, _status) -> None:
        self.main.statusBar().showMessage("导出完成" if code == 0 else f"导出失败 (退出码 {code})")


class PreannotateTab(QWidget):
    def __init__(self, main_window: "MainWindow") -> None:
        super().__init__()
        self.main = main_window
        self.settings = main_window.settings
        self.proc: QProcess | None = None

        root = QVBoxLayout(self)
        group = QGroupBox("智能预标注")
        form = QFormLayout(group)

        self.model_combo = QComboBox()
        self.model_combo.setEditable(True)
        self.model_combo.setInsertPolicy(QComboBox.NoInsert)
        for path in scan_models():
            self.model_combo.addItem(path)
        best = self.settings.value("pre_model", "")
        if best:
            self.model_combo.setCurrentText(best)
        btn_refresh = QPushButton("刷新")
        btn_refresh.setObjectName("ghost")
        btn_refresh.clicked.connect(self.refresh_models)
        mr = QHBoxLayout()
        mr.addWidget(self.model_combo, 1)
        mr.addWidget(btn_refresh)

        self.source_edit = QLineEdit(self.settings.value("pre_source", r"E:\tomato\new_images"))
        self.out_edit = QLineEdit(self.settings.value("pre_out", r"E:\tomato\preannotate_out"))
        self.conf = QDoubleSpinBox()
        self.conf.setRange(0.01, 0.99)
        self.conf.setSingleStep(0.05)
        self.conf.setValue(0.25)
        self.imgsz = QSpinBox()
        self.imgsz.setRange(320, 1920)
        self.imgsz.setSingleStep(32)
        self.imgsz.setValue(640)

        form.addRow("模型文件", mr)
        form.addRow("待标注图片文件夹", make_row(self.source_edit, self._browse_source))
        form.addRow("输出文件夹", make_row(self.out_edit, self._browse_out))
        form.addRow("置信度 conf", self.conf)
        form.addRow("图像尺寸 imgsz", self.imgsz)
        root.addWidget(group)

        buttons = QHBoxLayout()
        self.btn_run = QPushButton("开始预标注")
        self.btn_run.clicked.connect(self.run)
        self.btn_stop = QPushButton("停止")
        self.btn_stop.setObjectName("danger")
        self.btn_stop.clicked.connect(self.stop)
        self.btn_stop.setEnabled(False)
        self.btn_open = QPushButton("打开输出文件夹")
        self.btn_open.setObjectName("ghost")
        self.btn_open.clicked.connect(self.open_out)
        buttons.addWidget(self.btn_run)
        buttons.addWidget(self.btn_stop)
        buttons.addWidget(self.btn_open)
        buttons.addStretch(1)
        root.addLayout(buttons)

        self.log = LogPane()
        root.addWidget(self.log, 1)

    def refresh_models(self) -> None:
        current = self.model_combo.currentText()
        self.model_combo.clear()
        for path in scan_models():
            self.model_combo.addItem(path)
        if current:
            self.model_combo.setCurrentText(current)

    def _browse_source(self) -> None:
        path = QFileDialog.getExistingDirectory(self, "选择待标注图片文件夹")
        if path:
            self.source_edit.setText(path)

    def _browse_out(self) -> None:
        path = QFileDialog.getExistingDirectory(self, "选择输出文件夹")
        if path:
            self.out_edit.setText(path)

    def run(self) -> None:
        if self.proc is not None and self.proc.state() != QProcess.NotRunning:
            QMessageBox.warning(self, "提示", "预标注正在进行中。")
            return
        model = self.model_combo.currentText().strip().strip('"')
        source = self.source_edit.text().strip().strip('"')
        out = self.out_edit.text().strip().strip('"')
        if not os.path.isfile(model):
            QMessageBox.critical(self, "错误", f"找不到模型:\n{model}")
            return
        if not os.path.isdir(source):
            QMessageBox.critical(self, "错误", f"找不到图片文件夹:\n{source}")
            return

        cmd = [
            os.path.join(APP_DIR, "preannotate_worker.py"),
            "--model", model,
            "--source", source,
            "--conf", str(self.conf.value()),
            "--imgsz", str(self.imgsz.value()),
            "--out", out,
            "--save_txt", "1",
        ]
        self.log.append("\n" + "=" * 60)
        self.log.append(f"开始预标注: {os.path.basename(model)} conf={self.conf.value()}")
        self.log.append(f"图片文件夹: {source}")
        self.log.append("=" * 60)

        self.proc = QProcess(self)
        quiet_process(self.proc)
        self.proc.setProcessChannelMode(QProcess.MergedChannels)
        self.proc.readyReadStandardOutput.connect(self._read_output)
        self.proc.finished.connect(self._on_finished)
        self.proc.start(sys.executable, cmd)

        self.btn_run.setEnabled(False)
        self.btn_stop.setEnabled(True)
        self.main.statusBar().showMessage("预标注中 ...")

    def stop(self) -> None:
        if self.proc is not None and self.proc.state() != QProcess.NotRunning:
            self.proc.kill()
            self.log.append("\n[已发送停止指令]")
            self.main.statusBar().showMessage("正在停止 ...")

    def open_out(self) -> None:
        out = self.out_edit.text().strip().strip('"')
        if os.path.isdir(out):
            os.startfile(out)
        else:
            QMessageBox.information(self, "提示", f"输出文件夹不存在:\n{out}")

    def _read_output(self) -> None:
        assert self.proc is not None
        raw = bytes(self.proc.readAllStandardOutput()).decode("utf-8", errors="replace")
        for line in strip_ansi(raw).split("\n"):
            if line.strip():
                self.log.append(line)

    def _on_finished(self, code: int, _status) -> None:
        self.proc = None
        self.btn_run.setEnabled(True)
        self.btn_stop.setEnabled(False)
        if code == 0:
            self.main.statusBar().showMessage("预标注完成 ✓")
            self.log.append("\n[预标注完成] 标注: <输出>/pre/labels/*.txt  预览: <输出>/pre/*.jpg")
        else:
            self.main.statusBar().showMessage(f"预标注已停止 (退出码 {code})")
            self.log.append(f"\n[进程结束，退出码 {code}]")


class InferTab(QWidget):
    def __init__(self, main_window: "MainWindow") -> None:
        super().__init__()
        self.main = main_window
        self.proc: QProcess | None = None

        root = QVBoxLayout(self)
        group = QGroupBox("推理测试")
        form = QFormLayout(group)

        self.model_combo = QComboBox()
        self.model_combo.setEditable(True)
        self.model_combo.setInsertPolicy(QComboBox.NoInsert)
        for path in scan_models():
            self.model_combo.addItem(path)
        btn_refresh = QPushButton("刷新")
        btn_refresh.setObjectName("ghost")
        btn_refresh.clicked.connect(self.refresh_models)
        mr = QHBoxLayout()
        mr.addWidget(self.model_combo, 1)
        mr.addWidget(btn_refresh)

        self.source_edit = QLineEdit()
        self.source_edit.setPlaceholderText("选择一张图片或一个文件夹")
        sr = make_row(self.source_edit, self._browse_source)
        btn_file = QPushButton("选文件")
        btn_file.setObjectName("ghost")
        btn_file.clicked.connect(self._browse_file)
        sr.addWidget(btn_file)
        self.out_edit = QLineEdit(r"E:\tomato\infer_out")
        self.conf = QDoubleSpinBox()
        self.conf.setRange(0.01, 0.99)
        self.conf.setSingleStep(0.05)
        self.conf.setValue(0.25)
        self.imgsz = QSpinBox()
        self.imgsz.setRange(320, 1920)
        self.imgsz.setSingleStep(32)
        self.imgsz.setValue(640)

        form.addRow("模型文件", mr)
        form.addRow("图片 / 文件夹", sr)
        form.addRow("输出文件夹", make_row(self.out_edit, self._browse_out))
        form.addRow("置信度 conf", self.conf)
        form.addRow("图像尺寸 imgsz", self.imgsz)
        root.addWidget(group)

        buttons = QHBoxLayout()
        self.btn_run = QPushButton("开始推理")
        self.btn_run.clicked.connect(self.run)
        self.btn_stop = QPushButton("停止")
        self.btn_stop.setObjectName("danger")
        self.btn_stop.clicked.connect(self.stop)
        self.btn_stop.setEnabled(False)
        self.btn_open = QPushButton("打开结果")
        self.btn_open.setObjectName("ghost")
        self.btn_open.clicked.connect(self.open_out)
        buttons.addWidget(self.btn_run)
        buttons.addWidget(self.btn_stop)
        buttons.addWidget(self.btn_open)
        buttons.addStretch(1)
        root.addLayout(buttons)

        self.log = LogPane()
        root.addWidget(self.log, 1)

    def refresh_models(self) -> None:
        current = self.model_combo.currentText()
        self.model_combo.clear()
        for path in scan_models():
            self.model_combo.addItem(path)
        if current:
            self.model_combo.setCurrentText(current)

    def _browse_source(self) -> None:
        path = QFileDialog.getExistingDirectory(self, "选择文件夹")
        if path:
            self.source_edit.setText(path)

    def _browse_file(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "选择图片", "", "图片 (*.jpg *.jpeg *.png *.bmp *.webp);;所有文件 (*.*)"
        )
        if path:
            self.source_edit.setText(path)

    def _browse_out(self) -> None:
        path = QFileDialog.getExistingDirectory(self, "选择输出文件夹")
        if path:
            self.out_edit.setText(path)

    def run(self) -> None:
        if self.proc is not None and self.proc.state() != QProcess.NotRunning:
            QMessageBox.warning(self, "提示", "推理正在进行中。")
            return
        model = self.model_combo.currentText().strip().strip('"')
        source = self.source_edit.text().strip().strip('"')
        out = self.out_edit.text().strip().strip('"')
        if not os.path.isfile(model):
            QMessageBox.critical(self, "错误", f"找不到模型:\n{model}")
            return
        if not os.path.exists(source):
            QMessageBox.critical(self, "错误", f"找不到图片或文件夹:\n{source}")
            return

        cmd = [
            os.path.join(APP_DIR, "preannotate_worker.py"),
            "--model", model,
            "--source", source,
            "--conf", str(self.conf.value()),
            "--imgsz", str(self.imgsz.value()),
            "--out", out,
            "--save_txt", "0",
        ]
        self.log.append("\n" + "=" * 60)
        self.log.append(f"开始推理: {os.path.basename(model)} conf={self.conf.value()}")
        self.log.append(f"输入: {source}")
        self.log.append("=" * 60)

        self.proc = QProcess(self)
        quiet_process(self.proc)
        self.proc.setProcessChannelMode(QProcess.MergedChannels)
        self.proc.readyReadStandardOutput.connect(self._read_output)
        self.proc.finished.connect(self._on_finished)
        self.proc.start(sys.executable, cmd)

        self.btn_run.setEnabled(False)
        self.btn_stop.setEnabled(True)
        self.main.statusBar().showMessage("推理中 ...")

    def stop(self) -> None:
        if self.proc is not None and self.proc.state() != QProcess.NotRunning:
            self.proc.kill()
            self.log.append("\n[已发送停止指令]")
            self.main.statusBar().showMessage("正在停止 ...")

    def open_out(self) -> None:
        out = self.out_edit.text().strip().strip('"')
        if os.path.isdir(out):
            os.startfile(out)
        else:
            QMessageBox.information(self, "提示", f"输出文件夹不存在:\n{out}")

    def _read_output(self) -> None:
        assert self.proc is not None
        raw = bytes(self.proc.readAllStandardOutput()).decode("utf-8", errors="replace")
        for line in strip_ansi(raw).split("\n"):
            if line.strip():
                self.log.append(line)

    def _on_finished(self, code: int, _status) -> None:
        self.proc = None
        self.btn_run.setEnabled(True)
        self.btn_stop.setEnabled(False)
        if code == 0:
            self.main.statusBar().showMessage("推理完成 ✓")
            self.log.append("\n[推理完成] 结果保存在输出文件夹")
        else:
            self.main.statusBar().showMessage(f"推理已停止 (退出码 {code})")
            self.log.append(f"\n[进程结束，退出码 {code}]")


class WarnTab(QWidget):
    """监测预警：批量扫描图片，输出健康/病害三级预警与防治建议。"""

    def __init__(self, main_window: "MainWindow") -> None:
        super().__init__()
        self.main = main_window
        self.proc: QProcess | None = None

        root = QVBoxLayout(self)
        group = QGroupBox("监测参数")
        form = QFormLayout(group)

        self.model_combo = QComboBox()
        self.model_combo.setEditable(True)
        self.model_combo.setInsertPolicy(QComboBox.NoInsert)
        for path in scan_models():
            self.model_combo.addItem(path)
        btn_refresh = QPushButton("刷新")
        btn_refresh.setObjectName("ghost")
        btn_refresh.clicked.connect(self.refresh_models)
        mr = QHBoxLayout()
        mr.addWidget(self.model_combo, 1)
        mr.addWidget(btn_refresh)
        btn_browse = QPushButton("浏览")
        btn_browse.setObjectName("ghost")
        btn_browse.clicked.connect(lambda: browse_model(self.model_combo, self))
        mr.addWidget(btn_browse)

        self.source_edit = QLineEdit()
        self.source_edit.setPlaceholderText("选择单张图片或一个图片文件夹")
        sr = make_row(self.source_edit, self._browse_source)
        btn_file = QPushButton("选文件")
        btn_file.setObjectName("ghost")
        btn_file.clicked.connect(self._browse_file)
        sr.addWidget(btn_file)

        self.out_edit = QLineEdit(r"E:\tomato\warn_out")
        self.conf = QDoubleSpinBox()
        self.conf.setRange(0.01, 0.99)
        self.conf.setSingleStep(0.05)
        self.conf.setValue(0.25)
        self.area = QDoubleSpinBox()
        self.area.setRange(0.01, 1.0)
        self.area.setSingleStep(0.05)
        self.area.setValue(0.5)
        self.confirm = QSpinBox()
        self.confirm.setRange(1, 20)
        self.confirm.setValue(1)
        self.imgsz = QSpinBox()
        self.imgsz.setRange(320, 1920)
        self.imgsz.setSingleStep(32)
        self.imgsz.setValue(640)

        form.addRow("模型文件", mr)
        form.addRow("图片 / 文件夹", sr)
        form.addRow("报告文件夹", make_row(self.out_edit, self._browse_out))
        form.addRow("置信度 conf", self.conf)
        form.addRow("患病叶片占比阈值", self.area)
        form.addRow("连续帧确认", self.confirm)
        form.addRow("图片尺寸 imgsz", self.imgsz)
        root.addWidget(group)

        buttons = QHBoxLayout()
        self.btn_run = QPushButton("开始监测")
        self.btn_run.clicked.connect(self.run)
        self.btn_stop = QPushButton("停止")
        self.btn_stop.setObjectName("danger")
        self.btn_stop.clicked.connect(self.stop)
        self.btn_stop.setEnabled(False)
        self.btn_open = QPushButton("打开报告")
        self.btn_open.setObjectName("ghost")
        self.btn_open.clicked.connect(self.open_out)
        buttons.addWidget(self.btn_run)
        buttons.addWidget(self.btn_stop)
        buttons.addWidget(self.btn_open)
        buttons.addStretch(1)
        root.addLayout(buttons)

        self.log = LogPane()
        root.addWidget(self.log, 1)

    def refresh_models(self) -> None:
        current = self.model_combo.currentText()
        self.model_combo.clear()
        for path in scan_models():
            self.model_combo.addItem(path)
        if current:
            self.model_combo.setCurrentText(current)

    def _browse_source(self) -> None:
        path = QFileDialog.getExistingDirectory(self, "选择图片文件夹")
        if path:
            self.source_edit.setText(path)

    def _browse_file(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "选择图片", "", "图片 (*.jpg *.jpeg *.png *.bmp *.webp);;所有文件 (*.*)"
        )
        if path:
            self.source_edit.setText(path)

    def _browse_out(self) -> None:
        path = QFileDialog.getExistingDirectory(self, "选择报告文件夹")
        if path:
            self.out_edit.setText(path)

    def run(self) -> None:
        if self.proc is not None and self.proc.state() != QProcess.NotRunning:
            QMessageBox.warning(self, "提示", "监测任务正在运行中。")
            return
        model = self.model_combo.currentText().strip().strip('"')
        source = self.source_edit.text().strip().strip('"')
        out = self.out_edit.text().strip().strip('"')
        if not os.path.isfile(model):
            QMessageBox.critical(self, "错误", f"找不到模型:\n{model}")
            return
        if not os.path.exists(source):
            QMessageBox.critical(self, "错误", f"找不到图片或文件夹:\n{source}")
            return

        cmd = [
            os.path.join(APP_DIR, "warn_rules.py"),
            "--model", model,
            "--source", source,
            "--out", out,
            "--conf", str(self.conf.value()),
            "--imgsz", str(self.imgsz.value()),
            "--area", str(self.area.value()),
            "--confirm", str(self.confirm.value()),
        ]
        self.log.append("\n" + "=" * 60)
        self.log.append(f"开始监测: {os.path.basename(model)} conf={self.conf.value()} "
                        f"area={self.area.value()} confirm={self.confirm.value()}")
        self.log.append(f"图片源: {source}")
        self.log.append("=" * 60)

        self.proc = QProcess(self)
        quiet_process(self.proc)
        self.proc.setProcessChannelMode(QProcess.MergedChannels)
        self.proc.readyReadStandardOutput.connect(self._read_output)
        self.proc.finished.connect(self._on_finished)
        self.proc.start(sys.executable, cmd)

        self.btn_run.setEnabled(False)
        self.btn_stop.setEnabled(True)
        self.main.statusBar().showMessage("监测中 ...")

    def stop(self) -> None:
        if self.proc is not None and self.proc.state() != QProcess.NotRunning:
            self.proc.kill()
            self.log.append("\n[已发送停止指令]")
            self.main.statusBar().showMessage("正在停止 ...")

    def open_out(self) -> None:
        out = self.out_edit.text().strip().strip('"')
        if os.path.isdir(out):
            os.startfile(out)
        else:
            QMessageBox.information(self, "提示", f"报告文件夹不存在:\n{out}")

    def _read_output(self) -> None:
        assert self.proc is not None
        raw = bytes(self.proc.readAllStandardOutput()).decode("utf-8", errors="replace")
        for line in strip_ansi(raw).split("\n"):
            if line.strip():
                self.log.append(line)

    def _on_finished(self, code: int, _status) -> None:
        self.proc = None
        self.btn_run.setEnabled(True)
        self.btn_stop.setEnabled(False)
        if code == 0:
            self.main.statusBar().showMessage("监测完成")
            self.log.append("\n[监测完成] 报告与标注图已生成，可点击“打开报告”查看")
        else:
            self.main.statusBar().showMessage(f"监测异常停止 (退出码 {code})")
            self.log.append(f"\n[进程结束] 退出码 {code}")


class DatasetGenDialog(QDialog):
    """数据集生成：把图片+标注整理成 YOLO 结构并生成 data.yaml。"""

    def __init__(self, parent, on_done) -> None:
        super().__init__(parent)
        self.on_done = on_done
        self.setWindowTitle("数据集生成")
        self.setMinimumWidth(560)
        form = QFormLayout(self)

        self.images_edit = QLineEdit()
        self.images_edit.setPlaceholderText("存放待整理图片的文件夹")
        self.labels_edit = QLineEdit()
        self.labels_edit.setPlaceholderText("留空则与图片同文件夹")
        self.out_edit = QLineEdit(r"E:\tomato\new_dataset")
        self.train_ratio = QDoubleSpinBox()
        self.train_ratio.setRange(0.0, 1.0)
        self.train_ratio.setSingleStep(0.05)
        self.train_ratio.setValue(0.7)
        self.val_ratio = QDoubleSpinBox()
        self.val_ratio.setRange(0.0, 1.0)
        self.val_ratio.setSingleStep(0.05)
        self.val_ratio.setValue(0.2)
        self.test_ratio = QDoubleSpinBox()
        self.test_ratio.setRange(0.0, 1.0)
        self.test_ratio.setSingleStep(0.05)
        self.test_ratio.setValue(0.1)
        self.names_edit = QLineEdit("early_blight")
        self.seed = QSpinBox()
        self.seed.setRange(0, 999999)
        self.seed.setValue(42)

        form.addRow("图片文件夹", make_row(self.images_edit, self._browse_images))
        form.addRow("标注文件夹", make_row(self.labels_edit, self._browse_labels))
        form.addRow("输出位置", make_row(self.out_edit, self._browse_out))
        form.addRow("训练集占比", self.train_ratio)
        form.addRow("验证集占比", self.val_ratio)
        form.addRow("测试集占比", self.test_ratio)
        tip = QLabel("三者之和应为 1，例如 0.7 / 0.2 / 0.1")
        tip.setObjectName("caption")
        form.addRow("", tip)
        form.addRow("类别名(逗号分隔)", self.names_edit)
        form.addRow("随机种子", self.seed)

        btns = QHBoxLayout()
        ok = QPushButton("生成数据集")
        ok.clicked.connect(self.generate)
        cancel = QPushButton("取消")
        cancel.clicked.connect(self.reject)
        btns.addWidget(ok)
        btns.addWidget(cancel)
        btns.addStretch(1)
        form.addRow(btns)
        self.status = QLabel("")
        self.status.setWordWrap(True)
        form.addRow(self.status)

    def _browse_images(self) -> None:
        path = QFileDialog.getExistingDirectory(self, "选择图片文件夹")
        if path:
            self.images_edit.setText(path)

    def _browse_labels(self) -> None:
        path = QFileDialog.getExistingDirectory(self, "选择标注文件夹")
        if path:
            self.labels_edit.setText(path)

    def _browse_out(self) -> None:
        path = QFileDialog.getExistingDirectory(self, "选择输出位置")
        if path:
            self.out_edit.setText(path)

    def generate(self) -> None:
        images = self.images_edit.text().strip().strip('"')
        labels = self.labels_edit.text().strip().strip('"') or images
        out = self.out_edit.text().strip().strip('"')
        names = [n.strip() for n in self.names_edit.text().split(",") if n.strip()]
        if not os.path.isdir(images):
            self.status.setText("图片文件夹不存在")
            return
        if not os.path.isdir(labels):
            self.status.setText("标注文件夹不存在")
            return
        if not names:
            self.status.setText("请填写类别名")
            return
        tr = self.train_ratio.value()
        vr = self.val_ratio.value()
        tr2 = self.test_ratio.value()
        if abs(tr + vr + tr2 - 1.0) > 0.02:
            self.status.setText("三个占比之和应等于 1（例如 0.7 / 0.2 / 0.1）")
            return

        random.seed(self.seed.value())
        exts = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
        imgs = sorted(f for f in os.listdir(images) if f.lower().endswith(exts))
        valid = []
        for f in imgs:
            stem = os.path.splitext(f)[0]
            if os.path.isfile(os.path.join(labels, stem + ".txt")):
                valid.append(f)
        if not valid:
            self.status.setText("没有找到带同名 .txt 标注的图片")
            return

        random.shuffle(valid)
        total = len(valid)
        n_test = int(round(total * tr2))
        n_val = int(round(total * vr))
        if n_test + n_val > total:
            n_test = max(0, total - n_val)
        test_files = valid[:n_test]
        val_files = valid[n_test:n_test + n_val]
        train_files = [f for f in valid if f not in set(test_files) and f not in set(val_files)]

        for split, files in (("train", train_files), ("val", val_files), ("test", test_files)):
            img_dir = os.path.join(out, "images", split)
            lbl_dir = os.path.join(out, "labels", split)
            os.makedirs(img_dir, exist_ok=True)
            os.makedirs(lbl_dir, exist_ok=True)
            for f in files:
                stem = os.path.splitext(f)[0]
                shutil.copy2(os.path.join(images, f), os.path.join(img_dir, f))
                shutil.copy2(os.path.join(labels, stem + ".txt"), os.path.join(lbl_dir, stem + ".txt"))

        yaml_path = os.path.join(out, "data.yaml")
        with open(yaml_path, "w", encoding="utf-8") as f:
            f.write(f"path: {out.replace(os.sep, '/')}\n")
            f.write("train: images/train\n")
            f.write("val: images/val\n")
            f.write("test: images/test\n")
            f.write(f"nc: {len(names)}\n")
            f.write("names: " + repr(names) + "\n")

        self.status.setText(
            f"完成: 共 {total} 张 -> 训练 {len(train_files)} / 验证 {len(val_files)} / 测试 {len(test_files)}\n已生成 data.yaml"
        )
        self.on_done(yaml_path)


class DatasetBrowseDialog(QDialog):
    """数据集浏览：查看图片和标注框。"""

    def __init__(self, parent, data_yaml: str) -> None:
        super().__init__(parent)
        self.data_yaml = data_yaml
        info = parse_yaml_simple(data_yaml)
        self.base = info["path"] or os.path.dirname(data_yaml)
        self.names = info["names"]
        self.files: list[str] = []
        self.index = 0
        self.split = "train"

        self.setWindowTitle("数据集浏览")
        self.resize(920, 700)
        root = QVBoxLayout(self)

        top = QHBoxLayout()
        top.addWidget(QLabel("数据子集:"))
        self.split_combo = QComboBox()
        for s in ("train", "val", "test"):
            self.split_combo.addItem(s)
        self.split_combo.currentTextChanged.connect(self._load_split)
        top.addWidget(self.split_combo)
        self.count_label = QLabel("")
        top.addWidget(self.count_label)
        top.addStretch(1)
        self.btn_prev = QPushButton("上一张")
        self.btn_prev.setObjectName("ghost")
        self.btn_prev.clicked.connect(lambda: self._step(-1))
        self.btn_next = QPushButton("下一张")
        self.btn_next.setObjectName("ghost")
        self.btn_next.clicked.connect(lambda: self._step(1))
        top.addWidget(self.btn_prev)
        top.addWidget(self.btn_next)
        root.addLayout(top)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        self.image_label = QLabel("加载中 ...")
        self.image_label.setAlignment(Qt.AlignCenter)
        scroll.setWidget(self.image_label)
        root.addWidget(scroll, 1)
        self._load_split()

    def _load_split(self, split: str = "") -> None:
        if split:
            self.split = split
        img_dir = os.path.join(self.base, self.split, "images")
        if not os.path.isdir(img_dir):
            img_dir = os.path.join(self.base, "images", self.split)
        if not os.path.isdir(img_dir):
            self.files = []
            self.count_label.setText("该子集没有图片")
            return
        exts = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
        self.files = sorted(f for f in os.listdir(img_dir) if f.lower().endswith(exts))
        self.index = 0
        self.count_label.setText(f"共 {len(self.files)} 张")
        self._show()

    def _step(self, delta: int) -> None:
        if not self.files:
            return
        self.index = (self.index + delta) % len(self.files)
        self._show()

    def _show(self) -> None:
        if not self.files:
            self.image_label.setText("暂无图片")
            self.image_label.setPixmap(QPixmap())
            return
        img_dir = os.path.join(self.base, self.split, "images")
        if not os.path.isdir(img_dir):
            img_dir = os.path.join(self.base, "images", self.split)
        lbl_dir = os.path.join(self.base, self.split, "labels")
        if not os.path.isdir(lbl_dir):
            lbl_dir = os.path.join(self.base, "labels", self.split)
        name = self.files[self.index]
        img_path = os.path.join(img_dir, name)
        stem = os.path.splitext(name)[0]
        lbl_path = os.path.join(lbl_dir, stem + ".txt")
        pix = QPixmap(img_path)
        if pix.isNull():
            self.image_label.setText(f"无法读取图片: {img_path}")
            return
        if os.path.isfile(lbl_path):
            painter = QPainter(pix)
            pen = QPen(QColor(0, 200, 80), max(2, int(pix.width() / 300)))
            painter.setPen(pen)
            font = painter.font()
            font.setPointSize(max(8, int(pix.width() / 120)))
            painter.setFont(font)
            w, h = pix.width(), pix.height()
            try:
                with open(lbl_path, "r", encoding="utf-8", errors="replace") as f:
                    for line in f:
                        parts = line.split()
                        if len(parts) < 5:
                            continue
                        cls = int(parts[0])
                        x, y, bw, bh = (float(v) for v in parts[1:5])
                        rect = QRectF((x - bw / 2) * w, (y - bh / 2) * h, bw * w, bh * h)
                        painter.drawRect(rect)
                        label = self.names[cls] if cls < len(self.names) else str(cls)
                        painter.drawText(rect.topLeft(), label)
            except OSError:
                pass
            painter.end()
        self.image_label.setPixmap(pix.scaledToWidth(880, Qt.SmoothTransformation))


class WeightsDialog(QDialog):
    """权重配置：浏览/导入/选择模型权重。"""

    def __init__(self, parent, on_select) -> None:
        super().__init__(parent)
        self.on_select = on_select
        self.setWindowTitle("权重配置")
        self.resize(560, 420)
        root = QVBoxLayout(self)

        self.listw = QListWidget()
        root.addWidget(self.listw, 1)
        btns = QHBoxLayout()
        for text, slot in (
            ("设为当前模型", self.select),
            ("导入权重", self.import_weight),
            ("打开所在文件夹", self.open_folder),
            ("刷新", self.refresh),
            ("关闭", self.accept),
        ):
            b = QPushButton(text)
            b.setObjectName("ghost" if text != "设为当前模型" else "")
            b.clicked.connect(slot)
            btns.addWidget(b)
        btns.addStretch(1)
        root.addLayout(btns)
        self.status = QLabel("")
        root.addWidget(self.status)
        self.refresh()

    def refresh(self) -> None:
        self.listw.clear()
        for path in scan_models():
            size = os.path.getsize(path) / 1024 / 1024
            self.listw.addItem(f"{os.path.basename(path)}  ({size:.1f} MB)  {os.path.dirname(path)}")
            item = self.listw.item(self.listw.count() - 1)
            item.setData(Qt.UserRole, path)
        self.status.setText(f"共 {self.listw.count()} 个权重文件")

    def select(self) -> None:
        item = self.listw.currentItem()
        if item:
            self.on_select(item.data(Qt.UserRole))
            self.status.setText("已设为当前模型")

    def import_weight(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "导入权重", "", "模型 (*.pt *.onnx);;所有文件 (*.*)"
        )
        if not path:
            return
        dst_dir = r"E:\tomato\tomato_dataset"
        os.makedirs(dst_dir, exist_ok=True)
        dst = os.path.join(dst_dir, os.path.basename(path))
        shutil.copy2(path, dst)
        self.status.setText(f"已导入: {dst}")
        self.refresh()
        self.on_select(dst)

    def open_folder(self) -> None:
        item = self.listw.currentItem()
        if item:
            folder = os.path.dirname(item.data(Qt.UserRole))
            if os.path.isdir(folder):
                os.startfile(folder)


IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


class DataProcessTab(QWidget):
    def __init__(self, main_window: "MainWindow") -> None:
        super().__init__()
        self.main = main_window
        self.proc: QProcess | None = None

        root = QVBoxLayout(self)

        g1 = QGroupBox("1. 自动数据分类（用训练好的模型把图片按类别归类）")
        f1 = QFormLayout(g1)
        self.c_source = QLineEdit()
        self.c_source.setPlaceholderText("选择存放待分类图片的文件夹")
        self.c_model = QComboBox()
        self.c_model.setEditable(True)
        self.c_model.setInsertPolicy(QComboBox.NoInsert)
        for path in scan_models():
            self.c_model.addItem(path)
        btn_c_refresh = QPushButton("刷新")
        btn_c_refresh.setObjectName("ghost")
        btn_c_refresh.clicked.connect(self._refresh_classify_models)
        mr = QHBoxLayout()
        mr.addWidget(self.c_model, 1)
        mr.addWidget(btn_c_refresh)
        self.c_conf = QDoubleSpinBox()
        self.c_conf.setRange(0.01, 0.99)
        self.c_conf.setSingleStep(0.05)
        self.c_conf.setValue(0.25)
        self.c_imgsz = QSpinBox()
        self.c_imgsz.setRange(320, 1920)
        self.c_imgsz.setSingleStep(32)
        self.c_imgsz.setValue(640)
        f1.addRow("图片文件夹", make_row(self.c_source, self._browse_c_source))
        f1.addRow("分类模型", mr)
        f1.addRow("置信度 conf", self.c_conf)
        f1.addRow("图像尺寸 imgsz", self.c_imgsz)
        btn_classify = QPushButton("开始自动分类")
        btn_classify.clicked.connect(self.run_classify)
        f1.addRow(btn_classify)
        self.c_status = QLabel("输出: <图片文件夹>/自动分类/<类别名>/")
        self.c_status.setObjectName("caption")
        f1.addRow(self.c_status)
        root.addWidget(g1)

        g2 = QGroupBox("2. 批量重命名（分类文件夹内图片：英文/拼音前缀 + 数字）")
        f2 = QFormLayout(g2)
        self.r_folder = QLineEdit()
        self.r_folder.setPlaceholderText("选择要重命名的分类文件夹")
        self.r_prefix = QLineEdit()
        self.r_prefix.setPlaceholderText("留空自动取文件夹名，如 early_blight")
        self.r_start = QSpinBox()
        self.r_start.setRange(0, 99999)
        self.r_start.setValue(1)
        self.r_digits = QSpinBox()
        self.r_digits.setRange(1, 6)
        self.r_digits.setValue(3)
        self.r_sync = QCheckBox("同步重命名同名 .txt 标注文件")
        self.r_sync.setChecked(True)
        f2.addRow("分类文件夹", make_row(self.r_folder, self._browse_r_folder))
        f2.addRow("命名前缀", self.r_prefix)
        f2.addRow("起始数字", self.r_start)
        f2.addRow("数字位数", self.r_digits)
        f2.addRow("", self.r_sync)
        btn_rename = QPushButton("开始重命名")
        btn_rename.clicked.connect(self.run_rename)
        f2.addRow(btn_rename)
        root.addWidget(g2)

        g3 = QGroupBox("3. 数据整合（把分类文件夹汇总到 main，并生成 main-labels 标签）")
        f3 = QFormLayout(g3)
        self.i_project = QLineEdit()
        self.i_project.setPlaceholderText("选择项目文件夹（内含各分类子文件夹）")
        self.i_main = QLineEdit("main")
        self.i_labels = QLineEdit("main-labels")
        f3.addRow("项目文件夹", make_row(self.i_project, self._browse_i_project))
        f3.addRow("主图片文件夹", self.i_main)
        f3.addRow("标签文件夹", self.i_labels)
        btn_integrate = QPushButton("开始整合")
        btn_integrate.clicked.connect(self.run_integrate)
        f3.addRow(btn_integrate)
        root.addWidget(g3)

        self.log = LogPane()
        root.addWidget(self.log, 1)

    def _refresh_classify_models(self) -> None:
        current = self.c_model.currentText()
        self.c_model.clear()
        for path in scan_models():
            self.c_model.addItem(path)
        if current:
            self.c_model.setCurrentText(current)

    def _browse_c_source(self) -> None:
        path = QFileDialog.getExistingDirectory(self, "选择待分类图片文件夹")
        if path:
            self.c_source.setText(path)
            self.c_status.setText(f"输出: {os.path.join(path, '自动分类')}/<类别名>/")

    def _browse_r_folder(self) -> None:
        path = QFileDialog.getExistingDirectory(self, "选择分类文件夹")
        if path:
            self.r_folder.setText(path)
            if not self.r_prefix.text().strip():
                self.r_prefix.setText(re.sub(r"[^a-zA-Z0-9]+", "_", os.path.basename(path)).strip("_"))

    def _browse_i_project(self) -> None:
        path = QFileDialog.getExistingDirectory(self, "选择项目文件夹")
        if path:
            self.i_project.setText(path)

    # ---------- 1. 自动分类 ----------
    def run_classify(self) -> None:
        if self.proc is not None and self.proc.state() != QProcess.NotRunning:
            QMessageBox.warning(self, "提示", "已有任务在运行。")
            return
        source = self.c_source.text().strip().strip('"')
        model = self.c_model.currentText().strip().strip('"')
        if not os.path.isdir(source):
            QMessageBox.critical(self, "错误", f"找不到图片文件夹:\n{source}")
            return
        if not os.path.isfile(model):
            QMessageBox.critical(self, "错误", f"找不到模型:\n{model}")
            return

        cmd = [
            os.path.join(APP_DIR, "data_process_worker.py"),
            "--mode", "classify",
            "--source", source,
            "--model", model,
            "--conf", str(self.c_conf.value()),
            "--imgsz", str(self.c_imgsz.value()),
        ]
        self.log.append("\n" + "=" * 60)
        self.log.append(f"自动分类: {os.path.basename(model)} conf={self.c_conf.value()}")
        self.log.append(f"图片文件夹: {source}")
        self.log.append("=" * 60)
        self.proc = QProcess(self)
        quiet_process(self.proc)
        self.proc.setProcessChannelMode(QProcess.MergedChannels)
        self.proc.readyReadStandardOutput.connect(self._read_output)
        self.proc.finished.connect(self._on_classify_finished)
        self.proc.start(sys.executable, cmd)
        self.main.statusBar().showMessage("自动分类中 ...")

    def _on_classify_finished(self, code: int, _status) -> None:
        self.proc = None
        if code == 0:
            self.main.statusBar().showMessage("自动分类完成 ✓")
            self.log.append("\n[自动分类完成]")
        else:
            self.main.statusBar().showMessage(f"分类已停止 (退出码 {code})")
            self.log.append(f"\n[进程结束，退出码 {code}]")

    # ---------- 2. 批量重命名 ----------
    def run_rename(self) -> None:
        folder = self.r_folder.text().strip().strip('"')
        if not os.path.isdir(folder):
            QMessageBox.critical(self, "错误", f"找不到文件夹:\n{folder}")
            return
        images = sorted(f for f in os.listdir(folder) if f.lower().endswith(IMAGE_EXTS))
        if not images:
            QMessageBox.information(self, "提示", "该文件夹里没有图片。")
            return
        prefix = self.r_prefix.text().strip() or re.sub(
            r"[^a-zA-Z0-9]+", "_", os.path.basename(folder)
        ).strip("_") or "img"
        start = self.r_start.value()
        digits = self.r_digits.value()
        if QMessageBox.question(
            self, "确认", f"将重命名 {len(images)} 个文件（前缀 {prefix}_xxx）\n是否继续？"
        ) != QMessageBox.Yes:
            return

        used = set(os.listdir(folder))
        renamed = 0
        for i, name in enumerate(images):
            stem, ext = os.path.splitext(name)
            new_name = f"{prefix}_{start + i:0{digits}d}{ext.lower()}"
            extra = 1
            while new_name in used:
                new_name = f"{prefix}_{start + i:0{digits}d}_{extra}{ext.lower()}"
                extra += 1
            os.rename(os.path.join(folder, name), os.path.join(folder, new_name))
            used.discard(name)
            used.add(new_name)
            renamed += 1
            if self.r_sync.isChecked():
                lbl = os.path.join(folder, stem + ".txt")
                if os.path.isfile(lbl):
                    new_lbl = os.path.join(folder, os.path.splitext(new_name)[0] + ".txt")
                    os.rename(lbl, new_lbl)
                    used.discard(stem + ".txt")
                    used.add(os.path.basename(new_lbl))
        self.log.append(f"\n[批量重命名] 完成：{renamed} 个文件，示例: {prefix}_{start:0{digits}d}{os.path.splitext(images[0])[1].lower()}")
        self.main.statusBar().showMessage("重命名完成 ✓")

    # ---------- 3. 数据整合 ----------
    def run_integrate(self) -> None:
        project = self.i_project.text().strip().strip('"')
        main_name = self.i_main.text().strip().strip('"') or "main"
        labels_name = self.i_labels.text().strip().strip('"') or "main-labels"
        if not os.path.isdir(project):
            QMessageBox.critical(self, "错误", f"找不到项目文件夹:\n{project}")
            return
        skip = {main_name, labels_name, "自动分类"}
        class_dirs = sorted(
            d for d in os.listdir(project)
            if os.path.isdir(os.path.join(project, d)) and d not in skip
        )
        if not class_dirs:
            QMessageBox.information(self, "提示", "项目文件夹里没有分类子文件夹。")
            return
        if QMessageBox.question(
            self, "确认",
            f"将整合 {len(class_dirs)} 个分类文件夹:\n" + "\n".join(f"  {i}: {n}" for i, n in enumerate(class_dirs))
            + f"\n图片 -> {main_name}，标签 -> {labels_name}\n是否继续？",
        ) != QMessageBox.Yes:
            return

        main_dir = os.path.join(project, main_name)
        labels_dir = os.path.join(project, labels_name)
        os.makedirs(main_dir, exist_ok=True)
        os.makedirs(labels_dir, exist_ok=True)

        used = set(os.listdir(main_dir))
        total = 0
        for idx, cls in enumerate(class_dirs):
            src = os.path.join(project, cls)
            for f in sorted(os.listdir(src)):
                if not f.lower().endswith(IMAGE_EXTS):
                    continue
                stem, ext = os.path.splitext(f)
                dst_name = f
                if dst_name in used:
                    dst_name = f"{cls}_{f}"
                extra = 1
                while dst_name in used:
                    dst_name = f"{cls}_{os.path.splitext(f)[0]}_{extra}{ext.lower()}"
                    extra += 1
                shutil.copy2(os.path.join(src, f), os.path.join(main_dir, dst_name))
                used.add(dst_name)

                src_lbl = os.path.join(src, stem + ".txt")
                new_stem = os.path.splitext(dst_name)[0]
                if os.path.isfile(src_lbl):
                    shutil.copy2(src_lbl, os.path.join(labels_dir, new_stem + ".txt"))
                else:
                    with open(os.path.join(labels_dir, new_stem + ".txt"), "w", encoding="utf-8") as fh:
                        fh.write(f"{idx} 0.5 0.5 1.0 1.0\n")
                total += 1

        with open(os.path.join(labels_dir, "classes.txt"), "w", encoding="utf-8") as fh:
            fh.write("\n".join(class_dirs) + "\n")
        self.log.append(
            f"\n[数据整合] 完成：{total} 张图 -> {main_dir}\n标签 -> {labels_dir}（classes.txt 已生成）"
        )
        self.main.statusBar().showMessage("数据整合完成 ✓")

    def _read_output(self) -> None:
        assert self.proc is not None
        raw = bytes(self.proc.readAllStandardOutput()).decode("utf-8", errors="replace")
        for line in strip_ansi(raw).split("\n"):
            if line.strip():
                self.log.append(line)


class OverviewTab(QWidget):
    def __init__(self, main_window: "MainWindow") -> None:
        super().__init__()
        self.main = main_window
        self.settings = main_window.settings

        root = QVBoxLayout(self)
        title = QLabel("项目概览")
        title.setObjectName("title")
        root.addWidget(title)

        top = QHBoxLayout()
        self.gpu_label = QLabel("硬件环境: 检测中 ...")
        self.gpu_label.setObjectName("caption")
        self.gpu_label.setWordWrap(True)
        top.addWidget(self.gpu_label)
        top.addStretch(1)
        btn_refresh = QPushButton("刷新")
        btn_refresh.setObjectName("ghost")
        btn_refresh.clicked.connect(self.refresh)
        top.addWidget(btn_refresh)
        root.addLayout(top)

        cards = QHBoxLayout()
        ds = QGroupBox("数据集")
        dv = QVBoxLayout(ds)
        self.ds_path = QLabel()
        self.ds_path.setWordWrap(True)
        self.ds_info = QLabel()
        self.ds_info.setWordWrap(True)
        dbtn = QPushButton("打开数据集目录")
        dbtn.setObjectName("ghost")
        dbtn.clicked.connect(self._open_dataset)
        dv.addWidget(self.ds_path)
        dv.addWidget(self.ds_info)
        dv.addWidget(dbtn)
        cards.addWidget(ds, 1)

        md = QGroupBox("当前模型")
        mv = QVBoxLayout(md)
        self.md_path = QLabel()
        self.md_path.setWordWrap(True)
        self.md_info = QLabel()
        self.md_info.setWordWrap(True)
        mbtn = QPushButton("打开模型目录")
        mbtn.setObjectName("ghost")
        mbtn.clicked.connect(self._open_model_dir)
        mv.addWidget(self.md_path)
        mv.addWidget(self.md_info)
        mv.addWidget(mbtn)
        cards.addWidget(md, 1)
        root.addLayout(cards)

        runs = QGroupBox("最近训练 (双击打开结果)")
        rv = QVBoxLayout(runs)
        self.table = QTableWidget(0, 4)
        self.table.setHorizontalHeaderLabels(["训练", "mAP50-95", "轮数", "修改时间"])
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.table.setSelectionBehavior(QTableWidget.SelectRows)
        self.table.doubleClicked.connect(self._open_run)
        rv.addWidget(self.table)
        root.addWidget(runs, 1)

        self._query_gpu()
        self.refresh()

    def _query_gpu(self) -> None:
        proc = QProcess(self)
        quiet_process(proc)
        proc.finished.connect(lambda code, _s, p=proc: self._gpu_done(code, p))
        proc.start(sys.executable, [
            "-c",
            "import torch;print(torch.__version__);print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU only')",
        ])

    def _gpu_done(self, code: int, proc: QProcess) -> None:
        try:
            if code != 0:
                self.gpu_label.setText("硬件环境: 无法读取")
                return
            out = bytes(proc.readAllStandardOutput()).decode("utf-8", errors="replace").strip()
            lines = out.splitlines()
            torch_v = lines[0] if lines else "?"
            gpu = lines[1] if len(lines) > 1 else "?"
            self.gpu_label.setText(f"torch {torch_v}  |  GPU: {gpu}")
        except RuntimeError:
            pass  # 窗口已关闭

    def refresh(self) -> None:
        if not hasattr(self.main, "train_tab"):
            return
        data = self.main.train_tab.data_edit.text().strip().strip('"')
        model = self.main.train_tab.model_combo.currentText().strip().strip('"')
        if os.path.isfile(data):
            self.ds_path.setText(f"<b>{os.path.basename(data)}</b><br>{data}")
            s = dataset_summary(data)
            names = ", ".join(s["names"]) if s["names"] else "?"
            parts = [f"类别: {s['nc'] or '?'} ({names})"]
            for k, (ni, nl) in s["splits"].items():
                if ni:
                    parts.append(f"{k}: {ni}图/{nl}标")
            self.ds_info.setText(" | ".join(parts))
        else:
            self.ds_path.setText("未选择数据集")
            self.ds_info.setText("")
        if os.path.isfile(model):
            size = os.path.getsize(model) / 1024 / 1024
            self.md_path.setText(f"<b>{os.path.basename(model)}</b><br>{model}")
            self.md_info.setText(f"大小: {size:.1f} MB")
        else:
            self.md_path.setText("未选择模型")
            self.md_info.setText("")
        self._refresh_table(data)

    def _refresh_table(self, data: str) -> None:
        self.table.setRowCount(0)
        for folder in run_folders(data)[:10]:
            row = self.table.rowCount()
            self.table.insertRow(row)
            name = os.path.basename(folder)
            metrics = latest_metrics(folder)
            m = QTableWidgetItem(name)
            m.setData(Qt.UserRole, folder)
            m50 = QTableWidgetItem(metrics.get("metrics/mAP50-95(B)", "-"))
            ep = QTableWidgetItem(metrics.get("epoch", "-"))
            t = QTableWidgetItem(datetime.fromtimestamp(os.path.getmtime(folder)).strftime("%m-%d %H:%M"))
            for col, item in enumerate((m, m50, ep, t)):
                self.table.setItem(row, col, item)

    def _open_dataset(self) -> None:
        data = self.main.train_tab.data_edit.text().strip().strip('"')
        folder = os.path.dirname(data) if data else ""
        if os.path.isdir(folder):
            os.startfile(folder)

    def _open_model_dir(self) -> None:
        model = self.main.train_tab.model_combo.currentText().strip().strip('"')
        folder = os.path.dirname(model) if model else ""
        if os.path.isdir(folder):
            os.startfile(folder)

    def _open_run(self, index) -> None:
        item = self.table.item(index.row(), 0)
        if item:
            folder = item.data(Qt.UserRole)
            if folder and os.path.isdir(folder):
                os.startfile(folder)


class ReportTab(QWidget):
    def __init__(self, main_window: "MainWindow") -> None:
        super().__init__()
        self.main = main_window

        root = QVBoxLayout(self)
        title = QLabel("模型报告")
        title.setObjectName("title")
        root.addWidget(title)

        top = QHBoxLayout()
        self.run_combo = QComboBox()
        self.run_combo.setMinimumWidth(360)
        self.run_combo.currentIndexChanged.connect(self.load_run)
        btn_refresh = QPushButton("刷新")
        btn_refresh.setObjectName("ghost")
        btn_refresh.clicked.connect(self.refresh_runs)
        self.btn_open = QPushButton("打开文件夹")
        self.btn_open.setObjectName("ghost")
        self.btn_open.clicked.connect(self.open_run)
        self.btn_export = QPushButton("导出 ONNX")
        self.btn_export.setObjectName("ghost")
        self.btn_export.clicked.connect(self.export_onnx)
        self.btn_infer = QPushButton("用此模型推理")
        self.btn_infer.setObjectName("ghost")
        self.btn_infer.clicked.connect(self.use_for_infer)
        top.addWidget(QLabel("训练记录:"))
        top.addWidget(self.run_combo, 1)
        top.addWidget(btn_refresh)
        top.addWidget(self.btn_open)
        top.addWidget(self.btn_export)
        top.addWidget(self.btn_infer)
        root.addLayout(top)

        metrics_box = QGroupBox("验证指标 (最后一次)")
        mg = QGridLayout(metrics_box)
        self.metric_labels = {}
        names = ["P", "R", "mAP50", "mAP50-95"]
        for i, n in enumerate(names):
            cap = QLabel(n)
            cap.setObjectName("caption")
            cap.setAlignment(Qt.AlignCenter)
            val = QLabel("-")
            val.setObjectName("metric")
            val.setAlignment(Qt.AlignCenter)
            mg.addWidget(cap, 0, i)
            mg.addWidget(val, 1, i)
            self.metric_labels[n] = val
        self.extra_info = QLabel("")
        self.extra_info.setObjectName("caption")
        self.extra_info.setWordWrap(True)
        mg.addWidget(self.extra_info, 2, 0, 1, 4)
        root.addWidget(metrics_box)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        imgs = QWidget()
        iv = QVBoxLayout(imgs)
        self.img_results = QLabel("训练曲线：选择训练记录后显示")
        self.img_results.setAlignment(Qt.AlignTop)
        self.img_conf = QLabel("混淆矩阵：选择训练记录后显示")
        self.img_conf.setAlignment(Qt.AlignTop)
        iv.addWidget(QLabel("训练曲线 results.png"))
        iv.addWidget(self.img_results)
        iv.addWidget(QLabel("混淆矩阵 confusion_matrix.png"))
        iv.addWidget(self.img_conf)
        iv.addStretch(1)
        scroll.setWidget(imgs)
        root.addWidget(scroll, 1)

        self.refresh_runs()

    def refresh_runs(self) -> None:
        data = self.main.train_tab.data_edit.text().strip().strip('"')
        current = self.run_combo.currentData()
        self.run_combo.clear()
        for folder in run_folders(data):
            self.run_combo.addItem(os.path.basename(folder), folder)
        if current is not None:
            idx = self.run_combo.findData(current)
            if idx >= 0:
                self.run_combo.setCurrentIndex(idx)
        self.load_run()

    def load_run(self) -> None:
        folder = self.run_combo.currentData()
        if not folder:
            for lbl in self.metric_labels.values():
                lbl.setText("-")
            self.extra_info.setText("暂无训练记录")
            set_pixmap(self.img_results, "")
            set_pixmap(self.img_conf, "")
            return
        metrics = latest_metrics(folder)
        if metrics:
            self.metric_labels["P"].setText(metrics.get("metrics/precision(B)", "-"))
            self.metric_labels["R"].setText(metrics.get("metrics/recall(B)", "-"))
            self.metric_labels["mAP50"].setText(metrics.get("metrics/mAP50(B)", "-"))
            self.metric_labels["mAP50-95"].setText(metrics.get("metrics/mAP50-95(B)", "-"))
            self.extra_info.setText(
                f"epoch={metrics.get('epoch', '-')}  box_loss={metrics.get('train/box_loss', '-')}  "
                f"cls_loss={metrics.get('train/cls_loss', '-')}  "
                f"best.pt={os.path.join(folder, 'weights', 'best.pt')}"
            )
        else:
            for lbl in self.metric_labels.values():
                lbl.setText("-")
            self.extra_info.setText("该记录没有 results.csv")
        set_pixmap(self.img_results, os.path.join(folder, "results.png"))
        set_pixmap(self.img_conf, os.path.join(folder, "confusion_matrix.png"))

    def open_run(self) -> None:
        folder = self.run_combo.currentData()
        if folder and os.path.isdir(folder):
            os.startfile(folder)

    def export_onnx(self) -> None:
        folder = self.run_combo.currentData()
        best = os.path.join(folder, "weights", "best.pt") if folder else ""
        if os.path.isfile(best):
            self.main.train_tab.export_onnx(best)
            self.main.tabs.setCurrentIndex(1)
        else:
            QMessageBox.information(self, "提示", "该记录没有 best.pt")

    def use_for_infer(self) -> None:
        folder = self.run_combo.currentData()
        best = os.path.join(folder, "weights", "best.pt") if folder else ""
        if os.path.isfile(best):
            self.main.infer_tab.model_combo.setCurrentText(best)
            self.main.tabs.setCurrentIndex(3)
            self.main.statusBar().showMessage("已切换到推理测试，模型已选好")
        else:
            QMessageBox.information(self, "提示", "该记录没有 best.pt")


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.settings = QSettings("TomatoYOLO", "Trainer")
        self.setWindowTitle("cxy的yolo训练平台")
        self.resize(1080, 800)
        icon_path = os.path.join(APP_DIR, "icon.ico")
        if os.path.isfile(icon_path):
            self.setWindowIcon(QIcon(icon_path))

        self.tabs = QTabWidget()
        self.overview = OverviewTab(self)
        self.train_tab = TrainTab(self)
        self.pre_tab = PreannotateTab(self)
        self.infer_tab = InferTab(self)
        self.data_tab = DataProcessTab(self)
        self.report = ReportTab(self)
        self.warn_tab = WarnTab(self)
        self.tabs.addTab(self.overview, "项目概览")
        self.tabs.addTab(self.train_tab, "训练")
        self.tabs.addTab(self.pre_tab, "智能预标注")
        self.tabs.addTab(self.infer_tab, "推理测试")
        self.tabs.addTab(self.data_tab, "数据处理")
        self.tabs.addTab(self.report, "模型报告")
        self.tabs.addTab(self.warn_tab, "监测预警")
        self.setCentralWidget(self.tabs)
        self.overview.refresh()
        self.statusBar().showMessage("就绪")

    def closeEvent(self, event) -> None:
        self.settings.setValue("data", self.train_tab.data_edit.text())
        self.settings.setValue("pre_model", self.pre_tab.model_combo.currentText())
        self.settings.setValue("pre_source", self.pre_tab.source_edit.text())
        self.settings.setValue("pre_out", self.pre_tab.out_edit.text())
        for tab in (self.train_tab, self.pre_tab, self.infer_tab, self.data_tab, self.warn_tab):
            if tab.proc is not None and tab.proc.state() != QProcess.NotRunning:
                tab.proc.kill()
        event.accept()


def main() -> None:
    try:
        app = QApplication(sys.argv)
        app.setStyle("Fusion")
        app.setStyleSheet(STYLE)
        window = MainWindow()
        window.show()
        sys.exit(app.exec())
    except Exception:
        import traceback

        with open(os.path.join(APP_DIR, "startup_error.log"), "w", encoding="utf-8") as f:
            traceback.print_exc(file=f)
        raise


if __name__ == "__main__":
    main()
