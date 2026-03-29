import sys
import pandas as pd
import numpy as np
from PyQt5 import QtWidgets, QtCore
import pyqtgraph as pg
import json
import os
from PyQt5.QtWidgets import QFileDialog, QMessageBox
from PyQt5.QtGui import QPixmap, QPainter

class PlotDeck(QtWidgets.QMainWindow):

    def __init__(self):
        super().__init__()

        self.setWindowTitle("PlotDeck")
        self.resize(1800, 950)

        self.df = None
        self.x_data = None
        self.fft_windows = []
        self.current_plotset_file = None

        main_widget = QtWidgets.QWidget()
        self.setCentralWidget(main_widget)
        main_layout = QtWidgets.QHBoxLayout(main_widget)

        # -----------------------------
        # LEFT SIDE: plots
        # -----------------------------
        plot_container = QtWidgets.QVBoxLayout()
        main_layout.addLayout(plot_container, 4)

        control_bar = QtWidgets.QHBoxLayout()
        plot_container.addLayout(control_bar)

        self.load_button = QtWidgets.QPushButton("Load CSV")
        self.load_button.clicked.connect(self.load_csv)
        control_bar.addWidget(self.load_button)

        self.x_dropdown = QtWidgets.QComboBox()
        self.x_dropdown.currentIndexChanged.connect(self.update_x_axis)
        control_bar.addWidget(self.x_dropdown)

        # Save/load plot set buttons
        self.load_plotset_button = QtWidgets.QPushButton("Load Plot Set")
        self.load_plotset_button.clicked.connect(self.load_plot_set)
        control_bar.addWidget(self.load_plotset_button)

        self.save_plotset_button = QtWidgets.QPushButton("Save Plot Set")
        self.save_plotset_button.clicked.connect(self.save_plot_set)
        control_bar.addWidget(self.save_plotset_button)
        

        # -----------------------------
        # Save all plots as PNG
        # -----------------------------
        self.save_png_button = QtWidgets.QPushButton("Save All Plots as PNG")
        self.save_png_button.clicked.connect(self.save_all_plots_png)
        control_bar.addWidget(self.save_png_button)

        self.plots = []
        for i in range(4):
            plot = pg.PlotWidget()
            plot.showGrid(x=True, y=True)
            plot_container.addWidget(plot)
            if i > 0:
                plot.setXLink(self.plots[0])
            legend = plot.addLegend(labelTextColor='k')
            legend.setBrush(pg.mkBrush(255, 255, 255, 200))  # white, alpha=200 (semi-transparent)
            self.plots.append(plot)

        # -----------------------------
        # RIGHT SIDE: trees + FFT + clear buttons
        # -----------------------------
        right_panel = QtWidgets.QVBoxLayout()
        main_layout.addLayout(right_panel, 1)

        self.trees = []
        self.clear_buttons = []
        self.fft_buttons = []

        for i in range(4):
            label = QtWidgets.QLabel(f"Plot {i+1} Variables")
            right_panel.addWidget(label)

            # FFT button
            fft_btn = QtWidgets.QPushButton("Show FFT of Current Window")
            fft_btn.clicked.connect(lambda _, idx=i: self.plot_fft_button(idx))
            right_panel.addWidget(fft_btn)
            self.fft_buttons.append(fft_btn)

            # Bode button
            bode_btn = QtWidgets.QPushButton("Show Bode of Current Window (Setpoint → Measurement)")
            bode_btn.clicked.connect(lambda _, idx=i: self.plot_bode_button(idx))
            right_panel.addWidget(bode_btn)

            # Clear button
            clear_btn = QtWidgets.QPushButton("Clear Selection")
            clear_btn.clicked.connect(lambda _, idx=i: self.clear_selection(idx))
            right_panel.addWidget(clear_btn)
            self.clear_buttons.append(clear_btn)

            tree = QtWidgets.QTreeWidget()
            tree.setHeaderHidden(True)
            tree.itemChanged.connect(lambda item, col, idx=i: self.update_plots())
            tree.setVerticalScrollMode(QtWidgets.QAbstractItemView.ScrollPerPixel)
            right_panel.addWidget(tree)
            self.trees.append(tree)

        # WASD key support
        self.installEventFilter(self)

        # autoscale when x-range changes
        for plot in self.plots:
            plot.sigXRangeChanged.connect(self.autoscale_y)

    # -----------------------------
    # Save all plots as PNG
    # -----------------------------
    def save_all_plots_png(self):
        if not self.plots:
            return
        
        base_dir = os.path.dirname(os.path.abspath(__file__))
        images_dir = os.path.join(base_dir, "ImageExports")

        path, _ = QFileDialog.getSaveFileName(
            self, "Save Plots as PNG", images_dir, "PNG Files (*.png)"
        )
        if not path:
            return
        if not path.lower().endswith(".png"):
            path += ".png"

        # Grab each plot as QPixmap
        pixmaps = [plot.grab() for plot in self.plots]
        widths = [p.width() for p in pixmaps]
        heights = [p.height() for p in pixmaps]

        # Determine combined image size (2x2 grid)
        total_width = max(widths)
        total_height = sum(heights)

        combined_pixmap = QPixmap(total_width, total_height)
        combined_pixmap.fill(QtCore.Qt.white)
        painter = QPainter(combined_pixmap)

        y_offset = 0
        for pix in pixmaps:
            painter.drawPixmap(0, y_offset, pix)
            y_offset += pix.height()

        painter.end()
        combined_pixmap.save(path)
        QMessageBox.information(self, "Saved", f"All plots saved as:\n{path}")

    # -----------------------------
    # Plot FFT for button
    # -----------------------------
    def plot_fft_button(self, plot_idx):
        plot = self.plots[plot_idx]
        view = plot.getViewBox()
        xmin, xmax = view.viewRange()[0]

        fft_win = pg.GraphicsLayoutWidget(title=f"FFT Plot {plot_idx+1}")
        self.fft_windows.append(fft_win)  # KEEP IT ALIVE
        fft_plot = fft_win.addPlot(title="FFT (Visible X Range)")
        fft_plot.showGrid(x=True, y=True)
        fft_plot.addLegend(labelTextColor='k')
        fft_plot.legend.setBrush(pg.mkBrush(255, 255, 255, 200))  # white, alpha=200 (semi-transparent)

        for curve in plot.listDataItems():
            y = curve.yData
            if y is None:
                continue
            mask = (self.x_data >= xmin) & (self.x_data <= xmax)
            y_visible = y[mask]
            x_visible = self.x_data[mask]

            if len(x_visible) < 2:
                continue

            dt = np.mean(np.diff(x_visible))
            y_fft = np.fft.rfft(y_visible - np.mean(y_visible))
            freqs = np.fft.rfftfreq(len(y_visible), d=dt)
            mag = np.abs(y_fft)

            pen = curve.opts.get('pen', pg.mkPen('r'))
            fft_plot.plot(freqs, mag, pen=pen, name=curve.name())

        fft_plot.setLabel('bottom', 'Frequency', units='Hz')
        fft_plot.setLabel('left', 'Magnitude')
        fft_win.show()

    # -----------------------------
    # Plot bode button
    # -----------------------------
    def plot_bode_button(self, plot_idx):
        plot = self.plots[plot_idx]
        curves = plot.listDataItems()

        if len(curves) < 2:
            QMessageBox.warning(self, "Error", "Need at least 2 plotted signals.")
            return

        # -----------------------------------
        # Prompt user to pick setpoint + measurement
        # -----------------------------------
        names = [c.name() for c in curves if c.name() is not None]

        if len(names) < 2:
            QMessageBox.warning(self, "Error", "Curves must have names.")
            return

        setpoint_name, ok1 = QtWidgets.QInputDialog.getItem(
            self, "Select Setpoint", "Setpoint signal:", names, 0, False
        )
        if not ok1:
            return

        measurement_name, ok2 = QtWidgets.QInputDialog.getItem(
            self, "Select Measurement", "Measurement signal:", names, 1, False
        )
        if not ok2:
            return

        if setpoint_name == measurement_name:
            QMessageBox.warning(self, "Error", "Setpoint and measurement must differ.")
            return

        # Get curves
        setpoint_curve = next(c for c in curves if c.name() == setpoint_name)
        measurement_curve = next(c for c in curves if c.name() == measurement_name)

        # -----------------------------------
        # Extract visible window
        # -----------------------------------
        view = plot.getViewBox()
        xmin, xmax = view.viewRange()[0]
        mask = (self.x_data >= xmin) & (self.x_data <= xmax)

        x = self.x_data[mask]
        u = setpoint_curve.yData[mask]
        y = measurement_curve.yData[mask]

        if len(x) < 64:
            QMessageBox.warning(self, "Error", "Not enough data in visible window.")
            return

        dt = np.mean(np.diff(x))
        fs = 1.0 / dt

        # -----------------------------------
        # Welch parameters
        # -----------------------------------
        nperseg = min(1024, len(x))
        noverlap = nperseg // 2
        step = nperseg - noverlap

        window = np.hanning(nperseg)

        Suu = None
        Syy = None
        Syu = None

        count = 0

        # -----------------------------------
        # Segment loop (Welch averaging)
        # -----------------------------------
        for start in range(0, len(x) - nperseg + 1, step):
            u_seg = u[start:start+nperseg]
            y_seg = y[start:start+nperseg]

            # Detrend
            u_seg = u_seg - np.mean(u_seg)
            y_seg = y_seg - np.mean(y_seg)

            # Window
            u_seg *= window
            y_seg *= window

            # FFT
            U = np.fft.rfft(u_seg)
            Y = np.fft.rfft(y_seg)

            # Spectra
            Suu_seg = U * np.conj(U)
            Syy_seg = Y * np.conj(Y)
            Syu_seg = Y * np.conj(U)

            if Suu is None:
                Suu = Suu_seg
                Syy = Syy_seg
                Syu = Syu_seg
            else:
                Suu += Suu_seg
                Syy += Syy_seg
                Syu += Syu_seg

            count += 1

        if count == 0:
            QMessageBox.warning(self, "Error", "Failed to segment data.")
            return

        # Average
        Suu /= count
        Syy /= count
        Syu /= count

        freqs = np.fft.rfftfreq(nperseg, d=dt)

        # -----------------------------------
        # Transfer function + coherence
        # -----------------------------------
        eps = 1e-12
        H = Syu / (Suu + eps)

        coherence = np.abs(Syu)**2 / (Suu * Syy + eps)

        mag = 20 * np.log10(np.abs(H))

        phase = np.angle(H, deg=True)

        # Force real (critical for pyqtgraph)
        mag = np.real(mag)
        phase = np.real(phase)
        coherence = np.real(coherence)

        # -----------------------------------
        # Mask bad frequencies
        # -----------------------------------
        # Reject low excitation or low coherence
        valid = (np.abs(Suu) > 1e-8) & (coherence > 0.3) & (freqs > 0)

        freqs = freqs[valid]
        mag = mag[valid]
        phase = phase[valid]
        coherence = coherence[valid]

        if len(freqs) < 5:
            QMessageBox.warning(self, "Warning", "Low coherence — results may be unreliable.")
        
        # -----------------------------------
        # Create plots
        # -----------------------------------
        bode_win = pg.GraphicsLayoutWidget(title=f"Bode Plot {plot_idx+1}")
        self.fft_windows.append(bode_win)

        mag_plot = bode_win.addPlot(title="Magnitude (dB)")
        mag_plot.showGrid(x=True, y=True)

        bode_win.nextRow()

        phase_plot = bode_win.addPlot(title="Phase (deg)")
        phase_plot.showGrid(x=True, y=True)

        bode_win.nextRow()

        coh_plot = bode_win.addPlot(title="Coherence")
        coh_plot.showGrid(x=True, y=True)

        # Log frequency axis
        mag_plot.setLogMode(x=True, y=False)
        phase_plot.setLogMode(x=True, y=False)
        coh_plot.setLogMode(x=True, y=False)
        phase_plot.setXLink(mag_plot)
        coh_plot.setXLink(mag_plot)

        # Plot
        mag_plot.plot(freqs, mag, name="Magnitude", pen=pg.mkPen((50, 100, 255), width=2))
        phase_plot.plot(freqs, phase, name="Phase", pen=pg.mkPen((0, 255, 0), width=2))
        coh_plot.plot(freqs, coherence, name="Coherence", pen=pg.mkPen((255, 0, 0), width=2))

        # Labels
        mag_plot.setLabel('left', 'Magnitude', units='dB')
        mag_plot.setLabel('bottom', 'Frequency', units='Hz')

        phase_plot.setLabel('left', 'Phase', units='deg')
        phase_plot.setLabel('bottom', 'Frequency', units='Hz')

        coh_plot.setLabel('left', 'Coherence')
        coh_plot.setLabel('bottom', 'Frequency', units='Hz')

        bode_win.show()

    # -----------------------------
    # Save/Load Plot Sets
    # -----------------------------
    def save_plot_set(self):
        if not self.trees:
            return

        plot_set = {}
        for idx, tree in enumerate(self.trees):
            checked = []
            def collect(item):
                for i in range(item.childCount()):
                    child = item.child(i)
                    if child.childCount() > 0:
                        collect(child)
                    elif child.checkState(0) == QtCore.Qt.Checked:
                        checked.append(child.text(0))
            for i in range(tree.topLevelItemCount()):
                collect(tree.topLevelItem(i))
            plot_set[idx] = checked

        base_dir = os.path.dirname(os.path.abspath(__file__))
        plotsets_dir = os.path.join(base_dir, "PlotSets")

        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Save Plot Set", plotsets_dir, "JSON Files (*.json)")
        if path:
            if not path.lower().endswith(".json"):
                path += ".json"
            with open(path, "w") as f:
                json.dump(plot_set, f, indent=2)

    def load_plot_set(self):
        base_dir = os.path.dirname(os.path.abspath(__file__))
        plotsets_dir = os.path.join(base_dir, "PlotSets")

        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Load Plot Set", plotsets_dir, "JSON Files (*.json)")
        if not path or not os.path.exists(path):
            return

        with open(path, "r") as f:
            plot_set = json.load(f)

        # Uncheck all first
        for tree in self.trees:
            for i in range(tree.topLevelItemCount()):
                tree.topLevelItem(i).setCheckState(0, QtCore.Qt.Unchecked)

        # Check only variables that exist in the current CSV
        if self.df is not None:
            csv_cols = set(self.df.columns)
            for idx, items in plot_set.items():
                tree = self.trees[int(idx)]
                items = [itm for itm in items if itm in csv_cols]
                def collect(item):
                    for i in range(item.childCount()):
                        child = item.child(i)
                        if child.childCount() > 0:
                            collect(child)
                        elif child.text(0) in items:
                            child.setCheckState(0, QtCore.Qt.Checked)
                for i in range(tree.topLevelItemCount()):
                    collect(tree.topLevelItem(i))

        # Update plots
        self.update_plots()

        # Set the current plot set and update window title
        self.current_plotset_file = os.path.splitext(os.path.basename(path))[0]
        self.setWindowTitle(f"Flight Log Viewer - Plot Set: {self.current_plotset_file}")

    # -----------------------------
    # Load CSV
    # -----------------------------
    def load_csv(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Select CSV", "", "CSV Files (*.csv)")

        if not path:
            return

        # store current checked items per tree, filtered by columns in new CSV
        checked_items_per_tree = []
        new_df = pd.read_csv(path)
        new_df.columns = new_df.columns.str.strip()
        csv_cols = set(new_df.columns)

        for tree in self.trees:
            checked = set()
            def collect(item):
                for i in range(item.childCount()):
                    child = item.child(i)
                    if child.childCount() > 0:
                        collect(child)
                    elif child.checkState(0) == QtCore.Qt.Checked and child.text(0) in csv_cols:
                        checked.add(child.text(0))
            for i in range(tree.topLevelItemCount()):
                collect(tree.topLevelItem(i))
            checked_items_per_tree.append(checked)

        self.df = new_df

        # =========================
        # LOAD DERIVED COLUMN DEFS
        # =========================
        try:
            base_dir = os.path.dirname(os.path.abspath(__file__))
            config_path = os.path.join(base_dir, "userDerivedFields.json")

            if os.path.exists(config_path):
                with open(config_path, "r") as f:
                    config = json.load(f)

                derived_list = config.get("derived_data_columns", [])

                # Build safe evaluation context
                # Replace dots with underscores for eval compatibility
                df_eval = self.df.copy()
                col_map = {}
                for col in df_eval.columns:
                    safe_col = col.replace('.', '_')
                    col_map[col] = safe_col
                    df_eval.rename(columns={col: safe_col}, inplace=True)

                for item in derived_list:
                    name = item.get("name")
                    expr = item.get("expression")

                    if not name or not expr:
                        continue

                    try:
                        # Convert expression to safe column names
                        expr_safe = expr
                        for orig, safe in col_map.items():
                            expr_safe = expr_safe.replace(orig, safe)

                        # Evaluate
                        result = df_eval.eval(expr_safe)

                        # Only assign if successful
                        self.df[name] = result

                    except Exception:
                        print("Problem creating user derived data column: ", name)
                        # Gracefully skip bad expressions / missing variables
                        continue

        except Exception:
            pass  # fully silent fail (your call if you want logging)

        # =========================
        # X DROPDOWN
        # =========================
        self.x_dropdown.blockSignals(True)
        self.x_dropdown.clear()

        if "seconds" in self.df.columns:
            self.x_dropdown.addItem("seconds")

        for col in self.df.columns:
            self.x_dropdown.addItem(col)

        self.x_dropdown.blockSignals(False)

        # build hierarchical tree based on arbitrary '.' nesting
        tree_struct = {}
        standalone = []

        for col in self.df.columns:
            if '.' not in col:
                standalone.append(col)
                continue
            parts = col.split('.')
            d = tree_struct
            for p in parts:
                d = d.setdefault(p, {})
            d.setdefault('_leaf', []).append(col)

        def add_items(parent, struct, tree_idx):
            for key, val in struct.items():
                if key == '_leaf':
                    for leaf_name in val:
                        item = QtWidgets.QTreeWidgetItem(parent, [leaf_name])
                        item.setFlags(item.flags() | QtCore.Qt.ItemIsUserCheckable)
                        if leaf_name in checked_items_per_tree[tree_idx]:
                            item.setCheckState(0, QtCore.Qt.Checked)
                        else:
                            item.setCheckState(0, QtCore.Qt.Unchecked)
                else:
                    if list(val.keys()) == ['_leaf'] and len(val['_leaf']) == 1:
                        leaf_name = val['_leaf'][0]
                        item = QtWidgets.QTreeWidgetItem(parent, [leaf_name])
                        item.setFlags(item.flags() | QtCore.Qt.ItemIsUserCheckable)
                        if leaf_name in checked_items_per_tree[tree_idx]:
                            item.setCheckState(0, QtCore.Qt.Checked)
                        else:
                            item.setCheckState(0, QtCore.Qt.Unchecked)
                    else:
                        node = QtWidgets.QTreeWidgetItem(parent, [key])
                        node.setFlags(node.flags() | QtCore.Qt.ItemIsTristate | QtCore.Qt.ItemIsUserCheckable)
                        node.setCheckState(0, QtCore.Qt.Unchecked)
                        add_items(node, val, tree_idx)

        for tree_idx, tree in enumerate(self.trees):
            tree.blockSignals(True)
            tree.clear()
            add_items(tree, tree_struct, tree_idx)
            for col in standalone:
                item = QtWidgets.QTreeWidgetItem(tree, [col])
                item.setFlags(item.flags() | QtCore.Qt.ItemIsUserCheckable)
                if col in checked_items_per_tree[tree_idx]:
                    item.setCheckState(0, QtCore.Qt.Checked)
                else:
                    item.setCheckState(0, QtCore.Qt.Unchecked)
            tree.collapseAll()
            tree.blockSignals(False)

        self.update_x_axis()

        # reset X range
        if self.x_data is not None and len(self.x_data) > 0:
            xmin = np.min(self.x_data)
            xmax = np.max(self.x_data)
            for plot in self.plots:
                plot.setXRange(xmin, xmax, padding=0)

    # -----------------------------
    # Update X variable
    # -----------------------------
    def update_x_axis(self):
        if self.df is None:
            return
        x_name = self.x_dropdown.currentText()
        self.x_data = self.df[x_name].values
        for plot in self.plots:
            plot.setLabel("bottom", x_name)
        self.update_plots()

    # -----------------------------
    # Update plots
    # -----------------------------
    def update_plots(self):
        if self.df is None or self.x_data is None:
            return
    
        # Any modification clears the loaded plot set
        if self.current_plotset_file is not None:
            self.current_plotset_file = None
            self.setWindowTitle("Flight Log Viewer")

        colors = [
            (255,0,0),(0,255,0),(50,100,255),(200,200,0),
            (200,0,200),(0,200,200),(255,100,0),(100,255,0)
        ]

        for plot_idx, tree in enumerate(self.trees):
            plot = self.plots[plot_idx]
            plot.clear()
            legend = plot.addLegend(labelTextColor='k')
            legend.setBrush(pg.mkBrush(255, 255, 255, 200))  # white, alpha=200 (semi-transparent)
            color_idx = 0

            def process_item(item):
                nonlocal color_idx

                if item.childCount() == 0:
                    if item.checkState(0) == QtCore.Qt.Checked:
                        name = item.text(0)
                        if name in self.df.columns:  # protect against missing columns
                            col_data = self.df[name]
                            # Convert TRUE/FALSE to 1/0
                            if col_data.dtype == object:
                                y_series = col_data.astype(str).str.upper().map({'TRUE': 1, 'FALSE': 0})
                                if not y_series.isnull().all():
                                    y = y_series.fillna(0).astype(float).values
                                else:
                                    y = None
                            else:
                                y = col_data.values.astype(float)
                            pen = pg.mkPen(colors[color_idx % len(colors)], width=2)
                            plot.plot(self.x_data, y, pen=pen, name=name)
                            color_idx += 1

                for i in range(item.childCount()):
                    process_item(item.child(i))

            for i in range(tree.topLevelItemCount()):
                process_item(tree.topLevelItem(i))

        self.autoscale_y()

    # -----------------------------
    # Autoscale Y
    # -----------------------------
    def autoscale_y(self):
        if self.x_data is None:
            return
        for plot in self.plots:
            view = plot.getViewBox()
            xmin, xmax = view.viewRange()[0]
            mask = (self.x_data >= xmin) & (self.x_data <= xmax)
            if not np.any(mask):
                continue
            ymin = np.inf
            ymax = -np.inf
            for curve in plot.listDataItems():
                y = curve.yData
                if y is None: continue
                y_visible = y[mask]
                if len(y_visible) == 0: continue
                ymin = min(ymin, np.min(y_visible))
                ymax = max(ymax, np.max(y_visible))
            if ymin < ymax:
                plot.setYRange(ymin, ymax, padding=0.1)

    # -----------------------------
    # Clear selection
    # -----------------------------
    def clear_selection(self, idx):
        tree = self.trees[idx]
        for i in range(tree.topLevelItemCount()):
            item = tree.topLevelItem(i)
            item.setCheckState(0, QtCore.Qt.Unchecked)

    # -----------------------------
    # WASD navigation
    # -----------------------------
    def eventFilter(self, obj, event):
        if event.type() == QtCore.QEvent.KeyPress and self.plots:
            view = self.plots[0].getViewBox()
            xmin, xmax = view.viewRange()[0]
            width = xmax - xmin
            shift = width * 0.1
            if event.key() == QtCore.Qt.Key_A:
                view.setXRange(xmin - shift, xmax - shift, padding=0)
            if event.key() == QtCore.Qt.Key_D:
                view.setXRange(xmin + shift, xmax + shift, padding=0)
            if event.key() == QtCore.Qt.Key_W:
                view.setXRange(xmin + shift, xmax - shift, padding=0)
            if event.key() == QtCore.Qt.Key_S:
                view.setXRange(xmin - shift, xmax + shift, padding=0)
        return super().eventFilter(obj, event)


# -----------------------------
# Run app
# -----------------------------
app = QtWidgets.QApplication(sys.argv)
viewer = PlotDeck()
viewer.show()
sys.exit(app.exec_())