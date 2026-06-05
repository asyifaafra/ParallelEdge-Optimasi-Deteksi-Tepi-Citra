import sys
import cv2
import numpy as np
import matplotlib.pyplot as plt
import concurrent.futures # Library untuk komputasi paralel
import time # Untuk mengukur waktu eksekusi
from PyQt5 import QtWidgets, uic, QtGui
from PyQt5.QtWidgets import QFileDialog, QMessageBox

class MyGUI(QtWidgets.QMainWindow):
    def __init__(self):
        super(MyGUI, self).__init__()
        # 1. Memuat file UI
        try:
            uic.loadUi('Deteksi-tepi.ui', self)
        except Exception as e:
            print(f"Error: File UI tidak ditemukan! {e}")

        # Inisialisasi variabel
        self.original_image = None
        self.grayscale_image = None

        # 2. Menghubungkan Komponen UI
        self.button_LoadImage = self.findChild(QtWidgets.QPushButton, 'button_LoadImage')
        self.button_Grayscale = self.findChild(QtWidgets.QPushButton, 'button_Grayscale')
        
        # Action Menu Bar
        self.actionSobel = self.findChild(QtWidgets.QAction, 'actionSobel')
        self.actionPrewitt = self.findChild(QtWidgets.QAction, 'actionPrewitt')
        self.actionRoberts = self.findChild(QtWidgets.QAction, 'actionRoberts')
        self.actionSmoothing_Image = self.findChild(QtWidgets.QAction, 'actionSmoothing_Image')

        # 3. Event Listeners
        if self.button_LoadImage:
            self.button_LoadImage.clicked.connect(self.load_image)
        if self.button_Grayscale:
            self.button_Grayscale.clicked.connect(self.convert_to_grayscale)
            
        # Menghubungkan Menu Deteksi Tepi
        if self.actionSobel:
            self.actionSobel.triggered.connect(self.apply_sobel_edge)
        if self.actionPrewitt:
            self.actionPrewitt.triggered.connect(self.apply_prewitt_edge)
        if self.actionRoberts:
            self.actionRoberts.triggered.connect(self.apply_roberts_edge)
        if self.actionSmoothing_Image:
            self.actionSmoothing_Image.triggered.connect(self.apply_dft_smoothing)

    def display_image(self, img, label_name):
        """Menampilkan citra ke QLabel spesifik"""
        label = self.findChild(QtWidgets.QLabel, label_name)
        if label is None:
            return

        if len(img.shape) == 2: # Grayscale
            h, w = img.shape
            q_img = QtGui.QImage(img.data, w, h, w, QtGui.QImage.Format_Grayscale8)
        else: # BGR
            h, w, c = img.shape
            q_img = QtGui.QImage(img.data, w, h, w * c, QtGui.QImage.Format_RGB888).rgbSwapped()
        
        pixmap = QtGui.QPixmap.fromImage(q_img)
        label.setPixmap(pixmap.scaled(label.width(), label.height(), 1))

    def load_image(self):
        options = QFileDialog.Options()
        file_path, _ = QFileDialog.getOpenFileName(self, "Open Image", "", "Images (*.png *.jpg *.jpeg)", options=options)
        if file_path:
            self.original_image = cv2.imread(file_path)
            if self.original_image is not None:
                self.display_image(self.original_image, 'imgLabel')

    def convert_to_grayscale(self):
        """[Algoritma Step 1] Mengubah citra ke Grayscale """
        if self.original_image is not None:
            self.grayscale_image = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2GRAY)
            self.display_image(self.grayscale_image, 'hasilLabel')
        else:
            QMessageBox.warning(self, "Peringatan", "Load Image terlebih dahulu!")

    def edge_detection_core(self, kx, ky, title):
        """Prosedur Inti Deteksi Tepi menggunakan PARALELISASI"""
        if self.grayscale_image is None:
            QMessageBox.warning(self, "Peringatan", "Konversi ke Grayscale dulu!")
            return

        img = self.grayscale_image.astype(np.float32)

        start_time = time.time() # Memulai timer untuk melihat performa

        # --- BLOK KOMPUTASI PARALEL ---
        num_workers = 4 # Menentukan jumlah thread/pekerja 
        
        # 1. DIVIDE: Memecah gambar (matriks) menjadi 4 bagian secara horizontal
        chunks = np.array_split(img, num_workers)

        def process_chunk(chunk):
            """Fungsi pembantu untuk memproses tiap potongan gambar"""
            gx = cv2.filter2D(chunk, -1, kx)
            gy = cv2.filter2D(chunk, -1, ky)
            return np.sqrt((gx**2) + (gy**2))

        # 2. CONQUER: Menjalankan filter ke semua potongan gambar secara BERSAMAAN
        with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
            # executor.map akan mendistribusikan potongan gambar ke tiap worker
            results = list(executor.map(process_chunk, chunks))

        # 3. COMBINE: Menggabungkan kembali potongan gambar yang sudah terdeteksi tepinya
        gradient = np.vstack(results)
        # --- AKHIR BLOK PARALEL ---

        # [Step 7] Normalisasi dalam range 0-255 
        max_val = np.max(gradient)
        if max_val > 0:
            normalized = (gradient / max_val) * 255
        else:
            normalized = gradient
        
        result = normalized.astype(np.uint8)

        end_time = time.time()
        waktu_eksekusi = end_time - start_time
        print(f"[{title}] Selesai diproses paralel dalam {waktu_eksekusi:.4f} detik")

        self.display_image(result, 'edgeLabel')

        # [Step 8] Tampilkan di Matplotlib beserta info waktu
        plt.figure(f"Analisis Deteksi Tepi {title}")
        plt.imshow(result, cmap='gray', interpolation='bicubic')
        plt.title(f'Hasil Operator {title}\n(Waktu Paralel: {waktu_eksekusi:.4f} detik)')
        plt.show()

    def apply_sobel_edge(self):
        """[Step 2 & 3] Inisialisasi Kernel Sobel """
        kx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float32)
        ky = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=np.float32)
        self.edge_detection_core(kx, ky, "Sobel")

    def apply_prewitt_edge(self):
        """Implementasi Kernel Prewitt """
        kx = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]], dtype=np.float32)
        ky = np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]], dtype=np.float32)
        self.edge_detection_core(kx, ky, "Prewitt")

    def apply_roberts_edge(self):
        """Implementasi Kernel Roberts """
        kx = np.array([[1, 0], [0, -1]], dtype=np.float32)
        ky = np.array([[0, 1], [-1, 0]], dtype=np.float32)
        self.edge_detection_core(kx, ky, "Roberts")

    def apply_dft_smoothing(self):
        """Fungsi DFT Smoothing tetap menampilkan hasil ke 'hasilLabel'"""
        if self.grayscale_image is None: return
        img = self.grayscale_image
        dft = cv2.dft(np.float32(img), flags=cv2.DFT_COMPLEX_OUTPUT)
        dft_shift = np.fft.fftshift(dft)
        rows, cols = img.shape
        crow, ccol = int(rows/2), int(cols/2)
        mask = np.zeros((rows, cols, 2), np.uint8)
        mask[crow-30:crow+30, ccol-30:ccol+30] = 1
        fshift = dft_shift * mask
        f_ishift = np.fft.ifftshift(fshift)
        img_back = cv2.idft(f_ishift)
        img_back = cv2.magnitude(img_back[:,:,0], img_back[:,:,1])
        res = cv2.normalize(img_back, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        self.display_image(res, 'hasilLabel')

if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    window = MyGUI()
    window.show()
    sys.exit(app.exec_())