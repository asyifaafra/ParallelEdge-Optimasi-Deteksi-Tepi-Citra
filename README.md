# ParallelEdge: Optimasi Deteksi Tepi Citra Digital Berbasis Komputasi Paralel

Proyek ini merupakan pemenuhan tugas **Evaluasi 3** untuk mata kuliah **IFB 206 Komputasi Paralel**, Program Studi **Informatika**, **Institut Teknologi Nasional (Itenas) Bandung**.

* **Dosen Pengampu:** Dr. sc. Lisa Kristiana, ST., MT.
* **Kelas:** AA

---

## 📌 Deskripsi Proyek
`ParallelEdge` adalah sebuah aplikasi berbasis GUI (Graphical User Interface) yang dirancang untuk melakukan pemrosesan citra digital, khususnya dalam mendeteksi tepi objek (*edge detection*) menggunakan berbagai operator spasial (Sobel, Prewitt, dan Roberts) serta fitur *DFT Smoothing*. 

Fokus utama dari proyek ini adalah mengoptimalkan waktu eksekusi pemrosesan konvolusi citra yang biasanya berat menjadi lebih cepat dengan memanfaatkan arsitektur multiprosesor (*multithreading*) pada komputer melalui pustaka `concurrent.futures`.

---

## ⚙️ Arsitektur Komputasi Paralel: Metode *Divide and Conquer*
Aplikasi ini memotong alur pemrosesan sekuensial yang lambat dan menggantinya dengan sistem keroyokan paralel menggunakan pendekatan **Divide, Conquer, and Combine**:

1. **Divide (Membagi Data):** Citra masukan yang telah diubah ke dalam bentuk matriks *grayscale* akan dipecah secara horizontal menjadi 4 potongan bagian gambar (*chunks*) yang sama besar.
2. **Conquer (Memproses Paralel):** Menggunakan `ThreadPoolExecutor` dengan 4 *worker threads*, setiap potongan gambar dikirimkan ke *core* prosesor yang berbeda untuk dihitung nilai gradien konvolusinya secara bersamaan (simultan) di detik yang sama.
3. **Combine (Menyatukan Hasil):** Setelah seluruh *thread* menyelesaikan komputasinya, hasil deteksi tepi dari masing-masing potongan dijahit kembali secara vertikal (`np.vstack`) menjadi satu kesatuan citra utuh yang siap ditampilkan.

Dengan metode ini, beban kerja CPU terdistribusi secara merata, sehingga mempercepat waktu respon aplikasi secara signifikan, terutama saat memproses gambar beresolusi tinggi.

---

## 🚀 Fitur Utama aplikasii
* **Load Image:** Mengimpor citra digital dalam format `.png`, `.jpg`, atau `.jpeg`.
* **Grayscale Conversion:** Mengubah citra BGR menjadi derajat keabuan (8-bit) sebagai syarat awal pemrosesan tepi.
* **Multi-Operator Edge Detection:** * **Operator Sobel:** Menggunakan kernel sensitivitas tinggi terhadap efek *noise*.
  * **Operator Prewitt:** Menggunakan kernel berbasis perhitungan gradien horizontal & vertikal statis.
  * **Operator Roberts:** Menggunakan kernel silang 2x2 untuk komputasi tepi yang cepat.
* **DFT Smoothing:** Reduksi *noise* frekuensi tinggi menggunakan fungsi Discrete Fourier Transform (`cv2.dft`).
* **Performance Analytics:** Menghitung dan menampilkan durasi waktu eksekusi paralel (dalam satuan detik) secara langsung pada visualisasi Matplotlib.

---

## 🛠️ Spesifikasi Teknologi & Library
Aplikasi ini dibangun menggunakan bahasa pemrograman **Python 3** dengan ketergantungan pustaka sebagai berikut:
* **PyQt5:** Untuk membangun *framework* antarmuka grafis (GUI) yang responsif.
* **OpenCV (`cv2`):** Untuk penanganan fungsi dasar citra, konversi warna, filter, dan DFT.
* **NumPy:** Untuk manipulasi matriks tingkat tinggi dan pemecahan data (*chunking*).
* **Matplotlib:** Untuk visualisasi grafik hasil akhir dan analisis performa runtime.
* **Concurrent Futures:** Pustaka bawaan Python untuk manajemen *Thread Pool* komputasi paralel.

---

## 💻 Cara Menjalankan Aplikasi
1. Pastikan Anda telah mengunduh/mengkloning repositori ini.
2. Pastikan pustaka yang diperlukan sudah terinstal. Jika belum, instal via terminal/CMD:
   ```bash
   pip install PyQt5 opencv-python numpy matplotlib