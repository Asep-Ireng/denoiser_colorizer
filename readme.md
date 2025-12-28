Disusun oleh:
Kelompok 3
Reynard Sebastian Hartono (C14230155)
Juan Matthew Davidson (C14230124)
Bryan Alexander Limanto (C14230114)
Satrio Adi Rinekso (C14230112)
Rui Krisna (C14230277)
Dosen Pembimbing: Liliana, S.T., M.Eng., Ph.D.

1. Latar Belakang Masalah

Citra digital rentan terhadap degradasi kualitas, terutama dalam bentuk noise (fluktuasi acak akibat low-light atau keterbatasan sensor) dan degradasi warna (fading, color cast, atau saturasi tidak akurat). Degradasi ini mengurangi nilai informasi dan estetika.

Metode restorasi tradisional seringkali mengorbankan detail (blurring) atau bersifat subjektif dan memakan waktu (restorasi manual). Oleh karena itu, diusulkan sebuah penelitian untuk melakukan analisis investigatif terhadap arsitektur U-Net. Penelitian ini bertujuan untuk menganalisis dampak modifikasi arsitektural U-Net secara sistematis untuk tugas Image Denoising dan Color Restoration secara simultan.2. Tujuan Proyek
Mengembangkan Arsitektur Modifikasi: Mengimplementasikan varian U-Net yang menggabungkan Residual Connections, Attention Gates, dan Cross-Feedback terkontrol untuk image denoising dan color restoration terpadu.
Analisis Komparatif: Membandingkan kinerja model yang diusulkan terhadap model baseline (U-Net standar) menggunakan metrik kuantitatif, yaitu Peak Signal-to-Noise Ratio (PSNR) dan Structural Similarity Index Measure (SSIM).
Studi Ablasi: Menganalisis kontribusi masing-masing modul modifikasi (misalnya: efek penambahan Feature Pyramid Network (FPN) atau Depthwise Separable Convolution) terhadap efisiensi komputasi dan akurasi restorasi model. 3. Ruang Lingkup

Proyek ini berfokus pada implementasi dan studi ablasi terhadap arsitektur U-Net yang dimodifikasi. Modifikasi yang dianalisis meliputi:
Integrasi Residual Block (ResUnet) pada koneksi skip.
Penggunaan mekanisme Attention Gate atau self-attention (Transformer-style) pada jalur fitur.
Implementasi mekanisme Cross-Feedback terkontrol antara decoder Denoising dan Colorization.
Eksplorasi Multi-Scale Feature Fusion (mirip FPN) untuk pemulihan detail.
Penerapan Depthwise Separable Convolution untuk efisiensi model. 4. Arsitektur Model yang Diusulkan
Base: Arsitektur Encoder tunggal dengan Dual Decoder (satu untuk denoise grayscale, satu untuk colorizer RGB) dan Dual Discriminator (grayscale PatchGAN dan color PatchGAN).
Tipe Tugas: Ini adalah tugas Regresi (Regression), bukan klasifikasi, karena model memprediksi nilai piksel (warna dan kecerahan) secara kontinu.
Modifikasi Arsitektur (Prioritas Implementasi)
No.
Modifikasi
Detail Implementasi
Kelebihan & Tujuan
1
Residual Blocks
Mengganti Plain U-Net block dengan Residual Blocks (ResUnet). Menggunakan pre-activation dan 1x1 projection pada shortcut jika terdapat perbedaan jumlah channel.
Memperbaiki aliran gradient (gradient flow) dan menstabilkan proses pelatihan, terutama pada model GAN.
2
Attention Gates
Implementasi Attention Gate pada setiap koneksi skip (gating dari fitur decoder ke encoder). Opsional: penambahan self-attention multi-head pada resolusi fitur 1/8 untuk konteks global.
Memungkinkan model untuk berfokus pada bagian input yang paling relevan (meningkatkan global consistency), yang dapat menghemat sumber daya komputasi.
3
Cross-Feedback Terkontrol
Injeksi fitur warna dari colorizer (dengan stop gradient/detached) ke jalur denoiser melalui gate (kombinasi 1x1 conv + 3x3 conv).
Fitur warna membantu penyempurnaan detail pada gambar grayscale yang telah di-denoise, menghasilkan pinggiran lebih tajam dan mengurangi color bleeding artifact. Harus berhati-hati dalam siklus feedback agar tidak terjadi kebocoran fitur yang merusak.
4
Feature Pyramid Network (FPN)
Multi-scale feature fusion pada 2-3 level resolusi, menggunakan lateral 1x1 conv (64–128 channel) diikuti upsample, concat, dan 3x3 fuse.
Meningkatkan pemulihan detail, terutama pada scene dengan noise tinggi atau objek besar.
5
Depthwise Separable Convolution
Diterapkan secara parsial, hanya pada blok di bagian tengah jaringan (mid-network). Lapisan down-sampling dan up-sampling awal/akhir tetap menggunakan convolution biasa.
Membuat model lebih lightweight dan efisien secara komputasi.

5. Dataset dan Pra-pemrosesan
   Kategori
   Deskripsi
   Dataset Utama
   COCO (Common Objects in Context) sebagai sumber data utama karena variasi scene yang luas.
   Dataset Alternatif
   KITTI (untuk raw color frames), BSD68/DIV2K (untuk sanity check tugas denoising).
   Target Ground Truth (GT)
   Target RGB: Citra asli berwarna (Raw Frames). Target Grayscale: Konversi citra asli ke grayscale menggunakan standar luminance conversion.
   Input Model
   Citra grayscale yang telah ditambahkan noise sintetik (menggunakan distribusi Gaussian Uniform U(5,50) per gambar).

6. Metodologi Penelitian
   Studi Literatur: Mempelajari secara mendalam arsitektur U-Net, Residual Learning, mekanisme Attention, dan Generative Adversarial Networks (GAN) untuk restorasi citra.
   Pengumpulan Data: Mengakuisisi dan memverifikasi dataset citra standar yang akan digunakan untuk pelatihan dan pengujian.
   Pra-pemrosesan Data: Penyesuaian format data, konversi warna, dan pembangkitan noise sintetik yang seragam untuk pelatihan.
   Perancangan Arsitektur: Implementasi model modifikasi U-Net dengan Dual Decoder dan mekanisme Cross-Feedback yang telah ditentukan.
   Pelatihan Model (Training): Melatih model menggunakan Loss Function gabungan yang disesuaikan untuk tugas dual-task (denoising dan colorization).
   Evaluasi dan Analisis: Menguji performa model menggunakan metrik kuantitatif (PSNR/SSIM) dan menjalankan studi ablasi untuk menilai kontribusi setiap modul modifikasi.
7. Pembagian Tugas
   Nama Anggota
   NIM
   Tugas yang Diambil
   Rui Krisna
   C14230277
   Feature Pyramid Network
   Bryan Alexander Limanto
   C14230114
   Residual Blocks
   Satrio Adi Rinekso
   C14230112
   Controlled Cross Feedback
   Reynard Sebastian Hartono
   C14230155
   Depthwise Separable Convolution
   Juan Matthew Davidson
   C14230124
   Attention Gates

Referensi
COCO dataset: torchvision.datasets.CocoDetection
Denoizer: DPIR
Colorizer: DDColor
GAN (pix2pix): SMA-2020 paper 39
Sisa paper untuk colorizer (Perlu ditambahkan link spesifik jika sudah ditemukan)
