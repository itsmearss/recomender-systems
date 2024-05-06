# Laporan Proyek Machine Learning - Annur Riyadhus Solikhin

## Domain Proyek

Dengan ekspansi industri musik digital, lebih dari 400 juta pengguna aktif bulanan tercatat pada platform streaming seperti Spotify pada tahun 2022. Namun, semakin melimpahnya katalog lagu - mencapai lebih dari 70 juta lagu pada tahun tersebut - menciptakan tantangan baru dalam menavigasi dan menemukan konten yang sesuai dengan preferensi pengguna.
Menurut laporan terbaru, pasar streaming musik di Indonesia diyakini telah banyak membantu para seniman untuk memerangi konten bajakan baik fisik maupun digital, sekaligus sebagai penyelamat industri musik yang telah mengalami kerugian sebesar Rp14 triliun karena pelanggaran hak cipta ilegal. <br>
Pada tahun 2020, McKinsey & Company menyebut Indonesia sebagai salah satu dari empat negara paling potensial untuk industri musik digital di Asia Tenggara. Indonesia berkontribusi terhadap 34,7% pasar JOOX, 9,8% pasar Spotify, dan 10,2% pasar SoundCloud di Asia Tenggara. <br>
Menurut survei yang dilakukan oleh beberapa sumber, sekitar 60% pengguna melaporkan kesulitan menemukan lagu-lagu baru yang sesuai dengan selera mereka, sementara 75% mengungkapkan keinginan untuk mendengarkan konten yang lebih bervariasi. Dalam persaingan ketat antara platform streaming, meningkatkan retensi pengguna dan memperluas eksplorasi musik menjadi faktor kritis dalam keberhasilan jangka panjang. <br>
Dalam data internal Spotify, ditemukan bahwa pengguna yang menerima rekomendasi musik yang sesuai dengan preferensi mereka cenderung meningkatkan waktu aktif di platform dan memiliki kecenderungan untuk memperpanjang langganan premium mereka. Oleh karena itu, pengembangan sistem rekomendasi musik yang canggih dan dapat dipersonalisasi menjadi fokus utama. Dengan memanfaatkan data pengguna, preferensi musik historis, dan teknik kecerdasan buatan, tujuannya adalah untuk menciptakan pengalaman mendengarkan yang lebih menyenangkan, memuaskan, dan meningkatkan tingkat keterlibatan pengguna di platform Spotify.

## Business Understanding

### Problem Statements

Menjelaskan pernyataan masalah latar belakang:

- Sistem rekomendasi apa yang baik diterapkan pada kasus ini?
- Bagaimana cara membuat sistem rekomendasi musik di spotify yang akan merekomendasikan buku berdasarkan judulnya?

### Goals

Menjelaskan tujuan dari pernyataan masalah:

- Membuat sistem rekomendasi musik dengan judul musik sebagai fitur.
- Memberikan rekomendasi musik yang mungkin disukai pengguna.

## Data Understanding

Dataset ini, yang dikenal sebagai Spotify Million Song Dataset, merupakan kumpulan data yang mencakup informasi tentang nama lagu, nama artis, tautan ke lagu, dan lirik. Dataset ini memiliki potensi besar untuk digunakan dalam berbagai aplikasi seperti rekomendasi lagu, klasifikasi atau pengelompokan lagu berdasarkan berbagai fitur, serta analisis lebih lanjut terkait preferensi dan tren musik. Dengan informasi yang kaya seperti ini, dataset ini dapat menjadi sumber daya yang berharga untuk proyek-proyek kecerdasan buatan dan analisis data di bidang musik.

| Jenis          | Keterangan                                                                                                    |
| -------------- | ------------------------------------------------------------------------------------------------------------- |
| Sumber         | [Spotify Million Song Dataset](https://www.kaggle.com/datasets/notshrirang/spotify-million-song-dataset/data) |
| Kategori       | Musik                                                                                                         |
| Jenis & Ukuran | csv & 73mb                                                                                                    |

### Variabel-variabel pada Restaurant Stroke Prediction Dataset adalah sebagai berikut:

- song_name: Nama lagu (tipe data: string)
- artist_name: Nama artis yang membuat lagu (tipe data: string)
- link: Tautan atau URL ke lagu (tipe data: string)
- lyrics: Lirik dari lagu tersebut (tipe data: string)

## Data Preparation

### Langkah-langkah Pra Pemrosesan Data

1. Membaca Dataset <br>
   Menggunakan Pandas, dataset dibaca dari file CSV dan dimuat ke dalam DataFrame df. Ini adalah langkah awal untuk mendapatkan akses ke data yang akan diolah. <br>
   | artist | song | link | text |
   |------------|--------------------|------------------------------------------------------------------|---------------------------------------------------------------------------------------------------|
   | ABBA | Ahe's My Kind Of Girl | /a/abba/ahes+my+kind+of+girl_20598417.html | Look at her face, it's a wonderful face \r\nA... |
   | ABBA | Andante, Andante | /a/abba/andante+andante_20002708.html | Take it easy with me, please \r\nTouch me gen... |
   | ABBA | As Good As New | /a/abba/as+good+as+new_20003033.html | I'll never know why I had to go \r\nWhy I had... |
   | ABBA | Bang | /a/abba/bang_20598415.html | Making somebody happy is a question of give an... |
   | ABBA | Bang-A-Boomerang | /a/abba/bang+a+boomerang_20002668.html | Making somebody happy is a question of give an... |

2. Membatasi Jumlah Baris <br>
   Dalam contoh ini, hanya 5000 baris pertama dari dataset yang dipertahankan. Ini dapat berguna jika ingin mengurangi ukuran dataset untuk eksplorasi awal atau jika ada batasan komputasi.
3. Menghapus Kolom "link" <br>
   Menghapus kolom "link" dari DataFrame karena kolom ini mungkin tidak relevan untuk pembuatan model rekomendasi lagu.
4. Ganti Nama Kolom "text" Menjadi "lyrics" <br>
   Mengganti nama kolom "text" menjadi "lyrics" untuk membuatnya lebih deskriptif dan memudahkan pemahaman kontennya.
5. Menghapus duplikat berdasarkan judul lagu <br>
   Menghapus baris yang memiliki nilai yang sama dalam kolom "song". Hal ini dilakukan untuk memastikan bahwa setiap lagu dalam dataset adalah unik dan mencegah duplikasi yang tidak diinginkan.
6. Mereset Indeks DataFrame <br>
   Mereset indeks DataFrame setelah operasi penghapusan duplikat untuk mendapatkan indeks yang disusun ulang secara berurutan. <br>
   | artist | song | lyrics |
   |------------|------------------------|--------------------------------------------------------------------------|
   | ABBA | Ahe's My Kind Of Girl | Look at her face, it's a wonderful face \r\nA... |
   | ABBA | Andante, Andante | Take it easy with me, please \r\nTouch me gen... |
   | ABBA | As Good As New | I'll never know why I had to go \r\nWhy I had... |
   | ABBA | Bang | Making somebody happy is a question of give an... |
   | ABBA | Bang-A-Boomerang | Making somebody happy is a question of give an... |
7. Membuat Kolom Baru "combined_features" <br>
   Membuat kolom baru "combined_features" yang menggabungkan informasi dari kolom "artist", "song", dan "lyrics". Ini akan menjadi dasar untuk perhitungan kemiripan antar lagu. <br>
   | artist | song | lyrics | combined_features |
   |------------|------------------------|--------------------------------------------------------------------------|--------------------------------------------------------------------------|
   | ABBA | Ahe's My Kind Of Girl | Look at her face, it's a wonderful face \r\nA... | ABBA Ahe's My Kind Of Girl Look at her face, i... |
   | ABBA | Andante, Andante | Take it easy with me, please \r\nTouch me gen... | ABBA Andante, Andante Take it easy with me, pl... |
   | ABBA | As Good As New | I'll never know why I had to go \r\nWhy I had... | ABBA As Good As New I'll never know why I had ... |
   | ABBA | Bang | Making somebody happy is a question of give an... | ABBA Bang Making somebody happy is a question ... |
   | ABBA | Bang-A-Boomerang | Making somebody happy is a question of give an... | ABBA Bang-A-Boomerang Making somebody happy is... |

8. Pembersihan Teks Menggunakan Fungsi cleaning <br>
   Mengaplikasikan fungsi cleaning pada setiap nilai dalam kolom "combined_features". Fungsi ini membersihkan teks dengan menghapus karakter non-alphabetic, mengubah teks menjadi huruf kecil, dan menghapus stopwords dalam bahasa Inggris.

## Modeling

Dalam proses pembuatan model sistem rekomendasi berbasis konten untuk lagu-lagu, algoritma yang digunakan umumnya melibatkan beberapa tahap, termasuk pemrosesan teks, vektorisasi, perhitungan kemiripan, dan rekomendasi. Berikut langkah-langkahnya: <br>

- Pemrosesan Teks (Cleaning) <br>
  Pada tahap ini, teks dari lirik lagu diolah untuk membersihkannya dari karakter non-alphabetic, mengubahnya menjadi huruf kecil, dan menghapus kata-kata penghubung atau stopwords.
- Vektorisasi (TF-IDF)<br>
  Teks yang telah dibersihkan kemudian diubah menjadi representasi vektor menggunakan metode TF-IDF (Term Frequency-Inverse Document Frequency).
  TF-IDF memberikan bobot yang lebih tinggi pada kata-kata yang muncul frekuensinya tinggi dalam satu dokumen tetapi jarang muncul dalam seluruh koleksi dokumen.

- Perhitungan Kemiripan (Cosine Similarity)<br>
  Matriks kemiripan kosinus dihitung berdasarkan vektor TF-IDF yang dihasilkan sebelumnya.
  Cosine similarity digunakan untuk mengukur kemiripan antara vektor representasi dua lagu. Nilai cosine similarity mendekati 1 jika dua lagu mirip dan mendekati 0 jika tidak mirip.

- Pemilihan Lagu Referensi<br>
  Pengguna memilih lagu referensi yang akan digunakan sebagai dasar untuk mendapatkan rekomendasi.
  Misalnya, pengguna memilih lagu "Hope."

- Pencarian Lagu-Lagu Mirip<br>
  Mengidentifikasi indeks lagu referensi dalam dataset.
  Membuat daftar kemiripan antara lagu referensi dengan lagu-lagu lain berdasarkan cosine similarity.

- Pengurutan Lagu-Lagu Mirip<br>
  Daftar lagu yang mirip diurutkan berdasarkan nilai cosine similarity secara menurun.
  Lagu dengan kemiripan yang lebih tinggi mendapatkan peringkat yang lebih tinggi.

- Rekomendasi Lagu<br>
  Menghasilkan rekomendasi sepuluh lagu teratas yang memiliki kemiripan tertinggi dengan lagu referensi. <br>
  Hasil rekomendasi lagu dengan kata "Love":
  | results |
  |--------------------------|
  | I Do Love You |
  | Mystery Of Love |
  | I Love Her I Love Her |
  | You Wrote The Book On Love|
  | Love To Love You Baby |
  | Who Do You Love |
  | I Feel Love |
  | This Is My Love |
  | Some Love |
  | Love Is Made Of This |

  Algoritma ini berfokus pada kemiripan konten lirik lagu dalam membuat rekomendasi. Dengan menggunakan vektor TF-IDF dan cosine similarity, sistem dapat mengidentifikasi lagu-lagu yang memiliki konten serupa dengan lagu yang dipilih oleh pengguna, menciptakan pengalaman rekomendasi yang lebih kontekstual dan personal.

## Evaluation

Pada tahap ini akan digunakan Precision untuk mengevaluasi hasil dari rekomendasi pada tabel 8. Precision dapat didefinisikan sebagai berikut: <br>
<math xmlns="http://www.w3.org/1998/Math/MathML">
<mtext>Precision</mtext>
<mo>=</mo>
<mfrac>
<mi>r</mi>
<mi>i</mi>
</mfrac>
</math> <br>
r= total rekomendasi yang relevan <br>
i= jumlah rekomendasi yang diberikan <br>

Acuan similaritas untuk kasus ini adalah, judul lagu yang direkomendasikan memiliki kata yang ditentukan. <br>
Dari hasil rekomendasi pada tabel diatas, diketahui akan meminta rekomendasi lagu dengan judul yang memiliki kata "Love". Dari 10 judul yang direkomendasikan, 10 judul memiliki kategori kata "Love" (similar). Artinya, precision sistem sebesar 10/10 atau 100%.

### Kesimpulan

Sistem rekomendasi musik yang telah diimplementasikan menggunakan pendekatan berbasis konten dengan menerapkan metode TF-IDF dan cosine similarity untuk mengukur kemiripan antara lirik lagu. Proses dimulai dengan menciptakan representasi vektor TF-IDF dari lirik lagu, diikuti oleh perhitungan matriks kemiripan untuk menentukan sejauh mana lagu satu dengan yang lain. Pengguna dapat memilih lagu referensi, dan rekomendasi lagu diberikan berdasarkan kemiripan lirik dengan lagu referensi tersebut. Penting untuk memilih jumlah fitur maksimal dalam TF-IDF dengan hati-hati, dengan percobaan beberapa nilai untuk menemukan keseimbangan antara kompleksitas model dan kinerja yang baik. Evaluasi model dapat dilakukan dengan metrik seperti precision dan recall untuk mengukur seberapa baik sistem memberikan rekomendasi yang sesuai dengan preferensi pengguna. Selain itu, data yang representatif dan ukuran dataset yang memadai menjadi kunci dalam meningkatkan efektivitas sistem rekomendasi. Kesimpulannya, sistem rekomendasi ini memberikan pendekatan berbasis konten yang dapat disesuaikan dengan preferensi individu pengguna, namun perlu perhatian ekstra terhadap penyesuaian hyperparameter dan evaluasi yang cermat untuk meningkatkan kualitas rekomendasi.<br>
**---Ini adalah bagian akhir laporan---**
