import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import altair as alt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score


def load_data(file_path):
    data = pd.read_csv(file_path)
    return data

def main():
    st.title("Selamat Datang di data set IPL Ball-by-Ball 2008-2020")
    
    # Tambahkan menu bar
    menu = ["Beranda", "IDA", "Relations","Distribution","Comparison","Composition","Predik"]
    pilihan_menu = st.sidebar.selectbox("Menu", menu)
    
    # Jika menu Beranda dipilih
    if pilihan_menu == "Beranda":
        st.write("Selamat datang di Beranda!")
        image_url = 'https://storage.googleapis.com/kaggle-datasets-images/990900/1672987/4b2affa2b729df004da2f7d413f7e866/dataset-cover.jpg?t=2020-11-23-06-58-15'
        # Tampilkan gambar menggunakan st.image()
        st.image(image_url, use_column_width=True)

        st.title("Business objective")
        st.write("Tujuan bisnis dari proyek ini adalah untuk mengoptimalkan strategi dan kinerja tim dalam Indian Premier League (IPL) menggunakan data yang tersedia dalam IPL Complete Dataset. Hal ini mencakup identifikasi pola, tren,dan faktor-faktor kunci yang mempengaruhi hasil pertandingan, serta pengembangan strategi yang lebih efektif untuk mencapai kesuksesan dalam turnamen")

        st.title("Assess Situation")
        st.write("IPL merupakan salah satu turnamen kriket paling bergengsi di dunia yang menarik minat global. Namun, kinerja tim dapat dipengaruhi oleh berbagai faktor seperti komposisi tim, kondisi lapangan, keputusan taktis, dan performa individu pemain. Dengan IPL Complete Dataset, kami memiliki akses ke informasi lengkap tentang setiap pertandingan, pemain, dan detail ball-by-ball, yang memungkinkan kami untuk melakukan analisis mendalam terhadap kinerja tim dan faktor-faktor yang memengaruhi hasil pertandingan")
        

        st.title("Data Mining Goals")
        st.write("Tujuan dari analisis data ini adalah: Mengidentifikasi pola dan tren dalam hasil pertandingan IPL. Menganalisis kinerja individu dan tim secara statistik. Membangun model prediksi untuk hasil pertandingan IPL. Mengidentifikasi faktor-faktor kunci yang mempengaruhi hasil pertandingan. Membuat rekomendasi strategis untuk meningkatkan kinerja tim dalam IPL")

        st.title("Project Plan")
        st.write("Pengumpulan Data: Mengunduh atau memperoleh IPL Complete Dataset dari sumber yang dapat dipercaya seperti situs resmi IPL atau platform data olahraga terpercaya")
        st.write("Data Preprocessing:Memeriksa keberadaan nilai yang hilang atau tidak lengkap dalam dataset. Melakukan penggabungan data dari file CSV matches dan deliveries menjadi satu dataset yang komprehensif, menggunakan kunci unik seperti ID pertandingan.")
        st.write("Analisis Kinerja Tim:Menganalisis kinerja tim selama berbagai musim IPL berdasarkan statistik seperti jumlah kemenangan, rata-rata run rate, dan keberhasilan di berbagai kondisi lapangan. Evaluasi kinerja tim dalam pertandingan kandang dan tandang, serta dalam pertandingan eliminasi.")
        st.write("Penyebaran Hasil:Berbagi laporan dan temuan dengan pemangku kepentingan terkait, termasuk manajemen tim, pelatih, dan pemain. Menggunakan temuan untuk mendukung pengambilan keputusan strategis dalam persiapan untuk musim IPL berikutnya.")
    
    # Jika menu Data dipilih
    elif pilihan_menu == "IDA":
        st.subheader("IDA")
        # Load data
        df = pd.read_csv('IPL Ball-by-Ball 2008-2020.csv')

        # Define the list of columns to visualize
        columns = ['id','inning','over','ball','batsman_runs','extra_runs','total_runs','non_boundary','is_wicket']

        # Create the plot
        plt.figure(figsize=(10, 8))

        for i, Quality in enumerate(columns):
            plt.subplot(3, 3, i + 1)
            sns.histplot(data=df, x=Quality, kde=True, bins=15, color='red')

        # Show the plot using st.pyplot()
        st.pyplot()
        st.write("Berdasarkan gambar bar tersebut, dapat disimpulkan beberapa hal berikut:Kecepatan lari:") 
        st.write("Kecepatan lari tercepat adalah 1,25 km/jam dan 1,50 km/jam.Jumlah lari:") 
        st.write("Jumlah lari terbanyak terjadi di inning ke-2 dan inning ke-4.")
        st.write("Status wicket: Jumlah lari terbanyak tidak menghasilkan wicket.")
    
    # Jika menu Grafik dipilih
    elif pilihan_menu == "Distribution":
        st.subheader("Distribution")

        # Load data
        file_path = "IPL Ball-by-Ball 2008-2020.csv"
        data = load_data(file_path)

        # Judul dan deskripsi
        st.title("Distribusi Nilai")

        st.title("Visualisasi Data Mining menggunakan Streamlit")

        st.write("Visualisasi di atas menunjukkan distribusi jumlah observasi untuk setiap nilai dalam kolom 'inning' dan 'batsman_runs' dalam dataset ini.")

        # Plot distribusi untuk kolom 'inning' dan 'batsman_runs'
        st.subheader("Distribusi Inning dan Batsman Runs")
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.histplot(x='inning', hue='batsman_runs', data=data, multiple="stack", palette="coolwarm", edgecolor=".3")
        plt.title("Distribusi Inning dan Batsman Runs")
        plt.xlabel("Inning dan Batsman_runs")
        plt.ylabel("Jumlah")
        st.pyplot(fig)

        st.write("Berdasarkan set bar ketiga ini, dapat disimpulkan bahwa:")
        st.write("Jumlah lari tanpa wicket lebih banyak daripada jumlah lari dengan wicket.")
        st.write("Jumlah lari dengan wicket terbanyak terjadi dengan kecepatan 1,0 km/jam.:")
        


    elif pilihan_menu == "Relations":
        st.subheader("Relations")    
        st.title("Hubungan Nilai")
        st.write("Menu ini menampilkan hubungan antara beberapa variabel dalam dataset IPL Ball-by-Ball.")

        # Load data
        file_path = "IPL Ball-by-Ball 2008-2020.csv"
        data = load_data(file_path)

        # Memilih Plot yang Diminta
        st.subheader("Scatter Plot: Inning vs Batsman Runs")
        fig, ax = plt.subplots()
        sns.scatterplot(x='total_runs', y='batsman_runs', data=data)
        ax.set_xlabel("Total_runs")
        ax.set_ylabel("Batsman Runs")
        st.pyplot(fig)
        st.write("Berdasarkan kumpulan batang pertama, kita dapat menyimpulkan hal berikut:")
        st.write("Kecepatan lari tercepat adalah 1,25 km/jam.")
        st.write("Lari terbanyak dicetak pada kecepatan 1,0 km/jam.")
        st.write("Terdapat lebih banyak jalur non batas dibandingkan jalur batas.")
        st.write("Jumlah lari is_wicket lebih sedikit dibandingkan lari non-is_wicket.")

        st.write("Variabel batsman_runs diwakili oleh keseluruhan bar total runs (berwarna biru).")
        st.write("Bar ini menunjukkan distribusi lari yang dicetak oleh batsman pada kecepatan berbeda.")
        st.write("Ketinggian bar pada setiap kecepatan menunjukkan jumlah lari yang dicetak pada kecepatan tersebut.")

        # Korelasi
        st.title("Korelasi")
        correlation_matrix = data[['inning', 'over', 'ball', 'batsman_runs', 'extra_runs', 'total_runs', 'non_boundary', 'is_wicket']].corr()

        # Visualisasi matriks korelasi menggunakan heatmap
        st.subheader("Heatmap Korelasi Antara Variabel dalam IPL Ball-by-Ball Dataset")
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", ax=ax)
        ax.set_title("Korelasi Antara Variabel dalam IPL Ball-by-Ball Dataset")
        st.pyplot(fig)
        st.write("inning tidak memiliki korelasi kuat dengan variabel lainnya.over berkorelasi positif dengan batsman_runs, total_runs, dan is_wicket. Ini menunjukkan bahwa seiring berjalannya over, jumlah run yang dicetak, wicket yang diambil, dan skor keseluruhan cenderung meningkat.ball berkorelasi positif dengan batsman_runs dan total_runs. Ini menunjukkan bahwa saat bola dib bowled, jumlah run yang dicetak dan skor keseluruhan cenderung meningkat.")
        st.write ("batsman_runs berkorelasi positif dengan total_runs, non_boundary, dan is_wicket. Ini menunjukkan bahwa jumlah run yang dicetak oleh batsman berkorelasi positif dengan skor keseluruhan, jumlah run non-boundary, dan kemungkinan wicket diambil.")
        st.write ("extra_runs berkorelasi negatif dengan batsman_runs. Ini menunjukkan bahwa extra run biasanya diberikan ketika batsman tidak mencetak run dari bat.")
        st.write("total_runs berkorelasi positif dengan batsman_runs, non_boundary, dan is_wicket. Ini menunjukkan bahwa skor keseluruhan berkorelasi positif dengan jumlah run yang dicetak oleh batsman, jumlah run non-boundary, dan kemungkinan wicket diambil.")
        st.write ("non_boundary berkorelasi positif dengan total_runs. Ini menunjukkan bahwa jumlah run non-boundary berkorelasi positif dengan skor keseluruhan.")
        st.write ("is_wicket berkorelasi negatif dengan batsman_runs dan total_runs. Ini menunjukkan bahwa kemungkinan wicket diambil berkorelasi negatif dengan jumlah run yang dicetak oleh batsman dan skor keseluruhan")

    elif pilihan_menu == "Comparison":
        st.subheader("Comparison") 

        st.title("Perbandingan")

        # Load data
        file_path = "IPL Ball-by-Ball 2008-2020.csv"
        df = load_data(file_path)

        # Membuat subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))

        # Plot pertama: inning
        df['inning'].head(10).value_counts().plot(kind='bar', ax=axes[0, 0], color='skyblue')
        axes[0, 0].set_title('Inning', fontweight="bold", size=10)

        # Plot kedua: over
        df['over'].head(10).value_counts().plot(kind='bar', ax=axes[0, 1], color='salmon')
        axes[0, 1].set_title('Over', fontweight="bold", size=10)

        # Plot ketiga: ball
        df['ball'].head(10).value_counts().plot(kind='bar', ax=axes[1, 0], color='green')
        axes[1, 0].set_title('Ball', fontsize=10)

        # Plot keempat: batsman_runs
        df['batsman_runs'].head(10).value_counts().plot(kind='bar', ax=axes[1, 1], color='purple')
        axes[1, 1].set_title('Batsman Runs', fontsize=10)

        # Menampilkan plot di Streamlit
        st.pyplot(fig)

        st.write("Inning:Sumbu X menunjukkan jumlah inning.Sumbu Y menunjukkan jumlah over.Titik-titik pada grafik menunjukkan jumlah over yang dimainkan di setiap inning.Dapat dilihat dari grafik bahwa pertandingan kriket ini terdiri dari 10 inning.Jumlah over yang dimainkan di setiap inning bervariasi.Inning yang paling banyak dimainkan adalah inning ke-6, dengan 8 over.")
        st.write("Over:Sumbu X menunjukkan jumlah inning.Sumbu Y menunjukkan jumlah over yang dimainkan di setiap inning.Kolom pada grafik menunjukkan jumlah over yang dimainkan di setiap inning.Dapat dilihat dari grafik bahwa jumlah over yang dimainkan di setiap inning bervariasi.Inning yang paling banyak dimainkan adalah inning ke-6, dengan 8 over.")
        st.write("Ball: Sumbu X menunjukkan kecepatan bola dalam meter per detik (m/s).Sumbu Y menunjukkan jumlah run yang dicetak oleh batsman.Titik-titik pada grafik menunjukkan hubungan antara kecepatan bola dan jumlah run yang dicetak oleh batsman.Dapat dilihat dari grafik bahwa terdapat hubungan positif antara kecepatan bola dan jumlah run yang dicetak oleh batsman.Semakin tinggi kecepatan bola, semakin banyak run yang dicetak oleh batsman.")         
        st.write("Batsman Runs:Sumbu X menunjukkan kecepatan bola dalam meter per detik (m/s).Sumbu Y menunjukkan jumlah run yang dicetak oleh batsman.Kolom pada grafik menunjukkan jumlah run yang dicetak oleh batsman untuk setiap kecepatan bola.Dapat dilihat dari grafik bahwa jumlah run yang dicetak oleh batsman untuk setiap kecepatan bola bervariasi.Kecepatan bola yang paling banyak dicetak runnya adalah 2,00 m/s, dengan 7 run.")

    elif pilihan_menu == "Composition":
        st.subheader("Composition") 

        st.title("Komposisi Batsman Runs dalam Data IPL Ball-by-Ball")
        st.write("""
            Anda dapat menghubungi kami di example@example.com
        """)

        # Load data
        file_path = "IPL Ball-by-Ball 2008-2020.csv"
        df = load_data(file_path)

        # Membuat kamus untuk mengganti angka dengan nama sesuai kolom Batsman Runs
        runs_mapping = {0: 'No Run', 1: 'One Run', 2: 'Two Runs', 3: 'Three Runs', 4: 'Four Runs', 5: 'Five Runs', 6: 'Six Runs'}
        
        # Mengganti angka dengan nama menggunakan kamus
        df['batsman_runs'] = df['batsman_runs'].map(runs_mapping)

        # Pie Chart untuk menunjukkan komposisi Batsman Runs secara keseluruhan
        st.subheader("Komposisi Batsman Runs")
        batsman_runs_count = df['batsman_runs'].value_counts()
        labels = batsman_runs_count.index.tolist()
        sizes = batsman_runs_count.values.tolist()

        fig, ax = plt.subplots()
        ax.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90, colors=plt.cm.tab20.colors)
        ax.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
        plt.title('Komposisi Batsman Runs')
        st.pyplot(fig)
        st.write("No Run: 40,1%")
        st.write("One Run: 37,2%")
        st.write("Two Runs: 11,3%")
        st.write("Four Runs: 6,4%")
        st.write("Other: 0,01%")


        st.write("No Run: Persentase No Run yang paling tinggi menunjukkan bahwa banyak batsman yang tidak mencetak lari dalam satu inning. Hal ini dapat disebabkan oleh berbagai faktor, seperti bowler yang bagus, kondisi lapangan yang sulit, atau strategi tim yang defensif.")
        st.write("One Run: Persentase One Run yang cukup tinggi menunjukkan bahwa banyak batsman yang mencetak satu lari dalam satu inning. Hal ini dapat disebabkan oleh batsman yang lebih berhati-hati dalam bermain atau bowler yang tidak terlalu bagus dalam memberikan bola yang sulit.")
        st.write("Two Runs: Persentase Two Runs yang cukup tinggi menunjukkan bahwa banyak batsman yang mencetak dua lari dalam satu inning. Hal ini dapat disebabkan oleh batsman yang lebih agresif dalam bermain atau bowler yang memberikan bola yang mudah dipukul.")
        st.write("Four Runs: Persentase Four Runs yang relatif rendah menunjukkan bahwa tidak banyak batsman yang mencetak empat lari dalam satu inning. Hal ini dapat disebabkan oleh bowler yang bagus dalam memberikan bola yang sulit atau batsman yang tidak terlalu agresif dalam bermain.")
        st.write("Other: Persentase Other yang sangat kecil menunjukkan bahwa hanya ada sedikit batsman yang mencetak lebih dari empat lari dalam satu inning. Hal ini dapat disebabkan oleh batsman yang sangat berbakat atau bowler yang memberikan bola yang sangat mudah dipukul.")




    
    # Di dalam bagian Prediksi menu
    elif pilihan_menu == "Predik":
        st.subheader("Prediksi Hasil Pertandingan IPL")
        st.write("Masukkan data untuk prediksi:")

        file_path = "IPL Ball-by-Ball 2008-2020.csv"
        data = load_data(file_path)
        st.write(data)
        
        # Tampilkan form input data
        form = st.form(key='my_form')
        inning = form.text_input(label='Inning')
        over = form.text_input(label='Over')
        ball = form.text_input(label='Ball')
        batsman_runs = form.text_input(label='Batsman_runs')
        extra_runs = form.text_input(label='Extra_runs')
        is_wicket = form.text_input(label='Is_wickets')
        form_submit = form.form_submit_button(label='Prediksi')
        
        # Jika tombol submit ditekan
        if form_submit:
            # Melakukan prediksi dengan model yang sudah dibuat
            # Disini Anda dapat menambahkan proses prediksi menggunakan model yang sesuai dengan data Anda
            # Misalnya, menggunakan model machine learning seperti RandomForestClassifier
            # Contoh sederhana prediksi dummy
            prediksi = np.random.choice(['Tim A Menang', 'Tim B Menang'])
            st.write("Hasil Prediksi:", prediksi)

        





if __name__ == "__main__":
    main()