# Bir Pekiştirmeli Öğrenme Yaklaşımı: Makine Kestirimci Bakım Stratejilerinin Optimizasyonu
1. Giriş ve Problem Tanımı

Endüstri 4.0 süreçlerinde, üretim hatlarının duruş sürelerini minimize etmek operasyonel verimlilik için kritiktir. Bu proje, üretim hattında çalışan bir motorun sensör verilerini analiz ederek "ne zaman bakım yapılmalı?" sorusuna yanıt arayan bir karar destek mekanizması sunar. Geleneksel periyodik bakım yerine, makinenin anlık durumuna göre hareket eden Kestirimci Bakım (Predictive Maintenance) stratejisi, Reinforcement Learning (RL) prensipleriyle optimize edilmiştir.

2. Metodoloji ve Ortam Tasarımı

Projede, Markov Karar Süreci (MDP) altyapısı üzerine kurulu bir simülasyon ortamı kullanılmıştır. Model, sensör verilerini sürekli gözlemleyerek sistemin aşınma seviyesini analiz eder.

2.1. Durum Uzayı (State Space)

Sensörlerden gelen karmaşık veriler, mühendislik yaklaşımlarıyla 4 ana duruma indirgenmiştir:

🟢 Yeni (State 0): İdeal çalışma koşulları.

🟡 Normal (State 1): Stabil operasyon süreci.

🟠 Uyarı (State 2): Arıza öncesi sinyaller (Sıcaklık veya tork artışı).

🔴 Kritik (State 3): Yüksek arıza riski.

2.2. Veri Kaynağı

Analizlerde, 10.000 satırlık sentetik ancak gerçekçi değerlerden oluşan AI4I 2020 Predictive Maintenance veri seti kullanılmıştır. Model; sıcaklık farkı, tork ve takım aşınması gibi çoklu parametreleri eş zamanlı olarak değerlendirir.

3. Q-Learning ve Ödül Mekanizması

Ajanın stratejisi, Q-Learning algoritması kullanılarak eğitilmiştir. Modelin "riskten kaçınan" bir tutum sergilemesi için ödül fonksiyonu şu şekilde ölçeklendirilmiştir:
Olay	                Ödül (Reward)	    Açıklama
Başarılı Çalışma	    +50	              Sistemin aktif kalması için teşvik ödülü
Planlı Bakım	        -20	              Kontrollü duruş maliyeti
Makine Arızası	      -1000	            Beklenmedik duruş ve hasar cezası.

4. Deneysel Bulgular ve Analiz

Eğitim sürecinde (25.000 bölüm), ajanın öğrenme eğrisi ve karar mekanizması üzerinde yapılan gözlemler şu sonuçları ortaya koymuştur:

4.1. Öğrenme Eğrisi ve Trend Analizi

Eğitimin başlangıç evrelerinde ajan, makineyi patlatana kadar çalıştırma eğilimi gösterirken (yüksek ceza), ilerleyen evrelerde toplam ödülünü artırmak için stratejik bakımlar gerçekleştirmeye başlamıştır.

4.2. Optimal Politika Çıkarımı

Eğitilmiş Q-Table sonuçlarına göre, ajanın geliştirdiği nihai strateji şöyledir:

Yeni & Normal Durumlar: Kesintisiz operasyon (Devam Et).

Uyarı Durumu: Risk/Kazanç analizi yapılarak operasyona devam.

Kritik Durum: Hemen müdahale (Bakım Yap).

5. Sonuç

Bu çalışma, RL algoritmalarının endüstriyel bakım süreçlerinde maliyet optimizasyonu ve risk yönetimi için güçlü bir araç olabileceğini göstermektedir. Geliştirilen ajan, ağır cezadan (arıza) kaçınmak için küçük maliyetleri (bakım) kabul etmeyi matematiksel olarak öğrenmiştir.

<img width="4470" height="1767" alt="egitim_grafikleri" src="https://github.com/user-attachments/assets/e4a4e660-a8ef-4043-bc3b-ed96906a0de3" />


<img width="437" height="275" alt="image" src="https://github.com/user-attachments/assets/50da327c-7872-4d6a-8a0f-4942f496314b" />

Kurulum ve Çalıştırma
1-)Gerekli kütüphaneleri yükleyin:
pip install numpy pandas matplotlib gymnasium

2-)ai4i2020.csv dosyasının proje dizininde olduğundan emin olun.

3-)Python scriptini çalıştırın:
python maintenance_rl.py

