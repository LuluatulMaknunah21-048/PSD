import streamlit as st
import pandas as pd
from streamlit_option_menu import option_menu
from sklearn.ensemble import RandomForestClassifier
from pycaret.classification import *
import pickle

selected=option_menu(
    menu_title=None,
    options=['Data', 'Implementasi','Me'],
    default_index=0,
    orientation='horizontal',
    menu_icon=None,
    styles={
    "nav-link":{
        "font-size":"12px",
        "text-align":"center",
        "margin":"5px",
        "--hover-color":"pink",},
    "nav-link-selected":{
        "background-color":"purple"},
    })
if selected=='Data':
    st.title('APLIKASI KLASIFIKASI KELAS JAMUR')
    image_path = "strukturjamur.png"
    
    # Menampilkan gambar dari file lokal
    st.image(image_path, caption="gambar jamur", use_column_width=True, output_format="auto")
    st.write("Data yang digunakan diambil dari website UCI REPOSITORY : https://archive.ics.uci.edu/dataset/73/mushroom. Dataset mushroom digunakan untuk menganalisis faktor-faktor yang membedakan jamur yang dapat dikonsumsi dengan yang beracun berdasarkan berbagai ciri jamur atau karakteristik jamur.")

    st.write("**Deskripsi Dataset:**")
    st.write("1. **Jumlah Sampel:**")
    st.write("   Dataset 'Mushroom' terdiri dari 8124 baris data secara keseluruhan, atau jika berdasarkan kelas 4208 data berkelas e dan 3916 data berkelas p. Jika dilihat dari jumlah data, dataset mushroom datanya seimbang antar label.")

    st.write("2. **Missing Value:**")
    st.write("   Dataset Mushroom memiliki missing value pada Fitur 'Stalk Root (Akar Batang)', missing value pada dataset mushroom ditandai dengan tanda '?'. Untuk menangani nilai yang hilang pada fitur ini, dilakukan pengisian dengan modus dari data yang memiliki label yang sama dengan label pada missing value tersebut. Pendekatan ini dipilih karena dataset ini bersifat kategorikal.")

    st.write("3. **One-Hot Encoding:**")
    st.write("   Semua Fitur pada dataset mushroom bertype data kategorikal, maka dari itu akan dilakukan one-hot encoding pada preprocessing. One-hot encoding merupakan sebuah metode representasi data kategorikal di dalam bentuk yang dapat diolah oleh model pembelajaran mesin.")

    st.write("4. **Fitur-Fitur Pada Dataset**")
    st.write("   Terdapat 23 kolom pada dataset yang terdiri dari 22 Fitur Ciri dari mushroom dan Label mushroom. Berikut rincian dari fitur-fitur tersebut:")
    st.write("   1. **Label(Edible/Poisonous):**")
    st.write("       Label memiliki 2 kelas Menandakan apakah jamur tersebut dapat dikonsumsi (edible/e) atau beracun (poisonous/p).")
    st.write("   4. **Fitur-Fitur Pada Dataset (lanjutan)**")
    st.write("      2. **Cap Shape (Bentuk Tudung):**")
    st.write("         Kolom Cap Shape Menunjukkan bentuk fisik dari tudung jamur. Ada beberapa Bentuk tudung jamur seperti berikut ini:")
    st.write("         - bell=b: Bentuk seperti lonceng.")
    st.write("         - conical=c: Bentuk kerucut.")
    st.write("         - convex=x: Bentuk cembung.")
    st.write("         - flat=f: Bentuk datar.")
    st.write("         - knobbed=k: Bentuk dengan tonjolan.")
    st.write("         - sunken=s: Bentuk cekung.")

    st.write("      3. **Cap Surface (Permukaan Tudung):**")
    st.write("         Kolom Cap surface Menjelaskan tekstur permukaan tudung jamur. Berikut macam-macam Tekstur Permukaan tudung Jamur:")
    st.write("         - fibrous=f: Permukaan serat.")
    st.write("         - grooves=g: Permukaan berlekuk.")
    st.write("         - scaly=y: Permukaan bersisik.")
    st.write("         - smooth=s: Permukaan halus.")

    st.write("      4. **Cap Color (Warna Tudung):**")
    st.write("         Kolom cap color berisi macam-macam warna dari tudung jamur. berikut macam macam warna dari tudung jamur: ")
    st.write("         - brown=n: Coklat.")
    st.write("         - buff=b: Coklat muda.")
    st.write("         - cinnamon=c: Coklat kayu manis.")
    st.write("         - gray=g: Abu-abu.")
    st.write("         - green=r: Hijau.")
    st.write("         - pink=p: Merah muda.")
    st.write("         - purple=u: Ungu.")
    st.write("         - red=e: Merah.")
    st.write("         - white=w: Putih.")
    st.write("         - yellow=y: Kuning.")

    st.write("      5. **Bruises (Memar):**")
    st.write("         Nilai kolom Bruisses Menandakan apakah jamur tersebut akan memar saat disentuh atau tidak, berikut kode nilai yang ada dalam kolom bruisses :")
    st.write("         - bruises=t: Memar.")
    st.write("         - no=f: Tidak memar.")

    st.write("      6. **Odor (Bau):**")
    st.write("         Kolom ini Menunjukkan aroma yang dihasilkan oleh jamur, Berikut merupakan aroma dari jamur")
    st.write("         - almond=a: Aroma almond.")
    st.write("         - anise=l: Aroma adas manis.")
    st.write("         - creosote=c: Aroma creosote.")
    st.write("         - fishy=y: Aroma ikan.")
    st.write("         - foul=f: Aroma busuk.")
    st.write("         - musty=m: Aroma berjamur.")
    st.write("         - none=n: Tidak ada aroma.")
    st.write("         - pungent=p: Aroma tajam.")
    st.write("         - spicy=s: Aroma pedas.")

    st.write("      7. **Gill Attachment (Lampiran Gill):**")
    st.write("         Kolom Gill Attachment Menunjukkan bagaimana hubungan gill dengan batang jamur. Adapun hubungan nilai hubungannya seperti berikut :")
    st.write("         - attached=a: Melekat.")
    st.write("         - descending=d: Menurun.")
    st.write("         - free=f: Bebas.")
    st.write("         - notched=n: Tidak rata.")

    st.write("      8. **Gill Spacing (Jarak Gill):**")
    st.write("         Kolom ini Menunjukkan seberapa rapat gill jamur taraf ukurnya yaitu rapat, padat dan distant.")
    st.write("         - close=c: Rapat.")
    st.write("         - crowded=w: Padat.")
    st.write("         - distant=d: Jauh.")

    st.write("      9. **Gill Size (Ukuran Gill):**")
    st.write("         Kolom ini Menunjukkan ukuran gill jamur apakah termasuk ke kategori lebar atau sempit.")
    st.write("         - broad=b: Lebar.")
    st.write("         - narrow=n: Sempit.")

    st.write("      10. **Gill Color (Warna Gill):**")
    st.write("          Kolom ini Menunjukkan warna dari gill jamur. berikut warna-warna dari gill jamur : ")
    st.write("          - black=k: Hitam.")
    st.write("          - brown=n: Coklat.")
    st.write("          - buff=b: Coklat muda.")
    st.write("          - chocolate=h: Coklat tua.")
    st.write("          - gray=g: Abu-abu.")
    st.write("          - green=r: Hijau.")
    st.write("          - orange=o: Oranye.")
    st.write("          - pink=p: Merah muda.")
    st.write("          - purple=u: Ungu.")
    st.write("          - red=e: Merah.")
    st.write("          - white=w: Putih.")
    st.write("          - yellow=y: Kuning.")

    st.write("      11. **Stalk Shape (Bentuk Batang):**")
    st.write("          Kolom ini Menunjukkan apakah batang jamur membesar atau menyempit.")
    st.write("          - enlarging=e: Membesar.")
    st.write("          - tapering=t: Menyempit.")

    st.write("      12. **Stalk Root (Akar Batang):**")
    st.write("          Kolom ini Menjelaskan jenis akar batang jamur.")
    st.write("          - bulbous=b: Berbentuk bulat.")
    st.write("          - club=c: Berbentuk klub.")
    st.write("          - cup=u: Berbentuk cangkir.")
    st.write("          - equal=e: Sama panjang dengan batang.")
    st.write("          - rhizomorphs=z: Berbentuk seperti rizom.")
    st.write("          - rooted=r: Mempunyai akar.")
    st.write("          - missing=?: Tidak diketahui.")

    st.write("      13. **Stalk Surface Above Ring (Permukaan Batang di Atas Cincin):**")
    st.write("          Kolom ini Menunjukkan tekstur permukaan batang di atas cincin jamur.")
    st.write("          - fibrous=f: Permukaan serat.")
    st.write("          - scaly=y: Permukaan bersisik.")
    st.write("          - silky=k: Permukaan berkilau.")
    st.write("          - smooth=s: Permukaan halus.")

    st.write("      14. **Stalk Surface Below Ring (Permukaan Batang di Bawah Cincin):**")
    st.write("          Kolom ini Menunjukkan tekstur permukaan batang di bawah cincin jamur.")
    st.write("          - fibrous=f: Permukaan serat.")
    st.write("          - scaly=y: Permukaan bersisik.")
    st.write("          - silky=k: Permukaan berkilau.")
    st.write("          - smooth=s: Permukaan halus.")

    st.write("      15. **Stalk Color Above Ring (Warna Batang di Atas Cincin):**")
    st.write("          Kolom ini Menunjukkan warna batang di atas cincin jamur.")
    st.write("          - brown=n: Coklat.")
    st.write("          - buff=b: Coklat muda.")
    st.write("          - cinnamon=c: Coklat kayu manis.")
    st.write("          - gray=g: Abu-abu.")
    st.write("          - orange=o: Oranye.")
    st.write("          - pink=p: Merah muda.")
    st.write("          - red=e: Merah.")
    st.write("          - white=w: Putih.")
    st.write("          - yellow=y: Kuning.")

    st.write("      16. **Stalk Color Below Ring (Warna Batang di Bawah Cincin):**")
    st.write("          Kolom ini Menunjukkan warna batang di bawah cincin jamur.")
    st.write("          - brown=n: Coklat.")
    st.write("          - buff=b: Coklat muda.")
    st.write("          - cinnamon=c: Coklat kayu manis.")
    st.write("          - gray=g: Abu-abu.")
    st.write("          - orange=o: Oranye.")
    st.write("          - pink=p: Merah muda.")
    st.write("          - red=e: Merah.")
    st.write("          - white=w: Putih.")
    st.write("          - yellow=y: Kuning.")

    st.write("      17. **Veil Type (Tipe Tutup):**")
    st.write("          Kolom ini Menunjukkan tipe tutup (veil) jamur.")
    st.write("          - partial=p: Sebagian.")
    st.write("          - universal=u: Universal (menutup seluruhnya).")

    st.write("      18. **Veil Color (Warna Tutup):**")
    st.write("          Kolom ini Menunjukkan warna tutup jamur.")
    st.write("          - brown=n: Coklat.")
    st.write("          - orange=o: Oranye.")
    st.write("          - white=w: Putih.")
    st.write("          - yellow=y: Kuning.")

    st.write("      19. **Ring Number (Jumlah Cincin):**")
    st.write("          Kolom ini Menunjukkan jumlah cincin pada jamur.")
    st.write("          - none=n: Tidak ada.")
    st.write("          - one=o: Satu.")
    st.write("          - two=t: Dua.")

    st.write("      20. **Ring Type (Tipe Cincin):**")
    st.write("          Kolom ini Menunjukkan jenis cincin pada jamur.")
    st.write("          - cobwebby=c: Berbentuk seperti jaring laba-laba.")
    st.write("          - evanescent=e: Cepat menghilang.")
    st.write("          - flaring=f: Membentuk bentuk seperti terompet.")
    st.write("          - large=l: Besar.")
    st.write("          - none=n: Tidak ada.")
    st.write("          - pendant=p: Membentuk gantungan.")
    st.write("          - sheathing=s: Membentuk lapisan.")
    st.write("          - zone=z: Berbentuk zona.")

    st.write("      21. **Spore Print Color (Warna Spora):**")
    st.write("          Kolom ini Menunjukkan warna spora yang dihasilkan oleh jamur.")
    st.write("          - black=k: Hitam.")
    st.write("          - brown=n: Coklat.")
    st.write("          - buff=b: Coklat muda.")
    st.write("          - chocolate=h: Coklat tua.")
    st.write("          - green=r: Hijau.")
    st.write("          - orange=o: Oranye.")
    st.write("          - purple=u: Ungu.")
    st.write("          - white=w: Putih.")
    st.write("          - yellow=y: Kuning.")

    st.write("      22. **Population (Populasi):**")
    st.write("          Kolom ini Menunjukkan sebaran populasi jamur di suatu tempat.")
    st.write("          - abundant=a: Melimpah.")
    st.write("          - clustered=c: Berkumpul.")
    st.write("          - numerous=n: Banyak.")
    st.write("          - scattered=s: Tersebar.")
    st.write("          - several=v: Beberapa.")
    st.write("          - solitary=y: Tunggal.")

    st.write("      23. **Habitat (Habitat):**")
    st.write("          Kolom ini Menunjukkan habitat tempat tumbuhnya jamur, adapun tempat-tempat tumbuhnya jamur menurut deskripsi dataset seperti berikut : ")
    st.write("          - grasses=g: Di atas rumput.")
    st.write("          - leaves=l: Di atas daun.")
    st.write("          - meadows=m: Di padang rumput.")
    st.write("          - paths=p: Di jalur.")
    st.write("          - urban=u: Di perkotaan.")
    st.write("          - waste=w: Di tempat pembuangan.")
    st.write("          - woods=d: Di hutan.")



if selected=='Implementasi':
    kolom=['cap-shape_b','cap-shape_c','cap-shape_f','cap-shape_k','cap-shape_s','cap-shape_x','cap-surface_f','cap-surface_g','cap-surface_s','cap-surface_y','cap-color_b','cap-color_c','cap-color_e','cap-color_g','cap-color_n','cap-color_p','cap-color_r','cap-color_u','cap-color_w','cap-color_y','bruises_f','bruises_t','odor_a','odor_c','odor_f','odor_l','odor_m','odor_n','odor_p','odor_s','odor_y','gill-attachment_a','gill-attachment_f','gill-spacing_c','gill-spacing_w','gill-size_b','gill-size_n','gill-color_b','gill-color_e','gill-color_g','gill-color_h','gill-color_k','gill-color_n','gill-color_o','gill-color_p','gill-color_r','gill-color_u','gill-color_w','gill-color_y','stalk-shape_e','stalk-shape_t','stalk-root_b','stalk-root_c','stalk-root_e','stalk-root_r','stalk-surface-above-ring_f','stalk-surface-above-ring_k','stalk-surface-above-ring_s','stalk-surface-above-ring_y','stalk-surface-below-ring_f','stalk-surface-below-ring_k','stalk-surface-below-ring_s','stalk-surface-below-ring_y','stalk-color-above-ring_b','stalk-color-above-ring_c','stalk-color-above-ring_e','stalk-color-above-ring_g','stalk-color-above-ring_n','stalk-color-above-ring_o','stalk-color-above-ring_p','stalk-color-above-ring_w','stalk-color-above-ring_y','stalk-color-below-ring_b','stalk-color-below-ring_c','stalk-color-below-ring_e','stalk-color-below-ring_g','stalk-color-below-ring_n','stalk-color-below-ring_o','stalk-color-below-ring_p','stalk-color-below-ring_w','stalk-color-below-ring_y','veil-type_p','veil-color_n','veil-color_o','veil-color_w','veil-color_y','ring-number_n','ring-number_o','ring-number_t','ring-type_e','ring-type_f','ring-type_l','ring-type_n','ring-type_p','spore-print-color_b','spore-print-color_h','spore-print-color_k','spore-print-color_n','spore-print-color_o','spore-print-color_r','spore-print-color_u','spore-print-color_w','spore-print-color_y','population_a','population_c','population_n','population_s','population_v','population_y','habitat_d','habitat_g','habitat_l','habitat_m','habitat_p','habitat_u','habitat_w']

    st.title('APLIKASI KLASIFIKASI KELAS JAMUR')
    col1,col2=st.columns(2)

    df = pd.DataFrame(data=[[0]*len(kolom)], columns=kolom)

    capshape=['Silahkan Pilih','bell','conical','convex','flat','knobbed','sunken']
    capsurface=['Silahkan Pilih','fibrous','grooves','scaly','smooth']
    capcolor=['Silahkan Pilih','brown','buff','cinnamon','gray','green','pink','purple','red','white','yellow']
    bruises=['Silahkan Pilih','bruises','no']
    odor=['Silahkan Pilih','almond','anise','creosote','fishy','foul','musty','none','pungent','spicy']
    gill_attachment=['Silahkan Pilih','attached','free']
    gill_spacing=['Silahkan Pilih','close','crowded']
    gill_size=['Silahkan Pilih','broad','narrow']
    gill_color=['Silahkan Pilih','black','brown','buff','chocolate','gray','green','orange','pink','purple','red','white','yellow']
    stalk_shape=['Silahkan Pilih','enlarging','tapering']
    stalk_root=['Silahkan Pilih','bulbous','club','equal','rooted']
    stalk_surface_above_ring=['Silahkan Pilih','fibrous','scaly','silky','smooth']
    stalk_surface_below_ring=['Silahkan Pilih','fibrous','scaly','silky','smooth']
    stalk_color_above_ring=['Silahkan Pilih','brown','buff','cinnamon','gray','orange','pink','red','white','yellow']
    stalk_color_below_ring=['Silahkan Pilih','brown','buff','cinnamon','gray','orange','pink','red','white','yellow']
    veiltype=['Silahkan Pilih','partial']
    veil_color=['Silahkan Pilih','brown','orange','white','yellow']
    ring_number=['Silahkan Pilih','none','one','two']
    ringtype=['Silahkan Pilih','evanescent','flaring','large','none','pendant',]
    spore_print_color=['Silahkan Pilih','black','brown','buff','chocolate','green','orange','purple','white','yellow']
    population=['Silahkan Pilih','abundant','clustered','numerous','scattered','several','solitary']
    habitat=['Silahkan Pilih','grasses','leaves','meadows','paths','urban','waste','woods']

    with col1:
        cshap=st.selectbox('PILIH Cap Shape',capshape)
        if cshap=='bell':
            df['cap-shape_b']=1
        if cshap=='conical':
            df['cap-shape_c']=1
        if cshap=='convex':
            df['cap-shape_x']=1
        if cshap=='flat':
            df['cap-shape_f']=1
        if cshap=='knobbed':
            df['cap-shape_k']=1
        if cshap=='sunken':
            df['cap-shape_s']=1
            
        csur=st.selectbox('PILIH Cap Surface',capsurface)
        if csur=='fibrous':
            df['cap-surface_f']=1
        if csur=='grooves':
            df['cap-surface_g']=1
        if csur=='scaly':
            df['cap-surface_y']=1
        if csur=='smooth':
            df['cap-surface_s']=1
        
        cc=st.selectbox('PILIH Cap Color',capcolor)
        if cc=='brown':
            df['cap-color_n']=1
        if cc=='buff':
            df['cap-color_b']=1
        if cc=='cinnamon':
            df['cap-color_c']=1
        if cc=='gray':
            df['cap-color_g']=1
        if cc=='green':
            df['cap-color_r']=1
        if cc=='pink':
            df['cap-color_p']=1
        if cc=='purple':
            df['cap-color_u']=1
        if cc=='red':
            df['cap-color_e']=1
        if cc=='white':
            df['cap-color_w']=1
        if cc=='yellow':
            df['cap-color_y']=1    
            
        brus=st.selectbox('PILIH Bruises',bruises)
        if brus =='bruises':
            df['bruises_t']=1
        if brus =='no':
            df['bruises_f']=1
            
        odor=st.selectbox('PILIH Odor',odor)
        if odor =='almond':
            df['odor_a']=1
        if odor =='anise':
            df['odor_l']=1
        if odor =='creosote':
            df['odor_c']=1
        if odor =='fishy':
            df['odor_y']=1
        if odor =='foul':
            df['odor_f']=1
        if odor =='musty':
            df['odor_m']=1
        if odor =='none':
            df['odor_n']=1
        if odor =='pungent':
            df['odor_p']=1
        if odor =='spicy':
            df['odor_s']=1
        
        ga=st.selectbox('PILIH Gill Attachment',gill_attachment)
        if ga =='attached':
            df['gill-attachment_a']=1
        if ga =='free':
            df['gill-attachment_f']=1
        
        gsp=st.selectbox('PILIH Gill Spacing',gill_spacing)
        if gsp =='close':
            df['gill-spacing_c']=1
        if gsp =='crowded':
            df['gill-spacing_w']=1
        
        gsz=st.selectbox('PILIH Gill Size',gill_size)
        if gsz =='broad':
            df['gill-size_b']=1
        if gsz =='narrow':
            df['gill-size_n']=1
            
        gc=st.selectbox('PILIH Gill Color',gill_color)
        if gc =='black':
            df['gill-color_k']=1
        if gc =='brown':
            df['gill-color_n']=1
        if gc =='buff':
            df['gill-color_b']=1
        if gc =='chocolate':
            df['gill-color_h']=1
        if gc =='gray':
            df['gill-color_g']=1
        if gc =='green':
            df['gill-color_r']=1
        if gc =='orange':
            df['gill-color_o']=1
        if gc =='pink':
            df['gill-color_p']=1
        if gc =='purple':
            df['gill-color_u']=1
        if gc =='red':
            df['gill-color_e']=1
        if gc =='white':
            df['gill-color_w']=1
        if gc =='yellow':
            df['gill-color_y']=1    
            
        ssh=st.selectbox('PILIH Stalk Shape',stalk_shape)
        if ssh =='enlarging':
            df['stalk-shape_e']=1
        if ssh =='tapering':
            df['stalk-shape_t']=1
        
        sr=st.selectbox('PILIH Stalk Root',stalk_root)
        if sr =='bulbous':
            df['stalk-root_b']=1
        if sr =='club':
            df['stalk-root_c']=1
        if sr =='equal':
            df['stalk-root_e']=1
        if sr =='rooted':
            df['stalk-root_r']=1
        
    with col2:
        ssar=st.selectbox('PILIH Stalk Surface Above Ring',stalk_surface_above_ring)
        if ssar =='fibrous':
            df['stalk-surface-above-ring_f']=1
        if ssar =='scaly':
            df['stalk-surface-above-ring_y']=1
        if ssar =='silky':
            df['stalk-surface-above-ring_k']=1
        if ssar =='smooth':
            df['stalk-surface-above-ring_s']=1
            
        ssbr=st.selectbox('PILIH Stalk Surface Below Ring',stalk_surface_below_ring)
        if ssbr =='fibrous':
            df['stalk-surface-below-ring_f']=1
        if ssbr =='scaly':
            df['stalk-surface-below-ring_y']=1
        if ssbr =='silky':
            df['stalk-surface-below-ring_k']=1
        if ssbr =='smooth':
            df['stalk-surface-below-ring_s']=1
        
        scar=st.selectbox('PILIH Stalk Color Above Ring',stalk_color_above_ring)
        if scar=='brown':
            df['stalk-color-above-ring_n']=1
        if scar=='buff':
            df['stalk-color-above-ring_b']=1
        if scar=='cinnamon':
            df['stalk-color-above-ring_c']=1
        if scar=='gray':
            df['stalk-color-above-ring_g']=1
        if scar=='green':
            df['stalk-color-above-ring_r']=1
        if scar=='pink':
            df['stalk-color-above-ring_p']=1
        if scar=='purple':
            df['stalk-color-above-ring_u']=1
        if scar=='red':
            df['stalk-color-above-ring_e']=1
        if scar=='white':
            df['stalk-color-above-ring_w']=1
        if scar=='yellow':
            df['stalk-color-above-ring_y']=1
        if scar=='orange':
            df['stalk-color-above-ring_o']=1
        
        
        scbr=st.selectbox('PILIH Stalk Color Below Ring',stalk_color_below_ring)
        if scbr=='brown':
            df['stalk-color-below-ring_n']=1
        if scbr=='buff':
            df['stalk-color-below-ring_b']=1
        if scbr=='cinnamon':
            df['stalk-color-below-ring_c']=1
        if scbr=='gray':
            df['stalk-color-below-ring_g']=1
        if scbr=='green':
            df['stalk-color-below-ring_r']=1
        if scbr=='pink':
            df['stalk-color-below-ring_p']=1
        if scbr=='purple':
            df['stalk-color-below-ring_u']=1
        if scbr=='red':
            df['stalk-color-below-ring_e']=1
        if scbr=='white':
            df['stalk-color-below-ring_w']=1
        if scbr=='yellow':
            df['stalk-color-below-ring_y']=1 
        if scar=='orange':
            df['stalk-color-above-ring_o']=1 
            
        vt=st.selectbox('PILIH Veil Type',veiltype)
        if vt=='partial':
            df['veil-type_p']=1
        
        vc=st.selectbox('PILIH Veil Color',veil_color)
        if vc=='brown':
            df['veil-color_n']=1
        if vc=='orange':
            df['veil-color_o']=1
        if vc=='white':
            df['veil-color_w']=1
        if vc=='yellow':
            df['veil-color_y']=1
        
        rn=st.selectbox('PILIH Ring Number',ring_number)
        if rn=='none':
            df['ring-number_n']=1
        if rn=='one':
            df['ring-number_o']=1
        if rn=='two':
            df['ring-number_t']=1
        
        rt=st.selectbox('PILIH Ring Type',ringtype)
        if rt=='evanescent':
            df['ring-type_e']=1
        if rt=='flaring':
            df['ring-type_f']=1
        if rt=='large':
            df['ring-type_l']=1
        if rt=='none':
            df['ring-type_n']=1
        if rt=='pendant':
            df['ring-type_p']=1

        spc=st.selectbox('PILIH Spore Print Color',spore_print_color)
        if spc =='black':
            df['spore-print-color_k']=1
        if spc =='brown':
            df['spore-print-color_n']=1
        if spc =='buff':
            df['spore-print-color_b']=1
        if spc =='chocolate':
            df['spore-print-color_h']=1
        if spc =='green':
            df['spore-print-color_r']=1
        if spc =='orange':
            df['spore-print-color_o']=1
        if spc =='purple':
            df['spore-print-color_u']=1
        if spc =='white':
            df['spore-print-color_w']=1
        if spc =='yellow':
            df['spore-print-color_y']=1
        
        popul=st.selectbox('PILIH Population',population)
        if popul =='abundant':
            df['population_a']=1
        if popul =='clustered':
            df['population_c']=1
        if popul =='numerous':
            df['population_n']=1
        if popul =='scattered':
            df['population_s']=1
        if popul =='several':
            df['population_v']=1
        if popul =='solitary':
            df['population_y']=1
        
        habi=st.selectbox('PILIH Habitat',habitat)
        if habi =='grasses':
            df['habitat_g']=1
        if habi =='leaves':
            df['habitat_l']=1
        if habi =='meawods':
            df['habitat_m']=1
        if habi =='paths':
            df['habitat_p']=1
        if habi =='urban':
            df['habitat_u']=1
        if habi =='waste':
            df['habitat_w']=1
        if habi =='wood':
            df['habitat_d']=1
        
    button=st.button('KLASIFIKASI',use_container_width=500,type='primary')
    if button:
        #column= df.columns[df.eq(1).any()]
        #st.write(df[column])
        if cshap!='Silahkan Pilih'and csur!='Silahkan Pilih'and cc!='Silahkan Pilih'and brus!='Silahkan Pilih'and odor!='Silahkan Pilih'and ga!='Silahkan Pilih'and gsp!='Silahkan Pilih'and gsz!='Silahkan Pilih'and gc!='Silahkan Pilih'and ssh!='Silahkan Pilih'and sr!='Silahkan Pilih'and ssar!='Silahkan Pilih'and scar!='Silahkan Pilih'and scbr!='Silahkan Pilih'and vt!='Silahkan Pilih'and vc!='Silahkan Pilih'and rn!='Silahkan Pilih'and rt!='Silahkan Pilih'and spc!='Silahkan Pilih'and popul!='Silahkan Pilih'and habi!='Silahkan Pilih':
            loaded_model = load_model('knn_jamur')
            predictions = predict_model(loaded_model, data=df)
            prediction=predictions['prediction_label']
            for i in prediction:
                if i == 'p':
                    st.write('Status',i,', Jamur Beracun')
                if i == 'e':
                    st.write('Status',i,', Jamur Dapat di Makan')
        else:
            st.write('Mohon Isi semua Kolom Pertanyaan')
if selected=='Me':
    st.title('ABOUT ME')
    st.write("My Name is LU'LUATUL MAKNUNAH")
    st.write("Just Call Me LUNA")
    st.write("ID Number 210411100048")