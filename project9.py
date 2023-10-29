import streamlit as st
import pandas as pd
from streamlit_option_menu import option_menu
from sklearn.ensemble import RandomForestClassifier
import pickle

kolom=['cap-shape_b','cap-shape_c','cap-shape_f','cap-shape_k','cap-shape_s','cap-shape_x','cap-surface_f','cap-surface_g','cap-surface_s','cap-surface_y','cap-color_b','cap-color_c','cap-color_e','cap-color_g','cap-color_n','cap-color_p','cap-color_r','cap-color_u','cap-color_w','cap-color_y','bruises_f','bruises_t','odor_a','odor_c','odor_f','odor_l','odor_m','odor_n','odor_p','odor_s','odor_y','gill-attachment_a','gill-attachment_f','gill-spacing_c','gill-spacing_w','gill-size_b','gill-size_n','gill-color_b','gill-color_e','gill-color_g','gill-color_h','gill-color_k','gill-color_n','gill-color_o','gill-color_p','gill-color_r','gill-color_u','gill-color_w','gill-color_y','stalk-shape_e','stalk-shape_t','stalk-root_b','stalk-root_c','stalk-root_e','stalk-root_r','stalk-surface-above-ring_f','stalk-surface-above-ring_k','stalk-surface-above-ring_s','stalk-surface-above-ring_y','stalk-surface-below-ring_f','stalk-surface-below-ring_k','stalk-surface-below-ring_s','stalk-surface-below-ring_y','stalk-color-above-ring_b','stalk-color-above-ring_c','stalk-color-above-ring_e','stalk-color-above-ring_g','stalk-color-above-ring_n','stalk-color-above-ring_o','stalk-color-above-ring_p','stalk-color-above-ring_w','stalk-color-above-ring_y','stalk-color-below-ring_b','stalk-color-below-ring_c','stalk-color-below-ring_e','stalk-color-below-ring_g','stalk-color-below-ring_n','stalk-color-below-ring_o','stalk-color-below-ring_p','stalk-color-below-ring_w','stalk-color-below-ring_y','veil-type_p','veil-color_n','veil-color_o','veil-color_w','veil-color_y','ring-number_n','ring-number_o','ring-number_t','ring-type_e','ring-type_f','ring-type_l','ring-type_n','ring-type_p','spore-print-color_b','spore-print-color_h','spore-print-color_k','spore-print-color_n','spore-print-color_o','spore-print-color_r','spore-print-color_u','spore-print-color_w','spore-print-color_y','population_a','population_c','population_n','population_s','population_v','population_y','habitat_d','habitat_g','habitat_l','habitat_m','habitat_p','habitat_u','habitat_w']

st.title('APLIKASI KLASIFIKASI JENIS JAMUR')
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
    column= df.columns[df.eq(1).any()]
    st.write(df[column])
    if cshap!='Silahkan Pilih'and csur!='Silahkan Pilih'and cc!='Silahkan Pilih'and brus!='Silahkan Pilih'and odor!='Silahkan Pilih'and ga!='Silahkan Pilih'and gsp!='Silahkan Pilih'and gsz!='Silahkan Pilih'and gc!='Silahkan Pilih'and ssh!='Silahkan Pilih'and sr!='Silahkan Pilih'and ssar!='Silahkan Pilih'and scar!='Silahkan Pilih'and scbr!='Silahkan Pilih'and vt!='Silahkan Pilih'and vc!='Silahkan Pilih'and rn!='Silahkan Pilih'and rt!='Silahkan Pilih'and spc!='Silahkan Pilih'and popul!='Silahkan Pilih'and habi!='Silahkan Pilih':
        with open('PCA7rf.pkl', 'rb') as pca:
            loadpca= pickle.load(pca)
        f_pca=loadpca.transform(df)
        st.write(f_pca)
        with open('jamur.pkl', 'rb') as rf:
            randomfores= pickle.load(rf)
        klasifikasi=randomfores.predict(f_pca)
        for klas in klasifikasi:
            if klas == 'p':
                st.write('Status : ',klas,', JAMUR TIDAK BISA DI KONSUMSI')
            if klas =='e':
                st.write('Status : ',klas,', JAMUR BISA DI KONSUMSI')
    else:
        st.write('Mohon Isi semua Kolom Pertanyaan')
