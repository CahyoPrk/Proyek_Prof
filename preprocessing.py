import re
import pandas as pd
from nltk import word_tokenize
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from sklearn.feature_extraction.text import TfidfVectorizer
#from translate import Translator
#from googletrans import Translator
from mtranslate import translate

def translate_text(text):
    translation = translate(text, "id")
    return translation

def clean(text):
    text = text.strip()
    text = text.lower()
    text = re.sub(r'[^a-zA-Z]+', ' ', text)
    return text

def normalisasi(text):
    norm= {' beenilai ':' bernilai ', ' gak ':' tidak ', ' eksis ': ' ada ', ' yg ':' yang ', ' sdg ':' sedang ', ' dg ':' dengan ', ' nan ':' dan ',
        ' cociks ':' cocok ', ' heritage ': ' peninggalan ', ' trip ':' wisata ',' kl ':' kalo ', ' ga ':' tidak ', ' bosen ': ' bosan ',
        ' d ':' di ', ' k ':' ke ', ' plg ':' pulang ', ' ajah ':' aja ', ' bgt ':' banget ', ' fahami ':' pahami ', ' cermay ':' cermat ',
        'tmptna ':'tempatnya ', ' dpt ':' dapat ', ' kebudyaan ':' kebudayaan ', ' guide ':' pemandu ', ' dilaranf ':' dilarang ', ' lbh ':' lebih ',
        ' ayem ':' tenang ', ' dah ':' udah ', ' mager ': ' malas gerak ', ' bikin ':' buat ', ' dsana ':' disana ', ' dijaman ':' di zaman',
        ' kalo ':' kalau ', ' hape ':' gawai ', ' charge ':' biaya ', ' adem ':' tenang ', 'nguri uri ':'melestarikan ', ' lg ':' lagi ', ' ny ':' nya ',
        ' sejrh ':' sejarah ', ' thn ':' tahun ', ' utk ':' untuk ', ' mlht ':' melihat ', ' dn ':' dan ', ' gk ':' tidak ', ' klw ':' kalo ', ' trs ':' terus ',
        ' bs ': ' bisa ', ' jlan ':' jalan ', ' temp ':' tempat ', ' sy ':' saya ', ' trmpatnya ':' tempatnya ', 'srklsi ':'sekali ', ' mantafff ':' mantap ',
        ' karna ':' karena ', ' exis ':' tetap ', 'apek tenan':'bagus banget', ' tentram ':' tenteram', ' mgkn ':' mungkin ', ' sndri ':' sendiri ',
        ' boring ':' bosan ', ' lmyn ':' lumayan', ' krn ':' karena  ', ' pgn ':' pengen ', ' dlm ':' dalam ', ' bt ':' butuh ', ' dr ':' dari ',
        ' n ':' dan ', ' bbrp ':' beberapa ', ' skali ':' sekali ', ' bangeettt':' banget', 'snagt ':'sangat ', ' poto':' foto', ' photo ':' foto ',
        ' skr ':' sekarang ', ' hrs ':' harus ', ' sgt ':' sangat ', ' biasaaaaaaa':' biasa', ' jg':' juga', ' lwt ':' lewat ', ' orng ':' orang ',
        ' dlm ':' dalam ', ' apakh ':' apakah ', ' mnitip ':' menitip ', ' ttip ':' titip ', ' bsa ':' bisa ', ' tkg ':' tukang ', ' kmi ':' kami ',
        ' dibw ':' dibawa ', ' jdwl ':' jadwal ', ' bkrja ':' berkerja', ' bln  ':' bulan ', ' pnggntinya ': ' penggantinya ', ' semberawut ':' berantakan ',
        ' ptgs ': ' petugas ', ' rmptnya ':' rumputnya', ' petgsnya ':' petugasnya', ' jd ': ' jd ', ' kasultanan ':' kesultanan', 'mmf ': 'mohon maaf ',
        ' blm ':' belum ', ' ttp ':' tetapi ', ' sprti ':' seperti ', ' dtg ':' datang ', ' clana ':' celana ', ' luarbiasa ':' luar biasa', ' kbdynnya ':' kebudayaan ',
        ' prl ':' perlu ', ' dtngktkn ':' ditingkatkan lagi ', ' tlah ':' telah ', ' jm ':' jam ', 'jum at':'jumat', ' bnyak ':' banyak ', ' bgs ':' bagus ',
        ' tdk ':' tidak ', ' exis ':' ada ', ' rapih ':' rapi ', ' tp ':' tapi ', ' pas ':' waktu ', 'diterjemahkan oleh google ':'', 'smoga ':'semoga ', ' ank ':' anak ',
        'istana':'kraton', 'keraton':'kraton', 'ngayogyakarta':'yogyakarta', 'jogja ':'yogyakarta ', ' jogjakarta':' yogyakarta',
        ' rp ': ' rupiah ', ' rb ':' ribu ', ' b ':' biasa ', ' ngayogjokarto ':' yogyakarta ', ' th ': ' tahun '}
    for i in norm:
        text = text.replace(i, norm[i])
    return text
    
def tokenize(text):
    tokens = word_tokenize(text)
    return tokens

def stopword(tokens):
    factory = StopWordRemoverFactory()
    stopwords = factory.get_stop_words()
    filtered_tokens = [token for token in tokens if token not in stopwords]
    return filtered_tokens

def stemming(filtered_tokens):
    factory = StemmerFactory()
    stemmer = factory.create_stemmer()  
    stemmed_tokens = [stemmer.stem(token) for token in filtered_tokens]
    return stemmed_tokens

def join(stemmed_tokens):
    joined_text = ' '.join(stemmed_tokens)
    return joined_text

def label(rating):
    if rating == '5 stars' or rating == '4 stars':
        return '1'
    else:
        return '0'

def fix_label(df):
    kata_positif = ['bersih', 'murah', 'rapi', 'atur', 'puas', 'ramah', 'bagus', 'awat', 'oke', 'keren', 'kagum']
    df.loc[df['review'].str.contains('|'.join(kata_positif), case=False), 'sentimen'] = 1   
    kata_negatif = ['bau', 'kotor', 'mahal', 'panas', 'jelek', 'macet', 'ramai', 'kasar', 'tutup', 'lama']
    df.loc[df['review'].str.contains('|'.join(kata_negatif), case=False), 'sentimen'] = 0
    df = df[df['review'].apply(lambda x: len(x.split()) > 2)]
    df.reset_index(drop=True, inplace=True)
    return df

def prep_all(text):
    text = clean(text)
    text = normalisasi(text)
    text = tokenize(text)
    text = stopword(text)
    text = stemming(text)
    text = join(text)
 
    return pd.DataFrame({'review': [text]})

def prep_aspek(text):
    text = clean(text)
    text = normalisasi(text)
    text = tokenize(text)
    text = stopword(text)
    text = stemming(text)
    text = join(text)
    return text