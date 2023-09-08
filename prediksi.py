import pickle


filename = 'model-svm-RFE-rbf-10.0.pickle'
fs_label = filename[10:13] # RFE
text = ['pilkada akan dimenangkan oleh anies']

with open(filename, 'rb') as fin:
    if fs_label == 'RFE':
        vectorizer,clf,rfe = pickle.load(fin)
        tfidf_text_vectors = vectorizer.transform(text)
    else:
        vectorizer,clf = pickle.load(fin)
        tfidf_text_vectors = vectorizer.transform(text)
y_pred = clf.predict(tfidf_text_vectors)
print(y_pred)
if y_pred:
    sentimen_svm = 'Tweet positif'
else:
    sentimen_svm = 'Tweet negatif'

print('FS\t\t: ', fs_label)
print('Teks\t\t: ', text[0])
print('Sentimen\t: ', sentimen_svm) #
