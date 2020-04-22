import pandas as pd
import re
import numpy as np
import string
from sklearn.feature_extraction.text import CountVectorizer



def preprocess_imaging(df):

    # ct_vocab = '13gram_24surg_27nonsurg'
    # sboimg_vocab = 'sboimg_23gram_min12_43surg_52nonsurg'
    ct_vocab = 'ct_hand'
    sboimg_vocab = 'sboimg_hand'

    ct = pd.read_pickle('imaging/ct_full.pickle')
    sboimg = pd.read_pickle('imaging/sboimg_full.pickle')
    ct_text = preprocess_imaging_helper(ct, df, ct_vocab, 'ct_img')
    sboimg_text = preprocess_imaging_helper(sboimg, df, sboimg_vocab, 'sbo_img')

    res = pd.concat([df,
        #ct_text[['datetime','word_log_ratio_img', 'indicator_img']]]
        ct_text.drop(['text_len'],1),
        sboimg_text.drop(['text_len'],1)]
        , sort=True)
    return res

def preprocess_imaging_helper(ct, df, vocab, suff):
    # NOTE: not just for ct but didn't want to have to change the variables

    print('\nVocabulary: {}'.format(vocab))
    vocabulary_dict = pd.read_pickle(
        'vocabularies/{}.pickle'.format(vocab))
    surg_words = list(vocabulary_dict['surg'])
    non_surg_words = list(vocabulary_dict['non_surg'])

    # Only imaging for people in df
    ct = (ct.reset_index(level=0)
        .loc[df.index.get_level_values(1).unique(),:]
        .dropna()
        .reset_index().set_index(['mrn','id', 'datetime']))

    # Count occurrences using vocabulary
    count_vectorizer = CountVectorizer(
        vocabulary = surg_words + non_surg_words,
        binary=True)
    count_array = count_vectorizer.fit_transform(ct['full_text'].values)
    count_df = pd.DataFrame(count_array.todense(),
                            columns=count_vectorizer.get_feature_names(),
                            index=ct.index).add_suffix('_{}'.format(suff))
    surg_words = [col+'_{}'.format(suff) for col in surg_words]
    non_surg_words = [col+'_{}'.format(suff) for col in non_surg_words]

    ct_text = pd.concat([ct[['text_len']], count_df],1, sort=True).reset_index(level=2)

    surg_words_count = ct_text[surg_words].sum(axis=1)
    non_surg_words_count = ct_text[non_surg_words].sum(axis=1)
    surg_words_count_adj = surg_words_count / ct_text['text_len']
    non_surg_words_count_adj = non_surg_words_count / ct_text['text_len']
    #ct_enc_text['surg_words'] = np.log(ct_enc_text['surg_words_count_adj']+1) > 1
    #ct_enc_text['non_surg_words'] = np.log(ct_enc_text['non_surg_words_count_adj']+1) > 0.5
    #ct_enc_text['word_diff'] = (
    #    ct_enc_text['surg_words_count_adj'] - ct_enc_text['non_surg_words_count_adj']/2)
    #ct_text['word_log_ratio_{}'.format(suff)] = (
    #    np.log(
    #        (surg_words_count_adj + 1)
    #        / (non_surg_words_count_adj + 1)))

    ct_text['ind_event_{}'.format(suff)] = 1
    return ct_text


if __name__ == '__main__':
    preprocess_imaging(ct)
