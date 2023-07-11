from sentence_transformers import SentenceTransformer, util
from transformers import pipeline

import utils
import heapq
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import collections

def score_similarity(query_emb=None, docs_emb=np.array([])):
    #Compute dot score between query and all document embeddings
    scores = util.dot_score(query_emb, docs_emb)[0].cpu().tolist()
    return scores

def match_qncs(model, query, context, query_emb=None, context_emb=[]): # match questions with possible contexts

    #Encode query and documents
    query_emb = query_emb if query_emb else model.encode(query) 
    context_emb = context_emb if np.any(context_emb) else model.envode(context_emb)

    #Compute dot score between query and all document embeddings
    scores = score_similarity(query_emb, context_emb)

    #Combine context & scores
    score_pairs = list(zip(context, scores))

    #Sort by decreasing score
    score_pairs = sorted(score_pairs, key=lambda x: x[1], reverse=True)

    # #Output context & scores
    return score_pairs[0]

def match_all(model, questions=utils.varload('questions.txt'), context=utils.varload('context.txt')):
    context_prompt_input =("The document entitled {title} found at the URL link {url} has the following content: {content}")
    context = [context_prompt_input.format_map(ctx) for ctx in context]
    context_emb = model.encode(context)

    results = []
    for q in questions:
        results.append((q, match_qncs(model,q,context,context_emb=context_emb)))
    # sort results by similarity score between query/question and context
    results = sorted(results, key=lambda qctx: qctx[1][1], reverse=True)
    return results
    # return context_emb

# def makej_qnas(qncs, model_name='sjrhuschlee/flan-t5-large-squad2'):
def makej_qnas(qncs, full_context=None, model_name='deepset/roberta-base-squad2'): 
    """Make JSON ready dicts for questions&ansnwers - j_qnas"""
    question_answerer = pipeline("question-answering", 
                                 model=model_name,
                                 okenizer=model_name,
                                 trust_remote_code=True)

    j_qnas = []

    for qst, ctx in qncs:
        # ctx = ctx[0]
        ctx = [full_context, 100] if full_context else ctx
        j_qnas.append({"instruction": qst,
                     "input": {'context':ctx[0], 'score':ctx[1]},
                     "output": question_answerer({'question': qst, 
                                                  'context': ctx[0]})})

    return j_qnas

def convert_todf(j_qnas):
    instructions = []
    inputs = []
    outputs = []
    for qna in j_qnas:
        instructions.append(qna['instruction'])
        inputs.append(qna['input'])
        outputs.append(qna['output'])
    instructions = pd.DataFrame({'instruction':instructions})
    inputs = pd.DataFrame(inputs).rename(columns={'score':'qm_score'}) # question match score
    outputs = pd.DataFrame(outputs).rename(columns={'score':'ans_score'}) # answer score
    return pd.concat((instructions, inputs, outputs), axis=1)

# split dataset in favorites and outliers
def split_favout(df_qnas, lambda_ = 0.8, a_thold=0.10):

    # favorites = df_qnas[df_qnas.ans_score > df_qnas.ans_score.quantile(a_thold)]
    # outliers = df_qnas[df_qnas.ans_score <= df_qnas.ans_score.quantile(a_thold)]

    df_qnas['qmn_score'] = (df_qnas.qm_score - df_qnas.qm_score.min())/(df_qnas.qm_score.max()-df_qnas.qm_score.min())
    df_qnas['ansn_score'] = (df_qnas.ans_score - df_qnas.ans_score.min())/(df_qnas.ans_score.max()-df_qnas.ans_score.min())
    df_qnas['wm_score'] = (df_qnas.qmn_score * lambda_ + df_qnas.ansn_score * (1-lambda_))

    favorites = df_qnas[df_qnas.wm_score > a_thold]
    outliers = df_qnas[df_qnas.wm_score <= a_thold]

    return favorites, outliers

# determine a sorting via similarity too similar in question & answer (e.g. Maximal Marginal Relevance)
def argmax(keys, f): 
    return max(keys, key=f) 

def mmr_sorted(docs_emb, query_emb, lambda_, similarity_fn): # THIS DOES NOT SCALE WELL
    """Sort a list of docs by Maximal marginal relevance""" 
    ids = set(range(len(docs_emb)))
    selected = collections.OrderedDict() 
    while set(selected) != ids: 
        remaining = ids - set(selected) 
        mmr_score = lambda id1: lambda_*similarity_fn(docs_emb[id1], query_emb) - (1-lambda_)*max([similarity_fn(docs_emb[id1], docs_emb[id2]) for id2 in set(selected)-{id1}] or [0]) 
        next_selected = argmax(remaining, mmr_score) 
        selected[next_selected] = len(selected) 
    return list(selected.keys())

def sort_bydiversity(df_qnas, model=SentenceTransformer('sentence-transformers/msmarco-distilbert-base-tas-b'),
                     athold=0.15, simthold=90, redundancy=1 # unused
                     ):
    doc_format =("""The question to be answered is {q}.
                 The context for this question is: {c}.
                 The answer to the question is: {a}""")

    docs = [doc_format.format_map({"q": q,
                                   "c": c, 
                                   "a": a}) for q,c,a in df_qnas[['instruction','context','answer']].values]
    
    docs_emb = model.encode(docs)
    query_emb = model.encode('question and asnwer')
    similarity_fn = lambda x,y: score_similarity(query_emb=x,docs_emb=[y])[0]

    reorder_ids = mmr_sorted(docs_emb, query_emb, 0, similarity_fn)
    return [df_qnas.index.values[i] for i in reorder_ids] #[docs[idx] for idx in reorder_ids]

# keep alpha(%) for training and 1-alpha(%) for testing 
def split_trainval(df_qnas, alpha=0.9):
    df_len = df_qnas.shape[0]
    tr_split = int(df_len * alpha)

    return {'train':df_qnas.iloc[:tr_split],
            'test':df_qnas.iloc[tr_split:],}

def make_trainval(questions=utils.varload('questions.txt'), context=utils.varload('context.txt'), 
                  models={'sentence-transformer': 'sentence-transformers/msmarco-distilbert-base-tas-b',
                          'question-answering':'deepset/roberta-base-squad2'},
                  # params: to be evaled by looking at the dataset):
                  params={'fav_split_lambda':0.3, # 0.3 * q-ctx similarity match + 0.7 q-answer score: we give more bias to the answer score as this is the actual output
                          'fav_split_a_thold': 0.1, # outliers are answers with scores less 0.1 - this is evaluated through a look at the statistics/hist
                          'trainval_split_alpha': 0.8 # looking at the stats after reordering is done
                          }):
    sentence_model = SentenceTransformer(models['sentence-transformer'])
    # sorteby question-to-context match-in-similarity score
    qncs = match_all(sentence_model, questions, context)

    # generate/extract answers
    j_qnas = makej_qnas(qncs, model_name=models['question-answering'],full_context=None)
    
    # convert to pandast dataframe
    df_qnas = convert_todf(j_qnas)

    # split into favorites and outliers:
    # 0.3 * q-ctx similarity match + 0.7 q-answer score: we give more bias to the answer score as this is the actual output
    # outliers are answers with scores less 0.1 - this is evaluated through a look at the statistics/hist
    favorites, outliers = split_favout(df_qnas, lambda_=params['fav_split_lambda'], a_thold=params['fav_split_a_thold']) 

    # sort by diversity
    favorites = favorites.reindex(sort_bydiversity(favorites))
    outliers = outliers.reindex(sort_bydiversity(outliers))

    # do a split of both and merge back into train/test sets
    favorites = split_trainval(favorites, params['trainval_split_alpha'])
    outliers = split_trainval(outliers, params['trainval_split_alpha'])

    return favorites, outliers

# inspect difference between ds
def doheat(ds, sentence_model=SentenceTransformer('sentence-transformers/msmarco-distilbert-base-tas-b')):
    doc_format = ("""The question to be answered is {q}.
                    The context for this question is: {c}.
                    The answer to the question is: {a}""")
    docs = [doc_format.format_map({"q": q,
                                    "c": c,
                                    "a": a}) for q, c, a in ds[['instruction', 'context', 'answer']].values]

    docs_emb = sentence_model.encode(docs).tolist()
    scores_matrix = [score_similarity(d, docs_emb) for d in docs_emb]

    return np.array(scores_matrix)

def converttodict(dataframe_ds):
    instructions = [{'instruction': q,
                     'input': c,
                     'output': a} for q, c, a in dataframe_ds[['instruction', 'context', 'answer']].values]
    return instructions

