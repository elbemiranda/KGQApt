import requests
import json
import pandas as pd
import numpy as np
import json
from SPARQLWrapper import SPARQLWrapper, JSON
import requests
import itertools
import spotlight
import tagme
import inflect
import re
import sys
import requests
from nltk.stem.porter import *
import nltk
from nltk.stem import RSLPStemmer 
import gensim
from sklearn.metrics.pairwise import cosine_similarity
import re
import string
from kb.kb import KB
from common.container.uri import Uri
from kb.dbpedia import DBpedia
import urllib.parse, urllib.request, json

#nltk.download('stopwords')

p = inflect.engine()
tagme.GCUBE_TOKEN = ""


def sort_dict_by_values(dictionary):
    keys = []
    values = []
    for key, value in sorted(dictionary.items(), key=lambda item: (item[1], item[0]), reverse=True):
        keys.append(key)
        values.append(value)
    return keys, values


def preprocess_relations(file, stemmer, prop=False):
    relations = {}
    with open(file, encoding='utf-8') as f:
        content = f.readlines()
        for line in content:
            split_line = line.split()

            key = ' '.join(split_line[2:])[1:-3].lower()
            key = ' '.join([stemmer.stem(word) for word in key.split()])

            if key not in relations:
                relations[key] = []

            uri = split_line[0].replace('<', '').replace('>', '')

            if prop is True:
                uri_property = uri.replace('/ontology/', '/property/')
                relations[key].extend([uri, uri_property])
            else:
                relations[key].append(uri)
    return relations

def preprocess_relations_qakgpt(file, fasttext_model, stopwords, prop=False):
    relations = {}
    with open(file, encoding='utf-8') as f:
        content = f.readlines()
        for line in content:
            split_line = line.split()

            key = ' '.join(split_line[2:])[1:-3] #.lower()
            key = ' '.join([split_camel_case(word) for word in key.split()])
            key = ' '.join([split_hiphen(word) for word in key.split()])
            #key = ' '.join([stemmer.stem(word) for word in key.split()])
            for word in key.split():
                if (word in stopwords):
                    key = re.sub(r'\b' + word + r'\b', ' ', key)
            key = re.sub(' +', ' ', key)
            key = key.translate(str.maketrans('', '', string.punctuation))
            key = key.strip()
            key = ' '.join([w for w in key.split() if len(w.strip()) > 1])
            
            if key not in relations:
                relations[key] = {'uris':[],
                                  'vector':[]}
                
            words = key.split()
            if len(words) > 0:
                words_in_vocab = [word for word in words if word in fasttext_model.vocab]
                if len(words_in_vocab) > 0:
                    mean_vector = np.mean(fasttext_model[words_in_vocab], axis = 0)
                    relations[key]['vector'] = mean_vector
                else:
                    relations[key]['vector'] = np.zeros(300)
            else:
                relations[key]['vector'] = np.zeros(300)


            uri = split_line[0].replace('<', '').replace('>', '')

            if prop is True:
                uri_property = uri.replace('/ontology/', '/property/')
                relations[key]['uris'].extend([uri, uri_property])
            else:
                relations[key]['uris'].append(uri)
        for key in relations.keys():
            relations[key]['uris'] = list(set(relations[key]['uris']))
    return relations    


def get_earl_entities(query):

    result = {}
    result['question'] = query
    result['entities'] = []
    result['relations'] = []

    THRESHOLD = 0.1

    response = requests.post('http://ltdemos.informatik.uni-hamburg.de/earl/processQuery',
                             json={"nlquery": query, "pagerankflag": False})

    json_response = json.loads(response.text)
    type_list = []
    chunk = []
    for i in json_response['ertypes']:
        type_list.append(i)
    for i in json_response['chunktext']:
        chunk.append([i['surfacestart'], i['surfacelength']])

    keys = list(json_response['rerankedlists'].keys())
    reranked_lists = json_response['rerankedlists']
    for i in range(len(keys)):
        if type_list[i] == 'entity':
            entity = {}
            entity['uris'] = []
            entity['surface'] = chunk[i]
            for r in reranked_lists[keys[i]]:
                if r[0] > THRESHOLD:
                    uri = {}
                    uri['uri'] = r[1]
                    uri['confidence'] = r[0]
                    entity['uris'].append(uri)
            if entity['uris'] != []:
                result['entities'].append(entity)
        if type_list[i] == 'relation':
            relation = {}
            relation['uris'] = []
            relation['surface'] = chunk[i]
            for r in reranked_lists[keys[i]]:
                if r[0] > THRESHOLD:
                    uri = {}
                    uri['uri'] = r[1]
                    uri['confidence'] = r[0]
                    relation['uris'].append(uri)
            if relation['uris'] != []:
                result['relations'].append(relation)

    return result


def get_tag_me_entities(query):
    threshold = 0.1
    try:
        response = requests.get("https://tagme.d4science.org/tagme/tag?lang=en&gcube-token={}&text={}"
                                .format('1b4eb12e-d434-4b30-8c7f-91b3395b96e8-843339462', query))

        entities = []
        for annotation in json.loads(response.text)['annotations']:
            confidence = float(annotation['link_probability'])
            if confidence > threshold:
                entity = {}
                uris = {}
                uri = 'http://dbpedia.org/resource/' + annotation['title'].replace(' ', '_')
                uris['uri'] = uri
                uris['confidence'] = confidence
                surface = [annotation['start'], annotation['end']-annotation['start']]
                entity['uris'] = [uris]
                entity['surface'] = surface
                entities.append(entity)
    except:
        entities = []
        print('get_tag_me_entities: ', query)
    return entities


def get_nliwod_entities(query, hashmap, stemmer, language = "en"):
    ignore_list = []
    entities = []
    if(language == 'en'):
        singular_query = [stemmer.stem(word) if p.singular_noun(word) == False else stemmer.stem(p.singular_noun(word)) for
                      word in query.lower().split(' ')]
    else:
        singular_query = [stemmer.stem(word) for word in query.lower().split(' ')]

    string = ' '.join(singular_query)
    words = query.split(' ')
    indexlist = {}
    surface = []
    current = 0
    locate = 0
    for i in range(len(singular_query)):
        indexlist[current] = {}
        indexlist[current]['len'] = len(words[i])-1
        indexlist[current]['surface'] = [locate, len(words[i])-1]
        current += len(singular_query[i])+1
        locate += len(words[i])+1
    for key in hashmap.keys():
        if key in string and len(key) > 2 and key not in ignore_list:
            e_list = list(set(hashmap[key]))
            k_index = string.index(key)
            if k_index in indexlist.keys():
                surface = indexlist[k_index]['surface']
            else:
                for i in indexlist:
                    if k_index>i and k_index<(i+indexlist[i]['len']):
                        surface = indexlist[i]['surface']
                        break
            for e in e_list:
                r_e = {}
                r_e['surface'] = surface
                r_en = {}
                r_en['uri'] = e
                r_en['confidence'] = 0.3
                r_e['uris'] = [r_en]
                entities.append(r_e)
    return entities


def get_spotlight_entities(query, lang = 'en'):
    entities = []
    data = {
        'text': query,
        'confidence': '0.4',
        'support': '10'
    }
    headers = {"Accept": "application/json"}
    try:
        response = requests.post(f'https://api.dbpedia-spotlight.org/{lang}/annotate', data=data, headers=headers)
        response_json = response.text.replace('@', '')
        output = json.loads(response_json)
        if 'Resources' in output.keys():
            resource = output['Resources']
            for item in resource:
                entity = {}
                uri = {}
                uri['uri'] = item['URI']
                uri['confidence'] = float(item['similarityScore'])
                entity['uris'] = [uri]
                entity['surface'] = [int(item['offset']), len(item['surfaceForm'])]
                entities.append(entity)
    except Exception as ex:
        print('Spotlight: ', ex)
    return entities


def get_falcon_entities(query):

    entities = []
    relations = []
    headers = {
        'Content-Type': 'application/json',
    }
    params = (
        ('mode', 'long'),
    )
    data = "{\"text\": \"" + query + "\"}"
    response = requests.post('https://labs.tib.eu/falcon/api', headers=headers, params=params, data=data.encode('utf-8'))
    try:
        output = json.loads(response.text)
        for i in output['entities']:
            ent = {}
            ent['surface'] = ""
            ent_uri = {}
            ent_uri['confidence'] = 0.9
            ent_uri['uri'] = i[0]
            ent['uris'] = [ent_uri]
            entities.append(ent)
        for i in output['relations']:
            rel = {}
            rel['surface'] = ""
            rel_uri = {}
            rel_uri['confidence'] = 0.9
            rel_uri['uri'] = i[0]
            rel['uris'] = [rel_uri]
            relations.append(rel)
    except:
            print('get_falcon_entities: ', query)
    return entities, relations

def get_wikifier_entities(query, lang="pt", threshold=0.8):
    # Prepare the URL.
    data = {"text":query,
            "lang":lang,
            "userKey":"hsmxkxvbcjqueyoswokyyfqynoalpn",
            "pageRankSqThreshold": "%g" % threshold,
            "applyPageRankSqThreshold":"true",
            "nTopDfValuesToIgnore" :"200",
            "nWordsToIgnoreFromList":"200",
            "wikiDataClasses": "false",
            "wikiDataClassIds": "false",
            "support" : "true",
            "ranges": "false",
            "secondaryAnnotLanguage" : "en",
            #"minLinkFrequency": "1",
            "includeCosines": "false",
            "maxMentionEntropy":"3"
           }
    url = "http://www.wikifier.org/annotate-article"
    # Call the Wikifier and read the response.
    entities = []
    try:
        response = requests.post(url, data=data)
        response = json.loads(response.text)
        # Output the annotations.
        annotations = response["annotations"]
        for annotation in annotations:
            entity = {}
            uri = {}
            uri['uri'] = annotation['dbPediaIri']
            pageRank = 0
            for sup in annotation['support']:
                if sup['pageRank'] > pageRank:
                    uri['confidence'] = sup['prbConfidence']
                    surface = [sup['chFrom'], sup['chTo']]
                    
            entity['uris'] = [uri]
            entity['surface'] = surface
            entities.append(entity)
    except Exception as ex:
            print('get_wikifier_entities: ', ex)
    return entities    
def split_camel_case(word):
    word = re.sub('([A-Z][a-z]+)', r' \1', re.sub('([A-Z]+)', r' \1', word)).split()
    return(' '.join(word).lower())

def split_hiphen(word):
    word = ' '.join(word.split('-'))
    return(word.lower())

def calculate_cosine_similarity(word_vector, properties):
    similarities = []
    properties_similarities = {}
    for key in properties.keys():
        cosine_result = cosine_similarity([word_vector], [properties[key]['vector']])[0][0]
        similarities.append(cosine_result)
        properties_similarities[cosine_result] = properties[key]
    return([similarities, properties_similarities])

def get_qakgpt_relations(query, entities, fasttext_model, stopwords, properties):
    qakgpt_relations = []

    question = query
    question_no_entities = question
    print(question)
    for entity in entities:
        start = entity['surface'][0]
        end = entity['surface'][1]
        entity_text = question[start:start+end]
        question_no_entities = question_no_entities.replace(entity_text, "")
    
    clean_question = question_no_entities.lower()
    # if clean_question.startswith("quando"):
    #     clean_question = ' '.join(clean_question.split()[1:len(clean_question.split())])
    #     clean_question = "em que data " + clean_question

    # if clean_question.startswith("onde"):
    #     clean_question = ' '.join(clean_question.split()[1:len(clean_question.split())])
    #     clean_question = "em que local " + clean_question

    for word in clean_question.split():
        if (word in stopwords):
            clean_question = re.sub(r'\b' + word + r'\b', ' ', clean_question)
    clean_question = re.sub(' +', ' ', clean_question)
    clean_question = clean_question.translate(str.maketrans('', '', string.punctuation.replace('-', '')))
    clean_question = clean_question.strip()
    clean_question = ' '.join([w for w in clean_question.split() if len(w.strip()) > 1])
    clean_question = clean_question.split()
    print(clean_question)
    if len(clean_question) > 0:
        words = [word for word in clean_question if word in fasttext_model.vocab]
        if len(words) > 0:
            mean_vector = np.mean(fasttext_model[words], axis = 0)
            similarities, properties_similarities = calculate_cosine_similarity(mean_vector, properties)
            similarities = sorted(similarities, reverse = True)
            treshold = min(3, len(similarities))
            most_similars = similarities[0:treshold]
            confidence = np.mean(most_similars)
            relation_uris = []

            for most_similar in most_similars:
                uris = properties_similarities[most_similar]['uris']
                relation_uris.extend(uris)
            
            relation_uris = list(set(relation_uris))

            for relation_uri in relation_uris:
                rel = {}
                rel['surface'] = ""
                rel['uris'] = []
                rel_uri = {}
                rel_uri['confidence'] = float(str(confidence))
                rel_uri['uri'] = relation_uri
                rel['uris'].append(rel_uri)
                qakgpt_relations.append(rel)

    return qakgpt_relations

def get_sameas(kb, entity):
    if kb.check_server():
        query = "PREFIX owl:<http://www.w3.org/2002/07/owl#> SELECT ?obj WHERE { <" + entity + "> (owl:sameAs|^owl:sameAs)* ?obj FILTER (strstarts(str(?obj), 'http://dbpedia.org/resource'))}"
        result = kb.query(query)
        if result[0] == 200:
            bindings = result[1]['results']['bindings']
            if len(bindings) > 0:
                for binding in bindings:
                    binding_type = binding['obj']['type']
                    if binding_type == 'uri':
                        return binding['obj']['value']
        return None

def merge_entity(old_e, new_e):
    for i in new_e:
        exist = False
        for j in old_e:
            for k in j['uris']:
                if i['uris'][0]['uri'] == k['uri']:
                    k['confidence'] = max(k['confidence'], i['uris'][0]['confidence'])
                    exist = True
        if not exist:
            old_e.append(i)
    return old_e


def merge_relation(old_e, new_e):
    for i in range(len(new_e)):
        for j in range(len(old_e)):
            if new_e[i]['surface']==old_e[j]['surface']:
                for i1 in range(len(new_e[i]['uris'])):
                    notexist = True
                    for j1 in range(len(old_e[j]['uris'])):
                        if new_e[i]['uris'][i1]['uri']==old_e[j]['uris'][j1]['uri']:
                            old_e[j]['uris'][j1]['confidence'] = max(old_e[j]['uris'][j1]['confidence'], new_e[i]['uris'][i1]['confidence'])
                            notexist = False
                    if notexist:
                        old_e[j]['uris'].append(new_e[i]['uris'][i1])
    return old_e


if __name__ == "__main__":
    language = 'pt_BR'
    stemmer = RSLPStemmer() if language == 'pt_BR' else PorterStemmer() 

    with open('learning/treelstm/data/lc_quad/LCQuad_test.json', 'r', encoding='utf-8') as f:
        data = json.load(f)


    properties = preprocess_relations('dbpedia_3Por_property.ttl', stemmer, True)
    #print('properties: ', len(properties))

    fasttext_model = gensim.models.KeyedVectors.load_word2vec_format("./data/fasttext/wiki.pt.vec")
    stopwords = nltk.corpus.stopwords.words('portuguese')
    pronomes_interrogatorios =  ['que', 'quem', 'qual', 'quais', 'quanto', 'quanta', 'quantos', 'quantas', 'onde', 'quando', 'como', 'por que', 'liste']
    stopwords = stopwords + pronomes_interrogatorios
    properties_qakgpt = preprocess_relations_qakgpt('dbpedia_3Por_property.ttl', fasttext_model, stopwords, True)

    #with open('data\LC-QUAD\entity_lcquad - v0.json', 'r', encoding='utf-8') as f:
    #    lcquad_entities = json.load(f)

    # QASparql paper. QAKG in english
    #qasparql_entity_mapping_methods = ['earl', 'tagme', 'falcon', 'spotlight']
    #qasparql_relation_mapping_methods = ['earl','nliwod', 'falcon']

    # QAKGpt without ptRL
    #experiment = "without_ptrl"
    #qakgpt_entity_mapping_methods = ['spotlight', 'wikifier']
    #qakgpt_relation_mapping_methods = ['nliwod']

    # QAKGpt with ptRL
    experiment = "with_ptrl"
    qakgpt_entity_mapping_methods = ['spotlight', 'wikifier']
    qakgpt_relation_mapping_methods = ['nliwod', 'ptRL']    


    entity_mapping_methods = qakgpt_entity_mapping_methods
    relation_mapping_methods = qakgpt_relation_mapping_methods

    endpoint = 'https://dbpedia.org/sparql'
    kb = KB(endpoint)

    threshold = 0.85
    linked_data = []
    
    na_entity = []
    count = 0
    total = len(data)
    for q in data:
        query = q['question']
        mapped = {}
        mapped['question'] = query
        mapped['entities'] = []
        mapped['relations'] = []

        # EARL
        if ('earl' in entity_mapping_methods) | ('earl' in relation_mapping_methods):
            earl_e = get_earl_entities(query)
            if ('earl' in entity_mapping_methods) and len(earl_e['entities']) > 0:
                mapped['entities'] = merge_entity(mapped['entities'], earl_e['entities'])
            if ('earl' in relation_mapping_methods) and len(earl_e['relations']) > 0:
                mapped['relations'] = merge_entity(mapped['relations'], earl_e['relations'])
        
        # TAGME
        if ('tagme' in entity_mapping_methods):
            tagme_e = get_tag_me_entities(query)
            if len(tagme_e) > 0:
                mapped['entities'] = merge_entity(mapped['entities'], tagme_e)

        # NLIWOD
        if ('nliwod' in relation_mapping_methods):
            nliwod = get_nliwod_entities(query, properties, stemmer, language)
            if len(nliwod) > 0:
                mapped['relations'] = merge_entity(mapped['relations'], nliwod)

        # FALCON
        if ('falcon' in entity_mapping_methods) | ('falcon' in relation_mapping_methods) :
            falcon_e, falcon_r = get_falcon_entities(query)
            if ('falcon' in entity_mapping_methods):
                mapped['entities'] = merge_entity(mapped['entities'], falcon_e)
            if ('falcon' in relation_mapping_methods):
                mapped['entities'] = merge_entity(mapped['relations'], falcon_r)

        # DBPEDIA SPOTLIGHT
        if ('spotlight' in entity_mapping_methods):
            spot_en = get_spotlight_entities(query, 'en')
            if len(spot_en) > 0:
                mapped['entities'] = merge_entity(mapped['entities'], spot_en)

            if language == 'pt_BR':
                spot_pt = get_spotlight_entities(query, 'pt')
                if len(spot_pt) > 0:
                    for pt_entity in spot_pt:
                        en_entity = get_sameas(kb, pt_entity['uris'][0]['uri'])
                        if (en_entity != None and (en_entity != pt_entity['uris'][0]['uri'])):
                            #print(f"{pt_entity['uris'][0]['uri']} = {en_entity}")
                            new_entity = pt_entity
                            new_entity['uris'][0]['uri'] = en_entity
                            spot_pt.append(new_entity)

                    mapped['entities'] = merge_entity(mapped['entities'], spot_pt)
        
        # WIKIFIER
        if ('wikifier' in entity_mapping_methods):
            wikifier_en  = get_wikifier_entities(query, 'en', threshold)
            if len(wikifier_en) > 0:
                mapped['entities'] = merge_entity(mapped['entities'], wikifier_en)         

            if language == 'pt_BR':
                wikifier_pt  = get_wikifier_entities(query, 'pt', threshold)
                if len(wikifier_pt) > 0:
                    for pt_entity in wikifier_pt:
                        en_entity = get_sameas(kb, pt_entity['uris'][0]['uri'])
                        if (en_entity != None and (en_entity != pt_entity['uris'][0]['uri'])):
                            #print(f"{pt_entity['uris'][0]['uri']} = {en_entity}")
                            new_entity = pt_entity
                            new_entity['uris'][0]['uri'] = en_entity
                            wikifier_pt.append(new_entity)            
                    mapped['entities'] = merge_entity(mapped['entities'], wikifier_pt)  
                
        # PTRL
        if ('ptRL' in relation_mapping_methods and language == 'pt_BR'):
            ptRelations = get_qakgpt_relations(query, mapped['entities'], fasttext_model, stopwords, properties_qakgpt)
            if len(ptRelations) > 0:
                mapped['relations'] = merge_entity(mapped['relations'], ptRelations)         
            
        # Get entities with maximum confidence
        esim = []
        for i in mapped['entities']:
            i['uris'] = sorted(i['uris'], key=lambda k: k['confidence'], reverse=True)
            esim.append(max([j['confidence'] for j in i['uris']]))

        mapped['entities'] = np.array(mapped['entities'])
        esim = np.array(esim)
        inds = esim.argsort()[::-1]
        mapped['entities'] = mapped['entities'][inds]
        
        # Get relations with maximum confidence
        rsim = []
        for i in mapped['relations']:
            i['uris'] = sorted(i['uris'], key=lambda k: k['confidence'], reverse=True)
            rsim.append(max([j['confidence'] for j in i['uris']]))

        mapped['relations'] = np.array(mapped['relations'])
        rsim = np.array(rsim)
        inds = rsim.argsort()[::-1]
        mapped['relations'] = mapped['relations'][inds]

        mapped['entities'] = list(mapped['entities'])
        mapped['relations'] = list(mapped['relations'])

        #print(q['id'])
        count = count + 1
        print(f'{count} de {total}')
        linked_data.append(mapped)

    with open(f'data/LC-QUAD/entity_lcquad_test-{experiment}.json', "w", encoding="utf-8") as data_file:
        json.dump(linked_data, data_file, sort_keys=True, indent=4, separators=(',', ': '))
