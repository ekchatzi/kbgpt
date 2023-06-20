#!/bin/env python3

import argparse
import glob
import os
import shlex
import shutil
import time
import traceback
import math

import sqlite3
import hashlib
import json


import os
import re
import readline

import fitz
import html2text
import requests
from bs4 import BeautifulSoup
from urllib.parse import urlparse, urljoin

from sentence_transformers import SentenceTransformer
import numpy as np
import torch
import torch.nn.functional as F
import openai
import tiktoken
from urllib.parse import unquote

from dotenv import load_dotenv
load_dotenv()
openai.api_key = os.environ['OPENAI_API_KEY']

# TODO
# - doc files
# - better chunking, filtering


DEBUG_LOGS = False
def LOGD(*args, **kwargs):
    if not DEBUG_LOGS:
        return
    print(*args, **kwargs)

SCRAPE_DEPTH = 3

PATH = os.path.dirname(os.path.realpath(__file__))

GPTMODEL = "gpt-3.5-turbo"
TRUNCATE_LENGTH = 4096

# EMBED_MODEL = "text-embedding-ada-002"
# def encode_text(text):
#     response = openai.Embedding.create(
#         input=text,
#         model=EMBED_MODEL
#     )
#     embeddings = response['data'][0]['embedding']
#     return torch.tensor(embeddings)

ENCODE_MODEL = SentenceTransformer('all-MiniLM-L6-v2')
def encode_text(text):
    return ENCODE_MODEL.encode(text, convert_to_tensor=True)




KNOWLEDGE_BASES = {}
DB_CONNS = {}

def get_db_conn(output_dir):
    if output_dir in DB_CONNS:
        return DB_CONNS[output_dir]
    conn = sqlite3.connect(os.path.join(output_dir, 'data.db'))
    
    c = conn.cursor()
    c.execute("""
    CREATE TABLE IF NOT EXISTS tensors (
        key TEXT PRIMARY KEY,
        value BLOB
    )
    """)
    c.execute("""
    CREATE TABLE IF NOT EXISTS file_metadata  (
        key TEXT PRIMARY KEY,
        value TEXT
    )
    """)
    conn.commit()
    DB_CONNS[output_dir] = conn
    return DB_CONNS[output_dir]

def tensor_to_bytes(tensor):
    return tensor.cpu().numpy().tobytes()

def bytes_to_tensor(bytes):
    ten = torch.from_numpy(np.frombuffer(bytes, dtype=np.float32).copy())
    if torch.cuda.is_available():
        ten = ten.to('cuda')
    return ten

def save_to_knowledge_base(user, output_dir, path, tensor):
    if user in KNOWLEDGE_BASES:
        KNOWLEDGE_BASES[user][path] = tensor
    
    conn = get_db_conn(output_dir)
    c = conn.cursor()
    c.execute("INSERT OR REPLACE INTO tensors VALUES (?, ?)", (path, tensor_to_bytes(tensor)))
    conn.commit()


def load_user_knowledge_base(output_dir, user):
    ret = {}

    conn = get_db_conn(output_dir)
    c = conn.cursor()
    c.execute("SELECT key, value FROM tensors")
    for row in c:
        ret[row[0]] = bytes_to_tensor(row[1])
    return ret

def get_user_knowledge_base_map(output_dir, user):
    if user not in KNOWLEDGE_BASES:
        KNOWLEDGE_BASES[user] = load_user_knowledge_base(output_dir, user)
    return KNOWLEDGE_BASES[user]

def get_saved_file_metadata(user, output_dir, filename):
    conn = get_db_conn(output_dir)
    c = conn.cursor()
    c.execute("SELECT value FROM file_metadata WHERE key=?", (filename,))
    result = c.fetchone()
    if result is not None:
        return json.loads(result[0])
    return None

def set_saved_file_metadata(user, output_dir, filename, metadata):
    conn = get_db_conn(output_dir)
    c = conn.cursor()
    c.execute("INSERT OR REPLACE INTO file_metadata (key, value) VALUES (?,?)", (filename,json.dumps(metadata)))
    conn.commit()










def create_directories(docs_dir, user):
    os.makedirs(get_user_source_dir(docs_dir, user), exist_ok=True)
    os.makedirs(get_user_inter_dir(docs_dir, user), exist_ok=True)
    os.makedirs(get_user_output_dir(docs_dir, user), exist_ok=True)

def get_user_source_dir(docs_dir, user):
    return os.path.join(docs_dir, str(user), 'source')

def get_user_inter_dir(docs_dir, user):
    return os.path.join(docs_dir, str(user), 'inter')

def get_user_output_dir(docs_dir, user):
    return os.path.join(docs_dir, str(user), 'output')

def file_is_binary(file_path):
    text_chars = bytearray({7,8,9,10,12,13,27} | set(range(0x20, 0x100)) - {0x7f})
    is_binary_string = lambda bytes: bool(bytes.translate(None, text_chars))

    with open(file_path, 'rb') as f:
        return is_binary_string(f.read(1024))

def file_is_parseable(file_path):
    if file_path.endswith('.html') or file_path.endswith('.md') or file_path.endswith('.pdf'):
        return True
    return not file_is_binary(file_path) and (not file_path.endswith('.pem'))

def get_md5(file_path):
    hash_md5 = hashlib.md5()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()

def file_is_changed(user, file_path, inter_dir, output_dir):
    is_changed = True
    newmeta = {
        'md5': get_md5(file_path)
    }
    oldmeta = get_saved_file_metadata(user, output_dir, file_path)
    if oldmeta:
        is_changed = oldmeta['md5'] != newmeta['md5']
    
    return is_changed, newmeta


HTMLConverter = html2text.HTML2Text()
HTMLConverter.ignore_links = True
HTMLConverter.body_width = 0
def convert_file_to_text(full_path):
    try:
        if full_path.endswith('.pdf'):
            pages = []

            doc = fitz.open(full_path)
            for page in doc:
                text = page.get_text()
                pages.append(text)
            text = '\n\n-------\n\n'.join(pages)
            return text
        else:
            with open(full_path, 'r') as fp:
            
                text = fp.read()
                if full_path.endswith('html'):
                    text = HTMLConverter.handle(text)
                    text = re.sub(r'\[([^\]]+)\]\([^\)]+\)', r'\1', text)
            return text
    except Exception:
        traceback.print_exc()
    return ''

def split_text_by_line_start(text, delimiter):
    ret = []
    current_lines = []
    for line in text.split('\n'):
        print(line)
        if line.startswith(delimiter):
            ret.append('\n'.join(current_lines))
            current_lines = []
        current_lines.append(line)
    if len(current_lines):
        ret.append('\n'.join(current_lines))
    return ret

def convert_text_to_documents(text, full_path='.txt'):
    ret = []
    texts = []
    if full_path.endswith('.py'):
        texts = split_text_by_line_start(text, 'def ')
    elif full_path.endswith('.pdf'):
        texts = split_text_by_line_start(text, '-------')
    elif full_path.endswith('.js'):
        texts = split_text_by_line_start(text, 'function ')
    else:
        texts = [t for t in text.split('\n\n') if not t.isspace()]
    return [re.sub(r'^\s+', '', re.sub(r'\s+$', '', t)) for t in texts]
   
def process_doc(user, full_path, source_dir, inter_dir, output_dir):
    LOGD(f'Processing {full_path} for user={user} inter_dir={inter_dir}')
    text = convert_file_to_text(full_path)
    docs = convert_text_to_documents(text, full_path)
    for i, d in enumerate(docs):
        inter_text_path = os.path.join(inter_dir, full_path.replace(source_dir+'/', '').replace(' ','_').replace('+','-')+'+'+str(i)+'+.text')
        os.makedirs(os.path.dirname(inter_text_path), exist_ok=True)
        with open(inter_text_path, 'w') as fp:
            fp.write(d)
        tensor = encode_text(d)
        save_to_knowledge_base(user, output_dir, inter_text_path, tensor)
    

def process_docs(user, source_dir, inter_dir, output_dir):
    print(f'Processing docs for user={user}, source={source_dir}, inter={inter_dir}, output={output_dir}')

    start = time.time()
    for root, dirs, files in os.walk(source_dir):
        for file in files:
            full_path = os.path.join(root, file)
            if not file_is_parseable(full_path):
                LOGD(f'File {full_path} is not parseable, ignoring..')
                continue

            rel_path = full_path.replace(source_dir+'/', '')
            is_changed, metadata = file_is_changed(user, full_path, inter_dir, output_dir)
            if not is_changed:
                LOGD(f'File {full_path} is not CHANGED, ignoring..')
                continue

            process_doc(user, full_path, source_dir, inter_dir, output_dir)
            set_saved_file_metadata(user, output_dir, full_path, metadata)
    end = time.time()
    print(f'Processing docs for user={user} took {end - start} seconds')

def process_query(docs_dir, user, query):
    return encode_text(query)

def split_knowledge_base_to_tensors_and_indices(knowledge_base, filter=None):
    tensors = []
    indices = []
    for key, item in knowledge_base.items():
        if filter and not key in filter:
            continue

        tensors.append(item)
        indices.append(key)
    if len(tensors) == 0:
        return [],[]
    return torch.stack(tensors), indices

def get_doc_text(docs_dir, user, lookup):
    try:
        with open(lookup, 'r') as fp:
            return fp.read()
    except Exception:
        pass
    return ''

encoding = tiktoken.encoding_for_model(GPTMODEL)

def get_token_count(text):
    num_tokens = len(encoding.encode(text))
    return num_tokens

def truncate_text(text, length):
    tokens = encoding.encode(text)
    truncated_tokens = tokens[:length]
    truncated_text = ""
    for token in truncated_tokens:
        if token.startswith("Ġ"):
            truncated_text += token[1:]  # Add space before the token
        else:
            truncated_text += token  # No space needed

    return truncated_text

def find_documents(docs_dir, user, pquery, token_limit, filter=None):
    ret = []
    kb_map = get_user_knowledge_base_map(get_user_output_dir(docs_dir, user), user)
    tensors, indices = split_knowledge_base_to_tensors_and_indices(kb_map, filter)
    if len(tensors) == 0:
        LOGD('No matching library tensors')
        return []
    similarities = F.cosine_similarity(pquery.reshape(1, -1), tensors).cpu()

    LOGD(f'Looking for matches in {indices}')

    top_k = 10
    top_k_indices = torch.argsort(similarities, descending=True)[0:top_k]

    total_token_count = 0
    extra_docs = []
    doc_tensors = []
    for index in reversed(top_k_indices):  # Reverse to start with most similar
        doc_tensor = tensors[index]
        too_similar = False
        # if len(doc_tensors) != 0:
        #     self_similarities = F.cosine_similarity(doc_tensor.reshape(1, -1), torch.stack(doc_tensors))
        #     if (self_similarities > 0.9).any():
        #         print('Skipping because it is too similar')
        #         too_similar = True
        #         continue

        if not too_similar:
            doc_text = get_doc_text(docs_dir, user, indices[index])
            # LOGD(f"Similarity: {similarities[index]}")
            doc_tensors.append(doc_tensor)
            
            if len(doc_text) == 0:
                continue

            token_count = get_token_count(doc_text)
            # LOGD(f'doc_text={doc_text} token_count={token_count} total_count={total_token_count} Similarity: {similarities[index]}')
        
            if total_token_count + token_count > token_limit:
                LOGD('Maximum tokens reached, ending search')
                continue

            total_token_count += token_count
            ret.append(doc_text)

            for d in [1,2,3]:
                file = indices[index]
                parts = file.split('+')
                fileno = parts[1]
                side_doc = parts[0]+str(int(fileno)+d)+parts[2]
                if len(get_doc_text(docs_dir, user, side_doc)) > 0:
                    extra_doc.append(side_doc)

    if total_token_count < token_limit:
        for extra_doc in extra_docs:
            doc_text = get_doc_text(docs_dir, user, extra_doc)
            token_count = get_token_count(doc_text)
            if total_token_count + token_count > token_limit:
                LOGD('Maximum tokens reached, ending adding extra texts')
                break
            total_token_count += token_count
            ret.append(doc_text)
    return ret


def prepare_messages_with_documents(docs, previous_conversation, max_length=TRUNCATE_LENGTH, role="helpful assistant"):
    context = '\n-------\n\n\n\n'.join(docs)
    LOGD(context)
    messages = [
        {"role": "system", "content": "You are a "+role+". Some context:"+context},
    ]

    total_token_count = 0
    n = len(previous_conversation)
    for i in range(0, int(n/2)):
        token_count = get_token_count(previous_conversation[i*2]['content']) + get_token_count(previous_conversation[i*2+1]['content'])
        if token_count + total_token_count > max_length:
            break

        messages.append(previous_conversation[i*2])
        messages.append(previous_conversation[i*2+1])
        total_token_count += token_count
    if total_token_count > TRUNCATE_LENGTH:
        messages[0]['content'] = truncate_text(messages[0]['content'], TRUNCATE_LENGTH)
    return messages

def send_query(messages, query):
    messages.append({'role':'user','content':query})
    LOGD(messages)
    response = openai.ChatCompletion.create(
        model=GPTMODEL,
        messages=messages
    )
    return response
    

FORCED_CONTEXT = []
def clear_forced_context():
    global FORCED_CONTEXT
    FORCED_CONTEXT = []

def use_nocontext_forced_context():
    global FORCED_CONTEXT
    FORCED_CONTEXT = ['<NOCONTEXT>']

def add_files(docs_dir, user, files, force_context=False):
    if force_context:
        clear_forced_context()

    source_dir = get_user_source_dir(docs_dir, user)
    
    for file in files:
        if len(file) == 0:
            use_nocontext_forced_context()
            continue

        if file.startswith('http'):
            source_filename = get_website(docs_dir, user, file)
            print(f'Using file {source_filename}')
        else:
            ffile = file[1:] if file[0] == '/' else file
            source_filename = os.path.join(source_dir, ffile.replace(source_dir+'/', '').replace(' ','_').replace('+','-')) 
        
            print(f'Using file {file} ({source_filename})')
            os.makedirs(os.path.dirname(source_filename), exist_ok=True)
            shutil.copyfile(file, source_filename)
        if force_context or (len(FORCED_CONTEXT) > 0 and '<NOCONTEXT>' not in FORCED_CONTEXT):
            FORCED_CONTEXT.append(source_filename)
    init_for_user(docs_dir, user)
        

def get_filter_for_forced_context(docs_dir, user):
    if not FORCED_CONTEXT or len(FORCED_CONTEXT) == 0:
        return None
    
    if '<NOCONTEXT>' in FORCED_CONTEXT:
        return set(['<NOCONTEXT>'])
    
    ret = set()
    inter_dir = get_user_inter_dir(docs_dir, user)
    source_dir = get_user_source_dir(docs_dir, user)
    for file in FORCED_CONTEXT:
        inter_text_pattern = os.path.join(inter_dir, file.replace(source_dir+'/', '').replace(' ','_').replace('+','-')+'+*+.text')
        for docs_name in glob.glob(inter_text_pattern):
            lookup = docs_name.replace(source_dir+'/', '')
            ret.add(lookup)
    return ret


def process_anwser(user, query, ans, previous_conversation):
    LOGD(ans)
    message = ans['choices'][0]['message']['content']
    message = re.sub(r'[^.]*(AI language model|artificial intelligence language model),[^.]*[.!]( However, )?', '', message).strip()
    message = message[0].upper() + message[1:]
    previous_conversation.append({'role':'user', 'content': query})
    previous_conversation.append({'role':'assistant', 'content': message})
    return message

def anwser_query(docs_dir, user, query, previous_conversation, role="helpful assistant"):
    pquery = process_query(docs_dir, user, query)
    filter = get_filter_for_forced_context(docs_dir, user)
    docs = find_documents(docs_dir, user, pquery, (TRUNCATE_LENGTH - get_token_count(query))*0.6, filter)
    messages = prepare_messages_with_documents(docs, previous_conversation, TRUNCATE_LENGTH - get_token_count(query), role)
    ans = send_query(messages, query)
    ans = process_anwser(user, query, ans, previous_conversation)
    return ans    

def eval(docs_dir, user, query, previous_conversation, role="helpful assistant"):
    return anwser_query(docs_dir, user, query, previous_conversation, role)


def is_valid_url(url):
    parsed = urlparse(url)
    return bool(parsed.netloc) and bool(parsed.scheme) and (parsed.scheme == 'http' or parsed.scheme == 'https')


def scrape(url):
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/89.0.4389.82 Safari/537.36",
        "Accept-Language": "en-US,en;q=0.5",
        "Accept": "text/html,application/xhtml+xml,application/xml,application/pdf;q=0.9",
        "Cache-Control": "no-cache",
        # "Cookie: gdpr-cookie-notice=read; ph_phc_necIgLhdqRwJoaHaOHsuspAxndBYxRSGzhgMw4bNUX3_posthog=%7B%22distinct_id%22%3A%2218829bcc31535ab-06a6cca699f505-15462c6c-1fa400-18829bcc3166b52%22%2C%22%24device_id%22%3A%2218829bcc31535ab-06a6cca699f505-15462c6c-1fa400-18829bcc3166b52%22%2C%22%24referrer%22%3A%22https%3A%2F%2Fappmanager.gfi.com%2Forganization%2F34%2Fdashboard%2F34%2Finstances%2Fbff1e1ba-340f-4d24-a006-03bda79634c6%2Fmanage%2Fconfiguration%22%2C%22%24referring_domain%22%3A%22appmanager.gfi.com%22%2C%22%24sesid%22%3A%5B1685469334035%2C%221886dcd0a131aca-0a251b48d77e22-15462c6c-1fa400-1886dcd0a1430b4%22%2C1685469334035%5D%2C%22%24session_recording_enabled_server_side%22%3Atrue%2C%22%24active_feature_flags%22%3A%5B%5D%2C%22%24enabled_feature_flags%22%3A%7B%7D%7D; ph_POSTHOG_INCORRECT_PROJECT_KEY_posthog=%7B%22distinct_id%22%3A%22lefteris.chatzipetrou%40gfi.com%22%2C%22%24device_id%22%3A%2218829be09c77ea-01ffd006687128-15462c6c-1fa400-18829be09c84154%22%2C%22%24referrer%22%3A%22https%3A%2F%2Fappmanager.gfi.com%2F%3Fstate%3Da27dcafe1f2049e6b3ded1dae6480889%26session_state%3D4635670c-bcba-4a47-b27d-4180fbaf2be7%26code%3D3bcc9ed8-d43a-43d8-b6d2-70da39ada6ce.4635670c-bcba-4a47-b27d-4180fbaf2be7.330d060e-19dd-441e-8a55-9fc21645de2c%22%2C%22%24referring_domain%22%3A%22appmanager.gfi.com%22%2C%22%24sesid%22%3A%5B1686314774293%2C%22188a0316f16279f-0d385fe996849-15462c6c-1fa400-188a0316f17300e%22%2C1686314774293%5D%2C%22%24user_id%22%3A%22lefteris.chatzipetrou%40gfi.com%22%7D; CONCRETE=411391bd605c19665b894400207a9df3; BIGipServerGFI-Concrete5=885535498.20480.0000; colossus#lang=en; _gid=GA1.2.985898611.1686651650; _ga_JP8QT0RF3B=GS1.1.1686654223.33.1.1686654463.0.0.0; ph_phc_36r1FnsemYx7dsOeLOymk9HPXEG0QRaWCXKmaAeSdOB_posthog=%7B%22distinct_id%22%3A%22lefteris.chatzipetrou%40gfi.com%22%2C%22%24device_id%22%3A%22185216934fc33f-030aaec9a4d9c6-1f462c6d-1fa400-185216934fd3e5d%22%2C%22%24referrer%22%3A%22https%3A%2F%2Fappmanager-dev.gfi.com%2Forganization%2F3%2Fdashboard%2F5%2Finstances%2F8ec1ffae-b3ed-4ea4-a5ea-794bbd37ef57%2Fmanage%2Fconfiguration%3FproductUrl%3DaHR0cHM6Ly9hcHBtYW5hZ2VyLWRldi5nZmkuY29tL2tjb250cm9sLw%3D%3D%22%2C%22%24referring_domain%22%3A%22appmanager-dev.gfi.com%22%2C%22%24sesid%22%3A%5B1686654464262%2C%22188b470ae361e62-0f18fffc17409-15462c6c-1fa400-188b470ae3740d3%22%2C1686654463542%5D%2C%22%24session_recording_enabled_server_side%22%3Afalse%2C%22%24active_feature_flags%22%3A%5B%5D%2C%22%24enabled_feature_flags%22%3A%7B%7D%2C%22%24user_id%22%3A%22lefteris.chatzipetrou%40gfi.com%22%2C%22%24search_engine%22%3A%22google%22%7D; _ga_4EBH4LEWH3=GS1.1.1686659999.2.0.1686659999.0.0.0; _ga=GA1.2.1648977014.1685546275; _gat_UA-52423314-1=1
        "Pragma": "no-cache",
        "sec-ch-ua": '"Chromium";v="112", "Google Chrome";v="112", "Not:A-Brand";v="99"',
        "sec-ch-ua-mobile": "?0",
        "sec-ch-ua-platform": "Linux",
        "Sec-Fetch-Dest": "document",
        "Sec-Fetch-Mode": "navigate",
        "Sec-Fetch-Site": "none",
        "Sec-Fetch-User": "?1",
        "Upgrade-Insecure-Requests": "1",
    }

    urls = set()
    domain_name = urlparse(url).netloc
    LOGD(f"Getting all links for {url}")
    r = requests.get(url, headers=headers, timeout=5)
    LOGD(f"Request returned")
    content = r.text
    soup = BeautifulSoup(content, "html.parser")
    LOGD(f"Got soup for {url}")
    for a_tag in soup.findAll("a"):
        href = a_tag.attrs.get("href")
        if href == "" or href is None:
            continue
        # LOGD(href)

        href = urljoin(url, href)
        parsed_href = urlparse(href)
        href = parsed_href.scheme + "://" + parsed_href.netloc + parsed_href.path

        if not is_valid_url(href):
            continue
        if href in urls:
            continue
        if domain_name not in href:
            continue
        urls.add(href)
    return urls, content


def save_html_page(url, content, directory):
    parsed_link = urlparse(url)
    file_path = os.path.join(directory, parsed_link.netloc + unquote(parsed_link.path))
    if not file_path.endswith('.pdf') and not file_path.endswith('.html'):
        file_path += '.html'
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, 'w') as f:
        f.write(content)
    return file_path

def download_html_pages(url, directory, max_depth=SCRAPE_DEPTH, visited=set(), depth=0):
    if depth > max_depth:
        return
    
    if not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)

    print(f'Downloading {url}')
    visited.add(url)
    try:
        all_links, content = scrape(url)
        file_path = save_html_page(url, content, directory)

        for link in all_links:
            if link in visited:
                continue
            visited.add(link)
            download_html_pages(link, directory, max_depth, visited, depth+1)
        return file_path
    except Exception as e:
        print(f'Failed to fetch {url}: {e}')


def get_website(docs_dir, user, www):
    print(f'Getting website {www}')
    if not www.startswith('http'):
        www = 'https://'+www
    ret = download_html_pages(www, os.path.join(get_user_source_dir(docs_dir, user), www.replace('https://','').replace('http://','').replace('/','_')), max_depth=0)
    init_for_user(docs_dir, user)        
    return ret

def scrape_website(docs_dir, user, www):
    print(f'Scraping website {www}')
    if not www.startswith('http'):
        www = 'https://'+www
    ret = download_html_pages(www, os.path.join(get_user_source_dir(docs_dir, user), www.replace('https://','').replace('http://','').replace('/','_')))
    init_for_user(docs_dir, user)
    return ret

def repl_get_args(text):
    return shlex.split(text)

def repl_loop(docs_dir, user, role):
    quit = False
    previous_conversation = []
    code = ''
    while not quit:
        try:
            code = ''
            code = input("\033[0m>>> \033[1;33m").strip()
            print('\033[0m')
            if code == 'quit' or code == 'q' or code == 'exit' or code == 'e':
                quit = True
                break

            if code == 'reload':
                init_for_user(docs_dir, user)
            elif code.startswith('clear'):
                previous_conversation = []
                clear_forced_context()
            elif code.startswith('scrape '):
                scrape_website(docs_dir, user, code.split('scrape ')[1])
            elif code.startswith('get '):
                get_website(docs_dir, user, code.split('get ')[1])
            elif code.startswith('use '):
                add_files(docs_dir, user, repl_get_args(code.split('use ')[1]), force_context=True)
            elif code.startswith('add '):
                add_files(docs_dir, user, repl_get_args(code.split('add ')[1]), force_context=False)
            elif code.startswith('nocontext'):
                use_nocontext_forced_context()
            else:
                result = eval(docs_dir, user, code, previous_conversation, role)
                print('\033[1;32m')
                print(result)
                print('\033[0m')
        except Exception as e:
            print("Error:", str(e))
            traceback.print_exc()
            code = ''
        except KeyboardInterrupt as k:
            if len(code) == 0:
                quit = True
            code = ''
    
def init_for_user(docs_dir, user):
    LOGD(f"Initializing for user={user}, docs_dir={docs_dir}")

    create_directories(docs_dir, user)
    process_docs(user, get_user_source_dir(docs_dir, user), get_user_inter_dir(docs_dir, user), get_user_output_dir(docs_dir, user))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Search with GPT')
    parser.add_argument('--docs-dir','-d', default=os.path.join(PATH, 'docs'))
    parser.add_argument('--user','-u', type=str, default='0')
    parser.add_argument('--scrape-depth', type=int, default=3)
    parser.add_argument('--role','-r', type=str, default='helpful assistant')
    parser.add_argument('--verbose','-v', action='store_true', default=False)
    args = parser.parse_args()
    
    DEBUG_LOGS = args.verbose
    SCRAPE_DEPTH = args.scrape_depth
    init_for_user(args.docs_dir, args.user)
    repl_loop(args.docs_dir, args.user, args.role)
