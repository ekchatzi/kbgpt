# kbgpt

*kbgpt* is an command line interface to Large Language Models (LLM) that has the ability to use your own files and scrape websites. All you need is a python environment and an OpenAI API key.

It converts PDFs and HTML to text, converts them into chunks and creates embeddings from them. The embeddings are stored in a local database. Every time you search something, the most relevant chunks are pulled and sent as context to the LLM model along with the previous messages.

## API Key
Get an OpenAI [API Key](https://help.openai.com/en/articles/4936850-where-do-i-find-my-secret-api-key)

Set OPENAI_API_KEY env variable or create a .env file on the directory of `kbgpt`:
```
export OPENAI_API_KEY=sk-somethingsomething
```

### Install requirements
pip install -r requirements.txt 

## How to run
```
python kbgpt.py 
    --docs-dir DOCS_DIRS : where to keep generated data
    --user USER : namespace inside docs. Use to separate your documents
    --role ROLE : used on the prompt to set the tone of the LLM
    --scrape-depth SCRAPE_DEPTH : the maximum scrape depth
    --verbose, -v : print debug logs
```

## How to use
Just enter your prompts like you would on ChatGPT.

Special commands:
```
get WEBSITE : fetches the website and saves in the library
scrape WEBSITE : scrapes the website and saves in the library up to SCRAPE_DEPTH
add DOC1 DOC2 DOC3 URL1 ... : pull files into library 
use DOC1 DOC2 DOC3 URL1 ... : pull files into library and limit search only to these
clear : clear previous conversation and forced context set with `use`
nocontext : don't include anything from library, only use previous conversation
reload : process docs source directory again
```

It is possible to put many files at once into the library by putting them in DOCS_DIR/USER/source and the run `reload` or just restart. You can have subdirectories.
