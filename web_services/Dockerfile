FROM python:3.7.5

RUN ["mkdir", "-p", "/usr/src/app"]

WORKDIR /usr/src/app

COPY requirements.txt requirements.txt

RUN ["pip3", "install", "--upgrade", "pip"]

RUN ["pip3", "install", "numpy"]

RUN ["pip3", "install", "-r", "requirements.txt"]

RUN ["pip3", "install", "textblob", "nltk"] 

RUN ["python3", "-m", "textblob.download_corpora"]

RUN [ "python3", "-c", "import nltk; nltk.download('stopwords')" ]

COPY . .
