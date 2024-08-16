import os
import openai
import pandas as pd
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone, ServerlessSpec, PodSpec
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter
import streamlit as st


class PineconeManager:
    def __init__(self, api_key, openai_key, index_name, dimension=1536, use_serverless=True):
        self.api_key = api_key
        self.openai_key = openai_key
        self.index_name = index_name
        self.dimension = dimension
        self.use_serverless = use_serverless

        self._initialize_pinecone()
        self._initialize_openai()
        self._create_index()

    def _initialize_pinecone(self):
        os.environ['PINECONE_API_KEY'] = self.api_key
        
        self.pc = Pinecone(api_key=self.api_key)
        
        if self.use_serverless:
            self.spec = ServerlessSpec(cloud='aws', region='us-east-1')
        else:
            self.spec = PodSpec()

    def _initialize_openai(self):
        openai.api_key = self.openai_key
        self.embeddings = OpenAIEmbeddings(
            model="text-embedding-ada-002",
            openai_api_key=self.openai_key
        )

    def _create_index(self):
        # if self.index_name in Pinecone.list_indexes():
        if self.index_name in self.pc.list_indexes().names():
            self.pc.delete_index(self.index_name)

        self.pc.create_index(
            self.index_name,
            dimension=self.dimension,
            metric='cosine',
            spec=self.spec
        )
        self.index = self.pc.Index(self.index_name)

    def upsert_dataframe(self, df, text_field='content', batch_size=100):

        df[text_field].to_csv('dataset/data/vectors.txt', index=False, header=False)

        loader = TextLoader("dataset/data/vectors.txt")

        documents = loader.load()

        text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=2)

        docs = text_splitter.split_documents(documents)

        vectorstore_from_docs = PineconeVectorStore.from_documents(
                                                            docs,
                                                            index_name=self.index_name,
                                                            embedding=self.embeddings
                                                            ,
                                                        
                                                        )

    def query(self, query_text, top_k=3):
        vectorstore = PineconeVectorStore(
            self.index, self.embeddings
        )
        return vectorstore.similarity_search(query_text, k=top_k)

    def delete_index(self):
        self.pc.delete_index(self.index_name)






