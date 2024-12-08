from langchain.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from dotenv import load_dotenv


class FakeConfiguration:
    def __init__(self):
        self.openai_api_key = "fake"





def setup():
    config = FakeConfiguration()
    return config

def load_data():
    pass

def transform_data():
    pass

def create_embeddings():
    pass

def create_index():
    pass

def store_data():
    pass

def query():
    pass



def main():
    config = setup()

    print("Hello, World!")


if __name__ == "__main__":
    load_dotenv()
    main()