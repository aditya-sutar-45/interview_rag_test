from glob import glob
import os
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import MarkdownHeaderTextSplitter

def load_all_md_file():
    markdown_paths = glob("./data/*.md")
    all_docs = []

    for path in markdown_paths:
        loader = TextLoader(path, encoding="utf-8")
        data = loader.load()

        splitter = MarkdownHeaderTextSplitter(headers_to_split_on=[("###", "question")])
        docs = splitter.split_text(data[0].page_content)

        for i, doc in enumerate(docs):
            doc.metadata["source"] = path
            doc.metadata["question_number"] = i + 1
            doc.metadata["language"] = os.path.basename(path).lower().replace(".md", "")
        
        all_docs.extend(docs)

    return all_docs