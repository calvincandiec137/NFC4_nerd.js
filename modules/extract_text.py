import pymupdf
from langchain_text_splitters import RecursiveCharacterTextSplitter

PATH = "./database/sample_document.pdf"

structured = []


def main():

    doc = pymupdf.open(PATH)

    text = ""

    for page in doc:
        text+= page.get_text()
    
    # print(text)

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=0)
    texts = text_splitter.split_text(text)
    
    #print(texts)
    for i, chunk in enumerate(texts):
        structured.append({
            "id": i,
            "text": chunk,
            "source": PATH.split("/")[-1],
            "chunk_index": i
        })

if __name__=="__main__":
    main()