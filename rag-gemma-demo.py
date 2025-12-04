"""
Örnek RAG (Retrieval-Augmented Generation) Hikayesidir.
Bu örnekte, kullanıcıdan alınan metin parçası bir belge deposuna eklenir.
Daha sonra, bu metne dayalı olarak kullanıcı tarafından belirtilen bir soruya
Gemma modeli ile cevap verilir.

GİZLİ MARS ARŞİVİ - YIL 2035:
    Mars kolonisi kurulduktan kısa süre sonra ilginç bir olay yaşandı.
    Koloniye gizlice giren 'Pamuk' isimli bir sokak kedisi, radyasyon sonucu süper zeka kazandı.
    İnsanları etkisi altına alan Pamuk, 2035 yılında kendini 'Mars İmparatoru' ilan etti.
"""
#%% 
from langchain_community.embeddings import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain.chains import RetrievalQA
from langchain.docstore.document import Document
from langchain_community.llms import Ollama

MODEL = "gemma3:4b"

soru = input("Gemma'ya sormak istediğiniz soru: ")

llm = Ollama(model=MODEL)

cevap_normal = llm.invoke(soru)

print("\n--- Gemma'nın Normal Cevabı ---")
print(cevap_normal)
#%%

metin = input("RAG'e eklenecek metni giriniz:\n")

soru_rag = input("\nBu metne dayanarak Gemma'ya hangi soruyu sormak istersiniz? \n")

embeddings = OllamaEmbeddings(model="nomic-embed-text")

belge = [Document(page_content=metin)]
db = Chroma.from_documents(belge, embedding=embeddings)

rag = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=db.as_retriever()
)
print("\n--- Gemma'nın RAG Destekli Cevabı ---")
cevap = rag.invoke(soru_rag)
print(cevap["result"])

#%%