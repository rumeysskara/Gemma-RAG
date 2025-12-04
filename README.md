# Integrating Gemma into RAG Pipelines 🚀

Bu proje, Google'ın **Gemma** modellerini yerel ortamda (Ollama kullanarak) çalıştırıp, **LangChain** ve **ChromaDB** kullanarak basit bir RAG (Retrieval-Augmented Generation) hattının nasıl kurulacağını gösteren bir demodur.

Özellikle bu alana yeni başlayanlara; standart bir LLM cevabı ile RAG destekli cevap arasındaki farkı somut bir şekilde göstermeyi amaçlar.

## 📊 Sunum

Projenin teorik altyapısını ve mimari detaylarını içeren sunuma aşağıdaki bağlantıdan ulaşabilirsiniz:

👉 **[Canva Sunumu: Integrating Gemma into RAG Pipelines](https://www.canva.com/design/DAG5cMLunI4/UDbW1tlubl1zOw318Mfgdg/view?utm_content=DAG5cMLunI4&utm_campaign=designshare&utm_medium=link2&utm_source=uniquelinks&utlId=hf89e05b805)**

---

## 🛠️ Kurulum ve Ön Hazırlık

Bu projeyi çalıştırmak için bilgisayarınızda **Python** ve **Ollama** kurulu olmalıdır.

### 1. Ollama Kurulumu ve Modellerin Çekilmesi
Öncelikle [Ollama.com](https://ollama.com/) adresinden uygulamanın işletim sisteminize uygun sürümünü indirin. Ardından terminal üzerinden bu projede kullanılan LLM (`gemma3:4b`) ve Embedding (`nomic-embed-text`) modellerini indirin:

```bash
ollama pull gemma3:4b
ollama pull nomic-embed-text
