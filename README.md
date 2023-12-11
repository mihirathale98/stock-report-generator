# Stock Report Generator (CS5100 Project)

This repository is an implementation of the project "Stock Report Generator" for CS5100(Foundations of Artificial
Intelligence).

The code in this repository showcases how NLP and LLMs can be used to analyze Earnings Call Transcripts to find out
hidden insights and present the user with a summary report backed by evidence so that they can make informed decisions
about investing in the stock market.

## Setup

1. Make sure you have python>=3.7 installed
2. run `pip install -r requirements.txt`
3. Make sure all the packages were installed correctly
4. Make sure to have a GPU capable of loading a 7B parameter model

## Components

1. **Earning Call API**
    * We use `discountingcashflows` api to get the earnings call transcripts of publicly traded companies.
    * To get the transcripts of the required companies open the `fin_api.py` file and list the tickers of the companies
      required and then run the file.
    * The code will fetch the results for the mentioned years for each quarted and appropriately save the results in
      the `earnings_call_transcripts` directory.


2. **Text Encoder**
    * We use a transformer model based on the BERT architecture to encode the earnings call transcripts.
    * To encode the transcripts, we first use NLTK to sent tokenize the transcripts into dialogues.
    * Then we create a chain of dialogues such that each chain is under 512 tokens which is basically the max_length of
      tokens that the encoder model can handle.
    * We then use the encoder to encode the tokenized passages into embedding vectors, where each vector is of 768
      dimensions (i.e [1,768]). We use batched operations to encode the transcripts faster.
    * We then save the encodings into a npy file which can be loaded later into the index.


3. **Indexer**
    * We use FAISS(Facebook AI's Similarity Search) to index and retrieve the most relevant documents for the query.
    * To build the index and start the api, run 'retriever_api.py' with the correct path to the npy file saved in the
      previous step.
    * To retrieve the relevant passages, we first index the query using the same encoder used to index the passages.
    * Then we feed the encoded query to FAISS search, which performs cosine_similarity search over all the documents in
      the index and retrieves top k passages for the user.
    * Then we reverse map the passage ids to the original passages. We create passage to index map in the previous
      encoding step itself.
    * The reverse mapped passages and then further used to get the final answer from the LLM.


4. **NER**
    * We implement NER to recognize entities which are financially relevant so that using them we can guide the LLM to
      generate more specific answer.
    * For the Custom LM approach, we use BERT for token classification and try to train it on FINER-139 dataset. We are
      not able to achieve decent performance with this as the dataset is really huge(900000 samples) and due to time
      restrictions on compute we were not able to train it on the whole dataset.
    * So we try using a BERT finetuned for generic NER, implemented using the ner pipeline from huggingface. This implementation can easily load a finetuned model and make predictions with it.
    * To run the NER api just run the `ner_api.py` file.
    * We make NER step optional in our solution. To get more entity driven answer we can enable NER and use the
      entities to make LLM generate a more focused answer.


5. **Reader**
    * The most important component of our solution is the reader.
    * Reader takes in all the information(query, passages, prompt, entities) and generates a relevant answer for the
      user, based on which they can make an informed decision.
    * We use huggingface(transformers) library to implement the reader, which is basically a LLM(decoder only for this
      case).
    * We tried out two reader - MPT-7b-Instruct and LLama-7b-Chat.
    * We were able to achieve better quality results with the llama model in our demo, hence we chose to stick with it.
    * To start the reader api, just run the `reader_api.py` file. Make sure you are on a device capable of handling a 7B
      parameter model(atleast 2 T4s).


6. **Demo**
    * To build the demo we use streamlit and we also add OpenAI api to compare the results from the smaller 7B LLM to
      GPT-3.5-turbo(175B)
    * To run the demo just run `streamlit run sample-ui.py`
    * Make sure to plug in the api key for OpenAI, to make the demo work.

## Conclusion and Future Work

1. We are able to generate reports which are highly relevant to the retrieved passages. These reports can be useful for the user along with the relevant evidence present in our demo.
2. We can scale up this solution by indexing more earnings calls of various companies. We can even make the indexing process dynamic by adding support for the user to choose the company from the front end.
3. To improve NER we can fine tune the NER model on a custom dataset of the financial domain(more specific towards stock market), and use that NER model to find the relevant entities and then feed them into the LLM for a better answer.
4. We can even finetune the LLM over synthetically generated data, so that the responses are highly specific to the format we want to provide to the user and to make the LLM follow the custom instruction in a better way. Fine-tuning might also help LLM adopt to the financial domain.

## Contributors

* Mihir Athale
* Sai Gautham Yadavalli
* Daniel Chinn
* Fabian Christopher