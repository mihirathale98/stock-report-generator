import torch
import uvicorn
from transformers import AutoModelForCausalLM, AutoTokenizer, StoppingCriteria, StoppingCriteriaList, AutoConfig
from fastapi import FastAPI
from pyngrok import ngrok
from transformers import LlamaForCausalLM, LlamaTokenizer

app = FastAPI()

# model_ckpt = 'meta-llama/Llama-2-7b-chat-hf'
# hf_model_path = 'meta-llama/Llama-2-7b-chat-hf'
# tokenizer = AutoTokenizer.from_pretrained(hf_model_path, trust_remote_code=True)
# config = AutoConfig.from_pretrained(hf_model_path, trust_remote_code=True, device_map="auto")
# config.init_device = "cpu"
# config.max_seq_len = 4096
# device = 'cpu'
# #config.attn_config['attn_impl'] = 'triton'
# model = AutoModelForCausalLM.from_pretrained(model_ckpt, config=config, trust_remote_code=True,
#                                              torch_dtype=torch.bfloat16)
#
# # llama_config_path = ''
# config = AutoConfig.from_pretrained(llama_config_path, trust_remote_code=True, device_map="auto")
# tokenizer = LlamaTokenizer.from_pretrained("/output/path")
# model = LlamaForCausalLM.from_pretrained("/output/path")


class StoppingCriteriaSub(StoppingCriteria):

    def __init__(self, stops=[], encounters=1):
        super().__init__()
        self.stops = stops
        self.ENCOUNTERS = encounters

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor):
        stop_count = 0
        for stop in self.stops:
            stop_count = (stop == input_ids[0]).sum().item()

        if stop_count >= self.ENCOUNTERS:
            return True
        return False


stopping_criteria = StoppingCriteriaList([StoppingCriteriaSub(stops=[0], encounters=1)])


@app.post("/qa")
async def search(request: dict):
    '''
    Search api to get Response

    Args :
        request(dict): post request for search api
    '''
    query = request['query']
    max_new_tokens = int(request['max_new_tokens'])
    print(query)
    #
    # inputs = tokenizer(query, return_tensors='pt').to('cpu')
    # response = model.generate(**inputs, max_new_tokens=max_new_tokens, stopping_criteria=stopping_criteria)
    # response = tokenizer.decode(response[0])
    # return {'answer': response}


if __name__ == '__main__':
    config = uvicorn.Config("reader_api:app", port=8001, workers=2)
    server = uvicorn.Server(config)
    ngrok.set_auth_token(input("Enter ngrok auth token: "))
    public_url = ngrok.connect(8001, bind_tls=True)
    print(public_url)
    server.run()
    ngrok.disconnect(public_url.public_url)
