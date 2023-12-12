import torch
import uvicorn
from transformers import AutoModelForCausalLM, AutoTokenizer, StoppingCriteria, StoppingCriteriaList, AutoConfig
from fastapi import FastAPI

# Initialize the FastAPI app
app = FastAPI()

# Load the model
model_ckpt = 'meta-llama/Llama-2-7b-chat-hf'
hf_model_path = 'meta-llama/Llama-2-7b-chat-hf'
tokenizer = AutoTokenizer.from_pretrained(hf_model_path, trust_remote_code=True)
config = AutoConfig.from_pretrained(hf_model_path, trust_remote_code=True, device_map="auto")
config.init_device = "cuda"
config.max_seq_len = 4096
device = 'cuda'
model = AutoModelForCausalLM.from_pretrained(model_ckpt, config=config, trust_remote_code=True,
                                              torch_dtype=torch.bfloat16)

class StoppingCriteriaSub(StoppingCriteria):
    """
    Stopping criteria for stopping generation when a token is encountered a certain number of times in the output sequence

    """
    def __init__(self, stops=[], encounters=1):
        """
        Initialize the stopping criteria

        Args:
            stops (list): List of tokens to stop generation
            encounters (int): Number of times to encounter the token before stopping generation

        """
        super().__init__()
        self.stops = stops
        self.ENCOUNTERS = encounters

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor):
        """
        Check if the stopping criteria is met for the given input_ids and scores and return True if it is met else False

        Args:
            input_ids (torch.LongTensor): Input ids for the current step
            scores (torch.FloatTensor): Scores for the current step

        Returns:
            bool: True if the stopping criteria is met else False
            
        """
        stop_count = 0
        for stop in self.stops:
            stop_count = (stop == input_ids[0]).sum().item()

        if stop_count >= self.ENCOUNTERS:
            return True
        return False


stopping_criteria = StoppingCriteriaList([StoppingCriteriaSub(stops=tokenizer.eos_token_id, encounters=1)])


@app.post("/qa")
async def search(request: dict):
    '''
    Search api to get Response

    Args :
        request(dict): post request for search api
    '''
    query = request['query']
    max_new_tokens = int(request['max_new_tokens'])

    inputs = tokenizer(query, return_tensors='pt').to('cuda')
    response = model.generate(**inputs, max_new_tokens=max_new_tokens, stopping_criteria=stopping_criteria)
    response = tokenizer.decode(response[0])
    return {'answer': response}


if __name__ == '__main__':
    config = uvicorn.Config("reader_api:app",port=8001, workers=2)
    server = uvicorn.Server(config)
    server.run()
