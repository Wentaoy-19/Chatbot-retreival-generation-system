import os 
import sys 
ROOT_DIR = os.path.abspath("./HuggingFace/")
sys.path.append(ROOT_DIR)
import torch
import tensorrt as trt
from transformers import (
    T5ForConditionalGeneration,
    T5Tokenizer,
    T5Config,
)

from transformers import (
    OPTForCausalLM,
    GPT2Tokenizer,
    OPTConfig,
)

from transformers.generation_stopping_criteria import (
    MaxLengthCriteria,
    StoppingCriteriaList,
)
import GPT2
from NNDF.networks import NetworkMetadata, Precision
from T5.T5ModelConfig import T5ModelTRTConfig, T5Metadata
from GPT2.GPT2ModelConfig import GPT2ModelTRTConfig,GPT2Metadata
from GPT2.export import GPT2TorchFile,GPT2ONNXFile,GPT2TRTEngine
from T5.measurements import decoder_inference, encoder_inference, full_inference_greedy, full_inference_beam
from T5.export import T5EncoderTorchFile, T5DecoderTorchFile, T5EncoderTRTEngine, T5DecoderTRTEngine
from NNDF.networks import TimingProfile
from NNDF.torch_utils import expand_inputs_for_beam_search

from T5.export import T5DecoderONNXFile, T5EncoderONNXFile
from polygraphy.backend.trt import Profile


from T5.trt import T5TRTEncoder, T5TRTDecoder
from GPT2.trt import GPT2TRTDecoder
from GPT2.measurements import gpt2_inference


class trt_opt():
    def __init__(self,device,GPT2_VARIANT,trt_path):
        super(trt_opt,self).__init__()
        self.trt_path = trt_path
        self.device = device 
        self.GPT2_VARIANT = GPT2_VARIANT
        self.metadata = NetworkMetadata(variant=GPT2_VARIANT, precision=Precision(fp16=False), other=GPT2Metadata(kv_cache=False))
        self.config = GPT2ModelTRTConfig()
        self.tokenizer = GPT2Tokenizer.from_pretrained(GPT2_VARIANT)
        self.trt_model = self.load_model()
    def load_model(self):
        opt_engine = GPT2TRTEngine(self.trt_path,self.metadata)
        opt_trt = GPT2TRTDecoder(opt_engine, self.metadata, OPTConfig(self.GPT2_VARIANT))
        return opt_trt
    def inference_greedy(self,input_text:str):
        input_ids = self.tokenizer(input_text, return_tensors="pt").input_ids
        outputs = self.trt_model.generate(input_ids.to("cuda:0"), max_length=256)
        out_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)       
        # sample_output, full_e2e_runtime = GPT2.measurements.full_inference_greedy(
        #     self.trt_model,
        #     input_ids,
        #     TimingProfile(iterations=10, number=1, warmup=5, duration=0, percentile=50),
        #     max_length=self.config.MAX_SEQUENCE_LENGTH[self.GPT2_VARIANT],
        #     batch_size=1,
        #     early_stopping=True,
        # )
        # return sample_output
        # out_text, decoder_e2e_median_time = gpt2_inference(
        #     self.trt_model, input_ids, TimingProfile(iterations=10, number=1, warmup=1, duration=0, percentile=50)
        # )
        return out_text

class trt_t5():
    def __init__(self,device,T5_VARIANT,trt_encoder_path,trt_decoder_path):
        super(trt_t5,self).__init__()
        self.trt_encoder_path = trt_encoder_path
        self.trt_decoder_path = trt_decoder_path
        self.T5_VARIANT = T5_VARIANT
        self.device = device
        self.metadata=NetworkMetadata(variant=T5_VARIANT, precision=Precision(fp16=False), other=T5Metadata(kv_cache=False))
        self.config = T5ModelTRTConfig()
        self.tokenizer = T5Tokenizer.from_pretrained(T5_VARIANT)
        self.encoder,self.decoder =  self.load_model()
    def load_model(self,num_beams=1):
        t5_trt_encoder_engine = T5EncoderTRTEngine(self.trt_encoder_path, self.metadata)
        t5_trt_decoder_engine = T5DecoderTRTEngine(self.trt_decoder_path, self.metadata)
        tfm_config = T5Config(
            use_cache=True,
            num_layers=self.config.NUMBER_OF_LAYERS[self.T5_VARIANT],
        )  
        t5_trt_encoder = T5TRTEncoder(
                        t5_trt_encoder_engine, self.metadata, tfm_config
                    )
        t5_trt_decoder = T5TRTDecoder(
                        t5_trt_decoder_engine, self.metadata, tfm_config, num_beams=num_beams
                    )
        return t5_trt_encoder,t5_trt_decoder
    def inference_greedy(self,input_text):
        input_ids = self.tokenizer(input_text, return_tensors="pt").input_ids
        # decoder_output, full_e2e_median_runtime = full_inference_greedy(
        # self.encoder,
        # self.decoder,
        # input_ids,
        # self.tokenizer,
        # TimingProfile(iterations=10, number=1, warmup=1, duration=0, percentile=50),
        # max_length=self.config.MAX_SEQUENCE_LENGTH[self.metadata.variant],
        # use_cuda=True,
        # )
        max_length = self.config.MAX_SEQUENCE_LENGTH[self.metadata.variant]
        encoder_last_hidden_state = self.encoder(input_ids=input_ids)
        decoder_input_ids = torch.full(
            (1, 1), self.tokenizer.convert_tokens_to_ids(self.tokenizer.pad_token), dtype=torch.int32
        ).to("cuda:0")
        outputs = self.decoder.greedy_search(
                    input_ids=decoder_input_ids,
                    encoder_hidden_states=encoder_last_hidden_state,
                    stopping_criteria = StoppingCriteriaList([MaxLengthCriteria(max_length)])
                )
        out_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return out_text
    
if __name__ == '__main__':
    # my_t5 = trt_t5(device= torch.device("cuda:0"),
    #        T5_VARIANT = "google/flan-t5-large",
    #        trt_decoder_path = "./models/flan-t5/google-flan-t5-large-decoder-with-lm-head.onnx.engine",
    #        trt_encoder_path = "./models/flan-t5/google-flan-t5-large-encoder.onnx.engine")
    # prompt = '''
    # Answer question from context
    # Context: Turing was prosecuted in 1952 for homosexual acts. He accepted hormone treatment with DES, a procedure commonly referred to as chemical castration, as an alternative to prison. Turing died on 7 June 1954, 16 days before his 42nd birthday, from cyanide poisoning. An inquest determined his death as a suicide, but it has been noted that the known evidence is also consistent with accidental poisoning. Following a public campaign in 2009, the British Prime Minister Gordon Brown made an official public apology on behalf of the British government for "the appalling way [Turing] was treated". Queen Elizabeth II granted a posthumous pardon in 2013. The term "Alan Turing law" is now used informally to refer to a 2017 law in the United Kingdom that retroactively pardoned men cautioned or convicted under historical legislation that outlawed homosexual acts.
    # question: Who is Alan Turing? 
    # Answer:
    # '''
    # out_text = my_t5.inference_greedy(prompt)
    # print(out_text)
    my_opt = trt_opt(
        device = torch.device("cuda:0"),GPT2_VARIANT = "facebook/opt-1.3b",trt_path = "./models/opt/trt/facebook-opt-1.3b.onnx.engine"
    )
    prompt = "Turing machine is"
    out_text = my_opt.inference_greedy(prompt)
    print(out_text)
    