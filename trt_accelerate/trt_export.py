import os 
import sys 
ROOT_DIR = os.path.abspath("./HuggingFace")
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
from NNDF.networks import NetworkMetadata, Precision
from T5.T5ModelConfig import T5ModelTRTConfig, T5Metadata

from GPT2.export import GPT2TorchFile,GPT2ONNXFile,GPT2TRTEngine
from GPT2.GPT2ModelConfig import GPT2ModelTRTConfig, GPT2Metadata

from T5.measurements import decoder_inference, encoder_inference, full_inference_greedy, full_inference_beam
from T5.export import T5EncoderTorchFile, T5DecoderTorchFile, T5EncoderTRTEngine, T5DecoderTRTEngine
from NNDF.networks import TimingProfile
from NNDF.torch_utils import expand_inputs_for_beam_search

from T5.export import T5DecoderONNXFile, T5EncoderONNXFile
from polygraphy.backend.trt import Profile

def t5_model2onnx(T5_VARIANT,cp_path,onnx_model_path):
    
    t5_model = T5ForConditionalGeneration.from_pretrained(T5_VARIANT)
    cp = torch.load(cp_path)
    t5_model.load_state_dict(cp)
    
    metadata=NetworkMetadata(variant=T5_VARIANT, precision=Precision(fp16=False), other=T5Metadata(kv_cache=False))
    t5_file_name = T5_VARIANT.replace("/","-")
    encoder_onnx_model_fpath = t5_file_name + "-encoder.onnx"
    decoder_onnx_model_fpath = t5_file_name + "-decoder-with-lm-head.onnx"
    
    
    t5_encoder = T5EncoderTorchFile(t5_model.to('cpu'), metadata)
    t5_decoder = T5DecoderTorchFile(t5_model.to('cpu'), metadata)
    onnx_t5_encoder = t5_encoder.as_onnx_model(
    os.path.join(onnx_model_path, encoder_onnx_model_fpath), force_overwrite=False
    )
    onnx_t5_decoder = t5_decoder.as_onnx_model(
        os.path.join(onnx_model_path, decoder_onnx_model_fpath), force_overwrite=False
    )
    return 

def opt_model2onnx(GPT2_VARIANT,cp_path,onnx_model_path):
    model = OPTForCausalLM.from_pretrained(GPT2_VARIANT)
    cp = torch.load(cp_path)
    model.load_state_dict(cp)
    metadata = NetworkMetadata(variant=GPT2_VARIANT, precision=Precision(fp16=False), other=GPT2Metadata(kv_cache=False))
    opt_file_name = GPT2_VARIANT.replace("/","-")
    onnx_fpath = opt_file_name + ".onnx"
    gpt2 = GPT2TorchFile(model.to('cpu'), metadata)
    gpt2.as_onnx_model(os.path.join(onnx_model_path,onnx_fpath), force_overwrite=False)
    return
    
def opt_model2trt(onnx_fpath,
                  onnx_model_path,
                  tensorrt_model_path,
                  GPT2_VARIANT,
                  batch_size=1):
    metadata = NetworkMetadata(variant=GPT2_VARIANT, precision=Precision(fp16=False), other=GPT2Metadata(kv_cache=False))
    max_sequence_length = GPT2ModelTRTConfig.MAX_SEQUENCE_LENGTH[metadata.variant]
    profiles = [Profile().add(
    "input_ids",
    min=(batch_size, 1),
    opt=(batch_size, max_sequence_length // 2),
    max=(batch_size, max_sequence_length),
    )]
    if not os.path.exists(os.path.join(tensorrt_model_path, onnx_fpath) + ".engine"):
        gpt2_engine = GPT2ONNXFile(os.path.join(onnx_model_path,onnx_fpath), metadata).as_trt_engine(output_fpath=os.path.join(tensorrt_model_path, onnx_fpath) + ".engine", profiles=profiles)
    else:
        gpt2_engine = GPT2TRTEngine(os.path.join(tensorrt_model_path, onnx_fpath) + ".engine", metadata)
    return 

    
def t5_model2trt(encoder_onnx_model_fpath,
                 decoder_onnx_model_fpath,
                 onnx_model_path,
                 tensorrt_model_path,
                 T5_VARIANT,
                 batch_size = 1,
                 num_beams =1):
    metadata=NetworkMetadata(variant=T5_VARIANT, precision=Precision(fp16=False), other=T5Metadata(kv_cache=False))
    max_sequence_length = T5ModelTRTConfig.MAX_SEQUENCE_LENGTH[T5_VARIANT]
    decoder_profile = Profile()
    decoder_profile.add(
        "input_ids",
        min=(batch_size * num_beams, 1),
        opt=(batch_size * num_beams, max_sequence_length // 2),
        max=(batch_size * num_beams, max_sequence_length),
    )
    decoder_profile.add(
        "encoder_hidden_states",
        min=(batch_size * num_beams, 1, max_sequence_length),
        opt=(batch_size * num_beams, max_sequence_length // 2, max_sequence_length),
        max=(batch_size * num_beams, max_sequence_length, max_sequence_length),
    )

    # Encoder optimization profiles
    encoder_profile = Profile()
    encoder_profile.add(
        "input_ids",
        min=(batch_size, 1),
        opt=(batch_size, max_sequence_length // 2),
        max=(batch_size, max_sequence_length),
    )

    if not os.path.exists(os.path.join(tensorrt_model_path, encoder_onnx_model_fpath) + ".engine"):
        t5_trt_encoder_engine = T5EncoderONNXFile(
                        os.path.join(onnx_model_path, encoder_onnx_model_fpath), metadata
                    ).as_trt_engine(os.path.join(tensorrt_model_path, encoder_onnx_model_fpath) + ".engine", profiles=[encoder_profile])
    else:
        t5_trt_encoder_engine = T5EncoderTRTEngine(os.path.join(tensorrt_model_path, encoder_onnx_model_fpath) + ".engine", metadata)

    if not os.path.exists(os.path.join(tensorrt_model_path, decoder_onnx_model_fpath) + ".engine"):    
        t5_trt_decoder_engine = T5DecoderONNXFile(
                        os.path.join(onnx_model_path, decoder_onnx_model_fpath), metadata
                    ).as_trt_engine(os.path.join(tensorrt_model_path, decoder_onnx_model_fpath) + ".engine", profiles=[decoder_profile])
    else:
        t5_trt_decoder_engine = T5DecoderTRTEngine(os.path.join(tensorrt_model_path, decoder_onnx_model_fpath) + ".engine", metadata)
    
    

    return 



if __name__ == "__main__":
    # t5_model2onnx("google/flan-t5-large","./models/flan-t5/")
    # t5_model2trt(encoder_onnx_model_fpath = "google-flan-t5-large-encoder.onnx",
    #              decoder_onnx_model_fpath = "google-flan-t5-large-decoder-with-lm-head.onnx",
    #              onnx_model_path = "./models/flan-t5",
    #              tensorrt_model_path = "./models/flan-t5",
    #              T5_VARIANT = "google/flan-t5-large",
    #              batch_size = 1,
    #              num_beams =1)
    opt_model2onnx(GPT2_VARIANT= "facebook/opt-1.3b",onnx_model_path = "./models/opt/onnx")
    opt_model2trt(onnx_fpath = "facebook-opt-1.3b.onnx",
                  onnx_model_path = "./models/opt/onnx",
                  tensorrt_model_path = "./models/opt/trt",
                  GPT2_VARIANT = "facebook/opt-1.3b",
                  batch_size=1)