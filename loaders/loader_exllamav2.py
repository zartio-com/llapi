import time

from exllamav2 import (
    ExLlamaV2,
    ExLlamaV2Config,
    ExLlamaV2Cache,
    ExLlamaV2Tokenizer, ExLlamaV2Cache_8bit,
)
from exllamav2.generator import ExLlamaV2Sampler, ExLlamaV2BaseGenerator

from wrappers import ModelWrapper


class ExLlamaV2ModelWrapper(ModelWrapper):
    def __init__(self,
                 config: ExLlamaV2Config, model: ExLlamaV2, tokenizer: ExLlamaV2Tokenizer,
                 generator: ExLlamaV2BaseGenerator, settings: ExLlamaV2Sampler.Settings):
        super().__init__()

        self.config = config
        self.model = model
        self.tokenizer = tokenizer
        self.generator = generator
        self.settings = settings

    @staticmethod
    def load(model_dir: str) -> ModelWrapper:
        config = ExLlamaV2Config()
        config.model_dir = model_dir
        config.prepare()

        model = ExLlamaV2(config)

        cache = ExLlamaV2Cache_8bit(model, lazy=True)
        model.load_autosplit(cache)

        tokenizer = ExLlamaV2Tokenizer(config)

        generator = ExLlamaV2BaseGenerator(model, cache, tokenizer)

        settings = ExLlamaV2Sampler.Settings()
        settings.temperature = 0.85
        settings.top_k = 50
        settings.top_p = 0.8
        settings.token_repetition_penalty = 1.15
        settings.disallow_tokens(tokenizer, [tokenizer.eos_token_id])

        generator.warmup()

        return ExLlamaV2ModelWrapper(config, model, tokenizer, generator, settings)

    def predict(self, text_input: str, settings: dict) -> str:
        time_begin = time.time()

        llama_sampler_settings = self._default_settings()
        if 'sampler' in settings:
            sampler_settings: dict = settings['sampler']
            if 'temperature' in sampler_settings:
                llama_sampler_settings.temperature = sampler_settings['temperature']
            if 'top_k' in sampler_settings:
                llama_sampler_settings.top_k = sampler_settings['top_k']
            if 'top_p' in sampler_settings:
                llama_sampler_settings.top_p = sampler_settings['top_p']
            if 'token_repetition_penalty' in sampler_settings:
                llama_sampler_settings.token_repetition_penalty = sampler_settings['token_repetition_penalty']
            if 'disallow_eos_token' in sampler_settings and sampler_settings['disallow_eos_token']:
                llama_sampler_settings.disallow_tokens(self.tokenizer, [self.tokenizer.eos_token_id])
            if 'typical' in sampler_settings:
                llama_sampler_settings.typical = sampler_settings['typical']

        max_tokens = settings['tokens_limit'] if 'tokens_limit' in settings else 250
        seed = settings['seed'] if 'seed' in settings else 1234
        output = self.generator.generate_simple(text_input, llama_sampler_settings, max_tokens, seed=seed)

        time_total = time.time() - time_begin
        print(f'Response generated in {time_total:.2f} seconds, {250} tokens, {250 / time_total:.2f} tokens/second')

        return output

    def _default_settings(self) -> ExLlamaV2Sampler.Settings:
        settings = ExLlamaV2Sampler.Settings()
        settings.temperature = 0.85
        settings.top_k = 50
        settings.top_p = 0.8
        settings.token_repetition_penalty = 1.15
        return settings
