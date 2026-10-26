"""Export saved SFT weights through the supported Tinker external-weights path."""
import json
from tinker_cookbook import weights
from paper_character_v7 import RUN

if __name__=='__main__':
    checkpoint=json.loads((RUN/'sft/checkpoint.json').read_text())
    output=RUN/'hf_sft_export/adapter'
    path=weights.download(tinker_path=checkpoint['state_path'],output_dir=str(output))
    (RUN/'sft/export.json').write_text(json.dumps({'source':checkpoint,'adapter_directory':path,
        'method':'supported_external_weights_export','additional_optimizer_updates':0},indent=2))
    print(json.dumps({'exported_sft_adapter':path}),flush=True)
