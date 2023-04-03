# chilly
## Training
### Prepare instruction data
- [Export chat history](https://slack.com/help/articles/201658943-Export-your-workspace-data)
- Extract the `.zip` file into `training/data/slack-export` folder
- Run the script to generate `instructions.json` file:
  ```
  docker run -it --rm -v "$PWD":/usr/src/chilly -w /usr/src/chilly python:3 python training/prepare_data.py training/data/slack-export/ training/data/instructions.json
  ```

## Fine-tuning
- Clone [alpaca-lora](https://github.com/tloen/alpaca-lora) repo
- Install requirements:
  ```
  cd alpaca-lora/
  pip install -r requirements.txt
  ```
- Run the script (with **`INSTRUCTIONS_FILE_PATH`** is the path to `instructions.json` file):
  <pre>
  python finetune.py \
    --base_model='decapoda-research/llama-7b-hf' \
    --data_path='<b>INSTRUCTIONS_FILE_PATH</b>' \
    --num_epochs=4 \
    --cutoff_len=512 \
    --group_by_length \
    --output_dir='./lora-alpaca' \
    --lora_target_modules='[q_proj,k_proj,v_proj,o_proj]' \
    --lora_r=16 \
    --micro_batch_size=8
  </pre>
  <details><summary>Hint</summary>

  If you want to use Alpaca-LoRA weights instead of the official weights, add something similar to these lines to `finetune.py` file (see [`generate.py`](https://github.com/tloen/alpaca-lora/blob/main/generate.py#L47-L51) for more details):
  ```python
  if lora_weights != "":
    model = PeftModel.from_pretrained(model, lora_weights, torch_dtype=torch.float16)
  ```
  and pass `--lora_weights 'tloen/alpaca-lora-7b'` param when running the script.
  </details>
