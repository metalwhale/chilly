# chilly
## How to run
### Prepare data
- [Export chat history](https://slack.com/help/articles/201658943-Export-your-workspace-data)
- Put the `.zip` file into [`data`](./data/) directory

### Training
```
pip3 install -r requirements.txt
python3 train.py > data/train.log 2>&1 &
```

## Kudos
- [`tloen/alpaca-lora`](https://github.com/tloen/alpaca-lora)
