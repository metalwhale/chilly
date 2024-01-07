# chilly
Chilly chatbot

## How to run
### Prepare data
- [Export chat history](https://slack.com/help/articles/201658943-Export-your-workspace-data)
- Rename the `.zip` file to `slack.zip` and place it into [`data`](./data/) directory

### Training
```bash
pip3 install -r requirements.txt
python3 train.py
```

## Kudos
- [`mamba-chat`](https://github.com/havenhq/mamba-chat)
