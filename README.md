# chilly
## Training
### Prepare data
- [Export chat history](https://slack.com/help/articles/201658943-Export-your-workspace-data)
- Extract the `.zip` file into `training/data/slack-export` folder
- Run the script to generate `.json` data files:
  ```
  docker run -it --rm -v "$PWD":/usr/src/chilly -w /usr/src/chilly python:3 python training/prepare_data.py training/data/slack-export/ training/data/
  ```
