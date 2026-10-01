# REX data donation notebook

Why: researchers considering data donation need to see what a donated dataset actually contains and what it can answer, before they design a study around it. This notebook takes one export from a REX browser extension and shows that in a few charts, with no coding needed to run it. It was prepared for the Data Donation Symposium 2026.

REX (Research EXtensions) is an open-source framework from BRIC for building browser extensions that collect web and AI-chatbot data with participants' consent.

## What it shows

1. **What a donor gives.** How much of the browsing history arrives as a full web address, as the site name only, or as a category such as "email" in place of the site.
2. **When people browse and chat, and what a conversation looks like.** Activity by hour and weekday, prompts per conversation, and prompt length against reply length.
3. **What the model does behind the scenes.** The model that answered, how long it reasoned, and hidden steps such as writing to its memory of the user. ChatGPT's own data download leaves out the hidden steps and the exact reasoning times.
4. **Whether people search before they ask.** Browsing in the half hour around each ChatGPT conversation, with sites grouped rather than named.

## What you need

A REX export: the JSON file the REX extension saves when you download your data.

## Run it in your browser (no install)

[Open the notebook in Google Colab](https://colab.research.google.com/github/bric-digital/rex-data-donation-notebook/blob/main/rex_data_donation.ipynb)

Choose Runtime > Run all. When asked, choose your export file.

Colab uploads your export to Google's servers. Use it only for data you are free to share there. To keep the data on your own computer, run it locally instead.

## Run it on your own computer

With [uv](https://docs.astral.sh/uv/) installed, from this folder:

```
uv run --python 3.12 --with-requirements requirements.txt jupyter notebook rex_data_donation.ipynb
```

Without uv, using Python 3.11 or later:

```
python3 -m pip install -r requirements.txt
jupyter notebook rex_data_donation.ipynb
```

Then set `EXPORT_PATH` in the second cell to your export file and choose Run > Run All Cells.

You can also open `rex_data_donation.ipynb` in VS Code, using a Python environment that has pandas and matplotlib.

## Keeping data out of the repo

Once it has run, the notebook holds the participant's browsing and conversation data. If you change the notebook and want to contribute the change, clear its outputs first (Edit > Clear Outputs of All Cells). The `.gitignore` excludes `.json` files so exports are not committed by accident.

## Using the loader in your own code

`rex_export.py` loads an export into pandas tables:

```python
from rex_export import load_export

export = load_export("path/to/export.json")
export.visits         # one row per history visit, with its privacy level
export.conversations  # one row per ChatGPT conversation
export.messages       # one row per message, including hidden tool steps
export.events         # extension events, such as a crawl finishing
```

To run its tests:

```
uv run --python 3.12 --with-requirements requirements.txt --with pytest pytest tests
```

The tests build their own made-up records and never read real exports.

## License

Apache 2.0. See `LICENSE`.
