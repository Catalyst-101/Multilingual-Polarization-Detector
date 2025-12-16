This is a semester project about multilingual polarization detection / classification model. We have trained BERT based models for five different languages, English, Chinese, Urdu, Arabic and Turkish. 
In order to run it first clone it and then run the the app.py first by doing this:

uvicorn app:app --host 0.0.0.0 --port 8000

Then run the html page in the static folder.

also create a "models" folder in the root directory and clone the models from my hugging face to it before running the app.py.

git clone https://huggingface.co/Catalyst-101/Polarization-Classification-English = eng_model
git clone https://huggingface.co/Catalyst-101/Polarization-Classification-Chinese = zho_model
git clone https://huggingface.co/Catalyst-101/Polarization-Classification-Urdu = urd_model
git clone https://huggingface.co/Catalyst-101/Polarization-Classification-Arabic = arb_model
git clone https://huggingface.co/Catalyst-101/Polarization-Classification-Turkish = tur_model

your model folder should be like

models/
  eng_model/
  zho_model/
  urd_model/
  arb_model/
  tur_model/
